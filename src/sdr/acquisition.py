import numpy as np
from scipy.stats import chi2
from src.sdr.frontend import FS, N_MS, unpack_vesper_iq

F_L1 = 1575.42e6
BIT_MS = 20  # GPS navigation data bit duration (20 ms)

# --- FEATURE TOGGLES (Set to False to revert to original behavior) ---
USE_BIT_SYNC_20MS = True
USE_CODE_PHASE_AIDING = True


def generate_native_prns(fs: float = FS) -> dict:
    """Generates GPS L1 C/A FFT replicas sampled at 1.024 MHz."""
    prns = {}
    chip_idx = np.floor(np.arange(N_MS) * (1.023e6 / fs)).astype(np.int64) % 1023

    sv_taps = {
        1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9),
        6: (2, 10), 7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3),
        11: (3, 4), 12: (5, 6), 13: (6, 7), 14: (7, 8), 15: (8, 9),
        16: (9, 10), 17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7),
        21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6), 25: (5, 7),
        26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
        31: (3, 8), 32: (4, 9),
    }

    for prn, (t1, t2) in sv_taps.items():
        g1, g2 = np.ones(10, dtype=int), np.ones(10, dtype=int)
        code = np.empty(1023, dtype=np.float32)
        for i in range(1023):
            chip = g1[9] ^ g2[t1 - 1] ^ g2[t2 - 1]
            code[i] = 1.0 - 2.0 * chip
            new_g1 = g1[2] ^ g1[9]
            new_g2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
            g1 = np.roll(g1, 1); g1[0] = new_g1
            g2 = np.roll(g2, 1); g2[0] = new_g2

        prns[f"G{prn:02d}"] = np.conj(np.fft.fft(code[chip_idx])).astype(np.complex64)

    return prns


def parabolic_interpolation(power_vec: np.ndarray, peak_idx: int) -> float:
    """Original 3-point parabolic correlation peak interpolation."""
    N = len(power_vec)
    y_m1 = float(power_vec[(peak_idx - 1) % N])
    y_0  = float(power_vec[peak_idx])
    y_p1 = float(power_vec[(peak_idx + 1) % N])
    denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
    if denom <= 1e-12:
        return float(peak_idx)
    delta = (y_p1 - y_m1) / denom
    return (peak_idx + np.clip(delta, -0.5, 0.5)) % N


def _bit_aligned_20ms(corr1: np.ndarray, cand_lags: np.ndarray = None):
    """
    Tests all 20 navigation data-bit edge offsets (0..19 ms) and coherently
    integrates across full 20 ms data bits for +4 to +6 dB SNR gain.
    """
    K = corr1.shape[0]
    best = None
    for off in range(BIT_MS):
        G = (K - off) // BIT_MS
        if G < 1:
            break
        c = corr1[off : off + G * BIT_MS].reshape(G, BIT_MS, -1).sum(axis=1)
        pw = np.sum(c.real ** 2 + c.imag ** 2, axis=0)
        peak = float(pw[cand_lags].max()) if cand_lags is not None else float(pw.max())
        snr_metric = peak / max(float(pw.mean()), 1e-12)
        if best is None or snr_metric > best[0]:
            best = (snr_metric, pw, c, G)
    return best[1], best[2], best[3]


def _upsampled_peak(groups: np.ndarray, lag0: int, search: int = 3, factor: int = 8) -> float:
    """
    8x FFT zero-padded band-limited sinc interpolation on coherent groups.
    Eliminates 1-sample-per-chip quantization bias at 1.024 MHz.
    """
    G, L = groups.shape
    N = L * factor
    Cf = np.fft.fft(groups, axis=1)
    h = L // 2
    up = np.zeros((G, N), dtype=np.complex64)
    up[:, :h] = Cf[:, :h]
    up[:, -h:] = Cf[:, -h:]
    u = np.sum(np.abs(np.fft.ifft(up, axis=1)) ** 2, axis=0)

    idx = (lag0 * factor + np.arange(-search * factor, search * factor + 1)) % N
    j = int(idx[np.argmax(u[idx])])
    ym, y0, yp = float(u[(j - 1) % N]), float(u[j]), float(u[(j + 1) % N])
    den = ym - 2.0 * y0 + yp
    frac = 0.5 * (ym - yp) / den if den < -1e-12 else 0.0
    return float((j + np.clip(frac, -0.5, 0.5)) / factor) % L


def refine_satellite_bit_sync(
    seg: np.ndarray,
    code_f: np.ndarray,
    doppler_hz: float,
    coarse_lag: int,
):
    """
    Refines a detected satellite's Doppler, code phase, and C/N0 using
    1ms code-Doppler alignment, 20ms bit-edge synchronization, and 8x FFT upsampling.
    """
    K = len(seg) // N_MS
    if K < 2 * BIT_MS:
        return None

    t = np.arange(K * N_MS, dtype=np.float64) / FS
    freq_k = np.fft.fftfreq(N_MS, d=1.0 / N_MS)
    ms_times_s = (np.arange(K) + 0.5) * 1e-3

    # Mix carrier and remove code-Doppler walk on every 1 ms block
    w = seg[: K * N_MS] * np.exp(-2j * np.pi * doppler_hz * t).astype(np.complex64)
    F1 = np.fft.fft(w.reshape(K, N_MS), axis=1)
    shift_samples = (doppler_hz / F_L1) * FS * ms_times_s
    phase_ramp = np.exp(-2j * np.pi * shift_samples[:, None] * freq_k[None, :] / N_MS).astype(np.complex64)
    corr1 = np.fft.ifft((F1 * phase_ramp) * code_f[None, :], axis=1)

    cand = (coarse_lag + np.arange(-3, 4)) % N_MS
    power, groups, G = _bit_aligned_20ms(corr1, cand)
    best_lag = int(cand[np.argmax(power[cand])])

    # Refine Doppler using squared prompt correlations (wipes out 180-deg data bit flips)
    z = groups[:, best_lag]
    fine_dop = doppler_hz
    if z.size >= 2:
        z2 = z ** 2
        dphi = float(np.angle(np.sum(z2[1:] * np.conj(z2[:-1]))))
        fine_dop += float(np.clip(dphi / (2.0 * np.pi * 2.0 * BIT_MS * 1e-3), -25.0, 25.0))

    sub_sample = _upsampled_peak(groups, best_lag, search=3, factor=8)

    mask = np.ones(N_MS, dtype=bool)
    mask[(best_lag + np.arange(-3, 4)) % N_MS] = False
    noise = float(np.mean(power[mask])) / G
    sig = max(float(power[best_lag]) / G - noise, 1e-6)
    snr = max(sig / max(noise, 1e-12), 1e-6)
    cn0_20ms = float(10.0 * np.log10(snr / (BIT_MS * 1e-3)))

    return sub_sample, fine_dop, cn0_20ms


def fast_signal_gate(seg: np.ndarray, codes_f: np.ndarray, gate_ms: int = 80, coh_ms: int = 5) -> int:
    """
    Stage 1 Fast Gate: Screens the first 80ms at 200 Hz steps to check if at least
    2 satellites show signal presence before committing to a full sweep.
    """
    G = gate_ms // coh_ms
    if len(seg) < G * coh_ms * N_MS:
        return 0

    sub_seg = seg[: G * coh_ms * N_MS]
    t = np.arange(G * coh_ms * N_MS, dtype=np.float64) / FS
    doppler_bins = np.arange(-4800, 5000, 200)
    cfar_gate = float(chi2.isf(1e-3 / (N_MS * len(doppler_bins)), 2 * G) / (2 * G))

    best_snr = np.zeros(len(codes_f), dtype=np.float64)
    for doppler in doppler_bins:
        w = sub_seg * np.exp(-2j * np.pi * doppler * t).astype(np.complex64)
        F = np.fft.fft(w.reshape(G, coh_ms, N_MS), axis=2).sum(axis=1)
        corr = np.fft.ifft(F[None, :, :] * codes_f[:, None, :], axis=2)
        power = np.sum(corr.real ** 2 + corr.imag ** 2, axis=1)
        means = np.mean(power, axis=1)
        peaks = np.max(power, axis=1)
        snrs = np.divide(peaks, means, out=np.zeros_like(peaks), where=means > 0)
        best_snr = np.maximum(best_snr, snrs)

    return int(np.sum(best_snr >= cfar_gate))


def acquire_constellation(
    filepath: str,
    prn_fft_dict: dict,
    total_ms: int = 480,
    coh_ms: int = 8,
    pfa: float = 2e-3,
    use_fast_gate: bool = True,
):
    """
    Acquires GPS constellation with Code-Doppler walk compensation,
    parabolic interpolation, and optional 20ms Bit-Sync + 8x FFT Sinc refinement.
    """
    seg, actual_ms = unpack_vesper_iq(filepath, total_ms=total_ms)
    G = actual_ms // coh_ms
    if G < 10:
        return [], 999.0

    prn_list = list(prn_fft_dict.keys())
    codes_f = np.stack([prn_fft_dict[p] for p in prn_list])

    if use_fast_gate and fast_signal_gate(seg, codes_f) < 2:
        return [], 0.0

    seg = seg[: G * coh_ms * N_MS]
    t = np.arange(G * coh_ms * N_MS, dtype=np.float64) / FS

    doppler_step = 50.0
    doppler_bins = np.arange(-5000, 5050, doppler_step)
    cells = N_MS * len(doppler_bins)
    cfar_threshold = float(chi2.isf(pfa / cells, 2 * G) / (2 * G))

    n_prn = len(prn_list)
    n_dop = len(doppler_bins)
    snr_grid = np.zeros((n_prn, n_dop), dtype=np.float64)
    best_snr = np.zeros(n_prn, dtype=np.float64)
    best_dop_idx = np.zeros(n_prn, dtype=np.int64)
    best_profile = np.zeros((n_prn, N_MS), dtype=np.float64)

    freq_k = np.fft.fftfreq(N_MS, d=1.0 / N_MS)
    group_times_s = (np.arange(G) * coh_ms + 0.5 * coh_ms) * 1e-3

    for d_idx, doppler in enumerate(doppler_bins):
        w = seg * np.exp(-2j * np.pi * doppler * t).astype(np.complex64)
        F = np.fft.fft(w.reshape(G, coh_ms, N_MS), axis=2).sum(axis=1)

        shift_samples = (doppler / F_L1) * FS * group_times_s
        phase_ramp = np.exp(-2j * np.pi * shift_samples[:, None] * freq_k[None, :] / N_MS).astype(np.complex64)
        F_aligned = F * phase_ramp

        corr = np.fft.ifft(F_aligned[None, :, :] * codes_f[:, None, :], axis=2)
        power = np.sum(corr.real ** 2 + corr.imag ** 2, axis=1)

        means = np.mean(power, axis=1)
        peaks = np.max(power, axis=1)
        snrs = np.divide(peaks, means, out=np.zeros_like(peaks), where=means > 0)
        snr_grid[:, d_idx] = snrs

        improved = snrs > best_snr
        best_snr[improved] = snrs[improved]
        best_dop_idx[improved] = d_idx
        best_profile[improved] = power[improved]

    detections = []
    for i, prn in enumerate(prn_list):
        if best_snr[i] >= cfar_threshold:
            peak_bin = int(np.argmax(best_profile[i]))
            sub_sample = parabolic_interpolation(best_profile[i], peak_bin)

            d_idx = int(best_dop_idx[i])
            if 0 < d_idx < n_dop - 1:
                y_m1, y_0, y_p1 = snr_grid[i, d_idx - 1], snr_grid[i, d_idx], snr_grid[i, d_idx + 1]
                denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
                delta_d = np.clip((y_p1 - y_m1) / denom, -0.5, 0.5) if denom > 1e-12 else 0.0
                fine_doppler = float(doppler_bins[d_idx] + delta_d * doppler_step)
            else:
                fine_doppler = float(doppler_bins[d_idx])

            cn0_est = 10.0 * np.log10(max((best_snr[i] - 1.0) / (coh_ms * 1e-3), 1.0))

            # Optional 20ms Bit-Synchronized + 8x FFT Sinc Peak Refinement
            if USE_BIT_SYNC_20MS:
                ref = refine_satellite_bit_sync(seg, codes_f[i], fine_doppler, peak_bin)
                if ref is not None:
                    sub_sample, fine_doppler, cn0_est = ref

            tau_ms = sub_sample / float(N_MS)
            detections.append({
                "prn": prn,
                "metric": float(best_snr[i]),
                "cn0_dbhz": float(cn0_est),
                "doppler_hz": fine_doppler,
                "sample_phase": float(sub_sample),
                "code_phase_chips": float(tau_ms * 1023.0),
                "tau_ms": float(tau_ms),
            })

    detections.sort(key=lambda d: d["metric"], reverse=True)
    return detections, cfar_threshold


def acquire_ephemeris_aided(
    filepath: str,
    prn_fft_dict: dict,
    existing_detections: list,
    visible_sv_dopplers: dict,
    predicted_code_phases: dict = None,
    lag_window: int = 8,
    total_ms: int = 480,
    coh_ms: int = 8,
    pfa: float = 5e-3,
):
    """
    Stage 3 (Ephemeris-Aided Acquisition):
    Uses predicted Doppler (and optionally predicted code-phase arrival within
    +/- lag_window samples) to recover weaker visible satellites.
    """
    locked_prns = {d["prn"] for d in existing_detections}
    candidate_prns = [p for p in visible_sv_dopplers if p not in locked_prns and p in prn_fft_dict]
    if not candidate_prns:
        return existing_detections

    seg, actual_ms = unpack_vesper_iq(filepath, total_ms=total_ms)
    G = actual_ms // coh_ms
    if G < 10:
        return existing_detections

    seg = seg[: G * coh_ms * N_MS]
    t = np.arange(G * coh_ms * N_MS, dtype=np.float64) / FS
    freq_k = np.fft.fftfreq(N_MS, d=1.0 / N_MS)
    group_times_s = (np.arange(G) * coh_ms + 0.5 * coh_ms) * 1e-3

    doppler_step = 50.0
    n_bins = 13  # +/- 300 Hz window
    aided_detections = list(existing_detections)

    for prn in candidate_prns:
        center_dop = round(visible_sv_dopplers[prn] / doppler_step) * doppler_step
        doppler_bins = center_dop + np.arange(-6, 7) * doppler_step
        code_f = prn_fft_dict[prn]

        # Restrict code-phase search window if code-phase prediction is active & available
        if USE_CODE_PHASE_AIDING and predicted_code_phases and prn in predicted_code_phases:
            center_lag = int(round(predicted_code_phases[prn] * N_MS)) % N_MS
            lags = (center_lag + np.arange(-lag_window, lag_window + 1)) % N_MS
        else:
            lags = np.arange(N_MS)

        cfar_aided = float(chi2.isf(pfa / (len(lags) * n_bins), 2 * G) / (2 * G))

        snrs = np.zeros(len(doppler_bins), dtype=np.float64)
        best_lags = np.zeros(len(doppler_bins), dtype=np.int64)
        profiles = np.zeros((len(doppler_bins), N_MS), dtype=np.float64)

        for d_idx, doppler in enumerate(doppler_bins):
            w = seg * np.exp(-2j * np.pi * doppler * t).astype(np.complex64)
            F = np.fft.fft(w.reshape(G, coh_ms, N_MS), axis=2).sum(axis=1)
            shift_samples = (doppler / F_L1) * FS * group_times_s
            phase_ramp = np.exp(-2j * np.pi * shift_samples[:, None] * freq_k[None, :] / N_MS).astype(np.complex64)
            F_aligned = F * phase_ramp

            corr = np.fft.ifft(F_aligned * code_f[None, :], axis=1)
            power = np.sum(corr.real ** 2 + corr.imag ** 2, axis=0)
            mean_p = float(np.mean(power))
            sub_pw = power[lags]
            best_local = int(np.argmax(sub_pw))
            snrs[d_idx] = float(sub_pw[best_local]) / mean_p if mean_p > 0 else 0.0
            best_lags[d_idx] = int(lags[best_local])
            profiles[d_idx] = power

        best_d_idx = int(np.argmax(snrs))
        if snrs[best_d_idx] >= cfar_aided:
            peak_bin = int(best_lags[best_d_idx])
            sub_sample = parabolic_interpolation(profiles[best_d_idx], peak_bin)

            if 0 < best_d_idx < len(doppler_bins) - 1:
                y_m1, y_0, y_p1 = snrs[best_d_idx - 1], snrs[best_d_idx], snrs[best_d_idx + 1]
                denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
                delta_d = np.clip((y_p1 - y_m1) / denom, -0.5, 0.5) if denom > 1e-12 else 0.0
                fine_doppler = float(doppler_bins[best_d_idx] + delta_d * doppler_step)
            else:
                fine_doppler = float(doppler_bins[best_d_idx])

            cn0_est = 10.0 * np.log10(max((snrs[best_d_idx] - 1.0) / (coh_ms * 1e-3), 1.0))

            if USE_BIT_SYNC_20MS:
                ref = refine_satellite_bit_sync(seg, code_f, fine_doppler, peak_bin)
                if ref is not None:
                    sub_sample, fine_doppler, cn0_est = ref

            tau_ms = sub_sample / float(N_MS)
            aided_detections.append({
                "prn": prn,
                "metric": float(snrs[best_d_idx]),
                "cn0_dbhz": float(cn0_est),
                "doppler_hz": fine_doppler,
                "sample_phase": float(sub_sample),
                "code_phase_chips": float(tau_ms * 1023.0),
                "tau_ms": float(tau_ms),
            })

    aided_detections.sort(key=lambda d: d["metric"], reverse=True)
    return aided_detections

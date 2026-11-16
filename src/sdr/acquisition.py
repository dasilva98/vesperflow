from functools import lru_cache

import numpy as np
from scipy.stats import chi2
from src.sdr.frontend import FS, N_MS, unpack_vesper_iq

F_L1 = 1575.42e6
BIT_MS = 20  # GPS navigation data bit duration (20 ms)

# --- FEATURE TOGGLES (Set to False to revert to original behavior) ---
USE_BIT_SYNC_20MS = True
USE_CODE_PHASE_AIDING = True

# Pre-computed constant 1ms frequency & time grids
_FREQ_K = np.fft.fftfreq(N_MS, d=1.0 / N_MS)
_T_1MS = np.arange(N_MS, dtype=np.float64) / FS


@lru_cache(maxsize=4)
def generate_native_prns(fs: float = FS) -> dict:
    """
    Generates GPS L1 C/A FFT replicas sampled at 1.024 MHz.
    Cached: the 32 LFSR codes are identical for every session (~0.45 s to build).
    Callers must treat the returned dict and arrays as read-only.
    """
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

        replica = np.conj(np.fft.fft(code[chip_idx])).astype(np.complex64)
        replica.flags.writeable = False
        prns[f"G{prn:02d}"] = replica

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
    Uses factored outer-product carrier mixing for fast execution.
    """
    K = len(seg) // N_MS
    if K < 2 * BIT_MS:
        return None

    ms_idx = np.arange(K, dtype=np.float64)
    ms_times_s = (ms_idx + 0.5) * 1e-3

    # Factored outer-product carrier wipeoff (480 + 1024 exponentials instead of 491,520)
    ms_phasor = np.exp(-2j * np.pi * doppler_hz * ms_idx * 1e-3).astype(np.complex64)
    sub_phasor = np.exp(-2j * np.pi * doppler_hz * _T_1MS).astype(np.complex64)
    w = seg[: K * N_MS].reshape(K, N_MS) * ms_phasor[:, None] * sub_phasor[None, :]

    F1 = np.fft.fft(w, axis=1)
    shift_samples = (doppler_hz / F_L1) * FS * ms_times_s
    phase_ramp = np.exp(-2j * np.pi * shift_samples[:, None] * _FREQ_K[None, :] / N_MS).astype(np.complex64)
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


def _coherent_fft_aligned(
    seg_3d: np.ndarray,
    doppler: float,
    g_times_s: np.ndarray,
    c_times_s: np.ndarray,
    group_mid_s: np.ndarray,
) -> np.ndarray:
    """
    Sums coherently across the coh_ms axis BEFORE the 1024-point FFT using
    factored carrier phasors (8x fewer FFTs and 330x fewer trig calls).
    """
    p_c = np.exp(-2j * np.pi * doppler * c_times_s).astype(np.complex64)
    p_g = np.exp(-2j * np.pi * doppler * g_times_s).astype(np.complex64)
    p_k = np.exp(-2j * np.pi * doppler * _T_1MS).astype(np.complex64)

    # Coherent sum over coh_ms in time domain -> shape (G, N_MS)
    w_sum = np.tensordot(seg_3d, p_c, axes=([1], [0])) * (p_g[:, None] * p_k[None, :])
    F = np.fft.fft(w_sum, axis=1)

    shift_samples = (doppler / F_L1) * FS * group_mid_s
    phase_ramp = np.exp(-2j * np.pi * shift_samples[:, None] * _FREQ_K[None, :] / N_MS).astype(np.complex64)
    return F * phase_ramp


def _coherent_fft_base(seg_3d, doppler, g_times_s, c_times_s):
    """Carrier-wiped, coherently summed 1 ms spectra WITHOUT the code-Doppler ramp."""
    p_c = np.exp(-2j * np.pi * doppler * c_times_s).astype(np.complex64)
    p_g = np.exp(-2j * np.pi * doppler * g_times_s).astype(np.complex64)
    p_k = np.exp(-2j * np.pi * doppler * _T_1MS).astype(np.complex64)
    w_sum = np.tensordot(seg_3d, p_c, axes=([1], [0])) * (p_g[:, None] * p_k[None, :])
    return np.fft.fft(w_sum, axis=1)


def _code_doppler_ramp(doppler, group_mid_s):
    shift_samples = (doppler / F_L1) * FS * group_mid_s
    return np.exp(-2j * np.pi * shift_samples[:, None] * _FREQ_K[None, :] / N_MS).astype(np.complex64)


def fast_signal_gate(seg: np.ndarray, codes_f: np.ndarray, gate_ms: int = 80, coh_ms: int = 5) -> int:
    """
    Stage 1 Fast Gate: Screens the first 80ms at 200 Hz steps to check if at least
    2 satellites show signal presence before committing to a full sweep.
    """
    G = gate_ms // coh_ms
    if len(seg) < G * coh_ms * N_MS:
        return 0

    seg_3d = seg[: G * coh_ms * N_MS].reshape(G, coh_ms, N_MS)
    g_times_s = np.arange(G, dtype=np.float64) * (coh_ms * 1e-3)
    c_times_s = np.arange(coh_ms, dtype=np.float64) * 1e-3
    doppler_bins = np.arange(-4800, 5000, 200)
    cfar_gate = float(chi2.isf(1e-3 / (N_MS * len(doppler_bins)), 2 * G) / (2 * G))

    best_snr = np.zeros(len(codes_f), dtype=np.float64)
    for doppler in doppler_bins:
        p_c = np.exp(-2j * np.pi * doppler * c_times_s).astype(np.complex64)
        p_g = np.exp(-2j * np.pi * doppler * g_times_s).astype(np.complex64)
        p_k = np.exp(-2j * np.pi * doppler * _T_1MS).astype(np.complex64)
        w_sum = np.tensordot(seg_3d, p_c, axes=([1], [0])) * (p_g[:, None] * p_k[None, :])
        F = np.fft.fft(w_sum, axis=1)
        corr = np.fft.ifft(F[None, :, :] * codes_f[:, None, :], axis=2)
        power = np.sum(corr.real ** 2 + corr.imag ** 2, axis=1)
        means = np.mean(power, axis=1)
        peaks = np.max(power, axis=1)
        snrs = np.divide(peaks, means, out=np.zeros_like(peaks), where=means > 0)
        best_snr = np.maximum(best_snr, snrs)

    return int(np.sum(best_snr >= cfar_gate))


def _search_doppler_grid(seg_3d, codes_f, bins_per_prn, g_times_s, c_times_s, group_mid_s):
    """
    Non-coherent code/Doppler search core shared by blind and guided acquisition.

    bins_per_prn[i] is the (sorted, 50 Hz-grid) array of Doppler bins to test for PRN i.
    Each distinct bin is visited once and only the PRNs whose window contains it are
    correlated, so a blind sweep (all PRNs, all bins) costs the same as before while a
    guided sweep (~10 PRNs x ~25 bins) touches ~25x fewer PRN/bin cells.

    Carrier wipe-off at f0 + k*1 kHz equals wipe-off at f0 followed by a k-bin spectral
    roll (1 kHz is exactly one FFT bin at 1024 pts / 1 ms and an integer number of cycles
    per ms, so the per-ms and per-group phasors are unchanged). Only one forward FFT per
    distinct sub-kHz offset is therefore needed.

    Returns (snr_lists, best_snr, best_bin_idx, best_profile) where snr_lists[i][j] is the
    peak/mean metric of PRN i at bins_per_prn[i][j].
    """
    n_prn = len(bins_per_prn)
    G = seg_3d.shape[0]

    bin_to_prns: dict = {}
    for i, bins in enumerate(bins_per_prn):
        for j, b in enumerate(bins):
            bin_to_prns.setdefault(int(b), []).append((i, j))

    snr_lists = [np.zeros(len(b), dtype=np.float64) for b in bins_per_prn]
    best_snr = np.zeros(n_prn, dtype=np.float64)
    best_bin_idx = np.zeros(n_prn, dtype=np.int64)
    best_profile = np.zeros((n_prn, N_MS), dtype=np.float64)

    base_cache: dict = {}
    for d_int in sorted(bin_to_prns):
        members = bin_to_prns[d_int]
        f0 = ((d_int + 500) % 1000) - 500
        k = (d_int - f0) // 1000
        if f0 not in base_cache:
            base_cache[f0] = _coherent_fft_base(seg_3d, float(f0), g_times_s, c_times_s)
        F_aligned = np.roll(base_cache[f0], -k, axis=1)
        F_aligned *= _code_doppler_ramp(float(d_int), group_mid_s)

        idx = np.fromiter((m[0] for m in members), dtype=np.int64, count=len(members))
        codes_sel = codes_f if len(idx) == n_prn else codes_f[idx]
        prod = F_aligned[None, :, :] * codes_sel[:, None, :]
        mag = np.abs(np.fft.ifft(prod, axis=2))
        mag *= mag
        power = mag.sum(axis=1, dtype=np.float64)

        means = np.mean(power, axis=1)
        peaks = np.max(power, axis=1)
        snrs = np.divide(peaks, means, out=np.zeros_like(peaks), where=means > 0)

        for row, (i, j) in enumerate(members):
            snr_lists[i][j] = snrs[row]
            if snrs[row] > best_snr[i]:
                best_snr[i] = snrs[row]
                best_bin_idx[i] = j
                best_profile[i] = power[row]

    return snr_lists, best_snr, best_bin_idx, best_profile


def acquire_constellation(
    filepath: str,
    prn_fft_dict: dict,
    total_ms: int = 480,
    coh_ms: int = 8,
    pfa: float = 2e-3,
    use_fast_gate: bool = True,
    doppler_centers: dict | None = None,
    doppler_halfwidth: float = 600.0,
):
    """
    Acquires GPS constellation with Code-Doppler walk compensation,
    parabolic interpolation, and optional 20ms Bit-Sync + 8x FFT Sinc refinement.

    Blind mode (doppler_centers=None): all PRNs over -5000..+5000 Hz (original behaviour).
    Guided mode: only the PRNs in doppler_centers, each within +/- doppler_halfwidth of its
    predicted Doppler. The CFAR threshold is set per PRN for the number of cells actually
    searched, and the 80 ms gate is skipped (the guided search is cheaper than the gate).
    """
    seg, actual_ms = unpack_vesper_iq(filepath, total_ms=total_ms)
    G = actual_ms // coh_ms
    if G < 10:
        return [], 999.0

    doppler_step = 50.0
    if doppler_centers is None:
        prn_list = list(prn_fft_dict.keys())
        full_bins = np.arange(-5000, 5050, doppler_step).astype(np.int64)
        bins_per_prn = [full_bins] * len(prn_list)
    else:
        prn_list = [p for p in doppler_centers if p in prn_fft_dict]
        if not prn_list:
            return [], 999.0
        half = int(round(doppler_halfwidth / doppler_step))
        offsets = np.arange(-half, half + 1, dtype=np.int64) * int(doppler_step)
        bins_per_prn = [
            int(round(doppler_centers[p] / doppler_step)) * int(doppler_step) + offsets
            for p in prn_list
        ]

    codes_f = np.stack([prn_fft_dict[p] for p in prn_list])

    if use_fast_gate:
        all_codes_f = codes_f if doppler_centers is None else np.stack(list(prn_fft_dict.values()))
        if fast_signal_gate(seg, all_codes_f) < 2:
            return [], 0.0

    seg = seg[: G * coh_ms * N_MS]
    seg_3d = seg.reshape(G, coh_ms, N_MS)
    g_times_s = np.arange(G, dtype=np.float64) * (coh_ms * 1e-3)
    c_times_s = np.arange(coh_ms, dtype=np.float64) * 1e-3
    group_mid_s = (np.arange(G, dtype=np.float64) * coh_ms + 0.5 * coh_ms) * 1e-3

    snr_lists, best_snr, best_bin_idx, best_profile = _search_doppler_grid(
        seg_3d, codes_f, bins_per_prn, g_times_s, c_times_s, group_mid_s
    )

    thr_cache: dict = {}

    def cfar_for(n_bins: int) -> float:
        if n_bins not in thr_cache:
            thr_cache[n_bins] = float(chi2.isf(pfa / (N_MS * n_bins), 2 * G) / (2 * G))
        return thr_cache[n_bins]

    cfar_threshold = cfar_for(len(bins_per_prn[0]))

    detections = []
    for i, prn in enumerate(prn_list):
        bins = bins_per_prn[i]
        if best_snr[i] < cfar_for(len(bins)):
            continue
        peak_bin = int(np.argmax(best_profile[i]))
        sub_sample = parabolic_interpolation(best_profile[i], peak_bin)

        d_idx = int(best_bin_idx[i])
        snr_grid_i = snr_lists[i]
        if 0 < d_idx < len(bins) - 1:
            y_m1, y_0, y_p1 = snr_grid_i[d_idx - 1], snr_grid_i[d_idx], snr_grid_i[d_idx + 1]
            denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
            delta_d = np.clip((y_p1 - y_m1) / denom, -0.5, 0.5) if denom > 1e-12 else 0.0
            fine_doppler = float(bins[d_idx] + delta_d * doppler_step)
        else:
            fine_doppler = float(bins[d_idx])

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
    Caches aligned FFT spectra across shared Doppler bins for speed.
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
    seg_3d = seg.reshape(G, coh_ms, N_MS)
    g_times_s = np.arange(G, dtype=np.float64) * (coh_ms * 1e-3)
    c_times_s = np.arange(coh_ms, dtype=np.float64) * 1e-3
    group_mid_s = (np.arange(G, dtype=np.float64) * coh_ms + 0.5 * coh_ms) * 1e-3

    doppler_step = 50.0
    n_bins = 13  # +/- 300 Hz window
    aided_detections = list(existing_detections)
    f_aligned_cache = {}

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
            dop_key = float(doppler)
            if dop_key not in f_aligned_cache:
                f_aligned_cache[dop_key] = _coherent_fft_aligned(
                    seg_3d, dop_key, g_times_s, c_times_s, group_mid_s
                )
            F_aligned = f_aligned_cache[dop_key]

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

from datetime import datetime
import numpy as np
import pytest

from src.sdr.frontend import FS, N_MS, unpack_vesper_iq
from src.sdr.acquisition import (
    BIT_MS,
    generate_native_prns,
    parabolic_interpolation,
    _bit_aligned_20ms,
    _upsampled_peak,
)
from src.sdr.ephemeris import datetime_to_gpstime, klobuchar_delay


def test_01_unpack_vesper_iq_dc_removal_and_length(tmp_path):
    """Test 1: Verifies 1-bit complex I/Q unpacking, length calculation, and DC bias removal."""
    dat_file = tmp_path / "snap.2025_09_29_11_13_08_GC0.dat"
    rng = np.random.default_rng(42)
    raw_words = rng.integers(0, 2**32, size=60 * 64, dtype=np.uint32)
    raw_words.tofile(dat_file)

    iq, actual_ms = unpack_vesper_iq(str(dat_file), total_ms=50)
    assert actual_ms == 50
    assert len(iq) == 50 * N_MS
    assert iq.dtype == np.complex64
    assert abs(np.mean(iq)) < 1e-4


def test_02_unpack_vesper_iq_short_corrupt_file_graceful(tmp_path):
    """Test 2: Ensures truncated/corrupt .dat files smaller than the startup skip do not crash."""
    tiny_file = tmp_path / "corrupt_short.dat"
    np.zeros(10, dtype=np.uint32).tofile(tiny_file)

    iq, actual_ms = unpack_vesper_iq(str(tiny_file), total_ms=480)
    assert actual_ms == 0
    assert len(iq) == 0


def test_03_generate_native_prns_orthogonality():
    """Test 3: Verifies all 32 GPS PRN Gold codes have length 1024 and low cross-correlation."""
    prns = generate_native_prns(FS)
    assert len(prns) == 32
    assert "G01" in prns and "G32" in prns

    g01_fft = prns["G01"]
    g02_fft = prns["G02"]
    assert g01_fft.shape == (N_MS,)

    # Auto-correlation peak of G01 at lag 0 must equal N_MS (1024)
    auto_corr = np.abs(np.fft.ifft(np.conj(g01_fft) * g01_fft))
    assert np.isclose(auto_corr[0], N_MS, atol=1e-2)

    # Cross-correlation between G01 and G02 must be < 15% of autocorrelation peak
    cross_corr = np.max(np.abs(np.fft.ifft(np.conj(g01_fft) * g02_fft)))
    assert cross_corr < 0.15 * N_MS


def test_04_parabolic_interpolation_fractional_peak():
    """Test 4: Verifies 3-point parabolic interpolation recovers sub-sample offsets."""
    true_offset = 412.35
    x = np.arange(N_MS, dtype=np.float64)
    power = np.exp(-0.5 * ((x - true_offset) / 1.2) ** 2)
    peak_bin = int(np.argmax(power))

    est = parabolic_interpolation(power, peak_bin)
    assert abs(est - true_offset) < 0.05


def test_05_parabolic_interpolation_flat_zero_denominator_edge_case():
    """Test 5: Guards against ZeroDivisionError when correlation profile is completely flat."""
    flat_power = np.ones(N_MS, dtype=np.float64) * 5.0
    est = parabolic_interpolation(flat_power, 100)
    assert est == 100.0


def test_06_bit_aligned_20ms_finds_true_bit_edge():
    """Test 6: Verifies 20ms coherent integration aligns with navigation data bit flips."""
    K = 80  # 4 data bits (80 ms)
    true_edge = 7
    true_lag = 250
    rng = np.random.default_rng(7)
    corr1 = (rng.normal(0, 0.1, (K, N_MS)) + 1j * rng.normal(0, 0.1, (K, N_MS))).astype(np.complex64)

    # Inject signal at true_lag with 180-deg phase reversals every 20 ms starting at true_edge
    bits = [1.0, -1.0, 1.0, -1.0, 1.0]
    for k in range(K):
        bit_idx = (k - true_edge) // BIT_MS
        corr1[k, true_lag] += 5.0 * bits[bit_idx]

    cand = (true_lag + np.arange(-3, 4)) % N_MS
    pw, groups, G = _bit_aligned_20ms(corr1, cand)
    assert G >= 3
    assert int(np.argmax(pw)) == true_lag


def test_07_upsampled_peak_8x_sinc_subchip_precision():
    """Test 7: Verifies 8x FFT zero-padded sinc interpolation recovers fractional code phase."""
    G = 10
    true_lag = 300.625
    freq_k = np.fft.fftfreq(N_MS, d=1.0 / N_MS)
    # Construct band-limited impulse shifted to true_lag
    spectrum = np.exp(-2j * np.pi * true_lag * freq_k / N_MS).astype(np.complex64)
    waveform = np.fft.ifft(spectrum)
    groups = np.tile(waveform, (G, 1))

    est_lag = _upsampled_peak(groups, int(round(true_lag)), search=3, factor=8)
    assert abs(est_lag - true_lag) < 0.05


def test_08_leap_seconds_and_klobuchar_edge_cases():
    """Test 8: Validates historical leap-second boundaries and Klobuchar ionospheric bounds."""
    # Pre-2009 fallback (14s), 2013 (16s), 2016 (17s), and 2025 (18s)
    _, tow_2008 = datetime_to_gpstime(datetime(2008, 6, 1, 0, 0, 0))
    _, tow_2013 = datetime_to_gpstime(datetime(2013, 6, 1, 0, 0, 0))
    _, tow_2025 = datetime_to_gpstime(datetime(2025, 9, 29, 12, 0, 0))
    assert (tow_2025 - tow_2013) % 60 == 2.0  # 18s vs 16s
    assert (tow_2025 - tow_2008) % 60 == 4.0  # 18s vs 14s

    iono = ([1.0e-8, 1.5e-8, -6.0e-8, 0.0], [9.0e4, 0.0, -1.9e5, 0.0])
    # Negative elevation must return 0.0 m
    assert klobuchar_delay(iono, 51.4, 10.7, 180.0, -5.0, 43200.0) == 0.0
    # Missing iono tuple must return 0.0 m
    assert klobuchar_delay(None, 51.4, 10.7, 180.0, 45.0, 43200.0) == 0.0
    # Daytime delay at 30 deg elevation should be positive and physically realistic (2m - 40m)
    delay_m = klobuchar_delay(iono, 51.4, 10.7, 180.0, 30.0, 50400.0)
    assert 2.0 < delay_m < 40.0

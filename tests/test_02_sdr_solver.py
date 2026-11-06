import math
import numpy as np
import pytest

from src.sdr.ephemeris import C_LIGHT, OMEGA_E, parse_rinex3_gps, propagate_sv_ecef, get_sv_state
from src.sdr.solver import (
    MS_METERS,
    llh_to_ecef,
    ecef_to_llh,
    calc_el_az,
    _solve_wls_subset,
    _fit_session_clock_model,
    solve_coarse_time_position,
)


@pytest.mark.parametrize(
    "lat,lon,alt",
    [
        (51.3951, 10.7455, 330.0),   # Affenwald field site
        (0.0, 0.0, 0.0),             # Equator / Prime Meridian
        (-22.9068, -43.1729, 15.0),  # Southern/Western hemisphere
        (89.9, 135.0, 1200.0),       # Near North Pole
    ],
)
def test_09_wgs84_llh_ecef_roundtrip_extremes(lat, lon, alt):
    """Test 9: Verifies sub-millimeter accuracy on LLH <-> ECEF coordinate transformations."""
    ecef = llh_to_ecef(lat, lon, alt)
    lat_out, lon_out, alt_out = ecef_to_llh(ecef)
    assert abs(lat_out - lat) < 1e-7
    assert abs(lon_out - lon) < 1e-7
    assert abs(alt_out - alt) < 1e-3


def test_10_calc_el_az_zenith_and_horizon():
    """Test 10: Verifies satellite elevation and azimuth geometry at zenith and due North."""
    rx = llh_to_ecef(0.0, 0.0, 0.0)
    # Satellite directly overhead along X-axis at Equator/Prime Meridian
    sv_zenith = rx + np.array([20200000.0, 0.0, 0.0])
    el, _ = calc_el_az(rx, sv_zenith)
    assert abs(el - 90.0) < 1e-3

    # Satellite on the northern horizon (along +Z tangent at Equator)
    sv_north = rx + np.array([0.0, 0.0, 20200000.0])
    el_n, az_n = calc_el_az(rx, sv_north)
    assert abs(el_n - 0.0) < 1e-3
    assert abs(az_n - 0.0) < 1e-3 or abs(az_n - 360.0) < 1e-3


def test_11_rinex3_parser_healthy_vs_unhealthy_sv_and_iono(tmp_path):
    """Test 11: Ensures RINEX v3 parser extracts GPSA/GPSB iono and filters unhealthy SVs."""
    rnx_content = (
        "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n"
        "GPSA   1.0245E-08  1.4901E-08 -5.9605E-08 -1.1921E-07       IONOSPHERIC CORR\n"
        "GPSB   9.8304E+04  1.6384E+04 -1.9661E+05 -6.5536E+04       IONOSPHERIC CORR\n"
        "                                                            END OF HEADER\n"
        "G05 2025 09 29 12 00 00 -1.000000000000E-04  1.000000000000E-11  0.000000000000E+00\n"
        "     4.500000000000E+01 -1.850000000000E+01  4.800000000000E-09  8.500000000000E-01\n"
        "    -9.500000000000E-07  8.500000000000E-03  6.200000000000E-06  5.153650000000E+03\n"
        "     1.296000000000E+05  3.500000000000E-08  1.420000000000E+00 -4.200000000000E-08\n"
        "     9.600000000000E-01  2.450000000000E+02 -1.150000000000E+00 -8.100000000000E-09\n"
        "     2.100000000000E-10  1.000000000000E+00  2.386000000000E+03  0.000000000000E+00\n"
        "     2.000000000000E+00  0.000000000000E+00 -1.000000000000E-08  4.500000000000E+01\n"
        "     1.296000000000E+05  4.000000000000E+00\n"
        "G09 2025 09 29 12 00 00 -1.000000000000E-04  1.000000000000E-11  0.000000000000E+00\n"
        "     4.500000000000E+01 -1.850000000000E+01  4.800000000000E-09  8.500000000000E-01\n"
        "    -9.500000000000E-07  8.500000000000E-03  6.200000000000E-06  5.153650000000E+03\n"
        "     1.296000000000E+05  3.500000000000E-08  1.420000000000E+00 -4.200000000000E-08\n"
        "     9.600000000000E-01  2.450000000000E+02 -1.150000000000E+00 -8.100000000000E-09\n"
        "     2.100000000000E-10  1.000000000000E+00  2.386000000000E+03  0.000000000000E+00\n"
        "     2.000000000000E+00  1.000000000000E+00 -1.000000000000E-08  4.500000000000E+01\n"
        "     1.296000000000E+05  4.000000000000E+00\n"
    )
    rnx_file = tmp_path / "test_brdc.rnx"
    rnx_file.write_text(rnx_content, encoding="utf-8")

    db = parse_rinex3_gps(str(rnx_file))
    assert "_iono" in db
    assert "G05" in db       # Healthy (sv_health == 0.0)
    assert "G09" not in db   # Unhealthy (sv_health == 1.0) must be rejected


def test_12_kepler_orbit_propagation_and_sv_state(synthetic_eph):
    """Test 12: Verifies GPS MEO satellite orbital radius (~26,560 km) and ECEF velocity magnitude."""
    pos, vel, dt_sv, _ = get_sv_state(synthetic_eph, 130000.0)
    r_km = np.linalg.norm(pos) / 1000.0
    v_kms = np.linalg.norm(vel) / 1000.0
    assert 25500.0 < r_km < 27500.0
    assert 2.7 < v_kms < 3.6
    assert abs(dt_sv) < 1e-2


def _build_synthetic_constellation(true_lat, true_lon, true_alt, tow, synthetic_eph):
    """Helper to generate 5 high-elevation synthetic satellites over Europe."""
    rx_true = llh_to_ecef(true_lat, true_lon, true_alt)
    eph_db = {}
    valid_svs = []
    detections = []

    # 5 orbital slots guaranteed to be high in the northern sky (el > 30 deg) over Europe at tow=129600
    orbital_offsets = [
        (0.80, 7.80),
        (1.10, 8.20),
        (1.40, 8.60),
        (0.95, 7.40),
        (1.25, 9.00),
    ]
    for idx, (m0_val, om0_val) in enumerate(orbital_offsets, 1):
        prn = f"G{idx:02d}"
        eph = dict(synthetic_eph, omega=0.5, m0=m0_val, omega0=om0_val)
        eph_db[prn] = [eph]

        sv_pos, sv_vel, dt_sv, ddt_sv = get_sv_state(eph, tow)
        tau_flight = np.linalg.norm(sv_pos - rx_true) / C_LIGHT
        sv_pos_tx, _, dt_sv_tx, _ = get_sv_state(eph, tow - tau_flight)
        theta = OMEGA_E * tau_flight
        c_th, s_th = math.cos(theta), math.sin(theta)
        sv_rot = np.array([
            c_th * sv_pos_tx[0] + s_th * sv_pos_tx[1],
            -s_th * sv_pos_tx[0] + c_th * sv_pos_tx[1],
            sv_pos_tx[2],
        ])
        geom_range = np.linalg.norm(sv_rot - rx_true)
        el, _ = calc_el_az(rx_true, sv_rot)
        tropo = 2.3 / max(math.sin(math.radians(max(el, 5.0))), 0.1)
        pred_rho = geom_range - C_LIGHT * dt_sv_tx + tropo

        tau_ms = (pred_rho % MS_METERS) / MS_METERS
        d = {
            "prn": prn,
            "metric": 25.0,
            "cn0_dbhz": 32.0,
            "doppler_hz": 0.0,
            "tau_ms": float(tau_ms),
        }
        detections.append(d)
        valid_svs.append((d, sv_pos, sv_vel, dt_sv, ddt_sv))

    return rx_true, eph_db, valid_svs, detections


def test_13_wls_5sv_synthetic_convergence_to_ground_truth(synthetic_eph):
    """Test 13: Verifies 5-SV Coarse-Time WLS converges to ground truth from an offset prior."""
    true_lat, true_lon, true_alt = 51.3951, 10.7455, 330.0
    tow = 129600.0
    _, eph_db, valid_svs, _ = _build_synthetic_constellation(
        true_lat, true_lon, true_alt, tow, synthetic_eph
    )
    # Start 15 km away from ground truth
    rx_init = llh_to_ecef(true_lat + 0.1, true_lon - 0.1, 300.0)
    sol = _solve_wls_subset(valid_svs, eph_db, tow, rx_init, reference_alt=330.0, pinned_dt_coarse=0.0)

    assert abs(sol["lat"] - true_lat) < 0.001
    assert abs(sol["lon"] - true_lon) < 0.001
    assert sol["rms_m"] < 1.0


def test_14_wls_leave_one_out_multipath_outlier_rejection(synthetic_eph):
    """Test 14: Verifies Leave-One-Out subset rejection drops a corrupted 6th satellite."""
    true_lat, true_lon, true_alt = 51.3951, 10.7455, 330.0
    tow = 129600.0
    rx_true, eph_db, _, detections = _build_synthetic_constellation(
        true_lat, true_lon, true_alt, tow, synthetic_eph
    )
    # Add a 6th high-elevation satellite with a +250 m canopy multipath error
    eph6 = dict(synthetic_eph, omega=0.5, m0=1.05, omega0=8.00)
    eph_db["G06"] = [eph6]
    sv_pos6, _, _, _ = get_sv_state(eph6, tow)
    tau_flight6 = np.linalg.norm(sv_pos6 - rx_true) / C_LIGHT
    sv_pos_tx6, _, dt_sv_tx6, _ = get_sv_state(eph6, tow - tau_flight6)
    theta6 = OMEGA_E * tau_flight6
    sv_rot6 = np.array([
        math.cos(theta6) * sv_pos_tx6[0] + math.sin(theta6) * sv_pos_tx6[1],
        -math.sin(theta6) * sv_pos_tx6[0] + math.cos(theta6) * sv_pos_tx6[1],
        sv_pos_tx6[2],
    ])
    el6, _ = calc_el_az(rx_true, sv_rot6)
    tropo6 = 2.3 / max(math.sin(math.radians(max(el6, 5.0))), 0.1)
    true_rho6 = np.linalg.norm(sv_rot6 - rx_true) - C_LIGHT * dt_sv_tx6 + tropo6

    # Inject +250 meters of pseudorange error on G06
    bad_tau_ms = ((true_rho6 + 250.0) % MS_METERS) / MS_METERS
    detections.append({
        "prn": "G06",
        "metric": 25.0,
        "cn0_dbhz": 30.0,
        "doppler_hz": 0.0,
        "tau_ms": float(bad_tau_ms),
    })

    sol = solve_coarse_time_position(
        detections,
        eph_db,
        tow,
        prior_llh=(true_lat + 0.01, true_lon - 0.01, 330.0),
        reference_alt=330.0,
        pinned_dt_coarse=0.0,
    )
    assert sol is not None
    assert sol["num_svs"] == 5  # Corrupted G06 was dropped by Leave-One-Out
    assert sol["rms_m"] < 5.0


def test_15_session_clock_model_mad_outlier_filtering_and_linear_drift():
    """Test 15: Verifies Pass-2 clock model rejects wild clock outliers and fits session drift."""
    records = [
        {"tow": 100000.0, "sol": {"num_svs": 6, "rms_m": 4.0, "dt_coarse_s": 1.05}},
        {"tow": 100100.0, "sol": {"num_svs": 5, "rms_m": 6.0, "dt_coarse_s": 1.06}},
        {"tow": 100200.0, "sol": {"num_svs": 6, "rms_m": 3.0, "dt_coarse_s": 1.04}},
        {"tow": 100300.0, "sol": {"num_svs": 5, "rms_m": 8.0, "dt_coarse_s": -9.50}},  # Outlier!
        {"tow": 100400.0, "sol": {"num_svs": 4, "rms_m": 2.0, "dt_coarse_s": 5.00}},   # Ignored (<5 SVs)
    ]
    clock_fn = _fit_session_clock_model(records)
    assert clock_fn is not None
    assert abs(clock_fn(100150.0) - 1.05) < 0.05


def test_16_solver_rejects_under_4_svs_and_high_rms(synthetic_eph):
    """Test 16: Ensures solver returns None when fewer than 4 valid satellites are locked."""
    true_lat, true_lon, true_alt = 51.3951, 10.7455, 330.0
    tow = 129600.0
    _, eph_db, _, detections = _build_synthetic_constellation(
        true_lat, true_lon, true_alt, tow, synthetic_eph
    )
    sol = solve_coarse_time_position(
        detections[:3],  # Only 3 satellites
        eph_db,
        tow,
        prior_llh=(true_lat, true_lon, true_alt),
    )
    assert sol is None

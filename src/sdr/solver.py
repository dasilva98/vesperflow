import os
import re
import glob
import math
import logging
from datetime import datetime
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm

from src.core.constants import REGION_PRESETS
from src.core.menu import get_all_regions
from src.sdr.acquisition import (
    F_L1, generate_native_prns, acquire_constellation, acquire_ephemeris_aided, USE_CODE_PHASE_AIDING
)
from src.sdr.ephemeris import (
    C_LIGHT, OMEGA_E, datetime_to_gpstime, fetch_rinex_ephemeris,
    parse_rinex3_gps, select_ephemeris, get_sv_state, klobuchar_delay
)

logger = logging.getLogger("vesperflow")

LAMBDA_L1 = C_LIGHT / F_L1
MS_METERS = C_LIGHT * 1e-3
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# --- FEATURE TOGGLES (Set any to False to revert to original single-pass behavior) ---
USE_KLOBUCHAR_IONO = True
USE_SUBSET_OUTLIER_REJECT = True
USE_TWO_PASS_CLOCK_MODEL = True
WRITE_DIAGNOSTICS_CSV = True


def llh_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_llh(ecef: np.ndarray):
    x, y, z = ecef
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    alt = 0.0
    for _ in range(6):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
        alt = p / max(math.cos(lat), 1e-12) - N
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * (N / (N + alt))))
    return math.degrees(lat), math.degrees(lon), alt


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    return np.array([
        [-slon,          clon,         0.0],
        [-slat * clon, -slat * slon,  clat],
        [ clat * clon,  clat * slon,  slat],
    ])


def calc_el_az(rx_ecef: np.ndarray, sv_ecef: np.ndarray):
    lat, lon, _ = ecef_to_llh(rx_ecef)
    R = ecef_to_enu_matrix(lat, lon)
    diff = sv_ecef - rx_ecef
    dist = np.linalg.norm(diff)
    enu = R @ (diff / dist)
    el = math.degrees(math.asin(np.clip(enu[2], -1.0, 1.0)))
    az = math.degrees(math.atan2(enu[0], enu[1])) % 360.0
    return el, az


def estimate_initial_location_from_doppler(
    detections: list, eph_db: dict, tow: float, region="global", prior_llh=None
):
    sv_states = []
    for d in detections:
        if d["prn"] not in eph_db:
            continue
        eph = select_ephemeris(eph_db[d["prn"]], tow)
        pos, vel, dt_sv, ddt_sv = get_sv_state(eph, tow)
        sv_states.append((d, pos, vel, dt_sv, ddt_sv))

    if prior_llh is not None:
        return llh_to_ecef(*prior_llh), sv_states

    if isinstance(region, str):
        all_presets = get_all_regions()
        key = region.strip().lower().replace(" ", "_")
        lat_min, lat_max, lon_min, lon_max, step = all_presets.get(key, all_presets["global"])
    elif isinstance(region, (tuple, list)) and len(region) == 4:
        lat_min, lat_max, lon_min, lon_max = region
        step = max(0.25, min((lat_max - lat_min), (lon_max - lon_min)) / 20.0)
    else:
        lat_min, lat_max, lon_min, lon_max, step = REGION_PRESETS["global"]

    if (lat_max - lat_min) <= 1.0 and (lon_max - lon_min) <= 1.0:
        center_lat = (lat_min + lat_max) / 2.0
        center_lon = (lon_min + lon_max) / 2.0
        return llh_to_ecef(center_lat, center_lon, 250.0), sv_states

    core_svs = sv_states[: min(5, len(sv_states))]
    best_score = 1e18
    best_ll = ((lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0)

    lats = np.arange(lat_min, lat_max + 0.01, step)
    lons = np.arange(lon_min, lon_max + 0.01, step)

    for lat in lats:
        for lon in lons:
            rx = llh_to_ecef(lat, lon, 250.0)
            valid = True
            rr_diffs, weights = [], []
            for d, pos, vel, _, ddt_sv in core_svs[:4]:
                diff = pos - rx
                dist = np.linalg.norm(diff)
                if np.dot(diff / dist, rx / np.linalg.norm(rx)) < 0.05:
                    valid = False
                    break
                u = diff / dist
                sagnac_rate = (OMEGA_E / C_LIGHT) * (pos[0] * vel[1] - pos[1] * vel[0])
                pred_rr = np.dot(u, vel) + sagnac_rate - C_LIGHT * ddt_sv
                meas_rr = -LAMBDA_L1 * d["doppler_hz"]
                rr_diffs.append(meas_rr - pred_rr)
                weights.append(d["metric"])
            if not valid:
                continue
            w = np.array(weights) / np.sum(weights)
            r = np.array(rr_diffs)
            clock_drift = np.sum(w * r)
            cost = np.sum(w * (r - clock_drift) ** 2)
            if cost < best_score:
                best_score = cost
                best_ll = (lat, lon)

    def doppler_res(params):
        lat, lon, c_drift = params
        rx = llh_to_ecef(lat, lon, 250.0)
        res = []
        for d, pos, vel, _, ddt_sv in core_svs[:4]:
            diff = pos - rx
            dist = np.linalg.norm(diff)
            u = diff / dist
            sagnac_rate = (OMEGA_E / C_LIGHT) * (pos[0] * vel[1] - pos[1] * vel[0])
            pred_rr = np.dot(u, vel) + sagnac_rate - C_LIGHT * ddt_sv + c_drift
            meas_rr = -LAMBDA_L1 * d["doppler_hz"]
            res.append(math.sqrt(d["metric"]) * (meas_rr - pred_rr))
        return np.array(res)

    margin = max(step * 2.5, 0.5)
    lat_lo = max(lat_min, best_ll[0] - margin)
    lat_hi = min(lat_max, best_ll[0] + margin)
    lon_lo = max(lon_min, best_ll[1] - margin)
    lon_hi = min(lon_max, best_ll[1] + margin)

    opt = least_squares(
        doppler_res,
        x0=[np.clip(best_ll[0], lat_lo, lat_hi), np.clip(best_ll[1], lon_lo, lon_hi), 0.0],
        bounds=([lat_lo, lon_lo, -5000.0], [lat_hi, lon_hi, 5000.0]),
    )
    return llh_to_ecef(opt.x[0], opt.x[1], 250.0), sv_states


def get_visible_constellation_predictions(
    rx_approx: np.ndarray,
    detections: list,
    eph_db: dict,
    tow: float,
    prelim_sol: dict | None = None,
):
    """
    Predicts Doppler shifts (and fractional 1ms code phases if prelim_sol is given)
    for all visible satellites (el >= 5 deg).
    """
    freq_offsets = []
    for d in detections[:4]:
        if d["prn"] not in eph_db:
            continue
        eph = select_ephemeris(eph_db[d["prn"]], tow)
        pos, vel, _, ddt_sv = get_sv_state(eph, tow)
        diff = pos - rx_approx
        dist = np.linalg.norm(diff)
        u = diff / dist
        sagnac_rate = (OMEGA_E / C_LIGHT) * (pos[0] * vel[1] - pos[1] * vel[0])
        geom_dop = -(np.dot(u, vel) + sagnac_rate - C_LIGHT * ddt_sv) / LAMBDA_L1
        freq_offsets.append(d["doppler_hz"] - geom_dop)

    rx_freq_bias_hz = float(np.median(freq_offsets)) if freq_offsets else 0.0
    iono = eph_db.get("_iono") if USE_KLOBUCHAR_IONO else None

    visible_dopplers = {}
    predicted_code_phases = {}

    rx_use = prelim_sol["rx_ecef"] if prelim_sol is not None else rx_approx
    t_use = tow + (prelim_sol["dt_coarse_s"] if prelim_sol is not None else 0.0)
    lat_u, lon_u, _ = ecef_to_llh(rx_use)

    for prn, eph_list in eph_db.items():
        if prn.startswith("_"):
            continue
        eph = select_ephemeris(eph_list, t_use)
        pos, vel, _, ddt_sv = get_sv_state(eph, t_use)
        el, az = calc_el_az(rx_use, pos)
        if el >= 5.0:
            diff = pos - rx_use
            dist = np.linalg.norm(diff)
            u = diff / dist
            sagnac_rate = (OMEGA_E / C_LIGHT) * (pos[0] * vel[1] - pos[1] * vel[0])
            geom_dop = -(np.dot(u, vel) + sagnac_rate - C_LIGHT * ddt_sv) / LAMBDA_L1
            visible_dopplers[prn] = geom_dop + rx_freq_bias_hz

            if prelim_sol is not None and USE_CODE_PHASE_AIDING:
                tau_flight = dist / C_LIGHT
                sv_pos_tx, _, dt_sv_tx, _ = get_sv_state(eph, t_use - tau_flight)
                theta = OMEGA_E * tau_flight
                c_th, s_th = math.cos(theta), math.sin(theta)
                sv_rot = np.array([
                    c_th * sv_pos_tx[0] + s_th * sv_pos_tx[1],
                    -s_th * sv_pos_tx[0] + c_th * sv_pos_tx[1],
                    sv_pos_tx[2],
                ])
                geom_range = np.linalg.norm(sv_rot - rx_use)
                tropo = 2.3 / max(math.sin(math.radians(max(el, 5.0))), 0.1)
                iono_m = klobuchar_delay(iono, lat_u, lon_u, az, el, t_use)
                pred_rho = geom_range - C_LIGHT * dt_sv_tx + tropo + iono_m

                pred_frac_m = (pred_rho + prelim_sol["ref_offset_m"] + prelim_sol["c_dtrx_m"]) % MS_METERS
                predicted_code_phases[prn] = (prelim_sol["sign"] * (pred_frac_m / MS_METERS)) % 1.0

    return visible_dopplers, predicted_code_phases


def _solve_wls_subset(
    valid_svs: list,
    eph_db: dict,
    tow: float,
    rx_init: np.ndarray,
    reference_alt: float,
    pinned_dt_coarse: float | None = None,
    pinned_dt_sigma: float = 0.05,
):
    """
    Core WLS solver for a given subset of satellites.
    Pre-computes 2nd-order satellite orbital states and atmospheric delays outside
    the least-squares loop for ~400x faster vectorized residual evaluation.
    """
    lat0, lon0, _ = ecef_to_llh(rx_init)
    rx_init = llh_to_ecef(lat0, lon0, reference_alt)
    R_enu = ecef_to_enu_matrix(lat0, lon0)
    iono = eph_db.get("_iono") if USE_KLOBUCHAR_IONO else None

    n_sv = len(valid_svs)
    pos0_arr = np.empty((n_sv, 3), dtype=np.float64)
    vel0_arr = np.empty((n_sv, 3), dtype=np.float64)
    acc0_arr = np.empty((n_sv, 3), dtype=np.float64)
    dt0_arr = np.empty(n_sv, dtype=np.float64)
    ddt0_arr = np.empty(n_sv, dtype=np.float64)
    atmo_arr = np.empty(n_sv, dtype=np.float64)
    weights = np.empty(n_sv, dtype=np.float64)
    tau_arr = np.empty(n_sv, dtype=np.float64)

    for i, (d, _, _, _, _) in enumerate(valid_svs):
        eph = select_ephemeris(eph_db[d["prn"]], tow)
        pos0, vel0, dt0, ddt0 = get_sv_state(eph, tow)
        _, vel_p, _, _ = get_sv_state(eph, tow + 0.5)
        _, vel_m, _, _ = get_sv_state(eph, tow - 0.5)

        pos0_arr[i] = pos0
        vel0_arr[i] = vel0
        acc0_arr[i] = vel_p - vel_m
        dt0_arr[i] = dt0
        ddt0_arr[i] = ddt0

        el0, az0 = calc_el_az(rx_init, pos0)
        tropo0 = 2.3 / max(math.sin(math.radians(max(el0, 5.0))), 0.1)
        iono0 = klobuchar_delay(iono, lat0, lon0, az0, el0, tow)
        atmo_arr[i] = tropo0 + iono0
        weights[i] = math.sqrt(d["metric"]) * math.sin(math.radians(max(el0, 10.0)))
        tau_arr[i] = d["tau_ms"]

    weights /= np.mean(weights)
    alt_sigma = 15.0 if n_sv >= 6 else 7.0
    R_earth = 6371000.0
    best_overall = None

    for sign in (+1.0, -1.0):
        meas_fracs = ((sign * tau_arr) % 1.0) * MS_METERS

        def eval_residuals(x, dt_coarse):
            d_up = x[2]
            # 2nd-order Earth curvature correction keeps ellipsoid height exact without iterative ecef_to_llh
            d_up_eff = d_up - (x[0] * x[0] + x[1] * x[1]) / (2.0 * R_earth)
            rx = rx_init + R_enu.T @ np.array([x[0], x[1], d_up_eff])
            c_dtrx = x[3]

            # Propagate satellite positions to t_eval = tow + dt_coarse for flight-time estimate
            sv_pos = pos0_arr + vel0_arr * dt_coarse + 0.5 * acc0_arr * (dt_coarse * dt_coarse)
            tau_flight = np.linalg.norm(sv_pos - rx, axis=1) / C_LIGHT

            # Propagate to transmission time t_tx = tow + (dt_coarse - tau_flight)
            dt_tx = dt_coarse - tau_flight
            sv_pos_tx = pos0_arr + vel0_arr * dt_tx[:, None] + 0.5 * acc0_arr * (dt_tx * dt_tx)[:, None]
            dt_sv_tx = dt0_arr + ddt0_arr * dt_tx

            # Earth rotation (Sagnac) during signal flight time
            theta = OMEGA_E * tau_flight
            c_th, s_th = np.cos(theta), np.sin(theta)
            sv_rot_x = c_th * sv_pos_tx[:, 0] + s_th * sv_pos_tx[:, 1]
            sv_rot_y = -s_th * sv_pos_tx[:, 0] + c_th * sv_pos_tx[:, 1]
            sv_rot_z = sv_pos_tx[:, 2]

            dx = sv_rot_x - rx[0]
            dy = sv_rot_y - rx[1]
            dz = sv_rot_z - rx[2]
            geom_range = np.sqrt(dx * dx + dy * dy + dz * dz)

            pred_ranges = geom_range - C_LIGHT * dt_sv_tx + atmo_arr
            ref_offset = (meas_fracs[0] - pred_ranges[0]) % MS_METERS
            diffs = (meas_fracs - pred_ranges - ref_offset - c_dtrx + 0.5 * MS_METERS) % MS_METERS - 0.5 * MS_METERS

            res = np.empty(n_sv + 2, dtype=np.float64)
            res[:n_sv] = weights * diffs
            res[n_sv] = d_up / alt_sigma
            res[n_sv + 1] = (
                (dt_coarse - pinned_dt_coarse) / pinned_dt_sigma
                if pinned_dt_coarse is not None
                else dt_coarse * 3.0
            )
            return res

        dt_grid = [pinned_dt_coarse] if pinned_dt_coarse is not None else np.linspace(-2.0, 2.0, 17)
        for dt_c in dt_grid:
            opt = least_squares(
                eval_residuals,
                x0=np.zeros(4),
                args=(dt_c,),
                loss="soft_l1",
                f_scale=30.0,
                bounds=([-150000.0, -150000.0, -300.0, -150000.0],
                        [ 150000.0,  150000.0,  300.0,  150000.0]),
                xtol=1e-5,
            )
            if best_overall is None or opt.cost < best_overall[0]:
                best_overall = (opt.cost, sign, dt_c, opt.x)

    assert best_overall is not None
    _, best_sign, best_dtc, best_x = best_overall
    sign = best_sign
    meas_fracs = ((sign * tau_arr) % 1.0) * MS_METERS

    def joint_res(p):
        return eval_residuals(p[:4], p[4])

    dt_lo = (pinned_dt_coarse - 0.25) if pinned_dt_coarse is not None else -10.0
    dt_hi = (pinned_dt_coarse + 0.25) if pinned_dt_coarse is not None else  10.0

    opt_final = least_squares(
        joint_res,
        x0=np.append(best_x, best_dtc),
        loss="soft_l1",
        f_scale=30.0,
        bounds=([-150000.0, -150000.0, -300.0, -150000.0, dt_lo],
                [ 150000.0,  150000.0,  300.0,  150000.0, dt_hi]),
        xtol=1e-8,
    )

    # Exact final evaluation with full satellite details
    x_fin = opt_final.x[:4]
    dt_fin = float(opt_final.x[4])
    rx_flat = rx_init + R_enu.T @ np.array([x_fin[0], x_fin[1], 0.0])
    lat_f, lon_f, _ = ecef_to_llh(rx_flat)
    alt_f = reference_alt + float(x_fin[2])
    rx_final = llh_to_ecef(lat_f, lon_f, alt_f)
    t_eval = tow + dt_fin

    pred_ranges, details = [], []
    for i, (d, _, _, _, _) in enumerate(valid_svs):
        eph = select_ephemeris(eph_db[d["prn"]], t_eval)
        sv_pos, _, _, _ = get_sv_state(eph, t_eval)
        tau_flight = np.linalg.norm(sv_pos - rx_final) / C_LIGHT
        sv_pos_tx, _, dt_sv_tx, _ = get_sv_state(eph, t_eval - tau_flight)
        theta = OMEGA_E * tau_flight
        c_th, s_th = math.cos(theta), math.sin(theta)
        sv_rot = np.array([
            c_th * sv_pos_tx[0] + s_th * sv_pos_tx[1],
            -s_th * sv_pos_tx[0] + c_th * sv_pos_tx[1],
            sv_pos_tx[2],
        ])
        geom_range = np.linalg.norm(sv_rot - rx_final)
        el, az = calc_el_az(rx_final, sv_rot)
        tropo = 2.3 / max(math.sin(math.radians(max(el, 5.0))), 0.1)
        iono_m = klobuchar_delay(iono, lat_f, lon_f, az, el, t_eval)
        pred_rho = geom_range - C_LIGHT * dt_sv_tx + tropo + iono_m
        pred_ranges.append(pred_rho)
        details.append((d, el, az, pred_rho, float(meas_fracs[i])))

    ref_offset = (meas_fracs[0] - pred_ranges[0]) % MS_METERS
    raw_res_m = [
        float((meas_fracs[i] - pred_ranges[i] - ref_offset - x_fin[3] + 0.5 * MS_METERS) % MS_METERS - 0.5 * MS_METERS)
        for i in range(n_sv)
    ]
    rms_m = float(np.sqrt(np.mean(np.square(raw_res_m))))

    return {
        "lat": lat_f,
        "lon": lon_f,
        "alt": alt_f,
        "rx_ecef": rx_final,
        "dt_coarse_s": dt_fin,
        "c_dtrx_m": float(x_fin[3]),
        "ref_offset_m": float(ref_offset),
        "sign": float(sign),
        "rms_m": rms_m,
        "num_svs": n_sv,
        "residuals_m": raw_res_m,
        "sv_details": details,
    }


def solve_coarse_time_position(
    detections: list,
    eph_db: dict,
    tow: float,
    region="global",
    prior_llh=None,
    reference_alt: float = 250.0,
    filepath: str | None = None,
    prn_dict: dict | None = None,
    pinned_dt_coarse: float | None = None,
):
    rx_init, sv_states_all = estimate_initial_location_from_doppler(
        detections, eph_db, tow, region=region, prior_llh=prior_llh
    )

    valid_svs = [
        item for item in sv_states_all
        if calc_el_az(rx_init, item[1])[0] >= 5.0
    ]

    # Preliminary fix (if >= 4 blind SVs) to predict code phases for Stage 3 Assisted Pass
    prelim_sol = None
    if filepath is not None and prn_dict is not None and len(valid_svs) >= 4:
        prelim_sol = _solve_wls_subset(valid_svs, eph_db, tow, rx_init, reference_alt, pinned_dt_coarse)
        if prelim_sol["rms_m"] > 100.0:
            prelim_sol = None

    visible_dopplers, pred_code_phases = get_visible_constellation_predictions(
        rx_init, detections, eph_db, tow, prelim_sol=prelim_sol
    )
    svs_in_view = len(visible_dopplers)

    # Smart Skip: only run single-threaded Ephemeris-Aided pass if we don't already have a clean >= 6 SV 3D fix
    need_aided = (
        filepath is not None
        and prn_dict is not None
        and len(detections) < 8
        and not (prelim_sol is not None and len(valid_svs) >= 6 and prelim_sol["rms_m"] < 18.0)
    )

    prev_det_count = len(detections)
    if need_aided:
        detections = acquire_ephemeris_aided(
            filepath,
            prn_dict,
            detections,
            visible_dopplers,
            predicted_code_phases=pred_code_phases,
            total_ms=480,
            coh_ms=8,
        )
        if len(detections) > prev_det_count:
            sv_states_all = []
            for d in detections:
                if d["prn"] in eph_db:
                    eph = select_ephemeris(eph_db[d["prn"]], tow)
                    pos, vel, dt_sv, ddt_sv = get_sv_state(eph, tow)
                    sv_states_all.append((d, pos, vel, dt_sv, ddt_sv))

            valid_svs = []
            for item in sv_states_all:
                d, pos, vel, dt_sv, ddt_sv = item
                el, az = calc_el_az(rx_init, pos)
                if el >= 5.0:
                    d["el_deg"] = el
                    d["az_deg"] = az
                    valid_svs.append(item)

    if len(valid_svs) < 4:
        return None

    # Reuse prelim_sol if no new satellites were added by Ephemeris-Aided acquisition
    if prelim_sol is not None and len(detections) == prev_det_count:
        best_sol = prelim_sol
    else:
        best_sol = _solve_wls_subset(valid_svs, eph_db, tow, rx_init, reference_alt, pinned_dt_coarse)

    # Leave-One-Out Subset Outlier Rejection (sorted by worst residual first)
    if USE_SUBSET_OUTLIER_REJECT and len(valid_svs) >= 5 and best_sol["rms_m"] > 20.0:
        order = np.argsort(np.abs(best_sol["residuals_m"]))[::-1]
        for drop_idx in order:
            sub = [sv for idx, sv in enumerate(valid_svs) if idx != drop_idx]
            sub_sol = _solve_wls_subset(sub, eph_db, tow, rx_init, reference_alt, pinned_dt_coarse)
            if sub_sol["rms_m"] < min(25.0, 0.55 * best_sol["rms_m"]):
                best_sol = sub_sol

    if best_sol["rms_m"] > 120.0 or not (-90.0 <= best_sol["lat"] <= 90.0):
        return None

    details = best_sol["sv_details"]
    H = []
    for _, el, az, _, _ in details:
        el_r, az_r = math.radians(el), math.radians(az)
        H.append([-math.cos(el_r) * math.sin(az_r), -math.cos(el_r) * math.cos(az_r), -math.sin(el_r), 1.0])
    H.append([0.0, 0.0, 1.0, 0.0])
    H = np.array(H)
    try:
        Q = np.linalg.inv(H.T @ H)
        hdop = float(math.sqrt(Q[0, 0] + Q[1, 1]))
        vdop = float(math.sqrt(Q[2, 2]))
        pdop = float(math.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2]))
    except np.linalg.LinAlgError:
        hdop, vdop, pdop = 9.9, 0.9, 9.9

    lat0, lon0, _ = ecef_to_llh(rx_init)
    return {
        **best_sol,
        "detections": detections,
        "hdop": hdop,
        "vdop": vdop,
        "pdop": pdop,
        "svs_in_view": max(svs_in_view, best_sol["num_svs"]),
        "snr_avg": float(np.mean([d["cn0_dbhz"] for d, *_ in details])),
        "max_cn0": max(d["cn0_dbhz"] for d, *_ in details),
        "init_llh": (lat0, lon0, reference_alt),
    }


def _fit_session_clock_model(pass1_records: list):
    """
    Fits a robust median or linear clock-drift model to Pass-1 fixes with >= 5 SVs.
    """
    pts = [
        (rec["tow"], rec["sol"]["dt_coarse_s"])
        for rec in pass1_records
        if rec["sol"] is not None and rec["sol"]["num_svs"] >= 5 and rec["sol"]["rms_m"] < 40.0
    ]
    if len(pts) < 2:
        return None

    t = np.array([p[0] for p in pts])
    d = np.array([p[1] for p in pts])
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) * 1.4826
    keep = np.abs(d - med) <= max(2.5 * mad, 0.15)
    t, d = t[keep], d[keep]

    if len(t) >= 6 and (t.max() - t.min()) > 600.0:
        tm = float(t.mean())
        A = np.vstack([np.ones_like(t), t - tm]).T
        coef = np.linalg.lstsq(A, d, rcond=None)[0]
        return lambda tow_val, c=coef, t_m=tm: float(c[0] + c[1] * (tow_val - t_m))

    final_med = float(np.median(d))
    return lambda tow_val, m=final_med: m


def _worker_acquire(filepath: str, prn_dict: dict):
    """Parallel worker for Stage 1 + Stage 2 signal acquisition."""
    fname = os.path.basename(filepath)
    m = re.search(r"snap\.(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})", fname)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    dt = datetime(y, mo, d, h, mi, s)
    detections, _ = acquire_constellation(filepath, prn_dict, total_ms=480, coh_ms=8)
    return (filepath, fname, dt, detections)


def solve_session_directory(
    dat_folder: str,
    output_dir: str,
    region="europe",
    ephemeris_dir: str = "./data/ephemeris",
    bar_position: int = 0,
    master_pbar=None,
    max_files: int | None = None,
) -> tuple:
    """
    Processes .dat snapshots in dat_folder in parallel, optionally runs Pass-2
    session clock pinning, and writes 'Track-sdr.csv' (plus 'Track-sdr-diagnostics.csv').
    """
    dat_files = sorted(glob.glob(os.path.join(dat_folder, "snap.*.dat")))
    if max_files is not None:
        dat_files = dat_files[:max_files]

    if not dat_files:
        return False, f"No snap.*.dat files found in {dat_folder}"

    os.makedirs(output_dir, exist_ok=True)
    prn_dict = generate_native_prns()
    session_name = os.path.basename(dat_folder)

    # Load optional sub-second header offsets if subseconds.csv was written by gps_parser.py
    subsec_map = {}
    subsec_path = os.path.join(dat_folder, "subseconds.csv")
    if os.path.exists(subsec_path):
        try:
            with open(subsec_path, "r", encoding="utf-8") as sf:
                for line in sf:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        subsec_map[parts[0]] = float(parts[1])
        except Exception:
            pass

    # 1. Parallel SDR Signal Acquisition
    all_scanned = []
    acquired_snapshots = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_worker_acquire, path, prn_dict): path
            for path in dat_files
        }
        with tqdm(
            total=len(dat_files),
            desc=f"📡 SDR Acquire ({session_name})",
            unit="snap",
            position=bar_position,
            leave=False,
            mininterval=0.1,
            dynamic_ncols=True,
        ) as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    all_scanned.append(res)
                    if len(res[3]) >= 3:
                        acquired_snapshots.append(res)
                        if master_pbar is not None:
                            master_pbar.total += 1
                pbar.set_postfix(locks=len(acquired_snapshots), refresh=False)
                pbar.update(1)
                if master_pbar is not None:
                    master_pbar.update(1)

    all_scanned.sort(key=lambda x: x[2])
    acquired_snapshots.sort(key=lambda x: x[2])

    if not acquired_snapshots:
        return False, "No snapshots contained >= 3 satellite locks (indoor/noisy session)."

    # 2. Pre-load RINEX ephemeris for each unique UTC date in the session
    eph_cache = {}
    for _, _, dt, _ in acquired_snapshots:
        day_key = (dt.year, dt.timetuple().tm_yday)
        if day_key not in eph_cache:
            rnx_path = fetch_rinex_ephemeris(dt, cache_dir=ephemeris_dir)
            eph_cache[day_key] = parse_rinex3_gps(rnx_path)

    # 3. Pass 1: Solve Navigation Fixes + Assisted Acquisition
    pass1_records: list[dict[str, Any]] = []
    active_region = region
    active_alt = 320.0

    with tqdm(
        acquired_snapshots,
        desc=f"🧭 GNSS Solver ({session_name})",
        unit="fix",
        position=bar_position,
        leave=False,
        mininterval=0.1,
        dynamic_ncols=True,
    ) as solve_bar:
        for filepath, fname, dt, detections in solve_bar:
            day_key = (dt.year, dt.timetuple().tm_yday)
            eph_db = eph_cache[day_key]
            _, tow = datetime_to_gpstime(dt)
            tow += subsec_map.get(fname, 0.0)

            sol = solve_coarse_time_position(
                detections,
                eph_db,
                tow,
                region=active_region,
                reference_alt=active_alt,
                filepath=filepath,
                prn_dict=prn_dict,
            )
            if sol is not None:
                if sol["num_svs"] >= 5 and sol["rms_m"] < 50.0:
                    # Keep bounding box <= 0.7 deg so subsequent snapshots skip Doppler grid search
                    active_region = (
                        sol["lat"] - 0.35, sol["lat"] + 0.35,
                        sol["lon"] - 0.35, sol["lon"] + 0.35,
                    )
                active_alt = 0.75 * active_alt + 0.25 * sol["alt"]

            pass1_records.append({
                "filepath": filepath,
                "fname": fname,
                "dt": dt,
                "tow": tow,
                "eph_db": eph_db,
                "detections": sol["detections"] if sol is not None else detections,
                "sol": sol,
                "region": active_region,
                "ref_alt": active_alt,
            })
            solve_bar.set_postfix(
                fixes=sum(1 for r in pass1_records if r["sol"] is not None),
                refresh=False,
            )
            if master_pbar is not None:
                master_pbar.update(1)

    # 4. Optional Pass 2: Re-solve with Session Clock Model Pinned (no extra FFT cost)
    if USE_TWO_PASS_CLOCK_MODEL:
        clock_fn = _fit_session_clock_model(pass1_records)
        if clock_fn is not None:
            for rec in pass1_records:
                if len(rec["detections"]) >= 4:
                    pinned_dt = clock_fn(rec["tow"])
                    sol2 = solve_coarse_time_position(
                        rec["detections"],
                        rec["eph_db"],
                        rec["tow"],
                        region=rec["region"],
                        reference_alt=rec["ref_alt"],
                        filepath=None,  # Uses cached detections from Pass 1
                        prn_dict=None,
                        pinned_dt_coarse=pinned_dt,
                    )
                    if sol2 is not None:
                        rec["sol"] = sol2

    # 5. Build Output Rows & Optional Diagnostics CSV
    rows = []
    solved_by_name = {rec["fname"]: rec for rec in pass1_records}
    diag_rows = []

    for idx, (_, fname, dt, blind_dets) in enumerate(all_scanned, 1):
        rec = solved_by_name.get(fname)
        sol = rec["sol"] if rec is not None else None
        if sol is not None:
            rows.append({
                "Index": idx,
                "Status": "Valid",
                "Date": dt.strftime("%Y-%m-%d"),
                "UTC": dt.strftime("%H:%M:%S"),
                "Latitude [DD]": f"{sol['lat']:.6f}",
                "Longitude [DD]": f"{sol['lon']:.6f}",
                "Altitude [m]": f"{sol['alt']:.1f}",
                "PDOP": f"{sol['pdop']:.1f}",
                "HDOP": f"{sol['hdop']:.1f}",
                "VDOP": f"{sol['vdop']:.1f}",
                "SVs in Use": sol["num_svs"],
                "SVs in View": sol["svs_in_view"],
                "SNR Avg [dB]": f"{sol['snr_avg']:.1f}",
                "Pseudorange RMS [m]": f"{sol['rms_m']:.1f}",
                "Clock Offset [s]": f"{sol['dt_coarse_s']:+.3f}",
                "Snapshot": fname,
            })
            diag_rows.append({
                "Index": idx,
                "Snapshot": fname,
                "Status": "FIX",
                "SVs_Used": sol["num_svs"],
                "PRNs": " ".join(d["prn"] for d, *_ in sol["sv_details"]),
                "RMS_m": f"{sol['rms_m']:.1f}",
                "Clock_Offset_s": f"{sol['dt_coarse_s']:+.3f}",
            })
        else:
            n_cand = len(rec["detections"]) if rec is not None else len(blind_dets)
            reason = f"Only {n_cand} SVs acquired (<4)" if n_cand < 4 else "Solver rejected (high RMS)"
            diag_rows.append({
                "Index": idx,
                "Snapshot": fname,
                "Status": reason,
                "SVs_Used": n_cand,
                "PRNs": " ".join(d["prn"] for d in (rec["detections"] if rec else blind_dets)),
                "RMS_m": "",
                "Clock_Offset_s": "",
            })

    if WRITE_DIAGNOSTICS_CSV and diag_rows:
        diag_csv = os.path.join(output_dir, "Track-sdr-diagnostics.csv")
        pd.DataFrame(diag_rows).to_csv(diag_csv, index=False)

    if not rows:
        return False, "No valid position fixes converged."

    df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "Track-sdr.csv")
    df.to_csv(out_csv, index=False)

    logger.info(f"Native SDR Solver extracted {len(df)} position fixes -> {out_csv}")
    return True, f"SDR Solver extracted {len(df)} fixes"

import os
import re
import sys
import logging
from datetime import datetime

from src.sdr.acquisition import generate_native_prns, acquire_constellation
from src.sdr.ephemeris import datetime_to_gpstime, fetch_rinex_ephemeris, parse_rinex3_gps
from src.sdr.solver import solve_coarse_time_position

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("GPS_Inspector")


def inspect_snapshot(filepath: str, region: str = "europe"):
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return

    fname = os.path.basename(filepath)
    m = re.search(r"snap\.(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})", fname)
    if not m:
        logger.error(f"Cannot parse timestamp from filename: {fname}")
        return

    dt = datetime(*map(int, m.groups()))
    week, tow = datetime_to_gpstime(dt)

    prn_dict = generate_native_prns()
    detections, thresh = acquire_constellation(
        filepath, prn_dict, total_ms=350, coh_ms=5, use_fast_gate=False
    )

    logger.info("=" * 90)
    logger.info(f" SNAPSHOT: {fname} | UTC: {dt} | GPS Week: {week}, TOW: {tow:.1f}s")
    logger.info("=" * 90)
    logger.info(f"CFAR Threshold: {thresh:.2f} | Locked Satellites: {len(detections)}")

    if len(detections) < 4:
        logger.warning("Insufficient satellites (< 4) for a 3D position fix.")
        for d in detections:
            logger.info(f"  {d['prn']} | Metric: {d['metric']:.2f} | C/N0: {d['cn0_dbhz']:.1f} dB-Hz | Doppler: {d['doppler_hz']:.0f} Hz")
        return

    rnx_path = fetch_rinex_ephemeris(dt)
    eph_db = parse_rinex3_gps(rnx_path)
    sol = solve_coarse_time_position(detections, eph_db, tow, region=region)

    if not sol:
        logger.warning("Solver failed to converge on a valid fix.")
        return

    logger.info(f"Doppler Coarse Init : Lat {sol['init_llh'][0]:.4f}°, Lon {sol['init_llh'][1]:.4f}°")
    logger.info("-" * 90)
    logger.info(f"{'SV':<5} | {'METRIC':<7} | {'C/N0':<10} | {'DOPPLER':<9} | {'EL / AZ':<15} | {'CODE PHASE':<12} | {'RESIDUAL'}")
    logger.info("-" * 90)
    for (d, el, az, _, _), res_m in zip(sol["sv_details"], sol["residuals_m"]):
        logger.info(
            f"{d['prn']:<5} | {d['metric']:06.2f}  | {d['cn0_dbhz']:04.1f} dB-Hz | "
            f"{d['doppler_hz']:>6.0f} Hz | {el:04.1f}° / {az:05.1f}° | "
            f"{d['code_phase_chips']:>6.2f} chips | {res_m:>+7.1f} m"
        )
    logger.info("-" * 90)
    logger.info(f"📍 POSITION FIX : {sol['lat']:.6f}°, {sol['lon']:.6f}° (Alt: {sol['alt']:.1f} m)")
    logger.info(f"⏱️  CLOCK OFFSET : {sol['dt_coarse_s']:+.3f} s | Pseudorange RMS: {sol['rms_m']:.1f} m")
    logger.info(f"🌍 GOOGLE MAPS  : https://www.google.com/maps?q={sol['lat']:.6f},{sol['lon']:.6f}\n")


if __name__ == "__main__":
    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "./data/processed/gps/snapshots/2025_09_29-530E503C/snap.2025_09_29_13_14_52_GC0.dat"
    )
    inspect_snapshot(target, region="europe")

import os
import re
import gzip
import math
import logging
import urllib.request
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger("vesperflow")

C_LIGHT = 299792458.0
MU_EARTH = 3.986005e14
OMEGA_E = 7.2921151467e-5


# Historical GPS-UTC leap seconds table
_LEAP_TABLE = [
    (datetime(2017, 1, 1, tzinfo=timezone.utc), 18.0),
    (datetime(2015, 7, 1, tzinfo=timezone.utc), 17.0),
    (datetime(2012, 7, 1, tzinfo=timezone.utc), 16.0),
    (datetime(2009, 1, 1, tzinfo=timezone.utc), 15.0),
]


def datetime_to_gpstime(dt: datetime):
    """Converts UTC datetime to (GPS Week, Time of Week in seconds)."""
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
    dt_utc = dt.replace(tzinfo=timezone.utc)
    leap_s = next((ls for start, ls in _LEAP_TABLE if dt_utc >= start), 14.0)
    total_sec = (dt_utc - gps_epoch).total_seconds() + leap_s
    week = int(total_sec // 604800)
    tow = total_sec - week * 604800
    return week, tow


def fetch_rinex_ephemeris(dt: datetime, cache_dir: str = "./data/ephemeris") -> str:
    """Downloads daily broadcast ephemeris from BKG IGS archive if not cached."""
    os.makedirs(cache_dir, exist_ok=True)
    year = dt.year
    doy = dt.timetuple().tm_yday
    local_rnx = os.path.join(cache_dir, f"BRDC_{year}_{doy:03d}.rnx")

    if os.path.exists(local_rnx) and os.path.getsize(local_rnx) > 10000:
        return local_rnx

    urls = [
        f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/BRDC00WRD_R_{year}{doy:03d}0000_01D_MN.rnx.gz",
        f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz",
        f"https://igs.bkg.bund.de/root_ftp/EUREF/BRDC/{year}/{doy:03d}/BRDC00GOP_R_{year}{doy:03d}0000_01D_MN.rnx.gz",
    ]

    for url in urls:
        try:
            logger.info(f"Downloading IGS broadcast ephemeris ({year} DOY {doy:03d})...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                compressed = resp.read()
            raw_text = gzip.decompress(compressed).decode("utf-8", errors="ignore")
            with open(local_rnx, "w", encoding="utf-8") as f:
                f.write(raw_text)
            logger.info(f"Cached RINEX ephemeris at: {local_rnx}")
            return local_rnx
        except Exception as e:
            logger.debug(f"Ephemeris mirror failed ({e}), trying next...")

    raise RuntimeError(f"Could not download RINEX ephemeris for {year} DOY {doy:03d}.")


def _parse_floats(line: str):
    line = line.replace("D", "E").replace("d", "e")
    return [float(x) for x in re.findall(r"[+-]?\d+\.\d+[Ee][+-]\d+", line)]


def parse_rinex3_gps(rnx_path: str) -> dict:
    """Parses GPS (G01-G32) broadcast navigation records and Klobuchar iono coefficients."""
    with open(rnx_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    alpha, beta = None, None
    idx = 0
    while idx < len(lines) and "END OF HEADER" not in lines[idx]:
        ln = lines[idx]
        if "IONOSPHERIC CORR" in ln:
            if ln.startswith("GPSA"):
                alpha = _parse_floats(ln[5:60])
            elif ln.startswith("GPSB"):
                beta = _parse_floats(ln[5:60])
        elif "ION ALPHA" in ln:
            alpha = _parse_floats(ln[:60])
        elif "ION BETA" in ln:
            beta = _parse_floats(ln[:60])
        idx += 1
    idx += 1

    eph_db = {}
    if alpha and beta and len(alpha) == 4 and len(beta) == 4:
        eph_db["_iono"] = (alpha, beta)

    while idx + 7 < len(lines):
        line0 = lines[idx]
        if line0.startswith("G"):
            prn = line0[:3].strip()
            vals = _parse_floats(line0[23:])
            for k in range(1, 8):
                vals.extend(_parse_floats(lines[idx + k]))
            if len(vals) >= 29:
                rec = {
                    "af0": vals[0], "af1": vals[1], "af2": vals[2],
                    "iode": vals[3], "crs": vals[4], "deltan": vals[5], "m0": vals[6],
                    "cuc": vals[7], "e": vals[8], "cus": vals[9], "sqrtA": vals[10],
                    "toe": vals[11], "cic": vals[12], "omega0": vals[13], "cis": vals[14],
                    "i0": vals[15], "crc": vals[16], "omega": vals[17], "omegadot": vals[18],
                    "idot": vals[19], "week": vals[21], "sv_health": vals[24], "tgd": vals[25],
                }
                if rec["sv_health"] == 0.0:
                    eph_db.setdefault(prn, []).append(rec)
            idx += 8
        else:
            idx += 1
    return eph_db


def klobuchar_delay(iono, lat_deg: float, lon_deg: float, az_deg: float, el_deg: float, tow: float) -> float:
    """Computes Klobuchar L1 ionospheric group delay in meters."""
    if not iono or el_deg <= 0.0:
        return 0.0
    alpha, beta = iono
    el = math.radians(el_deg) / math.pi
    lat = math.radians(lat_deg) / math.pi
    lon = math.radians(lon_deg) / math.pi
    az_rad = math.radians(az_deg)

    psi = 0.0137 / (el + 0.11) - 0.022
    phi_i = max(min(lat + psi * math.cos(az_rad), 0.416), -0.416)
    lam_i = lon + psi * math.sin(az_rad) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)

    t = (43200.0 * lam_i + (tow % 86400.0)) % 86400.0
    f = 1.0 + 16.0 * (0.53 - el) ** 3
    amp = max(sum(alpha[n] * (phi_m ** n) for n in range(4)), 0.0)
    per = max(sum(beta[n] * (phi_m ** n) for n in range(4)), 72000.0)
    x = 2.0 * math.pi * (t - 50400.0) / per

    if abs(x) < 1.57:
        delay_s = f * (5e-9 + amp * (1.0 - (x ** 2) / 2.0 + (x ** 4) / 24.0))
    else:
        delay_s = f * 5e-9
    return delay_s * C_LIGHT


def select_ephemeris(eph_list: list, tow: float) -> dict:
    return min(eph_list, key=lambda r: abs((tow - r["toe"] + 302400) % 604800 - 302400))


def propagate_sv_ecef(eph: dict, tow: float):
    """Computes GPS satellite ECEF position (m) and clock bias (s) at GPS TOW."""
    A = eph["sqrtA"] ** 2
    n0 = math.sqrt(MU_EARTH / (A ** 3))
    tk = (tow - eph["toe"] + 302400) % 604800 - 302400

    n = n0 + eph["deltan"]
    Mk = eph["m0"] + n * tk

    Ek = Mk
    for _ in range(10):
        Ek = Mk + eph["e"] * math.sin(Ek)

    sin_E, cos_E = math.sin(Ek), math.cos(Ek)
    vk = math.atan2(math.sqrt(1.0 - eph["e"] ** 2) * sin_E, cos_E - eph["e"])
    phik = vk + eph["omega"]

    sin_2phi, cos_2phi = math.sin(2.0 * phik), math.cos(2.0 * phik)
    du = eph["cus"] * sin_2phi + eph["cuc"] * cos_2phi
    dr = eph["crs"] * sin_2phi + eph["crc"] * cos_2phi
    di = eph["cis"] * sin_2phi + eph["cic"] * cos_2phi

    uk = phik + du
    rk = A * (1.0 - eph["e"] * cos_E) + dr
    ik = eph["i0"] + di + eph["idot"] * tk

    xk_p = rk * math.cos(uk)
    yk_p = rk * math.sin(uk)
    omegak = eph["omega0"] + (eph["omegadot"] - OMEGA_E) * tk - OMEGA_E * eph["toe"]

    cos_O, sin_O = math.cos(omegak), math.sin(omegak)
    cos_i, sin_i = math.cos(ik), math.sin(ik)

    x = xk_p * cos_O - yk_p * cos_i * sin_O
    y = xk_p * sin_O + yk_p * cos_i * cos_O
    z = yk_p * sin_i

    dt_rel = -4.442807633e-10 * eph["e"] * eph["sqrtA"] * sin_E
    dt_sv = eph["af0"] + eph["af1"] * tk + eph["af2"] * (tk ** 2) + dt_rel - eph["tgd"]

    return np.array([x, y, z], dtype=np.float64), dt_sv


def get_sv_state(eph: dict, tow: float):
    pos, dt_sv = propagate_sv_ecef(eph, tow)
    pos_p, dt_p = propagate_sv_ecef(eph, tow + 0.5)
    pos_m, dt_m = propagate_sv_ecef(eph, tow - 0.5)
    vel = pos_p - pos_m
    ddt_sv = dt_p - dt_m
    return pos, vel, dt_sv, ddt_sv

"""
IMU BINARY FORMAT ANALYZER & DIAGNOSTIC TOOL
=============================================
Diagnostic tool for inspecting raw IMU .BIN files from Vesper Wildlife Tags.
Auto-detects packet format from the header bitmask, staying in sync with the
main parser at all times.

LOCATION:
---------
    src/tools/imu_inspector.py

USAGE (run from project root):
-------------------------------
1. Metadata Report (default):
   Prints header fields, bitmask decoded bit-by-bit, trailer (if present),
   and the auto-detected packet format configuration.
   $ python -m src.tools.imu_inspector ./data/raw/00M.BIN

2. Hex Dump (--hex):
   $ python -m src.tools.imu_inspector ./data/raw/00M.BIN --hex
   $ python -m src.tools.imu_inspector ./data/raw/00M.BIN --hex 400

3. Packet Size Scanner (--scan):
   $ python -m src.tools.imu_inspector ./data/raw/00M.BIN --scan

4. Decoded Packets (--data):
   $ python -m src.tools.imu_inspector ./data/raw/00M.BIN --data
   $ python -m src.tools.imu_inspector ./data/raw/00M.BIN --data 25

5. Hex Packets (--hexpackets):
   $ python -m src.tools.imu_inspector ./data/raw/00M.BIN --hexpackets
   $ python -m src.tools.imu_inspector ./data/raw/00M.BIN --hexpackets 50

6. Rhythm Analysis (--rhythm):
   Analyses the per-second packet count pattern across N sequential files
   (default N=3). Starting from the named file it finds the next N-1 files
   in the same folder by sequence index (e.g. 2M → 3M → 4M).
   Useful for understanding clock drift, the pkt_ts sec-field lag, and the
   packet-count pattern across second boundaries and file boundaries.
   Use --all to scan ALL files in the folder instead of just N.
   $ python -m src.tools.imu_inspector ./data/raw/2M.BIN --rhythm
   $ python -m src.tools.imu_inspector ./data/raw/2M.BIN --rhythm --n 5
   $ python -m src.tools.imu_inspector ./data/raw/0M.BIN --rhythm --all

ARGUMENTS SUMMARY:
----------
    file               Path to the .BIN file.
    --hex  [N]         Hex + ASCII dump of the first N bytes (default: 200).
    --scan             Packet size hypothesis scanner.
    --data [N]         Decode and print first N packets (default: 10).
    --hexpackets [N]   Structured hex dump of first N packets (default: 20).
    --rhythm [N]       Rhythm analysis across N sequential files (default: 3).
"""

import argparse
import os
import re
import struct
import sys
import math
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.binary_decoder import (
    TRAILER_SIZE,
    decode_binary_trailer,
    decode_binary_header,
)
from src.parsers.imu_parser import _build_dtype, _get_format_config

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
HEADER_SIZE = 144

KNOWN_BITS = {
    0: "Accelerometer active",
    1: "Gyroscope active",
    2: "Magnetometer active",
    3: "Extended format — 46B packets, adds temp + pressure fields",
    5: "Standard mode flag (observed in 0x27 — exact function undocumented)",
}

# Config0: observed values 0 or 1. Exact meaning undocumented.
# Earlier mutual-exclusivity hypothesis with bit 3 was not confirmed.

SCAN_ABS_LIMIT = 1e6
PACKET_SIZE_MIN = 32
PACKET_SIZE_MAX = 60


def _divider(char="=", width=72):
    return char * width


# ---------------------------------------------------------------------------
# MODE 1 — METADATA REPORT (default)
# ---------------------------------------------------------------------------
def print_metadata(filepath):
    print(f"\n{_divider()}")
    print("  METADATA REPORT")
    print(f"  {filepath}")
    print(_divider())

    meta = decode_binary_header(filepath, header_size=HEADER_SIZE)
    if not meta:
        print("ERROR: Could not decode header.")
        return

    fsize = os.path.getsize(filepath)

    # --- Basic fields ---
    print(f"\n{'File:':<20} {os.path.basename(filepath)}  ({fsize:,} bytes)")
    print(f"{'Device ID:':<20} {meta['DeviceID']}")
    print(f"{'Sensor Name:':<20} {meta['Sensor']}")
    print(f"{'FWID:':<20} {meta['FWID']}  (0x{meta['FWID']:04X})")
    print(f"{'HWID:':<20} {meta['HWID']}  (0x{meta['HWID']:04X})")
    print(f"{'Sample Rate:':<20} {meta['SampleRate']} Hz")
    print(f"{'WinLen / WinRate:':<20} {meta['WinLen']} / {meta['WinRate']}")
    print(
        f"{'Config 0-3:':<20} {meta['Config0']}, {meta['Config1']}, "
        f"{meta['Config2']}, {meta['Config3']}"
    )

    b136 = meta.get("Header_B136")
    b136_str = f"0x{b136:02X} ({b136})" if isinstance(b136, int) else "N/A"
    print(f"{'Header_B136:':<20} {b136_str}")
    print(
        f"{'  B136 note:':<20} Per-session byte, stable across all files in a session."
    )
    print(f"{'':20} Mirrored at trailer byte 8 — confirms preservation end-to-end.")
    print(f"{'':20} Leading theory: config preset or schedule slot index.")

    print(
        f"{'Start Time (BCD):':<20} {meta['Start_Time'].strftime('%d/%m/%Y %H:%M:%S')}"
    )
    with open(filepath, "rb") as f:
            f.seek(140)
            ss_res, ss_val = struct.unpack("<HH", f.read(4))
    print(f"{'Header subsec:':<20} ss_res={ss_res}, ss_val={ss_val}")

    # --- Bitmask decoded ---
    bitmask = meta.get("Bitmask", 0)
    print(f"\n{_divider('-')}")
    print(
        f"  BITMASK:  {bitmask} decimal  |  0x{bitmask:02X} hex  |  {bitmask:08b} binary"
    )
    print(_divider("-"))
    for bit in range(8):
        state = "ON " if (bitmask >> bit) & 1 else "OFF"
        meaning = KNOWN_BITS.get(bit, "Unknown")
        marker = " <--" if (bitmask >> bit) & 1 else ""
        print(f"  Bit {bit} [{state}]  {meaning}{marker}")

    fmt_tag = bitmask & 0x0F
    print(f"\n  pkt_ts format tag (bitmask & 0x0F): 0x{fmt_tag:02X}")
    print(f"  -> Every packet begins with [0x55][0x{fmt_tag:02X}] as a self-describing")
    print(f"     sync + format identifier.")

    # --- Auto-detected format config ---
    fmt = _get_format_config(bitmask)
    print(f"\n{_divider('-')}")
    print("  AUTO-DETECTED FORMAT CONFIG  (matches main parser)")
    print(_divider("-"))
    print(f"  {'Packet size:':<22} {fmt['packet_size']} bytes")
    print(
        f"  {'Gyro scale:':<22} /{fmt['gyro_scale']:.0f}  "
        f"({'output stays in mdps' if fmt['gyro_scale'] == 1.0 else 'mdps -> divide'})"
    )
    print(
        f"  {'Temp + Pressure:':<22} "
        f"{'YES — bytes 42-45 of each packet' if fmt['has_temp_pres'] else 'NO'}"
    )
    print(f"  {'pkt_ts (bytes 0-5):':<22} sync + fmt + min + sec + uint16 subsec ticks")

    # --- Trailer ---
    print(f"\n{_divider('-')}")
    print("  TRAILER (last 16 bytes)")
    print(_divider("-"))
    trailer = decode_binary_trailer(filepath)
    if trailer:
        print(f"  End Time:           {trailer['End_Time'].strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]}")
        print(f"  Trailer ss_res:      {trailer['End_ss_res']}")
        end_ms = 1000.0 * (trailer['End_ss_res'] - trailer['End_ss_val']) / (trailer['End_ss_res'] + 1.0)
        print(f"  Trailer ss_val:      {trailer['End_ss_val']}  ({end_ms:.2f}ms past second)")
        print(f"  Trailer B7:          0x{trailer['B7']:02X} — varies per file (counter or checksum?)")
        print(f"  Trailer B8:          0x{trailer['B8']:02X} — should mirror Header_B136 (0x{b136:02X})")
        if trailer["B8"] != b136:
            print(f"     <<< MISMATCH between Trailer B8 and Header B136!")
        else:
            print(f"     OK — B8 mirrors B136")

        # Compute session duration from header start to trailer end
        start_dt = meta["Start_Time"]
        end_dt = trailer["End_Time"]
        duration = (end_dt - start_dt).total_seconds()
        h_dur = int(duration // 3600)
        m_dur = int((duration % 3600) // 60)
        s_dur = int(duration % 60)
        print(
            f"  Session duration:   {h_dur}h {m_dur}m {s_dur}s  (from header BCD to trailer BCD)"
        )
    else:
        print("  No valid trailer — file may be truncated.")

    # --- Packet count ---
    payload = fsize - HEADER_SIZE - (TRAILER_SIZE if trailer else 0)
    n_packets = payload // fmt["packet_size"]
    remainder = payload % fmt["packet_size"]
    duration_s = n_packets / meta["SampleRate"] if meta["SampleRate"] > 0 else 0

    print(f"\n{_divider('-')}")
    print(
        f"  PACKET COUNT  ({fmt['packet_size']}B packets, trailer {'subtracted' if trailer else 'not present'})"
    )
    print(_divider("-"))
    print(f"  {'Payload bytes:':<22} {payload:,}")
    print(f"  {'Complete packets:':<22} {n_packets:,}")
    print(
        f"  {'Remainder bytes:':<22} {remainder}  "
        f"({'clean' if remainder == 0 else 'WARNING — not divisible'})"
    )
    print(
        f"  {'Implied duration:':<22} {int(duration_s // 3600)}h "
        f"{int((duration_s % 3600) // 60)}m {int(duration_s % 60)}s"
    )
    print(f"\n{_divider()}\n")


# ---------------------------------------------------------------------------
# MODE 2 — HEX DUMP (--hex)
# ---------------------------------------------------------------------------
def hex_inspector(filepath, limit=200):
    print(f"\n{_divider()}")
    print(f"  HEX INSPECTOR  —  First {limit} bytes")
    print(f"  {filepath}")
    print(_divider())
    print(f"\n{'OFFSET':<8} | {'HEX':^48} | ASCII")
    print("-" * 72)

    with open(filepath, "rb") as f:
        data = f.read(limit)

    for i in range(0, len(data), 16):
        chunk = data[i : i + 16]
        hex_str = " ".join(f"{b:02X}" for b in chunk)
        asc_str = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
        print(f"{i:<8} | {hex_str:<48} | {asc_str}")
        if i < HEADER_SIZE <= i + 16:
            print(
                f"{'-' * 8} | {'^ DATA PAYLOAD STARTS AT BYTE 144 ^':^48} | {'-' * 16}"
            )

    print(f"\n{_divider()}\n")


# ---------------------------------------------------------------------------
# MODE 3 — PACKET SIZE SCANNER (--scan)
# ---------------------------------------------------------------------------
def scan_packet_size(filepath):
    print(f"\n{_divider()}")
    print("  PACKET SIZE SCANNER")
    print(f"  {filepath}")
    print(_divider())

    # Subtract trailer if present so the scan doesn't see trailer bytes as junk
    trailer = decode_binary_trailer(filepath)
    file_size = os.path.getsize(filepath)
    payload_end = file_size - TRAILER_SIZE if trailer else file_size

    with open(filepath, "rb") as f:
        f.seek(HEADER_SIZE)
        payload = f.read(payload_end - HEADER_SIZE)

    total_bytes = len(payload)
    print(
        f"\n  Payload: {total_bytes:,} bytes (trailer {'excluded' if trailer else 'not present'})\n"
    )
    print(f"  Checking packet sizes {PACKET_SIZE_MIN} – {PACKET_SIZE_MAX} bytes.")
    print(
        f"  'Bad' = any of the 9 sensor float32s is NaN, Inf, or abs > {SCAN_ABS_LIMIT:.0e}\n"
    )
    print(
        f"  {'Size':>6} | {'Pkts':>7} | {'Rem':>4} | {'Bad':>7} | {'Bad %':>7} | {'Verdict'}"
    )
    print(f"  {'-' * 6}-+-{'-' * 7}-+-{'-' * 4}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 20}")

    results = []

    for psize in range(PACKET_SIZE_MIN, PACKET_SIZE_MAX + 1):
        n = total_bytes // psize
        remainder = total_bytes % psize
        if n == 0:
            continue

        check_n = min(500, n)
        bad = 0

        for i in range(check_n):
            chunk = payload[i * psize + 6 : i * psize + 42]
            try:
                vals = struct.unpack("<9f", chunk)
                if any(not (v == v) or abs(v) > SCAN_ABS_LIMIT for v in vals):
                    bad += 1
            except struct.error:
                bad += 1

        pct = bad / check_n * 100
        verdict = ""
        if pct == 0.0:
            verdict = "<<< CLEAN — strong candidate"
        elif pct < 2.0:
            verdict = "<<  likely candidate"
        elif pct < 10.0:
            verdict = "<   possible"

        results.append((psize, n, remainder, bad, pct, verdict))
        print(
            f"  {psize:>6} | {n:>7,} | {remainder:>4} | {bad:>7} | {pct:>6.1f}% | {verdict}"
        )

    best = [r for r in results if r[4] < 2.0]
    print(f"\n{_divider('-')}")
    if best:
        print(f"  Best candidate(s): {[r[0] for r in best]} bytes")
    else:
        print(
            f"  No clean packet size found in range {PACKET_SIZE_MIN}-{PACKET_SIZE_MAX}."
        )
    print(f"\n{_divider()}\n")


# ---------------------------------------------------------------------------
# MODE 4 — DECODED PACKETS (--data)
# ---------------------------------------------------------------------------
def decode_packets(filepath, n_packets=10):
    print(f"\n{_divider()}")
    print(f"  DECODED PACKETS  —  First {n_packets} packets")
    print(f"  {filepath}")
    print(_divider())

    meta = decode_binary_header(filepath, header_size=HEADER_SIZE)
    if not meta:
        print("ERROR: Could not decode header.")
        return

    bitmask = meta.get("Bitmask", 0)
    fmt = _get_format_config(bitmask)
    dt = _build_dtype(fmt)

    # Read body excluding trailer
    trailer = decode_binary_trailer(filepath)
    file_size = os.path.getsize(filepath)
    payload_end = file_size - TRAILER_SIZE if trailer else file_size
    n_in_file = (payload_end - HEADER_SIZE) // fmt["packet_size"]

    with open(filepath, "rb") as f:
        f.seek(HEADER_SIZE)
        raw = np.frombuffer(f.read(n_in_file * fmt["packet_size"]), dtype=dt)

    total_available = len(raw)
    if total_available == 0:
        print("  No data packets found.")
        return

    n_show = min(n_packets, total_available)
    subset = raw[:n_show]

    gyro_data = np.round(subset["gyro"] / fmt["gyro_scale"], 2)
    acc_data = np.round(subset["acc"], 3)
    mag_data = np.round(subset["mag"], 1)

    has_tp = fmt["has_temp_pres"]
    if has_tp:
        temp_data = np.round(subset["temp"].astype(float) * 0.01, 2)
        pres_data = np.round(subset["pres"].astype(float), 0)

    print(
        f"\n  Bitmask: 0x{bitmask:02X}  |  Packet size: {fmt['packet_size']}B  |  "
        f"Gyro: mdps  |  Temp+Pres: {'YES' if has_tp else 'NO'}"
    )
    print(f"  Showing {n_show} of {total_available:,} available packets.\n")

    h = f"  {'Pkt':>4} | {'Min':>3} {'Sec':>3} {'Sub':>5} | "
    h += f"{'Gyro X':>10} {'Gyro Y':>10} {'Gyro Z':>10} [mdps] | "
    h += f"{'Acc X':>9} {'Acc Y':>9} {'Acc Z':>9} [mg]   | "
    h += f"{'Mag X':>8} {'Mag Y':>8} {'Mag Z':>8} [mGauss]"
    if has_tp:
        h += f"  | {'Temp':>7} [C]  {'Pres':>7} [hPa]"
    print(h)
    print(f"  {'-' * (len(h) - 2)}")

    for i in range(n_show):
        gx, gy, gz = gyro_data[i]
        ax, ay, az = acc_data[i]
        mx, my, mz = mag_data[i]
        # Grab the raw 6-byte array for this specific row
        ts = subset["pkt_ts"][i]

        mn = ts[2]  # Minute is byte 2
        sc = ts[3]  # Second is byte 3
        # Combine byte 4 (low) and byte 5 (high) into a 16-bit integer
        sub = ts[4] | (ts[5] << 8)

        row = f"  {i:>4} | {mn:>3} {sc:>3} {sub:>5} | "
        row += f"{gx:>10.2f} {gy:>10.2f} {gz:>10.2f}       | "
        row += f"{ax:>9.3f} {ay:>9.3f} {az:>9.3f}        | "
        row += f"{mx:>8.1f} {my:>8.1f} {mz:>8.1f}"
        if has_tp:
            row += f"          | {temp_data[i]:>7.2f}        {pres_data[i]:>7.0f}"
        print(row)

    # --- Summary ---
    all_gyro = np.round(raw["gyro"] / fmt["gyro_scale"], 2)
    all_acc = np.round(raw["acc"], 3)
    all_mag = np.round(raw["mag"], 1)

    print(f"\n{_divider('-')}")
    print(f"  PER-SENSOR SUMMARY  (all {total_available:,} packets)")
    print(_divider("-"))

    def _sensor_summary(label, data, unit):
        flat = data.reshape(-1)
        valid = flat[np.isfinite(flat)]
        if len(valid) == 0:
            print(f"  {label:<8}  no finite values found")
            return
        print(
            f"  {label:<8} [{unit}]"
            f"  min={valid.min():>12.3f}"
            f"  max={valid.max():>12.3f}"
            f"  mean={valid.mean():>12.3f}"
            f"  NaN/Inf={len(flat) - len(valid)}"
        )

    _sensor_summary("Gyro", all_gyro, "mdps   ")
    _sensor_summary("Acc", all_acc, "mg     ")
    _sensor_summary("Mag", all_mag, "mGauss ")
    if has_tp:
        all_temp = np.round(raw["temp"].astype(float) * 0.01, 2)
        all_pres = np.round(raw["pres"].astype(float), 0)
        _sensor_summary("Temp", all_temp.reshape(-1, 1), "C      ")
        _sensor_summary("Pres", all_pres.reshape(-1, 1), "hPa    ")

    # --- pkt_ts summary ---
    # Grab the entire column of timestamps (Shape: N rows by 6 columns)
    all_ts = raw["pkt_ts"]

    mins = all_ts[:, 2]
    secs = all_ts[:, 3]
    subs = all_ts[:, 4].astype(int) | (all_ts[:, 5].astype(int) << 8)

    expected_sync1 = bitmask & 0x0F
    sync_ok = np.all(all_ts[:, 0] == 0x55) and np.all(all_ts[:, 1] == expected_sync1)

    print(f"\n{_divider('-')}")
    print("  PKT_TS SUMMARY")
    print(_divider("-"))
    print(f"  Minutes range:      {mins.min()} – {mins.max()}")
    print(f"  Seconds range:      {secs.min()} – {secs.max()}")
    print(f"  Subsec range:       {subs.min()} – {subs.max()}  (uint16 ticks, 0-{meta['ss_res']})")
    print(f"  Sync bytes [0x55, 0x{expected_sync1:02X}]: {'OK' if sync_ok else 'MISMATCH'}")

    if total_available > 1:
        # Simply check if the first 6-byte array is identical to the second 6-byte array
        dup = np.array_equal(all_ts[0], all_ts[1])
        print(f"  Startup duplicate:  {'YES - first packet duplicated (parser filters this)' if dup else 'NO'}")

    # --- Plausibility check ---
    print(f"\n{_divider('-')}")
    print("  UNIT PLAUSIBILITY CHECK")
    print(_divider("-"))

    gyro_flat = all_gyro.reshape(-1)
    gyro_valid = gyro_flat[np.isfinite(gyro_flat)]
    gyro_max = np.abs(gyro_valid).max() if len(gyro_valid) > 0 else 0

    if gyro_max > 5_000_000:
        print(f"  Gyro max abs = {gyro_max:.1f} — unexpectedly large even for mdps.")
    elif gyro_max > 2_000_000:
        print(
            f"  Gyro max abs = {gyro_max:.1f} — near sensor saturation (±2,000,000 mdps)."
        )
    else:
        print(f"  Gyro max abs = {gyro_max:.1f} — plausible for mdps.")

    acc_flat = all_acc.reshape(-1)
    acc_valid = acc_flat[np.isfinite(acc_flat)]
    acc_max = np.abs(acc_valid).max() if len(acc_valid) > 0 else 0

    if acc_max > 20000:
        print(f"  Acc  max abs = {acc_max:.1f} — exceeds ±16g sensor range.")
    else:
        print(f"  Acc  max abs = {acc_max:.1f} — plausible for mg.")

    print(f"\n{_divider()}\n")


# ---------------------------------------------------------------------------
# MODE 5 — HEX PACKETS (--hexpackets)
# ---------------------------------------------------------------------------
def hex_packets(filepath, n_packets=20):
    print(f"\n{_divider()}")
    print(f"  HEX PACKETS  —  First {n_packets} packets (payload only)")
    print(f"  {filepath}")
    print(_divider())

    meta = decode_binary_header(filepath, header_size=HEADER_SIZE)
    if not meta:
        print("ERROR: Could not decode header.")
        return

    bitmask = meta.get("Bitmask", 0)
    fmt = _get_format_config(bitmask)
    psize = fmt["packet_size"]
    has_tp = fmt["has_temp_pres"]

    trailer = decode_binary_trailer(filepath)
    file_size = os.path.getsize(filepath)
    payload_end = file_size - TRAILER_SIZE if trailer else file_size

    with open(filepath, "rb") as f:
        f.seek(HEADER_SIZE)
        payload = f.read(payload_end - HEADER_SIZE)

    total = len(payload) // psize
    n_show = min(n_packets, total)

    print(
        f"\n  Bitmask: 0x{bitmask:02X}  |  pkt_ts format tag: 0x{bitmask & 0x0F:02X}  |  "
        f"Packet size: {psize}B  |  Temp+Pres: {'YES' if has_tp else 'NO'}\n"
    )

    header = f"  {'PKT':>4} | {'PKT_TS (bytes 0-5)':<17} | {'GYRO (bytes 6-17)':<35} | {'ACC (bytes 18-29)':<35} | {'MAG (bytes 30-41)':<35}"
    if has_tp:
        header += f" | {'TEMP':<5} | {'PRES':<5}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for i in range(n_show):
        chunk = payload[i * psize : i * psize + psize]
        ts_hex = chunk[0:6].hex(" ").upper()
        gyro_hex = chunk[6:18].hex(" ").upper()
        acc_hex = chunk[18:30].hex(" ").upper()
        mag_hex = chunk[30:42].hex(" ").upper()

        row = f"  {i:>4} | {ts_hex} | {gyro_hex} | {acc_hex} | {mag_hex}"

        if has_tp:
            temp_hex = chunk[42:44].hex(" ").upper()
            pres_hex = chunk[44:46].hex(" ").upper()
            row += f" | {temp_hex} | {pres_hex}"

        print(row)

    print(f"\n  Showing {n_show} of {total:,} available packets.")
    print(f"\n{_divider()}\n")


# ---------------------------------------------------------------------------
# MODE 6 — RHYTHM ANALYSIS (--rhythm)
# ---------------------------------------------------------------------------
def _find_sequential_files(start_filepath, n_files, use_all=False):
    """
    Finds sequential IMU .BIN files in the same folder as start_filepath.
    Matches the pattern: <digits>M.BIN (e.g. 0M.BIN, 1M.BIN, 12M.BIN).
    Returns a sorted list of absolute paths starting from start_filepath.
    """
    folder = os.path.dirname(os.path.abspath(start_filepath))
    pattern = re.compile(r"^(\d+)M\.BIN$", re.IGNORECASE)

    candidates = {}
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            candidates[int(m.group(1))] = os.path.join(folder, fname)

    # Determine the start index
    start_m = pattern.match(os.path.basename(start_filepath))
    if not start_m:
        print(
            f"  Warning: '{os.path.basename(start_filepath)}' does not match NM.BIN pattern."
        )
        return [start_filepath]

    start_idx = int(start_m.group(1))
    sorted_indices = sorted(k for k in candidates if k >= start_idx)

    if use_all:
        return [candidates[i] for i in sorted_indices]
    else:
        return [candidates[i] for i in sorted_indices[:n_files] if i in candidates]


def _load_pkt_ts(filepath, fmt_config):
    """
    Reads all pkt_ts fields from a file's payload (excluding trailer).
    Returns numpy array of shape (N, 6), dtype uint8.
    """
    file_size = os.path.getsize(filepath)
    psize = fmt_config["packet_size"]
    trailer_ok = False

    # Check trailer
    with open(filepath, "rb") as f:
        f.seek(file_size - TRAILER_SIZE)
        trailer_sync = f.read(4)
    if trailer_sync == b"\xa5\x5a\x5a\xa5":
        trailer_ok = True
        payload_size = file_size - HEADER_SIZE - TRAILER_SIZE
    else:
        payload_size = file_size - HEADER_SIZE

    n_max = max(0, payload_size // psize)
    dt = _build_dtype(fmt_config)

    with open(filepath, "rb") as f:
        f.seek(HEADER_SIZE)
        raw = np.fromfile(f, dtype=dt, count=n_max)

    # Drop startup duplicate
    if len(raw) > 1 and np.array_equal(raw["pkt_ts"][0], raw["pkt_ts"][1]):
        raw = raw[1:]

    return raw["pkt_ts"].copy(), trailer_ok


def analyze_drift_rhythm(filepath, n_files=3, use_all=False):
    """
    Analyses the per-second packet count pattern across N sequential files.

    For each file it computes:
      - Tick delta stats (mean, std, values seen) → true sample rate
      - Per-(min,sec) group packet counts → apparent packets/second
      - Distribution of packet counts
      - Correlation of packet count with the last digit of the sec field

    At each file boundary it computes:
      - Trailer end time vs next header start time
      - Gap in ms between files
      - First packet tick of the new file vs trailer ss_val

    The sec field in pkt_ts is known to lag behind the tick wrap (by a variable
    number of packets — 1 to 20+ observed). This tool is specifically designed
    to quantify that lag and its effect on apparent packets-per-second counts.
    """
    files = _find_sequential_files(filepath, n_files, use_all=use_all)
    if not files:
        print("  No sequential files found.")
        return

    print(f"\n{_divider()}")
    print("  RHYTHM ANALYSIS")
    print(f"  Starting from: {os.path.basename(filepath)}")
    print(
        f"  Files to scan: {len(files)}"
        + (" (all in folder)" if use_all else f" (N={n_files})")
    )
    print(_divider())

    print(f"\n  Files found:")
    for i, fp in enumerate(files):
        size = os.path.getsize(fp)
        print(f"    {i + 1}. {os.path.basename(fp)}  ({size:,} bytes)")

    # -----------------------------------------------------------------------
    # Per-file analysis
    # -----------------------------------------------------------------------
    file_data = []  # store per-file results for boundary analysis

    for fp in files:
        meta = decode_binary_header(fp, header_size=HEADER_SIZE)
        if not meta:
            print(f"\n  ERROR: Could not decode header for {fp}")
            file_data.append(None)
            continue

        fmt = _get_format_config(meta.get("Bitmask", 0))
        pkt_ts, trailer_ok = _load_pkt_ts(fp, fmt)
        fname = os.path.basename(fp)
        n_total = len(pkt_ts)

        if n_total == 0:
            print(f"\n  {fname}: no packets found.")
            file_data.append(None)
            continue

        # --- Tick delta stats ---
        ticks = pkt_ts[:, 4].astype(np.int64) | (pkt_ts[:, 5].astype(np.int64) << 8)

        raw_deltas = np.diff(ticks.astype(np.int64))
        # Unwrap: large positive jump = wrap (counter reset upward)
        deltas = np.where(raw_deltas > 500, raw_deltas - 1024, raw_deltas)
        deltas = np.where(deltas < -500, deltas + 1024, deltas)
        # Tick counts DOWN, so deltas are negative — flip sign for readability
        deltas_ms = -deltas * 1000.0 / 1024.0

        # --- Per-(min,sec) group packet counts ---
        mins = pkt_ts[:, 2].astype(np.int64)
        secs = pkt_ts[:, 3].astype(np.int64)

        # Build ordered list of groups and counts
        group_counts = []
        group_keys = []
        seen_keys = {}
        for i in range(n_total):
            key = (int(mins[i]), int(secs[i]))
            if key not in seen_keys:
                seen_keys[key] = len(group_keys)
                group_keys.append(key)
                group_counts.append(0)
            group_counts[seen_keys[key]] += 1

        counts_arr = np.array(group_counts)

        # Exclude the first and last group (likely partial seconds at file edges)
        inner_counts = counts_arr[1:-1] if len(counts_arr) > 2 else counts_arr

        # Distribution of packet counts
        count_dist = Counter(group_counts)

        # Correlation with last digit of sec
        digit_groups = defaultdict(list)
        for (mn, sc), cnt in zip(group_keys, group_counts):
            digit_groups[sc % 10].append(cnt)

        # Store for boundary analysis
        trailer_data = None
        if trailer_ok:
            file_size = os.path.getsize(fp)
            with open(fp, "rb") as f:
                f.seek(file_size - TRAILER_SIZE)
                trailer_raw = f.read(TRAILER_SIZE)
            if len(trailer_raw) == TRAILER_SIZE:

                def bcd(b):
                    return (b // 16) * 10 + (b % 16)

                end_h = bcd(trailer_raw[4])
                end_m = bcd(trailer_raw[5])
                end_s = bcd(trailer_raw[6])
                end_mo = bcd(trailer_raw[9])
                end_da = bcd(trailer_raw[10])
                end_yr = 2000 + bcd(trailer_raw[11])
                ss_res = struct.unpack("<H", trailer_raw[12:14])[0]
                ss_val = struct.unpack("<H", trailer_raw[14:16])[0]
                end_ms = 1000.0 * (ss_res - ss_val) / (ss_res + 1.0)
                trailer_data = {
                    "h": end_h,
                    "m": end_m,
                    "s": end_s,
                    "ms": end_ms,
                    "ss_val": ss_val,
                    "mo": end_mo,
                    "da": end_da,
                    "yr": end_yr,
                }

        file_data.append(
            {
                "fp": fp,
                "fname": fname,
                "meta": meta,
                "ticks": ticks,
                "deltas_ms": deltas_ms,
                "group_keys": group_keys,
                "group_counts": group_counts,
                "counts_arr": counts_arr,
                "inner_counts": inner_counts,
                "count_dist": count_dist,
                "digit_groups": digit_groups,
                "n_total": n_total,
                "trailer_data": trailer_data,
            }
        )

        # ----------------------------------------------------------------
        # Print per-file section
        # ----------------------------------------------------------------
        print(f"\n{_divider('=')}")
        print(f"  FILE: {fname}  ({n_total:,} packets, {len(group_keys)} sec-groups)")
        print(_divider("="))

        # Tick delta stats
        print(f"\n  TICK DELTA STATS  (true sample cadence)")
        print(f"  {'Mean:':<14} {deltas_ms.mean():.4f} ms/packet")
        print(f"  {'Std:':<14} {deltas_ms.std():.4f} ms")
        print(f"  {'Min:':<14} {deltas_ms.min():.4f} ms")
        print(f"  {'Max:':<14} {deltas_ms.max():.4f} ms")
        print(
            f"  {'Values (ms):':<14} {sorted(set(round(d, 3) for d in deltas_ms.tolist()))}"
        )
        equiv_hz = 1000.0 / deltas_ms.mean() if deltas_ms.mean() > 0 else 0
        print(f"  {'Equiv. rate:':<14} {equiv_hz:.4f} Hz")

        # Per-(min,sec) group listing
        print(
            f"\n  PER-(MIN,SEC) GROUP PACKET COUNTS  "
            f"[first and last groups may be partial]"
        )
        print(f"  {'Group':<12} {'Count':>6}   {'Bar'}")
        print(f"  {'-' * 50}")
        for j, ((mn, sc), cnt) in enumerate(zip(group_keys, group_counts)):
            tag = ""
            if j == 0:
                tag = " ← first (may be partial)"
            elif j == len(group_keys) - 1:
                tag = " ← last  (may be partial)"
            bar = "█" * (cnt // 2)
            print(f"  {mn:02d}:{sc:02d} (d={sc % 10})  {cnt:>5}   {bar}{tag}")

        # Distribution of counts (inner groups only)
        print(f"\n  PACKET COUNT DISTRIBUTION  (inner groups only, excl. first/last)")
        print(f"  {'Count/sec':>10} | {'Occurrences':>12} | {'%':>7} | Bar")
        print(f"  {'-' * 55}")
        if len(inner_counts) > 0:
            inner_dist = Counter(inner_counts.tolist())
            for val in sorted(inner_dist.keys()):
                occ = inner_dist[val]
                pct = 100.0 * occ / len(inner_counts)
                bar = "█" * int(pct / 2)
                print(f"  {val:>10} | {occ:>12} | {pct:>6.1f}% | {bar}")
            print(
                f"\n  Inner groups: n={len(inner_counts)}"
                f"  mean={inner_counts.mean():.2f}"
                f"  median={np.median(inner_counts):.1f}"
                f"  std={inner_counts.std():.2f}"
                f"  min={inner_counts.min()}"
                f"  max={inner_counts.max()}"
            )
        else:
            print(f"  Not enough groups for inner analysis.")

        # Correlation with sec last digit
        print(f"\n  CORRELATION: PACKET COUNT vs LAST DIGIT OF SEC FIELD")
        print(
            f"  {'Digit':>6} | {'Groups':>7} | {'Avg':>7} | {'Count distribution (value×occurrences)'}"
        )
        print(f"  {'-' * 80}")
        for d in range(10):
            if d in digit_groups:
                vals = digit_groups[d]
                avg = np.mean(vals)
                dist = Counter(vals)
                dist_str = "  ".join(f"{k}×{v}" for k, v in sorted(dist.items(), reverse=True))
                print(f"  {d:>6} | {len(vals):>7} | {avg:>7.1f} | {dist_str}")
            else:
                print(f"  {d:>6} | {'—':>7} | {'—':>7} |")

    # -----------------------------------------------------------------------
    # Cross-file combined distribution
    # -----------------------------------------------------------------------
    all_inner = []
    for fd in file_data:
        if fd and len(fd["inner_counts"]) > 0:
            all_inner.extend(fd["inner_counts"].tolist())

    if all_inner:
        all_inner_arr = np.array(all_inner)
        print(f"\n{_divider('=')}")
        print(
            f"  COMBINED DISTRIBUTION  (all files, inner groups only, n={len(all_inner)})"
        )
        print(_divider("="))
        print(f"\n  {'Count/sec':>10} | {'Occurrences':>12} | {'%':>7} | Bar")
        print(f"  {'-' * 55}")
        combined_dist = Counter(all_inner)
        for val in sorted(combined_dist.keys()):
            occ = combined_dist[val]
            pct = 100.0 * occ / len(all_inner)
            bar = "█" * int(pct / 2)
            print(f"  {val:>10} | {occ:>12} | {pct:>6.1f}% | {bar}")
        print(
            f"\n  Overall: mean={all_inner_arr.mean():.2f}"
            f"  median={np.median(all_inner_arr):.1f}"
            f"  std={all_inner_arr.std():.2f}"
            f"  min={all_inner_arr.min()}"
            f"  max={all_inner_arr.max()}"
        )

        # Combined digit correlation
        combined_digit = defaultdict(list)
        for fd in file_data:
            if fd:
                for d, vals in fd["digit_groups"].items():
                    combined_digit[d].extend(vals)

        print(f"\n  COMBINED DIGIT CORRELATION  (all files)")
        print(
            f"  {'Digit':>6} | {'Groups':>7} | {'Avg':>7} | {'Counts seen  ([values]×occurrences)'}"
        )
        print(f"  {'-' * 80}")
        for d in range(10):
            if d in combined_digit:
                vals = combined_digit[d]
                avg = np.mean(vals)
                dist = Counter(vals)
                # Group packet-count values by their occurrence count
                # e.g. 34×36 and [54,63,78]×1 instead of 54×1  63×1  78×1
                occ_groups = defaultdict(list)
                for count_val, occ in sorted(dist.items()):
                    occ_groups[occ].append(count_val)
                parts = []
                for occ in sorted(occ_groups.keys(), reverse=True):
                    group_vals = occ_groups[occ]

                    if occ > n_files:
                        if len(group_vals) == 1:
                            parts.append(f"{group_vals[0]}×{occ}")
                        else:
                            parts.append(f"[{','.join(str(v) for v in group_vals)}]×{occ}")
                dist_str = " ".join(parts)
                print(f"  {d:>6} | {len(vals):>7} | {avg:>7.1f} | {dist_str}")
            else:
                print(f"  {d:>6} | {'—':>7} | {'—':>7} |")

    # -----------------------------------------------------------------------
    # File boundary analysis
    # -----------------------------------------------------------------------
    valid_data = [fd for fd in file_data if fd is not None]
    if len(valid_data) > 1:
        print(f"\n{_divider('=')}")
        print(f"  FILE BOUNDARY ANALYSIS")
        print(_divider("="))

        for i in range(len(valid_data) - 1):
            curr = valid_data[i]
            nxt = valid_data[i + 1]
            print(f"\n  {curr['fname']} → {nxt['fname']}")
            print(f"  {'-' * 60}")

            # Trailer of curr
            if curr["trailer_data"]:
                fd = curr["trailer_data"]
                print(
                    f"  Trailer end:     {fd['h']:02d}:{fd['m']:02d}:{fd['s']:02d}.{fd['ms']:03.0f}ms"
                    f"  (ss_val={fd['ss_val']})"
                )
            else:
                print(f"  Trailer:         MISSING — {curr['fname']} may be truncated")

            # Header of nxt
            nm = nxt["meta"]
            st = nm["Start_Time"]
            ss_val_hdr = None
            with open(nxt["fp"], "rb") as f:
                f.seek(142)
                ss_val_hdr = struct.unpack("<H", f.read(2))[0]
            ss_res_hdr = None
            with open(nxt["fp"], "rb") as f:
                f.seek(140)
                ss_res_hdr = struct.unpack("<H", f.read(2))[0]
            hdr_ms = (
                1000.0 * (ss_res_hdr - ss_val_hdr) / (ss_res_hdr + 1.0)
                if ss_res_hdr
                else 0
            )
            print(
                f"  Header start:   {st.hour:02d}:{st.minute:02d}:{st.second:02d}.{hdr_ms:03.0f}ms"
                f"  (ss_val={ss_val_hdr})"
            )

            # First packet tick of nxt
            if len(nxt["ticks"]) > 0:
                first_tick = int(nxt["ticks"][0])
                first_ms = 1000.0 * (1023 - first_tick) / 1024.0
                first_min = int(nxt["group_keys"][0][0])
                first_sec = int(nxt["group_keys"][0][1])
                print(
                    f"  1st pkt tick:   min={first_min:02d} sec={first_sec:02d}"
                    f"  tick={first_tick}  → {first_ms:.2f}ms within sec"
                )
                print(
                    f"  Header ss_val matches 1st pkt tick: "
                    f"{'YES ✓' if ss_val_hdr == first_tick else f'NO  (diff={ss_val_hdr - first_tick} ticks)'}"
                )

            # Gap between files
            if curr["trailer_data"]:
                fd = curr["trailer_data"]
                end_abs_ms = (fd["h"] * 3600 + fd["m"] * 60 + fd["s"]) * 1000 + fd["ms"]
                start_abs_ms = (
                    st.hour * 3600 + st.minute * 60 + st.second
                ) * 1000 + hdr_ms
                gap_ms = start_abs_ms - end_abs_ms
                n_samples_gap = gap_ms / (
                    deltas_ms.mean() if deltas_ms.mean() > 0 else 20.0
                )
                print(
                    f"  Gap between files: {gap_ms:+.2f}ms  (~{n_samples_gap:.1f} samples)"
                )

    print(f"\n{_divider()}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="IMU Binary Format Inspector - VesperFlow diagnostic tool.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("file", help="Path to the .BIN file to inspect.")
    parser.add_argument(
        "--hex",
        nargs="?",
        const=200,
        type=int,
        metavar="N",
        help="Hex + ASCII dump of the first N bytes (default: 200).",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help=f"Packet size scanner (sizes {PACKET_SIZE_MIN}-{PACKET_SIZE_MAX}).",
    )
    parser.add_argument(
        "--data",
        nargs="?",
        const=10,
        type=int,
        metavar="N",
        help="Decode and print the first N packets (default: 10).",
    )
    parser.add_argument(
        "--hexpackets",
        nargs="?",
        const=20,
        type=int,
        metavar="N",
        help="Hex dump of first N payload packets (default: 20).",
    )
    parser.add_argument(
        "--rhythm",
        action="store_true",
        help="Rhythm analysis: per-second packet count pattern across N files.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        metavar="N",
        help="Number of sequential files to scan in --rhythm mode (default: 3).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="scan_all",
        help="In --rhythm mode, scan ALL files in the folder instead of N.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"\nError: File not found: '{args.file}'\n")
        sys.exit(1)

    # --rhythm runs independently (it prints its own headers and scans multiple files)
    if args.rhythm:
        analyze_drift_rhythm(args.file, n_files=args.n, use_all=args.scan_all)
        return

    print_metadata(args.file)

    if args.hex is not None:
        hex_inspector(args.file, limit=args.hex)
    if args.scan:
        scan_packet_size(args.file)
    if args.data is not None:
        decode_packets(args.file, n_packets=args.data)
    if args.hexpackets is not None:
        hex_packets(args.file, n_packets=args.hexpackets)


if __name__ == "__main__":
    main()

import logging
import os

import numpy as np
import pandas as pd

from src.core.binary_decoder import (
    TRAILER_SIZE,
    decode_binary_trailer,
    decode_binary_header,
)

logger = logging.getLogger("vesperflow")

# --- CONSTANTS ---
# True binary file header ends at byte 143.
# Bytes 144-149 are the pkt_ts field of the first data packet.
HEADER_SIZE = 144

# Subsecond tick scale — every IMU file observed uses 1023-tick resolution.
# The header, trailer, and per-packet pkt_ts all share this scale.
TICK_RESOLUTION = 1023
TICK_DENOM = (
    TICK_RESOLUTION + 1
)  # 1024 — the divisor used in time-within-second formula

# Sensor output rounding — matches actual hardware resolution.
# Values beyond these decimal places are float32 noise, not real data.
#   Acc:  ±16g range, ~0.5 mg/LSB     → 3dp (0.001 mg)
#   Gyro: stored in mdps, ~1.25/LSB   → 2dp (0.01 mdps)
#   Mag:  1.5 mGauss steps            → 1dp (0.1 mGauss)
#   Temp: raw * 0.01                  → 2dp (handled at extraction)
#   Pres: integer hPa                 → 0dp
ACC_DECIMALS = 3
GYRO_DECIMALS = 2
MAG_DECIMALS = 1
PRES_DECIMALS = 0


# =============================================================================
# FORMAT CONFIGURATION REGISTRY
# =============================================================================
# Maps bitmask bit flags to packet format parameters.
# To support a new format variant, add a new block in _get_format_config()
# following the same pattern.
#
# Known bitmask bits:
#   Bit 0 (0x01): Accelerometer active
#   Bit 1 (0x02): Gyroscope active
#   Bit 2 (0x04): Magnetometer active
#   Bit 3 (0x08): Extended format — adds 2B temperature + 2B pressure per
#                 packet. Packet grows from 42 to 46 bytes.
#   Bit 5 (0x20): Standard mode flag (observed in bitmask 0x27).
#                 Exact meaning undocumented — no structural effect on packet.
#
# Packet layout (both formats):
#   pkt_ts   6B   [0-5]    Packet timestamp:
#                            byte 0:   sync marker (always 0x55)
#                            byte 1:   format tag (bitmask & 0x0F)
#                            byte 2:   minutes (uint8 raw decimal, 0-59)
#                            byte 3:   seconds (uint8 raw decimal, 0-59)
#                            bytes 4-5: subsecond tick counter (uint16 LE,
#                                       0-1023 scale, counts DOWN by ~20 per
#                                       sample at 50Hz, wraps at second boundary)
#   gyro    12B   [6-17]   Gyroscope X, Y, Z  (float32 LE, output kept in mdps)
#   acc     12B  [18-29]   Accelerometer X, Y, Z  (float32 LE, mg)
#   mag     12B  [30-41]   Magnetometer X, Y, Z  (float32 LE, mGauss)
#   -- extended only (bit 3) --
#   temp     2B  [42-43]   Temperature uint16 (raw * 0.01 = °C)
#   pres     2B  [44-45]   Pressure uint16 (raw = hPa)
# =============================================================================


def _get_format_config(bitmask):
    """
    Derives packet format parameters from the header bitmask.
    """
    config = {
        "packet_size": 42,
        "gyro_scale": 1.0,  # Output stays in mdps (raw stored unit)
        "has_temp_pres": False,
    }

    if bitmask & 0x08:
        config["packet_size"] = 46
        config["has_temp_pres"] = True

    return config


def _build_dtype(config):
    """
    Constructs the NumPy dtype for one data packet given a format config.
    """
    fields: list = [
        ("pkt_ts", "u1", (6,)),
        ("gyro", "<f4", (3,)),
        ("acc", "<f4", (3,)),
        ("mag", "<f4", (3,)),
    ]

    if config["has_temp_pres"]:
        fields.append(("temp", "<u2"))
        fields.append(("pres", "<u2"))

    return np.dtype(fields)


def parse_imu_file(filepath):
    """
    Parses IMU binary (.BIN).

    Returns DataFrame matching the structure of 'MBN.csv' files.

    FILE STRUCTURE:
    ------------------------------------------------------------
    |  HEADER (0 - 143 Bytes)                                  |
    |----------------------------------------------------------|
    | Offset  | Type     | Description                         |
    | 0-3     | UInt32   | Magic Number (0xDEAFDAC0)           |
    | 4-7     | UInt32   | Device ID                           |
    | 8-23    | String   | Sensor Name (ASCII, e.g., "IMU10")  |
    |         |          | (null-padded to 16 bytes)           |
    | 24-25   | UInt16   | FWID                                |
    | 26-27   | UInt16   | HWID                                |
    | 28-31   | UInt32   | Sample Rate (Hz)                    |
    | 32-35   | UInt32   | WinLen                              |
    | 36-39   | UInt32   | WinRate                             |
    | 40-43   | UInt32   | Bitmask (active sensors + format)   |
    | 44-47   | UInt32   | Config0                             |
    | 48-51   | UInt32   | Config1                             |
    | 52-55   | UInt32   | Config2                             |
    | 56-59   | UInt32   | Config3                             |
    | 60-127  | Pad      | Padding (0xFF)                      |
    | 128-131 | UInt32   | Timestamp Sync Number (0x5AA55AA5)  |
    | 132     | BCD      | Start Hour                          |
    | 133     | BCD      | Start Minute                        |
    | 134     | BCD      | Start Second                        |
    | 135     | Pad      | Always 0x00                         |
    | 136     | UInt8    | Header_B136 — per-session marker,   |
    |         |          | exact meaning undocumented          |
    | 137     | BCD      | Start Month                         |
    | 138     | BCD      | Start Day                           |
    | 139     | BCD      | Start Year (offset from 2000)       |
    | 140-141 | UInt16   | Subsecond Resolution (typically 1023|
    | 142-143 | UInt16   | Subsecond Value of first sample     |
    |         |          | (same 1023-tick scale as pkt_ts)    |
    |----------------------------------------------------------|
    |  DATA PAYLOAD (Repeating packets — see _build_dtype)     |
    |----------------------------------------------------------|
    | pkt_ts  | 6 bytes  | see pkt_ts byte map below           |
    | gyro    | 12 bytes | X, Y, Z float32 (mdps)              |
    | acc     | 12 bytes | X, Y, Z float32 (mg)                |
    | mag     | 12 bytes | X, Y, Z float32 (mGauss)            |
    | temp*   | 2 bytes  | uint16 raw * 0.01 = °C (ext only)   |
    | pres*   | 2 bytes  | uint16 raw = hPa   (ext only)       |
    ------------------------------------------------------------

    FORMAT VARIANTS (driven by Bitmask — see _get_format_config):
    ------------------------------------------------------------
    | Bitmask 0x07 | 42B packets | no temp/pres               |
    | Bitmask 0x27 | 42B packets | no temp/pres               |
    | Bitmask 0x0F | 46B packets | temp + pres                |
    ------------------------------------------------------------

    pkt_ts byte map:
    ------------------------------------------------------------
    | Byte 0   | 0x55          | Fixed sync marker             |
    | Byte 1   | bitmask&0x0F  | Format tag (lower nibble)     |
    |          |               | Self-describing: decoder can  |
    |          |               | identify format from any pkt  |
    | Byte 2   | uint8         | Minutes (raw decimal, 0-59)   |
    | Byte 3   | uint8         | Seconds (raw decimal, 0-59)   |
    | Bytes 4-5| uint16 LE     | Subsecond tick counter        |
    |          |               | 0-1023 scale (same as header) |
    |          |               | Counts DOWN ~20 ticks/sample  |
    |          |               | Wraps at every second boundary|
    |          |               | IS the authoritative per-     |
    |          |               | sample time — no correction   |
    |          |               | needed                        |
    ------------------------------------------------------------

    TRAILER (last 16 bytes of file):
    ------------------------------------------------------------
    | Byte 0-3 | A5 5A 5A A5  | End sync (NOT of header sync)  |
    | Byte 4   | BCD          | End Hour                       |
    | Byte 5   | BCD          | End Minute                     |
    | Byte 6   | BCD          | End Second                     |
    | Byte 7   | UInt8        | Trailer_B7 — varies per file,  |
    |          |              | undocumented                   |
    | Byte 8   | UInt8        | Trailer_B8 — mirrors Header_B136|
    | Byte 9   | BCD          | End Month                      |
    | Byte 10  | BCD          | End Day                        |
    | Byte 11  | BCD          | End Year (offset from 2000)    |
    | Byte12-13| UInt16       | Subsecond resolution (=1023)   |
    | Byte14-15| UInt16       | Subsec tick AFTER last sample  |
    ------------------------------------------------------------

    Returns:
        (status, message, df, meta)
        status: "SUCCESS", "EMPTY", "FAIL"
    """
    if not os.path.exists(filepath):
        return "FAIL", "File not found", None, None

    try:
        # --- PART 1: HEADER PARSING ---
        try:
            meta = decode_binary_header(filepath, header_size=HEADER_SIZE)
            if not meta:
                return "FAIL", "Header invalid (read returned None)", None, None
        except Exception as e:
            return "FAIL", f"Header parse error: {str(e)}", None, None

        # --- PART 2: TRAILER PARSING ---
        # The trailer carries the authoritative end timestamp (sub-millisecond
        # precision) and a B8 byte that mirrors the header's Header_B136.
        trailer = decode_binary_trailer(filepath)
        if trailer is None:
            logger.warning(
                f"Trailer missing in {os.path.basename(filepath)} — "
                f"recording was interrupted (Scheduled Sleep OR Battery Exhaustion)."
            )
            # Replace the generic "Truncated" flag with explicit, dual-cause statuses
            meta["Trailer_Present"] = "FALSE"
            meta["Recording_Status"] = "INTERRUPTED (Scheduled Sleep or Battery Exhaustion)"

            # (non-mandatory) If there is downstream code that still strictly expects a boolean
            # for the report card, you can keep meta["Truncated"] = True here as a fallback!
        else:
            meta["Trailer_Present"] = "TRUE"
            meta["Recording_Status"] = "COMPLETED"
            meta["End_Time"] = trailer["End_Time"]
            meta["Trailer_B7"] = trailer["Trailer_B7"]
            meta["Trailer_B8"] = trailer["Trailer_B8"]

            # Sanity check: trailer B8 should mirror header B136
            if trailer["Trailer_B8"] != meta["Header_B136"]:
                logger.debug(
                    f"Trailer B8 (0x{trailer['Trailer_B8']:02X}) differs from "
                    f"Header_B136 (0x{meta['Header_B136']:02X}) in {filepath}"
                )

        # --- PART 3: FORMAT DETECTION ---
        fmt = _get_format_config(meta.get("Bitmask", 0))
        dt = _build_dtype(fmt)
        logger.debug(
            f"IMU format: packet={fmt['packet_size']}B "
            f"temp_pres={fmt['has_temp_pres']} "
            f"(Bitmask=0x{meta.get('Bitmask', 0):02X})"
        )

        # --- PART 4: PARSE DATA PAYLOAD ---
        # Read the body between header and trailer. If trailer is missing, read
        # everything after the header (np.fromfile will stop at the last full
        # packet automatically, leaving any trailing bytes unparsed).
        try:
            file_size = os.path.getsize(filepath)
            if trailer is not None:
                payload_size = file_size - HEADER_SIZE - TRAILER_SIZE
            else:
                payload_size = file_size - HEADER_SIZE

            n_packets_max = max(0, payload_size // fmt["packet_size"])

            with open(filepath, "rb") as f:
                f.seek(HEADER_SIZE)
                raw_struct = np.fromfile(f, dtype=dt, count=n_packets_max)
        except Exception as e:
            return "FAIL", f"Payload read error: {str(e)}", None, meta

        # --- PART 5: EMPTY CHECK ---
        num_samples = len(raw_struct)
        if num_samples == 0:
            return "EMPTY", "No sensor data rows found", None, meta

        # --- PART 5b: STARTUP DUPLICATE FILTER ---
        # The firmware writes the first packet twice at startup (identical pkt_ts
        # bytes and sensor values). Detect and drop the duplicate if present.
        if num_samples > 1 and np.array_equal(
            raw_struct["pkt_ts"][0], raw_struct["pkt_ts"][1]
        ):
            raw_struct = raw_struct[1:]
            num_samples -= 1
            logger.debug(
                f"Startup duplicate packet removed from {os.path.basename(filepath)}"
            )

        # --- PART 6: TIMESTAMP EXTRACTION FROM pkt_ts ---
        # pkt_ts byte map (per packet):
        #   byte 0:    0x55 sync
        #   byte 1:    bitmask & 0x0F (format tag)
        #   byte 2:    minutes (uint8 raw decimal)
        #   byte 3:    seconds (uint8 raw decimal)
        #   bytes 4-5: subsecond TICK counter (uint16 LE, on 1023-tick scale)
        #
        # The tick counter is the AUTHORITATIVE per-sample timing source.
        # It counts DOWN from ~1023 to 0 by ~20 ticks per sample at 50Hz,
        # then wraps back to ~1023 at every second boundary.
        #
        # Time-within-second (ms) = 1000 * (TICK_RESOLUTION - tick) / TICK_DENOM
        #
        # Hour: pkt_ts has only minute and second. Hour comes from the header.
        # We track minute rollovers (59→0) to detect hour boundaries.

        pkt_ts = raw_struct["pkt_ts"]

        # Sync byte validation (warning only)
        expected_sync1 = meta.get("Bitmask", 0) & 0x0F
        if not (
            np.all(pkt_ts[:20, 0] == 0x55) and np.all(pkt_ts[:20, 1] == expected_sync1)
        ):
            logger.warning(
                f"pkt_ts sync byte mismatch in {filepath} "
                f"— expected [0x55, 0x{expected_sync1:02X}]"
            )

        pkt_min = pkt_ts[:, 2].astype(np.int64)
        pkt_sec = pkt_ts[:, 3].astype(np.int64)
        # Combine bytes 4 (lo) and 5 (hi) into uint16 little-endian.
        pkt_tick = pkt_ts[:, 4].astype(np.int64) | (pkt_ts[:, 5].astype(np.int64) << 8)

        # Hour rollover detection: minute jumps 59→0 indicate a new hour.
        min_diffs = np.diff(pkt_min, prepend=pkt_min[0])
        hour_offsets = np.cumsum(min_diffs < -30).astype(np.int64)

        # Non-contiguous recording detection: warn on large forward minute jumps.
        forward_jumps = np.where(min_diffs > 2)[0]
        if len(forward_jumps) > 0:
            gap_min = int(min_diffs[forward_jumps[0]])
            logger.warning(
                f"CRITICAL:Non-contiguous recording (maybe due to tag sleep) detected in {os.path.basename(filepath)}: "
                f"~{gap_min} minute gap at packet {forward_jumps[0]}. "
                f"This file may span multiple recording sessions."
            )
            logger.warning(
                f"CRITICAL: {gap_min}-minute forward jump detected in {os.path.basename(filepath)}"
                f"at packet {forward_jumps[0]}. Because Vesper hardware only embeds mm:ss in packets, "
                f"sleep gaps crossing hour boundaries cannot be safely tracked. "
                f"Absolute timestamps for the rest of this file are likely missing an hour offset!"
            )

        # Total seconds since midnight per packet.
        pkt_secs_from_midnight = (
            (meta["Start_Time"].hour + hour_offsets) * 3600 + pkt_min * 60 + pkt_sec
        )

        # Sub-second offset in microseconds, derived directly from pkt_ts tick.
        # Time within second (us) = 1_000_000 * (TICK_RESOLUTION - tick) / TICK_DENOM
        sub_us = (1_000_000 * (TICK_RESOLUTION - pkt_tick)) // TICK_DENOM

        # Build final timestamps from midnight of the recording date.
        base_dt = pd.Timestamp(
            year=meta["Start_Time"].year,
            month=meta["Start_Time"].month,
            day=meta["Start_Time"].day,
        )
        timestamps = base_dt + pd.to_timedelta(
            pkt_secs_from_midnight * 1_000_000 + sub_us, unit="us"
        )

        # Update meta with the precise per-packet derived start time.
        # (This supersedes the get_precise_start_time approximation for IMU.)
        meta["Start_Time"] = timestamps[0].to_pydatetime()
        meta["Start_Time_Str"] = meta["Start_Time"].strftime("%Y-%m-%d %H:%M:%S.%f")

        # --- PART 7: DATAFRAME CREATION ---
        acc_data = np.round(raw_struct["acc"] / 1.0, ACC_DECIMALS)
        gyro_data = np.round(raw_struct["gyro"] / fmt["gyro_scale"], GYRO_DECIMALS)
        mag_data = np.round(raw_struct["mag"] / 1.0, MAG_DECIMALS)

        if fmt["has_temp_pres"]:
            temp_data = np.round(raw_struct["temp"].astype(float) * 0.01, 2)
            pres_data = np.round(raw_struct["pres"].astype(float), PRES_DECIMALS)
        else:
            temp_data = np.zeros(num_samples, dtype=float)
            pres_data = np.zeros(num_samples, dtype=float)

        minutes = timestamps.minute.astype("int8")
        seconds = timestamps.second.astype("int8")
        millis = (timestamps.microsecond // 1000).astype("int16")

        data = {
            "Time": timestamps,
            "Minute": minutes,
            "Second": seconds,
            "Millisecond": millis,
            "Acc X [mg]": acc_data[:, 0],
            "Acc Y [mg]": acc_data[:, 1],
            "Acc Z [mg]": acc_data[:, 2],
            "Gyro X [mdps]": gyro_data[:, 0],
            "Gyro Y [mdps]": gyro_data[:, 1],
            "Gyro Z [mdps]": gyro_data[:, 2],
            "Mag X [mGauss]": mag_data[:, 0],
            "Mag Y [mGauss]": mag_data[:, 1],
            "Mag Z [mGauss]": mag_data[:, 2],
            "Temperature [C]": temp_data,
            "Bar Pressure [hPa]": pres_data,
        }

        df = pd.DataFrame(data)

        # --- FILE-SPLIT HICCUP FILTER ---
        # A hardware write glitch causes the gyroscope to occasionally
        # output exactly -8.75 (hex: 00 00 0C C1) across all three axes at the start of new files.

        # 1. Dynamically find the Gyro X, Y, and Z columns using a case-insensitive regex
        gyro_cols = df.filter(regex=r'(?i)gyro\s+[xyz]').columns

        # 2. Build the mask by checking if ALL dynamically found gyro columns equal -8.75
        hiccup_mask = (df[gyro_cols] == -8.75).all(axis=1)

        # 3. Convert the corrupted gyroscope readings to NaN (Not a Number)
        df.loc[hiccup_mask, gyro_cols] = np.nan

        cols_order = [
            "Time",
            "Minute",
            "Second",
            "Millisecond",
            "Acc X [mg]",
            "Acc Y [mg]",
            "Acc Z [mg]",
            "Gyro X [mdps]",
            "Gyro Y [mdps]",
            "Gyro Z [mdps]",
            "Mag X [mGauss]",
            "Mag Y [mGauss]",
            "Mag Z [mGauss]",
            "Temperature [C]",
            "Bar Pressure [hPa]",
        ]
        missing = [c for c in cols_order if c not in df.columns]
        if not missing:
            df = df[cols_order]

        return "SUCCESS", "Parsed Successfully", df, meta

    except Exception as e:
        return "FAIL", f"Crash: {str(e)}", None, None

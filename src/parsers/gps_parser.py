import os
import numpy as np
import logging

from src.core.binary_decoder import decode_gps_header, get_precise_start_time

logger = logging.getLogger("vesperflow")

def extract_gps_snapshots(filepath, output_dir):
    """
    Writes new .DAT files (raw GPS snapshots) from GPS binaries (.BIN)
    for the external GeoTag processing tool.

    Returns:
        (status, message)
        status: "SUCCESS", "EMPTY", "FAIL"
        message: Description of the result or error

    Includes:
    - Smart Header Detection (handling the 1024-byte 'warm-up' buffer).
    - Precision Timestamp calculation for logging.

    GPS Header: Should only be present in the first binary file
                of a recording/session such as '0G.BIN'
    Size: 16 Bytes
    Start: 0x5AA55AA5
    Timestamp Offset: 4 (BCD), 12 (SubSec)

    HEADER DETAIL (Offsets 0 - 15)
    ----------------------------------------------------
    | Offset | Type   | Description                    |
    |--------|--------|--------------------------------|
    | 0-3    | UInt32 | Sync Magic Number (0x5AA55AA5) |
    | 4-6    | BCD    | HH, MM, SS                     |
    | 7-8    | Bytes  | Padding/Config (0x07 at off 8) |
    | 9-11   | BCD    | MM, DD, YY                     |
    | 12-13  | UInt16 | Subsecond Resolution           |
    | 14-15  | UInt16 | Subsecond Value                |
    ----------------------------------------------------

    """
    if not os.path.exists(filepath):
        return "FAIL", "File Not Found"

    try:
        # --- 1. HEADER ANALYSIS ---
        # Handles both 'Compact' (16b) and 'Buffered' (1024b) GPS file headers
        meta = decode_gps_header(filepath)
        if not meta:
            return "FAIL", "Invalid or truncated GPS header"

        header_size = meta["Header_Size"]
        start_dt = meta["Start_Time"]

        # Standard GeoTag filename format: snap.YYYY_MM_DD_HH_MM_SS_GC0.dat
        filename = start_dt.strftime("snap.%Y_%m_%d_%H_%M_%S_GC0.dat")

        # Calculate precision metadata for the logs
        precise_start = get_precise_start_time(filepath, meta, sensor_type="GPS")
        precise_time_str = precise_start.strftime('%H:%M:%S.%f')[:-3]

        # --- 2. OUTPUT SETUP ---
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            return "SUCCESS", "Skipped (Already Exists)"

        # --- 3. PAYLOAD PROCESSING ---
        # Read file using the calculated header offset
        raw_data = np.fromfile(filepath, dtype='<u4', offset=header_size)

        if raw_data.size == 0:
            return "EMPTY", "Payload is empty (No GPS fixes)"

        # Word Swap (I/Q Correction)
        swapped_data = (raw_data << 16) | (raw_data >> 16)
        swapped_data.astype('<u4').tofile(output_path)

        # Save sub-second millisecond offset to a sidecar index (for Pass-2 clock pinning)
        if precise_time_str != "Time_Calc_Error":
            subsec_log = os.path.join(output_dir, "subseconds.csv")
            with open(subsec_log, "a", encoding="utf-8") as sf:
                sf.write(f"{filename},0.{precise_time_str.split('.')[-1]}\n")

        # Return success with the precise time we calculated
        return "SUCCESS", f"Parsed (Head:{header_size}b, T:{precise_time_str})"

    except Exception as e:
        logger.error(f"GPS Crash {os.path.basename(filepath)}: {e}")
        return "FAIL", f"Crash: {str(e)}"

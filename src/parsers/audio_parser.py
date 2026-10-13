import os
import struct
import logging
import numpy as np

from datetime import datetime
from src.core.binary_decoder import decode_binary_header, get_precise_start_time

logger = logging.getLogger("vesperflow")

def parse_audio_file(filepath):
    """
    Parses raw binary audio into standard WAV format.
    Includes artifact removal and BCD (Binary Coded Decimal) timestamp decoding.

    Returns:
        (status, message, audio_data, meta)
        status: "SUCCESS", "EMPTY", "FAIL"
        message: Description of the result or error

    Includes:
    - Decode 128-byte File Header + 16-byte Block Header with Precision Timing.
    - Remove 16-byte Metadata Trailers inserted every 64KB.
    - Remove "Startup Pop" (sensor initialization artifacts).
    - Return cleaned PCM data ready for .WAV export.

    FILE FORMAT SPECIFICATION:
    ===========================================================================
    - Codec: Signed 16-bit PCM (Little Endian).
    - Sample Rate: 48,000 Hz.
    - Structure: 128-byte File Header + 16-byte Block Header, followed by audio data.
    - Sync Markers:
        1. 64KB Page Trailers: Every 65,536 bytes, a 16-byte metadata trailer is inserted.
           [Magic: 4B] [Time: 4B] [Date: 4B] [SubsecFrac: 2B] [Subsec: 2B] = 16 Bytes.
           Magic = 0xABCDEFEF (Little Endian).
        2. Startup Pop: The first ~17ms contain sensor initialization data (0x8000).

    AUDIO (.BIN) FILE STRUCTURE
    ===========================================================================
    The file consists of a 128-byte File Header + 16-byte Block Header followed by a
    sequence of 64KB Audio Pages. Each page is terminated by a 16-byte Sync Marker.

    GLOBAL LAYOUT:
    ---------------------------------------------------------
    |  HEADER (0 - 144 Bytes)                               |
    |-------------------------------------------------------|
    |  AUDIO DATA PAGE 1 (~65,536 Bytes)                    |
    |-------------------------------------------------------|
    |  SYNC MARKER 1 (16 Bytes)                             |
    |-------------------------------------------------------|
    |  AUDIO DATA PAGE 2 (~65,536 Bytes)                    |
    |-------------------------------------------------------|
    |  ... (Repeats until EOF)                              |
    ---------------------------------------------------------

    1. HEADERS DETAIL (Offsets 0 - 143)
    ---------------------128-Byte File Header---------------------
    | Offset  | Type     | Value (Hex)     | Description --------|
    |---------|----------|-----------------|---------------------|
    | 0-3     | UInt32   | C0 DA AF DE     | Magic Number        |
    | 4-7     | UInt32   | 3C 50 0E 53     | Device ID           |
    | 8-14    | String   | ASCII "SPH0641" | Sensor Name         |
    | 28-31   | UInt32   | 80 BB 00 00     | Sample Rate = 48k   |
    | 60-127  |          | FF FF FF FF     | Padding             |
    |--------------------16-Byte Block Header--------------------|
    | 128-131 | UInt32   | 5A A5 5A A5     | Timestamp Sync Numb |
    |--------------------| Description --------------------------|
    | 132     | BCD      | 07 (7am)        | Start Hour          |
    | 133     | BCD      | 20 (20min)      | Start Minute        |
    | 134     | BCD      | 50 (50sec)      | Start Second        |
    | 135     | Pad      | 00 (00milisec?) | Always 0x00         |
    | 136     | UInt8    | Sometimes 07    | per-session marker, |
    |         |          |                 | meaning undocumented|
    | 137     | BCD      | 08 (Sept)       | Start Month         |
    | 138     | BCD      | 18 (18th)       | Start Day           |
    | 139     | BCD      | 25 (2025)       | Start Year          |
    | 140-141 | UInt16   | FF 03           | Subsecond Resolution|
    | 142-143 | UInt16   | CA 03           | Start Subsec Value  |
    --------------------------------------------------------------

    2. AUDIO PAYLOAD (Signed 16-bit PCM, Little Endian)
    ---------------------------------------------------------
    | Rel Byte| Value (Hex) | Int16 Val   | Description     |
    |---------|-------------|-------------|-----------------|
    | 0-1695  | 00 80       | -32768      | MUTE / STARTUP  |
    |         |             |             | (Sensor Wakeup) |
    | 1696+   | (Var)       | (Var)       | VALID AUDIO     |
    ---------------------------------------------------------

    3. SYNC MARKER (Inserted every ~64KB)
    ---------------------------------------------------------
    | Rel Byte| Value (Hex) | Description                   |
    |---------|-------------|-------------------------------|
    | 0-3     | EF EF CD AB | Magic Number                  |
    | 4-7     | HH MM SS XX | Time BCD (7 = Pad)            |
    | 8-11    | XX MM DD YY | Date BCD (8 = Pad)            |
    | 12-13   | FF 03       | 03FF => 1023  (Max Tick)      |
    | 14-15   | 0F 01       | 010F => 271   (Current Tick)  |
    ---------------------------------------------------------
    """

    # --- CONSTANTS ---
    # File Header (128) + Block Header (16) = 144 Bytes
    HEADER_SIZE = 144

    # Artifact Definition (64KB Page Trailer)
    TRAILER_MAGIC = b'\xEF\xEF\xCD\xAB' # 0xABCDEFEF (Little Endian)
    TRAILER_LEN = 16

    if not os.path.exists(filepath):
        # Return format: Status, Msg, Meta, AudioData, TrailerTimestamps
        return "FAIL", "File not found", None, None, []

    try:
        # --- 1. HEADER PARSING & PRECISION TIME ---
        # Decode standard metadata (IDs, SampleRate, Coarse Time)
        meta = decode_binary_header(filepath, header_size=HEADER_SIZE)

        if not meta:
            return "FAIL", "Header invalid/unreadable", None, None, []

        # Calculate Precision Start Time (Sub-millisecond)
        # It uses the shared logic for Offset 140-143 + 1-sample correction
        meta['Start_Time'] = get_precise_start_time(filepath, meta, sensor_type="AUD")
        meta['Start_Time_Str'] = meta['Start_Time'].strftime('%Y-%m-%d %H:%M:%S.%f')

        # --- 2. READ RAW FILE ---
        with open(filepath, 'rb') as f:
            f.seek(HEADER_SIZE)
            raw_bytes = f.read()

        if len(raw_bytes) == 0:
            return "EMPTY", "File has header but 0 bytes of audio", meta, None, []

        clean_byte_stream = bytearray()
        cursor = 0
        file_len = len(raw_bytes)
        timestamps = [] # Store timestamps found in trailers for debugging/validation

        # --- 3. ARTIFACT REMOVAL LOOP ---
        while cursor < file_len:
            # Search for the next metadata trailer
            next_trailer = raw_bytes.find(TRAILER_MAGIC, cursor)

            # If no trailer found, append the rest of the file and finish
            if next_trailer == -1:
                clean_byte_stream.extend(raw_bytes[cursor:])
                break

            # --- EXTRACT TRAILER TIMESTAMP (For Logging/Validation) ---
            # Trailer Structure: [Magic:4] [Time:4] [Date:4] [SubsecFrac:2] [Subsec:2]
            try:
                # We extract 12 bytes starting 4 bytes after the trailer magic
                # offsets relative to magic: 0-3=Magic, 4-7=Time, 8-11=Date, 12-15=Ticks
                ts_chunk = raw_bytes[next_trailer+4 : next_trailer+TRAILER_LEN]

                if len(ts_chunk) == TRAILER_LEN-4:
                    # Time: HH(0), MM(1), SS(2), Pad(3)
                    hh, mm, ss = ts_chunk[0], ts_chunk[1], ts_chunk[2]

                    # Date: Pad(4), Mon(5), Day(6), Year(7)
                    mon, day, yy  = ts_chunk[5], ts_chunk[6], ts_chunk[7]

                    # Ticks: SubsecFrac(8-9), Subsec(10-11)
                    subsec_frac, subsec = struct.unpack('<HH', ts_chunk[8:12])

                    # Calculate precise milliseconds using hardware math
                    if subsec_frac > 0:
                        milisecs = 1000.0 * ((subsec_frac - subsec) / (subsec_frac + 1.0))
                    else:
                        milisecs = 0.0

                    if milisecs < 0:
                        milisecs += 1000.0 # Safety net for rollover

                    # Use :02x to read BCD bytes directly as decimal strings
                    ts_str = f"20{yy:02x}-{mon:02x}-{day:02x} {hh:02x}:{mm:02x}:{ss:02x}.{int(milisecs):03d}"
                    timestamps.append(ts_str)
            except Exception:
                pass # Non-critical failure

            # --- CALCULATE CUTS ---
            # Cut point Left: Trailer Start and ensure we don't cut before the current cursor (overlap check)
            cut_start = max(cursor, next_trailer)

            # Append valid audio up to the cut point
            clean_byte_stream.extend(raw_bytes[cursor : cut_start])

            # Advance Cursor: Skip the 16-byte Trailer
            cursor = next_trailer + TRAILER_LEN

        # --- 4. FINALIZE AUDIO ---
        # Convert to Numpy Array (Signed 16-bit PCM)
        audio_data = np.frombuffer(clean_byte_stream, dtype='<i2')

        # --- 5. DYNAMIC STARTUP ARTIFACT MUTE ---
        # The microphone hardware takes variable time to stabilize, emitting garbage
        # or a train of -32768 (0x8000) samples. We dynamically scan the start of the
        # file and mute up to the last -32768 sample, requiring an 8-sample (16-byte)
        # clean streak to confirm the initialization pop is truly over.

        search_limit = min(len(audio_data), 10000)
        mute_end_idx = 0
        consecutive_clean = 0

        for i in range(search_limit):
            if audio_data[i] == -32768:
                consecutive_clean = 0
                mute_end_idx = i + 1  # We must mute at least up to this pop
            else:
                consecutive_clean += 1
                if consecutive_clean >= 8:
                    # 8 clean samples in a row confirms the artifact is over
                    break

        if mute_end_idx > 0:
            audio_data[:mute_end_idx] = 0

        # FINAL CHECK: Did we end up with valid data?
        if len(audio_data) == 0:
            return "EMPTY", "Silent (0 samples after processing)", meta, audio_data, timestamps

        # Calculate Duration for Meta
        if meta.get('SampleRate', 0) > 0:
            meta['Duration'] = len(audio_data) / meta['SampleRate']

        return "SUCCESS", "Parsed Successfully", meta, audio_data, timestamps

    except Exception as e:
        return "FAIL", f"Crash: {str(e)}", None, None, []

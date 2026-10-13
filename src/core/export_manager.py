import logging
import os
import shutil
from datetime import timedelta

import pandas as pd
from scipy.io import wavfile

logger = logging.getLogger("vesperflow")


class ExportManager:
    def __init__(self, processed_root):
        self.processed_root = processed_root
        self.structure = {
            "gps": os.path.join(processed_root, "gps"),
            "imu": os.path.join(processed_root, "imu"),
            "aud": os.path.join(processed_root, "aud"),
        }

        for path in self.structure.values():
            os.makedirs(path, exist_ok=True)

    def save_session_metadata(self, meta, end_time=None, time_stamps=None):
        """
        Saves the sidecar .txt file.
        Formats Configs and Bitmask as Hexadecimal to match the VesperApp output.
        """

        # --- Construct Filename and Path ---
        try:
            time_fmt = "%Y%m%d_%H%M%S"
            start_str = meta["Start_Time"].strftime(time_fmt)

            if end_time:
                end_str = end_time.strftime(time_fmt)
                txt_filename = f"{start_str}-{end_str}_{meta['DeviceID']}.txt"
            else:
                txt_filename = f"{start_str}_{meta['DeviceID']}.txt"

        except AttributeError:
            logger.error(f"Metadata error: Start_Time missing.")
            return

        if meta["Sensor"] == "IMU10":
            meta_dir = os.path.join(self.structure["imu"], "metadata")
        elif meta["Sensor"] == "SPH0641":
            meta_dir = os.path.join(self.structure["aud"], "metadata")
        else:
            return

        os.makedirs(meta_dir, exist_ok=True)

        txt_path = os.path.join(meta_dir, txt_filename)

        if os.path.exists(txt_path):
            return

        lines = [
            f"DeviceID:{meta['DeviceID']:X}"
            if isinstance(meta["DeviceID"], int)
            else f"DeviceID:{meta['DeviceID']}",
            f"HWID:{meta['HWID']:X}",
            f"FWID:{meta['FWID']:X}",
            f"Sensor:{meta['Sensor']}",
            f"SampleRate:{meta['SampleRate']}",
            f"WinRate:{meta['WinRate']}",
            f"WinLen:{meta['WinLen']}",
            f"Config0:{meta['Config0']:X}",
            f"Config1:{meta['Config1']:X}",
            f"Config2:{meta['Config2']:X}",
            f"Config3:{meta['Config3']:X}",
            f"Bitmask:{meta['Bitmask']:X}",
        ]

        # Header_B136: per-session byte at offset 136 of the binary header.
        # Stable across all files in a recording session; mirrored in the
        # trailer's B8. Exact meaning undocumented — logged for future analysis.
        # Leading theory: per-session configuration preset or schedule slot index.
        # Alternatives: session sequence counter, firmware state flag,
        #               schedule/program index, Cell-Guide internal field.
        b136 = meta.get("Header_B136")
        if isinstance(b136, int):
            lines.append(f"Header_B136:{b136:02X}")
        else:
            lines.append("Header_B136:N/A")

        # Trailer fields — only present for files with a valid (non-truncated) trailer.
        # Trailer_B7: varies between files within the same session. Exact meaning
        # undocumented — possibly a counter or checksum.
        # Trailer_B8: mirrors Header_B136 in well-formed files.
        if "Trailer_B7" in meta and isinstance(meta["Trailer_B7"], int):
            lines.append(f"Trailer_B7:{meta['Trailer_B7']:02X}")
        if "Trailer_B8" in meta and isinstance(meta["Trailer_B8"], int):
            lines.append(f"Trailer_B8:{meta['Trailer_B8']:02X}")

        # --- SCHEDULE-AWARE TRUNCATION LOGIC: Explicitly log if the trailer was found and the resulting recording status
        if "Trailer_Present" in meta:
            lines.append(f"Trailer_Present:{meta['Trailer_Present']}")
        if "Recording_Status" in meta:
            lines.append(f"Recording_Status:{meta['Recording_Status']}")

        # Fallback for older parser versions or audio files that still use the boolean
        elif meta.get("Truncated"):
            lines.append("Truncated:TRUE")

        # Append Audio Drift Timestamps (If present)
        if time_stamps and isinstance(time_stamps, list) and len(time_stamps) > 0:
            lines.append("")
            lines.append("=== EMBEDDED BLOCK TIMESTAMPS (Audio Drift Check) ===")
            for i, ts in enumerate(time_stamps):
                lines.append(f"Block_{i + 1}: {ts}")

        try:
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.error(f"Failed to write metadata txt: {e}")

    def save_imu_csv(self, dataframe, uid=None):
        """
        Saves the IMU DataFrame.
        """
        if dataframe is None or dataframe.empty:
            return False

        output_path = None
        try:
            start_dt = dataframe["Time"].iloc[0]
            end_dt = dataframe["Time"].iloc[-1]

            time_fmt_file = "%Y%m%d_%H%M%S"
            new_filename = f"{start_dt.strftime(time_fmt_file)}-{end_dt.strftime(time_fmt_file)}_{uid}.csv"
            output_path = os.path.join(self.structure["imu"], new_filename)

            df_export = dataframe.copy()

            time_str = df_export["Time"].dt.strftime("%d/%m/%Y %H:%M:%S")
            ms_str = (df_export["Time"].dt.microsecond // 1000).astype(str).str.zfill(3)
            df_export["Time"] = time_str + "." + ms_str

            # Format sensor columns to fixed decimal places before writing.
            # Without this, pandas drops trailing zeros (e.g. -25.010 → -25.01),
            # making the output look non-uniform even though values are correctly rounded.
            col_formats = {
                "Acc X [mg]": 3,
                "Acc Y [mg]": 3,
                "Acc Z [mg]": 3,
                "Gyro X [mdps]": 2,
                "Gyro Y [mdps]": 2,
                "Gyro Z [mdps]": 2,
                "Mag X [mGauss]": 1,
                "Mag Y [mGauss]": 1,
                "Mag Z [mGauss]": 1,
                "Temperature [C]": 2,
                "Bar Pressure [hPa]": 0,
            }
            for col, decimals in col_formats.items():
                if col in df_export.columns:
                    df_export[col] = df_export[col].map(
                        lambda x, d=decimals: f"{x:.{d}f}"
                    )

            # Zero-pad the Millisecond column to always show 3 digits (e.g. 2 → 002)
            if "Millisecond" in df_export.columns:
                df_export["Millisecond"] = (
                    df_export["Millisecond"].astype(int).map(lambda x: f"{x:03d}")
                )

            df_export.to_csv(output_path, index=False, sep=",")

            logger.info(f"Saved IMU CSV: {new_filename}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to save CSV {output_path if output_path else 'Unknown'}: {e}"
            )
            return False

    def save_aud_wav(self, audio_data, meta):
        """
        Saves the Audio_data to .WAV with a timestamped filename.
        """
        if audio_data is not None and len(audio_data) > 0:
            output_path = None
            try:
                duration_seconds = len(audio_data) / meta["SampleRate"]
                start_dt = meta["Start_Time"]
                end_dt = start_dt + timedelta(seconds=duration_seconds)

                time_fmt = "%Y%m%d_%H%M%S"
                start_string = start_dt.strftime(time_fmt)
                end_string = end_dt.strftime(time_fmt)

                new_filename = f"{start_string}-{end_string}_{meta['DeviceID']}.wav"
                output_path = os.path.join(self.structure["aud"], new_filename)

                wavfile.write(output_path, meta["SampleRate"], audio_data)

                logger.info(f"Saved Audio: {new_filename}")
                return True

            except Exception as e:
                logger.error(
                    f"Failed to save .WAV {output_path if output_path else 'Unknown'}: {e}"
                )
                return False
        else:
            return False

    def finalize_geotag_csv(self, decoded_folder, session_id):
        """
        Locates the generic 'Track-geoTag.csv' generated by GeoTag.exe,
        renames it using the Start-End timestamps found inside,
        and moves it to the main 'data/processed/gps/' folder.
        """
        source_csv = os.path.join(decoded_folder, "Track-geoTag.csv")

        if not os.path.exists(source_csv):
            logger.warning(f"GPS CSV not found at: {source_csv}")
            return False, 0

        try:
            df = pd.read_csv(source_csv)

            if df.empty:
                logger.warning(f"GPS CSV is empty: {source_csv}")
                return False, 0

            try:
                start_str = f"{df.iloc[0]['Date']} {df.iloc[0]['UTC']}"
                end_str = f"{df.iloc[-1]['Date']} {df.iloc[-1]['UTC']}"

                start_dt = pd.to_datetime(start_str)
                end_dt = pd.to_datetime(end_str)

                fmt = "%Y%m%d-%H%M%S"
                start_fmt = start_dt.strftime(fmt)
                end_fmt = end_dt.strftime(fmt)

            except Exception as e:
                logger.warning(
                    f"Could not parse dates from GPS CSV ({e}). Using generic name."
                )
                start_fmt = "UnknownStart"
                end_fmt = "UnknownEnd"

            new_filename = f"{start_fmt}_{end_fmt}_{session_id}.csv"
            dest_path = os.path.join(self.structure["gps"], new_filename)

            shutil.copy2(source_csv, dest_path)

            logger.info(f"Finalized GPS CSV: {new_filename}")
            return True, len(df)

        except Exception as e:
            logger.error(f"Failed to finalize GPS CSV for {session_id}: {e}")
            return False, 0

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.core.logger import AnimatedTqdm
from src.parsers.imu_parser import parse_imu_file

def process_imu_session(session_id, imu_files, exporter, reporter, sess_metrics, extract_sequence_index, master_pbar, logger):
    sess_metrics["files_total"] += len(imu_files)
    imu_files.sort(key=extract_sequence_index)

    imu_chunks = []
    session_device_id = None
    last_meta = None
    imu_glitches = 0
    imu_dups = 0

    if not imu_files:
        logger.warning(f"No IMU binary files found in {session_id}")
        return session_device_id

    logger.info(f"Starting IMU Parser on {len(imu_files)} binary files...")

    with tqdm(
        total=len(imu_files),
        desc=f"🧭 IMU ({session_id})",
        unit="file",
        position=0,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.1,
    ) as imu_bar:
        for filepath in imu_files:
            try:
                status, msg, df, meta = parse_imu_file(filepath)

                if status == "SUCCESS":
                    sess_metrics["imu_ok"] += 1
                    imu_chunks.append(df)

                    if meta:
                        imu_glitches += meta.get("Glitches_Masked", 0)
                        imu_dups += meta.get("Startup_Duplicates_Dropped", 0)

                    if session_device_id is None and meta:
                        session_device_id = meta.get("DeviceID", "UnknownTag")
                        sess_metrics["id"] = session_device_id
                        last_meta = meta

                reporter.log_file_result("imu", status, filepath, msg)
            except Exception as e:
                reporter.log_file_result("imu", "FAIL", filepath, f"Unexpected loop crash: {e}")
            finally:
                imu_bar.update(1)
                master_pbar.update(1)

        imu_df = pd.concat(imu_chunks, ignore_index=True) if imu_chunks else pd.DataFrame()

        if not imu_df.empty:
            try:
                imu_bar.set_description_str(f"💾 Saving IMU CSV ({session_id})", refresh=True)
                t_start = imu_df["Time"].iloc[0]
                t_end = imu_df["Time"].iloc[-1]

                # Update session calendar window natively (avoiding update_range helper)
                if sess_metrics["start_time"] is None or t_start < sess_metrics["start_time"]:
                    sess_metrics["start_time"] = t_start
                if sess_metrics["end_time"] is None or t_end > sess_metrics["end_time"]:
                    sess_metrics["end_time"] = t_end

                nominal_sr = float(last_meta.get("SampleRate", 50.0)) if last_meta else 50.0
                if nominal_sr <= 0: nominal_sr = 50.0
                sess_metrics["duration_imu_sec"] = len(imu_df) / nominal_sr

                time_diffs = imu_df["Time"].diff().dt.total_seconds()
                active_sec = float(time_diffs[(time_diffs > 0) & (time_diffs <= 1.0)].sum())
                eff_hz = (len(imu_df) - 1) / active_sec if active_sec > 0 else nominal_sr

                temp_col = imu_df["Temperature [C]"]
                pres_col = imu_df["Bar Pressure [hPa]"]
                has_env = bool(last_meta and (last_meta.get("Bitmask", 0) & 0x08)) or (temp_col.abs().max() > 0)

                sess_metrics["imu_analytics"] = {
                    "total_samples": len(imu_df),
                    "effective_hz": eff_hz,
                    "temp_range_str": f"{temp_col.min():.1f} to {temp_col.max():.1f}" if has_env else "N/A (Off)",
                    "pres_range_str": f"{pres_col.min():.0f} to {pres_col.max():.0f}" if has_env else "N/A (Off)",
                    "glitches_masked": imu_glitches,
                    "dups_dropped": imu_dups,
                }

                if last_meta:
                    last_meta["Start_Time"] = t_start

                success = exporter.save_imu_csv(imu_df, uid=session_device_id)
                if success and last_meta:
                    exporter.save_session_metadata(last_meta, end_time=t_end)

            except Exception as e:
                logger.error(f"Failed to finalize IMU session {session_id}: {e}")

    return session_device_id

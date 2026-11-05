import os
import re
import pandas as pd
from tqdm import tqdm

from src.parsers.gps_parser import extract_gps_snapshots
from src.sdr.solver import solve_session_directory
from src.wrappers.geotag_wrapper import run_geotag

def process_gps_session(session_id, gps_files, exporter, reporter, sess_metrics, session_device_id, extract_sequence_index, master_pbar, logger, config, processed_folder, GPS_TEST_LIMIT):
    gps_files.sort(key=extract_sequence_index)
    if GPS_TEST_LIMIT:
        gps_files = gps_files[:GPS_TEST_LIMIT]

    sess_metrics["files_total"] += len(gps_files)
    sess_metrics["gps_attempts"] += len(gps_files)

    session_gps_valid_count = 0
    session_ref_time = "Unknown Time"

    session_snap_dir = os.path.join(processed_folder, "gps", "snapshots", session_id)
    session_decode_dir = os.path.join(processed_folder, "gps", "decoded", session_id)
    abs_snap_dir = os.path.abspath(session_snap_dir)
    abs_decode_dir = os.path.abspath(session_decode_dir)

    if not gps_files:
        return

    logger.info(f"Starting GPS Parser: Extracting snapshots from {len(gps_files)} binary files...")

    for filepath in tqdm(
        gps_files,
        desc=f"🛰️  GPS Unpack ({session_id})",
        unit="file",
        position=0,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
    ):
        try:
            status, msg = extract_gps_snapshots(filepath, abs_snap_dir)

            if status == "SUCCESS":
                sess_metrics["gps_ok"] += 1
                session_gps_valid_count += 1

                if session_ref_time == "Unknown Time":
                    try:
                        generated_files = os.listdir(abs_snap_dir)
                        if generated_files:
                            match = re.search(r"snap\.(\d{4}_\d{2}_\d{2})", generated_files[0])
                            if match:
                                session_ref_time = match.group(1)
                    except Exception:
                        pass

            reporter.log_file_result("gps", status, filepath, msg)
        except Exception as e:
            reporter.log_file_result("gps", "FAIL", filepath, f"Unexpected loop crash: {e}")
        finally:
            master_pbar.update(1)

    final_id = session_device_id if (session_device_id and session_device_id != "UnknownTag") else session_id

    if session_ref_time != "Unknown Time":
        try:
            new_folder_name = f"{session_ref_time}-{final_id}"
            parent_snap = os.path.dirname(abs_snap_dir)
            parent_decode = os.path.dirname(abs_decode_dir)
            new_snap_dir = os.path.join(parent_snap, new_folder_name)
            new_decode_dir = os.path.join(parent_decode, new_folder_name)

            if not os.path.exists(new_snap_dir):
                if os.path.exists(abs_snap_dir):
                    os.rename(abs_snap_dir, new_snap_dir)
                    logger.info(f"   -> Renamed session folder to: {new_folder_name}")
                    abs_snap_dir = new_snap_dir
                    abs_decode_dir = new_decode_dir
            else:
                logger.info(f"   -> Folder {new_folder_name} exists. Using it.")
                abs_snap_dir = new_snap_dir
                abs_decode_dir = new_decode_dir

        except Exception as e:
            logger.warning(f"Could not rename session folder: {e}")

    if session_gps_valid_count > 0:
        solver_mode = config.get("gps_solver", "sdr").strip().lower()

        if solver_mode == "sdr":
            gps_region = config.get("gps_region", "europe")
            eph_dir = os.path.abspath(config.get("ephemeris_folder", "./data/ephemeris"))

            logger.info(f" -> Launching Native SDR Solver (Region: {gps_region})...")
            gps_success, gps_msg = solve_session_directory(
                dat_folder=abs_snap_dir,
                output_dir=abs_decode_dir,
                region=gps_region,
                ephemeris_dir=eph_dir,
                bar_position=0,
                master_pbar=master_pbar,
                max_files=len(gps_files),
            )
        else:
            logger.info(" -> Launching Legacy GeoTag.exe...")
            from src.main import resolve_config_path
            cli_path = resolve_config_path(config.get("gps_cli_path", "./external_tools/CG/GeoTag/GeoTag.exe"))
            eng_path = resolve_config_path(config.get("gps_engine_path", "./external_tools/CG/GeoTagEngine/GeoTagEngine.exe"))

            gps_success, gps_msg = run_geotag(
                dat_folder=abs_snap_dir,
                output_dir=abs_decode_dir,
                geotag_exe=cli_path,
                engine_exe=eng_path,
            )

        if gps_success:
            success, count = exporter.finalize_gps_csv(abs_decode_dir, final_id)
            if success:
                sess_metrics["gps_fixes"] = count
                track_csv = os.path.join(abs_decode_dir, "Track-sdr.csv")
                if os.path.exists(track_csv):
                    gdf = pd.read_csv(track_csv)
                    if not gdf.empty:
                        sv_col = gdf["SVs in Use"] if "SVs in Use" in gdf.columns else gdf["Satellites"]
                        cn0_col = gdf["SNR Avg [dB]"] if "SNR Avg [dB]" in gdf.columns else gdf["Max C/N0 [dB-Hz]"]
                        sess_metrics["gps_analytics"] = {
                            "fixes_3d": int((sv_col >= 5).sum()),
                            "fixes_2d": int((sv_col == 4).sum()),
                            "mean_cn0": float(pd.to_numeric(cn0_col, errors="coerce").mean()),
                            "med_hdop": float(pd.to_numeric(gdf.get("HDOP", pd.Series([0.0])), errors="coerce").median()),
                            "med_rms": float(pd.to_numeric(gdf.get("Pseudorange RMS [m]", pd.Series([0.0])), errors="coerce").median()),
                            "med_alt": float(pd.to_numeric(gdf["Altitude [m]"], errors="coerce").median()),
                            "med_clock_s": float(pd.to_numeric(gdf.get("Clock Offset [s]", pd.Series([0.0])), errors="coerce").median()),
                        }
        else:
            err_msg = f" [{final_id}]-> {gps_msg}"
            logger.error(err_msg)
            reporter.log_external_error(err_msg)

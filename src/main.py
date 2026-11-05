import os
import re
import sys
import traceback
import pandas as pd
import yaml

from src.core.constants import FULL_APP_NAME
from src.core.export_manager import ExportManager
from src.core.file_scanner import scan_raw_files
from src.core.logger import setup_logger, AnimatedTqdm
from src.core.run_reporter import RunReporter
from src.core.menu import run_startup_menu

from src.stages.stage_imu import process_imu_session
from src.stages.stage_audio import process_audio_session
from src.stages.stage_gps import process_gps_session


def load_config(config_path="config.yaml"):
    """Loads configuration from the YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_path(path):
    """
    Resolves a path from config.yaml relative to the Application Root.
    Handles 'frozen' state (running as .exe) vs script state.
    """
    if os.path.isabs(path):
        return path

    # Determine App Root
    if getattr(sys, "frozen", False):
        # If .exe, root is where the .exe is
        base_dir = os.path.dirname(sys.executable)
    else:
        # If script (src/main.py), root is two levels up (vesperflow/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.abspath(os.path.join(base_dir, path))


def extract_sequence_index(filepath):
    match = re.search(r"(\d+)", os.path.basename(filepath))
    return int(match.group(1)) if match else 0


def main():
    pd.set_option("display.max_columns", None)

    # Setup Logging
    logger = setup_logger("vesperflow", log_dir="./logs")
    logger.info(f"============== {FULL_APP_NAME} Started ===============")

    # Load Config
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Launch Interactive TUI (Bypassed if --headless or non-TTY)
    config = run_startup_menu(config)

    # Setup Paths & Objects
    raw_folder = config.get("raw_data_folder", "./data/raw")
    processed_folder = config.get("processed_folder", "./data/processed")

    reporter = RunReporter(
        report_mode=config.get("report_mode", "verbose"),
        custom_components=config.get("report_components", None),
    )
    exporter = ExportManager(processed_folder)

    # Scan all files in the input/raw folder
    all_sessions = scan_raw_files(raw_folder)

    # Read test limits from menu/config (defaults to None for full production runs)
    AUD_TEST_LIMIT = config.get("AUD_TEST_LIMIT", None)
    GPS_TEST_LIMIT = config.get("GPS_TEST_LIMIT", None)

    solver_mode = config.get("gps_solver", "sdr").strip().lower()
    gps_weight = 2 if solver_mode == "sdr" else 1 # 1 for .BIN unpack + 1 for SDR scan

    grand_total_files = sum(
        len(f["imu"])
        + (len(f["aud"][:AUD_TEST_LIMIT]) if AUD_TEST_LIMIT else len(f["aud"]))
        + gps_weight * (len(f["gps"][:GPS_TEST_LIMIT]) if GPS_TEST_LIMIT else len(f["gps"]))
        for f in all_sessions.values()
    )

    # Redirect logger output through tqdm so bars stay pinned cleanly at the bottom
    with AnimatedTqdm(
        total=grand_total_files,
        desc="📦 Total Pipeline Progress",
        unit="file",
        position=1,
        leave=False,
        mininterval=0.2,
        dynamic_ncols=True,
    ) as master_pbar:

        # ======== MAIN LOOP ======== (Iterates over each Recording/Tag/Session)
        for session_id, files_map in all_sessions.items():
            logger.info(f"Processing Session: {session_id}")

            # ---Initialize Session Metrics---
            sess_metrics = {
                "id": session_id,
                "start_time": None,
                "end_time": None,
                "imu_ok": 0, "aud_ok": 0, "gps_ok": 0, "gps_attempts": 0,
                "duration_imu_sec": 0.0, "aud_duration": 0.0,
                "gps_fixes": 0, "files_total": 0,
            }

            session_device_id = None

            # --- STAGE 1: IMU ---
            session_device_id = process_imu_session(
                session_id, files_map["imu"], exporter, reporter, sess_metrics,
                extract_sequence_index, master_pbar, logger
            ) or session_device_id

            # --- STAGE 2: AUDIO ---
            session_device_id = process_audio_session(
                session_id, files_map["aud"], exporter, reporter, sess_metrics,
                session_device_id, extract_sequence_index, master_pbar, logger, AUD_TEST_LIMIT
            ) or session_device_id

            # --- STAGE 3: GPS ---
            process_gps_session(
                session_id, files_map["gps"], exporter, reporter, sess_metrics,
                session_device_id, extract_sequence_index, master_pbar, logger, config, processed_folder, GPS_TEST_LIMIT
            )

            # SAVE SESSION STATS TO REPORTER
            reporter.add_session(sess_metrics)
    # Save Processing Report
    reporter.save_report(processed_folder, logger)
    logger.info("=" * 100)
    logger.info(f"{'[SUCCESS] Processing Complete!':^100}")
    logger.info("=" * 100)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main()
        input("Press Enter to exit...")
    except Exception as e:
        critic_err_message = "[CRITICAL ERROR] Something went wrong!"
        print("\n\n" + "!" * 100)
        print(f"{critic_err_message:^100}")
        print("!" * 100 + "\n")
        traceback.print_exc()
        input("[!] Press Enter to exit...")

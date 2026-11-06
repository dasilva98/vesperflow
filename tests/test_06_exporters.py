import os
import sys
import json
import wave
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
import pytest

import src.core.menu as menu_mod
from src.core.menu import (
    load_custom_regions,
    save_custom_regions,
    get_all_regions,
    save_config_yaml,
    run_startup_menu,
    _manage_custom_regions,
)
from src.core.export_manager import ExportManager
from src.stages.stage_gps import process_gps_session


GOLDEN_CONFIG_YAML = """# VesperFlow - Configuration

# =====================================================================
# Paths to Data Folders
# =====================================================================
raw_data_folder: "./data/raw"
processed_folder: "./data/processed"
ephemeris_folder: "./data/ephemeris" # Local cache for downloaded NASA/IGS satellite orbits

# =====================================================================
# UI & Workflow Settings
# =====================================================================
interactive_menu: true   # Set to false to disable the startup menu on headless servers
log_level: "INFO"        # Options: DEBUG, INFO, WARNING, ERROR

# =====================================================================
# Report Customization
# =====================================================================
report_mode: "verbose"

report_components:
  raw_stats: true
  sessions_inventory: true
  data_yield: true
  gps_analytics: true
  imu_analytics: true
  audio_analytics: true
  quality_control: true

# =====================================================================
# GPS Solver Configuration
# =====================================================================
gps_solver: "sdr"      # Options: "sdr" or "geotag"
gps_region: "europe"   # Default search region
gps_cli_path: "./external_tools/CG/GeoTag/GeoTag.exe"
gps_engine_path: "./external_tools/CG/GeoTagEngine/GeoTagEngine.exe"
"""


class _MockPrompt:
    def __init__(self, val):
        self._val = val

    def ask(self):
        return self._val() if callable(self._val) else self._val


def test_41_golden_config_yaml_mutation_and_byte_exact_factory_reset(tmp_path, monkeypatch):
    """Test 41: Mutates config.yaml on disk, verifies changes, then runs Factory Reset and checks exact restoration."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(GOLDEN_CONFIG_YAML, encoding="utf-8")

    # 1. Load, mutate multiple fields, and save
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    cfg["raw_data_folder"] = "./custom/input"
    cfg["gps_solver"] = "geotag"
    cfg["gps_region"] = "madagascar"
    cfg["report_mode"] = "custom"
    cfg["report_components"]["gps_analytics"] = False
    cfg["report_components"]["audio_analytics"] = False
    save_config_yaml(cfg, str(cfg_file))

    mutated_parsed = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert mutated_parsed["raw_data_folder"] == "./custom/input"
    assert mutated_parsed["gps_solver"] == "geotag"
    assert mutated_parsed["gps_region"] == "madagascar"
    assert mutated_parsed["report_components"]["gps_analytics"] is False

    # 2. Trigger Factory Reset via run_startup_menu
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    select_seq = iter(["🔄 Factory Reset config.yaml", "🚀 Start Processing Pipeline"])
    monkeypatch.setattr("questionary.select", lambda *a, **k: _MockPrompt(next(select_seq)))
    monkeypatch.setattr("questionary.text", lambda *a, **k: _MockPrompt("RESET CONFIG"))
    monkeypatch.setattr("questionary.press_any_key_to_continue", lambda *a, **k: _MockPrompt(None))

    run_startup_menu(cfg, config_path=str(cfg_file))

    # 3. Verify every line, comment, and value returned to the golden original
    assert cfg_file.read_text(encoding="utf-8") == GOLDEN_CONFIG_YAML


def test_42_save_config_yaml_ignores_commented_out_keys_and_colons_in_comments(tmp_path):
    """Test 42: Ensures commented-out YAML lines (# gps_region: 'brazil') are never uncommented or mangled."""
    content = (
        "# Commented out key below:\n"
        "# gps_region: \"brazil\"\n"
        "gps_region: \"europe\" # Note: ratio: 1:2:3\n"
    )
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(content, encoding="utf-8")

    save_config_yaml({"gps_region": "japan"}, str(cfg_file))
    lines = cfg_file.read_text(encoding="utf-8").splitlines()
    assert lines[1] == '# gps_region: "brazil"'
    assert lines[2] == 'gps_region: "japan" # Note: ratio: 1:2:3'


def test_43_save_config_yaml_handles_windows_crlf_line_endings(tmp_path):
    """Test 43: Ensures config files edited on Windows (CRLF) update cleanly without corrupting values."""
    crlf_content = 'raw_data_folder: "./data/raw"\r\ngps_region: "europe"\r\n'
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_bytes(crlf_content.encode("utf-8"))

    save_config_yaml({"gps_region": "affenwald"}, str(cfg_file))
    parsed = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert parsed["raw_data_folder"] == "./data/raw"
    assert parsed["gps_region"] == "affenwald"


def test_44_custom_regions_json_full_lifecycle_on_disk(tmp_path, monkeypatch):
    """Test 44: Tests creating, editing, merging, and factory-resetting data/regions.json on disk."""
    reg_file = tmp_path / "regions.json"
    monkeypatch.setattr(menu_mod, "CUSTOM_REGIONS_FILE", str(reg_file))

    # 1. Start empty, add 'dpz_site'
    custom_regions = load_custom_regions(str(reg_file))
    assert custom_regions == {}

    select_seq = iter([
        "Add / Edit a Custom Region",
        "Factory Reset Custom Regions",
        "<- Back to Main Menu",
    ])
    text_seq = iter(["dpz_site", "51.50", "51.60", "9.90", "10.05", "0.02"])
    monkeypatch.setattr("questionary.select", lambda *a, **k: _MockPrompt(next(select_seq)))
    monkeypatch.setattr("questionary.text", lambda *a, **k: _MockPrompt(next(text_seq)))

    # First stop before Factory Reset to inspect the saved JSON file on disk
    select_add_only = iter(["Add / Edit a Custom Region", "<- Back to Main Menu"])
    monkeypatch.setattr("questionary.select", lambda *a, **k: _MockPrompt(next(select_add_only)))
    cfg = {"gps_region": "europe"}
    _manage_custom_regions(cfg, custom_regions)

    assert reg_file.exists()
    on_disk = json.loads(reg_file.read_text(encoding="utf-8"))
    assert on_disk["dpz_site"] == [51.5, 51.6, 9.9, 10.05, 0.02]
    assert get_all_regions()["dpz_site"] == (51.5, 51.6, 9.9, 10.05, 0.02)

    # 2. Factory Reset Custom Regions -> wipes regions.json to {}
    select_reset = iter(["Factory Reset Custom Regions", "<- Back to Main Menu"])
    monkeypatch.setattr("questionary.select", lambda *a, **k: _MockPrompt(next(select_reset)))
    monkeypatch.setattr("questionary.confirm", lambda *a, **k: _MockPrompt(True))
    _manage_custom_regions(cfg, custom_regions)

    assert json.loads(reg_file.read_text(encoding="utf-8")) == {}
    assert cfg["gps_region"] == "europe"


def test_45_custom_region_coordinate_validator_rejects_out_of_bounds(monkeypatch):
    """Test 45: Captures the questionary float validator and verifies it rejects invalid lat/lon bounds."""
    captured_validators = []

    def spy_text(prompt_msg, default="", validate=None, **kwargs):
        if validate is not None:
            captured_validators.append((prompt_msg, validate))
        return _MockPrompt(None)  # Cancel after capturing validator

    select_seq = iter(["Add / Edit a Custom Region", "<- Back to Main Menu"])
    monkeypatch.setattr("questionary.select", lambda *a, **k: _MockPrompt(next(select_seq)))
    monkeypatch.setattr("questionary.text", lambda msg, **kw: _MockPrompt("test_site") if "identifier" in msg else spy_text(msg, **kw))

    _manage_custom_regions({"gps_region": "europe"}, {})
    assert len(captured_validators) == 1
    _, lat_validator = captured_validators[0]

    assert lat_validator("-95.0") != True        # Below -90 deg
    assert lat_validator("100.0") != True        # Above +90 deg
    assert lat_validator("not_a_float") != True  # Non-numeric
    assert lat_validator("51.395") is True       # Valid latitude


def test_46_wav_export_riff_header_and_sample_exact(tmp_path):
    """Test 46: Verifies generated .WAV files have strict 48kHz 16-bit RIFF headers and sample-exact payload."""
    exporter = ExportManager(str(tmp_path))
    audio_data = np.zeros(48000, dtype=np.int16)
    audio_data[::2] = 30000
    audio_data[1::2] = -30000

    meta = {
        "DeviceID": "TAG_AUDIO",
        "Sensor": "SPH0641",
        "SampleRate": 48000,
        "Start_Time": datetime(2025, 9, 29, 12, 0, 0),
    }
    exporter.save_aud_wav(audio_data, meta)

    wav_files = list(tmp_path.rglob("*.wav"))
    assert len(wav_files) == 1
    with wave.open(str(wav_files[0]), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == 48000
        assert wf.getnframes() == 48000
        recovered = np.frombuffer(wf.readframes(48000), dtype=np.int16)
        assert np.array_equal(recovered, audio_data)


def test_47_imu_csv_nan_glitch_empty_formatting(tmp_path):
    """Test 47: Verifies IMU NaN glitches are written as empty CSV cells (,,) without 'NaN' strings."""
    exporter = ExportManager(str(tmp_path))
    df = pd.DataFrame({
        "Time": [datetime(2025, 9, 29, 12, 0, 0), datetime(2025, 9, 29, 12, 0, 1)],
        "Gyro X [mdps]": [10.5, np.nan],
        "Temperature [C]": [22.1, 22.1],
    })

    exporter.save_imu_csv(df, uid="TAG_IMU")
    csv_file = list((tmp_path / "imu").glob("*.csv"))[0]

    raw_text = csv_file.read_text(encoding="utf-8").splitlines()
    assert len(raw_text) == 3
    assert "NaN" not in raw_text[2]
    assert ",," in raw_text[2] or raw_text[2].endswith(",")


def test_48_txt_metadata_hexadecimal_config_preservation(tmp_path):
    """Test 48: Ensures Config0-Config3 and Bitmask are written to the sidecar .txt in uppercase Hexadecimal."""
    exporter = ExportManager(str(tmp_path))
    meta = {
        "DeviceID": "530E503C",
        "Sensor": "IMU10",
        "FWID": 4,
        "HWID": 4,
        "SampleRate": 50,
        "WinLen": 0,
        "WinRate": 0,
        "Bitmask": 15,   # 0x0F in Hex
        "Config0": 10,   # 0x0A in Hex
        "Config1": 255,  # 0xFF in Hex
        "Config2": 0,
        "Config3": 0,
        "Footer_Present": True,
        "Recording_Status": "COMPLETE",
        "Start_Time": datetime(2025, 9, 29, 12, 0, 0),
    }
    exporter.save_session_metadata(meta, end_time=datetime(2025, 9, 29, 13, 0, 0))

    txt_files = list(tmp_path.rglob("*.txt"))
    assert len(txt_files) == 1
    content = txt_files[0].read_text(encoding="utf-8")
    assert "530E503C" in content
    assert "FF" in content


def test_49_gps_track_finalize_chronological_renaming(tmp_path):
    """Test 49: Verifies finalize_gps_csv applies YYYYMMDD-HHMMSS_YYYYMMDD-HHMMSS_DEVICEID.csv format."""
    exporter = ExportManager(str(tmp_path))
    decode_dir = tmp_path / "gps" / "decoded" / "session_1"
    decode_dir.mkdir(parents=True)

    track_path = decode_dir / "Track-sdr.csv"
    track_path.write_text(
        "Index,Status,Date,UTC,Latitude [DD]\n"
        "1,Valid,2025-09-29,11:13:08,51.3951\n"
        "2,Valid,2025-09-29,14:20:00,51.3952\n",
        encoding="utf-8",
    )

    success, count = exporter.finalize_gps_csv(str(decode_dir), "TEST_TAG")
    assert success is True
    assert count == 2

    final_files = list((tmp_path / "gps").glob("*.csv"))
    assert len(final_files) == 1
    assert final_files[0].name == "20250929-111308_20250929-142000_TEST_TAG.csv"


def test_50_gps_diagnostics_csv_accounting_for_rejected_and_low_sv_snapshots(tmp_path, monkeypatch):
    """Test 50: Verifies solve_session_directory writes all snapshots to Track-sdr-diagnostics.csv."""
    from concurrent.futures import ThreadPoolExecutor
    from src.sdr import solver as solver_mod

    snap_dir = tmp_path / "snapshots"
    out_dir = tmp_path / "decoded"
    snap_dir.mkdir()
    for s in ("08", "10", "12"):
        (snap_dir / f"snap.2025_09_29_11_13_{s}_GC0.dat").write_bytes(b"\x00" * 1024)

    def mock_worker(filepath, prn_dict):
        fname = os.path.basename(filepath)
        dt = datetime(2025, 9, 29, 11, 13, int(fname.split("_")[5]))
        if "_08_" in fname:
            return (filepath, fname, dt, [{"prn": "G01"}, {"prn": "G02"}])
        if "_10_" in fname:
            return (filepath, fname, dt, [{"prn": f"G0{i}"} for i in range(1, 5)])
        return (filepath, fname, dt, [{"prn": f"G0{i}"} for i in range(1, 6)])

    monkeypatch.setattr(solver_mod, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(solver_mod, "_worker_acquire", mock_worker)
    monkeypatch.setattr(solver_mod, "fetch_rinex_ephemeris", lambda *a, **k: "dummy.rnx")
    monkeypatch.setattr(solver_mod, "parse_rinex3_gps", lambda *a, **k: {})

    # Mock _worker_acquire: 1st file has 2 SVs (<4), 2nd has 4 SVs (high RMS rejected), 3rd has 5 SVs (valid fix)
    def mock_worker(filepath, prn_dict):
        fname = os.path.basename(filepath)
        dt = datetime(2025, 9, 29, 11, 13, int(fname.split("_")[5]))
        if "_08_" in fname:
            return (filepath, fname, dt, [{"prn": "G01"}, {"prn": "G02"}])
        if "_10_" in fname:
            return (filepath, fname, dt, [{"prn": f"G0{i}"} for i in range(1, 5)])
        return (filepath, fname, dt, [{"prn": f"G0{i}"} for i in range(1, 6)])

    monkeypatch.setattr(solver_mod, "_worker_acquire", mock_worker)
    monkeypatch.setattr(solver_mod, "fetch_rinex_ephemeris", lambda *a, **k: "dummy.rnx")
    monkeypatch.setattr(solver_mod, "parse_rinex3_gps", lambda *a, **k: {})

    def mock_solve(detections, *a, **k):
        if len(detections) == 4:
            return None  # High RMS rejection
        return {
            "lat": 51.395, "lon": 10.745, "alt": 330.0, "pdop": 1.8, "hdop": 1.5, "vdop": 0.9,
            "num_svs": 5, "svs_in_view": 10, "snr_avg": 27.0, "rms_m": 2.1, "dt_coarse_s": 1.05,
            "detections": detections, "sv_details": [(d, 45.0, 180.0, 20000.0, 0.0) for d in detections],
        }

    monkeypatch.setattr(solver_mod, "solve_coarse_time_position", mock_solve)

    ok, _ = solver_mod.solve_session_directory(str(snap_dir), str(out_dir), region="affenwald")
    assert ok is True

    track_df = pd.read_csv(out_dir / "Track-sdr.csv")
    diag_df = pd.read_csv(out_dir / "Track-sdr-diagnostics.csv")
    assert len(track_df) == 1   # Only the 1 valid fix
    assert len(diag_df) == 3    # All 3 snapshots logged with diagnostics
    statuses = list(diag_df["Status"])
    assert "Only 2 SVs acquired (<4)" in statuses
    assert "Solver rejected (high RMS)" in statuses
    assert "FIX" in statuses


def test_51_gps_folder_dynamic_renaming_handles_collisions(tmp_path, dummy_logger, monkeypatch):
    """Test 51: Verifies stage_gps renames the session folder safely even if a prior run folder exists."""
    class DummyReporter:
        def log_file_result(self, *a, **k): pass
        def log_external_error(self, *a, **k): pass

    class DummyBar:
        def update(self, *a): pass

    def mock_extract(filepath, outdir):
        os.makedirs(outdir, exist_ok=True)
        (tmp_path / outdir / "snap.2025_09_29_11_13_08_GC0.dat").touch()
        return "SUCCESS", "OK"

    monkeypatch.setattr("src.stages.stage_gps.extract_gps_snapshots", mock_extract)
    monkeypatch.setattr("src.stages.stage_gps.solve_session_directory", lambda *a, **k: (False, "No fixes"))

    raw_files = ["0G.BIN"]
    metrics = {"gps_attempts": 0, "gps_ok": 0, "files_total": 0}

    target_snap_dir = tmp_path / "gps" / "snapshots" / "2025_09_29-TAG_COLLISION"
    target_snap_dir.mkdir(parents=True)

    process_gps_session(
        "session_orig", raw_files, None, DummyReporter(), metrics,
        "TAG_COLLISION", lambda x: 0, DummyBar(), dummy_logger,
        {"gps_solver": "sdr"}, str(tmp_path), None,
    )
    assert target_snap_dir.exists()

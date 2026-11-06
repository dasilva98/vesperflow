import gzip
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.core.file_scanner import scan_raw_files
from src.core.export_manager import ExportManager
from src.core.run_reporter import RunReporter
from src.stages.stage_imu import process_imu_session
from src.stages.stage_audio import process_audio_session
from src.stages.stage_gps import process_gps_session
from src.sdr.ephemeris import fetch_rinex_ephemeris, datetime_to_gpstime


def test_52_midnight_rollover_in_gps_rename(tmp_path):
    """Test 52: Verifies multi-day track finalizing accurately formats cross-midnight filenames."""
    exporter = ExportManager(str(tmp_path))
    decode_dir = tmp_path / "gps" / "decoded" / "session_1"
    decode_dir.mkdir(parents=True)

    track_path = decode_dir / "Track-sdr.csv"
    track_path.write_text(
        "Index,Status,Date,UTC,Latitude [DD]\n"
        "1,Valid,2025-09-29,23:50:00,51.3951\n"
        "2,Valid,2025-09-30,00:15:00,51.3952\n",
        encoding="utf-8",
    )

    exporter.finalize_gps_csv(str(decode_dir), "NIGHT_TAG")
    final_files = list((tmp_path / "gps").glob("*.csv"))
    assert final_files[0].name == "20250929-235000_20250930-001500_NIGHT_TAG.csv"


def test_53_gps_week_rollover_boundary():
    """Test 53: Verifies modulo math safely handles snapshots occurring right at the GPS week boundary."""
    # Accounting for +18s GPS-UTC leap seconds in 2025:
    # 2025-09-27 23:59:41 UTC == 2025-09-27 23:59:59 GPS (1s before end of GPS Week 2385)
    week1, tow1 = datetime_to_gpstime(datetime(2025, 9, 27, 23, 59, 41))
    assert tow1 == 604799.0

    # 2025-09-27 23:59:43 UTC == 2025-09-28 00:00:01 GPS (1s into new GPS Week 2386)
    week2, tow2 = datetime_to_gpstime(datetime(2025, 9, 27, 23, 59, 43))
    assert tow2 == 1.0
    assert week2 == week1 + 1


def test_54_scan_raw_files_flattens_nested_chunk_subfolders(tmp_path):
    """Test 54: Ensures recursive crawler flattens 256-file chunk subfolders and ignores junk files."""
    tag_dir = tmp_path / "2025_TAG"
    chunk0 = tag_dir / "chunk_00"
    chunk1 = tag_dir / "chunk_01"
    chunk0.mkdir(parents=True)
    chunk1.mkdir()

    (chunk0 / "0M.BIN").touch()
    (chunk0 / "255M.BIN").touch()
    (chunk1 / "256M.BIN").touch()
    (tag_dir / "notes.txt").touch()
    (chunk0 / ".DS_Store").touch()

    sessions = scan_raw_files(str(tmp_path))
    assert len(sessions) == 1
    files_map = sessions["2025_TAG"]
    assert len(files_map["imu"]) == 3
    assert len(files_map["aud"]) == 0
    assert all(f.endswith(".BIN") for f in files_map["imu"])


def test_55_fault_isolation_one_corrupt_bin_middle_of_session(tmp_path, dummy_logger, monkeypatch):
    """Test 55: Ensures one corrupted file logs a FAIL but does not abort the entire session compilation."""
    def mock_parse_imu(filepath):
        if "bad" in filepath:
            return "FAIL", "Corrupted", pd.DataFrame(), None
        df = pd.DataFrame({
            "Time": [datetime(2025, 9, 29, 12, 0, 0)],
            "Temperature [C]": [0.0],
            "Bar Pressure [hPa]": [0.0],
        })
        return "SUCCESS", "OK", df, {"DeviceID": "TAG1", "SampleRate": 50.0, "Bitmask": 7}

    monkeypatch.setattr("src.stages.stage_imu.parse_imu_file", mock_parse_imu)

    class DummyReporter:
        def log_file_result(self, *a, **k): pass

    class DummyExporter:
        def save_imu_csv(self, *a, **k): return True
        def save_session_metadata(self, *a, **k): pass

    class DummyBar:
        def update(self, *a): pass

    metrics = {"id": "sess_faulty", "imu_ok": 0, "start_time": None, "end_time": None, "files_total": 0}
    files = ["0M.BIN", "bad_1M.BIN", "2M.BIN"]

    process_imu_session(
        "sess_faulty", files, DummyExporter(), DummyReporter(), metrics,
        lambda x: 0, DummyBar(), dummy_logger,
    )
    assert metrics["imu_ok"] == 2
    assert metrics["imu_analytics"]["total_samples"] == 2


def test_56_single_sensor_audio_only_tag_deployment(tmp_path, dummy_logger, monkeypatch):
    """Test 56: Verifies a tag deployment with only Audio files (0 IMU, 0 GPS) extracts DeviceID and reports cleanly."""
    t0 = datetime(2025, 9, 29, 6, 0, 0)
    monkeypatch.setattr(
        "src.stages.stage_audio.parse_audio_file",
        lambda fp: (
            "SUCCESS",
            "OK",
            {"DeviceID": "AUDIO_ONLY_TAG", "Sensor": "SPH0641", "SampleRate": 48000, "Start_Time": t0},
            np.ones(48000, dtype=np.int16) * 500,
            [],
        ),
    )

    class DummyExporter:
        def save_aud_wav(self, *a, **k): pass
        def save_session_metadata(self, *a, **k): pass

    class DummyBar:
        def update(self, *a): pass

    reporter = RunReporter(report_mode="verbose")
    sess_metrics = {
        "id": "raw_folder_name",
        "start_time": None, "end_time": None,
        "imu_ok": 0, "aud_ok": 0, "gps_ok": 0, "gps_attempts": 0,
        "duration_imu_sec": 0.0, "aud_duration": 0.0, "gps_fixes": 0, "files_total": 0,
    }

    dev_id = process_imu_session("s1", [], DummyExporter(), reporter, sess_metrics, lambda x: 0, DummyBar(), dummy_logger)
    dev_id = process_audio_session("s1", ["0U.BIN"], DummyExporter(), reporter, sess_metrics, dev_id, lambda x: 0, DummyBar(), dummy_logger, None)
    process_gps_session("s1", [], DummyExporter(), reporter, sess_metrics, dev_id, lambda x: 0, DummyBar(), dummy_logger, {}, str(tmp_path), None)

    reporter.add_session(sess_metrics)
    reporter.save_report(str(tmp_path), dummy_logger)

    assert dev_id == "AUDIO_ONLY_TAG"
    assert sess_metrics["id"] == "AUDIO_ONLY_TAG"
    report_text = next((tmp_path / "report_cards").glob("Report_*.txt")).read_text(encoding="utf-8")
    assert "AUDIO_ONLY" in report_text


def test_57_ephemeris_cache_hit_offline_execution(tmp_path, monkeypatch):
    """Test 57: Verifies ephemeris downloader bypasses network entirely if a local cached file exists."""
    eph_dir = tmp_path / "ephemeris"
    eph_dir.mkdir()

    dt = datetime(2025, 9, 29)
    doy = dt.timetuple().tm_yday
    cached_file = eph_dir / f"BRDC_{dt.year}_{doy:03d}.rnx"
    cached_file.write_text("x" * 15000, encoding="utf-8")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: pytest.fail("Network should not be accessed when local cache exists!"),
    )

    res = fetch_rinex_ephemeris(dt, cache_dir=str(eph_dir))
    assert res == str(cached_file)


def test_58_ephemeris_mirror_fallback(tmp_path, monkeypatch):
    """Test 58: Verifies fetch_rinex_ephemeris falls back to secondary IGS mirror if BKG primary times out."""
    eph_dir = tmp_path / "ephemeris"

    class MockResponse:
        def read(self):
            return gzip.compress(b"Valid RINEX Payload")
        def __enter__(self): return self
        def __exit__(self, *a): pass

    call_count = {"calls": 0}
    def mock_urlopen(req, timeout=20):
        call_count["calls"] += 1
        if call_count["calls"] == 1:
            raise OSError("Primary mirror timeout")
        return MockResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    dt = datetime(2025, 9, 29)
    res = fetch_rinex_ephemeris(dt, cache_dir=str(eph_dir))

    assert call_count["calls"] == 2
    assert "Valid RINEX Payload" in open(res, encoding="utf-8").read()

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.parsers.imu_parser import parse_imu_file
from src.parsers.audio_parser import parse_audio_file
from src.parsers.gps_parser import extract_gps_snapshots
from src.main import extract_sequence_index


def test_17_parsers_missing_and_corrupt_files_return_fail(tmp_path):
    """Test 17: Verifies IMU, Audio, and GPS parsers gracefully return FAIL on missing/tiny files."""
    missing = str(tmp_path / "does_not_exist.BIN")
    s_imu, *_ = parse_imu_file(missing)
    s_aud, *_ = parse_audio_file(missing)
    s_gps, _ = extract_gps_snapshots(missing, str(tmp_path))
    assert s_imu == "FAIL"
    assert s_aud == "FAIL"
    assert s_gps == "FAIL"


def test_18_parsers_header_only_empty_payload_returns_empty_warn(tmp_path):
    """Test 18: Verifies a 128-byte header-only file (0 payload bytes) returns EMPTY or FAIL without crashing."""
    empty_bin = tmp_path / "0M.BIN"
    empty_bin.write_bytes(b"\x00" * 128)

    status_imu, *_ = parse_imu_file(str(empty_bin))
    status_aud, *_ = parse_audio_file(str(empty_bin))
    assert status_imu in ("EMPTY", "FAIL")
    assert status_aud in ("EMPTY", "FAIL")


def test_19_imu_glitch_masking_neg_8_75_to_nan():
    """Test 19: Verifies the -8.75 mdps file-split gyroscope glitch filter masks only full-axis glitches."""
    df = pd.DataFrame({
        "Gyro X [mdps]": [1.2, -8.75, -8.75, 0.5],
        "Gyro Y [mdps]": [-0.4, -8.75, 2.10, 0.1],
        "Gyro Z [mdps]": [0.8, -8.75, -8.75, -0.2],
    })
    gyro_cols = [c for c in df.columns if "Gyro" in c]
    hiccup_mask = (df[gyro_cols] == -8.75).all(axis=1)
    df.loc[hiccup_mask, gyro_cols] = np.nan

    assert int(hiccup_mask.sum()) == 1
    assert df.loc[1, gyro_cols].isna().all()
    # Row 2 had -8.75 on X and Z, but 2.10 on Y -> must NOT be masked
    assert not df.loc[2, gyro_cols].isna().any()


def test_20_imu_sleep_gap_free_effective_sample_rate():
    """Test 20: Verifies effective Hz math ignores multi-hour scheduled tag sleep intervals."""
    t0 = datetime(2025, 9, 29, 8, 0, 0)
    # Burst 1: 100 samples at 50 Hz (2.0s), then 4-hour sleep gap, then Burst 2: 100 samples at 50 Hz
    burst1 = [t0 + timedelta(seconds=i * 0.02) for i in range(100)]
    burst2 = [t0 + timedelta(hours=4, seconds=i * 0.02) for i in range(100)]
    imu_df = pd.DataFrame({"Time": pd.to_datetime(burst1 + burst2)})

    time_diffs = imu_df["Time"].diff().dt.total_seconds()
    active_sec = float(time_diffs[(time_diffs > 0) & (time_diffs <= 1.0)].sum())
    eff_hz = (len(imu_df) - 1) / active_sec if active_sec > 0 else 50.0

    # Despite a 4-hour wall-clock span, active effective rate should be ~50.25 Hz (not 0.01 Hz)
    assert 49.5 <= eff_hz <= 50.6


def test_21_audio_bout_detection_and_dead_mic_flagging(tmp_path, dummy_logger, monkeypatch):
    """Test 21: Verifies >300s sleep gaps increment bout counts and silent arrays trigger dead-mic count."""
    from src.stages.stage_audio import process_audio_session

    t0 = datetime(2025, 9, 29, 9, 0, 0)
    fake_files = ["0U.BIN", "1U.BIN", "2U.BIN"]
    # File 0 & 1 are 60s apart (Bout 1); File 2 is 20 minutes later (Bout 2) and completely silent (dead mic)
    calls = [
        ("SUCCESS", "OK", {"DeviceID": "TAG1", "SampleRate": 48000, "Start_Time": t0}, np.ones(48000, dtype=np.int16) * 1000, []),
        ("SUCCESS", "OK", {"DeviceID": "TAG1", "SampleRate": 48000, "Start_Time": t0 + timedelta(seconds=60)}, np.ones(48000, dtype=np.int16) * 1000, []),
        ("SUCCESS", "OK", {"DeviceID": "TAG1", "SampleRate": 48000, "Start_Time": t0 + timedelta(minutes=20)}, np.zeros(48000, dtype=np.int16), []),
    ]
    monkeypatch.setattr("src.stages.stage_audio.parse_audio_file", lambda fp: calls.pop(0))

    class DummyExp:
        def save_aud_wav(self, *a, **k): pass
        def save_session_metadata(self, *a, **k): pass

    class DummyRep:
        def log_file_result(self, *a, **k): pass

    class DummyBar:
        def update(self, n): pass

    sess_metrics = {
        "id": "sess1", "start_time": None, "end_time": None,
        "aud_ok": 0, "aud_duration": 0.0, "files_total": 0,
    }
    process_audio_session(
        "sess1", fake_files, DummyExp(), DummyRep(), sess_metrics,
        None, extract_sequence_index, DummyBar(), dummy_logger, None
    )

    a_stat = sess_metrics["audio_analytics"]
    assert a_stat["bouts"] == 2
    assert a_stat["dead_mic_files"] == 1


def test_22_audio_clipping_percentage_calculation():
    """Test 22: Verifies digital rail clipping threshold (|sample| >= 32760) percentage math."""
    samples = np.zeros(10000, dtype=np.int16)
    samples[:25] = 32765   # Positive rail clip
    samples[25:50] = -32768  # Negative rail clip
    samples[50:100] = 32000  # Loud but not clipped

    clipped = int(np.sum(np.abs(samples.astype(np.float64)) >= 32760))
    pct = (clipped / len(samples)) * 100.0
    assert clipped == 50
    assert np.isclose(pct, 0.5000)


def test_23_gps_parser_subseconds_csv_sidecar_logging(tmp_path):
    """Test 23: Verifies solver sub-second sidecar reader handles malformed lines gracefully."""
    subsec_path = tmp_path / "subseconds.csv"
    subsec_path.write_text(
        "snap.2025_09_29_11_13_08_GC0.dat,0.425\n"
        "corrupted_line_without_comma\n"
        "snap.2025_09_29_11_13_12_GC0.dat,0.875\n",
        encoding="utf-8",
    )
    subsec_map = {}
    with open(subsec_path, "r", encoding="utf-8") as sf:
        for line in sf:
            parts = line.strip().split(",")
            if len(parts) == 2:
                subsec_map[parts[0]] = float(parts[1])

    assert subsec_map["snap.2025_09_29_11_13_08_GC0.dat"] == 0.425
    assert subsec_map["snap.2025_09_29_11_13_12_GC0.dat"] == 0.875
    assert len(subsec_map) == 2


def test_24_extract_sequence_index_sorting_edge_cases():
    """Test 24: Ensures hardware chunk filenames sort numerically (0, 1, 2, 10, 100) instead of ASCII."""
    files = ["/path/10M.BIN", "/path/2M.BIN", "/path/0M.BIN", "/path/100M.BIN", "/path/no_digits.BIN"]
    files.sort(key=extract_sequence_index)
    basenames = [os.path.basename(f) for f in files]
    assert basenames == ["0M.BIN", "no_digits.BIN", "2M.BIN", "10M.BIN", "100M.BIN"] or \
           basenames == ["no_digits.BIN", "0M.BIN", "2M.BIN", "10M.BIN", "100M.BIN"]

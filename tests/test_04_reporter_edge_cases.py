from datetime import datetime
import pytest

from src.core.run_reporter import RunReporter, DEFAULT_REPORT_COMPONENTS


def test_25_reporter_mode_resolution_compact_verbose_custom():
    """Test 25: Verifies compact disables the 3 analytics tables while verbose enables all 7."""
    rep_compact = RunReporter(report_mode="compact")
    assert rep_compact.components["raw_stats"] is True
    assert rep_compact.components["gps_analytics"] is False
    assert rep_compact.components["imu_analytics"] is False
    assert rep_compact.components["audio_analytics"] is False

    rep_verbose = RunReporter(report_mode="verbose")
    assert all(rep_verbose.components.values())

    rep_custom = RunReporter(
        report_mode="custom",
        custom_components={"raw_stats": False, "gps_analytics": True},
    )
    assert rep_custom.components["raw_stats"] is False
    assert rep_custom.components["gps_analytics"] is True
    assert rep_custom.components["data_yield"] is True  # Default fallback for omitted keys


def test_26_reporter_malformed_or_none_custom_dict_fallback():
    """Test 26: Ensures passing None or a non-dict to 'custom' mode falls back safely without crashing."""
    rep_none = RunReporter(report_mode="custom", custom_components=None)
    assert rep_none.components == DEFAULT_REPORT_COMPONENTS

    rep_bad = RunReporter(report_mode="UNKNOWN_GARBAGE_MODE", custom_components="not_a_dict")
    assert rep_bad.components == DEFAULT_REPORT_COMPONENTS


def test_27_reporter_zero_files_no_zero_division_error(tmp_path, dummy_logger):
    """Test 27: Ensures saving a report when 0 files were found does not raise ZeroDivisionError."""
    rep = RunReporter(report_mode="verbose")
    rep.save_report(str(tmp_path), dummy_logger)

    report_cards = list((tmp_path / "report_cards").glob("Report_*.txt"))
    assert len(report_cards) == 1
    content = report_cards[0].read_text(encoding="utf-8")
    assert "Total Files Found: 0" in content
    assert "0.0%" in content


def test_28_reporter_true_yield_ignores_multi_day_sleep_gaps():
    """Test 28: Verifies True Yield accumulates sample durations rather than calendar wall-clock span."""
    rep = RunReporter()
    # Session spans 4 days (96 hours) on the calendar, but only recorded 1 hour of IMU and 0.5h of Audio
    rep.add_session({
        "id": "530E503C",
        "start_time": datetime(2025, 9, 25, 8, 0),
        "end_time": datetime(2025, 9, 29, 8, 0),
        "duration_imu_sec": 3600.0,
        "aud_duration": 1800.0,
        "gps_fixes": 12,
    })
    assert rep._fmt_duration(rep.stats["duration_imu_sec"]) == "1h 0m 0s"
    assert rep._fmt_duration(rep.stats["duration_aud_sec"]) == "0h 30m 0s"
    assert rep.stats["gps_fixes"] == 12


def test_29_reporter_warning_truncation_over_25_files(tmp_path, dummy_logger):
    """Test 29: Verifies that >25 empty-file warnings are truncated cleanly in Quality Control."""
    rep = RunReporter(report_mode="verbose")
    for i in range(40):
        rep.log_file_result("imu", "EMPTY", f"/raw/{i}M.BIN", "File Empty")

    rep.save_report(str(tmp_path), dummy_logger)
    report_file = next((tmp_path / "report_cards").glob("Report_*.txt"))
    text = report_file.read_text(encoding="utf-8")
    assert "... and 15 additional warnings." in text


def test_30_reporter_custom_all_disabled_minimal_report(tmp_path, dummy_logger):
    """Test 30: Verifies report generation when the user unchecks all 7 custom sections."""
    all_off = {k: False for k in DEFAULT_REPORT_COMPONENTS}
    rep = RunReporter(report_mode="custom", custom_components=all_off)
    rep.log_file_result("aud", "SUCCESS", "0U.BIN")
    rep.save_report(str(tmp_path), dummy_logger)

    report_file = next((tmp_path / "report_cards").glob("Report_*.txt"))
    text = report_file.read_text(encoding="utf-8")
    assert "RAW DATA STATISTICS:" not in text
    assert "TOTAL DATA YIELD:" not in text
    assert "END OF REPORT" in text


def test_31_reporter_analytics_tables_with_zero_gps_fixes(tmp_path, dummy_logger):
    """Test 31: Verifies GPS analytics table prints N/A safely when one tag has fixes and another has 0."""
    rep = RunReporter(report_mode="verbose")
    rep.add_session({
        "id": "TAG_OK",
        "gps_attempts": 50,
        "gps_fixes": 10,
        "gps_analytics": {
            "fixes_3d": 8, "fixes_2d": 2, "mean_cn0": 27.5,
            "med_hdop": 1.6, "med_rms": 2.4, "med_alt": 335.0, "med_clock_s": 1.05,
        },
    })
    rep.add_session({
        "id": "TAG_BURROW",
        "gps_attempts": 50,
        "gps_fixes": 0,
        "gps_analytics": {},
    })
    rep.save_report(str(tmp_path), dummy_logger)
    report_file = next((tmp_path / "report_cards").glob("Report_*.txt"))
    text = report_file.read_text(encoding="utf-8")
    assert "TAG_OK" in text
    assert "TAG_BURROW" in text
    assert "0 / 0" in text

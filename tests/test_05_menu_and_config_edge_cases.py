import sys
import json
import yaml
import pytest

import src.core.menu as menu_mod
from src.core.menu import (
    load_custom_regions,
    save_custom_regions,
    save_config_yaml,
    run_startup_menu,
    _manage_custom_regions,
)


class _MockQuestionaryPrompt:
    """Helper to feed a deterministic sequence of answers into questionary prompts."""
    def __init__(self, answer):
        self._answer = answer

    def ask(self):
        return self._answer() if callable(self._answer) else self._answer


@pytest.mark.parametrize("flag", ["--headless", "--no-menu", "-y"])
def test_32_headless_bypass_cli_flags(monkeypatch, flag):
    """Test 32: Verifies --headless, --no-menu, and -y immediately bypass questionary prompts."""
    monkeypatch.setattr(sys, "argv", ["main.py", flag])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    # If questionary.select is called, fail the test immediately
    monkeypatch.setattr(
        "questionary.select",
        lambda *a, **k: pytest.fail("Menu should not launch when headless CLI flag is set!"),
    )
    cfg = {"gps_region": "europe", "interactive_menu": True}
    out = run_startup_menu(cfg)
    assert out["gps_region"] == "europe"


def test_33_headless_bypass_config_flag_and_non_tty_stdin(monkeypatch):
    """Test 33: Verifies interactive_menu: False or non-TTY stdin (Docker/SLURM) skips the menu."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr(
        "questionary.select",
        lambda *a, **k: pytest.fail("Menu should not launch in non-TTY or interactive_menu=False!"),
    )

    # Case A: interactive_menu: False in config.yaml
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    cfg_disabled = {"interactive_menu": False, "gps_region": "affenwald"}
    assert run_startup_menu(cfg_disabled)["gps_region"] == "affenwald"

    # Case B: interactive_menu: True, but running on HPC/Docker where stdin.isatty() is False
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    cfg_docker = {"interactive_menu": True, "gps_region": "germany"}
    assert run_startup_menu(cfg_docker)["gps_region"] == "germany"


def test_34_save_config_yaml_preserves_comments_and_inline_notes(tmp_path):
    """Test 34: Verifies regex save_config_yaml updates values in place while keeping # comments."""
    template = (
        "# VesperFlow Header Comment\n"
        "raw_data_folder: \"./data/raw\" # Keep this inline note!\n"
        "gps_solver: \"sdr\"\n"
        "gps_region: \"europe\"   # Region comment\n"
        "interactive_menu: true\n"
        "report_mode: \"verbose\"\n"
        "report_components:\n"
        "  raw_stats: true # Nested comment\n"
        "  gps_analytics: true\n"
    )
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(template, encoding="utf-8")

    updated_cfg = {
        "raw_data_folder": "./data/custom_raw",
        "gps_solver": "geotag",
        "gps_region": "affenwald",
        "interactive_menu": False,
        "report_mode": "custom",
        "report_components": {
            "raw_stats": False,
            "gps_analytics": True,
        },
    }
    save_config_yaml(updated_cfg, str(cfg_file))

    raw_text = cfg_file.read_text(encoding="utf-8")
    # 1. All comments must still be intact
    assert "# VesperFlow Header Comment" in raw_text
    assert "# Keep this inline note!" in raw_text
    assert "# Region comment" in raw_text
    assert "# Nested comment" in raw_text

    # 2. Parsed YAML values must reflect the updates
    parsed = yaml.safe_load(raw_text)
    assert parsed["raw_data_folder"] == "./data/custom_raw"
    assert parsed["gps_solver"] == "geotag"
    assert parsed["gps_region"] == "affenwald"
    assert parsed["interactive_menu"] is False
    assert parsed["report_mode"] == "custom"
    assert parsed["report_components"]["raw_stats"] is False
    assert parsed["report_components"]["gps_analytics"] is True


def test_35_save_config_yaml_excludes_transient_test_limits_and_handles_missing_file(tmp_path):
    """Test 35: Ensures AUD_TEST_LIMIT/GPS_TEST_LIMIT are never written to disk and missing files don't crash."""
    missing_path = tmp_path / "nonexistent_config.yaml"
    # Should return cleanly without raising FileNotFoundError
    save_config_yaml({"gps_region": "europe"}, str(missing_path))
    assert not missing_path.exists()

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('gps_region: "europe"\n', encoding="utf-8")
    save_config_yaml(
        {"gps_region": "madagascar", "AUD_TEST_LIMIT": 10, "GPS_TEST_LIMIT": 50},
        str(cfg_file),
    )
    text = cfg_file.read_text(encoding="utf-8")
    assert "AUD_TEST_LIMIT" not in text
    assert "GPS_TEST_LIMIT" not in text
    assert '"madagascar"' in text


def test_36_custom_regions_corrupted_json_and_invalid_tuple_lengths(tmp_path):
    """Test 36: Verifies load_custom_regions survives corrupted JSON and filters invalid bounding boxes."""
    reg_file = tmp_path / "regions.json"

    # Case A: Corrupted JSON syntax
    reg_file.write_text("{bad json syntax!!!", encoding="utf-8")
    assert load_custom_regions(str(reg_file)) == {}

    # Case B: Valid JSON, but one entry has only 3 numbers instead of 5 (lat_min, lat_max, lon_min, lon_max, step)
    reg_file.write_text(
        json.dumps({
            "valid_site": [51.0, 52.0, 10.0, 11.0, 0.05],
            "bad_short_site": [51.0, 52.0, 10.0],
            "bad_type_site": "not_a_list",
        }),
        encoding="utf-8",
    )
    loaded = load_custom_regions(str(reg_file))
    assert "valid_site" in loaded
    assert loaded["valid_site"] == (51.0, 52.0, 10.0, 11.0, 0.05)
    assert "bad_short_site" not in loaded
    assert "bad_type_site" not in loaded


def test_37_menu_factory_reset_config_rejects_wrong_confirmation_phrase(tmp_path, monkeypatch):
    """Test 37: Ensures typing anything other than 'RESET CONFIG' aborts the factory reset."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('gps_region: "madagascar"\n', encoding="utf-8")
    cfg = {"gps_region": "madagascar", "interactive_menu": True}

    select_answers = iter([
        "🔄 Factory Reset config.yaml",
        "🚀 Start Processing Pipeline",
    ])
    monkeypatch.setattr("questionary.select", lambda *a, **k: _MockQuestionaryPrompt(next(select_answers)))
    # User types lowercase 'reset config' instead of exact 'RESET CONFIG'
    monkeypatch.setattr("questionary.text", lambda *a, **k: _MockQuestionaryPrompt("reset config"))
    monkeypatch.setattr("questionary.press_any_key_to_continue", lambda *a, **k: _MockQuestionaryPrompt(None))

    out = run_startup_menu(cfg, config_path=str(cfg_file))
    assert out["gps_region"] == "madagascar"  # Unchanged!


def test_38_menu_factory_reset_config_accepts_exact_phrase_and_restores_defaults(tmp_path, monkeypatch):
    """Test 38: Ensures typing exact 'RESET CONFIG' restores config dictionary and updates config.yaml."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('gps_region: "madagascar"\nreport_mode: "compact"\n', encoding="utf-8")
    cfg = {"gps_region": "madagascar", "report_mode": "compact", "interactive_menu": True}

    select_answers = iter([
        "🔄 Factory Reset config.yaml",
        "🚀 Start Processing Pipeline",
    ])
    monkeypatch.setattr("questionary.select", lambda *a, **k: _MockQuestionaryPrompt(next(select_answers)))
    monkeypatch.setattr("questionary.text", lambda *a, **k: _MockQuestionaryPrompt("RESET CONFIG"))
    monkeypatch.setattr("questionary.press_any_key_to_continue", lambda *a, **k: _MockQuestionaryPrompt(None))

    out = run_startup_menu(cfg, config_path=str(cfg_file))
    assert out["gps_region"] == "europe"
    assert out["report_mode"] == "verbose"
    saved = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert saved["gps_region"] == "europe"


def test_39_menu_delete_active_custom_region_falls_back_to_europe(tmp_path, monkeypatch):
    """Test 39: Deleting the currently active custom region resets gps_region to 'europe'."""
    reg_file = tmp_path / "regions.json"
    monkeypatch.setattr(menu_mod, "CUSTOM_REGIONS_FILE", str(reg_file))

    custom_regions = {"kruger_park": (-25.0, -22.0, 30.0, 32.0, 0.1)}
    save_custom_regions(custom_regions, str(reg_file))
    cfg = {"gps_region": "kruger_park"}

    select_answers = iter([
        "Delete a Custom Region",
        "kruger_park",
        "<- Back to Main Menu",
    ])
    monkeypatch.setattr("questionary.select", lambda *a, **k: _MockQuestionaryPrompt(next(select_answers)))
    monkeypatch.setattr("questionary.confirm", lambda *a, **k: _MockQuestionaryPrompt(True))

    _manage_custom_regions(cfg, custom_regions)
    assert "kruger_park" not in custom_regions
    assert cfg["gps_region"] == "europe"


def test_40_menu_custom_test_limits_invalid_string_and_zero_unlimited_edge_cases(monkeypatch):
    """Test 40: Entering 0 sets test limits to None (unlimited), and invalid text does not crash."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    cfg = {"interactive_menu": True, "AUD_TEST_LIMIT": 10, "GPS_TEST_LIMIT": 50}
    select_answers = iter([
        "🧪 Toggle Quick Test Limits vs. Full Run",
        "custom",
        "🧪 Toggle Quick Test Limits vs. Full Run",
        "custom",
        "🚀 Start Processing Pipeline",
    ])
    text_answers = iter([
        "0", "0",             # First pass: 0 -> should set both limits to None (unlimited)
        "not_a_number", "5",  # Second pass: invalid ValueError -> caught gracefully without crashing
    ])
    monkeypatch.setattr("questionary.select", lambda *a, **k: _MockQuestionaryPrompt(next(select_answers)))
    monkeypatch.setattr("questionary.text", lambda *a, **k: _MockQuestionaryPrompt(next(text_answers)))

    out = run_startup_menu(cfg)
    assert out["AUD_TEST_LIMIT"] is None
    assert out["GPS_TEST_LIMIT"] is None

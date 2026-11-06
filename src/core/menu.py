import os
import sys
import re
import json
import yaml
import questionary
from questionary import Style

from src.core.constants import FULL_APP_NAME, REGION_PRESETS
from src.core.run_reporter import DEFAULT_REPORT_COMPONENTS

CUSTOM_REGIONS_FILE = "./data/regions.json"


MENU_STYLE = Style([
    ("qmark",       "fg:#feb854 bold"),     # Pastel Orange
    ("question",    "fg:#ffffff bold"),     # White
    ("answer",      "fg:#e03c32 bold"),     # Bright Red
    ("pointer",     "fg:#fe626e bold"),     # Watermelon Pink
    ("highlighted", "fg:#64a4a4 bold"),     # Teal
    ("selected",    "fg:#64a4a4"),          # Teal
    ("separator",   "fg:#64a4a4"),          # Teal
    ("instruction", "fg:#8a6042 italic"),   # Brown
])


COMPONENT_LABELS = {
    "raw_stats": "Raw Data Statistics (File counts & success % per sensor)",
    "sessions_inventory": "Sessions Inventory (Time windows, durations & file ratios)",
    "data_yield": "Total Data Yield (True sample-derived IMU/Audio hours & GPS fixes)",
    "gps_analytics": "GPS / GNSS Analytics (3D/2D split, C/N0, HDOP, RMS & clock offset)",
    "imu_analytics": "IMU & Environmental Analytics (Effective Hz, Temp/Pres & glitches)",
    "audio_analytics": "Audio Bioacoustics (Bout count, dBFS levels, clipping % & dead-mic)",
    "quality_control": "Quality Control Log (Detailed list of warnings & critical failures)",
}


def load_custom_regions(filepath: str = None) -> dict:
    """Loads user-defined region bounding boxes from JSON."""
    filepath = filepath or CUSTOM_REGIONS_FILE
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                k: tuple(v)
                for k, v in data.items()
                if isinstance(v, (list, tuple)) and len(v) == 5
            }
    except Exception:
        return {}


def save_custom_regions(regions: dict, filepath: str = None):
    """Persists user-defined regions to JSON."""
    filepath = filepath or CUSTOM_REGIONS_FILE
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    serializable = {k: list(v) for k, v in regions.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def get_all_regions() -> dict:
    """Returns built-in REGION_PRESETS merged with user regions from data/regions.json."""
    merged = dict(REGION_PRESETS)
    merged.update(load_custom_regions())
    return merged


def save_config_yaml(config: dict, config_path: str = "config.yaml"):
    """
    Saves persistent user settings back to config.yaml using only the standard library.
    Preserves all original # comments, inline comment alignment, blank lines, and structure.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return

    def format_val(v):
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, str):
            return f'"{v}"'
        return str(v)

    for i, line in enumerate(lines):
        # 1. Match top-level keys (captures key, ': ', value, and trailing whitespace/comment)
        top_match = re.match(
            r"^([a-zA-Z0-9_]+)(\s*:\s*)([^#\r\n]+?)(\s*(?:#.*)?\r?\n?)$", line
        )
        if top_match:
            key, sep, _, trailing = top_match.groups()
            if (
                key in config
                and key != "report_components"
                and key not in ("AUD_TEST_LIMIT", "GPS_TEST_LIMIT")
            ):
                if not trailing.endswith("\n"):
                    trailing += "\n"
                lines[i] = f"{key}{sep}{format_val(config[key])}{trailing}"
                continue

        # 2. Match nested keys under report_components
        nested_match = re.match(
            r"^(\s+[a-zA-Z0-9_]+)(\s*:\s*)([^#\r\n]+?)(\s*(?:#.*)?\r?\n?)$", line
        )
        if nested_match and isinstance(config.get("report_components"), dict):
            indented_key, sep, _, trailing = nested_match.groups()
            sub_key = indented_key.strip()
            if sub_key in config["report_components"]:
                if not trailing.endswith("\n"):
                    trailing += "\n"
                lines[i] = (
                    f"{indented_key}{sep}{format_val(config['report_components'][sub_key])}{trailing}"
                )

    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _print_banner(config: dict, custom_regions: dict):
    os.system("cls" if os.name == "nt" else "clear")
    solver = str(config.get("gps_solver", "sdr")).upper()
    region = str(config.get("gps_region", "europe"))
    r_tag = " (Custom)" if region in custom_regions else ""
    rep_mode = str(config.get("report_mode", "verbose")).upper()

    aud_lim = config.get("AUD_TEST_LIMIT")
    gps_lim = config.get("GPS_TEST_LIMIT")
    if aud_lim or gps_lim:
        limit_str = f"TEST MODE (Audio: {aud_lim or 'All'}, GPS: {gps_lim or 'All'})"
    else:
        limit_str = "FULL PRODUCTION RUN (All Files)"

    print("=" * 76)
    print(f"{FULL_APP_NAME + ' 🐒 ⚙️ Pre-Flight Configuration':^76}")
    print("=" * 76)
    print(f"  Input Folder    : {config.get('raw_data_folder', './data/raw')}")
    print(f"  Output Folder   : {config.get('processed_folder', './data/processed')}")
    print(f"  Run Scope       : {limit_str}")
    print(f"  GPS Engine      : {solver}  |  Search Region: {region}{r_tag}")
    print(f"  Report Mode     : {rep_mode}")
    print("=" * 76)


def _show_help_guide():
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 76)
    print(f"{'VESPERFLOW — USER GUIDE & SETTINGS REFERENCE':^76}")
    print("=" * 76)
    print("\n1. INPUT & OUTPUT FOLDERS (raw_data_folder / processed_folder)")
    print("   - Input Folder: Directory containing your raw tag session folders (.BIN).")
    print("     VesperFlow scans recursively and merges 256-file hardware chunks.")
    print("   - Output Folder: Destination for decoded IMU (.csv), Audio (.wav),")
    print("     GPS tracks (.csv), sidecar metadata (.txt), and report cards.")

    print("\n2. RUN SCOPE (Full Run vs. Quick Test Limits)")
    print("   - Full Run processes every binary file found across all sessions.")
    print("   - Quick Test Limits cap Audio (e.g., 10 files) and GPS (e.g., 50 files)")
    print("     so you can verify settings in ~30 seconds before a multi-hour run.")

    print("\n3. GPS ENGINE (sdr vs. geotag)")
    print("   - sdr (Default): Native cross-platform Python GNSS SDR solver. Works on")
    print("     Linux, macOS, and Windows. Automatically downloads NASA/IGS satellite")
    print("     orbits on the first run for a deployment date and caches them offline.")
    print("   - geotag (Legacy): Calls the manufacturer's Windows-only GeoTag.exe.")

    print("\n4. GPS SEARCH REGION (gps_region)")
    print("   - Snapshot GPS records only fractional millisecond code phases. Providing")
    print("     a bounding box (continent, country, or study site) bounds the initial")
    print("     Doppler search, accelerating the first fix and preventing ambiguity.")
    print("   - You can add your own study sites in 'Manage Custom GPS Regions'.")

    print("\n5. REPORT CUSTOMIZATION (compact / verbose / custom)")
    print("   - compact : Prints only file counts, session inventory, yield, and errors.")
    print("   - verbose : Adds deep-dive GPS, IMU/Environment, and Audio analytics tables.")
    print("   - custom  : Lets you toggle each of the 7 report sections ON or OFF.")

    print("\n6. HEADLESS / UNIVERSITY SERVER & DOCKER MODE")
    print("   - Run with `python -m src.main --headless` (or set `interactive_menu: false`")
    print("     in config.yaml) to skip this menu on HPC clusters, SLURM, or Docker.")
    print("=" * 76)
    questionary.press_any_key_to_continue("Press any key to return to the menu...").ask()


def _manage_custom_regions(config: dict, custom_regions: dict):
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 76)
        print(f"{'CUSTOM GPS REGION MANAGER (data/regions.json)':^76}")
        print("=" * 76)
        if custom_regions:
            print("  Saved User Regions:")
            for name, bounds in custom_regions.items():
                print(
                    f"   • {name:<18} : Lat [{bounds[0]}, {bounds[1]}], "
                    f"Lon [{bounds[2]}, {bounds[3]}], Step {bounds[4]}°"
                )
        else:
            print("  (No custom user regions saved yet. Built-in presets are active.)")
        print("=" * 76)

        action = questionary.select(
            "Region Manager Options:",
            choices=[
                "Select Active GPS Region (Built-in + Custom)",
                "Add / Edit a Custom Region",
                "Delete a Custom Region",
                "Factory Reset Custom Regions",
                "<- Back to Main Menu",
            ],
            style=MENU_STYLE,
        ).ask()

        if action is None or action == "<- Back to Main Menu":
            break

        if action == "Select Active GPS Region (Built-in + Custom)":
            all_reg = get_all_regions()
            choices = [
                questionary.Choice(
                    title=f"{k:<18} ({v[0]}..{v[1]}°, {v[2]}..{v[3]}°, step {v[4]}°)"
                    + (" [CUSTOM]" if k in custom_regions else ""),
                    value=k,
                )
                for k, v in all_reg.items()
            ]
            selected = questionary.select(
                "Choose active GPS region:",
                choices=choices,
                default=config.get("gps_region", "europe")
                if config.get("gps_region") in all_reg
                else None,
                style=MENU_STYLE,
            ).ask()
            if selected:
                config["gps_region"] = selected

        elif action == "Add / Edit a Custom Region":
            name = questionary.text(
                "Enter region identifier (e.g., 'kruger_park' or 'dpz_gottingen'):",
                style=MENU_STYLE,
            ).ask()
            if not name or not name.strip():
                continue
            key = name.strip().lower().replace(" ", "_")

            def _ask_float(prompt_msg, default_val, lo, hi):
                def _val(text):
                    try:
                        v = float(text)
                        return True if lo <= v <= hi else f"Enter a value between {lo} and {hi}"
                    except ValueError:
                        return "Please enter a valid decimal number."

                res = questionary.text(
                    prompt_msg, default=str(default_val), validate=_val, style=MENU_STYLE
                ).ask()
                return float(res) if res is not None else None

            existing = custom_regions.get(key, (-10.0, 10.0, -10.0, 10.0, 0.25))
            lat_min = _ask_float("Minimum Latitude (-90 to 90):", existing[0], -90.0, 90.0)
            if lat_min is None:
                continue
            lat_max = _ask_float("Maximum Latitude (-90 to 90):", existing[1], lat_min, 90.0)
            if lat_max is None:
                continue
            lon_min = _ask_float("Minimum Longitude (-180 to 180):", existing[2], -180.0, 180.0)
            if lon_min is None:
                continue
            lon_max = _ask_float("Maximum Longitude (-180 to 180):", existing[3], lon_min, 180.0)
            if lon_max is None:
                continue
            default_step = round(
                max(0.005, min(lat_max - lat_min, lon_max - lon_min) / 20.0), 3
            )
            step = _ask_float("Grid Step in Degrees (0.001 to 5.0):", default_step, 0.001, 5.0)
            if step is None:
                continue

            custom_regions[key] = (lat_min, lat_max, lon_min, lon_max, step)
            save_custom_regions(custom_regions)
            config["gps_region"] = key

        elif action == "Delete a Custom Region":
            if not custom_regions:
                questionary.press_any_key_to_continue("No custom regions to delete.").ask()
                continue
            to_del = questionary.select(
                "Select custom region to delete:",
                choices=list(custom_regions.keys()) + ["Cancel"],
                style=MENU_STYLE,
            ).ask()
            if to_del and to_del != "Cancel":
                if questionary.confirm(
                    f"Delete custom region '{to_del}'?", default=False, style=MENU_STYLE
                ).ask():
                    custom_regions.pop(to_del, None)
                    save_custom_regions(custom_regions)
                    if config.get("gps_region") == to_del:
                        config["gps_region"] = "europe"

        elif action == "Factory Reset Custom Regions":
            if not custom_regions:
                questionary.press_any_key_to_continue(
                    "Custom regions are already empty (at factory defaults)."
                ).ask()
                continue
            confirmed = questionary.confirm(
                "WARNING: This will permanently delete all custom regions in data/regions.json "
                "and restore built-in defaults. Are you sure?",
                default=False,
                style=MENU_STYLE,
            ).ask()
            if confirmed:
                custom_regions.clear()
                save_custom_regions(custom_regions)
                if config.get("gps_region") not in REGION_PRESETS:
                    config["gps_region"] = "europe"


def _customize_report_menu(config: dict):
    mode = questionary.select(
        "Select Processing Report Mode:",
        choices=[
            questionary.Choice(
                "verbose — Full scientific report (all 7 sections & analytics tables)",
                value="verbose",
            ),
            questionary.Choice(
                "compact — Summary only (file counts, session inventory, yield & errors)",
                value="compact",
            ),
            questionary.Choice(
                "custom  — Pick and choose individual report sections...",
                value="custom",
            ),
        ],
        default=config.get("report_mode", "verbose"),
        style=MENU_STYLE,
    ).ask()

    if not mode:
        return
    config["report_mode"] = mode

    if mode == "custom":
        current_comps = config.get("report_components") or dict(DEFAULT_REPORT_COMPONENTS)
        choices = [
            questionary.Choice(
                title=label,
                value=key,
                checked=bool(current_comps.get(key, True)),
            )
            for key, label in COMPONENT_LABELS.items()
        ]
        selected_keys = questionary.checkbox(
            "Toggle Report Components (Spacebar to toggle, Enter to confirm):",
            choices=choices,
            style=MENU_STYLE,
        ).ask()

        if selected_keys is not None:
            config["report_components"] = {
                k: (k in selected_keys) for k in DEFAULT_REPORT_COMPONENTS
            }


def run_startup_menu(config: dict, config_path: str = "config.yaml") -> dict:
    """
    Runs the interactive pre-flight menu unless bypassed by:
      1. CLI flag (--headless / --no-menu / -y)
      2. config.yaml setting (interactive_menu: false)
      3. Non-interactive environment (Docker / SLURM / SSH batch where stdin is not a TTY)
    """
    cli_flags = set(sys.argv[1:])
    if (
        "--headless" in cli_flags
        or "--no-menu" in cli_flags
        or "-y" in cli_flags
        or not config.get("interactive_menu", True)
        or not sys.stdin.isatty()
    ):
        return config

    custom_regions = load_custom_regions()

    current_selection = "🚀 Start Processing Pipeline"

    while True:
        _print_banner(config, custom_regions)
        choice = questionary.select(
            "Select an action:",
            choices=[
                "🚀 Start Processing Pipeline",
                "📂 Change Input / Output Paths",
                "🧪 Toggle Quick Test Limits vs. Full Run",
                "📡 Change GPS Engine or Search Region",
                "📍 Manage Custom GPS Regions (Add / Edit / Delete / Reset)",
                "📜 Customize Processing Report (Compact / Verbose / Custom)",
                "💾 Save Current Settings to config.yaml",
                "🔄 Factory Reset config.yaml",
                "📖 Help & Settings Reference Guide",
                "🚪 Exit VesperFlow",
            ],
            default=current_selection,
            style=MENU_STYLE,
        ).ask()

        if choice is None or choice == "🚪 Exit VesperFlow":
            print("Exiting VesperFlow.")
            sys.exit(0)

        current_selection = choice

        if choice == "🚀 Start Processing Pipeline":
            break

        elif choice == "📂 Change Input / Output Paths":
            raw_p = questionary.text(
                "Raw input data folder:",
                default=config.get("raw_data_folder", "./data/raw"),
                style=MENU_STYLE,
            ).ask()
            if raw_p:
                config["raw_data_folder"] = raw_p.strip()

            proc_p = questionary.text(
                "Processed output folder:",
                default=config.get("processed_folder", "./data/processed"),
                style=MENU_STYLE,
            ).ask()
            if proc_p:
                config["processed_folder"] = proc_p.strip()

        elif choice == "🧪 Toggle Quick Test Limits vs. Full Run":
            scope = questionary.select(
                "Choose pipeline run scope:",
                choices=[
                    questionary.Choice("Full Production Run (Process all files)", value="full"),
                    questionary.Choice(
                        "Quick Test Mode (10 Audio + 50 GPS files per tag)", value="quick"
                    ),
                    questionary.Choice("Custom File Limits...", value="custom"),
                ],
                style=MENU_STYLE,
            ).ask()
            if scope == "full":
                config["AUD_TEST_LIMIT"] = None
                config["GPS_TEST_LIMIT"] = None
            elif scope == "quick":
                config["AUD_TEST_LIMIT"] = 10
                config["GPS_TEST_LIMIT"] = 50
            elif scope == "custom":
                a_lim = questionary.text(
                    "Max Audio files per session (0 for unlimited):",
                    default="10",
                    style=MENU_STYLE,
                ).ask()
                g_lim = questionary.text(
                    "Max GPS files per session (0 for unlimited):",
                    default="50",
                    style=MENU_STYLE,
                ).ask()
                try:
                    config["AUD_TEST_LIMIT"] = int(a_lim) if int(a_lim) > 0 else None
                    config["GPS_TEST_LIMIT"] = int(g_lim) if int(g_lim) > 0 else None
                except (ValueError, TypeError):
                    pass

        elif choice == "📡 Change GPS Engine or Search Region":
            eng = questionary.select(
                "Select GPS Solver Engine:",
                choices=[
                    questionary.Choice(
                        "sdr    — Native Cross-Platform Python GNSS Solver (Recommended)",
                        value="sdr",
                    ),
                    questionary.Choice(
                        "geotag — Legacy Windows GeoTag.exe Wrapper", value="geotag"
                    ),
                ],
                default=config.get("gps_solver", "sdr"),
                style=MENU_STYLE,
            ).ask()
            if eng:
                config["gps_solver"] = eng

            all_reg = get_all_regions()
            reg_choices = [
                questionary.Choice(
                    title=f"{k:<18} ({v[0]}..{v[1]}°, {v[2]}..{v[3]}°)"
                    + (" [CUSTOM]" if k in custom_regions else ""),
                    value=k,
                )
                for k, v in all_reg.items()
            ]
            reg = questionary.select(
                "Select GPS Search Region:",
                choices=reg_choices,
                default=config.get("gps_region", "europe")
                if config.get("gps_region") in all_reg
                else None,
                style=MENU_STYLE,
            ).ask()
            if reg:
                config["gps_region"] = reg

        elif choice == "📍 Manage Custom GPS Regions (Add / Edit / Delete / Reset)":
            _manage_custom_regions(config, custom_regions)

        elif choice == "📜 Customize Processing Report (Compact / Verbose / Custom)":
            _customize_report_menu(config)

        elif choice == "💾 Save Current Settings to config.yaml":
            confirm = questionary.text(
                "WARNING: Overwrite config.yaml with current changes? (y/n):",
                style=MENU_STYLE
            ).ask()

            if confirm and confirm.lower() in ("yes", "y"):
                save_config_yaml(config, config_path)
                questionary.text(
                    f"Settings saved to {config_path}! Press [ENTER] to continue...",
                    default="",
                    style=MENU_STYLE
                ).ask()
            else:
                questionary.text(
                    "Saving canceled! Press [ENTER] to continue...",
                    default="",
                    style=MENU_STYLE
                ).ask()




        elif choice == "🔄 Factory Reset config.yaml":
            confirm = questionary.text(
                "WARNING: This overrides config.yaml with defaults. Type 'RESET CONFIG' to confirm:",
                style=MENU_STYLE
            ).ask()

            if confirm == "RESET CONFIG":
                default_cfg = {
                    "raw_data_folder": "./data/raw",
                    "processed_folder": "./data/processed",
                    "ephemeris_folder": "./data/ephemeris",
                    "interactive_menu": True,
                    "log_level": "INFO",
                    "report_mode": "verbose",
                    "report_components": dict(DEFAULT_REPORT_COMPONENTS),
                    "gps_solver": "sdr",
                    "gps_region": "europe",
                    "gps_cli_path": "./external_tools/CG/GeoTag/GeoTag.exe",
                    "gps_engine_path": "./external_tools/CG/GeoTagEngine/GeoTagEngine.exe",
                }


                config.clear()
                config.update(default_cfg)
                save_config_yaml(config, config_path)
                questionary.press_any_key_to_continue(
                    "Factory reset complete! config.yaml has been restored. Press any key..."
                ).ask()
            else:
                questionary.press_any_key_to_continue(
                    "Reset canceled (invalid confirmation). Press any key..."
                ).ask()

        elif choice == "📖 Help & Settings Reference Guide":
            _show_help_guide()

    return config

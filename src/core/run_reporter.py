import os
from datetime import datetime
from typing import Any

from src.core.constants import FULL_APP_NAME

DEFAULT_REPORT_COMPONENTS = {
    "raw_stats": True,
    "sessions_inventory": True,
    "data_yield": True,
    "gps_analytics": True,
    "imu_analytics": True,
    "audio_analytics": True,
    "quality_control": True,
}


class RunReporter:
    """
    Manages execution statistics and generates the final Processing Report.
    Supports 'compact', 'verbose', and 'custom' component-level toggling.
    """

    def __init__(self, report_mode="verbose", custom_components=None):
        self.start_time = datetime.now()
        self.report_mode = (report_mode or "verbose").strip().lower()
        self.components = self._resolve_components(self.report_mode, custom_components)

        self.stats: dict[str, Any] = {
            "total": 0,
            "total_imu": 0, "success_imu": 0, "warn_imu": 0, "failed_imu": 0,
            "total_aud": 0, "success_aud": 0, "warn_aud": 0, "failed_aud": 0,
            "total_gps": 0, "success_gps": 0, "warn_gps": 0, "failed_gps": 0,
            "duration_imu_sec": 0.0,
            "duration_aud_sec": 0.0,
            "gps_fixes": 0,
            "sessions": [],
            "errors": [],
        }

    @staticmethod
    def _resolve_components(mode: str, custom: dict | None = None) -> dict:
        if custom == None:
            custom = {}
        if mode == "compact":
            return {
                "raw_stats": True,
                "sessions_inventory": True,
                "data_yield": True,
                "gps_analytics": False,
                "imu_analytics": False,
                "audio_analytics": False,
                "quality_control": True,
            }
        if mode == "custom" and isinstance(custom, dict):
            return {k: bool(custom.get(k, v)) for k, v in DEFAULT_REPORT_COMPONENTS.items()}
        return dict(DEFAULT_REPORT_COMPONENTS)

    def log_file_result(self, sensor_type, status, filepath, msg=None):
        s_key = sensor_type.lower()
        self.stats[f"total_{s_key}"] += 1
        self.stats["total"] += 1

        if status == "SUCCESS":
            self.stats[f"success_{s_key}"] += 1
        elif status == "EMPTY":
            self.stats[f"warn_{s_key}"] += 1
            self.stats["errors"].append({
                "type": "WARN",
                "file": filepath,
                "reason": msg or "File Empty",
            })
        else:
            self.stats[f"failed_{s_key}"] += 1
            self.stats["errors"].append({
                "type": "CRITICAL",
                "file": filepath,
                "reason": msg or "Processing Failed",
            })

    def add_session(self, session_metrics):
        self.stats["sessions"].append(session_metrics)
        self.stats["gps_fixes"] += session_metrics.get("gps_fixes", 0)
        self.stats["duration_imu_sec"] += session_metrics.get("duration_imu_sec", 0.0)
        self.stats["duration_aud_sec"] += session_metrics.get("aud_duration", 0.0)

    def log_external_error(self, reason):
        self.stats["errors"].append({
            "type": "CRITICAL",
            "file": "External Tool",
            "reason": reason,
        })

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        m, s = divmod(int(round(seconds)), 60)
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"

    def save_report(self, processed_folder, logger):
        end_time = datetime.now()
        run_secs = (end_time - self.start_time).total_seconds()
        rm, rs = divmod(int(run_secs), 60)
        run_str = f"{rm:02d}m{rs:02d}s"
        date_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append("=" * 100)
        title = f"{FULL_APP_NAME.upper()} - PROCESSING REPORT ({self.report_mode.upper()})"
        lines.append(f"{title:^100}")
        lines.append("=" * 100)
        header_meta = f"Date: {date_str}  |  Total Files Found: {self.stats['total']}  |  Run Time: {run_str}"
        lines.append(f"{header_meta:^100}")
        lines.append("")

        # ------------------------------------------------------------------
        # 1. RAW DATA STATISTICS
        # ------------------------------------------------------------------
        if self.components.get("raw_stats", True):
            lines.append("RAW DATA STATISTICS:")
            lines.append("-" * 59)
            lines.append("| SENSOR   | TOTAL  | OK     | WARN   | FAIL   | RATE (%) |")
            lines.append("-" * 59)
            for label, key in [("IMU", "imu"), ("AUDIO", "aud"), ("GPS", "gps")]:
                tot = self.stats[f"total_{key}"]
                ok = self.stats[f"success_{key}"]
                warn = self.stats[f"warn_{key}"]
                fail = self.stats[f"failed_{key}"]
                rate = (ok / tot * 100.0) if tot > 0 else 0.0
                lines.append(
                    f"| {label:<8} | {tot:<6} | {ok:<6} | {warn:<6} | {fail:<6} | {rate:>6.1f}%  |"
                )
            lines.append("-" * 59)
            tot_all = self.stats["total"]
            ok_all = self.stats["success_imu"] + self.stats["success_aud"] + self.stats["success_gps"]
            warn_all = self.stats["warn_imu"] + self.stats["warn_aud"] + self.stats["warn_gps"]
            fail_all = self.stats["failed_imu"] + self.stats["failed_aud"] + self.stats["failed_gps"]
            rate_all = (ok_all / tot_all * 100.0) if tot_all > 0 else 0.0
            lines.append(
                f"| {'ALL':<8} | {tot_all:<6} | {ok_all:<6} | {warn_all:<6} | {fail_all:<6} | {rate_all:>6.1f}%  |"
            )
            lines.append("-" * 59)
            lines.append("")

        # ------------------------------------------------------------------
        # 2. SESSIONS INVENTORY & METRICS
        # ------------------------------------------------------------------
        if self.components.get("sessions_inventory", True) and self.stats["sessions"]:
            lines.append("SESSIONS INVENTORY & METRICS:")
            lines.append("-" * 105)
            lines.append(
                "| DEVICE ID  | WINDOW (Start -> End)                  | IMU (h) | AUD (h) | GPS (Fix/Try) | I/A/G       |"
            )
            lines.append("-" * 105)
            for s in self.stats["sessions"]:
                dev_id = str(s.get("id", "Unknown"))[:10]
                st, et = s.get("start_time"), s.get("end_time")
                if st and et:
                    win_str = f"{st.strftime('%Y-%m-%d %H:%M')} -> {et.strftime('%Y-%m-%d %H:%M')}"
                else:
                    win_str = "N/A"
                imu_h = s.get("duration_imu_sec", 0.0) / 3600.0
                aud_h = s.get("aud_duration", 0.0) / 3600.0
                gps_str = f"{s.get('gps_fixes', 0)}/{s.get('gps_attempts', 0)}"
                iag_str = f"{s.get('imu_ok', 0)}/{s.get('aud_ok', 0)}/{s.get('gps_ok', 0)}"
                lines.append(
                    f"| {dev_id:<10} | {win_str:<38} | {imu_h:<7.1f} | {aud_h:<7.1f} | {gps_str:<13} | {iag_str:<11} |"
                )
            lines.append("-" * 105)
            lines.append("")

        # ------------------------------------------------------------------
        # 3. TOTAL DATA YIELD
        # ------------------------------------------------------------------
        if self.components.get("data_yield", True):
            lines.append("TOTAL DATA YIELD:")
            lines.append(f"   > Total IMU Duration:   {self._fmt_duration(self.stats['duration_imu_sec'])}")
            lines.append(f"   > Total Audio Duration: {self._fmt_duration(self.stats['duration_aud_sec'])}")
            lines.append(f"   > Total GPS Fixes:      {self.stats['gps_fixes']} valid coordinates")
            lines.append("")

        # ------------------------------------------------------------------
        # 4. GPS / GNSS SIGNAL & TRAJECTORY ANALYTICS
        # ------------------------------------------------------------------
        if self.components.get("gps_analytics", True) and any(s.get("gps_fixes", 0) > 0 for s in self.stats["sessions"]):
            lines.append("GPS / GNSS SIGNAL & TRAJECTORY ANALYTICS:")
            lines.append("-" * 108)
            lines.append(
                "| DEVICE ID  | YIELD (%) | 3D / 2D FIXES | MEAN C/N0  | MED HDOP | MED RMS [m] | ALTITUDE [m] | RTC OFFSET |"
            )
            lines.append("-" * 108)
            for s in self.stats["sessions"]:
                dev_id = str(s.get("id", "Unknown"))[:10]
                tries = s.get("gps_attempts", 0)
                fixes = s.get("gps_fixes", 0)
                yield_pct = (fixes / tries * 100.0) if tries > 0 else 0.0
                g_stat = s.get("gps_analytics", {})
                if fixes > 0 and g_stat:
                    f3d = g_stat.get("fixes_3d", 0)
                    f2d = g_stat.get("fixes_2d", 0)
                    fix_split = f"{f3d} / {f2d}"
                    cn0_str = f"{g_stat.get('mean_cn0', 0.0):.1f} dB"
                    hdop_str = f"{g_stat.get('med_hdop', 0.0):.1f}"
                    rms_str = f"{g_stat.get('med_rms', 0.0):.1f} m"
                    alt_str = f"{g_stat.get('med_alt', 0.0):.0f} m"
                    clk_str = f"{g_stat.get('med_clock_s', 0.0):+.2f} s"
                else:
                    fix_split, cn0_str, hdop_str, rms_str, alt_str, clk_str = "0 / 0", "N/A", "N/A", "N/A", "N/A", "N/A"
                lines.append(
                    f"| {dev_id:<10} | {yield_pct:>7.1f}%  | {fix_split:<13} | {cn0_str:<10} | {hdop_str:<8} | {rms_str:<11} | {alt_str:<12} | {clk_str:<10} |"
                )
            lines.append("-" * 108)
            lines.append("")

        # ------------------------------------------------------------------
        # 5. IMU & ENVIRONMENTAL HEALTH ANALYTICS
        # ------------------------------------------------------------------
        if self.components.get("imu_analytics", True) and any(s.get("imu_ok", 0) > 0 for s in self.stats["sessions"]):
            lines.append("IMU & ENVIRONMENTAL HEALTH ANALYTICS:")
            lines.append("-" * 100)
            lines.append(
                "| DEVICE ID  | SAMPLES    | EFF. RATE   | TEMP RANGE [C]    | PRESSURE [hPa]    | GLITCHES MASKED  |"
            )
            lines.append("-" * 100)
            for s in self.stats["sessions"]:
                dev_id = str(s.get("id", "Unknown"))[:10]
                i_stat = s.get("imu_analytics", {})
                if s.get("imu_ok", 0) > 0 and i_stat:
                    n_samp = f"{i_stat.get('total_samples', 0):,}"
                    eff_hz = f"{i_stat.get('effective_hz', 0.0):.2f} Hz"
                    t_range = i_stat.get("temp_range_str", "N/A (Off)")
                    p_range = i_stat.get("pres_range_str", "N/A (Off)")
                    glitches = f"{i_stat.get('glitches_masked', 0)} gyro / {i_stat.get('dups_dropped', 0)} dup"
                else:
                    n_samp, eff_hz, t_range, p_range, glitches = "0", "N/A", "N/A", "N/A", "0"
                lines.append(
                    f"| {dev_id:<10} | {n_samp:<10} | {eff_hz:<11} | {t_range:<17} | {p_range:<17} | {glitches:<16} |"
                )
            lines.append("-" * 100)
            lines.append("")

        # ------------------------------------------------------------------
        # 6. AUDIO BIOACOUSTICS & SIGNAL INTEGRITY
        # ------------------------------------------------------------------
        if self.components.get("audio_analytics", True) and any(s.get("aud_ok", 0) > 0 for s in self.stats["sessions"]):
            lines.append("AUDIO BIOACOUSTICS & SIGNAL INTEGRITY:")
            lines.append("-" * 100)
            lines.append(
                "| DEVICE ID  | BOUTS | MEAN FILE | MEAN LEVEL   | PEAK LEVEL   | CLIPPING (%)   | DEAD-MIC FILES   |"
            )
            lines.append("-" * 100)
            for s in self.stats["sessions"]:
                dev_id = str(s.get("id", "Unknown"))[:10]
                a_stat = s.get("audio_analytics", {})
                if s.get("aud_ok", 0) > 0 and a_stat:
                    bouts = str(a_stat.get("bouts", 1))
                    mean_len = f"{a_stat.get('mean_file_sec', 0.0):.1f} s"
                    rms_db = f"{a_stat.get('mean_dbfs', -99.0):.1f} dBFS"
                    peak_db = f"{a_stat.get('peak_dbfs', -99.0):.1f} dBFS"
                    clip_pct = f"{a_stat.get('clipping_pct', 0.0):.4f}%"
                    dead_cnt = f"{a_stat.get('dead_mic_files', 0)} files"
                else:
                    bouts, mean_len, rms_db, peak_db, clip_pct, dead_cnt = "0", "N/A", "N/A", "N/A", "0.0000%", "0 files"
                lines.append(
                    f"| {dev_id:<10} | {bouts:<5} | {mean_len:<9} | {rms_db:<12} | {peak_db:<12} | {clip_pct:<14} | {dead_cnt:<16} |"
                )
            lines.append("-" * 100)
            lines.append("")

        lines.append("=" * 100)

        # ------------------------------------------------------------------
        # 7. QUALITY CONTROL (CRITICAL FAILURES & WARNINGS)
        # ------------------------------------------------------------------
        if self.components.get("quality_control", True):
            errors = [e for e in self.stats["errors"] if e.get("type") == "CRITICAL"]
            if errors:
                lines.append(f"{'[X] CRITICAL FAILURES (Action Required)':^100}")
                lines.append("-" * 100)
                for err in errors:
                    lines.append(f"   {err['reason']}")
                    if "file" in err and err["file"] != "External Tool":
                        lines.append(f"      File: {err['file']}")
                lines.append("=" * 100)

            warnings = [e for e in self.stats["errors"] if e.get("type") == "WARN"]
            if warnings:
                lines.append(f"{'[!] WARNINGS (Empty or Interrupted Files)':^100}")
                lines.append("-" * 100)
                for w in warnings[:25]:
                    lines.append(f"   {w['reason']} -> {w.get('file', '')}")
                if len(warnings) > 25:
                    lines.append(f"   ... and {len(warnings) - 25} additional warnings.")
                lines.append("=" * 100)

        lines.append("END OF REPORT")

        # Write to file and log to console
        report_dir = os.path.join(processed_folder, "report_cards")
        os.makedirs(report_dir, exist_ok=True)
        report_filename = f"Report_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(report_dir, report_filename)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        for line in lines:
            if "[X] CRITICAL FAILURES" in line:
                logger.error(line)
            else:
                logger.info(line)

        logger.info(f"[Report] Saved to: {report_path}")

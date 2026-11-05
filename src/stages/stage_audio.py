import numpy as np
from tqdm import tqdm
from datetime import timedelta

from src.parsers.audio_parser import parse_audio_file

def process_audio_session(session_id, aud_files, exporter, reporter, sess_metrics, session_device_id, extract_sequence_index, master_pbar, logger, AUD_TEST_LIMIT):
    aud_files.sort(key=extract_sequence_index)
    if AUD_TEST_LIMIT:
        aud_files = aud_files[:AUD_TEST_LIMIT]

    sess_metrics["files_total"] += len(aud_files)

    aud_total_samples = 0
    aud_clipped_samples = 0
    aud_sum_sq = 0.0
    aud_max_peak = 0.0
    aud_dead_files = 0
    aud_bouts = 0
    prev_aud_end = None

    if not aud_files:
        logger.warning(f"No audio binary files found in {session_id}")
        return session_device_id

    logger.info(f"Starting Audio Parser on {len(aud_files)} binary files...")

    for filepath in tqdm(
        aud_files,
        desc=f"🔊 Audio ({session_id})",
        unit="file",
        position=0,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
    ):
        try:
            status, msg, meta, audio_data, timestamps = parse_audio_file(filepath)

            if status == "SUCCESS":
                sess_metrics["aud_ok"] += 1

                if session_device_id is None and meta:
                    session_device_id = meta.get("DeviceID", "UnknownTag")
                    sess_metrics["id"] = session_device_id

                if meta and audio_data is not None:
                    duration = len(audio_data) / meta["SampleRate"]
                    end_time = meta["Start_Time"] + timedelta(seconds=duration)
                    sess_metrics["aud_duration"] += duration

                    if sess_metrics["start_time"] is None or meta["Start_Time"] < sess_metrics["start_time"]:
                        sess_metrics["start_time"] = meta["Start_Time"]
                    if sess_metrics["end_time"] is None or end_time > sess_metrics["end_time"]:
                        sess_metrics["end_time"] = end_time

                    if prev_aud_end is None or (meta["Start_Time"] - prev_aud_end).total_seconds() > 300:
                        aud_bouts += 1
                    prev_aud_end = end_time

                    arr_f = audio_data.astype(np.float64)
                    n_s = len(arr_f)
                    aud_total_samples += n_s
                    aud_clipped_samples += int(np.sum(np.abs(arr_f) >= 32760))
                    file_sq = float(np.sum(arr_f ** 2))
                    aud_sum_sq += file_sq
                    aud_max_peak = max(aud_max_peak, float(np.max(np.abs(arr_f))) if n_s > 0 else 0.0)

                    file_rms = np.sqrt(file_sq / max(n_s, 1))
                    if 20.0 * np.log10(max(file_rms / 32768.0, 1e-9)) < -75.0:
                        aud_dead_files += 1

                    exporter.save_aud_wav(audio_data, meta)
                    exporter.save_session_metadata(meta, end_time=end_time, time_stamps=timestamps)

            reporter.log_file_result("aud", status, filepath, msg)
        except Exception as e:
            reporter.log_file_result("aud", "FAIL", filepath, f"Unexpected loop crash: {e}")
        finally:
            master_pbar.update(1)

    if sess_metrics["aud_ok"] > 0 and aud_total_samples > 0:
        global_rms = np.sqrt(aud_sum_sq / aud_total_samples)
        sess_metrics["audio_analytics"] = {
            "bouts": aud_bouts,
            "mean_file_sec": sess_metrics["aud_duration"] / sess_metrics["aud_ok"],
            "mean_dbfs": 20.0 * np.log10(max(global_rms / 32768.0, 1e-9)),
            "peak_dbfs": 20.0 * np.log10(max(aud_max_peak / 32768.0, 1e-9)),
            "clipping_pct": (aud_clipped_samples / aud_total_samples) * 100.0,
            "dead_mic_files": aud_dead_files,
        }

    logger.info(f"Saved {sess_metrics['aud_ok']} Audio .WAV files for {session_id}")
    return session_device_id

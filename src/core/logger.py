import itertools
import logging
import os
import sys
import threading
import time
from datetime import datetime
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that uses tqdm.write to
    ensure log messages don't break the progress bar.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            # tqdm.write prints safely above the progress bar
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class AnimatedTqdm(tqdm):
    """A tqdm progress bar with a flicker-free Braille spinner for the main pipeline bar."""
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, *args, **kwargs):
        self._base_desc = kwargs.get("desc", "Processing")
        self._stop_spin = threading.Event()
        # Hide terminal cursor to prevent row-jumping flicker on multi-line bars
        sys.stderr.write("\033[?25l")
        sys.stderr.flush()
        super().__init__(*args, **kwargs)
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _spin_loop(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_spin.is_set():
                break
            try:
                with self.get_lock():
                    self.desc = f"{frame} {self._base_desc}"
                    self.refresh(nolock=True)
            except Exception:
                break
            time.sleep(0.25)

    def close(self):
        if hasattr(self, "_stop_spin"):
            self._stop_spin.set()
        super().close()
        # Restore terminal cursor when finished
        sys.stderr.write("\033[?25h")
        sys.stderr.flush()


def setup_logger(name, log_dir="."):
    """Sets up a logger that writes to console and a file."""

    # --- 1. RENAME LEVELS ---
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.CRITICAL, "CRIT")
    logging.addLevelName(logging.ERROR, "ERR")
    logging.addLevelName(logging.DEBUG, "DBUG")

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log filename based on time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"vesperflow_run_{timestamp}.log")

    # Define the format
    log_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-4s | %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )

    # File handler (writes to disk)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_formatter)

    # TQDM console handler (writes to terminal safely)
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

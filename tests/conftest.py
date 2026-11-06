import os
import sys
import logging
import pytest

# Ensure project root is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@pytest.fixture(autouse=True)
def prevent_screen_clear(monkeypatch):
    """Prevents os.system('clear'/'cls') in menu.py from wiping pytest output."""
    monkeypatch.setattr(os, "system", lambda *args, **kwargs: 0)


@pytest.fixture
def dummy_logger():
    logger = logging.getLogger("vesperflow_test")
    logger.setLevel(logging.CRITICAL)
    return logger


@pytest.fixture
def synthetic_eph():
    """Returns a realistic broadcast ephemeris record for GPS PRN G01."""
    return {
        "af0": -1.2e-4,
        "af1": 1.5e-11,
        "af2": 0.0,
        "iode": 45.0,
        "crs": -18.5,
        "deltan": 4.8e-9,
        "m0": 0.85,
        "cuc": -9.5e-7,
        "e": 0.0085,
        "cus": 6.2e-6,
        "sqrtA": 5153.65,
        "toe": 129600.0,
        "cic": 3.5e-8,
        "omega0": 1.42,
        "cis": -4.2e-8,
        "i0": 0.96,
        "crc": 245.0,
        "omega": -1.15,
        "omegadot": -8.1e-9,
        "idot": 2.1e-10,
        "week": 2386.0,
        "sv_health": 0.0,
        "tgd": -1.0e-8,
    }

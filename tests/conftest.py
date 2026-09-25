"""Shared pytest fixtures.

The platform keeps all state under ``data_store/`` and caches its Settings and DB
engine as process-wide singletons. For tests we want each test to run against a
*fresh, throwaway* store so they don't clobber real data or each other.

``isolated_store`` points ``FTP_DATA_DIR`` at a tmp dir, disables the simulated
trainer's per-step sleep (so training is instant), and resets the cached Settings
+ DB engine so they're rebuilt against the tmp store.
"""

from __future__ import annotations

import pytest

import ftp.db.session as db_session
from ftp.config import get_settings


def _reset_singletons() -> None:
    get_settings.cache_clear()
    db_session._engine = None
    db_session._SessionLocal = None


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("FTP_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("FTP_SIM_STEP_DELAY", "0")  # instant simulated training
    _reset_singletons()
    yield
    _reset_singletons()

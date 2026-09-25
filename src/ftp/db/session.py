"""Database engine + session management.

We use SQLite via SQLAlchemy 2.0. SQLite is a deliberate choice for a local-first
platform (ADR 0004): zero setup, a single file under data_store/, and more than
enough for a single-node control plane that stores metadata (datasets, jobs,
models) — not bulk data.

Swapping to Postgres later is a one-line URL change because we never use SQLite-
specific SQL.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ftp.config import get_settings

from .models import Base

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def _init() -> None:
    """Lazily create the engine + tables on first use."""
    global _engine, _SessionLocal
    if _engine is not None:
        return
    settings = get_settings()
    settings.ensure_dirs()
    # check_same_thread=False: FastAPI may touch the session from different
    # threads (e.g. the sync background job runner). Safe here because we scope a
    # fresh Session per unit of work rather than sharing one across threads.
    _engine = create_engine(
        f"sqlite:///{settings.db_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def get_engine():
    _init()
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error, always close.

    Usage:
        with session_scope() as s:
            s.add(obj)
    """
    _init()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yields a session, commits, closes.

    FastAPI calls this per-request via Depends(get_db).
    """
    _init()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

"""Database session management and SQLite connection lifecycle.

Configures SQLAlchemy engine with SQLite-specific WAL mode, foreign keys,
and connection pooling suitable for single-host concurrent access.
"""

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings


def _set_sqlite_pragmas(dbapi_connection, connection_record) -> None:  # noqa: ANN001
    """Enforces SQLite pragmas for concurrency, integrity, and performance."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=5000;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.close()


def get_engine(database_url: str | None = None) -> Engine:
    """Creates and configures an SQLAlchemy Engine for SQLite."""
    url = database_url or get_settings().database_url

    # Ensure parent directory exists for file-based SQLite databases
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        db_path = Path(url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}

    engine = create_engine(
        url,
        connect_args=connect_args,
        echo=False,
    )

    if url.startswith("sqlite"):
        event.listen(engine, "connect", _set_sqlite_pragmas)

    return engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Returns a sessionmaker configured with the provided or default engine."""
    eng = engine or get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=eng)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    configured_settings = get_settings()
    engine = get_engine(configured_settings.database_url)
    session_factory = get_session_factory(engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def now_utc() -> datetime:
    """Helper returning timezone-aware UTC datetime."""
    return datetime.now(UTC)

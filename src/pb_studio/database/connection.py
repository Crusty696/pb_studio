"""
Database-Verbindungs-Manager für PB_studio

Verwaltet Verbindungen zu SQLite (Persistenz) und DuckDB (Analytics).
Dual-Database-Architektur wie in Projektvorgaben definiert.

Security Fixes:
- K-01: Parametrisierte Queries fuer DuckDB (SQL Injection Prevention)
- H-01: Thread-safe Singleton-Initialisierung
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None  # type: ignore

from ..core.config import get_config
from ..utils.logger import get_logger

logger = get_logger()


# SQLite connection optimization
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Aktiviert WAL-Modus und Foreign Keys für SQLite.

    WAL (Write-Ahead Logging) verbessert Parallelität und Performance.
    """
    if hasattr(dbapi_connection, "execute"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        # PERF-FIX: Verhindert "database locked" Fehler bei concurrent access
        cursor.execute("PRAGMA busy_timeout=30000")  # 30s timeout
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.close()


class DatabaseManager:
    """Manager für SQLite-Datenbankverbindungen."""

    def __init__(self, db_path: str | None = None):
        """
        Initialisiert den Database Manager.

        Args:
            db_path: Pfad zur SQLite-Datenbank (optional)
        """
        config = get_config()

        if db_path is None:
            db_path = config.get("Database", "sqlite_path", "data/project.db")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # SQLAlchemy Engine mit Threading-Support (Phase 2)
        # PERF-OPT: Explicit connection pooling for 2-5% faster queries
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,  # Set to True for SQL debugging
            future=True,
            # BUG-FIX: Add SERIALIZABLE isolation level for better transaction safety
            isolation_level="SERIALIZABLE",
            # PERF-OPT: Connection pooling configuration
            poolclass=QueuePool,
            pool_size=5,  # Max 5 connections in pool
            max_overflow=10,  # Allow 10 overflow connections
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,  # Recycle connections after 1 hour
            # Threading-Optimierungen (Phase 2)
            connect_args={
                "check_same_thread": False,  # Erlaubt Multi-Threading
                "timeout": 30.0,  # 30s Lock-Timeout für concurrent access
            },
        )

        # Session Factory
        self.SessionLocal = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False, expire_on_commit=False
        )

        logger.info(f"SQLite Database Manager initialisiert: {self.db_path}")

    def get_session(self) -> Session:
        """
        Erstellt eine neue SQLAlchemy Session.

        Returns:
            SQLAlchemy Session-Objekt
        """
        return self.SessionLocal()

    def init_db(self, base) -> None:
        """
        Initialisiert die Datenbank mit allen Tabellen.

        Args:
            base: SQLAlchemy Base-Klasse mit allen Models
        """
        base.metadata.create_all(bind=self.engine)
        logger.info("Datenbank-Schema erstellt")

    def drop_all(self, base) -> None:
        """
        Löscht alle Tabellen (VORSICHT!).

        Args:
            base: SQLAlchemy Base-Klasse
        """
        base.metadata.drop_all(bind=self.engine)
        logger.warning("Alle Datenbank-Tabellen gelöscht!")

    def close(self) -> None:
        """Schließt die Datenbankverbindung."""
        self.engine.dispose()
        logger.info("SQLite Database Manager geschlossen")

    def create_backup(self, max_backups: int = 5) -> Path | None:
        """
        Create a timestamped backup of the SQLite database.

        Args:
            max_backups: Maximum number of backups to retain (oldest are deleted).

        Returns:
            Path to the created backup file, or None if backup failed.
        """
        import shutil
        from datetime import datetime

        if not self.db_path.exists():
            logger.warning(f"Cannot backup: Database file not found: {self.db_path}")
            return None

        # Backup directory
        backup_dir = self.db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.db_path.stem}_{timestamp}.db"
        backup_path = backup_dir / backup_name

        try:
            # Copy database file
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backup created: {backup_path}")

            # Rotate old backups (keep only max_backups)
            self._rotate_backups(backup_dir, max_backups)

            return backup_path
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return None

    def _rotate_backups(self, backup_dir: Path, max_backups: int) -> None:
        """Delete old backups, keeping only the most recent max_backups."""
        try:
            # Get all backup files sorted by modification time (newest first)
            backups = sorted(
                backup_dir.glob(f"{self.db_path.stem}_*.db"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            # Delete older backups
            for old_backup in backups[max_backups:]:
                old_backup.unlink()
                logger.debug(f"Deleted old backup: {old_backup.name}")

            if len(backups) > max_backups:
                logger.info(
                    f"Rotated backups: kept {max_backups}, deleted {len(backups) - max_backups}"
                )
        except Exception as e:
            logger.warning(f"Backup rotation failed: {e}")


class DuckDBManager:
    """Manager für DuckDB-Analytics-Verbindungen."""

    def __init__(self, db_path: str | None = None):
        """
        Initialisiert den DuckDB Manager.

        Args:
            db_path: Pfad zur DuckDB-Datenbank (optional)
        """
        if not DUCKDB_AVAILABLE:
            logger.warning("DuckDB nicht verfügbar - Analytics-Features deaktiviert")
            self.connection = None
            return

        config = get_config()

        if db_path is None:
            db_path = config.get("Database", "duckdb_path", "data/analytics.duckdb")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # DuckDB Connection
        self.connection = duckdb.connect(str(self.db_path))
        logger.info(f"DuckDB Manager initialisiert: {self.db_path}")

    def execute(self, query: str, params: tuple[Any, ...] | None = None):
        """
        Führt eine DuckDB-Abfrage aus mit parametrisierten Queries.

        K-01 FIX: SQL Injection Prevention durch Parametrisierung.

        Args:
            query: SQL-Query mit Platzhaltern (? oder $1, $2, ...)
            params: Optionale Parameter-Tuple für die Query

        Returns:
            DuckDB ResultSet

        Example:
            # Sicher (parametrisiert):
            db.execute("SELECT * FROM clips WHERE id = ?", (clip_id,))

            # UNSICHER (nicht verwenden!):
            db.execute(f"SELECT * FROM clips WHERE id = {clip_id}")
        """
        if self.connection is None:
            logger.error("DuckDB nicht verfügbar")
            return None

        try:
            if params is not None:
                return self.connection.execute(query, params)
            return self.connection.execute(query)
        except Exception as e:
            logger.error(f"DuckDB Query fehlgeschlagen: {e}")
            raise

    def query(self, query: str, params: tuple[Any, ...] | None = None) -> list[tuple]:
        """
        Führt eine Query aus und gibt Resultate als Liste zurück.

        K-01 FIX: SQL Injection Prevention durch Parametrisierung.

        Args:
            query: SQL-Query mit Platzhaltern (? oder $1, $2, ...)
            params: Optionale Parameter-Tuple für die Query

        Returns:
            Liste von Tupeln mit Ergebnissen

        Example:
            # Sicher (parametrisiert):
            results = db.query("SELECT * FROM clips WHERE name LIKE ?", (f"%{search}%",))
        """
        if self.connection is None:
            logger.error("DuckDB nicht verfügbar")
            return []

        try:
            if params is not None:
                result = self.connection.execute(query, params)
            else:
                result = self.connection.execute(query)
            return result.fetchall()
        except Exception as e:
            logger.error(f"DuckDB Query fehlgeschlagen: {e}")
            return []

    def close(self) -> None:
        """Schließt die DuckDB-Verbindung."""
        if self.connection is not None:
            self.connection.close()
            logger.info("DuckDB Manager geschlossen")


# Globale Manager-Instanzen
_db_manager: DatabaseManager | None = None
_duckdb_manager: DuckDBManager | None = None

# H-01 FIX: Thread-Locks fuer sichere Singleton-Initialisierung
_db_manager_lock = threading.Lock()
_duckdb_manager_lock = threading.Lock()


def get_db_manager() -> DatabaseManager:
    """
    Gibt den globalen DatabaseManager zurück.

    H-01 FIX: Thread-safe Singleton mit Double-Check Locking.

    Returns:
        DatabaseManager-Instanz
    """
    global _db_manager
    if _db_manager is None:
        with _db_manager_lock:
            # Double-check nach Lock-Acquisition
            if _db_manager is None:
                _db_manager = DatabaseManager()
    return _db_manager


def get_duckdb_manager() -> DuckDBManager:
    """
    Gibt den globalen DuckDBManager zurück.

    H-01 FIX: Thread-safe Singleton mit Double-Check Locking.

    Returns:
        DuckDBManager-Instanz
    """
    global _duckdb_manager
    if _duckdb_manager is None:
        with _duckdb_manager_lock:
            # Double-check nach Lock-Acquisition
            if _duckdb_manager is None:
                _duckdb_manager = DuckDBManager()
    return _duckdb_manager


def get_db_session() -> Session:
    """
    Convenience-Funktion für neue SQLAlchemy Session.

    Returns:
        SQLAlchemy Session
    """
    return get_db_manager().get_session()


@contextmanager
def managed_session(session: Session | None = None) -> Generator[Session, None, None]:
    """
    Context Manager für automatisches Session-Handling.

    BUG-FIX: Added exception logging for better debugging.

    Vereinfacht CRUD-Operationen durch automatisches:
    - Session-Erstellung (falls nicht übergeben)
    - Commit bei Erfolg
    - Rollback bei Fehler
    - Session-Schließen (nur wenn selbst erstellt)

    Args:
        session: Optionale existierende Session

    Yields:
        SQLAlchemy Session

    Example:
        with managed_session() as db:
            project = Project(name="Test")
            db.add(project)
            # Auto-commit am Ende
    """
    close_on_exit = session is None
    if close_on_exit:
        session = get_db_session()

    try:
        yield session
        session.commit()
    except Exception as e:
        # BUG-FIX: Log exception details before rollback
        logger.error(
            f"Database transaction failed, rolling back: {type(e).__name__}: {e}", exc_info=True
        )
        session.rollback()
        raise
    finally:
        if close_on_exit:
            session.close()

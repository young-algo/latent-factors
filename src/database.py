"""
Robust SQLite Database Management for Equity Factors Research
==============================================================

This module provides a re-architected database system that addresses
the persistent locking issues in the original implementation.

Key Improvements
----------------
1. **DELETE Journal Mode**: Uses standard rollback journal instead of WAL
   - WAL caused persistent locks with -wal/-shm files
   - DELETE mode is simpler and more reliable for single-process use
   
2. **Busy Timeout**: All connections have 30-second busy timeout
   - Waits for locks instead of failing immediately
   - Configurable via environment variable
   
3. **Connection Per Operation**: No connection pooling
   - Each operation gets fresh connection
   - Always properly closed via context managers
   - Eliminates stale connection issues
   
4. **Lock Detection & Recovery**: Automatic handling of stuck locks
   - Detects orphaned -wal files
   - Automatic recovery procedures
   
5. **Thread Safety**: Proper threading model for SQLite
   - Uses thread-local storage for connections
   - No cross-thread connection sharing

Usage
-----
>>> from src.database import get_db_connection, check_database_health
>>>
>>> # Use connection with automatic cleanup
>>> with get_db_connection() as conn:
...     df = pd.read_sql("SELECT * FROM prices", conn)
>>>
>>> # Check database health
>>> health = check_database_health()
>>> print(f"Database size: {health['size_mb']:.1f} MB")
"""

from __future__ import annotations
import os
import sqlite3
import threading
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator

import pandas as pd

from .config import config

_LOGGER = logging.getLogger(__name__)

# Thread-local storage for database connections
_thread_local = threading.local()

# Default database path
DEFAULT_DB_PATH = Path("./av_cache.db")

# Busy timeout in milliseconds (30 seconds default)
BUSY_TIMEOUT_MS = int(os.getenv('SQLITE_BUSY_TIMEOUT_MS', '30000'))


class DatabaseLockError(Exception):
    """Raised when database is locked and cannot be accessed."""
    pass


class DatabaseCorruptedError(Exception):
    """Raised when database is corrupted and needs recovery."""
    pass


def get_database_path() -> Path:
    """Get the database path from config or default."""
    return config.get_cache_path("av_cache.db") if hasattr(config, 'get_cache_path') else DEFAULT_DB_PATH


def _configure_connection(conn: sqlite3.Connection) -> None:
    """
    Configure SQLite connection with optimal settings.
    
    Settings Applied:
    -----------------
    - busy_timeout: Wait up to 30 seconds for locks
    - journal_mode: DELETE (standard rollback journal, more reliable than WAL)
    - synchronous: NORMAL (balance between safety and speed)
    - cache_size: -64000 (64MB page cache)
    - temp_store: MEMORY (store temp tables in memory)
    """
    # Set busy timeout - CRITICAL for avoiding "database is locked" errors
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    
    # Use DELETE journal mode instead of WAL
    # WAL mode causes persistent locks with -wal/-shm files
    conn.execute("PRAGMA journal_mode = DELETE")
    
    # Normal sync mode for performance/safety balance
    conn.execute("PRAGMA synchronous = NORMAL")
    
    # Larger cache for better performance
    conn.execute("PRAGMA cache_size = -64000")
    
    # Store temporary tables in memory
    conn.execute("PRAGMA temp_store = MEMORY")
    
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    
    conn.commit()


def _get_thread_connection(db_path: Path) -> sqlite3.Connection:
    """
    Get a connection for the current thread.
    
    Uses thread-local storage to ensure each thread has its own connection.
    This is the proper way to handle SQLite in multi-threaded applications.
    """
    if not hasattr(_thread_local, 'connection') or _thread_local.connection is None:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        _configure_connection(conn)
        _thread_local.connection = conn
        _thread_local.db_path = str(db_path)
    
    return _thread_local.connection


def _close_thread_connection() -> None:
    """Close the current thread's connection."""
    if hasattr(_thread_local, 'connection') and _thread_local.connection is not None:
        try:
            _thread_local.connection.close()
        except Exception:
            pass
        _thread_local.connection = None


@contextmanager
def get_db_connection(
    db_path: Optional[Path] = None,
    timeout: float = 30.0
) -> Generator[sqlite3.Connection, None, None]:
    """
    Get a configured database connection with proper cleanup.
    
    This is the primary way to access the database. It provides:
    - Automatic configuration with optimal settings
    - Busy timeout to handle concurrent access
    - Guaranteed cleanup even if exceptions occur
    
    Parameters
    ----------
    db_path : Path, optional
        Path to database file. Uses default if not specified.
    timeout : float, default 30.0
        Connection timeout in seconds
        
    Yields
    ------
    sqlite3.Connection
        Configured database connection
        
    Raises
    ------
    DatabaseLockError
        If database is locked after timeout
    DatabaseCorruptedError
        If database is corrupted
        
    Examples
    --------
    >>> with get_db_connection() as conn:
    ...     cursor = conn.execute("SELECT COUNT(*) FROM prices")
    ...     count = cursor.fetchone()[0]
    
    >>> with get_db_connection() as conn:
    ...     df = pd.read_sql("SELECT * FROM meta", conn)
    """
    db_path = db_path or get_database_path()
    db_path = Path(db_path).expanduser()
    
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = None
    try:
        # Create fresh connection with timeout
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        _configure_connection(conn)
        
        yield conn
        
        # Commit any pending transactions
        conn.commit()
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            _LOGGER.error(f"Database locked: {db_path}")
            raise DatabaseLockError(
                f"Database is locked: {db_path}. "
                "Try: 1) Close other applications using the database, "
                "2) Run 'python -m src clean --all' to reset, or "
                "3) Wait a few minutes and retry."
            ) from e
        elif "database disk image is malformed" in str(e).lower():
            _LOGGER.error(f"Database corrupted: {db_path}")
            raise DatabaseCorruptedError(
                f"Database is corrupted: {db_path}. "
                "Run 'python -m src clean --all' to recreate."
            ) from e
        raise
        
    finally:
        # Always close connection
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                _LOGGER.warning(f"Error closing database connection: {e}")


@contextmanager
def get_db_cursor(
    db_path: Optional[Path] = None
) -> Generator[sqlite3.Cursor, None, None]:
    """
    Get a database cursor with automatic connection management.
    
    Convenience wrapper that yields a cursor directly.
    
    Parameters
    ----------
    db_path : Path, optional
        Path to database file
        
    Yields
    ------
    sqlite3.Cursor
        Database cursor
        
    Examples
    --------
    >>> with get_db_cursor() as cursor:
    ...     cursor.execute("SELECT * FROM prices WHERE ticker = ?", ("AAPL",))
    ...     rows = cursor.fetchall()
    """
    with get_db_connection(db_path) as conn:
        yield conn.cursor()


def check_database_health(db_path: Optional[Path] = None) -> dict:
    """
    Check the health status of the database.
    
    Returns comprehensive information about database state including:
    - File existence and size
    - Table counts and integrity
    - Lock file status
    - Connection test
    
    Parameters
    ----------
    db_path : Path, optional
        Path to database file
        
    Returns
    -------
    dict
        Health status with keys:
        - exists: bool
        - size_mb: float
        - tables: list of table names
        - record_counts: dict of table -> count
        - has_wal_files: bool
        - is_locked: bool
        - is_healthy: bool
        - errors: list of any errors found
    """
    db_path = db_path or get_database_path()
    db_path = Path(db_path)
    
    health = {
        'path': str(db_path),
        'exists': False,
        'size_mb': 0.0,
        'tables': [],
        'record_counts': {},
        'has_wal_files': False,
        'is_locked': False,
        'is_healthy': False,
        'errors': []
    }
    
    try:
        # Check file existence
        if not db_path.exists():
            health['errors'].append(f"Database file not found: {db_path}")
            return health
        
        health['exists'] = True
        health['size_mb'] = db_path.stat().st_size / (1024 * 1024)
        
        # Check for WAL files
        wal_file = db_path.with_suffix('.db-wal')
        shm_file = db_path.with_suffix('.db-shm')
        health['has_wal_files'] = wal_file.exists() or shm_file.exists()
        
        # Try to connect and get info
        try:
            with get_db_connection(db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                
                # Get list of tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                health['tables'] = [row[0] for row in cursor.fetchall()]
                
                # Get record counts
                for table in health['tables']:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        health['record_counts'][table] = cursor.fetchone()[0]
                    except Exception as e:
                        health['record_counts'][table] = f"Error: {e}"
                
                # Run integrity check
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                if integrity_result != "ok":
                    health['errors'].append(f"Integrity check failed: {integrity_result}")
                
                health['is_healthy'] = len(health['errors']) == 0
                
        except DatabaseLockError:
            health['is_locked'] = True
            health['errors'].append("Database is currently locked")
        except Exception as e:
            health['errors'].append(f"Connection error: {e}")
            
    except Exception as e:
        health['errors'].append(f"Health check error: {e}")
    
    return health


def reset_database(db_path: Optional[Path] = None, backup: bool = True) -> Path:
    """
    Reset the database by deleting and optionally backing up.
    
    This is the nuclear option for fixing persistent lock issues.
    
    Parameters
    ----------
    db_path : Path, optional
        Path to database file
    backup : bool, default True
        Whether to create a backup before deleting
        
    Returns
    -------
    Path
        Path to the new (empty) database file
        
    Raises
    ------
    DatabaseLockError
        If database files cannot be deleted due to locks
    """
    db_path = db_path or get_database_path()
    db_path = Path(db_path)
    
    wal_file = db_path.with_suffix('.db-wal')
    shm_file = db_path.with_suffix('.db-shm')
    
    # Create backup if requested
    if backup and db_path.exists():
        backup_path = db_path.with_suffix(f'.db.backup.{int(time.time())}')
        try:
            import shutil
            shutil.copy2(db_path, backup_path)
            _LOGGER.info(f"Database backed up to: {backup_path}")
        except Exception as e:
            _LOGGER.warning(f"Could not create backup: {e}")
    
    # Close any thread-local connections
    _close_thread_connection()
    
    # Try to delete files
    files_to_delete = [db_path, wal_file, shm_file]
    for file_path in files_to_delete:
        if file_path.exists():
            try:
                file_path.unlink()
                _LOGGER.info(f"Deleted: {file_path}")
            except PermissionError as e:
                raise DatabaseLockError(
                    f"Cannot delete {file_path} - file is locked by another process. "
                    "Close all Python processes and retry."
                ) from e
            except Exception as e:
                _LOGGER.warning(f"Could not delete {file_path}: {e}")
    
    return db_path


def ensure_schema(db_path: Optional[Path] = None) -> None:
    """
    Ensure database schema exists.
    
    Creates tables if they don't exist. Safe to call multiple times.
    
    Parameters
    ----------
    db_path : Path, optional
        Path to database file
    """
    with get_db_connection(db_path) as conn:
        # Price data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT,
                date TEXT,
                adj_close REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                ticker TEXT PRIMARY KEY,
                last_update TEXT
            )
        """)
        
        # Fundamentals
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker TEXT PRIMARY KEY,
                last_update TEXT,
                json TEXT
            )
        """)
        
        # ETF holdings
        conn.execute("""
            CREATE TABLE IF NOT EXISTS etf_holdings (
                etf TEXT,
                constituent TEXT,
                weight REAL,
                retrieved TEXT,
                PRIMARY KEY (etf, constituent)
            )
        """)
        
        conn.commit()
        _LOGGER.debug("Database schema ensured")


def optimize_database(db_path: Optional[Path] = None) -> dict:
    """
    Optimize the database by running VACUUM and ANALYZE.
    
    This should be run periodically to reclaim space and update statistics.
    
    Parameters
    ----------
    db_path : Path, optional
        Path to database file
        
    Returns
    -------
    dict
        Optimization results with before/after sizes
    """
    db_path = db_path or get_database_path()
    db_path = Path(db_path)
    
    result = {
        'path': str(db_path),
        'size_before_mb': 0.0,
        'size_after_mb': 0.0,
        'success': False,
        'error': None
    }
    
    if not db_path.exists():
        result['error'] = "Database does not exist"
        return result
    
    result['size_before_mb'] = db_path.stat().st_size / (1024 * 1024)
    
    try:
        with get_db_connection(db_path) as conn:
            # Update statistics
            conn.execute("ANALYZE")
            
            # Vacuum to reclaim space
            conn.execute("VACUUM")
            
            conn.commit()
        
        result['size_after_mb'] = db_path.stat().st_size / (1024 * 1024)
        result['success'] = True
        
        _LOGGER.info(
            f"Database optimized: {result['size_before_mb']:.1f} MB -> "
            f"{result['size_after_mb']:.1f} MB"
        )
        
    except Exception as e:
        result['error'] = str(e)
        _LOGGER.error(f"Database optimization failed: {e}")
    
    return result


# ============================================================================
# Migration utilities for upgrading from old database format
# ============================================================================

def migrate_from_wal_mode(db_path: Optional[Path] = None) -> bool:
    """
    Migrate database from WAL mode to DELETE mode.
    
    This fixes databases that were created with the old WAL-based connection pool.
    
    Parameters
    ----------
    db_path : Path, optional
        Path to database file
        
    Returns
    -------
    bool
        True if migration was successful
    """
    db_path = db_path or get_database_path()
    db_path = Path(db_path)
    
    wal_file = db_path.with_suffix('.db-wal')
    shm_file = db_path.with_suffix('.db-shm')
    
    if not wal_file.exists() and not shm_file.exists():
        # No WAL files to migrate
        return True
    
    _LOGGER.info("Migrating database from WAL mode to DELETE mode...")
    
    try:
        # Connect and checkpoint WAL
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.close()
        
        # Delete WAL files if they still exist
        for file_path in [wal_file, shm_file]:
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    _LOGGER.warning(f"Could not delete {file_path}: {e}")
        
        _LOGGER.info("Database migration complete")
        return True
        
    except Exception as e:
        _LOGGER.error(f"Database migration failed: {e}")
        return False

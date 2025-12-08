# src/rotator_library/utils/resilient_io.py
"""
Resilient I/O utilities for handling file operations gracefully.

Provides two main patterns:
1. ResilientStateWriter - For stateful files (usage.json, credentials, cache)
   that should be buffered in memory and retried on disk failure.
2. safe_log_write / safe_write_json - For logs that can be dropped on failure.
"""

import json
import os
import shutil
import tempfile
import threading
import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union


class ResilientStateWriter:
    """
    Manages resilient writes for stateful files (usage stats, credentials, cache).

    Design:
    - Caller hands off data via write() - always succeeds (memory update)
    - Attempts disk write immediately
    - If disk fails, retries periodically in background
    - On recovery, writes full current state (not just new data)

    Thread-safe for use in async contexts with sync file I/O.

    Usage:
        writer = ResilientStateWriter("data.json", logger)
        writer.write({"key": "value"})  # Always succeeds
        # ... later ...
        if not writer.is_healthy:
            logger.warning("Disk writes failing, data in memory only")
    """

    def __init__(
        self,
        path: Union[str, Path],
        logger: logging.Logger,
        retry_interval: float = 30.0,
        serializer: Optional[Callable[[Any], str]] = None,
    ):
        """
        Initialize the resilient writer.

        Args:
            path: File path to write to
            logger: Logger for warnings/errors
            retry_interval: Seconds between retry attempts when disk is unhealthy
            serializer: Custom serializer function (defaults to JSON with indent=2)
        """
        self.path = Path(path)
        self.logger = logger
        self.retry_interval = retry_interval
        self._serializer = serializer or (lambda d: json.dumps(d, indent=2))

        self._current_state: Optional[Any] = None
        self._disk_healthy = True
        self._last_attempt: float = 0
        self._last_success: Optional[float] = None
        self._failure_count = 0
        self._lock = threading.Lock()

    def write(self, data: Any) -> bool:
        """
        Update state and attempt disk write.

        Always updates in-memory state (guaranteed to succeed).
        Attempts disk write - if it fails, schedules for retry.

        Args:
            data: Data to persist (must be serializable)

        Returns:
            True if disk write succeeded, False if failed (data still in memory)
        """
        with self._lock:
            self._current_state = data
            return self._try_disk_write()

    def retry_if_needed(self) -> bool:
        """
        Retry disk write if unhealthy and retry interval has passed.

        Call this periodically (e.g., on each save attempt) to recover
        from transient disk failures.

        Returns:
            True if healthy (no retry needed or retry succeeded)
        """
        with self._lock:
            if self._disk_healthy:
                return True

            if self._current_state is None:
                return True

            now = time.time()
            if now - self._last_attempt < self.retry_interval:
                return False

            return self._try_disk_write()

    def _try_disk_write(self) -> bool:
        """
        Attempt atomic write to disk. Updates health status.

        Uses tempfile + move pattern for atomic writes on POSIX systems.
        On Windows, uses direct write (still safe for our use case).
        """
        if self._current_state is None:
            return True

        self._last_attempt = time.time()

        try:
            # Ensure directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize data
            content = self._serializer(self._current_state)

            # Atomic write: write to temp file, then move
            tmp_fd = None
            tmp_path = None
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=self.path.parent, prefix=".tmp_", suffix=".json", text=True
                )

                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    tmp_fd = None  # fdopen closes the fd

                # Atomic move
                shutil.move(tmp_path, self.path)
                tmp_path = None

            finally:
                # Cleanup on failure
                if tmp_fd is not None:
                    try:
                        os.close(tmp_fd)
                    except OSError:
                        pass
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            # Success - update health
            self._disk_healthy = True
            self._last_success = time.time()
            self._failure_count = 0
            return True

        except (OSError, PermissionError, IOError) as e:
            self._disk_healthy = False
            self._failure_count += 1

            # Log warning (rate-limited to avoid flooding)
            if self._failure_count == 1 or self._failure_count % 10 == 0:
                self.logger.warning(
                    f"Failed to write {self.path.name}: {e}. "
                    f"Data retained in memory (failure #{self._failure_count})."
                )
            return False

    @property
    def is_healthy(self) -> bool:
        """Check if disk writes are currently working."""
        return self._disk_healthy

    @property
    def current_state(self) -> Optional[Any]:
        """Get the current in-memory state (for inspection/debugging)."""
        return self._current_state

    def get_health_info(self) -> Dict[str, Any]:
        """
        Get detailed health information for monitoring.

        Returns dict with:
            - healthy: bool
            - failure_count: int
            - last_success: Optional[float] (timestamp)
            - last_attempt: float (timestamp)
            - path: str
        """
        return {
            "healthy": self._disk_healthy,
            "failure_count": self._failure_count,
            "last_success": self._last_success,
            "last_attempt": self._last_attempt,
            "path": str(self.path),
        }


def safe_write_json(
    path: Union[str, Path],
    data: Dict[str, Any],
    logger: logging.Logger,
    atomic: bool = True,
    indent: int = 2,
    ensure_ascii: bool = True,
    secure_permissions: bool = False,
) -> bool:
    """
    Write JSON data to file with error handling. No buffering or retry.

    Suitable for one-off writes where failure is acceptable (e.g., logs).
    Creates parent directories if needed.

    Args:
        path: File path to write to
        data: JSON-serializable data
        logger: Logger for warnings
        atomic: Use atomic write pattern (tempfile + move)
        indent: JSON indentation level (default: 2)
        ensure_ascii: Escape non-ASCII characters (default: True)
        secure_permissions: Set file permissions to 0o600 (default: False)

    Returns:
        True on success, False on failure (never raises)
    """
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

        if atomic:
            tmp_fd = None
            tmp_path = None
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=path.parent, prefix=".tmp_", suffix=".json", text=True
                )
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    tmp_fd = None

                # Set secure permissions if requested (before move for security)
                if secure_permissions:
                    try:
                        os.chmod(tmp_path, 0o600)
                    except (OSError, AttributeError):
                        # Windows may not support chmod, ignore
                        pass

                shutil.move(tmp_path, path)
                tmp_path = None
            finally:
                if tmp_fd is not None:
                    try:
                        os.close(tmp_fd)
                    except OSError:
                        pass
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            # Set secure permissions if requested
            if secure_permissions:
                try:
                    os.chmod(path, 0o600)
                except (OSError, AttributeError):
                    pass

        return True

    except (OSError, PermissionError, IOError, TypeError, ValueError) as e:
        logger.warning(f"Failed to write JSON to {path}: {e}")
        return False


def safe_log_write(
    path: Union[str, Path],
    content: str,
    logger: logging.Logger,
    mode: str = "a",
) -> bool:
    """
    Write content to log file with error handling. No buffering or retry.

    Suitable for log files where occasional loss is acceptable.
    Creates parent directories if needed.

    Args:
        path: File path to write to
        content: String content to write
        logger: Logger for warnings
        mode: File mode ('a' for append, 'w' for overwrite)

    Returns:
        True on success, False on failure (never raises)
    """
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        return True

    except (OSError, PermissionError, IOError) as e:
        logger.warning(f"Failed to write log to {path}: {e}")
        return False


def safe_mkdir(path: Union[str, Path], logger: logging.Logger) -> bool:
    """
    Create directory with error handling.

    Args:
        path: Directory path to create
        logger: Logger for warnings

    Returns:
        True on success (or already exists), False on failure
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except (OSError, PermissionError) as e:
        logger.warning(f"Failed to create directory {path}: {e}")
        return False

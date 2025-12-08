# src/rotator_library/utils/resilient_io.py
"""
Resilient I/O utilities for handling file operations gracefully.

Provides three main patterns:
1. BufferedWriteRegistry - Global singleton for buffered writes with periodic
   retry and shutdown flush. Ensures data is saved on app exit (Ctrl+C).
2. ResilientStateWriter - For stateful files (usage.json) that should be
   buffered in memory and retried on disk failure.
3. safe_write_json (with buffer_on_failure) - For critical files (auth tokens)
   that should be buffered and retried if write fails.
4. safe_log_write - For logs that can be dropped on failure.
"""

import atexit
import json
import os
import shutil
import tempfile
import threading
import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union


# =============================================================================
# BUFFERED WRITE REGISTRY (SINGLETON)
# =============================================================================


class BufferedWriteRegistry:
    """
    Global singleton registry for buffered writes with periodic retry and shutdown flush.

    This ensures that critical data (auth tokens, usage stats) is saved even if
    disk writes fail temporarily. On app exit (including Ctrl+C), all pending
    writes are flushed.

    Features:
    - Per-file buffering: each file path has its own pending write
    - Periodic retries: background thread retries failed writes every N seconds
    - Shutdown flush: atexit hook ensures final write attempt on app exit
    - Thread-safe: safe for concurrent access from multiple threads

    Usage:
        # Get the singleton instance
        registry = BufferedWriteRegistry.get_instance()

        # Register a pending write (usually called by safe_write_json on failure)
        registry.register_pending(path, data, serializer_fn, options)

        # Manual flush (optional - atexit handles this automatically)
        results = registry.flush_all()
    """

    _instance: Optional["BufferedWriteRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self, retry_interval: float = 30.0):
        """
        Initialize the registry. Use get_instance() instead of direct construction.

        Args:
            retry_interval: Seconds between retry attempts (default: 30)
        """
        self._pending: Dict[str, Tuple[Any, Callable[[Any], str], Dict[str, Any]]] = {}
        self._retry_interval = retry_interval
        self._lock = threading.Lock()
        self._running = False
        self._retry_thread: Optional[threading.Thread] = None
        self._logger = logging.getLogger("rotator_library.resilient_io")

        # Start background retry thread
        self._start_retry_thread()

        # Register atexit handler for shutdown flush
        atexit.register(self._atexit_handler)

    @classmethod
    def get_instance(cls, retry_interval: float = 30.0) -> "BufferedWriteRegistry":
        """
        Get or create the singleton instance.

        Args:
            retry_interval: Seconds between retry attempts (only used on first call)

        Returns:
            The singleton BufferedWriteRegistry instance
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(retry_interval)
        return cls._instance

    def _start_retry_thread(self) -> None:
        """Start the background retry thread."""
        if self._running:
            return

        self._running = True
        self._retry_thread = threading.Thread(
            target=self._retry_loop,
            name="BufferedWriteRegistry-Retry",
            daemon=True,  # Daemon so it doesn't block app exit
        )
        self._retry_thread.start()

    def _retry_loop(self) -> None:
        """Background thread: periodically retry pending writes."""
        while self._running:
            time.sleep(self._retry_interval)
            if not self._running:
                break
            self._retry_pending()

    def _retry_pending(self) -> None:
        """Attempt to write all pending files."""
        with self._lock:
            if not self._pending:
                return

            # Copy paths to avoid modifying dict during iteration
            paths = list(self._pending.keys())

        for path_str in paths:
            self._try_write(path_str, remove_on_success=True)

    def register_pending(
        self,
        path: Union[str, Path],
        data: Any,
        serializer: Callable[[Any], str],
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a pending write for later retry.

        If a write is already pending for this path, it is replaced with the new data
        (we always want to write the latest state).

        Args:
            path: File path to write to
            data: Data to serialize and write
            serializer: Function to serialize data to string
            options: Additional options (e.g., secure_permissions)
        """
        path_str = str(Path(path).resolve())
        with self._lock:
            self._pending[path_str] = (data, serializer, options or {})
            self._logger.debug(f"Registered pending write for {Path(path).name}")

    def unregister(self, path: Union[str, Path]) -> None:
        """
        Remove a pending write (called when write succeeds elsewhere).

        Args:
            path: File path to remove from pending
        """
        path_str = str(Path(path).resolve())
        with self._lock:
            self._pending.pop(path_str, None)

    def _try_write(self, path_str: str, remove_on_success: bool = True) -> bool:
        """
        Attempt to write a pending file.

        Args:
            path_str: Resolved path string
            remove_on_success: Remove from pending if successful

        Returns:
            True if write succeeded, False otherwise
        """
        with self._lock:
            if path_str not in self._pending:
                return True
            data, serializer, options = self._pending[path_str]

        path = Path(path_str)
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize data
            content = serializer(data)

            # Atomic write
            tmp_fd = None
            tmp_path = None
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=path.parent, prefix=".tmp_", suffix=".json", text=True
                )
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    tmp_fd = None

                # Set secure permissions if requested
                if options.get("secure_permissions"):
                    try:
                        os.chmod(tmp_path, 0o600)
                    except (OSError, AttributeError):
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

            # Success - remove from pending
            if remove_on_success:
                with self._lock:
                    self._pending.pop(path_str, None)

            self._logger.debug(f"Retry succeeded for {path.name}")
            return True

        except (OSError, PermissionError, IOError) as e:
            self._logger.debug(f"Retry failed for {path.name}: {e}")
            return False

    def flush_all(self) -> Dict[str, bool]:
        """
        Attempt to write all pending files immediately.

        Returns:
            Dict mapping file paths to success status
        """
        with self._lock:
            paths = list(self._pending.keys())

        results = {}
        for path_str in paths:
            results[path_str] = self._try_write(path_str, remove_on_success=True)

        return results

    def _atexit_handler(self) -> None:
        """Called on app exit to flush pending writes."""
        self._running = False

        with self._lock:
            pending_count = len(self._pending)

        if pending_count == 0:
            return

        self._logger.info(f"Flushing {pending_count} pending write(s) on shutdown...")
        results = self.flush_all()

        succeeded = sum(1 for v in results.values() if v)
        failed = pending_count - succeeded

        if failed > 0:
            self._logger.warning(
                f"Shutdown flush: {succeeded} succeeded, {failed} failed"
            )
            for path_str, success in results.items():
                if not success:
                    self._logger.warning(f"  Failed to save: {Path(path_str).name}")
        else:
            self._logger.info(f"Shutdown flush: all {succeeded} write(s) succeeded")

    def get_pending_count(self) -> int:
        """Get the number of pending writes."""
        with self._lock:
            return len(self._pending)

    def get_pending_paths(self) -> list:
        """Get list of paths with pending writes (for monitoring)."""
        with self._lock:
            return [Path(p).name for p in self._pending.keys()]

    def shutdown(self) -> Dict[str, bool]:
        """
        Manually trigger shutdown: stop retry thread and flush all pending writes.

        Returns:
            Dict mapping file paths to success status
        """
        self._running = False
        if self._retry_thread and self._retry_thread.is_alive():
            self._retry_thread.join(timeout=1.0)
        return self.flush_all()


# =============================================================================
# RESILIENT STATE WRITER
# =============================================================================


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
        Attempts disk write - if disk is unhealthy, respects retry_interval
        before attempting again to avoid flooding with failed writes.

        Args:
            data: Data to persist (must be serializable)

        Returns:
            True if disk write succeeded, False if failed (data still in memory)
        """
        with self._lock:
            self._current_state = data

            # If disk is unhealthy, only retry after retry_interval has passed
            if not self._disk_healthy:
                now = time.time()
                if now - self._last_attempt < self.retry_interval:
                    # Too soon to retry, data is safe in memory
                    return False

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

        Also registers/unregisters with BufferedWriteRegistry for shutdown flush.
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

            # Success - update health and unregister from shutdown flush
            self._disk_healthy = True
            self._last_success = time.time()
            self._failure_count = 0
            BufferedWriteRegistry.get_instance().unregister(self.path)
            return True

        except (OSError, PermissionError, IOError) as e:
            self._disk_healthy = False
            self._failure_count += 1

            # Register with BufferedWriteRegistry for shutdown flush
            registry = BufferedWriteRegistry.get_instance()
            registry.register_pending(
                self.path,
                self._current_state,
                self._serializer,
                {},  # No special options for ResilientStateWriter
            )

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
    buffer_on_failure: bool = False,
) -> bool:
    """
    Write JSON data to file with error handling and optional buffering.

    When buffer_on_failure is True, failed writes are registered with the
    BufferedWriteRegistry for periodic retry and shutdown flush. This ensures
    critical data (like auth tokens) is eventually saved.

    Args:
        path: File path to write to
        data: JSON-serializable data
        logger: Logger for warnings
        atomic: Use atomic write pattern (tempfile + move)
        indent: JSON indentation level (default: 2)
        ensure_ascii: Escape non-ASCII characters (default: True)
        secure_permissions: Set file permissions to 0o600 (default: False)
        buffer_on_failure: Register with BufferedWriteRegistry on failure (default: False)

    Returns:
        True on success, False on failure (never raises)
    """
    path = Path(path)

    # Create serializer function that matches the requested formatting
    def serializer(d: Any) -> str:
        return json.dumps(d, indent=indent, ensure_ascii=ensure_ascii)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = serializer(data)

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

        # Success - remove from pending if it was there
        if buffer_on_failure:
            BufferedWriteRegistry.get_instance().unregister(path)

        return True

    except (OSError, PermissionError, IOError, TypeError, ValueError) as e:
        logger.warning(f"Failed to write JSON to {path}: {e}")

        # Register for retry if buffering is enabled
        if buffer_on_failure:
            registry = BufferedWriteRegistry.get_instance()
            registry.register_pending(
                path,
                data,
                serializer,
                {"secure_permissions": secure_permissions},
            )
            logger.debug(f"Buffered {path.name} for retry on next interval or shutdown")

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

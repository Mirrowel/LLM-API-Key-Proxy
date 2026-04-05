"""
Codex WebSocket Transport

Persistent WebSocket connection pool for the OpenAI Responses API WebSocket mode.
Maintains long-lived connections per credential with session affinity for
previous_response_id chaining.

Protocol reference: https://developers.openai.com/api/docs/guides/websocket-mode
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

import websockets
import websockets.asyncio.client
import websockets.exceptions
from websockets.protocol import State as WebSocketState

lib_logger = logging.getLogger("rotator_library")

POOL_ACQUIRE_TIMEOUT = 5.0

def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return default

WS_RECV_TIMEOUT = float(_env_int("CODEX_WS_RECV_TIMEOUT", 90))


@dataclass
class _SessionState:
    """Tracks the last response_id for a given session on a specific connection."""
    connection_id: str
    response_id: str
    timestamp: float = field(default_factory=time.time)


class CodexWebSocketConnection:
    """Single managed WebSocket connection to OpenAI Responses API."""

    def __init__(self, connection_id: str, ws_endpoint: str, headers: Dict[str, str],
                 connection_ttl: float = 3300.0):
        self._id = connection_id
        self._ws_endpoint = ws_endpoint
        self._headers = headers
        self._connection_ttl = connection_ttl
        self._ws: Optional[websockets.asyncio.client.ClientConnection] = None
        self._created_at: float = 0
        self._in_use: bool = False
        self._last_response_id: Optional[str] = None
        self._dead: bool = False

    @property
    def id(self) -> str:
        return self._id

    @property
    def in_use(self) -> bool:
        return self._in_use

    @property
    def is_dead(self) -> bool:
        return self._dead

    @property
    def last_response_id(self) -> Optional[str]:
        return self._last_response_id

    @property
    def is_expired(self) -> bool:
        if self._created_at == 0:
            return False
        return (time.time() - self._created_at) > self._connection_ttl

    @property
    def is_connected(self) -> bool:
        return (
            self._ws is not None
            and not self._dead
            and self._ws.state is WebSocketState.OPEN
        )

    def mark_dead(self) -> None:
        """Mark connection as dead (unusable, pending removal)."""
        self._dead = True
        self._last_response_id = None

    async def connect(self) -> None:
        """Establish the WebSocket connection."""
        if self.is_connected and not self.is_expired:
            return

        await self.close()

        additional_headers = websockets.Headers(self._headers)
        self._ws = await websockets.asyncio.client.connect(
            self._ws_endpoint,
            additional_headers=additional_headers,
            max_size=2**24,  # 16MB max frame (reasoning outputs can be large)
            close_timeout=5,
            ping_interval=20,
            ping_timeout=20,
        )
        self._created_at = time.time()
        self._last_response_id = None
        self._dead = False
        lib_logger.debug(f"[Codex-WS] Connection {self._id} established to {self._ws_endpoint}")

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._last_response_id = None

    async def send_response_create(
        self,
        payload: Dict[str, Any],
        previous_response_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a response.create event and yield parsed JSON events from the server.

        Args:
            payload: The Responses API request body (model, input, tools, etc.)
            previous_response_id: If set, sent for incremental continuation

        Yields:
            Parsed JSON event dicts from the server
        """
        if not self._in_use:
            raise RuntimeError("send_response_create called on unacquired connection")
        if not self.is_connected:
            raise ConnectionError("WebSocket not connected")

        event = {
            "type": "response.create",
            **payload,
        }
        # Remove transport-specific fields not used in WS mode
        event.pop("stream", None)
        event.pop("background", None)

        if previous_response_id:
            event["previous_response_id"] = previous_response_id

        await self._ws.send(json.dumps(event))

        response_id: Optional[str] = None
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(
                        self._ws.recv(), timeout=WS_RECV_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    lib_logger.warning(
                        f"[Codex-WS] Connection {self._id} recv timeout after {WS_RECV_TIMEOUT}s"
                    )
                    break

                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="ignore")

                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    lib_logger.debug(
                        f"[Codex-WS] Non-JSON message on {self._id}: {raw[:200]!r}"
                    )
                    continue

                # Track response_id for chaining
                if isinstance(evt.get("response"), dict):
                    rid = evt["response"].get("id")
                    if rid:
                        response_id = rid

                yield evt

                kind = evt.get("type", "")
                if kind in ("response.completed", "response.incomplete", "response.failed"):
                    break
                if kind == "error":
                    break

        finally:
            if response_id:
                self._last_response_id = response_id
            else:
                self._last_response_id = None


class CodexWebSocketPool:
    """
    Pool of WebSocket connections per credential.

    Provides session affinity so that requests with the same session_id
    are routed to the same connection for previous_response_id chaining.

    Uses asyncio.Condition for efficient wake-on-release instead of polling.
    """

    def __init__(self, ws_endpoint: str, max_per_credential: int = 3,
                 connection_ttl: float = 3300.0):
        self._ws_endpoint = ws_endpoint
        self._max_per_credential = max_per_credential
        self._connection_ttl = connection_ttl
        # credential_path -> list of connections
        self._pools: Dict[str, List[CodexWebSocketConnection]] = {}
        # session_id -> session state (which connection + last response_id)
        self._session_map: Dict[str, _SessionState] = {}
        self._lock = asyncio.Lock()
        self._available = asyncio.Condition(self._lock)
        self._counter = 0
        self._reaper_task: Optional[asyncio.Task] = None
        self.start_reaper()

    def start_reaper(self) -> None:
        """Start background task to close expired connections.

        Safe to call multiple times; only the first call with an active event
        loop creates the task.  Must be called while holding self._lock (the
        _acquire_inner path satisfies this).
        """
        if self._reaper_task is not None and not self._reaper_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._reaper_task = loop.create_task(self._reaper_loop())
        except RuntimeError:
            self._reaper_task = None

    async def _reaper_loop(self) -> None:
        """Periodically close expired connections."""
        while True:
            await asyncio.sleep(60)
            try:
                await self._reap_expired()
            except Exception as e:
                lib_logger.debug(f"[Codex-WS] Reaper error: {e}")

    async def _reap_expired(self) -> None:
        """Close connections that have exceeded their TTL."""
        to_close: List[CodexWebSocketConnection] = []
        removed_conn_ids: List[str] = []

        # Collect expired connections under lock, but don't close them yet
        async with self._lock:
            for cred, conns in list(self._pools.items()):
                remaining = []
                for conn in conns:
                    if (conn.is_expired or conn.is_dead) and not conn.in_use:
                        to_close.append(conn)
                        removed_conn_ids.append(conn.id)
                    else:
                        remaining.append(conn)
                self._pools[cred] = remaining

            # Evict session-map entries pointing to removed connections
            self._evict_sessions_for_connections(removed_conn_ids)

            # Also clean up stale session entries (>60 min old)
            now = time.time()
            stale = [
                sid for sid, state in self._session_map.items()
                if (now - state.timestamp) > 3600
            ]
            for sid in stale:
                del self._session_map[sid]

        # Close connections outside the lock (avoids blocking pool operations)
        for conn in to_close:
            try:
                await conn.close()
            except Exception:
                pass
            lib_logger.debug(f"[Codex-WS] Reaped expired/dead connection {conn.id}")

    def _evict_sessions_for_connections(self, conn_ids: List[str]) -> None:
        """Remove session_map entries that reference any of the given connection IDs.
        Must be called while holding self._lock."""
        if not conn_ids:
            return
        conn_id_set = set(conn_ids)
        to_remove = [
            sid for sid, state in self._session_map.items()
            if state.connection_id in conn_id_set
        ]
        for sid in to_remove:
            del self._session_map[sid]

    async def acquire(
        self,
        credential_path: str,
        headers: Dict[str, str],
        session_id: Optional[str] = None,
    ) -> tuple[CodexWebSocketConnection, Optional[str]]:
        """
        Acquire a connection from the pool.

        Returns:
            Tuple of (connection, previous_response_id or None)

        Connection establishment happens outside the lock to avoid blocking
        the entire pool during TLS handshake.  If acquisition times out, any
        connection that was marked in-use is cleaned up to prevent leaks.
        """
        deadline = time.monotonic() + POOL_ACQUIRE_TIMEOUT
        conn: Optional[CodexWebSocketConnection] = None

        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ConnectionError(
                        f"[Codex-WS] Pool exhausted for credential (max={self._max_per_credential}), "
                        f"timed out waiting for available connection"
                    )

                conn, prev_resp_id, needs_connect = await asyncio.wait_for(
                    self._acquire_inner(credential_path, headers, session_id),
                    timeout=remaining,
                )

                if not needs_connect:
                    return conn, prev_resp_id

                # Connect outside the lock so other pool operations aren't blocked
                try:
                    await conn.connect()
                    return conn, prev_resp_id
                except Exception as e:
                    lib_logger.warning(f"[Codex-WS] Failed to connect {conn.id}: {e}")
                    async with self._available:
                        pool = self._pools.get(credential_path, [])
                        if conn in pool:
                            pool.remove(conn)
                        conn._in_use = False
                        self._available.notify_all()
                    raise

        except (asyncio.CancelledError, ConnectionError):
            # On timeout or cancellation, release any connection we reserved
            if conn is not None and conn.in_use:
                async with self._available:
                    conn._in_use = False
                    self._available.notify_all()
            raise

    async def _acquire_inner(
        self,
        credential_path: str,
        headers: Dict[str, str],
        session_id: Optional[str],
    ) -> tuple[CodexWebSocketConnection, Optional[str], bool]:
        """Inner acquire logic.

        Returns (connection, previous_response_id, needs_connect).
        When needs_connect is True the caller must call conn.connect()
        outside the lock before using the connection.
        """
        async with self._available:
            self.start_reaper()

            while True:
                pool = self._pools.setdefault(credential_path, [])

                # Proactively collect expired/dead idle connections
                expired = [c for c in pool if (c.is_expired or c.is_dead) and not c.in_use]
                if expired:
                    expired_ids = [c.id for c in expired]
                    pool[:] = [c for c in pool if c not in expired]
                    self._evict_sessions_for_connections(expired_ids)
                    for c in expired:
                        try:
                            await c.close()
                        except Exception:
                            pass

                # Session affinity: try to reuse the same connection
                if session_id and session_id in self._session_map:
                    state = self._session_map[session_id]
                    for conn in pool:
                        if (conn.id == state.connection_id
                                and not conn.in_use
                                and conn.is_connected
                                and not conn.is_expired):
                            conn._in_use = True
                            state.timestamp = time.time()
                            lib_logger.debug(
                                f"[Codex-WS] Session affinity hit: session={session_id[:8]}..., "
                                f"conn={conn.id}, prev_resp={state.response_id}"
                            )
                            return conn, state.response_id, False

                # Find any already-connected idle connection
                for conn in pool:
                    if not conn.in_use and not conn.is_expired and not conn.is_dead:
                        if conn.is_connected:
                            conn._in_use = True
                            conn._headers = headers
                            return conn, None, False

                # Find a disconnected connection to reconnect (outside the lock)
                for conn in pool:
                    if not conn.in_use and not conn.is_expired and not conn.is_dead:
                        conn._in_use = True
                        conn._headers = headers
                        return conn, None, True

                # Create a new connection slot if under limit
                if len(pool) < self._max_per_credential:
                    self._counter += 1
                    conn_id = f"ws-{self._counter}"
                    conn = CodexWebSocketConnection(
                        conn_id, self._ws_endpoint, headers,
                        connection_ttl=self._connection_ttl,
                    )
                    conn._in_use = True
                    pool.append(conn)
                    return conn, None, True

                # Pool is full — wait for a release
                await self._available.wait()

    async def release(
        self,
        conn: CodexWebSocketConnection,
        session_id: Optional[str] = None,
    ) -> None:
        """Release a connection back to the pool and update session state.

        Wakes any tasks waiting for a free connection.
        """
        async with self._available:
            if session_id and conn.last_response_id:
                self._session_map[session_id] = _SessionState(
                    connection_id=conn.id,
                    response_id=conn.last_response_id,
                )
            conn._in_use = False
            self._available.notify_all()

    async def mark_dead_and_evict(self, conn: CodexWebSocketConnection) -> None:
        """Mark a connection as dead, close it, and remove it from the pool.

        Called when a WS connection fails mid-stream.
        """
        conn.mark_dead()
        conn._in_use = False
        async with self._available:
            for cred, pool in self._pools.items():
                if conn in pool:
                    pool.remove(conn)
                    self._evict_sessions_for_connections([conn.id])
                    break
            self._available.notify_all()
        try:
            await conn.close()
        except Exception:
            pass

    async def clear_session(self, session_id: str) -> None:
        """Clear a session's previous_response_id mapping (e.g. on not_found error)."""
        async with self._lock:
            self._session_map.pop(session_id, None)

    async def close_all(self) -> None:
        """Shut down all connections (for graceful shutdown)."""
        if self._reaper_task and not self._reaper_task.done():
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except (asyncio.CancelledError, Exception):
                pass
        async with self._lock:
            for pool in self._pools.values():
                for conn in pool:
                    await conn.close()
            self._pools.clear()
            self._session_map.clear()

# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Freebuff Authentication Base

Manages Freebuff session and run lifecycle including:
- Free session management (create, poll, refresh, end)
- Agent run lifecycle (start, finish, rotate)
- Model-to-agent mapping from Codebuff free-agents source
- Per-token state tracking (sessions, runs)
"""

import asyncio
import json
import logging
import os
import random
import re
import string
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

FREEBUFF_DEFAULT_BASE_URL = "https://codebuff.com"
FREEBUFF_USER_AGENT = "ai-sdk/openai-compatible/1.0.25/codebuff"
FREE_AGENTS_SOURCE_URL = (
    "https://raw.githubusercontent.com/CodebuffAI/codebuff/main/common/src/constants/free-agents.ts"
)
MODEL_REFRESH_INTERVAL = 6 * 3600
SESSION_POLL_INTERVAL = 5.0
SESSION_RETRY_DELAY = 10.0
RUN_ROTATION_INTERVAL = 6 * 3600
REQUEST_TIMEOUT = 900.0

HARDCODED_AGENT_MODELS: Dict[str, List[str]] = {
    "base2-free": ["minimax/minimax-m2.7", "z-ai/glm-5.1"],
    "file-picker": ["google/gemini-2.5-flash-lite"],
    "file-picker-max": ["google/gemini-3.1-flash-lite-preview"],
    "file-lister": ["google/gemini-3.1-flash-lite-preview"],
    "researcher-web": ["google/gemini-3.1-flash-lite-preview"],
    "researcher-docs": ["google/gemini-3.1-flash-lite-preview"],
    "basher": ["google/gemini-3.1-flash-lite-preview"],
    "editor-lite": ["minimax/minimax-m2.7", "z-ai/glm-5.1"],
    "code-reviewer-lite": ["minimax/minimax-m2.7", "z-ai/glm-5.1"],
}


def _generate_client_session_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=13))


def _parse_optional_time(value: str) -> Optional[datetime]:
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class CachedSession:
    __slots__ = ("status", "instance_id", "expires_at")

    def __init__(
        self,
        status: str,
        instance_id: str = "",
        expires_at: Optional[datetime] = None,
    ):
        self.status = status
        self.instance_id = instance_id
        self.expires_at = expires_at


class ManagedRun:
    __slots__ = ("run_id", "agent_id", "started_at", "inflight", "request_count", "finishing")

    def __init__(self, run_id: str, agent_id: str):
        self.run_id = run_id
        self.agent_id = agent_id
        self.started_at = time.monotonic()
        self.inflight = 0
        self.request_count = 0
        self.finishing = False


class TokenPoolState:
    def __init__(self, token: str, name: str):
        self.token = token
        self.name = name
        self.session: Optional[CachedSession] = None
        self.session_refresh_lock = asyncio.Lock()
        self.runs: Dict[str, ManagedRun] = {}
        self.draining: List[ManagedRun] = []
        self.cooldown_until: float = 0.0
        self.last_error: str = ""

    def is_cooling_down(self) -> bool:
        return time.monotonic() < self.cooldown_until

    def ready_session_instance(self) -> Optional[str]:
        if self.session is None:
            return None
        if self.session.status == "disabled":
            return ""
        if self.session.status == "active" and self.session.instance_id:
            if self.session.expires_at is None or datetime.now(timezone.utc) < self.session.expires_at.replace(
                tzinfo=timezone.utc
            ) - __import__("datetime").timedelta(seconds=5):
                return self.session.instance_id
        return None


class FreebuffAuthBase:
    """
    Authentication and session management base for Freebuff provider.

    Handles:
    - Model-agent mapping (fetched from Codebuff repo with hardcoded fallback)
    - Free session lifecycle per auth token
    - Agent run lifecycle per auth token
    - Token pool round-robin selection
    """

    def __init__(self):
        self.base_url = os.getenv("FREEBUFF_API_BASE", FREEBUFF_DEFAULT_BASE_URL).rstrip("/")
        self._agent_models: Dict[str, List[str]] = {}
        self._model_to_agent: Dict[str, str] = {}
        self._all_models: List[str] = []
        self._token_pools: Dict[str, TokenPoolState] = {}
        self._next_pool_index = 0
        self._model_refresh_lock = asyncio.Lock()
        self._last_model_refresh: float = 0.0
        self._initialized = False
        self._load_model_mapping_fallback()

    async def initialize_credentials(self, credential_paths: List[str]) -> None:
        """
        Initialize token pool states from credential paths.

        Credential paths are either file paths or env:// references.
        The actual token value is stored in the pool state.
        """
        if self._initialized:
            return
        self._initialized = True

        self._load_model_mapping_fallback()

        for i, cred_path in enumerate(credential_paths):
            name = f"token-{i + 1}"
            token = self._resolve_token(cred_path)
            if token:
                self._token_pools[cred_path] = TokenPoolState(token, name)
                lib_logger.info(f"Freebuff: initialized token pool {name}")

        lib_logger.info(
            f"Freebuff: initialized {len(self._token_pools)} token pools, "
            f"{len(self._all_models)} models available"
        )

    def _resolve_token(self, credential_path: str) -> Optional[str]:
        if credential_path.startswith("env://"):
            return os.getenv(credential_path[6:])
        if os.path.isfile(credential_path):
            try:
                with open(credential_path) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    default = data.get("default", data)
                    if isinstance(default, dict):
                        return default.get("authToken", default.get("token"))
                    return str(default)
                return str(data)
            except (json.JSONDecodeError, OSError) as e:
                lib_logger.warning(f"Freebuff: failed to read credential file {credential_path}: {e}")
                return None
        return credential_path

    def get_available_models(self) -> List[str]:
        return list(self._all_models)

    def get_agent_for_model(self, model: str) -> Optional[str]:
        return self._model_to_agent.get(model)

    def _load_model_mapping_fallback(self) -> None:
        model_to_agent: Dict[str, str] = {}
        all_models: List[str] = []
        for agent_id, models in HARDCODED_AGENT_MODELS.items():
            for model in models:
                if model not in model_to_agent:
                    model_to_agent[model] = agent_id
                    all_models.append(model)
        all_models.sort()
        self._agent_models = dict(HARDCODED_AGENT_MODELS)
        self._model_to_agent = model_to_agent
        self._all_models = all_models

    async def refresh_model_mapping(self, client: httpx.AsyncClient) -> None:
        async with self._model_refresh_lock:
            now = time.monotonic()
            if now - self._last_model_refresh < MODEL_REFRESH_INTERVAL:
                return
            try:
                resp = await client.get(
                    FREE_AGENTS_SOURCE_URL,
                    headers={"Accept": "text/plain"},
                    timeout=30.0,
                )
                resp.raise_for_status()
                source = resp.text
                parsed = self._parse_free_agents_source(source)
                if parsed:
                    model_to_agent, all_models = self._build_model_mapping(parsed)
                    self._agent_models = parsed
                    self._model_to_agent = model_to_agent
                    self._all_models = all_models
                    self._last_model_refresh = now
                    lib_logger.info(
                        f"Freebuff: refreshed model mapping: {len(parsed)} agents, {len(all_models)} models"
                    )
            except Exception as e:
                lib_logger.debug(f"Freebuff: model mapping refresh failed: {e}")

    def _parse_free_agents_source(self, source: str) -> Dict[str, List[str]]:
        block_re = re.compile(r"'([^']+)':\s*new\s+Set\(\[([^\]]*)\]\)")
        model_re = re.compile(r"'([^']+)'")
        result: Dict[str, List[str]] = {}
        for match in block_re.finditer(source):
            agent_id = match.group(1)
            models_str = match.group(2)
            models = [m.strip() for m in model_re.findall(models_str) if m.strip()]
            if models:
                result[agent_id] = models
        return result

    def _build_model_mapping(
        self, agent_models: Dict[str, List[str]]
    ) -> Tuple[Dict[str, str], List[str]]:
        model_agents: Dict[str, List[str]] = {}
        for agent_id, models in agent_models.items():
            for model in models:
                model_agents.setdefault(model, []).append(agent_id)
        model_to_agent = {m: random.choice(a) for m, a in model_agents.items()}
        all_models = sorted(model_to_agent.keys())
        return model_to_agent, all_models

    def _get_pool(self, credential_path: str) -> Optional[TokenPoolState]:
        return self._token_pools.get(credential_path)

    def _select_pool(self) -> Optional[Tuple[str, TokenPoolState]]:
        pools = list(self._token_pools.items())
        if not pools:
            return None
        start = self._next_pool_index % len(pools)
        ready = []
        not_ready = []
        for offset in range(len(pools)):
            path, pool = pools[(start + offset) % len(pools)]
            if pool.ready_session_instance() is not None:
                ready.append((path, pool))
            else:
                not_ready.append((path, pool))
        candidates = ready + not_ready
        for path, pool in candidates:
            if not pool.is_cooling_down():
                self._next_pool_index += 1
                return path, pool
        return None

    async def ensure_session(
        self, client: httpx.AsyncClient, pool: TokenPoolState
    ) -> Optional[str]:
        instance = pool.ready_session_instance()
        if instance is not None:
            return instance

        async with pool.session_refresh_lock:
            instance = pool.ready_session_instance()
            if instance is not None:
                return instance
            try:
                session, instance_id = await self._refresh_session(client, pool)
                pool.session = session
                pool.last_error = ""
                if session and session.status == "active" and session.expires_at:
                    asyncio.ensure_future(
                        self._watch_session_expiry(client, pool, session)
                    )
                return instance_id
            except Exception as e:
                pool.session = None
                pool.last_error = str(e)
                lib_logger.warning(f"Freebuff [{pool.name}]: session refresh failed: {e}")
                return None

    async def _refresh_session(
        self, client: httpx.AsyncClient, pool: TokenPoolState
    ) -> Tuple[Optional[CachedSession], Optional[str]]:
        state = await self._create_or_refresh_session(client, pool.token)
        while True:
            status = state.get("status", "").strip()
            if status == "disabled":
                return CachedSession("disabled"), ""
            elif status == "active":
                instance_id = state.get("instanceId", "").strip()
                if not instance_id:
                    raise ValueError("active session missing instanceId")
                expires_at = _parse_optional_time(state.get("expiresAt", ""))
                return (
                    CachedSession("active", instance_id, expires_at),
                    instance_id,
                )
            elif status == "queued":
                instance_id = state.get("instanceId", "").strip()
                if not instance_id:
                    raise ValueError("queued session missing instanceId")
                wait_ms = state.get("estimatedWaitMs", 0)
                delay = max(1.0, min(wait_ms / 1000.0, SESSION_POLL_INTERVAL))
                await asyncio.sleep(delay)
                state = await self._get_session(client, pool.token, instance_id)
            else:
                state = await self._create_or_refresh_session(client, pool.token)

    async def _create_or_refresh_session(
        self, client: httpx.AsyncClient, token: str
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/freebuff/session"
        resp = await client.post(
            url,
            json={},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": FREEBUFF_USER_AGENT,
            },
            timeout=30.0,
        )
        if resp.status_code == 404:
            return {"status": "disabled"}
        resp.raise_for_status()
        data = resp.json()
        if not data.get("status", "").strip():
            raise ValueError("session response missing status")
        return data

    async def _get_session(
        self, client: httpx.AsyncClient, token: str, instance_id: str
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/freebuff/session"
        resp = await client.get(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": FREEBUFF_USER_AGENT,
                "x-freebuff-instance-id": instance_id,
            },
            timeout=30.0,
        )
        if resp.status_code == 404:
            return {"status": "disabled"}
        resp.raise_for_status()
        return resp.json()

    async def _end_session(
        self, client: httpx.AsyncClient, token: str
    ) -> None:
        url = f"{self.base_url}/api/v1/freebuff/session"
        try:
            await client.delete(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "User-Agent": FREEBUFF_USER_AGENT,
                },
                timeout=15.0,
            )
        except Exception as e:
            lib_logger.debug(f"Freebuff: end session failed: {e}")

    async def _watch_session_expiry(
        self, client: httpx.AsyncClient, pool: TokenPoolState, session: CachedSession
    ) -> None:
        if not session.expires_at:
            return
        now = datetime.now(timezone.utc)
        expires = session.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        delay = max(0, (expires - now).total_seconds() + 1)
        await asyncio.sleep(delay)
        if pool.session is session and session.status == "active":
            pool.session = None
            lib_logger.info(f"Freebuff [{pool.name}]: session expired, will refresh on next request")

    def invalidate_session(self, pool: TokenPoolState, reason: str = "") -> None:
        pool.session = None
        if reason:
            pool.last_error = reason

    async def start_run(
        self, client: httpx.AsyncClient, pool: TokenPoolState, agent_id: str
    ) -> str:
        url = f"{self.base_url}/api/v1/agent-runs"
        payload = {"action": "START", "agentId": agent_id}
        resp = await client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {pool.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": FREEBUFF_USER_AGENT,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        run_id = data.get("runId", "").strip()
        if not run_id:
            raise ValueError(f"start run response missing runId: {data}")
        old_run = pool.runs.get(agent_id)
        run = ManagedRun(run_id, agent_id)
        pool.runs[agent_id] = run
        if old_run:
            pool.draining.append(old_run)
            asyncio.ensure_future(self._finish_draining_run(client, pool, old_run))
        lib_logger.debug(f"Freebuff [{pool.name}]: started run {run_id} for agent {agent_id}")
        return run_id

    async def ensure_run(
        self, client: httpx.AsyncClient, pool: TokenPoolState, agent_id: str
    ) -> ManagedRun:
        run = pool.runs.get(agent_id)
        needs_rotate = run is None or (time.monotonic() - run.started_at) >= RUN_ROTATION_INTERVAL
        if needs_rotate:
            await self.start_run(client, pool, agent_id)
            run = pool.runs.get(agent_id)
        if run is None:
            raise RuntimeError(f"run missing for agent {agent_id} after rotation")
        return run

    async def finish_run(
        self, client: httpx.AsyncClient, pool: TokenPoolState, run: ManagedRun
    ) -> None:
        if run.finishing:
            return
        run.finishing = True
        url = f"{self.base_url}/api/v1/agent-runs"
        payload = {
            "action": "FINISH",
            "runId": run.run_id,
            "status": "completed",
            "totalSteps": run.request_count,
            "directCredits": 0,
            "totalCredits": 0,
        }
        try:
            await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {pool.token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": FREEBUFF_USER_AGENT,
                },
                timeout=15.0,
            )
        except Exception as e:
            run.finishing = False
            pool.last_error = str(e)
            lib_logger.debug(f"Freebuff [{pool.name}]: finish run {run.run_id} failed: {e}")
            return
        pool.draining = [r for r in pool.draining if r is not run]

    def invalidate_run(self, pool: TokenPoolState, run: ManagedRun, reason: str = "") -> None:
        if pool.runs.get(run.agent_id) is run:
            del pool.runs[run.agent_id]
        pool.draining = [r for r in pool.draining if r is not run]
        if reason:
            pool.last_error = reason

    async def _finish_draining_run(
        self, client: httpx.AsyncClient, pool: TokenPoolState, run: ManagedRun
    ) -> None:
        if run.inflight > 0:
            return
        await self.finish_run(client, pool, run)

    def acquire_run(self, run: ManagedRun) -> None:
        run.inflight += 1
        run.request_count += 1

    def release_run(self, pool: TokenPoolState, run: ManagedRun) -> None:
        if run.inflight > 0:
            run.inflight -= 1
        if run.inflight == 0 and pool.runs.get(run.agent_id) is not run:
            asyncio.ensure_future(self._finish_draining_run_run(pool, run))

    async def _finish_draining_run_run(self, pool: TokenPoolState, run: ManagedRun) -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            await self._finish_draining_run(client, pool, run)

    def is_session_invalid_error(self, status_code: int, error_body: str) -> bool:
        if status_code < 400:
            return False
        session_errors = {
            "freebuff_update_required",
            "waiting_room_required",
            "waiting_room_queued",
            "session_superseded",
            "session_expired",
        }
        try:
            data = json.loads(error_body)
            error = data.get("error", "")
            return error in session_errors
        except (json.JSONDecodeError, TypeError):
            return False

    def is_run_invalid_error(self, status_code: int, error_body: str) -> bool:
        if status_code != 400:
            return False
        msg = error_body.lower()
        return "runid not found" in msg or "runid not running" in msg

    def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        pool = self._get_pool(credential_identifier)
        token = pool.token if pool else credential_identifier
        return {"Authorization": f"Bearer {token}"}

    async def prewarm(
        self, client: httpx.AsyncClient, credential_paths: List[str]
    ) -> None:
        agent_ids = list(self._agent_models.keys())
        for path in credential_paths:
            pool = self._get_pool(path)
            if not pool:
                continue
            try:
                await self.ensure_session(client, pool)
            except Exception as e:
                lib_logger.debug(f"Freebuff [{pool.name}]: session prewarm failed: {e}")
            for agent_id in agent_ids:
                try:
                    await self.start_run(client, pool, agent_id)
                except Exception as e:
                    lib_logger.debug(f"Freebuff [{pool.name}]: run prewarm for {agent_id} failed: {e}")
        lib_logger.info("Freebuff: prewarm complete")

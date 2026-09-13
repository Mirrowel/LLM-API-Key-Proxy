# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Usage data storage.

Persists per-credential usage state to the shared ``usage`` storage engine:
one row per credential plus one small metadata row for the accessor index and
global fair-cycle state. The former whole-file ``usage_*.json`` medium is no
longer read or written (operator ruling: fresh start, old files untouched).
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..types import (
    WindowStats,
    TotalStats,
    ModelStats,
    GroupStats,
    CredentialState,
    CooldownInfo,
    FairCycleState,
)
from ...storage.engine import get_engine
from ...error_handler import mask_credential
from ..identity.registry import derive_accessor_id
from ...core.constants import (
    DEFAULT_MAX_CONCURRENT_PER_KEY,
    DEFAULT_OPTIMAL_CONCURRENT_PER_KEY,
)

lib_logger = logging.getLogger("rotator_library")


def _format_timestamp(ts: Optional[float]) -> Optional[str]:
    """Format a unix timestamp as a human-readable local time string."""
    if ts is None:
        return None
    try:
        # Use local timezone for human readability
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return None


_TIMESTAMP_FIELDS = (
    "started_at",
    "reset_at",
    "max_recorded_at",
    "first_used_at",
    "last_used_at",
    "created_at",
    "last_updated",
    "until",
    "exhausted_at",
)


def with_human_timestamps(value: Any) -> Any:
    """Regenerate display-only ``*_human`` strings on a stored-shaped value.

    Human-readable timestamps are derived view fields, so they are stripped
    before storage and regenerated here when a display path needs them.
    """

    if isinstance(value, dict):
        for key in list(value):
            item = value[key]
            if key in _TIMESTAMP_FIELDS and item is not None:
                value.setdefault(f"{key}_human", _format_timestamp(item))
            else:
                with_human_timestamps(item)
    elif isinstance(value, list):
        for item in value:
            with_human_timestamps(item)
    return value


class UsageStorage:
    """
    Handles persistence of usage data to the storage engine.

    Features:
    - One engine row per credential (no whole-file rewrite)
    - Debounced saves to reduce I/O
    - Derived accessors: raw upstream keys never reach storage
    """

    CURRENT_SCHEMA_VERSION = 3
    _META_SUFFIX = "__meta__"

    def __init__(
        self,
        file_path: Union[str, Path],
        save_debounce_seconds: float = 5.0,
    ):
        """
        Initialize storage.

        Args:
            file_path: Legacy usage.json path; used only to namespace rows
            save_debounce_seconds: Minimum time between saves
        """
        self.file_path = Path(file_path)
        self.save_debounce_seconds = save_debounce_seconds
        self._namespace = hashlib.sha256(
            str(self.file_path).encode("utf-8")
        ).hexdigest()[:16]
        self._engine = get_engine("usage")

        self._last_save: float = 0
        self._pending_save: bool = False
        self._save_lock = asyncio.Lock()
        self._dirty: bool = False

    def _row_key(self, stable_id: str) -> str:
        return f"{self._namespace}:{stable_id}"

    def _meta_key(self) -> str:
        return f"{self._namespace}:{self._META_SUFFIX}"

    def _prefix(self) -> str:
        return f"{self._namespace}:"

    async def load(
        self,
    ) -> tuple[Dict[str, CredentialState], Dict[str, Dict[str, Any]], bool]:
        """
        Load usage data from the engine.

        Returns:
            Tuple of (states dict, fair_cycle_global dict, loaded bool)
        """
        async with self._save_lock:
            try:
                meta = await self._engine.aget(self._meta_key())
                fair_cycle_global: Dict[str, Dict[str, Any]] = {}
                if meta:
                    try:
                        meta_data = json.loads(meta)
                    except (ValueError, TypeError):
                        meta_data = None
                    if isinstance(meta_data, dict):
                        fair_cycle_global = (
                            meta_data.get("fair_cycle_global", {}) or {}
                        )

                states: Dict[str, CredentialState] = {}
                rows = await asyncio.to_thread(
                    lambda: list(self._engine.iterate(prefix=self._prefix()))
                )
                for key, raw, _meta in rows:
                    if key == self._meta_key():
                        continue
                    stable_id = key[len(self._prefix()) :]
                    try:
                        cred_data = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(cred_data, dict):
                        continue
                    state = self._parse_credential_state(stable_id, cred_data)
                    if state:
                        states[stable_id] = state

                loaded = bool(states) or meta is not None
                lib_logger.info(
                    f"Loaded {len(states)} credentials from usage engine "
                    f"namespace {self._namespace}"
                )
                return states, fair_cycle_global, loaded
            except Exception as e:
                lib_logger.error(f"Failed to load usage data: {e}")
                return {}, {}, False

    async def save(
        self,
        states: Dict[str, CredentialState],
        fair_cycle_global: Optional[Dict[str, Dict[str, Any]]] = None,
        force: bool = False,
    ) -> bool:
        """
        Save usage data to the engine, one row per credential.

        Args:
            states: Dict of stable_id -> CredentialState
            fair_cycle_global: Global fair cycle state
            force: Force save even if debounce not elapsed

        Returns:
            True if saved, False if skipped or failed
        """
        now = time.time()

        # Check debounce
        if not force and (now - self._last_save) < self.save_debounce_seconds:
            self._dirty = True
            return False

        async with self._save_lock:
            try:
                accessor_index: Dict[str, str] = {}
                keep: set[str] = set()
                for stable_id, state in states.items():
                    keep.add(self._row_key(stable_id))
                    serialized = self._serialize_credential_state(state)
                    if not str(state.accessor).startswith("private:"):
                        accessor_index[
                            derive_accessor_id(state.accessor)
                        ] = stable_id
                    await self._engine.aset(
                        self._row_key(stable_id),
                        json.dumps(serialized).encode("utf-8"),
                    )

                meta = {
                    "schema_version": self.CURRENT_SCHEMA_VERSION,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "accessor_index": accessor_index,
                    "fair_cycle_global": fair_cycle_global or {},
                }
                keep.add(self._meta_key())
                await self._engine.aset(
                    self._meta_key(), json.dumps(meta).encode("utf-8")
                )

                existing = await asyncio.to_thread(
                    lambda: list(self._engine.keys(prefix=self._prefix()))
                )
                for key in existing:
                    if key not in keep:
                        await self._engine.adelete(key)

                self._last_save = now
                self._dirty = False
                lib_logger.debug(
                    f"Saved {len(states)} credentials to usage engine "
                    f"namespace {self._namespace}"
                )
                return True

            except Exception as e:
                lib_logger.error(f"Failed to save usage data: {e}")
                self._dirty = True
                return False

    async def save_if_dirty(
        self,
        states: Dict[str, CredentialState],
        fair_cycle_global: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> bool:
        """
        Save if there are pending changes.

        Args:
            states: Dict of stable_id -> CredentialState
            fair_cycle_global: Global fair cycle state

        Returns:
            True if saved, False otherwise
        """
        if self._dirty:
            return await self.save(states, fair_cycle_global, force=True)
        return False

    def mark_dirty(self) -> None:
        """Mark data as changed, needing save."""
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._dirty

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _parse_window_stats(self, name: str, data: Dict[str, Any]) -> WindowStats:
        """Parse window stats from storage data."""
        return WindowStats(
            name=name,
            request_count=data.get("request_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            thinking_tokens=data.get("thinking_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            prompt_tokens_cache_read=data.get("prompt_tokens_cache_read", 0),
            prompt_tokens_cache_write=data.get("prompt_tokens_cache_write", 0),
            total_tokens=data.get("total_tokens", 0),
            approx_cost=data.get("approx_cost", 0.0),
            started_at=data.get("started_at"),
            reset_at=data.get("reset_at"),
            limit=data.get("limit"),
            max_recorded_requests=data.get("max_recorded_requests"),
            max_recorded_at=data.get("max_recorded_at"),
            first_used_at=data.get("first_used_at"),
            last_used_at=data.get("last_used_at"),
        )

    def _serialize_window_stats(self, window: WindowStats) -> Dict[str, Any]:
        """Serialize window stats for storage (display fields excluded)."""
        return {
            "request_count": window.request_count,
            "success_count": window.success_count,
            "failure_count": window.failure_count,
            "prompt_tokens": window.prompt_tokens,
            "completion_tokens": window.completion_tokens,
            "thinking_tokens": window.thinking_tokens,
            "output_tokens": window.output_tokens,
            "prompt_tokens_cache_read": window.prompt_tokens_cache_read,
            "prompt_tokens_cache_write": window.prompt_tokens_cache_write,
            "total_tokens": window.total_tokens,
            "approx_cost": window.approx_cost,
            "started_at": window.started_at,
            "reset_at": window.reset_at,
            "limit": window.limit,
            "max_recorded_requests": window.max_recorded_requests,
            "max_recorded_at": window.max_recorded_at,
            "first_used_at": window.first_used_at,
            "last_used_at": window.last_used_at,
        }

    def _parse_total_stats(self, data: Dict[str, Any]) -> TotalStats:
        """Parse total stats from storage data."""
        return TotalStats(
            request_count=data.get("request_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            thinking_tokens=data.get("thinking_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            prompt_tokens_cache_read=data.get("prompt_tokens_cache_read", 0),
            prompt_tokens_cache_write=data.get("prompt_tokens_cache_write", 0),
            total_tokens=data.get("total_tokens", 0),
            approx_cost=data.get("approx_cost", 0.0),
            first_used_at=data.get("first_used_at"),
            last_used_at=data.get("last_used_at"),
        )

    def _serialize_total_stats(self, totals: TotalStats) -> Dict[str, Any]:
        """Serialize total stats for storage (display fields excluded)."""
        return {
            "request_count": totals.request_count,
            "success_count": totals.success_count,
            "failure_count": totals.failure_count,
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            "thinking_tokens": totals.thinking_tokens,
            "output_tokens": totals.output_tokens,
            "prompt_tokens_cache_read": totals.prompt_tokens_cache_read,
            "prompt_tokens_cache_write": totals.prompt_tokens_cache_write,
            "total_tokens": totals.total_tokens,
            "approx_cost": totals.approx_cost,
            "first_used_at": totals.first_used_at,
            "last_used_at": totals.last_used_at,
        }

    def _parse_model_stats(self, data: Dict[str, Any]) -> ModelStats:
        """Parse model stats from storage data."""
        windows = {}
        for name, wdata in data.get("windows", {}).items():
            # Skip legacy "total" window - now tracked in totals
            if name == "total":
                continue
            windows[name] = self._parse_window_stats(name, wdata)

        totals = self._parse_total_stats(data.get("totals", {}))

        return ModelStats(windows=windows, totals=totals)

    def _serialize_model_stats(self, stats: ModelStats) -> Dict[str, Any]:
        """Serialize model stats for storage."""
        return {
            "windows": {
                name: self._serialize_window_stats(window)
                for name, window in stats.windows.items()
            },
            "totals": self._serialize_total_stats(stats.totals),
        }

    def _parse_group_stats(self, data: Dict[str, Any]) -> GroupStats:
        """Parse group stats from storage data."""
        windows = {}
        for name, wdata in data.get("windows", {}).items():
            # Skip legacy "total" window - now tracked in totals
            if name == "total":
                continue
            windows[name] = self._parse_window_stats(name, wdata)

        totals = self._parse_total_stats(data.get("totals", {}))

        return GroupStats(windows=windows, totals=totals)

    def _serialize_group_stats(self, stats: GroupStats) -> Dict[str, Any]:
        """Serialize group stats for storage."""
        return {
            "windows": {
                name: self._serialize_window_stats(window)
                for name, window in stats.windows.items()
            },
            "totals": self._serialize_total_stats(stats.totals),
        }

    def _parse_credential_state(
        self,
        stable_id: str,
        data: Dict[str, Any],
    ) -> Optional[CredentialState]:
        """Parse a credential state from storage data."""
        try:
            # Parse model_usage
            model_usage = {}
            for key, usage_data in data.get("model_usage", {}).items():
                model_usage[key] = self._parse_model_stats(usage_data)

            # Parse group_usage
            group_usage = {}
            for key, usage_data in data.get("group_usage", {}).items():
                group_usage[key] = self._parse_group_stats(usage_data)

            # Parse credential-level totals
            totals = self._parse_total_stats(data.get("totals", {}))

            # Parse cooldowns
            cooldowns = {}
            for key, cdata in data.get("cooldowns", {}).items():
                cooldowns[key] = CooldownInfo(
                    reason=cdata.get("reason", "unknown"),
                    until=cdata.get("until", 0),
                    started_at=cdata.get("started_at", 0),
                    source=cdata.get("source", "system"),
                    model_or_group=cdata.get("model_or_group"),
                    backoff_count=cdata.get("backoff_count", 0),
                )

            # Parse fair cycle
            fair_cycle = {}
            for key, fcdata in data.get("fair_cycle", {}).items():
                fair_cycle[key] = FairCycleState(
                    exhausted=fcdata.get("exhausted", False),
                    exhausted_at=fcdata.get("exhausted_at"),
                    exhausted_reason=fcdata.get("exhausted_reason"),
                    cycle_request_count=fcdata.get("cycle_request_count", 0),
                    model_or_group=key,
                )

            raw_max_concurrent = data.get(
                "max_concurrent", DEFAULT_MAX_CONCURRENT_PER_KEY
            )
            try:
                max_concurrent = int(raw_max_concurrent)
            except (TypeError, ValueError):
                max_concurrent = DEFAULT_MAX_CONCURRENT_PER_KEY
            if max_concurrent <= 0:
                max_concurrent = -1

            raw_optimal_concurrent = data.get(
                "optimal_concurrent", DEFAULT_OPTIMAL_CONCURRENT_PER_KEY
            )
            try:
                optimal_concurrent = int(raw_optimal_concurrent)
            except (TypeError, ValueError):
                optimal_concurrent = DEFAULT_OPTIMAL_CONCURRENT_PER_KEY
            if optimal_concurrent <= 0:
                optimal_concurrent = -1

            return CredentialState(
                stable_id=stable_id,
                provider=data.get("provider", "unknown"),
                accessor=data.get("accessor", stable_id),
                display_name=data.get("display_name"),
                tier=data.get("tier"),
                priority=data.get("priority", 999),
                model_usage=model_usage,
                group_usage=group_usage,
                totals=totals,
                cooldowns=cooldowns,
                fair_cycle=fair_cycle,
                active_requests=0,  # Always starts at 0
                optimal_concurrent=optimal_concurrent,
                max_concurrent=max_concurrent,
                created_at=data.get("created_at"),
                last_updated=data.get("last_updated"),
            )

        except Exception as e:
            lib_logger.warning(
                f"Failed to parse credential {mask_credential(stable_id, style='full')}: {e}"
            )
            return None

    def _serialize_credential_state(self, state: CredentialState) -> Dict[str, Any]:
        """Serialize a credential state for storage (display fields excluded)."""
        # Serialize cooldowns (only active ones)
        now = time.time()
        cooldowns = {}
        for key, cd in state.cooldowns.items():
            if cd.until > now:  # Only save active cooldowns
                cooldowns[key] = {
                    "reason": cd.reason,
                    "until": cd.until,
                    "started_at": cd.started_at,
                    "source": cd.source,
                    "model_or_group": cd.model_or_group,
                    "backoff_count": cd.backoff_count,
                }

        # Serialize fair cycle
        fair_cycle = {}
        for key, fc in state.fair_cycle.items():
            fair_cycle[key] = {
                "exhausted": fc.exhausted,
                "exhausted_at": fc.exhausted_at,
                "exhausted_reason": fc.exhausted_reason,
                "cycle_request_count": fc.cycle_request_count,
            }

        return {
            "provider": state.provider,
            "accessor": derive_accessor_id(state.accessor),
            "private": str(state.accessor).startswith("private:"),
            "display_name": state.display_name,
            "tier": state.tier,
            "priority": state.priority,
            "model_usage": {
                key: self._serialize_model_stats(stats)
                for key, stats in state.model_usage.items()
            },
            "group_usage": {
                key: self._serialize_group_stats(stats)
                for key, stats in state.group_usage.items()
            },
            "totals": self._serialize_total_stats(state.totals),
            "cooldowns": cooldowns,
            "fair_cycle": fair_cycle,
            "optimal_concurrent": state.optimal_concurrent,
            "max_concurrent": state.max_concurrent,
            "created_at": state.created_at,
            "last_updated": state.last_updated,
        }

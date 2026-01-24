# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Usage data storage.

Handles loading and saving usage data to JSON files.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..types import (
    WindowStats,
    TotalStats,
    ModelStats,
    GroupStats,
    CredentialState,
    CooldownInfo,
    FairCycleState,
    GlobalFairCycleState,
    StorageSchema,
)
from ...utils.resilient_io import ResilientStateWriter, safe_read_json
from ...error_handler import mask_credential

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


class UsageStorage:
    """
    Handles persistence of usage data to JSON files.

    Features:
    - Async file I/O with aiofiles
    - Atomic writes (write to temp, then rename)
    - Automatic schema migration
    - Debounced saves to reduce I/O
    """

    CURRENT_SCHEMA_VERSION = 2

    def __init__(
        self,
        file_path: Union[str, Path],
        save_debounce_seconds: float = 5.0,
    ):
        """
        Initialize storage.

        Args:
            file_path: Path to the usage.json file
            save_debounce_seconds: Minimum time between saves
        """
        self.file_path = Path(file_path)
        self.save_debounce_seconds = save_debounce_seconds

        self._last_save: float = 0
        self._pending_save: bool = False
        self._save_lock = asyncio.Lock()
        self._dirty: bool = False
        self._writer = ResilientStateWriter(self.file_path, lib_logger)

    async def load(
        self,
    ) -> tuple[Dict[str, CredentialState], Dict[str, Dict[str, Any]], bool]:
        """
        Load usage data from file.

        Returns:
            Tuple of (states dict, fair_cycle_global dict, loaded_from_file bool)
        """
        if not self.file_path.exists():
            return {}, {}, False

        try:
            async with self._file_lock():
                data = safe_read_json(self.file_path, lib_logger, parse_json=True)

            if not data:
                return {}, {}, True

            # Check schema version
            version = data.get("schema_version", 1)
            if version < self.CURRENT_SCHEMA_VERSION:
                lib_logger.info(
                    f"Migrating usage data from v{version} to v{self.CURRENT_SCHEMA_VERSION}"
                )
                data = self._migrate(data, version)

            # Parse credentials
            states = {}
            for stable_id, cred_data in data.get("credentials", {}).items():
                state = self._parse_credential_state(stable_id, cred_data)
                if state:
                    states[stable_id] = state

            lib_logger.info(f"Loaded {len(states)} credentials from {self.file_path}")
            return states, data.get("fair_cycle_global", {}), True

        except json.JSONDecodeError as e:
            lib_logger.error(f"Failed to parse usage file: {e}")
            return {}, {}, True
        except Exception as e:
            lib_logger.error(f"Failed to load usage file: {e}")
            return {}, {}, True

    async def save(
        self,
        states: Dict[str, CredentialState],
        fair_cycle_global: Optional[Dict[str, Dict[str, Any]]] = None,
        force: bool = False,
    ) -> bool:
        """
        Save usage data to file.

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
                # Build storage data
                data = {
                    "schema_version": self.CURRENT_SCHEMA_VERSION,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "credentials": {},
                    "accessor_index": {},
                    "fair_cycle_global": fair_cycle_global or {},
                }

                for stable_id, state in states.items():
                    data["credentials"][stable_id] = self._serialize_credential_state(
                        state
                    )
                    data["accessor_index"][state.accessor] = stable_id

                saved = self._writer.write(data)

                if saved:
                    self._last_save = now
                    self._dirty = False
                    lib_logger.debug(
                        f"Saved {len(states)} credentials to {self.file_path}"
                    )
                    return True

                self._dirty = True
                return False

            except Exception as e:
                lib_logger.error(f"Failed to save usage file: {e}")
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

    def _file_lock(self):
        """Get a lock for file operations."""
        return self._save_lock

    def _migrate(self, data: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """Migrate data from older schema versions."""
        if from_version == 1:
            # v1 -> v2: Add accessor_index, restructure credentials
            data["schema_version"] = 2
            data.setdefault("accessor_index", {})
            data.setdefault("fair_cycle_global", {})

            # v1 used file paths as keys, v2 uses stable_ids
            # For migration, treat paths as stable_ids
            old_credentials = data.get("credentials", data.get("key_states", {}))
            new_credentials = {}

            for key, cred_data in old_credentials.items():
                # Use path as temporary stable_id
                stable_id = cred_data.get("stable_id", key)
                new_credentials[stable_id] = cred_data
                new_credentials[stable_id]["accessor"] = key

            data["credentials"] = new_credentials

        return data

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
        """Serialize window stats for storage."""
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
            "started_at_human": _format_timestamp(window.started_at),
            "reset_at": window.reset_at,
            "reset_at_human": _format_timestamp(window.reset_at),
            "limit": window.limit,
            "max_recorded_requests": window.max_recorded_requests,
            "max_recorded_at": window.max_recorded_at,
            "max_recorded_at_human": _format_timestamp(window.max_recorded_at),
            "first_used_at": window.first_used_at,
            "first_used_at_human": _format_timestamp(window.first_used_at),
            "last_used_at": window.last_used_at,
            "last_used_at_human": _format_timestamp(window.last_used_at),
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
        """Serialize total stats for storage."""
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
            "first_used_at_human": _format_timestamp(totals.first_used_at),
            "last_used_at": totals.last_used_at,
            "last_used_at_human": _format_timestamp(totals.last_used_at),
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
                max_concurrent=data.get("max_concurrent"),
                created_at=data.get("created_at"),
                last_updated=data.get("last_updated"),
            )

        except Exception as e:
            lib_logger.warning(
                f"Failed to parse credential {mask_credential(stable_id, style='full')}: {e}"
            )
            return None

    def _serialize_credential_state(self, state: CredentialState) -> Dict[str, Any]:
        """Serialize a credential state for storage."""
        # Serialize cooldowns (only active ones)
        now = time.time()
        cooldowns = {}
        for key, cd in state.cooldowns.items():
            if cd.until > now:  # Only save active cooldowns
                cooldowns[key] = {
                    "reason": cd.reason,
                    "until": cd.until,
                    "until_human": _format_timestamp(cd.until),
                    "started_at": cd.started_at,
                    "started_at_human": _format_timestamp(cd.started_at),
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
                "exhausted_at_human": _format_timestamp(fc.exhausted_at),
                "exhausted_reason": fc.exhausted_reason,
                "cycle_request_count": fc.cycle_request_count,
            }

        return {
            "provider": state.provider,
            "accessor": state.accessor,
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
            "max_concurrent": state.max_concurrent,
            "created_at": state.created_at,
            "created_at_human": _format_timestamp(state.created_at),
            "last_updated": state.last_updated,
            "last_updated_human": _format_timestamp(state.last_updated),
        }

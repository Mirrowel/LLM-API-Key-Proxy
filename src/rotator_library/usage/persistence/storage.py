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
    UsageStats,
    WindowStats,
    CredentialState,
    CooldownInfo,
    FairCycleState,
    GlobalFairCycleState,
    StorageSchema,
)

lib_logger = logging.getLogger("rotator_library")


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

    async def load(
        self,
    ) -> tuple[Dict[str, CredentialState], Dict[str, Dict[str, Any]]]:
        """
        Load usage data from file.

        Returns:
            Dict of stable_id -> CredentialState
        """
        if not self.file_path.exists():
            lib_logger.info(f"No usage file found at {self.file_path}, starting fresh")
            return {}, {}

        try:
            async with self._file_lock():
                content = await self._read_file()

            if not content:
                return {}, {}

            data = json.loads(content)

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
            return states, data.get("fair_cycle_global", {})

        except json.JSONDecodeError as e:
            lib_logger.error(f"Failed to parse usage file: {e}")
            return {}, {}
        except Exception as e:
            lib_logger.error(f"Failed to load usage file: {e}")
            return {}, {}

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

                # Write atomically
                await self._write_file(json.dumps(data, indent=2))

                self._last_save = now
                self._dirty = False

                lib_logger.debug(f"Saved {len(states)} credentials to {self.file_path}")
                return True

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

    async def _read_file(self) -> str:
        """Read file contents asynchronously."""
        try:
            import aiofiles

            async with aiofiles.open(self.file_path, "r", encoding="utf-8") as f:
                return await f.read()
        except ImportError:
            # Fallback to sync read
            with open(self.file_path, "r", encoding="utf-8") as f:
                return f.read()

    async def _write_file(self, content: str) -> None:
        """Write file contents atomically."""
        temp_path = self.file_path.with_suffix(".tmp")

        try:
            import aiofiles

            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                await f.write(content)
        except ImportError:
            # Fallback to sync write
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)

        # Atomic rename
        temp_path.replace(self.file_path)

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

    def _parse_usage_stats(self, data: Dict[str, Any]) -> UsageStats:
        """Parse usage stats from storage data."""
        windows = {}
        for name, wdata in data.get("windows", {}).items():
            windows[name] = WindowStats(
                name=name,
                request_count=wdata.get("request_count", 0),
                total_tokens=wdata.get("total_tokens", 0),
                prompt_tokens=wdata.get("prompt_tokens", 0),
                prompt_tokens_cached=wdata.get("prompt_tokens_cached", 0),
                completion_tokens=wdata.get("completion_tokens", 0),
                approx_cost=wdata.get("approx_cost", 0.0),
                started_at=wdata.get("started_at"),
                reset_at=wdata.get("reset_at"),
                limit=wdata.get("limit"),
            )

        return UsageStats(
            windows=windows,
            total_requests=data.get("total_requests", 0),
            total_successes=data.get("total_successes", 0),
            total_failures=data.get("total_failures", 0),
            total_tokens=data.get("total_tokens", 0),
            total_prompt_tokens_cached=data.get("total_prompt_tokens_cached", 0),
            total_approx_cost=data.get("total_approx_cost", 0.0),
            first_used_at=data.get("first_used_at"),
            last_used_at=data.get("last_used_at"),
            model_request_counts=dict(data.get("model_request_counts", {})),
        )

    def _serialize_usage_stats(self, usage: UsageStats) -> Dict[str, Any]:
        """Serialize usage stats for storage."""
        windows = {}
        for name, window in usage.windows.items():
            windows[name] = {
                "request_count": window.request_count,
                "total_tokens": window.total_tokens,
                "prompt_tokens": window.prompt_tokens,
                "prompt_tokens_cached": window.prompt_tokens_cached,
                "completion_tokens": window.completion_tokens,
                "approx_cost": window.approx_cost,
                "started_at": window.started_at,
                "reset_at": window.reset_at,
                "limit": window.limit,
            }

        return {
            "windows": windows,
            "total_requests": usage.total_requests,
            "total_successes": usage.total_successes,
            "total_failures": usage.total_failures,
            "total_tokens": usage.total_tokens,
            "total_prompt_tokens_cached": usage.total_prompt_tokens_cached,
            "total_approx_cost": usage.total_approx_cost,
            "first_used_at": usage.first_used_at,
            "last_used_at": usage.last_used_at,
            "model_request_counts": usage.model_request_counts,
        }

    def _parse_credential_state(
        self,
        stable_id: str,
        data: Dict[str, Any],
    ) -> Optional[CredentialState]:
        """Parse a credential state from storage data."""
        try:
            usage = self._parse_usage_stats(data)

            model_usage = {
                key: self._parse_usage_stats(usage_data)
                for key, usage_data in data.get("model_usage", {}).items()
            }
            group_usage = {
                key: self._parse_usage_stats(usage_data)
                for key, usage_data in data.get("group_usage", {}).items()
            }

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
                usage=usage,
                model_usage=model_usage,
                group_usage=group_usage,
                cooldowns=cooldowns,
                fair_cycle=fair_cycle,
                active_requests=0,  # Always starts at 0
                max_concurrent=data.get("max_concurrent"),
                created_at=data.get("created_at"),
                last_updated=data.get("last_updated"),
            )

        except Exception as e:
            lib_logger.warning(f"Failed to parse credential {stable_id}: {e}")
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
            "accessor": state.accessor,
            "display_name": state.display_name,
            "tier": state.tier,
            "priority": state.priority,
            **self._serialize_usage_stats(state.usage),
            "model_usage": {
                key: self._serialize_usage_stats(usage)
                for key, usage in state.model_usage.items()
            },
            "group_usage": {
                key: self._serialize_usage_stats(usage)
                for key, usage in state.group_usage.items()
            },
            "cooldowns": cooldowns,
            "fair_cycle": fair_cycle,
            "max_concurrent": state.max_concurrent,
            "created_at": state.created_at,
            "last_updated": state.last_updated,
        }

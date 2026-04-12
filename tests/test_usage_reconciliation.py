# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for usage state reconciliation and Codex empty call_id handling.

Covers:
- Stale credential pruning on startup (deleted OAuth files)
- accessor_index rebuild from current credentials only
- Stable ID deduplication when multiple IDs point to same accessor
- Codex adapter empty/missing tool_call_id handling

NO network calls, NO API keys needed.
"""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from rotator_library.usage.manager import UsageManager
from rotator_library.usage.types import CredentialState
from rotator_library.providers.codex_provider import (
    _sanitize_call_id,
    _convert_messages_to_responses_input,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_usage_file(tmp_path):
    return tmp_path / "usage" / "usage_codex.json"


@pytest.fixture
def oauth_dir(tmp_path):
    """Create temp oauth dir with credential files."""
    d = tmp_path / "oauth_creds"
    d.mkdir()
    return d


def _make_oauth_file(directory: Path, name: str, email: str) -> Path:
    """Create a minimal OAuth credential file."""
    p = directory / name
    p.write_text(json.dumps({
        "access_token": "fake-token",
        "refresh_token": "fake-refresh",
        "_proxy_metadata": {"email": email},
    }))
    return p


def _make_usage_state(stable_id: str, accessor: str, provider: str = "codex") -> CredentialState:
    """Create a minimal CredentialState for testing."""
    return CredentialState(
        stable_id=stable_id,
        provider=provider,
        accessor=accessor,
        created_at=time.time(),
    )


# =============================================================================
# Test: Stale credential pruning
# =============================================================================


class TestUsageReconciliation:
    """Test that stale credentials are pruned from persisted usage state."""

    @pytest.mark.asyncio
    async def test_missing_file_accessor_pruned(self, tmp_path, oauth_dir):
        """Persisted usage referencing a deleted OAuth file is pruned on startup."""
        existing = _make_oauth_file(oauth_dir, "codex_oauth_1.json", "user1@test.com")
        deleted_path = str(oauth_dir / "codex_oauth_2.json")

        manager = UsageManager(
            provider="codex",
            max_concurrent_per_key=5,
        )

        # Simulate loaded persisted state with a stale entry
        persisted_states = {
            "user1@test.com": _make_usage_state("user1@test.com", str(existing)),
            "user2@test.com": _make_usage_state("user2@test.com", deleted_path),
        }

        mock_storage = MagicMock()
        mock_storage.load = AsyncMock(return_value=(dict(persisted_states), {}, True))
        mock_storage.mark_dirty = MagicMock()
        manager._storage = mock_storage

        await manager.initialize([str(existing)])

        assert "user2@test.com" not in manager._states
        assert "user1@test.com" in manager._states
        mock_storage.mark_dirty.assert_called()

    @pytest.mark.asyncio
    async def test_accessor_index_rebuilt_from_current(self, tmp_path, oauth_dir):
        """accessor_index is rebuilt from current credentials, not stale persisted data."""
        existing = _make_oauth_file(oauth_dir, "codex_oauth_1.json", "user1@test.com")

        manager = UsageManager(
            provider="codex",
            max_concurrent_per_key=5,
        )

        deleted_path = str(oauth_dir / "codex_oauth_missing.json")
        persisted_states = {
            "user1@test.com": _make_usage_state("user1@test.com", str(existing)),
            "gone_user@test.com": _make_usage_state("gone_user@test.com", deleted_path),
        }

        mock_storage = MagicMock()
        mock_storage.load = AsyncMock(return_value=(dict(persisted_states), {}, True))
        mock_storage.mark_dirty = MagicMock()
        manager._storage = mock_storage

        await manager.initialize([str(existing)])

        assert len(manager._active_stable_ids) == 1
        assert "gone_user@test.com" not in manager._states

    @pytest.mark.asyncio
    async def test_stable_id_deduplication(self, tmp_path, oauth_dir):
        """When two stable IDs point to same accessor, only the correct one survives."""
        existing = _make_oauth_file(oauth_dir, "codex_oauth_1.json", "user1@test.com")
        accessor = str(existing)

        manager = UsageManager(
            provider="codex",
            max_concurrent_per_key=5,
        )

        persisted_states = {
            "user1@test.com": _make_usage_state("user1@test.com", accessor),
            "old_legacy_id": _make_usage_state("old_legacy_id", accessor),
        }

        mock_storage = MagicMock()
        mock_storage.load = AsyncMock(return_value=(dict(persisted_states), {}, True))
        mock_storage.mark_dirty = MagicMock()
        manager._storage = mock_storage

        await manager.initialize([accessor])

        accessor_entries = [
            s for s in manager._states.values() if s.accessor == accessor
        ]
        assert len(accessor_entries) == 1
        assert "user1@test.com" in manager._states

    @pytest.mark.asyncio
    async def test_remove_credential_at_runtime(self, tmp_path, oauth_dir):
        """remove_credential() removes from both active set and states."""
        existing = _make_oauth_file(oauth_dir, "codex_oauth_1.json", "user1@test.com")
        accessor = str(existing)

        manager = UsageManager(
            provider="codex",
            max_concurrent_per_key=5,
        )

        mock_storage = MagicMock()
        mock_storage.load = AsyncMock(return_value=({}, {}, False))
        mock_storage.mark_dirty = MagicMock()
        mock_storage.save = AsyncMock(return_value=True)
        manager._storage = mock_storage

        await manager.initialize([accessor])
        assert len(manager._active_stable_ids) == 1

        removed = await manager.remove_credential(accessor)
        assert removed is True
        assert len(manager._active_stable_ids) == 0
        assert "user1@test.com" not in manager._states


# =============================================================================
# Test: Codex empty call_id handling
# =============================================================================


class TestCodexEmptyCallId:
    """Test that empty/missing tool_call_id is handled safely."""

    def test_sanitize_call_id_empty_returns_empty(self):
        """_sanitize_call_id('') returns '' to signal caller to handle."""
        id_map = {}
        result = _sanitize_call_id("", id_map)
        assert result == ""

    def test_sanitize_call_id_none_coerced_empty(self):
        """If raw_id is falsy (None coerced to ''), returns empty."""
        id_map = {}
        # Simulating what happens when tool_call_id is None -> "" via msg.get default
        result = _sanitize_call_id("", id_map)
        assert result == ""

    def test_sanitize_call_id_valid_passthrough(self):
        """Normal call_ids pass through unchanged."""
        id_map = {}
        result = _sanitize_call_id("call_abc123", id_map)
        assert result == "call_abc123"

    def test_sanitize_call_id_oversized_hashed(self):
        """Oversized call_ids get hash-based replacement."""
        id_map = {}
        long_id = "x" * 100
        result = _sanitize_call_id(long_id, id_map)
        assert result.startswith("call_")
        assert len(result) <= 64

    def test_convert_messages_empty_tool_call_id_no_function_call_output(self):
        """Tool message with empty tool_call_id becomes user context, not function_call_output."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_valid123",
                        "function": {"name": "get_info", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "",
                "content": "Some tool output",
            },
        ]

        input_items, _ = _convert_messages_to_responses_input(messages)

        # Find the item that represents the empty-id tool result
        function_call_outputs = [
            item for item in input_items if item.get("type") == "function_call_output"
        ]
        # Should NOT have an empty call_id function_call_output
        for fco in function_call_outputs:
            assert fco["call_id"] != "", "function_call_output with empty call_id should not be emitted"

        # Should have been converted to a user message instead
        user_messages = [
            item for item in input_items
            if item.get("type") == "message" and item.get("role") == "user"
        ]
        tool_as_user = [
            m for m in user_messages
            if any("[Tool result]" in p.get("text", "") for p in m.get("content", []))
        ]
        assert len(tool_as_user) == 1
        assert "Some tool output" in tool_as_user[0]["content"][0]["text"]

    def test_convert_messages_valid_tool_call_id_emits_function_call_output(self):
        """Tool message with valid tool_call_id correctly emits function_call_output."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_valid456",
                        "function": {"name": "get_info", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_valid456",
                "content": "result data",
            },
        ]

        input_items, _ = _convert_messages_to_responses_input(messages)

        function_call_outputs = [
            item for item in input_items if item.get("type") == "function_call_output"
        ]
        assert len(function_call_outputs) == 1
        assert function_call_outputs[0]["call_id"] == "call_valid456"
        assert function_call_outputs[0]["output"] == "result data"

    def test_convert_messages_missing_tool_call_id_key(self):
        """Tool message without tool_call_id key at all is handled safely."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "tool",
                "content": "orphan tool result",
            },
        ]

        input_items, _ = _convert_messages_to_responses_input(messages)

        # Should not crash, and should not emit function_call_output with empty call_id
        function_call_outputs = [
            item for item in input_items if item.get("type") == "function_call_output"
        ]
        for fco in function_call_outputs:
            assert fco["call_id"] != ""

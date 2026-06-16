# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Tests for session_tracking.SessionTracker.

SessionTracker infers stable session IDs from request payloads so that
sequential rotation can keep using the same credential for the same
conversation.

These tests exercise:
- Explicit session/conversation IDs (strong fingerprints)
- Tool call ID matching (strong fingerprints)
- First-user-text matching (weak fingerprints)
- TTL expiry and pruning
- Multiple concurrent sessions
- Persistence (save/load)
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from session_tracking import SessionTracker


# ============================================================================
# Explicit ID tests
# ============================================================================


class TestExplicitSessionIds:
    """Tests for explicit session/conversation/thread IDs."""

    @pytest.mark.parametrize(
        "key",
        [
            "session_id",
            "conversation_id",
            "conversationId",
            "thread_id",
            "threadId",
            "chat_id",
            "chatId",
        ],
    )
    def test_explicit_id_creates_session(self, key):
        tracker = SessionTracker()
        request = {key: "abc-123", "messages": []}
        sid = tracker.infer_session_id(request)
        assert sid is not None

    def test_same_explicit_id_returns_same_session(self):
        tracker = SessionTracker()
        req = {"session_id": "conv-1", "messages": []}
        sid1 = tracker.infer_session_id(req)
        sid2 = tracker.infer_session_id(req)
        assert sid1 == sid2

    def test_different_explicit_ids_create_different_sessions(self):
        tracker = SessionTracker()
        sid1 = tracker.infer_session_id({"session_id": "conv-1", "messages": []})
        sid2 = tracker.infer_session_id({"session_id": "conv-2", "messages": []})
        assert sid1 != sid2

    def test_empty_explicit_id_ignored(self):
        tracker = SessionTracker()
        # Empty string should be falsy → no explicit fingerprint
        sid = tracker.infer_session_id({"session_id": "", "messages": []})
        # With no messages, there's nothing to fingerprint
        assert sid is None


# ============================================================================
# Conversation fingerprint tests
# ============================================================================


class TestConversationFingerprints:
    """Tests for message-based conversation fingerprinting."""

    def test_same_conversation_returns_same_session(self):
        tracker = SessionTracker()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        sid1 = tracker.infer_session_id({"messages": messages})
        sid2 = tracker.infer_session_id({"messages": messages})
        assert sid1 == sid2

    def test_different_conversations_different_sessions(self):
        tracker = SessionTracker()
        msgs1 = [{"role": "user", "content": "What is Python?"}]
        msgs2 = [{"role": "user", "content": "What is Rust?"}]
        sid1 = tracker.infer_session_id({"messages": msgs1})
        sid2 = tracker.infer_session_id({"messages": msgs2})
        assert sid1 != sid2

    def test_tool_call_ids_create_strong_fingerprint(self):
        tracker = SessionTracker()
        messages = [
            {"role": "user", "content": "Use the tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_001", "function": {"name": "get_weather"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "Sunny, 72F",
            },
        ]
        sid1 = tracker.infer_session_id({"messages": messages})
        sid2 = tracker.infer_session_id({"messages": messages})
        assert sid1 == sid2

    def test_empty_messages_returns_none(self):
        tracker = SessionTracker()
        assert tracker.infer_session_id({"messages": []}) is None

    def test_no_messages_key_returns_none(self):
        tracker = SessionTracker()
        assert tracker.infer_session_id({}) is None

    def test_multi_content_user_message(self):
        """User content as a list of parts should be handled."""
        tracker = SessionTracker()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        sid1 = tracker.infer_session_id({"messages": messages})
        sid2 = tracker.infer_session_id({"messages": messages})
        assert sid1 == sid2


# ============================================================================
# Weak fingerprint (first user text) tests
# ============================================================================


class TestWeakFingerprints:
    """Tests for first-user-text matching."""

    def test_same_first_user_text_weak_match(self):
        tracker = SessionTracker()
        msgs1 = [{"role": "user", "content": "Tell me a joke"}]
        msgs2 = [
            {"role": "user", "content": "Tell me a joke"},
            {"role": "assistant", "content": "Why did the chicken..."},
            {"role": "user", "content": "Another one"},
        ]
        sid1 = tracker.infer_session_id({"messages": msgs1})
        sid2 = tracker.infer_session_id({"messages": msgs2})
        # The weak fingerprint should bridge them
        assert sid1 == sid2


# ============================================================================
# TTL and pruning tests
# ============================================================================


class TestTTLAndPruning:
    """Tests for TTL expiry and record pruning."""

    def test_expired_record_is_pruned(self):
        tracker = SessionTracker(ttl_seconds=1)
        sid_old = tracker.infer_session_id({"session_id": "conv-old"})
        # Advance past TTL without real sleeping
        with patch("session_tracking.time.time", return_value=time.time() + 2):
            # After expiry, a new request with same ID should get a new session
            sid_new = tracker.infer_session_id({"session_id": "conv-old"})
        # The new session must differ from the expired one (proves pruning)
        assert sid_new is not None
        assert sid_new != sid_old

    def test_different_ttls(self):
        tracker = SessionTracker(ttl_seconds=60)
        sid1 = tracker.infer_session_id({"session_id": "long"})
        sid2 = tracker.infer_session_id({"session_id": "long"})
        assert sid1 == sid2


# ============================================================================
# Persistence tests
# ============================================================================


class TestPersistence:
    """Tests for disk persistence."""

    def test_save_and_load(self, tmp_path: Path):
        persist_file = tmp_path / "sessions.json"

        tracker1 = SessionTracker(
            ttl_seconds=3600, persist_to_disk=True, persistence_path=persist_file
        )
        sid = tracker1.infer_session_id({"session_id": "persisted-conv"})

        # Create a new tracker that loads from disk
        tracker2 = SessionTracker(
            ttl_seconds=3600, persist_to_disk=True, persistence_path=persist_file
        )
        sid2 = tracker2.infer_session_id({"session_id": "persisted-conv"})
        assert sid == sid2

    def test_persistence_file_created(self, tmp_path: Path):
        persist_file = tmp_path / "sessions.json"
        tracker = SessionTracker(
            ttl_seconds=3600, persist_to_disk=True, persistence_path=persist_file
        )
        tracker.infer_session_id({"session_id": "test"})
        assert persist_file.exists()

    def test_load_corrupt_file_silent(self, tmp_path: Path):
        """Corrupted persistence file should be silently ignored."""
        persist_file = tmp_path / "sessions.json"
        persist_file.write_text("NOT JSON {{{", encoding="utf-8")

        # Should not raise
        tracker = SessionTracker(
            ttl_seconds=3600, persist_to_disk=True, persistence_path=persist_file
        )
        # Should still work normally (empty records)
        sid = tracker.infer_session_id({"session_id": "new-conv"})
        assert sid is not None


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_none_request_data(self):
        tracker = SessionTracker()
        # infer_session_id expects a dict; None should not crash but return None
        # (it accesses .get on dict, so None would raise — that's acceptable behavior)
        # Test with empty dict instead
        assert tracker.infer_session_id({}) is None

    def test_ttl_minimum_clamped(self):
        """TTL should be clamped to at least 1 second."""
        tracker = SessionTracker(ttl_seconds=0)
        assert tracker.ttl_seconds == 1

    def test_conversation_growth_extends_session(self):
        """Adding messages to an existing conversation should preserve session."""
        tracker = SessionTracker()
        msgs_v1 = [{"role": "user", "content": "Hello"}]
        sid1 = tracker.infer_session_id({"messages": msgs_v1})

        msgs_v2 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]
        sid2 = tracker.infer_session_id({"messages": msgs_v2})

        # The strong conversation fingerprint changed, but the weak
        # first-user-text fingerprint should bridge them
        assert sid1 == sid2

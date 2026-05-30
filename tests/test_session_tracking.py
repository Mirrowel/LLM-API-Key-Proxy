import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rotator_library.session_tracking import SessionTracker, SessionTrackingHints
from rotator_library.client.streaming import StreamingHandler


class SessionTrackerTests(unittest.TestCase):
    def test_weak_first_user_only_does_not_reuse_session(self):
        tracker = SessionTracker(ttl_seconds=3600)
        request = {"messages": [{"role": "user", "content": "hello world"}]}

        session_a = tracker.infer_session(request, provider="gemini", model="pro")
        session_b = tracker.infer_session(request, provider="gemini", model="pro")

        self.assertIsNotNone(session_a.session_id)
        self.assertIsNotNone(session_b.session_id)
        self.assertNotEqual(session_a.session_id, session_b.session_id)
        self.assertIsNone(session_a.affinity_key)

    def test_single_long_prompt_does_not_become_sticky_by_itself(self):
        tracker = SessionTracker(ttl_seconds=3600)
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": "Please perform a careful detailed analysis of this standalone prompt and provide a structured answer with several paragraphs.",
                }
            ]
        }

        session_a = tracker.infer_session(request, provider="gemini", model="pro")
        session_b = tracker.infer_session(request, provider="gemini", model="pro")

        self.assertNotEqual(session_a.session_id, session_b.session_id)
        self.assertIsNone(session_a.affinity_key)

    def test_shared_system_and_user_prompt_does_not_become_sticky_by_itself(self):
        tracker = SessionTracker(ttl_seconds=3600)
        request = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a shared coding harness with stable instructions used by many independent sessions.",
                },
                {
                    "role": "user",
                    "content": "Please review this standalone request carefully and provide the exact structured output.",
                },
            ]
        }

        session_a = tracker.infer_session(request, provider="gemini", model="pro")
        session_b = tracker.infer_session(request, provider="gemini", model="pro")

        self.assertNotEqual(session_a.session_id, session_b.session_id)
        self.assertIsNone(session_a.affinity_key)

    def test_explicit_ids_are_weak_unless_configured_as_trusted(self):
        request = {"conversation_id": "stable-client-id"}

        conservative = SessionTracker(ttl_seconds=3600)
        first = conservative.infer_session(request, provider="gemini", model="pro")
        second = conservative.infer_session(request, provider="gemini", model="pro")
        self.assertNotEqual(first.session_id, second.session_id)

        trusted = SessionTracker(
            ttl_seconds=3600,
            trusted_explicit_fields=["conversation_id"],
        )
        first = trusted.infer_session(request, provider="gemini", model="pro")
        second = trusted.infer_session(request, provider="gemini", model="pro")
        self.assertEqual(first.session_id, second.session_id)

    def test_trusted_explicit_fields_can_come_from_env(self):
        request = {"conversation_id": "stable-client-id"}

        with patch.dict(os.environ, {"TRUSTED_SESSION_ID_FIELDS": "conversation_id"}):
            tracker = SessionTracker(ttl_seconds=3600)

        first = tracker.infer_session(request, provider="gemini", model="pro")
        second = tracker.infer_session(request, provider="gemini", model="pro")

        self.assertEqual(first.session_id, second.session_id)

    def test_multiple_message_anchors_reuse_session_when_tools_are_pruned(self):
        tracker = SessionTracker(ttl_seconds=3600)

        original = {
            "messages": [
                {"role": "user", "content": "Please inspect the account quota and explain the cooldown status in detail."},
                {
                    "role": "assistant",
                    "content": "I will inspect the quota files and compare the cooldown windows.",
                    "tool_calls": [
                        {"id": "call_123", "type": "function", "function": {"name": "read"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "quota payload"},
                {"role": "user", "content": "Now summarize the result and note what changed since yesterday."},
            ]
        }
        pruned_tools = {
            "messages": [
                {"role": "user", "content": "Please inspect the account quota and explain the cooldown status in detail."},
                {
                    "role": "assistant",
                    "content": "I will inspect the quota files and compare the cooldown windows.",
                },
                {"role": "user", "content": "Now summarize the result and note what changed since yesterday."},
            ]
        }

        session_a = tracker.infer_session(original, provider="gemini", model="pro")
        session_b = tracker.infer_session(pruned_tools, provider="gemini", model="pro")

        self.assertEqual(session_a.session_id, session_b.session_id)
        self.assertIn(session_b.confidence, {"strong", "probable"})

    def test_response_anchors_bridge_next_request(self):
        tracker = SessionTracker(ttl_seconds=3600)
        request = {
            "messages": [
                {"role": "user", "content": "Investigate why the nightly quota reset failed for the Gemini account."}
            ]
        }
        inferred = tracker.infer_session(request, provider="gemini", model="pro")
        tracker.record_response(
            inferred.session_id,
            provider="gemini",
            model="pro",
            response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The nightly quota reset failed because the reset timestamp was parsed in local time.",
                        }
                    }
                ]
            },
        )

        next_request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "The nightly quota reset failed because the reset timestamp was parsed in local time.",
                },
                {"role": "user", "content": "Patch the timestamp parser now."},
            ]
        }
        continued = tracker.infer_session(next_request, provider="gemini", model="pro")

        self.assertEqual(inferred.session_id, continued.session_id)

    def test_record_response_uses_stored_namespace_when_tracking_namespace_omitted(self):
        tracker = SessionTracker(ttl_seconds=3600)
        hints = SessionTrackingHints(session_scope="quota-group-pro")
        request = {
            "messages": [
                {"role": "user", "content": "Investigate scoped response tracking with enough text."},
                {"role": "assistant", "content": "The response anchor must stay in the provider session scope."},
            ]
        }
        inferred = tracker.infer_session(
            request, provider="gemini", model="pro", scope_key="gemini", hints=hints
        )
        tracker.record_response(
            inferred.session_id,
            provider="gemini",
            model="pro",
            scope_key="gemini",
            response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The scoped response anchor survived without passing the tracking namespace.",
                        }
                    }
                ]
            },
        )

        continued = tracker.infer_session(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "The scoped response anchor survived without passing the tracking namespace.",
                    },
                    {"role": "user", "content": "Continue within the same provider session scope."},
                ]
            },
            provider="gemini",
            model="pro",
            scope_key="gemini",
            hints=hints,
        )

        self.assertEqual(inferred.session_id, continued.session_id)

    def test_provider_model_scope_is_isolated(self):
        tracker = SessionTracker(ttl_seconds=3600)
        request = {
            "messages": [
                {"role": "user", "content": "Compare the provider model cache behavior using the same long prompt."},
                {"role": "assistant", "content": "The same text should not cross provider or model scopes."},
            ]
        }

        gemini = tracker.infer_session(request, provider="gemini", model="pro")
        openai = tracker.infer_session(request, provider="openai", model="pro")
        gemini_flash = tracker.infer_session(request, provider="gemini", model="flash")

        self.assertNotEqual(gemini.session_id, openai.session_id)
        self.assertNotEqual(gemini.session_id, gemini_flash.session_id)

    def test_allowed_usage_scope_is_isolated(self):
        tracker = SessionTracker(ttl_seconds=3600)
        request = {
            "messages": [
                {"role": "user", "content": "Compare scoped routing behavior using enough text for anchors."},
                {"role": "assistant", "content": "The same conversation text must not cross classifier scopes."},
            ]
        }

        public = tracker.infer_session(
            request, provider="gemini", model="pro", scope_key="gemini"
        )
        scoped = tracker.infer_session(
            request, provider="gemini", model="pro", scope_key="classifier:user-a:gemini"
        )

        self.assertNotEqual(public.session_id, scoped.session_id)

    def test_provider_strong_hint_reuses_session(self):
        tracker = SessionTracker(ttl_seconds=3600)
        hints = SessionTrackingHints(strong_anchors=["native-session-abc"])

        first = tracker.infer_session({}, provider="custom", model="m", hints=hints)
        second = tracker.infer_session({}, provider="custom", model="m", hints=hints)

        self.assertEqual(first.session_id, second.session_id)
        self.assertIsNotNone(first.affinity_key)

    def test_provider_session_scope_can_override_model_inside_allowed_scope(self):
        tracker = SessionTracker(ttl_seconds=3600)
        hints = SessionTrackingHints(
            strong_anchors=["native-session-abc"],
            session_scope="quota-group-pro",
        )

        pro = tracker.infer_session(
            {}, provider="gemini", model="pro", scope_key="gemini", hints=hints
        )
        flash = tracker.infer_session(
            {}, provider="gemini", model="flash", scope_key="gemini", hints=hints
        )
        other_scope = tracker.infer_session(
            {},
            provider="gemini",
            model="flash",
            scope_key="classifier:user-a:gemini",
            hints=hints,
        )

        self.assertEqual(pro.session_id, flash.session_id)
        self.assertNotEqual(pro.session_id, other_scope.session_id)

    def test_compaction_lineage_creates_new_session_without_strong_anchor(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original = {
            "messages": [
                {"role": "user", "content": "Please analyze the routing state and remember this detailed anchor text."},
                {"role": "assistant", "content": "I analyzed the routing state and found a useful continuity anchor."},
            ]
        }
        compacted = {
            "messages": [
                {
                    "role": "system",
                    "content": "Summary of previous conversation: I analyzed the routing state and found a useful continuity anchor.",
                },
                {"role": "user", "content": "Continue from the compressed context."},
            ]
        }

        parent = tracker.infer_session(original, provider="gemini", model="pro")
        child = tracker.infer_session(compacted, provider="gemini", model="pro")

        self.assertNotEqual(parent.session_id, child.session_id)
        self.assertTrue(child.possible_compaction)
        self.assertEqual(parent.session_id, child.lineage_parent_session_id)

    def test_compaction_detection_is_conservative_to_early_system_messages(self):
        tracker = SessionTracker(ttl_seconds=3600)
        user_summary = {
            "messages": [
                {"role": "user", "content": "Summary of previous conversation: continue this task."}
            ]
        }

        inferred = tracker.infer_session(user_summary, provider="gemini", model="pro")

        self.assertFalse(inferred.possible_compaction)

    def test_persistence_round_trips_current_schema_with_anchor_metadata(self):
        request = {
            "messages": [
                {"role": "user", "content": "Persist this first detailed user anchor for the session tracker."},
                {"role": "assistant", "content": "Persist this second detailed assistant anchor as well."},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session_stickiness.json"
            tracker = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
                persistence_flush_interval_seconds=0,
            )
            first = tracker.infer_session(request, provider="gemini", model="pro")
            tracker.flush()

            restored = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
                persistence_flush_interval_seconds=0,
            )
            second = restored.infer_session(request, provider="gemini", model="pro")

        self.assertEqual(first.session_id, second.session_id)

    def test_unversioned_persistence_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session_stickiness.json"
            path.write_text(
                json.dumps(
                    {
                        "sessions": {
                            "old-session": {
                                "namespace": "provider:gemini:model:pro",
                                "expires_at": 9999999999,
                                "anchors": [],
                            }
                        },
                        "anchors": {},
                    }
                ),
                encoding="utf-8",
            )
            tracker = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
            )

        self.assertEqual(tracker._sessions, {})
        self.assertEqual(tracker._anchors, {})

    def test_streaming_chunk_collector_preserves_response_anchors(self):
        handler = StreamingHandler()
        assistant_parts = []
        tool_call_ids = []

        handler._collect_session_response_anchors(
            'data: {"choices":[{"delta":{"content":"hello ","tool_calls":[{"id":"call_1"}]}}]}\n\n',
            assistant_parts,
            tool_call_ids,
        )

        self.assertEqual(assistant_parts, ["hello "])
        self.assertEqual(tool_call_ids, ["call_1"])


if __name__ == "__main__":
    unittest.main()

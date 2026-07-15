import asyncio
import os
import sys
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rotator_library.session_tracking import (
    SessionAnchor,
    SessionTracker,
    SessionTrackingHints,
    _AnchorRecord,
)
from rotator_library.client.streaming import StreamingHandler


class SessionTrackerTests(unittest.TestCase):
    @staticmethod
    def _long_text(label, repeats=12):
        return " ".join([f"{label} carries deterministic substantial history evidence"] * repeats)

    @staticmethod
    def _response(content, response_id=None):
        response = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        if response_id:
            response["id"] = response_id
        return response

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

    def test_response_id_anchor_bridges_responses_previous_response_id(self):
        tracker = SessionTracker(ttl_seconds=3600)
        inferred = tracker.infer_session(
            {"messages": [{"role": "user", "content": "Start a Responses API conversation."}]},
            provider="openai",
            model="gpt-test",
        )
        tracker.record_response(
            inferred.session_id,
            provider="openai",
            model="gpt-test",
            response={"id": "resp_parent", "object": "response"},
        )

        continued = tracker.infer_session(
            {"messages": [{"role": "user", "content": "Continue."}]},
            provider="openai",
            model="gpt-test",
            hints={"strong_anchors": ["responses_previous_response_id:resp_parent"]},
        )

        self.assertEqual(inferred.session_id, continued.session_id)
        self.assertEqual(continued.confidence, "strong")

    def test_one_response_user_copy_does_not_create_compaction_lineage(self):
        tracker = SessionTracker(ttl_seconds=3600)
        request = {
            "messages": [
                {"role": "user", "content": "Prepare a detailed durable summary for later compaction."}
            ]
        }
        parent = tracker.infer_session(request, provider="gemini", model="pro")
        summary = " ".join(
            [
                "The compaction summary says the routing investigation found that response anchors",
                "should identify a parent session without continuing sticky routing for the child context.",
            ]
            * 8
        )
        tracker.record_response(
            parent.session_id,
            provider="gemini",
            model="pro",
            response={"choices": [{"message": {"role": "assistant", "content": summary}}]},
        )

        independent = tracker.infer_session(
            {"messages": [{"role": "user", "content": summary}]},
            provider="gemini",
            model="pro",
        )
        repeated = tracker.infer_session(
            {"messages": [{"role": "user", "content": summary}]},
            provider="gemini",
            model="pro",
        )

        self.assertFalse(independent.possible_compaction)
        self.assertIsNone(independent.lineage_parent_session_id)
        self.assertFalse(repeated.possible_compaction)
        self.assertNotEqual(parent.session_id, independent.session_id)

    def test_unmarked_summary_requires_two_distinct_response_events(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original = {
            "messages": [
                {"role": "system", "content": "Stable harness instructions."},
                {"role": "user", "content": self._long_text("initial-user")},
                {"role": "assistant", "content": self._long_text("initial-assistant")},
                {"role": "user", "content": self._long_text("follow-up-user")},
            ]
        }
        parent = tracker.infer_session(original, provider="gemini", model="pro")
        response_a = " ".join(f"alpha{i:02d}longword" for i in range(8))
        response_b = " ".join(f"bravo{i:02d}longword" for i in range(8))
        tracker.record_response(
            parent.session_id,
            provider="gemini",
            model="pro",
            response=self._response(response_a),
        )
        continued = tracker.infer_session(
            {
                "messages": [
                    *original["messages"],
                    {"role": "assistant", "content": response_a},
                    {"role": "user", "content": "Continue the investigation with another result."},
                ]
            },
            provider="gemini",
            model="pro",
        )
        self.assertEqual(parent.session_id, continued.session_id)
        tracker.record_response(
            parent.session_id,
            provider="gemini",
            model="pro",
            response=self._response(response_b),
        )

        filler_a = " ".join(f"middle{i:02d}longword" for i in range(8))
        filler_b = " ".join(f"ending{i:02d}longword" for i in range(8))
        summary = f"{response_a} {filler_a} {response_b} {filler_b}"
        child_request = {"messages": [{"role": "user", "content": summary}]}
        child = tracker.infer_session(child_request, provider="gemini", model="pro")
        replay = tracker.infer_session(child_request, provider="gemini", model="pro")

        self.assertTrue(child.possible_compaction)
        self.assertEqual(parent.session_id, child.lineage_parent_session_id)
        self.assertNotEqual(parent.session_id, child.session_id)
        self.assertEqual(child.session_id, replay.session_id)
        self.assertEqual(parent.session_id, replay.lineage_parent_session_id)

    def test_duplicate_response_content_counts_as_one_response_event(self):
        tracker = SessionTracker(ttl_seconds=3600)
        parent_request = {
            "messages": [
                {"role": "user", "content": self._long_text("parent-user")},
                {"role": "assistant", "content": self._long_text("parent-assistant")},
                {"role": "user", "content": self._long_text("parent-follow-up")},
            ]
        }
        parent = tracker.infer_session(parent_request, provider="gemini", model="pro")
        repeated_response = self._long_text("same-response", repeats=16)
        for _ in range(2):
            tracker.record_response(
                parent.session_id,
                provider="gemini",
                model="pro",
                response=self._response(repeated_response),
            )

        inferred = tracker.infer_session(
            {"messages": [{"role": "user", "content": repeated_response}]},
            provider="gemini",
            model="pro",
        )

        self.assertFalse(inferred.possible_compaction)

    def test_normal_history_with_recorded_assistant_response_is_not_compaction(self):
        tracker = SessionTracker(ttl_seconds=3600)
        user_text = self._long_text("ordinary-user", repeats=16)
        assistant_text = self._long_text("ordinary-assistant", repeats=16)
        parent = tracker.infer_session(
            {"messages": [{"role": "user", "content": user_text}]},
            provider="deepseek",
            model="chat",
        )
        tracker.record_response(
            parent.session_id,
            provider="deepseek",
            model="chat",
            response=self._response(assistant_text),
        )

        continued = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                    {"role": "user", "content": "Continue with the next ordinary turn."},
                ]
            },
            provider="deepseek",
            model="chat",
        )

        self.assertFalse(continued.possible_compaction)
        self.assertEqual(parent.session_id, continued.session_id)

    def test_long_classifier_instruction_quoting_one_response_is_independent(self):
        tracker = SessionTracker(ttl_seconds=3600)
        parent = tracker.infer_session(
            {
                "messages": [
                    {"role": "system", "content": self._long_text("roleplay-system")},
                    {"role": "user", "content": self._long_text("roleplay-user")},
                    {"role": "assistant", "content": self._long_text("roleplay-history")},
                    {"role": "user", "content": self._long_text("roleplay-next")},
                ]
            },
            provider="mistral",
            model="large",
        )
        response_text = " ".join(f"scene{i:02d}longword" for i in range(8))
        tracker.record_response(
            parent.session_id,
            provider="mistral",
            model="large",
            response=self._response(response_text),
        )
        preamble = " ".join(f"classifier{i:02d}longword" for i in range(8))
        suffix = " ".join(f"location{i:02d}longword" for i in range(16))
        classifier_prompt = f"{preamble} {response_text} {suffix}"
        response_chunks = set(tracker._content_chunk_hashes(response_text))
        classifier_chunks = set(tracker._content_chunk_hashes(classifier_prompt))
        self.assertTrue(response_chunks & classifier_chunks)

        classified = tracker.infer_session(
            {"messages": [{"role": "user", "content": classifier_prompt}]},
            provider="mistral",
            model="large",
        )

        self.assertFalse(classified.possible_compaction)
        self.assertIsNone(classified.lineage_parent_session_id)
        self.assertNotEqual(parent.session_id, classified.session_id)

    def test_ordinary_context_contraction_without_summary_stays_on_parent(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": self._long_text(f"branch-history-{index}"),
            }
            for index in range(5)
        ]
        parent = tracker.infer_session(
            {"messages": original},
            provider="deepseek",
            model="chat",
        )
        contracted = {"messages": [original[0], original[1], original[4]]}

        first = tracker.infer_session(contracted, provider="deepseek", model="chat")
        replay = tracker.infer_session(contracted, provider="deepseek", model="chat")

        self.assertFalse(first.possible_compaction)
        self.assertEqual(parent.session_id, first.session_id)
        self.assertEqual(parent.session_id, replay.session_id)
        self.assertEqual(len(tracker._sessions[parent.session_id].history_signatures), 5)

    def test_response_provenance_survives_echo_in_request_history(self):
        tracker = SessionTracker(ttl_seconds=3600)
        user_text = self._long_text("provenance-user")
        response_text = self._long_text("provenance-response")
        parent = tracker.infer_session(
            {"messages": [{"role": "user", "content": user_text}]},
            provider="gemini",
            model="pro",
        )
        tracker.record_response(
            parent.session_id,
            provider="gemini",
            model="pro",
            response=self._response(response_text),
        )
        response_records = {
            value: record.group
            for value, record in tracker._anchors.items()
            if record.session_id == parent.session_id
            and record.source == "response"
            and record.group
            and record.group.startswith("response_event:")
        }

        tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": "Continue after preserving response provenance."},
                ]
            },
            provider="gemini",
            model="pro",
        )

        self.assertTrue(response_records)
        for value, group in response_records.items():
            self.assertEqual(tracker._anchors[value].source, "response")
            self.assertEqual(tracker._anchors[value].group, group)

    def test_system_or_developer_quote_of_one_response_does_not_bridge_session(self):
        for role in ("system", "developer"):
            with self.subTest(role=role):
                tracker = SessionTracker(ttl_seconds=3600)
                parent = tracker.infer_session(
                    {
                        "messages": [
                            {"role": "user", "content": self._long_text(f"{role}-quote-user")},
                            {
                                "role": "assistant",
                                "content": self._long_text(f"{role}-quote-history"),
                            },
                            {"role": "user", "content": self._long_text(f"{role}-quote-next")},
                        ]
                    },
                    provider="gemini",
                    model="pro",
                )
                response_text = self._long_text(f"{role}-quoted-response", repeats=16)
                tracker.record_response(
                    parent.session_id,
                    provider="gemini",
                    model="pro",
                    response=self._response(response_text),
                )

                quoted = tracker.infer_session(
                    {
                        "messages": [
                            {"role": role, "content": response_text},
                            {"role": "user", "content": "Start an independent request."},
                        ]
                    },
                    provider="gemini",
                    model="pro",
                )

                self.assertFalse(quoted.possible_compaction)
                self.assertNotEqual(parent.session_id, quoted.session_id)

    def test_size_only_long_prompt_overlap_does_not_create_compaction_lineage(self):
        tracker = SessionTracker(ttl_seconds=3600)
        long_text = " ".join(
            [
                "This is a normal long prompt without compaction markers that repeats",
                "ordinary request context and should not be treated as a summary descendant.",
            ]
            * 8
        )
        original = {
            "messages": [
                {"role": "user", "content": long_text},
                {"role": "assistant", "content": "A normal second message adds continuity evidence."},
            ]
        }
        repeated = {"messages": [{"role": "user", "content": long_text}]}

        tracker.infer_session(original, provider="gemini", model="pro")
        inferred = tracker.infer_session(repeated, provider="gemini", model="pro")

        self.assertFalse(inferred.possible_compaction)
        self.assertIsNone(inferred.lineage_parent_session_id)

    def test_size_only_two_message_history_overlap_does_not_create_compaction_lineage(self):
        tracker = SessionTracker(ttl_seconds=3600)
        first = " ".join(["First ordinary long request message without summary markers."] * 30)
        second = " ".join(["Second ordinary long request message without summary markers."] * 30)
        request = {
            "messages": [
                {"role": "user", "content": first},
                {"role": "assistant", "content": second},
            ]
        }

        tracker.infer_session(request, provider="gemini", model="pro")
        inferred = tracker.infer_session(request, provider="gemini", model="pro")

        self.assertFalse(inferred.possible_compaction)

    def test_middle_only_summary_replacement_remains_same_session(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original_messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": self._long_text(f"middle-history-{index}"),
            }
            for index in range(8)
        ]
        parent = tracker.infer_session(
            {"messages": original_messages},
            provider="gemini",
            model="pro",
        )
        middle_summary = {
            "role": "developer",
            "content": "Summary of previous conversation: " + original_messages[3]["content"],
        }
        partially_replaced = {
            "messages": [
                *original_messages[:3],
                middle_summary,
                *original_messages[4:],
            ]
        }

        continued = tracker.infer_session(
            partially_replaced,
            provider="gemini",
            model="pro",
        )

        self.assertFalse(continued.possible_compaction)
        self.assertEqual(parent.session_id, continued.session_id)
        self.assertEqual(len(tracker._sessions[parent.session_id].history_signatures), 8)

    def test_early_summary_replacing_a_minority_of_history_remains_same_session(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original_messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": self._long_text(f"minority-history-{index}"),
            }
            for index in range(8)
        ]
        parent = tracker.infer_session(
            {"messages": original_messages},
            provider="gemini",
            model="pro",
        )
        summary = {
            "role": "developer",
            "content": "Summary of previous conversation: " + original_messages[1]["content"],
        }
        partially_replaced = {
            "messages": [summary, original_messages[0], *original_messages[3:]]
        }

        continued = tracker.infer_session(
            partially_replaced,
            provider="gemini",
            model="pro",
        )

        self.assertFalse(continued.possible_compaction)
        self.assertEqual(parent.session_id, continued.session_id)
        self.assertEqual(len(tracker._sessions[parent.session_id].history_signatures), 8)

    def test_retaining_exactly_half_of_history_is_not_compaction(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original_messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": self._long_text(f"half-history-{index}"),
            }
            for index in range(8)
        ]
        parent = tracker.infer_session(
            {"messages": original_messages},
            provider="gemini",
            model="pro",
        )
        summary = {
            "role": "system",
            "content": "Compacted context: " + original_messages[1]["content"],
        }
        half_retained = {"messages": [summary, *original_messages[4:]]}

        continued = tracker.infer_session(
            half_retained,
            provider="gemini",
            model="pro",
        )

        self.assertFalse(continued.possible_compaction)
        self.assertEqual(parent.session_id, continued.session_id)

    def test_marked_summary_replacing_most_history_creates_child(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original_messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": self._long_text(f"most-history-{index}"),
            }
            for index in range(8)
        ]
        parent = tracker.infer_session(
            {"messages": original_messages},
            provider="gemini",
            model="pro",
        )
        summary = {
            "role": "system",
            "content": "Compacted context: " + original_messages[1]["content"],
        }
        mostly_replaced = {"messages": [summary, *original_messages[-2:]]}

        child = tracker.infer_session(
            mostly_replaced,
            provider="gemini",
            model="pro",
        )

        self.assertTrue(child.possible_compaction)
        self.assertEqual(parent.session_id, child.lineage_parent_session_id)
        self.assertNotEqual(parent.session_id, child.session_id)

    def test_unmarked_two_response_summary_replacing_only_a_minority_is_not_compaction(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original_messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": self._long_text(f"unmarked-partial-{index}"),
            }
            for index in range(8)
        ]
        parent = tracker.infer_session(
            {"messages": original_messages},
            provider="gemini",
            model="pro",
        )
        response_a = " ".join(f"partialalpha{i:02d}word" for i in range(8))
        response_b = " ".join(f"partialbravo{i:02d}word" for i in range(8))
        tracker.record_response(
            parent.session_id,
            provider="gemini",
            model="pro",
            response=self._response(response_a),
        )
        tracker.record_response(
            parent.session_id,
            provider="gemini",
            model="pro",
            response=self._response(response_b),
        )
        filler_a = " ".join(f"partialmiddle{i:02d}word" for i in range(8))
        filler_b = " ".join(f"partialending{i:02d}word" for i in range(8))
        summary = {
            "role": "user",
            "content": f"{response_a} {filler_a} {response_b} {filler_b}",
        }
        mostly_retained = {"messages": [summary, *original_messages[3:]]}

        continued = tracker.infer_session(
            mostly_retained,
            provider="gemini",
            model="pro",
        )

        self.assertFalse(continued.possible_compaction)
        self.assertEqual(parent.session_id, continued.session_id)

    def test_unmarked_two_response_summary_replacing_most_history_creates_child(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original_messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": self._long_text(f"unmarked-most-{index}"),
            }
            for index in range(8)
        ]
        parent = tracker.infer_session(
            {"messages": original_messages},
            provider="gemini",
            model="pro",
        )
        response_a = " ".join(f"mostalpha{i:02d}longword" for i in range(8))
        response_b = " ".join(f"mostbravo{i:02d}longword" for i in range(8))
        tracker.record_response(
            parent.session_id,
            provider="gemini",
            model="pro",
            response=self._response(response_a),
        )
        tracker.record_response(
            parent.session_id,
            provider="gemini",
            model="pro",
            response=self._response(response_b),
        )
        filler_a = " ".join(f"mostmiddle{i:02d}longword" for i in range(8))
        filler_b = " ".join(f"mostending{i:02d}longword" for i in range(8))
        mostly_replaced = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{response_a} {filler_a} {response_b} {filler_b}",
                },
                original_messages[-1],
            ]
        }

        child = tracker.infer_session(
            mostly_replaced,
            provider="gemini",
            model="pro",
        )

        self.assertTrue(child.possible_compaction)
        self.assertEqual(parent.session_id, child.lineage_parent_session_id)
        self.assertNotEqual(parent.session_id, child.session_id)

    def test_compaction_evidence_cannot_cross_usage_scope(self):
        tracker = SessionTracker(ttl_seconds=3600)
        parent = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": self._long_text("scope-parent-user")},
                    {"role": "assistant", "content": self._long_text("scope-parent-assistant")},
                    {"role": "user", "content": self._long_text("scope-parent-follow-up")},
                ]
            },
            provider="gemini",
            model="pro",
            scope_key="scope-a",
        )
        response_a = " ".join(f"scopealpha{i:02d}longword" for i in range(8))
        response_b = " ".join(f"scopebravo{i:02d}longword" for i in range(8))
        tracker.record_response(
            parent.session_id,
            tracking_namespace=parent.tracking_namespace,
            response=self._response(response_a),
        )
        tracker.record_response(
            parent.session_id,
            tracking_namespace=parent.tracking_namespace,
            response=self._response(response_b),
        )
        filler_a = " ".join(f"scopemiddle{i:02d}longword" for i in range(8))
        filler_b = " ".join(f"scopeending{i:02d}longword" for i in range(8))
        summary = f"{response_a} {filler_a} {response_b} {filler_b}"

        inferred = tracker.infer_session(
            {"messages": [{"role": "user", "content": summary}]},
            provider="gemini",
            model="pro",
            scope_key="scope-b",
        )

        self.assertFalse(inferred.possible_compaction)
        self.assertNotEqual(parent.session_id, inferred.session_id)

    def test_trusted_identity_keeps_live_session_through_compaction(self):
        tracker = SessionTracker(
            ttl_seconds=3600,
            trusted_explicit_fields=["conversation_id"],
        )
        original_messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": self._long_text(f"trusted-history-{index}"),
            }
            for index in range(6)
        ]
        parent = tracker.infer_session(
            {"conversation_id": "trusted-session", "messages": original_messages},
            provider="gemini",
            model="pro",
        )
        compacted = tracker.infer_session(
            {
                "conversation_id": "trusted-session",
                "messages": [
                    {
                        "role": "system",
                        "content": "Summary of previous conversation: "
                        + original_messages[1]["content"],
                    },
                    {"role": "user", "content": "Continue from the summary."},
                ],
            },
            provider="gemini",
            model="pro",
        )

        self.assertTrue(compacted.possible_compaction)
        self.assertEqual(parent.session_id, compacted.session_id)
        self.assertIsNone(compacted.lineage_parent_session_id)

    def test_strong_tool_identity_keeps_live_session_through_compaction(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original_messages = [
            {"role": "user", "content": self._long_text("tool-parent-user")},
            {
                "role": "assistant",
                "content": self._long_text("tool-parent-assistant"),
                "tool_calls": [{"id": "call_persistent", "type": "function"}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_persistent",
                "content": self._long_text("tool-parent-result"),
            },
            {"role": "user", "content": self._long_text("tool-parent-follow-up")},
            {"role": "assistant", "content": self._long_text("tool-parent-later")},
            {"role": "user", "content": self._long_text("tool-parent-final")},
        ]
        parent = tracker.infer_session(
            {"messages": original_messages},
            provider="gemini",
            model="pro",
        )
        compacted = tracker.infer_session(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Compacted context: " + original_messages[1]["content"],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_persistent",
                        "content": "The retained tool result remains authoritative.",
                    },
                ]
            },
            provider="gemini",
            model="pro",
        )

        self.assertTrue(compacted.possible_compaction)
        self.assertEqual(parent.session_id, compacted.session_id)

    def test_changed_non_probe_history_does_not_hit_exact_compaction_replay(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original = {
            "messages": [
                {"role": "user", "content": self._long_text("replay-parent-user")},
                {"role": "assistant", "content": self._long_text("replay-parent-assistant")},
                {"role": "user", "content": self._long_text("replay-parent-follow-up")},
                {"role": "assistant", "content": self._long_text("replay-parent-later")},
            ]
        }
        parent = tracker.infer_session(original, provider="gemini", model="pro")
        summary = {
            "role": "system",
            "content": "Summary of previous conversation: "
            + original["messages"][1]["content"],
        }
        first_request = {
            "messages": [summary, {"role": "user", "content": "Continue branch one."}]
        }
        changed_request = {
            "messages": [summary, {"role": "user", "content": "Continue branch two."}]
        }

        first_child = tracker.infer_session(
            first_request,
            provider="gemini",
            model="pro",
        )
        changed_child = tracker.infer_session(
            changed_request,
            provider="gemini",
            model="pro",
        )

        self.assertEqual(parent.session_id, first_child.lineage_parent_session_id)
        self.assertEqual(parent.session_id, changed_child.lineage_parent_session_id)
        self.assertNotEqual(first_child.session_id, changed_child.session_id)

    def test_marker_without_parent_evidence_does_not_create_compaction(self):
        tracker = SessionTracker(ttl_seconds=3600)

        inferred = tracker.infer_session(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Summary of previous conversation: no known parent evidence exists.",
                    },
                    {"role": "user", "content": "Continue."},
                ]
            },
            provider="gemini",
            model="pro",
        )

        self.assertFalse(inferred.possible_compaction)
        self.assertIsNone(inferred.lineage_parent_session_id)

    def test_all_compaction_markers_are_recognized(self):
        tracker = SessionTracker(ttl_seconds=3600)
        markers = (
            "summary of previous conversation",
            "summary of the previous conversation",
            "summarized conversation",
            "compressed context",
            "compacted context",
            "conversation so far",
            "previous conversation",
            "context reminder",
        )

        for marker in markers:
            with self.subTest(marker=marker):
                self.assertTrue(tracker._has_compaction_marker(f"Prefix {marker} suffix"))

    def test_compaction_probe_size_boundaries(self):
        tracker = SessionTracker(ttl_seconds=3600)

        self.assertFalse(tracker._is_compaction_probe_text("x" * 399))
        self.assertTrue(tracker._is_compaction_probe_text("x" * 400))
        self.assertFalse(tracker._is_compaction_probe_text(" ".join(["x"] * 79)))
        self.assertTrue(tracker._is_compaction_probe_text(" ".join(["x"] * 80)))

    def test_oversized_assistant_history_is_never_a_compaction_probe(self):
        tracker = SessionTracker(ttl_seconds=3600)
        assistant_text = self._long_text("oversized-assistant", repeats=20)
        parent = tracker.infer_session(
            {
                "messages": [
                    {"role": "assistant", "content": assistant_text},
                    {"role": "user", "content": "Continue this established conversation normally."},
                ]
            },
            provider="gemini",
            model="pro",
        )
        repeated = tracker.infer_session(
            {
                "messages": [
                    {"role": "assistant", "content": assistant_text},
                    {"role": "user", "content": "Continue this established conversation normally."},
                ]
            },
            provider="gemini",
            model="pro",
        )

        self.assertFalse(repeated.possible_compaction)
        self.assertEqual(parent.session_id, repeated.session_id)

    def test_oversized_tool_result_is_never_a_compaction_probe(self):
        tracker = SessionTracker(ttl_seconds=3600)
        tool_text = self._long_text("oversized-tool", repeats=20)
        request = {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_stable",
                    "content": tool_text,
                },
                {"role": "user", "content": "Continue after the tool result."},
            ]
        }
        parent = tracker.infer_session(request, provider="gemini", model="pro")
        repeated = tracker.infer_session(request, provider="gemini", model="pro")

        self.assertFalse(repeated.possible_compaction)
        self.assertEqual(parent.session_id, repeated.session_id)

    def test_stale_persistence_job_does_not_overwrite_newer_snapshot(self):
        writes = []

        class FakeWriter:
            def write(self, payload):
                writes.append(payload)
                return True

        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=Path(temp_dir) / "session_stickiness.json",
                persistence_flush_interval_seconds=0,
            )
            tracker._writer = FakeWriter()

            with tracker._lock:
                tracker._dirty = True
                tracker._dirty_generation = 5
                old_job = tracker._prepare_save_locked(force=True)
                tracker._dirty_generation = 6
                new_job = tracker._prepare_save_locked(force=True)

            tracker._write_save_job(new_job)
            tracker._write_save_job(old_job)

        self.assertEqual(len(writes), 1)
        self.assertEqual(tracker._last_persisted_generation, 6)

    def test_failed_persistence_write_stays_dirty_and_recovers(self):
        class FlakyWriter:
            def __init__(self):
                self.outcomes = [False, True]

            def write(self, payload):
                return self.outcomes.pop(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=Path(temp_dir) / "session_stickiness.json",
                persistence_flush_interval_seconds=0,
            )
            tracker._writer = FlakyWriter()
            tracker.infer_session(
                {"messages": [{"role": "user", "content": self._long_text("failed-write")}]},
                provider="gemini",
                model="pro",
            )

            self.assertTrue(tracker._dirty)
            self.assertEqual(tracker._last_persisted_generation, 0)
            generation = tracker._dirty_generation
            tracker.flush()

        self.assertFalse(tracker._dirty)
        self.assertEqual(tracker._last_persisted_generation, generation)

    def test_unexpected_writer_exception_does_not_escape_request(self):
        class RaisingWriter:
            def write(self, payload):
                raise RuntimeError("synthetic writer failure")

        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=Path(temp_dir) / "session_stickiness.json",
                persistence_flush_interval_seconds=0,
            )
            tracker._writer = RaisingWriter()

            inferred = tracker.infer_session(
                {"messages": [{"role": "user", "content": self._long_text("writer-exception")}]},
                provider="gemini",
                model="pro",
            )

        self.assertIsNotNone(inferred.session_id)
        self.assertTrue(tracker._dirty)

    def test_flush_interval_throttles_automatic_write_but_force_flushes(self):
        writes = []

        class FakeWriter:
            def write(self, payload):
                writes.append(payload)
                return True

        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=Path(temp_dir) / "session_stickiness.json",
                persistence_flush_interval_seconds=100,
            )
            tracker._writer = FakeWriter()
            with patch("rotator_library.session_tracking.time.time", return_value=1000.0):
                tracker.infer_session(
                    {"messages": [{"role": "user", "content": self._long_text("throttle-one")}]},
                    provider="gemini",
                    model="pro",
                )
            with patch("rotator_library.session_tracking.time.time", return_value=1050.0):
                tracker.infer_session(
                    {"messages": [{"role": "user", "content": self._long_text("throttle-two")}]},
                    provider="gemini",
                    model="pro",
                )

            self.assertEqual(len(writes), 1)
            self.assertTrue(tracker._dirty)
            tracker.flush()

        self.assertEqual(len(writes), 2)
        self.assertFalse(tracker._dirty)

    def test_disk_write_does_not_hold_session_state_lock(self):
        entered_write = threading.Event()
        release_write = threading.Event()

        class BlockingWriter:
            def write(self, payload):
                entered_write.set()
                if not release_write.wait(timeout=5):
                    raise TimeoutError("test did not release blocked writer")
                return True

        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=Path(temp_dir) / "session_stickiness.json",
                persistence_flush_interval_seconds=0,
            )
            tracker._writer = BlockingWriter()
            worker = threading.Thread(
                target=lambda: tracker.infer_session(
                    {"messages": [{"role": "user", "content": self._long_text("blocked-write")}]},
                    provider="gemini",
                    model="pro",
                )
            )
            worker.start()
            self.assertTrue(entered_write.wait(timeout=2))

            acquired = tracker._lock.acquire(timeout=1)
            if acquired:
                tracker._lock.release()
            release_write.set()
            worker.join(timeout=5)

        self.assertTrue(acquired)
        self.assertFalse(worker.is_alive())

    def test_concurrent_infer_record_and_flush_preserve_anchor_ownership(self):
        tracker = SessionTracker(ttl_seconds=3600)

        def exercise(index):
            user_text = self._long_text(f"concurrent-user-{index}")
            response_text = self._long_text(f"concurrent-response-{index}")
            inferred = tracker.infer_session(
                {"messages": [{"role": "user", "content": user_text}]},
                provider="gemini",
                model="pro",
            )
            tracker.record_response(
                inferred.session_id,
                provider="gemini",
                model="pro",
                response=self._response(response_text),
            )
            tracker.infer_session(
                {
                    "messages": [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": response_text},
                        {"role": "user", "content": f"Continue unique conversation {index}."},
                    ]
                },
                provider="gemini",
                model="pro",
            )
            tracker.flush()

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(exercise, range(24)))

        for value, record in tracker._anchors.items():
            self.assertIn(record.session_id, tracker._sessions)
            self.assertIn(value, tracker._sessions[record.session_id].anchors)
        for state in tracker._sessions.values():
            for value in state.anchors:
                self.assertIn(value, tracker._anchors)
                self.assertEqual(tracker._anchors[value].session_id, state.session_id)

    def test_best_match_tie_breaks_deterministically(self):
        tracker = SessionTracker(ttl_seconds=3600)
        now = 1000.0
        namespace = "scope:test:provider:gemini:model:pro"
        tracker._anchors["anchor-a"] = _AnchorRecord(
            session_id="session-a",
            namespace=namespace,
            strength="medium",
            source="message",
            group="message:0",
            expires_at=now + 100,
            last_seen=now,
        )
        tracker._anchors["anchor-b"] = _AnchorRecord(
            session_id="session-b",
            namespace=namespace,
            strength="medium",
            source="message",
            group="message:1",
            expires_at=now + 100,
            last_seen=now,
        )

        match = tracker._best_match(
            [
                SessionAnchor("anchor-b", "medium", source="message", group="message:1"),
                SessionAnchor("anchor-a", "medium", source="message", group="message:0"),
            ],
            namespace,
            now,
        )

        self.assertEqual(match.session_id, "session-b")

    def test_best_match_prefers_distinct_response_events_on_equal_score(self):
        tracker = SessionTracker(ttl_seconds=3600)
        namespace = "scope:test:provider:gemini:model:pro"
        now = 1000.0
        records = {
            "single-a": ("session-z-single", "response_event:one"),
            "single-b": ("session-z-single", "response_event:one"),
            "diverse-a": ("session-a-diverse", "response_event:one"),
            "diverse-b": ("session-a-diverse", "response_event:two"),
        }
        for value, (session_id, group) in records.items():
            tracker._anchors[value] = _AnchorRecord(
                session_id=session_id,
                namespace=namespace,
                strength="medium",
                source="response",
                group=group,
                expires_at=now + 100,
                last_seen=now,
            )
        anchors = [
            SessionAnchor(
                value,
                "medium",
                source="compaction_probe",
                group="compaction_probe:0",
            )
            for value in records
        ]

        match = tracker._best_match(anchors, namespace, now)

        self.assertEqual(match.session_id, "session-a-diverse")
        self.assertEqual(len(match.response_groups), 2)

    def test_ttl_boundary_prunes_session_and_late_response_cannot_resurrect_it(self):
        tracker = SessionTracker(ttl_seconds=10)
        with patch("rotator_library.session_tracking.time.time", return_value=1000.0):
            inferred = tracker.infer_session(
                {"messages": [{"role": "user", "content": self._long_text("ttl-parent")}]},
                provider="gemini",
                model="pro",
            )
        with patch("rotator_library.session_tracking.time.time", return_value=1010.0):
            tracker.record_response(
                inferred.session_id,
                provider="gemini",
                model="pro",
                response=self._response(self._long_text("late-response")),
            )

        self.assertNotIn(inferred.session_id, tracker._sessions)
        self.assertFalse(
            any(record.session_id == inferred.session_id for record in tracker._anchors.values())
        )

    def test_shared_content_anchor_keeps_first_live_owner(self):
        tracker = SessionTracker(ttl_seconds=3600)
        shared = self._long_text("shared-content", repeats=16)
        first = tracker.infer_session(
            {"messages": [{"role": "user", "content": shared}]},
            provider="gemini",
            model="pro",
        )
        first_owned = {
            value
            for value, record in tracker._anchors.items()
            if record.session_id == first.session_id and ":chunk:" in value
        }
        second = tracker.infer_session(
            {"messages": [{"role": "user", "content": shared}]},
            provider="gemini",
            model="pro",
        )

        self.assertNotEqual(first.session_id, second.session_id)
        self.assertTrue(first_owned)
        for value in first_owned:
            self.assertEqual(tracker._anchors[value].session_id, first.session_id)
            self.assertNotIn(value, tracker._sessions[second.session_id].anchors)

    def test_session_anchor_trim_preserves_bidirectional_ownership(self):
        tracker = SessionTracker(
            ttl_seconds=3600,
            max_anchors_per_session=16,
        )
        messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": self._long_text(f"trim-history-{index}", repeats=16),
            }
            for index in range(12)
        ]
        inferred = tracker.infer_session(
            {"messages": messages},
            provider="gemini",
            model="pro",
        )
        state = tracker._sessions[inferred.session_id]

        self.assertLessEqual(len(state.anchors), 16)
        for value in state.anchors:
            self.assertEqual(tracker._anchors[value].session_id, state.session_id)
        for value, record in tracker._anchors.items():
            if record.session_id == state.session_id:
                self.assertIn(value, state.anchors)

    def test_global_anchor_trim_preserves_bidirectional_ownership(self):
        tracker = SessionTracker(
            ttl_seconds=3600,
            max_anchor_records=100,
        )
        for index in range(24):
            unique_text = " ".join(
                f"global{index:02d}word{word:02d}" for word in range(24)
            )
            tracker.infer_session(
                {"messages": [{"role": "user", "content": unique_text}]},
                provider="gemini",
                model="pro",
            )

        self.assertLessEqual(len(tracker._anchors), 100)
        for value, record in tracker._anchors.items():
            self.assertIn(value, tracker._sessions[record.session_id].anchors)
        for state in tracker._sessions.values():
            for value in state.anchors:
                self.assertEqual(tracker._anchors[value].session_id, state.session_id)

    def test_affinity_is_deterministic_across_tracker_instances(self):
        request = {
            "messages": [
                {"role": "user", "content": self._long_text("affinity-user")},
                {"role": "assistant", "content": self._long_text("affinity-assistant")},
            ]
        }
        first = SessionTracker(ttl_seconds=3600).infer_session(
            request,
            provider="gemini",
            model="pro",
        )
        second = SessionTracker(ttl_seconds=3600).infer_session(
            request,
            provider="gemini",
            model="pro",
        )

        self.assertIsNotNone(first.affinity_key)
        self.assertEqual(first.affinity_key, second.affinity_key)

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
            persisted = json.loads(path.read_text(encoding="utf-8"))
            original_state = tracker._sessions[first.session_id]
            original_records = {
                value: (record.strength, record.source, record.group)
                for value, record in tracker._anchors.items()
                if record.session_id == first.session_id
            }

            restored = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
                persistence_flush_interval_seconds=0,
            )
            second = restored.infer_session(request, provider="gemini", model="pro")
            restored_state = restored._sessions[first.session_id]

        self.assertEqual(persisted["schema_version"], 2)
        self.assertNotIn("anchors", persisted["sessions"][first.session_id])
        self.assertEqual(first.session_id, second.session_id)
        self.assertEqual(first.affinity_key, second.affinity_key)
        self.assertEqual(original_state.history_signatures, restored_state.history_signatures)
        for value, metadata in original_records.items():
            restored_record = restored._anchors[value]
            self.assertEqual(
                (restored_record.strength, restored_record.source, restored_record.group),
                metadata,
            )

    def test_persistence_restart_preserves_compaction_and_replay_binding(self):
        original = {
            "messages": [
                {"role": "user", "content": self._long_text("persist-parent-user")},
                {"role": "assistant", "content": self._long_text("persist-parent-assistant")},
                {"role": "user", "content": self._long_text("persist-parent-follow-up")},
            ]
        }
        response_a = " ".join(f"persistalpha{i:02d}word" for i in range(8))
        response_b = " ".join(f"persistbravo{i:02d}word" for i in range(8))
        filler_a = " ".join(f"persistmiddle{i:02d}word" for i in range(8))
        filler_b = " ".join(f"persistending{i:02d}word" for i in range(8))
        child_request = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{response_a} {filler_a} {response_b} {filler_b}",
                }
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
            parent = tracker.infer_session(original, provider="gemini", model="pro")
            tracker.record_response(
                parent.session_id,
                provider="gemini",
                model="pro",
                response=self._response(response_a),
            )
            tracker.record_response(
                parent.session_id,
                provider="gemini",
                model="pro",
                response=self._response(response_b),
            )
            tracker.flush()

            restored_parent = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
                persistence_flush_interval_seconds=0,
            )
            child = restored_parent.infer_session(
                child_request,
                provider="gemini",
                model="pro",
            )
            restored_parent.flush()

            restored_child = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
                persistence_flush_interval_seconds=0,
            )
            replay = restored_child.infer_session(
                child_request,
                provider="gemini",
                model="pro",
            )

        self.assertTrue(child.possible_compaction)
        self.assertEqual(parent.session_id, child.lineage_parent_session_id)
        self.assertEqual(child.session_id, replay.session_id)
        self.assertEqual(parent.session_id, replay.lineage_parent_session_id)

    def test_malformed_persistence_payloads_are_ignored_without_startup_failure(self):
        payloads = [
            "{not-json",
            json.dumps([]),
            json.dumps({"schema_version": 1, "sessions": {}, "anchors": {}}),
            json.dumps({"schema_version": 2, "sessions": [], "anchors": {}}),
            json.dumps(
                {
                    "schema_version": 2,
                    "sessions": {
                        "bad": {
                            "namespace": "scope:x:provider:y:model:z",
                            "expires_at": "not-a-number",
                        }
                    },
                    "anchors": {},
                }
            ),
        ]

        for payload in payloads:
            with self.subTest(payload=payload[:30]), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "session_stickiness.json"
                path.write_text(payload, encoding="utf-8")
                tracker = SessionTracker(
                    ttl_seconds=3600,
                    persist_to_disk=True,
                    persistence_path=path,
                )
                self.assertEqual(tracker._sessions, {})
                self.assertEqual(tracker._anchors, {})

    def test_persistence_loader_rebuilds_only_valid_anchor_ownership(self):
        namespace = "scope:gemini:provider:gemini:model:pro"
        expires_at = time.time() + 3600
        good_value = f"{namespace}:message:user:{'a' * 64}"
        payload = {
            "schema_version": 2,
            "sessions": {
                "good-session": {
                    "namespace": namespace,
                    "expires_at": expires_at,
                    "last_seen": expires_at - 10,
                    "affinity_key": "stable-affinity",
                    "history_signatures": ["b" * 64, "invalid"],
                },
                "expired-session": {
                    "namespace": namespace,
                    "expires_at": time.time() - 1,
                    "last_seen": time.time() - 10,
                    "history_signatures": ["c" * 64],
                },
            },
            "anchors": {
                good_value: {
                    "session_id": "good-session",
                    "namespace": namespace,
                    "strength": "medium",
                    "source": "message",
                    "group": "message:0:user",
                    "expires_at": expires_at + 100,
                    "last_seen": expires_at - 10,
                },
                f"{namespace}:orphan": {
                    "session_id": "missing-session",
                    "namespace": namespace,
                    "strength": "medium",
                    "source": "message",
                    "group": "message:0:user",
                    "expires_at": expires_at,
                    "last_seen": expires_at - 10,
                },
                f"{namespace}:wrong-namespace": {
                    "session_id": "good-session",
                    "namespace": "scope:other:provider:gemini:model:pro",
                    "strength": "medium",
                    "source": "message",
                    "group": "message:0:user",
                    "expires_at": expires_at,
                    "last_seen": expires_at - 10,
                },
                f"{namespace}:bad-strength": {
                    "session_id": "good-session",
                    "namespace": namespace,
                    "strength": "certain",
                    "source": "message",
                    "group": "message:0:user",
                    "expires_at": expires_at,
                    "last_seen": expires_at - 10,
                },
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session_stickiness.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            tracker = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
            )

        state = tracker._sessions["good-session"]
        self.assertNotIn("expired-session", tracker._sessions)
        self.assertEqual(state.history_signatures, ("b" * 64,))
        self.assertEqual(state.anchors, {good_value})
        self.assertEqual(set(tracker._anchors), {good_value})
        self.assertEqual(tracker._anchors[good_value].expires_at, state.expires_at)

    def test_persistence_loader_enforces_caps_without_orphaning_session_sets(self):
        namespace = "scope:gemini:provider:gemini:model:pro"
        expires_at = time.time() + 3600
        sessions = {}
        anchors = {}
        for session_index in range(12):
            session_id = f"session-{session_index}"
            sessions[session_id] = {
                "namespace": namespace,
                "expires_at": expires_at,
                "last_seen": expires_at - session_index,
                "history_signatures": [f"{session_index:064x}"],
            }
            for anchor_index in range(24):
                value = f"{namespace}:loaded:{session_index}:{anchor_index}"
                anchors[value] = {
                    "session_id": session_id,
                    "namespace": namespace,
                    "strength": "medium",
                    "source": "message",
                    "group": f"message:{anchor_index}:user",
                    "expires_at": expires_at,
                    "last_seen": expires_at - anchor_index,
                }
        payload = {
            "schema_version": 2,
            "sessions": sessions,
            "anchors": anchors,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session_stickiness.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            tracker = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
                max_anchors_per_session=16,
                max_anchor_records=100,
            )

        self.assertLessEqual(len(tracker._anchors), 100)
        for state in tracker._sessions.values():
            self.assertLessEqual(len(state.anchors), 16)
            for value in state.anchors:
                self.assertIn(value, tracker._anchors)
                self.assertEqual(tracker._anchors[value].session_id, state.session_id)
        for value, record in tracker._anchors.items():
            self.assertIn(value, tracker._sessions[record.session_id].anchors)

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

    def test_streaming_chunk_collector_ignores_non_evidence_payloads(self):
        handler = StreamingHandler()
        assistant_parts = []
        tool_call_ids = []
        payloads = (
            "event: message\n\n",
            "data: [DONE]\n\n",
            "data: not-json\n\n",
            "data: []\n\n",
            "data: null\n\n",
            "data: 42\n\n",
            'data: {"choices":[null,"invalid",{"delta":"invalid"}]}\n\n',
            'data: {"choices":[{"delta":{"tool_calls":42}}]}\n\n',
        )

        for payload in payloads:
            handler._collect_session_response_anchors(
                payload,
                assistant_parts,
                tool_call_ids,
            )

        self.assertEqual(assistant_parts, [])
        self.assertEqual(tool_call_ids, [])

    def test_streaming_chunk_collector_accepts_event_frames_and_data_without_space(self):
        handler = StreamingHandler()
        assistant_parts = []
        tool_call_ids = []
        handler._collect_session_response_anchors(
            'event: message\ndata:{"choices":[{"delta":{"content":"hello ",'
            '"tool_calls":[{"id":"call_1"}]}}]}\n\n',
            assistant_parts,
            tool_call_ids,
        )
        handler._collect_session_response_anchors(
            'data: {"choices":[{"delta":{"content":"world",'
            '"tool_calls":[{"id":"call_1"}]}}]}\n\n',
            assistant_parts,
            tool_call_ids,
        )

        self.assertEqual(assistant_parts, ["hello ", "world"])
        self.assertEqual(tool_call_ids, ["call_1"])

    def test_stream_eof_without_completion_signal_does_not_record_response_identity(self):
        handler = StreamingHandler()
        responses = []

        async def partial_stream():
            yield {
                "choices": [
                    {
                        "delta": {"content": "partial assistant evidence"},
                        "finish_reason": None,
                    }
                ]
            }

        async def consume():
            return [
                item
                async for item in handler.wrap_stream(
                    partial_stream(),
                    "credential",
                    "model",
                    response_callback=responses.append,
                )
            ]

        output = asyncio.run(consume())

        self.assertEqual(responses, [])
        self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_raw_stream_early_finish_reason_without_usage_does_not_record_identity(self):
        handler = StreamingHandler()
        responses = []

        async def truncated_stream():
            yield (
                'data: {"choices":[{"delta":{"content":"truncated evidence"},'
                '"finish_reason":"stop"}]}\n\n'
            )

        async def consume():
            return [
                item
                async for item in handler.wrap_stream(
                    truncated_stream(),
                    "credential",
                    "model",
                    response_callback=responses.append,
                )
            ]

        output = asyncio.run(consume())

        self.assertEqual(responses, [])
        self.assertEqual(output[-1], "data: [DONE]\n\n")

    def test_raw_sse_completion_requires_done_or_usage_backing(self):
        handler = StreamingHandler()

        self.assertFalse(
            handler._sse_has_completion_signal(
                'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
            )
        )
        self.assertTrue(
            handler._sse_has_completion_signal(
                'data: {"choices":[{"finish_reason":"stop"}],"usage":{}}\n\n'
            )
        )
        self.assertTrue(
            handler._sse_has_completion_signal(
                'data: {"choices":[],"usage":{"completion_tokens":3}}\n\n'
            )
        )
        self.assertTrue(handler._sse_has_completion_signal("data: [DONE]\n\n"))

    def test_stream_finish_reason_records_completed_response_identity(self):
        handler = StreamingHandler()
        responses = []

        async def complete_stream():
            yield {
                "choices": [
                    {
                        "delta": {"content": "complete assistant evidence"},
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "total_tokens": 7,
                },
            }

        async def consume():
            return [
                item
                async for item in handler.wrap_stream(
                    complete_stream(),
                    "credential",
                    "model",
                    response_callback=responses.append,
                )
            ]

        output = asyncio.run(consume())

        self.assertEqual(len(responses), 1)
        self.assertEqual(
            responses[0]["choices"][0]["message"]["content"],
            "complete assistant evidence",
        )
        self.assertEqual(output[-1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()

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
from rotator_library.client.rotating_client import _resolve_session_persistence_settings
from rotator_library.client.request_builder import RequestContextBuilder


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

    @staticmethod
    def _tool_call_message(call_id, name, arguments="{}"):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }

    @staticmethod
    def _tool_result(call_id, content="ok"):
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    @staticmethod
    def _persistent_tracker(path):
        """Create a tracker whose every mutation is immediately restart-safe."""

        return SessionTracker(
            ttl_seconds=3600,
            persist_to_disk=True,
            persistence_path=path,
            persistence_flush_interval_seconds=0,
        )

    def test_proxy_session_persistence_environment_settings(self):
        """Proxy defaults expose persistence without overriding explicit callers."""

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                _resolve_session_persistence_settings(None, None),
                (False, 5.0),
            )

        with patch.dict(
            os.environ,
            {
                "SESSION_PERSISTENCE_ENABLED": "true",
                "SESSION_PERSISTENCE_FLUSH_INTERVAL_SECONDS": "0",
            },
        ):
            self.assertEqual(
                _resolve_session_persistence_settings(None, None),
                (True, 0.0),
            )
            self.assertEqual(
                _resolve_session_persistence_settings(False, 2.5),
                (False, 2.5),
            )

        with patch.dict(
            os.environ,
            {"SESSION_PERSISTENCE_FLUSH_INTERVAL_SECONDS": "invalid"},
        ):
            with self.assertLogs("rotator_library", level="WARNING") as captured:
                enabled, interval = _resolve_session_persistence_settings(False, None)
        self.assertFalse(enabled)
        self.assertEqual(interval, 5.0)
        self.assertIn("Invalid SESSION_PERSISTENCE_FLUSH_INTERVAL_SECONDS", captured.output[0])

    def test_warning_lineage_logs_new_and_continued_session_ids(self):
        tracker = SessionTracker(ttl_seconds=3600)
        request = {
            "messages": [
                {"role": "user", "content": self._long_text("lineage-user")},
                {"role": "assistant", "content": self._long_text("lineage-assistant")},
            ]
        }

        with self.assertLogs("rotator_library", level="WARNING") as captured:
            first = tracker.infer_session(request, provider="gemini", model="pro")
            second = tracker.infer_session(request, provider="gemini", model="pro")

        self.assertEqual(first.session_id, second.session_id)
        self.assertIn(f"action=new session_id={first.session_id}", captured.output[0])
        self.assertIn("matched_session_id=-", captured.output[0])
        self.assertIn("candidate_session_id=-", captured.output[0])
        self.assertIn(f"action=continue session_id={first.session_id}", captured.output[1])
        self.assertIn(f"matched_session_id={first.session_id}", captured.output[1])
        self.assertIn("origin=memory", captured.output[1])

    def test_warning_lineage_separates_rejected_candidate_from_accepted_match(self):
        tracker = SessionTracker(ttl_seconds=3600)
        opening = "Tell me a story now"
        established = {
            "messages": [
                {"role": "user", "content": opening},
                {"role": "assistant", "content": self._long_text("candidate-assistant")},
                {"role": "user", "content": "Add another paragraph to the story."},
            ]
        }
        first = tracker.infer_session(established, provider="deepseek", model="chat")

        with self.assertLogs("rotator_library", level="WARNING") as captured:
            independent = tracker.infer_session(
                {"messages": [{"role": "user", "content": opening}]},
                provider="deepseek",
                model="chat",
            )

        self.assertNotEqual(first.session_id, independent.session_id)
        self.assertIn(f"action=new session_id={independent.session_id}", captured.output[0])
        self.assertIn("matched_session_id=-", captured.output[0])
        self.assertIn(f"candidate_session_id={first.session_id}", captured.output[0])
        self.assertIn("confidence=weak score=5", captured.output[0])

    def test_warning_lineage_logs_compaction_child_and_exact_replay(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original = {
            "messages": [
                {"role": "user", "content": self._long_text("log-parent-user")},
                {"role": "assistant", "content": self._long_text("log-parent-assistant")},
                {"role": "user", "content": self._long_text("log-parent-follow-up")},
                {"role": "assistant", "content": self._long_text("log-parent-later")},
            ]
        }
        parent = tracker.infer_session(original, provider="gemini", model="pro")
        compacted = {
            "messages": [
                {
                    "role": "system",
                    "content": "Summary of previous conversation: "
                    + original["messages"][1]["content"],
                },
                {"role": "user", "content": "Continue from compacted context."},
            ]
        }

        with self.assertLogs("rotator_library", level="WARNING") as captured:
            child = tracker.infer_session(compacted, provider="gemini", model="pro")
            replay = tracker.infer_session(compacted, provider="gemini", model="pro")

        self.assertEqual(child.session_id, replay.session_id)
        self.assertIn(f"action=compaction_child session_id={child.session_id}", captured.output[0])
        self.assertIn(f"parent_session_id={parent.session_id}", captured.output[0])
        self.assertIn("marker=True", captured.output[0])
        self.assertIn(f"action=compaction_replay session_id={child.session_id}", captured.output[1])
        self.assertIn(f"matched_session_id={child.session_id}", captured.output[1])
        self.assertIn(f"parent_session_id={parent.session_id}", captured.output[1])

    def test_warning_lineage_logs_persisted_session_origin(self):
        request = {
            "messages": [
                {"role": "user", "content": self._long_text("persisted-log-user")},
                {
                    "role": "assistant",
                    "content": self._long_text("persisted-log-assistant"),
                },
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
            original = tracker.infer_session(request, provider="gemini", model="pro")
            tracker.flush()
            restored = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
                persistence_flush_interval_seconds=0,
            )

            with self.assertLogs("rotator_library", level="WARNING") as captured:
                continued = restored.infer_session(request, provider="gemini", model="pro")

        self.assertEqual(original.session_id, continued.session_id)
        self.assertIn(f"action=continue session_id={original.session_id}", captured.output[0])
        self.assertIn(f"matched_session_id={original.session_id}", captured.output[0])
        self.assertIn("origin=persisted", captured.output[0])

    def test_warning_lineage_logs_untracked_request(self):
        tracker = SessionTracker(ttl_seconds=3600)

        with self.assertLogs("rotator_library", level="WARNING") as captured:
            inferred = tracker.infer_session({}, provider="gemini", model="pro")

        self.assertIsNone(inferred.session_id)
        self.assertIn("action=untracked session_id=-", captured.output[0])
        self.assertIn("origin=none", captured.output[0])

    def test_agentic_tool_loop_persists_compacts_replays_and_continues_child(self):
        """Exercise a complete persisted agent loop and compacted-child lifecycle."""

        routes = [
            ("deepseek", "deepseek/v4"),
            ("deepseek", "deepseek/v4-flash"),
            ("anthropic", "anthropic/claude-sonnet"),
            ("anthropic", "anthropic/claude-opus"),
            ("openai", "openai/gpt-5"),
            ("deepseek", "deepseek/v4"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "agentic_session_stickiness.json"
            tracker = self._persistent_tracker(path)
            history = [
                {
                    "role": "user",
                    "content": self._long_text("agentic-root-task", repeats=16),
                }
            ]
            root_session_id = None
            assistant_responses = []

            for round_index in range(6):
                provider, model = routes[round_index]
                if round_index in {2, 4}:
                    tracker.flush()
                    tracker = self._persistent_tracker(path)
                    self.assertTrue(tracker._sessions[root_session_id].loaded_from_persistence)

                inferred = tracker.infer_session(
                    {"messages": list(history)},
                    provider=provider,
                    model=model,
                )
                if root_session_id is None:
                    root_session_id = inferred.session_id
                self.assertEqual(inferred.session_id, root_session_id)
                self.assertFalse(inferred.possible_compaction)
                self.assertEqual(inferred.tracking_namespace, "session-domain:public")
                if round_index:
                    self.assertIn(inferred.confidence, {"probable", "strong"})

                tool_id = f"call_agentic_{round_index}"
                assistant_text = self._long_text(
                    f"agentic-assistant-analysis-{round_index}",
                    repeats=14,
                )
                assistant_responses.append(assistant_text)
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_text,
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": f"agent_tool_{round_index}",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
                tracker.record_response(
                    inferred.session_id,
                    provider=provider,
                    model=model,
                    tracking_namespace=inferred.tracking_namespace,
                    response={"choices": [{"message": assistant_message}]},
                )
                history.extend(
                    [
                        assistant_message,
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": self._long_text(
                                f"agentic-tool-result-{round_index}",
                                repeats=12,
                            ),
                        },
                        {
                            "role": "user",
                            "content": self._long_text(
                                f"agentic-next-instruction-{round_index}",
                                repeats=10,
                            ),
                        },
                    ]
                )

            # Six responses were appended, but the sixth response/tool/user tail
            # has not yet appeared in a subsequent request. The high-water request
            # therefore contains 16 rather than all 19 locally accumulated messages.
            parent_high_water = tracker._sessions[root_session_id].history_signatures
            self.assertEqual(len(parent_high_water), 16)
            tracker.flush()
            tracker = self._persistent_tracker(path)

            # A real compacted payload drops every strong tool ID and replaces
            # almost all prior messages with one durable summary.
            compacted = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Compacted context: "
                        + assistant_responses[1]
                        + " "
                        + assistant_responses[4],
                    },
                    {
                        "role": "user",
                        "content": self._long_text("agentic-resume-after-compaction"),
                    },
                ]
            }
            child = tracker.infer_session(
                compacted,
                provider="anthropic",
                model="anthropic/claude-opus",
            )
            self.assertTrue(child.possible_compaction)
            self.assertEqual(child.lineage_parent_session_id, root_session_id)
            self.assertNotEqual(child.session_id, root_session_id)

            child_response = self._long_text("agentic-child-response", repeats=14)
            tracker.record_response(
                child.session_id,
                provider="anthropic",
                model="anthropic/claude-opus",
                tracking_namespace=child.tracking_namespace,
                response=self._response(child_response),
            )
            tracker.flush()
            tracker = self._persistent_tracker(path)

            replay = tracker.infer_session(
                compacted,
                provider="openai",
                model="openai/gpt-5",
            )
            self.assertEqual(replay.session_id, child.session_id)
            self.assertEqual(replay.lineage_parent_session_id, root_session_id)
            self.assertTrue(replay.possible_compaction)
            self.assertEqual(replay.confidence, "strong")

            child_follow_up = {
                "messages": [
                    *compacted["messages"],
                    {"role": "assistant", "content": child_response},
                    {
                        "role": "user",
                        "content": self._long_text("agentic-child-next-step"),
                    },
                ]
            }
            continued_child = tracker.infer_session(
                child_follow_up,
                provider="deepseek",
                model="deepseek/v4-flash",
            )
            self.assertEqual(continued_child.session_id, child.session_id)
            self.assertNotEqual(continued_child.session_id, root_session_id)
            self.assertFalse(continued_child.possible_compaction)
            self.assertEqual(continued_child.lineage_parent_session_id, root_session_id)

            parent_continuation = tracker.infer_session(
                {"messages": list(history)},
                provider="openai",
                model="openai/gpt-5",
            )
            self.assertEqual(parent_continuation.session_id, root_session_id)
            self.assertNotEqual(parent_continuation.session_id, child.session_id)

    def test_long_normal_conversation_tracks_every_turn_across_restarts(self):
        """Grow a long ordinary chat turn-by-turn without false compaction."""

        routes = [
            ("deepseek", "deepseek/chat"),
            ("openai", "openai/gpt-5"),
            ("anthropic", "anthropic/claude-sonnet"),
            ("deepseek", "deepseek/reasoner"),
            ("openai", "openai/gpt-5-mini"),
            ("anthropic", "anthropic/claude-opus"),
            ("deepseek", "deepseek/chat"),
            ("openai", "openai/gpt-5"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "normal_session_stickiness.json"
            tracker = self._persistent_tracker(path)
            history = []
            session_id = None
            first_response = None
            request_lengths = []

            for turn_index in range(8):
                provider, model = routes[turn_index]
                if turn_index in {3, 6}:
                    tracker.flush()
                    tracker = self._persistent_tracker(path)
                    self.assertTrue(tracker._sessions[session_id].loaded_from_persistence)

                history.append(
                    {
                        "role": "user",
                        "content": self._long_text(
                            f"normal-conversation-question-{turn_index}",
                            repeats=14,
                        ),
                    }
                )
                inferred = tracker.infer_session(
                    {"messages": list(history)},
                    provider=provider,
                    model=model,
                )
                if session_id is None:
                    session_id = inferred.session_id
                self.assertEqual(inferred.session_id, session_id)
                self.assertFalse(inferred.possible_compaction)
                self.assertEqual(inferred.tracking_namespace, "session-domain:public")
                request_lengths.append(len(history))
                self.assertEqual(
                    len(tracker._sessions[session_id].history_signatures),
                    len(history),
                )

                response_text = (
                    first_response
                    if turn_index == 4
                    else self._long_text(
                        f"normal-conversation-answer-{turn_index}",
                        repeats=16,
                    )
                )
                if first_response is None:
                    first_response = response_text
                tracker.record_response(
                    inferred.session_id,
                    provider=provider,
                    model=model,
                    tracking_namespace=inferred.tracking_namespace,
                    response=self._response(response_text),
                )
                history.append({"role": "assistant", "content": response_text})

            self.assertEqual(request_lengths, [1, 3, 5, 7, 9, 11, 13, 15])
            self.assertEqual(
                len(tracker._sessions[session_id].history_signatures),
                request_lengths[-1],
            )
            response_groups = {
                record.group
                for record in tracker._anchors.values()
                if record.session_id == session_id
                and record.source == "response"
                and record.group
                and record.group.startswith("response_event:")
            }
            self.assertEqual(len(response_groups), 7)

    def test_roleplay_redo_edits_rollback_and_middle_rewrite_stay_on_session(self):
        """Model realistic roleplay regeneration and branch editing behavior."""

        routes = [
            ("mistral", "mistral/large"),
            ("anthropic", "anthropic/claude-sonnet"),
            ("mistral", "mistral/large"),
            ("openai", "openai/gpt-5"),
            ("anthropic", "anthropic/claude-opus"),
            ("mistral", "mistral/large"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "roleplay_session_stickiness.json"
            tracker = self._persistent_tracker(path)
            history = [
                {
                    "role": "system",
                    "content": self._long_text("roleplay-world-rules", repeats=14),
                },
                {
                    "role": "user",
                    "content": self._long_text("roleplay-opening-scene", repeats=12),
                },
            ]
            session_id = None
            request_snapshots = []

            for turn_index in range(6):
                provider, model = routes[turn_index]
                request_snapshot = [dict(message) for message in history]
                request_snapshots.append(request_snapshot)
                inferred = tracker.infer_session(
                    {"messages": request_snapshot},
                    provider=provider,
                    model=model,
                )
                if session_id is None:
                    session_id = inferred.session_id
                self.assertEqual(inferred.session_id, session_id)
                self.assertFalse(inferred.possible_compaction)
                self.assertIsNone(inferred.lineage_parent_session_id)

                original_response = self._long_text(
                    f"roleplay-assistant-scene-{turn_index}",
                    repeats=15,
                )
                tracker.record_response(
                    session_id,
                    provider=provider,
                    model=model,
                    tracking_namespace=inferred.tracking_namespace,
                    response=self._response(original_response),
                )

                if turn_index == 2:
                    # Retrying generation sends the byte-identical request without
                    # appending the response that is being regenerated.
                    same_provider_redo = tracker.infer_session(
                        {"messages": [dict(message) for message in request_snapshot]},
                        provider=provider,
                        model=model,
                    )
                    self.assertEqual(same_provider_redo.session_id, session_id)
                    self.assertFalse(same_provider_redo.possible_compaction)

                    exact_redo = tracker.infer_session(
                        {"messages": [dict(message) for message in request_snapshot]},
                        provider="openai",
                        model="openai/gpt-5-mini",
                    )
                    self.assertEqual(exact_redo.session_id, session_id)
                    self.assertFalse(exact_redo.possible_compaction)
                    chosen_response = self._long_text(
                        "roleplay-assistant-scene-2-edited-redo",
                        repeats=15,
                    )
                    tracker.record_response(
                        session_id,
                        provider="openai",
                        model="openai/gpt-5-mini",
                        tracking_namespace=exact_redo.tracking_namespace,
                        response=self._response(chosen_response),
                    )
                else:
                    chosen_response = original_response

                history.append({"role": "assistant", "content": chosen_response})
                if turn_index < 5:
                    history.append(
                        {
                            "role": "user",
                            "content": self._long_text(
                                f"roleplay-user-action-{turn_index + 1}",
                                repeats=11,
                            ),
                        }
                    )

                if turn_index == 3:
                    tracker.flush()
                    tracker = self._persistent_tracker(path)
                    self.assertTrue(tracker._sessions[session_id].loaded_from_persistence)

            high_water_before_edits = tracker._sessions[session_id].history_signatures
            self.assertEqual(len(high_water_before_edits), len(request_snapshots[-1]))

            # Rewrite an assistant response in the middle while preserving every
            # later turn. Seven-plus surviving messages must dominate the edit.
            middle_edited_history = [dict(message) for message in history]
            middle_assistant_index = 6
            self.assertEqual(middle_edited_history[middle_assistant_index]["role"], "assistant")
            middle_edited_history[middle_assistant_index] = {
                "role": "assistant",
                "content": self._long_text("roleplay-middle-response-rewritten", repeats=15),
            }
            middle_edit = tracker.infer_session(
                {"messages": middle_edited_history},
                provider="anthropic",
                model="anthropic/claude-opus",
            )
            self.assertEqual(middle_edit.session_id, session_id)
            self.assertFalse(middle_edit.possible_compaction)
            self.assertIsNone(middle_edit.lineage_parent_session_id)

            # Roll back to an older request snapshot, then resume a different
            # branch. The shorter request must not reduce the high-water profile.
            # This rollback retains less than half the high-water history. It is
            # still ordinary branching because it has neither a summary marker nor
            # response-derived probe evidence.
            rollback_request = [dict(message) for message in request_snapshots[2]]
            high_water_before_rollback = tracker._sessions[session_id].history_signatures
            rolled_back = tracker.infer_session(
                {"messages": rollback_request},
                provider="deepseek",
                model="deepseek/v4-flash",
            )
            self.assertEqual(rolled_back.session_id, session_id)
            self.assertFalse(rolled_back.possible_compaction)
            self.assertIsNone(rolled_back.lineage_parent_session_id)
            self.assertEqual(
                tracker._sessions[session_id].history_signatures,
                high_water_before_rollback,
            )

            branch_response = self._long_text("roleplay-rollback-new-response", repeats=15)
            tracker.record_response(
                session_id,
                provider="deepseek",
                model="deepseek/v4-flash",
                tracking_namespace=rolled_back.tracking_namespace,
                response=self._response(branch_response),
            )
            resumed_branch = [
                *rollback_request,
                {"role": "assistant", "content": branch_response},
                {
                    "role": "user",
                    "content": self._long_text("roleplay-rollback-new-user-action"),
                },
            ]
            resumed = tracker.infer_session(
                {"messages": resumed_branch},
                provider="openai",
                model="openai/gpt-5",
            )
            self.assertEqual(resumed.session_id, session_id)
            self.assertFalse(resumed.possible_compaction)
            self.assertIsNone(resumed.lineage_parent_session_id)
            self.assertEqual(
                tracker._sessions[session_id].history_signatures,
                high_water_before_rollback,
            )

    def test_long_changed_tail_continues_persisted_compaction_context(self):
        """A long changing user tail must not become part of the summary key."""

        provider = "gemini"
        model = "pro"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "long_tail_session_stickiness.json"
            tracker = self._persistent_tracker(path)
            parent_request = {
                "messages": [
                    {"role": "user", "content": self._long_text("long-tail-parent-user-1")},
                    {
                        "role": "assistant",
                        "content": self._long_text("long-tail-parent-assistant-1"),
                    },
                    {"role": "user", "content": self._long_text("long-tail-parent-user-2")},
                    {
                        "role": "assistant",
                        "content": self._long_text("long-tail-parent-assistant-2"),
                    },
                ]
            }
            parent = tracker.infer_session(parent_request, provider=provider, model=model)
            summary = {
                "role": "system",
                "content": "Summary of previous conversation: "
                + parent_request["messages"][1]["content"],
            }
            first_compacted = {
                "messages": [
                    summary,
                    {
                        "role": "user",
                        "content": self._long_text("long-tail-first-instruction", repeats=20),
                    },
                ]
            }
            child = tracker.infer_session(first_compacted, provider=provider, model=model)
            self.assertTrue(child.possible_compaction)
            self.assertEqual(child.lineage_parent_session_id, parent.session_id)
            tracker.flush()
            tracker = self._persistent_tracker(path)

            changed_tail = {
                "messages": [
                    summary,
                    {
                        "role": "user",
                        "content": self._long_text("long-tail-second-instruction", repeats=20),
                    },
                ]
            }
            continued = tracker.infer_session(changed_tail, provider=provider, model=model)

            self.assertEqual(continued.session_id, child.session_id)
            self.assertEqual(continued.lineage_parent_session_id, parent.session_id)
            self.assertFalse(continued.possible_compaction)
            self.assertEqual(continued.confidence, "strong")

    def test_expired_compaction_child_is_recreated_from_live_parent(self):
        """Expired child bindings cannot revive, while a live parent can fork anew."""

        with patch("rotator_library.session_tracking.time.time", return_value=1000.0):
            tracker = SessionTracker(ttl_seconds=10)
            parent_request = {
                "messages": [
                    {"role": "user", "content": self._long_text("ttl-parent-user")},
                    {
                        "role": "assistant",
                        "content": self._long_text("ttl-parent-assistant"),
                    },
                    {"role": "user", "content": self._long_text("ttl-parent-follow-up")},
                ]
            }
            parent = tracker.infer_session(parent_request, provider="gemini", model="pro")
            tracker.record_response(
                parent.session_id,
                provider="gemini",
                model="pro",
                tracking_namespace=parent.tracking_namespace,
                response=self._response(parent_request["messages"][1]["content"]),
            )
            compacted = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Compacted context: "
                        + parent_request["messages"][1]["content"],
                    },
                    {"role": "user", "content": "Continue after compaction."},
                ]
            }
            child = tracker.infer_session(compacted, provider="gemini", model="pro")

        with patch("rotator_library.session_tracking.time.time", return_value=1005.0):
            refreshed_parent = tracker.infer_session(
                parent_request,
                provider="openai",
                model="gpt",
            )
            self.assertEqual(refreshed_parent.session_id, parent.session_id)

        with patch("rotator_library.session_tracking.time.time", return_value=1011.0):
            after_expiry = tracker.infer_session(compacted, provider="gemini", model="pro")

        self.assertNotEqual(after_expiry.session_id, child.session_id)
        self.assertTrue(after_expiry.possible_compaction)
        self.assertEqual(after_expiry.lineage_parent_session_id, parent.session_id)

    def test_compaction_context_survives_anchor_caps_and_restart(self):
        """Anchor caps retain both durable child bindings deterministically."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "capped_context_stickiness.json"
            tracker = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
                persistence_flush_interval_seconds=0,
                max_anchors_per_session=16,
            )
            parent_request = {
                "messages": [
                    {"role": "user", "content": self._long_text("cap-parent-user")},
                    {
                        "role": "assistant",
                        "content": self._long_text("cap-parent-assistant"),
                    },
                    {"role": "user", "content": self._long_text("cap-parent-follow-up")},
                ]
            }
            tracker.infer_session(parent_request, provider="gemini", model="pro")
            compacted = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Summary of previous conversation: "
                        + parent_request["messages"][1]["content"],
                    },
                    {"role": "user", "content": "Continue capped context."},
                ]
            }
            child = tracker.infer_session(compacted, provider="gemini", model="pro")
            child_response = self._long_text("cap-child-response")
            tracker.record_response(
                child.session_id,
                provider="gemini",
                model="pro",
                tracking_namespace=child.tracking_namespace,
                response=self._response(child_response),
            )
            pressure_messages = [
                *compacted["messages"],
                {"role": "assistant", "content": child_response},
            ]
            for index in range(8):
                pressure_messages.append(
                    {
                        "role": "user" if index % 2 == 0 else "assistant",
                        "content": self._long_text(f"cap-pressure-{index}"),
                    }
                )
            pressured = tracker.infer_session(
                {"messages": pressure_messages},
                provider="gemini",
                model="pro",
            )
            self.assertEqual(pressured.session_id, child.session_id)
            child_sources = {
                tracker._anchors[value].source
                for value in tracker._sessions[child.session_id].anchors
            }
            self.assertIn("compaction_context", child_sources)
            self.assertIn("compaction_replay", child_sources)
            self.assertLessEqual(len(tracker._sessions[child.session_id].anchors), 16)
            tracker.flush()

            restored = SessionTracker(
                ttl_seconds=3600,
                persist_to_disk=True,
                persistence_path=path,
                persistence_flush_interval_seconds=0,
                max_anchors_per_session=16,
            )
            replay = restored.infer_session(compacted, provider="gemini", model="pro")
            self.assertEqual(replay.session_id, child.session_id)
            self.assertTrue(replay.possible_compaction)
            changed_tail = {
                "messages": [
                    compacted["messages"][0],
                    {"role": "user", "content": "Continue capped context differently."},
                ]
            }
            continued = restored.infer_session(
                changed_tail,
                provider="gemini",
                model="pro",
            )

            self.assertEqual(continued.session_id, child.session_id)
            self.assertFalse(continued.possible_compaction)

    def test_compaction_context_cannot_cross_usage_scope(self):
        """A validated child binding remains isolated to its usage namespace."""

        tracker = SessionTracker(ttl_seconds=3600)
        parent_request = {
            "messages": [
                {"role": "user", "content": self._long_text("context-scope-user")},
                {
                    "role": "assistant",
                    "content": self._long_text("context-scope-assistant"),
                },
                {"role": "user", "content": self._long_text("context-scope-follow-up")},
            ]
        }
        parent = tracker.infer_session(
            parent_request,
            provider="gemini",
            model="pro",
            scope_key="A",
        )
        compacted = {
            "messages": [
                {
                    "role": "system",
                    "content": "Compacted context: "
                    + parent_request["messages"][1]["content"],
                },
                {"role": "user", "content": "Continue scoped context."},
            ]
        }
        child_a = tracker.infer_session(
            compacted,
            provider="gemini",
            model="pro",
            scope_key="A",
        )
        same_payload_b = tracker.infer_session(
            compacted,
            provider="gemini",
            model="pro",
            scope_key="B",
        )

        self.assertEqual(child_a.lineage_parent_session_id, parent.session_id)
        self.assertNotEqual(same_payload_b.session_id, child_a.session_id)
        self.assertFalse(same_payload_b.possible_compaction)
        self.assertIsNone(same_payload_b.lineage_parent_session_id)

    def test_trusted_and_provider_identity_override_compaction_bindings(self):
        """Authoritative identity must win over an older replay/context anchor."""

        tracker = SessionTracker(
            ttl_seconds=3600,
            trusted_explicit_fields=["conversation_id"],
        )
        parent_request = {
            "messages": [
                {"role": "user", "content": self._long_text("identity-parent-user")},
                {
                    "role": "assistant",
                    "content": self._long_text("identity-parent-assistant"),
                },
                {"role": "user", "content": self._long_text("identity-parent-follow-up")},
                {
                    "role": "assistant",
                    "content": self._long_text("identity-parent-later"),
                },
            ]
        }
        tracker.infer_session(parent_request, provider="gemini", model="pro")
        compacted = {
            "messages": [
                {
                    "role": "system",
                    "content": "Summary of previous conversation: "
                    + parent_request["messages"][1]["content"],
                },
                {"role": "user", "content": "Continue from the compacted state."},
            ]
        }
        compacted_child = tracker.infer_session(compacted, provider="gemini", model="pro")

        explicit_session = tracker.infer_session(
            {
                "conversation_id": "trusted-conversation-B",
                "messages": [
                    {"role": "user", "content": self._long_text("identity-explicit-session")}
                ],
            },
            provider="gemini",
            model="pro",
        )
        explicit_replay = tracker.infer_session(
            {**compacted, "conversation_id": "trusted-conversation-B"},
            provider="gemini",
            model="pro",
        )
        self.assertEqual(explicit_replay.session_id, explicit_session.session_id)
        self.assertNotEqual(explicit_replay.session_id, compacted_child.session_id)
        self.assertFalse(explicit_replay.possible_compaction)
        self.assertIsNone(explicit_replay.lineage_parent_session_id)

        unseen_explicit = tracker.infer_session(
            {**compacted, "conversation_id": "trusted-conversation-C"},
            provider="gemini",
            model="pro",
        )
        self.assertNotEqual(unseen_explicit.session_id, compacted_child.session_id)
        self.assertNotEqual(unseen_explicit.session_id, explicit_session.session_id)

        provider_hints = SessionTrackingHints(strong_anchors=["provider-native-session-B"])
        provider_session = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": self._long_text("identity-provider-session")}
                ]
            },
            provider="gemini",
            model="pro",
            hints=provider_hints,
        )
        provider_replay = tracker.infer_session(
            compacted,
            provider="gemini",
            model="pro",
            hints=provider_hints,
        )
        self.assertEqual(provider_replay.session_id, provider_session.session_id)
        self.assertNotEqual(provider_replay.session_id, compacted_child.session_id)
        self.assertFalse(provider_replay.possible_compaction)
        self.assertIsNone(provider_replay.lineage_parent_session_id)

        opaque_tool_id = "call_92d1e7b45ac843f6"
        tool_session = tracker.infer_session(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": opaque_tool_id, "type": "function"}],
                    }
                ]
            },
            provider="gemini",
            model="pro",
        )
        tool_bound_context = tracker.infer_session(
            {
                "messages": [
                    *compacted["messages"],
                    {
                        "role": "tool",
                        "tool_call_id": opaque_tool_id,
                        "content": "Opaque tool continuation evidence.",
                    },
                ]
            },
            provider="gemini",
            model="pro",
        )
        self.assertNotEqual(tool_bound_context.session_id, tool_session.session_id)
        self.assertEqual(tool_bound_context.session_id, compacted_child.session_id)

    def test_record_response_normalizes_cross_namespace_fallback_across_restart(self):
        """A fallback response stays with its original logical session scope."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "namespace_session_stickiness.json"
            tracker = self._persistent_tracker(path)
            session_a = tracker.infer_session(
                {
                    "messages": [
                        {"role": "user", "content": self._long_text("namespace-a-request")}
                    ]
                },
                provider="gemini",
                model="pro",
                scope_key="A",
            )
            namespace_b = tracker._namespace("gemini", "pro", scope_key="B")
            response_text = self._long_text("namespace-mismatched-response")

            with self.assertLogs("rotator_library", level="WARNING") as captured:
                tracker.record_response(
                    session_a.session_id,
                    provider="gemini",
                    model="pro",
                    tracking_namespace=namespace_b,
                    response=self._response(response_text),
                )

            self.assertIn("normalized response namespace mismatch", captured.output[0])
            self.assertEqual(
                tracker._sessions[session_a.session_id].namespace,
                session_a.tracking_namespace,
            )
            tracker.flush()
            tracker = self._persistent_tracker(path)
            self.assertEqual(
                tracker._sessions[session_a.session_id].namespace,
                session_a.tracking_namespace,
            )

            scope_a_continuation = tracker.infer_session(
                {
                    "messages": [
                        {"role": "assistant", "content": response_text},
                        {"role": "user", "content": "Continue in logical scope A."},
                    ]
                },
                provider="gemini",
                model="pro",
                scope_key="A",
            )
            self.assertEqual(scope_a_continuation.session_id, session_a.session_id)

            scope_b_request = tracker.infer_session(
                {
                    "messages": [
                        {"role": "assistant", "content": response_text},
                        {"role": "user", "content": "Continue in scope B."},
                    ]
                },
                provider="gemini",
                model="pro",
                scope_key="B",
            )
            self.assertNotEqual(scope_b_request.session_id, session_a.session_id)

    def test_raw_tool_ids_do_not_bind_independent_sessions(self):
        """Neither short nor entropy-looking request tool IDs are authoritative."""

        tracker = SessionTracker(ttl_seconds=3600)
        generic_owner = tracker.infer_session(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": "call_1", "type": "function"}],
                    }
                ]
            },
            provider="gemini",
            model="pro",
        )
        unrelated_generic = tracker.infer_session(
            {
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "An unrelated tool result from another conversation.",
                    }
                ]
            },
            provider="gemini",
            model="pro",
        )
        self.assertNotEqual(unrelated_generic.session_id, generic_owner.session_id)
        self.assertEqual(unrelated_generic.confidence, "weak")

        opaque_id = "call_7f4e2a91c8b64d37"
        opaque_owner = tracker.infer_session(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": opaque_id, "type": "function"}],
                    }
                ]
            },
            provider="gemini",
            model="pro",
        )
        opaque_result = tracker.infer_session(
            {
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": opaque_id,
                        "content": "The matching opaque tool result.",
                    }
                ]
            },
            provider="gemini",
            model="pro",
        )
        self.assertNotEqual(opaque_result.session_id, opaque_owner.session_id)
        self.assertEqual(opaque_result.confidence, "weak")

        compound_tracker = SessionTracker(ttl_seconds=3600)
        compound_owner = compound_tracker.infer_session(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "call_1", "type": "function"},
                            {"id": "call_2", "type": "function"},
                        ],
                    }
                ]
            },
            provider="gemini",
            model="pro",
        )
        compound_unrelated = compound_tracker.infer_session(
            {
                "messages": [
                    {"role": "tool", "tool_call_id": "call_1", "content": "first"},
                    {"role": "tool", "tool_call_id": "call_2", "content": "second"},
                ]
            },
            provider="gemini",
            model="pro",
        )
        self.assertNotEqual(compound_unrelated.session_id, compound_owner.session_id)
        self.assertEqual(compound_unrelated.confidence, "weak")

    def test_closed_tool_event_plus_message_group_continues_across_provider(self):
        tracker = SessionTracker(ttl_seconds=3600)
        user_message = {
            "role": "user",
            "content": self._long_text("closed-tool-event-user"),
        }
        call = self._tool_call_message(
            "call_read_1",
            "read_file",
            '{"path":"src/session.py","line":10}',
        )
        root = tracker.infer_session(
            {"messages": [user_message]},
            provider="deepseek",
            model="chat",
        )
        tracker.record_response(
            root.session_id,
            tracking_namespace=root.tracking_namespace,
            response={"choices": [{"message": call}]},
        )
        pending_events = [
            record
            for record in tracker._anchors.values()
            if record.session_id == root.session_id
            and record.source == "tool_event"
        ]
        self.assertEqual(len(pending_events), 1)
        self.assertEqual(pending_events[0].strength, "weak")

        continued = tracker.infer_session(
            {"messages": [user_message, call, self._tool_result("call_read_1")]},
            provider="anthropic",
            model="claude",
        )

        self.assertEqual(continued.session_id, root.session_id)
        self.assertEqual(continued.confidence, "probable")

    def test_two_distinct_closed_tool_events_continue_sparse_agentic_turn(self):
        tracker = SessionTracker(ttl_seconds=3600)
        first_call = self._tool_call_message("call_1", "read_file", '{"path":"a.py"}')
        second_call = self._tool_call_message("call_2", "grep", '{"pattern":"needle"}')
        response_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [*first_call["tool_calls"], *second_call["tool_calls"]],
        }
        root = tracker.infer_session(
            {"messages": [{"role": "user", "content": "continue"}]},
            provider="openai",
            model="gpt",
        )
        tracker.record_response(
            root.session_id,
            tracking_namespace=root.tracking_namespace,
            response={"choices": [{"message": response_message}]},
        )

        continued = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": "continue"},
                    response_message,
                    self._tool_result("call_1", "a"),
                    self._tool_result("call_2", "b"),
                ]
            },
            provider="deepseek",
            model="chat",
        )

        self.assertEqual(continued.session_id, root.session_id)
        self.assertEqual(continued.confidence, "probable")

    def test_one_closed_tool_event_without_other_medium_group_remains_weak(self):
        tracker = SessionTracker(ttl_seconds=3600)
        call = self._tool_call_message("call_1", "read_file", '{"path":"a.py"}')
        root = tracker.infer_session(
            {"messages": [{"role": "user", "content": "continue"}]},
            provider="openai",
            model="gpt",
        )
        tracker.record_response(
            root.session_id,
            tracking_namespace=root.tracking_namespace,
            response={"choices": [{"message": call}]},
        )

        ambiguous = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": "continue"},
                    call,
                    self._tool_result("call_1"),
                ]
            },
            provider="openai",
            model="gpt",
        )

        self.assertNotEqual(ambiguous.session_id, root.session_id)
        self.assertEqual(ambiguous.confidence, "weak")

    def test_tool_event_arguments_are_canonical_and_result_edits_do_not_fragment(self):
        tracker = SessionTracker(ttl_seconds=3600)
        user_message = {
            "role": "user",
            "content": self._long_text("canonical-tool-event-user"),
        }
        emitted = self._tool_call_message(
            "call_1",
            "read_file",
            '{"path":"a.py","line":10}',
        )
        reordered = self._tool_call_message(
            "call_1",
            "read_file",
            '{ "line": 10, "path": "a.py" }',
        )
        root = tracker.infer_session(
            {"messages": [user_message]}, provider="openai", model="gpt"
        )
        tracker.record_response(
            root.session_id,
            tracking_namespace=root.tracking_namespace,
            response={"choices": [{"message": emitted}]},
        )
        first = tracker.infer_session(
            {
                "messages": [
                    user_message,
                    reordered,
                    self._tool_result("call_1", "original result"),
                ]
            },
            provider="anthropic",
            model="claude",
        )
        edited = tracker.infer_session(
            {
                "messages": [
                    user_message,
                    reordered,
                    self._tool_result("call_1", "edited result"),
                ]
            },
            provider="deepseek",
            model="chat",
        )

        self.assertEqual(first.session_id, root.session_id)
        self.assertEqual(edited.session_id, root.session_id)

    def test_same_raw_id_with_different_call_structure_does_not_bind(self):
        tracker = SessionTracker(ttl_seconds=3600)
        user_message = {
            "role": "user",
            "content": self._long_text("tool-event-collision-user"),
        }
        owner_call = self._tool_call_message(
            "call_1", "read_file", '{"path":"owner.py"}'
        )
        owner = tracker.infer_session(
            {"messages": [user_message]}, provider="openai", model="gpt"
        )
        tracker.record_response(
            owner.session_id,
            tracking_namespace=owner.tracking_namespace,
            response={"choices": [{"message": owner_call}]},
        )

        unrelated = tracker.infer_session(
            {
                "messages": [
                    user_message,
                    self._tool_call_message(
                        "call_1", "read_file", '{"path":"other.py"}'
                    ),
                    self._tool_result("call_1"),
                ]
            },
            provider="openai",
            model="gpt",
        )

        self.assertNotEqual(unrelated.session_id, owner.session_id)
        self.assertEqual(unrelated.confidence, "weak")

    def test_duplicate_tool_results_count_as_one_closed_event(self):
        tracker = SessionTracker(ttl_seconds=3600)
        call = self._tool_call_message("call_1", "read_file", '{"path":"a.py"}')
        root = tracker.infer_session(
            {"messages": [{"role": "user", "content": "continue"}]},
            provider="openai",
            model="gpt",
        )
        tracker.record_response(
            root.session_id,
            tracking_namespace=root.tracking_namespace,
            response={"choices": [{"message": call}]},
        )

        duplicated = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": "continue"},
                    call,
                    self._tool_result("call_1", "first"),
                    self._tool_result("call_1", "duplicate"),
                ]
            },
            provider="openai",
            model="gpt",
        )

        self.assertNotEqual(duplicated.session_id, root.session_id)
        self.assertEqual(duplicated.confidence, "weak")

    def test_multiple_unpaired_structural_tool_calls_remain_weak(self):
        tracker = SessionTracker(ttl_seconds=3600)
        calls = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                self._tool_call_message(
                    "call_1", "read_file", '{"path":"a.py"}'
                )["tool_calls"][0],
                self._tool_call_message(
                    "call_2", "grep", '{"pattern":"needle"}'
                )["tool_calls"][0],
            ],
        }

        first = tracker.infer_session(
            {"messages": [calls]}, provider="openai", model="gpt"
        )
        second = tracker.infer_session(
            {"messages": [calls]}, provider="openai", model="gpt"
        )

        self.assertNotEqual(second.session_id, first.session_id)
        self.assertEqual(second.confidence, "weak")

    def test_previously_closed_event_does_not_upgrade_later_unpaired_call(self):
        tracker = SessionTracker(ttl_seconds=3600)
        user_message = {
            "role": "user",
            "content": self._long_text("request-local-tool-closure"),
        }
        call = self._tool_call_message("call_1", "read_file", '{"path":"a.py"}')
        root = tracker.infer_session(
            {"messages": [user_message]}, provider="openai", model="gpt"
        )
        tracker.record_response(
            root.session_id,
            tracking_namespace=root.tracking_namespace,
            response={"choices": [{"message": call}]},
        )
        closed = tracker.infer_session(
            {
                "messages": [
                    user_message,
                    call,
                    self._tool_result("call_1"),
                ]
            },
            provider="openai",
            model="gpt",
        )
        self.assertEqual(closed.session_id, root.session_id)

        unpaired = tracker.infer_session(
            {"messages": [user_message, call]},
            provider="anthropic",
            model="claude",
        )

        self.assertNotEqual(unpaired.session_id, root.session_id)
        self.assertEqual(unpaired.confidence, "weak")

    def test_one_result_closes_only_one_of_multiple_same_id_calls(self):
        tracker = SessionTracker(ttl_seconds=3600)
        calls = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                self._tool_call_message(
                    "call_1", "read_file", '{"path":"a.py"}'
                )["tool_calls"][0],
                self._tool_call_message(
                    "call_1", "read_file", '{"path":"b.py"}'
                )["tool_calls"][0],
            ],
        }
        root = tracker.infer_session(
            {"messages": [{"role": "user", "content": "continue"}]},
            provider="openai",
            model="gpt",
        )
        tracker.record_response(
            root.session_id,
            tracking_namespace=root.tracking_namespace,
            response={"choices": [{"message": calls}]},
        )

        ambiguous = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": "continue"},
                    calls,
                    self._tool_result("call_1"),
                ]
            },
            provider="openai",
            model="gpt",
        )

        self.assertNotEqual(ambiguous.session_id, root.session_id)
        self.assertEqual(ambiguous.confidence, "weak")

    def test_non_json_and_mapping_tool_arguments_are_canonical(self):
        cases = (
            ("path   =   a.py", "path = a.py"),
            ({"path": "a.py", "line": 10}, {"line": 10, "path": "a.py"}),
        )
        for emitted_arguments, returned_arguments in cases:
            with self.subTest(arguments=emitted_arguments):
                tracker = SessionTracker(ttl_seconds=3600)
                user_message = {
                    "role": "user",
                    "content": self._long_text("non-json-tool-arguments"),
                }
                emitted = self._tool_call_message(
                    "call_1", "read_file", emitted_arguments
                )
                returned = self._tool_call_message(
                    "call_1", "read_file", returned_arguments
                )
                root = tracker.infer_session(
                    {"messages": [user_message]},
                    provider="openai",
                    model="gpt",
                )
                tracker.record_response(
                    root.session_id,
                    tracking_namespace=root.tracking_namespace,
                    response={"choices": [{"message": emitted}]},
                )

                continued = tracker.infer_session(
                    {
                        "messages": [
                            user_message,
                            returned,
                            self._tool_result("call_1"),
                        ]
                    },
                    provider="openai",
                    model="gpt",
                )

                self.assertEqual(continued.session_id, root.session_id)

    def test_function_name_is_part_of_tool_event_identity(self):
        tracker = SessionTracker(ttl_seconds=3600)
        user_message = {
            "role": "user",
            "content": self._long_text("tool-function-name-collision"),
        }
        emitted = self._tool_call_message("call_1", "read_file", '{"path":"a.py"}')
        root = tracker.infer_session(
            {"messages": [user_message]}, provider="openai", model="gpt"
        )
        tracker.record_response(
            root.session_id,
            tracking_namespace=root.tracking_namespace,
            response={"choices": [{"message": emitted}]},
        )

        unrelated = tracker.infer_session(
            {
                "messages": [
                    user_message,
                    self._tool_call_message(
                        "call_1", "write_file", '{"path":"a.py"}'
                    ),
                    self._tool_result("call_1"),
                ]
            },
            provider="openai",
            model="gpt",
        )

        self.assertNotEqual(unrelated.session_id, root.session_id)
        self.assertEqual(unrelated.confidence, "weak")

    def test_legacy_function_result_closes_matching_tool_event(self):
        tracker = SessionTracker(ttl_seconds=3600)
        user_message = {
            "role": "user",
            "content": self._long_text("legacy-function-tool-event"),
        }
        call = self._tool_call_message("call_1", "lookup", '{"query":"state"}')
        root = tracker.infer_session(
            {"messages": [user_message]}, provider="openai", model="gpt"
        )
        tracker.record_response(
            root.session_id,
            tracking_namespace=root.tracking_namespace,
            response={"choices": [{"message": call}]},
        )

        continued = tracker.infer_session(
            {
                "messages": [
                    user_message,
                    call,
                    {
                        "role": "function",
                        "tool_call_id": "call_1",
                        "content": "ok",
                    },
                ]
            },
            provider="openai",
            model="gpt",
        )

        self.assertEqual(continued.session_id, root.session_id)

    def test_nameless_tool_call_never_creates_structural_event(self):
        tracker = SessionTracker(ttl_seconds=3600)
        anchors = tracker._build_anchors(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"arguments": '{"path":"a.py"}'},
                            }
                        ],
                    },
                    self._tool_result("call_1"),
                ]
            },
            "session-domain:public",
            None,
        )

        self.assertFalse(any(anchor.source == "tool_event" for anchor in anchors))

    def test_closed_tool_events_support_continuity_but_not_compaction_probes(self):
        tracker = SessionTracker(ttl_seconds=3600)
        calls = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                self._tool_call_message(
                    "call_1", "read_file", '{"path":"a.py"}'
                )["tool_calls"][0],
                self._tool_call_message(
                    "call_2", "grep", '{"pattern":"needle"}'
                )["tool_calls"][0],
            ],
        }
        history = [
            calls,
            self._tool_result("call_1", "a"),
            self._tool_result("call_2", "b"),
        ]
        root = tracker.infer_session(
            {"messages": history}, provider="openai", model="gpt"
        )

        replay = tracker.infer_session(
            {"messages": history}, provider="anthropic", model="claude"
        )

        self.assertEqual(replay.session_id, root.session_id)
        self.assertEqual(replay.confidence, "probable")
        self.assertFalse(replay.possible_compaction)

    def test_closed_tool_event_survives_persistence_and_never_crosses_domain(self):
        user_message = {
            "role": "user",
            "content": self._long_text("persisted-tool-event-user"),
        }
        call = self._tool_call_message(
            "call_persisted", "lookup", '{"query":"status"}'
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tool_event_session.json"
            tracker = self._persistent_tracker(path)
            root = tracker.infer_session(
                {"messages": [user_message]},
                provider="openai",
                model="gpt",
                scope_key="domain-a",
            )
            tracker.record_response(
                root.session_id,
                tracking_namespace=root.tracking_namespace,
                response={"choices": [{"message": call}]},
            )
            tracker.flush()
            tracker = self._persistent_tracker(path)
            request = {
                "messages": [
                    user_message,
                    call,
                    self._tool_result("call_persisted", "changed safely"),
                ]
            }

            continued = tracker.infer_session(
                request,
                provider="anthropic",
                model="claude",
                scope_key="domain-a",
            )
            isolated = tracker.infer_session(
                request,
                provider="anthropic",
                model="claude",
                scope_key="domain-b",
            )

        self.assertEqual(continued.session_id, root.session_id)
        self.assertNotEqual(isolated.session_id, root.session_id)
        self.assertFalse(continued.possible_compaction)

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
            provider="anthropic",
            model="claude-test",
            hints={"global_strong_anchors": ["responses_previous_response_id:resp_parent"]},
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
        request_evidence = " ".join(
            f"initialrequest{i:02d}durableevidenceword" for i in range(8)
        )
        original = {
            "messages": [
                {"role": "system", "content": "Stable harness instructions."},
                {"role": "user", "content": request_evidence},
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

        summary = f"{request_evidence} {response_a} {response_b}"
        child_request = {"messages": [{"role": "user", "content": summary}]}
        child = tracker.infer_session(child_request, provider="gemini", model="pro")
        replay = tracker.infer_session(child_request, provider="gemini", model="pro")

        self.assertTrue(child.possible_compaction)
        self.assertEqual(parent.session_id, child.lineage_parent_session_id)
        self.assertNotEqual(parent.session_id, child.session_id)
        self.assertEqual(child.session_id, replay.session_id)
        self.assertEqual(parent.session_id, replay.lineage_parent_session_id)

    def test_unmarked_aggregator_quoting_two_responses_is_not_compaction(self):
        """Two quoted outputs without request-side history remain independent."""

        tracker = SessionTracker(ttl_seconds=3600)
        parent = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": self._long_text("aggregator-parent-user")},
                    {
                        "role": "assistant",
                        "content": self._long_text("aggregator-parent-assistant"),
                    },
                    {"role": "user", "content": self._long_text("aggregator-parent-follow-up")},
                ]
            },
            provider="gemini",
            model="pro",
        )
        response_a = " ".join(f"aggregatealpha{i:02d}longword" for i in range(8))
        response_b = " ".join(f"aggregatebravo{i:02d}longword" for i in range(8))
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
        filler_a = " ".join(f"aggregatefillera{i:02d}longword" for i in range(8))
        filler_b = " ".join(f"aggregatefillerb{i:02d}longword" for i in range(8))
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{response_a} {filler_a} {response_b} {filler_b}",
                }
            ]
        }
        namespace = parent.tracking_namespace
        probe_indexes = tracker._compaction_probe_indexes(request)
        probe_anchors = tracker._build_compaction_probe_anchors(
            request,
            namespace,
            probe_indexes=probe_indexes,
        )
        candidate = tracker._best_match(probe_anchors, namespace, time.time())
        self.assertEqual(len(candidate.response_groups), 2)
        self.assertFalse(candidate.request_groups)

        independent = tracker.infer_session(request, provider="gemini", model="pro")

        self.assertFalse(independent.possible_compaction)
        self.assertNotEqual(independent.session_id, parent.session_id)
        self.assertIsNone(independent.lineage_parent_session_id)

    def test_unmarked_compaction_does_not_bind_shared_system_harness(self):
        """Only the probe carrying parent evidence may become a context key."""

        tracker = SessionTracker(ttl_seconds=3600)
        request_evidence = " ".join(
            f"sharedrequest{i:02d}durableevidenceword" for i in range(8)
        )
        parent = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": request_evidence},
                    {"role": "assistant", "content": self._long_text("shared-parent-a")},
                    {"role": "user", "content": self._long_text("shared-parent-u")},
                ]
            },
            provider="gemini",
            model="pro",
        )
        response_a = " ".join(f"sharedalpha{i:02d}longword" for i in range(8))
        response_b = " ".join(f"sharedbravo{i:02d}longword" for i in range(8))
        for response_text in (response_a, response_b):
            tracker.record_response(
                parent.session_id,
                provider="gemini",
                model="pro",
                response=self._response(response_text),
            )
        shared_harness = self._long_text("shared-static-system-harness", repeats=18)
        summary = f"{request_evidence} {response_a} {response_b}"
        compacted = {
            "messages": [
                {"role": "system", "content": shared_harness},
                {"role": "user", "content": summary},
            ]
        }
        child = tracker.infer_session(compacted, provider="gemini", model="pro")
        self.assertTrue(child.possible_compaction)

        independent = tracker.infer_session(
            {
                "messages": [
                    {"role": "system", "content": shared_harness},
                    {"role": "user", "content": "Start a distinct unrelated task."},
                ]
            },
            provider="gemini",
            model="pro",
        )

        self.assertNotEqual(independent.session_id, child.session_id)
        self.assertFalse(independent.possible_compaction)
        self.assertIsNone(independent.lineage_parent_session_id)

    def test_unmarked_compaction_does_not_bind_shared_user_harness(self):
        """Request overlap qualifies lineage but response overlap keys its child."""

        tracker = SessionTracker(ttl_seconds=3600)
        shared_harness = self._long_text("shared-user-harness", repeats=18)
        request_evidence = " ".join(
            f"shareduserrequest{i:02d}durableword" for i in range(8)
        )
        parent = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": shared_harness},
                    {"role": "assistant", "content": self._long_text("shared-user-parent-a")},
                    {"role": "user", "content": request_evidence},
                    {"role": "assistant", "content": self._long_text("shared-user-parent-b")},
                ]
            },
            provider="gemini",
            model="pro",
        )
        response_a = " ".join(f"useralpha{i:02d}durableword" for i in range(8))
        response_b = " ".join(f"userbravo{i:02d}durableword" for i in range(8))
        for response_text in (response_a, response_b):
            tracker.record_response(
                parent.session_id,
                provider="gemini",
                model="pro",
                response=self._response(response_text),
            )
        summary = f"{request_evidence} {response_a} {response_b}"
        child = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": shared_harness},
                    {"role": "user", "content": summary},
                ]
            },
            provider="gemini",
            model="pro",
        )
        self.assertTrue(child.possible_compaction)

        independent = tracker.infer_session(
            {
                "messages": [
                    {"role": "user", "content": shared_harness},
                    {"role": "user", "content": "Start another unrelated task."},
                ]
            },
            provider="gemini",
            model="pro",
        )

        self.assertNotEqual(independent.session_id, child.session_id)
        self.assertFalse(independent.possible_compaction)

    def test_marked_compaction_does_not_bind_retained_ordinary_user_probe(self):
        """A marked summary may bind itself, never another retained probe."""

        tracker = SessionTracker(ttl_seconds=3600)
        shared_user = self._long_text("marked-retained-user", repeats=18)
        parent_request = {
            "messages": [
                {"role": "user", "content": shared_user},
                {
                    "role": "assistant",
                    "content": self._long_text("marked-retained-assistant"),
                },
                {"role": "user", "content": self._long_text("marked-retained-follow-up")},
                {
                    "role": "assistant",
                    "content": self._long_text("marked-retained-later"),
                },
            ]
        }
        parent = tracker.infer_session(parent_request, provider="gemini", model="pro")
        child = tracker.infer_session(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Summary of previous conversation: "
                        + parent_request["messages"][1]["content"],
                    },
                    {"role": "user", "content": shared_user},
                ]
            },
            provider="gemini",
            model="pro",
        )
        self.assertTrue(child.possible_compaction)
        self.assertEqual(child.lineage_parent_session_id, parent.session_id)

        standalone = tracker.infer_session(
            {"messages": [{"role": "user", "content": shared_user}]},
            provider="gemini",
            model="pro",
        )

        self.assertNotEqual(standalone.session_id, child.session_id)
        self.assertFalse(standalone.possible_compaction)

    def test_unmarked_compaction_context_continues_changed_tail_after_restart(self):
        """The evidence-bearing unmarked summary remains a stable child base."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unmarked_context_stickiness.json"
            tracker = self._persistent_tracker(path)
            request_evidence = " ".join(
                f"unmarkedcontext{i:02d}requestevidence" for i in range(8)
            )
            parent = tracker.infer_session(
                {
                    "messages": [
                        {"role": "user", "content": request_evidence},
                        {
                            "role": "assistant",
                            "content": self._long_text("unmarked-context-parent"),
                        },
                        {"role": "user", "content": self._long_text("unmarked-context-next")},
                    ]
                },
                provider="gemini",
                model="pro",
            )
            response_a = " ".join(f"contextalpha{i:02d}longword" for i in range(8))
            response_b = " ".join(f"contextbravo{i:02d}longword" for i in range(8))
            for response_text in (response_a, response_b):
                tracker.record_response(
                    parent.session_id,
                    provider="gemini",
                    model="pro",
                    response=self._response(response_text),
                )
            summary = f"{request_evidence} {response_a} {response_b}"
            first = {"messages": [{"role": "user", "content": summary}]}
            child = tracker.infer_session(first, provider="gemini", model="pro")
            self.assertTrue(child.possible_compaction)
            tracker.flush()
            tracker = self._persistent_tracker(path)

            changed_tail = {
                "messages": [
                    {"role": "user", "content": summary},
                    {"role": "user", "content": "Take a different next action."},
                ]
            }
            continued = tracker.infer_session(
                changed_tail,
                provider="gemini",
                model="pro",
            )

            self.assertEqual(continued.session_id, child.session_id)
            self.assertEqual(continued.lineage_parent_session_id, parent.session_id)
            self.assertFalse(continued.possible_compaction)

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
        request_evidence = " ".join(
            f"mostrequest{i:02d}durableevidenceword" for i in range(8)
        )
        original_messages = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": (
                    request_evidence
                    if index == 0
                    else self._long_text(f"unmarked-most-{index}")
                ),
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
        mostly_replaced = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{request_evidence} {response_a} {response_b}",
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

    def test_raw_tool_identity_does_not_override_validated_compaction(self):
        tracker = SessionTracker(ttl_seconds=3600)
        original_messages = [
            {"role": "user", "content": self._long_text("tool-parent-user")},
            {
                "role": "assistant",
                "content": self._long_text("tool-parent-assistant"),
                "tool_calls": [
                    {"id": "call_7f4e2a91c8b64d37", "type": "function"}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_7f4e2a91c8b64d37",
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
                        "tool_call_id": "call_7f4e2a91c8b64d37",
                        "content": "The retained tool result remains authoritative.",
                    },
                ]
            },
            provider="gemini",
            model="pro",
        )

        self.assertTrue(compacted.possible_compaction)
        self.assertNotEqual(parent.session_id, compacted.session_id)
        self.assertEqual(compacted.lineage_parent_session_id, parent.session_id)

    def test_changed_post_summary_history_continues_validated_compaction_child(self):
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
        with self.assertLogs("rotator_library", level="WARNING") as captured:
            changed_child = tracker.infer_session(
                changed_request,
                provider="gemini",
                model="pro",
            )

        self.assertEqual(parent.session_id, first_child.lineage_parent_session_id)
        self.assertEqual(parent.session_id, changed_child.lineage_parent_session_id)
        self.assertEqual(first_child.session_id, changed_child.session_id)
        self.assertFalse(changed_child.possible_compaction)
        self.assertEqual(changed_child.confidence, "strong")
        self.assertIn("action=compaction_continue", captured.output[0])

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
        namespace = "session-domain:test"
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
        namespace = "session-domain:test"
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

    def test_provider_and_model_changes_keep_global_logical_session(self):
        tracker = SessionTracker(ttl_seconds=3600)
        request = {
            "messages": [
                {"role": "user", "content": "Compare the provider model cache behavior using the same long prompt."},
                {"role": "assistant", "content": "The same logical session should cross providers and models."},
            ]
        }

        gemini = tracker.infer_session(request, provider="gemini", model="pro")
        openai = tracker.infer_session(request, provider="openai", model="pro")
        gemini_flash = tracker.infer_session(request, provider="gemini", model="flash")

        self.assertEqual(gemini.session_id, openai.session_id)
        self.assertEqual(gemini.session_id, gemini_flash.session_id)
        self.assertEqual(gemini.tracking_namespace, "session-domain:public")

    def test_cross_provider_first_turn_reroll_remains_ambiguous(self):
        """Removing the sole response leaves only one conversational group."""

        tracker = SessionTracker(ttl_seconds=3600)
        request = {
            "messages": [
                {"role": "system", "content": self._long_text("shared-agent-harness")},
                {"role": "developer", "content": self._long_text("shared-tool-rules")},
                {"role": "user", "content": "Write one opening scene for this story."},
            ]
        }
        deepseek = tracker.infer_session(request, provider="deepseek", model="chat")
        response = self._long_text("first-turn-response")
        tracker.record_response(
            deepseek.session_id,
            provider="deepseek",
            model="chat",
            tracking_namespace=deepseek.tracking_namespace,
            response=self._response(response),
        )

        reroll = tracker.infer_session(request, provider="openai", model="gpt")

        self.assertNotEqual(reroll.session_id, deepseek.session_id)
        self.assertEqual(reroll.confidence, "weak")

    def test_cross_provider_complete_first_turn_and_follow_up_continue(self):
        tracker = SessionTracker(ttl_seconds=3600)
        initial = {
            "messages": [
                {"role": "system", "content": self._long_text("shared-agent-harness")},
                {"role": "user", "content": self._long_text("global-first-user")},
            ]
        }
        deepseek = tracker.infer_session(initial, provider="deepseek", model="chat")
        response = self._long_text("global-first-response")
        tracker.record_response(
            deepseek.session_id,
            provider="deepseek",
            model="chat",
            tracking_namespace=deepseek.tracking_namespace,
            response=self._response(response),
        )
        continued_request = {
            "messages": [
                *initial["messages"],
                {"role": "assistant", "content": response},
                {"role": "user", "content": self._long_text("global-follow-up")},
            ]
        }

        anthropic = tracker.infer_session(
            continued_request,
            provider="anthropic",
            model="claude",
        )

        self.assertEqual(anthropic.session_id, deepseek.session_id)
        self.assertEqual(anthropic.confidence, "probable")

    def test_trusted_client_identity_crosses_provider_inside_domain(self):
        tracker = SessionTracker(
            ttl_seconds=3600,
            trusted_explicit_fields=["conversation_id"],
        )
        request = {"conversation_id": "client-global-session"}

        deepseek = tracker.infer_session(request, provider="deepseek", model="chat")
        openai = tracker.infer_session(request, provider="openai", model="gpt")

        self.assertEqual(openai.session_id, deepseek.session_id)
        self.assertEqual(openai.confidence, "strong")

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

    def test_request_isolation_keys_are_stable_secret_free_and_strict(self):
        self.assertEqual(
            RequestContextBuilder._session_isolation_key(None, None, None, False),
            "public",
        )
        classifier_a = RequestContextBuilder._session_isolation_key(
            "agent-a", None, None, False
        )
        classifier_a_again = RequestContextBuilder._session_isolation_key(
            "agent-a", None, None, False
        )
        classifier_b = RequestContextBuilder._session_isolation_key(
            "agent-b", None, None, False
        )
        classifier_a_with_rotated_credentials = RequestContextBuilder._session_isolation_key(
            "agent-a",
            {"openai": ["rotated-secret"]},
            {"openai": {"api_base": "https://rotated.example"}},
            True,
        )
        self.assertEqual(classifier_a, classifier_a_again)
        self.assertEqual(classifier_a, classifier_a_with_rotated_credentials)
        self.assertNotEqual(classifier_a, classifier_b)

        bundle_a = RequestContextBuilder._session_isolation_key(
            None,
            {"deepseek": ["secret-b", "secret-a"], "openai": "secret-c"},
            {"openai": {"api_base": "https://one.example"}},
            True,
        )
        bundle_a_reordered = RequestContextBuilder._session_isolation_key(
            None,
            {"openai": "secret-c", "deepseek": ["secret-a", "secret-b"]},
            {"openai": {"api_base": "https://one.example"}},
            True,
        )
        bundle_a_mixed_case = RequestContextBuilder._session_isolation_key(
            None,
            {"OpenAI": "secret-c", "DeepSeek": ["secret-a", "secret-b"]},
            {"OpenAI": {"api_base": "https://one.example"}},
            True,
        )
        bundle_b = RequestContextBuilder._session_isolation_key(
            None,
            {"deepseek": ["different-secret"], "openai": "secret-c"},
            {"openai": {"api_base": "https://one.example"}},
            True,
        )
        self.assertEqual(bundle_a, bundle_a_reordered)
        self.assertEqual(bundle_a, bundle_a_mixed_case)
        self.assertNotEqual(bundle_a, bundle_b)
        self.assertNotIn("secret", bundle_a)

    def test_global_evidence_never_crosses_isolation_domains(self):
        tracker = SessionTracker(ttl_seconds=3600)
        request = {
            "messages": [
                {"role": "user", "content": self._long_text("strict-domain-user")},
                {"role": "assistant", "content": self._long_text("strict-domain-assistant")},
            ]
        }

        domain_a = tracker.infer_session(
            request,
            provider="deepseek",
            model="chat",
            scope_key="classifier:a",
        )
        domain_b = tracker.infer_session(
            request,
            provider="openai",
            model="gpt",
            scope_key="classifier:b",
        )

        self.assertNotEqual(domain_a.session_id, domain_b.session_id)
        self.assertNotEqual(domain_a.tracking_namespace, domain_b.tracking_namespace)

    def test_raw_scope_text_cannot_impersonate_derived_bundle_domain(self):
        tracker = SessionTracker(ttl_seconds=3600)
        derived_domain = "bundle:" + ("a" * 64)
        request = {
            "messages": [
                {"role": "user", "content": self._long_text("scope-marker-collision-user")},
                {
                    "role": "assistant",
                    "content": self._long_text("scope-marker-collision-assistant"),
                },
            ]
        }

        trusted = tracker.infer_session(
            request,
            provider="openai",
            model="gpt",
            scope_key=derived_domain,
            _trusted_isolation_key=True,
        )
        raw = tracker.infer_session(
            request,
            provider="openai",
            model="gpt",
            scope_key=derived_domain,
        )

        self.assertNotEqual(raw.session_id, trusted.session_id)
        self.assertNotEqual(raw.tracking_namespace, trusted.tracking_namespace)

    def test_provider_strong_hint_reuses_session(self):
        tracker = SessionTracker(ttl_seconds=3600)
        hints = SessionTrackingHints(strong_anchors=["native-session-abc"])

        first = tracker.infer_session({}, provider="custom", model="m", hints=hints)
        second = tracker.infer_session({}, provider="custom", model="m", hints=hints)

        self.assertEqual(first.session_id, second.session_id)
        self.assertIsNotNone(first.affinity_key)

    def test_provider_native_hint_does_not_cross_provider(self):
        tracker = SessionTracker(ttl_seconds=3600)
        hints = SessionTrackingHints(strong_anchors=["native-session-abc"])

        gemini = tracker.infer_session({}, provider="gemini", model="pro", hints=hints)
        openai = tracker.infer_session({}, provider="openai", model="gpt", hints=hints)

        self.assertNotEqual(gemini.session_id, openai.session_id)
        self.assertNotEqual(gemini.affinity_key, openai.affinity_key)

    def test_provider_session_scope_cannot_cross_caller_isolation_domain(self):
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

    def test_provider_session_scope_partitions_only_native_evidence(self):
        tracker = SessionTracker(ttl_seconds=3600)
        model_a_hints = SessionTrackingHints(
            strong_anchors=["native-session-abc"],
            session_scope="model-a",
        )
        model_b_hints = SessionTrackingHints(
            strong_anchors=["native-session-abc"],
            session_scope="model-b",
        )

        model_a = tracker.infer_session(
            {}, provider="custom", model="a", hints=model_a_hints
        )
        model_b = tracker.infer_session(
            {}, provider="custom", model="b", hints=model_b_hints
        )

        self.assertNotEqual(model_a.session_id, model_b.session_id)
        self.assertEqual(model_a.tracking_namespace, model_b.tracking_namespace)
        self.assertNotEqual(model_a.affinity_key, model_b.affinity_key)

    def test_provider_native_scope_partition_survives_persistence_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "provider_native_scope.json"
            tracker = self._persistent_tracker(path)
            hints_a = SessionTrackingHints(
                strong_anchors=["provider-native-secret-a"],
                session_scope="family-a",
            )
            hints_b = SessionTrackingHints(
                strong_anchors=["provider-native-secret-a"],
                session_scope="family-b",
            )
            session_a = tracker.infer_session(
                {}, provider="custom", model="a", hints=hints_a
            )
            tracker.flush()
            restored = self._persistent_tracker(path)

            continued_a = restored.infer_session(
                {}, provider="custom", model="a2", hints=hints_a
            )
            independent_b = restored.infer_session(
                {}, provider="custom", model="b", hints=hints_b
            )

        self.assertEqual(continued_a.session_id, session_a.session_id)
        self.assertNotEqual(independent_b.session_id, session_a.session_id)

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

        self.assertEqual(persisted["schema_version"], 3)
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

    def test_schema_three_persistence_hashes_all_external_identifiers(self):
        secrets = {
            "explicit": "client-conversation-secret",
            "provider": "provider-native-secret",
            "affinity": "provider-affinity-secret",
            "tool": "tool-call-secret",
            "response": "response-id-secret",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session_stickiness.json"
            tracker = SessionTracker(
                ttl_seconds=3600,
                trusted_explicit_fields=["conversation_id"],
                persist_to_disk=True,
                persistence_path=path,
                persistence_flush_interval_seconds=0,
            )
            inferred = tracker.infer_session(
                {
                    "conversation_id": secrets["explicit"],
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [{"id": secrets["tool"], "type": "function"}],
                        }
                    ],
                },
                provider="custom",
                model="model-a",
                hints=SessionTrackingHints(
                    strong_anchors=[secrets["provider"]],
                    affinity_key=secrets["affinity"],
                ),
            )
            tracker.record_response(
                inferred.session_id,
                tracking_namespace=inferred.tracking_namespace,
                response={"id": secrets["response"], "object": "response"},
            )
            tracker.flush()
            persisted_text = path.read_text(encoding="utf-8")

        for secret in secrets.values():
            self.assertNotIn(secret, persisted_text)

    def test_schema_three_loader_bounds_sessions_and_rejects_unknown_sources(self):
        now = time.time()
        namespace = "session-domain:test"
        sessions = {
            f"session-{index}": {
                "namespace": namespace,
                "expires_at": now + 3600,
                "last_seen": now + index,
                "history_signatures": [],
            }
            for index in range(3)
        }
        evil_anchor = f"{namespace}:evil:{'a' * 64}"
        payload = {
            "schema_version": 3,
            "sessions": sessions,
            "anchors": {
                evil_anchor: {
                    "session_id": "session-2",
                    "namespace": namespace,
                    "strength": "strong",
                    "source": "compaction_probe",
                    "group": "forged",
                    "expires_at": now + 3600,
                    "last_seen": now,
                }
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session_stickiness.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(SessionTracker, "_MAX_PERSISTED_SESSIONS", 2):
                tracker = SessionTracker(
                    ttl_seconds=3600,
                    persist_to_disk=True,
                    persistence_path=path,
                )

        self.assertEqual(set(tracker._sessions), {"session-1", "session-2"})
        self.assertNotIn(evil_anchor, tracker._anchors)

    def test_schema_three_loader_rejects_oversized_state_before_json_parsing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "session_stickiness.json"
            path.write_text("{" + ("x" * 128), encoding="utf-8")
            with patch.object(SessionTracker, "_MAX_PERSISTED_FILE_BYTES", 64):
                tracker = SessionTracker(
                    ttl_seconds=3600,
                    persist_to_disk=True,
                    persistence_path=path,
                )

        self.assertFalse(tracker._sessions)
        self.assertFalse(tracker._anchors)

    def test_persistence_restart_preserves_compaction_and_replay_binding(self):
        request_evidence = " ".join(
            f"persistrequest{i:02d}durableevidenceword" for i in range(8)
        )
        original = {
            "messages": [
                {"role": "user", "content": request_evidence},
                {"role": "assistant", "content": self._long_text("persist-parent-assistant")},
                {"role": "user", "content": self._long_text("persist-parent-follow-up")},
            ]
        }
        response_a = " ".join(f"persistalpha{i:02d}word" for i in range(8))
        response_b = " ".join(f"persistbravo{i:02d}word" for i in range(8))
        child_request = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{request_evidence} {response_a} {response_b}",
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
            json.dumps({"schema_version": 2, "sessions": {}, "anchors": {}}),
            json.dumps({"schema_version": 3, "sessions": [], "anchors": {}}),
            json.dumps(
                {
                    "schema_version": 3,
                    "sessions": {
                        "bad": {
                            "namespace": "session-domain:x",
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
        namespace = "session-domain:test"
        expires_at = time.time() + 3600
        good_value = f"{namespace}:message:user:{'a' * 64}"
        payload = {
            "schema_version": 3,
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
                    "namespace": "session-domain:other",
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
        namespace = "session-domain:test"
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
            "schema_version": 3,
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

    def test_streaming_collector_reassembles_structural_tool_event(self):
        handler = StreamingHandler()
        assistant_parts = []
        tool_call_ids = []
        tool_call_events = {}
        handler._collect_session_response_anchors(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            '"function":{"name":"read_file","arguments":"{\\"path\\":"}}]}}]}'
            "\n\n",
            assistant_parts,
            tool_call_ids,
            tool_call_events,
        )
        handler._collect_session_response_anchors(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"\\"a.py\\"}"}}]}}]}\n\n',
            assistant_parts,
            tool_call_ids,
            tool_call_events,
        )

        assembled = handler._assembled_tool_calls(tool_call_ids, tool_call_events)

        self.assertEqual(
            assembled,
            [
                {
                    "id": "call_1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"a.py"}',
                    },
                }
            ],
        )

    def test_streaming_tool_events_separate_choices_with_shared_indexes(self):
        handler = StreamingHandler()
        tool_call_ids = []
        tool_call_events = {}
        handler._collect_session_response_anchors(
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_a",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"a.py"}',
                                        },
                                    }
                                ]
                            }
                        },
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_b",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"b.py"}',
                                        },
                                    }
                                ]
                            }
                        },
                    ]
                }
            )
            + "\n\n",
            [],
            tool_call_ids,
            tool_call_events,
        )

        assembled = handler._assembled_tool_calls(tool_call_ids, tool_call_events)

        self.assertEqual([call["id"] for call in assembled], ["call_a", "call_b"])
        self.assertEqual(
            [call["function"]["arguments"] for call in assembled],
            ['{"path":"a.py"}', '{"path":"b.py"}'],
        )

    def test_streaming_tool_events_use_choice_index_across_separate_frames(self):
        handler = StreamingHandler()
        tool_call_ids = []
        tool_call_events = {}
        for choice_index, call_id, path in (
            (0, "call_a", "a.py"),
            (1, "call_b", "b.py"),
        ):
            handler._collect_session_response_anchors(
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "index": choice_index,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": call_id,
                                            "function": {
                                                "name": "read_file",
                                                "arguments": json.dumps({"path": path}),
                                            },
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                )
                + "\n\n",
                [],
                tool_call_ids,
                tool_call_events,
            )

        assembled = handler._assembled_tool_calls(tool_call_ids, tool_call_events)

        self.assertEqual([call["id"] for call in assembled], ["call_a", "call_b"])
        self.assertEqual(
            [call["function"]["arguments"] for call in assembled],
            ['{"path": "a.py"}', '{"path": "b.py"}'],
        )

    def test_streaming_cumulative_argument_snapshots_replace_instead_of_append(self):
        handler = StreamingHandler()
        tool_call_ids = []
        tool_call_events = {}
        for arguments in ('{"path":', '{"path":"a.py"}'):
            handler._collect_session_response_anchors(
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": arguments,
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
                + "\n\n",
                [],
                tool_call_ids,
                tool_call_events,
            )

        assembled = handler._assembled_tool_calls(tool_call_ids, tool_call_events)

        self.assertEqual(
            assembled[0]["function"]["arguments"],
            '{"path":"a.py"}',
        )

    def test_completed_streamed_tool_call_bridges_closed_event_across_provider(self):
        handler = StreamingHandler()
        tracker = SessionTracker(ttl_seconds=3600)
        user_message = {
            "role": "user",
            "content": self._long_text("streamed-tool-event-user"),
        }
        root = tracker.infer_session(
            {"messages": [user_message]}, provider="openai", model="gpt"
        )
        responses = []

        def record(response):
            responses.append(response)
            tracker.record_response(
                root.session_id,
                tracking_namespace=root.tracking_namespace,
                response=response,
            )

        async def stream():
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_streamed",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"a.py"}'},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ) + "\n\n"

        async def consume():
            return [
                chunk
                async for chunk in handler.wrap_stream(
                    stream(),
                    "credential",
                    "model",
                    response_callback=record,
                )
            ]

        asyncio.run(consume())
        assistant = responses[0]["choices"][0]["message"]
        continued = tracker.infer_session(
            {
                "messages": [
                    user_message,
                    assistant,
                    self._tool_result("call_streamed"),
                ]
            },
            provider="anthropic",
            model="claude",
        )

        self.assertEqual(
            assistant["tool_calls"][0]["function"]["arguments"],
            '{"path":"a.py"}',
        )
        self.assertEqual(continued.session_id, root.session_id)
        self.assertEqual(continued.confidence, "probable")

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

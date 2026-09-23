# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Model-string arguments: ``provider/model:arg`` request hints."""

import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rotator_library.client.model_args import (
    apply_model_args_to_kwargs,
    split_model_args,
)
from rotator_library.client.request_builder import RequestContextBuilder


class SplitModelArgsTests(unittest.TestCase):
    def test_split_known_suffixes(self):
        self.assertEqual(
            split_model_args("deepseek/deepseek-flash:high"),
            ("deepseek/deepseek-flash", {"reasoning_effort": "high"}),
        )
        self.assertEqual(
            split_model_args("provider/model:max"),
            ("provider/model", {"reasoning_effort": "max"}),
        )
        # Case-insensitive match, canonical value applied
        clean, args = split_model_args("provider/model:HIGH")
        self.assertEqual(clean, "provider/model")
        self.assertEqual(args, {"reasoning_effort": "high"})

    def test_unknown_suffixes_ride_verbatim(self):
        # OpenRouter :free variants, Ollama model:tag ids
        for model in ("openrouter/kimi-k2.5:free", "ollama/llama3:8b"):
            clean, args = split_model_args(model)
            self.assertEqual((clean, args), (model, {}))
        # Consumption stops at the first non-vocabulary segment from the end
        clean, args = split_model_args("provider/model:high:8b")
        self.assertEqual((clean, args), ("provider/model:high:8b", {}))
        # Trailing empty segment is not an argument
        clean, args = split_model_args("provider/model:")
        self.assertEqual((clean, args), ("provider/model:", {}))

    def test_multiple_known_args_endmost_wins(self):
        clean, args = split_model_args("provider/model:low:max")
        self.assertEqual(clean, "provider/model")
        self.assertEqual(args, {"reasoning_effort": "max"})

    def test_no_colon_or_empty(self):
        self.assertEqual(split_model_args("provider/model"), ("provider/model", {}))
        self.assertEqual(split_model_args(""), ("", {}))

    def test_registry_extension(self):
        from rotator_library.client import model_args

        model_args.register_model_arg("json", "response_format", "json_object")
        try:
            clean, args = split_model_args("prov/m:json")
            self.assertEqual(clean, "prov/m")
            self.assertEqual(args, {"response_format": "json_object"})
        finally:
            model_args._REGISTRY.pop("json", None)


class ApplyModelArgsTests(unittest.TestCase):
    def test_apply_only_when_absent(self):
        kwargs = {}
        self.assertTrue(apply_model_args_to_kwargs(kwargs, {"reasoning_effort": "high"}))
        self.assertEqual(kwargs["reasoning_effort"], "high")

    def test_explicit_param_wins(self):
        kwargs = {"reasoning_effort": "low"}
        self.assertFalse(apply_model_args_to_kwargs(kwargs, {"reasoning_effort": "high"}))
        self.assertEqual(kwargs["reasoning_effort"], "low")

    def test_falsy_explicit_value_is_filled(self):
        # Matches the source semantics: an empty explicit effort counts
        # as absent and the hint fills it.
        kwargs = {"reasoning_effort": ""}
        self.assertTrue(apply_model_args_to_kwargs(kwargs, {"reasoning_effort": "high"}))
        self.assertEqual(kwargs["reasoning_effort"], "high")

    def test_empty_args_is_noop(self):
        kwargs = {"model": "m"}
        self.assertFalse(apply_model_args_to_kwargs(kwargs, {}))
        self.assertEqual(kwargs, {"model": "m"})


class BuilderIntegrationTests(unittest.TestCase):
    """The real request path: ``provider/model:high`` keys provider
    extraction and alias resolution on the clean id, and the hint lands
    in the request kwargs only when the client sent none."""

    def _make_builder(self, seen_models):
        async def resolve_scope_for_provider(provider, classifier, keys, providers, private):
            return {
                "credentials": ["cred-1"],
                "usage_manager_key": "scope-key",
                "provider_config": None,
                "credential_secrets": {},
                "classifier": None,
            }

        resolver = SimpleNamespace(resolve_model_id=lambda model, provider: model)

        def infer_session(kwargs, provider, model, scope_key, hints):
            seen_models.append(model)
            return SimpleNamespace(
                session_id="s",
                affinity_key="a",
                possible_compaction=False,
                lineage_parent_session_id=None,
                tracking_namespace="ns",
            )

        tracker = SimpleNamespace(infer_session=infer_session)
        return RequestContextBuilder(
            resolve_scope_for_provider=resolve_scope_for_provider,
            model_resolver=resolver,
            session_tracker=tracker,
            get_global_timeout=lambda: 60,
            get_enable_request_logging=lambda: False,
        )

    def test_builder_splits_and_applies(self):
        seen = []
        builder = self._make_builder(seen)
        kwargs = {"model": "deepseek/deepseek-flash:high"}
        context = asyncio.run(builder.build_completion_context(None, None, kwargs))
        # Clean id keys routing, alias resolution, and the session tracker
        self.assertEqual(context.provider, "deepseek")
        self.assertEqual(context.model, "deepseek/deepseek-flash")
        self.assertEqual(context.kwargs["model"], "deepseek/deepseek-flash")
        self.assertEqual(seen, ["deepseek/deepseek-flash"])
        # The hint behaves as a client-sent parameter
        self.assertEqual(context.kwargs["reasoning_effort"], "high")

    def test_builder_explicit_param_wins(self):
        builder = self._make_builder([])
        kwargs = {
            "model": "deepseek/deepseek-flash:high",
            "reasoning_effort": "low",
        }
        context = asyncio.run(builder.build_completion_context(None, None, kwargs))
        self.assertEqual(context.model, "deepseek/deepseek-flash")
        self.assertEqual(context.kwargs["reasoning_effort"], "low")

    def test_builder_unknown_suffix_rides_verbatim(self):
        builder = self._make_builder([])
        kwargs = {"model": "openrouter/kimi-k2.5:free"}
        context = asyncio.run(builder.build_completion_context(None, None, kwargs))
        self.assertEqual(context.provider, "openrouter")
        self.assertEqual(context.model, "openrouter/kimi-k2.5:free")
        self.assertNotIn("reasoning_effort", context.kwargs)


if __name__ == "__main__":
    unittest.main()

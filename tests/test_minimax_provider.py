import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import httpx
import litellm

from rotator_library.minimax_config import (
    ANTHROPIC_PROTOCOL,
    CN_ZH,
    GLOBAL_EN,
    OPENAI_PROTOCOL,
    get_minimax_endpoint,
)
from rotator_library.client.transforms import ProviderTransforms
from rotator_library.model_info_service import ModelRegistry
from rotator_library.provider_config import ProviderConfig
from rotator_library.providers.minimax_provider import MinimaxProvider


class _CaptureHandler(BaseHTTPRequestHandler):
    captured_paths = []
    captured_bodies = []

    def do_POST(self):  # noqa: N802
        self.__class__.captured_paths.append(self.path)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.__class__.captured_bodies.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
                if self.path.endswith("/messages")
                else {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
        )

    def log_message(self, format, *args):  # noqa: A002
        return


class MiniMaxProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_catalog_preserves_only_explicit_extra_models(self):
        provider = MinimaxProvider()
        async with httpx.AsyncClient() as client:
            with patch.object(
                provider.model_definitions, "get_all_provider_models", return_value=[]
            ), patch.object(client, "get", side_effect=AssertionError("No discovery")):
                self.assertEqual(
                    await provider.get_models("test-key", client),
                    ["minimax/MiniMax-M3.1-Flash", "minimax/MiniMax-M3"],
                )
            with patch.object(
                provider.model_definitions,
                "get_all_provider_models",
                return_value=[
                    "minimax/MiniMax-M2.7",
                    "minimax/custom-model",
                    "minimax/MiniMax-M3",
                ],
            ):
                self.assertEqual(
                    await provider.get_models("test-key", client),
                    [
                        "minimax/MiniMax-M3.1-Flash",
                        "minimax/MiniMax-M3",
                        "minimax/MiniMax-M2.7",
                        "minimax/custom-model",
                    ],
                )

    async def test_anthropic_request_capture_appends_v1_messages(self):
        _CaptureHandler.captured_paths = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}/anthropic"
            with patch.dict(
                os.environ,
                {
                    "MINIMAX_API_PROTOCOL": ANTHROPIC_PROTOCOL,
                    "MINIMAX_API_REGION": GLOBAL_EN,
                    "MINIMAX_GLOBAL_ANTHROPIC_BASE_URL": base_url,
                },
            ):
                kwargs = {
                    "model": "minimax/MiniMax-M3",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1,
                }
                await MinimaxProvider().transform_request(
                    kwargs, kwargs["model"], "test-key"
                )
                kwargs["api_key"] = "test-key"
                await litellm.acompletion(**kwargs)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(_CaptureHandler.captured_paths, ["/anthropic/v1/messages"])

    async def test_serialized_requests_preserve_models_regions_and_history(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for region in (GLOBAL_EN, CN_ZH):
                for protocol in (OPENAI_PROTOCOL, ANTHROPIC_PROTOCOL):
                    for model in (
                        "MiniMax-M3.1-Flash",
                        "MiniMax-M3",
                        "MiniMax-M2.7",
                        "custom-model",
                    ):
                        with self.subTest(
                            region=region, protocol=protocol, model=model
                        ):
                            suffix = (
                                "v1" if protocol == OPENAI_PROTOCOL else "anthropic"
                            )
                            region_key = "GLOBAL" if region == GLOBAL_EN else "CN"
                            endpoint = f"http://127.0.0.1:{server.server_port}/{suffix}"
                            with patch.dict(
                                os.environ,
                                {
                                    "MINIMAX_API_REGION": region,
                                    "MINIMAX_API_PROTOCOL": protocol,
                                    f"MINIMAX_{region_key}_{protocol.upper()}_BASE_URL": endpoint,
                                },
                            ):
                                messages = [
                                    {"role": "user", "content": "Remember this"},
                                    {"role": "assistant", "content": "Remembered"},
                                    {"role": "user", "content": "What did I say?"},
                                ]
                                transforms = ProviderTransforms(
                                    {"minimax": MinimaxProvider}, ProviderConfig()
                                )
                                kwargs = await transforms.apply(
                                    "minimax",
                                    f"minimax/{model}",
                                    "test-key",
                                    {
                                        "model": f"minimax/{model}",
                                        "messages": messages,
                                        "max_tokens": 16,
                                    },
                                )
                                await litellm.acompletion(api_key="test-key", **kwargs)
                            body = _CaptureHandler.captured_bodies[-1]
                            self.assertEqual(body["model"], model)
                            self.assertEqual(
                                _CaptureHandler.captured_paths[-1],
                                (
                                    "/v1/chat/completions"
                                    if protocol == OPENAI_PROTOCOL
                                    else "/anthropic/v1/messages"
                                ),
                            )
                            self.assertEqual(
                                [m["role"] for m in body["messages"]],
                                ["user", "assistant", "user"],
                            )
                            self.assertIn("Remember this", json.dumps(body["messages"]))
                            self.assertNotIn("service_tier", body)
                            self.assertNotIn("reasoning_split", body)
                            self.assertNotEqual(body.get("reasoning_effort"), "none")
                            self.assertNotEqual(
                                body.get("thinking"), {"type": "disabled"}
                            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    async def test_flash_effort_and_multimodal_tool_history_on_wire(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for protocol in (OPENAI_PROTOCOL, ANTHROPIC_PROTOCOL):
                for effort in ("low", "medium", "high", "xhigh", "max"):
                    with self.subTest(protocol=protocol, effort=effort):
                        suffix = "v1" if protocol == OPENAI_PROTOCOL else "anthropic"
                        with patch.dict(
                            os.environ,
                            {
                                "MINIMAX_API_PROTOCOL": protocol,
                                "MINIMAX_API_REGION": GLOBAL_EN,
                                f"MINIMAX_GLOBAL_{protocol.upper()}_BASE_URL": f"http://127.0.0.1:{server.server_port}/{suffix}",
                            },
                        ):
                            kwargs = await ProviderTransforms(
                                {"minimax": MinimaxProvider}, ProviderConfig()
                            ).apply(
                                "minimax",
                                "minimax/MiniMax-M3.1-Flash",
                                "test-key",
                                {
                                    "model": "minimax/MiniMax-M3.1-Flash",
                                    "max_tokens": 16,
                                    "reasoning_effort": effort,
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": "Inspect this",
                                                },
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": "data:image/png;base64,aGVsbG8="
                                                    },
                                                },
                                            ],
                                        },
                                        {
                                            "role": "assistant",
                                            "content": "Checking",
                                            "tool_calls": [
                                                {
                                                    "id": "call_1",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "lookup",
                                                        "arguments": "{}",
                                                    },
                                                }
                                            ],
                                        },
                                        {
                                            "role": "tool",
                                            "tool_call_id": "call_1",
                                            "content": "Found",
                                        },
                                    ],
                                    "tools": [
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "lookup",
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {},
                                                },
                                            },
                                        }
                                    ],
                                },
                            )
                            await litellm.acompletion(api_key="test-key", **kwargs)
                        body = _CaptureHandler.captured_bodies[-1]
                        if protocol == OPENAI_PROTOCOL:
                            self.assertEqual(body["reasoning_effort"], effort)
                        else:
                            self.assertEqual(body["output_config"]["effort"], effort)
                            self.assertNotIn("thinking", body)
                        serialized = json.dumps(body)
                        for preserved in ("aGVsbG8=", "call_1", "Found", "lookup"):
                            self.assertIn(preserved, serialized)
                        self.assertNotIn("service_tier", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    async def test_saved_effort_options_are_adapted_without_mutating_configuration(
        self,
    ):
        saved = {"reasoning_effort": "xhigh", "temperature": 0.7}
        provider = MinimaxProvider()
        with patch.dict(
            os.environ, {"MINIMAX_API_PROTOCOL": ANTHROPIC_PROTOCOL}
        ), patch.object(
            provider.model_definitions, "get_model_options", return_value=saved
        ):
            options = provider.get_model_options("minimax/MiniMax-M3.1-Flash")
        self.assertEqual(options["output_config"], {"effort": "xhigh"})
        self.assertNotIn("reasoning_effort", options)
        self.assertEqual(saved, {"reasoning_effort": "xhigh", "temperature": 0.7})

    async def test_native_model_metadata_and_endpoint_regions(self):
        registry = ModelRegistry()
        metadata = registry.lookup("minimax/MiniMax-M3")

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.limits.context_window, 1_000_000)
        self.assertEqual(metadata.pricing.prompt, 0.3 / 1_000_000)
        self.assertEqual(metadata.pricing.completion, 1.2 / 1_000_000)
        self.assertEqual(metadata.pricing.cached_input, 0.06 / 1_000_000)
        self.assertEqual(metadata.input_types, ["text", "image", "video"])
        self.assertTrue(metadata.capabilities.tools)
        self.assertTrue(metadata.capabilities.functions)
        self.assertTrue(metadata.capabilities.interleaved)
        self.assertEqual(metadata.capabilities.thinking_modes, ["adaptive", "disabled"])
        self.assertFalse(metadata.pricing.tiers)

        flash = registry.lookup("minimax/MiniMax-M3.1-Flash")
        self.assertEqual(flash.limits.context_window, 1_000_000)
        self.assertEqual(flash.limits.max_output, 524_288)
        self.assertEqual(flash.pricing.prompt, 0.15 / 1_000_000)
        self.assertEqual(flash.pricing.completion, 0.60 / 1_000_000)
        self.assertEqual(flash.pricing.cached_input, 0.015 / 1_000_000)
        self.assertIsNone(flash.pricing.cache_write)
        self.assertEqual(flash.capabilities.thinking_modes, ["adaptive"])
        self.assertTrue(flash.capabilities.tools)
        self.assertTrue(flash.capabilities.functions)
        self.assertTrue(flash.capabilities.interleaved)
        self.assertEqual(flash.input_types, ["text", "image", "video"])

        secondary = registry.lookup("minimax/MiniMax-M2.7")
        self.assertIsNotNone(secondary)
        self.assertEqual(secondary.limits.context_window, 204_800)
        self.assertEqual(secondary.pricing.cache_write, 0.375 / 1_000_000)
        self.assertEqual(secondary.input_types, ["text"])
        self.assertEqual(secondary.capabilities.thinking_modes, ["always_on"])
        self.assertFalse(secondary.supported_parameters)
        self.assertEqual(
            get_minimax_endpoint(GLOBAL_EN, OPENAI_PROTOCOL),
            "https://api.minimax.io/v1",
        )
        self.assertEqual(
            get_minimax_endpoint(CN_ZH, ANTHROPIC_PROTOCOL),
            "https://api.minimaxi.com/anthropic",
        )

    async def test_provider_hook_endpoint_survives_global_override(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_BASE": "https://global.example/v1",
                "MINIMAX_API_REGION": GLOBAL_EN,
                "MINIMAX_API_PROTOCOL": OPENAI_PROTOCOL,
            },
        ):
            transforms = ProviderTransforms(
                {"minimax": MinimaxProvider}, ProviderConfig()
            )
            result = await transforms.apply(
                "minimax",
                "minimax/MiniMax-M3",
                "test-key",
                {"model": "minimax/MiniMax-M3"},
            )

        self.assertEqual(result["api_base"], "https://api.minimax.io/v1")
        self.assertEqual(result["custom_llm_provider"], "openai")

    async def test_invalid_anthropic_override_uses_selected_default(self):
        with patch.dict(
            os.environ,
            {
                "MINIMAX_API_REGION": GLOBAL_EN,
                "MINIMAX_API_PROTOCOL": ANTHROPIC_PROTOCOL,
                "MINIMAX_GLOBAL_ANTHROPIC_BASE_URL": "https://invalid.example/v1",
            },
        ):
            endpoint = get_minimax_endpoint(protocol=ANTHROPIC_PROTOCOL)

        self.assertEqual(endpoint, "https://api.minimax.io/anthropic")

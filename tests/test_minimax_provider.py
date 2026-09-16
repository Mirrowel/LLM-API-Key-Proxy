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

    def do_POST(self):  # noqa: N802
        self.__class__.captured_paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "MiniMax-M3",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()
        )

    def log_message(self, format, *args):  # noqa: A002
        return


class MiniMaxProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_models_preserve_dynamic_models(self):
        async def handler(request):
            return httpx.Response(
                200,
                json={"data": [{"id": "MiniMax-Extra"}, {"id": "MiniMax-M3"}]},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            models = await MinimaxProvider().get_models("test-key", client)

        self.assertEqual(
            models[:2], ["minimax/MiniMax-M3", "minimax/MiniMax-M2.7"]
        )
        self.assertIn("minimax/MiniMax-Extra", models)
        self.assertEqual(models.count("minimax/MiniMax-M3"), 1)

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

    async def test_native_model_metadata_and_endpoint_regions(self):
        registry = ModelRegistry()
        metadata = registry.lookup("minimax/MiniMax-M3")

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.limits.context_window, 1_000_000)
        self.assertEqual(metadata.pricing.prompt, 0.6 / 1_000_000)
        self.assertEqual(metadata.pricing.completion, 2.4 / 1_000_000)
        self.assertEqual(metadata.pricing.cached_input, 0.12 / 1_000_000)
        self.assertEqual(metadata.input_types, ["text", "image", "video"])
        self.assertTrue(metadata.capabilities.tools)
        self.assertTrue(metadata.capabilities.functions)
        self.assertTrue(metadata.capabilities.interleaved)
        self.assertEqual(
            metadata.capabilities.thinking_modes, ["adaptive", "disabled"]
        )
        self.assertFalse(metadata.pricing.tiers)

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

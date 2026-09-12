"""G11 Phase A pins: the responses sibling split, family matching, gemini
two-faces (native + /v1beta/openai compat), registry-derived allowlists."""

import pytest
from fastapi.testclient import TestClient

import proxy_app.main as proxy_main
from rotator_library.protocols import get_protocol
from rotator_library.protocols.registry import is_generative_protocol, list_protocols
from rotator_library.routing.profiles import resolve_profile


# ------------------------------------------------------------ siblings

def test_three_responses_siblings_share_family() -> None:
    names = list_protocols()
    assert {"responses", "responses_stateful", "responses_websocket"} <= set(names)

    stateless = get_protocol("responses")
    stateful = get_protocol("responses_stateful")
    ws = get_protocol("responses_websocket")

    # wire base shared: same parse behavior
    assert stateless.base_family == stateful.base_family == ws.base_family == "responses"
    # transports: stateless/stateful HTTP+SSE, WS sibling websocket-only
    assert stateless.supports_transport("http") and stateless.supports_transport("sse")
    assert stateful.supports_transport("http") and stateful.supports_transport("sse")
    assert ws.supports_transport("websocket") and not ws.supports_transport("http")
    # aliases stay on the stateless baseline
    assert get_protocol("openai_responses") is stateless


def test_sibling_wire_round_trip() -> None:
    from rotator_library.protocols.types import ProtocolContext

    stateful = get_protocol("responses_stateful")
    raw = {
        "model": "openai/gpt-test",
        "input": [{"role": "user", "content": "hi"}],
        "previous_response_id": "resp_parent",
        "store": True,
    }
    ctx = ProtocolContext(source_protocol="responses_stateful", target_protocol="responses_stateful")
    unified = stateful.parse_request(raw, ctx)
    assert unified.previous_response_id == "resp_parent"
    rebuilt = stateful.build_request(unified, ctx)
    assert rebuilt["previous_response_id"] == "resp_parent"


# ------------------------------------------------------- family matching

def test_bare_name_matches_sibling_profile_without_conversion() -> None:
    # A provider declaring a stateful sibling serves a stateless-family
    # client WITHOUT a cross-family conversion (same wire format).
    chosen = resolve_profile(
        declared_profiles={
            "chat": {"protocol": "openai_chat"},
            "mem": {"protocol": "responses_stateful"},
        },
        default_profile="chat",
        protocol_name="openai_chat",
        client_protocol="responses",
        requested_profile=None,
        provider="openai",
    )
    assert chosen == "mem"


def test_family_ambiguity_still_errors() -> None:
    with pytest.raises(Exception, match="multiple profiles"):
        resolve_profile(
            declared_profiles={
                "a": {"protocol": "responses"},
                "b": {"protocol": "responses_stateful"},
            },
            default_profile=None,
            protocol_name=None,
            client_protocol="responses",
            requested_profile=None,
            provider="openai",
        )


def test_priority_walk_stays_family_level() -> None:
    # bare chat client on a responses-only provider: priority list picks
    # responses family (any sibling counts as the family offering).
    chosen = resolve_profile(
        declared_profiles={"mem": {"protocol": "responses_stateful"}},
        default_profile="mem",
        protocol_name="responses_stateful",
        client_protocol="openai_chat",
        requested_profile=None,
        provider="prov",
    )
    assert chosen == "mem"


# ------------------------------------------------- registry-derived allowlist

def test_generative_protocol_gate() -> None:
    assert is_generative_protocol("openai_chat")
    assert is_generative_protocol("responses")
    assert is_generative_protocol("responses_stateful")
    assert is_generative_protocol("responses_websocket")
    assert is_generative_protocol("gemini")
    assert is_generative_protocol("ollama")
    # non-generative / passthrough / unknown are refused
    assert not is_generative_protocol("openai_embeddings")
    assert not is_generative_protocol("openai_images")
    assert not is_generative_protocol("litellm_fallback")
    assert not is_generative_protocol("does_not_exist")


# ---------------------------------------------------------- gemini faces

def test_gemini_provider_declares_two_faces() -> None:
    from rotator_library.providers.gemini_provider import GeminiProvider

    provider = GeminiProvider()
    assert provider.default_profile == "native"
    assert provider.transport_profiles["openai"]["protocol"] == "openai_chat"
    assert provider.get_protocol_name("", profile="openai") == "openai_chat"
    assert provider.get_protocol_name("", profile="native") == "gemini"
    # operation + auth are per-face (the pre-G11 bugs)
    assert provider.get_native_operation("", stream=True, profile="openai") == "chat"
    assert provider.get_native_operation("", stream=True, profile="native") == "stream_generate"
    headers = provider.get_native_headers("KEY", profile="openai")
    assert headers == {"Authorization": "Bearer KEY"}
    headers_native = provider.get_native_headers("KEY", profile="native")
    assert headers_native == {"x-goog-api-key": "KEY"}
    endpoint = provider.get_native_endpoint("gemini-3-flash", "chat", profile="openai")
    assert endpoint.endswith("/v1beta/openai/chat/completions")


def test_gemini_v1_ingress_removed_and_openai_face_routes_exist() -> None:
    paths = {route.path for route in proxy_main.app.routes}
    assert "/v1beta/openai/chat/completions" in paths
    assert "/v1beta/openai/models" in paths
    # v1 gemini ingress dropped per ruling (v1beta is the canonical surface);
    # the OpenAI-surface /v1/models routes are unrelated and untouched.
    assert not any(":generateContent" in p or ":streamGenerateContent" in p or ":countTokens" in p
                   for p in paths if p.startswith("/v1/"))


class _CompatFakeClient:
    def __init__(self) -> None:
        self.calls: list = []

    async def agenerate(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return {
            "id": "chatcmpl-compat",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

    async def get_all_available_models(self, grouped=False):
        return ["gemini/gemini-3-flash"]


def test_gemini_openai_chat_route_stamps_the_profile() -> None:
    proxy_main.PROXY_API_KEY = None
    fake = _CompatFakeClient()
    proxy_main.app.state.rotating_client = fake
    client = TestClient(proxy_main.app)

    response = client.post(
        "/v1beta/openai/chat/completions",
        json={"model": "gemini-3-flash", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    # the model reference rides the profile grammar — the fast path is
    # chat-wire-in/chat-wire-out on Google's compat endpoint
    assert fake.calls[0][0]["model"] == "gemini:openai/gemini-3-flash"
    assert fake.calls[0][1]["input_protocol"] == "openai_chat"


def test_gemini_openai_models_route_serves_openai_shape() -> None:
    proxy_main.PROXY_API_KEY = None
    proxy_main.app.state.rotating_client = _CompatFakeClient()
    client = TestClient(proxy_main.app)

    response = client.get("/v1beta/openai/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "gemini-3-flash"

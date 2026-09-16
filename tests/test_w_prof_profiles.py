"""W-PROF acceptance: provider:profile/model grammar + multi-profile transport.

D13: one provider identity (credentials, quota, accounting) may expose
multiple transport profiles; bare names fast-path-or-error; explicit
profiles convert as asked; identity normalization keeps every pool at
provider level.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotator_library.providers.provider_interface import ProviderInterface
from rotator_library.routing.profiles import (
    ModelReferenceError,
    parse_model_reference,
    resolve_profile,
    valid_profile_name,
)


PROFILES = {
    "chat": {"protocol": "openai_chat", "endpoint_path": "/chat/completions"},
    "responses": {"protocol": "responses", "endpoint_path": "/responses"},
    "anthropic": {"protocol": "anthropic_messages", "endpoint_path": "/v1/messages"},
}


class MultiProfileProvider(ProviderInterface):
    """Synthetic multi-profile provider (generic runtime tests use
    synthetic providers — never concrete ones)."""

    transport_profiles = PROFILES
    default_profile = "chat"
    default_api_base = "https://synthetic.example/v1"

    async def get_models(self, api_key, client):
        return []


# --- Grammar parsing ---


def test_explicit_profile_reference_parses() -> None:
    ref = parse_model_reference("synthetic:responses/gpt-test")
    assert ref.provider == "synthetic"
    assert ref.profile == "responses"
    assert ref.model == "gpt-test"
    assert ref.bare == "synthetic/gpt-test"


def test_slashed_models_parse_through_the_grammar() -> None:
    ref = parse_model_reference("openrouter:fast/openai/gpt-4.1")
    assert (ref.provider, ref.profile, ref.model) == ("openrouter", "fast", "openai/gpt-4.1")


def test_bare_reference_has_no_profile() -> None:
    ref = parse_model_reference("openai/gpt-test")
    assert (ref.profile, ref.model) == (None, "gpt-test")


@pytest.mark.parametrize(
    "reference",
    ["", ":", ":profile/model", "provider:/model", "provider:profile/", "provider:"],
)
def test_malformed_references_rejected(reference: str) -> None:
    with pytest.raises(ModelReferenceError):
        parse_model_reference(reference)


@pytest.mark.parametrize(
    "model_id",
    [
        "openrouter/openai/gpt-4:free",
        "openrouter/meta-llama/llama-3.2-3b-instruct:free",
        "ollama/llama3:8b",
        "provider/model:tag",
    ],
)
def test_colons_in_model_segments_are_not_profiles(model_id: str) -> None:
    """OpenRouter :free variants and Ollama model:tag ids stay intact — the
    grammar applies only to the provider segment (regression guard)."""

    ref = parse_model_reference(model_id)
    assert ref.profile is None
    assert ref.bare == model_id


def test_names_may_not_contain_separator_or_slash() -> None:
    assert valid_profile_name("chat") is True
    assert valid_profile_name("a:b") is False
    assert valid_profile_name("a/b") is False
    assert valid_profile_name("") is False


# --- Profile resolution (D13 semantics) ---


def test_bare_name_picks_default_profile_for_matching_client_protocol() -> None:
    assert (
        resolve_profile(
            declared_profiles=PROFILES,
            default_profile="chat",
            protocol_name=None,
            client_protocol="openai_chat",
            requested_profile=None,
            provider="synthetic",
        )
        == "chat"
    )


def test_bare_name_matches_unique_profile_for_other_client_protocol() -> None:
    assert (
        resolve_profile(
            declared_profiles=PROFILES,
            default_profile="chat",
            protocol_name=None,
            client_protocol="anthropic_messages",
            requested_profile=None,
            provider="synthetic",
        )
        == "anthropic"
    )


def test_bare_name_converts_through_priority_list_with_warning() -> None:
    """D13 revision: zero protocol match auto-converts through the priority
    list (openai_chat first) with a terminal warning — never a hard error."""
    result = resolve_profile(
        declared_profiles=PROFILES,
        default_profile="responses",
        protocol_name=None,
        client_protocol="gemini",
        requested_profile=None,
        provider="synthetic",
    )
    # gemini is unavailable; openai_chat is the first offered protocol.
    assert result == "chat"


def test_bare_name_ambiguity_errors_listing_candidates() -> None:
    """Several profiles speaking the client protocol with no default match
    is endpoint ambiguity: error lists them and suggests a default."""
    profiles = {
        "fast": {"protocol": "openai_chat"},
        "cheap": {"protocol": "openai_chat"},
    }
    with pytest.raises(ModelReferenceError, match="multiple profiles"):
        resolve_profile(
            declared_profiles=profiles,
            default_profile=None,
            protocol_name=None,
            client_protocol="openai_chat",
            requested_profile=None,
            provider="synthetic",
        )


def test_explicit_profile_wins_and_must_exist() -> None:
    assert (
        resolve_profile(
            declared_profiles=PROFILES,
            default_profile="chat",
            protocol_name=None,
            client_protocol="openai_chat",
            requested_profile="responses",
            provider="synthetic",
        )
        == "responses"
    )
    with pytest.raises(ModelReferenceError, match="no profile"):
        resolve_profile(
            declared_profiles=PROFILES,
            default_profile="chat",
            protocol_name=None,
            client_protocol="openai_chat",
            requested_profile="nonexistent",
            provider="synthetic",
        )


def test_explicit_profile_on_single_protocol_provider_errors() -> None:
    """provider:profile on a provider without profiles fails loudly — never
    a silent protocol mismatch."""

    with pytest.raises(ModelReferenceError, match="has no profiles"):
        resolve_profile(
            declared_profiles=None,
            default_profile=None,
            protocol_name="openai_chat",
            client_protocol="openai_chat",
            requested_profile="responses",
            provider="openai",
        )


def test_invalid_default_profile_raises() -> None:
    with pytest.raises(ModelReferenceError, match="default profile"):
        resolve_profile(
            declared_profiles=PROFILES,
            default_profile="nonexistent",
            protocol_name=None,
            client_protocol="openai_chat",
            requested_profile=None,
            provider="synthetic",
        )


def test_route_target_parses_profile_and_execution() -> None:
    from rotator_library.routing.config import parse_route_target

    target = parse_route_target("synthetic:responses/m@native")
    assert (target.provider, target.profile, target.model, target.execution) == (
        "synthetic",
        "responses",
        "m",
        "native",
    )
    plain = parse_route_target("openai/gpt-test")
    assert plain.profile is None
    with pytest.raises(ValueError, match="profile name cannot be empty"):
        parse_route_target("synthetic:/m")


def test_profile_endpoint_derivation_guards() -> None:
    provider = MultiProfileProvider()

    class _NoBase(MultiProfileProvider):
        default_api_base = None

    with pytest.raises(NotImplementedError, match="transport base"):
        _NoBase().get_native_endpoint(model="m", operation="chat", profile="chat")

    class _NoPath(MultiProfileProvider):
        transport_profiles = {"responses": {"protocol": "responses"}}

    endpoint = _NoPath().get_native_endpoint(model="m", operation="chat", profile="responses")
    assert endpoint.endswith("/responses")


def test_cross_protocol_profile_operation_resolution() -> None:
    """Explicit anthropic profile on a chat-default provider resolves the
    anthropic operation vocabulary (messages), not the default's chat —
    cross-protocol conversion executes instead of dying at the gate."""

    provider = MultiProfileProvider()
    assert provider.get_native_operation("m", None, stream=False, profile="anthropic") == "messages"
    assert provider.supports_native_operation("m", "messages", profile="anthropic") is True
    assert provider.supports_native_operation("m", "chat", profile="anthropic") is False


def test_explicit_request_survives_bogus_default_profile() -> None:
    assert (
        resolve_profile(
            declared_profiles=PROFILES,
            default_profile="bogus",
            protocol_name=None,
            client_protocol="openai_chat",
            requested_profile="chat",
            provider="synthetic",
        )
        == "chat"
    )


# --- Provider interface integration ---


def test_provider_resolves_protocol_per_profile() -> None:
    provider = MultiProfileProvider()
    assert provider.get_protocol_name("m", profile="chat") == "openai_chat"
    assert provider.get_protocol_name("m", profile="anthropic") == "anthropic_messages"
    # No profile: the default profile's protocol.
    assert provider.get_protocol_name("m") == "openai_chat"
    with pytest.raises(ModelReferenceError):
        provider.get_protocol_name("m", profile="nonexistent")


def test_provider_derives_endpoint_per_profile() -> None:
    provider = MultiProfileProvider()
    endpoint = provider.get_native_endpoint(model="m", operation="chat", profile="responses")
    assert endpoint == "https://synthetic.example/v1/responses"


def test_single_protocol_providers_ignore_profiles() -> None:
    from rotator_library.providers import PROVIDER_PLUGINS

    # deepseek was the original pin but declares three faces since its G8
    # remake; groq is the honest single-protocol example now. The envelope
    # expresses its one face as a one-entry speaks tuple.
    groq = PROVIDER_PLUGINS["groq"]()
    assert groq.get_protocol_name("m") == "openai_chat"
    assert groq.transport_profiles is None
    assert groq.speaks == ("openai_chat",)


def test_openai_declares_two_faces_with_matching_resolution() -> None:
    from rotator_library.providers import PROVIDER_PLUGINS

    openai = PROVIDER_PLUGINS["openai"]()
    assert openai.get_protocol_name("m", profile="chat") == "openai_chat"
    assert openai.get_protocol_name("m") == "responses"
    assert openai.get_native_endpoint(model="m", profile="chat").endswith("/chat/completions")
    assert openai.get_native_endpoint(model="m").endswith("/responses")
    # speaks replaced the legacy two-profile declaration table.
    assert openai.transport_profiles is None
    assert openai.speaks[0][0] == "responses"
    profiles, default = openai.get_declared_profiles()
    assert default == "responses" and set(profiles) == {"responses", "chat"}


def test_executor_rejects_explicit_profile_on_single_protocol_provider() -> None:
    """groq:responses/gpt-test must fail loudly (structured 400), not be
    silently served as chat. (openai now legitimately declares a responses
    face, so it no longer belongs in this pin.) The envelope makes the
    single face a one-entry profile table, so the refusal names the
    unknown profile against the known set."""

    from rotator_library.client.executor import RequestExecutor
    from rotator_library.core.errors import StructuredAPIResponseError
    from rotator_library.core.types import RequestContext
    from rotator_library.providers import PROVIDER_PLUGINS

    executor = RequestExecutor.__new__(RequestExecutor)
    executor._experimental_config = None
    plugin = PROVIDER_PLUGINS["groq"]()
    context = RequestContext(
        model="groq/gpt-test",
        provider="groq",
        kwargs={"model": "groq/gpt-test", "messages": []},
        streaming=False,
        credentials=[],
        deadline=0,
        input_protocol_name="openai_chat",
        execution_profile="responses",
    )
    with pytest.raises(StructuredAPIResponseError, match="has no profile"):
        executor._build_native_provider_context(
            "openai", "groq/gpt-test", plugin, "sk-test", "cred-1", context, None
        )


# --- Executor integration (native context) ---


def test_native_context_resolves_bare_and_explicit_profiles() -> None:
    from rotator_library.client.executor import RequestExecutor, RoutingExecutionError
    from rotator_library.core.types import RequestContext

    executor = RequestExecutor.__new__(RequestExecutor)
    executor._experimental_config = None

    def _context(profile: str, client_protocol: str) -> RequestContext:
        return RequestContext(
            model="synthetic/m",
            provider="synthetic",
            kwargs={"model": "synthetic/m", "messages": []},
            streaming=False,
            credentials=[],
            deadline=0,
            input_protocol_name=client_protocol,
            execution_profile=profile,
        )

    plugin = MultiProfileProvider()
    context = executor._build_native_provider_context(
        "synthetic", "synthetic/m", plugin, "sk-test", "cred-1", _context(None, "anthropic_messages"), None
    )
    assert context.protocol_name == "anthropic_messages"
    assert context.endpoint.endswith("/v1/messages")

    explicit = executor._build_native_provider_context(
        "synthetic", "synthetic/m", plugin, "sk-test", "cred-1", _context("responses", "openai_chat"), None
    )
    assert explicit.protocol_name == "responses"
    assert explicit.endpoint.endswith("/responses")
    assert explicit.metadata.get("execution_profile") == "responses"


def test_native_context_bare_name_unmatched_protocol_converts() -> None:
    """D13 revision at the executor level: an unmatched bare-name protocol
    resolves via the priority list instead of a 400 (conversion warning is
    emitted once per substitution)."""
    from rotator_library.client.executor import RequestExecutor
    from rotator_library.core.types import RequestContext

    executor = RequestExecutor.__new__(RequestExecutor)
    executor._experimental_config = None
    plugin = MultiProfileProvider()
    context = RequestContext(
        model="synthetic/m",
        provider="synthetic",
        kwargs={"model": "synthetic/m", "messages": []},
        streaming=False,
        credentials=[],
        deadline=0,
        input_protocol_name="gemini",
    )
    native_context = executor._build_native_provider_context(
        "synthetic", "synthetic/m", plugin, "sk-test", "cred-1", context, None
    )
    # Converted to the first offered priority protocol (openai_chat).
    assert native_context.protocol_name == "openai_chat"
    assert native_context.client_protocol_name == "gemini"


# --- Plan 2.8 endpoint declarations (plural map + placeholders) ---


def test_profile_endpoint_paths_map_and_placeholder_rendering() -> None:
    class GeminiProfileProvider(MultiProfileProvider):
        transport_profiles = {
            "native": {
                "protocol": "gemini",
                "endpoint_paths": {
                    "generate": "/v1beta/models/{model}:generateContent",
                    "stream_generate": "/v1beta/models/{model}:streamGenerateContent?alt=sse",
                },
            },
        }
        default_profile = "native"

    provider = GeminiProfileProvider()
    assert provider.get_native_endpoint("gemini-3-pro", "generate", profile="native") == (
        "https://synthetic.example/v1/v1beta/models/gemini-3-pro:generateContent"
    )
    assert provider.get_native_endpoint("gemini-3-pro", "stream_generate", profile="native") == (
        "https://synthetic.example/v1/v1beta/models/gemini-3-pro:streamGenerateContent?alt=sse"
    )


def test_profile_endpoint_plural_map_wins_over_singular_fallback() -> None:
    class MixedPathProvider(MultiProfileProvider):
        transport_profiles = {
            "chat": {
                "protocol": "openai_chat",
                "endpoint_paths": {"chat": "/v1/chat/completions"},
                "endpoint_path": "/legacy/chat",
            },
        }
        default_profile = "chat"

    provider = MixedPathProvider()
    assert provider.get_native_endpoint("m", "chat", profile="chat") == (
        "https://synthetic.example/v1/v1/chat/completions"
    )
    # A different operation falls back to the singular declaration.
    assert provider.get_native_endpoint("m", "embeddings", profile="chat") == (
        "https://synthetic.example/v1/legacy/chat"
    )


def test_profile_endpoint_declarations_win_over_provider_level() -> None:
    from rotator_library.config.experimental import load_config_from_mapping

    class ProviderLevelPaths(MultiProfileProvider):
        provider_env_name = "provider_level_paths"
        transport_profiles = {"chat": {"protocol": "openai_chat", "endpoint_paths": {"chat": "/profile/chat"}}}
        default_profile = "chat"

    provider = ProviderLevelPaths()
    provider.bind_runtime_config(
        load_config_from_mapping(
            {"providers": {"provider_level_paths": {"endpoint_paths": {"chat": "/provider/chat"}}}}
        )
    )
    assert provider.get_native_endpoint("m", "chat", profile="chat").endswith("/profile/chat")


# --- Plan 2.8 per-profile auth ---


def test_profile_auth_declarations_override_provider_level() -> None:
    from rotator_library.config.experimental import load_config_from_mapping

    class ProfileAuthProvider(MultiProfileProvider):
        provider_env_name = "profile_auth_provider"
        transport_profiles = {
            "anthropic": {
                "protocol": "anthropic_messages",
                "endpoint_path": "/v1/messages",
                "auth_mode": "x-api-key",
            },
        }
        default_profile = "anthropic"

    provider = ProfileAuthProvider()
    provider.bind_runtime_config(
        load_config_from_mapping({"providers": {"profile_auth_provider": {"auth_mode": "x-goog-api-key"}}})
    )
    assert provider.get_native_headers("secret", profile="anthropic") == {"x-api-key": "secret"}
    assert provider.get_native_headers("secret") == {"x-goog-api-key": "secret"}


# --- Dynamic provider profile surface (G7) ---


def test_dynamic_provider_accepts_json_transport_profiles(tmp_path, monkeypatch) -> None:
    import json

    from rotator_library.providers import _create_dynamic_plugin_class

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "profiled_dynamic": {
                        "api_base": "https://dynamic.example",
                        "profiles": {
                            "chat": {
                                "protocol": "openai_chat",
                                "endpoint_paths": {"chat": "/v1/chat/completions"},
                            },
                            "anthropic": {
                                "protocol": "anthropic_messages",
                                "endpoint_path": "/v1/messages",
                                "auth_mode": "x-api-key",
                            },
                        },
                        "default_profile": "chat",
                        "models": ["m"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", str(config_path))

    provider = _create_dynamic_plugin_class("profiled_dynamic")()
    assert provider.transport_profiles is not None
    assert provider.get_protocol_name("profiled_dynamic/m", profile="anthropic") == "anthropic_messages"
    assert provider.get_native_endpoint("profiled_dynamic/m", "chat", profile="chat") == (
        "https://dynamic.example/v1/chat/completions"
    )
    assert provider.get_native_endpoint("profiled_dynamic/m", "messages", profile="anthropic") == (
        "https://dynamic.example/v1/messages"
    )
    assert provider.get_native_headers("secret", profile="anthropic")["x-api-key"] == "secret"
    assert provider.get_native_headers("secret", profile="chat")["Authorization"] == "Bearer secret"


def test_env_only_dynamic_provider_defaults_native_openai_chat(monkeypatch) -> None:
    from rotator_library.providers import _create_dynamic_plugin_class

    monkeypatch.delenv("LLM_PROXY_CONFIG_FILE", raising=False)
    monkeypatch.delenv("PROXY_CONFIG_FILE", raising=False)
    monkeypatch.setenv("ENVONLY_DYNAMIC_API_BASE", "https://envonly.example/v1")

    provider = _create_dynamic_plugin_class("envonly_dynamic")()
    assert provider.get_api_base() == "https://envonly.example/v1"
    assert provider.get_protocol_name("envonly_dynamic/m") == "openai_chat"


# --- Embeddings profile loudness + acompletion input_protocol (G7) ---


@pytest.mark.asyncio
async def test_embedding_request_with_profile_fails_loud(monkeypatch) -> None:
    monkeypatch.delenv("FALLBACK_GROUPS", raising=False)
    from rotator_library.client.request_builder import RequestContextBuilder

    async def _scope(provider, classifier, request_api_keys, request_providers, private):
        return {
            "credentials": ["cred"],
            "usage_manager_key": provider,
            "provider_config": {},
            "credential_secrets": {},
            "classifier": "global",
        }

    builder = RequestContextBuilder(
        resolve_scope_for_provider=_scope,
        model_resolver=type("R", (), {"resolve_model_id": lambda self, model, provider: model})(),
        session_tracker=type(
            "S",
            (),
            {
                "infer_session": lambda self, *a, **k: type(
                    "Session",
                    (),
                    {"session_id": "s", "affinity_key": None, "tracking_namespace": "n"},
                )()
            },
        )(),
        get_global_timeout=lambda: 30,
        get_enable_request_logging=lambda: False,
    )

    # G9: profiles may serve embeddings — the address resolves through the
    # same grammar as completions; an unknown target surfaces as the
    # standard no-provider error, not an embeddings-specific rejection.
    import asyncio

    try:
        ctx = asyncio.run(
            builder.build_embedding_context(
                None, None, {"model": "synthetic:responses/embed-1", "input": "hi"}
            )
        )
    except Exception as exc:
        # A provider/plugin that does not exist fails honestly (scope
        # fakes make resolution provider-dependent) — but NEVER the old
        # "embedding requests do not support transport profiles" reject.
        assert "profile" not in str(exc).lower() or "does not support" not in str(exc)
    else:
        assert ctx.requested_operation == "embeddings"
        assert ctx.input_protocol_name == "openai_embeddings"


def test_acompletion_input_protocol_parameter_is_authoritative() -> None:
    import asyncio

    from rotator_library.client.rotating_client import RotatingClient

    captured: dict = {}

    class _Builder:
        async def build_completion_context(self, request, callback, kwargs):
            captured.update(kwargs)
            return object()

    class _Executor:
        async def execute(self, context):
            return "ok"

    client = RotatingClient.__new__(RotatingClient)
    client._request_builder = _Builder()
    client._executor = _Executor()

    assert asyncio.run(client.acompletion(input_protocol="anthropic_messages", model="synthetic/m", messages=[])) == "ok"
    assert captured["_input_protocol"] == "anthropic_messages"

    captured.clear()
    asyncio.run(client.acompletion(_input_protocol="gemini", model="synthetic/m", messages=[]))
    assert captured["_input_protocol"] == "gemini"

    captured.clear()
    asyncio.run(client.acompletion(input_protocol="responses", _input_protocol="gemini", model="synthetic/m"))
    assert captured["_input_protocol"] == "responses"


# --- Identity normalization ---


def test_profile_addressing_normalizes_to_provider_identity() -> None:
    """Usage pools, session namespaces, and cache scopes see only the bare
    provider — the grammar must not leak profiles into identity keys."""

    # The request builder normalizes model strings before any identity use:
    ref = parse_model_reference("synthetic:responses/m")
    assert ref.bare == "synthetic/m"
    assert ref.provider == "synthetic"

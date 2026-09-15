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


def test_bare_name_errors_when_no_profile_speaks_client_protocol() -> None:
    with pytest.raises(ModelReferenceError, match="no endpoint"):
        resolve_profile(
            declared_profiles=PROFILES,
            default_profile="chat",
            protocol_name=None,
            client_protocol="gemini",
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

    openai = PROVIDER_PLUGINS["openai"]()
    assert openai.get_protocol_name("m") == "openai_chat"


def test_executor_rejects_explicit_profile_on_single_protocol_provider() -> None:
    """openai:responses/gpt-test must fail loudly (structured 400), not be
    silently served as chat."""

    from rotator_library.client.executor import RequestExecutor
    from rotator_library.core.errors import StructuredAPIResponseError
    from rotator_library.core.types import RequestContext
    from rotator_library.providers import PROVIDER_PLUGINS

    executor = RequestExecutor.__new__(RequestExecutor)
    executor._experimental_config = None
    plugin = PROVIDER_PLUGINS["openai"]()
    context = RequestContext(
        model="openai/gpt-test",
        provider="openai",
        kwargs={"model": "openai/gpt-test", "messages": []},
        streaming=False,
        credentials=[],
        deadline=0,
        input_protocol_name="openai_chat",
        execution_profile="responses",
    )
    with pytest.raises(StructuredAPIResponseError, match="has no profiles"):
        executor._build_native_provider_context(
            "openai", "openai/gpt-test", plugin, "sk-test", "cred-1", context, None
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


def test_native_context_bare_name_unmatched_protocol_errors() -> None:
    from rotator_library.client.executor import RequestExecutor
    from rotator_library.core.errors import StructuredAPIResponseError
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
    with pytest.raises(StructuredAPIResponseError, match="no endpoint"):
        executor._build_native_provider_context(
            "synthetic", "synthetic/m", plugin, "sk-test", "cred-1", context, None
        )


# --- Identity normalization ---


def test_profile_addressing_normalizes_to_provider_identity() -> None:
    """Usage pools, session namespaces, and cache scopes see only the bare
    provider — the grammar must not leak profiles into identity keys."""

    from rotator_library.routing.profiles import split_profile_from_provider

    assert split_profile_from_provider("synthetic") == ("synthetic", None)
    assert split_profile_from_provider("synthetic:responses") == ("synthetic", "responses")
    # The request builder normalizes model strings before any identity use:
    ref = parse_model_reference("synthetic:responses/m")
    assert ref.bare == "synthetic/m"
    assert ref.provider == "synthetic"

"""G11 verification-fix pins: family-aware wire dispatch, provider-owned
continuation, never-fail row construction, compat model normalization."""

import pytest

from rotator_library.protocols.canonical import family_wire_name, format_stop_reason, is_same_protocol
from rotator_library.protocols.types import ProtocolContext
from rotator_library.protocols.validation import validate_generative_request
from rotator_library.protocols import get_protocol


# --------------------------------------------------- family wire dispatch

def test_family_wire_name_mapping() -> None:
    assert family_wire_name("responses") == "responses"
    assert family_wire_name("responses_stateful") == "responses"
    assert family_wire_name("responses_websocket") == "responses"
    assert family_wire_name("openai_chat") == "openai_chat"
    assert family_wire_name("not_a_protocol") == "not_a_protocol"
    assert family_wire_name("") == ""


def test_sibling_target_is_same_wire_not_conversion() -> None:
    # THE verifier P0: a responses client on a responses_stateful endpoint
    # is SAME-WIRE traffic — raw replay preserved, no cross-protocol path.
    ctx = ProtocolContext(source_protocol="responses", target_protocol="responses_stateful")
    assert is_same_protocol(ctx, "responses_stateful") is True
    assert is_same_protocol(ctx, "openai_chat") is False


def test_sibling_target_keeps_stateful_fields_and_content() -> None:
    # validate_generative_request must NOT hard-reject previous_response_id /
    # conversation on a sibling target (the exact fields that DEFINE the
    # stateful variant), and must not warn-drop ordinary content.
    adapter = get_protocol("responses")
    raw = {
        "model": "prov/gpt-x",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ],
        "previous_response_id": "resp_parent",
    }
    ctx = ProtocolContext(source_protocol="responses", target_protocol="responses_stateful")
    unified = adapter.parse_request(raw, ctx)
    target = get_protocol("responses_stateful")
    # target sibling build: stateful fields survive, content survives
    validate_generative_request(unified, "responses_stateful", ProtocolContext(source_protocol="responses", target_protocol="responses_stateful"))
    built = target.build_request(unified, ProtocolContext(source_protocol="responses", target_protocol="responses_stateful"))
    assert built["previous_response_id"] == "resp_parent"


def test_sibling_target_formats_status_and_reason() -> None:
    # Stop-reason table lookup is family-level: a sibling target renders
    # a real status value instead of None.
    assert format_stop_reason("stop", "responses_stateful") is not None
    assert format_stop_reason("stop", "responses_websocket") is not None


def test_native_operation_family_aware() -> None:
    from rotator_library.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.protocol_name = "responses_stateful"
    # family dispatch via the base method: a sibling-protocol provider maps
    # to the responses operation, not the chat fallthrough.
    assert provider.get_native_operation("m", stream=False) == "responses"


# ------------------------------------------- provider-owned continuation

@pytest.mark.asyncio
async def test_provider_owned_row_rides_the_id_not_local_replay() -> None:
    """THE second-turn P0: a locally-mirrored provider-minted row continues
    on the provider's own chain (returns None → id passthrough), instead of
    replaying the stored suffix and severing the provider's prefix."""

    import time as _time

    from rotator_library.providers import PROVIDER_PLUGINS
    from rotator_library.responses.service import ResponsesService
    from rotator_library.responses.store import InMemoryResponsesStore
    from rotator_library.responses.types import StoredResponse

    class _ResponsesPlugin:
        protocol_name = "responses"
        transport_profiles = None
        default_profile = None

        def get_protocol_name(self, model: str = "", profile=None) -> str:
            return "responses"

    PROVIDER_PLUGINS["respprovider"] = _ResponsesPlugin
    try:
        service = ResponsesService.__new__(ResponsesService)
        service.store = InMemoryResponsesStore()

        row = StoredResponse(
            id="resp_prov_2",
            created_at=_time.time(),
            model="respprovider/gpt-x",
            status="completed",
            request={"model": "respprovider/gpt-x", "input": "hi", "previous_response_id": "resp_prov_1"},
            response={"id": "resp_prov_2"},
            input_items=[],
            output_items=[],
            metadata={"provider": "respprovider", "provider_owned": True},
            scope_key="public",
        )
        await service.store.save(row)

        parent = await service._load_previous_response(
            "resp_prov_2",
            None,
            expected_scope_key="public",
            provider_passthrough=True,
            raw_request_dict={"model": "respprovider/gpt-x", "previous_response_id": "resp_prov_2"},
        )
        assert parent is None  # rides the provider's chain
    finally:
        PROVIDER_PLUGINS.pop("respprovider", None)


@pytest.mark.asyncio
async def test_proxy_owned_row_still_replays_locally() -> None:
    import time as _time

    from rotator_library.responses.service import ResponsesService
    from rotator_library.responses.store import InMemoryResponsesStore
    from rotator_library.responses.types import StoredResponse

    service = ResponsesService.__new__(ResponsesService)
    service.store = InMemoryResponsesStore()

    row = StoredResponse(
        id="resp_local_1",
        created_at=_time.time(),
        model="respprovider/gpt-x",
        status="completed",
        request={"model": "respprovider/gpt-x", "input": "hi"},
        response={"id": "resp_local_1"},
        input_items=[{"type": "message", "role": "user", "content": "hi"}],
        output_items=[],
        metadata={"provider_owned": False},
        scope_key="public",
    )
    await service.store.save(row)

    parent = await service._load_previous_response(
        "resp_local_1",
        None,
        expected_scope_key="public",
        provider_passthrough=True,
        raw_request_dict={"model": "respprovider/gpt-x"},
    )
    assert parent is not None and parent.id == "resp_local_1"


@pytest.mark.asyncio
async def test_provider_owned_row_replays_when_target_switched() -> None:
    """Provider switched (fallback): the foreign provider can never resolve
    the old provider's chain — local replay is the honest degradation."""

    import time as _time

    from rotator_library.responses.service import ResponsesService
    from rotator_library.responses.store import InMemoryResponsesStore
    from rotator_library.responses.types import StoredResponse

    service = ResponsesService.__new__(ResponsesService)
    service.store = InMemoryResponsesStore()

    row = StoredResponse(
        id="resp_prov_2",
        created_at=_time.time(),
        model="respprovider/gpt-x",
        status="completed",
        request={"model": "respprovider/gpt-x", "input": "hi"},
        response={"id": "resp_prov_2"},
        input_items=[],
        output_items=[],
        metadata={"provider": "respprovider", "provider_owned": True},
        scope_key="public",
    )
    await service.store.save(row)

    # target is a chat-family provider → not eligible → local replay
    parent = await service._load_previous_response(
        "resp_prov_2",
        None,
        expected_scope_key="public",
        provider_passthrough=True,
        raw_request_dict={"model": "gemini/gemini-3-flash"},
    )
    assert parent is not None and parent.id == "resp_prov_2"


@pytest.mark.asyncio
async def test_provider_owned_row_replays_when_same_family_other_provider() -> None:
    """A provider-owned row is only safe to ride with the SAME provider.

    Both targets speak the Responses family, but the row was minted on
    provider A and the request now targets provider B (fallback): B can never
    resolve A's hidden chain, so local replay is the honest outcome.
    """

    import time as _time

    from rotator_library.providers import PROVIDER_PLUGINS
    from rotator_library.responses.service import ResponsesService
    from rotator_library.responses.store import InMemoryResponsesStore
    from rotator_library.responses.types import StoredResponse

    class _ResponsesPlugin:
        protocol_name = "responses"
        transport_profiles = None
        default_profile = None

        def get_protocol_name(self, model: str = "", profile=None) -> str:
            return "responses"

    PROVIDER_PLUGINS["respa"] = _ResponsesPlugin
    PROVIDER_PLUGINS["respb"] = _ResponsesPlugin
    try:
        service = ResponsesService.__new__(ResponsesService)
        service.store = InMemoryResponsesStore()

        row = StoredResponse(
            id="resp_a",
            created_at=_time.time(),
            model="respa/gpt-x",
            status="completed",
            request={"model": "respa/gpt-x"},
            response={"id": "resp_a"},
            input_items=[],
            output_items=[],
            metadata={"provider": "respa", "provider_owned": True},
            scope_key="public",
        )
        await service.store.save(row)

        parent = await service._load_previous_response(
            "resp_a",
            None,
            expected_scope_key="public",
            provider_passthrough=True,
            raw_request_dict={"model": "respb/gpt-x", "previous_response_id": "resp_a"},
        )
        assert parent is not None and parent.id == "resp_a"
    finally:
        PROVIDER_PLUGINS.pop("respa", None)
        PROVIDER_PLUGINS.pop("respb", None)


# ------------------------------------------------- never-fail row build

@pytest.mark.asyncio
async def test_idless_payload_never_kills_the_answer() -> None:
    from rotator_library.responses.service import ResponsesService
    from rotator_library.responses.store import InMemoryResponsesStore

    service = ResponsesService.__new__(ResponsesService)
    service.store = InMemoryResponsesStore()

    saved = await service._safe_store(
        {"model": "prov/x"},
        {"status": "completed", "output": []},  # NO id — the old KeyError path
        None,
        None,
        None,
        "responses_store_response",
    )
    assert saved is False  # logged, contained, answer untouched


# ------------------------------------------------ compat model routing

def test_gemini_openai_model_normalization() -> None:
    from proxy_app.main import _gemini_openai_model

    assert _gemini_openai_model("gemini-3-flash") == "gemini:openai/gemini-3-flash"
    assert _gemini_openai_model("gemini/gemini-3-flash") == "gemini:openai/gemini-3-flash"
    assert _gemini_openai_model("models/gemini-3-flash") == "gemini:openai/gemini-3-flash"
    assert _gemini_openai_model("google/gemini-3-flash") == "gemini:openai/gemini-3-flash"
    # client-supplied profile addressing cannot rebind the face
    assert _gemini_openai_model("gemini:native/gemini-3-flash") == "gemini:openai/gemini-3-flash"
    assert _gemini_openai_model("") == "gemini:openai/"

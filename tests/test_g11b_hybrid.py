"""G11 Phase B pins: continuation hybrid (provider passthrough, provenance)."""

import pytest

from rotator_library.responses.service import _provider_continuation_eligible
from rotator_library.responses.types import StoredResponse


# ------------------------------------------------- eligibility resolution

class _ResponsesFamilyPlugin:
    # G11 verify-fix: eligibility instantiates plugin CLASSES (the registry
    # convention) — the fake must be a class, not a pre-built instance.
    protocol_name = "responses"

    def get_protocol_name(self, model: str = "", profile=None) -> str:
        return "responses"


def test_eligibility_requires_responses_family_target(monkeypatch) -> None:
    from rotator_library.providers import PROVIDER_PLUGINS

    monkeypatch.setitem(PROVIDER_PLUGINS, "respprovider", _ResponsesFamilyPlugin)
    assert _provider_continuation_eligible({"model": "respprovider/gpt-x", "previous_response_id": "resp_foreign"}) is True


def test_eligibility_false_for_chat_family(monkeypatch) -> None:
    # The gemini provider speaks the gemini family: not eligible.
    assert _provider_continuation_eligible({"model": "gemini/gemini-3-flash"}) is False


def test_eligibility_false_for_unknown_provider() -> None:
    assert _provider_continuation_eligible({"model": "nonexistent_provider/model-x"}) is False


def test_eligibility_false_without_model() -> None:
    assert _provider_continuation_eligible({"previous_response_id": "resp_x"}) is False


# ---------------------------------------------------- passthrough behavior

@pytest.mark.asyncio
async def test_public_miss_on_responses_target_passes_through_instead_of_404(tmp_path) -> None:
    """The confirmed defect: same-dialect native provider continuations
    always lost provider state (encrypted reasoning chains, cache keys) —
    now a public-scope miss on a responses-family target lets the id ride
    through verbatim."""

    from rotator_library.responses.service import ResponsesService
    from rotator_library.responses.store import InMemoryResponsesStore
    from rotator_library.responses.types import ResponsesStoreSettings

    service = ResponsesService.__new__(ResponsesService)
    service.store = InMemoryResponsesStore()
    service.store_settings = ResponsesStoreSettings()

    parent = await service._load_previous_response(
        "resp_foreign_id",
        None,
        expected_scope_key="public",
        provider_passthrough=True,
    )
    assert parent is None  # no local lineage — the provider owns the chain


@pytest.mark.asyncio
async def test_public_miss_without_eligibility_still_404s(tmp_path) -> None:
    from rotator_library.responses.service import ResponsesService, ResponsesServiceError
    from rotator_library.responses.store import InMemoryResponsesStore
    from rotator_library.responses.types import ResponsesStoreSettings

    service = ResponsesService.__new__(ResponsesService)
    service.store = InMemoryResponsesStore()
    service.store_settings = ResponsesStoreSettings()

    with pytest.raises(ResponsesServiceError) as raised:
        await service._load_previous_response(
            "resp_unknown",
            None,
            expected_scope_key="public",
            provider_passthrough=False,
        )
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_scoped_miss_never_passes_through() -> None:
    """The capability gate stays local: scoped misses 404 regardless of
    provider eligibility (anti cross-tenant injection)."""

    from rotator_library.responses.service import ResponsesService, ResponsesServiceError
    from rotator_library.responses.store import InMemoryResponsesStore
    from rotator_library.responses.types import ResponsesStoreSettings

    service = ResponsesService.__new__(ResponsesService)
    service.store = InMemoryResponsesStore()
    service.store_settings = ResponsesStoreSettings()

    with pytest.raises(ResponsesServiceError) as raised:
        await service._load_previous_response(
            "resp_scoped",
            None,
            expected_scope_key="scope:classified:prov",
            access_token="token-x",
            provider_passthrough=True,
        )
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_local_parent_still_wins_over_passthrough() -> None:
    """Local store hit is the first branch: proxy-owned chains replay
    locally even when the provider could also own it."""

    from rotator_library.responses.service import ResponsesService
    from rotator_library.responses.store import InMemoryResponsesStore
    from rotator_library.responses.types import ResponsesStoreSettings

    service = ResponsesService.__new__(ResponsesService)
    settings = ResponsesStoreSettings()
    service.store = InMemoryResponsesStore()
    service.store_settings = settings

    import time as _time

    stored = StoredResponse(
        id="resp_local",
        created_at=_time.time(),
        model="respprovider/gpt-x",
        status="completed",
        request={"model": "respprovider/gpt-x", "input": []},
        response={"id": "resp_local"},
        input_items=[{"type": "message", "role": "user", "content": "hi"}],
        output_items=[],
        metadata={},
        scope_key="public",
    )
    await service.store.save(stored)

    parent = await service._load_previous_response(
        "resp_local",
        None,
        expected_scope_key="public",
        provider_passthrough=True,
    )
    assert parent is not None and parent.id == "resp_local"


# ----------------------------------------------------------- provenance

def test_stored_response_carries_provider_provenance() -> None:
    from rotator_library.responses.service import ResponsesService
    from rotator_library.responses.types import ResponsesStoreSettings

    service = ResponsesService.__new__(ResponsesService)
    service.store_settings = ResponsesStoreSettings()
    service.request_scope_key = lambda raw: "public"  # type: ignore[method-assign]

    stored = service._stored_response(
        {"model": "respprovider/gpt-x", "input": "hi"},
        {"id": "resp_1", "model": "gpt-x", "created_at": 1.0, "output": []},
        None,
        session_info={"provider": "respprovider", "scope_key": "public"},
    )
    assert stored.metadata["provider"] == "respprovider"

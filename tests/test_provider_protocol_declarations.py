from __future__ import annotations

from rotator_library.field_cache import FieldCacheRule
from rotator_library.providers.provider_interface import ProviderInterface


class DeclarationProvider(ProviderInterface):
    protocol_name = "openai_chat"
    adapter_names = ("model_override", "suppress_developer_role")
    field_cache_rules = (
        FieldCacheRule(name="reasoning_content", source="response", path="choices.*.message.reasoning_content"),
    )

    async def get_models(self, api_key, client):
        return []


def test_provider_interface_defaults_are_noop_for_protocol_stack() -> None:
    provider = DeclarationProvider()

    assert provider.get_protocol_name("model") == "openai_chat"
    assert provider.get_adapter_names("model") == ("model_override", "suppress_developer_role")
    assert provider.get_adapter_config("model") == {}
    assert provider.get_field_cache_rules("model")[0].name == "reasoning_content"


def test_provider_interface_methods_can_be_model_specific() -> None:
    class ModelSpecificProvider(DeclarationProvider):
        def get_protocol_name(self, model: str = ""):
            return "responses" if "response" in model else super().get_protocol_name(model)

        def get_adapter_config(self, model: str = ""):
            return {"model_override": {"model": f"native/{model}"}}

    provider = ModelSpecificProvider()

    assert provider.get_protocol_name("response-model") == "responses"
    assert provider.get_protocol_name("chat-model") == "openai_chat"
    assert provider.get_adapter_config("chat-model") == {"model_override": {"model": "native/chat-model"}}

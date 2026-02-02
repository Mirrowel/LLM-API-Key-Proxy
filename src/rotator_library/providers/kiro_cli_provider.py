import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm
from litellm.exceptions import RateLimitError

from .provider_interface import ProviderInterface
from .kiro_auth_base import AuthType, KiroAuthManager
from .utilities.kiro_converters import (
    build_kiro_payload,
    convert_openai_messages_to_unified,
    convert_openai_tools_to_unified,
)
from .utilities.kiro_http_client import KiroHttpClient
from .utilities.kiro_streaming import (
    collect_stream_response,
    stream_kiro_to_openai_chunks,
)
from .utilities.kiro_utils import generate_conversation_id, get_kiro_headers
from ..model_definitions import ModelDefinitions


lib_logger = logging.getLogger("rotator_library")


class KiroCliProvider(ProviderInterface):
    skip_cost_calculation = True
    provider_env_name: str = "kiro_cli"

    def __init__(self) -> None:
        self.model_definitions = ModelDefinitions()
        self._auth_managers: Dict[str, KiroAuthManager] = {}

    def _get_auth_manager(self, credential_identifier: str) -> KiroAuthManager:
        if credential_identifier not in self._auth_managers:
            region = os.getenv("KIRO_REGION", "us-east-1")
            refresh_token = os.getenv("KIRO_REFRESH_TOKEN") or os.getenv("REFRESH_TOKEN", "")
            profile_arn = os.getenv("PROFILE_ARN") or os.getenv("KIRO_PROFILE_ARN")

            # Determine credential type based on prefix
            if credential_identifier.startswith("env:token:"):
                # Direct refresh token from environment variable
                if not refresh_token:
                    raise ValueError(
                        "KIRO_REFRESH_TOKEN or REFRESH_TOKEN environment variable is required"
                    )
                auth_manager = KiroAuthManager(
                    refresh_token=refresh_token,
                    profile_arn=profile_arn,
                    region=region,
                )
                lib_logger.debug("Created KiroAuthManager with env refresh token")
            elif credential_identifier.startswith("json:"):
                # JSON credentials file
                json_path = credential_identifier[5:]  # Remove "json:" prefix
                auth_manager = KiroAuthManager(
                    refresh_token=refresh_token or None,
                    profile_arn=profile_arn,
                    region=region,
                    json_creds_file=json_path,
                )
                lib_logger.debug(f"Created KiroAuthManager with JSON file: {json_path}")
            else:
                # SQLite database
                auth_manager = KiroAuthManager(
                    refresh_token=refresh_token or None,
                    profile_arn=profile_arn,
                    region=region,
                    sqlite_db=credential_identifier,
                )
                lib_logger.debug(f"Created KiroAuthManager with SQLite DB: {credential_identifier}")

            self._auth_managers[credential_identifier] = auth_manager
        return self._auth_managers[credential_identifier]

    def _resolve_model_id(self, model: str) -> str:
        model_name = model.split("/")[-1] if "/" in model else model
        model_id = self.model_definitions.get_model_id("kiro_cli", model_name)
        return model_id or model_name

    async def initialize_token(self, credential_identifier: str) -> None:
        auth_manager = self._get_auth_manager(credential_identifier)
        await auth_manager.get_access_token()

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        auth_manager = self._get_auth_manager(credential_identifier)
        token = await auth_manager.get_access_token()
        return {"Authorization": f"Bearer {token}"}

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        models: List[str] = []
        env_models = self.model_definitions.get_all_provider_models("kiro_cli")
        env_ids = set()
        if env_models:
            for model in env_models:
                model_name = model.split("/")[-1] if "/" in model else model
                model_id = self.model_definitions.get_model_id("kiro_cli", model_name)
                if model_id:
                    env_ids.add(model_id)
                models.append(model)

        if os.getenv("KIRO_CLI_ENABLE_DYNAMIC_MODELS", "true").lower() in (
            "true",
            "1",
            "yes",
        ):
            try:
                auth_manager = self._get_auth_manager(api_key)
                token = await auth_manager.get_access_token()
                headers = get_kiro_headers(auth_manager, token)
                params = {"origin": "AI_EDITOR"}
                if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
                    params["profileArn"] = auth_manager.profile_arn

                url = f"{auth_manager.q_host}/ListAvailableModels"
                response = await client.get(url, headers=headers, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    for model_info in data.get("models", []):
                        model_id = model_info.get("modelId") or model_info.get("model_id")
                        if model_id and model_id not in env_ids:
                            models.append(f"kiro_cli/{model_id}")
                            env_ids.add(model_id)
            except Exception as exc:
                lib_logger.debug(f"Dynamic model discovery failed for kiro_cli: {exc}")

        return models

    def has_custom_logic(self) -> bool:
        return True

    async def _build_payload(
        self, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        system_prompt, unified_messages = convert_openai_messages_to_unified(messages)
        unified_tools = convert_openai_tools_to_unified(tools)
        conversation_id = generate_conversation_id(messages)
        model_id = self._resolve_model_id(model)
        return build_kiro_payload(
            messages=unified_messages,
            system_prompt=system_prompt,
            model_id=model_id,
            tools=unified_tools,
            conversation_id=conversation_id,
            profile_arn="",
            inject_thinking=True,
        ).payload

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        model = kwargs["model"]
        credential_path = kwargs.pop("credential_identifier")
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")
        stream = kwargs.get("stream", False)

        auth_manager = self._get_auth_manager(credential_path)
        profile_arn = ""
        if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
            profile_arn = auth_manager.profile_arn

        payload = await self._build_payload(model, messages, tools)
        if profile_arn:
            payload["profileArn"] = profile_arn

        url = f"{auth_manager.api_host}/generateAssistantResponse"

        if stream:
            http_client = KiroHttpClient(auth_manager, shared_client=None)
        else:
            http_client = KiroHttpClient(auth_manager, shared_client=client)

        response = await http_client.request_with_retry(
            "POST", url, payload, stream=True
        )
        if response.status_code != 200:
            error_body = await response.aread()
            if response.status_code == 429:
                raise RateLimitError(
                    message=f"Kiro rate limit exceeded: {error_body!r}",
                    llm_provider="kiro_cli",
                    model=model,
                    response=response,
                )
            raise httpx.HTTPStatusError(
                f"Kiro API error: {response.status_code} {error_body!r}",
                request=response.request,
                response=response,
            )

        if stream:
            async def stream_handler():
                try:
                    async for chunk in stream_kiro_to_openai_chunks(
                        response, model
                    ):
                        yield litellm.ModelResponse(**chunk)
                finally:
                    await http_client.close()

            return stream_handler()

        try:
            openai_response = await collect_stream_response(response, model)
            return litellm.ModelResponse(**openai_response)
        finally:
            await http_client.close()

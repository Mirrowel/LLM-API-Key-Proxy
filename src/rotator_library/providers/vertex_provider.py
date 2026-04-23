# SPDX-License-Identifier: LGPL-3.0-only

import os
import copy
import json
import httpx
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

from .provider_interface import ProviderInterface
from .provider_cache import ProviderCache
from ..utils.paths import get_cache_dir

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


def _get_vertex_signature_cache_file():
    return get_cache_dir(subdir="vertex") / "gemini3_signatures.json"


class VertexProvider(ProviderInterface):
    """
    Provider for Google Vertex AI using Express Mode API keys.

    Express mode API keys use `x-goog-api-key` header authentication
    against the Vertex AI OpenAI-compatible endpoint:
      https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global/endpoints/openapi

    Environment variables:
        VERTEX_PROJECT   - Default GCP project ID (used when key doesn't embed project)
        VERTEX_LOCATION  - GCP location (default: "global")
        VERTEX_API_KEY_N - API keys. Two formats supported:
            1. Plain key:    VERTEX_API_KEY_1=AQ.Ab8...
               (uses VERTEX_PROJECT as the project)
            2. project:key:  VERTEX_API_KEY_1=my-project:AQ.Ab8...
               (each key specifies its own project)

    Models are advertised with the "google/" prefix from the upstream API
    and re-prefixed as "vertex/" for the proxy's internal routing.
    """

    # The upstream Vertex AI OpenAI-compatible endpoint returns standard
    # OpenAI-format responses, so cost calculation can use litellm's defaults.
    skip_cost_calculation: bool = False

    def __init__(self):
        self.default_project = os.getenv("VERTEX_PROJECT")
        self.location = os.getenv("VERTEX_LOCATION", "global")

        self._preserve_thought_signatures = os.getenv(
            "VERTEX_PRESERVE_THOUGHT_SIGNATURES", "true"
        ).lower() in ("true", "1", "yes")
        self._signature_cache = ProviderCache(
            _get_vertex_signature_cache_file(),
            int(os.getenv("VERTEX_SIGNATURE_CACHE_TTL", "3600")),
            int(os.getenv("VERTEX_SIGNATURE_DISK_TTL", "86400")),
            env_prefix="VERTEX_SIGNATURE",
        )

        lib_logger.info(
            f"VertexProvider initialized: default_project={self.default_project}, "
            f"location={self.location}"
        )

    @staticmethod
    def _iter_assistant_tool_calls(messages: list):
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []) or []:
                if tc.get("type", "function") == "function":
                    yield tc

    def _messages_indicate_thinking(self, messages: list) -> bool:
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            if msg.get("reasoning_content"):
                return True

        for tc in self._iter_assistant_tool_calls(messages):
            if self._get_thought_signature(tc):
                return True
            tc_id = tc.get("id", "")
            if tc_id and self._signature_cache.retrieve(tc_id):
                return True

        return False

    def _should_enable_thinking(self, messages: list, thinking: Any) -> bool:
        if isinstance(thinking, dict):
            return thinking.get("type") != "disabled"

        return self._messages_indicate_thinking(messages)

    @staticmethod
    def _get_thought_signature(tc: dict) -> Optional[str]:
        sig = tc.get("thought_signature")
        if sig:
            return sig
        ec = tc.get("extra_content")
        if isinstance(ec, dict):
            g = ec.get("google")
            if isinstance(g, dict):
                return g.get("thought_signature")
        return None

    @staticmethod
    def _set_thought_signature(tc: dict, sig: str) -> None:
        ec = tc.setdefault("extra_content", {})
        g = ec.setdefault("google", {})
        g["thought_signature"] = sig

    def _inject_thought_signatures(self, messages: list, thinking: Any) -> list:
        if not self._should_enable_thinking(messages, thinking):
            return messages

        messages = copy.deepcopy(messages)

        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                continue

            first_func = True
            for tc in tool_calls:
                if tc.get("type", "function") != "function":
                    continue

                sig = self._get_thought_signature(tc)
                if not sig:
                    tc_id = tc.get("id", "")
                    sig = self._signature_cache.retrieve(tc_id)

                if sig:
                    self._set_thought_signature(tc, sig)
                elif first_func:
                    self._set_thought_signature(tc, "skip_thought_signature_validator")

                first_func = False

        return messages

    def _extract_thought_signatures(self, data: dict) -> None:
        if not self._preserve_thought_signatures:
            return

        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            for tc in msg.get("tool_calls", []):
                sig = self._get_thought_signature(tc)
                if sig and tc.get("id"):
                    self._signature_cache.store(tc["id"], sig)

    def _extract_thought_signatures_stream(self, data: dict) -> None:
        if not self._preserve_thought_signatures:
            return

        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            for tc in delta.get("tool_calls", []):
                sig = self._get_thought_signature(tc)
                if sig and tc.get("id"):
                    self._signature_cache.store(tc["id"], sig)

    @staticmethod
    def _sanitize_tool_schemas(tools: list) -> list:
        """
        Sanitize tool schemas for Vertex AI compatibility.

        Vertex's OpenAI-compatible endpoint validates JSON Schema strictly
        and rejects common schema issues from client SDKs:
        - "ref" keys (Pydantic internal markers, not valid JSON Schema)
        - "$schema" meta-property (not needed for function declarations)

        Recursively removes these issues in tool parameter schemas.
        """
        if not tools:
            return tools

        tools = copy.deepcopy(tools)

        ref_count = 0
        schema_count = 0

        def fix_schema(obj):
            nonlocal ref_count, schema_count
            if isinstance(obj, dict):
                # Remove "ref" keys with string values — these are Pydantic
                # internal markers (e.g. {"ref": "QuestionPrompt"}) that are
                # not valid JSON Schema and cause Vertex to reject the request.
                # We preserve "ref" when it's a dict (legitimate property def).
                if "ref" in obj and isinstance(obj["ref"], str):
                    obj.pop("ref")
                    ref_count += 1
                # Remove "$schema" meta-property
                if "$schema" in obj:
                    obj.pop("$schema")
                    schema_count += 1
                # Recursively fix nested objects
                for value in obj.values():
                    fix_schema(value)
            elif isinstance(obj, list):
                for item in obj:
                    fix_schema(item)

        for tool in tools:
            func = tool.get("function", {})
            params = func.get("parameters")
            if params:
                fix_schema(params)

        lib_logger.debug(f"Schema sanitization: removed {ref_count} ref, removed {schema_count} $schema")
        return tools

    def _parse_credential(self, credential: str) -> tuple:
        """
        Parse a credential string into (project_id, api_key).

        Supports two formats:
        - "project_id:api_key" — project embedded in credential
        - "api_key"           — uses self.default_project

        Returns:
            Tuple of (project_id, api_key)

        Raises:
            ValueError: If no project can be determined
        """
        if ":" in credential:
            project, api_key = credential.split(":", 1)
            return project, api_key
        elif self.default_project:
            return self.default_project, credential
        else:
            raise ValueError(
                "Cannot determine project for credential. Either use "
                "'project_id:api_key' format or set VERTEX_PROJECT env var."
            )

    def _build_api_base(self, project: str) -> str:
        """Build the OpenAI-compatible base URL for a given project."""
        # Use v1beta1 for Express Mode API key support
        version = "v1beta1"
        if self.location == "global":
            return (
                f"https://aiplatform.googleapis.com/{version}/projects/{project}"
                f"/locations/global/endpoints/openapi"
            )
        else:
            return (
                f"https://{self.location}-aiplatform.googleapis.com/{version}/projects/{project}"
                f"/locations/{self.location}/endpoints/openapi"
            )

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Return the list of available Vertex AI models.

        Vertex AI does not expose an OpenAI-compatible /v1/models endpoint,
        so model discovery cannot be performed dynamically.

        Sources (checked in order):
        1. VERTEX_MODELS env var — comma-separated list of bare model names
           (e.g. "gemini-2.5-pro,gemini-3-flash-preview").
        2. Built-in defaults — a curated list of known-active Vertex models,
           updated manually when Google publishes new ones.

        When a new model becomes available on Vertex AI, add it to
        VERTEX_MODELS or update the defaults list below.
        """
        # 1. Check for explicit env-var override
        env_models_raw = os.getenv("VERTEX_MODELS", "").strip()
        if env_models_raw:
            env_models = [m.strip() for m in env_models_raw.split(",") if m.strip()]
            models = [f"vertex/{m}" for m in env_models]
            lib_logger.info(
                f"Using {len(models)} Vertex models from VERTEX_MODELS env var"
            )
            return models

        # 2. Static defaults — known-active Vertex AI models
        default_models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3.1-pro-preview",
            "gemma-4-26b-a4b-it-maas",
        ]
        models = [f"vertex/{m}" for m in default_models]
        lib_logger.info(f"Using {len(models)} default models for Vertex provider")
        return models

    def has_custom_logic(self) -> bool:
        """
        Returns True — we handle the HTTP call ourselves because the
        Vertex AI API key must be sent as `x-goog-api-key` header,
        not `Authorization: Bearer`.
        """
        return True

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Return the x-goog-api-key header for Vertex AI API key auth."""
        _, api_key = self._parse_credential(credential_identifier)
        return {"x-goog-api-key": api_key}

    def calculate_cost(
        self, 
        model: str, 
        prompt_tokens: int, 
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0
    ) -> float:
        """
        Calculate cost using the proxy's ModelRegistry pricing data.

        The executor calls this before falling back to litellm.completion_cost().
        Since litellm doesn't know our vertex/ prefix, we use the registry
        which has fuzzy-matched pricing from modelsdev/openrouter.
        """
        try:
            from ..model_info_service import get_model_info_service
            registry = get_model_info_service()
            if registry:
                cost = registry.calculate_cost(
                    model, 
                    prompt_tokens, 
                    completion_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens
                )
                if cost is not None:
                    return cost
        except Exception as e:
            lib_logger.debug(f"Registry cost calculation failed for {model}: {e}")
        return 0.0

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Any:
        """
        Make a chat completion request to the Vertex AI OpenAI-compatible endpoint.

        Handles both streaming and non-streaming requests.
        """
        credential = kwargs.pop("credential_identifier", None)
        if not credential:
            raise ValueError("No credential_identifier provided")

        # Parse credential to get project-specific API key and base URL
        project, api_key = self._parse_credential(credential)
        api_base = self._build_api_base(project)

        # Extract model name — strip our provider prefix
        model = kwargs.get("model", "")
        if model.startswith("vertex/"):
            # The Vertex AI OpenAI-compat endpoint expects "google/" prefix
            bare_model = model.replace("vertex/", "", 1)
            model = f"google/{bare_model}"

        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        thinking = kwargs.get("thinking")

        messages = self._inject_thought_signatures(messages, thinking)

        # Sanitize tool schemas for Vertex AI compatibility
        if "tools" in kwargs and kwargs["tools"]:
            lib_logger.debug(f"Sanitizing {len(kwargs['tools'])} tool schemas for Vertex")
            kwargs["tools"] = self._sanitize_tool_schemas(kwargs["tools"])

        # Build the request payload (OpenAI-compatible format)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # Forward supported OpenAI params
        for param in [
            "temperature", "max_tokens", "top_p", "n", "stop",
            "frequency_penalty", "presence_penalty", "tools", "tool_choice",
            "response_format", "stream",
        ]:
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]

        # Resolve thinking config. Sources (highest priority first):
        #   1. top-level kwargs["thinking"] (explicit client/transforms request)
        #   2. continuation mode when the conversation already contains
        #      reasoning_content or stored Google thought signatures
        # The global _guard_thinking_tool_calls skips Vertex, so we won't
        # see a "disabled" override for missing reasoning_content.
        reasoning_effort = kwargs.get("reasoning_effort")

        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            pass
        elif isinstance(thinking, dict):
            payload.setdefault("extra_body", {})
            google = payload["extra_body"].setdefault("google", {})
            google["thinking_config"] = {
                "include_thoughts": thinking.get("include_thoughts", True),
            }
            if "budget_tokens" in thinking:
                google["thinking_config"]["thinking_budget"] = thinking["budget_tokens"]
            google.setdefault("thought_tag_marker", "thought")
        elif self._should_enable_thinking(messages, thinking) and reasoning_effort is None:
            payload.setdefault("extra_body", {})
            google = payload["extra_body"].setdefault("google", {})
            google["thinking_config"] = {"include_thoughts": True}
            google.setdefault("thought_tag_marker", "thought")

        if reasoning_effort is not None and "extra_body" not in payload:
            payload["reasoning_effort"] = reasoning_effort

        headers = {
            "Content-Type": "application/json",
        }

        # For Agent Platform API (Express Mode), the key MUST be passed as a query parameter
        url = f"{api_base}/chat/completions?key={api_key}"

        if stream:
            return await self._stream_completion(client, url, headers, payload)
        else:
            return await self._non_stream_completion(client, url, headers, payload)

    async def _non_stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> Any:
        """Make a non-streaming completion request."""
        from litellm import ModelResponse
        from litellm.types.utils import Usage, Message, Choices

        response = await client.post(
            url,
            headers=headers,
            json=payload,
            timeout=120.0,
        )

        if response.status_code != 200:
            await self._raise_api_error(response)

        data = response.json()

        self._extract_thought_signatures(data)

        # Convert to LiteLLM ModelResponse format
        model_response = ModelResponse()
        model_response.id = data.get("id", "")
        model_response.model = data.get("model", payload.get("model", ""))
        model_response.object = "chat.completion"
        model_response.created = data.get("created", 0)

        # Parse choices
        choices = []
        for i, choice_data in enumerate(data.get("choices", [])):
            msg_data = choice_data.get("message", {})
            message = Message(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content"),
            )
            if "tool_calls" in msg_data:
                message.tool_calls = msg_data["tool_calls"]
            if msg_data.get("reasoning_content") is not None:
                message.reasoning_content = msg_data["reasoning_content"]

            choice = Choices(
                index=i,
                message=message,
                finish_reason=choice_data.get("finish_reason", "stop"),
            )
            choices.append(choice)

        model_response.choices = choices

        # Parse usage
        usage_data = data.get("usage", {})
        model_response.usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        ctd = usage_data.get("completion_tokens_details")
        if ctd and isinstance(ctd, dict):
            model_response.usage.completion_tokens_details = ctd

        return model_response

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> AsyncGenerator:
        """Make a streaming completion request."""
        from litellm import ModelResponse
        from litellm.types.utils import Delta, StreamingChoices, Usage

        payload["stream"] = True

        # Use httpx streaming
        request = client.build_request(
            "POST", url, headers=headers, json=payload, timeout=120.0
        )
        response = await client.send(request, stream=True)

        if response.status_code != 200:
            body = await response.aread()
            await response.aclose()
            self._raise_api_error_sync(response.status_code, body)

        async def generate():
            try:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            # Yield final DONE
                            return
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        self._extract_thought_signatures_stream(data)

                        # Convert to LiteLLM streaming format
                        chunk = ModelResponse(stream=True)
                        chunk.id = data.get("id", "")
                        chunk.model = data.get("model", "")
                        chunk.object = "chat.completion.chunk"
                        chunk.created = data.get("created", 0)

                        streaming_choices = []
                        for choice_data in data.get("choices", []):
                            delta_data = choice_data.get("delta", {})
                            delta = Delta(
                                role=delta_data.get("role"),
                                content=delta_data.get("content"),
                            )
                            if "tool_calls" in delta_data:
                                delta.tool_calls = delta_data["tool_calls"]
                            if delta_data.get("reasoning_content") is not None:
                                delta.reasoning_content = delta_data["reasoning_content"]

                            sc = StreamingChoices(
                                index=choice_data.get("index", 0),
                                delta=delta,
                                finish_reason=choice_data.get("finish_reason"),
                            )
                            streaming_choices.append(sc)

                        chunk.choices = streaming_choices

                        # Include usage if present (final chunk)
                        if "usage" in data:
                            usage_data = data["usage"]
                            chunk.usage = Usage(
                                prompt_tokens=usage_data.get("prompt_tokens", 0),
                                completion_tokens=usage_data.get("completion_tokens", 0),
                                total_tokens=usage_data.get("total_tokens", 0),
                            )
                            ctd = usage_data.get("completion_tokens_details")
                            if ctd and isinstance(ctd, dict):
                                chunk.usage.completion_tokens_details = ctd

                        yield chunk
            finally:
                await response.aclose()

        return generate()

    async def _raise_api_error(self, response: httpx.Response) -> None:
        """Raise an appropriate litellm error from an HTTP error response."""
        body = response.text
        status = response.status_code

        self._raise_api_error_sync(status, body.encode())

    def _raise_api_error_sync(self, status: int, body: bytes) -> None:
        """Raise an appropriate litellm error given status code and body."""
        import litellm

        body_str = body.decode("utf-8", errors="replace")

        if status == 429:
            raise litellm.RateLimitError(
                message=f"VertexError - {body_str}",
                llm_provider="vertex_ai",
                model="",
            )
        elif status == 401 or status == 403:
            raise litellm.AuthenticationError(
                message=f"VertexError - {body_str}",
                llm_provider="vertex_ai",
                model="",
            )
        elif status == 400:
            raise litellm.BadRequestError(
                message=f"VertexError - {body_str}",
                llm_provider="vertex_ai",
                model="",
            )
        elif status == 404:
            raise litellm.NotFoundError(
                message=f"VertexError - {body_str}",
                llm_provider="vertex_ai",
                model="",
            )
        else:
            raise litellm.APIError(
                message=f"VertexError (HTTP {status}) - {body_str}",
                llm_provider="vertex_ai",
                model="",
                status_code=status,
            )

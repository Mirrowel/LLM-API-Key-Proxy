# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import os
import logging
import time
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, TYPE_CHECKING
import httpx
import litellm
import openai

if TYPE_CHECKING:
    from ..usage import UsageManager

from .provider_interface import ProviderInterface
from .utilities.opencode_quota_tracker import OpencodeQuotaTracker
from ..model_definitions import ModelDefinitions
from ..error_handler import mask_credential

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


class OpencodeProvider(OpencodeQuotaTracker, ProviderInterface):
    """
    Provider for OpenCode 'Go' service - OpenAI-compatible API.
    """

    provider_env_name = "opencode_go"

    SUPPORTED_PARAMS = {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "tools",
        "tool_choice",
        "presence_penalty",
        "frequency_penalty",
        "n",
        "stop",
        "seed",
        "metadata",
        "logit_bias",
        "top_logprobs",
        "logprobs",
        "extra_headers",
        "extra_body",
        "api_key",
        "api_base",
        "custom_llm_provider",
        "client",
    }

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse OpenCode-specific quota errors.

        OpenCode returns:
        - "Monthly usage limit reached. Resets in 3 days." → quota exhaustion
        - "5-hour usage limit reached" / "Weekly usage limit reached" → shorter-term quota
        """
        import re

        body = error_body
        if not body:
            if hasattr(error, "response") and hasattr(error.response, "text"):
                body = error.response.text
            elif hasattr(error, "body"):
                body = str(error.body) if not isinstance(error.body, str) else error.body
            else:
                body = str(error)

        body_lower = body.lower() if body else ""

        if "usage limit" not in body_lower and "limit reached" not in body_lower:
            return None

        retry_after = None
        days_match = re.search(r"resets? in\s*(\d+)\s*days?", body_lower)
        if days_match:
            retry_after = int(days_match.group(1)) * 86400

        if "monthly" in body_lower:
            return {"retry_after": retry_after, "reason": "monthly_quota_exhausted"}
        if "weekly" in body_lower:
            return {"retry_after": retry_after, "reason": "weekly_quota_exhausted"}
        if "5-hour" in body_lower or "5 hour" in body_lower or "rolling" in body_lower:
            return {"retry_after": retry_after, "reason": "rolling_quota_exhausted"}

        return {"retry_after": retry_after, "reason": "quota_exhausted"}

    # Quota groups: display-only time windows + a hidden global group for blocking.
    # The tiered windows (5hr < weekly < monthly) are for dashboard visibility.
    # "opencode_go-global" is the key CooldownChecker uses during credential selection.
    model_quota_groups = {
        "5hr": ["5hr"],
        "weekly": ["weekly"],
        "monthly": ["monthly"],
        "opencode_go-global": [],
    }

    hidden_quota_groups = frozenset({"opencode_go-global"})

    # Tier hierarchy: higher-tier exhaustion implies all lower tiers are blocked.
    # monthly > weekly > 5hr.  When monthly is exhausted the credential cannot be used
    # even if 5hr/weekly windows show remaining capacity (that capacity is unreachable).
    QUOTA_TIER_HIERARCHY = ["5hr", "weekly", "monthly"]

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """All real models share the opencode_go-global quota pool."""
        clean = model.split("/")[-1] if "/" in model else model
        if clean in ("5hr", "weekly", "monthly"):
            return clean
        return "opencode_go-global"

    def __init__(self):
        super().__init__()
        self.api_base = os.getenv("OPENCODE_GO_API_BASE", "https://opencode.ai/zen/v1")
        self.global_workspace_id = os.getenv("OPENCODE_WORKSPACE_ID")
        self._balance_cache = {}
        self._quota_refresh_interval = 300
        self.model_definitions = ModelDefinitions()
        
        masked_wrk = mask_credential(self.global_workspace_id) if self.global_workspace_id else "None"
        lib_logger.debug(f"OpencodeProvider initialized: base={self.api_base}, global_wrk={masked_wrk}")

    def _get_headers(self, auth_cookie: Optional[str] = None) -> Dict[str, str]:
        """Return the custom headers required by OpenCode."""
        headers = {
            "HTTP-Referer": "https://opencode.ai/",
            "X-Title": "opencode",
            "User-Agent": "opencode/1.0",
        }
        if auth_cookie:
            headers["Cookie"] = f"auth={auth_cookie}; oc_locale=en"
        return headers

    def _parse_credential(self, credential_identifier: str) -> Dict[str, str]:
        """
        Parse the credential identifier into component parts.
        Format: sk-key (required) or api_key:workspace_id:auth_cookie (workspace and cookie optional)
        """
        result = {
            "api_key": credential_identifier,
            "workspace_id": self.global_workspace_id,
            "auth_cookie": None
        }

        if ":" in credential_identifier:
            parts = credential_identifier.split(":")
            # Part 0: API Key (Required)
            result["api_key"] = parts[0]
            
            # Part 1: Workspace ID (Optional)
            if len(parts) > 1 and parts[1]:
                result["workspace_id"] = parts[1]
                
            # Part 2: Auth Cookie (Optional)
            if len(parts) > 2 and parts[2]:
                rest = parts[2]
                if rest.startswith("auth="):
                    result["auth_cookie"] = rest[5:]
                else:
                    result["auth_cookie"] = rest
        
        # Fallback for simple Fe26.2** cookies passed as the only identifier
        if not result["auth_cookie"] and result["api_key"].startswith("Fe26.2**"):
            result["auth_cookie"] = result["api_key"]
            
        return result

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns models from the upstream go endpoint. Quota tracking virtual
        models are NOT returned here to keep them out of the public /v1/models list.
        """
        # 1. Check env var override first
        static_models = self.model_definitions.get_all_provider_models("opencode_go")
        if static_models:
            return static_models

        # 2. Query upstream go models endpoint
        try:
            go_base = self.api_base.replace("/zen/v1", "/zen/go/v1").rstrip("/")
            models_url = f"{go_base}/models"
            response = await client.get(
                models_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            discovered = [
                f"opencode_go/{m['id']}"
                for m in data.get("data", [])
                if m.get("id")
            ]
            if discovered:
                lib_logger.info(f"Discovered {len(discovered)} models from go endpoint")
                return discovered
        except Exception as e:
            lib_logger.warning(f"Failed to fetch go models: {e}")

        # 3. Graceful fallback
        return [
            "opencode_go/deepseek-v4-pro",
            "opencode_go/glm-5.1",
            "opencode_go/kimi-k2.6",
        ]

    # Models that only accept plain string content (no multipart arrays)
    TEXT_ONLY_MODELS = frozenset({"kimi", "moonshot", "glm"})

    def _is_deepseek_v4(self, model: str) -> bool:
        """Check if model is a DeepSeek V4 variant."""
        return "deepseek-v4" in model.lower()

    def _is_moonshot(self, model: str) -> bool:
        """Check if model is a Moonshot (Kimi) variant."""
        return "kimi" in model.lower() or "moonshot" in model.lower()

    def _requires_string_content(self, model: str) -> bool:
        """Check if model requires plain string content (no multipart arrays)."""
        model_lower = model.lower()
        return any(k in model_lower for k in self.TEXT_ONLY_MODELS)

    @staticmethod
    def _flatten_content_to_string(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Flatten multipart content arrays to plain strings.

        Providers like Kimi/GLM reject the OpenAI multipart format
        [{"type": "text", "text": "..."}] and only accept string content.
        Image parts are discarded since these models don't support vision.
        """
        new_messages = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                new_messages.append({**msg, "content": "\n".join(text_parts) if text_parts else ""})
            else:
                new_messages.append(msg)
        return new_messages

    def _fix_moonshot_json_schema(self, schema: Any) -> Any:
        """
        Recursively fix JSON schema for Moonshot models.
        - If 'anyOf' is present, the parent 'type' must be removed.
        - 'additionalProperties' is rejected by Moonshot/Azure AI validation.
        """
        if not isinstance(schema, dict):
            return schema

        new_schema = {}
        for k, v in schema.items():
            # Moonshot rejects additionalProperties inside tool schemas;
            # stripping it lets the schema pass validation.
            if k == "additionalProperties":
                continue
            if isinstance(v, dict):
                new_schema[k] = self._fix_moonshot_json_schema(v)
            elif isinstance(v, list):
                new_schema[k] = [
                    self._fix_moonshot_json_schema(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                new_schema[k] = v

        if "anyOf" in new_schema and "type" in new_schema:
            # Moonshot rejects schemas that have both 'type' and 'anyOf' at the same level.
            # The 'type' should be defined in the anyOf items instead.
            parent_type = new_schema.pop("type")
            for item in new_schema["anyOf"]:
                if isinstance(item, dict) and "type" not in item:
                    item["type"] = parent_type

        return new_schema

    def _apply_moonshot_fixes(self, kwargs: Dict[str, Any]) -> None:
        """Apply Moonshot-specific fixes to tools in kwargs."""
        tools = kwargs.get("tools")
        if not tools or not isinstance(tools, list):
            return

        new_tools = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                new_tools.append(tool)
                continue

            func = tool.get("function")
            if not func or not isinstance(func, dict):
                new_tools.append(tool)
                continue

            params = func.get("parameters")
            if params:
                func["parameters"] = self._fix_moonshot_json_schema(params)
            
            new_tools.append(tool)
        
        kwargs["tools"] = new_tools

    def _ensure_deepseek_v4_tool_choice(self, kwargs: Dict[str, Any], model: str) -> None:
        """
        Remove tool_choice for DeepSeek V4 reasoner models.

        deepseek-reasoner rejects tool_choice (even 'auto'). The reasoning
        models handle tool calling implicitly based on whether tools are
        present in the request.
        """
        if not self._is_deepseek_v4(model):
            return
        if "tool_choice" in kwargs:
            del kwargs["tool_choice"]
            lib_logger.debug(
                f"opencode_go: removed tool_choice for {model} "
                f"(not supported by reasoner models)"
            )

    def _ensure_deepseek_v4_reasoning(self, kwargs: Dict[str, Any], model: str) -> None:
        """
        Enable thinking mode and pad reasoning_content for DeepSeek V4.

        DeepSeek V4 models default to thinking mode ON. The API requires
        ``reasoning_content`` on assistant messages that contain tool_calls
        (the global ``_guard_thinking_tool_calls`` transform handles the
        unrecoverable case where the client dropped reasoning from a
        tool-call turn by setting ``thinking: disabled`` before we get here).

        This method:
        1. Enables thinking if not already configured (respects the guard).
        2. Pads non-tool-call assistant messages missing reasoning_content
           with ``""`` — the API ignores this field on non-tool-call turns.

        References:
        - https://api-docs.deepseek.com/guides/thinking_mode
        - OpenCode issue #24722 (reasoning_content mandatory for tool-call turns)
        """
        if not self._is_deepseek_v4(model):
            return

        extra_body = kwargs.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        if "thinking" not in extra_body:
            extra_body = {**extra_body, "thinking": {"type": "enabled"}}
            kwargs["extra_body"] = extra_body

        messages = kwargs.get("messages")
        if not messages:
            return

        new_messages: List[Dict[str, Any]] = []
        modified = False
        for msg in messages:
            if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                new_messages.append({**msg, "reasoning_content": ""})
                modified = True
            else:
                new_messages.append(msg)

        if modified:
            kwargs["messages"] = new_messages
            lib_logger.debug(
                f"opencode_go: padded reasoning_content for {model} "
                f"({sum(1 for m in new_messages if m.get('role') == 'assistant')} assistant messages)"
            )

    def _ensure_moonshot_reasoning(self, kwargs: Dict[str, Any], model: str) -> None:
        """
        Moonshot/Kimi models with thinking mode enabled require a non-empty
        ``reasoning_content`` on every assistant message.  Standard OpenAI
        clients omit this field on historical turns, causing a 400 error:

            "thinking is enabled but reasoning_content is missing
             in assistant tool call message at index N"

        The Kimi API treats empty-string ``""`` as absent, so we pad with a
        single space ``" "``, matching LiteLLM's MoonshotChatConfig approach.
        """
        if not self._is_moonshot(model):
            return

        extra_body = kwargs.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        if "thinking" not in extra_body:
            extra_body = {**extra_body, "thinking": {"type": "enabled"}}
            kwargs["extra_body"] = extra_body

        messages = kwargs.get("messages")
        if not messages:
            return

        new_messages: List[Dict[str, Any]] = []
        patched = 0
        for msg in messages:
            if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
                new_messages.append({**msg, "reasoning_content": " "})
                patched += 1
            else:
                new_messages.append(msg)

        if patched:
            kwargs["messages"] = new_messages
            lib_logger.debug(
                f"opencode_go: padded reasoning_content on {patched} "
                f"assistant message(s) for {model}"
            )

    def has_custom_logic(self) -> bool:
        return True

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential = kwargs.pop("credential_identifier", "")
        cred = self._parse_credential(credential)
        kwargs.pop("transaction_context", None)
        model = kwargs.get("model", "")
        model_bare = model.split("/")[-1] if "/" in model else model

        # Flatten multipart content for models that only accept strings
        if self._requires_string_content(model):
            messages = kwargs.get("messages")
            if messages:
                kwargs["messages"] = self._flatten_content_to_string(messages)

        # Apply model-specific fixes
        self._ensure_deepseek_v4_tool_choice(kwargs, model)
        self._ensure_deepseek_v4_reasoning(kwargs, model)
        if self._is_moonshot(model):
            self._apply_moonshot_fixes(kwargs)
            self._ensure_moonshot_reasoning(kwargs, model)

        # Ensure only one of max_tokens or max_completion_tokens is sent.
        # Some OpenCode models (like GLM-5.1) reject requests containing both.
        if kwargs.get("max_tokens") is not None and kwargs.get("max_completion_tokens") is not None:
            lib_logger.debug(f"opencode_go: both max_tokens and max_completion_tokens present for {model}, dropping max_tokens")
            kwargs.pop("max_tokens")

        kwargs["model"] = "openai/" + model_bare
        extra_headers = self._get_headers(cred["auth_cookie"])
        existing_headers = kwargs.get("extra_headers") or {}
        kwargs["extra_headers"] = {**existing_headers, **extra_headers}
        actual_key = cred["api_key"]
        if not actual_key or actual_key == "dummy":
            actual_key = cred["auth_cookie"]
        kwargs["api_key"] = actual_key
        api_base = self.api_base
        if "/zen/v1" in api_base and not "/zen/go/v1" in api_base:
             api_base = api_base.replace("/zen/v1", "/zen/go/v1")
        kwargs["api_base"] = api_base
        kwargs["custom_llm_provider"] = "openai"
        kwargs["client"] = openai.AsyncOpenAI(
            api_key=actual_key,
            base_url=api_base,
            http_client=client,
        )
        unsupported = set(kwargs.keys()) - self.SUPPORTED_PARAMS
        if unsupported:
            lib_logger.debug(f"opencode_go: stripping unsupported params for {model}: {unsupported}")
            kwargs = {k: v for k, v in kwargs.items() if k in self.SUPPORTED_PARAMS}
        return await litellm.acompletion(**kwargs)

    async def aembedding(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> litellm.EmbeddingResponse:
        credential = kwargs.pop("credential_identifier", "")
        cred = self._parse_credential(credential)
        kwargs.pop("transaction_context", None)
        model = kwargs.get("model", "")
        model_bare = model.split("/")[-1] if "/" in model else model
        kwargs["model"] = "openai/" + model_bare
        extra_headers = self._get_headers(cred["auth_cookie"])
        existing_headers = kwargs.get("extra_headers") or {}
        kwargs["extra_headers"] = {**existing_headers, **extra_headers}
        actual_key = cred["api_key"]
        if not actual_key or actual_key == "dummy":
            actual_key = cred["auth_cookie"]
        kwargs["api_key"] = actual_key
        kwargs["api_base"] = self.api_base
        kwargs["custom_llm_provider"] = "openai"
        kwargs["client"] = openai.AsyncOpenAI(
            api_key=actual_key,
            base_url=self.api_base,
            http_client=client,
        )
        return await litellm.aembedding(**kwargs)

    async def refresh_balance(
        self,
        api_key: str,
        credential_identifier: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        cred = self._parse_credential(credential_identifier)
        auth_cookie = cred["auth_cookie"]
        workspace_id = cred["workspace_id"]
        if not auth_cookie or not workspace_id:
            return {"status": "skipped", "reason": "missing credentials"}
        return await super().refresh_balance(
            auth_cookie, credential_identifier, workspace_id=workspace_id, client=client
        )

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        return {
            "interval": self._quota_refresh_interval,
            "name": "opencode_go_quota_refresh",
            "run_on_start": True,
        }

    async def fetch_initial_baselines(
        self,
        credentials: List[str],
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch live quota/balance for all credentials.
        This is called by the RotatingClient during force_refresh.
        """
        results = {}
        # Use provided client or create a temporary one
        async def _fetch():
            for ident in credentials:
                try:
                    balance_data = await self.refresh_balance(ident, ident, client=client)
                    results[ident] = balance_data
                except Exception as e:
                    results[ident] = {"status": "error", "message": str(e)}

        if client:
            await _fetch()
        else:
            async with httpx.AsyncClient(timeout=30.0) as new_client:
                client = new_client
                await _fetch()
        
        return results

    async def _store_baselines_to_usage_manager(
        self,
        results: Dict[str, Any],
        usage_manager: "UsageManager",
        force: bool = False,
        is_initial_fetch: bool = False,
    ) -> int:
        """
        Store the fetched quota results into the usage manager.
        """
        stored_count = 0
        for ident, balance_data in results.items():
            if balance_data.get("status") == "success":
                usage_raw = balance_data.get("usage_raw", {})
                now = balance_data.get("fetched_at", time.time())
                windows_map = {
                    "rollingUsage": "5hr",
                    "weeklyUsage": "weekly",
                    "monthlyUsage": "monthly"
                }

                # Collect per-window data for hierarchical exhaustion
                window_data = {}
                for raw_key, model_key in windows_map.items():
                    win_data = usage_raw.get(raw_key, {})
                    if isinstance(win_data, dict):
                        usage_percent = win_data.get("usagePercent", 0)
                        reset_in = win_data.get("resetInSec")
                        reset_ts = now + (reset_in if reset_in is not None else 0)
                        window_data[model_key] = {
                            "usage_percent": usage_percent,
                            "reset_ts": reset_ts,
                        }

                # Store each display window baseline
                for model_key, wd in window_data.items():
                    val_to_store = round(float(wd["usage_percent"]), 2)
                    if val_to_store <= 0:
                        val_to_store = 0.0001

                    await usage_manager.update_quota_baseline(
                        ident,
                        model_key,
                        quota_max_requests=100,
                        quota_used=val_to_store,
                        quota_reset_ts=wd["reset_ts"],
                        force=force,
                        apply_exhaustion=False,
                    )

                # Hierarchical exhaustion: monthly > weekly > 5hr.
                # The highest-tier exhausted window determines if the credential
                # is blocked, using its reset_ts as the cooldown duration.
                global_exhausted = False
                global_reset_ts = None
                for tier_key in reversed(self.QUOTA_TIER_HIERARCHY):
                    wd = window_data.get(tier_key)
                    if wd and wd["usage_percent"] >= 100.0 and wd["reset_ts"] > now:
                        global_exhausted = True
                        global_reset_ts = wd["reset_ts"]
                        break

                await usage_manager.update_quota_baseline(
                    ident,
                    "opencode_go-global",
                    quota_max_requests=100,
                    quota_used=100 if global_exhausted else 0,
                    quota_reset_ts=global_reset_ts,
                    quota_group="opencode_go-global",
                    force=force,
                    apply_exhaustion=global_exhausted,
                )

                stored_count += 1
        return stored_count

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            results = await self.fetch_initial_baselines(credentials, client=client)
            await self._store_baselines_to_usage_manager(results, usage_manager, force=True)

    def calculate_cost(
        self, 
        model: str, 
        prompt_tokens: int, 
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0
    ) -> float:
        """
        Calculate cost for the request using ModelInfoService.
        """
        try:
            from ..model_info_service import get_model_info_service
            registry = get_model_info_service()
            if registry and registry.is_ready:
                cost = registry.calculate_cost(
                    model, 
                    prompt_tokens, 
                    completion_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens
                )
                return cost if cost is not None else 0.0
        except Exception as e:
            lib_logger.debug(f"Opencode Go cost calculation failed for {model}: {e}")
        return 0.0


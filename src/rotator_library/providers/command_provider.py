# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Command Code Provider

Custom provider for Command Code (commandcode.ai).
Uses the CLI generate API (/alpha/generate) to route completions
to open-source models using browser session cookies or environment API keys,
bypassing plan gating constraints on direct completions.
"""

import os
import json
import base64
import urllib.parse
import httpx
import uuid
import datetime
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

import litellm

from .provider_interface import ProviderInterface
from ..core.errors import StreamedAPIError
from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

# Default absolute path for session cookies
COOKIES_PATH = "/home/b3nw/projects/core/LLM-API-Key-Proxy/command_code_cookies.json"

# Models supported by Command Code
COMMAND_MODELS = [
    "command/deepseek-v4-pro",
    "command/deepseek-v4-flash",
    "command/qwen-3.7-max",
    "command/qwen-3.6-plus",
    "command/qwen-3.6-max-preview",
    "command/kimi-k2.6",
    "command/kimi-k2.5",
    "command/glm-5.1",
    "command/glm-5",
    "command/minimax-m3",
    "command/minimax-m2.7",
    "command/minimax-m2.5",
    "command/step-3.7-flash",
    "command/step-3.5-flash",
    "command/mimo-v2.5-pro",
    "command/mimo-v2.5",
    "command/gemini-3.5-flash",
    "command/gemini-3.1-flash-lite",
]

# Internal model mapping to Command Code canonical model names
MODEL_MAPPING = {
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "qwen-3.7-max": "Qwen/Qwen3.7-Max",
    "qwen-3.6-plus": "Qwen/Qwen3.6-Plus",
    "qwen-3.6-max-preview": "Qwen/Qwen3.6-Max-Preview",
    "kimi-k2.6": "moonshotai/Kimi-K2.6",
    "kimi-k2.5": "moonshotai/Kimi-K2.5",
    "glm-5.1": "zai-org/GLM-5.1",
    "glm-5": "zai-org/GLM-5",
    "minimax-m3": "MiniMaxAI/MiniMax-M3",
    "minimax-m2.7": "MiniMaxAI/MiniMax-M2.7",
    "minimax-m2.5": "MiniMaxAI/MiniMax-M2.5",
    "step-3.7-flash": "stepfun/Step-3.7-Flash",
    "step-3.5-flash": "stepfun/Step-3.5-Flash",
    "mimo-v2.5-pro": "xiaomi/mimo-v2.5-pro",
    "mimo-v2.5": "xiaomi/mimo-v2.5",
    "gemini-3.5-flash": "google/gemini-3.5-flash",
    "gemini-3.1-flash-lite": "google/gemini-3.1-flash-lite",
}


class CommandProvider(ProviderInterface):
    """
    Provider for Command Code API.
    Routes requests directly to `/alpha/generate` using browser session cookies.
    """

    provider_env_name = "command"
    skip_cost_calculation = True

    _latest_npm_version = "0.30.2"
    _latest_npm_version_fetched = 0.0

    # Quota groups for tracking monthly credit limits
    model_quota_groups = {
        "command_credits": ["command/_quota"],
    }

    def __init__(self):
        self.model_definitions = ModelDefinitions()

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """All models share the same monthly credit quota."""
        return "command_credits"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """Returns the virtual models in the quota group."""
        if group == "command_credits":
            return ["command/_quota"]
        return []

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """Monthly usage reset configuration."""
        return {
            "mode": "per_model",
            "window_seconds": 2592000,  # ~30 days (monthly credits)
            "field_name": "models",
        }

    async def _fetch_credits(self, api_key: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        """Fetch credits from billing API."""
        url = "https://api.commandcode.ai/alpha/billing/credits"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
            "x-command-code-version": "0.30.2",
            "x-cli-environment": "production",
            "Authorization": f"Bearer {api_key}"
        }
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            else:
                lib_logger.warning(f"Credits API returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            lib_logger.warning(f"Failed to fetch credits from Command Code API: {e}")
        return None

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """Configure periodic quota usage refresh."""
        return {
            "interval": 300,  # Refresh every 5 minutes
            "name": "command_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: Any,
        credentials: List[str],
    ) -> None:
        """Refresh credit usage baseline from Command Code API."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            for api_key in credentials:
                try:
                    usage_data = await self._fetch_credits(api_key, client)
                    if usage_data and "credits" in usage_data:
                        credits_info = usage_data["credits"]
                        monthly = credits_info.get("monthlyCredits", 0.0)
                        purchased = credits_info.get("purchasedCredits", 0.0)
                        free = credits_info.get("freeCredits", 0.0)
                        
                        total_remaining = monthly + purchased + free
                        max_limit = 10.00
                        
                        # Convert to cents for integer representation
                        max_requests = int(max_limit * 100)
                        
                        # Round down the remaining credits to the nearest cent to match TUI/website
                        remaining_cents = int(total_remaining * 100)
                        quota_used = max(0, max_requests - remaining_cents)
                        
                        # Set reset to end of current month
                        now = datetime.datetime.now(datetime.timezone.utc)
                        if now.month == 12:
                            next_month = datetime.datetime(now.year + 1, 1, 1, tzinfo=datetime.timezone.utc)
                        else:
                            next_month = datetime.datetime(now.year, now.month + 1, 1, tzinfo=datetime.timezone.utc)
                        reset_ts = next_month.timestamp()
                        
                        await usage_manager.update_quota_baseline(
                            accessor=api_key,
                            model="command/_quota",
                            quota_max_requests=max_requests,
                            quota_used=quota_used,
                            quota_reset_ts=reset_ts,
                            quota_group="command_credits",
                            force=True
                        )
                        
                        if total_remaining <= 0.0:
                            stable_id = usage_manager.registry.get_stable_id(
                                api_key, usage_manager.provider
                            )
                            state = usage_manager.states.get(stable_id)
                            if state:
                                await usage_manager.tracking.apply_cooldown(
                                    state=state,
                                    reason="quota_exhausted",
                                    until=reset_ts,
                                    model_or_group="command_credits",
                                    source="api_quota"
                                )
                except Exception as e:
                    lib_logger.warning(f"Failed to refresh credits for key in background: {e}")

    async def fetch_initial_baselines(
        self,
        credential_paths: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch initial credit baselines on startup."""
        results = {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            for api_key in credential_paths:
                try:
                    usage_data = await self._fetch_credits(api_key, client)
                    if usage_data:
                        results[api_key] = usage_data
                except Exception as e:
                    lib_logger.warning(f"Failed to fetch initial baseline for Command key: {e}")
        return results

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Dynamically fetch available models from the Command Code models API."""
        static_models = self.model_definitions.get_all_provider_models("command")
        if static_models:
            return static_models

        try:
            url = "https://api.commandcode.ai/provider/v1/models"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
                "Authorization": f"Bearer {api_key}"
            }
            resp = await client.get(url, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return [f"command/{m.get('id')}" for m in data if m.get("id")]
            else:
                lib_logger.warning(f"Models API returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            lib_logger.warning(f"Failed to dynamically fetch Command Code models: {e}")
        
        return COMMAND_MODELS

    def _load_session_credentials(self) -> tuple[List[str], Optional[str], Optional[str]]:
        """
        Load browser session cookies and parse the session token & user agent.
        """
        if not os.path.exists(COOKIES_PATH):
            return [], None, None

        try:
            with open(COOKIES_PATH, "r") as f:
                cookies_data = json.load(f)

            cookies_list = []
            session_token = None
            user_agent = None

            for cookie in cookies_data:
                name = cookie.get("name")
                value = cookie.get("value")
                cookies_list.append(f"{name}={value}")
                
                if name == "__Secure-commandcode_prod_.session_token":
                    session_token = urllib.parse.unquote(value)
                elif name == "__Secure-commandcode_prod_.session_data":
                    try:
                        decoded_val = urllib.parse.unquote(value)
                        padded = decoded_val + "=" * (4 - len(decoded_val) % 4)
                        decoded_bytes = base64.b64decode(padded, altchars=b'-_')
                        decoded_str = decoded_bytes.decode("utf-8", errors="ignore")
                        if "userAgent" in decoded_str:
                            import re
                            match = re.search(r'"userAgent"\s*:\s*"([^"]+)"', decoded_str)
                            if match:
                                user_agent = match.group(1)
                    except Exception:
                        pass

            return cookies_list, session_token, user_agent
        except Exception as e:
            lib_logger.error(f"Failed to read cookies from {COOKIES_PATH}: {e}")
            return [], None, None

    async def _get_latest_version(self, client: httpx.AsyncClient) -> str:
        """
        Fetch the latest version of the command-code package from NPM registry,
        caching the result for 24 hours to prevent high latency on completions.
        """
        import time
        now = time.time()
        if now - CommandProvider._latest_npm_version_fetched < 86400.0:
            return CommandProvider._latest_npm_version

        try:
            url = "https://registry.npmjs.org/command-code/latest"
            resp = await client.get(url, timeout=1.5)
            if resp.status_code == 200:
                version = resp.json().get("version")
                if version:
                    CommandProvider._latest_npm_version = version
                    CommandProvider._latest_npm_version_fetched = now
                    lib_logger.info(f"Discovered latest command-code package version: {version}")
                    return version
        except Exception as e:
            lib_logger.warning(f"Failed to fetch latest version from NPM registry: {e}")

        # Update cache timestamp even on failure to avoid spamming the registry on subsequent calls
        CommandProvider._latest_npm_version_fetched = now
        return CommandProvider._latest_npm_version

    def _translate_tools_to_anthropic(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        translated = []
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
                func = t["function"]
                translated.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
            else:
                translated.append(t)
        return translated

    def _translate_tool_choice_to_anthropic(self, tool_choice: Any) -> Optional[Dict[str, Any]]:
        if not tool_choice:
            return None
        if isinstance(tool_choice, str):
            choice_str = tool_choice.lower()
            if choice_str == "auto":
                return {"type": "auto"}
            elif choice_str == "required":
                return {"type": "any"}
            elif choice_str == "none":
                return {"type": "none"}
            return {"type": "auto"}
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function" and "function" in tool_choice:
                func_name = tool_choice["function"].get("name")
                if func_name:
                    return {"type": "tool", "name": func_name}
            return tool_choice
        return {"type": "auto"}

    def _clean_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess messages to:
        1. Extract all system messages.
        2. Translate OpenAI tool calls & tool response messages to Vercel AI SDK format.
        3. Prepend system messages to the first user message.
        """
        system_prompts = []
        translated_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                if isinstance(content, str):
                    system_prompts.append(content)
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    system_prompts.append(" ".join(text_parts))
                continue
                
            if role == "user":
                translated_messages.append({
                    "role": "user",
                    "content": content if content else ""
                })
                continue
                
            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    content_parts = []
                    if isinstance(content, str) and content:
                        content_parts.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                content_parts.append({"type": "text", "text": part.get("text", "")})
                    
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            func = tc.get("function", {})
                            arguments = func.get("arguments", "{}")
                            if isinstance(arguments, dict):
                                input_data = arguments
                            else:
                                try:
                                    input_data = json.loads(arguments)
                                except Exception:
                                    input_data = {}
                            
                            content_parts.append({
                                "type": "tool-call",
                                "toolCallId": tc.get("id", ""),
                                "toolName": func.get("name", ""),
                                "input": input_data
                            })
                    translated_messages.append({
                        "role": "assistant",
                        "content": content_parts
                    })
                else:
                    translated_messages.append({
                        "role": "assistant",
                        "content": content if content else ""
                    })
                continue
                
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                tool_name = msg.get("name", "")
                tool_content = content
                
                # Extract plain text string value from tool_content
                text_value = ""
                if isinstance(tool_content, str):
                    text_value = tool_content
                elif isinstance(tool_content, list):
                    text_parts = []
                    for block in tool_content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool-result":
                                output = block.get("output")
                                if isinstance(output, dict) and "value" in output:
                                    text_parts.append(output["value"])
                                else:
                                    text_parts.append(str(block.get("result", "")))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    text_value = "\n".join(text_parts)
                else:
                    text_value = str(tool_content)

                translated_messages.append({
                    "role": "tool",
                    "content": [{
                        "type": "tool-result",
                        "toolCallId": tool_call_id,
                        "toolName": tool_name,
                        "output": {
                            "type": "text",
                            "value": text_value
                        }
                    }]
                })
                continue
                
            # Unknown role - copy as is
            translated_messages.append(dict(msg))

        # Now group consecutive messages of the same role
        grouped_messages = []
        for msg in translated_messages:
            if not grouped_messages:
                grouped_messages.append(dict(msg))
                continue
            
            last_msg = grouped_messages[-1]
            if last_msg["role"] == msg["role"]:
                # Merge
                if msg["role"] == "tool":
                    last_msg["content"] = (last_msg.get("content") or []) + (msg.get("content") or [])
                elif msg["role"] in ("user", "assistant"):
                    last_content = last_msg.get("content")
                    new_content = msg.get("content")
                    
                    if isinstance(last_content, str) and isinstance(new_content, str):
                        last_msg["content"] = (last_content + "\n\n" + new_content).strip()
                    else:
                        if isinstance(last_content, str):
                            last_blocks = [{"type": "text", "text": last_content}] if last_content else []
                        else:
                            last_blocks = list(last_content) if last_content else []
                            
                        if isinstance(new_content, str):
                            new_blocks = [{"type": "text", "text": new_content}] if new_content else []
                        else:
                            new_blocks = list(new_content) if new_content else []
                            
                        last_msg["content"] = last_blocks + new_blocks
            else:
                grouped_messages.append(dict(msg))

        if system_prompts:
            system_text = "\n".join(system_prompts)
            user_msg_idx = -1
            for idx, msg in enumerate(grouped_messages):
                if msg.get("role") == "user":
                    user_msg_idx = idx
                    break
            
            if user_msg_idx != -1:
                orig_content = grouped_messages[user_msg_idx].get("content") or ""
                if isinstance(orig_content, str):
                    grouped_messages[user_msg_idx]["content"] = f"System Prompt:\n{system_text}\n\n{orig_content}"
                elif isinstance(orig_content, list):
                    grouped_messages[user_msg_idx]["content"] = [
                        {"type": "text", "text": f"System Prompt:\n{system_text}\n\n"}
                    ] + orig_content
            else:
                grouped_messages.insert(0, {
                    "role": "user",
                    "content": f"System Prompt:\n{system_text}"
                })
                
        return grouped_messages

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs,
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion by translating the payload and routing to the generate API.
        """
        # Extract internal rotator params
        credential = kwargs.pop("credential_identifier", "")
        kwargs.pop("transaction_context", None)
        kwargs.pop("litellm_params", None)

        model = kwargs.get("model", "")
        if model.startswith("command/"):
            model_bare = model[8:]  # Strip "command/" prefix
        else:
            model_bare = model.split("/", 1)[1] if "/" in model else model
            
        cc_model = MODEL_MAPPING.get(model_bare, model_bare)

        stream = kwargs.get("stream", False)
        messages = self._clean_messages(kwargs.get("messages", []))
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p")
        max_tokens = kwargs.get("max_tokens") or 8192

        # Translate tools and tool_choice
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        parallel_tool_calls = kwargs.get("parallel_tool_calls")
        
        translated_tools = self._translate_tools_to_anthropic(tools)
        translated_tool_choice = self._translate_tool_choice_to_anthropic(tool_choice)

        # 1. Load session cookies & token
        cookies_list, session_token, user_agent = self._load_session_credentials()

        if not user_agent:
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"

        # Get the latest NPM version of the package dynamically
        cc_version = await self._get_latest_version(client)

        # 2. Build headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": user_agent,
            "x-command-code-version": cc_version,
            "x-cli-environment": "production",
        }

        # Authenticate using the environment credentials (API key) or fallback to session token from cookies
        auth_token = credential if credential else session_token
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        # Only send Cookie header if we are falling back to the session token
        if not credential and cookies_list:
            headers["Cookie"] = "; ".join(cookies_list)

        # 3. Build required config payload to satisfy Zod validation schema
        payload = {
            "config": {
                "editor": "vscode",
                "shell": "bash",
                "version": cc_version,
                "workingDir": "/home/b3nw/projects/core/LLM-API-Key-Proxy",
                "date": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
                "environment": "production",
                "structure": [],
                "isGitRepo": False,
                "currentBranch": "",
                "mainBranch": "",
                "gitStatus": "",
                "recentCommits": []
            },
            "memory": "",
            "taste": "",
            "skills": "",
            "params": {
                "messages": messages,
                "model": cc_model,
                "stream": True,
                "max_tokens": max_tokens
            },
            "threadId": str(uuid.uuid4())
        }

        if translated_tools is not None:
            payload["params"]["tools"] = translated_tools
        if translated_tool_choice is not None:
            payload["params"]["tool_choice"] = translated_tool_choice
        if parallel_tool_calls is not None:
            payload["params"]["parallel_tool_calls"] = parallel_tool_calls

        if temperature is not None:
            payload["params"]["temperature"] = temperature
        if top_p is not None:
            payload["params"]["top_p"] = top_p

        url = "https://api.commandcode.ai/alpha/generate"

        if stream:
            return self._stream_completion(client, url, headers, payload, model)
        else:
            return await self._non_stream_completion(client, url, headers, payload, model)

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Stream chat completions from generate endpoint."""
        created = int(datetime.datetime.now(datetime.UTC).timestamp())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        async with client.stream("POST", url, headers=headers, json=payload, timeout=TimeoutConfig.streaming()) as response:
            if response.status_code >= 400:
                body = await response.aread()
                error_msg = body.decode("utf-8", errors="ignore")
                raise StreamedAPIError(f"Command Code API Error ({response.status_code}): {error_msg}", data=None)

            tool_calls_registry = {}

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue

                evt_type = evt.get("type")
                if evt_type == "text-delta":
                    yield litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                        choices=[{
                            "index": 0,
                            "delta": {"content": evt.get("text", ""), "role": "assistant"},
                            "finish_reason": None
                        }]
                    )
                elif evt_type == "reasoning-delta":
                    yield litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                        choices=[{
                            "index": 0,
                            "delta": {"reasoning_content": evt.get("text", ""), "role": "assistant"},
                            "finish_reason": None
                        }]
                    )
                elif evt_type == "tool-input-start":
                    tc_id = evt.get("id", "")
                    tool_name = evt.get("toolName", "")
                    tc_idx = len(tool_calls_registry)
                    tool_calls_registry[tc_id] = {
                        "name": tool_name,
                        "index": tc_idx,
                        "arguments": ""
                    }
                    yield litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                        choices=[{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [{
                                    "index": tc_idx,
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": ""
                                    }
                                }]
                            },
                            "finish_reason": None
                        }]
                    )
                elif evt_type == "tool-input-delta":
                    tc_id = evt.get("id", "")
                    delta = evt.get("delta", "")
                    tc_info = tool_calls_registry.get(tc_id)
                    if tc_info:
                        tc_info["arguments"] += delta
                        yield litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=model,
                            object="chat.completion.chunk",
                            choices=[{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "tool_calls": [{
                                        "index": tc_info["index"],
                                        "id": tc_id,
                                        "function": {
                                            "arguments": delta
                                        }
                                    }]
                                },
                                "finish_reason": None
                            }]
                        )
                elif evt_type == "finish":
                    usage = evt.get("totalUsage", {})
                    input_tokens = usage.get("inputTokens", 0)
                    output_tokens = usage.get("outputTokens", 0)
                    cached_input_tokens = usage.get("cachedInputTokens", 0)

                    final_chunk = litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                        choices=[{
                            "index": 0,
                            "delta": {},
                            "finish_reason": evt.get("finishReason") or "stop"
                        }]
                    )
                    final_chunk.usage = litellm.Usage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                        cache_read_tokens=cached_input_tokens if cached_input_tokens else None
                    )
                    yield final_chunk
                elif evt_type == "error":
                    err_msg = evt.get("error", {}).get("message") or "Unknown stream error"
                    raise StreamedAPIError(f"Command Code Stream Error: {err_msg}", data=None)

    async def _non_stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> litellm.ModelResponse:
        """Handle non-streaming completion requests."""
        created = int(datetime.datetime.now(datetime.UTC).timestamp())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        response = await client.post(url, headers=headers, json=payload, timeout=TimeoutConfig.non_streaming())

        if response.status_code >= 400:
            error_msg = response.text
            raise RuntimeError(f"Command Code API Error ({response.status_code}): {error_msg}")

        full_text = ""
        reasoning_text = ""
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        finish_reason = "stop"
        tool_calls = []

        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            evt_type = obj.get("type")
            if evt_type == "text-delta":
                full_text += obj.get("text", "")
            elif evt_type == "reasoning-delta":
                reasoning_text += obj.get("text", "")
            elif evt_type == "tool-call":
                tc_id = obj.get("toolCallId")
                tc_name = obj.get("toolName")
                tc_input = obj.get("input", {})
                arguments_str = json.dumps(tc_input) if isinstance(tc_input, dict) else str(tc_input)
                tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc_name,
                        "arguments": arguments_str
                    }
                })
            elif evt_type == "finish":
                finish_reason = obj.get("rawFinishReason") or obj.get("finishReason") or "stop"
                usage = obj.get("totalUsage", {})
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)
                cached_input_tokens = usage.get("cachedInputTokens", 0)
            elif evt_type == "error":
                err_msg = obj.get("error", {}).get("message") or "Unknown error"
                raise RuntimeError(f"Command Code API Error: {err_msg}")

        message_dict = {
            "role": "assistant",
            "content": full_text if full_text else None,
        }
        if reasoning_text:
            message_dict["reasoning_content"] = reasoning_text
        if tool_calls:
            message_dict["tool_calls"] = tool_calls

        resp = litellm.ModelResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[{
                "index": 0,
                "message": message_dict,
                "finish_reason": finish_reason
            }],
            usage=litellm.Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cache_read_tokens=cached_input_tokens if cached_input_tokens else None
            )
        )
        return resp

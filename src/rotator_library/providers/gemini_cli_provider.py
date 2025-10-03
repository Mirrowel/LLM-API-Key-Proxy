# src/rotator_library/providers/gemini_cli_provider.py

import json
import httpx
import logging
import time
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Union, Optional, Tuple
from .provider_interface import ProviderInterface
from .gemini_auth_base import GeminiAuthBase
import litellm
from litellm.exceptions import RateLimitError
import os
from pathlib import Path

lib_logger = logging.getLogger('rotator_library')

CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com/v1internal"

# [NEW] Hardcoded model list based on Kilo example
HARDCODED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025"
]

class GeminiCliProvider(GeminiAuthBase, ProviderInterface):
    skip_cost_calculation = True

    def __init__(self):
        super().__init__()
        self.project_id_cache: Dict[str, str] = {} # Cache project ID per credential path

    async def _discover_project_id(self, credential_path: str, access_token: str, litellm_params: Dict[str, Any]) -> str:
        """Discovers the Google Cloud Project ID, with caching and onboarding for new accounts."""
        if credential_path in self.project_id_cache:
            return self.project_id_cache[credential_path]

        if litellm_params.get("project_id"):
            project_id = litellm_params["project_id"]
            lib_logger.info(f"Using configured Gemini CLI project ID: {project_id}")
            self.project_id_cache[credential_path] = project_id
            return project_id

        headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
        
        async with httpx.AsyncClient() as client:
            # 1. Try discovery endpoint with onboarding logic from Kilo example
            try:
                initial_project_id = "default"
                client_metadata = {
                    "ideType": "IDE_UNSPECIFIED", "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI", "duetProject": initial_project_id,
                }
                load_request = {"cloudaicompanionProject": initial_project_id, "metadata": client_metadata}
                
                response = await client.post(f"{CODE_ASSIST_ENDPOINT}:loadCodeAssist", headers=headers, json=load_request, timeout=20)
                response.raise_for_status()
                data = response.json()

                if data.get('cloudaicompanionProject'):
                    project_id = data['cloudaicompanionProject']
                    lib_logger.info(f"Discovered Gemini project ID via loadCodeAssist: {project_id}")
                    self.project_id_cache[credential_path] = project_id
                    return project_id
                
                # 2. If no project ID, trigger onboarding
                lib_logger.info("No existing Gemini project found, attempting to onboard user...")
                tier_id = next((t.get('id', 'free-tier') for t in data.get('allowedTiers', []) if t.get('isDefault')), 'free-tier')
                onboard_request = {"tierId": tier_id, "cloudaicompanionProject": initial_project_id, "metadata": client_metadata}

                lro_response = await client.post(f"{CODE_ASSIST_ENDPOINT}:onboardUser", headers=headers, json=onboard_request, timeout=30)
                lro_response.raise_for_status()
                lro_data = lro_response.json()

                for i in range(30): # Poll for up to 60 seconds
                    if lro_data.get('done'): break
                    await asyncio.sleep(2)
                    lib_logger.debug(f"Polling onboarding status... (Attempt {i+1}/30)")
                    lro_response = await client.post(f"{CODE_ASSIST_ENDPOINT}:onboardUser", headers=headers, json=onboard_request, timeout=30)
                    lro_response.raise_for_status()
                    lro_data = lro_response.json()
                
                if not lro_data.get('done'): raise ValueError("Onboarding process timed out.")

                project_id = lro_data.get('response', {}).get('cloudaicompanionProject', {}).get('id')
                if not project_id: raise ValueError("Onboarding completed, but no project ID was returned.")
                
                lib_logger.info(f"Successfully onboarded user and discovered project ID: {project_id}")
                self.project_id_cache[credential_path] = project_id
                return project_id

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                lib_logger.warning(f"Gemini onboarding/discovery failed: {e}. Falling back to project listing.")

        # 3. Fallback to listing all available GCP projects (last resort)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("https://cloudresourcemanager.googleapis.com/v1/projects", headers=headers, timeout=20)
                response.raise_for_status()
                projects = response.json().get('projects', [])
                if active_projects := [p for p in projects if p.get('lifecycleState') == 'ACTIVE']:
                    project_id = active_projects[0]['projectId']
                    lib_logger.info(f"Discovered Gemini project ID from active projects list: {project_id}")
                    self.project_id_cache[credential_path] = project_id
                    return project_id
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403: lib_logger.warning("Failed to list GCP projects due to a 403 Forbidden error.")
            else: lib_logger.error(f"Failed to list GCP projects: {e}")
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to list GCP projects: {e}")

        raise ValueError("Could not auto-discover Gemini project ID. Onboarding may have failed or the account lacks permissions. Please set GEMINI_CLI_PROJECT_ID in your .env file.")
    def has_custom_logic(self) -> bool:
        return True

    def _transform_messages(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        system_instruction = None
        gemini_contents = []
        
        # Separate system prompt from other messages
        if messages and messages[0].get('role') == 'system':
            system_prompt_content = messages.pop(0).get('content', '')
            if system_prompt_content:
                system_instruction = {
                    "role": "user",
                    "parts": [{"text": system_prompt_content}]
                }

        tool_call_id_to_name = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    if tool_call.get("type") == "function":
                        tool_call_id_to_name[tool_call["id"]] = tool_call["function"]["name"]

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            parts = []
            gemini_role = "model" if role == "assistant" else "tool" if role == "tool" else "user"

            if role == "user":
                text_content = ""
                if isinstance(content, str):
                    text_content = content
                elif isinstance(content, list):
                    text_content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
                if text_content:
                    parts.append({"text": text_content})

            elif role == "assistant":
                if isinstance(content, str):
                    parts.append({"text": content})
                if msg.get("tool_calls"):
                    for tool_call in msg["tool_calls"]:
                        if tool_call.get("type") == "function":
                            try:
                                args_dict = json.loads(tool_call["function"]["arguments"])
                            except (json.JSONDecodeError, TypeError):
                                args_dict = {}
                            parts.append({"functionCall": {"name": tool_call["function"]["name"], "args": args_dict}})

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                function_name = tool_call_id_to_name.get(tool_call_id)
                if function_name:
                    # Per Go example, wrap the tool response in a 'result' object
                    response_content = {"result": content}
                    parts.append({"functionResponse": {"name": function_name, "response": response_content}})

            if parts:
                gemini_contents.append({"role": gemini_role, "parts": parts})

        if not any(c['role'] == 'user' for c in gemini_contents):
             gemini_contents.insert(0, {"role": "user", "parts": [{"text": ""}]})
        elif gemini_contents and gemini_contents[0]["role"] == "model":
            gemini_contents.insert(0, {"role": "user", "parts": [{"text": ""}]})

        return system_instruction, gemini_contents

    def _handle_reasoning_parameters(self, payload: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
        custom_reasoning_budget = payload.get("custom_reasoning_budget", False)
        reasoning_effort = payload.get("reasoning_effort")

        if "thinkingConfig" in payload.get("generationConfig", {}):
            return None

        # Only apply reasoning logic to the gemini-2.5 model family
        if "gemini-2.5" not in model:
            payload.pop("reasoning_effort", None)
            payload.pop("custom_reasoning_budget", None)
            return None

        if not reasoning_effort:
            return {"thinkingBudget": -1, "include_thoughts": True}

        # If reasoning_effort is provided, calculate the budget
        budget = -1  # Default for 'auto' or invalid values
        if "gemini-2.5-pro" in model:
            budgets = {"low": 8192, "medium": 16384, "high": 32768}
        elif "gemini-2.5-flash" in model:
            budgets = {"low": 6144, "medium": 12288, "high": 24576}
        else:
            # Fallback for other gemini-2.5 models
            budgets = {"low": 1024, "medium": 2048, "high": 4096}

        budget = budgets.get(reasoning_effort, -1)
        if reasoning_effort == "disable":
            budget = 0
        
        if not custom_reasoning_budget:
            budget = budget // 4

        # Clean up the original payload
        payload.pop("reasoning_effort", None)
        payload.pop("custom_reasoning_budget", None)
        
        return {"thinkingBudget": budget, "include_thoughts": True}

    def _convert_chunk_to_openai(self, chunk: Dict[str, Any], model_id: str):
        lib_logger.debug(f"Converting Gemini chunk: {json.dumps(chunk)}")
        response_data = chunk.get('response', chunk)
        candidates = response_data.get('candidates', [])
        if not candidates:
            return

        candidate = candidates[0]
        parts = candidate.get('content', {}).get('parts', [])

        for part in parts:
            delta = {}
            finish_reason = None

            if 'functionCall' in part:
                function_call = part['functionCall']
                delta['tool_calls'] = [{
                    "index": 0,
                    "id": f"tool-call-{time.time()}",
                    "type": "function",
                    "function": {
                        "name": function_call.get('name'),
                        "arguments": json.dumps(function_call.get('args', {}))
                    }
                }]
            elif 'text' in part:
                # Use a lenient check for the 'thought' flag, as its type can be inconsistent
                if str(part.get('thought')).lower() == 'true':
                    delta['reasoning_content'] = part['text']
                else:
                    delta['content'] = part['text']
            
            if not delta:
                continue

            raw_finish_reason = candidate.get('finishReason')
            if raw_finish_reason:
                mapping = {'STOP': 'stop', 'MAX_TOKENS': 'length', 'SAFETY': 'content_filter'}
                finish_reason = mapping.get(raw_finish_reason, 'stop')

            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            
            openai_chunk = {
                "choices": [choice], "model": model_id, "object": "chat.completion.chunk",
                "id": f"chatcmpl-geminicli-{time.time()}", "created": int(time.time())
            }

            if 'usageMetadata' in response_data:
                usage = response_data['usageMetadata']
                openai_chunk["usage"] = {
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens": usage.get("totalTokenCount", 0),
                }
            
            yield openai_chunk

    async def acompletion(self, client: httpx.AsyncClient, **kwargs) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        model = kwargs["model"]
        credential_path = kwargs.pop("credential_identifier")
        
        async def do_call():
            # Get auth header once, it's needed for the request anyway
            auth_header = await self.get_auth_header(credential_path)

            # Discover project ID only if not already cached
            project_id = self.project_id_cache.get(credential_path)
            if not project_id:
                access_token = auth_header['Authorization'].split(' ')[1]
                project_id = await self._discover_project_id(credential_path, access_token, kwargs.get("litellm_params", {}))
            
            # Handle :thinking suffix from Kilo example
            model_name = model.split('/')[-1].replace(':thinking', '')

            gen_config = {
                "maxOutputTokens": kwargs.get("max_tokens", 64000), # Increased default
            }
            if "temperature" in kwargs:
                gen_config["temperature"] = kwargs["temperature"]
            else:
                gen_config["temperature"] = 0.7
            if "top_k" in kwargs:
                gen_config["topK"] = kwargs["top_k"]
            if "top_p" in kwargs:
                gen_config["topP"] = kwargs["top_p"]

            # Use the sophisticated reasoning logic
            thinking_config = self._handle_reasoning_parameters(kwargs, model_name)
            if thinking_config:
                gen_config["thinkingConfig"] = thinking_config

            system_instruction, contents = self._transform_messages(kwargs.get("messages", []))
            request_payload = {
                "model": model_name,
                "project": project_id,
                "request": {
                    "contents": contents,
                    "generationConfig": gen_config,
                },
            }

            if system_instruction:
                request_payload["request"]["systemInstruction"] = system_instruction

            if "tools" in kwargs and kwargs["tools"]:
                function_declarations = [
                    tool["function"] for tool in kwargs["tools"] if tool.get("type") == "function"
                ]
                if function_declarations:
                    request_payload["request"]["tools"] = [{"functionDeclarations": function_declarations}]
            
            # Log the final payload for debugging, as requested
            lib_logger.debug(f"Gemini CLI Request Payload: {json.dumps(request_payload, indent=2)}")
            url = f"{CODE_ASSIST_ENDPOINT}:streamGenerateContent"

            async def stream_handler():
                final_headers = auth_header.copy()
                final_headers.update({
                    "User-Agent": "google-api-nodejs-client/9.15.1",
                    "X-Goog-Api-Client": "gl-node/22.17.0",
                    "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
                })
                async with client.stream("POST", url, headers=final_headers, json=request_payload, params={"alt": "sse"}, timeout=600) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == "[DONE]": break
                            try:
                                chunk = json.loads(data_str)
                                for openai_chunk in self._convert_chunk_to_openai(chunk, model):
                                    yield litellm.ModelResponse(**openai_chunk)
                            except json.JSONDecodeError:
                                lib_logger.warning(f"Could not decode JSON from Gemini CLI: {line}")
            return stream_handler()

        try:
            response_gen = await do_call()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                lib_logger.warning("Gemini CLI returned 401. Forcing token refresh and retrying once.")
                creds = await self._load_credentials(credential_path)
                await self._refresh_token(credential_path, creds, force=True)
                response_gen = await do_call()
            elif e.response.status_code == 429:
                raise RateLimitError(
                    message=f"Gemini CLI rate limit exceeded: {e.response.text}",
                    llm_provider="gemini_cli",
                    response=e.response
                )
            else:
                raise e

        if kwargs.get("stream", False):
            return response_gen
        else:
            # Accumulate stream for non-streaming response
            chunks = [chunk async for chunk in response_gen]
            return litellm.utils.stream_to_completion_response(chunks)

    # Use the shared GeminiAuthBase for auth logic
    # get_models is not applicable for this custom provider
    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        """Returns a hardcoded list of known compatible Gemini CLI models."""
        return [f"gemini_cli/{model_id}" for model_id in HARDCODED_MODELS]
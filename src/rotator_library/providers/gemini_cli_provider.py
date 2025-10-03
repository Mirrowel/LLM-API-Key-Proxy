# src/rotator_library/providers/gemini_cli_provider.py

import json
import httpx
import logging
import time
from typing import List, Dict, Any, AsyncGenerator, Union, Optional
from .provider_interface import ProviderInterface
from .gemini_auth_base import GeminiAuthBase
import litellm
import os
from pathlib import Path

lib_logger = logging.getLogger('rotator_library')

CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com/v1internal"

class GeminiCliProvider(GeminiAuthBase, ProviderInterface):
    def __init__(self):
        super().__init__()
        self.project_id_cache: Dict[str, str] = {} # Cache project ID per credential path

    async def _discover_project_id(self, credential_path: str, access_token: str, litellm_params: Dict[str, Any]) -> str:
        """Discovers the Google Cloud Project ID, with caching."""
        if credential_path in self.project_id_cache:
            return self.project_id_cache[credential_path]

        if litellm_params.get("project_id"):
            project_id = litellm_params["project_id"]
            lib_logger.info(f"Using configured Gemini CLI project ID: {project_id}")
            self.project_id_cache[credential_path] = project_id
            return project_id

        headers = {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
        
        # 1. Try Gemini-specific discovery endpoint
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{CODE_ASSIST_ENDPOINT}:loadCodeAssist", headers=headers, json={"metadata": {"pluginType": "GEMINI"}})
                response.raise_for_status()
                data = response.json()
                if data.get('cloudaicompanionProject'):
                    project_id = data['cloudaicompanionProject']
                    lib_logger.info(f"Discovered Gemini project ID via loadCodeAssist: {project_id}")
                    self.project_id_cache[credential_path] = project_id
                    return project_id
        except httpx.RequestError as e:
            lib_logger.warning(f"Gemini loadCodeAssist failed, falling back to project listing: {e}")

        # 2. Fallback to listing all available GCP projects
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("https://cloudresourcemanager.googleapis.com/v1/projects", headers=headers)
                response.raise_for_status()
                projects = response.json().get('projects', [])
                active_projects = [p for p in projects if p.get('lifecycleState') == 'ACTIVE']
                if active_projects:
                    project_id = active_projects[0]['projectId']
                    lib_logger.info(f"Discovered Gemini project ID from active projects list: {project_id}")
                    self.project_id_cache[credential_path] = project_id
                    return project_id
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to list GCP projects: {e}")

        raise ValueError(
            "Could not auto-discover Gemini project ID. Please set GEMINI_CLI_PROJECT_ID in your .env file."
        )
    def has_custom_logic(self) -> bool:
        return True

    def _transform_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # As seen in Kilo examples, system prompts are injected into the first user message.
        gemini_contents = []
        system_prompt = ""
        if messages and messages[0].get('role') == 'system':
            system_prompt = messages.pop(0).get('content', '')

        for msg in messages:
            role = "model" if msg.get("role") == "assistant" else "user"
            content = msg.get("content", "")
            if system_prompt and role == "user":
                content = f"{system_prompt}\n\n{content}"
                system_prompt = "" # Inject only once
            gemini_contents.append({"role": role, "parts": [{"text": content}]})
        return gemini_contents

    def _convert_chunk_to_openai(self, chunk: Dict[str, Any], model_id: str) -> dict:
        response_data = chunk.get('response', chunk)
        candidate = response_data.get('candidates', [{}])[0]
        
        delta = {}
        finish_reason = None

        # Correctly handle reasoning vs. content based on 'thought' flag from Kilo example
        if 'content' in candidate and 'parts' in candidate['content']:
            part = candidate['content']['parts'][0]
            if part.get('text'):
                if part.get('thought') is True:
                    # This is a reasoning/thinking step
                    delta['reasoning_content'] = part['text']
                else:
                    delta['content'] = part['text']

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
        
        return openai_chunk

    async def acompletion(self, client: httpx.AsyncClient, **kwargs) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        model = kwargs["model"]
        credential_path = kwargs.pop("credential_identifier")
        
        async def do_call():
            auth_header = await self.get_auth_header(credential_path)
            project_id = await self._discover_project_id(credential_path, auth_header['Authorization'].split(' ')[1], kwargs.get("litellm_params", {}))
            
            # Handle :thinking suffix from Kilo example
            model_name = model.split('/')[-1]
            enable_thinking = model_name.endswith(':thinking')
            if enable_thinking:
                model_name = model_name.replace(':thinking', '')

            gen_config = {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 8192),
            }
            if enable_thinking:
                gen_config["thinkingConfig"] = {"thinkingBudget": -1}

            request_payload = {
                "model": model_name,
                "project": project_id,
                "request": {
                    "contents": self._transform_messages(kwargs.get("messages", [])),
                    "generationConfig": gen_config,
                },
            }

            url = f"{CODE_ASSIST_ENDPOINT}:streamGenerateContent"
            async def stream_handler():
                async with client.stream("POST", url, headers=auth_header, json=request_payload, params={"alt": "sse"}, timeout=600) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == "[DONE]": break
                            try:
                                chunk = json.loads(data_str)
                                openai_chunk = self._convert_chunk_to_openai(chunk, model)
                                yield litellm.ModelResponse(**openai_chunk)
                            except json.JSONDecodeError:
                                lib_logger.warning(f"Could not decode JSON from Gemini CLI: {line}")
            return stream_handler()

        try:
            response_gen = await do_call()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                lib_logger.warning("Gemini CLI returned 401. Forcing token refresh and retrying once.")
                await self._refresh_token(credential_path, force=True)
                response_gen = await do_call()
            else:
                raise e

        if kwargs.get("stream", False):
            return response_gen
        else:
            # Accumulate stream for non-streaming response
            chunks = [chunk async for chunk in response_gen]
            return litellm.utils.stream_to_completion_response(chunks)

    # [NEW] Hardcoded model list based on Kilo example
    HARDCODED_MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ]
    # Use the shared GeminiAuthBase for auth logic
    # get_models is not applicable for this custom provider
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Returns a hardcoded list of known compatible Gemini CLI models."""
        return [f"gemini_cli/{model_id}" for model_id in HARDCODED_MODELS]
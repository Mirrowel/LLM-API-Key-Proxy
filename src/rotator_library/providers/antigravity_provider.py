# src/rotator_library/providers/antigravity_provider.py

import json
import httpx
import logging
import time
import asyncio
import random
import uuid
import copy
from typing import List, Dict, Any, AsyncGenerator, Union, Optional, Tuple
from .provider_interface import ProviderInterface
from .antigravity_auth_base import AntigravityAuthBase
from ..model_definitions import ModelDefinitions
import litellm
from litellm.exceptions import RateLimitError
from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

lib_logger = logging.getLogger('rotator_library')

# Antigravity base URLs with fallback order
# Priority: daily (sandbox) → autopush (sandbox) → production
BASE_URLS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com/v1internal",
    "https://cloudcode-pa.googleapis.com/v1internal"  # Production fallback
]

# Hardcoded models available via Antigravity
HARDCODED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-computer-use-preview-10-2025"
]


class AntigravityProvider(AntigravityAuthBase, ProviderInterface):
    """
    Antigravity provider implementation for Gemini models.
    
    Antigravity is an experimental internal Google API that provides access to Gemini models
    including Gemini 3 with thinking/reasoning capabilities. It wraps standard Gemini API
    requests with additional metadata and uses sandbox endpoints.
    
    Key features:
    - Model aliasing (gemini-3-pro-high ↔ gemini-3-pro-preview)
    - Gemini 3 thinkingLevel support
    - Thinking signature preservation for multi-turn conversations
    - Sophisticated tool response grouping
    - Base URL fallback (sandbox → production)
    """
    skip_cost_calculation = True

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()
        self._current_base_url = BASE_URLS[0]  # Start with daily sandbox
        self._base_url_index = 0

    # ============================================================================
    # MODEL ALIAS SYSTEM
    # ============================================================================

    def _model_name_to_alias(self, model_name: str) -> str:
        """
        Convert internal Antigravity model names to public aliases.
        
        Args:
            model_name: Internal model name
            
        Returns:
            Public alias name, or empty string if model should be excluded
        """
        alias_map = {
            "rev19-uic3-1p": "gemini-2.5-computer-use-preview-10-2025",
            "gemini-3-pro-image": "gemini-3-pro-image-preview",
            "gemini-3-pro-high": "gemini-3-pro-preview",
            "claude-sonnet-4-5": "gemini-claude-sonnet-4-5",
            "claude-sonnet-4-5-thinking": "gemini-claude-sonnet-4-5-thinking",
        }
        
        # Filter out excluded models (return empty string to skip)
        excluded = [
            "chat_20706", "chat_23310", "gemini-2.5-flash-thinking",
            "gemini-3-pro-low", "gemini-2.5-pro"
        ]
        if model_name in excluded:
            return ""
        
        return alias_map.get(model_name, model_name)

    def _alias_to_model_name(self, alias: str) -> str:
        """
        Convert public aliases to internal Antigravity model names.
        
        Args:
            alias: Public alias name
            
        Returns:
            Internal model name
        """
        reverse_map = {
            "gemini-2.5-computer-use-preview-10-2025": "rev19-uic3-1p",
            "gemini-3-pro-image-preview": "gemini-3-pro-image",
            "gemini-3-pro-preview": "gemini-3-pro-high",
            "gemini-claude-sonnet-4-5": "claude-sonnet-4-5",
            "gemini-claude-sonnet-4-5-thinking": "claude-sonnet-4-5-thinking",
        }
        return reverse_map.get(alias, alias)

    # ============================================================================
    # RANDOM ID GENERATION
    # ============================================================================

    @staticmethod
    def generate_request_id() -> str:
        """Generate Antigravity request ID: agent-{uuid}"""
        return f"agent-{uuid.uuid4()}"

    @staticmethod
    def generate_session_id() -> str:
        """Generate Antigravity session ID: -{random_number}"""
        # Generate random 19-digit number
        n = random.randint(1_000_000_000_000_000_000, 9_999_999_999_999_999_999)
        return f"-{n}"

    @staticmethod
    def generate_project_id() -> str:
        """Generate fake project ID: {adj}-{noun}-{random}"""
        adjectives = ["useful", "bright", "swift", "calm", "bold"]
        nouns = ["fuze", "wave", "spark", "flow", "core"]
        adj = random.choice(adjectives)
        noun = random.choice(nouns)
        random_part = str(uuid.uuid4())[:5].lower()
        return f"{adj}-{noun}-{random_part}"

    # ============================================================================
    # MESSAGE TRANSFORMATION (OpenAI → Gemini CLI format)
    # ============================================================================

    def _transform_messages(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform OpenAI messages to Gemini CLI format.
        Reused from GeminiCliProvider with modifications for Antigravity.
        
        Returns:
            Tuple of (system_instruction, gemini_contents)
        """
        system_instruction = None
        gemini_contents = []
        
        # Make a copy to avoid modifying original
        messages = copy.deepcopy(messages)
        
        # Separate system prompt from other messages
        if messages and messages[0].get('role') == 'system':
            system_prompt_content = messages.pop(0).get('content', '')
            if system_prompt_content:
                system_instruction = {
                    "role": "user",
                    "parts": [{"text": system_prompt_content}]
                }

        # Build tool call ID to name mapping
        tool_call_id_to_name = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    if tool_call.get("type") == "function":
                        tool_call_id_to_name[tool_call["id"]] = tool_call["function"]["name"]

        # Convert each message
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            parts = []
            gemini_role = "model" if role == "assistant" else "tool" if role == "tool" else "user"

            if role == "user":
                if isinstance(content, str):
                    # Simple text content
                    if content:
                        parts.append({"text": content})
                elif isinstance(content, list):
                    # Multi-part content (text, images, etc.)
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                parts.append({"text": text})
                        elif item.get("type") == "image_url":
                            # Handle image data URLs
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                try:
                                    # Parse: data:image/png;base64,iVBORw0KG...
                                    header, data = image_url.split(",", 1)
                                    mime_type = header.split(":")[1].split(";")[0]
                                    parts.append({
                                        "inlineData": {
                                            "mimeType": mime_type,
                                            "data": data
                                        }
                                    })
                                except Exception as e:
                                    lib_logger.warning(f"Failed to parse image data URL: {e}")

            elif role == "assistant":
                if isinstance(content, str) and content:
                    parts.append({"text": content})
                if msg.get("tool_calls"):
                    for tool_call in msg["tool_calls"]:
                        if tool_call.get("type") == "function":
                            try:
                                args_dict = json.loads(tool_call["function"]["arguments"])
                            except (json.JSONDecodeError, TypeError):
                                args_dict = {}
                            
                            # Add function call part with thoughtSignature
                            func_call_part = {
                                "functionCall": {
                                    "name": tool_call["function"]["name"],
                                    "args": args_dict
                                },
                                "thoughtSignature": "skip_thought_signature_validator"
                            }
                            parts.append(func_call_part)

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                function_name = tool_call_id_to_name.get(tool_call_id)
                if function_name:
                    # Wrap the tool response in a 'result' object
                    response_content = {"result": content}
                    parts.append({"functionResponse": {"name": function_name, "response": response_content}})

            if parts:
                gemini_contents.append({"role": gemini_role, "parts": parts})

        # Ensure first message is from user
        if not gemini_contents or gemini_contents[0]['role'] != 'user':
            gemini_contents.insert(0, {"role": "user", "parts": [{"text": ""}]})

        return system_instruction, gemini_contents

    # ============================================================================
    # THINKING/REASONING CONFIGURATION
    # ============================================================================

    def _map_reasoning_effort_to_thinking_config(
        self,
        reasoning_effort: Optional[str],
        model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Map OpenAI reasoning_effort to Gemini thinking configuration.
        Handles Gemini 3 thinkingLevel vs other models thinkingBudget.
        
        Args:
            reasoning_effort: OpenAI reasoning_effort value
            model: Model name (public alias)
            
        Returns:
            Dictionary with thinkingConfig or None
        """
        internal_model = self._alias_to_model_name(model)
        is_gemini_3 = internal_model.startswith("gemini-3-")
        
        # Default for gemini-3-pro-preview when no reasoning_effort specified
        if not reasoning_effort:
            if model == "gemini-3-pro-preview" or internal_model == "gemini-3-pro-high":
                return {
                    "thinkingBudget": -1,
                    "include_thoughts": True
                }
            return None
        
        if reasoning_effort == "none":
            return {
                "thinkingBudget": 0,
                "include_thoughts": False
            }
        
        if reasoning_effort == "auto":
            # Auto always uses thinkingBudget=-1, even for Gemini 3
            return {
                "thinkingBudget": -1,
                "include_thoughts": True
            }
        
        if is_gemini_3:
            # Gemini 3: Use thinkingLevel
            level_map = {
                "low": "low",
                "medium": "high",  # Medium not released yet, map to high
                "high": "high"
            }
            level = level_map.get(reasoning_effort, "high")
            return {
                "thinkingLevel": level,
                "include_thoughts": True
            }
        else:
            # Non-Gemini-3: Use thinkingBudget with normalization
            budget_map = {
                "low": 1024,
                "medium": 8192,
                "high": 32768
            }
            budget = budget_map.get(reasoning_effort, -1)
            # TODO: Add model-specific normalization via model registry
            return {
                "thinkingBudget": budget,
                "include_thoughts": True
            }

    # ============================================================================
    # TOOL RESPONSE GROUPING
    # ============================================================================

    def _fix_tool_response_grouping(self, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group function calls with their responses for Antigravity compatibility.
        
        Converts linear format (function call, response, function call, response)
        to grouped format (model with calls, function role with all responses).
        
        Args:
            contents: List of Gemini content objects
            
        Returns:
            List of grouped content objects
        """
        new_contents = []
        pending_groups = []  # Groups awaiting responses
        collected_responses = []  # Standalone responses to match
        
        for content in contents:
            role = content.get("role")
            parts = content.get("parts", [])
            
            # Check if this content has function responses
            response_parts = [p for p in parts if "functionResponse" in p]
            
            if response_parts:
                # Collect responses
                collected_responses.extend(response_parts)
                
                # Try to satisfy pending groups
                for i in range(len(pending_groups) - 1, -1, -1):
                    group = pending_groups[i]
                    if len(collected_responses) >= group["responses_needed"]:
                        # Take needed responses
                        group_responses = collected_responses[:group["responses_needed"]]
                        collected_responses = collected_responses[group["responses_needed"]:]
                        
                        # Create merged function response content
                        function_response_content = {
                            "parts": group_responses,
                            "role": "function"  # Changed from tool
                        }
                        new_contents.append(function_response_content)
                        
                        # Remove satisfied group
                        pending_groups.pop(i)
                        break
                
                continue  # Skip adding this content
            
            # If this is model content with function calls, create a group
            if role == "model":
                function_calls = [p for p in parts if "functionCall" in p]
                
                if function_calls:
                    # Add model content first
                    new_contents.append(content)
                    
                    # Create pending group
                    pending_groups.append({
                        "model_content": content,
                        "function_calls": function_calls,
                        "responses_needed": len(function_calls)
                    })
                else:
                    # Regular model content without function calls
                    new_contents.append(content)
            else:
                # Non-model content (user, etc.)
                new_contents.append(content)
        
        # Handle remaining pending groups
        for group in pending_groups:
            if len(collected_responses) >= group["responses_needed"]:
                group_responses = collected_responses[:group["responses_needed"]]
                collected_responses = collected_responses[group["responses_needed"]:]
                
                function_response_content = {
                    "parts": group_responses,
                    "role": "function"
                }
                new_contents.append(function_response_content)
        
        return new_contents

    # ============================================================================
    # ANTIGRAVITY REQUEST TRANSFORMATION
    # ============================================================================

    def _transform_to_antigravity_format(
        self,
        gemini_cli_payload: Dict[str, Any],
        model: str
    ) -> Dict[str, Any]:
        """
        Transform Gemini CLI format to complete Antigravity format.
        
        Args:
            gemini_cli_payload: Request in Gemini CLI format
            model: Model name (public alias)
            
        Returns:
            Complete Antigravity request payload
        """
        internal_model = self._alias_to_model_name(model)
        
        # 1. Wrap in Antigravity envelope
        antigravity_payload = {
            "project": self.generate_project_id(),
            "userAgent": "antigravity",
            "requestId": self.generate_request_id(),
            "model": internal_model,  # Use internal name
            "request": copy.deepcopy(gemini_cli_payload)
        }
        
        # 2. Add session ID
        antigravity_payload["request"]["sessionId"] = self.generate_session_id()
        
        # 3. Remove fields that Antigravity doesn't support
        antigravity_payload["request"].pop("safetySettings", None)
        if "generationConfig" in antigravity_payload["request"]:
            antigravity_payload["request"]["generationConfig"].pop("maxOutputTokens", None)
        
        # 4. Set toolConfig mode
        if "toolConfig" not in antigravity_payload["request"]:
            antigravity_payload["request"]["toolConfig"] = {}
        if "functionCallingConfig" not in antigravity_payload["request"]["toolConfig"]:
            antigravity_payload["request"]["toolConfig"]["functionCallingConfig"] = {}
        antigravity_payload["request"]["toolConfig"]["functionCallingConfig"]["mode"] = "VALIDATED"
        
        # 5. Handle Gemini 3 specific thinking logic
        # For non-Gemini-3 models, convert thinkingLevel to thinkingBudget
        if not internal_model.startswith("gemini-3-"):
            gen_config = antigravity_payload["request"].get("generationConfig", {})
            thinking_config = gen_config.get("thinkingConfig", {})
            if "thinkingLevel" in thinking_config:
                # Remove thinkingLevel for non-Gemini-3 models
                del thinking_config["thinkingLevel"]
                # Set thinkingBudget to -1 (auto/dynamic)
                thinking_config["thinkingBudget"] = -1
        
        # 6. Preserve/add thoughtSignature to ALL function calls in model role content
        for content in antigravity_payload["request"].get("contents", []):
            if content.get("role") == "model":
                for part in content.get("parts", []):
                    # Add signature to function calls OR preserve if already exists
                    if "functionCall" in part and "thoughtSignature" not in part:
                        part["thoughtSignature"] = "skip_thought_signature_validator"
                    # If thoughtSignature already exists, preserve it (important for Gemini 3)
        
        # 7. Handle Claude models (special tool schema conversion)
        if internal_model.startswith("claude-sonnet-"):
            # For Claude models, convert parametersJsonSchema back to parameters
            for tool in antigravity_payload["request"].get("tools", []):
                for func_decl in tool.get("functionDeclarations", []):
                    if "parametersJsonSchema" in func_decl:
                        func_decl["parameters"] = func_decl.pop("parametersJsonSchema")
                        # Remove $schema if present
                        if "parameters" in func_decl and "$schema" in func_decl["parameters"]:
                            del func_decl["parameters"]["$schema"]
        
        return antigravity_payload

    #============================================================================
    # BASE URL FALLBACK LOGIC
    # ============================================================================

    def _get_current_base_url(self) -> str:
        """Get the current base URL from the fallback list."""
        return self._current_base_url

    def _try_next_base_url(self) -> bool:
        """
        Switch to the next base URL in the fallback list.
        
        Returns:
            True if successfully switched to next URL, False if no more URLs available
        """
        if self._base_url_index < len(BASE_URLS) - 1:
            self._base_url_index += 1
            self._current_base_url = BASE_URLS[self._base_url_index]
            lib_logger.info(f"Switching to fallback Antigravity base URL: {self._current_base_url}")
            return True
        return False

    def _reset_base_url(self):
        """Reset to the primary base URL (daily sandbox)."""
        self._base_url_index = 0
        self._current_base_url = BASE_URLS[0]

    # ============================================================================
    # RESPONSE TRANSFORMATION (Antigravity → OpenAI)
    # ============================================================================

    def _unwrap_antigravity_response(self, antigravity_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Gemini response from Antigravity envelope.
        
        Args:
            antigravity_response: Response from Antigravity API
            
        Returns:
            Gemini response (unwrapped)
        """
        # For both streaming and non-streaming, response is in 'response' field
        return antigravity_response.get("response", antigravity_response)

    def _gemini_to_openai_chunk(self, gemini_chunk: Dict[str, Any], model: str) -> litellm.ModelResponse:
        """
        Convert a single Gemini response chunk to OpenAI format.
        Based on GeminiCliProvider logic.
        
        Args:
            gemini_chunk: Gemini response chunk
            model: Model name
            
        Returns:
            OpenAI-format ModelResponse
        """
        # Extract candidate
        candidates = gemini_chunk.get("candidates", [])
        if not candidates:
            # Empty chunk, return minimal response
            return litellm.ModelResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=model,
                choices=[]
            )
        
        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        
        # Extract text, tool calls, and thinking
        text_content = ""
        tool_calls = []
        
        for part in content_parts:
            # Extract text
            if "text" in part:
                text_content += part["text"]
            
            # Extract function calls (tool calls)
            if "functionCall" in part:
                func_call = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("args", {}))
                    }
                })
        
        # Build delta
        delta = {}
        if text_content:
            delta["content"] = text_content
        if tool_calls:
            delta["tool_calls"] = tool_calls
        
        # Get finish reason
        finish_reason = candidate.get("finishReason", "").lower() if candidate.get("finishReason") else None
        if finish_reason == "stop":
            finish_reason = "stop"
        elif finish_reason == "max_tokens":
            finish_reason = "length"
        
        # Build choice
        choice = {
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }
        
        # Extract usage (if present)
        usage_metadata = gemini_chunk.get("usageMetadata", {})
        usage = None
        if usage_metadata:
            usage = {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0)
            }
        
        return litellm.ModelResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage
        )

    # ============================================================================
    # PROVIDER INTERFACE IMPLEMENTATION
    # ============================================================================

    def has_custom_logic(self) -> bool:
        """Antigravity uses custom translation logic."""
        return True

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Get OAuth authorization header for Antigravity.
        
        Args:
            credential_identifier: Credential file path or "env"
            
        Returns:
            Dict with Authorization header
        """
        access_token = await self.get_valid_token(credential_identifier)
        return {"Authorization": f"Bearer {access_token}"}

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from Antigravity.
        
        For Antigravity, we use the fetchAvailableModels endpoint and apply
        alias mapping to convert internal names to public names.
        
        Args:
            api_key: Credential path (not a traditional API key)
            client: HTTP client
            
        Returns:
            List of public model names
        """
        credential_path = api_key  # For OAuth providers, this is the credential path
        
        try:
            access_token = await self.get_valid_token(credential_path)
            base_url = self._get_current_base_url()
            
            # Generate required IDs
            project_id = self.generate_project_id()
            request_id = self.generate_request_id()
            
            # Fetch models endpoint
            url = f"{base_url}/fetchAvailableModels"
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "project": project_id,
                "requestId": request_id,
                "userAgent": "antigravity"
            }
            
            lib_logger.debug(f"Fetching Antigravity models from: {url}")
            
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract model names and apply aliasing
            models = []
            if "models" in data:
                for model_info in data["models"]:
                    internal_name = model_info.get("name", "").replace("models/", "")
                    if internal_name:
                        public_name = self._model_name_to_alias(internal_name)
                        if public_name:  # Skip excluded models (empty string)
                            models.append(public_name)
            
            if models:
                lib_logger.info(f"Discovered {len(models)} Antigravity models")
                return models
            else:
                lib_logger.warning("No models returned from Antigravity, using hardcoded list")
                return HARDCODED_MODELS
                
        except Exception as e:
            lib_logger.warning(f"Failed to fetch Antigravity models: {e}, using hardcoded list")
            return HARDCODED_MODELS

    async def acompletion(
        self,
        client: httpx.AsyncClient,
        **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion requests for Antigravity.
        
        This is the main entry point that:
        1. Extracts the model and credential path
        2. Transforms OpenAI request → Gemini CLI → Antigravity format
        3. Makes the API call with fallback logic
        4. Transforms Antigravity response → Gemini → OpenAI format
        
        Args:
            client: HTTP client
            **kwargs: LiteLLM completion parameters
            
        Returns:
            ModelResponse (non-streaming) or AsyncGenerator (streaming)
        """
        # Extract key parameters
        model = kwargs.get("model", "gemini-2.5-pro")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        credential_path = kwargs.pop("credential_identifier", kwargs.get("api_key", ""))
        tools = kwargs.get("tools")
        reasoning_effort = kwargs.get("reasoning_effort")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p")
        max_tokens = kwargs.get("max_tokens")
        
        lib_logger.info(f"Antigravity completion: model={model}, stream={stream}, messages={len(messages)}")
        
        # Step 1: Transform messages (OpenAI → Gemini CLI)
        system_instruction, gemini_contents = self._transform_messages(messages)
        
        # Apply tool response grouping
        gemini_contents = self._fix_tool_response_grouping(gemini_contents)
        
        # Step 2: Build Gemini CLI payload
        gemini_cli_payload = {
            "contents": gemini_contents
        }
        
        if system_instruction:
            gemini_cli_payload["system_instruction"] = system_instruction
        
        # Add generation config
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if top_p is not None:
            generation_config["topP"] = top_p
        
        # Handle thinking config
        thinking_config = self._map_reasoning_effort_to_thinking_config(reasoning_effort, model)
        if thinking_config:
            generation_config.setdefault("thinkingConfig", {}).update(thinking_config)
        
        if generation_config:
            gemini_cli_payload["generationConfig"] = generation_config
        
        # Add tools
        if tools:
            gemini_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    schema = _build_vertex_schema(parameters=func.get("parameters", {}))
                    gemini_tools.append({
                        "functionDeclarations": [{
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "parametersJsonSchema": schema
                        }]
                    })
            if gemini_tools:
                gemini_cli_payload["tools"] = gemini_tools
        
        # Step 3: Transform to Antigravity format
        antigravity_payload = self._transform_to_antigravity_format(gemini_cli_payload, model)
        
        # Step 4: Make API call
        access_token = await self.get_valid_token(credential_path)
        base_url = self._get_current_base_url()
        
        endpoint = ":streamGenerateContent" if stream else ":generateContent"
        url = f"{base_url}{endpoint}"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        lib_logger.debug(f"Antigravity request to: {url}")
        
        try:
            if stream:
                return self._handle_streaming(client, url, headers, antigravity_payload, model)
            else:
                return await self._handle_non_streaming(client, url, headers, antigravity_payload, model)
        except Exception as e:
            # Try fallback URL if available
            if self._try_next_base_url():
                lib_logger.warning(f"Retrying Antigravity request with fallback URL: {e}")
                base_url = self._get_current_base_url()
                url = f"{base_url}{endpoint}"
                
                if stream:
                    return self._handle_streaming(client, url, headers, antigravity_payload, model)
                else:
                    return await self._handle_non_streaming(client, url, headers, antigravity_payload, model)
            else:
                raise

    async def _handle_non_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str
    ) -> litellm.ModelResponse:
        """Handle non-streaming completion."""
        response = await client.post(url, headers=headers, json=payload, timeout=120.0)
        response.raise_for_status()
        
        antigravity_response = response.json()
        
        # Unwrap Antigravity envelope
        gemini_response = self._unwrap_antigravity_response(antigravity_response)
        
        # Convert to OpenAI format
        return self._gemini_to_openai_chunk(gemini_response, model)

    async def _handle_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming completion."""
        async with client.stream("POST", url, headers=headers, json=payload, timeout=120.0) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        antigravity_chunk = json.loads(data_str)
                        
                        # Unwrap Antigravity envelope
                        gemini_chunk = self._unwrap_antigravity_response(antigravity_chunk)
                        
                        # Convert to OpenAI format
                        openai_chunk = self._gemini_to_openai_chunk(gemini_chunk, model)
                        
                        yield openai_chunk
                    except json.JSONDecodeError:
                        lib_logger.warning(f"Failed to parse Antigravity chunk: {data_str[:100]}")
                        continue

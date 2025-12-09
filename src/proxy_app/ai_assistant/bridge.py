"""
LLM Bridge for the AI Assistant.

Provides the bridge between the async RotatingClient and the synchronous GUI thread.
Handles streaming, tool call parsing, and model list fetching.
"""

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tools import ToolCall

logger = logging.getLogger(__name__)


@dataclass
class StreamCallbacks:
    """Callbacks for streaming response handling."""

    on_thinking_chunk: Optional[Callable[[str], None]] = None
    on_content_chunk: Optional[Callable[[str], None]] = None
    on_tool_calls: Optional[Callable[[List[ToolCall]], None]] = None
    on_error: Optional[Callable[[str], None]] = None
    on_complete: Optional[Callable[[], None]] = None


@dataclass
class ParsedChunk:
    """Parsed data from a streaming chunk."""

    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    is_done: bool = False


class LLMBridge:
    """
    Bridge between async RotatingClient and synchronous GUI thread.

    Handles:
    - RotatingClient lifecycle management
    - Thread/async coordination using threading.Thread + asyncio.run()
    - Streaming chunk processing
    - Tool call parsing
    - Model list fetching
    """

    def __init__(
        self,
        schedule_on_gui: Callable[[Callable], None],
        ignore_models: Optional[Dict[str, List[str]]] = None,
        whitelist_models: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the LLM Bridge.

        Args:
            schedule_on_gui: Function to schedule callbacks on the GUI thread
                             (typically window.after(0, callback))
            ignore_models: Model patterns to ignore (passed to RotatingClient)
            whitelist_models: Model patterns to whitelist (passed to RotatingClient)
        """
        self._schedule_on_gui = schedule_on_gui
        self._ignore_models = ignore_models
        self._whitelist_models = whitelist_models
        self._client = None
        self._current_thread: Optional[threading.Thread] = None
        self._cancel_requested = False
        self._models_cache: Optional[Dict[str, List[str]]] = None

    def _get_client(self):
        """Get or create the RotatingClient instance."""
        if self._client is None:
            # Import here to avoid circular imports and reduce startup time
            import os

            from dotenv import load_dotenv

            from rotator_library import RotatingClient
            from rotator_library.credential_manager import CredentialManager

            # Load environment variables
            load_dotenv(override=True)

            # Discover API keys from environment variables (same as main.py)
            api_keys = {}
            for key, value in os.environ.items():
                if "_API_KEY" in key and key != "PROXY_API_KEY":
                    provider = key.split("_API_KEY")[0].lower()
                    if provider not in api_keys:
                        api_keys[provider] = []
                    api_keys[provider].append(value)

            # Discover OAuth credentials via CredentialManager
            cred_manager = CredentialManager(os.environ)
            oauth_credentials = cred_manager.discover_and_prepare()

            # Discover model filtering rules from environment (same as main.py)
            ignore_models = self._ignore_models or {}
            whitelist_models = self._whitelist_models or {}

            # Load per-provider ignore/whitelist from env vars
            for key, value in os.environ.items():
                if key.startswith("IGNORE_MODELS_") and value:
                    provider = key.replace("IGNORE_MODELS_", "").lower()
                    patterns = [p.strip() for p in value.split(",") if p.strip()]
                    if patterns:
                        ignore_models[provider] = patterns
                elif key.startswith("WHITELIST_MODELS_") and value:
                    provider = key.replace("WHITELIST_MODELS_", "").lower()
                    patterns = [p.strip() for p in value.split(",") if p.strip()]
                    if patterns:
                        whitelist_models[provider] = patterns

            self._client = RotatingClient(
                api_keys=api_keys,
                oauth_credentials=oauth_credentials,
                ignore_models=ignore_models if ignore_models else None,
                whitelist_models=whitelist_models if whitelist_models else None,
                configure_logging=False,  # Use existing logging config
            )
        return self._client

    def cancel_stream(self) -> None:
        """Request cancellation of the current stream."""
        self._cancel_requested = True
        logger.info("Stream cancellation requested")

    def is_streaming(self) -> bool:
        """Check if a stream is currently in progress."""
        return self._current_thread is not None and self._current_thread.is_alive()

    def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: str,
        callbacks: StreamCallbacks,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """
        Start a streaming completion request in a background thread.

        Args:
            messages: The message history in OpenAI format
            tools: Tool definitions in OpenAI format
            model: The model to use (e.g., "openai/gpt-4o")
            callbacks: Callbacks for handling streaming events
            reasoning_effort: Optional reasoning effort level ("low", "medium", "high")
        """
        self._cancel_requested = False

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(
                    self._stream_async(
                        messages, tools, model, callbacks, reasoning_effort
                    )
                )
            except Exception as e:
                logger.exception("Error in streaming thread")
                if callbacks.on_error:
                    self._schedule_on_gui(lambda: callbacks.on_error(str(e)))
            finally:
                # Clean up pending tasks
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                finally:
                    loop.close()

        self._current_thread = threading.Thread(target=run_in_thread, daemon=True)
        self._current_thread.start()

    async def _stream_async(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: str,
        callbacks: StreamCallbacks,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Async implementation of streaming completion."""
        client = self._get_client()

        # Accumulate tool calls across chunks
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

        try:
            # Build completion kwargs
            completion_kwargs = {
                "model": model,
                "messages": messages,
                "tools": tools if tools else None,
                "stream": True,
            }

            # Add reasoning_effort if specified
            if reasoning_effort and reasoning_effort in ("low", "medium", "high"):
                completion_kwargs["reasoning_effort"] = reasoning_effort

            response = client.acompletion(**completion_kwargs)

            async for chunk in response:
                if self._cancel_requested:
                    logger.info("Stream cancelled by user")
                    break

                parsed = self._parse_chunk(chunk)

                if parsed.is_done:
                    break

                # Handle reasoning/thinking content
                if parsed.reasoning_content and callbacks.on_thinking_chunk:
                    content = parsed.reasoning_content
                    self._schedule_on_gui(
                        lambda c=content: callbacks.on_thinking_chunk(c)
                    )

                # Handle regular content
                if parsed.content and callbacks.on_content_chunk:
                    content = parsed.content
                    self._schedule_on_gui(
                        lambda c=content: callbacks.on_content_chunk(c)
                    )

                # Accumulate tool calls
                if parsed.tool_calls:
                    for tc in parsed.tool_calls:
                        index = tc.get("index", 0)
                        if index not in accumulated_tool_calls:
                            accumulated_tool_calls[index] = {
                                "id": tc.get("id", ""),
                                "name": "",
                                "arguments": "",
                            }

                        # Accumulate ID if provided
                        if tc.get("id"):
                            accumulated_tool_calls[index]["id"] = tc["id"]

                        # Accumulate function name and arguments
                        func = tc.get("function", {})
                        if func.get("name"):
                            accumulated_tool_calls[index]["name"] = func["name"]
                        if func.get("arguments"):
                            accumulated_tool_calls[index]["arguments"] += func[
                                "arguments"
                            ]

            # Process accumulated tool calls
            if accumulated_tool_calls and callbacks.on_tool_calls:
                tool_calls = []
                for index in sorted(accumulated_tool_calls.keys()):
                    tc_data = accumulated_tool_calls[index]
                    try:
                        arguments = (
                            json.loads(tc_data["arguments"])
                            if tc_data["arguments"]
                            else {}
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse tool arguments: {tc_data['arguments']}"
                        )
                        arguments = {}

                    tool_calls.append(
                        ToolCall(
                            id=tc_data["id"],
                            name=tc_data["name"],
                            arguments=arguments,
                        )
                    )

                if tool_calls:
                    self._schedule_on_gui(
                        lambda tc=tool_calls: callbacks.on_tool_calls(tc)
                    )

            # Signal completion
            if callbacks.on_complete and not self._cancel_requested:
                self._schedule_on_gui(callbacks.on_complete)

        except Exception as e:
            logger.exception("Error during streaming")
            if callbacks.on_error:
                error_msg = self._format_error(e)
                self._schedule_on_gui(lambda msg=error_msg: callbacks.on_error(msg))

    def _parse_chunk(self, chunk: str) -> ParsedChunk:
        """
        Parse a streaming chunk.

        Args:
            chunk: SSE-formatted chunk string (e.g., "data: {...}\n\n")

        Returns:
            ParsedChunk with extracted data
        """
        result = ParsedChunk()

        # Handle SSE format
        if isinstance(chunk, str):
            chunk = chunk.strip()
            if chunk.startswith("data: "):
                chunk = chunk[6:]

            if chunk == "[DONE]":
                result.is_done = True
                return result

            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                return result
        elif hasattr(chunk, "choices"):
            # It's already a parsed object (litellm response)
            data = chunk
        else:
            return result

        # Extract from choices
        if hasattr(data, "choices"):
            choices = data.choices
        elif isinstance(data, dict):
            choices = data.get("choices", [])
        else:
            return result

        if not choices:
            return result

        choice = choices[0] if isinstance(choices, list) else choices

        # Get delta (for streaming) or message
        if hasattr(choice, "delta"):
            delta = choice.delta
        elif isinstance(choice, dict):
            delta = choice.get("delta", choice.get("message", {}))
        else:
            delta = {}

        # Extract content
        if hasattr(delta, "content"):
            result.content = delta.content
        elif isinstance(delta, dict):
            result.content = delta.get("content")

        # Extract reasoning content (for models that support it)
        if hasattr(delta, "reasoning_content"):
            result.reasoning_content = delta.reasoning_content
        elif isinstance(delta, dict):
            result.reasoning_content = delta.get("reasoning_content")

        # Extract tool calls
        if hasattr(delta, "tool_calls"):
            tool_calls = delta.tool_calls
            if tool_calls:
                result.tool_calls = [
                    {
                        "index": getattr(tc, "index", i),
                        "id": getattr(tc, "id", None),
                        "function": {
                            "name": getattr(tc.function, "name", None)
                            if hasattr(tc, "function")
                            else None,
                            "arguments": getattr(tc.function, "arguments", "")
                            if hasattr(tc, "function")
                            else "",
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ]
        elif isinstance(delta, dict) and "tool_calls" in delta:
            result.tool_calls = delta["tool_calls"]

        # Extract finish reason
        if hasattr(choice, "finish_reason"):
            result.finish_reason = choice.finish_reason
        elif isinstance(choice, dict):
            result.finish_reason = choice.get("finish_reason")

        return result

    def _format_error(self, error: Exception) -> str:
        """Format an exception into a user-friendly error message."""
        error_str = str(error)

        # Check for common error types
        if "rate_limit" in error_str.lower() or "429" in error_str:
            return "Rate limit exceeded. Please try again in a moment."
        elif "quota" in error_str.lower():
            return "API quota exceeded. Please check your API usage limits."
        elif "authentication" in error_str.lower() or "401" in error_str:
            return "Authentication failed. Please check your API credentials."
        elif "connection" in error_str.lower() or "network" in error_str.lower():
            return "Connection error. Please check your network connection."
        elif "timeout" in error_str.lower():
            return "Request timed out. Please try again."
        elif "all_credentials_exhausted" in error_str.lower():
            return "All API credentials exhausted. Please add more credentials or wait for rate limits to reset."

        # Return truncated error for unknown errors
        if len(error_str) > 200:
            return f"Error: {error_str[:200]}..."
        return f"Error: {error_str}"

    def fetch_models(
        self,
        on_success: Callable[[Dict[str, List[str]]], None],
        on_error: Callable[[str], None],
    ) -> None:
        """
        Fetch available models in a background thread.

        Models are grouped by provider.

        Args:
            on_success: Callback with dict of provider -> model list
            on_error: Callback with error message
        """

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                client = self._get_client()
                models = loop.run_until_complete(
                    client.get_all_available_models(grouped=True)
                )
                self._models_cache = models
                self._schedule_on_gui(lambda: on_success(models))
            except Exception as e:
                logger.exception("Error fetching models")
                self._schedule_on_gui(lambda: on_error(str(e)))
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                loop.close()

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

    def get_cached_models(self) -> Optional[Dict[str, List[str]]]:
        """Get the cached model list, if available."""
        return self._models_cache

    def close(self) -> None:
        """Close the client and clean up resources."""
        if self._client is not None:
            # Run close in a new event loop since it's async
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._client.close())
                loop.close()
            except Exception as e:
                logger.error(f"Error closing client: {e}")
            finally:
                self._client = None

# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Streaming response handler.

Extracts streaming logic from client.py _safe_streaming_wrapper (lines 904-1117).
Handles:
- Chunk processing with finish_reason logic
- JSON reassembly for fragmented responses
- Error detection in streamed data
- Usage tracking from final chunks
- Client disconnect handling
"""

import codecs
import json
import logging
import re
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Optional, TYPE_CHECKING

import litellm

from ..core.errors import StreamedAPIError, CredentialNeedsReauthError
from ..core.types import ProcessedChunk

if TYPE_CHECKING:
    from ..usage.manager import CredentialContext

lib_logger = logging.getLogger("rotator_library")


class StreamingHandler:
    """
    Process streaming responses with error handling and usage tracking.

    This class extracts the streaming logic that was in _safe_streaming_wrapper
    and provides a clean interface for processing LiteLLM streams.

    Usage recording is handled via CredentialContext passed to wrap_stream().
    """

    async def wrap_stream(
        self,
        stream: AsyncIterator[Any],
        credential: str,
        model: str,
        request: Optional[Any] = None,
        cred_context: Optional["CredentialContext"] = None,
        skip_cost_calculation: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Wrap a LiteLLM stream with error handling and usage tracking.

        FINISH_REASON HANDLING:
        - Strip finish_reason from intermediate chunks (litellm defaults to "stop")
        - Track accumulated_finish_reason with priority: tool_calls > length/content_filter > stop
        - Only emit finish_reason on final chunk (detected by usage.completion_tokens > 0)

        Args:
            stream: The async iterator from LiteLLM
            credential: Credential identifier (for logging)
            model: Model name for usage recording
            request: Optional FastAPI request for disconnect detection
            cred_context: CredentialContext for marking success/failure

        Yields:
            SSE-formatted strings: "data: {...}\\n\\n"
        """
        stream_completed = False
        error_buffer = StreamBuffer()  # Use StreamBuffer for JSON reassembly
        accumulated_finish_reason: Optional[str] = None
        has_tool_calls = False
        prompt_tokens = 0
        prompt_tokens_cached = 0
        prompt_tokens_cache_write = 0
        prompt_tokens_uncached = 0
        completion_tokens = 0
        thinking_tokens = 0

        # Use manual iteration to allow continue after partial JSON errors
        stream_iterator = stream.__aiter__()

        try:
            while True:
                try:
                    # Check client disconnect before waiting for next chunk
                    if request and await request.is_disconnected():
                        lib_logger.info(
                            f"Client disconnected. Aborting stream for model {model}."
                        )
                        break

                    chunk = await stream_iterator.__anext__()

                    # Clear error buffer on successful chunk receipt
                    error_buffer.reset()

                    # Process chunk
                    processed = self._process_chunk(
                        chunk,
                        accumulated_finish_reason,
                        has_tool_calls,
                    )

                    # Update tracking state
                    if processed.has_tool_calls:
                        has_tool_calls = True
                        accumulated_finish_reason = "tool_calls"
                    if processed.finish_reason and not has_tool_calls:
                        # Only update if not already tool_calls (highest priority)
                        accumulated_finish_reason = processed.finish_reason
                    if processed.usage and isinstance(processed.usage, dict):
                        # Extract token counts from final chunk
                        prompt_tokens = processed.usage.get("prompt_tokens", 0)
                        completion_tokens = processed.usage.get("completion_tokens", 0)
                        prompt_details = processed.usage.get("prompt_tokens_details")
                        if prompt_details:
                            if isinstance(prompt_details, dict):
                                prompt_tokens_cached = (
                                    prompt_details.get("cached_tokens", 0) or 0
                                )
                                prompt_tokens_cache_write = (
                                    prompt_details.get("cache_creation_tokens", 0) or 0
                                )
                            else:
                                prompt_tokens_cached = (
                                    getattr(prompt_details, "cached_tokens", 0) or 0
                                )
                                prompt_tokens_cache_write = (
                                    getattr(prompt_details, "cache_creation_tokens", 0)
                                    or 0
                                )
                        completion_details = processed.usage.get(
                            "completion_tokens_details"
                        )
                        if completion_details:
                            if isinstance(completion_details, dict):
                                thinking_tokens = (
                                    completion_details.get("reasoning_tokens", 0) or 0
                                )
                            else:
                                thinking_tokens = (
                                    getattr(completion_details, "reasoning_tokens", 0)
                                    or 0
                                )
                        if processed.usage.get("cache_read_tokens") is not None:
                            prompt_tokens_cached = (
                                processed.usage.get("cache_read_tokens") or 0
                            )
                        if processed.usage.get("cache_creation_tokens") is not None:
                            prompt_tokens_cache_write = (
                                processed.usage.get("cache_creation_tokens") or 0
                            )
                        if thinking_tokens and completion_tokens >= thinking_tokens:
                            completion_tokens = completion_tokens - thinking_tokens
                        prompt_tokens_uncached = max(
                            0, prompt_tokens - prompt_tokens_cached
                        )

                    yield processed.sse_string

                except StopAsyncIteration:
                    # Stream ended normally
                    stream_completed = True
                    break

                except CredentialNeedsReauthError as e:
                    # Credential needs re-auth - wrap for outer retry loop
                    if cred_context:
                        from ..error_handler import classify_error

                        cred_context.mark_failure(classify_error(e))
                    raise StreamedAPIError("Credential needs re-authentication", data=e)

                except json.JSONDecodeError as e:
                    # Partial JSON - accumulate and continue
                    error_buffer.append(str(e))
                    if error_buffer.is_complete:
                        # We have complete JSON now
                        raise StreamedAPIError(
                            "Provider error", data=error_buffer.content
                        )
                    # Continue waiting for more chunks
                    continue

                except Exception as e:
                    # Try to extract JSON from fragmented response
                    error_str = str(e)
                    error_buffer.append(error_str)

                    # Check if buffer now has complete JSON
                    if error_buffer.is_complete:
                        if cred_context:
                            from ..error_handler import classify_error

                            cred_context.mark_failure(classify_error(e))
                        raise StreamedAPIError(
                            "Provider error in stream", data=error_buffer.content
                        )

                    # Try pattern matching for error extraction
                    extracted = self._try_extract_error(e, error_buffer.content)
                    if extracted:
                        if cred_context:
                            from ..error_handler import classify_error

                            cred_context.mark_failure(classify_error(e))
                        raise StreamedAPIError(
                            "Provider error in stream", data=extracted
                        )

                    # Not a JSON-related error, re-raise
                    raise

        except StreamedAPIError:
            # Re-raise for retry loop
            raise

        finally:
            # Record usage if stream completed
            if stream_completed:
                if cred_context:
                    approx_cost = 0.0
                    if not skip_cost_calculation:
                        approx_cost = self._calculate_stream_cost(
                            model,
                            prompt_tokens_uncached + prompt_tokens_cached,
                            completion_tokens + thinking_tokens,
                        )
                    cred_context.mark_success(
                        prompt_tokens=prompt_tokens_uncached,
                        completion_tokens=completion_tokens,
                        thinking_tokens=thinking_tokens,
                        prompt_tokens_cache_read=prompt_tokens_cached,
                        prompt_tokens_cache_write=prompt_tokens_cache_write,
                        approx_cost=approx_cost,
                    )

                # Yield [DONE] for completed streams
                yield "data: [DONE]\n\n"

    def _process_chunk(
        self,
        chunk: Any,
        accumulated_finish_reason: Optional[str],
        has_tool_calls: bool,
    ) -> ProcessedChunk:
        """
        Process a single streaming chunk.

        Handles finish_reason logic:
        - Strip from intermediate chunks
        - Apply correct finish_reason on final chunk

        Args:
            chunk: Raw chunk from LiteLLM
            accumulated_finish_reason: Current accumulated finish reason
            has_tool_calls: Whether any chunk has had tool_calls

        Returns:
            ProcessedChunk with SSE string and metadata
        """
        # Convert chunk to dict
        if hasattr(chunk, "model_dump"):
            chunk_dict = chunk.model_dump()
        elif hasattr(chunk, "dict"):
            chunk_dict = chunk.dict()
        else:
            chunk_dict = chunk

        # Extract metadata before modifying
        usage = chunk_dict.get("usage")
        finish_reason = None
        chunk_has_tool_calls = False

        if "choices" in chunk_dict and chunk_dict["choices"]:
            choice = chunk_dict["choices"][0]
            delta = choice.get("delta", {})

            # Check for tool_calls
            if delta.get("tool_calls"):
                chunk_has_tool_calls = True

            # Detect final chunk: has usage with completion_tokens > 0
            has_completion_tokens = (
                usage
                and isinstance(usage, dict)
                and usage.get("completion_tokens", 0) > 0
            )

            if has_completion_tokens:
                # FINAL CHUNK: Determine correct finish_reason
                if has_tool_calls or chunk_has_tool_calls:
                    choice["finish_reason"] = "tool_calls"
                elif accumulated_finish_reason:
                    choice["finish_reason"] = accumulated_finish_reason
                else:
                    choice["finish_reason"] = "stop"
                finish_reason = choice["finish_reason"]
            else:
                # INTERMEDIATE CHUNK: Never emit finish_reason
                choice["finish_reason"] = None

        return ProcessedChunk(
            sse_string=f"data: {json.dumps(chunk_dict)}\n\n",
            usage=usage,
            finish_reason=finish_reason,
            has_tool_calls=chunk_has_tool_calls,
        )

    def _try_extract_error(
        self,
        exception: Exception,
        buffer: str,
    ) -> Optional[Dict]:
        """
        Try to extract error JSON from exception or buffer.

        Handles multiple error formats:
        - Google-style bytes representation: b'{...}'
        - "Received chunk:" prefix
        - JSON in buffer accumulation

        Args:
            exception: The caught exception
            buffer: Current JSON buffer content

        Returns:
            Parsed error dict or None
        """
        error_str = str(exception)

        # Pattern 1: Google-style bytes representation
        match = re.search(r"b'(\{.*\})'", error_str, re.DOTALL)
        if match:
            try:
                decoded = codecs.decode(match.group(1), "unicode_escape")
                return json.loads(decoded)
            except (json.JSONDecodeError, ValueError):
                pass

        # Pattern 2: "Received chunk:" prefix
        if "Received chunk:" in error_str:
            chunk = error_str.split("Received chunk:")[-1].strip()
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                pass

        # Pattern 3: Buffer accumulation
        if buffer:
            try:
                return json.loads(buffer)
            except json.JSONDecodeError:
                pass

        return None

    def _calculate_stream_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        try:
            model_info = litellm.get_model_info(model)
            input_cost = model_info.get("input_cost_per_token")
            output_cost = model_info.get("output_cost_per_token")
            total_cost = 0.0
            if input_cost:
                total_cost += prompt_tokens * input_cost
            if output_cost:
                total_cost += completion_tokens * output_cost
            return total_cost
        except Exception as exc:
            lib_logger.debug(f"Stream cost calculation failed for {model}: {exc}")
            return 0.0


class StreamBuffer:
    """
    Buffer for reassembling fragmented JSON in streams.

    Some providers send JSON split across multiple chunks, especially
    for error responses. This class handles accumulation and parsing.
    """

    def __init__(self):
        self._buffer = ""
        self._complete = False

    def append(self, chunk: str) -> Optional[Dict]:
        """
        Append a chunk and try to parse.

        Args:
            chunk: Raw chunk string

        Returns:
            Parsed dict if complete, None if still accumulating
        """
        self._buffer += chunk

        try:
            result = json.loads(self._buffer)
            self._complete = True
            return result
        except json.JSONDecodeError:
            return None

    def reset(self) -> None:
        """Reset the buffer."""
        self._buffer = ""
        self._complete = False

    @property
    def content(self) -> str:
        """Get current buffer content."""
        return self._buffer

    @property
    def is_complete(self) -> bool:
        """Check if buffer contains complete JSON."""
        return self._complete

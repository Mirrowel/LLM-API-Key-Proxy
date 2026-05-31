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
import time
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, List, Optional, TYPE_CHECKING

from ..core.errors import StreamedAPIError, CredentialNeedsReauthError
from ..core.types import ProcessedChunk
from ..core.utils import normalize_usage_for_response
from ..streaming import StreamEvent, StreamMonitor, stream_event_from_sse_chunk
from ..usage.accounting import UsageRecord, extract_usage_record
from ..usage.costs import CostBreakdown, CostCalculator

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
        response_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        transaction_logger: Optional[Any] = None,
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
        usage_record = UsageRecord(source="stream")
        assistant_parts: List[str] = []
        tool_call_ids: List[str] = []
        monitor = StreamMonitor(clock=time.monotonic)
        self._log_stream_lifecycle(
            transaction_logger,
            "stream_started",
            monitor,
            StreamEvent("started", protocol="openai_chat"),
        )

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

                    raw_event = StreamEvent(
                        "raw_chunk",
                        protocol="openai_chat",
                        raw=chunk,
                        data={"chunk_index": monitor.metrics.chunk_count + 1},
                    )
                    if monitor.metrics.first_byte_at is None:
                        self._log_stream_lifecycle(
                            transaction_logger,
                            "stream_first_byte",
                            monitor,
                            raw_event,
                        )

                    # Clear error buffer on successful chunk receipt
                    error_buffer.reset()

                    # Process chunk
                    processed = self._process_chunk(
                        chunk,
                        accumulated_finish_reason,
                        has_tool_calls,
                        model,
                    )
                    self._collect_session_response_anchors(
                        processed.sse_string,
                        assistant_parts,
                        tool_call_ids,
                    )
                    event = stream_event_from_sse_chunk(processed.sse_string)
                    first_visible = (
                        event.visible_output
                        and monitor.metrics.first_visible_output_at is None
                    )
                    monitor.record_event(event)
                    if first_visible:
                        self._log_stream_lifecycle(
                            transaction_logger,
                            "stream_first_visible_output",
                            monitor,
                            event,
                        )

                    # Update tracking state
                    if processed.has_tool_calls:
                        has_tool_calls = True
                        accumulated_finish_reason = "tool_calls"
                    if processed.finish_reason and not has_tool_calls:
                        # Only update if not already tool_calls (highest priority)
                        accumulated_finish_reason = processed.finish_reason
                    if processed.usage and isinstance(processed.usage, dict):
                        usage_record = extract_usage_record(
                            processed.usage,
                            model=model,
                            source="stream_final_chunk",
                        )
                        prompt_tokens = usage_record.input_tokens + usage_record.cache_read_tokens
                        completion_tokens = usage_record.completion_tokens
                        thinking_tokens = usage_record.reasoning_tokens
                        prompt_tokens_cached = usage_record.cache_read_tokens
                        prompt_tokens_cache_write = usage_record.cache_write_tokens
                        prompt_tokens_uncached = usage_record.prompt_tokens_for_mark_success

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
                    monitor.metrics.error_count += 1
                    raise

        except StreamedAPIError:
            # Re-raise for retry loop
            raise

        finally:
            # Record usage if stream completed
            if stream_completed:
                if cred_context:
                    cost_breakdown = self._calculate_stream_cost_breakdown(
                        model,
                        usage_record,
                        skip_cost_calculation=skip_cost_calculation,
                    )
                    self._log_stream_usage_accounting(
                        transaction_logger,
                        usage_record,
                        cost_breakdown,
                    )
                    cred_context.mark_success(
                        prompt_tokens=prompt_tokens_uncached,
                        completion_tokens=completion_tokens,
                        thinking_tokens=thinking_tokens,
                        prompt_tokens_cache_read=prompt_tokens_cached,
                        prompt_tokens_cache_write=prompt_tokens_cache_write,
                        approx_cost=cost_breakdown.total_cost,
                    )

                if response_callback and (assistant_parts or tool_call_ids):
                    # Intentionally only record response anchors after a complete
                    # stream. Partial/aborted streams can contain text the client
                    # never accepted, so using them for identity would over-bind
                    # failed or disconnected sessions.
                    response_callback(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "".join(assistant_parts),
                                        "tool_calls": [
                                            {"id": call_id} for call_id in tool_call_ids
                                        ],
                                    }
                                }
                            ]
                        }
                    )

                monitor.complete()
                self._log_stream_lifecycle(
                    transaction_logger,
                    "stream_completed",
                    monitor,
                    StreamEvent("completed", protocol="openai_chat"),
                )
                self._log_stream_lifecycle(
                    transaction_logger,
                    "stream_metrics_final",
                    monitor,
                    StreamEvent("metadata", protocol="openai_chat"),
                )

                # Yield [DONE] for completed streams
                yield "data: [DONE]\n\n"

            elif request and await request.is_disconnected():
                monitor.cancel()
                self._log_stream_lifecycle(
                    transaction_logger,
                    "stream_cancelled",
                    monitor,
                    StreamEvent("cancelled", protocol="openai_chat"),
                )

    @staticmethod
    def _log_stream_lifecycle(
        transaction_logger: Optional[Any],
        pass_name: str,
        monitor: StreamMonitor,
        event: StreamEvent,
    ) -> None:
        """Emit stream lifecycle metrics without affecting stream delivery."""

        if not transaction_logger:
            return
        try:
            transaction_logger.log_transform_pass(
                pass_name,
                {"event": event.to_dict(), "metrics": monitor.metrics.to_dict()},
                direction="stream",
                stage="client",
                protocol=event.protocol,
                transport="sse",
                metadata={"event_type": event.event_type},
                snapshot=False,
            )
        except Exception as exc:
            lib_logger.debug("Stream lifecycle trace failed: %s", exc)

    def _collect_session_response_anchors(
        self,
        sse_string: str,
        assistant_parts: List[str],
        tool_call_ids: List[str],
    ) -> None:
        """Collect lightweight response evidence for session tracking.

        Streaming providers emit assistant text and tool-call IDs across many
        chunks. We keep a synthetic assistant message so the core tracker can use
        the same response-anchor path as non-streaming responses.
        """
        if not sse_string.startswith("data: "):
            return
        payload = sse_string[6:].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                assistant_parts.append(str(content))
            for tool_call in delta.get("tool_calls") or []:
                call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
                if call_id:
                    tool_call_ids.append(str(call_id))

    def _process_chunk(
        self,
        chunk: Any,
        accumulated_finish_reason: Optional[str],
        has_tool_calls: bool,
        model: str = "",
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

            # Get source finish_reason before we potentially modify it
            source_finish_reason = choice.get("finish_reason")

            # Detect final chunk using multiple signals:
            # 1. Primary: has usage with any meaningful token count > 0
            # 2. Secondary: has usage (even empty) + source has finish_reason (Fallback case)
            has_meaningful_usage = (
                usage
                and isinstance(usage, dict)
                and any(
                    usage.get(k, 0) > 0
                    for k in [
                        "completion_tokens",
                        "prompt_tokens",
                        "total_tokens",
                        "reasoning_tokens",
                    ]
                )
            )
            has_usage_with_finish = (
                usage is not None
                and isinstance(usage, dict)
                and source_finish_reason is not None
            )
            is_final_chunk = has_meaningful_usage or has_usage_with_finish

            if is_final_chunk:
                # FINAL CHUNK: Determine correct finish_reason
                # Priority: tool_calls > source_finish_reason > accumulated > "stop"
                if has_tool_calls or chunk_has_tool_calls:
                    choice["finish_reason"] = "tool_calls"
                elif source_finish_reason:
                    choice["finish_reason"] = source_finish_reason
                elif accumulated_finish_reason:
                    choice["finish_reason"] = accumulated_finish_reason
                else:
                    choice["finish_reason"] = "stop"
                finish_reason = choice["finish_reason"]
            else:
                # INTERMEDIATE CHUNK: Never emit finish_reason
                choice["finish_reason"] = None

        usage = chunk_dict.get("usage")
        if isinstance(usage, dict):
            normalize_usage_for_response(usage, model)

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

    def _calculate_stream_cost_breakdown(
        self,
        model: str,
        usage_record: UsageRecord,
        *,
        skip_cost_calculation: bool,
    ) -> CostBreakdown:
        """Calculate advisory stream cost through the shared cost helper."""

        if skip_cost_calculation:
            return CostBreakdown(pricing_source="skipped")
        return CostCalculator().calculate(usage_record, model=model)

    @staticmethod
    def _log_stream_usage_accounting(
        transaction_logger: Optional[Any],
        usage_record: UsageRecord,
        cost_breakdown: CostBreakdown,
    ) -> None:
        """Trace normalized stream usage without affecting stream delivery."""

        if not transaction_logger:
            return
        transaction_logger.log_transform_pass(
            "usage_accounting_summary",
            {"usage": usage_record.to_dict(), "cost": cost_breakdown.to_dict()},
            direction="metadata",
            stage="final",
            transport="sse",
            metadata={
                "source": usage_record.source,
                "pricing_source": cost_breakdown.pricing_source,
            },
            snapshot=False,
        )


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

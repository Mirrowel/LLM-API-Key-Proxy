"""
AI Assistant Core orchestration logic.

Manages conversation sessions, tool execution, context injection,
and coordinates between the UI, LLM bridge, and window adapter.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .bridge import LLMBridge, StreamCallbacks
from .checkpoint import CheckpointManager
from .context import WindowContextAdapter
from .prompts import BASE_ASSISTANT_PROMPT
from .tools import ToolCall, ToolCallSummary, ToolExecutor, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str  # "user" | "assistant" | "tool" | "system"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None  # Thinking (from reasoning_content field)
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response messages
    timestamp: datetime = field(default_factory=datetime.now)

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible message format."""
        msg: Dict[str, Any] = {"role": self.role}

        # Content handling: when tool_calls are present, some providers require
        # content to be present (even if null/empty)
        if self.content is not None:
            msg["content"] = self.content
        elif self.tool_calls:
            # Ensure content is present when tool_calls exist
            msg["content"] = None

        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]

        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id

        return msg


@dataclass
class ChatSession:
    """Manages conversation state and message history."""

    session_id: str
    model: str
    messages: List[Message] = field(default_factory=list)
    pending_message: Optional[str] = None  # Queued user message
    is_streaming: bool = False
    current_checkpoint_position: int = -1
    last_known_context_hash: str = ""

    # Retry tracking
    consecutive_invalid_tool_calls: int = 0
    max_tool_retries: int = 4


class AIAssistantCore:
    """
    Core orchestration for the AI Assistant.

    Manages:
    - ChatSession lifecycle
    - Tool execution with checkpoints
    - Context injection and diffing
    - Message queuing
    - Agentic loops (multi-turn tool execution)
    """

    def __init__(
        self,
        window_adapter: WindowContextAdapter,
        schedule_on_gui: Callable[[Callable], None],
        default_model: str = "openai/gpt-4o",
    ):
        """
        Initialize the AI Assistant Core.

        Args:
            window_adapter: The window-specific context adapter
            schedule_on_gui: Function to schedule callbacks on GUI thread
            default_model: Default model to use
        """
        self._adapter = window_adapter
        self._schedule_on_gui = schedule_on_gui

        # Create session
        session_id = str(uuid.uuid4())[:8]
        self._session = ChatSession(session_id=session_id, model=default_model)

        # Initialize components
        self._bridge = LLMBridge(schedule_on_gui, session_id=session_id)
        self._checkpoint_manager = CheckpointManager(session_id)
        self._tool_executor = ToolExecutor(window_adapter.get_tools())

        # Callbacks for UI updates
        self._ui_callbacks: Dict[str, Callable] = {}

        # Current response state
        self._current_thinking: str = ""
        self._current_content: str = ""
        self._pending_tool_calls: List[ToolCall] = []

        # Reasoning effort setting (None = auto/don't send)
        self._reasoning_effort: Optional[str] = None

    @property
    def session(self) -> ChatSession:
        """Get the current chat session."""
        return self._session

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        """Get the checkpoint manager."""
        return self._checkpoint_manager

    @property
    def bridge(self) -> LLMBridge:
        """Get the LLM bridge."""
        return self._bridge

    def set_ui_callbacks(
        self,
        on_thinking_chunk: Optional[Callable[[str], None]] = None,
        on_content_chunk: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[ToolCall], None]] = None,
        on_tool_result: Optional[Callable[[ToolCall, ToolResult], None]] = None,
        on_message_complete: Optional[Callable[[Message], None]] = None,
        on_error: Optional[
            Callable[[str, bool], None]
        ] = None,  # (message, is_retryable)
        on_stream_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Set UI callbacks for response handling.

        Args:
            on_thinking_chunk: Called with each thinking/reasoning chunk
            on_content_chunk: Called with each content chunk
            on_tool_start: Called when a tool execution begins
            on_tool_result: Called when a tool execution completes
            on_message_complete: Called when a full message is added to history
            on_error: Called on errors (message, is_retryable)
            on_stream_complete: Called when streaming/processing is fully complete
        """
        self._ui_callbacks = {
            "on_thinking_chunk": on_thinking_chunk,
            "on_content_chunk": on_content_chunk,
            "on_tool_start": on_tool_start,
            "on_tool_result": on_tool_result,
            "on_message_complete": on_message_complete,
            "on_error": on_error,
            "on_stream_complete": on_stream_complete,
        }

    def set_model(self, model: str) -> None:
        """Set the model to use."""
        self._session.model = model
        logger.info(f"Model set to: {model}")

    def set_reasoning_effort(self, effort: Optional[str]) -> None:
        """
        Set the reasoning effort level.

        Args:
            effort: One of "low", "medium", "high", or None (auto/don't send)
        """
        if effort is not None and effort not in ("low", "medium", "high"):
            logger.warning(f"Invalid reasoning effort: {effort}")
            effort = None
        self._reasoning_effort = effort
        logger.info(f"Reasoning effort set to: {effort or 'auto'}")

    def send_message(self, content: str) -> bool:
        """
        Send a user message.

        If currently streaming, the message is queued.

        Args:
            content: The user message content

        Returns:
            True if message was sent/queued, False if invalid
        """
        content = content.strip()
        if not content:
            return False

        if self._session.is_streaming:
            # Queue the message (replaces any existing queued message)
            self._session.pending_message = content
            logger.info("Message queued (currently streaming)")
            return True

        # Add user message to history
        user_message = Message(role="user", content=content)
        self._session.messages.append(user_message)

        # Notify UI
        if self._ui_callbacks.get("on_message_complete"):
            self._ui_callbacks["on_message_complete"](user_message)

        # Start the response
        self._start_response()
        return True

    def _start_response(self) -> None:
        """Start generating an LLM response."""
        self._session.is_streaming = True
        self._current_thinking = ""
        self._current_content = ""
        self._pending_tool_calls = []
        self._session.consecutive_invalid_tool_calls = 0

        # Signal checkpoint manager
        self._checkpoint_manager.start_response()

        # Notify adapter that AI is starting
        self._adapter.on_ai_started()

        # Build messages array
        messages = self._build_messages()

        # Get tools in OpenAI format
        tools = self._tool_executor.get_tools_openai_format()

        # Set up callbacks
        callbacks = StreamCallbacks(
            on_thinking_chunk=self._handle_thinking_chunk,
            on_content_chunk=self._handle_content_chunk,
            on_tool_calls=self._handle_tool_calls,
            on_error=self._handle_error,
            on_complete=self._handle_stream_complete,
        )

        # Start streaming
        self._bridge.stream_completion(
            messages=messages,
            tools=tools,
            model=self._session.model,
            callbacks=callbacks,
            reasoning_effort=self._reasoning_effort,
        )

    def _build_messages(self) -> List[Dict[str, Any]]:
        """Build the messages array for the LLM request."""
        messages = []

        # System prompt (base + window-specific)
        system_prompt = (
            BASE_ASSISTANT_PROMPT + "\n\n" + self._adapter.get_window_system_prompt()
        )

        # Add current context
        context = self._adapter.get_full_context()
        context_str = json.dumps(context, indent=2, default=str)
        system_prompt += f"\n\n### Current Context\n\n```json\n{context_str}\n```"

        messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        for msg in self._session.messages:
            messages.append(msg.to_openai_format())

        # Log summary of messages being sent
        role_counts = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            # Log tool_calls in assistant messages
            if role == "assistant" and msg.get("tool_calls"):
                tc_names = [tc["function"]["name"] for tc in msg.get("tool_calls", [])]
                logger.debug(f"  Assistant message has tool_calls: {tc_names}")
            # Log tool responses
            if role == "tool":
                logger.debug(
                    f"  Tool response: id={msg.get('tool_call_id')}, "
                    f"content_preview={str(msg.get('content', ''))[:100]}"
                )

        logger.debug(f"Built messages array: {role_counts}")

        return messages

    def _handle_thinking_chunk(self, chunk: str) -> None:
        """Handle a thinking/reasoning chunk."""
        self._current_thinking += chunk

        if self._ui_callbacks.get("on_thinking_chunk"):
            self._ui_callbacks["on_thinking_chunk"](chunk)

    def _handle_content_chunk(self, chunk: str) -> None:
        """Handle a content chunk."""
        self._current_content += chunk

        if self._ui_callbacks.get("on_content_chunk"):
            self._ui_callbacks["on_content_chunk"](chunk)

    def _handle_tool_calls(self, tool_calls: List[ToolCall]) -> None:
        """Handle tool calls from the LLM."""
        self._pending_tool_calls = tool_calls

    def _handle_error(self, error: str) -> None:
        """Handle a streaming error."""
        logger.error(f"Streaming error: {error}")
        self._session.is_streaming = False
        self._adapter.on_ai_completed()

        if self._ui_callbacks.get("on_error"):
            # Most errors are retryable
            is_retryable = "authentication" not in error.lower()
            self._ui_callbacks["on_error"](error, is_retryable)

    def _handle_stream_complete(self) -> None:
        """Handle stream completion."""
        logger.info(
            f"Stream complete: content_len={len(self._current_content)}, "
            f"thinking_len={len(self._current_thinking)}, "
            f"tool_calls={len(self._pending_tool_calls)}"
        )

        # Create assistant message
        assistant_message = Message(
            role="assistant",
            content=self._current_content if self._current_content else None,
            reasoning_content=self._current_thinking
            if self._current_thinking
            else None,
            tool_calls=self._pending_tool_calls if self._pending_tool_calls else None,
        )
        self._session.messages.append(assistant_message)

        logger.debug(
            f"Added assistant message to history. Total messages: {len(self._session.messages)}"
        )

        # Notify UI of message completion
        if self._ui_callbacks.get("on_message_complete"):
            self._ui_callbacks["on_message_complete"](assistant_message)

        # Execute tool calls if any
        if self._pending_tool_calls:
            logger.info(
                f"Processing {len(self._pending_tool_calls)} pending tool call(s)"
            )
            self._execute_tool_calls(self._pending_tool_calls)
        else:
            # No tool calls - response is complete
            logger.info("No tool calls - finishing response")
            self._finish_response()

    def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> None:
        """Execute pending tool calls and handle results."""
        logger.info(f"Executing {len(tool_calls)} tool call(s)")

        # Check if we need to create a checkpoint
        if self._tool_executor.has_write_tools(tool_calls):
            if self._checkpoint_manager.should_create_checkpoint():
                # Create checkpoint before executing write tools
                state = self._adapter.get_full_context()
                summaries = [
                    ToolCallSummary(
                        name=tc.name,
                        arguments=tc.arguments,
                        success=True,  # Will be updated
                        message="Pending",
                    )
                    for tc in tool_calls
                ]
                self._checkpoint_manager.create_checkpoint(
                    state=state,
                    tool_calls=summaries,
                    message_index=len(self._session.messages) - 1,
                )

        # Execute each tool
        all_results: List[Message] = []
        has_errors = False
        assistant_logger = self._bridge.current_logger

        for tool_call in tool_calls:
            logger.info(
                f"Executing tool: {tool_call.name} (id={tool_call.id}) "
                f"args={json.dumps(tool_call.arguments)[:200]}"
            )

            # Notify UI
            if self._ui_callbacks.get("on_tool_start"):
                self._ui_callbacks["on_tool_start"](tool_call)

            # Execute
            result = self._tool_executor.execute(tool_call, self._adapter)
            tool_call.result = result

            logger.info(
                f"Tool result: {tool_call.name} -> "
                f"{'SUCCESS' if result.success else 'FAILED'}: {result.message[:100]}"
            )

            # Log to assistant logger
            if assistant_logger:
                assistant_logger.log_tool_execution(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    result_success=result.success,
                    result_message=result.message,
                    result_data=result.data,
                    error_code=result.error_code,
                )

            # Notify UI
            if self._ui_callbacks.get("on_tool_result"):
                self._ui_callbacks["on_tool_result"](tool_call, result)

            # Create tool response message
            tool_message = Message(
                role="tool",
                content=result.to_json(),
                tool_call_id=tool_call.id,
            )
            all_results.append(tool_message)

            if not result.success:
                has_errors = True

        # Add tool results to history
        self._session.messages.extend(all_results)
        logger.info(
            f"Added {len(all_results)} tool result message(s) to history. "
            f"Total messages: {len(self._session.messages)}"
        )

        # Handle retry logic for invalid tool calls
        if has_errors:
            self._session.consecutive_invalid_tool_calls += 1
            if (
                self._session.consecutive_invalid_tool_calls
                >= self._session.max_tool_retries
            ):
                # Max retries exceeded - show error to user
                logger.warning("Max tool retries exceeded")
                if self._ui_callbacks.get("on_error"):
                    self._ui_callbacks["on_error"](
                        "Tool execution failed after multiple retries. Please try a different approach.",
                        False,
                    )
                self._finish_response()
                return

            # Show retry indicator after 2nd failure
            if self._session.consecutive_invalid_tool_calls >= 2:
                logger.info("Tool retry in progress (shown to user)")
                # UI will show "Retrying..." based on the failed tool results

        # Continue the agentic loop - get next response from LLM
        self._continue_agentic_loop()

    def _continue_agentic_loop(self) -> None:
        """Continue the conversation after tool execution."""
        logger.info("Continuing agentic loop after tool execution")

        # Reset current response state
        self._current_thinking = ""
        self._current_content = ""
        self._pending_tool_calls = []

        # Build messages (includes tool results)
        messages = self._build_messages()
        tools = self._tool_executor.get_tools_openai_format()

        # Log the continuation request
        assistant_logger = self._bridge.current_logger
        if assistant_logger:
            assistant_logger.log_messages_sent(messages)

        logger.debug(
            f"Continuation request: {len(messages)} messages, "
            f"last message role={messages[-1]['role'] if messages else 'N/A'}"
        )

        callbacks = StreamCallbacks(
            on_thinking_chunk=self._handle_thinking_chunk,
            on_content_chunk=self._handle_content_chunk,
            on_tool_calls=self._handle_tool_calls,
            on_error=self._handle_error,
            on_complete=self._handle_stream_complete,
        )

        self._bridge.stream_completion(
            messages=messages,
            tools=tools,
            model=self._session.model,
            callbacks=callbacks,
            reasoning_effort=self._reasoning_effort,
        )

    def _finish_response(self) -> None:
        """Finish the response cycle."""
        self._session.is_streaming = False
        self._adapter.on_ai_completed()

        # Notify UI
        if self._ui_callbacks.get("on_stream_complete"):
            self._ui_callbacks["on_stream_complete"]()

        # Check for queued message
        if self._session.pending_message:
            queued = self._session.pending_message
            self._session.pending_message = None
            logger.info("Processing queued message")
            self.send_message(queued)

    def cancel(self) -> None:
        """Cancel the current response."""
        if not self._session.is_streaming:
            return

        self._bridge.cancel_stream()
        self._session.is_streaming = False
        self._adapter.on_ai_completed()

        # Discard partial response (don't add to history)
        # The last user message stays in history

        logger.info("Response cancelled")

    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Rollback to a specific checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to rollback to

        Returns:
            True if rollback succeeded
        """
        if self._session.is_streaming:
            logger.warning("Cannot rollback while streaming")
            return False

        # Get the state and message index
        state = self._checkpoint_manager.rollback_to(checkpoint_id)
        if state is None:
            return False

        message_index = self._checkpoint_manager.get_message_index_at(checkpoint_id)
        if message_index is not None:
            # Truncate conversation history
            self._session.messages = self._session.messages[: message_index + 1]

        # Apply state to window
        try:
            self._adapter.apply_state(state)
            logger.info(f"Rolled back to checkpoint {checkpoint_id}")
            return True
        except Exception as e:
            logger.exception(f"Failed to apply state for checkpoint {checkpoint_id}")
            return False

    def new_session(self) -> None:
        """Start a new session (clear history and checkpoints)."""
        if self._session.is_streaming:
            self.cancel()

        # Clear history
        self._session.messages.clear()
        self._session.pending_message = None
        self._session.consecutive_invalid_tool_calls = 0

        # Clear checkpoints
        self._checkpoint_manager.clear()

        # Generate new session ID
        new_session_id = str(uuid.uuid4())[:8]
        self._session.session_id = new_session_id
        self._checkpoint_manager.session_id = new_session_id
        self._bridge.set_session_id(new_session_id)

        logger.info(f"Started new session: {new_session_id}")

    def retry_last(self) -> bool:
        """
        Retry the last user message.

        Returns:
            True if retry was started
        """
        if self._session.is_streaming:
            return False

        # Find the last user message
        last_user_idx = -1
        for i in range(len(self._session.messages) - 1, -1, -1):
            if self._session.messages[i].role == "user":
                last_user_idx = i
                break

        if last_user_idx < 0:
            return False

        # Remove all messages after (and including) the last user message
        last_user_content = self._session.messages[last_user_idx].content
        self._session.messages = self._session.messages[:last_user_idx]

        # Re-send the message
        if last_user_content:
            self.send_message(last_user_content)
            return True

        return False

    def close(self) -> None:
        """Clean up resources."""
        self._bridge.close()
        self._checkpoint_manager.clear_temp_file()

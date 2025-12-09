"""
AI Chat Window - Main Pop-Out Window.

The primary UI for interacting with the AI assistant.
"""

import customtkinter as ctk
import tkinter as tk
from typing import Any, Callable, Dict, List, Optional

from ..checkpoint import Checkpoint
from ..core import AIAssistantCore, Message
from ..context import WindowContextAdapter
from ..tools import ToolCall, ToolResult
from .checkpoint_ui import CheckpointDropdown
from .message_view import MessageView
from .model_selector import ModelSelector
from .styles import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    BG_HOVER,
    BG_PRIMARY,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER_COLOR,
    FONT_FAMILY,
    FONT_SIZE_NORMAL,
    INPUT_MAX_HEIGHT,
    INPUT_MIN_HEIGHT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_font,
)


class AIChatWindow(ctk.CTkToplevel):
    """
    Pop-out AI Chat Window.

    Features:
    - Full message display with streaming
    - Model selector (grouped by provider)
    - Checkpoint dropdown for undo
    - Multi-line input with keyboard shortcuts
    - New Session and Send buttons
    """

    def __init__(
        self,
        parent,
        window_adapter: WindowContextAdapter,
        title: str = "AI Assistant",
        default_model: str = "openai/gpt-4o",
        **kwargs,
    ):
        """
        Initialize the chat window.

        Args:
            parent: Parent window
            window_adapter: The window context adapter
            title: Window title
            default_model: Default model to use
            **kwargs: Additional toplevel arguments
        """
        super().__init__(parent, **kwargs)

        self._parent = parent
        self._adapter = window_adapter
        self._default_model = default_model

        # Initialize core
        self._core = AIAssistantCore(
            window_adapter=window_adapter,
            schedule_on_gui=lambda fn: self.after(0, fn),
            default_model=default_model,
        )

        # Window configuration
        self.title(title)
        self.configure(fg_color=BG_PRIMARY)
        self.geometry("600x700")
        self.minsize(450, 500)

        # Track pending tool calls for UI updates
        self._pending_tool_frames: Dict[str, ctk.CTkFrame] = {}

        self._create_widgets()
        self._setup_callbacks()
        self._setup_bindings()

        # Load models
        self._load_models()

        # Focus input
        self.after(100, lambda: self.input_text.focus_set())

    def _create_widgets(self) -> None:
        """Create the UI widgets."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=3, minsize=200)  # Messages
        self.grid_rowconfigure(2, weight=0)  # Input
        self.grid_rowconfigure(3, weight=0)  # Buttons

        # Header
        self._create_header()

        # Message display
        self.message_view = MessageView(self)
        self.message_view.grid(row=1, column=0, sticky="nsew", padx=8, pady=(8, 0))

        # Input area
        self._create_input_area()

        # Buttons
        self._create_buttons()

    def _create_header(self) -> None:
        """Create the header with model selector, reasoning effort, and checkpoints."""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        header.grid_columnconfigure(1, weight=1)  # Spacer column

        # Left side: Model selector
        self.model_selector = ModelSelector(
            header,
            on_model_changed=self._on_model_changed,
        )
        self.model_selector.grid(row=0, column=0, sticky="w")

        # Middle: Reasoning effort selector
        reasoning_frame = ctk.CTkFrame(header, fg_color="transparent")
        reasoning_frame.grid(row=0, column=1, sticky="w", padx=(16, 0))

        ctk.CTkLabel(
            reasoning_frame,
            text="Reasoning:",
            font=get_font("normal"),
            text_color=TEXT_SECONDARY,
        ).pack(side="left", padx=(0, 8))

        self._reasoning_effort_var = ctk.StringVar(value="Auto")
        self.reasoning_dropdown = ctk.CTkComboBox(
            reasoning_frame,
            values=["Auto", "Low", "Medium", "High"],
            variable=self._reasoning_effort_var,
            font=get_font("normal"),
            dropdown_font=get_font("normal"),
            fg_color=BG_TERTIARY,
            border_color=BORDER_COLOR,
            button_color=BG_SECONDARY,
            button_hover_color=BG_HOVER,
            dropdown_fg_color=BG_SECONDARY,
            dropdown_hover_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            dropdown_text_color=TEXT_PRIMARY,
            width=100,
            state="readonly",
            command=self._on_reasoning_changed,
        )
        self.reasoning_dropdown.pack(side="left")

        # Right side: Checkpoint dropdown
        self.checkpoint_dropdown = CheckpointDropdown(
            header,
            on_rollback=self._on_rollback,
        )
        self.checkpoint_dropdown.grid(row=0, column=2, sticky="e")

    def _create_input_area(self) -> None:
        """Create the input area with buttons stacked on the right."""
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(8, 0))
        input_frame.grid_columnconfigure(0, weight=1)  # Input takes remaining space
        input_frame.grid_columnconfigure(1, weight=0)  # Buttons column fixed

        # Multi-line text input (left side)
        self.input_text = ctk.CTkTextbox(
            input_frame,
            font=get_font("normal"),
            fg_color=BG_TERTIARY,
            text_color=TEXT_PRIMARY,
            border_width=1,
            border_color=BORDER_COLOR,
            corner_radius=8,
            height=INPUT_MIN_HEIGHT,
            wrap="word",
        )
        self.input_text.grid(row=0, column=0, sticky="nsew")

        # Button stack (right side)
        btn_stack = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_stack.grid(row=0, column=1, sticky="ns", padx=(8, 0))

        # New Session button (top)
        self.new_session_btn = ctk.CTkButton(
            btn_stack,
            text="New Session",
            font=get_font("small"),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            border_width=1,
            border_color=BORDER_COLOR,
            width=90,
            height=26,
            command=self._on_new_session,
        )
        self.new_session_btn.pack(side="top", pady=(0, 4))

        # Send button (bottom)
        self.send_btn = ctk.CTkButton(
            btn_stack,
            text="Send →",
            font=get_font("small", bold=True),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8eef",
            text_color=TEXT_PRIMARY,
            width=90,
            height=26,
            command=self._on_send,
        )
        self.send_btn.pack(side="top")

        # Placeholder text handling
        self._placeholder_active = True
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        """Show placeholder text in input."""
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", "Type your message here... (Ctrl+Enter to send)")
        self.input_text.configure(text_color=TEXT_MUTED)
        self._placeholder_active = True

    def _hide_placeholder(self) -> None:
        """Hide placeholder text."""
        if self._placeholder_active:
            self.input_text.delete("1.0", "end")
            self.input_text.configure(text_color=TEXT_PRIMARY)
            self._placeholder_active = False

    def _create_buttons(self) -> None:
        """Create the status bar (buttons moved to input area)."""
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 8))

        # Status label
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="",
            font=get_font("small"),
            text_color=TEXT_MUTED,
        )
        self.status_label.pack(side="left")

    def _setup_callbacks(self) -> None:
        """Set up AI core callbacks."""
        self._core.set_ui_callbacks(
            on_thinking_chunk=self._on_thinking_chunk,
            on_content_chunk=self._on_content_chunk,
            on_tool_start=self._on_tool_start,
            on_tool_result=self._on_tool_result,
            on_message_complete=self._on_message_complete,
            on_error=self._on_error,
            on_stream_complete=self._on_stream_complete,
        )

    def _setup_bindings(self) -> None:
        """Set up keyboard bindings."""
        # Ctrl+Enter to send
        self.input_text.bind("<Control-Return>", lambda e: self._on_send())

        # Escape to cancel or clear
        self.bind("<Escape>", self._on_escape)

        # Focus handling for placeholder
        self.input_text.bind("<FocusIn>", lambda e: self._hide_placeholder())
        self.input_text.bind("<FocusOut>", self._on_input_focus_out)

        # Window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_input_focus_out(self, event) -> None:
        """Handle input focus out."""
        content = self.input_text.get("1.0", "end").strip()
        if not content or content == "Type your message here... (Ctrl+Enter to send)":
            self._show_placeholder()

    def _load_models(self) -> None:
        """Load available models."""
        self.model_selector.set_loading()

        self._core.bridge.fetch_models(
            on_success=self._on_models_loaded,
            on_error=self._on_models_error,
        )

    def _on_models_loaded(self, models: Dict[str, List[str]]) -> None:
        """Handle models loaded."""
        self.model_selector.set_models(models)

        # Try to select default model
        if not self.model_selector.set_selected_model(self._default_model):
            # Use first available
            pass

    def _on_models_error(self, error: str) -> None:
        """Handle models load error."""
        self.model_selector.set_error(f"Failed: {error[:30]}...")

    def _on_model_changed(self, model: str) -> None:
        """Handle model selection change."""
        self._core.set_model(model)

    def _on_reasoning_changed(self, choice: str) -> None:
        """Handle reasoning effort selection change."""
        # Map UI values to API values
        effort_map = {
            "Auto": None,  # Don't send the parameter
            "Low": "low",
            "Medium": "medium",
            "High": "high",
        }
        effort = effort_map.get(choice)
        self._core.set_reasoning_effort(effort)

    def _on_send(self) -> None:
        """Handle send button click."""
        if self._placeholder_active:
            return

        content = self.input_text.get("1.0", "end").strip()
        if not content:
            return

        # Clear input
        self.input_text.delete("1.0", "end")

        # Send message
        if self._core.send_message(content):
            self._set_streaming_state(True)

    def _on_new_session(self) -> None:
        """Handle new session button click."""
        self._core.new_session()
        self.message_view.clear()
        self.checkpoint_dropdown.set_checkpoints([])
        self._set_streaming_state(False)
        self.status_label.configure(text="")

    def _on_escape(self, event=None) -> None:
        """Handle escape key."""
        if self._core.session.is_streaming:
            self._core.cancel()
            self._set_streaming_state(False)
            self.status_label.configure(text="Cancelled")
        else:
            # Clear input
            self.input_text.delete("1.0", "end")
            self._show_placeholder()

    def _on_rollback(self, checkpoint_id: str) -> None:
        """Handle checkpoint rollback."""
        if self._core.rollback_to_checkpoint(checkpoint_id):
            # Rebuild message display
            self.message_view.clear()
            for msg in self._core.session.messages:
                self.message_view.add_message(msg)

            # Update checkpoints
            self.checkpoint_dropdown.set_checkpoints(
                self._core.checkpoint_manager.get_checkpoints()
            )

            self.status_label.configure(text="Rolled back")

    # =========================================================================
    # AI Core Callbacks
    # =========================================================================

    def _on_thinking_chunk(self, chunk: str) -> None:
        """Handle thinking chunk from AI."""
        self.message_view.append_thinking(chunk)

    def _on_content_chunk(self, chunk: str) -> None:
        """Handle content chunk from AI."""
        self.message_view.append_content(chunk)

    def _on_tool_start(self, tool_call: ToolCall) -> None:
        """Handle tool execution start."""
        frame = self.message_view.add_tool_call(tool_call)
        self._pending_tool_frames[tool_call.id] = frame
        self.status_label.configure(text=f"Executing: {tool_call.name}...")

    def _on_tool_result(self, tool_call: ToolCall, result: ToolResult) -> None:
        """Handle tool execution result."""
        if tool_call.id in self._pending_tool_frames:
            frame = self._pending_tool_frames.pop(tool_call.id)
            self.message_view.update_tool_result(frame, result)

        if result.success:
            self.status_label.configure(text=f"✓ {tool_call.name}")
        else:
            self.status_label.configure(
                text=f"✗ {tool_call.name}: {result.message[:30]}..."
            )

    def _on_message_complete(self, message: Message) -> None:
        """Handle complete message."""
        if message.role == "user":
            self.message_view.add_user_message(message.content or "", message.timestamp)
            # Start AI message for streaming
            self.message_view.start_ai_message()
        elif message.role == "assistant":
            # Finish streaming message
            self.message_view.finish_ai_message()

    def _on_error(self, error: str, is_retryable: bool) -> None:
        """Handle error."""
        self._set_streaming_state(False)

        self.message_view.add_error(
            message=error,
            is_retryable=is_retryable,
            on_retry=self._on_retry if is_retryable else None,
            on_cancel=lambda: self.status_label.configure(text=""),
        )

        self.status_label.configure(text="Error occurred")

    def _on_retry(self) -> None:
        """Handle retry."""
        if self._core.retry_last():
            self._set_streaming_state(True)

    def _on_stream_complete(self) -> None:
        """Handle stream completion."""
        self._set_streaming_state(False)

        # Update checkpoints
        self.checkpoint_dropdown.set_checkpoints(
            self._core.checkpoint_manager.get_checkpoints()
        )

        self.status_label.configure(text="")

    # =========================================================================
    # State Management
    # =========================================================================

    def _set_streaming_state(self, is_streaming: bool) -> None:
        """Update UI for streaming state."""
        if is_streaming:
            self.send_btn.configure(
                text="Cancel", fg_color=BG_SECONDARY, command=self._on_escape
            )
            self.new_session_btn.configure(state="disabled")
            self.status_label.configure(text="Thinking...")
        else:
            self.send_btn.configure(
                text="Send →", fg_color=ACCENT_BLUE, command=self._on_send
            )
            self.new_session_btn.configure(state="normal")

    def _on_close(self) -> None:
        """Handle window close."""
        # Cancel any streaming
        if self._core.session.is_streaming:
            self._core.cancel()

        # Clean up
        self._core.close()

        # Destroy window
        self.destroy()

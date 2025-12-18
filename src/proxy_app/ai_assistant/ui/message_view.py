"""
Message Display Widget for the AI Assistant.

Provides a scrollable view for displaying conversation messages,
including user messages, AI responses, thinking sections, and tool results.
"""

import customtkinter as ctk
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from ..core import Message
from ..tools import ToolCall, ToolResult
from .styles import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    AI_MESSAGE_BG,
    BG_HOVER,
    BG_PRIMARY,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER_COLOR,
    CORNER_RADIUS,
    ERROR_BG,
    FONT_FAMILY,
    FONT_SIZE_NORMAL,
    FONT_SIZE_SMALL,
    MESSAGE_PADDING,
    MESSAGE_SPACING,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    THINKING_BG,
    THINKING_TEXT,
    TOOL_BG,
    TOOL_FAILURE_COLOR,
    TOOL_SUCCESS_COLOR,
    USER_MESSAGE_BG,
    get_font,
)
from .thinking import ThinkingSection


@dataclass
class StreamingState:
    """State for the currently streaming message."""

    thinking_text: str = ""
    content_text: str = ""
    is_streaming: bool = False
    thinking_widget: Optional[ThinkingSection] = None
    content_widget: Optional[ctk.CTkLabel] = None
    message_frame: Optional[ctk.CTkFrame] = None


class MessageView(ctk.CTkFrame):
    """
    Scrollable message display for conversation history.

    Features:
    - User messages (right-aligned, accent background)
    - AI messages (left-aligned)
    - Collapsible thinking sections
    - Tool execution display with results
    - Streaming text support
    - Auto-scroll to bottom
    """

    def __init__(self, parent, **kwargs):
        """
        Initialize the message view.

        Args:
            parent: Parent widget
            **kwargs: Additional frame arguments
        """
        super().__init__(parent, fg_color=BG_TERTIARY, **kwargs)

        self._streaming = StreamingState()
        self._message_widgets: List[ctk.CTkFrame] = []

        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the UI widgets."""
        # Scrollable container
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=BG_TERTIARY,
            corner_radius=0,
        )
        self.scroll_frame.pack(fill="both", expand=True)

        # Configure scroll frame grid
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        # Welcome message
        self._show_welcome()

    def _show_welcome(self) -> None:
        """Show the welcome message."""
        welcome = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        welcome.pack(fill="x", pady=20)

        ctk.CTkLabel(
            welcome,
            text="AI Assistant",
            font=get_font("title", bold=True),
            text_color=TEXT_PRIMARY,
        ).pack()

        ctk.CTkLabel(
            welcome,
            text="Ask me to help configure your model filters.\nI can add rules, explain model statuses, and more.",
            font=get_font("normal"),
            text_color=TEXT_SECONDARY,
            justify="center",
        ).pack(pady=(8, 0))

        self._welcome_widget = welcome

    def _hide_welcome(self) -> None:
        """Hide the welcome message."""
        if hasattr(self, "_welcome_widget") and self._welcome_widget:
            self._welcome_widget.destroy()
            self._welcome_widget = None

    def add_user_message(
        self, content: str, timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add a user message to the display.

        Args:
            content: Message content
            timestamp: Optional timestamp
        """
        self._hide_welcome()

        # Container (right-aligned)
        container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        container.pack(fill="x", pady=(MESSAGE_SPACING, 0), padx=MESSAGE_PADDING)

        # Message bubble (right side)
        bubble = ctk.CTkFrame(
            container,
            fg_color=USER_MESSAGE_BG,
            corner_radius=CORNER_RADIUS,
        )
        bubble.pack(side="right", anchor="e")

        # Content
        label = ctk.CTkLabel(
            bubble,
            text=content,
            font=get_font("normal"),
            text_color=TEXT_PRIMARY,
            wraplength=400,
            justify="left",
            anchor="w",
        )
        label.pack(padx=MESSAGE_PADDING, pady=MESSAGE_PADDING)

        # Timestamp
        if timestamp:
            time_str = timestamp.strftime("%H:%M")
            time_label = ctk.CTkLabel(
                container,
                text=time_str,
                font=get_font("small"),
                text_color=TEXT_MUTED,
            )
            time_label.pack(side="right", padx=(0, 8))

        self._message_widgets.append(container)
        self._scroll_to_bottom()

    def start_ai_message(self) -> None:
        """Start a new AI message (for streaming)."""
        self._hide_welcome()

        # Container
        container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        container.pack(fill="x", pady=(MESSAGE_SPACING, 0), padx=MESSAGE_PADDING)

        # Message bubble (left side)
        bubble = ctk.CTkFrame(
            container,
            fg_color=AI_MESSAGE_BG,
            corner_radius=CORNER_RADIUS,
        )
        bubble.pack(side="left", anchor="w", fill="x", expand=True)

        # Thinking section (initially hidden, created when needed)
        self._streaming = StreamingState(
            is_streaming=True,
            message_frame=bubble,
        )

        self._message_widgets.append(container)
        self._scroll_to_bottom()

    def append_thinking(self, chunk: str) -> None:
        """
        Append thinking content to the current streaming message.

        Args:
            chunk: Thinking text chunk
        """
        if not self._streaming.is_streaming:
            return

        self._streaming.thinking_text += chunk

        # Create thinking widget if not exists
        if self._streaming.thinking_widget is None:
            self._streaming.thinking_widget = ThinkingSection(
                self._streaming.message_frame,
                initial_expanded=True,
            )
            self._streaming.thinking_widget.pack(
                fill="x", padx=MESSAGE_PADDING, pady=(MESSAGE_PADDING, 0)
            )

        self._streaming.thinking_widget.append_text(chunk)
        self._scroll_to_bottom()

    def append_content(self, chunk: str) -> None:
        """
        Append content to the current streaming message.

        Args:
            chunk: Content text chunk
        """
        if not self._streaming.is_streaming:
            return

        self._streaming.content_text += chunk

        # Auto-collapse thinking when content arrives
        if (
            self._streaming.thinking_widget
            and self._streaming.thinking_widget.is_expanded
        ):
            self._streaming.thinking_widget.auto_collapse()

        # Create or update content widget
        if self._streaming.content_widget is None:
            self._streaming.content_widget = ctk.CTkLabel(
                self._streaming.message_frame,
                text=self._streaming.content_text,
                font=get_font("normal"),
                text_color=TEXT_PRIMARY,
                wraplength=500,
                justify="left",
                anchor="w",
            )
            self._streaming.content_widget.pack(
                fill="x", padx=MESSAGE_PADDING, pady=MESSAGE_PADDING
            )
        else:
            self._streaming.content_widget.configure(text=self._streaming.content_text)

        self._scroll_to_bottom()

    def finish_ai_message(self) -> None:
        """Finish the current streaming AI message."""
        self._streaming.is_streaming = False
        self._streaming = StreamingState()

    def add_ai_message(
        self,
        content: Optional[str],
        thinking: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Add a complete AI message (non-streaming).

        Args:
            content: Message content
            thinking: Optional thinking content
            timestamp: Optional timestamp
        """
        self._hide_welcome()

        # Container
        container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        container.pack(fill="x", pady=(MESSAGE_SPACING, 0), padx=MESSAGE_PADDING)

        # Message bubble
        bubble = ctk.CTkFrame(
            container,
            fg_color=AI_MESSAGE_BG,
            corner_radius=CORNER_RADIUS,
        )
        bubble.pack(side="left", anchor="w", fill="x", expand=True)

        # Thinking section (collapsed by default)
        if thinking:
            thinking_section = ThinkingSection(bubble, initial_expanded=False)
            thinking_section.set_text(thinking)
            thinking_section.pack(
                fill="x", padx=MESSAGE_PADDING, pady=(MESSAGE_PADDING, 0)
            )

        # Content
        if content:
            label = ctk.CTkLabel(
                bubble,
                text=content,
                font=get_font("normal"),
                text_color=TEXT_PRIMARY,
                wraplength=500,
                justify="left",
                anchor="w",
            )
            label.pack(fill="x", padx=MESSAGE_PADDING, pady=MESSAGE_PADDING)

        self._message_widgets.append(container)
        self._scroll_to_bottom()

    def add_tool_call(
        self, tool_call: ToolCall, result: Optional[ToolResult] = None
    ) -> ctk.CTkFrame:
        """
        Add a tool call display.

        Args:
            tool_call: The tool call
            result: Optional result (if already executed)

        Returns:
            The tool frame widget (for updating with result later)
        """
        # Container
        container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        container.pack(fill="x", pady=(4, 0), padx=MESSAGE_PADDING)

        # Tool bubble
        bubble = ctk.CTkFrame(
            container,
            fg_color=TOOL_BG,
            corner_radius=6,
        )
        bubble.pack(side="left", anchor="w")

        # Header with icon
        header = ctk.CTkFrame(bubble, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(6, 2))

        # Icon (pending, success, or failure)
        if result is None:
            icon = "⋯"
            icon_color = TEXT_MUTED
        elif result.success:
            icon = "✓"
            icon_color = TOOL_SUCCESS_COLOR
        else:
            icon = "✗"
            icon_color = TOOL_FAILURE_COLOR

        icon_label = ctk.CTkLabel(
            header,
            text=icon,
            font=get_font("normal", bold=True),
            text_color=icon_color,
            width=16,
        )
        icon_label.pack(side="left")

        # Tool name and arguments
        args_str = ", ".join(f'{k}="{v}"' for k, v in tool_call.arguments.items())
        tool_text = f"{tool_call.name}({args_str})"

        ctk.CTkLabel(
            header,
            text=tool_text,
            font=get_font("small", monospace=True),
            text_color=TEXT_SECONDARY,
        ).pack(side="left", padx=(4, 0))

        # Result message
        if result:
            result_label = ctk.CTkLabel(
                bubble,
                text=f"→ {result.message}",
                font=get_font("small"),
                text_color=TEXT_MUTED,
                anchor="w",
            )
            result_label.pack(fill="x", padx=8, pady=(0, 6))

        # Store references for updating
        bubble._icon_label = icon_label
        bubble._result_container = bubble

        self._message_widgets.append(container)
        self._scroll_to_bottom()

        return bubble

    def update_tool_result(self, tool_frame: ctk.CTkFrame, result: ToolResult) -> None:
        """
        Update a tool call display with its result.

        Args:
            tool_frame: The tool frame returned by add_tool_call
            result: The tool result
        """
        if hasattr(tool_frame, "_icon_label"):
            if result.success:
                tool_frame._icon_label.configure(
                    text="✓", text_color=TOOL_SUCCESS_COLOR
                )
            else:
                tool_frame._icon_label.configure(
                    text="✗", text_color=TOOL_FAILURE_COLOR
                )

        # Add result message
        result_label = ctk.CTkLabel(
            tool_frame,
            text=f"→ {result.message}",
            font=get_font("small"),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        result_label.pack(fill="x", padx=8, pady=(0, 6))

        self._scroll_to_bottom()

    def add_error(
        self,
        message: str,
        is_retryable: bool = True,
        on_retry: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ) -> ctk.CTkFrame:
        """
        Add an error display.

        Args:
            message: Error message
            is_retryable: Whether retry is possible
            on_retry: Retry callback
            on_cancel: Cancel callback

        Returns:
            The error frame (for removal on retry success)
        """
        # Container
        container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        container.pack(fill="x", pady=(MESSAGE_SPACING, 0), padx=MESSAGE_PADDING)

        # Error bubble
        bubble = ctk.CTkFrame(
            container,
            fg_color=ERROR_BG,
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=ACCENT_RED,
        )
        bubble.pack(side="left", anchor="w", fill="x", expand=True)

        # Icon and title
        header = ctk.CTkFrame(bubble, fg_color="transparent")
        header.pack(fill="x", padx=MESSAGE_PADDING, pady=(MESSAGE_PADDING, 4))

        ctk.CTkLabel(
            header,
            text="⚠",
            font=get_font("large"),
            text_color=ACCENT_RED,
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text="Error",
            font=get_font("normal", bold=True),
            text_color=ACCENT_RED,
        ).pack(side="left", padx=(8, 0))

        # Message
        ctk.CTkLabel(
            bubble,
            text=message,
            font=get_font("normal"),
            text_color=TEXT_PRIMARY,
            wraplength=450,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=MESSAGE_PADDING)

        # Buttons
        if is_retryable or on_cancel:
            btn_frame = ctk.CTkFrame(bubble, fg_color="transparent")
            btn_frame.pack(fill="x", padx=MESSAGE_PADDING, pady=(8, MESSAGE_PADDING))

            if is_retryable and on_retry:
                ctk.CTkButton(
                    btn_frame,
                    text="Retry",
                    font=get_font("small"),
                    fg_color=ACCENT_BLUE,
                    hover_color="#3a8eef",
                    height=26,
                    width=60,
                    command=lambda: self._handle_retry(container, on_retry),
                ).pack(side="left", padx=(0, 8))

            if on_cancel:
                ctk.CTkButton(
                    btn_frame,
                    text="Cancel",
                    font=get_font("small"),
                    fg_color=BG_TERTIARY,
                    hover_color=BG_HOVER,
                    border_width=1,
                    border_color=BORDER_COLOR,
                    height=26,
                    width=60,
                    command=on_cancel,
                ).pack(side="left")

        self._message_widgets.append(container)
        self._scroll_to_bottom()

        return container

    def _handle_retry(
        self, error_frame: ctk.CTkFrame, on_retry: Callable[[], None]
    ) -> None:
        """Handle retry button click."""
        error_frame.destroy()
        if error_frame in self._message_widgets:
            self._message_widgets.remove(error_frame)
        on_retry()

    def remove_error(self, error_frame: ctk.CTkFrame) -> None:
        """Remove an error display."""
        if error_frame.winfo_exists():
            error_frame.destroy()
        if error_frame in self._message_widgets:
            self._message_widgets.remove(error_frame)

    def add_message(self, message: Message) -> None:
        """
        Add a Message object to the display.

        Args:
            message: The Message to display
        """
        if message.role == "user":
            self.add_user_message(message.content or "", message.timestamp)
        elif message.role == "assistant":
            if message.tool_calls:
                # Show AI message with tool calls
                if message.content or message.reasoning_content:
                    self.add_ai_message(
                        message.content, message.reasoning_content, message.timestamp
                    )
                for tc in message.tool_calls:
                    self.add_tool_call(tc, tc.result)
            else:
                self.add_ai_message(
                    message.content, message.reasoning_content, message.timestamp
                )
        # Tool messages are typically displayed as part of tool_calls

    def clear(self) -> None:
        """Clear all messages."""
        for widget in self._message_widgets:
            if widget.winfo_exists():
                widget.destroy()
        self._message_widgets.clear()
        self._streaming = StreamingState()
        self._show_welcome()

    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the message list."""
        self.scroll_frame.update_idletasks()
        self.scroll_frame._parent_canvas.yview_moveto(1.0)

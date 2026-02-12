"""
UI Components for the AI Assistant.

Provides all the visual components for the chat interface.
"""

from .chat_window import AIChatWindow
from .checkpoint_ui import CheckpointDropdown, CheckpointItem, CheckpointPopup
from .message_view import MessageView
from .model_selector import ModelSelector
from .styles import (
    AI_MESSAGE_BG,
    ERROR_BG,
    MESSAGE_PADDING,
    MESSAGE_SPACING,
    THINKING_BG,
    THINKING_TEXT,
    TOOL_BG,
    TOOL_FAILURE_COLOR,
    TOOL_SUCCESS_COLOR,
    USER_MESSAGE_BG,
    apply_button_style,
    get_font,
    get_scrollbar_style,
)
from .thinking import ThinkingSection

__all__ = [
    # Main window
    "AIChatWindow",
    # Components
    "MessageView",
    "ModelSelector",
    "CheckpointDropdown",
    "CheckpointPopup",
    "CheckpointItem",
    "ThinkingSection",
    # Styles
    "get_font",
    "apply_button_style",
    "get_scrollbar_style",
    "USER_MESSAGE_BG",
    "AI_MESSAGE_BG",
    "THINKING_BG",
    "THINKING_TEXT",
    "TOOL_BG",
    "TOOL_SUCCESS_COLOR",
    "TOOL_FAILURE_COLOR",
    "ERROR_BG",
    "MESSAGE_SPACING",
    "MESSAGE_PADDING",
]

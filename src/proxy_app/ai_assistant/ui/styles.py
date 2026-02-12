"""
UI Styles and Constants for the AI Assistant.

Imports colors from model_filter_gui.py for consistency
and defines assistant-specific styles.
"""

# Import base colors from model_filter_gui
# Using the same color scheme for visual consistency
from ...model_filter_gui import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_YELLOW,
    BG_HOVER,
    BG_PRIMARY,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER_COLOR,
    FONT_FAMILY,
    FONT_SIZE_LARGE,
    FONT_SIZE_NORMAL,
    FONT_SIZE_SMALL,
    FONT_SIZE_TITLE,
    HIGHLIGHT_BG,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

# Re-export base colors
__all__ = [
    # Base colors (from model_filter_gui)
    "BG_PRIMARY",
    "BG_SECONDARY",
    "BG_TERTIARY",
    "BG_HOVER",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "TEXT_MUTED",
    "ACCENT_BLUE",
    "ACCENT_GREEN",
    "ACCENT_RED",
    "ACCENT_YELLOW",
    "BORDER_COLOR",
    "HIGHLIGHT_BG",
    "FONT_FAMILY",
    "FONT_SIZE_SMALL",
    "FONT_SIZE_NORMAL",
    "FONT_SIZE_LARGE",
    "FONT_SIZE_TITLE",
    # Assistant-specific
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
    "CORNER_RADIUS",
    "INPUT_MIN_HEIGHT",
    "INPUT_MAX_HEIGHT",
]

# ============================================================================
# Assistant-Specific Colors
# ============================================================================

# Message backgrounds
USER_MESSAGE_BG = "#2a3f5f"  # Blue-tinted, right-aligned
AI_MESSAGE_BG = BG_SECONDARY  # Same as card backgrounds

# Thinking section
THINKING_BG = "#1a1a2a"  # Slightly darker
THINKING_TEXT = TEXT_MUTED  # Muted text for thinking

# Tool execution
TOOL_BG = "#1e2838"  # Subtle background for tool displays
TOOL_SUCCESS_COLOR = ACCENT_GREEN  # Checkmark color
TOOL_FAILURE_COLOR = ACCENT_RED  # X color

# Error display
ERROR_BG = "#3d2020"  # Dark red tint for error messages

# ============================================================================
# Layout Constants
# ============================================================================

# Message display
MESSAGE_SPACING = 12  # Vertical spacing between messages
MESSAGE_PADDING = 12  # Internal padding for message boxes
CORNER_RADIUS = 8  # Border radius for message boxes

# Input area
INPUT_MIN_HEIGHT = 55  # Minimum height (about 2 lines)
INPUT_MAX_HEIGHT = 200  # Maximum height before scrolling

# ============================================================================
# Font Configurations
# ============================================================================


def get_font(
    size: str = "normal", bold: bool = False, monospace: bool = False
) -> tuple:
    """
    Get a font tuple for CTk widgets.

    Args:
        size: "small", "normal", "large", or "title"
        bold: Whether to use bold weight
        monospace: Whether to use monospace font

    Returns:
        Tuple of (family, size, weight)
    """
    family = "Consolas" if monospace else FONT_FAMILY

    sizes = {
        "small": FONT_SIZE_SMALL,
        "normal": FONT_SIZE_NORMAL,
        "large": FONT_SIZE_LARGE,
        "title": FONT_SIZE_TITLE,
    }
    font_size = sizes.get(size, FONT_SIZE_NORMAL)

    if bold:
        return (family, font_size, "bold")
    return (family, font_size)


# ============================================================================
# Widget Style Helpers
# ============================================================================


def apply_button_style(button, style: str = "primary") -> None:
    """
    Apply a predefined style to a CTkButton.

    Args:
        button: The CTkButton to style
        style: "primary", "secondary", "danger", or "ghost"
    """
    styles = {
        "primary": {
            "fg_color": ACCENT_BLUE,
            "hover_color": "#3a8eef",
            "text_color": TEXT_PRIMARY,
        },
        "secondary": {
            "fg_color": BG_SECONDARY,
            "hover_color": BG_HOVER,
            "text_color": TEXT_PRIMARY,
            "border_width": 1,
            "border_color": BORDER_COLOR,
        },
        "danger": {
            "fg_color": ACCENT_RED,
            "hover_color": "#c0392b",
            "text_color": TEXT_PRIMARY,
        },
        "ghost": {
            "fg_color": "transparent",
            "hover_color": BG_HOVER,
            "text_color": TEXT_SECONDARY,
        },
        "success": {
            "fg_color": ACCENT_GREEN,
            "hover_color": "#27ae60",
            "text_color": TEXT_PRIMARY,
        },
    }

    if style in styles:
        button.configure(**styles[style])


def get_scrollbar_style() -> dict:
    """Get the style configuration for scrollbars."""
    return {
        "button_color": BG_HOVER,
        "button_hover_color": ACCENT_BLUE,
        "fg_color": BG_TERTIARY,
    }

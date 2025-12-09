"""
Collapsible Thinking Section Widget.

Displays the AI's thinking/reasoning content in a collapsible panel.
Auto-collapses when content starts arriving, can be manually toggled.
"""

import customtkinter as ctk
from typing import Optional

from .styles import (
    BG_TERTIARY,
    BORDER_COLOR,
    FONT_FAMILY,
    FONT_SIZE_SMALL,
    TEXT_MUTED,
    TEXT_SECONDARY,
    THINKING_BG,
    THINKING_TEXT,
    get_font,
)


class ThinkingSection(ctk.CTkFrame):
    """
    Collapsible section for displaying AI thinking/reasoning.

    Features:
    - Click header to expand/collapse
    - Auto-collapse when content starts arriving
    - Streaming text support
    - Muted styling to distinguish from main content
    """

    def __init__(
        self,
        parent,
        initial_expanded: bool = True,
        max_collapsed_preview: int = 100,
        **kwargs,
    ):
        """
        Initialize the thinking section.

        Args:
            parent: Parent widget
            initial_expanded: Whether to start expanded
            max_collapsed_preview: Max characters to show when collapsed
            **kwargs: Additional frame arguments
        """
        super().__init__(parent, fg_color=THINKING_BG, corner_radius=6, **kwargs)

        self._expanded = initial_expanded
        self._max_preview = max_collapsed_preview
        self._full_text = ""
        self._auto_collapsed = False

        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the UI widgets."""
        # Header (clickable)
        self.header = ctk.CTkFrame(self, fg_color="transparent", cursor="hand2")
        self.header.pack(fill="x", padx=8, pady=(6, 0))

        # Expand/collapse indicator
        self.indicator = ctk.CTkLabel(
            self.header,
            text="▼" if self._expanded else "▶",
            font=get_font("small"),
            text_color=TEXT_MUTED,
            width=16,
        )
        self.indicator.pack(side="left")

        # Title
        self.title_label = ctk.CTkLabel(
            self.header,
            text="Thinking",
            font=get_font("small"),
            text_color=TEXT_MUTED,
        )
        self.title_label.pack(side="left", padx=(4, 0))

        # Preview text (shown when collapsed)
        self.preview_label = ctk.CTkLabel(
            self.header,
            text="",
            font=get_font("small"),
            text_color=TEXT_MUTED,
            anchor="w",
        )

        # Content container
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        if self._expanded:
            self.content_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        # Text display
        self.text_display = ctk.CTkTextbox(
            self.content_frame,
            font=get_font("small"),
            fg_color=THINKING_BG,
            text_color=THINKING_TEXT,
            wrap="word",
            height=100,
            activate_scrollbars=True,
            state="disabled",
        )
        self.text_display.pack(fill="both", expand=True)

        # Bind click events
        self.header.bind("<Button-1>", self._toggle)
        self.indicator.bind("<Button-1>", self._toggle)
        self.title_label.bind("<Button-1>", self._toggle)
        self.preview_label.bind("<Button-1>", self._toggle)

    def _toggle(self, event=None) -> None:
        """Toggle expanded/collapsed state."""
        self._expanded = not self._expanded
        self._auto_collapsed = False
        self._update_display()

    def _update_display(self) -> None:
        """Update the display based on current state."""
        if self._expanded:
            self.indicator.configure(text="▼")
            self.preview_label.pack_forget()
            self.content_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        else:
            self.indicator.configure(text="▶")
            self.content_frame.pack_forget()
            # Show preview
            preview = self._full_text[: self._max_preview]
            if len(self._full_text) > self._max_preview:
                preview += "..."
            preview = preview.replace("\n", " ")
            self.preview_label.configure(text=f" - {preview}" if preview else "")
            self.preview_label.pack(side="left", padx=(8, 0), fill="x", expand=True)

    def append_text(self, text: str) -> None:
        """
        Append text to the thinking content.

        Args:
            text: Text chunk to append
        """
        self._full_text += text

        # Update text display
        self.text_display.configure(state="normal")
        self.text_display.insert("end", text)
        self.text_display.configure(state="disabled")
        self.text_display.see("end")

        # Update preview if collapsed
        if not self._expanded:
            self._update_display()

    def set_text(self, text: str) -> None:
        """
        Set the full thinking text.

        Args:
            text: Complete thinking text
        """
        self._full_text = text

        self.text_display.configure(state="normal")
        self.text_display.delete("1.0", "end")
        self.text_display.insert("1.0", text)
        self.text_display.configure(state="disabled")

        self._update_display()

    def auto_collapse(self) -> None:
        """Auto-collapse when content starts arriving."""
        if self._expanded and not self._auto_collapsed:
            self._expanded = False
            self._auto_collapsed = True
            self._update_display()

    def expand(self) -> None:
        """Expand the section."""
        if not self._expanded:
            self._expanded = True
            self._auto_collapsed = False
            self._update_display()

    def collapse(self) -> None:
        """Collapse the section."""
        if self._expanded:
            self._expanded = False
            self._update_display()

    def clear(self) -> None:
        """Clear all content."""
        self._full_text = ""
        self.text_display.configure(state="normal")
        self.text_display.delete("1.0", "end")
        self.text_display.configure(state="disabled")
        self._update_display()

    @property
    def is_expanded(self) -> bool:
        """Check if currently expanded."""
        return self._expanded

    @property
    def text(self) -> str:
        """Get the full thinking text."""
        return self._full_text

    @property
    def has_content(self) -> bool:
        """Check if there is any thinking content."""
        return bool(self._full_text.strip())

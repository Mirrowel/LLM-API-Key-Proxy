"""
Checkpoint UI Components.

Provides a dropdown/popup for viewing and rolling back to checkpoints.
"""

import customtkinter as ctk
import tkinter as tk
from typing import Callable, List, Optional

from ..checkpoint import Checkpoint
from .styles import (
    ACCENT_BLUE,
    ACCENT_RED,
    ACCENT_YELLOW,
    BG_HOVER,
    BG_PRIMARY,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER_COLOR,
    FONT_FAMILY,
    FONT_SIZE_NORMAL,
    FONT_SIZE_SMALL,
    HIGHLIGHT_BG,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_font,
)


class CheckpointDropdown(ctk.CTkFrame):
    """
    Dropdown button that opens a checkpoint selection popup.

    Shows the list of checkpoints and allows rolling back to any point.
    """

    def __init__(
        self, parent, on_rollback: Optional[Callable[[str], None]] = None, **kwargs
    ):
        """
        Initialize the checkpoint dropdown.

        Args:
            parent: Parent widget
            on_rollback: Callback when rollback is confirmed (receives checkpoint_id)
            **kwargs: Additional frame arguments
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._on_rollback = on_rollback
        self._checkpoints: List[Checkpoint] = []
        self._popup: Optional[CheckpointPopup] = None

        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the UI widgets."""
        # Dropdown button
        self.button = ctk.CTkButton(
            self,
            text="Checkpoints",
            font=get_font("normal"),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            border_width=1,
            border_color=BORDER_COLOR,
            width=120,
            height=28,
            command=self._toggle_popup,
        )
        self.button.pack(side="left")

        # Count badge (hidden when 0)
        self.badge = ctk.CTkLabel(
            self,
            text="0",
            font=get_font("small"),
            fg_color=ACCENT_BLUE,
            text_color=TEXT_PRIMARY,
            corner_radius=10,
            width=20,
            height=20,
        )
        # Initially hidden

    def _toggle_popup(self) -> None:
        """Toggle the checkpoint popup."""
        if self._popup is not None and self._popup.winfo_exists():
            self._popup.destroy()
            self._popup = None
        else:
            self._show_popup()

    def _show_popup(self) -> None:
        """Show the checkpoint selection popup."""
        if not self._checkpoints:
            # Show a simple message if no checkpoints
            self._popup = CheckpointPopup(
                self,
                checkpoints=[],
                on_rollback=self._handle_rollback,
                on_close=self._close_popup,
            )
        else:
            self._popup = CheckpointPopup(
                self,
                checkpoints=self._checkpoints,
                on_rollback=self._handle_rollback,
                on_close=self._close_popup,
            )

        # Position below the button
        x = self.button.winfo_rootx()
        y = self.button.winfo_rooty() + self.button.winfo_height() + 4
        self._popup.geometry(f"+{x}+{y}")

    def _handle_rollback(self, checkpoint_id: str) -> None:
        """Handle rollback confirmation."""
        self._close_popup()
        if self._on_rollback:
            self._on_rollback(checkpoint_id)

    def _close_popup(self) -> None:
        """Close the popup."""
        if self._popup is not None:
            self._popup.destroy()
            self._popup = None

    def set_checkpoints(self, checkpoints: List[Checkpoint]) -> None:
        """
        Update the list of checkpoints.

        Args:
            checkpoints: List of Checkpoint objects
        """
        self._checkpoints = checkpoints

        # Update badge
        count = len(checkpoints)
        if count > 0:
            self.badge.configure(text=str(count))
            self.badge.pack(side="left", padx=(4, 0))
        else:
            self.badge.pack_forget()

        # Update popup if open
        if self._popup is not None and self._popup.winfo_exists():
            self._popup.update_checkpoints(checkpoints)

    @property
    def checkpoint_count(self) -> int:
        """Get the number of checkpoints."""
        return len(self._checkpoints)


class CheckpointPopup(ctk.CTkToplevel):
    """
    Popup window for checkpoint selection.

    Shows a list of checkpoints with timestamps and descriptions,
    allows selecting one for rollback.
    """

    def __init__(
        self,
        parent,
        checkpoints: List[Checkpoint],
        on_rollback: Callable[[str], None],
        on_close: Callable[[], None],
        **kwargs,
    ):
        """
        Initialize the popup.

        Args:
            parent: Parent widget
            checkpoints: List of checkpoints to display
            on_rollback: Callback when rollback is confirmed
            on_close: Callback when popup is closed
        """
        super().__init__(parent, **kwargs)

        self._checkpoints = checkpoints
        self._on_rollback = on_rollback
        self._on_close = on_close
        self._selected_id: Optional[str] = None

        # Window configuration
        self.title("Checkpoints")
        self.configure(fg_color=BG_PRIMARY)
        self.overrideredirect(True)  # No window decorations
        self.attributes("-topmost", True)

        # Bind focus loss
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Escape>", lambda e: self._close())

        self._create_widgets()

        # Focus the window
        self.focus_force()

    def _create_widgets(self) -> None:
        """Create the UI widgets."""
        # Main container with border
        self.container = ctk.CTkFrame(
            self,
            fg_color=BG_SECONDARY,
            border_width=1,
            border_color=BORDER_COLOR,
            corner_radius=8,
        )
        self.container.pack(fill="both", expand=True, padx=2, pady=2)

        # Header
        header = ctk.CTkFrame(self.container, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(
            header,
            text="Checkpoints",
            font=get_font("large", bold=True),
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        ctk.CTkButton(
            header,
            text="×",
            font=get_font("large"),
            fg_color="transparent",
            hover_color=BG_HOVER,
            text_color=TEXT_MUTED,
            width=24,
            height=24,
            command=self._close,
        ).pack(side="right")

        # Scrollable list container
        self.list_frame = ctk.CTkScrollableFrame(
            self.container,
            fg_color=BG_TERTIARY,
            corner_radius=4,
            height=250,
            width=350,
        )
        self.list_frame.pack(fill="both", expand=True, padx=8, pady=4)

        # Populate list
        self._populate_list()

        # Buttons
        button_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        button_frame.pack(fill="x", padx=8, pady=(4, 8))

        self.rollback_btn = ctk.CTkButton(
            button_frame,
            text="Rollback to Selected",
            font=get_font("normal"),
            fg_color=ACCENT_YELLOW,
            hover_color="#d4a910",
            text_color=BG_PRIMARY,
            height=28,
            state="disabled",
            command=self._confirm_rollback,
        )
        self.rollback_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(
            button_frame,
            text="Cancel",
            font=get_font("normal"),
            fg_color=BG_TERTIARY,
            hover_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            border_width=1,
            border_color=BORDER_COLOR,
            height=28,
            width=70,
            command=self._close,
        ).pack(side="right")

    def _populate_list(self) -> None:
        """Populate the checkpoint list."""
        # Clear existing items
        for widget in self.list_frame.winfo_children():
            widget.destroy()

        if not self._checkpoints:
            ctk.CTkLabel(
                self.list_frame,
                text="No checkpoints yet.\n\nCheckpoints are created when\nthe AI makes changes.",
                font=get_font("normal"),
                text_color=TEXT_MUTED,
                justify="center",
            ).pack(pady=40)
            return

        # Current state indicator
        current_item = CheckpointItem(
            self.list_frame,
            text="Current State",
            subtext="No changes to undo",
            is_current=True,
            on_select=lambda: self._select(None),
        )
        current_item.pack(fill="x", pady=(0, 4))

        # Add separator
        sep = ctk.CTkFrame(self.list_frame, fg_color=BORDER_COLOR, height=1)
        sep.pack(fill="x", pady=4)

        # Add checkpoints (newest first)
        for checkpoint in reversed(self._checkpoints):
            item = CheckpointItem(
                self.list_frame,
                text=checkpoint.get_display_text(),
                subtext=self._get_checkpoint_subtext(checkpoint),
                checkpoint_id=checkpoint.id,
                on_select=lambda cid=checkpoint.id: self._select(cid),
            )
            item.pack(fill="x", pady=2)

    def _get_checkpoint_subtext(self, checkpoint: Checkpoint) -> str:
        """Get subtext for a checkpoint."""
        if checkpoint.tool_calls:
            first = checkpoint.tool_calls[0]
            return f"→ {first.message}"
        return checkpoint.description

    def _select(self, checkpoint_id: Optional[str]) -> None:
        """Select a checkpoint."""
        self._selected_id = checkpoint_id

        # Update button state
        if checkpoint_id is None:
            self.rollback_btn.configure(state="disabled")
        else:
            self.rollback_btn.configure(state="normal")

    def _confirm_rollback(self) -> None:
        """Confirm and execute rollback."""
        if self._selected_id:
            self._on_rollback(self._selected_id)

    def _close(self) -> None:
        """Close the popup."""
        self._on_close()

    def _on_focus_out(self, event) -> None:
        """Handle focus loss."""
        # Check if focus went to a child widget
        if event.widget == self:
            # Small delay to allow button clicks to register
            self.after(100, self._check_focus)

    def _check_focus(self) -> None:
        """Check if we should close after focus loss."""
        try:
            focused = self.focus_get()
            if focused is None or not str(focused).startswith(str(self)):
                self._close()
        except tk.TclError:
            self._close()

    def update_checkpoints(self, checkpoints: List[Checkpoint]) -> None:
        """Update the checkpoint list."""
        self._checkpoints = checkpoints
        self._selected_id = None
        self._populate_list()
        self.rollback_btn.configure(state="disabled")


class CheckpointItem(ctk.CTkFrame):
    """
    Individual checkpoint item in the list.
    """

    def __init__(
        self,
        parent,
        text: str,
        subtext: str = "",
        checkpoint_id: Optional[str] = None,
        is_current: bool = False,
        on_select: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        """
        Initialize the checkpoint item.

        Args:
            parent: Parent widget
            text: Main text (timestamp + tool)
            subtext: Description text
            checkpoint_id: ID of the checkpoint (None for current state)
            is_current: Whether this is the current state indicator
            on_select: Callback when selected
        """
        super().__init__(
            parent,
            fg_color=HIGHLIGHT_BG if is_current else "transparent",
            corner_radius=4,
            cursor="hand2",
            **kwargs,
        )

        self._checkpoint_id = checkpoint_id
        self._is_current = is_current
        self._is_selected = False
        self._on_select = on_select

        self._create_widgets(text, subtext)
        self._bind_events()

    def _create_widgets(self, text: str, subtext: str) -> None:
        """Create the UI widgets."""
        # Radio indicator
        self.radio = ctk.CTkLabel(
            self,
            text="○" if not self._is_current else "●",
            font=get_font("normal"),
            text_color=ACCENT_BLUE if self._is_current else TEXT_MUTED,
            width=20,
        )
        self.radio.pack(side="left", padx=(8, 4), pady=6)

        # Text container
        text_frame = ctk.CTkFrame(self, fg_color="transparent")
        text_frame.pack(side="left", fill="x", expand=True, pady=4)

        # Main text
        self.text_label = ctk.CTkLabel(
            text_frame,
            text=text,
            font=get_font("normal"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        self.text_label.pack(fill="x")

        # Subtext
        if subtext:
            self.subtext_label = ctk.CTkLabel(
                text_frame,
                text=subtext,
                font=get_font("small"),
                text_color=TEXT_MUTED,
                anchor="w",
            )
            self.subtext_label.pack(fill="x")

    def _bind_events(self) -> None:
        """Bind mouse events."""
        widgets = [self, self.radio, self.text_label]
        if hasattr(self, "subtext_label"):
            widgets.append(self.subtext_label)

        for widget in widgets:
            widget.bind("<Button-1>", self._on_click)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

    def _on_click(self, event=None) -> None:
        """Handle click."""
        if not self._is_current and self._on_select:
            self._is_selected = True
            self.radio.configure(text="●", text_color=ACCENT_BLUE)
            self.configure(fg_color=HIGHLIGHT_BG)
            self._on_select()

    def _on_enter(self, event=None) -> None:
        """Handle mouse enter."""
        if not self._is_current and not self._is_selected:
            self.configure(fg_color=BG_HOVER)

    def _on_leave(self, event=None) -> None:
        """Handle mouse leave."""
        if not self._is_current and not self._is_selected:
            self.configure(fg_color="transparent")

"""
Grouped Model Selector Widget.

Dropdown for selecting LLM models, grouped by provider.
"""

import customtkinter as ctk
from typing import Callable, Dict, List, Optional

from .styles import (
    ACCENT_BLUE,
    BG_HOVER,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER_COLOR,
    FONT_FAMILY,
    FONT_SIZE_NORMAL,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_font,
)


class ModelSelector(ctk.CTkFrame):
    """
    Dropdown for selecting LLM models, grouped by provider.

    Features:
    - Models grouped by provider (openai, gemini, anthropic, etc.)
    - Search/filter capability (future)
    - Displays current selection
    - Callback on selection change
    """

    def __init__(
        self, parent, on_model_changed: Optional[Callable[[str], None]] = None, **kwargs
    ):
        """
        Initialize the model selector.

        Args:
            parent: Parent widget
            on_model_changed: Callback when model selection changes
            **kwargs: Additional frame arguments
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._on_model_changed = on_model_changed
        self._models: Dict[str, List[str]] = {}  # provider -> [model_ids]
        self._flat_models: List[str] = []  # All model IDs
        self._current_model: Optional[str] = None

        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create the UI widgets."""
        # Label
        self.label = ctk.CTkLabel(
            self,
            text="Model:",
            font=get_font("normal"),
            text_color=TEXT_SECONDARY,
        )
        self.label.pack(side="left", padx=(0, 8))

        # Dropdown
        self.dropdown = ctk.CTkComboBox(
            self,
            values=["Loading..."],
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
            width=280,
            state="readonly",
            command=self._on_selection,
        )
        self.dropdown.pack(side="left")
        self.dropdown.set("Loading...")

        # Bind mousewheel to dropdown for scrolling through options
        self._bind_mousewheel()

    def set_models(self, models: Dict[str, List[str]]) -> None:
        """
        Set the available models.

        Args:
            models: Dict of provider -> list of model IDs
        """
        self._models = models
        self._flat_models = []

        # Build flat list with group headers
        display_values = []

        for provider in sorted(models.keys()):
            provider_models = models[provider]
            if not provider_models:
                continue

            # Add models (full provider/model format)
            for model_id in sorted(provider_models):
                # Model ID might already include provider prefix
                if "/" in model_id:
                    display_values.append(model_id)
                    self._flat_models.append(model_id)
                else:
                    full_id = f"{provider}/{model_id}"
                    display_values.append(full_id)
                    self._flat_models.append(full_id)

        if display_values:
            self.dropdown.configure(values=display_values, state="readonly")

            # Keep current selection if still valid
            if self._current_model and self._current_model in self._flat_models:
                self.dropdown.set(self._current_model)
            else:
                # Select first model
                self._current_model = display_values[0]
                self.dropdown.set(display_values[0])
        else:
            self.dropdown.configure(values=["No models available"], state="disabled")
            self.dropdown.set("No models available")
            self._current_model = None

    def _on_selection(self, choice: str) -> None:
        """Handle dropdown selection."""
        if choice and choice != "Loading..." and choice != "No models available":
            self._current_model = choice
            if self._on_model_changed:
                self._on_model_changed(choice)

    def get_selected_model(self) -> Optional[str]:
        """Get the currently selected model ID."""
        return self._current_model

    def set_selected_model(self, model_id: str) -> bool:
        """
        Set the selected model.

        Args:
            model_id: The model ID to select

        Returns:
            True if model was found and selected
        """
        if model_id in self._flat_models:
            self._current_model = model_id
            self.dropdown.set(model_id)
            return True
        return False

    def set_loading(self) -> None:
        """Set the dropdown to loading state."""
        self.dropdown.configure(values=["Loading..."], state="disabled")
        self.dropdown.set("Loading...")

    def set_error(self, message: str = "Failed to load models") -> None:
        """Set the dropdown to error state."""
        self.dropdown.configure(values=[message], state="disabled")
        self.dropdown.set(message)

    @property
    def has_models(self) -> bool:
        """Check if models are loaded."""
        return bool(self._flat_models)

    def _bind_mousewheel(self) -> None:
        """Bind mousewheel events to cycle through models when dropdown is focused."""

        def on_mousewheel(event):
            if not self._flat_models:
                return

            current = self._current_model
            if current not in self._flat_models:
                return

            current_idx = self._flat_models.index(current)

            # Scroll direction (Windows uses event.delta, Linux uses event.num)
            if event.delta > 0 or event.num == 4:
                # Scroll up - previous model
                new_idx = max(0, current_idx - 1)
            else:
                # Scroll down - next model
                new_idx = min(len(self._flat_models) - 1, current_idx + 1)

            if new_idx != current_idx:
                new_model = self._flat_models[new_idx]
                self._current_model = new_model
                self.dropdown.set(new_model)
                if self._on_model_changed:
                    self._on_model_changed(new_model)

        # Bind to the dropdown widget (works when hovering over it)
        self.dropdown.bind("<MouseWheel>", on_mousewheel)  # Windows
        self.dropdown.bind("<Button-4>", on_mousewheel)  # Linux scroll up
        self.dropdown.bind("<Button-5>", on_mousewheel)  # Linux scroll down

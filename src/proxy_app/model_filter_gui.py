"""
Model Filter GUI - Visual editor for model ignore/whitelist rules.

A CustomTkinter application that provides a friendly interface for managing
which models are available per provider through ignore lists and whitelists.

Features:
- Two synchronized model lists showing all fetched models and their filtered status
- Color-coded rules with visual association to affected models
- Real-time filtering preview as you type patterns
- Click interactions to highlight rule-model relationships
- Right-click context menus for quick actions
- Comprehensive help documentation
"""

import customtkinter as ctk
from tkinter import Menu
import asyncio
import threading
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Set
from dotenv import load_dotenv, set_key, unset_key


# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

# Window settings
WINDOW_TITLE = "Model Filter Configuration"
WINDOW_DEFAULT_SIZE = "1000x750"
WINDOW_MIN_WIDTH = 850
WINDOW_MIN_HEIGHT = 600

# Color scheme (dark mode)
BG_PRIMARY = "#1a1a2e"  # Main background
BG_SECONDARY = "#16213e"  # Card/panel background
BG_TERTIARY = "#0f0f1a"  # Input fields, lists
BG_HOVER = "#1f2b47"  # Hover state
BORDER_COLOR = "#2a2a4a"  # Subtle borders
TEXT_PRIMARY = "#e8e8e8"  # Main text
TEXT_SECONDARY = "#a0a0a0"  # Muted text
TEXT_MUTED = "#666680"  # Very muted text
ACCENT_BLUE = "#4a9eff"  # Primary accent
ACCENT_GREEN = "#2ecc71"  # Success/normal
ACCENT_RED = "#e74c3c"  # Danger/ignore
ACCENT_YELLOW = "#f1c40f"  # Warning

# Status colors
NORMAL_COLOR = "#2ecc71"  # Green - models not affected by any rule
HIGHLIGHT_BG = "#2a3a5a"  # Background for highlighted items

# Ignore rules - warm color progression (reds/oranges)
IGNORE_COLORS = [
    "#e74c3c",  # Bright red
    "#c0392b",  # Dark red
    "#e67e22",  # Orange
    "#d35400",  # Dark orange
    "#f39c12",  # Gold
    "#e91e63",  # Pink
    "#ff5722",  # Deep orange
    "#f44336",  # Material red
    "#ff6b6b",  # Coral
    "#ff8a65",  # Light deep orange
]

# Whitelist rules - cool color progression (blues/teals)
WHITELIST_COLORS = [
    "#3498db",  # Blue
    "#2980b9",  # Dark blue
    "#1abc9c",  # Teal
    "#16a085",  # Dark teal
    "#9b59b6",  # Purple
    "#8e44ad",  # Dark purple
    "#00bcd4",  # Cyan
    "#2196f3",  # Material blue
    "#64b5f6",  # Light blue
    "#4dd0e1",  # Light cyan
]

# Font configuration
FONT_FAMILY = "Segoe UI"
FONT_SIZE_SMALL = 11
FONT_SIZE_NORMAL = 12
FONT_SIZE_LARGE = 14
FONT_SIZE_TITLE = 16
FONT_SIZE_HEADER = 20


# ════════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class FilterRule:
    """Represents a single filter rule (ignore or whitelist pattern)."""

    pattern: str
    color: str
    rule_type: str  # 'ignore' or 'whitelist'
    affected_count: int = 0
    affected_models: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash((self.pattern, self.rule_type))

    def __eq__(self, other):
        if not isinstance(other, FilterRule):
            return False
        return self.pattern == other.pattern and self.rule_type == other.rule_type


@dataclass
class ModelStatus:
    """Status information for a single model."""

    model_id: str
    status: str  # 'normal', 'ignored', 'whitelisted'
    color: str
    affecting_rule: Optional[FilterRule] = None

    @property
    def display_name(self) -> str:
        """Get the model name without provider prefix for display."""
        if "/" in self.model_id:
            return self.model_id.split("/", 1)[1]
        return self.model_id

    @property
    def provider(self) -> str:
        """Extract provider from model ID."""
        if "/" in self.model_id:
            return self.model_id.split("/")[0]
        return ""


# ════════════════════════════════════════════════════════════════════════════════
# FILTER ENGINE
# ════════════════════════════════════════════════════════════════════════════════


class FilterEngine:
    """
    Core filtering logic with rule management.

    Handles pattern matching, rule storage, and status calculation.
    Tracks changes for save/discard functionality.
    """

    def __init__(self):
        self.ignore_rules: List[FilterRule] = []
        self.whitelist_rules: List[FilterRule] = []
        self._ignore_color_index = 0
        self._whitelist_color_index = 0
        self._original_ignore_patterns: Set[str] = set()
        self._original_whitelist_patterns: Set[str] = set()
        self._current_provider: Optional[str] = None

    def reset(self):
        """Clear all rules and reset state."""
        self.ignore_rules.clear()
        self.whitelist_rules.clear()
        self._ignore_color_index = 0
        self._whitelist_color_index = 0
        self._original_ignore_patterns.clear()
        self._original_whitelist_patterns.clear()

    def _get_next_ignore_color(self) -> str:
        """Get next color for ignore rules (cycles through palette)."""
        color = IGNORE_COLORS[self._ignore_color_index % len(IGNORE_COLORS)]
        self._ignore_color_index += 1
        return color

    def _get_next_whitelist_color(self) -> str:
        """Get next color for whitelist rules (cycles through palette)."""
        color = WHITELIST_COLORS[self._whitelist_color_index % len(WHITELIST_COLORS)]
        self._whitelist_color_index += 1
        return color

    def add_ignore_rule(self, pattern: str) -> Optional[FilterRule]:
        """Add a new ignore rule. Returns the rule if added, None if duplicate."""
        pattern = pattern.strip()
        if not pattern:
            return None

        # Check for duplicates
        for rule in self.ignore_rules:
            if rule.pattern == pattern:
                return None

        rule = FilterRule(
            pattern=pattern, color=self._get_next_ignore_color(), rule_type="ignore"
        )
        self.ignore_rules.append(rule)
        return rule

    def add_whitelist_rule(self, pattern: str) -> Optional[FilterRule]:
        """Add a new whitelist rule. Returns the rule if added, None if duplicate."""
        pattern = pattern.strip()
        if not pattern:
            return None

        # Check for duplicates
        for rule in self.whitelist_rules:
            if rule.pattern == pattern:
                return None

        rule = FilterRule(
            pattern=pattern,
            color=self._get_next_whitelist_color(),
            rule_type="whitelist",
        )
        self.whitelist_rules.append(rule)
        return rule

    def remove_ignore_rule(self, pattern: str) -> bool:
        """Remove an ignore rule by pattern. Returns True if removed."""
        for i, rule in enumerate(self.ignore_rules):
            if rule.pattern == pattern:
                self.ignore_rules.pop(i)
                return True
        return False

    def remove_whitelist_rule(self, pattern: str) -> bool:
        """Remove a whitelist rule by pattern. Returns True if removed."""
        for i, rule in enumerate(self.whitelist_rules):
            if rule.pattern == pattern:
                self.whitelist_rules.pop(i)
                return True
        return False

    def _pattern_matches(self, model_id: str, pattern: str) -> bool:
        """
        Check if a pattern matches a model ID.

        Supports:
        - Exact match: "gpt-4" matches only "gpt-4"
        - Prefix wildcard: "gpt-4*" matches "gpt-4", "gpt-4-turbo", etc.
        - Match all: "*" matches everything
        """
        # Extract model name without provider prefix
        if "/" in model_id:
            provider_model_name = model_id.split("/", 1)[1]
        else:
            provider_model_name = model_id

        if pattern == "*":
            return True
        elif pattern.endswith("*"):
            prefix = pattern[:-1]
            return provider_model_name.startswith(prefix) or model_id.startswith(prefix)
        else:
            # Exact match against full ID or provider model name
            return model_id == pattern or provider_model_name == pattern

    def get_model_status(self, model_id: str) -> ModelStatus:
        """
        Determine the status of a model based on current rules.

        Priority: Whitelist > Ignore > Normal
        """
        # Check whitelist first (takes priority)
        for rule in self.whitelist_rules:
            if self._pattern_matches(model_id, rule.pattern):
                return ModelStatus(
                    model_id=model_id,
                    status="whitelisted",
                    color=rule.color,
                    affecting_rule=rule,
                )

        # Then check ignore
        for rule in self.ignore_rules:
            if self._pattern_matches(model_id, rule.pattern):
                return ModelStatus(
                    model_id=model_id,
                    status="ignored",
                    color=rule.color,
                    affecting_rule=rule,
                )

        # Default: normal
        return ModelStatus(
            model_id=model_id, status="normal", color=NORMAL_COLOR, affecting_rule=None
        )

    def get_all_statuses(self, models: List[str]) -> List[ModelStatus]:
        """Get status for all models."""
        return [self.get_model_status(m) for m in models]

    def update_affected_counts(self, models: List[str]):
        """Update the affected_count and affected_models for all rules."""
        # Reset counts
        for rule in self.ignore_rules + self.whitelist_rules:
            rule.affected_count = 0
            rule.affected_models = []

        # Count affected models
        for model_id in models:
            status = self.get_model_status(model_id)
            if status.affecting_rule:
                status.affecting_rule.affected_count += 1
                status.affecting_rule.affected_models.append(model_id)

    def get_available_count(self, models: List[str]) -> Tuple[int, int]:
        """Returns (available_count, total_count)."""
        available = 0
        for model_id in models:
            status = self.get_model_status(model_id)
            if status.status != "ignored":
                available += 1
        return available, len(models)

    def preview_pattern(
        self, pattern: str, rule_type: str, models: List[str]
    ) -> List[str]:
        """
        Preview which models would be affected by a pattern without adding it.
        Returns list of affected model IDs.
        """
        affected = []
        pattern = pattern.strip()
        if not pattern:
            return affected

        for model_id in models:
            if self._pattern_matches(model_id, pattern):
                affected.append(model_id)

        return affected

    def load_from_env(self, provider: str):
        """Load ignore/whitelist rules for a provider from environment."""
        self.reset()
        self._current_provider = provider
        load_dotenv(override=True)

        # Load ignore list
        ignore_key = f"IGNORE_MODELS_{provider.upper()}"
        ignore_value = os.getenv(ignore_key, "")
        if ignore_value:
            patterns = [p.strip() for p in ignore_value.split(",") if p.strip()]
            for pattern in patterns:
                self.add_ignore_rule(pattern)
            self._original_ignore_patterns = set(patterns)

        # Load whitelist
        whitelist_key = f"WHITELIST_MODELS_{provider.upper()}"
        whitelist_value = os.getenv(whitelist_key, "")
        if whitelist_value:
            patterns = [p.strip() for p in whitelist_value.split(",") if p.strip()]
            for pattern in patterns:
                self.add_whitelist_rule(pattern)
            self._original_whitelist_patterns = set(patterns)

    def save_to_env(self, provider: str) -> bool:
        """
        Save current rules to .env file.
        Returns True if successful.
        """
        env_path = Path.cwd() / ".env"

        try:
            ignore_key = f"IGNORE_MODELS_{provider.upper()}"
            whitelist_key = f"WHITELIST_MODELS_{provider.upper()}"

            # Save ignore patterns
            ignore_patterns = [rule.pattern for rule in self.ignore_rules]
            if ignore_patterns:
                set_key(str(env_path), ignore_key, ",".join(ignore_patterns))
            else:
                # Remove the key if no patterns
                unset_key(str(env_path), ignore_key)

            # Save whitelist patterns
            whitelist_patterns = [rule.pattern for rule in self.whitelist_rules]
            if whitelist_patterns:
                set_key(str(env_path), whitelist_key, ",".join(whitelist_patterns))
            else:
                unset_key(str(env_path), whitelist_key)

            # Update original state
            self._original_ignore_patterns = set(ignore_patterns)
            self._original_whitelist_patterns = set(whitelist_patterns)

            return True
        except Exception as e:
            print(f"Error saving to .env: {e}")
            return False

    def has_unsaved_changes(self) -> bool:
        """Check if current rules differ from saved state."""
        current_ignore = set(rule.pattern for rule in self.ignore_rules)
        current_whitelist = set(rule.pattern for rule in self.whitelist_rules)

        return (
            current_ignore != self._original_ignore_patterns
            or current_whitelist != self._original_whitelist_patterns
        )

    def discard_changes(self):
        """Reload rules from environment, discarding unsaved changes."""
        if self._current_provider:
            self.load_from_env(self._current_provider)


# ════════════════════════════════════════════════════════════════════════════════
# MODEL FETCHER
# ════════════════════════════════════════════════════════════════════════════════

# Global cache for fetched models (persists across provider switches)
_model_cache: Dict[str, List[str]] = {}


class ModelFetcher:
    """
    Handles async model fetching from providers.

    Runs fetching in a background thread to avoid blocking the GUI.
    Includes caching to avoid refetching on every provider switch.
    """

    @staticmethod
    def get_cached_models(provider: str) -> Optional[List[str]]:
        """Get cached models for a provider, if available."""
        return _model_cache.get(provider)

    @staticmethod
    def clear_cache(provider: Optional[str] = None):
        """Clear model cache. If provider specified, only clear that provider."""
        if provider:
            _model_cache.pop(provider, None)
        else:
            _model_cache.clear()

    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of providers that have credentials configured."""
        providers = set()
        load_dotenv(override=True)

        # Scan environment for API keys (handles numbered keys like GEMINI_API_KEY_1)
        for key in os.environ:
            if "_API_KEY" in key and "PROXY_API_KEY" not in key:
                # Extract provider: NVIDIA_NIM_API_KEY_1 -> nvidia_nim
                provider = key.split("_API_KEY")[0].lower()
                providers.add(provider)

        # Check for OAuth providers
        oauth_dir = Path("oauth_creds")
        if oauth_dir.exists():
            for file in oauth_dir.glob("*_oauth_*.json"):
                provider = file.name.split("_oauth_")[0]
                providers.add(provider)

        return sorted(list(providers))

    @staticmethod
    def _find_credential(provider: str) -> Optional[str]:
        """Find a credential for a provider (handles numbered keys)."""
        load_dotenv(override=True)
        provider_upper = provider.upper()

        # Try exact match first (e.g., GEMINI_API_KEY)
        exact_key = f"{provider_upper}_API_KEY"
        if os.getenv(exact_key):
            return os.getenv(exact_key)

        # Look for numbered keys (e.g., GEMINI_API_KEY_1, NVIDIA_NIM_API_KEY_1)
        for key, value in os.environ.items():
            if key.startswith(f"{provider_upper}_API_KEY") and value:
                return value

        # Check for OAuth credentials
        oauth_dir = Path("oauth_creds")
        if oauth_dir.exists():
            oauth_files = list(oauth_dir.glob(f"{provider}_oauth_*.json"))
            if oauth_files:
                return str(oauth_files[0])

        return None

    @staticmethod
    async def _fetch_models_async(provider: str) -> Tuple[List[str], Optional[str]]:
        """
        Async implementation of model fetching.
        Returns: (models_list, error_message_or_none)
        """
        try:
            import httpx
            from rotator_library.providers import PROVIDER_PLUGINS

            # Get credential
            credential = ModelFetcher._find_credential(provider)
            if not credential:
                return [], f"No credentials found for '{provider}'"

            # Get provider class
            provider_class = PROVIDER_PLUGINS.get(provider.lower())
            if not provider_class:
                return [], f"Unknown provider: '{provider}'"

            # Fetch models
            async with httpx.AsyncClient(timeout=30.0) as client:
                instance = provider_class()
                models = await instance.get_models(credential, client)
                return models, None

        except ImportError as e:
            return [], f"Import error: {e}"
        except Exception as e:
            return [], f"Failed to fetch: {str(e)}"

    @staticmethod
    def fetch_models(
        provider: str,
        on_success: Callable[[List[str]], None],
        on_error: Callable[[str], None],
        on_start: Optional[Callable[[], None]] = None,
        force_refresh: bool = False,
    ):
        """
        Fetch models in a background thread.

        Args:
            provider: Provider name (e.g., 'openai', 'gemini')
            on_success: Callback with list of model IDs
            on_error: Callback with error message
            on_start: Optional callback when fetching starts
            force_refresh: If True, bypass cache and fetch fresh
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = ModelFetcher.get_cached_models(provider)
            if cached is not None:
                on_success(cached)
                return

        def run_fetch():
            if on_start:
                on_start()

            try:
                # Run async fetch in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    models, error = loop.run_until_complete(
                        ModelFetcher._fetch_models_async(provider)
                    )
                    # Clean up any pending tasks to avoid warnings
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                finally:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()

                if error:
                    on_error(error)
                else:
                    # Cache the results
                    _model_cache[provider] = models
                    on_success(models)

            except Exception as e:
                on_error(str(e))

        thread = threading.Thread(target=run_fetch, daemon=True)
        thread.start()


# ════════════════════════════════════════════════════════════════════════════════
# HELP WINDOW
# ════════════════════════════════════════════════════════════════════════════════


class HelpWindow(ctk.CTkToplevel):
    """
    Modal help popup with comprehensive filtering documentation.
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.title("Help - Model Filtering")
        self.geometry("700x600")
        self.minsize(600, 500)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Configure appearance
        self.configure(fg_color=BG_PRIMARY)

        # Build content
        self._create_content()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

        # Focus
        self.focus_force()

        # Bind escape to close
        self.bind("<Escape>", lambda e: self.destroy())

    def _create_content(self):
        """Build the help content."""
        # Main scrollable frame
        main_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=BG_PRIMARY,
            scrollbar_fg_color=BG_SECONDARY,
            scrollbar_button_color=BORDER_COLOR,
        )
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title = ctk.CTkLabel(
            main_frame,
            text="📖 Model Filtering Guide",
            font=(FONT_FAMILY, FONT_SIZE_HEADER, "bold"),
            text_color=TEXT_PRIMARY,
        )
        title.pack(anchor="w", pady=(0, 20))

        # Sections
        sections = [
            (
                "🎯 Overview",
                """
Model filtering allows you to control which models are available through your proxy for each provider.

• Use the IGNORE list to block specific models
• Use the WHITELIST to ensure specific models are always available
• Whitelist ALWAYS takes priority over Ignore""",
            ),
            (
                "⚖️ Filtering Priority",
                """
When a model is checked, the following order is used:

1. WHITELIST CHECK
   If the model matches any whitelist pattern → AVAILABLE
   (Whitelist overrides everything else)

2. IGNORE CHECK  
   If the model matches any ignore pattern → BLOCKED

3. DEFAULT
   If no patterns match → AVAILABLE""",
            ),
            (
                "✏️ Pattern Syntax",
                """
Three types of patterns are supported:

EXACT MATCH
  Pattern: gpt-4
  Matches: only "gpt-4", nothing else
  
PREFIX WILDCARD  
  Pattern: gpt-4*
  Matches: "gpt-4", "gpt-4-turbo", "gpt-4-preview", etc.
  
MATCH ALL
  Pattern: *
  Matches: every model for this provider""",
            ),
            (
                "💡 Common Patterns",
                """
BLOCK ALL, ALLOW SPECIFIC:
  Ignore:    *
  Whitelist: gpt-4o, gpt-4o-mini
  Result:    Only gpt-4o and gpt-4o-mini available

BLOCK PREVIEW MODELS:
  Ignore:    *-preview, *-preview*
  Result:    All preview variants blocked

BLOCK SPECIFIC SERIES:
  Ignore:    o1*, dall-e*
  Result:    All o1 and DALL-E models blocked

ALLOW ONLY LATEST:
  Ignore:    *
  Whitelist: *-latest
  Result:    Only models ending in "-latest" available""",
            ),
            (
                "🖱️ Interface Guide",
                """
PROVIDER DROPDOWN
  Select which provider to configure

MODEL LISTS
  • Left list: All fetched models (unfiltered)
  • Right list: Same models with colored status
  • Green = Available (normal)
  • Red/Orange tones = Blocked (ignored)
  • Blue/Teal tones = Whitelisted

SEARCH BOX
  Filter both lists to find specific models quickly

CLICKING MODELS
  • Left-click: Highlight the rule affecting this model
  • Right-click: Context menu with quick actions

CLICKING RULES
  • Highlights all models affected by that rule
  • Shows which models will be blocked/allowed

RULE INPUT
  • Enter patterns separated by commas
  • Press Add or Enter to create rules
  • Preview updates in real-time as you type

DELETE RULES
  • Click the × button on any rule to remove it""",
            ),
            (
                "⌨️ Keyboard Shortcuts",
                """
Ctrl+S     Save changes
Ctrl+R     Refresh models from provider
Ctrl+F     Focus search box
F1         Open this help window
Escape     Clear search / Close dialogs""",
            ),
            (
                "💾 Saving Changes",
                """
Changes are saved to your .env file in this format:

  IGNORE_MODELS_OPENAI=pattern1,pattern2*
  WHITELIST_MODELS_OPENAI=specific-model

Click "Save" to persist changes, or "Discard" to revert.
Closing the window with unsaved changes will prompt you.""",
            ),
        ]

        for title_text, content in sections:
            self._add_section(main_frame, title_text, content)

        # Close button
        close_btn = ctk.CTkButton(
            main_frame,
            text="Got it!",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            height=40,
            width=120,
            command=self.destroy,
        )
        close_btn.pack(pady=20)

    def _add_section(self, parent, title: str, content: str):
        """Add a help section."""
        # Section title
        title_label = ctk.CTkLabel(
            parent,
            text=title,
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=ACCENT_BLUE,
        )
        title_label.pack(anchor="w", pady=(15, 5))

        # Separator
        sep = ctk.CTkFrame(parent, height=1, fg_color=BORDER_COLOR)
        sep.pack(fill="x", pady=(0, 10))

        # Content
        content_label = ctk.CTkLabel(
            parent,
            text=content.strip(),
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=TEXT_SECONDARY,
            justify="left",
            anchor="w",
        )
        content_label.pack(anchor="w", fill="x")


# ════════════════════════════════════════════════════════════════════════════════
# CUSTOM DIALOG
# ════════════════════════════════════════════════════════════════════════════════


class UnsavedChangesDialog(ctk.CTkToplevel):
    """Modal dialog for unsaved changes confirmation."""

    def __init__(self, parent):
        super().__init__(parent)

        self.result: Optional[str] = None  # 'save', 'discard', 'cancel'

        self.title("Unsaved Changes")
        self.geometry("400x180")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Configure appearance
        self.configure(fg_color=BG_PRIMARY)

        # Build content
        self._create_content()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

        # Focus
        self.focus_force()

        # Bind escape to cancel
        self.bind("<Escape>", lambda e: self._on_cancel())

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _create_content(self):
        """Build dialog content."""
        # Icon and message
        msg_frame = ctk.CTkFrame(self, fg_color="transparent")
        msg_frame.pack(fill="x", padx=30, pady=(25, 15))

        icon = ctk.CTkLabel(
            msg_frame, text="⚠️", font=(FONT_FAMILY, 32), text_color=ACCENT_YELLOW
        )
        icon.pack(side="left", padx=(0, 15))

        text_frame = ctk.CTkFrame(msg_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="x", expand=True)

        title = ctk.CTkLabel(
            text_frame,
            text="Unsaved Changes",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            text_frame,
            text="You have unsaved filter changes.\nWhat would you like to do?",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=TEXT_SECONDARY,
            anchor="w",
            justify="left",
        )
        subtitle.pack(anchor="w")

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(10, 25))

        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=100,
            command=self._on_cancel,
        )
        cancel_btn.pack(side="right", padx=(10, 0))

        discard_btn = ctk.CTkButton(
            btn_frame,
            text="Discard",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=ACCENT_RED,
            hover_color="#c0392b",
            width=100,
            command=self._on_discard,
        )
        discard_btn.pack(side="right", padx=(10, 0))

        save_btn = ctk.CTkButton(
            btn_frame,
            text="Save",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=ACCENT_GREEN,
            hover_color="#27ae60",
            width=100,
            command=self._on_save,
        )
        save_btn.pack(side="right")

    def _on_save(self):
        self.result = "save"
        self.destroy()

    def _on_discard(self):
        self.result = "discard"
        self.destroy()

    def _on_cancel(self):
        self.result = "cancel"
        self.destroy()

    def show(self) -> Optional[str]:
        """Show dialog and return result."""
        self.wait_window()
        return self.result


# ════════════════════════════════════════════════════════════════════════════════
# TOOLTIP
# ════════════════════════════════════════════════════════════════════════════════


class ToolTip:
    """Simple tooltip implementation for CustomTkinter widgets."""

    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.after_id = None

        widget.bind("<Enter>", self._schedule_show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<Button>", self._hide)

    def _schedule_show(self, event=None):
        self._hide()
        self.after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tooltip_window:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tooltip_window = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(fg_color=BG_SECONDARY)

        # Add border effect
        frame = ctk.CTkFrame(
            tw,
            fg_color=BG_SECONDARY,
            border_width=1,
            border_color=BORDER_COLOR,
            corner_radius=6,
        )
        frame.pack(fill="both", expand=True)

        label = ctk.CTkLabel(
            frame,
            text=self.text,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
            padx=10,
            pady=5,
        )
        label.pack()

        # Ensure tooltip is on top
        tw.lift()

    def _hide(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def update_text(self, text: str):
        """Update tooltip text."""
        self.text = text


# ════════════════════════════════════════════════════════════════════════════════
# RULE CHIP COMPONENT
# ════════════════════════════════════════════════════════════════════════════════


class RuleChip(ctk.CTkFrame):
    """
    Individual rule display showing pattern, affected count, and delete button.

    The pattern text is colored with the rule's assigned color.
    """

    def __init__(
        self,
        master,
        rule: FilterRule,
        on_delete: Callable[[str], None],
        on_click: Callable[[FilterRule], None],
    ):
        super().__init__(
            master,
            fg_color=BG_TERTIARY,
            corner_radius=6,
            border_width=1,
            border_color=BORDER_COLOR,
        )

        self.rule = rule
        self.on_delete = on_delete
        self.on_click = on_click
        self._is_highlighted = False

        self._create_content()

        # Click binding
        self.bind("<Button-1>", self._handle_click)

    def _create_content(self):
        """Build chip content."""
        # Container for horizontal layout
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="x", padx=8, pady=6)

        # Pattern text (colored)
        self.pattern_label = ctk.CTkLabel(
            content,
            text=self.rule.pattern,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=self.rule.color,
            anchor="w",
        )
        self.pattern_label.pack(side="left", fill="x", expand=True)
        self.pattern_label.bind("<Button-1>", self._handle_click)

        # Affected count
        self.count_label = ctk.CTkLabel(
            content,
            text=f"({self.rule.affected_count})",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_MUTED,
            width=35,
        )
        self.count_label.pack(side="left", padx=(5, 5))
        self.count_label.bind("<Button-1>", self._handle_click)

        # Delete button
        delete_btn = ctk.CTkButton(
            content,
            text="×",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            fg_color="transparent",
            hover_color=ACCENT_RED,
            text_color=TEXT_MUTED,
            width=24,
            height=24,
            corner_radius=4,
            command=self._handle_delete,
        )
        delete_btn.pack(side="right")

        # Tooltip showing affected models
        self._update_tooltip()

    def _handle_click(self, event=None):
        """Handle click on rule chip."""
        self.on_click(self.rule)

    def _handle_delete(self):
        """Handle delete button click."""
        self.on_delete(self.rule.pattern)

    def update_count(self, count: int, affected_models: List[str]):
        """Update the affected count and tooltip."""
        self.rule.affected_count = count
        self.rule.affected_models = affected_models
        self.count_label.configure(text=f"({count})")
        self._update_tooltip()

    def _update_tooltip(self):
        """Update tooltip with affected models."""
        if self.rule.affected_models:
            if len(self.rule.affected_models) <= 5:
                models_text = "\n".join(self.rule.affected_models)
            else:
                models_text = "\n".join(self.rule.affected_models[:5])
                models_text += f"\n... and {len(self.rule.affected_models) - 5} more"
            ToolTip(self, f"Matches:\n{models_text}")
        else:
            ToolTip(self, "No models match this pattern")

    def set_highlighted(self, highlighted: bool):
        """Set highlighted state."""
        self._is_highlighted = highlighted
        if highlighted:
            self.configure(border_color=self.rule.color, border_width=2)
        else:
            self.configure(border_color=BORDER_COLOR, border_width=1)


# ════════════════════════════════════════════════════════════════════════════════
# RULE PANEL COMPONENT
# ════════════════════════════════════════════════════════════════════════════════


class RulePanel(ctk.CTkFrame):
    """
    Panel containing rule chips, input field, and add button.

    Handles adding and removing rules, with callbacks for changes.
    """

    def __init__(
        self,
        master,
        title: str,
        rule_type: str,  # 'ignore' or 'whitelist'
        on_rules_changed: Callable[[], None],
        on_rule_clicked: Callable[[FilterRule], None],
        on_input_changed: Callable[[str, str], None],  # (text, rule_type)
    ):
        super().__init__(master, fg_color=BG_SECONDARY, corner_radius=8)

        self.title = title
        self.rule_type = rule_type
        self.on_rules_changed = on_rules_changed
        self.on_rule_clicked = on_rule_clicked
        self.on_input_changed = on_input_changed
        self.rule_chips: Dict[str, RuleChip] = {}

        self._create_content()

    def _create_content(self):
        """Build panel content."""
        # Title
        title_label = ctk.CTkLabel(
            self,
            text=self.title,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            text_color=TEXT_PRIMARY,
        )
        title_label.pack(anchor="w", padx=12, pady=(12, 8))

        # Rules container (scrollable)
        self.rules_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            height=120,
            scrollbar_fg_color=BG_TERTIARY,
            scrollbar_button_color=BORDER_COLOR,
        )
        self.rules_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Empty state label
        self.empty_label = ctk.CTkLabel(
            self.rules_frame,
            text="No rules configured\nAdd patterns below",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_MUTED,
            justify="center",
        )
        self.empty_label.pack(expand=True, pady=20)

        # Input frame
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(fill="x", padx=8, pady=(0, 8))

        # Pattern input
        self.input_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="pattern1, pattern2*, ...",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_TERTIARY,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_MUTED,
            height=36,
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.input_entry.bind("<Return>", self._on_add_clicked)
        self.input_entry.bind("<KeyRelease>", self._on_input_key)

        # Add button
        add_btn = ctk.CTkButton(
            input_frame,
            text="+ Add",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            width=70,
            height=36,
            command=self._on_add_clicked,
        )
        add_btn.pack(side="right")

    def _on_input_key(self, event=None):
        """Handle key release in input field - for real-time preview."""
        text = self.input_entry.get().strip()
        self.on_input_changed(text, self.rule_type)

    def _on_add_clicked(self, event=None):
        """Handle add button click."""
        text = self.input_entry.get().strip()
        if text:
            # Parse comma-separated patterns
            patterns = [p.strip() for p in text.split(",") if p.strip()]
            if patterns:
                self.input_entry.delete(0, "end")
                for pattern in patterns:
                    self._emit_add_pattern(pattern)

    def _emit_add_pattern(self, pattern: str):
        """Emit request to add a pattern (handled by parent)."""
        # This will be connected to the main window's add method
        if hasattr(self, "_add_pattern_callback"):
            self._add_pattern_callback(pattern)

    def set_add_callback(self, callback: Callable[[str], None]):
        """Set the callback for adding patterns."""
        self._add_pattern_callback = callback

    def add_rule_chip(self, rule: FilterRule):
        """Add a rule chip to the panel."""
        if rule.pattern in self.rule_chips:
            return

        # Hide empty label
        self.empty_label.pack_forget()

        chip = RuleChip(
            self.rules_frame,
            rule,
            on_delete=self._on_rule_delete,
            on_click=self.on_rule_clicked,
        )
        chip.pack(fill="x", pady=2)
        self.rule_chips[rule.pattern] = chip

    def remove_rule_chip(self, pattern: str):
        """Remove a rule chip from the panel."""
        if pattern in self.rule_chips:
            self.rule_chips[pattern].destroy()
            del self.rule_chips[pattern]

        # Show empty label if no rules
        if not self.rule_chips:
            self.empty_label.pack(expand=True, pady=20)

    def _on_rule_delete(self, pattern: str):
        """Handle rule deletion."""
        if hasattr(self, "_delete_pattern_callback"):
            self._delete_pattern_callback(pattern)

    def set_delete_callback(self, callback: Callable[[str], None]):
        """Set the callback for deleting patterns."""
        self._delete_pattern_callback = callback

    def update_rule_counts(self, rules: List[FilterRule], models: List[str]):
        """Update affected counts for all rule chips."""
        for rule in rules:
            if rule.pattern in self.rule_chips:
                self.rule_chips[rule.pattern].update_count(
                    rule.affected_count, rule.affected_models
                )

    def highlight_rule(self, pattern: str):
        """Highlight a specific rule chip."""
        for p, chip in self.rule_chips.items():
            chip.set_highlighted(p == pattern)

    def clear_highlights(self):
        """Clear all rule highlights."""
        for chip in self.rule_chips.values():
            chip.set_highlighted(False)

    def clear_all(self):
        """Remove all rule chips."""
        for chip in list(self.rule_chips.values()):
            chip.destroy()
        self.rule_chips.clear()
        self.empty_label.pack(expand=True, pady=20)

    def get_input_text(self) -> str:
        """Get current input text."""
        return self.input_entry.get().strip()

    def clear_input(self):
        """Clear the input field."""
        self.input_entry.delete(0, "end")


# ════════════════════════════════════════════════════════════════════════════════
# MODEL LIST ITEM
# ════════════════════════════════════════════════════════════════════════════════


class ModelListItem(ctk.CTkFrame):
    """
    Single model row in the list.

    Shows model name with appropriate coloring based on status.
    """

    def __init__(
        self,
        master,
        status: ModelStatus,
        show_status_indicator: bool = False,
        on_click: Optional[Callable[[str], None]] = None,
        on_right_click: Optional[Callable[[str, any], None]] = None,
    ):
        super().__init__(master, fg_color="transparent", height=28)

        self.status = status
        self.on_click = on_click
        self.on_right_click = on_right_click
        self._is_highlighted = False
        self._show_status_indicator = show_status_indicator

        self._create_content()

    def _create_content(self):
        """Build item content."""
        self.pack_propagate(False)

        # Container
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=4, pady=1)

        # Status indicator (for filtered list)
        if self._show_status_indicator:
            indicator_text = {"normal": "●", "ignored": "✗", "whitelisted": "★"}.get(
                self.status.status, "●"
            )

            self.indicator = ctk.CTkLabel(
                self.container,
                text=indicator_text,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=self.status.color,
                width=18,
            )
            self.indicator.pack(side="left")
            self.indicator.bind("<Button-1>", self._handle_click)
            self.indicator.bind("<Button-3>", self._handle_right_click)

        # Model name
        self.name_label = ctk.CTkLabel(
            self.container,
            text=self.status.display_name,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=self.status.color
            if self._show_status_indicator
            else TEXT_PRIMARY,
            anchor="w",
        )
        self.name_label.pack(side="left", fill="x", expand=True)
        self.name_label.bind("<Button-1>", self._handle_click)
        self.name_label.bind("<Button-3>", self._handle_right_click)

        # Bindings for the frame itself
        self.bind("<Button-1>", self._handle_click)
        self.bind("<Button-3>", self._handle_right_click)
        self.container.bind("<Button-1>", self._handle_click)
        self.container.bind("<Button-3>", self._handle_right_click)

        # Hover effect
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.container.bind("<Enter>", self._on_enter)
        self.container.bind("<Leave>", self._on_leave)

    def _handle_click(self, event=None):
        """Handle left click."""
        if self.on_click:
            self.on_click(self.status.model_id)

    def _handle_right_click(self, event):
        """Handle right click."""
        if self.on_right_click:
            self.on_right_click(self.status.model_id, event)

    def _on_enter(self, event=None):
        """Mouse enter - show hover state."""
        if not self._is_highlighted:
            self.container.configure(fg_color=BG_HOVER)

    def _on_leave(self, event=None):
        """Mouse leave - hide hover state."""
        if not self._is_highlighted:
            self.container.configure(fg_color="transparent")

    def update_status(self, status: ModelStatus):
        """Update the model's status and appearance."""
        self.status = status

        if self._show_status_indicator:
            indicator_text = {"normal": "●", "ignored": "✗", "whitelisted": "★"}.get(
                status.status, "●"
            )
            self.indicator.configure(text=indicator_text, text_color=status.color)
            self.name_label.configure(text_color=status.color)
        else:
            self.name_label.configure(text_color=TEXT_PRIMARY)

    def set_highlighted(self, highlighted: bool):
        """Set highlighted state."""
        self._is_highlighted = highlighted
        if highlighted:
            self.container.configure(fg_color=HIGHLIGHT_BG)
        else:
            self.container.configure(fg_color="transparent")

    def matches_search(self, query: str) -> bool:
        """Check if this item matches a search query."""
        if not query:
            return True
        return query.lower() in self.status.model_id.lower()


# ════════════════════════════════════════════════════════════════════════════════
# SYNCHRONIZED MODEL LIST PANEL
# ════════════════════════════════════════════════════════════════════════════════


class SyncModelListPanel(ctk.CTkFrame):
    """
    Two synchronized scrollable model lists side by side.

    Left list: All fetched models (plain display)
    Right list: Same models with colored status indicators

    Both lists scroll together and filter together.
    """

    def __init__(
        self,
        master,
        on_model_click: Callable[[str], None],
        on_model_right_click: Callable[[str, any], None],
    ):
        super().__init__(master, fg_color="transparent")

        self.on_model_click = on_model_click
        self.on_model_right_click = on_model_right_click

        self.models: List[str] = []
        self.statuses: Dict[str, ModelStatus] = {}
        self.left_items: Dict[str, ModelListItem] = {}
        self.right_items: Dict[str, ModelListItem] = {}
        self.search_query: str = ""

        self._create_content()

    def _create_content(self):
        """Build the dual list layout."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Left header
        left_header = ctk.CTkLabel(
            self,
            text="All Fetched Models",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            text_color=TEXT_PRIMARY,
        )
        left_header.grid(row=0, column=0, sticky="w", padx=8, pady=(0, 5))

        self.left_count_label = ctk.CTkLabel(
            self, text="(0)", font=(FONT_FAMILY, FONT_SIZE_SMALL), text_color=TEXT_MUTED
        )
        self.left_count_label.grid(row=0, column=0, sticky="e", padx=8, pady=(0, 5))

        # Right header
        right_header = ctk.CTkLabel(
            self,
            text="Filtered Status",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            text_color=TEXT_PRIMARY,
        )
        right_header.grid(row=0, column=1, sticky="w", padx=8, pady=(0, 5))

        self.right_count_label = ctk.CTkLabel(
            self, text="", font=(FONT_FAMILY, FONT_SIZE_SMALL), text_color=TEXT_MUTED
        )
        self.right_count_label.grid(row=0, column=1, sticky="e", padx=8, pady=(0, 5))

        # Left list container
        left_frame = ctk.CTkFrame(self, fg_color=BG_TERTIARY, corner_radius=6)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        self.left_canvas = ctk.CTkCanvas(
            left_frame,
            bg=self._apply_appearance_mode(BG_TERTIARY),
            highlightthickness=0,
        )
        self.left_scrollbar = ctk.CTkScrollbar(left_frame, command=self._sync_scroll)
        self.left_inner = ctk.CTkFrame(self.left_canvas, fg_color="transparent")

        self.left_canvas.pack(side="left", fill="both", expand=True)
        self.left_scrollbar.pack(side="right", fill="y")

        self.left_canvas_window = self.left_canvas.create_window(
            (0, 0), window=self.left_inner, anchor="nw"
        )

        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        # Right list container
        right_frame = ctk.CTkFrame(self, fg_color=BG_TERTIARY, corner_radius=6)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        self.right_canvas = ctk.CTkCanvas(
            right_frame,
            bg=self._apply_appearance_mode(BG_TERTIARY),
            highlightthickness=0,
        )
        self.right_scrollbar = ctk.CTkScrollbar(right_frame, command=self._sync_scroll)
        self.right_inner = ctk.CTkFrame(self.right_canvas, fg_color="transparent")

        self.right_canvas.pack(side="left", fill="both", expand=True)
        self.right_scrollbar.pack(side="right", fill="y")

        self.right_canvas_window = self.right_canvas.create_window(
            (0, 0), window=self.right_inner, anchor="nw"
        )

        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        # Bind scroll events
        self.left_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.right_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.left_inner.bind("<MouseWheel>", self._on_mousewheel)
        self.right_inner.bind("<MouseWheel>", self._on_mousewheel)

        # Bind resize
        self.left_inner.bind("<Configure>", self._on_inner_configure)
        self.left_canvas.bind("<Configure>", self._on_canvas_configure)

        # Loading state
        self.loading_frame = ctk.CTkFrame(self, fg_color=BG_TERTIARY, corner_radius=6)
        self.loading_label = ctk.CTkLabel(
            self.loading_frame,
            text="Loading...",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=TEXT_MUTED,
        )
        self.loading_label.pack(expand=True)

        # Error state
        self.error_frame = ctk.CTkFrame(self, fg_color=BG_TERTIARY, corner_radius=6)
        self.error_label = ctk.CTkLabel(
            self.error_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=ACCENT_RED,
        )
        self.error_label.pack(expand=True, pady=20)

        self.retry_btn = ctk.CTkButton(
            self.error_frame,
            text="Retry",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            width=100,
        )
        self.retry_btn.pack()

    def _apply_appearance_mode(self, color):
        """Apply appearance mode to color."""
        return color

    def _sync_scroll(self, *args):
        """Synchronized scroll handler."""
        self.left_canvas.yview(*args)
        self.right_canvas.yview(*args)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        delta = -1 * (event.delta // 120)
        self.left_canvas.yview_scroll(delta, "units")
        self.right_canvas.yview_scroll(delta, "units")
        return "break"

    def _on_inner_configure(self, event=None):
        """Update scroll region when inner frame changes."""
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        """Update inner frame width when canvas resizes."""
        width = self.left_canvas.winfo_width()
        self.left_canvas.itemconfig(self.left_canvas_window, width=width)
        self.right_canvas.itemconfig(self.right_canvas_window, width=width)

    def show_loading(self, provider: str):
        """Show loading state."""
        self.loading_label.configure(text=f"Fetching models from {provider}...")
        self.loading_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.error_frame.grid_forget()

    def show_error(self, message: str, on_retry: Callable):
        """Show error state."""
        self.error_label.configure(text=f"❌ {message}")
        self.retry_btn.configure(command=on_retry)
        self.error_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.loading_frame.grid_forget()

    def hide_overlays(self):
        """Hide loading and error overlays."""
        self.loading_frame.grid_forget()
        self.error_frame.grid_forget()

    def set_models(self, models: List[str], statuses: List[ModelStatus]):
        """Set the models to display."""
        self.models = models
        self.statuses = {s.model_id: s for s in statuses}

        self._rebuild_lists()
        self._update_counts()
        self.hide_overlays()

    def _rebuild_lists(self):
        """Rebuild both model lists."""
        # Clear existing items
        for item in self.left_items.values():
            item.destroy()
        for item in self.right_items.values():
            item.destroy()
        self.left_items.clear()
        self.right_items.clear()

        # Create new items
        for model_id in self.models:
            status = self.statuses.get(
                model_id,
                ModelStatus(model_id=model_id, status="normal", color=NORMAL_COLOR),
            )

            # Left item (plain)
            left_item = ModelListItem(
                self.left_inner,
                status,
                show_status_indicator=False,
                on_click=self.on_model_click,
                on_right_click=self.on_model_right_click,
            )
            left_item.pack(fill="x")
            self.left_items[model_id] = left_item

            # Right item (with status colors)
            right_item = ModelListItem(
                self.right_inner,
                status,
                show_status_indicator=True,
                on_click=self.on_model_click,
                on_right_click=self.on_model_right_click,
            )
            right_item.pack(fill="x")
            self.right_items[model_id] = right_item

        # Apply current search filter
        self._apply_search_filter()

    def update_statuses(self, statuses: List[ModelStatus]):
        """Update status display for all models."""
        self.statuses = {s.model_id: s for s in statuses}

        for model_id, status in self.statuses.items():
            if model_id in self.right_items:
                self.right_items[model_id].update_status(status)

        self._update_counts()

    def _update_counts(self):
        """Update the count labels."""
        visible_count = sum(
            1
            for item in self.left_items.values()
            if item.winfo_viewable() or item.matches_search(self.search_query)
        )
        total = len(self.models)

        # Count available (not ignored)
        available = sum(1 for s in self.statuses.values() if s.status != "ignored")

        self.left_count_label.configure(text=f"({total})")
        self.right_count_label.configure(text=f"{available} available")

    def filter_by_search(self, query: str):
        """Filter models by search query."""
        self.search_query = query
        self._apply_search_filter()

    def _apply_search_filter(self):
        """Apply the current search filter to items."""
        for model_id in self.models:
            left_item = self.left_items.get(model_id)
            right_item = self.right_items.get(model_id)

            if left_item and right_item:
                matches = left_item.matches_search(self.search_query)
                if matches:
                    left_item.pack(fill="x")
                    right_item.pack(fill="x")
                else:
                    left_item.pack_forget()
                    right_item.pack_forget()

    def highlight_models_by_rule(self, rule: FilterRule):
        """Highlight all models affected by a rule."""
        self.clear_highlights()

        first_match = None
        for model_id in rule.affected_models:
            if model_id in self.left_items:
                self.left_items[model_id].set_highlighted(True)
                if first_match is None:
                    first_match = model_id
            if model_id in self.right_items:
                self.right_items[model_id].set_highlighted(True)

        # Scroll to first match
        if first_match:
            self._scroll_to_model(first_match)

    def highlight_model(self, model_id: str):
        """Highlight a specific model."""
        self.clear_highlights()

        if model_id in self.left_items:
            self.left_items[model_id].set_highlighted(True)
        if model_id in self.right_items:
            self.right_items[model_id].set_highlighted(True)

    def clear_highlights(self):
        """Clear all model highlights."""
        for item in self.left_items.values():
            item.set_highlighted(False)
        for item in self.right_items.values():
            item.set_highlighted(False)

    def _scroll_to_model(self, model_id: str):
        """Scroll to make a model visible."""
        if model_id not in self.left_items:
            return

        item = self.left_items[model_id]

        # Calculate position
        self.update_idletasks()
        item_y = item.winfo_y()
        inner_height = self.left_inner.winfo_height()
        canvas_height = self.left_canvas.winfo_height()

        if inner_height > canvas_height:
            # Calculate scroll fraction
            scroll_pos = item_y / inner_height
            scroll_pos = max(0, min(scroll_pos, 1))

            self.left_canvas.yview_moveto(scroll_pos)
            self.right_canvas.yview_moveto(scroll_pos)

    def scroll_to_affected(self, affected_models: List[str]):
        """Scroll to first affected model in the list."""
        for model_id in self.models:
            if model_id in affected_models:
                self._scroll_to_model(model_id)
                break

    def get_model_at_position(self, model_id: str) -> Optional[ModelStatus]:
        """Get the status of a model."""
        return self.statuses.get(model_id)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION WINDOW
# ════════════════════════════════════════════════════════════════════════════════


class ModelFilterGUI(ctk.CTk):
    """
    Main application window for model filter configuration.

    Provides a visual interface for managing IGNORE_MODELS_* and WHITELIST_MODELS_*
    environment variables per provider.
    """

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title(WINDOW_TITLE)
        self.geometry(WINDOW_DEFAULT_SIZE)
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.configure(fg_color=BG_PRIMARY)

        # State
        self.current_provider: Optional[str] = None
        self.models: List[str] = []
        self.filter_engine = FilterEngine()
        self.available_providers: List[str] = []
        self._preview_pattern: str = ""
        self._preview_rule_type: str = ""
        self._update_scheduled: bool = False
        self._pending_providers_to_fetch: List[str] = []
        self._fetch_in_progress: bool = False
        self._preview_after_id: Optional[str] = None

        # Build UI
        self._create_header()
        self._create_search_bar()
        self._create_model_lists()
        self._create_rule_panels()
        self._create_status_bar()
        self._create_action_buttons()

        # Context menu
        self._create_context_menu()

        # Load providers and start fetching all models
        self._load_providers()

        # Bind keyboard shortcuts
        self._bind_shortcuts()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Focus and raise window after it's fully loaded
        self.after(100, self._activate_window)

    def _activate_window(self):
        """Activate and focus the window."""
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False))

    def _create_header(self):
        """Create the header with provider selector and buttons."""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(15, 10))

        # Title
        title = ctk.CTkLabel(
            header,
            text="🎯 Model Filter Configuration",
            font=(FONT_FAMILY, FONT_SIZE_HEADER, "bold"),
            text_color=TEXT_PRIMARY,
        )
        title.pack(side="left")

        # Help button
        help_btn = ctk.CTkButton(
            header,
            text="?",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=36,
            height=36,
            corner_radius=18,
            command=self._show_help,
        )
        help_btn.pack(side="right", padx=(10, 0))
        ToolTip(help_btn, "Help (F1)")

        # Refresh button
        refresh_btn = ctk.CTkButton(
            header,
            text="🔄 Refresh",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=100,
            height=36,
            command=self._refresh_models,
        )
        refresh_btn.pack(side="right", padx=(10, 0))
        ToolTip(refresh_btn, "Refresh models (Ctrl+R)")

        # Provider selector
        provider_frame = ctk.CTkFrame(header, fg_color="transparent")
        provider_frame.pack(side="right")

        provider_label = ctk.CTkLabel(
            provider_frame,
            text="Provider:",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=TEXT_SECONDARY,
        )
        provider_label.pack(side="left", padx=(0, 8))

        self.provider_dropdown = ctk.CTkComboBox(
            provider_frame,
            values=["Loading..."],
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            dropdown_font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            border_color=BORDER_COLOR,
            button_color=BORDER_COLOR,
            button_hover_color=BG_HOVER,
            dropdown_fg_color=BG_SECONDARY,
            dropdown_hover_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            width=180,
            height=36,
            state="readonly",
            command=self._on_provider_changed,
        )
        self.provider_dropdown.pack(side="left")

    def _create_search_bar(self):
        """Create the search bar."""
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=20, pady=(0, 10))

        search_icon = ctk.CTkLabel(
            search_frame,
            text="🔍",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=TEXT_MUTED,
        )
        search_icon.pack(side="left", padx=(0, 8))

        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Search models...",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_MUTED,
            height=36,
        )
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_entry.bind("<KeyRelease>", self._on_search_changed)

        # Clear button
        clear_btn = ctk.CTkButton(
            search_frame,
            text="×",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            fg_color="transparent",
            hover_color=BG_HOVER,
            text_color=TEXT_MUTED,
            width=36,
            height=36,
            command=self._clear_search,
        )
        clear_btn.pack(side="left")

    def _create_model_lists(self):
        """Create the synchronized model list panel."""
        self.model_list_panel = SyncModelListPanel(
            self,
            on_model_click=self._on_model_clicked,
            on_model_right_click=self._on_model_right_clicked,
        )
        self.model_list_panel.pack(fill="both", expand=True, padx=20, pady=(0, 10))

    def _create_rule_panels(self):
        """Create the ignore and whitelist rule panels."""
        rules_frame = ctk.CTkFrame(self, fg_color="transparent")
        rules_frame.pack(fill="x", padx=20, pady=(0, 10))
        rules_frame.grid_columnconfigure(0, weight=1)
        rules_frame.grid_columnconfigure(1, weight=1)

        # Ignore panel
        self.ignore_panel = RulePanel(
            rules_frame,
            title="🚫 Ignore Rules",
            rule_type="ignore",
            on_rules_changed=self._on_rules_changed,
            on_rule_clicked=self._on_rule_clicked,
            on_input_changed=self._on_rule_input_changed,
        )
        self.ignore_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.ignore_panel.set_add_callback(self._add_ignore_pattern)
        self.ignore_panel.set_delete_callback(self._remove_ignore_pattern)

        # Whitelist panel
        self.whitelist_panel = RulePanel(
            rules_frame,
            title="✓ Whitelist Rules",
            rule_type="whitelist",
            on_rules_changed=self._on_rules_changed,
            on_rule_clicked=self._on_rule_clicked,
            on_input_changed=self._on_rule_input_changed,
        )
        self.whitelist_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.whitelist_panel.set_add_callback(self._add_whitelist_pattern)
        self.whitelist_panel.set_delete_callback(self._remove_whitelist_pattern)

    def _create_status_bar(self):
        """Create the status bar showing available count and action buttons."""
        # Combined status bar and action buttons in one row
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.pack(fill="x", padx=20, pady=(5, 15))

        # Status label (left side)
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Select a provider to begin",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=TEXT_SECONDARY,
        )
        self.status_label.pack(side="left")

        # Unsaved indicator (after status)
        self.unsaved_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=ACCENT_YELLOW,
        )
        self.unsaved_label.pack(side="left", padx=(15, 0))

        # Buttons (right side)
        # Discard button
        discard_btn = ctk.CTkButton(
            self.status_frame,
            text="↩️ Discard",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=110,
            height=36,
            command=self._discard_changes,
        )
        discard_btn.pack(side="right", padx=(10, 0))

        # Save button
        save_btn = ctk.CTkButton(
            self.status_frame,
            text="💾 Save",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=ACCENT_GREEN,
            hover_color="#27ae60",
            width=110,
            height=36,
            command=self._save_changes,
        )
        save_btn.pack(side="right")
        ToolTip(save_btn, "Save changes (Ctrl+S)")

    def _create_action_buttons(self):
        """Action buttons are now part of status bar - this is a no-op for compatibility."""
        pass

    def _create_context_menu(self):
        """Create the right-click context menu."""
        self.context_menu = Menu(self, tearoff=0, bg=BG_SECONDARY, fg=TEXT_PRIMARY)
        self.context_menu.add_command(
            label="➕ Add to Ignore List",
            command=lambda: self._add_model_to_list("ignore"),
        )
        self.context_menu.add_command(
            label="➕ Add to Whitelist",
            command=lambda: self._add_model_to_list("whitelist"),
        )
        self.context_menu.add_separator()
        self.context_menu.add_command(
            label="🔍 View Affecting Rule", command=self._view_affecting_rule
        )
        self.context_menu.add_command(
            label="📋 Copy Model Name", command=self._copy_model_name
        )

        self._context_model_id: Optional[str] = None

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.bind("<Control-s>", lambda e: self._save_changes())
        self.bind("<Control-r>", lambda e: self._refresh_models())
        self.bind("<Control-f>", lambda e: self.search_entry.focus_set())
        self.bind("<F1>", lambda e: self._show_help())
        self.bind("<Escape>", self._on_escape)

    def _on_escape(self, event=None):
        """Handle escape key."""
        # Clear search if has content
        if self.search_entry.get():
            self._clear_search()
        else:
            # Clear highlights
            self.model_list_panel.clear_highlights()
            self.ignore_panel.clear_highlights()
            self.whitelist_panel.clear_highlights()

    # ─────────────────────────────────────────────────────────────────────────────
    # Provider Management
    # ─────────────────────────────────────────────────────────────────────────────

    def _load_providers(self):
        """Load available providers and start fetching all models in background."""
        self.available_providers = ModelFetcher.get_available_providers()

        if self.available_providers:
            self.provider_dropdown.configure(values=self.available_providers)
            self.provider_dropdown.set(self.available_providers[0])

            # Start fetching all provider models in background
            self._pending_providers_to_fetch = list(self.available_providers)
            self.status_label.configure(text="Loading models for all providers...")
            self._fetch_next_provider()

            # Load the first provider immediately
            self._on_provider_changed(self.available_providers[0])
        else:
            self.provider_dropdown.configure(values=["No providers found"])
            self.provider_dropdown.set("No providers found")
            self.status_label.configure(
                text="No providers with credentials found. Add API keys to .env first."
            )

    def _fetch_next_provider(self):
        """Fetch models for the next provider in the queue (background prefetch)."""
        if not self._pending_providers_to_fetch or self._fetch_in_progress:
            return

        self._fetch_in_progress = True
        provider = self._pending_providers_to_fetch.pop(0)

        # Skip if already cached
        if ModelFetcher.get_cached_models(provider) is not None:
            self._fetch_in_progress = False
            self.after(10, self._fetch_next_provider)
            return

        def on_done(models):
            self._fetch_in_progress = False
            # If this is the current provider, update display
            if provider == self.current_provider:
                self._on_models_loaded(models)
            # Continue with next provider
            self.after(100, self._fetch_next_provider)

        def on_error(error):
            self._fetch_in_progress = False
            # Continue with next provider even on error
            self.after(100, self._fetch_next_provider)

        ModelFetcher.fetch_models(
            provider,
            on_success=on_done,
            on_error=on_error,
            force_refresh=False,
        )

    def _on_provider_changed(self, provider: str):
        """Handle provider selection change."""
        if provider == self.current_provider:
            return

        # Check for unsaved changes
        if self.current_provider and self.filter_engine.has_unsaved_changes():
            result = self._show_unsaved_dialog()
            if result == "cancel":
                # Reset dropdown
                self.provider_dropdown.set(self.current_provider)
                return
            elif result == "save":
                self._save_changes()

        self.current_provider = provider
        self.models = []

        # Clear UI
        self.ignore_panel.clear_all()
        self.whitelist_panel.clear_all()
        self.model_list_panel.clear_highlights()

        # Load rules for this provider
        self.filter_engine.load_from_env(provider)
        self._populate_rule_panels()

        # Try to load from cache first
        cached_models = ModelFetcher.get_cached_models(provider)
        if cached_models is not None:
            self._on_models_loaded(cached_models)
        else:
            # Fetch models (will cache automatically)
            self._fetch_models()

    def _fetch_models(self, force_refresh: bool = False):
        """Fetch models for current provider."""
        if not self.current_provider:
            return

        self.model_list_panel.show_loading(self.current_provider)
        self.status_label.configure(
            text=f"Fetching models from {self.current_provider}..."
        )

        ModelFetcher.fetch_models(
            self.current_provider,
            on_success=self._on_models_loaded,
            on_error=self._on_models_error,
            on_start=None,
            force_refresh=force_refresh,
        )

    def _on_models_loaded(self, models: List[str]):
        """Handle successful model fetch."""
        self.models = sorted(models)

        # Update filter engine counts
        self.filter_engine.update_affected_counts(self.models)

        # Update UI (must be on main thread)
        self.after(0, self._update_model_display)

    def _on_models_error(self, error: str):
        """Handle model fetch error."""
        self.after(
            0,
            lambda: self.model_list_panel.show_error(
                error, on_retry=self._refresh_models
            ),
        )
        self.after(
            0,
            lambda: self.status_label.configure(
                text=f"Failed to fetch models: {error}"
            ),
        )

    def _update_model_display(self):
        """Update the model list display."""
        statuses = self.filter_engine.get_all_statuses(self.models)
        self.model_list_panel.set_models(self.models, statuses)

        # Update rule counts
        self.ignore_panel.update_rule_counts(
            self.filter_engine.ignore_rules, self.models
        )
        self.whitelist_panel.update_rule_counts(
            self.filter_engine.whitelist_rules, self.models
        )

        # Update status
        self._update_status()

    def _refresh_models(self):
        """Refresh models from provider (force bypass cache)."""
        if self.current_provider:
            ModelFetcher.clear_cache(self.current_provider)
            self._fetch_models(force_refresh=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # Rule Management
    # ─────────────────────────────────────────────────────────────────────────────

    def _populate_rule_panels(self):
        """Populate rule panels from filter engine."""
        for rule in self.filter_engine.ignore_rules:
            self.ignore_panel.add_rule_chip(rule)

        for rule in self.filter_engine.whitelist_rules:
            self.whitelist_panel.add_rule_chip(rule)

    def _add_ignore_pattern(self, pattern: str):
        """Add an ignore pattern."""
        rule = self.filter_engine.add_ignore_rule(pattern)
        if rule:
            self.ignore_panel.add_rule_chip(rule)
            self._on_rules_changed()

    def _add_whitelist_pattern(self, pattern: str):
        """Add a whitelist pattern."""
        rule = self.filter_engine.add_whitelist_rule(pattern)
        if rule:
            self.whitelist_panel.add_rule_chip(rule)
            self._on_rules_changed()

    def _remove_ignore_pattern(self, pattern: str):
        """Remove an ignore pattern."""
        self.filter_engine.remove_ignore_rule(pattern)
        self.ignore_panel.remove_rule_chip(pattern)
        self._on_rules_changed()

    def _remove_whitelist_pattern(self, pattern: str):
        """Remove a whitelist pattern."""
        self.filter_engine.remove_whitelist_rule(pattern)
        self.whitelist_panel.remove_rule_chip(pattern)
        self._on_rules_changed()

    def _on_rules_changed(self):
        """Handle any rule change - uses debouncing to reduce lag."""
        if self._update_scheduled:
            return

        self._update_scheduled = True
        self.after(50, self._perform_rules_update)

    def _perform_rules_update(self):
        """Actually perform the rules update (called via debounce)."""
        self._update_scheduled = False

        # Update affected counts
        self.filter_engine.update_affected_counts(self.models)

        # Update model statuses
        statuses = self.filter_engine.get_all_statuses(self.models)
        self.model_list_panel.update_statuses(statuses)

        # Update rule counts
        self.ignore_panel.update_rule_counts(
            self.filter_engine.ignore_rules, self.models
        )
        self.whitelist_panel.update_rule_counts(
            self.filter_engine.whitelist_rules, self.models
        )

        # Update status
        self._update_status()

    def _on_rule_input_changed(self, text: str, rule_type: str):
        """Handle real-time input change for preview - debounced."""
        self._preview_pattern = text
        self._preview_rule_type = rule_type

        # Cancel any pending preview update
        if hasattr(self, "_preview_after_id") and self._preview_after_id:
            self.after_cancel(self._preview_after_id)

        # Debounce preview updates
        self._preview_after_id = self.after(
            100, lambda: self._perform_preview_update(text, rule_type)
        )

    def _perform_preview_update(self, text: str, rule_type: str):
        """Actually perform the preview update."""
        if not text or not self.models:
            self.model_list_panel.clear_highlights()
            return

        # Parse comma-separated patterns
        patterns = [p.strip() for p in text.split(",") if p.strip()]

        # Find all affected models
        affected = []
        for pattern in patterns:
            affected.extend(
                self.filter_engine.preview_pattern(pattern, rule_type, self.models)
            )

        # Highlight affected models
        if affected:
            # Create temporary statuses for preview
            for model_id in affected:
                if model_id in self.model_list_panel.right_items:
                    self.model_list_panel.right_items[model_id].set_highlighted(True)

            # Scroll to first affected
            self.model_list_panel.scroll_to_affected(affected)
        else:
            self.model_list_panel.clear_highlights()

    def _on_rule_clicked(self, rule: FilterRule):
        """Handle click on a rule chip."""
        # Highlight affected models
        self.model_list_panel.highlight_models_by_rule(rule)

        # Highlight the clicked rule
        if rule.rule_type == "ignore":
            self.ignore_panel.highlight_rule(rule.pattern)
            self.whitelist_panel.clear_highlights()
        else:
            self.whitelist_panel.highlight_rule(rule.pattern)
            self.ignore_panel.clear_highlights()

    # ─────────────────────────────────────────────────────────────────────────────
    # Model Interactions
    # ─────────────────────────────────────────────────────────────────────────────

    def _on_model_clicked(self, model_id: str):
        """Handle left-click on a model."""
        status = self.model_list_panel.get_model_at_position(model_id)

        if status and status.affecting_rule:
            # Highlight the affecting rule
            rule = status.affecting_rule
            if rule.rule_type == "ignore":
                self.ignore_panel.highlight_rule(rule.pattern)
                self.whitelist_panel.clear_highlights()
            else:
                self.whitelist_panel.highlight_rule(rule.pattern)
                self.ignore_panel.clear_highlights()

            # Also highlight the model
            self.model_list_panel.highlight_model(model_id)
        else:
            # No affecting rule - just show highlight briefly
            self.model_list_panel.highlight_model(model_id)
            self.ignore_panel.clear_highlights()
            self.whitelist_panel.clear_highlights()

    def _on_model_right_clicked(self, model_id: str, event):
        """Handle right-click on a model."""
        self._context_model_id = model_id

        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _add_model_to_list(self, list_type: str):
        """Add the context menu model to ignore or whitelist."""
        if not self._context_model_id:
            return

        # Extract model name without provider prefix
        if "/" in self._context_model_id:
            pattern = self._context_model_id.split("/", 1)[1]
        else:
            pattern = self._context_model_id

        if list_type == "ignore":
            self._add_ignore_pattern(pattern)
        else:
            self._add_whitelist_pattern(pattern)

    def _view_affecting_rule(self):
        """View the rule affecting the context menu model."""
        if not self._context_model_id:
            return

        self._on_model_clicked(self._context_model_id)

    def _copy_model_name(self):
        """Copy the context menu model name to clipboard."""
        if self._context_model_id:
            self.clipboard_clear()
            self.clipboard_append(self._context_model_id)

    # ─────────────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────────────

    def _on_search_changed(self, event=None):
        """Handle search input change."""
        query = self.search_entry.get()
        self.model_list_panel.filter_by_search(query)

    def _clear_search(self):
        """Clear search field."""
        self.search_entry.delete(0, "end")
        self.model_list_panel.filter_by_search("")

    # ─────────────────────────────────────────────────────────────────────────────
    # Status & UI Updates
    # ─────────────────────────────────────────────────────────────────────────────

    def _update_status(self):
        """Update the status bar."""
        if not self.models:
            self.status_label.configure(text="No models loaded")
            return

        available, total = self.filter_engine.get_available_count(self.models)
        ignored = total - available

        if ignored > 0:
            text = f"✅ {available} of {total} models available ({ignored} ignored)"
        else:
            text = f"✅ All {total} models available"

        self.status_label.configure(text=text)

        # Update unsaved indicator
        if self.filter_engine.has_unsaved_changes():
            self.unsaved_label.configure(text="● Unsaved changes")
        else:
            self.unsaved_label.configure(text="")

    # ─────────────────────────────────────────────────────────────────────────────
    # Dialogs
    # ─────────────────────────────────────────────────────────────────────────────

    def _show_help(self):
        """Show help window."""
        HelpWindow(self)

    def _show_unsaved_dialog(self) -> str:
        """Show unsaved changes dialog. Returns 'save', 'discard', or 'cancel'."""
        dialog = UnsavedChangesDialog(self)
        return dialog.show() or "cancel"

    # ─────────────────────────────────────────────────────────────────────────────
    # Save / Discard
    # ─────────────────────────────────────────────────────────────────────────────

    def _save_changes(self):
        """Save current rules to .env file."""
        if not self.current_provider:
            return

        if self.filter_engine.save_to_env(self.current_provider):
            self.status_label.configure(text="✅ Changes saved successfully!")
            self.unsaved_label.configure(text="")

            # Reset to show normal status after a moment
            self.after(2000, self._update_status)
        else:
            self.status_label.configure(text="❌ Failed to save changes")

    def _discard_changes(self):
        """Discard unsaved changes."""
        if not self.current_provider:
            return

        if not self.filter_engine.has_unsaved_changes():
            return

        # Reload from env
        self.filter_engine.discard_changes()

        # Rebuild rule panels
        self.ignore_panel.clear_all()
        self.whitelist_panel.clear_all()
        self._populate_rule_panels()

        # Update display
        self._on_rules_changed()

        self.status_label.configure(text="Changes discarded")
        self.after(2000, self._update_status)

    # ─────────────────────────────────────────────────────────────────────────────
    # Window Close
    # ─────────────────────────────────────────────────────────────────────────────

    def _on_close(self):
        """Handle window close."""
        if self.filter_engine.has_unsaved_changes():
            result = self._show_unsaved_dialog()
            if result == "cancel":
                return
            elif result == "save":
                self._save_changes()

        self.destroy()


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════


def run_model_filter_gui():
    """
    Launch the Model Filter GUI application.

    This function configures CustomTkinter for dark mode and starts the
    main application loop. It blocks until the window is closed.
    """
    # Force dark mode
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # Create and run app
    app = ModelFilterGUI()
    app.mainloop()


if __name__ == "__main__":
    run_model_filter_gui()

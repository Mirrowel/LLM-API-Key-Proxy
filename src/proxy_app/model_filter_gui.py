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
WINDOW_MIN_WIDTH = 600
WINDOW_MIN_HEIGHT = 400

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
    Uses caching for performance with large model lists.
    """

    def __init__(self):
        self.ignore_rules: List[FilterRule] = []
        self.whitelist_rules: List[FilterRule] = []
        self._ignore_color_index = 0
        self._whitelist_color_index = 0
        self._original_ignore_patterns: Set[str] = set()
        self._original_whitelist_patterns: Set[str] = set()
        self._current_provider: Optional[str] = None

        # Caching for performance
        self._status_cache: Dict[str, ModelStatus] = {}
        self._available_count_cache: Optional[Tuple[int, int]] = None
        self._cache_valid: bool = False

    def _invalidate_cache(self):
        """Mark cache as stale (call when rules change)."""
        self._status_cache.clear()
        self._available_count_cache = None
        self._cache_valid = False

    def reset(self):
        """Clear all rules and reset state."""
        self.ignore_rules.clear()
        self.whitelist_rules.clear()
        self._ignore_color_index = 0
        self._whitelist_color_index = 0
        self._original_ignore_patterns.clear()
        self._original_whitelist_patterns.clear()
        self._invalidate_cache()

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
        self._invalidate_cache()
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
        self._invalidate_cache()
        return rule

    def remove_ignore_rule(self, pattern: str) -> bool:
        """Remove an ignore rule by pattern. Returns True if removed."""
        for i, rule in enumerate(self.ignore_rules):
            if rule.pattern == pattern:
                self.ignore_rules.pop(i)
                self._invalidate_cache()
                return True
        return False

    def remove_whitelist_rule(self, pattern: str) -> bool:
        """Remove a whitelist rule by pattern. Returns True if removed."""
        for i, rule in enumerate(self.whitelist_rules):
            if rule.pattern == pattern:
                self.whitelist_rules.pop(i)
                self._invalidate_cache()
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

    def pattern_is_covered_by(self, new_pattern: str, existing_pattern: str) -> bool:
        """
        Check if new_pattern is already covered by existing_pattern.

        A pattern A is covered by pattern B if every model that would match A
        would also match B.

        Examples:
        - "gpt-4" is covered by "gpt-4*" (prefix covers exact)
        - "gpt-4-turbo" is covered by "gpt-4*" (prefix covers longer)
        - "gpt-4*" is covered by "gpt-*" (broader prefix covers narrower)
        - Anything is covered by "*" (match-all covers everything)
        - "gpt-4" is covered by "gpt-4" (exact duplicate)
        """
        # Exact duplicate
        if new_pattern == existing_pattern:
            return True

        # Existing is wildcard-all - covers everything
        if existing_pattern == "*":
            return True

        # If existing is a prefix wildcard
        if existing_pattern.endswith("*"):
            existing_prefix = existing_pattern[:-1]

            # New is exact match - check if it starts with existing prefix
            if not new_pattern.endswith("*"):
                return new_pattern.startswith(existing_prefix)

            # New is also a prefix wildcard - check if new prefix starts with existing
            new_prefix = new_pattern[:-1]
            return new_prefix.startswith(existing_prefix)

        # Existing is exact match - only covers exact duplicate (already handled)
        return False

    def is_pattern_covered(self, new_pattern: str, rule_type: str) -> bool:
        """
        Check if a new pattern is already covered by any existing rule of the same type.
        """
        rules = self.ignore_rules if rule_type == "ignore" else self.whitelist_rules
        for rule in rules:
            if self.pattern_is_covered_by(new_pattern, rule.pattern):
                return True
        return False

    def get_covered_patterns(self, new_pattern: str, rule_type: str) -> List[str]:
        """
        Get list of existing patterns that would be covered (made redundant)
        by adding new_pattern.

        Used for smart merge: when adding a broader pattern, remove the
        narrower patterns it covers.
        """
        rules = self.ignore_rules if rule_type == "ignore" else self.whitelist_rules
        covered = []
        for rule in rules:
            if self.pattern_is_covered_by(rule.pattern, new_pattern):
                # The existing rule would be covered by the new pattern
                covered.append(rule.pattern)
        return covered

    def _compute_status(self, model_id: str) -> ModelStatus:
        """
        Compute the status of a model based on current rules (no caching).

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

    def get_model_status(self, model_id: str) -> ModelStatus:
        """Get status for a model (uses cache if available)."""
        if model_id in self._status_cache:
            return self._status_cache[model_id]
        return self._compute_status(model_id)

    def _rebuild_cache(self, models: List[str]):
        """Rebuild the entire status cache in one efficient pass."""
        self._status_cache.clear()

        # Reset rule counts
        for rule in self.ignore_rules + self.whitelist_rules:
            rule.affected_count = 0
            rule.affected_models = []

        available = 0
        for model_id in models:
            status = self._compute_status(model_id)
            self._status_cache[model_id] = status

            if status.affecting_rule:
                status.affecting_rule.affected_count += 1
                status.affecting_rule.affected_models.append(model_id)

            if status.status != "ignored":
                available += 1

        self._available_count_cache = (available, len(models))
        self._cache_valid = True

    def get_all_statuses(self, models: List[str]) -> List[ModelStatus]:
        """Get status for all models (rebuilds cache if invalid)."""
        if not self._cache_valid:
            self._rebuild_cache(models)
        return [self._status_cache.get(m, self._compute_status(m)) for m in models]

    def update_affected_counts(self, models: List[str]):
        """Update the affected_count and affected_models for all rules."""
        # This now just ensures cache is valid - counts are updated in _rebuild_cache
        if not self._cache_valid:
            self._rebuild_cache(models)

    def get_available_count(self, models: List[str]) -> Tuple[int, int]:
        """Returns (available_count, total_count) from cache."""
        if not self._cache_valid:
            self._rebuild_cache(models)
        return self._available_count_cache or (0, 0)

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
    Uses CTkTextbox for proper scrolling with dark theme styling.
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
        """Build the help content using CTkTextbox for proper scrolling."""
        # Main container
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=(20, 10))

        # Use CTkTextbox - CustomTkinter's styled text widget with built-in scrolling
        self.text_box = ctk.CTkTextbox(
            main_frame,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            text_color=TEXT_SECONDARY,
            corner_radius=8,
            wrap="word",
            activate_scrollbars=True,
        )
        self.text_box.pack(fill="both", expand=True)

        # Configure text tags for formatting
        # Access the underlying tk.Text widget for tag configuration
        text_widget = self.text_box._textbox

        text_widget.tag_configure(
            "title",
            font=(FONT_FAMILY, FONT_SIZE_HEADER, "bold"),
            foreground=TEXT_PRIMARY,
            spacing1=5,
            spacing3=15,
        )
        text_widget.tag_configure(
            "section_title",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            foreground=ACCENT_BLUE,
            spacing1=20,
            spacing3=8,
        )
        text_widget.tag_configure(
            "separator",
            font=(FONT_FAMILY, 6),
            foreground=BORDER_COLOR,
            spacing3=5,
        )
        text_widget.tag_configure(
            "content",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            foreground=TEXT_SECONDARY,
            spacing1=2,
            spacing3=5,
            lmargin1=5,
            lmargin2=5,
        )

        # Insert content
        self._insert_help_content()

        # Make read-only by disabling
        self.text_box.configure(state="disabled")

        # Bind mouse wheel for faster scrolling on the internal canvas
        self.text_box.bind("<MouseWheel>", self._on_mousewheel)
        # Also bind on the textbox's internal widget
        self.text_box._textbox.bind("<MouseWheel>", self._on_mousewheel)

        # Close button at bottom
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 15))

        close_btn = ctk.CTkButton(
            btn_frame,
            text="Got it!",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            height=40,
            width=120,
            command=self.destroy,
        )
        close_btn.pack()

    def _on_mousewheel(self, event):
        """Handle mouse wheel with faster scrolling."""
        # CTkTextbox uses _textbox internally
        self.text_box._textbox.yview_scroll(-1 * (event.delta // 40), "units")
        return "break"

    def _insert_help_content(self):
        """Insert all help text with formatting."""
        # Access internal text widget for inserting with tags
        text_widget = self.text_box._textbox

        # Title
        text_widget.insert("end", "📖 Model Filtering Guide\n", "title")

        # Sections with emojis
        sections = [
            (
                "🎯 Overview",
                """Model filtering allows you to control which models are available through your proxy for each provider.

• Use the IGNORE list to block specific models
• Use the WHITELIST to ensure specific models are always available
• Whitelist ALWAYS takes priority over Ignore""",
            ),
            (
                "⚖️ Filtering Priority",
                """When a model is checked, the following order is used:

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
                """Three types of patterns are supported:

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
                """BLOCK ALL, ALLOW SPECIFIC:
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
                """PROVIDER DROPDOWN
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

RULE INPUT (Merge Mode)
  • Enter patterns separated by commas
  • Only adds patterns not covered by existing rules
  • Press Add or Enter to create rules

IMPORT BUTTON (Replace Mode)
  • Replaces ALL existing rules with imported ones
  • Paste comma-separated patterns

DELETE RULES
  • Click the × button on any rule to remove it""",
            ),
            (
                "⌨️ Keyboard Shortcuts",
                """Ctrl+S     Save changes
Ctrl+R     Refresh models from provider
Ctrl+F     Focus search box
F1         Open this help window
Escape     Clear search / Close dialogs""",
            ),
            (
                "💾 Saving Changes",
                """Changes are saved to your .env file in this format:

  IGNORE_MODELS_OPENAI=pattern1,pattern2*
  WHITELIST_MODELS_OPENAI=specific-model

Click "Save" to persist changes, or "Discard" to revert.
Closing the window with unsaved changes will prompt you.""",
            ),
        ]

        for section_title, content in sections:
            text_widget.insert("end", f"\n{section_title}\n", "section_title")
            text_widget.insert("end", "─" * 50 + "\n", "separator")
            text_widget.insert("end", content.strip() + "\n", "content")


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


class ImportRulesDialog(ctk.CTkToplevel):
    """Modal dialog for importing rules from comma-separated text."""

    def __init__(self, parent, rule_type: str):
        super().__init__(parent)

        self.result: Optional[List[str]] = None
        self.rule_type = rule_type

        title_text = (
            "Import Ignore Rules" if rule_type == "ignore" else "Import Whitelist Rules"
        )
        self.title(title_text)
        self.geometry("500x300")
        self.minsize(400, 250)

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
        self.text_box.focus_set()

        # Bind escape to cancel
        self.bind("<Escape>", lambda e: self._on_cancel())

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _create_content(self):
        """Build dialog content."""
        # Instructions at TOP
        instruction_frame = ctk.CTkFrame(self, fg_color="transparent")
        instruction_frame.pack(fill="x", padx=20, pady=(15, 10))

        instruction = ctk.CTkLabel(
            instruction_frame,
            text="Paste comma-separated patterns below (will REPLACE all existing rules):",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        instruction.pack(anchor="w")

        example = ctk.CTkLabel(
            instruction_frame,
            text="Example: gpt-4*, claude-3*, model-name",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        example.pack(anchor="w")

        # Buttons at BOTTOM - pack BEFORE textbox to reserve space
        btn_frame = ctk.CTkFrame(self, fg_color="transparent", height=50)
        btn_frame.pack(side="bottom", fill="x", padx=20, pady=(10, 15))
        btn_frame.pack_propagate(False)

        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=100,
            height=32,
            command=self._on_cancel,
        )
        cancel_btn.pack(side="right", padx=(10, 0))

        import_btn = ctk.CTkButton(
            btn_frame,
            text="Replace All",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            width=110,
            height=32,
            command=self._on_import,
        )
        import_btn.pack(side="right")

        # Text box fills MIDDLE space - pack LAST
        self.text_box = ctk.CTkTextbox(
            self,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_TERTIARY,
            border_color=BORDER_COLOR,
            border_width=1,
            text_color=TEXT_PRIMARY,
            wrap="word",
        )
        self.text_box.pack(fill="both", expand=True, padx=20, pady=(0, 0))

        # Bind Ctrl+Enter to import
        self.text_box.bind("<Control-Return>", lambda e: self._on_import())

    def _on_import(self):
        """Parse and return the patterns."""
        text = self.text_box.get("1.0", "end").strip()
        if text:
            # Parse comma-separated patterns
            patterns = [p.strip() for p in text.split(",") if p.strip()]
            self.result = patterns
        else:
            self.result = []
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

    def show(self) -> Optional[List[str]]:
        """Show dialog and return list of patterns, or None if cancelled."""
        self.wait_window()
        return self.result


class ImportResultDialog(ctk.CTkToplevel):
    """Simple dialog showing import results."""

    def __init__(self, parent, added: int, skipped: int, is_replace: bool = False):
        super().__init__(parent)

        self.title("Import Complete")
        self.geometry("380x160")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Configure appearance
        self.configure(fg_color=BG_PRIMARY)

        # Build content
        self._create_content(added, skipped, is_replace)

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

        # Focus
        self.focus_force()

        # Bind escape and enter to close
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Return>", lambda e: self.destroy())

    def _create_content(self, added: int, skipped: int, is_replace: bool):
        """Build dialog content."""
        # Icon and message
        msg_frame = ctk.CTkFrame(self, fg_color="transparent")
        msg_frame.pack(fill="x", padx=30, pady=(25, 15))

        icon = ctk.CTkLabel(
            msg_frame,
            text="✅" if added > 0 else "ℹ️",
            font=(FONT_FAMILY, 28),
            text_color=ACCENT_GREEN if added > 0 else ACCENT_BLUE,
        )
        icon.pack(side="left", padx=(0, 15))

        text_frame = ctk.CTkFrame(msg_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="x", expand=True)

        # Title text differs based on mode
        if is_replace:
            if added > 0:
                added_text = f"Replaced with {added} rule{'s' if added != 1 else ''}"
            else:
                added_text = "All rules cleared"
        else:
            if added > 0:
                added_text = f"Added {added} rule{'s' if added != 1 else ''}"
            else:
                added_text = "No new rules added"

        title = ctk.CTkLabel(
            text_frame,
            text=added_text,
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        title.pack(anchor="w")

        # Subtitle for skipped/duplicates
        if skipped > 0:
            skip_text = f"{skipped} duplicate{'s' if skipped != 1 else ''} skipped"
            subtitle = ctk.CTkLabel(
                text_frame,
                text=skip_text,
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                text_color=TEXT_MUTED,
                anchor="w",
            )
            subtitle.pack(anchor="w")

        # OK button
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(0, 20))

        ok_btn = ctk.CTkButton(
            btn_frame,
            text="OK",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            width=80,
            command=self.destroy,
        )
        ok_btn.pack(side="right")


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
# VIRTUAL MODEL LIST (Canvas-based for performance)
# ════════════════════════════════════════════════════════════════════════════════

# Constants for virtual list
ITEM_HEIGHT = 24  # Height of each row in pixels
INDICATOR_WIDTH = 18  # Width of status indicator


class VirtualModelList:
    """
    High-performance virtual list that only renders visible items.

    Uses a raw tkinter Canvas to draw text directly rather than
    creating individual widgets per row. This reduces widget count
    from O(n) to O(visible_rows).
    """

    def __init__(
        self,
        parent,
        show_status_indicator: bool = False,
        on_click: Optional[Callable[[str], None]] = None,
        on_right_click: Optional[Callable[[str, any], None]] = None,
    ):
        self.parent = parent
        self.show_status_indicator = show_status_indicator
        self.on_click = on_click
        self.on_right_click = on_right_click

        # Data
        self.models: List[str] = []
        self.statuses: Dict[str, ModelStatus] = {}
        self.filtered_models: List[str] = []  # Models after search filter
        self.search_query: str = ""
        self.highlighted_models: Set[str] = set()

        # UI state
        self._hover_index: Optional[int] = None

        # Create container frame
        self.frame = ctk.CTkFrame(parent, fg_color=BG_TERTIARY, corner_radius=6)

        # Create canvas (use raw tk.Canvas for performance)
        import tkinter as tk

        self.canvas = tk.Canvas(
            self.frame,
            bg=BG_TERTIARY,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar
        self.scrollbar = ctk.CTkScrollbar(self.frame, command=self._on_scroll)
        self.scrollbar.pack(side="right", fill="y")

        # Link canvas to scrollbar
        self.canvas.configure(yscrollcommand=self._on_canvas_scroll)

        # Bind events
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

    def grid(self, **kwargs):
        """Grid the container frame."""
        self.frame.grid(**kwargs)

    def grid_forget(self):
        """Hide the container frame."""
        self.frame.grid_forget()

    def pack(self, **kwargs):
        """Pack the container frame."""
        self.frame.pack(**kwargs)

    def pack_forget(self):
        """Hide the container frame."""
        self.frame.pack_forget()

    def set_models(self, models: List[str], statuses: Dict[str, ModelStatus]):
        """Set the model list and statuses."""
        self.models = models
        self.statuses = statuses
        self._apply_filter()
        self._update_scroll_region()
        self._render()

    def update_statuses(self, statuses: Dict[str, ModelStatus]):
        """Update just the statuses (no model list change)."""
        self.statuses = statuses
        self._render()

    def filter_by_search(self, query: str):
        """Filter models by search query."""
        self.search_query = query.lower().strip()
        self._apply_filter()
        self._update_scroll_region()
        self._render()

    def _apply_filter(self):
        """Apply current search filter to models."""
        if not self.search_query:
            self.filtered_models = list(self.models)
        else:
            self.filtered_models = [
                m for m in self.models if self.search_query in m.lower()
            ]

    def highlight_models(self, model_ids: Set[str]):
        """Set which models should be highlighted."""
        self.highlighted_models = model_ids
        self._render()

    def clear_highlights(self):
        """Clear all highlights."""
        self.highlighted_models.clear()
        self._render()

    def scroll_to_model(self, model_id: str):
        """Scroll to make a model visible."""
        if model_id not in self.filtered_models:
            return

        index = self.filtered_models.index(model_id)
        total_height = len(self.filtered_models) * ITEM_HEIGHT
        canvas_height = self.canvas.winfo_height()

        if total_height <= canvas_height:
            return

        # Calculate position to center the item
        item_y = index * ITEM_HEIGHT
        target_scroll = (item_y - canvas_height / 2 + ITEM_HEIGHT / 2) / total_height
        target_scroll = max(0, min(1, target_scroll))

        self.canvas.yview_moveto(target_scroll)
        self._render()

    def _update_scroll_region(self):
        """Update the scrollable region based on item count."""
        total_height = max(len(self.filtered_models) * ITEM_HEIGHT, 1)
        self.canvas.configure(scrollregion=(0, 0, 100, total_height))

    def _on_scroll(self, *args):
        """Handle scrollbar command."""
        self.canvas.yview(*args)
        self._render()

    def _on_canvas_scroll(self, first: float, last: float):
        """Handle canvas scroll update - just update scrollbar."""
        self.scrollbar.set(first, last)

    def _on_configure(self, event=None):
        """Handle canvas resize."""
        self._update_scroll_region()
        self._render()

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        delta = -1 * (event.delta // 120)
        self.canvas.yview_scroll(delta, "units")
        self._render()
        return "break"

    def _get_index_at_y(self, y: int) -> Optional[int]:
        """Get the model index at a y coordinate."""
        if not self.filtered_models:
            return None

        # Convert window y coordinate to canvas (scrollregion) coordinate
        canvas_y = self.canvas.canvasy(y)

        # Calculate index from absolute position
        index = int(canvas_y // ITEM_HEIGHT)

        if 0 <= index < len(self.filtered_models):
            return index
        return None

    def _on_left_click(self, event):
        """Handle left click."""
        index = self._get_index_at_y(event.y)
        if index is not None and self.on_click:
            model_id = self.filtered_models[index]
            self.on_click(model_id)

    def _on_right_click(self, event):
        """Handle right click."""
        index = self._get_index_at_y(event.y)
        if index is not None and self.on_right_click:
            model_id = self.filtered_models[index]
            self.on_right_click(model_id, event)

    def _on_mouse_motion(self, event):
        """Handle mouse motion for hover effect."""
        new_hover = self._get_index_at_y(event.y)
        if new_hover != self._hover_index:
            self._hover_index = new_hover
            self._render()

    def _on_mouse_leave(self, event):
        """Handle mouse leaving canvas."""
        if self._hover_index is not None:
            self._hover_index = None
            self._render()

    def _render(self):
        """Render only the visible items."""
        self.canvas.delete("all")

        if not self.filtered_models:
            # Show empty state
            canvas_height = self.canvas.winfo_height()
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                canvas_height // 2,
                text="No models",
                fill=TEXT_MUTED,
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            )
            return

        canvas_height = self.canvas.winfo_height()
        canvas_width = self.canvas.winfo_width()
        total_height = len(self.filtered_models) * ITEM_HEIGHT

        # Calculate visible range based on scroll position
        scroll_position = self.canvas.yview()[0]
        scroll_offset = scroll_position * total_height
        first_visible = int(scroll_offset // ITEM_HEIGHT)
        visible_count = int(canvas_height // ITEM_HEIGHT) + 2  # +2 for partial rows

        # Clamp to valid range
        first_visible = max(0, first_visible)
        last_visible = min(len(self.filtered_models), first_visible + visible_count)

        # Draw visible items at ABSOLUTE positions
        # The canvas scrollregion + yview handles showing the correct portion
        for i in range(first_visible, last_visible):
            model_id = self.filtered_models[i]
            status = self.statuses.get(
                model_id,
                ModelStatus(model_id=model_id, status="normal", color=NORMAL_COLOR),
            )

            # Absolute y position in the virtual list
            y = i * ITEM_HEIGHT
            y_center = y + ITEM_HEIGHT // 2

            # Background for hover/highlight
            is_highlighted = model_id in self.highlighted_models
            is_hovered = i == self._hover_index

            if is_highlighted:
                self.canvas.create_rectangle(
                    0, y, canvas_width, y + ITEM_HEIGHT, fill=HIGHLIGHT_BG, outline=""
                )
            elif is_hovered:
                self.canvas.create_rectangle(
                    0, y, canvas_width, y + ITEM_HEIGHT, fill=BG_HOVER, outline=""
                )

            # Status indicator (for right list)
            x_offset = 8
            if self.show_status_indicator:
                indicator_text = {
                    "normal": "●",
                    "ignored": "✗",
                    "whitelisted": "★",
                }.get(status.status, "●")
                self.canvas.create_text(
                    x_offset + INDICATOR_WIDTH // 2,
                    y_center,
                    text=indicator_text,
                    fill=status.color,
                    font=(FONT_FAMILY, FONT_SIZE_SMALL),
                )
                x_offset += INDICATOR_WIDTH

            # Model name
            text_color = status.color if self.show_status_indicator else TEXT_PRIMARY
            display_name = status.display_name

            self.canvas.create_text(
                x_offset,
                y_center,
                text=display_name,
                fill=text_color,
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                anchor="w",
            )

    def get_scroll_position(self) -> float:
        """Get current scroll position (0-1) directly from canvas."""
        return self.canvas.yview()[0]

    def set_scroll_position(self, pos: float, render: bool = True):
        """Set scroll position (0-1) and optionally render."""
        self.canvas.yview_moveto(pos)
        if render:
            self._render()


class VirtualSyncModelLists(ctk.CTkFrame):
    """
    Container with two synchronized virtual model lists.

    Left list: All fetched models (plain display)
    Right list: Same models with colored status indicators

    Both lists scroll together.
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
        self._syncing_scroll = False

        self._create_content()

    def _create_content(self):
        """Build the dual list layout."""
        # Don't let content dictate size - let parent grid control height
        self.grid_propagate(False)

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Left header frame
        left_header_frame = ctk.CTkFrame(self, fg_color="transparent")
        left_header_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(0, 5))

        left_header = ctk.CTkLabel(
            left_header_frame,
            text="All Fetched Models",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            text_color=TEXT_PRIMARY,
        )
        left_header.pack(side="left")

        self.left_count_label = ctk.CTkLabel(
            left_header_frame,
            text="(0)",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_MUTED,
        )
        self.left_count_label.pack(side="left", padx=(5, 0))

        # Copy button for all models
        self.left_copy_btn = ctk.CTkButton(
            left_header_frame,
            text="Copy",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=50,
            height=20,
            command=self._copy_all_models,
        )
        self.left_copy_btn.pack(side="right")
        ToolTip(self.left_copy_btn, "Copy all model names (comma-separated)")

        # Right header frame
        right_header_frame = ctk.CTkFrame(self, fg_color="transparent")
        right_header_frame.grid(row=0, column=1, sticky="ew", padx=8, pady=(0, 5))

        right_header = ctk.CTkLabel(
            right_header_frame,
            text="Filtered Status",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            text_color=TEXT_PRIMARY,
        )
        right_header.pack(side="left")

        self.right_count_label = ctk.CTkLabel(
            right_header_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_MUTED,
        )
        self.right_count_label.pack(side="left", padx=(5, 0))

        # Copy button for filtered models
        self.right_copy_btn = ctk.CTkButton(
            right_header_frame,
            text="Copy",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=50,
            height=20,
            command=self._copy_filtered_models,
        )
        self.right_copy_btn.pack(side="right")
        ToolTip(self.right_copy_btn, "Copy available model names (comma-separated)")

        # Create virtual lists
        self.left_list = VirtualModelList(
            self,
            show_status_indicator=False,
            on_click=self.on_model_click,
            on_right_click=self.on_model_right_click,
        )
        self.left_list.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        self.right_list = VirtualModelList(
            self,
            show_status_indicator=True,
            on_click=self.on_model_click,
            on_right_click=self.on_model_right_click,
        )
        self.right_list.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        # Synchronize scrolling
        self._setup_scroll_sync()

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

    def _setup_scroll_sync(self):
        """Setup synchronized scrolling between both lists."""
        # Override the scroll handlers to sync both lists
        original_left_scroll = self.left_list._on_scroll
        original_right_scroll = self.right_list._on_scroll
        original_left_wheel = self.left_list._on_mousewheel
        original_right_wheel = self.right_list._on_mousewheel

        def sync_scroll_left(*args):
            if self._syncing_scroll:
                return
            self._syncing_scroll = True
            original_left_scroll(*args)
            # Sync to right - get position after scroll completed
            pos = self.left_list.get_scroll_position()
            self.right_list.set_scroll_position(pos)
            self._syncing_scroll = False

        def sync_scroll_right(*args):
            if self._syncing_scroll:
                return
            self._syncing_scroll = True
            original_right_scroll(*args)
            # Sync to left - get position after scroll completed
            pos = self.right_list.get_scroll_position()
            self.left_list.set_scroll_position(pos)
            self._syncing_scroll = False

        def sync_wheel_left(event):
            if self._syncing_scroll:
                return "break"
            self._syncing_scroll = True
            original_left_wheel(event)
            # Sync to right - get position after scroll completed
            pos = self.left_list.get_scroll_position()
            self.right_list.set_scroll_position(pos)
            self._syncing_scroll = False
            return "break"

        def sync_wheel_right(event):
            if self._syncing_scroll:
                return "break"
            self._syncing_scroll = True
            original_right_wheel(event)
            # Sync to left - get position after scroll completed
            pos = self.right_list.get_scroll_position()
            self.left_list.set_scroll_position(pos)
            self._syncing_scroll = False
            return "break"

        # Override the method references
        self.left_list._on_scroll = sync_scroll_left
        self.right_list._on_scroll = sync_scroll_right

        # IMPORTANT: Reconfigure scrollbars to use the new sync handlers
        # The scrollbars were created with command=_on_scroll before we overrode it
        self.left_list.scrollbar.configure(command=sync_scroll_left)
        self.right_list.scrollbar.configure(command=sync_scroll_right)

        # Rebind mouse wheel events
        self.left_list.canvas.bind("<MouseWheel>", sync_wheel_left)
        self.right_list.canvas.bind("<MouseWheel>", sync_wheel_right)

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

        self.left_list.set_models(models, self.statuses)
        self.right_list.set_models(models, self.statuses)

        self._update_counts()
        self.hide_overlays()

    def update_statuses(self, statuses: List[ModelStatus]):
        """Update status display for all models."""
        self.statuses = {s.model_id: s for s in statuses}
        self.left_list.update_statuses(self.statuses)
        self.right_list.update_statuses(self.statuses)
        self._update_counts()

    def _update_counts(self):
        """Update the count labels."""
        total = len(self.models)
        available = sum(1 for s in self.statuses.values() if s.status != "ignored")

        self.left_count_label.configure(text=f"({total})")
        self.right_count_label.configure(text=f"{available} available")

    def filter_by_search(self, query: str):
        """Filter models by search query."""
        self.left_list.filter_by_search(query)
        self.right_list.filter_by_search(query)

    def highlight_models_by_rule(self, rule: FilterRule):
        """Highlight all models affected by a rule."""
        model_set = set(rule.affected_models)
        self.left_list.highlight_models(model_set)
        self.right_list.highlight_models(model_set)

        # Scroll to first match
        if rule.affected_models:
            self.left_list.scroll_to_model(rule.affected_models[0])
            # Sync right list scroll
            pos = self.left_list.get_scroll_position()
            self.right_list.set_scroll_position(pos)

    def highlight_model(self, model_id: str):
        """Highlight a specific model."""
        model_set = {model_id}
        self.left_list.highlight_models(model_set)
        self.right_list.highlight_models(model_set)

    def clear_highlights(self):
        """Clear all model highlights."""
        self.left_list.clear_highlights()
        self.right_list.clear_highlights()

    def scroll_to_affected(self, affected_models: List[str]):
        """Scroll to first affected model."""
        if affected_models:
            self.left_list.scroll_to_model(affected_models[0])
            pos = self.left_list.get_scroll_position()
            self.right_list.set_scroll_position(pos)

    def _get_model_display_name(self, model_id: str) -> str:
        """Get model name without provider prefix."""
        if "/" in model_id:
            return model_id.split("/", 1)[1]
        return model_id

    def _copy_all_models(self):
        """Copy all model names to clipboard (comma-separated, without provider prefix)."""
        if not self.models:
            return
        names = [self._get_model_display_name(m) for m in self.models]
        text = ", ".join(names)
        self.clipboard_clear()
        self.clipboard_append(text)

    def _copy_filtered_models(self):
        """Copy filtered/available model names to clipboard (comma-separated)."""
        if not self.models:
            return
        # Get only models that are not ignored
        available = [
            self._get_model_display_name(m)
            for m in self.models
            if self.statuses.get(m) and self.statuses[m].status != "ignored"
        ]
        # Also include models with no status (default to available)
        for m in self.models:
            if m not in self.statuses:
                available.append(self._get_model_display_name(m))
        text = ", ".join(available)
        self.clipboard_clear()
        self.clipboard_append(text)

    def get_model_at_position(self, model_id: str) -> Optional[ModelStatus]:
        """Get the status of a model."""
        return self.statuses.get(model_id)


# ════════════════════════════════════════════════════════════════════════════════
# VIRTUAL RULE LIST (Canvas-based for performance)
# ════════════════════════════════════════════════════════════════════════════════

# Constants for virtual rule list
RULE_ITEM_HEIGHT = 32  # Height of each rule row
RULE_DELETE_WIDTH = 24  # Width of delete button area
RULE_COUNT_WIDTH = 40  # Width of count area
RULE_PADDING = 8  # Horizontal padding


class VirtualRuleList:
    """
    High-performance virtual list for filter rules.

    Uses a raw tkinter Canvas to draw rules directly rather than
    creating individual widgets per row.
    """

    def __init__(
        self,
        parent,
        rule_type: str,  # 'ignore' or 'whitelist'
        on_rule_click: Callable[[FilterRule], None],
        on_rule_delete: Callable[[str], None],
    ):
        self.parent = parent
        self.rule_type = rule_type
        self.on_rule_click = on_rule_click
        self.on_rule_delete = on_rule_delete

        # Data
        self.rules: List[FilterRule] = []
        self.highlighted_pattern: Optional[str] = None

        # UI state
        self._hover_index: Optional[int] = None
        self._hover_delete: bool = False  # True if hovering over delete button

        # Tooltip state
        self._tooltip_window = None
        self._tooltip_after_id = None
        self._tooltip_rule_index: Optional[int] = None

        # Create container frame
        self.frame = ctk.CTkFrame(parent, fg_color="transparent")

        # Create canvas
        import tkinter as tk

        self.canvas = tk.Canvas(
            self.frame,
            bg=BG_SECONDARY,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar
        self.scrollbar = ctk.CTkScrollbar(self.frame, command=self._on_scroll)
        self.scrollbar.pack(side="right", fill="y")

        # Link canvas to scrollbar
        self.canvas.configure(yscrollcommand=self._on_canvas_scroll)

        # Bind events
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

    def pack(self, **kwargs):
        """Pack the container frame."""
        self.frame.pack(**kwargs)

    def set_rules(self, rules: List[FilterRule]):
        """Set the rules to display."""
        self.rules = rules
        self._update_scroll_region()
        self._render()

    def add_rule(self, rule: FilterRule):
        """Add a rule to the list."""
        # Check for duplicates
        if any(r.pattern == rule.pattern for r in self.rules):
            return
        self.rules.append(rule)
        self._update_scroll_region()
        self._render()

    def remove_rule(self, pattern: str):
        """Remove a rule by pattern."""
        self.rules = [r for r in self.rules if r.pattern != pattern]
        self._update_scroll_region()
        self._render()

    def update_rule_counts(self, rules: List[FilterRule]):
        """Update affected counts from new rule data."""
        rule_map = {r.pattern: r for r in rules}
        for rule in self.rules:
            if rule.pattern in rule_map:
                rule.affected_count = rule_map[rule.pattern].affected_count
                rule.affected_models = rule_map[rule.pattern].affected_models
        self._render()

    def highlight_rule(self, pattern: Optional[str]):
        """Highlight a specific rule."""
        self.highlighted_pattern = pattern
        if pattern:
            self._scroll_to_rule(pattern)
        self._render()

    def clear_highlights(self):
        """Clear all highlights."""
        self.highlighted_pattern = None
        self._render()

    def clear_all(self):
        """Remove all rules."""
        self.rules = []
        self._update_scroll_region()
        self._render()

    def _scroll_to_rule(self, pattern: str):
        """Scroll to make a rule visible."""
        for i, rule in enumerate(self.rules):
            if rule.pattern == pattern:
                total_height = len(self.rules) * RULE_ITEM_HEIGHT
                canvas_height = self.canvas.winfo_height()

                if total_height <= canvas_height:
                    return

                item_y = i * RULE_ITEM_HEIGHT
                target_scroll = (
                    item_y - canvas_height / 2 + RULE_ITEM_HEIGHT / 2
                ) / total_height
                target_scroll = max(0, min(1, target_scroll))

                self.canvas.yview_moveto(target_scroll)
                self._render()
                return

    def _update_scroll_region(self):
        """Update the scrollable region."""
        total_height = max(len(self.rules) * RULE_ITEM_HEIGHT, 1)
        self.canvas.configure(scrollregion=(0, 0, 100, total_height))

    def _on_scroll(self, *args):
        """Handle scrollbar command."""
        self.canvas.yview(*args)
        self._render()

    def _on_canvas_scroll(self, first: float, last: float):
        """Handle canvas scroll update."""
        self.scrollbar.set(first, last)

    def _on_configure(self, event=None):
        """Handle canvas resize."""
        self._update_scroll_region()
        self._render()

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        delta = -1 * (event.delta // 120)
        self.canvas.yview_scroll(delta, "units")
        self._render()
        return "break"

    def _get_index_at_y(self, y: int) -> Optional[int]:
        """Get the rule index at a y coordinate."""
        if not self.rules:
            return None

        canvas_y = self.canvas.canvasy(y)
        index = int(canvas_y // RULE_ITEM_HEIGHT)

        if 0 <= index < len(self.rules):
            return index
        return None

    def _is_over_delete(self, x: int) -> bool:
        """Check if x coordinate is over the delete button."""
        canvas_width = self.canvas.winfo_width()
        delete_start = canvas_width - RULE_DELETE_WIDTH - RULE_PADDING
        return x >= delete_start

    def _on_left_click(self, event):
        """Handle left click."""
        index = self._get_index_at_y(event.y)
        if index is None:
            return

        rule = self.rules[index]

        if self._is_over_delete(event.x):
            # Click on delete button
            self.on_rule_delete(rule.pattern)
        else:
            # Click on rule
            self.on_rule_click(rule)

    def _on_mouse_motion(self, event):
        """Handle mouse motion for hover effect."""
        new_hover = self._get_index_at_y(event.y)
        new_hover_delete = (
            self._is_over_delete(event.x) if new_hover is not None else False
        )

        if new_hover != self._hover_index or new_hover_delete != self._hover_delete:
            self._hover_index = new_hover
            self._hover_delete = new_hover_delete
            self._render()

        # Handle tooltip
        if new_hover != self._tooltip_rule_index:
            self._hide_tooltip()
            if new_hover is not None and not new_hover_delete:
                self._schedule_tooltip(new_hover)

    def _on_mouse_leave(self, event):
        """Handle mouse leaving canvas."""
        if self._hover_index is not None:
            self._hover_index = None
            self._hover_delete = False
            self._render()
        self._hide_tooltip()

    def _schedule_tooltip(self, index: int):
        """Schedule tooltip to appear."""
        self._tooltip_rule_index = index
        self._tooltip_after_id = self.canvas.after(
            500, lambda: self._show_tooltip(index)
        )

    def _show_tooltip(self, index: int):
        """Show tooltip for a rule."""
        if index != self._tooltip_rule_index or index >= len(self.rules):
            return

        rule = self.rules[index]

        # Build tooltip text
        if rule.affected_models:
            if len(rule.affected_models) <= 5:
                models_text = "\n".join(rule.affected_models)
            else:
                models_text = "\n".join(rule.affected_models[:5])
                models_text += f"\n... and {len(rule.affected_models) - 5} more"
            text = f"Matches:\n{models_text}"
        else:
            text = "No models match this pattern"

        # Position tooltip
        x = self.canvas.winfo_rootx() + 20
        y = (
            self.canvas.winfo_rooty()
            + (index + 1) * RULE_ITEM_HEIGHT
            - int(self.canvas.canvasy(0))
        )

        # Create tooltip window
        self._tooltip_window = tw = ctk.CTkToplevel(self.canvas)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(fg_color=BG_SECONDARY)

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
            text=text,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
            padx=10,
            pady=5,
        )
        label.pack()
        tw.lift()

    def _hide_tooltip(self):
        """Hide the tooltip."""
        if self._tooltip_after_id:
            self.canvas.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None
        if self._tooltip_window:
            self._tooltip_window.destroy()
            self._tooltip_window = None
        self._tooltip_rule_index = None

    def _render(self):
        """Render only the visible rules."""
        self.canvas.delete("all")

        if not self.rules:
            # Show empty state
            canvas_height = self.canvas.winfo_height()
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                canvas_height // 2,
                text="No rules configured\nAdd patterns below",
                fill=TEXT_MUTED,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                justify="center",
            )
            return

        canvas_height = self.canvas.winfo_height()
        canvas_width = self.canvas.winfo_width()
        total_height = len(self.rules) * RULE_ITEM_HEIGHT

        # Calculate visible range
        scroll_position = self.canvas.yview()[0]
        scroll_offset = scroll_position * total_height
        first_visible = int(scroll_offset // RULE_ITEM_HEIGHT)
        visible_count = int(canvas_height // RULE_ITEM_HEIGHT) + 2

        first_visible = max(0, first_visible)
        last_visible = min(len(self.rules), first_visible + visible_count)

        # Draw visible rules
        for i in range(first_visible, last_visible):
            rule = self.rules[i]

            # Absolute y position
            y = i * RULE_ITEM_HEIGHT
            y_center = y + RULE_ITEM_HEIGHT // 2

            # Background
            is_highlighted = rule.pattern == self.highlighted_pattern
            is_hovered = i == self._hover_index

            if is_highlighted:
                # Highlighted - use rule color for border effect
                self.canvas.create_rectangle(
                    2,
                    y + 2,
                    canvas_width - 2,
                    y + RULE_ITEM_HEIGHT - 2,
                    fill=BG_TERTIARY,
                    outline=rule.color,
                    width=2,
                )
            elif is_hovered:
                self.canvas.create_rectangle(
                    2,
                    y + 2,
                    canvas_width - 2,
                    y + RULE_ITEM_HEIGHT - 2,
                    fill=BG_HOVER,
                    outline=BORDER_COLOR,
                    width=1,
                )
            else:
                self.canvas.create_rectangle(
                    2,
                    y + 2,
                    canvas_width - 2,
                    y + RULE_ITEM_HEIGHT - 2,
                    fill=BG_TERTIARY,
                    outline=BORDER_COLOR,
                    width=1,
                )

            # Pattern text (colored)
            self.canvas.create_text(
                RULE_PADDING + 4,
                y_center,
                text=rule.pattern,
                fill=rule.color,
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                anchor="w",
            )

            # Count text
            count_x = canvas_width - RULE_DELETE_WIDTH - RULE_COUNT_WIDTH - RULE_PADDING
            self.canvas.create_text(
                count_x,
                y_center,
                text=f"({rule.affected_count})",
                fill=TEXT_MUTED,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                anchor="w",
            )

            # Delete button
            delete_x = (
                canvas_width - RULE_DELETE_WIDTH - RULE_PADDING + RULE_DELETE_WIDTH // 2
            )
            delete_color = (
                ACCENT_RED if (is_hovered and self._hover_delete) else TEXT_MUTED
            )
            self.canvas.create_text(
                delete_x,
                y_center,
                text="×",
                fill=delete_color,
                font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            )


# ════════════════════════════════════════════════════════════════════════════════
# RULE PANEL COMPONENT
# ════════════════════════════════════════════════════════════════════════════════


class RulePanel(ctk.CTkFrame):
    """
    Panel containing rule chips, input field, and add button.

    Uses VirtualRuleList for high-performance rendering of rules.
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

        self._create_content()

    def _create_content(self):
        """Build panel content."""
        # Title row at top (compact) with count and buttons
        title_frame = ctk.CTkFrame(self, fg_color="transparent", height=22)
        title_frame.pack(side="top", fill="x", padx=10, pady=(4, 2))
        title_frame.pack_propagate(False)

        # Base title (without count)
        self._base_title = self.title
        self._rule_count = 0

        self.title_label = ctk.CTkLabel(
            title_frame,
            text=f"{self.title}: 0",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            text_color=TEXT_PRIMARY,
        )
        self.title_label.pack(side="left")

        # Import button (right side)
        import_btn = ctk.CTkButton(
            title_frame,
            text="Import",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_TERTIARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=50,
            height=18,
            command=self._on_import_clicked,
        )
        import_btn.pack(side="right", padx=(4, 0))
        ToolTip(import_btn, "Import rules from comma-separated text")

        # Copy button
        copy_btn = ctk.CTkButton(
            title_frame,
            text="Copy",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_TERTIARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=45,
            height=18,
            command=self._on_copy_clicked,
        )
        copy_btn.pack(side="right")
        ToolTip(copy_btn, "Copy all rules (comma-separated)")

        # Input frame at BOTTOM - pack BEFORE rule_list to reserve space
        input_frame = ctk.CTkFrame(self, fg_color="transparent", height=32)
        input_frame.pack(side="bottom", fill="x", padx=6, pady=(2, 4))
        input_frame.pack_propagate(False)  # Prevent children from changing frame height

        # Pattern input
        self.input_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="pattern1, pattern2*, ...",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_TERTIARY,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_MUTED,
            height=28,
        )
        self.input_entry.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.input_entry.bind("<Return>", self._on_add_clicked)
        self.input_entry.bind("<KeyRelease>", self._on_input_key)

        # Add button
        add_btn = ctk.CTkButton(
            input_frame,
            text="+ Add",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            width=55,
            height=28,
            command=self._on_add_clicked,
        )
        add_btn.pack(side="right")

        # Virtual rule list fills REMAINING middle space - pack LAST
        self.rule_list = VirtualRuleList(
            self,
            rule_type=self.rule_type,
            on_rule_click=self.on_rule_clicked,
            on_rule_delete=self._on_rule_delete,
        )
        self.rule_list.pack(side="top", fill="both", expand=True, padx=6, pady=(0, 2))

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
        if hasattr(self, "_add_pattern_callback"):
            self._add_pattern_callback(pattern)

    def set_add_callback(self, callback: Callable[[str], None]):
        """Set the callback for adding patterns."""
        self._add_pattern_callback = callback

    def add_rule_chip(self, rule: FilterRule):
        """Add a rule to the panel."""
        self.rule_list.add_rule(rule)

    def remove_rule_chip(self, pattern: str):
        """Remove a rule from the panel."""
        self.rule_list.remove_rule(pattern)

    def _on_rule_delete(self, pattern: str):
        """Handle rule deletion."""
        if hasattr(self, "_delete_pattern_callback"):
            self._delete_pattern_callback(pattern)

    def set_delete_callback(self, callback: Callable[[str], None]):
        """Set the callback for deleting patterns."""
        self._delete_pattern_callback = callback

    def update_rule_counts(self, rules: List[FilterRule], models: List[str]):
        """Update affected counts for all rules."""
        self.rule_list.update_rule_counts(rules)
        self._update_title_count(len(rules))

    def _update_title_count(self, count: int):
        """Update the rule count in the title."""
        self._rule_count = count
        self.title_label.configure(text=f"{self._base_title}: {count}")

    def highlight_rule(self, pattern: str):
        """Highlight a specific rule and scroll to it."""
        self.rule_list.highlight_rule(pattern)

    def clear_highlights(self):
        """Clear all rule highlights."""
        self.rule_list.clear_highlights()

    def clear_all(self):
        """Remove all rules."""
        self.rule_list.clear_all()

    def get_input_text(self) -> str:
        """Get current input text."""
        return self.input_entry.get().strip()

    def clear_input(self):
        """Clear the input field."""
        self.input_entry.delete(0, "end")

    def _on_copy_clicked(self):
        """Copy all rule patterns to clipboard as comma-separated string."""
        patterns = [r.pattern for r in self.rule_list.rules]
        if patterns:
            text = ", ".join(patterns)
            self.clipboard_clear()
            self.clipboard_append(text)

    def _on_import_clicked(self):
        """
        Open import dialog and REPLACE ALL existing rules.

        This is a full replace operation - all existing rules are removed
        and replaced with the imported patterns.
        """
        dialog = ImportRulesDialog(self.winfo_toplevel(), self.rule_type)
        patterns = dialog.show()

        if patterns is None:
            # Cancelled
            return

        if not patterns:
            # Empty input - show message
            ImportResultDialog(self.winfo_toplevel(), 0, 0, is_replace=True)
            return

        # Deduplicate the imported patterns (keep first occurrence)
        seen = set()
        unique_patterns = []
        duplicates_in_import = 0
        for p in patterns:
            if p not in seen:
                seen.add(p)
                unique_patterns.append(p)
            else:
                duplicates_in_import += 1

        # Clear all existing rules first
        if hasattr(self, "_clear_all_callback"):
            self._clear_all_callback()

        # Add all unique patterns (skip coverage check since we're replacing)
        added = 0
        if hasattr(self, "_replace_add_callback"):
            for pattern in unique_patterns:
                if self._replace_add_callback(pattern):
                    added += 1

        # Show result dialog
        ImportResultDialog(
            self.winfo_toplevel(), added, duplicates_in_import, is_replace=True
        )

    def set_clear_all_callback(self, callback: Callable[[], None]):
        """Set the callback for clearing all rules (used by replace import)."""
        self._clear_all_callback = callback

    def set_replace_add_callback(self, callback: Callable[[str], bool]):
        """Set the callback for adding patterns in replace mode (skips coverage check)."""
        self._replace_add_callback = callback

    def get_all_patterns(self) -> List[str]:
        """Get all rule patterns."""
        return [r.pattern for r in self.rule_list.rules]


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

        # Build UI with grid layout for responsive sizing
        self._create_main_layout()

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

    def _create_main_layout(self):
        """Create the main layout with grid weights for 3:1 ratio."""
        # Main content frame - regular frame with grid layout
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=15, pady=(5, 8))

        # Configure grid with proper weights for 3:1 ratio
        self.content_frame.grid_columnconfigure(0, weight=1)

        # Row 0: Header - fixed height
        self.content_frame.grid_rowconfigure(0, weight=0)
        # Row 1: Search - fixed height
        self.content_frame.grid_rowconfigure(1, weight=0)
        # Row 2: Model lists - weight=3 for 3:1 ratio, minimum 100px
        self.content_frame.grid_rowconfigure(2, weight=3, minsize=200)
        # Row 3: Rule panels - weight=1 for 3:1 ratio, minimum 55px
        self.content_frame.grid_rowconfigure(3, weight=1, minsize=55)
        # Row 4: Status bar - fixed height
        self.content_frame.grid_rowconfigure(4, weight=0)

        # Create all sections
        self._create_header()
        self._create_search_bar()
        self._create_model_lists()
        self._create_rule_panels()
        self._create_status_bar()
        self._create_action_buttons()

    def _activate_window(self):
        """Activate and focus the window."""
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False))

    def _create_header(self):
        """Create the header with provider selector and buttons (compact)."""
        header = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        # Title (smaller font)
        title = ctk.CTkLabel(
            header,
            text="🎯 Model Filter Configuration",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=TEXT_PRIMARY,
        )
        title.pack(side="left")

        # Help button (smaller)
        help_btn = ctk.CTkButton(
            header,
            text="?",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=26,
            height=26,
            corner_radius=13,
            command=self._show_help,
        )
        help_btn.pack(side="right", padx=(8, 0))
        ToolTip(help_btn, "Help (F1)")

        # Refresh button (smaller)
        refresh_btn = ctk.CTkButton(
            header,
            text="🔄 Refresh",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=80,
            height=26,
            command=self._refresh_models,
        )
        refresh_btn.pack(side="right", padx=(8, 0))
        ToolTip(refresh_btn, "Refresh models (Ctrl+R)")

        # Provider selector (compact)
        provider_frame = ctk.CTkFrame(header, fg_color="transparent")
        provider_frame.pack(side="right")

        provider_label = ctk.CTkLabel(
            provider_frame,
            text="Provider:",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        )
        provider_label.pack(side="left", padx=(0, 6))

        self.provider_dropdown = ctk.CTkComboBox(
            provider_frame,
            values=["Loading..."],
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            dropdown_font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SECONDARY,
            border_color=BORDER_COLOR,
            button_color=BORDER_COLOR,
            button_hover_color=BG_HOVER,
            dropdown_fg_color=BG_SECONDARY,
            dropdown_hover_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            width=160,
            height=26,
            state="readonly",
            command=self._on_provider_changed,
        )
        self.provider_dropdown.pack(side="left")

    def _create_search_bar(self):
        """Create the search bar (compact version)."""
        search_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        search_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))

        search_icon = ctk.CTkLabel(
            search_frame,
            text="🔍",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_MUTED,
        )
        search_icon.pack(side="left", padx=(0, 6))

        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Search models...",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_MUTED,
            height=28,
        )
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_entry.bind("<KeyRelease>", self._on_search_changed)

        # Clear button
        clear_btn = ctk.CTkButton(
            search_frame,
            text="×",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color="transparent",
            hover_color=BG_HOVER,
            text_color=TEXT_MUTED,
            width=28,
            height=28,
            command=self._clear_search,
        )
        clear_btn.pack(side="left")

    def _create_model_lists(self):
        """Create the synchronized model list panel."""
        # Use the virtual list implementation for performance
        self.model_list_panel = VirtualSyncModelLists(
            self.content_frame,
            on_model_click=self._on_model_clicked,
            on_model_right_click=self._on_model_right_clicked,
        )
        self.model_list_panel.grid(row=2, column=0, sticky="nsew", pady=(0, 5))

    def _create_rule_panels(self):
        """Create the ignore and whitelist rule panels."""
        self.rules_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.rules_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 5))
        # Don't let content dictate size - let parent grid control height
        self.rules_frame.grid_propagate(False)
        self.rules_frame.grid_columnconfigure(0, weight=1)
        self.rules_frame.grid_columnconfigure(1, weight=1)
        self.rules_frame.grid_rowconfigure(0, weight=1)

        # Ignore panel
        self.ignore_panel = RulePanel(
            self.rules_frame,
            title="🚫 Ignore Rules",
            rule_type="ignore",
            on_rules_changed=self._on_rules_changed,
            on_rule_clicked=self._on_rule_clicked,
            on_input_changed=self._on_rule_input_changed,
        )
        self.ignore_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.ignore_panel.set_add_callback(self._add_ignore_pattern)
        self.ignore_panel.set_delete_callback(self._remove_ignore_pattern)
        self.ignore_panel.set_clear_all_callback(self._clear_all_ignore_rules)
        self.ignore_panel.set_replace_add_callback(
            lambda p: self._add_ignore_pattern(p, skip_coverage_check=True)
        )

        # Whitelist panel
        self.whitelist_panel = RulePanel(
            self.rules_frame,
            title="✓ Whitelist Rules",
            rule_type="whitelist",
            on_rules_changed=self._on_rules_changed,
            on_rule_clicked=self._on_rule_clicked,
            on_input_changed=self._on_rule_input_changed,
        )
        self.whitelist_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.whitelist_panel.set_add_callback(self._add_whitelist_pattern)
        self.whitelist_panel.set_delete_callback(self._remove_whitelist_pattern)
        self.whitelist_panel.set_clear_all_callback(self._clear_all_whitelist_rules)
        self.whitelist_panel.set_replace_add_callback(
            lambda p: self._add_whitelist_pattern(p, skip_coverage_check=True)
        )

    def _create_status_bar(self):
        """Create the status bar showing available count and action buttons (compact)."""
        # Combined status bar and action buttons in one row
        self.status_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.status_frame.grid(row=4, column=0, sticky="ew", pady=(3, 3))

        # Status label (left side, smaller font)
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Select a provider to begin",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
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
        self.unsaved_label.pack(side="left", padx=(10, 0))

        # Buttons (right side, smaller)
        # Discard button
        discard_btn = ctk.CTkButton(
            self.status_frame,
            text="↩️ Discard",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=85,
            height=26,
            command=self._discard_changes,
        )
        discard_btn.pack(side="right", padx=(8, 0))

        # Save button
        save_btn = ctk.CTkButton(
            self.status_frame,
            text="💾 Save",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg_color=ACCENT_GREEN,
            hover_color="#27ae60",
            width=75,
            height=26,
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
        # Deduplicate while preserving order, then sort
        self.models = sorted(list(dict.fromkeys(models)))

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

    def _add_ignore_pattern(self, pattern: str, skip_coverage_check: bool = False):
        """
        Add an ignore pattern with smart merge logic.

        If skip_coverage_check is False (default - from main input):
        - Skip if pattern is already covered by existing rules
        - Remove existing patterns that would be covered by this new pattern

        If skip_coverage_check is True (from replace import):
        - Just add without coverage checks
        """
        if not skip_coverage_check:
            # Check if this pattern is already covered
            if self.filter_engine.is_pattern_covered(pattern, "ignore"):
                return False  # Pattern already covered, skip

            # Remove patterns that this new pattern would cover
            covered = self.filter_engine.get_covered_patterns(pattern, "ignore")
            for covered_pattern in covered:
                self._remove_ignore_pattern(covered_pattern)

        rule = self.filter_engine.add_ignore_rule(pattern)
        if rule:
            self.ignore_panel.add_rule_chip(rule)
            self._on_rules_changed()
            return True
        return False

    def _add_whitelist_pattern(self, pattern: str, skip_coverage_check: bool = False):
        """
        Add a whitelist pattern with smart merge logic.

        If skip_coverage_check is False (default - from main input):
        - Skip if pattern is already covered by existing rules
        - Remove existing patterns that would be covered by this new pattern

        If skip_coverage_check is True (from replace import):
        - Just add without coverage checks
        """
        if not skip_coverage_check:
            # Check if this pattern is already covered
            if self.filter_engine.is_pattern_covered(pattern, "whitelist"):
                return False  # Pattern already covered, skip

            # Remove patterns that this new pattern would cover
            covered = self.filter_engine.get_covered_patterns(pattern, "whitelist")
            for covered_pattern in covered:
                self._remove_whitelist_pattern(covered_pattern)

        rule = self.filter_engine.add_whitelist_rule(pattern)
        if rule:
            self.whitelist_panel.add_rule_chip(rule)
            self._on_rules_changed()
            return True
        return False

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

    def _clear_all_ignore_rules(self):
        """Clear all ignore rules (used by replace import)."""
        # Remove all rules from engine
        patterns = [r.pattern for r in self.filter_engine.ignore_rules]
        for pattern in patterns:
            self.filter_engine.remove_ignore_rule(pattern)
        # Clear the panel
        self.ignore_panel.clear_all()
        self._on_rules_changed()

    def _clear_all_whitelist_rules(self):
        """Clear all whitelist rules (used by replace import)."""
        # Remove all rules from engine
        patterns = [r.pattern for r in self.filter_engine.whitelist_rules]
        for pattern in patterns:
            self.filter_engine.remove_whitelist_rule(pattern)
        # Clear the panel
        self.whitelist_panel.clear_all()
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

        # Highlight affected models using new virtual list API
        if affected:
            affected_set = set(affected)
            self.model_list_panel.left_list.highlight_models(affected_set)
            self.model_list_panel.right_list.highlight_models(affected_set)

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

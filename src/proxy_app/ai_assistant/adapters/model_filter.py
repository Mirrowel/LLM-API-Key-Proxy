"""
ModelFilterGUI Window Context Adapter.

Implements WindowContextAdapter for the Model Filter GUI, providing:
- Full context extraction from FilterEngine and UI state
- All tools for manipulating filter rules
- State application for checkpoint rollback
"""

import copy
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..context import WindowContextAdapter
from ..prompts import MODEL_FILTER_SYSTEM_PROMPT
from ..tools import ToolDefinition, ToolResult, assistant_tool

if TYPE_CHECKING:
    from ...model_filter_gui import ModelFilterGUI

logger = logging.getLogger(__name__)


class ModelFilterWindowContext(WindowContextAdapter):
    """
    Window context adapter for the Model Filter GUI.

    Provides complete context and tool access for the AI assistant
    to help users configure model filtering rules.
    """

    def __init__(self, gui: "ModelFilterGUI"):
        """
        Initialize the adapter.

        Args:
            gui: The ModelFilterGUI instance
        """
        self._gui = gui
        self._last_context_hash: str = ""
        self._changes: List[Dict[str, Any]] = []
        self._tracking_changes: bool = False

    # =========================================================================
    # WindowContextAdapter Implementation
    # =========================================================================

    def get_full_context(self) -> Dict[str, Any]:
        """Get the complete structured state of the window."""
        gui = self._gui
        engine = gui.filter_engine

        # Get current models
        models = gui.models or []

        # Build model list with statuses
        model_items = []
        engine.get_all_statuses(models)  # Ensure cache is valid

        for model_id in models:
            status = engine.get_model_status(model_id)
            item = {
                "id": model_id,
                "display_name": status.display_name,
                "status": status.status,  # "normal", "ignored", "whitelisted"
            }
            if status.affecting_rule:
                item["affecting_rule"] = {
                    "pattern": status.affecting_rule.pattern,
                    "type": status.affecting_rule.rule_type,
                }
            else:
                item["affecting_rule"] = None
            model_items.append(item)

        # Build rules lists
        ignore_rules = []
        for rule in engine.ignore_rules:
            ignore_rules.append(
                {
                    "pattern": rule.pattern,
                    "affected_count": rule.affected_count,
                    "affected_models": rule.affected_models[
                        :10
                    ],  # Limit for context size
                }
            )

        whitelist_rules = []
        for rule in engine.whitelist_rules:
            whitelist_rules.append(
                {
                    "pattern": rule.pattern,
                    "affected_count": rule.affected_count,
                    "affected_models": rule.affected_models[:10],
                }
            )

        # Get counts
        available, total = engine.get_available_count(models)

        # Build context
        context = {
            "window_type": "model_filter_gui",
            "current_provider": gui.current_provider,
            "models": {
                "total_count": total,
                "available_count": available,
                "items": model_items,
            },
            "rules": {
                "ignore": ignore_rules,
                "whitelist": whitelist_rules,
            },
            "ui_state": {
                "search_query": gui.search_entry.get()
                if hasattr(gui, "search_entry")
                else "",
                "has_unsaved_changes": engine.has_unsaved_changes(),
            },
            "available_providers": gui.available_providers or [],
        }

        # Add changes since last message if tracking
        if self._changes:
            context["changes_since_last_message"] = self._changes.copy()
            self._changes.clear()

        return context

    def get_window_system_prompt(self) -> str:
        """Get window-specific instructions for the AI."""
        return MODEL_FILTER_SYSTEM_PROMPT

    def get_tools(self) -> List[ToolDefinition]:
        """Get the list of tools available for this window."""
        # Collect all methods with @assistant_tool decorator
        tools = []
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, "_tool_definition"):
                tools.append(method._tool_definition)
        return tools

    def apply_state(self, state: Dict[str, Any]) -> None:
        """Restore the window to a given state (for checkpoint rollback)."""
        gui = self._gui
        engine = gui.filter_engine

        try:
            # Clear current rules
            engine.ignore_rules.clear()
            engine.whitelist_rules.clear()
            engine._invalidate_cache()

            # Restore ignore rules
            for rule_data in state.get("rules", {}).get("ignore", []):
                engine.add_ignore_rule(rule_data["pattern"])

            # Restore whitelist rules
            for rule_data in state.get("rules", {}).get("whitelist", []):
                engine.add_whitelist_rule(rule_data["pattern"])

            # Restore search query
            search_query = state.get("ui_state", {}).get("search_query", "")
            if hasattr(gui, "search_entry"):
                gui.search_entry.delete(0, "end")
                if search_query:
                    gui.search_entry.insert(0, search_query)

            # Trigger UI refresh
            gui._on_rules_changed()

            logger.info("State restored successfully")

        except Exception as e:
            logger.exception("Failed to apply state")
            raise

    def get_state_hash(self) -> str:
        """Get a quick hash of the current state for change detection."""
        engine = self._gui.filter_engine

        # Hash based on rules only (quick check)
        ignore_patterns = [r.pattern for r in engine.ignore_rules]
        whitelist_patterns = [r.pattern for r in engine.whitelist_rules]

        state_str = json.dumps(
            {
                "ignore": sorted(ignore_patterns),
                "whitelist": sorted(whitelist_patterns),
            }
        )
        return hashlib.md5(state_str.encode()).hexdigest()

    def lock_window(self) -> None:
        """Lock the window to prevent user interaction during AI execution."""
        gui = self._gui
        try:
            # Change cursor to indicate busy
            gui.configure(cursor="wait")

            # Disable interactive widgets
            self._set_widgets_state("disabled")

            logger.debug("Window locked")
        except Exception as e:
            logger.warning(f"Failed to lock window: {e}")

    def unlock_window(self) -> None:
        """Unlock the window after AI execution completes."""
        gui = self._gui
        try:
            # Restore cursor
            gui.configure(cursor="")

            # Re-enable widgets
            self._set_widgets_state("normal")

            logger.debug("Window unlocked")
        except Exception as e:
            logger.warning(f"Failed to unlock window: {e}")

    def _set_widgets_state(self, state: str) -> None:
        """Set the state of interactive widgets."""
        gui = self._gui

        # Disable/enable key widgets
        widgets_to_toggle = [
            "provider_combo",
            "search_entry",
            "refresh_btn",
            "help_btn",
        ]

        for widget_name in widgets_to_toggle:
            if hasattr(gui, widget_name):
                widget = getattr(gui, widget_name)
                try:
                    widget.configure(state=state)
                except Exception:
                    pass

        # Handle rule panels
        if hasattr(gui, "ignore_panel") and hasattr(gui.ignore_panel, "pattern_entry"):
            try:
                gui.ignore_panel.pattern_entry.configure(state=state)
            except Exception:
                pass

        if hasattr(gui, "whitelist_panel") and hasattr(
            gui.whitelist_panel, "pattern_entry"
        ):
            try:
                gui.whitelist_panel.pattern_entry.configure(state=state)
            except Exception:
                pass

    def on_ai_started(self) -> None:
        """Called when the AI starts processing a request."""
        self._tracking_changes = True
        self._changes.clear()
        self.lock_window()

    def on_ai_completed(self) -> None:
        """Called when the AI finishes processing."""
        self._tracking_changes = False
        self.unlock_window()

    def _record_change(self, change_type: str, **kwargs) -> None:
        """Record a change for context updates."""
        if self._tracking_changes:
            change = {
                "type": change_type,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }
            self._changes.append(change)

    # =========================================================================
    # Tool Implementations - Rule Management
    # =========================================================================

    @assistant_tool(
        name="add_ignore_rule",
        description="Add a pattern to the ignore list. Models matching this pattern will be blocked from the proxy.",
        parameters={
            "pattern": {
                "type": "string",
                "description": "The pattern to ignore. Supports * wildcard for prefix matching (e.g., 'gpt-4*' matches all gpt-4 models).",
            }
        },
        required=["pattern"],
        is_write=True,
    )
    def tool_add_ignore_rule(self, pattern: str) -> ToolResult:
        """Add an ignore rule."""
        gui = self._gui
        engine = gui.filter_engine
        models = gui.models or []

        pattern = pattern.strip()
        if not pattern:
            return ToolResult(
                success=False,
                message="Pattern cannot be empty",
                error_code="invalid_pattern",
            )

        # Check if already covered
        if engine.is_pattern_covered(pattern, "ignore"):
            covering_rules = [
                r.pattern
                for r in engine.ignore_rules
                if engine.pattern_is_covered_by(pattern, r.pattern)
            ]
            return ToolResult(
                success=False,
                message=f"Pattern '{pattern}' is already covered by existing rule(s): {covering_rules}",
                data={"covering_rules": covering_rules},
                error_code="pattern_covered",
            )

        # Check for patterns this would cover (smart merge)
        covered = engine.get_covered_patterns(pattern, "ignore")

        # Add the rule
        rule = engine.add_ignore_rule(pattern)
        if rule is None:
            return ToolResult(
                success=False,
                message=f"Pattern '{pattern}' already exists",
                error_code="duplicate_pattern",
            )

        # Remove covered patterns (smart merge)
        for covered_pattern in covered:
            engine.remove_ignore_rule(covered_pattern)

        # Update UI
        engine.update_affected_counts(models)
        gui._on_rules_changed()

        # Get affected models
        affected = engine.preview_pattern(pattern, "ignore", models)

        self._record_change("rule_added", rule_type="ignore", pattern=pattern)

        message = f"Added ignore rule: {pattern}. {len(affected)} model(s) now blocked."
        if covered:
            message += f" Removed {len(covered)} redundant rule(s): {covered}"

        return ToolResult(
            success=True,
            message=message,
            data={
                "pattern": pattern,
                "affected_count": len(affected),
                "affected_models": affected[:10],
                "removed_redundant": covered,
            },
        )

    @assistant_tool(
        name="remove_ignore_rule",
        description="Remove a pattern from the ignore list. Models previously blocked by this pattern will become available.",
        parameters={
            "pattern": {
                "type": "string",
                "description": "The exact pattern to remove from the ignore list.",
            }
        },
        required=["pattern"],
        is_write=True,
    )
    def tool_remove_ignore_rule(self, pattern: str) -> ToolResult:
        """Remove an ignore rule."""
        gui = self._gui
        engine = gui.filter_engine
        models = gui.models or []

        # Get affected models before removal
        affected = engine.preview_pattern(pattern, "ignore", models)

        if engine.remove_ignore_rule(pattern):
            gui._on_rules_changed()
            self._record_change("rule_removed", rule_type="ignore", pattern=pattern)

            return ToolResult(
                success=True,
                message=f"Removed ignore rule: {pattern}. {len(affected)} model(s) now available.",
                data={
                    "pattern": pattern,
                    "models_now_available": affected[:10],
                },
            )
        else:
            return ToolResult(
                success=False,
                message=f"Pattern '{pattern}' not found in ignore rules",
                error_code="pattern_not_found",
            )

    @assistant_tool(
        name="add_whitelist_rule",
        description="Add a pattern to the whitelist. Models matching this pattern will always be available, even if they match ignore rules.",
        parameters={
            "pattern": {
                "type": "string",
                "description": "The pattern to whitelist. Supports * wildcard for prefix matching.",
            }
        },
        required=["pattern"],
        is_write=True,
    )
    def tool_add_whitelist_rule(self, pattern: str) -> ToolResult:
        """Add a whitelist rule."""
        gui = self._gui
        engine = gui.filter_engine
        models = gui.models or []

        pattern = pattern.strip()
        if not pattern:
            return ToolResult(
                success=False,
                message="Pattern cannot be empty",
                error_code="invalid_pattern",
            )

        # Check if already covered
        if engine.is_pattern_covered(pattern, "whitelist"):
            covering_rules = [
                r.pattern
                for r in engine.whitelist_rules
                if engine.pattern_is_covered_by(pattern, r.pattern)
            ]
            return ToolResult(
                success=False,
                message=f"Pattern '{pattern}' is already covered by existing rule(s): {covering_rules}",
                data={"covering_rules": covering_rules},
                error_code="pattern_covered",
            )

        # Add the rule
        rule = engine.add_whitelist_rule(pattern)
        if rule is None:
            return ToolResult(
                success=False,
                message=f"Pattern '{pattern}' already exists",
                error_code="duplicate_pattern",
            )

        # Update UI
        engine.update_affected_counts(models)
        gui._on_rules_changed()

        affected = engine.preview_pattern(pattern, "whitelist", models)
        self._record_change("rule_added", rule_type="whitelist", pattern=pattern)

        return ToolResult(
            success=True,
            message=f"Added whitelist rule: {pattern}. {len(affected)} model(s) are now guaranteed available.",
            data={
                "pattern": pattern,
                "affected_count": len(affected),
                "affected_models": affected[:10],
            },
        )

    @assistant_tool(
        name="remove_whitelist_rule",
        description="Remove a pattern from the whitelist.",
        parameters={
            "pattern": {
                "type": "string",
                "description": "The exact pattern to remove from the whitelist.",
            }
        },
        required=["pattern"],
        is_write=True,
    )
    def tool_remove_whitelist_rule(self, pattern: str) -> ToolResult:
        """Remove a whitelist rule."""
        gui = self._gui
        engine = gui.filter_engine

        if engine.remove_whitelist_rule(pattern):
            gui._on_rules_changed()
            self._record_change("rule_removed", rule_type="whitelist", pattern=pattern)

            return ToolResult(
                success=True,
                message=f"Removed whitelist rule: {pattern}",
                data={"pattern": pattern},
            )
        else:
            return ToolResult(
                success=False,
                message=f"Pattern '{pattern}' not found in whitelist rules",
                error_code="pattern_not_found",
            )

    @assistant_tool(
        name="clear_all_ignore_rules",
        description="Remove all ignore rules. All models will become available (unless blocked by other means).",
        parameters={},
        required=[],
        is_write=True,
    )
    def tool_clear_all_ignore_rules(self) -> ToolResult:
        """Clear all ignore rules."""
        gui = self._gui
        engine = gui.filter_engine

        count = len(engine.ignore_rules)
        if count == 0:
            return ToolResult(
                success=True,
                message="No ignore rules to remove",
                data={"removed_count": 0},
            )

        patterns = [r.pattern for r in engine.ignore_rules]
        engine.ignore_rules.clear()
        engine._invalidate_cache()
        gui._on_rules_changed()

        self._record_change("rules_cleared", rule_type="ignore", count=count)

        return ToolResult(
            success=True,
            message=f"Removed all {count} ignore rule(s)",
            data={
                "removed_count": count,
                "removed_patterns": patterns,
            },
        )

    @assistant_tool(
        name="clear_all_whitelist_rules",
        description="Remove all whitelist rules.",
        parameters={},
        required=[],
        is_write=True,
    )
    def tool_clear_all_whitelist_rules(self) -> ToolResult:
        """Clear all whitelist rules."""
        gui = self._gui
        engine = gui.filter_engine

        count = len(engine.whitelist_rules)
        if count == 0:
            return ToolResult(
                success=True,
                message="No whitelist rules to remove",
                data={"removed_count": 0},
            )

        patterns = [r.pattern for r in engine.whitelist_rules]
        engine.whitelist_rules.clear()
        engine._invalidate_cache()
        gui._on_rules_changed()

        self._record_change("rules_cleared", rule_type="whitelist", count=count)

        return ToolResult(
            success=True,
            message=f"Removed all {count} whitelist rule(s)",
            data={
                "removed_count": count,
                "removed_patterns": patterns,
            },
        )

    # =========================================================================
    # Tool Implementations - Query
    # =========================================================================

    @assistant_tool(
        name="get_models_matching_pattern",
        description="Preview which models would be affected by a pattern without adding it as a rule. Useful for testing patterns before applying them.",
        parameters={
            "pattern": {
                "type": "string",
                "description": "The pattern to test. Supports * wildcard.",
            }
        },
        required=["pattern"],
        is_write=False,
    )
    def tool_get_models_matching_pattern(self, pattern: str) -> ToolResult:
        """Get models matching a pattern."""
        gui = self._gui
        engine = gui.filter_engine
        models = gui.models or []

        pattern = pattern.strip()
        if not pattern:
            return ToolResult(
                success=False,
                message="Pattern cannot be empty",
                error_code="invalid_pattern",
            )

        matches = engine.preview_pattern(pattern, "ignore", models)

        if not matches:
            # Suggest using wildcards
            hint = ""
            if "*" not in pattern:
                hint = " Hint: Use wildcards like '*preview*' to match models containing 'preview'."
            return ToolResult(
                success=True,
                message=f"No models match pattern '{pattern}'.{hint}",
                data={
                    "pattern": pattern,
                    "match_count": 0,
                    "matches": [],
                },
            )

        return ToolResult(
            success=True,
            message=f"Pattern '{pattern}' matches {len(matches)} model(s)",
            data={
                "pattern": pattern,
                "match_count": len(matches),
                "matches": matches,
            },
        )

    @assistant_tool(
        name="get_model_details",
        description="Get detailed information about a specific model, including its current status and any rules affecting it.",
        parameters={
            "model_id": {
                "type": "string",
                "description": "The model ID to look up (e.g., 'gpt-4' or 'openai/gpt-4').",
            }
        },
        required=["model_id"],
        is_write=False,
    )
    def tool_get_model_details(self, model_id: str) -> ToolResult:
        """Get details about a specific model."""
        gui = self._gui
        engine = gui.filter_engine
        models = gui.models or []

        # Try to find the model
        found_model = None
        for m in models:
            if m == model_id or m.endswith(f"/{model_id}"):
                found_model = m
                break

        if not found_model:
            return ToolResult(
                success=False,
                message=f"Model '{model_id}' not found in current provider",
                error_code="model_not_found",
            )

        status = engine.get_model_status(found_model)

        data = {
            "model_id": found_model,
            "display_name": status.display_name,
            "status": status.status,
            "color": status.color,
        }

        if status.affecting_rule:
            data["affecting_rule"] = {
                "pattern": status.affecting_rule.pattern,
                "type": status.affecting_rule.rule_type,
            }

        status_text = {
            "normal": "available (no rules affecting it)",
            "ignored": f"blocked by ignore rule '{status.affecting_rule.pattern}'"
            if status.affecting_rule
            else "blocked",
            "whitelisted": f"whitelisted by rule '{status.affecting_rule.pattern}'"
            if status.affecting_rule
            else "whitelisted",
        }

        return ToolResult(
            success=True,
            message=f"Model '{status.display_name}' is {status_text.get(status.status, status.status)}",
            data=data,
        )

    @assistant_tool(
        name="explain_model_status",
        description="Explain why a model has its current status (normal, ignored, or whitelisted). Shows the rule priority and which rules are affecting it.",
        parameters={
            "model_id": {"type": "string", "description": "The model ID to explain."}
        },
        required=["model_id"],
        is_write=False,
    )
    def tool_explain_model_status(self, model_id: str) -> ToolResult:
        """Explain why a model has its status."""
        gui = self._gui
        engine = gui.filter_engine
        models = gui.models or []

        # Find the model
        found_model = None
        for m in models:
            if m == model_id or m.endswith(f"/{model_id}"):
                found_model = m
                break

        if not found_model:
            return ToolResult(
                success=False,
                message=f"Model '{model_id}' not found",
                error_code="model_not_found",
            )

        # Check all matching rules
        matching_ignore = []
        matching_whitelist = []

        for rule in engine.ignore_rules:
            if engine._pattern_matches(found_model, rule.pattern):
                matching_ignore.append(rule.pattern)

        for rule in engine.whitelist_rules:
            if engine._pattern_matches(found_model, rule.pattern):
                matching_whitelist.append(rule.pattern)

        status = engine.get_model_status(found_model)

        explanation = []
        explanation.append(f"Model: {status.display_name}")
        explanation.append(f"Current status: {status.status.upper()}")
        explanation.append("")
        explanation.append("Rule priority: Whitelist > Ignore > Normal")
        explanation.append("")

        if matching_whitelist:
            explanation.append(f"Matching WHITELIST rules: {matching_whitelist}")
        else:
            explanation.append("No matching whitelist rules")

        if matching_ignore:
            explanation.append(f"Matching IGNORE rules: {matching_ignore}")
        else:
            explanation.append("No matching ignore rules")

        explanation.append("")

        if status.status == "whitelisted":
            explanation.append(
                f"Result: Model is AVAILABLE because whitelist rule '{status.affecting_rule.pattern}' takes priority"
            )
        elif status.status == "ignored":
            explanation.append(
                f"Result: Model is BLOCKED because ignore rule '{status.affecting_rule.pattern}' matches and no whitelist overrides it"
            )
        else:
            explanation.append(
                "Result: Model is AVAILABLE by default (no rules affect it)"
            )

        return ToolResult(
            success=True,
            message="\n".join(explanation),
            data={
                "model_id": found_model,
                "status": status.status,
                "matching_ignore_rules": matching_ignore,
                "matching_whitelist_rules": matching_whitelist,
                "affecting_rule": status.affecting_rule.pattern
                if status.affecting_rule
                else None,
            },
        )

    # =========================================================================
    # Tool Implementations - Provider
    # =========================================================================

    @assistant_tool(
        name="switch_provider",
        description="Switch to a different provider to view and configure its model filters.",
        parameters={
            "provider": {
                "type": "string",
                "description": "The provider name to switch to (e.g., 'openai', 'gemini', 'anthropic').",
            }
        },
        required=["provider"],
        is_write=False,
    )
    def tool_switch_provider(self, provider: str) -> ToolResult:
        """Switch to a different provider."""
        gui = self._gui
        available = gui.available_providers or []

        provider = provider.lower().strip()

        if provider not in available:
            return ToolResult(
                success=False,
                message=f"Provider '{provider}' is not available. Available providers: {available}",
                data={"available_providers": available},
                error_code="provider_not_found",
            )

        if provider == gui.current_provider:
            return ToolResult(
                success=True,
                message=f"Already on provider '{provider}'",
                data={"provider": provider},
            )

        # Switch provider via the combo box
        if hasattr(gui, "provider_combo"):
            gui.provider_combo.set(provider)
            gui._on_provider_changed(provider)

        self._record_change(
            "provider_changed",
            from_provider=gui.current_provider,
            to_provider=provider,
        )

        return ToolResult(
            success=True,
            message=f"Switched to provider '{provider}'",
            data={"provider": provider},
        )

    @assistant_tool(
        name="refresh_models",
        description="Refresh the model list from the current provider. Use this if models seem outdated or missing.",
        parameters={},
        required=[],
        is_write=False,
    )
    def tool_refresh_models(self) -> ToolResult:
        """Refresh the model list."""
        gui = self._gui

        # Trigger refresh
        gui._refresh_models()

        self._record_change("models_refreshed", provider=gui.current_provider)

        return ToolResult(
            success=True,
            message=f"Refreshing models for provider '{gui.current_provider}'... The model list will update shortly.",
            data={"provider": gui.current_provider},
        )

    # =========================================================================
    # Tool Implementations - State
    # =========================================================================

    @assistant_tool(
        name="save_changes",
        description="Save the current rules to the .env file. Changes will persist across restarts.",
        parameters={},
        required=[],
        is_write=True,
    )
    def tool_save_changes(self) -> ToolResult:
        """Save changes to .env file."""
        gui = self._gui
        engine = gui.filter_engine

        if not engine.has_unsaved_changes():
            return ToolResult(
                success=True,
                message="No unsaved changes to save",
                data={"saved": False},
            )

        if engine.save_to_env(gui.current_provider):
            gui._update_status()
            self._record_change("changes_saved")

            return ToolResult(
                success=True,
                message=f"Saved rules for provider '{gui.current_provider}' to .env file",
                data={
                    "saved": True,
                    "provider": gui.current_provider,
                    "ignore_rules": [r.pattern for r in engine.ignore_rules],
                    "whitelist_rules": [r.pattern for r in engine.whitelist_rules],
                },
            )
        else:
            return ToolResult(
                success=False,
                message="Failed to save changes to .env file",
                error_code="save_failed",
            )

    @assistant_tool(
        name="discard_changes",
        description="Discard all unsaved changes and reload rules from the .env file.",
        parameters={},
        required=[],
        is_write=True,
    )
    def tool_discard_changes(self) -> ToolResult:
        """Discard unsaved changes."""
        gui = self._gui
        engine = gui.filter_engine

        if not engine.has_unsaved_changes():
            return ToolResult(
                success=True,
                message="No unsaved changes to discard",
                data={"discarded": False},
            )

        engine.discard_changes()
        gui._on_rules_changed()

        self._record_change("changes_discarded")

        return ToolResult(
            success=True,
            message="Discarded unsaved changes and reloaded rules from .env",
            data={
                "discarded": True,
                "ignore_rules": [r.pattern for r in engine.ignore_rules],
                "whitelist_rules": [r.pattern for r in engine.whitelist_rules],
            },
        )

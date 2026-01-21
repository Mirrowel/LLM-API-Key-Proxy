# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/utilities/gemini_tool_handler.py
"""
Shared tool handling mixin for Gemini-based providers.

Provides tool schema transformation, response grouping, and tool choice translation
methods used by both GeminiCliProvider and AntigravityProvider.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .gemini_shared_utils import GEMINI3_TOOL_RENAMES

lib_logger = logging.getLogger("rotator_library")


class GeminiToolHandler:
    """
    Mixin providing tool schema transformation and response grouping for Gemini-based providers.

    Provides shared methods for:
    - Tool response grouping (fixing ID mismatches)
    - Type hint formatting for tool descriptions
    - Tool choice translation (OpenAI → Gemini)
    - Strict schema enforcement for Gemini 3

    Providers must define these attributes:
    - _gemini3_tool_prefix: str - Namespace prefix for Gemini 3 tools
    - _enable_gemini3_tool_fix: bool - Whether to apply Gemini 3 fixes

    Providers must implement:
    - _is_gemini_3(model: str) -> bool - Check if model is Gemini 3
    """

    # Class attributes - should be overridden by providers
    _gemini3_tool_prefix: str = "gemini3_"
    _enable_gemini3_tool_fix: bool = True

    def _is_gemini_3(self, model: str) -> bool:
        """Check if model is Gemini 3. Must be implemented by provider."""
        raise NotImplementedError("Subclass must implement _is_gemini_3")

    def _strip_gemini3_prefix(self, name: str) -> str:
        """
        Strip the Gemini 3 namespace prefix from a tool name.

        Also reverses any tool renames that were applied to avoid Gemini conflicts.

        Args:
            name: Tool name that may have a prefix

        Returns:
            Original tool name without prefix
        """
        from .gemini_shared_utils import GEMINI3_TOOL_RENAMES_REVERSE

        if name and name.startswith(self._gemini3_tool_prefix):
            stripped = name[len(self._gemini3_tool_prefix) :]
            # Reverse any renames
            return GEMINI3_TOOL_RENAMES_REVERSE.get(stripped, stripped)
        return name

    def _fix_tool_response_grouping(
        self, contents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group function calls with their responses for Gemini CLI compatibility.

        Converts linear format (call, response, call, response)
        to grouped format (model with calls, user with all responses).

        IMPORTANT: Preserves ID-based pairing to prevent mismatches.
        When IDs don't match, attempts recovery by:
        1. Matching by function name first
        2. Matching by order if names don't match
        3. Inserting placeholder responses if responses are missing
        4. Inserting responses at the CORRECT position (after their corresponding call)

        Args:
            contents: List of Gemini-format messages

        Returns:
            Reorganized messages with proper call/response grouping
        """
        new_contents = []
        # Each pending group tracks:
        # - ids: expected response IDs
        # - func_names: expected function names (for orphan matching)
        # - insert_after_idx: position in new_contents where model message was added
        pending_groups = []
        collected_responses = {}  # Dict mapping ID -> response_part

        for content in contents:
            role = content.get("role")
            parts = content.get("parts", [])

            response_parts = [p for p in parts if "functionResponse" in p]

            if response_parts:
                # Collect responses by ID (ignore duplicates - keep first occurrence)
                for resp in response_parts:
                    resp_id = resp.get("functionResponse", {}).get("id", "")
                    if resp_id:
                        if resp_id in collected_responses:
                            lib_logger.warning(
                                f"[Grouping] Duplicate response ID detected: {resp_id}. "
                                f"Ignoring duplicate - this may indicate malformed conversation history."
                            )
                            continue
                        collected_responses[resp_id] = resp

                # Try to satisfy pending groups (newest first)
                for i in range(len(pending_groups) - 1, -1, -1):
                    group = pending_groups[i]
                    group_ids = group["ids"]

                    # Check if we have ALL responses for this group
                    if all(gid in collected_responses for gid in group_ids):
                        # Extract responses in the same order as the function calls
                        group_responses = [
                            collected_responses.pop(gid) for gid in group_ids
                        ]
                        new_contents.append({"parts": group_responses, "role": "user"})
                        pending_groups.pop(i)
                        break
                continue

            if role == "model":
                func_calls = [p for p in parts if "functionCall" in p]
                new_contents.append(content)
                if func_calls:
                    call_ids = [
                        fc.get("functionCall", {}).get("id", "") for fc in func_calls
                    ]
                    call_ids = [cid for cid in call_ids if cid]  # Filter empty IDs

                    # Also extract function names for orphan matching
                    func_names = [
                        fc.get("functionCall", {}).get("name", "") for fc in func_calls
                    ]

                    if call_ids:
                        pending_groups.append(
                            {
                                "ids": call_ids,
                                "func_names": func_names,
                                "insert_after_idx": len(new_contents) - 1,
                            }
                        )
            else:
                new_contents.append(content)

        # Handle remaining groups (shouldn't happen in well-formed conversations)
        # Attempt recovery by matching orphans to unsatisfied calls
        # Process in REVERSE order of insert_after_idx so insertions don't shift indices
        pending_groups.sort(key=lambda g: g["insert_after_idx"], reverse=True)

        for group in pending_groups:
            group_ids = group["ids"]
            group_func_names = group.get("func_names", [])
            insert_idx = group["insert_after_idx"] + 1
            group_responses = []

            lib_logger.debug(
                f"[Grouping Recovery] Processing unsatisfied group: "
                f"ids={group_ids}, names={group_func_names}, insert_at={insert_idx}"
            )

            for i, expected_id in enumerate(group_ids):
                expected_name = group_func_names[i] if i < len(group_func_names) else ""

                if expected_id in collected_responses:
                    # Direct ID match
                    group_responses.append(collected_responses.pop(expected_id))
                    lib_logger.debug(
                        f"[Grouping Recovery] Direct ID match for '{expected_id}'"
                    )
                elif collected_responses:
                    # Try to find orphan with matching function name first
                    matched_orphan_id = None

                    # First pass: match by function name
                    for orphan_id, orphan_resp in collected_responses.items():
                        orphan_name = orphan_resp.get("functionResponse", {}).get(
                            "name", ""
                        )
                        # Match if names are equal
                        if orphan_name == expected_name:
                            matched_orphan_id = orphan_id
                            lib_logger.debug(
                                f"[Grouping Recovery] Matched orphan '{orphan_id}' by name '{orphan_name}'"
                            )
                            break

                    # Second pass: if no name match, try "unknown_function" orphans
                    if not matched_orphan_id:
                        for orphan_id, orphan_resp in collected_responses.items():
                            orphan_name = orphan_resp.get("functionResponse", {}).get(
                                "name", ""
                            )
                            if orphan_name == "unknown_function":
                                matched_orphan_id = orphan_id
                                lib_logger.debug(
                                    f"[Grouping Recovery] Matched unknown_function orphan '{orphan_id}' "
                                    f"to expected '{expected_name}'"
                                )
                                break

                    # Third pass: if still no match, take first available (order-based)
                    if not matched_orphan_id:
                        matched_orphan_id = next(iter(collected_responses))
                        lib_logger.debug(
                            f"[Grouping Recovery] No name match, using first available orphan '{matched_orphan_id}'"
                        )

                    if matched_orphan_id:
                        orphan_resp = collected_responses.pop(matched_orphan_id)

                        # Fix the ID in the response to match the call
                        old_id = orphan_resp["functionResponse"].get("id", "")
                        orphan_resp["functionResponse"]["id"] = expected_id

                        # Fix the name if it was "unknown_function"
                        if (
                            orphan_resp["functionResponse"].get("name")
                            == "unknown_function"
                            and expected_name
                        ):
                            orphan_resp["functionResponse"]["name"] = expected_name
                            lib_logger.info(
                                f"[Grouping Recovery] Fixed function name from 'unknown_function' to '{expected_name}'"
                            )

                        lib_logger.warning(
                            f"[Grouping] Auto-repaired ID mismatch: mapped response '{old_id}' "
                            f"to call '{expected_id}' (function: {expected_name})"
                        )
                        group_responses.append(orphan_resp)
                else:
                    # No responses available - create placeholder
                    placeholder_resp = {
                        "functionResponse": {
                            "name": expected_name or "unknown_function",
                            "response": {
                                "result": {
                                    "error": "Tool response was lost during context processing. "
                                    "This is a recovered placeholder.",
                                    "recovered": True,
                                }
                            },
                            "id": expected_id,
                        }
                    }
                    lib_logger.warning(
                        f"[Grouping Recovery] Created placeholder response for missing tool: "
                        f"id='{expected_id}', name='{expected_name}'"
                    )
                    group_responses.append(placeholder_resp)

            if group_responses:
                # Insert at the correct position (right after the model message with the calls)
                new_contents.insert(
                    insert_idx, {"parts": group_responses, "role": "user"}
                )
                lib_logger.info(
                    f"[Grouping Recovery] Inserted {len(group_responses)} responses at position {insert_idx} "
                    f"(expected {len(group_ids)})"
                )

        # Warn about unmatched responses
        if collected_responses:
            lib_logger.warning(
                f"[Grouping] {len(collected_responses)} unmatched responses remaining: "
                f"ids={list(collected_responses.keys())}"
            )

        return new_contents

    def _format_type_hint(self, prop_data: Dict[str, Any], depth: int = 0) -> str:
        """
        Format a detailed type hint for a property schema.

        Generates human-readable type descriptions for tool parameter documentation.
        Handles enums, const values, arrays, and nested objects.

        Args:
            prop_data: Property schema definition
            depth: Current recursion depth (limits nested formatting)

        Returns:
            Human-readable type hint string
        """
        type_hint = prop_data.get("type", "unknown")

        # Handle enum values - show allowed options
        if "enum" in prop_data:
            enum_vals = prop_data["enum"]
            if len(enum_vals) <= 5:
                return f"string ENUM[{', '.join(repr(v) for v in enum_vals)}]"
            return f"string ENUM[{len(enum_vals)} options]"

        # Handle const values
        if "const" in prop_data:
            return f"string CONST={repr(prop_data['const'])}"

        if type_hint == "array":
            items = prop_data.get("items", {})
            if isinstance(items, dict):
                item_type = items.get("type", "unknown")
                if item_type == "object":
                    nested_props = items.get("properties", {})
                    nested_req = items.get("required", [])
                    if nested_props:
                        nested_list = []
                        for n, d in nested_props.items():
                            if isinstance(d, dict):
                                # Recursively format nested types (limit depth)
                                if depth < 1:
                                    t = self._format_type_hint(d, depth + 1)
                                else:
                                    t = d.get("type", "unknown")
                                req = " REQUIRED" if n in nested_req else ""
                                nested_list.append(f"{n}: {t}{req}")
                        return f"ARRAY_OF_OBJECTS[{', '.join(nested_list)}]"
                    return "ARRAY_OF_OBJECTS"
                return f"ARRAY_OF_{item_type.upper()}"
            return "ARRAY"

        if type_hint == "object":
            nested_props = prop_data.get("properties", {})
            nested_req = prop_data.get("required", [])
            if nested_props and depth < 1:
                nested_list = []
                for n, d in nested_props.items():
                    if isinstance(d, dict):
                        t = d.get("type", "unknown")
                        req = " REQUIRED" if n in nested_req else ""
                        nested_list.append(f"{n}: {t}{req}")
                return f"object{{{', '.join(nested_list)}}}"

        return type_hint

    def _translate_tool_choice(
        self, tool_choice: Union[str, Dict[str, Any]], model: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Translate OpenAI's `tool_choice` to Gemini's `toolConfig`.

        Handles Gemini 3 namespace prefixes for specific tool selection.

        Args:
            tool_choice: OpenAI tool_choice value ("auto", "none", "required", or function spec)
            model: Model name (used for Gemini 3 prefix logic)

        Returns:
            Gemini toolConfig dict, or None if no translation needed
        """
        if not tool_choice:
            return None

        config = {}
        mode = "AUTO"  # Default to auto
        is_gemini_3 = self._is_gemini_3(model)

        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                mode = "AUTO"
            elif tool_choice == "none":
                mode = "NONE"
            elif tool_choice == "required":
                mode = "ANY"
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function_name = tool_choice.get("function", {}).get("name")
            if function_name:
                # Add Gemini 3 prefix if needed (and rename problematic tools)
                if is_gemini_3 and self._enable_gemini3_tool_fix:
                    function_name = GEMINI3_TOOL_RENAMES.get(
                        function_name, function_name
                    )
                    function_name = f"{self._gemini3_tool_prefix}{function_name}"

                mode = "ANY"  # Force a call, but only to this function
                config["functionCallingConfig"] = {
                    "mode": mode,
                    "allowedFunctionNames": [function_name],
                }
                return config

        config["functionCallingConfig"] = {"mode": mode}
        return config

    def _enforce_strict_schema(self, schema: Any) -> Any:
        """
        Enforce strict JSON schema for Gemini 3 to prevent hallucinated parameters.

        Adds 'additionalProperties: false' to object schemas with 'properties',
        which tells the model it CANNOT add properties not in the schema.

        IMPORTANT: Preserves 'additionalProperties: true' (or {}) when explicitly
        set in the original schema. This is critical for "freeform" parameter objects
        like batch/multi_tool's nested parameters which need to accept arbitrary
        tool parameters that aren't pre-defined in the schema.

        Args:
            schema: JSON schema to enforce strictness on

        Returns:
            Schema with additionalProperties: false added where appropriate
        """
        if not isinstance(schema, dict):
            return schema

        result = {}
        preserved_additional_props = None

        for key, value in schema.items():
            # Preserve additionalProperties as-is if it's truthy
            # This is critical for "freeform" parameter objects like batch's
            # nested parameters which need to accept arbitrary tool parameters
            if key == "additionalProperties":
                if value is not False:
                    # Preserve the original value (true, {}, {"type": "string"}, etc.)
                    preserved_additional_props = value
                continue
            if isinstance(value, dict):
                result[key] = self._enforce_strict_schema(value)
            elif isinstance(value, list):
                result[key] = [
                    self._enforce_strict_schema(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        # Add additionalProperties: false to object schemas with properties,
        # BUT only if we didn't preserve a value from the original schema
        if result.get("type") == "object" and "properties" in result:
            if preserved_additional_props is not None:
                result["additionalProperties"] = preserved_additional_props
            else:
                result["additionalProperties"] = False

        return result

    def _inject_signature_into_description(
        self, func_decl: Dict[str, Any], description_prompt: str
    ) -> Dict[str, Any]:
        """
        Inject parameter signatures into tool description.

        Appends a structured parameter signature to the tool's description
        to help Gemini 3 use the correct parameter names.

        Args:
            func_decl: Function declaration dict with name, description, and schema
                      (either 'parametersJsonSchema' or 'parameters' key)
            description_prompt: Template string with {params} placeholder

        Returns:
            Modified function declaration with signature appended to description
        """
        # Support both parametersJsonSchema and parameters keys
        schema = func_decl.get("parametersJsonSchema") or func_decl.get(
            "parameters", {}
        )
        if not schema:
            return func_decl

        required = schema.get("required", [])
        properties = schema.get("properties", {})

        if not properties:
            return func_decl

        param_list = []
        for prop_name, prop_data in properties.items():
            if not isinstance(prop_data, dict):
                continue

            type_hint = self._format_type_hint(prop_data)
            is_required = prop_name in required
            param_list.append(
                f"{prop_name} ({type_hint}{', REQUIRED' if is_required else ''})"
            )

        if param_list:
            sig_str = description_prompt.replace("{params}", ", ".join(param_list))
            func_decl["description"] = func_decl.get("description", "") + sig_str

        return func_decl

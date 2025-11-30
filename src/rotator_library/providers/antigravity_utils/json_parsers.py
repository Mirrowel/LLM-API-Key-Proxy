# src/rotator_library/providers/antigravity_utils/json_parsers.py
"""
JSON parsing utilities for Antigravity API responses.

Provides functions to handle JSON-stringified values and malformed JSON
that can appear in Antigravity API responses.
"""

import json
import logging
from typing import Any

lib_logger = logging.getLogger('rotator_library')


def recursively_parse_json_strings(obj: Any) -> Any:
    """
    Recursively parse JSON strings in nested data structures.
    
    Antigravity sometimes returns tool arguments with JSON-stringified values:
    {"files": "[{...}]"} instead of {"files": [{...}]}.
    
    Additionally handles:
    - Malformed double-encoded JSON (extra trailing '}' or ']')
    - Escaped string content (\\n, \\t, \\", etc.)
    
    Args:
        obj: Data structure to parse (dict, list, str, or primitive)
        
    Returns:
        Parsed data structure with JSON strings converted to objects
        
    Example:
        >>> obj = {"files": '[{"path": "test.py"}]', "count": "5"}
        >>> recursively_parse_json_strings(obj)
        {"files": [{"path": "test.py"}], "count": "5"}
        
        >>> obj = {"content": "Line 1\\nLine 2"}
        >>> recursively_parse_json_strings(obj)
        {"content": "Line 1\nLine 2"}
    """
    if isinstance(obj, dict):
        return {k: recursively_parse_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursively_parse_json_strings(item) for item in obj]
    elif isinstance(obj, str):
        stripped = obj.strip()
        
        # Check if it looks like JSON (starts with { or [)
        if stripped and stripped[0] in ('{', '['):
            # Try standard parsing first
            if (stripped.startswith('{') and stripped.endswith('}')) or \
               (stripped.startswith('[') and stripped.endswith(']')):
                try:
                    parsed = json.loads(obj)
                    return recursively_parse_json_strings(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Handle malformed JSON: array that doesn't end with ]
            # e.g., '[{"path": "..."}]}' instead of '[{"path": "..."}]'
            if stripped.startswith('[') and not stripped.endswith(']'):
                try:
                    # Find the last ] and truncate there
                    last_bracket = stripped.rfind(']')
                    if last_bracket > 0:
                        cleaned = stripped[:last_bracket+1]
                        parsed = json.loads(cleaned)
                        lib_logger.warning(
                            f"[Antigravity] Auto-corrected malformed JSON string: "
                            f"truncated {len(stripped) - len(cleaned)} extra chars"
                        )
                        return recursively_parse_json_strings(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Handle malformed JSON: object that doesn't end with }
            if stripped.startswith('{') and not stripped.endswith('}'):
                try:
                    # Find the last } and truncate there
                    last_brace = stripped.rfind('}')
                    if last_brace > 0:
                        cleaned = stripped[:last_brace+1]
                        parsed = json.loads(cleaned)
                        lib_logger.warning(
                            f"[Antigravity] Auto-corrected malformed JSON string: "
                            f"truncated {len(stripped) - len(cleaned)} extra chars"
                        )
                        return recursively_parse_json_strings(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
        
        # For non-JSON strings, check if they contain escape sequences that need unescaping
        # This handles cases where diff content or other text has literal \n instead of newlines
        if '\\n' in obj or '\\t' in obj or '\\"' in obj or '\\\\' in obj:
            try:
                # Use json.loads with quotes to properly unescape the string
                # This converts \n -> newline, \t -> tab, \" -> quote, etc.
                unescaped = json.loads(f'"{obj}"')
                lib_logger.debug(
                    f"[Antigravity] Unescaped string content: "
                    f"{len(obj) - len(unescaped)} chars changed"
                )
                return unescaped
            except (json.JSONDecodeError, ValueError):
                # If unescaping fails, continue with original processing
                pass
    return obj
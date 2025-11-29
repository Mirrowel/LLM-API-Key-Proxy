# src/rotator_library/providers/antigravity_utils/schema_transformers.py
"""
Schema transformation utilities for Antigravity API.

Provides functions to normalize and clean JSON schemas for compatibility
with Proto-based Antigravity API and Claude models.
"""

from typing import Any


def normalize_type_arrays(schema: Any) -> Any:
    """
    Normalize type arrays in JSON Schema for Proto-based Antigravity API.
    
    Converts `"type": ["string", "null"]` → `"type": "string"` by removing
    null from type arrays, as Proto doesn't support nullable type arrays.
    
    Args:
        schema: JSON schema to normalize (dict, list, or primitive)
        
    Returns:
        Normalized schema with type arrays simplified
        
    Example:
        >>> schema = {"type": ["string", "null"], "description": "Name"}
        >>> normalize_type_arrays(schema)
        {"type": "string", "description": "Name"}
    """
    if isinstance(schema, dict):
        normalized = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, list):
                # Remove "null" from type array and take first non-null type
                non_null = [t for t in value if t != "null"]
                normalized[key] = non_null[0] if non_null else value[0]
            else:
                normalized[key] = normalize_type_arrays(value)
        return normalized
    elif isinstance(schema, list):
        return [normalize_type_arrays(item) for item in schema]
    return schema


def clean_claude_schema(schema: Any) -> Any:
    """
    Recursively clean JSON Schema for Antigravity/Google's Proto-based API.
    
    Removes unsupported fields and converts certain constructs to supported equivalents:
    - Removes: $schema, additionalProperties, minItems, maxItems, pattern, etc.
    - Converts: 'const' to 'enum' with single value
    - Converts: 'anyOf'/'oneOf' to first option (Claude doesn't support these)
    
    Args:
        schema: JSON schema to clean (dict, list, or primitive)
        
    Returns:
        Cleaned schema compatible with Proto-based API
        
    Note:
        Claude via Antigravity rejects JSON Schema draft 2020-12 validation keywords.
        This function strips them to prevent API errors.
        
    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {"name": {"type": "string", "minLength": 1}},
        ...     "additionalProperties": False
        ... }
        >>> clean_claude_schema(schema)
        {"type": "object", "properties": {"name": {"type": "string"}}}
    """
    if not isinstance(schema, dict):
        return schema
    
    # Fields not supported by Antigravity/Google's Proto-based API
    # Note: Claude via Antigravity rejects JSON Schema draft 2020-12 validation keywords
    incompatible = {
        '$schema', 'additionalProperties', 'minItems', 'maxItems', 'pattern',
        'minLength', 'maxLength', 'minimum', 'maximum', 'default',
        'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf', 'format',
        'minProperties', 'maxProperties', 'uniqueItems', 'contentEncoding',
        'contentMediaType', 'contentSchema', 'deprecated', 'readOnly', 'writeOnly',
        'examples', '$id', '$ref', '$defs', 'definitions', 'title',
    }
    
    # Handle 'anyOf' by taking the first option (Claude doesn't support anyOf)
    if 'anyOf' in schema and isinstance(schema['anyOf'], list) and schema['anyOf']:
        first_option = clean_claude_schema(schema['anyOf'][0])
        if isinstance(first_option, dict):
            return first_option
    
    # Handle 'oneOf' similarly
    if 'oneOf' in schema and isinstance(schema['oneOf'], list) and schema['oneOf']:
        first_option = clean_claude_schema(schema['oneOf'][0])
        if isinstance(first_option, dict):
            return first_option
    

    cleaned = {}
    
    # Handle 'const' by converting to 'enum' with single value
    if 'const' in schema:
        const_value = schema['const']
        cleaned['enum'] = [const_value]
    
    for key, value in schema.items():
        if key in incompatible or key == 'const':
            continue
        if isinstance(value, dict):
            cleaned[key] = clean_claude_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_claude_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    
    return cleaned
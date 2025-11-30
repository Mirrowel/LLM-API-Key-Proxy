# src/rotator_library/providers/antigravity_validators.py
"""
Validation utilities for Antigravity provider.

Provides input validation functions to ensure data integrity
and provide clear error messages for public methods.
"""

import logging
from typing import Any, Dict, List, Optional, Union

lib_logger = logging.getLogger('rotator_library')


def validate_completion_params(
    model: str,
    messages: List[Dict[str, Any]],
    **kwargs
) -> None:
    """
    Validate completion request parameters.
    
    Args:
        model: Model name to validate
        messages: List of message dictionaries
        **kwargs: Additional parameters to validate
        
    Raises:
        ValueError: If model or messages are invalid
        TypeError: If parameter types are incorrect
    """
    # Validate model
    if not model:
        raise ValueError("model parameter is required and cannot be empty")
    
    if not isinstance(model, str):
        raise TypeError(f"model must be a string, got {type(model).__name__}")
    
    # Note: Model validation is done at runtime by the provider
    # as supported models may be dynamically discovered
    
    # Validate messages
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")
    
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")
    
    # Validate each message
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"Message at index {i} must be a dictionary, got {type(message).__name__}")
        
        # Check for required fields
        if 'role' not in message:
            raise ValueError(f"Message at index {i} missing required 'role' field")
        
        if 'content' not in message:
            raise ValueError(f"Message at index {i} missing required 'content' field")
        
        # Validate role
        valid_roles = {'user', 'assistant', 'system'}
        if message['role'] not in valid_roles:
            raise ValueError(
                f"Message at index {i} has invalid role '{message['role']}'. "
                f"Must be one of: {', '.join(valid_roles)}"
            )
        
        # Validate content
        if not isinstance(message['content'], str):
            raise TypeError(f"Message content must be a string, got {type(message['content']).__name__}")
    
    # Validate numeric parameters
    if 'temperature' in kwargs and kwargs['temperature'] is not None:
        temp = kwargs['temperature']
        if not isinstance(temp, (int, float)):
            raise TypeError(f"temperature must be a number, got {type(temp).__name__}")
        if not (0.0 <= temp <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
    
    if 'top_p' in kwargs and kwargs['top_p'] is not None:
        top_p = kwargs['top_p']
        if not isinstance(top_p, (int, float)):
            raise TypeError(f"top_p must be a number, got {type(top_p).__name__}")
        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
    
    if 'max_tokens' in kwargs and kwargs['max_tokens'] is not None:
        max_tokens = kwargs['max_tokens']
        if not isinstance(max_tokens, int):
            raise TypeError(f"max_tokens must be an integer, got {type(max_tokens).__name__}")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
    
    if 'max_output_tokens' in kwargs and kwargs['max_output_tokens'] is not None:
        max_output_tokens = kwargs['max_output_tokens']
        if not isinstance(max_output_tokens, int):
            raise TypeError(f"max_output_tokens must be an integer, got {type(max_output_tokens).__name__}")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than 0")


def validate_models_params(api_key: str) -> None:
    """
    Validate get_models method parameters.
    
    Args:
        api_key: API key to validate
        
    Raises:
        ValueError: If api_key is invalid
        TypeError: If parameter types are incorrect
    """
    if not api_key:
        raise ValueError("api_key parameter is required and cannot be empty")
    
    if not isinstance(api_key, str):
        raise TypeError(f"api_key must be a string, got {type(api_key).__name__}")


def validate_count_tokens_params(
    model: str,
    messages: List[Dict[str, Any]]
) -> None:
    """
    Validate count_tokens method parameters.
    
    Args:
        model: Model name to validate
        messages: List of message dictionaries
        
    Raises:
        ValueError: If parameters are invalid
        TypeError: If parameter types are incorrect
    """
    if not model:
        raise ValueError("model parameter is required and cannot be empty")
    
    if not isinstance(model, str):
        raise TypeError(f"model must be a string, got {type(model).__name__}")
    
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")
    
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")
    
    # Validate messages (same as in completion)
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"Message at index {i} must be a dictionary, got {type(message).__name__}")
        
        if 'content' not in message:
            raise ValueError(f"Message at index {i} missing required 'content' field")
        
        if not isinstance(message['content'], str):
            raise TypeError(f"Message content must be a string, got {type(message['content']).__name__}")


def validate_tool_parameters(tools: Optional[List[Dict[str, Any]]]) -> None:
    """
    Validate tools parameter for function calling.
    
    Args:
        tools: Optional list of tool definitions
        
    Raises:
        TypeError: If tools parameter has incorrect type
        ValueError: If tools parameter has invalid structure
    """
    if tools is not None:
        if not isinstance(tools, list):
            raise TypeError(f"tools must be a list or None, got {type(tools).__name__}")
        
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise TypeError(f"Tool at index {i} must be a dictionary, got {type(tool).__name__}")
            
            if 'function' not in tool:
                raise ValueError(f"Tool at index {i} missing required 'function' field")
            
            function = tool['function']
            if not isinstance(function, dict):
                raise TypeError(f"Tool function must be a dictionary, got {type(function).__name__}")
            
            # Validate function fields
            if 'name' not in function:
                raise ValueError(f"Tool function at index {i} missing required 'name' field")
            
            if 'description' not in function:
                raise ValueError(f"Tool function at index {i} missing required 'description' field")
            
            # Validate parameters if present
            if 'parameters' in function:
                params = function['parameters']
                if not isinstance(params, dict):
                    raise TypeError(f"Tool parameters must be a dictionary, got {type(params).__name__}")
                
                if 'properties' not in params:
                    raise ValueError("Tool parameters must include 'properties' field")
                
                if 'type' not in params:
                    raise ValueError("Tool parameters must include 'type' field")
                
                if params['type'] != 'object':
                    raise ValueError("Tool parameters must have type 'object'")


def validate_reasoning_effort(reasoning_effort: Optional[str]) -> None:
    """
    Validate reasoning_effort parameter for Claude models.
    
    Args:
        reasoning_effort: Optional reasoning effort level
        
    Raises:
        ValueError: If reasoning_effort is invalid
    """
    if reasoning_effort is not None:
        valid_levels = {'low', 'medium', 'high'}
        if reasoning_effort not in valid_levels:
            raise ValueError(
                f"reasoning_effort must be one of {', '.join(valid_levels)}, got '{reasoning_effort}'"
            )
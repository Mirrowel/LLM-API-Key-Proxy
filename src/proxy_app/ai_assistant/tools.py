"""
Tool definition system for the AI Assistant.

Provides the @assistant_tool decorator, ToolDefinition, ToolResult, and ToolExecutor classes.
"""

import json
import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    message: str  # Human-readable description
    data: Optional[Dict[str, Any]] = None  # Structured data for AI
    error_code: Optional[str] = None  # Machine-readable error type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"success": self.success, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        if self.error_code is not None:
            result["error_code"] = self.error_code
        return result

    def to_json(self) -> str:
        """Convert to JSON string for LLM tool response."""
        return json.dumps(self.to_dict())


@dataclass
class ToolDefinition:
    """Definition of an assistant tool."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for parameters
    required: List[str] = field(default_factory=list)
    is_write: bool = False  # If True, triggers checkpoint creation
    handler: Optional[Callable[..., ToolResult]] = None

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[ToolResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class ToolCallSummary:
    """Summary of a tool call for checkpoint description."""

    name: str
    arguments: Dict[str, Any]
    success: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "arguments": self.arguments,
            "success": self.success,
            "message": self.message,
        }


def assistant_tool(
    name: str,
    description: str,
    parameters: Dict[str, Dict[str, Any]],
    required: Optional[List[str]] = None,
    is_write: bool = False,
) -> Callable:
    """
    Decorator to mark a method as an assistant tool.

    Args:
        name: The tool name (used in LLM function calls)
        description: Human-readable description of what the tool does
        parameters: JSON Schema properties dict for the tool parameters
        required: List of required parameter names
        is_write: If True, a checkpoint will be created before execution

    Example:
        @assistant_tool(
            name="add_ignore_rule",
            description="Add a pattern to the ignore list.",
            parameters={
                "pattern": {
                    "type": "string",
                    "description": "The pattern to ignore. Supports * wildcard."
                }
            },
            required=["pattern"],
            is_write=True
        )
        def tool_add_ignore_rule(self, pattern: str) -> ToolResult:
            ...
    """
    if required is None:
        required = []

    def decorator(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        # Store tool metadata on the function
        func._tool_definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            required=required,
            is_write=is_write,
            handler=func,
        )

        @wraps(func)
        def wrapper(*args, **kwargs) -> ToolResult:
            return func(*args, **kwargs)

        # Copy the tool definition to the wrapper
        wrapper._tool_definition = func._tool_definition
        return wrapper

    return decorator


class ToolExecutor:
    """
    Executes tool calls with validation.

    Collects tools from a WindowContextAdapter and executes them
    when called by the LLM.
    """

    def __init__(self, tools: List[ToolDefinition]):
        """
        Initialize with a list of tool definitions.

        Args:
            tools: List of ToolDefinition objects
        """
        self._tools: Dict[str, ToolDefinition] = {tool.name: tool for tool in tools}
        self._timeout: float = 30.0  # Default timeout in seconds

    @property
    def tools(self) -> List[ToolDefinition]:
        """Get list of all registered tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def has_write_tools(self, tool_calls: List[ToolCall]) -> bool:
        """Check if any of the tool calls are write operations."""
        for call in tool_calls:
            tool = self._tools.get(call.name)
            if tool and tool.is_write:
                return True
        return False

    def get_tools_openai_format(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI-compatible format."""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def validate_tool_call(self, tool_call: ToolCall) -> Optional[str]:
        """
        Validate a tool call before execution.

        Returns None if valid, or an error message if invalid.
        """
        tool = self._tools.get(tool_call.name)

        if tool is None:
            available = list(self._tools.keys())
            return f"Unknown tool: '{tool_call.name}'. Available tools: {available}"

        # Check required parameters
        for param in tool.required:
            if param not in tool_call.arguments:
                return f"Missing required parameter: '{param}'"

        # Check for unknown parameters
        known_params = set(tool.parameters.keys())
        provided_params = set(tool_call.arguments.keys())
        unknown = provided_params - known_params

        if unknown:
            # Provide helpful suggestions for typos
            suggestions = []
            for unk in unknown:
                for known in known_params:
                    if unk.lower() == known.lower() or (
                        len(unk) > 2 and unk[:-1] == known[:-1]
                    ):
                        suggestions.append(f"'{unk}' -> did you mean '{known}'?")
            if suggestions:
                return f"Invalid parameter(s): {unknown}. {' '.join(suggestions)}"
            return f"Invalid parameter(s): {unknown}. Valid parameters: {known_params}"

        # Type validation (basic)
        for param_name, param_value in tool_call.arguments.items():
            if param_name not in tool.parameters:
                continue
            param_schema = tool.parameters[param_name]
            expected_type = param_schema.get("type")

            if expected_type == "string" and not isinstance(param_value, str):
                return f"Parameter '{param_name}' must be a string, got {type(param_value).__name__}"
            elif expected_type == "number" and not isinstance(
                param_value, (int, float)
            ):
                return f"Parameter '{param_name}' must be a number, got {type(param_value).__name__}"
            elif expected_type == "integer" and not isinstance(param_value, int):
                return f"Parameter '{param_name}' must be an integer, got {type(param_value).__name__}"
            elif expected_type == "boolean" and not isinstance(param_value, bool):
                return f"Parameter '{param_name}' must be a boolean, got {type(param_value).__name__}"
            elif expected_type == "array" and not isinstance(param_value, list):
                return f"Parameter '{param_name}' must be an array, got {type(param_value).__name__}"
            elif expected_type == "object" and not isinstance(param_value, dict):
                return f"Parameter '{param_name}' must be an object, got {type(param_value).__name__}"

        return None  # Valid

    def execute(self, tool_call: ToolCall, context: Any) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: The tool call to execute
            context: The WindowContextAdapter instance (passed as self to the tool method)

        Returns:
            ToolResult with success/failure status
        """
        # Validate first
        validation_error = self.validate_tool_call(tool_call)
        if validation_error:
            logger.warning(
                f"Tool validation failed for {tool_call.name}: {validation_error}"
            )
            return ToolResult(
                success=False,
                message=validation_error,
                error_code="invalid_parameters",
            )

        tool = self._tools[tool_call.name]

        try:
            # Execute the tool handler
            result = tool.handler(context, **tool_call.arguments)

            if not isinstance(result, ToolResult):
                logger.error(
                    f"Tool {tool_call.name} returned non-ToolResult: {type(result)}"
                )
                return ToolResult(
                    success=False,
                    message=f"Tool returned invalid result type: {type(result).__name__}",
                    error_code="internal_error",
                )

            if result.success:
                logger.info(f"Tool {tool_call.name} succeeded: {result.message}")
            else:
                logger.warning(f"Tool {tool_call.name} failed: {result.message}")

            return result

        except Exception as e:
            logger.exception(f"Tool {tool_call.name} raised exception")
            return ToolResult(
                success=False,
                message=f"Tool execution error: {str(e)}",
                error_code="execution_error",
            )

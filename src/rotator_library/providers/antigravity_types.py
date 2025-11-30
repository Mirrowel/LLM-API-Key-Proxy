# src/rotator_library/providers/antigravity_types.py
"""
Type definitions for Antigravity provider.

Provides TypedDict definitions for complex nested structures to improve
type safety and IDE support.
"""

from typing import TypedDict, Literal, List, Dict, Any, Optional, Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
    import litellm


# =============================================================================
# GEMINI API TYPES
# =============================================================================

class GeminiPart(TypedDict, total=False):
    """Single part of a Gemini content message."""
    text: str
    inlineData: Dict[str, str]
    functionCall: Dict[str, Any]
    functionResponse: Dict[str, Any]
    thought: bool
    thoughtSignature: str


class GeminiContent(TypedDict):
    """Gemini content message with role and parts."""
    role: Literal["user", "model"]
    parts: List[GeminiPart]


class SystemInstruction(TypedDict):
    """System instruction for Gemini API."""
    role: Literal["user"]
    parts: List[GeminiPart]


class ThinkingConfig(TypedDict, total=False):
    """Thinking configuration for models with reasoning capabilities."""
    thinkingBudget: int
    thinkingLevel: str
    include_thoughts: bool


class GenerationConfig(TypedDict, total=False):
    """Generation configuration for Gemini API."""
    topP: float
    temperature: float
    maxOutputTokens: int
    thinkingConfig: ThinkingConfig


class ToolDeclaration(TypedDict):
    """Tool declaration for function calling."""
    name: str
    description: str
    parametersJsonSchema: Dict[str, Any]


class Tool(TypedDict):
    """Tool container for Gemini API."""
    functionDeclarations: List[ToolDeclaration]


class GeminiRequest(TypedDict):
    """Complete Gemini API request structure."""
    contents: List[GeminiContent]
    generationConfig: GenerationConfig
    tools: List[Tool]
    system_instruction: SystemInstruction
    safetySettings: List[Dict[str, str]]


# =============================================================================
# ANTIGRAVITY ENVELOPE TYPES
# =============================================================================

class AntigravityRequest(TypedDict):
    """Antigravity request envelope structure."""
    project: str
    userAgent: str
    requestId: str
    model: str
    request: Dict[str, Any]
    sessionId: str


# =============================================================================
# RESPONSE TYPES
# =============================================================================

class FunctionCall(TypedDict):
    """Function call from model response."""
    name: str
    args: Dict[str, Any]
    id: str


class ToolCall(TypedDict):
    """OpenAI-style tool call."""
    id: str
    type: Literal["function"]
    index: int
    function: Dict[str, str]


class UsageMetadata(TypedDict, total=False):
    """Token usage metadata from Gemini API."""
    promptTokenCount: int
    thoughtsTokenCount: int
    candidatesTokenCount: int
    totalTokenCount: int


class CompletionDelta(TypedDict, total=False):
    """Streaming completion delta."""
    content: str
    reasoning_content: str
    tool_calls: List[ToolCall]
    role: Literal["assistant"]


class CompletionChoice(TypedDict):
    """Completion choice in OpenAI format."""
    index: int
    delta: CompletionDelta
    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]]


class CompletionChunk(TypedDict):
    """Streaming completion chunk."""
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Dict[str, int]]


class CompletionMessage(TypedDict, total=False):
    """Non-streaming completion message."""
    role: Literal["assistant"]
    content: str
    reasoning_content: str
    tool_calls: List[ToolCall]


class CompletionChoiceNonStreaming(TypedDict):
    """Non-streaming completion choice."""
    index: int
    message: CompletionMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"]


class CompletionResponse(TypedDict):
    """Complete non-streaming response."""
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[CompletionChoiceNonStreaming]
    usage: Optional[Dict[str, int]]


# =============================================================================
# PARAMETER TYPES
# =============================================================================

class CompletionParameters(TypedDict):
    """Extracted completion parameters."""
    model: str
    messages: List[Dict[str, Any]]
    stream: bool
    tools: Optional[List[Dict[str, Any]]]
    tool_choice: Optional[Union[str, Dict[str, Any]]]
    reasoning_effort: Optional[str]
    top_p: Optional[float]
    temperature: Optional[float]
    max_tokens: Optional[int]
    custom_budget: bool
    credential_path: str


class ModelTypeInfo(TypedDict):
    """Cached model type information."""
    is_gemini_25: bool
    is_gemini_3: bool
    is_claude: bool
    internal_name: str


# =============================================================================
# ERROR TYPES
# =============================================================================

class APIErrorInfo(TypedDict):
    """API error information for logging and debugging."""
    status_code: int
    error_type: str
    message: str
    request_id: Optional[str]
    model: Optional[str]


# =============================================================================
# UTILITY TYPES
# =============================================================================

class ConversationState(TypedDict):
    """Conversation state analysis result."""
    in_tool_loop: bool
    last_assistant_idx: int
    last_assistant_has_thinking: bool
    last_assistant_has_tool_calls: bool
    pending_tool_results: bool
    thinking_block_indices: List[int]


class SanitizationResult(TypedDict):
    """Thinking sanitization result."""
    sanitized_messages: List[Dict[str, Any]]
    force_disable_thinking: bool
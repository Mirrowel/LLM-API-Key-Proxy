"""
AI Assistant System for GUI Windows.

A reusable AI assistant that can be integrated into any GUI tool window.
Provides full context awareness, tool execution, checkpoints for undo,
and streaming responses with thinking visibility.

Main Components:
- AIAssistantCore: Main orchestration class
- WindowContextAdapter: Abstract base for window integration
- LLMBridge: Async LLM communication bridge
- CheckpointManager: Undo/rollback capability
- Tool system: @assistant_tool decorator and execution

Usage:
    from proxy_app.ai_assistant import AIAssistantCore, WindowContextAdapter

    class MyWindowAdapter(WindowContextAdapter):
        # Implement abstract methods
        ...

    core = AIAssistantCore(
        window_adapter=my_adapter,
        schedule_on_gui=lambda fn: window.after(0, fn),
    )
"""

from .bridge import LLMBridge, StreamCallbacks
from .checkpoint import Checkpoint, CheckpointManager
from .context import (
    WindowContextAdapter,
    apply_delta,
    compute_context_diff,
    compute_delta,
)
from .core import AIAssistantCore, ChatSession, Message
from .prompts import BASE_ASSISTANT_PROMPT, MODEL_FILTER_SYSTEM_PROMPT
from .tools import (
    ToolCall,
    ToolCallSummary,
    ToolDefinition,
    ToolExecutor,
    ToolResult,
    assistant_tool,
)

__all__ = [
    # Core
    "AIAssistantCore",
    "ChatSession",
    "Message",
    # Bridge
    "LLMBridge",
    "StreamCallbacks",
    # Checkpoint
    "CheckpointManager",
    "Checkpoint",
    # Context
    "WindowContextAdapter",
    "compute_context_diff",
    "compute_delta",
    "apply_delta",
    # Tools
    "assistant_tool",
    "ToolDefinition",
    "ToolResult",
    "ToolCall",
    "ToolCallSummary",
    "ToolExecutor",
    # Prompts
    "BASE_ASSISTANT_PROMPT",
    "MODEL_FILTER_SYSTEM_PROMPT",
]

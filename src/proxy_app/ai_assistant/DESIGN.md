# AI Assistant System - Design Document

**Version**: 1.1  
**Status**: Draft  
**Target Window**: Model Filter Configuration GUI (prototype)

---

## 1. Overview

### 1.1 Purpose

A reusable AI assistant system that can be integrated into any GUI tool window. The assistant has full context of the window's state, can execute actions via tools, maintains checkpoints for undo capability, and supports streaming responses with thinking visibility.

### 1.2 Core Principles

| Principle | Description |
|-----------|-------------|
| **Reusability** | Window-agnostic core with window-specific adapters |
| **Full Context** | Assistant always has complete visibility into window state |
| **Agentic** | Multi-tool execution, self-correction, error handling |
| **Non-destructive** | Checkpoint system prevents data loss |
| **Responsive** | Streaming responses with thinking display |

### 1.3 Implementation Strategy

**Prototype Phase**: Pop-out window only, no embedded panel. The embedded compact mode will be added after the pop-out is tested and refined.

---

## 2. System Architecture

### 2.1 Component Hierarchy

```
+-------------------------------------------------------------------+
|                        GUI Window                                  |
|  (e.g., ModelFilterGUI)                                           |
|  +---------------------------------------------------------------+|
|  |                   WindowContextAdapter                         ||
|  |  - Implements window-specific context extraction               ||
|  |  - Registers window-specific tools                             ||
|  |  - Provides window-specific system prompt                      ||
|  +---------------------------------------------------------------+|
+-------------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                      AIAssistantCore                               |
|  +--------------+  +--------------+  +--------------------------+ |
|  | ChatSession  |  | ToolExecutor |  | CheckpointManager        | |
|  | - History    |  | - Registry   |  | - Snapshots + Deltas     | |
|  | - Context    |  | - Validation |  | - Hybrid storage         | |
|  | - Streaming  |  | - Execution  |  | - Temp file backup       | |
|  +--------------+  +--------------+  +--------------------------+ |
+-------------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                      LLMBridge                                     |
|  - Wraps RotatingClient                                           |
|  - Thread/async coordination                                      |
|  - Streaming chunk processing                                     |
|  - Model selection                                                |
+-------------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                      AIChatWindow (UI)                             |
|  +---------------------------------------------------------------+|
|  |                    Popped-Out Mode                             ||
|  |  - Full message display (dynamically sized)                    ||
|  |  - Expanded input                                              ||
|  |  - Model selector (grouped by provider)                        ||
|  |  - Thinking sections (collapsible, auto-collapse)              ||
|  |  - Checkpoint dropdown                                         ||
|  |  - All features enabled                                        ||
|  +---------------------------------------------------------------+|
+-------------------------------------------------------------------+
```

### 2.2 Data Flow

```
User Input
    |
    v
[Queue if busy] ------------------------------------------+
    |                                                      |
    v                                                      |
WindowContextAdapter.get_full_context()                    |
    |                                                      |
    v                                                      |
Diff against last_known_context                            |
    |                                                      |
    v                                                      |
Build messages array:                                      |
  - Base system prompt                                     |
  - Window-specific system prompt                          |
  - Context injection                                      |
  - Conversation history                                   |
  - User message                                           |
    |                                                      |
    v                                                      |
LLMBridge.stream_completion()                              |
    |                                                      |
    +---> [Thinking chunks] ---> Display (collapsible)     |
    |                                                      |
    +---> [Content chunks] ---> Display streaming          |
    |                                                      |
    +---> [Tool calls] ---+                                |
                          v                                |
                Parse tool calls                           |
                          |                                |
                          v                                |
                Has write tools?                           |
                 |           |                             |
                YES         NO                             |
                 |           |                             |
                 v           |                             |
          Create checkpoint  |                             |
          (if not exists     |                             |
           for this response)|                             |
                 |           |                             |
                 v           v                             |
             Execute tools sequentially                    |
                 |                                         |
                 v                                         |
           Collect results (success/failure)               |
                 |                                         |
                 v                                         |
           Feed results back to LLM                        |
                 |                                         |
                 +-----> [Continue if more tool calls] ----+
                 |
                 v
           [Response complete]
```

---

## 3. Core Components

### 3.1 WindowContextAdapter (Abstract Base Class)

**Purpose**: Interface that each window must implement to connect to the assistant.

**Required Methods**:

| Method | Return Type | Description |
|--------|-------------|-------------|
| `get_full_context()` | `Dict[str, Any]` | Complete structured state of the window |
| `get_window_system_prompt()` | `str` | Window-specific instructions for the AI |
| `get_tools()` | `List[ToolDefinition]` | Available tools for this window |
| `apply_state(state: Dict)` | `None` | Restore window to a given state (for checkpoints) |
| `get_state_hash()` | `str` | Quick hash for change detection |

**Example Context Structure for ModelFilterGUI**:

```python
{
    "window_type": "model_filter_gui",
    "current_provider": "openai",
    "models": {
        "total_count": 45,
        "available_count": 38,
        "items": [
            {
                "id": "openai/gpt-4o",
                "display_name": "gpt-4o",
                "status": "normal",  # "normal" | "ignored" | "whitelisted"
                "affecting_rule": null
            },
            {
                "id": "openai/gpt-4-turbo",
                "display_name": "gpt-4-turbo", 
                "status": "ignored",
                "affecting_rule": {
                    "pattern": "gpt-4-turbo*", 
                    "type": "ignore"
                }
            }
            # ... all models
        ]
    },
    "rules": {
        "ignore": [
            {
                "pattern": "gpt-4-turbo*", 
                "affected_count": 3, 
                "affected_models": ["gpt-4-turbo", "gpt-4-turbo-preview", "..."]
            },
            {
                "pattern": "*-preview", 
                "affected_count": 5, 
                "affected_models": ["..."]
            }
        ],
        "whitelist": [
            {
                "pattern": "gpt-4o", 
                "affected_count": 1, 
                "affected_models": ["gpt-4o"]
            }
        ]
    },
    "ui_state": {
        "search_query": "",
        "has_unsaved_changes": true,
        "highlighted_rule": null,
        "highlighted_models": []
    },
    "available_providers": ["openai", "gemini", "anthropic"],
    "changes_since_last_message": [
        {
            "type": "rule_added", 
            "rule_type": "ignore", 
            "pattern": "o1*", 
            "timestamp": "..."
        },
        {
            "type": "provider_changed", 
            "from": "gemini", 
            "to": "openai", 
            "timestamp": "..."
        }
    ]
}
```

### 3.2 Tool Definition System

**Tool Decorator Syntax**:

```python
@assistant_tool(
    name="add_ignore_rule",
    description="Add a pattern to the ignore list. Models matching this pattern will be blocked.",
    parameters={
        "pattern": {
            "type": "string",
            "description": "The pattern to ignore. Supports * wildcard."
        }
    },
    required=["pattern"],
    is_write=True  # Triggers checkpoint creation
)
def tool_add_ignore_rule(self, pattern: str) -> ToolResult:
    """Add an ignore rule."""
    success = self._add_ignore_pattern(pattern)
    if success:
        return ToolResult(
            success=True,
            message=f"Added ignore rule: {pattern}",
            data={
                "pattern": pattern, 
                "affected_models": self._get_affected_models(pattern)
            }
        )
    else:
        return ToolResult(
            success=False,
            message=f"Pattern '{pattern}' is already covered by existing rule",
            data={"existing_rules": self._get_covering_rules(pattern)}
        )
```

**ToolResult Structure**:

```python
@dataclass
class ToolResult:
    success: bool
    message: str                          # Human-readable description
    data: Optional[Dict[str, Any]] = None # Structured data for AI
    error_code: Optional[str] = None      # Machine-readable error type
```

**Tool Categories for ModelFilterGUI**:

| Category | Tool Name | Write? | Description |
|----------|-----------|--------|-------------|
| **Rules** | `add_ignore_rule` | Yes | Add pattern to ignore list |
| | `remove_ignore_rule` | Yes | Remove pattern from ignore list |
| | `add_whitelist_rule` | Yes | Add pattern to whitelist |
| | `remove_whitelist_rule` | Yes | Remove pattern from whitelist |
| | `clear_all_ignore_rules` | Yes | Clear all ignore rules |
| | `clear_all_whitelist_rules` | Yes | Clear all whitelist rules |
| | `import_rules` | Yes | Bulk import rules |
| **Query** | `get_models_matching_pattern` | No | Preview pattern matches |
| | `get_model_details` | No | Get details for specific model |
| | `explain_model_status` | No | Explain why a model has its status |
| **Provider** | `switch_provider` | No | Change active provider |
| | `refresh_models` | No | Refetch models from provider |
| **State** | `save_changes` | Yes | Save to .env file |
| | `discard_changes` | Yes | Revert to saved state |

### 3.3 CheckpointManager

**Checkpoint Strategy**: Hybrid snapshot/delta approach.

- Full snapshot stored every 10 checkpoints
- Deltas stored between snapshots
- All checkpoints persisted to temp file for crash recovery
- On rollback: Load nearest full snapshot, apply deltas forward to target

```
[Full #0] -> D1 -> D2 -> ... -> D9 -> [Full #10] -> D11 -> ... -> D19 -> [Full #20]
```

**Checkpoint Structure**:

```python
@dataclass
class Checkpoint:
    id: str                                  # UUID
    timestamp: datetime
    description: str                         # Auto-generated from tools
    tool_calls: List[ToolCallSummary]        # What tools were called
    message_index: int                       # Conversation position at checkpoint time
    
    # One of these will be populated:
    full_state: Optional[Dict[str, Any]]     # Full snapshot (every Nth)
    delta: Optional[Dict[str, Any]]          # Changes from previous
    
    is_full_snapshot: bool
```

**Delta Format**:

```python
{
    "added": {
        "rules.ignore": [{"pattern": "gpt-4*", "...": "..."}]
    },
    "removed": {
        "rules.whitelist": [{"pattern": "claude*", "...": "..."}]
    },
    "modified": {
        "ui_state.search_query": {"old": "", "new": "gpt"}
    }
}
```

**Checkpoint Creation Logic**:

1. Before executing the first `is_write=True` tool in a response
2. Check if a checkpoint already exists for this response
3. If not, create one and proceed
4. If yes (shouldn't happen), log warning and proceed anyway

**Rollback Algorithm**:

```python
def rollback_to(checkpoint_id: str):
    target_index = find_checkpoint_index(checkpoint_id)
    
    # Find nearest full snapshot at or before target
    snapshot_index = find_nearest_snapshot_before(target_index)
    
    # Load full snapshot
    state = load_full_snapshot(snapshot_index)
    
    # Apply deltas from snapshot to target
    for i in range(snapshot_index + 1, target_index + 1):
        state = apply_delta(state, checkpoints[i].delta)
    
    # Apply state to window (atomic operation)
    window_context.apply_state(state)
    
    # Rollback conversation history to that point
    truncate_conversation_to_checkpoint(checkpoint_id)
    
    # Truncate checkpoint list (remove all after target)
    checkpoints = checkpoints[:target_index + 1]
    
    # Mark current position
    current_checkpoint_position = target_index
```

**Apply State Mechanics**:

The `WindowContextAdapter.apply_state()` method restores window state atomically:

1. Update internal data structures (e.g., `filter_engine.ignore_rules`)
2. Update UI input fields (e.g., search query)
3. Call existing refresh methods (e.g., `_on_rules_changed()`, `_update_model_display()`)

This reuses existing UI update logic rather than duplicating it. The operation is atomic:
if any step fails, the entire restore is aborted and the window remains in its pre-restore state.

**Conversation Rollback**:

When rolling back to a checkpoint, conversation history is also rolled back:
- All messages after the checkpoint are removed
- The conversation state matches what it was when the checkpoint was created
- This ensures AI context and window state are always synchronized

### 3.4 LLMBridge

**Purpose**: Bridge between async `RotatingClient` and sync GUI thread.

**Key Responsibilities**:

- Manage `RotatingClient` lifecycle
- Handle thread/async coordination using `threading.Thread` + `asyncio.run()`
- Process streaming chunks and route to appropriate handlers
- Parse tool calls from responses (OpenAI-compatible native JSON tools)
- Manage model list fetching (same list as `/v1/models` endpoint)

**Streaming Callback Interface**:

```python
callbacks = {
    "on_thinking_chunk": Callable[[str], None],
    "on_content_chunk": Callable[[str], None],
    "on_tool_calls": Callable[[List[ToolCall]], None],
    "on_error": Callable[[str], None],
    "on_complete": Callable[[], None],
}
```

**Thread Coordination Pattern**:

```python
def stream_completion(messages, tools, model, callbacks):
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            async def stream():
                response = await client.acompletion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=True
                )
                
                async for chunk in response:
                    parsed = parse_chunk(chunk)
                    
                    if parsed.reasoning_content:
                        schedule_on_gui_thread(
                            callbacks["on_thinking_chunk"], 
                            parsed.reasoning_content
                        )
                    
                    if parsed.content:
                        schedule_on_gui_thread(
                            callbacks["on_content_chunk"], 
                            parsed.content
                        )
                    
                    if parsed.tool_calls:
                        schedule_on_gui_thread(
                            callbacks["on_tool_calls"], 
                            parsed.tool_calls
                        )
                
                schedule_on_gui_thread(callbacks["on_complete"])
            
            loop.run_until_complete(stream())
        except Exception as e:
            schedule_on_gui_thread(callbacks["on_error"], str(e))
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
```

### 3.5 ChatSession

**Purpose**: Manages conversation state and message history.

**State**:

```python
@dataclass
class ChatSession:
    session_id: str
    model: str
    messages: List[Message]
    pending_message: Optional[str]        # Queued user message
    is_streaming: bool
    current_checkpoint_position: int
    last_known_context_hash: str
    
    # Retry tracking
    consecutive_invalid_tool_calls: int
    max_tool_retries: int = 4
```

**Message Types**:

```python
@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: Optional[str]
    reasoning_content: Optional[str]          # Thinking (from reasoning_content field)
    tool_calls: Optional[List[ToolCall]]
    tool_call_id: Optional[str]               # For tool response messages
    timestamp: datetime
    
@dataclass  
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[ToolResult]              # Populated after execution
```

**Model Selection**:

- Per-window, per-session persistence
- Model list fetched from `RotatingClient.get_all_available_models()` when window opens
- Model list is refreshed each time the AI chat window is opened
- Same filtered list that proxy serves at `/v1/models` endpoint
- If selected model becomes unavailable: show error, user must select another

**Message Queue**:

- If user sends message while streaming: queue it
- Process queued message after current response completes (not the entire agentic chain)
- Only one message can be queued at a time (new message replaces queued)

**New Session Button**:

When clicked:
1. Clear all conversation history
2. Clear all checkpoints
3. Model selection is preserved
4. Window state is NOT reset (current rules, models, etc. remain)
5. Fresh context snapshot is taken for the new session

---

## 4. UI Components

### 4.1 Design Principles

**Dynamic Resizing**:

- All panels and boxes use weight-based grid layout
- Only fixed sizes for: buttons, input field heights, padding
- Window is freely resizable by user
- Reference: Current `ModelFilterGUI` implementation using `grid_rowconfigure(weight=N)`

**Consistency**:

- Use same color constants as `model_filter_gui.py`
- Same font family and size scale
- Same border styles and corner radii

### 4.2 AIChatWindow (Pop-Out) Layout

```
+------------------------------------------------------------------------+
|  AI Assistant - Model Filter Configuration              [-] [_] [X]    |
+------------------------------------------------------------------------+
| +------------------------------------------+ +------------------------+ |
| | Model: [openai/gpt-4o              v]    | | Checkpoints       [v] | |
| +------------------------------------------+ +------------------------+ |
+------------------------------------------------------------------------+
|                                                                        |
|  +------------------------------------------------------------------+  |
|  |                        Message Display                            |  |
|  |                    (scrollable, weight=3)                         |  |
|  | ----------------------------------------------------------------- |  |
|  |                                                                   |  |
|  |  v Thinking (collapsed - click to expand)                        |  |
|  |  ---------------------------------------------------------------- |  |
|  |  AI: I'll help you configure the model filters. I can see you    |  |
|  |  have 45 models from OpenAI, with 7 currently ignored.           |  |
|  |                                                                   |  |
|  |  ---------------------------------------------------------------- |  |
|  |                                                                   |  |
|  |  You: Block all preview and experimental models                  |  |
|  |                                                                   |  |
|  |  ---------------------------------------------------------------- |  |
|  |                                                                   |  |
|  |  > Thinking (expanded)                                           |  |
|  |  +-------------------------------------------------------------+ |  |
|  |  | I need to identify patterns that match preview and          | |  |
|  |  | experimental models. Looking at the model list, I see:      | |  |
|  |  | - gpt-4-turbo-preview                                       | |  |
|  |  | - gpt-4o-preview                                            | |  |
|  |  | ...                                                         | |  |
|  |  +-------------------------------------------------------------+ |  |
|  |                                                                   |  |
|  |  AI: I'll add two patterns to block these:                       |  |
|  |  - `*-preview` - blocks 5 preview models                         |  |
|  |  - `*-experimental` - blocks 2 experimental models               |  |
|  |                                                                   |  |
|  |  [checkmark] Tool: add_ignore_rule(pattern="*-preview")          |  |
|  |    Result: Added. 5 models now blocked.                          |  |
|  |                                                                   |  |
|  |  [checkmark] Tool: add_ignore_rule(pattern="*-experimental")     |  |
|  |    Result: Added. 2 models now blocked.                          |  |
|  |                                                                   |  |
|  +------------------------------------------------------------------+  |
|                                                                        |
+------------------------------------------------------------------------+
| +------------------------------------------------------------------+  |
| |                                                                   |  |
| | Type your message here...                                         |  |
| |                                                                   |  |
| | (scrollable input, 3+ lines visible)                              |  |
| |                                                                   |  |
| +------------------------------------------------------------------+  |
|                                         [New Session]  [Send  ->]     |
+------------------------------------------------------------------------+
```

**Grid Layout Specification**:

```python
# Window grid configuration
window.grid_columnconfigure(0, weight=1)

# Row 0: Header (model selector + checkpoints) - fixed height
window.grid_rowconfigure(0, weight=0)

# Row 1: Message display - weight=3 (takes most space)
window.grid_rowconfigure(1, weight=3, minsize=200)

# Row 2: Input area - weight=1 (grows but less than messages)
window.grid_rowconfigure(2, weight=1, minsize=80)

# Row 3: Buttons - fixed height
window.grid_rowconfigure(3, weight=0)
```

### 4.3 Component Details

#### 4.3.1 Model Selector

- Grouped dropdown by provider
- Format: `provider/model-name`
- Groups: openai, gemini, anthropic, etc.
- Persists selection for session

```
+--------------------------------+
| openai/gpt-4o            [v]   |
+--------------------------------+
| -- openai --                   |
|   gpt-4o                       |
|   gpt-4o-mini                  |
|   gpt-4-turbo                  |
| -- gemini --                   |
|   gemini-2.0-flash             |
|   gemini-1.5-pro               |
| -- anthropic --                |
|   claude-3-5-sonnet            |
+--------------------------------+
```

#### 4.3.2 Checkpoint Dropdown

Clicking opens a popup/dropdown list:

```
+----------------------------------------------------------+
|  Checkpoints                                       [X]   |
+----------------------------------------------------------+
|  (*) Current State                                       |
|  --------------------------------------------------------|
|  ( ) 14:32:15 - add_ignore_rule("*-preview")            |
|      -> Added 5 models to ignore                         |
|  --------------------------------------------------------|
|  ( ) 14:31:42 - add_ignore_rule("gpt-4*")               |
|      -> Added 3 models to ignore                         |
|  --------------------------------------------------------|
|  ( ) 14:30:00 - Session Start                           |
|      -> Initial state                                    |
+----------------------------------------------------------+
|              [Rollback to Selected]  [Cancel]            |
+----------------------------------------------------------+
```

#### 4.3.3 Message Display

Canvas-based virtual list for performance (reference: `VirtualModelList` in `model_filter_gui.py`).

**Message Styling**:

| Element | Style |
|---------|-------|
| User message | Right-aligned, accent background |
| AI message | Left-aligned, secondary background |
| Thinking block | Muted color, smaller font, collapsible |
| Tool execution | Monospace, subtle background, icon prefix |
| Tool success | Green checkmark prefix |
| Tool failure | Red X prefix |
| Timestamp | Muted, small, right-aligned |

**Thinking Behavior**:

- Starts expanded while streaming
- Auto-collapses when first chunk with `content` but no `reasoning_content` arrives
- Click to expand/collapse manually at any time
- Styled: muted text color, slightly smaller font, distinct background

#### 4.3.4 Input Area

- Multi-line text input (CTkTextbox)
- Minimum 3 lines visible
- Scrollable for longer input
- Keyboard shortcuts:
  - `Ctrl+Enter`: Send message
  - `Escape`: Clear input / cancel if streaming

#### 4.3.5 Error Display

Errors appear inline where response would be, replacing on success:

```
+----------------------------------------------------------+
|  [!] Connection Error                                    |
|  Failed to reach model. Check your network connection.   |
|                                        [Retry] [Cancel]  |
+----------------------------------------------------------+
```

- Styled with warning colors
- Replaced by response if retry succeeds
- Not added to message history

---

## 5. Error Handling

### 5.1 Invalid Tool Calls

**Retry Logic**:

1. If AI generates invalid tool call (bad parameters, unknown tool)
2. Silently feed error back to AI: "Tool call failed: [error]. Please correct and retry."
3. After 2nd retry failure: show subtle "Retrying..." indicator in UI
4. After 4 total failures: show error to user

**Error Message to AI**:

```json
{
    "role": "tool",
    "tool_call_id": "call_xyz",
    "content": {
        "success": false,
        "error": "Invalid parameter: 'patern' is not a valid parameter. Did you mean 'pattern'?",
        "hint": "Please review the tool schema and retry."
    }
}
```

### 5.2 Partial Tool Execution

If AI calls 3 tools and 2nd fails:

1. Tool 1 result: applied, success fed back
2. Tool 2 result: NOT applied, error fed back
3. Tool 3: still executed (errors are per-tool, not chain-breaking)

AI receives all results and can decide how to proceed.

### 5.3 Model Unavailability

If selected model becomes unavailable (credential issue, rate limit):

1. Show error in UI (not as chat message)
2. User must select different model from dropdown
3. No auto-fallback

### 5.4 LLM Connection Errors

1. Display inline error where response would appear
2. Provide Retry and Cancel buttons
3. If retry succeeds, error replaced by response
4. Error is NOT added to conversation history

### 5.5 Tool Execution Timeout

All tools have a default timeout. If a tool execution exceeds the timeout:

1. Tool returns failure result with timeout error
2. Error is fed back to AI like any other tool failure
3. AI can decide to retry or inform user

### 5.6 Streaming Cancellation

If user presses Escape or clicks Cancel during streaming:

1. Streaming is immediately stopped
2. Partial response is discarded (not added to conversation history)
3. Any tool calls that were pending are NOT executed
4. UI returns to ready state for new input

### 5.7 Context Window Limits

If conversation history approaches token limits:

1. Show warning to user: "Conversation is getting long. Consider starting a new session."
2. Do NOT automatically truncate or summarize
3. User can click "New Session" to start fresh

Given typical context windows of 120k-250k+ tokens, this should be rare.

### 5.8 Concurrency and Window Locking

When the assistant's turn begins (request is sent):

1. The main window (e.g., ModelFilterGUI) is locked for user interaction
2. User cannot click buttons or modify fields during agent execution
3. Lock is released when agent turn completes (after all tool calls finish)

This prevents race conditions between manual user actions and AI tool execution.

---

## 6. System Prompts

### 6.1 Base Assistant Prompt (All Windows)

```
You are an AI assistant embedded in a GUI application. Your role is to help users 
accomplish tasks within this window by understanding their intent and executing 
actions using the available tools.

## Core Behaviors

1. **Full Context Awareness**: You have complete visibility into the window's state. 
   Use this information to provide accurate, contextual help.

2. **Tool Execution**: When the user requests an action, use the appropriate tools 
   to execute it. You may call multiple tools in sequence to accomplish complex tasks.

3. **Verbose Feedback**: After executing tools, clearly explain what was done, 
   what changed, and any important consequences. Both you and the user will see
   the tool results.

4. **Error Handling**: If a tool fails, explain why and suggest alternatives. 
   If you receive an error about an invalid tool call, carefully re-examine the 
   tool schema and try again with corrected parameters.

5. **Proactive Assistance**: If you notice potential issues or improvements, 
   mention them to the user.

## Tool Execution Guidelines

- Always confirm understanding before making destructive changes
- For bulk operations, summarize what will happen before executing
- If uncertain about user intent, ask for clarification
- Report all tool results, including partial successes
- You may call multiple tools in a single response when appropriate

## Context Updates

You will receive updates about changes to the window state in the 
`changes_since_last_message` field. Use this to stay aware of what 
the user may have done manually between messages.
```

### 6.2 Window-Specific Prompt (ModelFilterGUI)

```
## Model Filter Configuration Assistant

You are helping the user configure model filtering rules for an LLM proxy server.

### Domain Knowledge

- **Ignore Rules**: Patterns that block models from being available through the proxy
- **Whitelist Rules**: Patterns that ensure models are always available (override ignore rules)
- **Pattern Syntax**:
  - Exact match: `gpt-4` matches only "gpt-4"
  - Wildcard: `gpt-4*` matches "gpt-4", "gpt-4-turbo", "gpt-4-vision", etc.
  - Match anywhere: `*preview*` matches any model containing "preview"

### Rule Priority

Whitelist > Ignore > Default (available)

A model that matches both an ignore rule and a whitelist rule will be AVAILABLE 
(whitelist wins).

### Common Tasks

1. "Block all preview models" -> Use pattern `*-preview` or `*preview*`
2. "Only allow GPT-4o" -> Ignore `*`, whitelist `gpt-4o`
3. "What models are blocked?" -> Query the ignore rules and their affected models

### Important Notes

- Changes are not saved until the user explicitly saves (or you use save_changes tool)
- The `has_unsaved_changes` field in context tells you if there are pending changes
- Always inform the user if there are unsaved changes that might be lost

### Current Context

[Context will be injected dynamically from WindowContextAdapter.get_full_context()]
```

---

## 7. Message Flow Examples

### 7.1 Simple Tool Execution

**User**: "Add gpt-4* to the ignore list"

**Processing**:
1. Parse intent: add ignore rule
2. Check checkpoint: first write tool, create checkpoint
3. Execute: `add_ignore_rule(pattern="gpt-4*")`
4. Tool returns: `{success: true, message: "Added...", data: {affected_models: [...]}}`
5. Generate response incorporating result

**Response**: 
"I've added `gpt-4*` to the ignore list. This blocks 4 models: gpt-4, gpt-4-turbo, gpt-4-vision, and gpt-4-turbo-preview."

### 7.2 Multi-Tool with Partial Failure

**User**: "Block all Claude and GPT-3 models"

**Tool Calls**:
1. `add_ignore_rule(pattern="claude*")` -> Success, applied
2. `add_ignore_rule(pattern="gpt-3*")` -> Fails: "Pattern covered by existing 'gpt-*'"

**Response**: 
"I've added `claude*` to the ignore list, blocking 3 Claude models.

However, I couldn't add `gpt-3*` because it's already covered by your existing rule `gpt-*`. All GPT-3 models are already blocked by that broader pattern."

### 7.3 Invalid Tool Call with Self-Correction

**User**: "Show me what matches 'preview'"

**Attempt 1** (user doesn't see this):
`get_models_matching_pattern()` - missing required parameter
-> Error fed to AI: "Missing required parameter: pattern"

**Attempt 2** (user doesn't see this):
`get_models_matching_pattern(pattern="preview")` 
-> Error: "No matches. Hint: use wildcards for partial matching"

**Attempt 3**:
`get_models_matching_pattern(pattern="*preview*")`
-> Success

**Response** (user only sees final result):
"Here are 5 models containing 'preview': gpt-4-turbo-preview, gpt-4o-preview, ..."

### 7.4 Agentic Multi-Step Task

**User**: "Set up the filter so only the latest models from each major provider are available"

**AI Thinking** (visible, collapsible):
"I need to:
1. First understand what models are available
2. Identify 'latest' models per provider
3. Create an ignore-all rule
4. Whitelist the latest models

Let me check the current model list..."

**Tool Calls** (sequence):
1. `get_model_details()` - Query current state
2. `clear_all_ignore_rules()` - Start fresh
3. `add_ignore_rule(pattern="*")` - Block everything
4. `add_whitelist_rule(pattern="gpt-4o")` - Latest OpenAI
5. `add_whitelist_rule(pattern="gemini-2.0-flash")` - Latest Gemini
6. `add_whitelist_rule(pattern="claude-3-5-sonnet")` - Latest Anthropic

**Response**:
"I've configured the filter to only allow the latest models:

**Whitelisted (available):**
- openai/gpt-4o
- gemini/gemini-2.0-flash  
- anthropic/claude-3-5-sonnet

**Blocked:** All other 42 models

Note: These changes are not saved yet. Use the Save button or ask me to save when you're satisfied with the configuration."

---

## 8. Implementation Phases

### Phase 1: Core Infrastructure

**Directory**: `src/proxy_app/ai_assistant/`

| File | Contents |
|------|----------|
| `__init__.py` | Public exports |
| `core.py` | `AIAssistantCore`, `ChatSession`, `Message` classes |
| `tools.py` | `@assistant_tool` decorator, `ToolDefinition`, `ToolResult`, `ToolExecutor` |
| `checkpoint.py` | `CheckpointManager`, `Checkpoint`, delta/snapshot logic |
| `bridge.py` | `LLMBridge`, threading/async coordination |
| `context.py` | `WindowContextAdapter` ABC, context diffing utilities |
| `prompts.py` | Base system prompt constant |

### Phase 2: UI Components

**Directory**: `src/proxy_app/ai_assistant/ui/`

| File | Contents |
|------|----------|
| `__init__.py` | Public exports |
| `chat_window.py` | `AIChatWindow` - main pop-out window |
| `message_view.py` | Canvas-based message display widget |
| `thinking.py` | Collapsible thinking section widget |
| `checkpoint_ui.py` | Checkpoint dropdown/popup widget |
| `model_selector.py` | Grouped model dropdown widget |
| `styles.py` | Colors, fonts, shared constants |

### Phase 3: ModelFilterGUI Integration

**Directory**: `src/proxy_app/ai_assistant/adapters/`

| File | Contents |
|------|----------|
| `__init__.py` | Public exports |
| `model_filter.py` | `ModelFilterWindowContext` implementation, all tools |

**Modifications to existing files**:

| File | Changes |
|------|---------|
| `model_filter_gui.py` | Add button to open AI assistant, wire up context adapter |

### Phase 4: Polish & Edge Cases

| Task | Description |
|------|-------------|
| Checkpoint persistence | Save/load checkpoints to temp file |
| Model list caching | Efficient model list refresh |
| Error handling | Retry logic, user-facing errors |
| Message queue | Queue messages during streaming |
| Silent retry | 4-attempt retry for invalid tools |

---

## 9. File Structure

```
src/proxy_app/
+-- ai_assistant/
|   +-- __init__.py
|   +-- core.py                 # AIAssistantCore, ChatSession
|   +-- tools.py                # Tool decorator and executor
|   +-- checkpoint.py           # CheckpointManager
|   +-- bridge.py               # LLMBridge (RotatingClient wrapper)
|   +-- context.py              # WindowContextAdapter ABC
|   +-- prompts.py              # Base system prompt
|   +-- ui/
|   |   +-- __init__.py
|   |   +-- chat_window.py      # AIChatWindow (pop-out)
|   |   +-- message_view.py     # Message display canvas
|   |   +-- thinking.py         # Collapsible thinking widget
|   |   +-- checkpoint_ui.py    # Checkpoint dropdown/popup
|   |   +-- model_selector.py   # Grouped model dropdown
|   |   +-- styles.py           # UI constants
|   +-- adapters/
|   |   +-- __init__.py
|   |   +-- model_filter.py     # ModelFilterWindowContext
|   +-- DESIGN.md               # This document
+-- model_filter_gui.py         # Modified to include AI assistant button
+-- ...
```

---

## 10. Keyboard Shortcuts

| Shortcut | Context | Action |
|----------|---------|--------|
| `Ctrl+Enter` | Input focused | Send message |
| `Escape` | Input focused | Clear input |
| `Escape` | Streaming | Cancel generation, discard partial response |

---

## 11. Future Considerations (Out of Scope for v1)

These items are noted for future planning but not implemented in v1:

1. **Embedded Compact Mode**: After pop-out is stable, add compact panel for embedding in windows
2. **Conversation Persistence**: Save/load conversation history across sessions
3. **Conversation Export**: Export chat as markdown/text
4. **Custom Model Aliases**: User-defined shortcuts like "smart" -> "openai/gpt-4o"
5. **Multiple Sessions**: Support multiple concurrent assistant windows
6. **Voice Input**: Speech-to-text for input
7. **Image Support**: For multimodal models, support image context

---

## 12. Dependencies

**Required**:
- `customtkinter` - Already used by ModelFilterGUI
- `threading` - Standard library
- `asyncio` - Standard library
- `json` - Standard library
- `hashlib` - For context hashing
- `tempfile` - For checkpoint persistence
- `uuid` - For checkpoint IDs
- `dataclasses` - For data structures
- `abc` - For WindowContextAdapter
- `functools` - For decorator implementation
- `logging` - Standard library, for error/warning logging

**From existing codebase**:
- `rotator_library.client.RotatingClient` - LLM communication
- UI constants from `model_filter_gui.py` - Colors, fonts, etc.

---

## 13. Logging

The AI assistant system logs errors, warnings, and important events to file.

**Log Levels**:

| Level | What is logged |
|-------|----------------|
| ERROR | Tool execution failures, LLM connection errors, checkpoint restore failures |
| WARNING | Invalid tool call retries, context size approaching limits, timeout occurrences |
| INFO | Session start/end, checkpoint creation, model changes |
| DEBUG | Full request/response payloads (disabled by default) |

**Log Location**: Uses existing application logging infrastructure.

**What is NOT logged**:
- Full conversation history (privacy)
- User input content (unless DEBUG level)
- Sensitive context data

Request-response logs from `RotatingClient` already capture LLM interaction details, so the assistant layer focuses on assistant-specific events.

---

## Appendix A: Context Diff Format

When tracking changes between LLM calls, the diff format is:

```python
{
    "changes_since_last_message": [
        {
            "type": "rule_added",
            "rule_type": "ignore",        # or "whitelist"
            "pattern": "gpt-4*",
            "timestamp": "2024-01-15T14:32:15Z"
        },
        {
            "type": "rule_removed",
            "rule_type": "ignore",
            "pattern": "old-pattern*",
            "timestamp": "2024-01-15T14:32:10Z"
        },
        {
            "type": "provider_changed",
            "from": "openai",
            "to": "gemini",
            "timestamp": "2024-01-15T14:31:00Z"
        },
        {
            "type": "models_refreshed",
            "provider": "openai",
            "new_count": 45,
            "timestamp": "2024-01-15T14:30:00Z"
        },
        {
            "type": "search_changed",
            "query": "gpt",
            "timestamp": "2024-01-15T14:29:00Z"
        },
        {
            "type": "changes_saved",
            "timestamp": "2024-01-15T14:28:00Z"
        },
        {
            "type": "changes_discarded",
            "timestamp": "2024-01-15T14:27:00Z"
        }
    ]
}
```

---

## Appendix B: OpenAI Tool Format

Tools are sent to the LLM in OpenAI-compatible format:

```json
{
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "add_ignore_rule",
                "description": "Add a pattern to the ignore list...",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The pattern to ignore..."
                        }
                    },
                    "required": ["pattern"]
                }
            }
        }
    ]
}
```

Tool calls are received as:

```json
{
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "add_ignore_rule",
                "arguments": "{\"pattern\": \"gpt-4*\"}"
            }
        }
    ]
}
```

Tool results are sent back as:

```json
{
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": "{\"success\": true, \"message\": \"Added...\", \"data\": {...}}"
}
```

---

*End of Design Document*

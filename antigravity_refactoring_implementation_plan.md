# Antigravity Provider Refactoring - Implementation Plan

## Overview
This document outlines the implementation plan for HIGH priority refactoring items identified during the code review of the Antigravity provider.

## Scope
Focus on HIGH priority items only (items 1-4 from the refactoring analysis):
1. Extract helper functions into dedicated utility modules
2. Add comprehensive type hints using TypedDict for complex nested structures
3. Implement proper input validation for public methods with clear error messages
4. Replace broad exception handlers with specific exception types and proper error propagation

## Implementation Strategy

### Phase 1: Create Utility Modules (Item #1)
**Estimated Lines Affected:** ~400 lines  
**Risk Level:** LOW  
**Benefits:** Reduces main file size, improves testability, enables code reuse

#### Step 1.1: Create utility module structure
```
src/rotator_library/providers/antigravity_utils/
├── __init__.py
├── request_helpers.py      # Request ID, session ID, project ID generators
├── schema_transformers.py  # Schema normalization and cleaning functions
└── json_parsers.py         # JSON parsing and transformation utilities
```

#### Step 1.2: Extract functions to `request_helpers.py`
- `_generate_request_id()` (line 251-253)
- `_generate_session_id()` (line 256-259)  
- `_generate_project_id()` (line 262-266)

#### Step 1.3: Extract functions to `schema_transformers.py`
- `_normalize_type_arrays()` (line 269-285)
- `_clean_claude_schema()` (line 368-419)

#### Step 1.4: Extract functions to `json_parsers.py`
- `_recursively_parse_json_strings()` (line 288-365)

#### Step 1.5: Update imports in `antigravity_provider.py`
Add imports for extracted utilities while maintaining existing functionality.

---

### Phase 2: Add Type Definitions (Item #2)
**Estimated Signatures Affected:** ~30 methods  
**Risk Level:** LOW  
**Benefits:** Improved IDE support, type safety, documentation

#### Step 2.1: Create type definitions file
```
src/rotator_library/providers/antigravity_types.py
```

#### Step 2.2: Define core types
```python
from typing import TypedDict, Literal, List, Dict, Any, Optional

# Gemini API types
class GeminiPart(TypedDict, total=False):
    text: str
    inlineData: Dict[str, str]
    functionCall: Dict[str, Any]
    functionResponse: Dict[str, Any]
    thought: bool
    thoughtSignature: str

class GeminiContent(TypedDict):
    role: Literal["user", "model"]
    parts: List[GeminiPart]

class SystemInstruction(TypedDict):
    role: Literal["user"]
    parts: List[GeminiPart]

# Antigravity envelope types
class AntigravityRequest(TypedDict):
    project: str
    userAgent: str
    requestId: str
    model: str
    request: Dict[str, Any]

# Response types
class UsageMetadata(TypedDict, total=False):
    promptTokenCount: int
    thoughtsTokenCount: int
    candidatesTokenCount: int
    totalTokenCount: int
```

#### Step 2.3: Update key method signatures
Priority methods to type:
- `_transform_messages()` (line 1034)
- `_transform_to_antigravity_format()` (line 1672)
- `_gemini_to_openai_chunk()` (line 1794)
- `_gemini_to_openai_non_streaming()` (line 1895)
- `acompletion()` (line 2138)

---

### Phase 3: Input Validation (Item #3)
**Estimated Lines Added:** ~80 lines  
**Risk Level:** LOW  
**Benefits:** Clear error messages, fail-fast behavior

#### Step 3.1: Add validation helper
```python
def _validate_completion_params(
    self,
    model: str,
    messages: List[Dict[str, Any]],
    **kwargs
) -> None:
    """Validate completion request parameters."""
    if not messages:
        raise ValueError("messages parameter is required and cannot be empty")
    
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")
    
    if not all(isinstance(msg, dict) for msg in messages):
        raise TypeError("All messages must be dictionaries")
    
    if not model:
        raise ValueError("model parameter is required")
    
    # Validate model is supported
    internal_model = self._alias_to_internal(model)
    supported = set(AVAILABLE_MODELS + list(MODEL_ALIAS_MAP.keys()) + list(MODEL_ALIAS_MAP.values()))
    if internal_model not in supported and model not in supported:
        raise ValueError(
            f"Unsupported model: {model}. "
            f"Supported models: {', '.join(AVAILABLE_MODELS)}"
        )
```

#### Step 3.2: Add validation to public methods
- `acompletion()` - add parameter validation at start
- `get_models()` - validate api_key parameter
- `count_tokens()` - validate required parameters

---

### Phase 4: Error Handling (Item #4)
**Estimated Lines Affected:** ~50 lines  
**Risk Level:** MEDIUM  
**Benefits:** Better error messages, proper exception hierarchy

#### Step 4.1: Replace broad exception handlers in `acompletion()`
Current code (line 2271-2284):
```python
except Exception as e:  # TOO BROAD
    if self._try_next_base_url():
        # retry
    raise
```

Proposed:
```python
except httpx.HTTPStatusError as e:
    if e.response.status_code >= 500:
        if self._try_next_base_url():
            lib_logger.warning(f"Server error, retrying with fallback URL: {e}")
            # retry logic
        raise litellm.ServiceUnavailableError(
            f"Antigravity API server error: {e.response.status_code}"
        )
    elif e.response.status_code == 401:
        raise litellm.AuthenticationError(f"Invalid authentication credentials: {e}")
    elif e.response.status_code == 429:
        raise litellm.RateLimitError(f"Rate limit exceeded: {e}")
    raise litellm.APIError(f"API request failed: {e}")
except httpx.ConnectError as e:
    if self._try_next_base_url():
        lib_logger.warning(f"Connection failed, trying fallback URL: {e}")
        # retry logic
    raise litellm.APIConnectionError(f"Failed to connect to Antigravity API: {e}")
except httpx.TimeoutException as e:
    raise litellm.Timeout(f"Request timeout: {e}")
```

#### Step 4.2: Update other exception handlers
- `get_models()` (line 2104-2136)
- `count_tokens()` (line 2401-2449)
- `_handle_non_streaming()` (line 2305-2325)

---

## Implementation Order

### Week 1: Utility Extraction
- [ ] Day 1-2: Create utility module structure
- [ ] Day 2-3: Extract and test helper functions
- [ ] Day 3-4: Update imports and verify functionality

### Week 2: Type Safety
- [ ] Day 1-2: Define TypedDict classes
- [ ] Day 2-4: Update method signatures
- [ ] Day 4-5: Verify type checking works

### Week 3: Validation & Error Handling
- [ ] Day 1-2: Implement input validation
- [ ] Day 3-4: Update exception handlers
- [ ] Day 5: Integration testing

---

## Testing Strategy

Since the project has no formal test framework:

### Manual Testing Checklist
- [ ] Test with Gemini 2.5 models (Pro, Flash)
- [ ] Test with Gemini 3 models
- [ ] Test with Claude Sonnet 4.5
- [ ] Test streaming responses
- [ ] Test non-streaming responses
- [ ] Test tool calling
- [ ] Test error scenarios (invalid model, empty messages, auth errors)
- [ ] Verify base URL fallback logic
- [ ] Test thinking/reasoning features

### Regression Prevention
- Document current behavior before changes
- Test against existing conversation logs
- Verify OAuth token handling still works

---

## Rollback Strategy

If issues arise:
1. All changes are isolated to specific modules
2. Git history allows clean reversion
3. Each phase is independently reversible
4. Original code preserved through git tags

---

## Success Metrics

- ✅ Main file reduced by ~400 lines (16% reduction)
- ✅ Zero runtime errors after refactoring
- ✅ Type checking passes with mypy/pyright
- ✅ All manual tests pass
- ✅ Error messages are clear and actionable
- ✅ No performance regression

---

## Notes

- Keep backwards compatibility - no API changes
- Maintain async/await patterns throughout
- Follow existing code style and conventions
- Update docstrings as methods change
- Use `lib_logger` for all logging
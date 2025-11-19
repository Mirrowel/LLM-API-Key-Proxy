# HiveMind (Swarm/Fusion) - Implementation Plan (REVISED)

## Goal Description

Implement a sophisticated orchestration engine called "HiveMind" that enables two distinct modes of parallel model execution:

1. **Swarm Mode**: Multiple parallel calls to the **same model** (called "Drones") with optional configuration for temperature variation, adversarial critique, and recursive self-correction.
2. **Fusion Mode**: Multiple parallel calls to **different models** (called "Models" or "Specialists" when roles are assigned) with optional role-based routing and context-aware synthesis.

Both modes use an "Arbiter" (judge model) to synthesize responses with configurable strategies and optional recursive refinement.

---

## Terminology

- **HiveMind**: The overall feature/system
- **Swarm**: Parallel execution of the same model
  - **Drone**: Individual instance in a Swarm
- **Fusion**: Parallel execution of different models  
  - **Model**: Individual model in a Fusion (generic term)
  - **Specialist**: A Model with an assigned role and weight
- **Arbiter**: The judge/synthesizer model that produces the final response

---

## Architecture Overview

### Request Flow

```
User Request (model: "gemini-1.5-flash[swarm]")
    ↓
EnsembleManager.is_ensemble()? → Yes
    ↓
EnsembleManager.handle_request()
    ↓
┌─────────────────────────────────────────┐
│ 1. Configuration Resolution             │
│    - Load config for this ensemble      │
│    - Determine: Swarm or Fusion?        │
│    - Get Arbiter config                 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. Drone/Model Preparation              │
│    For Swarm:                           │
│      - Create N Drones (same model)     │
│      - Apply temp jitter (optional)     │
│      - Mark M as adversarial (optional) │
│    For Fusion:                          │
│      - Load constituent models          │
│      - Apply role prompts (optional)    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Parallel Execution                   │
│    - asyncio.gather() all calls         │
│    - Each call uses RotatingClient      │
│    - Apply retry logic per drone/model  │
│    - Collect responses + metadata       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Response Processing                  │
│    - Apply blind switch (optional)      │
│    - Format for Arbiter consumption     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. Arbitration                          │
│    - Load strategy prompt               │
│    - Inject role/weight context         │
│    - For Recursive Mode:                │
│      • Give arbiter autonomy            │
│      • Arbiter decides Round 2          │
│    - For Non-Recursive:                 │
│      • Direct synthesis only            │
│    - Call Arbiter (with streaming)      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 6. Final Output                         │
│    - Stream Arbiter's response to user  │
│    - Aggregate usage from all calls     │
│    - Log execution summary              │
└─────────────────────────────────────────┘
```

---

## Core Components

### 1. EnsembleManager Class

**File**: `src/rotator_library/ensemble_manager.py`

**Responsibilities**:
- Load and validate `ensemble_config.json`
- Detect Swarm requests (`[swarm]` notation) vs Fusion requests (config-based)
- Orchestrate parallel execution with retry logic
- Manage arbitration with streaming support
- Handle recursive refinement (single arbiter call with autonomous decision)

**Key Methods**:

#### `__init__(self, config_path, rotating_client)`
- Load configuration file
- Store reference to RotatingClient
- Build lookup tables for fast ensemble detection
- Validate configuration schema
- Initialize usage aggregator

#### `is_ensemble(self, model_id: str) -> bool`
- Check if model_id matches a Fusion config (exact match from config)
- Check if model_id contains `[swarm]` notation
- Handle conflict detection (if provider has real model with same name)
- Return: `True` if ensemble, `False` otherwise

#### `resolve_conflicts(self, base_model: str) -> str`
- Default format: `base_model[swarm]`
- Check if this conflicts with provider's real models
- If conflict, try: `base_model[hive]`, `base_model[max]`, etc.
- Log warning about conflict resolution
- Return: Final ensemble ID to use

#### `handle_request(self, request_params: dict) -> AsyncGenerator`
Main orchestration method. Returns a streaming generator for the Arbiter's response.

**Steps**:
1. **Identify Type**: Swarm or Fusion
2. **Load Config**: Get specific config or use defaults
3. **Prepare Drones/Models**:
   - Build list of execution targets
   - Apply temperature jitter (Swarm)
   - Apply role prompts (Fusion)
   - Mark adversarial instances
4. **Execute Parallel Calls**:
   - Use `asyncio.gather()` with exception handling
   - Each call goes through RotatingClient (inherits retry logic)
   - Require at least 1 successful response
   - Log failures as errors
5. **Aggregate Usage**:
   - Sum all `prompt_tokens`, `completion_tokens`, `total_tokens`
   - Calculate combined cost (using existing cost calculation)
6. **Process Responses**:
   - Extract content from each response
   - Apply blind switch if enabled (keep roles, strip model names)
   - Format for Arbiter
7. **Build Arbiter Prompt**:
   - Load strategy prompt template
   - Inject adversarial context (if applicable)
   - Inject role/weight context (Fusion)
   - For recursive mode: Add autonomous decision instructions
8. **Call Arbiter with Streaming**:
   - Stream Arbiter's synthesis to user
   - Parse internal markers (if recursive mode)
   - Aggregate Arbiter's usage into total
9. **Return**: Stream final response with combined usage metadata

#### `_prepare_drones(self, config: dict, base_model: str, request_params: dict) -> List[dict]`
For Swarm mode:
- Create N copies of request params
- **Temperature Jitter**:
  ```python
  base_temp = request_params.get('temperature', 0.7)
  jitter_config = config.get('temperature_jitter', {})
  if jitter_config.get('enabled', False):
      delta = jitter_config.get('delta', 0.0)
      for i in range(count):
          temp = base_temp + random.uniform(-delta, delta)
          temp = max(0.0, min(2.0, temp))  # Clamp
          drones[i]['temperature'] = temp
  ```
- **Adversarial Prompts**:
  ```python
  adv_config = config.get('adversarial_config', {})
  if adv_config.get('enabled', False):
      count = adv_config['count']
      prompt = adv_config['prompt']
      for i in range(count):
          drones[i]['messages'].insert(0, {
              'role': 'system',
              'content': prompt
          })
          drones[i]['_is_adversarial'] = True  # Metadata for logging
  ```
- **Model ID**: All drones use `base_model` (without `[swarm]` suffix)

#### `_prepare_models(self, config: dict, request_params: dict) -> List[dict]`
For Fusion mode:
- For each model in fusion config:
  - Clone request params
  - Set model ID from config
  - If role defined:
    - Apply `system_prompt_append` (prepend to messages)
    - Store role metadata for context
  - If weight defined:
    - Store weight for arbiter context
- Return list of prepared calls with metadata

#### `_execute_parallel(self, prepared_calls: List[dict]) -> Tuple[List[dict], dict]`
- Execute all calls in parallel:
  ```python
  results = await asyncio.gather(
      *[self.rotating_client.acompletion(**params) for params in prepared_calls],
      return_exceptions=True
  )
  ```
- Filter out exceptions/None values
- Log each failure as ERROR (drones should not fail)
- Require at least 1 success, else raise exception
- Aggregate usage:
  ```python
  total_usage = {
      'prompt_tokens': sum(r.usage.prompt_tokens for r in results if r),
      'completion_tokens': sum(r.usage.completion_tokens for r in results if r),
      'total_tokens': sum(r.usage.total_tokens for r in results if r)
  }
  ```
- Return: `(successful_responses, total_usage)`

#### `_format_for_arbiter(self, responses: List[dict], config: dict, mode: str, metadata: List[dict]) -> str`
Build formatted text for arbiter input.

**Blind Switch Logic**:
- If `blind=True`:
  - Labels: "Response 1 (Architect role)", "Response 2 (Security role)"
  - Do NOT include model names
- If `blind=False`:
  - Labels: "Response 1 (GPT-4o - Architect)", "Response 2 (Claude-3-opus - Security)"

**Adversarial Context** (if adversarial drones present):
```
NOTE: Responses marked [ADVERSARIAL] were specifically prompted to critique and find flaws.
Their purpose is to stress-test the solution. Consider their critiques when synthesizing.
```

**Format**:
```
Response 1 (GPT-4o - Architect):
[content]

Response 2 (Claude-3-opus - Security):
[content]

Response 3 [ADVERSARIAL]:
[content]
```

#### `_build_arbiter_prompt(self, formatted_responses: str, config: dict, mode: str) -> List[dict]`
Build complete messages array for arbiter.

**System Prompt Components**:
1. **Base Strategy**: Load from `arbitration_strategies[strategy_name]`
2. **Role/Weight Context** (Fusion only):
   ```
   You are synthesizing responses from specialists with the following expertise:
   - GPT-4o (Architect): Expert in system design and scalability. Trust this model for architectural decisions.
   - Claude-3-opus (Security): Expert in vulnerability assessment. Trust this model for security concerns.
   ```
3. **Adversarial Context** (if applicable):
   ```
   Some responses are marked [ADVERSARIAL]. These drones were specifically instructed to critique
   and find edge cases. Their purpose is quality assurance through skeptical analysis.
   ```
4. **Recursive Mode Instructions** (if enabled):
   ```
   AUTONOMOUS DECISION PROTOCOL:
   1. Analyze the responses and assess consensus (agreement level 1-10)
   2. If consensus >= 7/10: Proceed directly to synthesis
   3. If consensus < 7/10:
      a. Identify specific conflict points
      b. Internally trigger a critique phase
      c. For each response, reason about how it would address the conflicts
      d. Then synthesize the final answer
   
   Log your internal reasoning with markers:
   [CONSENSUS: X/10]
   [CONFLICTS: bullet list]
   [CRITIQUE REASONING: ...]
   [FINAL SYNTHESIS:]
   
   IMPORTANT: Only return the FINAL SYNTHESIS to the user. All internal reasoning
   should be wrapped in [INTERNAL] tags for logging purposes only.
   ```
5. **Output Format**:
   ```
   Provide your synthesis as a complete, high-quality response to the user's original query.
   Do not mention that you are combining responses unless directly relevant.
   ```

**User Message**: Original user query + formatted responses

Return: Complete messages array for arbiter call

#### `_call_arbiter_streaming(self, messages: List[dict], arbiter_model: str, original_params: dict) -> AsyncGenerator`
Call arbiter and stream response.

- Clone original request params
- Set model to `arbiter_model`
- Set `messages` to constructed arbiter prompt
- Set `stream=True`
- Call via RotatingClient.acompletion (returns async generator)
- **Parse Stream**:
  - Extract internal markers (consensus score, conflicts) for logging
  - Strip `[INTERNAL]` sections from user-facing output
  - Yield only synthesis content to user
- **Aggregate Usage**: Track arbiter's usage separately
- Return: Streaming generator

---

### 2. Configuration Structure

**Folder-Based Approach**: Instead of a single config file, HiveMind uses a directory structure:

```
ensemble_configs/
├── swarms/
│   ├── default.json          # Default swarm settings
│   ├── gemini-flash.json     # Custom swarm for gemini-flash
│   └── gpt4o.json            # Custom swarm for gpt-4o
├── fusions/
│   ├── dev-team.json         # Dev team fusion
│   └── creative-writers.json # Creative writers fusion
└── strategies/
    ├── synthesis.txt         # Synthesis strategy prompt
    ├── best_of_n.txt         # Best-of-N strategy
    └── code_review.txt       # Code review strategy
```

**Loading Logic**:
- Load all JSON files from each subfolder
- Merge swarm configs (specific model configs override defaults)
- Detect duplicate fusion IDs → apply conflict resolution
- Load strategy templates from `.txt` files

**Benefits**:
- Easy to add new configs (drop file in folder)
- Version control friendly (one file per fusion/config)
- Community sharing (share individual fusion configs)

---

### 3. Configuration Schemas

#### Swarm Config

**File**: `ensemble_configs/swarms/default.json`

```json
{
  "suffix": "[swarm]",
  "count": 3,
  
  "temperature_jitter": {
    "enabled": true,
    "delta": 0.2
  },
  
  "arbiter": {
    "model": "self",
    "strategy": "synthesis",
    "blind": true,
    "note": "Arbiter should be a decent reasoning model (e.g., GPT-4o, Claude 3+, Gemini 1.5 Pro+)"
  },
  
  "adversarial_config": {
    "enabled": false,
    "count": 1,
    "prompt": "You are a Senior Principal Engineer with 15+ years of experience..."
  },
  
  "recursive_mode": {
    "enabled": false,
    "consensus_threshold": 7,
    "note": "Requires a reasoning-capable arbiter model"
  }
}
```

#### Model-Specific Swarm Config

**File**: `ensemble_configs/swarms/gemini-flash.json`

```json
{
  "model": "gemini-1.5-flash",
  "arbiter": {
    "model": "gpt-4o",
    "strategy": "synthesis",
    "blind": true
  }
}
```

#### Fusion Config

**File**: `ensemble_configs/fusions/dev-team.json`

```json
{
  "id": "dev-team",
  "description": "A team of specialized models for software development",
  "models": [
    {
      "model": "gpt-4o",
      "role": "Architect",
      "system_prompt_append": "Focus on architectural patterns, scalability, and system design.",
      "weight": "Expert in system design and scalability. Trust for architectural decisions and structural integrity."
    },
    {
      "model": "claude-3-opus",
      "role": "Security Specialist",
      "system_prompt_append": "Focus on security vulnerabilities, edge cases, and potential exploits.",
      "weight": "Expert in security and vulnerability assessment. Trust for identifying security flaws and attack vectors."
    },
    {
      "model": "gemini-1.5-pro",
      "role": "Code Reviewer",
      "system_prompt_append": "Focus on code quality, performance, and best practices.",
      "weight": "Expert in code quality and performance optimization. Trust for maintainability and efficiency concerns."
    }
  ],
  "arbiter": {
    "model": "gpt-4o",
    "strategy": "synthesis",
    "blind": true,
    "note": "Requires a reasoning-capable model for best results"
  },
  "recursive_mode": {
    "enabled": false,
    "consensus_threshold": 7
  }
}
```

#### Strategy Template

**File**: `ensemble_configs/strategies/synthesis.txt`

```
You are an expert synthesizer. Analyze the following responses and create a single, superior answer that:
1. Combines the best elements from each response
2. Resolves any conflicts or contradictions
3. Ensures completeness and accuracy
4. Maintains coherence and clarity

Your goal is to produce the BEST possible answer by leveraging the strengths of each response.

Responses:
{responses}
```


---

## Detailed Feature Specifications

### 1. Temperature Jitter (Swarm Only)

**Purpose**: Introduce controlled randomness to increase response diversity.

**Configuration**:
```json
"temperature_jitter": {
  "enabled": true,
  "delta": 0.2
}
```

**Implementation**:
- Get base temperature from request (default 0.7)
- For each Drone: `temp = base_temp + random.uniform(-delta, delta)`
- Clamp to `[0.0, 2.0]`
- If request has `temperature=0`, disable jitter automatically

---

### 2. Adversarial Mode (Swarm Only)

**Purpose**: Inject critical analysis to stress-test solutions.

**Configuration**:
```json
"adversarial_config": {
  "enabled": false,
  "count": 1,
  "prompt": "You are a Senior Principal Engineer..."
}
```

**Implementation**:
- Select first N drones as adversarial
- Prepend adversarial system prompt
- Tag responses as `[ADVERSARIAL]` in arbiter input
- **Arbiter Context**: Explain adversarial purpose:
  ```
  NOTE: This mode is designed for SYNTHESIS strategy. Adversarial responses
  critique the solution to ensure all angles are considered. Integrate their
  insights to strengthen the final answer.
  ```

---

### 3. Role Assignment & Weights (Fusion Only)

**Purpose**: Specialize models and guide arbiter on expertise.

**Configuration** (per model):
```json
{
  "model": "gpt-4o",
  "role": "Architect",
  "system_prompt_append": "Focus on scalability.",
  "weight": "Expert in system design. Trust for architectural decisions."
}
```

**Fields**:
- `role`: Display name (for user reference and arbiter labels)
- `system_prompt_append`: Instructions sent to the model
- `weight`: Context for arbiter (what to trust this model for)

**Arbiter Context Injection**:
```
Specialist Expertise:
- Architect (GPT-4o): Expert in system design. Trust for architectural decisions.
- Security (Claude): Expert in vulnerabilities. Trust for security concerns.
```

---

### 4. Arbitration Strategies

**Purpose**: Flexible synthesis logic via prompt engineering.

**Built-in**:
- `synthesis`: Combine all responses into best version
- `best_of_n`: Select and refine the strongest response
- `code_review`: Code-specific evaluation

**User-Defined**: Users add custom strategies to `arbitration_strategies` config.

**Template Variables**:
- `{responses}`: Formatted response text
- `{role_context}`: Weight/expertise descriptions
- `{adversarial_note}`: Context about adversarial drones

---

### 5. Blind Switch

**Purpose**: Remove model identifiers to prevent bias, while keeping role context.

**Default**: `blind: true` (enabled by default)

**Per-Config**: Each swarm config and fusion config can override:

```json
"arbiter": {
  "blind": true
```

**Implementation**:
- `blind=true`: "Response 1 (Architect role)", "Response 2 (Security role)"
- `blind=false`: "Response 1 (GPT-4o - Architect)", "Response 2 (Claude - Security)"

**Key Change**: Roles are ALWAYS preserved. Only model names are stripped.

---

### 6. Recursive/Reflective Mode

**Purpose**: Multi-round refinement for low-consensus situations.

**Configuration**:
```json
"recursive_mode": {
  "enabled": false,
  "consensus_threshold": 7,
  "note": "Arbiter model must be capable of internal reasoning (e.g., GPT-4o, Claude 3.5+, Gemini 1.5 Pro+)"
}
```

**REVISED APPROACH** (Single Arbiter Call):

Instead of multiple requests, the arbiter is given **autonomous decision-making** via prompt.

> [!NOTE]
> The arbiter model should be a **decent reasoning model** to handle internal critique and consensus analysis effectively. Models like GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro are recommended.

**Arbiter Prompt** (when recursive enabled):
```
You have autonomous decision-making authority. Follow this protocol:

1. ASSESSMENT PHASE:
   - Analyze the provided responses
   - Rate consensus level (1-10)
   - Log: [CONSENSUS: X/10]

2. DECISION PHASE:
   If consensus >= 7/10:
     - Proceed directly to synthesis
   
   If consensus < 7/10:
     - Identify conflict points
     - Log: [CONFLICTS: ...]
     - For each response, reason internally about how it would address conflicts
     - Log: [CRITIQUE REASONING: ...]
   
3. SYNTHESIS PHASE:
   - Create final answer incorporating all insights
   - Log: [FINAL SYNTHESIS:]

IMPORTANT: Wrap all internal reasoning in [INTERNAL] tags. Only the content
after [FINAL SYNTHESIS:] will be shown to the user.
```

**Stream Processing**:
- EnsembleManager parses the stream
- Extract `[CONSENSUS: X/10]` → Log at WARN level if < threshold
- Extract `[CONFLICTS: ...]` → Log conflicts
- Strip all `[INTERNAL]` sections from user output
- Yield only `[FINAL SYNTHESIS:]` content to user

**Logging**:
```
[HiveMind] Recursive mode active. Consensus: 5/10 [WARN]
[HiveMind] Conflicts identified: [list]
[HiveMind] Arbiter performing internal critique...
[HiveMind] Final synthesis complete
```

---

### 7. Streaming Support

**Behavior**: Respects the `stream` boolean from the original request.

**Implementation**:
- Drone/Model calls are NOT streamed (collected in parallel)
- Arbiter call respects `stream` parameter:
  - If `stream=true`: Stream arbiter's response
  - If `stream=false`: Return complete arbiter response
- EnsembleManager passes through arbiter's streaming behavior
- Parse and filter internal markers during streaming
- Return clean synthesis to user

**Flow**:
```python
async def handle_request(...) -> AsyncGenerator:
    # 1. Collect drone responses (non-streaming)
    responses = await self._execute_parallel(...)
    
    # 2. Build arbiter prompt
    messages = self._build_arbiter_prompt(...)
    
    # 3. Stream arbiter response
    arbiter_stream = self._call_arbiter_streaming(...)
    
    # 4. Parse and yield
    async for chunk in arbiter_stream:
        # Filter [INTERNAL] sections
        if not chunk.startswith('[INTERNAL]'):
            yield chunk
```

---

### 8. Usage & Cost Tracking

**Aggregation**:
- Track usage from each Drone/Model call
- Track usage from Arbiter call
- Sum ALL usage fields:
  ```python
  total_usage = {
      'prompt_tokens': sum(all_calls),
      'completion_tokens': sum(all_calls),
      'cached_tokens': sum(all_calls),  # If available
      'reasoning_tokens': sum(all_calls),  # If available
      'total_tokens': sum(all_calls),
      # Include any other usage fields from responses
  }
  ```

**Cost Calculation**:
- Use `UsageManager.calculate_cost()` if available (preferred)
- Fallback to `litellm.completion_cost()` if needed
- Calculate cost per call
- Sum total cost
- **Note**: This should be one of the last features to implement
- Include in final response metadata

**Response Format**:
```json
{
  "usage": {
    "prompt_tokens": 5000,
    "completion_tokens": 800,
    "total_tokens": 5800,
    "hivemind_details": {
      "drone_count": 3,
      "arbiter_tokens": 1200,
      "total_cost_usd": 0.045
    }
  }
}
```

---

## Integration Points

### 1. RotatingClient Modification

**File**: `src/rotator_library/client.py`

```python
class RotatingClient:
    def __init__(self, ...):
        # Existing init
        self.ensemble_manager = EnsembleManager(
            config_path=os.path.join(os.path.dirname(__file__), '../../ensemble_config.json'),
            rotating_client=self
        )
    
    def acompletion(self, request=None, **kwargs):
        model = kwargs.get('model')
        
        # Check if ensemble
        if self.ensemble_manager.is_ensemble(model):
            # Return streaming generator from ensemble manager
            return self.ensemble_manager.handle_request(
                request=request,
                **kwargs
            )
        
        # Normal flow
        if kwargs.get('stream'):
            return self._streaming_acompletion_with_retry(...)
        else:
            return self._execute_with_retry(...)
```

---

### 2. Model List Integration

```python
async def get_all_available_models(self, grouped=True):
    # Existing provider models
    all_provider_models = await self._fetch_provider_models()
    
    # Add fusion models
    fusion_ids = self.ensemble_manager.get_fusion_ids()
    if fusion_ids:
        all_provider_models['hivemind'] = fusion_ids
    
    return all_provider_models
```

**Note**: Swarm model listing is **TBD**. The user notes it's "not infinite" and needs to design a better discovery system.

---

### 3. Logging

**Log Levels**:
- INFO: Normal operations (starting swarm, drone count, completion)
- DEBUG: Detailed execution (per-drone temps, prompt construction)
- WARN: Low consensus, conflicts, partial failures
- ERROR: Drone failures, arbiter failures

**Examples**:
```python
lib_logger.info(f"[HiveMind] Processing Swarm: {model_id} ({count} Drones)")
lib_logger.debug(f"[HiveMind] Drone {i+1}: temp={temp:.2f}, adversarial={is_adv}")
lib_logger.warn(f"[HiveMind] Recursive mode: Consensus 5/10 - below threshold")
lib_logger.error(f"[HiveMind] Drone {i+1} failed: {error}")
lib_logger.info(f"[HiveMind] Total cost: ${total_cost:.4f} ({total_tokens} tokens)")
```

---

## Edge Cases & Error Handling

### 1. Partial Failures

**Scenario**: Some Drones fail due to errors.

**Handling**:
- Each drone call uses RotatingClient → **inherits existing retry/key rotation logic**
- If a drone still fails after retries, log as ERROR
- Continue with successful responses
- **Minimum**: Require at least 1 successful response
- If all fail, raise exception with details

**No Special Logic Needed**: RotatingClient already handles retries, rate limits, key rotation.

---

### 2. Arbiter Failure

**Scenario**: Arbiter call fails.

**Handling**:
- Arbiter call uses RotatingClient → **inherits retry/resilience logic**
- If arbiter fails after retries:
  - Log ERROR
  - Fallback: Return first **non-adversarial** drone response
  - Log: `[HiveMind] Arbiter failed. Returning first non-adversarial response.`

---

### 3. Naming Conflicts

**Scenario**: Provider has `gemini-1.5-flash[swarm]` as real model, or duplicate fusion IDs exist.

**Handling**:
- Default naming: `model-name[swarm]` or fusion ID from config
- On conflict detected:
  - Append numeric suffix: `-1`, `-2`, `-3`, etc.
  - Example: `gemini-1.5-flash[swarm]` → `gemini-1.5-flash[swarm]-1`
  - Example: `dev-team` → `dev-team-1`
- Log: `[HiveMind] Conflict detected. Renamed 'dev-team' to 'dev-team-1'.`
- Store resolved names in runtime cache
- **Applies to**: Both swarm suffixes AND fusion IDs

---

### 4. Streaming Parse Errors

**Scenario**: Can't parse `[CONSENSUS: X/10]` from recursive mode stream.

**Handling**:
- Log warning
- Continue streaming synthesis
- Skip logging consensus score

---

### 5. Invalid Configuration

**Scenario**: User config has invalid fusion (missing model, invalid strategy).

**Handling**:
- On startup, validate all fusions
- Log errors for invalid configs
- Skip invalid fusions
- Continue with valid ones

---

## Implementation Phases

### **Phase 1: Foundation (Core Infrastructure)**

**Goal**: Set up basic structure and config loading.

**Tasks**:
1. Create `ensemble_manager.py` skeleton
   - Define `EnsembleManager` class
   - Implement `__init__` with folder-based config loading
   - Load and merge configs from `ensemble_configs/` directory
   - Add config validation (JSON schema)
   
2. Create config directory structure
   - `ensemble_configs/swarms/default.json`
   - `ensemble_configs/fusions/` (empty initially)
   - `ensemble_configs/strategies/synthesis.txt`

3. Integrate into `RotatingClient`
   - Import `EnsembleManager`
   - Initialize in `__init__` with config directory path
   - Add placeholder check in `acompletion`

4. Implement `is_ensemble()`
   - Detect `[swarm]` suffix
   - Detect fusion IDs from config
   - Add conflict detection logic

**Deliverables**:
- ✅ Folder-based config structure created
- ✅ Configs load and merge correctly
- ✅ Ensemble detection works
- ✅ Conflict resolution (numeric suffixes) works
- ✅ No runtime errors

**Testing**:
- Unit test folder-based config loading
- Unit test config merging (swarm defaults + model-specific)
- Unit test `is_ensemble()` with various inputs
- Test conflict detection and numeric suffix generation
- Test duplicate fusion ID handling

---

### **Phase 2: Basic Swarm (Non-Streaming)**

**Goal**: Get basic swarm working without advanced features.

**Tasks**:
1. Implement `_prepare_drones()`
   - Clone request params N times
   - Set model to base (strip `[swarm]`)
   - No jitter or adversarial yet

2. Implement `_execute_parallel()`
   - Use `asyncio.gather()` with drone calls
   - Handle exceptions gracefully
   - Aggregate usage stats

3. Implement `_format_for_arbiter()`
   - Basic formatting (numbered responses)
   - No blind switch yet

4. Implement `_build_arbiter_prompt()`
   - Load synthesis strategy
   - Simple system prompt + user message
   - No recursive mode yet

5. Implement `_call_arbiter()` (NON-streaming first)
   - Call arbiter via RotatingClient
   - Return complete response
   - Aggregate arbiter usage

6. Wire up `handle_request()` (non-streaming)
   - Connect all steps
   - Return arbiter's response
   - Include combined usage

**Deliverables**:
- ✅ Swarm executes 3 drones in parallel
- ✅ Arbiter synthesizes responses
- ✅ Final response returned (non-streaming)
- ✅ Usage aggregated correctly

**Testing**:
- Integration test: Call `gemini-1.5-flash[swarm]`
- Verify 3 drone calls + 1 arbiter call
- Verify synthesis quality (manual)
- Verify usage statistics

---

### **Phase 3: Streaming Support**

**Goal**: Enable streaming for arbiter response.

**Tasks**:
1. Modify `_call_arbiter()` to `_call_arbiter_streaming()`
   - Set `stream=True`
   - Return async generator
   - Track usage from stream

2. Update `handle_request()` to return generator
   - Yield arbiter stream chunks
   - Aggregate usage at end

3. Test streaming end-to-end
   - Verify chunks arrive in real-time
   - Verify complete response matches non-streaming

**Deliverables**:
- ✅ Arbiter response streams to user
- ✅ No buffering of full response
- ✅ Usage still aggregated correctly

**Testing**:
- Integration test with streaming
- Compare output to non-streaming version
- Test error handling mid-stream

---

### **Phase 4: Advanced Swarm Features**

**Goal**: Add jitter, adversarial, blind switch.

**Tasks**:
1. **Temperature Jitter**:
   - Add jitter logic to `_prepare_drones()`
   - Test with different delta values
   - Verify clamping

2. **Adversarial Mode**:
   - Inject adversarial prompts
   - Tag responses in formatting
   - Add arbiter context explanation

3. **Blind Switch**:
   - Modify `_format_for_arbiter()`
   - Strip model names when `blind=true`
   - Keep roles always

**Deliverables**:
- ✅ Jitter produces varied temps
- ✅ Adversarial drones produce critiques
- ✅ Blind mode strips model names

**Testing**:
- Test each feature independently
- Test combinations (jitter + adversarial)
- Manual review of adversarial effectiveness

---

### **Phase 5: Fusion Mode**

**Goal**: Enable multi-model mixtures with roles.

**Tasks**:
1. Implement `_prepare_models()`
   - Load models from fusion config
   - Apply role system prompts
   - Store metadata for arbiter

2. Update `_format_for_arbiter()` for roles
   - Include role labels
   - Apply blind switch for model names

3. Implement role/weight context injection
   - Build specialist expertise text
   - Inject into arbiter system prompt

4. Add example fusion to config
   - "dev-team" with 3 specialists

**Deliverables**:
- ✅ Fusion calls multiple models
- ✅ Arbiter receives role context
- ✅ Synthesis respects expertise weights

**Testing**:
- Test "dev-team" fusion with coding question
- Verify role prompts are applied
- Manual review: Does arbiter trust specialists appropriately?

---

### **Phase 6: Recursive Mode**

**Goal**: Enable autonomous arbiter decision-making for low consensus.

**Tasks**:
1. Update `_build_arbiter_prompt()` for recursive
   - Add autonomous protocol instructions
   - Define `[INTERNAL]` marker format
   - Include consensus threshold

2. Implement stream parsing in `_call_arbiter_streaming()`
   - Extract `[CONSENSUS: X/10]`
   - Extract `[CONFLICTS: ...]`
   - Strip `[INTERNAL]` sections from user output

3. Add logging for recursive flow
   - Log consensus score at WARN if low
   - Log identified conflicts
   - Log critique phase activation

**Deliverables**:
- ✅ Arbiter autonomously decides Round 2
- ✅ Internal reasoning logged but not shown to user
- ✅ Low consensus triggers critique

**Testing**:
- Test with intentionally ambiguous prompt
- Verify arbiter produces `[CONSENSUS: 4/10]`
- Verify critique reasoning appears in logs
- Verify final synthesis is improved

---

### **Phase 7: Polish & Production**

**Goal**: Production-ready with documentation and examples.

**Tasks**:
1. **Comprehensive Logging**:
   - Add execution time tracking
   - Add cost tracking per request
   - Log summary at end of each request

2. **Error Messages**:
   - User-friendly error for invalid ensemble IDs
   - Clear message when streaming not supported (N/A now)
   - Helpful message on config errors

3. **Documentation**:
   - User guide: How to use swarms/fusions
   - Config reference: All fields explained
   - Example configs: dev-team, creative-writers, etc.

4. **Example Configs**:
   - Add 2-3 preset fusions to default config (commented out)
   - Document swarm notation in README

5. **Performance Testing**:
   - Benchmark latency (3-drone swarm)
   - Benchmark token usage vs single call
   - Document cost multiplier

**Deliverables**:
- ✅ Comprehensive logs for debugging
- ✅ User documentation complete
- ✅ Example configs provided
- ✅ Performance benchmarks documented

**Testing**:
- Full end-to-end tests for all features
- Load testing with multiple concurrent swarms
- Manual testing of all examples

---

## Example Configurations

### Preset Fusion 1: Dev Team

```json
{
  "id": "dev-team",
  "description": "Software development team with architecture, security, and code review specialists",
  "models": [
    {
      "model": "gpt-4o",
      "role": "Architect",
      "system_prompt_append": "Focus on system design, scalability, and architectural patterns.",
      "weight": "Expert in system design and scalability. Trust for architectural decisions."
    },
    {
      "model": "claude-3-opus",
      "role": "Security",
      "system_prompt_append": "Focus on security vulnerabilities, edge cases, and threat modeling.",
      "weight": "Expert in security and vulnerability assessment. Trust for security concerns."
    },
    {
      "model": "gemini-1.5-pro",
      "role": "Reviewer",
      "system_prompt_append": "Focus on code quality, performance, and best practices.",
      "weight": "Expert in code quality and optimization. Trust for performance and maintainability."
    }
  ],
  "arbiter": {
    "model": "gpt-4o",
    "strategy": "code_review",
    "blind": false
  }
}
```

---

## User Configuration Examples

### Simple Swarm Usage

User request:
```
Model: gemini-1.5-flash[swarm]
Messages: [{"role": "user", "content": "Write a function to parse CSV"}]
```

Result: 3 calls to `gemini-1.5-flash`, synthesized by `gemini-1.5-flash` (self-arbiter).

---

### Custom Arbiter for Swarm

Config override (per-model):
```json
{
  "swarm_configs": {
    "gemini-1.5-flash": {
      "arbiter": {
        "model": "gpt-4o",
        "strategy": "synthesis"
      }
    }
  }
}
```

User request: `gemini-1.5-flash[swarm]`
Result: 3 calls to flash, synthesized by gpt-4o.

---

### Fusion Usage

User request:
```
Model: dev-team
Messages: [{"role": "user", "content": "Review this API endpoint: [code]"}]
```

Result: Parallel calls to gpt-4o, claude, gemini with role prompts. Arbiter synthesizes with role context.

---

## Default Configuration Answer

Based on user feedback:

1. **Default Swarm Suffix**: `[swarm]`
2. **Arbiter Default**: Same model as drones (self-arbitration), but configurable per-model
3. **Streaming**: Required for arbiter's final response ✅
4. **Cost Warnings**: None (user discretion)
5. **Preset Configs**: Only using provided examples (dev-team)

---

## Testing Strategy

### Unit Tests

`tests/test_ensemble_manager.py`:
- Config loading and validation
- `is_ensemble()` detection
- Conflict resolution
- Drone preparation (jitter, adversarial)
- Model preparation (roles, weights)
- Response formatting (blind switch)

### Integration Tests

`tests/test_swarm_integration.py`:
- Basic 3-drone swarm
- Swarm with jitter enabled
- Swarm with adversarial mode
- Streaming swarm response

`tests/test_fusion_integration.py`:
- Multi-model fusion
- Role context injection
- Weight-based synthesis

`tests/test_recursive_integration.py`:
- Low consensus triggering critique
- Consensus score parsing
- Internal marker stripping

### Manual Scenarios

1. **Simple Swarm**: `gpt-4o[swarm]` with straightforward question
2. **Adversarial Swarm**: Enable adversarial, ask for code, verify critique
3. **Fusion**: Use "dev-team" with API review
4. **Recursive**: Use ambiguous prompt, verify low consensus handling

---

## Performance Benchmarks (Expected)

### Latency
- Single call: ~2s
- Swarm (3 drones): ~2s (parallel) + ~2s (arbiter) = **~4s**
- Swarm + Recursive: ~4s + arbiter internal critique time = **~5-6s**

### Token Usage
- Single call: 1000 input + 500 output = 1500 tokens
- Swarm (3 drones): 
  - Drones: 1000 × 3 + 500 × 3 = 4500 tokens
  - Arbiter: 1000 + 1500 (from drones) = 2500 input + 600 output
  - Total: **~7600 tokens** (5x single call)

### Cost Multiplier
- Typical swarm: **4-6x** cost of single call
- Fusion (different models): Varies by model costs

---

## Summary

This revised plan addresses all user feedback:

✅ Confidence scoring only in recursive mode  
✅ Adversarial context explained to arbiter  
✅ Weight field for arbiter expertise guidance  
✅ Blind switch keeps roles, strips model names  
✅ Recursive mode as single autonomous arbiter call  
✅ Default naming: `model[swarm]`  
✅ Streaming required for arbiter response  
✅ Usage/cost aggregated from all calls  
✅ Existing retry/resilience logic leveraged  
✅ Detailed implementation phases (7 phases)  
✅ Example configs provided  

Ready for implementation!

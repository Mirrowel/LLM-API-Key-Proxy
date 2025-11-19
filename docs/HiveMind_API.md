# HiveMind API Reference

## EnsembleManager

Main class for orchestrating HiveMind requests.

### `__init__(rotating_client, config_dir=None)`

Initialize the ensemble manager.

**Parameters:**
- `rotating_client` (RotatingClient): Reference to the RotatingClient instance
- `config_dir` (str, optional): Path to ensemble_configs directory. Defaults to `src/rotator_library/ensemble_configs`

**Example:**
```python
client = RotatingClient()
# EnsembleManager is automatically initialized
manager = client.ensemble_manager
```

### `is_ensemble(model_id: str) -> bool`

Check if a model ID represents an ensemble request.

**Parameters:**
- `model_id` (str): Full model ID from user request

**Returns:**
- `bool`: True if ensemble (swarm or fusion), False otherwise

**Example:**
```python
manager.is_ensemble("gpt-4o[swarm]")  # True
manager.is_ensemble("dev-team")  # True
manager.is_ensemble("gpt-4o")  # False
```

### `get_base_model(swarm_id: str) -> str`

Extract base model name from swarm ID.

**Parameters:**
- `swarm_id` (str): Swarm model ID (e.g., "gemini-1.5-flash[swarm]")

**Returns:**
- `str`: Base model name (e.g., "gemini-1.5-flash")

**Example:**
```python
base = manager.get_base_model("gpt-4o[swarm]")  # "gpt-4o"
```

### `get_fusion_ids() -> List[str]`

Get list of all configured fusion IDs.

**Returns:**
- `List[str]`: List of fusion identifiers

**Example:**
```python
fusion_ids = manager.get_fusion_ids()  # ["dev-team", "creative-writers"]
```

### `handle_request(request, **kwargs) -> Response | AsyncGenerator`

Main entry point for ensemble execution.

**Parameters:**
- `request`: Original request object
- `**kwargs`: Request parameters (model, messages, stream, etc.)

**Returns:**
- `Response`: Complete response (if stream=False)
- `AsyncGenerator`: Streaming response generator (if stream=True)

**Example:**
```python
# Non-streaming
response = await client.acompletion(
    model="gpt-4o[swarm]",
    messages=[{"role": "user", "content": "Test"}],
    stream=False
)

# Streaming
async for chunk in client.acompletion(
    model="gpt-4o[swarm]",
    messages=[{"role": "user", "content": "Test"}],
    stream=True
):
    print(chunk)
```

---

## ConfigLoader

Manages configuration loading for ensemble modes.

### `load_all() -> None`

Load all configurations from directory structure.

**Side Effects:**
- Populates `swarm_default`, `swarm_configs`, `fusion_configs`, `strategies`

### `get_swarm_config(model: str) -> Dict[str, Any]`

Get swarm configuration for a specific model.

**Parameters:**
- `model` (str): Base model name (without [swarm] suffix)

**Returns:**
- `Dict[str, Any]`: Merged configuration (default + model-specific)

### `get_fusion_config(fusion_id: str) -> Optional[Dict[str, Any]]`

Get fusion configuration by ID.

**Parameters:**
- `fusion_id` (str): Fusion identifier

**Returns:**
- `Dict[str, Any]` | `None`: Fusion configuration or None if not found

### `get_strategy(strategy_name: str) -> Optional[str]`

Get strategy template by name.

**Parameters:**
- `strategy_name` (str): Strategy identifier

**Returns:**
- `str` | `None`: Strategy template or None if not found

### `get_all_fusion_ids() -> List[str]`

Get list of all fusion IDs.

**Returns:**
- `List[str]`: List of fusion identifiers

---

## Response Object

HiveMind responses follow the standard OpenAI response format with additional usage details.

### `Response.usage`

Usage statistics for the request.

**Standard Fields (OpenAI-Compatible):**

These fields contain the **complete aggregated totals** from all models (drones/specialists + arbiter). They are fully compatible with existing tooling and billing systems.

- `prompt_tokens` (int): **Total** prompt tokens from all models
- `completion_tokens` (int): **Total** completion tokens from all models
- `total_tokens` (int): **Total** tokens (sum of prompt + completion)
- `cached_tokens` (int, optional): **Total** cached tokens if supported
- `reasoning_tokens` (int, optional): **Total** reasoning tokens if supported

**HiveMind-Specific Fields (Supplementary):**

- `hivemind_details` (dict): **Breakdown information** for observability (does NOT replace standard fields)

**Important**: Always use the standard fields for billing, quotas, and analytics. They contain the correct aggregated totals. The `hivemind_details` provides additional context for debugging and understanding HiveMind execution.

### `Response.usage.hivemind_details`

Supplementary breakdown dictionary containing:

**Common Fields:**
- `mode` (str): "swarm" or "fusion"
- `arbiter_tokens` (int): Tokens used by arbiter
- `total_cost_usd` (float): Estimated total cost in USD
- `latency_ms` (float): Total execution time in milliseconds

**Swarm-Specific:**
- `drone_count` (int): Number of drones executed
- `drone_tokens` (int): Total tokens from all drones

**Fusion-Specific:**
- `specialist_count` (int): Number of specialists executed
- `specialist_tokens` (int): Total tokens from all specialists

**Example:**
```python
response = await client.acompletion(model="gpt-4o[swarm]", ...)

# Standard fields contain TOTAL aggregated usage
usage = response.usage
print(f"Total tokens: {usage.total_tokens}")  # e.g., 650 (drones 450 + arbiter 200)
print(f"Prompt tokens: {usage.prompt_tokens}")  # e.g., 400 (all models combined)
print(f"Completion tokens: {usage.completion_tokens}")  # e.g., 250 (all models combined)

# Supplementary breakdown for observability
details = usage.hivemind_details
print(f"Mode: {details['mode']}")  # "swarm"
print(f"Drone count: {details['drone_count']}")  # 3
print(f"Drone tokens: {details['drone_tokens']}")  # 450 (breakdown)
print(f"Arbiter tokens: {details['arbiter_tokens']}")  # 200 (breakdown)
print(f"Cost: ${details['total_cost_usd']}")  # 0.00123
print(f"Latency: {details['latency_ms']}ms")  # 1523.45

# Note: drone_tokens + arbiter_tokens = total_tokens
# The standard usage fields are what billing systems should use
```

---

## Configuration Schema

### Swarm Configuration

**File Location:** `ensemble_configs/swarms/*.json`

**Schema:**
```json
{
  "model": "string (optional, only for model-specific configs)",
  "suffix": "string (default: '[swarm]')",
  "count": "integer (default: 3)",
  
  "temperature_jitter": {
    "enabled": "boolean",
    "delta": "float (temperature variance)"
  },
  
  "arbiter": {
    "model": "string ('self' or model ID)",
    "strategy": "string (strategy name)",
    "blind": "boolean (default: true)"
  },
  
  "adversarial_config": {
    "enabled": "boolean",
    "count": "integer (number of adversarial drones)",
    "prompt": "string (system prompt for adversarial drones)"
  },
  
  "recursive_mode": {
    "enabled": "boolean",
    "consensus_threshold": "integer (1-10 scale)"
  }
}
```

### Fusion Configuration

**File Location:** `ensemble_configs/fusions/*.json`

**Schema:**
```json
{
  "id": "string (unique fusion identifier)",
  "description": "string (optional)",
  
  "specialists": [
    {
      " model": "string (model ID)",
      "role": "string (specialist role name)",
      "system_prompt": "string (role-specific instructions)",
      "weight": "float (importance weight, default: 1.0)"
    }
  ],
  
  "arbiter": {
    "model": "string (model ID)",
    "strategy": "string (strategy name)",
    "blind": "boolean (default: true)"
  },
  
  "recursive_mode": {
    "enabled": "boolean",
    "consensus_threshold": "integer (1-10 scale)"
  }
}
```

### Strategy Template

**File Location:** `ensemble_configs/strategies/*.txt`

**Format:**
Plain text file with `{responses}` placeholder.

**Example:**
```
You are an expert synthesizer. Analyze the following responses and create a single, superior answer.

{responses}

Provide your synthesis as a complete, high-quality response.
```

---

## Error Handling

### Common Exceptions

**`ValueError`**: Invalid model ID or configuration
```python
try:
    response = await client.acompletion(model="invalid-fusion", ...)
except ValueError as e:
    print(f"Configuration error: {e}")
```

**`RuntimeError`**: All drones/specialists failed
```python
try:
    response = await client.acompletion(model="gpt-4o[swarm]", ...)
except RuntimeError as e:
    print(f"Execution error: {e}")
```

### Partial Failures

If some drones/specialists fail but at least one succeeds, HiveMind continues with successful responses and logs warnings.

**Logs:**
```
[ERROR] [HiveMind] Drone 2/3 failed: Rate limit exceeded
[WARNING] [HiveMind] 1/3 drones failed. Proceeding with 2 successful responses.
```

---

## Logging

HiveMind uses the `rotator_library.ensemble` logger.

**Log Levels:**
- `INFO`: Normal operations (processing, completion)
- `DEBUG`: Detailed execution (temperatures, prompts)
- `WARNING`: Low consensus, partial failures, conflicts
- `ERROR`: Drone failures, critical issues

**Example Configuration:**
```python
import logging

# Enable HiveMind debug logging
logging.getLogger("rotator_library.ensemble").setLevel(logging.DEBUG)

# Example logs:
# [INFO] [HiveMind] Processing Swarm request: gpt-4o[swarm] (base: gpt-4o, 3 drones, streaming: False)
# [DEBUG] [HiveMind] Drone 1: temperature=0.82, adversarial=False
# [DEBUG] [HiveMind] Arbiter prompt built: 2 messages
# [INFO] [HiveMind] Swarm completed successfully. Total usage: 650 tokens. Latency: 1234.56ms, Cost: $0.001200
```

---

## Advanced Usage

### Custom Arbiter Models

Use different arbiter models for different fusions:

```json
{
  "id": "research-team",
  "specialists": [...],
  "arbiter": {
    "model": "gpt-4o",  // Use GPT-4o specifically
    "strategy": "synthesis"
  }
}
```

### Self-Arbiter

Use the same model as arbiter (saves one API call):

```json
{
  "arbiter": {
    "model": "self",  // Use base model as arbiter
    "strategy": "best_of_n"
  }
}
```

### Multiple Strategies

Create task-specific strategies:

**`ensemble_configs/strategies/math_solver.txt`:**
```
You are a mathematics expert. Review these solutions:

{responses}

Identify the correct approach, verify calculations, and provide the final answer with step-by-step explanation.
```

Usage:
```json
{
  "arbiter": {
    "strategy": "math_solver"
  }
}
```

---

## Migration Guide

### From Single Model to Swarm

**Before:**
```python
response = await client.acompletion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain AI"}]
)
```

**After:**
```python
response = await client.acompletion(
    model="gpt-4o-mini[swarm]",  # Add [swarm] suffix
    messages=[{"role": "user", "content": "Explain AI"}]
)
```

### From Multiple Calls to Fusion

**Before:**
```python
arch_response = await client.acompletion(model="gpt-4o", ...)
sec_response = await client.acompletion(model="claude-3-opus", ...)
# Manually combine responses
```

**After:**
Create fusion config, then:
```python
response = await client.acompletion(
    model="dev-team",  # All in one call
    messages=[...]
)
```

---

## Performance Metrics

Typical latencies (3 drones/specialists, non-streaming):

| Model Type | Drones/Specialists | Avg Latency |
|------------|-------------------|-------------|
| gpt-4o-mini[swarm] | 3 | 1.2-2.0s |
| gpt-4o[swarm] | 3 | 2.0-3.5s |
| dev-team (fusion) | 3 | 2.5-4.0s |

**Note**: Streaming reduces perceived latency as arbiter output begins immediately after drone/specialist completion.

---

## Limitations

1. **Cost**: Multiple API calls increase costs proportionally
2. **Rate Limits**: May hit rate limits faster with parallel calls
3. **Latency**: Total time = max(drone time) + arbiter time
4. **Model Availability**: All models must be available simultaneously
5. **Token Limits**: Large responses may exceed context windows

---

## Support

For issues, questions, or feature requests:
- Check logs (`rotator_library.ensemble`)
- Review configuration files
- Verify API keys and model availability
- See [User Guide](./HiveMind_User_Guide.md) for common patterns

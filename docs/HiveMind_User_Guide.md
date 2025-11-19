# HiveMind User Guide

## Overview

**HiveMind** is a powerful ensemble feature that enables parallel model execution with intelligent arbitration. It supports two modes:

- **Swarm Mode**: Multiple parallel calls to the **same model** (called "Drones")
- **Fusion Mode**: Multiple parallel calls to **different models** (called "Specialists")

Both modes use an "Arbiter" model to synthesize the responses into a single, high-quality answer.

---

## Quick Start

### Swarm Mode

Call the same model multiple times in parallel and synthesize results:

```python
from rotator_library.client import RotatingClient

client = RotatingClient()

# Basic swarm - adds `[swarm]` suffix to any model
response = await client.acompletion(
    model="gpt-4o-mini[swarm]",  # 3 drones by default
    messages=[{"role": "user", "content": "What is quantum computing?"}],
    stream=False
)

print(response.choices[0].message.content)
print(f"Total tokens: {response.usage.total_tokens}")
print(f"Drone count: {response.usage.hivemind_details['drone_count']}")
print(f"Cost: ${response.usage.hivemind_details['total_cost_usd']}")
```

### Fusion Mode

Use multiple specialized models working together:

```python
# dev-team fusion uses 3 specialist models
response = await client.acompletion(
    model="dev-team",
    messages=[{"role": "user", "content": "Review this function"}],
    stream=False
)

print(response.choices[0].message.content)
print(f"Specialists: {response.usage.hivemind_details['specialist_count']}")
```

---

## Swarm Mode

### How It Works

1. **Preparation**: Creates N copies of your request (N drones)
2. **Execution**: Runs all drones in parallel
3. **Arbitration**: An arbiter model synthesizes all responses
4. **Result**: Returns the arbiter's synthesis

### Configuration

Swarm behavior is configured in `src/rotator_library/ensemble_configs/swarms/`:

**`default.json`** - Global swarm settings:
```json
{
  "suffix": "[swarm]",
  "count": 3,
  "temperature_jitter": {
    "enabled": true,
    "delta": 0.2
  },
  "arb iter": {
    "model": "self",
    "strategy": "synthesis",
    "blind": true
  },
  "adversarial_config": {
    "enabled": false,
    "count": 1,
    "prompt": "You are a critical reviewer..."
  },
  "recursive_mode": {
    "enabled": false,
    "consensus_threshold": 7
  }
}
```

**Model-specific configs** (e.g., `gemini-flash.json`):
```json
{
  "model": "gemini-1.5-flash",
  "arbiter": {
    "model": "gpt-4o",
    "strategy": "synthesis"
  }
}
```

### Advanced Features

#### Temperature Jitter

Introduces randomness to increase response diversity:

```json
"temperature_jitter": {
  "enabled": true,
  "delta": 0.2  // ±0.2 variance
}
```

Each drone gets a slightly different temperature: `base_temp ± delta`

#### Adversarial Mode

Adds critical drones to stress-test solutions:

```json
"adversarial_config": {
  "enabled": true,
  "count": 1,
  "prompt": "You are a Senior Principal Engineer with 15+ years of experience. Your job is to find flaws, edge cases, and potential issues."
}
```

#### Blind Switch

Removes model names from arbiter input (enabled by default):

```json
"arbiter": {
  "blind": true  // Arbiter sees "Response 1", not "Response 1 (GPT-4o)"
}
```

#### Recursive Mode

Enables autonomous arbiter decision-making for low-consensus scenarios:

```json
"recursive_mode": {
  "enabled": true,
  "consensus_threshold": 7  // If consensus < 7/10, arbiter performs internal critique
}
```

---

## Fusion Mode

### How It Works

1. **Preparation**: Assigns role-specific prompts to each specialist
2. **Execution**: Runs all specialists in parallel
3. **Arbitration**: Arbiter synthesizes with role context
4. **Result**: Returns the arbiter's synthesis

### Configuration

Fusion models are configured in `src/rotator_library/ensemble_configs/fusions/`:

**`dev-team.json`** - Example fusion:
```json
{
  "id": "dev-team",
  "description": "Software development team with specialized roles",
  "specialists": [ 
    {
      "model": "gpt-4o",
      "role": "Architect",
      "system_prompt": "Focus on architectural patterns, scalability, and system design.",
      "weight": 1.5
    },
    {
      "model": "claude-3-opus",
      "role": "Security Specialist",
      "system_prompt": "Focus on security vulnerabilities and potential exploits.",
      "weight": 1.0
    },
    {
      "model": "gemini-1.5-pro",
      "role": "Code Reviewer",
      "system_prompt": "Focus on code quality, performance, and best practices.",
      "weight": 1.0
    }
  ],
  "arbiter": {
    "model": "gpt-4o",
    "strategy": "synthesis",
    "blind": true
  }
}
```

### Creating Custom Fusions

1. Create a new JSON file in `ensemble_configs/fusions/`
2. Define specialists with roles and prompts
3. Choose an arbiter model and strategy
4. Use the fusion ID as the model name

Example: `creative-writers.json`:
```json
{
  "id": "creative-writers",
  "description": "Creative writing team",
  "specialists": [
    {
      "model": "claude-3-opus",
      "role": "Storyteller",
      "system_prompt": "Focus on narrative, character development, and plot.",
      "weight": 1.5
    },
    {
      "model": "gpt-4o",
      "role": "Editor",
      "system_prompt": "Focus on clarity, grammar, and style.",
      "weight": 1.0
    }
  ],
  "arbiter": {
    "model": "gpt-4o",
    "strategy": "synthesis"
  }
}
```

Usage:
```python
response = await client.acompletion(
    model="creative-writers",
    messages=[{"role": "user", "content": "Write a short story about AI"}]
)
```

---

## Arbitration Strategies

Strategies are text prompts in `ensemble_configs/strategies/`:

**`synthesis.txt`** - Combine all responses:
```
You are an expert synthesizer. Analyze the following responses and create a single, superior answer that:
1. Combines the best elements from each response
2. Resolves any conflicts or contradictions
3. Ensures completeness and accuracy
4. Maintains coherence and clarity

{responses}
```

**`best_of_n.txt`** - Select and refine the best:
```
Review these responses and identify the strongest one. Then refine and enhance it.

{responses}
```

**`code_review.txt`** - Code-specific evaluation:
```
You are a senior code reviewer. Analyze these code responses and provide:
1. Best implementation approach
2. Security considerations
3. Performance optimization suggestions
4. Final recommended code

{responses}
```

### Creating Custom Strategies

Create a `.txt` file in `ensemble_configs/strategies/` with your prompt template. Use `{responses}` as a placeholder for the formatted responses.

---

## Streaming Support

HiveMind respects the `stream` parameter:

```python
# Streaming swarm
async for chunk in client.acompletion(
    model="gpt-4o[swarm]",
    messages=[{"role": "user", "content": "Explain AI"}],
    stream=True  # Stream arbiter's response
):
    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

**Note**: Drones/specialists execute in parallel (not streamed). Only the arbiter's final synthesis is streamed.

---

## Usage & Cost Tracking

All HiveMind responses include detailed usage information in **standard OpenAI-compatible fields** plus additional HiveMind-specific breakdown:

```python
response = await client.acompletion(
    model="gpt-4o-mini[swarm]",
    messages=[{"role": "user", "content": "Test"}]
)

# ✅ STANDARD usage fields (compatible with all tooling)
# These contain the TOTAL aggregated usage (drones/specialists + arbiter)
print(f"Prompt tokens: {response.usage.prompt_tokens}")  # Total from all models
print(f"Completion tokens: {response.usage.completion_tokens}")  # Total from all models
print(f"Total tokens: {response.usage.total_tokens}")  # Grand total

# ✅ SUPPLEMENTARY HiveMind details (breakdown for observability)
# These provide additional context but do NOT replace standard fields
details = response.usage.hivemind_details
print(f"Mode: {details['mode']}")  # "swarm" or "fusion"
print(f"Drone/Specialist count: {details.get('drone_count') or details.get('specialist_count')}")
print(f"Drone/Specialist tokens: {details.get('drone_tokens') or details.get('specialist_tokens')}")
print(f"Arbiter tokens: {details['arbiter_tokens']}")
print(f"Total cost: ${details['total_cost_usd']}")
print(f"Latency: {details['latency_ms']}ms")
```

**Important**: Consumers should use the standard usage fields (`prompt_tokens`, `completion_tokens`, `total_tokens`) for billing and analytics. These already include the complete totals. The `hivemind_details` field provides a breakdown for debugging and observability.

---

## Best Practices

### Model Selection

**Sw arm Mode**:
- Use for: Same model, different parameters (temperature jitter)
- Best for: Brainstorming, diverse perspectives, consensus building
- Models: Fast models (gpt-4o-mini, gemini-flash) for cost efficiency

**Fusion Mode**:
- Use for: Different models, specialized expertise
- Best for: Complex tasks requiring multiple skill sets
- Models: Mix strengths (GPT for reasoning, Claude for safety, Gemini for code)

### Cost Optimization

1. **Use smaller models for drones**: `gpt-4o-mini[swarm]` instead of `gpt-4o[swarm]`
2. **Limit drone count**: Default is 3, but 2 is often sufficient
3. **Use "self" arbiter**: Saves one API call
4. **Monitor `hivemind_details`**: Track costs per request

### Performance Tips

1. **Parallel execution is fast**: All drones/specialists run simultaneously
2. **Streaming reduces perceived latency**: Users see output immediately
3. **Check latency_ms**: Identify slow requests

---

## Troubleshooting

### No ensemble detected

**Problem**: Model isn't recognized as ensemble
**Solution**: Check spelling, ensure `[swarm]` suffix or fusion ID exists

### All drones failed

**Problem**: All parallel calls failed
**Solution**: Check API keys, rate limits, model availability

### High costs

**Problem**: HiveMind is expensive
**Solution**: Reduce drone count, use smaller models, limit to critical requests

### Poor synthesis quality

**Problem**: Arbiter output isn't good
**Solution**: Use a better arbiter model (gpt-4o, claude-3-opus), try different strategy

---

## API Reference

See [API.md](./API.md) for detailed API documentation.

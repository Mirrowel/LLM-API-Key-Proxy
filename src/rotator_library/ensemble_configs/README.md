# HiveMind Ensemble Configuration Guide

This directory contains the configuration for HiveMind Ensemble (Swarm/Fusion) feature.

## Directory Structure

```
ensemble_configs/
├── swarms/          # Swarm preset configurations
│   ├── default.json # Default global settings (fallback)
│   └── *.json       # Preset configurations (e.g., aggressive.json, balanced.json)
├── fusions/         # Fusion configurations (multi-model teams)
│   └── *.json       # Individual fusion definitions or arrays of fusions
├── strategies/      # Arbitration strategy templates
│   └── *.txt        # Strategy prompt templates with {responses} placeholder
└── roles/           # Reusable role template definitions
    └── *.json       # Role templates for fusion specialists
```

## Configuration Files

### Swarm Configuration (Preset-Based)

HiveMind uses a **preset-based system** for swarm configurations. Each preset defines a configuration that can be applied to multiple base models.

**Format Options**:
- Explicit: `{base_model}-{preset_id}[swarm]`
- Short (if `omit_id: true`): `{base_model}[swarm]`

**Example**: 
- `gpt-4o-mini-aggressive[swarm]` - explicitly uses the `aggressive.json` preset
- `gpt-4o-mini[swarm]` - uses `default.json` preset OR a custom preset with `omit_id: true`
- `gpt-4o-mini-default[swarm]` - always uses `default.json` even if omit_id preset exists

**Preset File Structure** (`swarms/{preset_id}.json`):
```json
{
  "id": "aggressive",
  "description": "High diversity swarm with adversarial critique",
  "base_models": ["gpt-4o-mini", "gemini-1.5-flash", "claude-3-haiku"],
  "count": 5,
  "temperature_jitter": {
    "enabled": true,
    "delta": 0.3
  },
  "adversarial_config": {
    "enabled": true,
    "count": 2,
    "prompt": "You are a critical reviewer. Find flaws and edge cases."
  },
  "arbiter": {
    "model": "self",
    "strategy": "synthesis",
    "blind": true
  },
  "recursive_mode": {
    "enabled": true,
    "consensus_threshold": 6
  }
}
```

**Key Fields**:
- `id`: Preset identifier (must match filename)
- `base_models`: List of models this preset applies to (enables discovery)
- `omit_id` (optional): If `true`, this preset becomes the default for its `base_models` when using `{model}[swarm]` syntax
- `count`: Number of drones to spawn
- `temperature_jitter`: Randomize temperature for diversity
- `adversarial_config`: Enable critical analysis drones
- `arbiter`: Synthesis configuration
- `recursive_mode`: Autonomous low-consensus handling

**Omit ID Feature**: When a preset has `"omit_id": true`, it becomes the default for its specified models:
- `gpt-4o-mini[swarm]` → uses the `omit_id` preset instead of `default.json`
- `gpt-4o-mini-default[swarm]` → always uses `default.json` (explicit fallback)
- `gpt-4o-mini-aggressive[swarm]` → always uses `aggressive.json` (explicit)

**Important**: `omit_id` controls ONLY what appears in `/v1/models` for discoverability, not what works at runtime:
- Explicit format (`model-preset[swarm]`) always works regardless of `omit_id` or `base_models`
- You can use ANY model with ANY preset explicitly (e.g., `claude-3-opus-aggressive[swarm]` works even if Claude isn't in aggressive's base_models)

**Discovery Rules** (`/v1/models` endpoint):
- Preset WITH `base_models` + `omit_id: true` → Shows as `{model}[swarm]` only (explicit form hidden to avoid clutter)
- Preset WITH `base_models` + `omit_id: false` → Shows as `{model}-{preset}[swarm]` only  
- Preset WITHOUT `base_models` → Never shown (invisible preset, but still usable with explicit syntax)

**`base_models` Purpose**: 
- Controls ONLY which models appear in `/v1/models` for this preset
- Does NOT restrict runtime usage - any model can use any preset with explicit syntax
- If empty/missing, preset is "invisible" but fully functional when explicitly referenced

### Fusion Configuration (Multi-Model Teams)

Fusions combine responses from different specialized models. Each fusion can have role-based routing and specialist expertise.

**Single Fusion Format** (`fusions/{fusion-id}.json`):
```json
{
  "id": "dev-team",
  "description": "Software development team with specialized roles",
  "specialists": [
    {
      "model": "gpt-4o",
      "role": "Architect",
      "system_prompt": "Focus on scalability and system design.",
      "weight": 1.5,
      "weight_description": "Expert in architecture. Trust for design decisions."
    },
    {
      "model": "claude-3-opus",
      "role_template": "security-expert"
    }
  ],
  "arbiter": {
    "model": "gpt-4o",
    "strategy": "code_review",
    "blind": false
  },
  "recursive_mode": {
    "enabled": false,
    "consensus_threshold": 7
  }
}
```

**Array Format** (multiple fusions in one file):
```json
{
  "fusions": [
    {
      "id": "dev-team",
      "specialists": [...]
    },
    {
      "id": "creative-writers",
      "specialists": [...]
    }
  ]
}
```

**Specialist Fields**:
- `model`: Provider/model ID
- `role`: Display name for this specialist
- `system_prompt`: Role-specific instructions sent to the model
- `weight`: Numeric importance (for future use)
- `weight_description`: Expertise description for arbiter context
- `role_template`: Reference to a reusable role template (see Roles section)

**Arbiter Configuration**:
- `model`: Model ID for synthesis (or "self" to use first specialist)
- `strategy`: Strategy template name (from `strategies/` directory)
- `blind`: If `true`, hides model names from arbiter (preserves roles)

### Role Templates (Reusable Configurations)

Role templates allow you to define reusable specialist configurations that can be referenced by multiple fusions.

**Single Role Format** (`roles/{role-id}.json`):
```json
{
  "name": "Security Expert",
  "system_prompt": "You are a cybersecurity expert. Focus on vulnerabilities, edge cases, and threat modeling.",
  "weight": 1.2,
  "weight_description": "Expert in security and vulnerability assessment. Trust for security concerns."
}
```

**Array Format** (multiple roles in one file):
```json
{
  "roles": [
    {
      "name": "Architect",
      "system_prompt": "Focus on system design and scalability.",
      "weight_description": "Expert in architectural patterns."
    },
    {
      "name": "Security Expert",
      "system_prompt": "Focus on vulnerabilities and threats.",
      "weight_description": "Expert in security assessment."
    }
  ]
}
```

**Usage in Fusions**:
```json
{
  "specialists": [
    {
      "model": "claude-3-opus",
      "role_template": "security-expert"
    }
  ]
}
```

**Override Behavior**: Specialist configs can override any field from the referenced template.

### Strategy Templates

Each strategy is a plain text file defining how the arbiter should synthesize responses.

**File Location**: `strategies/{strategy-name}.txt`

**Placeholder**: Use `{responses}` where formatted responses should be injected.

**Example** (`strategies/synthesis.txt`):
```
You are an expert synthesizer. Analyze the following responses and create a single, superior answer that:
1. Combines the best elements from each response
2. Resolves any conflicts or contradictions
3. Ensures completeness and accuracy
4. Maintains coherence and clarity

{responses}

Provide your synthesis as a complete, high-quality response.
```

## Adding New Configurations

1. **New Swarm Preset**: Create `{preset_id}.json` in `swarms/` with `id` and `base_models` fields
2. **New Fusion**: Create `{fusion_id}.json` in `fusions/` OR add to an existing array file
3. **New Strategy**: Create `{strategy_name}.txt` in `strategies/`
4. **New Role Template**: Create `{role_id}.json` in `roles/` OR add to an existing array file

All configs are loaded automatically on startup!

## Advanced Features

### Temperature Jitter (Swarm)
Randomizes temperature across drones to increase response diversity:
```json
"temperature_jitter": {
  "enabled": true,
  "delta": 0.2
}
```
Each drone gets `base_temp ± delta` (clamped to [0.0, 2.0]).

### Adversarial Mode (Swarm)
Dedicates N drones as critical reviewers:
```json
"adversarial_config": {
  "enabled": true,
  "count": 1,
  "prompt": "You are a Senior Principal Engineer. Find flaws and edge cases."
}
```
Last N drones receive the adversarial prompt. Responses are marked `[ADVERSARIAL]` in arbiter input.

### Recursive Mode (Swarm & Fusion)
Enables autonomous arbiter decision-making:
```json
"recursive_mode": {
  "enabled": true,
  "consensus_threshold": 7
}
```
If consensus < threshold, arbiter performs internal critique before synthesis. All internal reasoning is logged but hidden from user.

### Blind Switch
Controls whether model names are shown to arbiter:
```json
"arbiter": {
  "blind": true
}
```
- `true`: "Response 1 (Architect role)" (hides model names)
- `false`: "Response 1 (GPT-4o - Architect)" (shows models)

Roles are **always preserved** regardless of blind setting.

## Usage Examples

**Swarm Request**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-d '{"model": "gpt-4o-mini-aggressive[swarm]", "messages": [...]}'
```

**Fusion Request**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-d '{"model": "dev-team[fusion]", "messages": [...]}'
```

For detailed usage and API reference, see:
- [HiveMind User Guide](../../../docs/HiveMind_User_Guide.md)
- [HiveMind API Reference](../../../docs/HiveMind_API.md)

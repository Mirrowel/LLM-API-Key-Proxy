# HiveMind Ensemble Configuration Guide

This directory contains the configuration for HiveMind Ensemble (Swarm/Fusion) feature.

## Directory Structure

```
ensemble_configs/
├── swarms/          # Swarm configurations
│   ├── default.json # Default swarm settings (applied to all swarms)
│   └── *.json       # Model-specific swarm overrides
├── fusions/         # Fusion configurations
│   └── *.json       # Individual fusion definitions
└── strategies/      # Arbitration strategy templates
    └── *.txt        # Strategy prompt templates
```

## Configuration Files

### Swarm Configuration

**Default**: `swarms/default.json` - Applied to all swarm requests

**Model-Specific**: `swarms/{model-name}.json` - Overrides for specific models

Example model-specific config:
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

### Fusion Configuration

Each fusion is defined in its own file: `fusions/{fusion-id}.json`

See `dev-team.json` for a complete example.

### Strategy Templates

Each strategy is a text file in `strategies/{strategy-name}.txt`

Use `{responses}` placeholder for injecting formatted responses.

## Adding New Configurations

1. **New Swarm Override**: Drop a JSON file in `swarms/` with model-specific settings
2. **New Fusion**: Drop a JSON file in `fusions/` with fusion definition
3. **New Strategy**: Drop a .txt file in `strategies/` with prompt template

All configs are loaded automatically on startup!

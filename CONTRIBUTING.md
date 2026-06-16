# Contributing to LLM-API-Key-Proxy

Thanks for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/Mirrowel/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy

# Using uv (recommended)
uv venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
uv pip install -r requirements.txt

# Run the proxy
python src/proxy_app/main.py
```

## Project Structure

```
src/
├── rotator_library/    # Core library (LGPL-3.0-only)
│   └── providers/      # LLM provider implementations
└── proxy_app/          # Proxy application (MIT)
```

## Making Changes

### License Headers

All new source files **must** include the appropriate license header:

**For `src/rotator_library/` files:**
```python
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel
```

**For `src/proxy_app/` files:**
```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel
```

### Adding a New Provider

1. Create your provider in `src/rotator_library/providers/` following the existing pattern
2. Implement the `ProviderInterface` from `provider_interface.py`
3. Register the provider in `provider_factory.py`
4. Update `model_definitions.py` if needed
5. Add documentation to README.md and DOCUMENTATION.md
6. Reference the feature request issue in your PR

### Code Style

- Follow existing code patterns in the codebase
- Use type hints where practical
- Keep functions focused and well-documented

### Running Tests

The project uses `pytest` for testing. The test suite covers self-contained modules
in the resilience library (`rotator_library`).

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run the full test suite
pytest

# Run a specific test module
pytest tests/test_request_sanitizer.py

# Run with verbose output
pytest -v
```

When adding new features or fixing bugs, include tests in the `tests/` directory.
Follow the existing test file naming convention (`test_*.py`) and use descriptive
class names to group related tests.

## Pull Requests

1. Fork the repository and create a feature branch
2. Make your changes with clear commit messages
3. Reference related issues in commits: `feat(providers): add X provider (#123)`
4. Open a PR with a clear description of what changed and why
5. Ensure your changes include necessary documentation updates
6. Add or update tests for any changed behavior

## Reporting Issues

Please use the issue templates:
- **Bug Report** - For bugs and unexpected behavior
- **Feature Request** - For new features, enhancements, or provider requests

When reporting bugs, include:
- Which branch/version you're using
- Your deployment method (binary, Docker, source)
- Steps to reproduce
- Error logs if available

## Questions?

Open a discussion or issue if you need help getting started.

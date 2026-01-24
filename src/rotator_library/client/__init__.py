# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Client package for LLM API key rotation.

This package provides the RotatingClient and associated components
for intelligent credential rotation and retry logic.

Public API:
    RotatingClient: Main client class for making API requests
    StreamedAPIError: Exception for streaming errors

Components (for advanced usage):
    RequestExecutor: Unified retry/rotation logic
    CredentialFilter: Tier compatibility filtering
    ModelResolver: Model name resolution
    ProviderTransforms: Provider-specific transforms
    StreamingHandler: Streaming response processing
"""

from .rotating_client import RotatingClient
from ..core.errors import StreamedAPIError

# Also expose components for advanced usage
from .executor import RequestExecutor
from .filters import CredentialFilter
from .models import ModelResolver
from .transforms import ProviderTransforms
from .streaming import StreamingHandler
from .anthropic import AnthropicHandler
from .types import AvailabilityStats, RetryState, ExecutionResult

__all__ = [
    # Main public API
    "RotatingClient",
    "StreamedAPIError",
    # Components
    "RequestExecutor",
    "CredentialFilter",
    "ModelResolver",
    "ProviderTransforms",
    "StreamingHandler",
    "AnthropicHandler",
    # Types
    "AvailabilityStats",
    "RetryState",
    "ExecutionResult",
]

# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Core package for the rotator library.

Provides shared infrastructure used by both client and usage manager:
- types: Shared dataclasses and type definitions
- errors: All custom exceptions
- config: ConfigLoader for centralized configuration
- constants: Default values and magic numbers
"""

from .types import (
    CredentialInfo,
    RequestContext,
    ProcessedChunk,
    FilterResult,
    FairCycleConfig,
    CustomCapConfig,
    ProviderConfig,
    WindowConfig,
    RequestCompleteResult,
)

from .errors import (
    # Base exceptions
    NoAvailableKeysError,
    PreRequestCallbackError,
    CredentialNeedsReauthError,
    EmptyResponseError,
    TransientQuotaError,
    StreamedAPIError,
    # Error classification
    ClassifiedError,
    RequestErrorAccumulator,
    classify_error,
    should_rotate_on_error,
    should_retry_same_key,
    mask_credential,
    is_abnormal_error,
    get_retry_after,
)

from .config import ConfigLoader

__all__ = [
    # Types
    "CredentialInfo",
    "RequestContext",
    "ProcessedChunk",
    "FilterResult",
    "FairCycleConfig",
    "CustomCapConfig",
    "ProviderConfig",
    "WindowConfig",
    "RequestCompleteResult",
    # Errors
    "NoAvailableKeysError",
    "PreRequestCallbackError",
    "CredentialNeedsReauthError",
    "EmptyResponseError",
    "TransientQuotaError",
    "StreamedAPIError",
    "ClassifiedError",
    "RequestErrorAccumulator",
    "classify_error",
    "should_rotate_on_error",
    "should_retry_same_key",
    "mask_credential",
    "is_abnormal_error",
    "get_retry_after",
    # Config
    "ConfigLoader",
]

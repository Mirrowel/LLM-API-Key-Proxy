# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Configurable provider field-cache rules and helpers."""

from .paths import FieldCachePathError, extract_path, inject_path, parse_path
from .engine import FieldCacheEngine, FieldCacheOperation, build_cache_key
from .store import FieldCacheStore, InMemoryFieldCacheStore, ProviderCacheFieldStore
from .types import FieldCacheContext, FieldCacheInjection, FieldCacheRule
from .compat import CompatibilityRegistry, ModelRef, get_compatibility_registry, reset_compatibility_registry
from .replay import compile_cache_replay, parse_cache_replay_config

__all__ = [
    "CompatibilityRegistry",
    "FieldCacheContext",
    "FieldCacheEngine",
    "FieldCacheInjection",
    "FieldCacheOperation",
    "FieldCachePathError",
    "FieldCacheRule",
    "FieldCacheStore",
    "InMemoryFieldCacheStore",
    "ModelRef",
    "ProviderCacheFieldStore",
    "build_cache_key",
    "compile_cache_replay",
    "extract_path",
    "get_compatibility_registry",
    "inject_path",
    "parse_cache_replay_config",
    "parse_path",
    "reset_compatibility_registry",
]

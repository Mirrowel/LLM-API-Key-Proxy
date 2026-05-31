# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Shared protocol operation names.

Operations are deliberately plain strings instead of a closed enum. The native
protocol system is meant to be extended by local/custom providers, so the core
must provide well-known names without blocking new operations that are not known
today.
"""

from __future__ import annotations

from typing import Final

OPERATION_UNKNOWN: Final[str] = "unknown"
OPERATION_CHAT: Final[str] = "chat"
OPERATION_MESSAGES: Final[str] = "messages"
OPERATION_RESPONSES: Final[str] = "responses"
OPERATION_COUNT_TOKENS: Final[str] = "count_tokens"
OPERATION_EMBEDDINGS: Final[str] = "embeddings"
OPERATION_IMAGE_GENERATION: Final[str] = "image_generation"
OPERATION_IMAGE_EDIT: Final[str] = "image_edit"
OPERATION_IMAGE_VARIATION: Final[str] = "image_variation"
OPERATION_AUDIO_TRANSCRIPTION: Final[str] = "audio_transcription"
OPERATION_AUDIO_TRANSLATION: Final[str] = "audio_translation"
OPERATION_SPEECH: Final[str] = "speech"
OPERATION_OLLAMA_CHAT: Final[str] = "ollama_chat"
OPERATION_OLLAMA_GENERATE: Final[str] = "ollama_generate"
OPERATION_MCP: Final[str] = "mcp"


def normalize_operation(operation: str | None) -> str:
    """Normalize an operation name while preserving custom extensions."""

    if not operation:
        return OPERATION_UNKNOWN
    return str(operation).strip().lower() or OPERATION_UNKNOWN

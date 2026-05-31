# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Small JSON-path-like selector used by field-cache rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


class FieldCachePathError(ValueError):
    """Raised when a field-cache path is malformed or cannot be injected."""


@dataclass(frozen=True)
class PathToken:
    kind: Literal["key", "index", "wildcard"]
    value: str | int | None = None


def parse_path(path: str) -> tuple[PathToken, ...]:
    """Parse a minimal dotted path with indexes and wildcards.

    Supported examples: `choices.0.message.content`, `choices.*.message`,
    `messages[-1].reasoning_content`. Escaping is intentionally unsupported in
    Phase 3 so malformed provider configs fail early and clearly.
    """

    if not path or path.startswith(".") or path.endswith(".") or ".." in path:
        raise FieldCachePathError(f"Malformed field-cache path: {path!r}")
    tokens: list[PathToken] = []
    for segment in path.split("."):
        if not segment:
            raise FieldCachePathError(f"Malformed field-cache path: {path!r}")
        _parse_segment(segment, tokens, path)
    return tuple(tokens)


def _parse_segment(segment: str, tokens: list[PathToken], full_path: str) -> None:
    if segment == "*":
        tokens.append(PathToken("wildcard"))
        return
    cursor = 0
    while cursor < len(segment):
        bracket = segment.find("[", cursor)
        if bracket == -1:
            value = segment[cursor:]
            if value:
                if value.lstrip("-").isdigit():
                    tokens.append(PathToken("index", int(value)))
                else:
                    tokens.append(PathToken("key", value))
            return
        if bracket > cursor:
            tokens.append(PathToken("key", segment[cursor:bracket]))
        close = segment.find("]", bracket)
        if close == -1:
            raise FieldCachePathError(f"Unclosed index in field-cache path: {full_path!r}")
        raw_index = segment[bracket + 1 : close]
        if raw_index == "*":
            tokens.append(PathToken("wildcard"))
        else:
            try:
                tokens.append(PathToken("index", int(raw_index)))
            except ValueError as exc:
                raise FieldCachePathError(f"Invalid index {raw_index!r} in field-cache path: {full_path!r}") from exc
        cursor = close + 1


def extract_path(payload: Any, path: str) -> list[Any]:
    """Return every value matching path in stable traversal order."""

    tokens = parse_path(path)
    current = [payload]
    for token in tokens:
        next_values: list[Any] = []
        for value in current:
            next_values.extend(_extract_token(value, token))
        current = next_values
        if not current:
            break
    return current


def _extract_token(value: Any, token: PathToken) -> list[Any]:
    if token.kind == "key":
        if isinstance(value, dict) and token.value in value:
            return [value[token.value]]
        return []
    if token.kind == "index":
        if isinstance(value, list) and value:
            index = int(token.value)
            if -len(value) <= index < len(value):
                return [value[index]]
        return []
    if token.kind == "wildcard":
        if isinstance(value, dict):
            return list(value.values())
        if isinstance(value, list):
            return list(value)
        return []
    raise FieldCachePathError(f"Unknown path token: {token}")


def inject_path(payload: Any, path: str, injected_value: Any, *, when_missing_only: bool = False, insert: bool = False) -> bool:
    """Inject a value at a simple path, creating dict containers as needed.

    Wildcard injection is rejected because creating multiple branches can be
    ambiguous and provider-specific. List indexes must already exist; this keeps
    mutation predictable for message-tail use cases like `messages[-1].field`.
    `insert=True` is intentionally limited to final list-index tokens so rules
    cannot accidentally create provider-specific list structures.
    """

    tokens = parse_path(path)
    if any(token.kind == "wildcard" for token in tokens):
        raise FieldCachePathError("Wildcard injection is not supported")
    if not isinstance(payload, dict):
        raise FieldCachePathError("Field-cache injection root must be a dict")
    current = payload
    for index, token in enumerate(tokens):
        is_last = index == len(tokens) - 1
        if token.kind == "key":
            if not isinstance(current, dict):
                raise FieldCachePathError(f"Cannot inject key {token.value!r} into non-dict value")
            key = str(token.value)
            if is_last:
                if insert:
                    raise FieldCachePathError("insert=True requires a final list index token")
                if when_missing_only and key in current:
                    return False
                changed = current.get(key) != injected_value
                current[key] = injected_value
                return changed
            if key not in current or current[key] is None:
                current[key] = [] if tokens[index + 1].kind == "index" else {}
            current = current[key]
            continue
        if token.kind == "index":
            if not isinstance(current, list) or not current:
                raise FieldCachePathError("Cannot inject into missing or empty list")
            list_index = int(token.value)
            if not (-len(current) <= list_index < len(current)):
                raise FieldCachePathError(f"List index out of range for field-cache injection: {list_index}")
            if is_last:
                if insert:
                    if when_missing_only:
                        return False
                    current.insert(list_index, injected_value)
                    return True
                if when_missing_only and current[list_index] is not None:
                    return False
                changed = current[list_index] != injected_value
                current[list_index] = injected_value
                return changed
            current = current[list_index]
            continue
        raise FieldCachePathError(f"Unsupported injection token: {token}")
    return False

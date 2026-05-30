from __future__ import annotations

import pytest

from rotator_library.field_cache.paths import FieldCachePathError, extract_path, inject_path, parse_path
from rotator_library.field_cache.types import FieldCacheInjection, FieldCacheRule


def test_parse_path_supports_keys_indexes_wildcards_and_tail_index() -> None:
    tokens = parse_path("choices.*.message.parts[-1]")

    assert [token.kind for token in tokens] == ["key", "wildcard", "key", "key", "index"]
    assert tokens[-1].value == -1


def test_extract_path_handles_nested_dict_list_and_wildcard() -> None:
    payload = {
        "choices": [
            {"message": {"reasoning_content": "a"}},
            {"message": {"reasoning_content": "b"}},
        ]
    }

    assert extract_path(payload, "choices.*.message.reasoning_content") == ["a", "b"]
    assert extract_path(payload, "choices.1.message.reasoning_content") == ["b"]


def test_extract_path_missing_values_are_noop() -> None:
    assert extract_path({"choices": []}, "choices.*.message.reasoning_content") == []
    assert extract_path({}, "missing.path") == []


def test_extract_path_tail_index() -> None:
    payload = {"messages": [{"content": "first"}, {"content": "last"}]}

    assert extract_path(payload, "messages[-1].content") == ["last"]


def test_inject_path_creates_dict_containers() -> None:
    payload = {"messages": [{"role": "assistant"}]}

    changed = inject_path(payload, "messages[-1].reasoning_content", "hidden")

    assert changed is True
    assert payload["messages"][-1]["reasoning_content"] == "hidden"


def test_inject_path_respects_when_missing_only() -> None:
    payload = {"metadata": {"prompt_cache_key": "existing"}}

    changed = inject_path(payload, "metadata.prompt_cache_key", "new", when_missing_only=True)

    assert changed is False
    assert payload["metadata"]["prompt_cache_key"] == "existing"


def test_inject_path_rejects_wildcards_and_missing_lists() -> None:
    with pytest.raises(FieldCachePathError):
        inject_path({"choices": []}, "choices.*.message.reasoning_content", "x")
    with pytest.raises(FieldCachePathError):
        inject_path({}, "messages[-1].reasoning_content", "x")


def test_malformed_paths_and_rules_raise_useful_errors() -> None:
    with pytest.raises(FieldCachePathError):
        parse_path("choices..message")
    with pytest.raises(FieldCachePathError):
        parse_path("messages[abc]")
    with pytest.raises(ValueError):
        FieldCacheRule(name="bad/name", source="response", path="x")
    assert FieldCacheRule(
        name="reasoning_content",
        source="response",
        path="choices.*.message.reasoning_content",
        inject=FieldCacheInjection(target="request", path="messages[-1].reasoning_content"),
    ).scope == ("provider", "model", "classifier", "session")

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.transaction_logger import (
    TransactionLogger,
    FRAMEWORK_KEYS,
    _strip_framework_keys,
)


def _artifact(log_dir, name):
    compressed = Path(log_dir) / (name + ".zst")
    return compressed if compressed.exists() else Path(log_dir) / name


@pytest.fixture(autouse=True)
def _trace_level_2(monkeypatch):
    """Trace mechanics live at L2 (D15 tiers)."""
    monkeypatch.setenv("TRANSACTION_LOG_LEVEL", "2")


def _make_logger(tmp_path):
    # Isolated from the developer's real log store: nested under tmp_path
    # (parent_dir set → no root-logger retention pruning of real logs).
    return TransactionLogger(
        provider="nvidia_nim",
        model="mistral-medium-3.5",
        enabled=True,
        parent_dir=tmp_path,
    )


def _trace_entries(log_dir):
    from rotator_library.utils import zstd_io

    return zstd_io.read_jsonl_any(Path(log_dir) / "transform_trace.jsonl")

class FakeLogger:
    def __init__(self, log_dir):
        self.enabled = True
        self.log_dir = log_dir
        self._dir_available = True
        self._context = None
        self.request_id = "test1234"
        self.streaming = False


def test_strip_framework_keys():
    data = {
        "model": "test",
        "api_key": "secret",
        "api_base": "https://api.example.com",
        "custom_llm_provider": "openai",
        "transaction_context": {"trace_id": "abc"},
        "messages": [],
    }
    stripped = _strip_framework_keys(data)
    assert "model" in stripped
    assert "messages" in stripped
    for key in FRAMEWORK_KEYS:
        assert key not in stripped


def test_no_file_when_identical(tmp_path):
    logger = _make_logger(tmp_path)
    original = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    transformed = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    logger.log_transformed_request(transformed, original)
    assert not (Path(logger.log_dir) / "request_transformed.json").exists()


def test_no_file_when_only_framework_keys_differ(tmp_path):
    logger = _make_logger(tmp_path)
    original = {"model": "test", "messages": []}
    transformed = {
        "model": "test",
        "messages": [],
        "api_key": "secret",
        "api_base": "https://api.example.com",
        "custom_llm_provider": "openai",
        "transaction_context": {"trace_id": "abc"},
    }
    logger.log_transformed_request(transformed, original)
    assert not (Path(logger.log_dir) / "request_transformed.json").exists()


def test_file_written_when_transform_differs(tmp_path):
    logger = _make_logger(tmp_path)
    original = {"model": "test", "messages": []}
    transformed = {
        "model": "test",
        "messages": [],
        "reasoning_effort": "high",
        "api_base": "https://api.example.com",
    }
    logger.log_transformed_request(transformed, original)
    transformed_file = _artifact(Path(logger.log_dir), "request_transformed.json")
    assert transformed_file.exists()
    from rotator_library.utils import zstd_io
    data = zstd_io.read_json_any(Path(logger.log_dir) / "request_transformed.json")
    assert data["request_id"] == logger.request_id
    assert "reasoning_effort" in data["data"]
    assert data["data"]["reasoning_effort"] == "high"
    assert "api_key" not in data["data"]
    assert "transaction_context" not in data["data"]


def test_file_content_excludes_framework_keys(tmp_path):
    logger = _make_logger(tmp_path)
    original = {"model": "test", "messages": []}
    transformed = {
        "model": "test",
        "messages": [],
        "reasoning_effort": "high",
        "api_key": "secret",
        "custom_llm_provider": "openai",
        "transaction_context": {"ctx": True},
    }
    logger.log_transformed_request(transformed, original)
    transformed_file = _artifact(Path(logger.log_dir), "request_transformed.json")
    from rotator_library.utils import zstd_io
    data = zstd_io.read_json_any(Path(logger.log_dir) / "request_transformed.json")
    for key in FRAMEWORK_KEYS:
        assert key not in data["data"]


def test_no_file_when_disabled(tmp_path):
    logger = TransactionLogger(
        provider="nvidia_nim",
        model="test",
        enabled=False,
    )
    original = {"model": "test", "messages": []}
    transformed = {"model": "test", "messages": [], "reasoning_effort": "high"}
    logger.log_transformed_request(transformed, original)
    assert logger.log_dir is None


def test_extra_body_triggers_diff(tmp_path):
    logger = _make_logger(tmp_path)
    original = {
        "model": "nvidia_nim/deepseek-ai/deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hi"}],
    }
    transformed = {
        "model": "nvidia_nim/deepseek-ai/deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "max",
            }
        },
        "api_base": "https://integrate.api.nvidia.com/v1",
    }
    logger.log_transformed_request(transformed, original)
    transformed_file = _artifact(Path(logger.log_dir), "request_transformed.json")
    assert transformed_file.exists()
    from rotator_library.utils import zstd_io
    data = zstd_io.read_json_any(Path(logger.log_dir) / "request_transformed.json")
    assert data["data"]["extra_body"]["chat_template_kwargs"]["thinking"] is True
    assert data["data"]["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "max"
    assert "api_key" not in data["data"]

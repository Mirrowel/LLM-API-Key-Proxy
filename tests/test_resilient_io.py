import os
import json
import shutil
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, ANY

import pytest

from rotator_library.utils.resilient_io import safe_write_json, BufferedWriteRegistry


@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)


@pytest.mark.parametrize("atomic", [True, False])
def test_safe_write_json_happy_path(tmp_path, mock_logger, atomic):
    """Test basic write creates file with correct content."""
    file_path = tmp_path / f"test_{'atomic' if atomic else 'nonatomic'}.json"
    data = {"key": "value", "number": 42}

    result = safe_write_json(
        path=file_path,
        data=data,
        logger=mock_logger,
        atomic=atomic
    )

    assert result is True
    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == data


@pytest.mark.parametrize("atomic", [True, False])
def test_safe_write_json_secure_permissions(tmp_path, mock_logger, atomic):
    """Test secure permissions are set for writes."""
    file_path = tmp_path / f"test_secure_{'atomic' if atomic else 'nonatomic'}.json"
    data = {"secret": "data"}

    with patch("os.chmod") as mock_chmod:
        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=atomic,
            secure_permissions=True
        )

        assert result is True
        # For atomic, it applies to tmp file first. For non-atomic, it applies directly.
        if atomic:
            mock_chmod.assert_any_call(mock_chmod.call_args[0][0], 0o600)
        else:
            mock_chmod.assert_called_with(file_path, 0o600)


def test_safe_write_json_secure_permissions_fallback(tmp_path, mock_logger):
    """Test secure permissions fallback on OS without chmod support (e.g. Windows)."""
    file_path = tmp_path / "test_secure_fallback.json"
    data = {"secret": "data"}

    with patch("os.chmod", side_effect=OSError("Operation not supported")):
        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=True,
            secure_permissions=True
        )

        assert result is True
        assert file_path.exists()

        with open(file_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == data


def test_safe_write_json_error_handling(tmp_path, mock_logger):
    """Test error handling when atomic or non-atomic write fails."""
    file_path = tmp_path / "test_error.json"
    data = {"key": "value"}

    # Mock atomic failure
    with patch("shutil.move", side_effect=OSError("Permission denied")):
        result_atomic = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=True
        )
        assert result_atomic is False
        assert not file_path.exists()

    # Reset mock
    mock_logger.reset_mock()

    # Mock non-atomic failure
    mock_open_file = mock_open()
    mock_open_file.side_effect = OSError("Permission denied")

    with patch("builtins.open", mock_open_file):
        result_nonatomic = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=False
        )
        assert result_nonatomic is False

    # Both should have triggered a warning
    assert mock_logger.warning.call_count == 1
    assert "Failed to write JSON" in mock_logger.warning.call_args[0][0]


def test_safe_write_json_buffer_on_failure(tmp_path, mock_logger):
    """Test failed write registers with BufferedWriteRegistry when buffer_on_failure=True."""
    file_path = tmp_path / "test_buffer.json"
    data = {"critical": "data"}

    mock_registry = MagicMock()

    with patch("shutil.move", side_effect=OSError("Disk full")), \
         patch.object(BufferedWriteRegistry, "get_instance", return_value=mock_registry):

        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=True,
            buffer_on_failure=True,
            secure_permissions=True
        )

        assert result is False

        # Verify register_pending was called correctly
        mock_registry.register_pending.assert_called_once_with(
            file_path,
            data,
            ANY,
            {"secure_permissions": True}
        )


def test_safe_write_json_cleanup_on_failure(tmp_path, mock_logger):
    """Test temporary file is cleaned up if atomic write fails."""
    file_path = tmp_path / "test_cleanup.json"
    data = {"key": "value"}

    with patch("shutil.move", side_effect=OSError("Failed to move")), \
         patch("os.unlink") as mock_unlink:

        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=True
        )

        assert result is False
        # Ensure unlink was called to clean up the temporary file
        mock_unlink.assert_called_once()


def test_safe_write_json_nonserializable_data(tmp_path, mock_logger):
    """Test handling of data that cannot be JSON serialized."""
    file_path = tmp_path / "test_nonserializable.json"
    data = {"key": lambda: None}

    result = safe_write_json(
        path=file_path,
        data=data,
        logger=mock_logger,
        atomic=True
    )

    assert result is False
    assert not file_path.exists()
    mock_logger.warning.assert_called_once()
    assert "Failed to write JSON" in mock_logger.warning.call_args[0][0]


def test_safe_write_json_empty_dict(tmp_path, mock_logger):
    """Test writing an empty dictionary succeeds."""
    file_path = tmp_path / "test_empty.json"
    data = {}

    result = safe_write_json(
        path=file_path,
        data=data,
        logger=mock_logger,
        atomic=True
    )

    assert result is True
    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == data


def test_safe_write_json_invalid_path(tmp_path, mock_logger):
    """Test attempting to write to an unwritable path fails gracefully."""
    # Create a file, then remove write permissions
    file_path = tmp_path / "readonly.json"
    file_path.touch(mode=0o444)

    # Also restrict the directory so atomic write (which creates a new file and moves) fails
    tmp_path.chmod(0o555)

    data = {"key": "value"}

    try:
        result = safe_write_json(
            path=file_path,
            data=data,
            logger=mock_logger,
            atomic=True
        )

        assert result is False
        mock_logger.warning.assert_called_once()
        assert "Failed to write JSON" in mock_logger.warning.call_args[0][0]
    finally:
        # Restore permissions for cleanup
        tmp_path.chmod(0o777)
        file_path.chmod(0o666)

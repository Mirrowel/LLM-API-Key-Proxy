import pytest
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path
from unittest import mock

from rotator_library.utils.resilient_io import safe_write_json

def test_safe_write_json_oserror_buffering(tmp_path):
    logger = logging.getLogger("test")
    path = tmp_path / "test.json"
    data = {"key": "value"}

    with patch("tempfile.mkstemp", side_effect=OSError("Mocked OSError")) as mock_mkstemp, \
         patch("rotator_library.utils.resilient_io.BufferedWriteRegistry") as MockRegistry:

        mock_registry_instance = MagicMock()
        MockRegistry.get_instance.return_value = mock_registry_instance

        result = safe_write_json(
            path, data, logger, buffer_on_failure=True
        )

        assert result is False
        mock_mkstemp.assert_called_once()
        mock_registry_instance.register_pending.assert_called_once_with(
            path, data, mock.ANY, {"secure_permissions": False}
        )

def test_safe_write_json_permissionerror_buffering(tmp_path):
    logger = logging.getLogger("test")
    path = tmp_path / "test.json"
    data = {"key": "value"}

    with patch("tempfile.mkstemp", side_effect=PermissionError("Mocked PermissionError")) as mock_mkstemp, \
         patch("rotator_library.utils.resilient_io.BufferedWriteRegistry") as MockRegistry:

        mock_registry_instance = MagicMock()
        MockRegistry.get_instance.return_value = mock_registry_instance

        result = safe_write_json(
            path, data, logger, buffer_on_failure=True
        )

        assert result is False
        mock_mkstemp.assert_called_once()
        mock_registry_instance.register_pending.assert_called_once_with(
            path, data, mock.ANY, {"secure_permissions": False}
        )

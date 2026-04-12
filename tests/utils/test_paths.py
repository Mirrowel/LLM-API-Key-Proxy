import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from rotator_library.utils.paths import get_default_root

def test_get_default_root_not_frozen(mocker):
    """Test get_default_root when sys.frozen is False (standard script/library)."""
    # Use mocker to safely mock properties of sys
    mocker.patch.object(sys, "frozen", False, create=True)
    mock_cwd = mocker.patch("rotator_library.utils.paths.Path.cwd")
    mock_cwd.return_value = Path("/mock/cwd")

    result = get_default_root()

    assert result == Path("/mock/cwd")
    mock_cwd.assert_called_once()

def test_get_default_root_frozen(mocker):
    """Test get_default_root when sys.frozen is True (PyInstaller executable)."""
    mocker.patch.object(sys, "frozen", True, create=True)
    mocker.patch.object(sys, "executable", "/mock/bin/executable", create=True)

    result = get_default_root()

    assert result == Path("/mock/bin")

def test_get_default_root_no_frozen_attr(mocker):
    """Test get_default_root when sys has no 'frozen' attribute."""
    if hasattr(sys, 'frozen'):
        mocker.patch.object(sys, 'frozen', None)
        delattr(sys, 'frozen')

    mock_cwd = mocker.patch("rotator_library.utils.paths.Path.cwd")
    mock_cwd.return_value = Path("/mock/cwd2")

    result = get_default_root()

    assert result == Path("/mock/cwd2")
    mock_cwd.assert_called_once()

def test_get_default_root_cwd_failure(mocker):
    """Test get_default_root when Path.cwd() raises an OSError."""
    mocker.patch.object(sys, "frozen", False, create=True)
    mock_cwd = mocker.patch("rotator_library.utils.paths.Path.cwd")
    mock_cwd.side_effect = OSError("CWD not accessible")

    mock_home = mocker.patch("rotator_library.utils.paths.Path.home")
    # Return a mock Path object instead of real Path, so we can mock exists() on it
    mock_path_obj = MagicMock(spec=Path)
    mock_path_obj.exists.return_value = True
    mock_home.return_value = mock_path_obj

    result = get_default_root()

    # Should fallback to home dir
    assert result == mock_path_obj

def test_get_default_root_cwd_failure_no_home(mocker):
    """Test get_default_root when Path.cwd() raises OSError and home doesn't exist."""
    mocker.patch.object(sys, "frozen", False, create=True)
    mock_cwd = mocker.patch("rotator_library.utils.paths.Path.cwd")
    mock_cwd.side_effect = OSError("CWD not accessible")

    mock_home = mocker.patch("rotator_library.utils.paths.Path.home")
    mock_path_obj = MagicMock(spec=Path)
    mock_path_obj.exists.return_value = False
    mock_home.return_value = mock_path_obj

    result = get_default_root()

    # Should fallback to root dir
    assert result == Path("/")

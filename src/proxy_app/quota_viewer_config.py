"""
Configuration management for the Quota Viewer.

Handles remote proxy configurations including:
- Multiple remote proxies (local, VPS, etc.)
- API key storage per remote
- Default and last-used remote tracking
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class QuotaViewerConfig:
    """Manages quota viewer configuration including remote proxies."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the config manager.

        Args:
            config_path: Path to config file. Defaults to quota_viewer_config.json
                        in the current directory or EXE directory.
        """
        if config_path is None:
            import sys

            if getattr(sys, "frozen", False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path.cwd()
            config_path = base_dir / "quota_viewer_config.json"

        self.config_path = config_path
        self.config = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load config from file or return defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                # Ensure required fields exist
                if "remotes" not in config:
                    config["remotes"] = []
                return config
            except (json.JSONDecodeError, IOError):
                pass

        # Return default config with Local remote
        return {
            "remotes": [
                {
                    "name": "Local",
                    "host": "127.0.0.1",
                    "port": 8000,
                    "api_key": None,
                    "is_default": True,
                }
            ],
            "last_used": "Local",
        }

    def _save(self) -> bool:
        """Save config to file. Returns True on success."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
            return True
        except IOError:
            return False

    def get_remotes(self) -> List[Dict[str, Any]]:
        """Get list of all configured remotes."""
        return self.config.get("remotes", [])

    def get_remote_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a remote by name."""
        for remote in self.config.get("remotes", []):
            if remote["name"] == name:
                return remote
        return None

    def get_default_remote(self) -> Optional[Dict[str, Any]]:
        """Get the default remote."""
        for remote in self.config.get("remotes", []):
            if remote.get("is_default"):
                return remote
        # Fallback to first remote
        remotes = self.config.get("remotes", [])
        return remotes[0] if remotes else None

    def get_last_used_remote(self) -> Optional[Dict[str, Any]]:
        """Get the last used remote, or default if not set."""
        last_used_name = self.config.get("last_used")
        if last_used_name:
            remote = self.get_remote_by_name(last_used_name)
            if remote:
                return remote
        return self.get_default_remote()

    def set_last_used(self, name: str) -> bool:
        """Set the last used remote name."""
        self.config["last_used"] = name
        return self._save()

    def add_remote(
        self,
        name: str,
        host: str,
        port: int = 8000,
        api_key: Optional[str] = None,
        is_default: bool = False,
    ) -> bool:
        """
        Add a new remote configuration.

        Args:
            name: Display name for the remote
            host: Hostname or IP address
            port: Port number (default 8000)
            api_key: Optional API key for authentication
            is_default: Whether this should be the default remote

        Returns:
            True on success, False if name already exists
        """
        # Check for duplicate name
        if self.get_remote_by_name(name):
            return False

        # If setting as default, clear default from others
        if is_default:
            for remote in self.config.get("remotes", []):
                remote["is_default"] = False

        remote = {
            "name": name,
            "host": host,
            "port": port,
            "api_key": api_key,
            "is_default": is_default,
        }
        self.config.setdefault("remotes", []).append(remote)
        return self._save()

    def update_remote(self, name: str, **kwargs) -> bool:
        """
        Update an existing remote configuration.

        Args:
            name: Name of the remote to update
            **kwargs: Fields to update (host, port, api_key, is_default, new_name)

        Returns:
            True on success, False if remote not found
        """
        remote = self.get_remote_by_name(name)
        if not remote:
            return False

        # Handle rename
        if "new_name" in kwargs:
            new_name = kwargs.pop("new_name")
            if new_name != name and self.get_remote_by_name(new_name):
                return False  # New name already exists
            remote["name"] = new_name
            # Update last_used if it was this remote
            if self.config.get("last_used") == name:
                self.config["last_used"] = new_name

        # If setting as default, clear default from others
        if kwargs.get("is_default"):
            for r in self.config.get("remotes", []):
                r["is_default"] = False

        # Update other fields
        for key in ("host", "port", "api_key", "is_default"):
            if key in kwargs:
                remote[key] = kwargs[key]

        return self._save()

    def delete_remote(self, name: str) -> bool:
        """
        Delete a remote configuration.

        Args:
            name: Name of the remote to delete

        Returns:
            True on success, False if remote not found or is the only one
        """
        remotes = self.config.get("remotes", [])
        if len(remotes) <= 1:
            return False  # Don't delete the last remote

        for i, remote in enumerate(remotes):
            if remote["name"] == name:
                remotes.pop(i)
                # Update last_used if it was this remote
                if self.config.get("last_used") == name:
                    self.config["last_used"] = remotes[0]["name"] if remotes else None
                return self._save()
        return False

    def set_default_remote(self, name: str) -> bool:
        """Set a remote as the default."""
        remote = self.get_remote_by_name(name)
        if not remote:
            return False

        # Clear default from all remotes
        for r in self.config.get("remotes", []):
            r["is_default"] = False

        # Set new default
        remote["is_default"] = True
        return self._save()

    def sync_with_launcher_config(self) -> None:
        """
        Sync the Local remote with launcher_config.json if it exists.

        This ensures the Local remote always matches the launcher settings.
        """
        import sys

        if getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).parent
        else:
            base_dir = Path.cwd()

        launcher_config_path = base_dir / "launcher_config.json"

        if launcher_config_path.exists():
            try:
                with open(launcher_config_path, "r", encoding="utf-8") as f:
                    launcher_config = json.load(f)

                host = launcher_config.get("host", "127.0.0.1")
                port = launcher_config.get("port", 8000)

                # Update Local remote
                local_remote = self.get_remote_by_name("Local")
                if local_remote:
                    local_remote["host"] = host
                    local_remote["port"] = port
                    self._save()
                else:
                    # Create Local remote if it doesn't exist
                    self.add_remote("Local", host, port, is_default=True)

            except (json.JSONDecodeError, IOError):
                pass

    def get_api_key_from_env(self) -> Optional[str]:
        """
        Get PROXY_API_KEY from .env file for Local remote.

        Returns:
            API key string or None
        """
        import sys

        if getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).parent
        else:
            base_dir = Path.cwd()

        env_path = base_dir / ".env"
        if not env_path.exists():
            return None

        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("PROXY_API_KEY="):
                        value = line.split("=", 1)[1].strip()
                        # Remove quotes if present
                        if value and value[0] in ('"', "'") and value[-1] == value[0]:
                            value = value[1:-1]
                        return value if value else None
        except IOError:
            pass
        return None

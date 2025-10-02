import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

lib_logger = logging.getLogger('rotator_library')

OAUTH_BASE_DIR = Path.cwd() / "oauth_creds"
OAUTH_BASE_DIR.mkdir(exist_ok=True)

# Standard paths where tools like `gemini login` store credentials.
DEFAULT_OAUTH_PATHS = {
    "gemini": Path.home() / ".gemini" / "oauth_creds.json",
    "qwen": Path.home() / ".qwen" / "oauth_creds.json",
    # Add other providers like 'claude' here if they have a standard CLI path
}

class CredentialManager:
    """
    Discovers OAuth credential files from standard locations, copies them locally,
    and updates the configuration to use the local paths.
    """
    def __init__(self, oauth_config: Dict[str, List[str]]):
        self.oauth_config = oauth_config

    def discover_and_prepare(self) -> Dict[str, List[str]]:
        """
        Processes the initial OAuth config. If a path is empty, it tries to
        discover the file from a default location. It then copies the file
        locally if it doesn't already exist and returns the updated config
        pointing to the local paths.
        """
        updated_config = {}
        for provider, paths in self.oauth_config.items():
            updated_paths = []
            for i, path_str in enumerate(paths):
                account_id = i + 1
                source_path = self._resolve_source_path(provider, path_str)
                
                if not source_path or not source_path.exists():
                    lib_logger.warning(f"Could not find OAuth file for {provider} account #{account_id}. Skipping.")
                    continue

                local_filename = f"{provider}_oauth_{account_id}.json"
                local_path = OAUTH_BASE_DIR / local_filename

                if not local_path.exists():
                    try:
                        shutil.copy(source_path, local_path)
                        lib_logger.info(f"Copied '{source_path}' to local credentials at '{local_path}'.")
                    except Exception as e:
                        lib_logger.error(f"Failed to copy OAuth file for {provider} account #{account_id}: {e}")
                        continue
                
                updated_paths.append(str(local_path.resolve()))
            
            if updated_paths:
                updated_config[provider] = updated_paths
        
        return updated_config

    def _resolve_source_path(self, provider: str, specified_path: Optional[str]) -> Optional[Path]:
        """Determines the source path for a credential file."""
        if specified_path:
            # If a path is given, use it directly.
            return Path(specified_path).expanduser()
        
        # If no path is given, try the default location.
        return DEFAULT_OAUTH_PATHS.get(provider)
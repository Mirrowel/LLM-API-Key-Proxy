import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

lib_logger = logging.getLogger('rotator_library')

OAUTH_BASE_DIR = Path.cwd() / "oauth_creds"
OAUTH_BASE_DIR.mkdir(exist_ok=True)

# Standard directories where tools like `gemini login` store credentials.
DEFAULT_OAUTH_DIRS = {
    "gemini_cli": Path.home() / ".gemini",
    "qwen_code": Path.home() / ".qwen",
    # Add other providers like 'claude' here if they have a standard CLI path
}

class CredentialManager:
    """
    Discovers OAuth credential files from standard locations, copies them locally,
    and updates the configuration to use the local paths.
    """
    def __init__(self, env_vars: Dict[str, str]):
        self.env_vars = env_vars

    def discover_and_prepare(self) -> Dict[str, List[str]]:
        lib_logger.info("Starting automated OAuth credential discovery...")
        final_config = {}

        # Extract OAuth paths from environment variables first
        env_oauth_paths = {}
        for key, value in self.env_vars.items():
            if "_OAUTH_" in key:
                provider = key.split("_OAUTH_")[0].lower()
                if provider not in env_oauth_paths:
                    env_oauth_paths[provider] = []
                if value: # Only consider non-empty values
                    env_oauth_paths[provider].append(value)

        for provider, default_dir in DEFAULT_OAUTH_DIRS.items():
            # Check for existing local credentials first. If found, use them and skip discovery.
            local_provider_creds = sorted(list(OAUTH_BASE_DIR.glob(f"{provider}_oauth_*.json")))
            if local_provider_creds:
                lib_logger.info(f"Found {len(local_provider_creds)} existing local credential(s) for {provider}. Skipping discovery.")
                final_config[provider] = [str(p.resolve()) for p in local_provider_creds]
                continue

            # If no local credentials exist, proceed with a one-time discovery and copy.
            discovered_paths = set()

            # 1. Add paths from environment variables first, as they are overrides
            for path_str in env_oauth_paths.get(provider, []):
                path = Path(path_str).expanduser()
                if path.exists():
                    discovered_paths.add(path)
            
            # 2. If no overrides are provided via .env, scan the default directory
            if not discovered_paths and default_dir.exists():
                for json_file in default_dir.glob('*.json'):
                    discovered_paths.add(json_file)
            
            if not discovered_paths:
                lib_logger.debug(f"No credential files found for provider: {provider}")
                continue

            prepared_paths = []
            # Sort paths to ensure consistent numbering for the initial copy
            for i, source_path in enumerate(sorted(list(discovered_paths))):
                account_id = i + 1
                local_filename = f"{provider}_oauth_{account_id}.json"
                local_path = OAUTH_BASE_DIR / local_filename

                try:
                    # Since we've established no local files exist, we can copy directly.
                    shutil.copy(source_path, local_path)
                    lib_logger.info(f"Copied '{source_path.name}' to local pool at '{local_path}'.")
                    prepared_paths.append(str(local_path.resolve()))
                except Exception as e:
                    lib_logger.error(f"Failed to process OAuth file from '{source_path}': {e}")
            
            if prepared_paths:
                lib_logger.info(f"Discovered and prepared {len(prepared_paths)} credential(s) for provider: {provider}")
                final_config[provider] = prepared_paths

        lib_logger.info("OAuth credential discovery complete.")
        return final_config

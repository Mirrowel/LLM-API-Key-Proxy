import os
import re
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

lib_logger = logging.getLogger('rotator_library')

OAUTH_BASE_DIR = Path.cwd() / "oauth_creds"
OAUTH_BASE_DIR.mkdir(exist_ok=True)

# Standard directories where tools like `gemini login` store credentials.
DEFAULT_OAUTH_DIRS = {
    "gemini_cli": Path.home() / ".gemini",
    "qwen_code": Path.home() / ".qwen",
    "iflow": Path.home() / ".iflow",
    "antigravity": Path.home() / ".antigravity",
    # Add other providers like 'claude' here if they have a standard CLI path
}

# OAuth providers that support environment variable-based credentials
# Maps provider name to the ENV_PREFIX used by the provider
ENV_OAUTH_PROVIDERS = {
    "gemini_cli": "GEMINI_CLI",
    "antigravity": "ANTIGRAVITY",
    "qwen_code": "QWEN_CODE",
    "iflow": "IFLOW",
}


class CredentialManager:
    """
    Discovers OAuth credential files from standard locations, copies them locally,
    and updates the configuration to use the local paths.
    
    Also discovers environment variable-based OAuth credentials for stateless deployments.
    Supports two env var formats:
    
    1. Single credential (legacy): PROVIDER_ACCESS_TOKEN, PROVIDER_REFRESH_TOKEN
    2. Multiple credentials (numbered): PROVIDER_1_ACCESS_TOKEN, PROVIDER_2_ACCESS_TOKEN, etc.
    
    When env-based credentials are detected, virtual paths like "env://provider/1" are created.
    """
    def __init__(self, env_vars: Dict[str, str]):
        self.env_vars = env_vars

    def _discover_env_oauth_credentials(self) -> Dict[str, List[str]]:
        """
        Discover OAuth credentials defined via environment variables.
        
        Supports two formats:
        1. Single credential: ANTIGRAVITY_ACCESS_TOKEN + ANTIGRAVITY_REFRESH_TOKEN
        2. Multiple credentials: ANTIGRAVITY_1_ACCESS_TOKEN + ANTIGRAVITY_1_REFRESH_TOKEN, etc.
        
        Returns:
            Dict mapping provider name to list of virtual paths (e.g., "env://antigravity/1")
        """
        env_credentials: Dict[str, Set[str]] = {}
        
        # [DEBUG-REMOVE] Log all environment variable keys for OAuth providers
        print(f"[DEBUG-REMOVE] === Scanning environment for OAuth credentials ===")
        print(f"[DEBUG-REMOVE] ENV_OAUTH_PROVIDERS: {list(ENV_OAUTH_PROVIDERS.keys())}")
        
        for provider, env_prefix in ENV_OAUTH_PROVIDERS.items():
            found_indices: Set[str] = set()
            print(f"[DEBUG-REMOVE] Scanning for provider '{provider}' with prefix '{env_prefix}'")
            
            # Check for numbered credentials (PROVIDER_N_ACCESS_TOKEN pattern)
            # Pattern: ANTIGRAVITY_1_ACCESS_TOKEN, ANTIGRAVITY_2_ACCESS_TOKEN, etc.
            numbered_pattern = re.compile(rf"^{env_prefix}_(\d+)_ACCESS_TOKEN$")
            
            # [DEBUG-REMOVE] Show all matching environment variable keys
            matching_keys = [k for k in self.env_vars.keys() if env_prefix in k]
            if matching_keys:
                print(f"[DEBUG-REMOVE]   Found {len(matching_keys)} keys with '{env_prefix}': {matching_keys}")
            else:
                print(f"[DEBUG-REMOVE]   No keys found with '{env_prefix}' prefix")
            
            for key in self.env_vars.keys():
                match = numbered_pattern.match(key)
                if match:
                    index = match.group(1)
                    # Verify refresh token also exists
                    refresh_key = f"{env_prefix}_{index}_REFRESH_TOKEN"
                    if refresh_key in self.env_vars and self.env_vars[refresh_key]:
                        found_indices.add(index)
                        print(f"[DEBUG-REMOVE]   ✓ Found numbered credential {index} for {provider}")
                    else:
                        print(f"[DEBUG-REMOVE]   ✗ Missing REFRESH_TOKEN for {provider} credential {index}")
            
            # Check for legacy single credential (PROVIDER_ACCESS_TOKEN pattern)
            # Only use this if no numbered credentials exist
            if not found_indices:
                access_key = f"{env_prefix}_ACCESS_TOKEN"
                refresh_key = f"{env_prefix}_REFRESH_TOKEN"
                print(f"[DEBUG-REMOVE]   Checking legacy format: {access_key}, {refresh_key}")
                if (access_key in self.env_vars and self.env_vars[access_key] and
                    refresh_key in self.env_vars and self.env_vars[refresh_key]):
                    # Use "0" as the index for legacy single credential
                    found_indices.add("0")
                    print(f"[DEBUG-REMOVE]   ✓ Found legacy single credential for {provider}")
                else:
                    print(f"[DEBUG-REMOVE]   ✗ No legacy credential found for {provider}")
            
            if found_indices:
                env_credentials[provider] = found_indices
                lib_logger.info(f"Found {len(found_indices)} env-based credential(s) for {provider}")
                print(f"[DEBUG-REMOVE]   RESULT: {len(found_indices)} credential(s) registered for {provider}")
            else:
                print(f"[DEBUG-REMOVE]   RESULT: No credentials found for {provider}")
        
        # Convert to virtual paths
        result: Dict[str, List[str]] = {}
        for provider, indices in env_credentials.items():
            # Sort indices numerically for consistent ordering
            sorted_indices = sorted(indices, key=lambda x: int(x))
            result[provider] = [f"env://{provider}/{idx}" for idx in sorted_indices]
        
        return result

    def discover_and_prepare(self) -> Dict[str, List[str]]:
        lib_logger.info("Starting automated OAuth credential discovery...")
        final_config = {}

        # PHASE 1: Discover environment variable-based OAuth credentials
        # These take priority for stateless deployments
        env_oauth_creds = self._discover_env_oauth_credentials()
        for provider, virtual_paths in env_oauth_creds.items():
            lib_logger.info(f"Using {len(virtual_paths)} env-based credential(s) for {provider}")
            final_config[provider] = virtual_paths

        # Extract OAuth file paths from environment variables
        env_oauth_paths = {}
        for key, value in self.env_vars.items():
            if "_OAUTH_" in key:
                provider = key.split("_OAUTH_")[0].lower()
                if provider not in env_oauth_paths:
                    env_oauth_paths[provider] = []
                if value: # Only consider non-empty values
                    env_oauth_paths[provider].append(value)

        # PHASE 2: Discover file-based OAuth credentials
        for provider, default_dir in DEFAULT_OAUTH_DIRS.items():
            # Skip if already discovered from environment variables
            if provider in final_config:
                lib_logger.debug(f"Skipping file discovery for {provider} - using env-based credentials")
                continue
            
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
            # [MODIFIED] This logic is now disabled to prefer local-first credential management.
            # if not discovered_paths and default_dir.exists():
            #     for json_file in default_dir.glob('*.json'):
            #         discovered_paths.add(json_file)
            
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

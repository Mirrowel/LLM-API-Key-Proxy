# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

import os
import sys
import platform
import subprocess


def get_providers():
    """
    Scans the 'src/rotator_library/providers' directory to find all provider modules.
    Returns a list of hidden import arguments for PyInstaller.
    """
    hidden_imports = []
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the providers directory relative to this script's location
    providers_path = os.path.join(script_dir, "..", "rotator_library", "providers")

    if not os.path.isdir(providers_path):
        print(f"Error: Directory not found at '{os.path.abspath(providers_path)}'")
        return []

    for filename in os.listdir(providers_path):
        if filename.endswith("_provider.py") and filename != "__init__.py":
            module_name = f"rotator_library.providers.{filename[:-3]}"
            hidden_imports.append(f"--hidden-import={module_name}")
    return hidden_imports


def main():
    """
    Constructs and runs the PyInstaller command to build the executable.
    """
    # Base PyInstaller command with optimizations
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "proxy_app",
        "--paths",
        "../",
        "--paths",
        ".",
        # Core imports
        "--hidden-import=rotator_library",
        "--hidden-import=tiktoken_ext.openai_public",
        "--hidden-import=tiktoken_ext",
        "--collect-data",
        "litellm",
        # Optimization: Exclude unused heavy modules
        "--exclude-module=matplotlib",
        "--exclude-module=IPython",
        "--exclude-module=jupyter",
        "--exclude-module=notebook",
        "--exclude-module=PIL.ImageTk",
        # Optimization: Enable UPX compression (if available)
        "--upx-dir=upx"
        if platform.system() != "Darwin"
        else "--noupx",  # macOS has issues with UPX
        # Optimization: Strip debug symbols (smaller binary)
        "--strip"
        if platform.system() != "Windows"
        else "--console",  # Windows gets clean console
    ]

    # Add hidden imports for providers
    provider_imports = get_providers()
    if not provider_imports:
        print(
            "Warning: No providers found. The build might not include any LLM providers."
        )
    command.extend(provider_imports)

    # Add the main script
    command.append("main.py")

    # Execute the command
    print(f"Running command: {' '.join(command)}")
    try:
        # Run PyInstaller from the script's directory to ensure relative paths are correct
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(command, check=True, cwd=script_dir)
        print("Build successful!")
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
    except FileNotFoundError:
        print("Error: PyInstaller is not installed or not in the system's PATH.")


if __name__ == "__main__":
    main()

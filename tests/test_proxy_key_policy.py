# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""Proxy API key policy tests (default key + localhost binding).

Subprocess-based: the policy runs inside main.py's module flow, so the
real entry point (`python src/proxy_app/main.py`) is the only faithful
harness — argv, env, and stdin shape all matter.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "src" / "proxy_app" / "main.py"

DEFAULT_KEY = "VerysecretKey"


def _launch_cmd(
    cmd: list[str],
    extra_env: dict[str, str],
    cwd: Path | None = None,
    stdin: subprocess.IO | None = None,
) -> subprocess.Popen:
    env = {**os.environ, **extra_env}
    # Strip proxy-affecting variables so tests control the scenario.
    for var in ("PROXY_API_KEY", "UVICORN_HOST", "UVICORN_PORT"):
        env.pop(var, None)
    env.update(extra_env)
    # A bare cwd would trigger the interactive first-time onboarding flow
    # (no .env found); scenarios only exercise the key policy, so seed a
    # stub .env to mark the deployment as onboarded.
    if cwd is not None and not (cwd / ".env").exists():
        (cwd / ".env").write_text("# policy-test stub\n", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        stdin=stdin if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _launch(
    extra_env: dict[str, str],
    *cli_args: str,
    stdin: subprocess.IO | None = None,
    cwd: Path | None = None,
) -> subprocess.Popen:
    return _launch_cmd(
        [sys.executable, str(MAIN), *cli_args],
        extra_env,
        cwd=cwd,
        stdin=stdin,
    )


def _run_until_ready_or_exit(proc: subprocess.Popen, timeout: float = 45.0) -> tuple[int, str]:
    import time

    output: list[str] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        if line:
            output.append(line)
            if "Uvicorn running on" in line or "Application startup complete" in line:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return 0, "".join(output)
        elif proc.poll() is not None:
            return proc.returncode or 0, "".join(output)
    proc.kill()
    return -1, "".join(output)


def test_default_key_non_localheadless_blocks(tmp_path: Path) -> None:
    """No tty + default key + 0.0.0.0 -> the proxy refuses to serve."""
    proc = _launch({"PROXY_API_KEY": DEFAULT_KEY}, "--host", "0.0.0.0", "--port", "18123", cwd=tmp_path)
    code, output = _run_until_ready_or_exit(proc, timeout=60)
    assert "REFUSING TO START" in output
    assert code == 1


def test_default_key_localhost_boots(tmp_path: Path) -> None:
    """Default key on a loopback bind is the sanctioned local-dev path."""
    proc = _launch({"PROXY_API_KEY": DEFAULT_KEY}, "--host", "127.0.0.1", "--port", "18124", cwd=tmp_path)
    code, output = _run_until_ready_or_exit(proc)
    assert "Uvicorn running on" in output or "Application startup complete" in output
    assert "REFUSING TO START" not in output


def test_custom_key_non_local_boots(tmp_path: Path) -> None:
    """A real key never trips the policy."""
    proc = _launch({"PROXY_API_KEY": "k_test_custom_key_value_123456"}, "--host", "0.0.0.0", "--port", "18125", cwd=tmp_path)
    code, output = _run_until_ready_or_exit(proc)
    assert "Uvicorn running on" in output or "Application startup complete" in output
    assert "REFUSING TO START" not in output


def test_no_key_boots_open(tmp_path: Path) -> None:
    """Unset key keeps the documented open-access behavior (out of scope)."""
    proc = _launch({}, "--host", "0.0.0.0", "--port", "18126", cwd=tmp_path)
    code, output = _run_until_ready_or_exit(proc)
    assert "Uvicorn running on" in output or "Application startup complete" in output


def test_relative_import_regression(tmp_path: Path) -> None:
    """The launcher crash: relative imports under direct script execution."""
    proc = _launch({"PROXY_API_KEY": DEFAULT_KEY}, "--host", "127.0.0.1", "--port", "18127", cwd=tmp_path)
    code, output = _run_until_ready_or_exit(proc)
    assert "ImportError" not in output
    assert "attempted relative import" not in output


def test_m_uvicorn_public_bind_blocks(tmp_path: Path) -> None:
    """`python -m uvicorn` (argv0 becomes __main__.py) still hits the policy."""
    proc = _launch_cmd(
        [sys.executable, "-m", "uvicorn", "proxy_app.main:app", "--host", "0.0.0.0", "--port", "18131"],
        {"PROXY_API_KEY": DEFAULT_KEY, "PYTHONPATH": str(ROOT / "src")},
        cwd=tmp_path,
    )
    code, output = _run_until_ready_or_exit(proc, timeout=90)
    assert "REFUSING TO START" in output
    assert code == 1


def test_uvicorn_env_host_bypass_closed(tmp_path: Path) -> None:
    """UVICORN_HOST=0.0.0.0 (click envvar, argv lacks --host) cannot serve the default key."""
    proc = _launch_cmd(
        [sys.executable, "-m", "uvicorn", "proxy_app.main:app", "--port", "18132"],
        {"PROXY_API_KEY": DEFAULT_KEY, "UVICORN_HOST": "0.0.0.0", "PYTHONPATH": str(ROOT / "src")},
        cwd=tmp_path,
    )
    code, output = _run_until_ready_or_exit(proc, timeout=90)
    assert "REFUSING TO START" in output
    assert code == 1


def test_uvicorn_cli_localhost_default_boots(tmp_path: Path) -> None:
    """uvicorn CLI with no host override (its documented localhost default) boots."""
    proc = _launch_cmd(
        [sys.executable, "-m", "uvicorn", "proxy_app.main:app", "--port", "18133"],
        {"PROXY_API_KEY": DEFAULT_KEY, "PYTHONPATH": str(ROOT / "src")},
        cwd=tmp_path,
    )
    code, output = _run_until_ready_or_exit(proc, timeout=90)
    assert "Uvicorn running on" in output or "Application startup complete" in output
    assert "REFUSING TO START" not in output


def test_programmatic_uvicorn_run_blocks(tmp_path: Path) -> None:
    """uvicorn.run() imports are stack-detected; unproven bind fails closed."""
    proc = _launch_cmd(
        [sys.executable, "-c", "import uvicorn\nuvicorn.run('proxy_app.main:app', host='127.0.0.1', port=18134)"],
        {"PROXY_API_KEY": DEFAULT_KEY, "PYTHONPATH": str(ROOT / "src")},
        cwd=tmp_path,
    )
    code, output = _run_until_ready_or_exit(proc, timeout=90)
    assert "REFUSING TO START" in output
    assert code == 1


def test_import_does_not_launch_tui(tmp_path: Path) -> None:
    """`import proxy_app.main` never blocks on the interactive TUI."""
    proc = _launch_cmd(
        [sys.executable, "-c", "import proxy_app.main; print('IMPORT-NO-TUI-OK')"],
        {"PROXY_API_KEY": DEFAULT_KEY, "PYTHONPATH": str(ROOT / "src")},
        cwd=tmp_path,
    )
    code, output = _run_until_ready_or_exit(proc, timeout=90)
    assert code == 0
    assert "IMPORT-NO-TUI-OK" in output
    assert "REFUSING TO START" not in output


class TestPureHelpers:
    def test_is_localhost_bind(self) -> None:
        from proxy_app.key_policy import is_localhost_bind

        assert is_localhost_bind("127.0.0.1")
        assert is_localhost_bind("localhost")
        assert is_localhost_bind("::1")
        assert is_localhost_bind("[::1]")
        assert is_localhost_bind("LOCALHOST  ")
        assert not is_localhost_bind("0.0.0.0")
        assert not is_localhost_bind("")
        assert not is_localhost_bind(None)
        assert not is_localhost_bind("192.168.1.5")
        # All of 127.0.0.0/8 and IPv6 loopback spellings are loopback.
        assert is_localhost_bind("127.0.0.5")
        assert is_localhost_bind("0:0:0:0:0:0:0:1")
        # Unknown hostnames fail closed (treated as public).
        assert not is_localhost_bind("proxy.internal.corp")

    def test_generate_shape(self) -> None:
        from proxy_app.key_policy import generate_proxy_api_key

        key = generate_proxy_api_key()
        assert len(key) >= 24
        assert key.isalnum() or all(c.isalnum() or c in "-_" for c in key)

    def test_mask_fixed_shape_no_length(self) -> None:
        from proxy_app.startup_display import mask_secret_for_display

        masked = mask_secret_for_display("abcdefghijklmnop")
        assert masked == "abc…nop"
        assert mask_secret_for_display("short") == "Set (redacted)"
        assert "18" not in masked  # no char-count leakage


class TestLaunchDetection:
    """serving_bind_host: every launch mode resolves to a bind or None."""

    def test_m_uvicorn_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from proxy_app import key_policy

        # `python -m uvicorn` puts the package's __main__.py in argv[0].
        monkeypatch.setattr(sys, "argv", [r"C:\py\site-packages\uvicorn\__main__.py"], raising=False)
        assert key_policy.serving_bind_host() == "127.0.0.1"  # uvicorn's localhost default
        monkeypatch.setattr(sys, "argv", [r"C:\py\site-packages\uvicorn\__main__.py", "--host", "0.0.0.0"], raising=False)
        assert key_policy.serving_bind_host() == "0.0.0.0"

    def test_uvicorn_env_host_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from proxy_app import key_policy

        monkeypatch.setattr(sys, "argv", [r"C:\x\Scripts\uvicorn.exe"], raising=False)
        monkeypatch.delenv("UVICORN_HOST", raising=False)
        assert key_policy.serving_bind_host() == "127.0.0.1"
        monkeypatch.setenv("UVICORN_HOST", "0.0.0.0")
        assert key_policy.serving_bind_host() == "0.0.0.0"

    def test_gunicorn_hypercorn_unproven_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from proxy_app import key_policy

        monkeypatch.setattr(sys, "argv", ["/usr/bin/gunicorn", "-w", "4", "app:app"], raising=False)
        assert key_policy.serving_bind_host() == key_policy.UNPROVEN_BIND_HOST
        assert not key_policy.is_localhost_bind(key_policy.serving_bind_host())
        monkeypatch.setattr(sys, "argv", [r"C:\py\hypercorn\__main__.py"], raising=False)
        assert key_policy.serving_bind_host() == key_policy.UNPROVEN_BIND_HOST

    def test_plain_import_is_non_serving(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from proxy_app import key_policy

        monkeypatch.setattr(sys, "argv", ["tool.py", "--flag"], raising=False)
        assert key_policy.serving_bind_host() is None

    def test_programmatic_run_frames_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A uvicorn frame on the import stack (uvicorn.run of the app string)
        detects the launch even though argv belongs to the user's script."""
        from proxy_app import key_policy

        namespace: dict[str, object] = {"__name__": "uvicorn.config"}
        monkeypatch.setattr(sys, "argv", ["run_server.py"], raising=False)
        exec(
            "from proxy_app import key_policy\n"
            "detected = key_policy.serving_bind_host()",
            namespace,
        )
        assert namespace["detected"] == key_policy.UNPROVEN_BIND_HOST

    def test_port_parity_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from proxy_app import key_policy

        monkeypatch.setattr(sys, "argv", [r"C:\x\uvicorn.exe", "--port", "9001"], raising=False)
        assert key_policy.uvicorn_bind_port() == 9001
        monkeypatch.setattr(sys, "argv", [r"C:\x\uvicorn.exe"], raising=False)
        monkeypatch.delenv("UVICORN_PORT", raising=False)
        assert key_policy.uvicorn_bind_port() is None
        monkeypatch.setenv("UVICORN_PORT", "9002")
        assert key_policy.uvicorn_bind_port() == 9002


class TestPromptFlow:
    def test_skip_keeps_default(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        from proxy_app import key_policy

        monkeypatch.setenv("PROXY_API_KEY", DEFAULT_KEY)
        monkeypatch.setattr(key_policy, "_interactive_stream_available", lambda: True)
        answers = iter(["3"])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))
        result = key_policy.enforce_proxy_key_policy("0.0.0.0")
        assert result is None

    def test_generate_adopts_and_saves(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from proxy_app import key_policy

        monkeypatch.setenv("PROXY_API_KEY", DEFAULT_KEY)
        monkeypatch.setattr(key_policy, "_interactive_stream_available", lambda: True)
        monkeypatch.setattr(key_policy, "resolve_env_file", lambda: tmp_path / ".env")
        answers = iter(["2", ""])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))
        result = key_policy.enforce_proxy_key_policy("10.0.0.4")
        assert result is not None
        assert os.environ["PROXY_API_KEY"] == result
        saved = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "PROXY_API_KEY=" in saved and result in saved

    def test_enter_own_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from proxy_app import key_policy

        monkeypatch.setenv("PROXY_API_KEY", DEFAULT_KEY)
        monkeypatch.setattr(key_policy, "_interactive_stream_available", lambda: True)
        monkeypatch.setattr(key_policy, "resolve_env_file", lambda: tmp_path / ".env")
        answers = iter(["1", "my_own_secret_key", ""])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))
        result = key_policy.enforce_proxy_key_policy("example.internal")
        assert result == "my_own_secret_key"
        assert "my_own_secret_key" in (tmp_path / ".env").read_text(encoding="utf-8")

    def test_localhost_never_prompts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from proxy_app import key_policy

        monkeypatch.setenv("PROXY_API_KEY", DEFAULT_KEY)

        def _fail(*_: object, **__: object) -> str:
            raise AssertionError("policy must not prompt on localhost")

        monkeypatch.setattr("builtins.input", _fail)
        assert key_policy.enforce_proxy_key_policy("127.0.0.1") is None

    def test_full_loopback_subnet_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from proxy_app import key_policy

        monkeypatch.setenv("PROXY_API_KEY", DEFAULT_KEY)

        def _fail(*_: object, **__: object) -> str:
            raise AssertionError("policy must not prompt on 127.0.0.5")

        monkeypatch.setattr("builtins.input", _fail)
        assert key_policy.enforce_proxy_key_policy("127.0.0.5") is None

    def test_unknown_hostname_treated_public(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bind we cannot prove local fails closed (D4): prompt/block."""
        from proxy_app import key_policy

        monkeypatch.setenv("PROXY_API_KEY", DEFAULT_KEY)
        monkeypatch.setattr(key_policy, "_interactive_stream_available", lambda: False)
        with pytest.raises(SystemExit):
            key_policy.enforce_proxy_key_policy("proxy.internal.corp")

    def test_unproven_bind_blocks_headless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from proxy_app import key_policy

        monkeypatch.setenv("PROXY_API_KEY", DEFAULT_KEY)
        monkeypatch.setattr(key_policy, "_interactive_stream_available", lambda: False)
        with pytest.raises(SystemExit):
            key_policy.enforce_proxy_key_policy(key_policy.UNPROVEN_BIND_HOST)

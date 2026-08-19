#!/usr/bin/env python3
"""Mirrobot share-link decryptor.

Recovers the session share URL from an encrypted `MRB1.<base64>` blob
produced by .github/scripts/share-filter.sh in CI.

Only the URL is encrypted - the surrounding `context:` line (repo, PR,
head SHA, review type, run, actor) is public and may be pasted along; it
is parsed and displayed/logging next to the recovered link.

Usage:
  python decrypt_share_link.py                 # interactive TUI
  python decrypt_share_link.py "<pasted text>" # one-shot decrypt
  python decrypt_share_link.py "<blob>" --open # decrypt + open browser
  python decrypt_share_link.py setup           # keygen + push repo secret
  python decrypt_share_link.py config show
  python decrypt_share_link.py config set <key> <value>
  python decrypt_share_link.py --self-test

Config: ~/.config/mirrobot/config.json
  private_key_path  (default ~/.config/mirrobot/share-link.pem, env
                     MIRROBOT_SHARE_KEY overrides)
  persist           (default false - nothing is written to disk until
                     you explicitly enable it; the file is created only
                     when a setting changes)
  log_path          (default ~/.config/mirrobot/decryptions.jsonl)

Requires `openssl` on PATH (Git for Windows, macOS, Linux all ship it).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import webbrowser
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BLOB_RE = re.compile(r"MRB1\.([A-Za-z0-9+/=]+)")
CTX_RE = re.compile(r"^context:\s*(.+)$", re.MULTILINE)

CONFIG_DIR = Path.home() / ".config" / "mirrobot"
CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_KEY = CONFIG_DIR / "share-link.pem"
DEFAULT_LOG = CONFIG_DIR / "decryptions.jsonl"

DEFAULTS = {
    "private_key_path": str(DEFAULT_KEY),
    "persist": False,
    "log_path": str(DEFAULT_LOG),
}


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    if os.environ.get("MIRROBOT_SHARE_KEY"):
        cfg["private_key_path"] = os.environ["MIRROBOT_SHARE_KEY"]
    return cfg


def save_config(cfg: dict) -> None:
    clean = {k: cfg[k] for k in DEFAULTS if k in cfg}
    # env override must not leak into the persisted file
    if os.environ.get("MIRROBOT_SHARE_KEY"):
        clean["private_key_path"] = DEFAULTS["private_key_path"]
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(clean, indent=2) + "\n", encoding="utf-8"
    )


# ----------------------------------------------------------------------------
# crypto (openssl subprocess)
# ----------------------------------------------------------------------------

def find_openssl() -> str:
    exe = shutil.which("openssl")
    if not exe:
        sys.exit("error: openssl not found on PATH (install Git for Windows / openssl)")
    return exe


def openssl_decrypt(openssl: str, key: Path, ciphertext: bytes) -> bytes:
    proc = subprocess.run(
        [openssl, "pkeyutl", "-decrypt", "-inkey", str(key),
         "-pkeyopt", "rsa_padding_mode:oaep", "-pkeyopt", "rsa_oaep_md:sha256"],
        input=ciphertext, capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"openssl decrypt failed: {err}")
    return proc.stdout


def decrypt_blob(blob_b64: str, key_path: Path) -> str:
    openssl = find_openssl()
    try:
        ciphertext = base64.b64decode(blob_b64, validate=True)
    except Exception as exc:
        raise RuntimeError(f"invalid base64 in blob: {exc}") from exc
    plaintext = openssl_decrypt(openssl, key_path, ciphertext)
    url = plaintext.decode("utf-8", errors="strict")
    if not re.fullmatch(r"https://[A-Za-z0-9._/-]+", url):
        raise RuntimeError(f"decrypted value is not a URL: {url!r}")
    return url


# ----------------------------------------------------------------------------
# parsing / persistence
# ----------------------------------------------------------------------------

def parse_input(text: str) -> tuple[str | None, str | None]:
    m = BLOB_RE.search(text)
    blob = m.group(1) if m else None
    c = CTX_RE.search(text)
    ctx = c.group(1).strip() if c else None
    return blob, ctx


def parse_ctx_fields(ctx: str | None) -> dict:
    fields = {}
    for part in re.split(r"\s*\|\s*", ctx or ""):
        if ":" in part:
            k, v = part.split(":", 1)
            fields[k.strip()] = v.strip()
        elif part.strip():
            fields.setdefault("head", part.strip()) if part.strip().startswith("head") else None
    return fields


def append_log(cfg: dict, url: str, ctx: str | None) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": url,
        "context": ctx or "",
    }
    log = Path(cfg["log_path"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# presentation
# ----------------------------------------------------------------------------

def show_result(url: str, ctx: str | None, opened: bool = False) -> None:
    print()
    print("\N{BOX DRAWINGS LIGHT HORIZONTAL}" * 52)
    print("  share link decrypted:")
    print()
    print(url)  # bare URL on its own line - clickable in most terminals
    print()
    if ctx:
        for part in ctx.split("|"):
            print(f"    {part.strip()}")
        print()
    if opened:
        print("  (opened in browser)")
    print("\N{BOX DRAWINGS LIGHT HORIZONTAL}" * 52)


def decrypt_flow(text: str, cfg: dict, open_browser: bool) -> str | None:
    blob, ctx = parse_input(text)
    if not blob:
        print("no MRB1.<base64> blob found in the input")
        return None
    key = Path(cfg["private_key_path"])
    if not key.exists():
        print(f"private key not found: {key}")
        print("set private_key_path in config or $MIRROBOT_SHARE_KEY")
        return None
    try:
        url = decrypt_blob(blob, key)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return None
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    show_result(url, ctx, open_browser)
    if cfg.get("persist"):
        append_log(cfg, url, ctx)
        print("  [logged]")
    return url


# ----------------------------------------------------------------------------
# gh / GitHub API helpers (for `setup` and the run browser)
# ----------------------------------------------------------------------------

def find_gh() -> str | None:
    return shutil.which("gh")


def gh_repo(gh: str) -> str | None:
    try:
        p = subprocess.run([gh, "repo", "view", "--json", "nameWithOwner",
                            "--jq", ".nameWithOwner"],
                           capture_output=True, text=True, timeout=30)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    except Exception:
        pass
    return None


def gh_token(gh: str) -> str | None:
    try:
        p = subprocess.run([gh, "auth", "token"], capture_output=True,
                           text=True, timeout=30)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    except Exception:
        pass
    return os.environ.get("GITHUB_TOKEN")


def derive_pubkey(openssl: str, priv: Path) -> str:
    p = subprocess.run([openssl, "pkey", "-in", str(priv), "-pubout"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"error: could not derive public key from {priv}: {p.stderr.strip()}")
    return p.stdout


def api_get(url: str, token: str | None) -> bytes:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "mirrobot-decryptor",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


# ----------------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------------

SETUP_STEPS = """
Manual steps (gh CLI not available or not in a repo):
  1. Generate the keypair (once, keep the private key private):
       openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 \\
         -out ~/.config/mirrobot/share-link.pem
  2. Export the public key:
       openssl pkey -in ~/.config/mirrobot/share-link.pem -pubout
  3. Add it as the repo secret SHARE_LINK_PUBKEY:
       Settings -> Secrets and variables -> Actions -> New repository secret
       (or:  gh secret set SHARE_LINK_PUBKEY --repo OWNER/REPO < pub.pem )
  4. Make sure the workflows pipe agent output through share-filter.sh
     (already wired in repos carrying the Mirrobot platform).
"""


def cmd_setup() -> int:
    cfg = load_config()
    openssl = find_openssl()
    key = Path(cfg["private_key_path"])

    # 1. keypair
    if key.exists():
        print(f"  private key already exists: {key}")
    else:
        key.parent.mkdir(parents=True, exist_ok=True)
        print(f"  generating new keypair at {key} ...")
        subprocess.run([openssl, "genpkey", "-algorithm", "RSA",
                        "-pkeyopt", "rsa_keygen_bits:2048",
                        "-out", str(key)], check=True)
        try:  # best-effort restriction on POSIX-ish systems
            key.chmod(0o600)
        except OSError:
            pass
        print("  generated.")
    pub_pem = derive_pubkey(openssl, key)

    # 2. repo detection
    gh = find_gh()
    repo = gh_repo(gh) if gh else None
    if not repo:
        print("\n" + SETUP_STEPS)
        print(f"  Your public key (also needed for step 3):\n")
        print(pub_pem)
        return 0

    # 3. secret (only if missing, per spec)
    lst = subprocess.run([gh, "secret", "list", "--repo", repo, "--json",
                          "name"], capture_output=True, text=True)
    have = False
    try:
        have = any(s.get("name") == "SHARE_LINK_PUBKEY"
                   for s in json.loads(lst.stdout or "[]"))
    except json.JSONDecodeError:
        pass
    if have:
        print(f"  secret SHARE_LINK_PUBKEY already set on {repo} (left untouched)")
    else:
        p = subprocess.run([gh, "secret", "set", "SHARE_LINK_PUBKEY",
                            "--repo", repo], input=pub_pem,
                           capture_output=True, text=True)
        if p.returncode == 0:
            print(f"  secret SHARE_LINK_PUBKEY pushed to {repo}")
        else:
            print(f"  could not set secret: {p.stderr.strip()}")
            print(SETUP_STEPS)
            return 1

    # 4. wiring check
    wired = any(Path(p).is_dir() for p in (".github/workflows",))
    if wired:
        hits = subprocess.run(
            ["grep", "-rl", "SHARE_LINK_PUBKEY", ".github/workflows"],
            capture_output=True, text=True)
        if hits.stdout.strip():
            print(f"  workflows wired: {hits.stdout.strip().splitlines()}")
        else:
            print("  WARNING: no workflow in this repo references "
                  "SHARE_LINK_PUBKEY - the share filter is not wired here.")
    print("  setup complete.")
    return 0


# ----------------------------------------------------------------------------
# run browser
# ----------------------------------------------------------------------------

def list_runs(gh: str, repo: str, limit: int = 15) -> list[dict]:
    token = gh_token(gh)
    data = json.loads(api_get(
        f"https://api.github.com/repos/{repo}/actions/runs"
        f"?per_page={limit}", token))
    return data.get("workflow_runs", [])


def runs_blobs(gh: str, repo: str, run_id: int) -> list[tuple[str, str]]:
    """Returns [(blob, context-or-'')] found in the run's logs."""
    token = gh_token(gh)
    raw = api_get(
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs",
        token)
    found: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            text = zf.read(name).decode("utf-8", errors="replace")
            for m in BLOB_RE.finditer(text):
                ctx_m = CTX_RE.search(text)
                found.append((m.group(1), ctx_m.group(1) if ctx_m else ""))
    # dedupe, keep order
    seen, out = set(), []
    for b, c in found:
        if b not in seen:
            seen.add(b)
            out.append((b, c))
    return out


def tui_browse(cfg: dict) -> None:
    gh = find_gh()
    if not gh:
        print("gh CLI not found - the run browser needs it "
              "(https://cli.github.com/)")
        return
    repo = gh_repo(gh)
    if not repo:
        print("not inside a GitHub repository (or gh is not authenticated)")
        return
    runs = list_runs(gh, repo)
    if not runs:
        print(f"no workflow runs found in {repo}")
        return
    print(f"\n  recent runs in {repo}:")
    for i, r in enumerate(runs, 1):
        print(f"  {i:2}) [{r.get('conclusion') or r.get('status', '?'):>8}] "
              f"{r.get('name', '?')[:38]:<38} "
              f"{r.get('created_at', '')[5:16].replace('T', ' ')}  "
              f"#{r.get('run_number', '?')}")
    sel = input("\n  run number to inspect (Enter = cancel): ").strip()
    if not sel.isdigit() or not (1 <= int(sel) <= len(runs)):
        return
    run = runs[int(sel) - 1]
    print(f"  fetching logs for "
          f"{run.get('name')} #{run.get('run_number')} ...")
    try:
        blobs = runs_blobs(gh, repo, run["id"])
    except Exception as exc:
        print(f"  could not fetch logs: {exc}")
        return
    if not blobs:
        print("  no encrypted share links found in this run's logs "
              "(pre-filter era, or agent step never ran)")
        return
    key = Path(cfg["private_key_path"])
    if not key.exists():
        print(f"  private key not found: {key}")
        return
    for blob, ctx in blobs:
        try:
            url = decrypt_blob(blob, key)
        except RuntimeError as exc:
            print(f"  [undecryptable: {exc}]")
            continue
        show_result(url, ctx or None)





# ----------------------------------------------------------------------------
# TUI
# ----------------------------------------------------------------------------

MENU = """
\N{BOX DRAWINGS LIGHT DOWN AND RIGHT}{line}\N{BOX DRAWINGS LIGHT DOWN AND LEFT}
  Mirrobot Share-Link Decryptor
\N{BOX DRAWINGS LIGHT UP AND RIGHT}{line}\N{BOX DRAWINGS LIGHT UP AND LEFT}
  1) decrypt a share link        4) browse workflow runs
  2) view decryption log         5) setup / keygen for this repo
  3) settings                    q) quit
"""


def read_paste() -> str:
    print("paste the encrypted block (finish with an empty line):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines)


def tui_decrypt(cfg: dict) -> None:
    text = read_paste()
    url = decrypt_flow(text, cfg, open_browser=False)
    if url:
        choice = input("[o] open in browser, [Enter] back: ").strip().lower()
        if choice == "o":
            webbrowser.open(url)


def tui_log(cfg: dict) -> None:
    log = Path(cfg["log_path"])
    if not log.exists():
        print("no decryption log yet"
              + ("" if cfg.get("persist") else " (persistence is off)"))
        return
    for raw in log.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(raw)
            print(f"  {e.get('ts','?')}  {e.get('url','?')}")
            if e.get("context"):
                print(f"      {e['context']}")
        except json.JSONDecodeError:
            print(f"  {raw}")


def tui_settings(cfg: dict) -> dict:
    while True:
        print()
        print(f"  private key : {cfg['private_key_path']}")
        print(f"  persist     : {'on' if cfg.get('persist') else 'off'}")
        print(f"  log file    : {cfg['log_path']}")
        print(f"  config file : {CONFIG_PATH}"
              + ("" if CONFIG_PATH.exists() else "  (not created yet)"))
        print()
        print("  1) toggle persist   2) set private key path")
        print("  3) set log path     b) back")
        c = input("> ").strip().lower()
        if c == "1":
            cfg["persist"] = not cfg.get("persist", False)
            save_config(cfg)
            print(f"  persist is now {'on' if cfg['persist'] else 'off'}"
                  " - config file written")
        elif c == "2":
            v = input("  private key path: ").strip()
            if v:
                cfg["private_key_path"] = v
                save_config(cfg)
                print("  saved")
        elif c == "3":
            v = input("  log path: ").strip()
            if v:
                cfg["log_path"] = v
                save_config(cfg)
                print("  saved")
        elif c in ("b", "q", ""):
            return cfg


def tui() -> None:
    cfg = load_config()
    while True:
        print(MENU.format(line="\N{BOX DRAWINGS LIGHT HORIZONTAL}" * 42))
        c = input("> ").strip().lower()
        if c == "1":
            tui_decrypt(cfg)
        elif c == "2":
            tui_log(cfg)
        elif c == "3":
            cfg = tui_settings(cfg)
        elif c == "4":
            tui_browse(cfg)
        elif c == "5":
            cmd_setup()
        elif c in ("q", ""):
            return


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test() -> bool:
    openssl = find_openssl()
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        priv, pub = tdir / "k.pem", tdir / "k.pub.pem"
        subprocess.run([openssl, "genpkey", "-algorithm", "RSA",
                        "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(priv)],
                       capture_output=True, check=True)
        subprocess.run([openssl, "pkey", "-in", str(priv), "-pubout",
                        "-out", str(pub)], capture_output=True, check=True)
        url = "https://opencd.ai/share/sElFtEsT01"
        ct = subprocess.run(
            [openssl, "pkeyutl", "-encrypt", "-pubin", "-inkey", str(pub),
             "-pkeyopt", "rsa_padding_mode:oaep",
             "-pkeyopt", "rsa_oaep_md:sha256"],
            input=url.encode(), capture_output=True, check=True).stdout
        blob = base64.b64encode(ct).decode()
        # decrypt via the module under test
        got = openssl_decrypt(openssl, priv, ct)
        assert got.decode() == url, "round-trip mismatch"
        # parse via the module under test
        pasted = f"~ [share link encrypted] MRB1.{blob}\ncontext: test/repo | PR #1"
        b, ctx = parse_input(pasted)
        assert b == blob and ctx == "test/repo | PR #1", "parse mismatch"
        assert BLOB_RE.search(f"noise MRB1.{blob} trailing") is not None
        print("self-test: OK (round-trip, blob parse, context parse)")
        return True


# ----------------------------------------------------------------------------
# entry
# ----------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return 0 if self_test() else 1
    args = [a for a in argv[1:] if a != "--open"]
    want_open = "--open" in argv
    if args and args[0] == "setup":
        return cmd_setup()
    if args and args[0] == "config":
        cfg = load_config()
        if len(args) >= 2 and args[1] == "set" and len(args) == 4:
            key, value = args[2], args[3]
            if key not in DEFAULTS:
                print(f"unknown key: {key} (valid: {', '.join(DEFAULTS)})")
                return 1
            if key == "persist":
                value = value.lower() in ("1", "true", "on", "yes")
            cfg[key] = value
            save_config(cfg)
            print(f"saved {key} = {value} -> {CONFIG_PATH}")
            return 0
        print(json.dumps(cfg, indent=2))
        return 0
    if args:
        return 0 if decrypt_flow(" ".join(args), load_config(), want_open) else 1
    tui()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

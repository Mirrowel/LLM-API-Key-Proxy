#!/usr/bin/env python3
"""Passive session export: family exports + offline-viewable snapshot,
sealed into an MRB2 Actions artifact. Always-on counterpart to optional
live sharing; never fails the run - a visible warning is the failure
signal.

Flow:
  1. Resolve the newest ROOT session + its subagent family in the
     opencode db; `opencode export` every member (native format).
  2. Build a self-contained snapshot HTML:
     - online: a live share exists (JSON lines in the OpenShare urlFile)
       -> osh-snapshot.mjs pulls the snapshot from the worker;
     - offline fallback: osh-snapshot-local.mjs builds it from the raw
       exports with the viewer-dist release asset (no worker involved);
     - degrade: raw exports only, snapshot null, note recorded.
     Tools + dist come from the OpenShare repo at the ref the bundle URL
     variable points at (absent = master). Never print secrets.
  3. Seal {meta, exports, snapshot} with MRB2 (gzip + AES + RSA-OAEP)
     -> session-<runid>.enc artifact + encrypted meta blob for the run
     summary (share-link.enc + share-link.ctx + session-meta.txt).

Env: SESSION_ARTIFACT_PUBKEY (repo secret), OPENSHARE_URL_FILE (plugin urlFile;
JSON lines when the config sets urlFileFormat json - bare URLs also
accepted), OPENSHARE_BUNDLE_URL (repo variable, ref source),
OPENSHARE_CONFIG (the config secret, parsed in-memory for the share
read-auth pair; used for snapshot login, never logged).

Requires: python3 (stdlib only) + openssl + opencode; node for snapshots.
The artifact is decryptable by decrypt_share_link.py (MRB2).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

WIRE_PREFIX = "MRB2"
OPENSHARE_REPO = "Mirrowel/OpenShare"
DEFAULT_BUNDLE_URL = f"https://raw.githubusercontent.com/{OPENSHARE_REPO}/master/share-plugin/ci/openshare.js"
TOOLS_ONLINE = ["tools/osh-snapshot.mjs", "tools/snapshot-builder.mjs"]
TOOLS_OFFLINE = ["tools/osh-snapshot-local.mjs", "tools/snapshot-builder.mjs", "tools/osh-session.mjs"]
UA = {"user-agent": "mirrobot-agent-passive/1.0"}


# ----------------------------------------------------------------------------
# MRB2 hybrid encryption (openssl subprocess; no third-party deps)
# ----------------------------------------------------------------------------

def openssl() -> str:
    exe = shutil.which("openssl")
    if not exe:
        sys.exit("error: openssl not found on PATH")
    return exe


def mrb2_encrypt(payload: bytes, pub_pem: str) -> str:
    """AES-256-CBC(payload) with RSA-OAEP(SHA-256)-wrapped key.

    Wire format: MRB2.<b64(rsakey)>.<b64(iv)>.<b64(ct)>
    """
    key = secrets.token_bytes(32)
    iv = secrets.token_bytes(16)
    enc = subprocess.run(
        [openssl(), "enc", "-aes-256-cbc", "-K", key.hex(), "-iv", iv.hex()],
        input=payload, capture_output=True,
    )
    if enc.returncode != 0:
        sys.exit(f"error: aes encrypt failed: {enc.stderr.decode(errors='replace').strip()}")
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pem") as tf:
        tf.write(pub_pem.encode() if isinstance(pub_pem, str) else pub_pem)
        pubpath = tf.name
    try:
        wrap = subprocess.run(
            [openssl(), "pkeyutl", "-encrypt", "-pubin", "-inkey", pubpath,
             "-pkeyopt", "rsa_padding_mode:oaep", "-pkeyopt", "rsa_oaep_md:sha256"],
            input=key, capture_output=True,
        )
    finally:
        os.unlink(pubpath)
    if wrap.returncode != 0:
        sys.exit(f"error: rsa wrap failed: {wrap.stderr.decode(errors='replace').strip()}")
    b = base64.b64encode
    return f"{WIRE_PREFIX}.{b(wrap.stdout).decode()}.{b(iv).decode()}.{b(enc.stdout).decode()}"


# ----------------------------------------------------------------------------
# session discovery + export
# ----------------------------------------------------------------------------

def find_db(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"error: db not found: {p}")
        return p
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    p = Path(base) / "opencode" / "opencode.db"
    if not p.exists():
        sys.exit(f"error: opencode db not found at {p} (pass --db)")
    return p


def newest_session_id(db: Path) -> str:
    """Newest ROOT session (parentless). On the single-run runner db the
    newest session overall may be a subagent child that finished last —
    the family walk from a child never reaches the main session."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT id FROM session WHERE parent_id IS NULL "
            "ORDER BY time_updated DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    if not row or not row[0]:
        sys.exit("error: no root sessions found in the opencode db")
    return str(row[0])


def opencode_export(session_id: str, exe: str) -> dict:
    cmd = [exe] if exe else ["opencode"]
    cmd += ["export", session_id]
    # Capture BYTES + decode UTF-8 explicitly: text=True uses the locale
    # codec on Windows (cp1251 etc.) and dies on non-ASCII session content,
    # leaving stdout as None mid-read (live-caught).
    p = subprocess.run(cmd, capture_output=True, timeout=300)
    if p.returncode != 0:
        sys.exit(f"error: opencode export failed: "
                 f"{p.stderr.decode('utf-8', errors='replace').strip()[:400]}")
    raw = p.stdout.decode("utf-8", errors="replace")
    i = raw.find("{")
    if i < 0:
        sys.exit("error: export produced no JSON")
    try:
        return json.loads(raw[i:])
    except json.JSONDecodeError as exc:
        sys.exit(f"error: export JSON parse failed: {exc}")


def session_family(db: Path, main_id: str) -> list[str]:
    """The main session plus every session in the db whose root ancestor is
    the main session (subagent children, recursively)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT id, parent_id FROM session").fetchall()
    except sqlite3.OperationalError:
        return [main_id]
    finally:
        con.close()
    parent = {sid: (pid or None) for sid, pid in rows}

    def root_of(sid: str) -> str:
        seen = set()
        while parent.get(sid) and sid not in seen:
            seen.add(sid)
            sid = parent[sid]
        return sid

    return [sid for sid in parent if sid == main_id or root_of(sid) == main_id]


# ----------------------------------------------------------------------------
# urlFile (JSON lines preferred; bare URL lines accepted)
# ----------------------------------------------------------------------------

def parse_url_file(path: str | None) -> list[dict]:
    """One entry per share: {id, url, status, gated} when JSON format,
    {url} for legacy bare lines. Malformed lines are skipped loudly."""
    if not path or not Path(path).exists():
        return []
    out: list[dict] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj.get("url"), str) and obj["url"].startswith("http"):
                    out.append({
                        "id": str(obj.get("id") or ""),
                        "url": obj["url"],
                        "status": str(obj.get("status") or ""),
                        "gated": bool(obj.get("gated", False)),
                    })
                    continue
            except json.JSONDecodeError:
                pass
            print(f"note: urlFile line {n} is malformed JSON - skipped")
        elif line.startswith("http"):
            out.append({"id": "", "url": line, "status": "", "gated": False})
    return out


# ----------------------------------------------------------------------------
# OpenShare tool + dist fetching (from the repo at the bundle ref)
# ----------------------------------------------------------------------------

def bundle_ref() -> str:
    """The ref the bundle URL points at (OPENSHARE_BUNDLE_URL variable;
    absent = master, the released line)."""
    url = os.environ.get("OPENSHARE_BUNDLE_URL", "").strip() or DEFAULT_BUNDLE_URL
    m = re.search(rf"/{OPENSHARE_REPO}/([^/]+)/share-plugin/", url)
    return m.group(1) if m else "master"


def fetch_raw(ref: str, repo_path: str, dest: Path) -> None:
    url = f"https://raw.githubusercontent.com/{OPENSHARE_REPO}/{ref}/{repo_path}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        dest.write_bytes(r.read())
    if not dest.stat().st_size:
        raise RuntimeError(f"empty fetch: {repo_path}")


def fetch_tools(ref: str, names: list[str], workdir: Path) -> None:
    for repo_path in names:
        dest = workdir / Path(repo_path).name
        fetch_raw(ref, repo_path, dest)
    # the CLIs resolve their sibling imports relative to their own file
    (workdir / "node_modules").mkdir(exist_ok=True)


def resolve_dist_zip_url(ref: str) -> str | None:
    """viewer-dist release asset for the ref's channel: exact tag -> that
    release; dev/other branch -> newest prerelease carrying the asset;
    master -> newest stable. Refs with no release (docs-only) -> None."""
    api = f"https://api.github.com/repos/{OPENSHARE_REPO}/releases?per_page=40"
    with urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=30) as r:
        releases = json.load(r)

    def asset(rel: dict) -> str | None:
        for a in rel.get("assets", []):
            if str(a.get("name", "")).startswith("viewer-dist-"):
                return a.get("browser_download_url")
        return None

    if ref.startswith("v"):
        for rel in releases:
            if rel.get("tag_name") == ref:
                return asset(rel)
        return None
    want_pre = ref not in ("master", "main")
    for rel in releases:
        if bool(rel.get("prerelease")) == want_pre and asset(rel):
            return asset(rel)
    for rel in releases:  # channel fallback: any release with the asset
        if asset(rel):
            return asset(rel)
    return None


def fetch_dist(ref: str, workdir: Path) -> Path | None:
    url = resolve_dist_zip_url(ref)
    if not url:
        print(f"note: no viewer-dist release asset for ref '{ref}' - offline snapshot unavailable")
        return None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            data = r.read()
        dist = workdir / "viewer-dist"
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            # flat layout (root index.html) extracts as-is; a single
            # wrapper directory is stripped
            if any(n == "index.html" for n in names):
                strip = 0
            else:
                top = {n.split("/")[0] for n in names if "/" in n}
                strip = 1 if len(top) == 1 else 0
            for n in names:
                if n.endswith("/"):
                    continue
                parts = n.split("/")[strip:]
                if not parts:
                    continue
                dest = dist.joinpath(*parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(n))
        if not (dist / "index.html").exists():
            raise RuntimeError("dist zip missing index.html")
        return dist
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the run
        print(f"note: viewer-dist fetch/extract failed ({exc}) - offline snapshot unavailable")
        return None


def auth_pair() -> tuple[str, str]:
    """Read-auth pair from the config secret (in-memory only, never printed)."""
    raw = os.environ.get("OPENSHARE_CONFIG", "").strip()
    if not raw:
        return "", ""
    try:
        cfg = json.loads(raw)
        return str(cfg.get("authLogin") or ""), str(cfg.get("authPassword") or "")
    except json.JSONDecodeError:
        return "", ""


def run_node(args: list[str], workdir: Path) -> subprocess.CompletedProcess:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node not found on PATH")
    return subprocess.run([node, *args], cwd=workdir, capture_output=True, timeout=420)


def build_snapshot_online(share_url: str, login: str, password: str,
                          workdir: Path) -> tuple[Path, str] | None:
    """Snapshot pulled from the worker by osh-snapshot.mjs (the worker is
    the asset source; no dist needed)."""
    try:
        fetch_tools(bundle_ref(), TOOLS_ONLINE, workdir)
        cmd = [str(workdir / "osh-snapshot.mjs"), share_url, "-o", str(workdir / "snapshot.html"), "--json"]
        if login and password:
            # creds ride argv; the config leaves are add-masked at boot so
            # they cannot appear in logs, and the tool never echoes them
            cmd += ["--login", f"{login}:{password}"]
        p = run_node(cmd, workdir)
        if p.returncode != 0:
            raise RuntimeError(f"exit {p.returncode}: {p.stderr.decode('utf-8', 'replace').strip()[:200]}")
        out = workdir / "snapshot.html"
        if not out.exists():
            raise RuntimeError("tool wrote no snapshot")
        detail = p.stdout.decode("utf-8", "replace").strip().splitlines()[-1] if p.stdout.strip() else ""
        return out, detail
    except Exception as exc:  # noqa: BLE001 - fall back to offline build
        print(f"note: online snapshot failed ({exc}) - trying offline build")
        return None


def build_snapshot_offline(exports: dict[str, dict], workdir: Path) -> Path | None:
    """Snapshot built locally from the raw exports: no worker, no share -
    tools + viewer dist fetched from the OpenShare repo/release."""
    try:
        ref = bundle_ref()
        dist = fetch_dist(ref, workdir)
        if not dist:
            return None
        fetch_tools(ref, TOOLS_OFFLINE, workdir)
        files = []
        for sid, export in exports.items():
            f = workdir / f"export-{sid}.json"
            f.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
            files.append(str(f))
        cmd = [str(workdir / "osh-snapshot-local.mjs"), *files,
               "--viewer", str(dist), "-o", str(workdir / "snapshot.html")]
        p = run_node(cmd, workdir)
        if p.returncode != 0:
            raise RuntimeError(f"exit {p.returncode}: {p.stderr.decode('utf-8', 'replace').strip()[:200]}")
        out = workdir / "snapshot.html"
        if not out.exists():
            raise RuntimeError("tool wrote no snapshot")
        return out
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the run
        print(f"note: offline snapshot failed ({exc}) - artifact carries raw exports only")
        return None


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def _short(ident: str) -> str:
    """Log-safe short form: the share id IS the share URL slug - the raw
    value never appears in plaintext output, only inside the encrypted
    blob (and as a 6-char hint for eyeballing)."""
    return ident[:6] + "…" if len(ident) > 6 else ident


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp")
    ap.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "unknown/repo"))
    ap.add_argument("--db", default=None)
    ap.add_argument("--url-file", default=os.environ.get("OPENSHARE_URL_FILE"),
                    help="OpenShare urlFile (JSON lines preferred); first share feeds the summary + online snapshot")
    ap.add_argument("--opencode", default=None)
    ap.add_argument("--session-id", default=None,
                    help="export this session (default: newest ROOT in the db — correct for the "
                         "single-run runner db; use explicitly on dev machines)")
    ap.add_argument("--plain", action="store_true",
                    help="debug: also write the unencrypted payload JSON")
    args = ap.parse_args(argv[1:])

    pub = os.environ.get("SESSION_ARTIFACT_PUBKEY", "").strip()
    if not pub:
        sys.exit("error: SESSION_ARTIFACT_PUBKEY not set (repo secret)")

    warnings: list[str] = []
    shares = parse_url_file(args.url_file)
    live = shares[0] if shares else None
    login, password = auth_pair()
    if live is not None:
        # the share id/url are capability locators: mask them job-wide and
        # keep plaintext prints to the 6-char hint (step summaries are NOT
        # covered by add-mask - live-caught)
        for secretish in (live["id"], live["url"]):
            if secretish and len(secretish) >= 8:
                print(f"::add-mask::{secretish}")
        if live["gated"]:
            print(f"share gate: login-gated (id {_short(live['id']) or 'n/a'}, status {live['status'] or 'n/a'})")
        elif login:
            msg = ("OpenShare share is NOT login-gated although read auth is configured "
                   "- check the worker's auth handling (gated=false)")
            warnings.append(msg)
            print(f"::warning::{msg}")
        else:
            print("note: share is ungated (no read auth configured in the config secret)")

    db = find_db(args.db)
    sid = args.session_id or newest_session_id(db)
    family = session_family(db, sid)
    if not family:
        sys.exit(f"error: session family resolved empty for {sid} - refusing to ship an empty artifact")
    print(f"session: {sid} (newest ROOT in {db}); family: {len(family)} session(s)")

    exports: dict[str, dict] = {}
    for fsid in family:
        exports[fsid] = opencode_export(fsid, args.opencode)
    n_msgs = sum(1 for e in exports.values() for m in (e.get("messages") or []) if isinstance(m, dict))
    print(f"exported: {n_msgs} messages across {len(family)} session(s)")

    workdir = Path(tempfile.mkdtemp(prefix="osh-passive-"))
    snapshot_html: str | None = None
    snapshot_src = None
    snapshot_note = None
    if live is not None:
        snap = build_snapshot_online(live["url"], login, password, workdir)
        if snap:
            snapshot_html = snap[0].read_text(encoding="utf-8", errors="replace")
            snapshot_src = "online"
    if snapshot_html is None:
        snap2 = build_snapshot_offline(exports, workdir)
        if snap2:
            snapshot_html = snap2.read_text(encoding="utf-8", errors="replace")
            snapshot_src = "offline"
    if snapshot_src:
        print(f"snapshot: {snapshot_src} ({len(snapshot_html)} bytes)")
    elif not snapshot_note:
        snapshot_note = "snapshot unavailable (see notes above); raw exports intact"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": "osh-passive",
        "v": 2,
        "repo": args.repo,
        "run_id": str(args.run_id),
        "session_id": sid,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **({"live": live} if live else {"live": None}),
        "snapshot": snapshot_src,
        **({"snapshot_note": snapshot_note} if snapshot_note else {}),
        "exports": exports,
        "snapshot_html": snapshot_html,
    }
    blob = mrb2_encrypt(gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8")), pub)
    art = out / f"session-{args.run_id}.enc"
    art.write_text(blob + "\n", encoding="utf-8")
    print(f"artifact: {art} ({art.stat().st_size} bytes encrypted)")

    meta_payload = json.dumps({
        "v": 2,
        "kind": "meta",
        "repo": args.repo,
        "run_id": str(args.run_id),
        "session_id": sid,
        "artifact": art.name,
        **({"url": live["url"], "share_id": live["id"], "share_status": live["status"],
            "share_gated": live["gated"], "snapshot": snapshot_src} if live else {}),
    }, ensure_ascii=False).encode("utf-8")
    meta_blob = mrb2_encrypt(meta_payload, pub)

    (out / "share-link.enc").write_text(meta_blob + "\n", encoding="utf-8")
    live_desc = (f"live={_short(live['id']) or live['url']} status={live['status'] or '?'} "
                 f"gated={'yes' if live['gated'] else 'no'}") if live else "live=none"
    (out / "share-link.ctx").write_text(
        f"{args.repo} run {args.run_id}: {live_desc}, snapshot={snapshot_src or 'none'}",
        encoding="utf-8",
    )
    meta = out / "session-meta.txt"
    meta_lines = [
        f"session artifact (MRB2, decrypt with decrypt_share_link.py):",
        f"MRB2 encrypted: {meta_blob}",
        f"artifact: {art.name}",
        f"context: {live_desc}, snapshot={snapshot_src or 'none'}",
    ]
    meta_lines += [f"WARNING: {w}" for w in warnings]
    meta.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    print(f"meta: {meta}")

    if args.plain:
        plain = out / f"session-{args.run_id}.payload.json"
        plain.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"debug plain: {plain}")

    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

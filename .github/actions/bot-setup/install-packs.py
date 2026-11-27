#!/usr/bin/env python3
"""Content pack installer for the Mirrobot agent.

A content pack is a directory with an allowlisted layout; everything else in
the pack is ignored:

    .agents/skills/<name>/SKILL.md   skills (opencode's universal skills dir)
    plugins/<name>.js                opencode plugins, auto-discovered
    agents/<name>.md                 opencode agent definitions
    commands/<name>.md               slash commands
    extensions/<name>/install.sh     boot-time installer, run in the background
    opencode.json                    optional config fragment (never `permission`, at any depth)

Skills listed in `.agents/.skill-lock.json` (written by `npx skills`) that are
not present as directories are fetched from their upstream repository at the
recorded commit and verified against the recorded tree SHA. A committed
directory always wins.

Sources come from the CONTENT_PACKS variable (public sources) and the
CONTENT_PACKS_PRIVATE secret (private archive links; its entries follow the
variable's), each a flat comma list of `owner/repo[@ref]` (GitHub repository,
shallow-fetched) or `https://...` (zip or tar.gz archive). Later entries win.
The private list rides a secret because the runner echoes variables into the
log in clear.

Subcommands, in the order bot-setup runs them:
    fetch          download every source into $RUNNER_TEMP/mirrobot-packs
    start-skills   launch the upstream skill fetch in the background
    merge-config   merge `opencode.json` fragments into the generated config
    install        copy allowlisted content into the opencode homes
    start          launch extensions in the background
    wait           wait for the background jobs, relay their results, report
"""
import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

ROOT_NAMES = {".agents", "plugins", "agents", "commands", "extensions", "instructions", "opencode.json"}
KIND_DIRS = {"plugins", "agents", "commands"}
LOCK_FIELDS = ("sourceUrl", "commit", "skillPath", "skillFolderHash")
REPO_SPEC = re.compile(r"^(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?(?:@(?P<ref>.+))?$")
MAX_FILES = 20000
MAX_BYTES = 256 * 1024 * 1024
SECRET_ENV = re.compile(r"TOKEN|SECRET|PASSWORD|KEY|CREDENTIAL")

STATE = Path(os.environ.get("RUNNER_TEMP") or "/tmp") / "mirrobot-packs"
JOBS = STATE / "jobs"
SKILLS_HOME = Path.home() / ".agents" / "skills"
CONFIG_HOME = Path.home() / ".config" / "opencode"


class FetchError(Exception):
    """A pack could not be fetched or is not usable; the message never carries a URL."""


def say(message, command=None):
    """Print a log line, or a GitHub workflow command (`warning`, `error`) when `command` is given; a command is always one line."""
    if command:
        print(f"::{command}::{str(message).replace(chr(10), ' ').replace(chr(13), ' ')[:4000]}", flush=True)
    else:
        print(message, flush=True)


def remove_tree(path):
    """Delete a directory tree even when it holds read-only files, which git object stores do on Windows."""
    if not Path(path).exists():
        return

    def writable(func, target, _exc):
        """Clear the read-only bit, then retry the operation that failed."""
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, **({"onexc": writable} if sys.version_info >= (3, 12) else {"onerror": writable}))


def git_reason(stderr):
    """Map git's stderr to a short reason that carries no URL, token or repository name."""
    low = (stderr or "").lower()
    if any(s in low for s in ("not found", "authentication failed", "403", "404", "could not read username", "terminal prompts disabled")):
        return "not found or not accessible"
    if any(s in low for s in ("couldn't find remote ref", "unadvertised", "ambiguous argument")):
        return "ref or commit not found"
    if any(s in low for s in ("timed out", "could not resolve host", "network", "rpc failed", "early eof", "ssl")):
        return "network problem"
    return "git fetch failed"


def git(args, cwd, env=None):
    """Run a git command in `cwd` and return its stdout; raise FetchError with a sanitized reason on failure."""
    proc = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FetchError(f"{args[0]} failed: {git_reason(proc.stderr)}")
    return proc.stdout.strip()


def git_env(token):
    """Return a git environment that authenticates through a one-shot askpass helper, so the token never reaches a command line or config."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    if token:
        helper = STATE / "askpass.sh"
        helper.write_text('#!/bin/sh\nprintf "%s\\n" "$PACK_GIT_TOKEN"\n')
        helper.chmod(0o700)
        env["GIT_ASKPASS"] = str(helper)
        env["PACK_GIT_TOKEN"] = token
    return env


def safe_name(name):
    """Normalize an archive entry name to a relative POSIX path; refuse absolute paths and traversal."""
    name = name.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    if name in ("", "."):
        return None
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or "\x00" in name:
        raise FetchError("archive holds an absolute path")
    if ".." in name.split("/"):
        raise FetchError("archive holds a path traversal")
    return "/".join(part for part in name.split("/") if part not in ("", "."))


class Budget:
    """Running total of extracted files and bytes, capped so an archive cannot fill the disk."""

    def __init__(self):
        """Start both counters at zero."""
        self.files = 0
        self.total = 0

    def spend(self, size):
        """Add one file of `size` bytes; raise FetchError past either cap."""
        self.files += 1
        self.total += size
        if self.files > MAX_FILES or self.total > MAX_BYTES:
            raise FetchError("archive is larger than the extraction limit")


def write_member(source, target, mode):
    """Copy one extracted member to `target`, creating parents and applying `mode` when it is non-zero."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as out:
        shutil.copyfileobj(source, out)
    if mode:
        target.chmod(mode)


def extract_zip(archive, dest, budget):
    """Extract regular files from a zip; refuse link entries and names `safe_name` rejects."""
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = safe_name(info.filename)
            if name is None or info.is_dir():
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise FetchError("archive holds a link")
            budget.spend(info.file_size)
            with zf.open(info) as src:
                write_member(src, dest / name, (info.external_attr >> 16) & 0o777)


def extract_tar(archive, dest, budget):
    """Extract regular files from a tar (any compression); refuse every link, device or special member."""
    with tarfile.open(archive, "r:*") as tf:
        for member in tf:
            if member.isdir():
                continue
            if not member.isfile():
                raise FetchError("archive holds a link or special file")
            name = safe_name(member.name)
            if name is None:
                continue
            budget.spend(member.size)
            with tf.extractfile(member) as src:
                write_member(src, dest / name, member.mode & 0o777)


def extract_archive(archive, dest):
    """Extract a zip or tar archive into `dest`; return the pack root, unwrapping a single top-level folder."""
    dest.mkdir(parents=True)
    with open(archive, "rb") as fh:
        magic = fh.read(4)
    budget = Budget()
    if magic.startswith(b"PK"):
        extract_zip(archive, dest, budget)
    elif magic[:2] == b"\x1f\x8b":
        extract_tar(archive, dest, budget)
    else:
        raise FetchError("the file is not a zip or tar.gz archive")
    top = [p for p in dest.iterdir() if p.name != "__MACOSX"]
    if len(top) == 1 and top[0].is_dir() and top[0].name not in ROOT_NAMES:
        return top[0]
    return dest


def fetch_repo(repo, ref, token, work):
    """Shallow-fetch `repo` at `ref` (its default branch when None) into `work`; return the commit SHA."""
    work.mkdir(parents=True)
    env = git_env(token)
    git(["init", "-q"], work, env)
    git(["remote", "add", "origin", f"https://github.com/{repo}.git"], work, env)
    git(["fetch", "-q", "--depth", "1", "origin", *([ref] if ref else [])], work, env)
    sha = git(["rev-parse", "FETCH_HEAD"], work, env)
    git(["checkout", "-q", "--force", "FETCH_HEAD"], work, env)
    return sha


def curl_config(pairs):
    """Render lines for curl's `--config`; a quote or line break inside a value could add a directive, so it is refused."""
    lines = []
    for key, value in pairs:
        if any(ch in value for ch in '"\r\n'):
            raise FetchError(f"the {key} for a content pack contains a quote or a line break")
        lines.append(f'{key} = "{value}"')
    return "\n".join(lines) + "\n"


def fetch_url(url, token, work):
    """Download the archive at `url` and extract it safely into `work`; return the archive's SHA-256 prefix and the pack root."""
    work.mkdir(parents=True)
    archive = work / "archive"
    # A host with hotlink protection wants a Referer from its own origin
    # (qu.ax answers 302 without one); everywhere else the header is harmless.
    parts = urlsplit(url)
    directives = [("referer", f"{parts.scheme}://{parts.netloc}/")]
    if token:
        directives.append(("header", f"Authorization: Bearer {token}"))
    cmd = ["curl", "-fsSL", "--retry", "2", "--max-time", "180", "--max-filesize", str(MAX_BYTES), "--config", "-",
           "-o", str(archive), url]
    proc = subprocess.run(cmd, input=curl_config(directives), capture_output=True, text=True)
    if proc.returncode == 63:
        raise FetchError("the file at this link is larger than the extraction cap")
    if proc.returncode != 0:
        raise FetchError(f"download failed (curl exit {proc.returncode})")
    if archive.stat().st_size == 0:
        raise FetchError("the file at this link is empty")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    root = extract_archive(archive, work / "root")
    archive.unlink()
    return digest.hexdigest()[:12], root


def parse_sources(raw):
    """Split the CONTENT_PACKS value into (kind, value, ref) entries, warning about and skipping malformed ones."""
    sources = []
    for item in (part.strip() for part in (raw or "").split(",")):
        if not item:
            continue
        if re.match(r"^https?://", item):
            sources.append(("url", item, None))
            continue
        match = REPO_SPEC.match(item)
        if match:
            sources.append(("repo", match.group("repo"), match.group("ref")))
        else:
            say("an entry is neither owner/repo[@ref] nor an http(s) link; skipped", "warning")
    return sources


def count_files(root):
    """Count the files under `root`, ignoring git metadata."""
    return sum(1 for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts)


def load_manifest():
    """Return the list of successfully fetched packs, or an empty list when nothing was fetched."""
    path = STATE / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else []


def save_json(name, data):
    """Write `data` to the state directory under `name`, creating the directory first."""
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / name).write_text(json.dumps(data))


EXCLUDE_NAME = "content-packs.exclude.txt"


def parse_exclude_file(path):
    """Parse one exclusion list: skill directory names, # comments ignored, invalid names warned and skipped."""
    names = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name = line.split("#", 1)[0].strip()
        if not name:
            continue
        if name in (".", "..") or "/" in name or "\\" in name:
            say(f"{EXCLUDE_NAME}: ignoring invalid skill name {name!r} ({path})", "warning")
            continue
        names.add(name)
    return names


def pack_exclusions(manifest):
    """Union of the root exclusion lists of every fetched pack.

    Returns (found, names): a pack that carries an empty list is authoritative
    (nothing excluded) and stops the consumer fallback.
    """
    found = False
    names = set()
    for pack in manifest:
        path = Path(pack["root"]) / EXCLUDE_NAME
        if path.is_file():
            found = True
            names |= parse_exclude_file(path)
    return found, names


def consumer_exclusions():
    """Fallback exclusion list from the consumer repository (base-branch content under bot-setup)."""
    workspace = Path(os.environ.get("GITHUB_WORKSPACE") or os.getcwd())
    for path in (workspace / ".github" / EXCLUDE_NAME, Path(".github") / EXCLUDE_NAME):
        if path.is_file():
            return parse_exclude_file(path)
    return set()


def excluded_skills(manifest=None):
    """The exclusion set for this installation.

    Pack lists are primary and combine across packs; only when no pack carries
    one does the consumer repository's .github file apply. The fetch step saves
    the resolved set so install and the background skill job agree in separate
    processes.
    """
    saved = STATE / "exclude.json"
    if saved.is_file():
        try:
            return set(json.loads(saved.read_text()))
        except (ValueError, TypeError):
            say("exclude.json is not valid; recomputing the exclusion set", "warning")
    if manifest is not None:
        found, names = pack_exclusions(manifest)
        if found:
            return names
    return consumer_exclusions()


def skill_plan(manifest):
    """Map every skill name to its provider: a committed pack directory, or a lock entry to fetch upstream. Later packs win."""
    plan = {}
    excluded = excluded_skills(manifest)
    for index, pack in enumerate(manifest):
        root = Path(pack["root"])
        skills_dir = root / ".agents" / "skills"
        committed = {p.name: p for p in skills_dir.iterdir() if p.is_dir()} if skills_dir.is_dir() else {}
        lock_path = root / ".agents" / ".skill-lock.json"
        try:
            lock = json.loads(lock_path.read_text()).get("skills", {}) if lock_path.is_file() else {}
        except ValueError:
            say(f"pack {index + 1}: .agents/.skill-lock.json is not valid JSON; its skills will not be fetched", "warning")
            lock = {}
        for name in sorted(set(committed) | set(lock)):
            if name in excluded:
                continue
            if name in committed:
                plan[name] = {"from": "pack", "dir": str(committed[name])}
            else:
                plan[name] = {"from": "upstream", **{field: lock[name].get(field) for field in LOCK_FIELDS}}
    return plan


def cmd_fetch(args):
    """Fetch every configured pack into the state directory and record what succeeded."""
    token = os.environ.get("PACK_TOKEN", "")
    for value in (args.packs, token):
        if value and len(value) >= 8:
            say(value, "add-mask")
    sources = parse_sources(args.packs)
    if not sources:
        say("content packs: none configured")
    save_json("manifest.json", [])
    manifest = []
    for index, (kind, value, ref) in enumerate(sources, start=1):
        work = STATE / "src" / str(index)
        remove_tree(work)
        try:
            if kind == "repo":
                detail = fetch_repo(value, ref, token, work)
                root = work
            else:
                detail, root = fetch_url(value, token, work)
        except FetchError as error:
            say(f"content pack {index} ({kind}) could not be fetched: {error}", "warning")
            continue
        entry = {"index": index, "kind": kind, "ref": detail, "root": str(root), "files": count_files(root)}
        manifest.append(entry)
        say(f"content pack {index} of {len(sources)}: {kind}, {'commit' if kind == 'repo' else 'sha256'} {detail[:12]}, {entry['files']} files")
    save_json("manifest.json", manifest)
    found, names = pack_exclusions(manifest)
    excluded = names if found else consumer_exclusions()
    save_json("exclude.json", sorted(excluded))
    if excluded:
        source = "from the packs" if found else "from the consumer repository"
        say(f"skills excluded ({source}): {', '.join(sorted(excluded))}")
    save_json("skills.json", skill_plan(manifest))


def extension_env():
    """Return a child environment with no GitHub, Actions or credential-bearing variables in it."""
    keep = {}
    for name, value in os.environ.items():
        if name in ("RUNNER_TEMP", "RUNNER_OS") or not name.startswith(("GITHUB_", "ACTIONS_", "RUNNER_")):
            if not SECRET_ENV.search(name):
                keep[name] = value
    return keep


def launch(name, label, argv, cwd):
    """Start `argv` detached with its output in a log file; its exit code lands in a status file when it ends."""
    JOBS.mkdir(parents=True, exist_ok=True)
    log = JOBS / f"{name}.log"
    status = JOBS / f"{name}.status"
    for path in (log, status):
        path.unlink(missing_ok=True)
    script = 'log="$1"; status="$2"; shift 2; "$@" >"$log" 2>&1; echo $? >"$status"'
    child = subprocess.Popen([shutil.which("bash") or "/bin/bash", "-c", script, "_", str(log), str(status), *argv], cwd=cwd,
                             env=extension_env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    return {"name": name, "label": label, "pid": child.pid, "log": str(log), "status": str(status), "started": time.time()}


def cmd_start_skills(args):
    """Launch the upstream skill fetch in the background; the extensions follow in `start`."""
    jobs = []
    plan = json.loads((STATE / "skills.json").read_text()) if (STATE / "skills.json").is_file() else {}
    if any(item["from"] == "upstream" for item in plan.values()):
        jobs.append(launch("upstream-skills", "upstream skills", [sys.executable, str(Path(__file__).resolve()), "fetch-skills"], os.getcwd()))
    save_json("jobs.json", jobs)


def cmd_start(args):
    """Launch every pack extension in the background and add it to the job list the wait step reads."""
    jobs = json.loads((STATE / "jobs.json").read_text()) if (STATE / "jobs.json").is_file() else []
    for pack in load_manifest():
        extensions = Path(pack["root"]) / "extensions"
        for script in sorted(extensions.glob("*/install.sh")):
            name = f"pack{pack['index']}-{script.parent.name}"
            jobs.append(launch(name, f"{script.parent.name} (pack {pack['index']})", ["bash", str(script)], script.parent))
    save_json("jobs.json", jobs)
    say(f"content: started {len(jobs)} background job(s)")


def cmd_install(args):
    """Copy allowlisted content from every fetched pack into the opencode homes, in pack order."""
    manifest = load_manifest()
    kinds = {kind: 0 for kind in sorted(KIND_DIRS)}
    for pack in manifest:
        root = Path(pack["root"])
        for kind in KIND_DIRS:
            source = root / kind
            if source.is_dir():
                target = CONFIG_HOME / kind
                target.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target, dirs_exist_ok=True)
                kinds[kind] += sum(1 for p in source.rglob("*") if p.is_file())
    plan = skill_plan(manifest)
    from_packs = sorted(name for name, item in plan.items() if item["from"] == "pack")
    for name in from_packs:
        target = SKILLS_HOME / name
        remove_tree(target)
        SKILLS_HOME.mkdir(parents=True, exist_ok=True)
        shutil.copytree(plan[name]["dir"], target)
    excluded = excluded_skills()
    for name in sorted(excluded):
        remove_tree(SKILLS_HOME / name)
    upstream = [name for name, item in plan.items() if item["from"] == "upstream"]
    say(f"content: {len(manifest)} pack(s) installed; skills from packs: {len(from_packs)}"
        + (f" ({', '.join(from_packs)})" if from_packs else "")
        + f"; skills to fetch upstream: {len(upstream)}; "
        + (f"excluded by content-packs.exclude.txt: {', '.join(sorted(excluded))}; " if excluded else "")
        + "; ".join(f"{kind}: {count}" for kind, count in kinds.items()))


def resolve_by_tree(entry, work, env):
    """Find the upstream commit whose tree for the skill folder equals the tree SHA recorded in the lock; None when history holds none."""
    folder = posixpath.dirname(entry.get("skillPath") or "")
    wanted = entry.get("skillFolderHash")
    if not folder or not wanted:
        return None
    for commit in git(["log", "--format=%H", "FETCH_HEAD", "--", folder], work, env).splitlines():
        if git(["rev-parse", f"{commit}:{folder}"], work, env) == wanted:
            return commit
    return None


def fetch_skill_group(url, entries, work_root):
    """Fetch one upstream repository and materialize each skill folder listed in `entries`; return per-skill (name, lines), each commit group preceded by its duration."""
    out = []
    groups = {}
    for entry in entries:
        groups.setdefault(entry.get("commit") or "", []).append(entry)
    source = "/".join(url.rstrip("/").removesuffix(".git").split("/")[-2:])
    for commit, group in groups.items():
        began = time.monotonic()
        lines = upstream_commit(url, commit, group, work_root)
        out.append(("", [f"skills from {source}: {len(group)} in {time.monotonic() - began:.1f}s"]))
        out.extend(lines)
    return out


def upstream_commit(url, commit, group, work_root):
    """Fetch one upstream repository at `commit` and materialize the skills in `group`; return per-skill (name, lines)."""
    lines = []
    work = work_root / hashlib.sha256((url + commit).encode()).hexdigest()[:12]
    remove_tree(work)
    try:
        env = git_env(None)
        work.mkdir(parents=True)
        git(["init", "-q"], work, env)
        git(["remote", "add", "origin", url], work, env)
        depth = ["--depth", "1"] if commit else []
        git(["fetch", "-q", *depth, "--filter=blob:none", "origin", *([commit] if commit else [])], work, env)
    except FetchError as error:
        return [(entry["name"], [f"::warning::skill {entry['name']} was not fetched: {error}"]) for entry in group]
    for entry in group:
        resolved = commit or resolve_by_tree(entry, work, env)
        if not resolved:
            lines.append((entry["name"], [f"::warning::skill {entry['name']} was not fetched: no upstream commit matches the lock's tree SHA "
                                          "(run npx skills update to record a commit for it)"]))
            continue
        lines.append((entry["name"], materialize_skill(entry, resolved, work, work_root)))
    return lines


def materialize_skill(entry, resolved, work, work_root):
    """Extract one skill folder at `resolved`, verify it against the lock entry, and install it; return log lines."""
    name = entry["name"]
    if not re.fullmatch(r"[\w.-]+", name):
        return ["::warning::a skill name in the lock is not a plain directory name and was skipped"]
    folder = posixpath.dirname(entry.get("skillPath") or "")
    env = git_env(None)
    try:
        if not folder:
            raise FetchError("the lock entry has no skillPath")
        tree = git(["rev-parse", f"{resolved}:{folder}"], work, env)
        expected_tree = entry.get("skillFolderHash")
        if expected_tree and tree != expected_tree:
            raise FetchError("upstream content does not match the lock's tree SHA")
        expected = git(["ls-tree", "-r", "--name-only", resolved, "--", folder], work, env).splitlines()
        staging = work_root / "staging" / hashlib.sha256(name.encode()).hexdigest()[:12]
        remove_tree(staging)
        staging.mkdir(parents=True)
        archive = subprocess.Popen(["git", "-C", str(work), "archive", resolved, "--", folder], stdout=subprocess.PIPE)
        extract = subprocess.run(["tar", "-x", "-f", "-", "-C", str(staging)], stdin=archive.stdout)
        archive.stdout.close()
        if archive.wait() != 0 or extract.returncode != 0:
            raise FetchError("extracting the skill folder failed")
        source = staging / folder
        if sum(1 for p in source.rglob("*") if p.is_file()) != len(expected):
            raise FetchError("the extracted skill folder is incomplete")
        if any(p.is_symlink() for p in source.rglob("*")):
            raise FetchError("the skill folder holds a link")
        for meta in source.glob("metadata.json"):
            meta.unlink()
        target = SKILLS_HOME / name
        remove_tree(target)
        shutil.copytree(source, target)
        installed = sum(1 for p in target.rglob("*") if p.is_file())
    except FetchError as error:
        return [f"::warning::skill {name} was not fetched: {error}"]
    return [f"skill {name}: fetched from upstream at {resolved[:12]}, {installed} files, tree verified"]


def cmd_fetch_skills(args):
    """Fetch every upstream skill in the plan, one shallow fetch per source repository, repositories in parallel."""
    plan = json.loads((STATE / "skills.json").read_text()) if (STATE / "skills.json").is_file() else {}
    entries = [{"name": name, **item} for name, item in plan.items() if item["from"] == "upstream"]
    by_url = {}
    for entry in entries:
        by_url.setdefault(entry.get("sourceUrl") or "", []).append(entry)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(lambda kv: fetch_skill_group(kv[0], kv[1], STATE / "upstream"), by_url.items())
    if not entries:
        say("skills: nothing to fetch upstream")
    for group in results:
        for _, lines in group:
            for line in lines:
                say(line)


def merge_config(base, fragment):
    """Deep-merge `fragment` under `base`: objects merge, arrays are unioned, scalars keep the base value."""
    if isinstance(base, dict) and isinstance(fragment, dict):
        out = dict(base)
        for key, value in fragment.items():
            out[key] = merge_config(base[key], value) if key in base else value
        return out
    if isinstance(base, list) and isinstance(fragment, list):
        return base + [item for item in fragment if item not in base]
    return base


def strip_permissions(node):
    """Drop every `permission` key at any depth (the top level and agent definitions alike); return the cleaned node and how many were dropped."""
    if not isinstance(node, dict):
        return node, 0
    cleaned, dropped = {}, 0
    for key, value in node.items():
        if key == "permission":
            dropped += 1
            continue
        cleaned[key], removed = strip_permissions(value)
        dropped += removed
    return cleaned, dropped


def cmd_merge_config(args):
    """Merge every pack's `opencode.json` fragment into the generated config; fragments never contribute `permission`, at any depth."""
    path = Path(args.config)
    config = json.loads(path.read_text())
    for pack in load_manifest():
        fragment_path = Path(pack["root"]) / "opencode.json"
        if not fragment_path.is_file():
            continue
        try:
            fragment = json.loads(fragment_path.read_text())
        except ValueError:
            say(f"content pack {pack['index']}: opencode.json is not valid JSON and was not merged", "warning")
            continue
        if not isinstance(fragment, dict):
            say(f"content pack {pack['index']}: opencode.json is not an object and was not merged", "warning")
            continue
        fragment, dropped = strip_permissions(fragment)
        if dropped:
            say(f"content pack {pack['index']}: a pack cannot change permissions; {dropped} permission key(s) were ignored", "warning")
        config = merge_config(config, fragment)
        say(f"content pack {pack['index']}: config fragment merged")
    path.write_text(json.dumps(config))


def relay_extension(job):
    """Print an extension's result: its summary line, its workflow commands, and the log tail when it failed."""
    log = Path(job["log"])
    if not log.is_file():
        say(f"extension '{job['label']}' left no log; it most likely never started", "warning")
        return
    lines = log.read_text(errors="replace").splitlines()
    summaries = [line for line in lines if line.startswith("EXT-SUMMARY:")]
    for line in lines:
        if line.startswith(("::warning::", "::error::", "::notice::")):
            # An extension cannot fail the run from its own log: its errors relay as warnings.
            level = "notice" if line.startswith("::notice::") else "warning"
            say(f"{job['label']}: {line.split('::', 2)[2]}", level)
    status = Path(job["status"]).read_text().strip() if Path(job["status"]).is_file() else "?"
    for line in summaries:
        say(f"[{job['label']}] {line.removeprefix('EXT-SUMMARY:').strip()}")
    if status != "0":
        say(f"extension '{job['label']}' ended with status {status}", "warning")
        for line in lines[-15:]:
            say(f"    {line}")


def opencode_binary():
    """Return the opencode executable, found without relying on PATH; None when it is not installed."""
    found = shutil.which("opencode")
    if found:
        return found
    fallback = Path.home() / ".opencode" / "bin" / "opencode"
    return str(fallback) if fallback.is_file() else None


def opencode_report(binary):
    """Print any plugin load error and the names of the skills opencode discovers."""
    # One boot answers both: with plugins loaded, its stderr carries plugin errors and its
    # stdout the skill listing. Project config is switched off so this boot never reads
    # anything from the workspace, which can still be an unscrubbed checkout at this point.
    # Loading plugins waits for the plugin SDK opencode installs into each of its config
    # directories before the first one loads (about 5 seconds per directory on a fresh
    # runner, 20 to 70 when the package registry is slow).
    env = dict(os.environ, OPENCODE_DISABLE_PROJECT_CONFIG="1")
    with tempfile.TemporaryFile() as buffer:
        # opencode can exit before a large stdout pipe has drained (the listing was cut at
        # 128 KiB on a Linux runner), so it writes to an unlinked temp file instead. The
        # listing carries every skill's full text, which may be private: only names leave it.
        try:
            boot = subprocess.run([binary, "debug", "skill", "--print-logs", "--log-level", "ERROR"], stdout=buffer,
                                  stderr=subprocess.PIPE, env=env, timeout=240)
        except (OSError, subprocess.TimeoutExpired) as error:
            say(f"content check did not finish: {type(error).__name__}", "warning")
            return
        buffer.seek(0)
        listing = buffer.read()
    problems = [line for line in boot.stderr.decode("utf-8", errors="replace").splitlines() if "plugin" in line.lower()]
    for line in problems:
        say(f"plugin error: {line[-300:]}", "warning")
    if not problems:
        say("plugin load check: no errors")
    try:
        names = sorted(str(item.get("name")) for item in json.loads(listing.decode("utf-8", errors="replace")))
    except (ValueError, AttributeError) as error:
        say(f"could not list discovered skills: opencode exited with {boot.returncode} and wrote {len(listing)} bytes,"
            f" which is not the expected listing ({type(error).__name__})", "warning")
        return
    say(f"skills discovered by opencode ({len(names)}): {', '.join(names)}")


def cmd_wait(args):
    """Wait for every background job, relay what they reported, and print the content report."""
    jobs = json.loads((STATE / "jobs.json").read_text()) if (STATE / "jobs.json").is_file() else []
    deadline = time.monotonic() + args.timeout
    while any(not Path(job["status"]).is_file() for job in jobs) and time.monotonic() < deadline:
        time.sleep(1)
    for job in jobs:
        if not Path(job["status"]).is_file():
            say(f"background job '{job['label']}' did not finish in {args.timeout}s and was stopped", "warning")
            try:
                os.kill(job["pid"], getattr(signal, "SIGKILL", signal.SIGTERM))
            except OSError:
                pass
    for job in jobs:
        if job["name"] == "upstream-skills":
            if Path(job["log"]).is_file():
                for line in Path(job["log"]).read_text(errors="replace").splitlines():
                    if line.strip():
                        say(line)
        else:
            relay_extension(job)
        # How long the wait was held by each job: the log shows what setup waited for.
        if Path(job["status"]).is_file():
            say(f"content: background job '{job['label']}' finished in {os.path.getmtime(job['status']) - job.get('started', time.time()):.1f}s")
    binary = opencode_binary()
    if binary:
        opencode_report(binary)
    else:
        # Never a silent skip: a green setup with no content check must say so.
        say("opencode is not installed, so the content check was skipped (no plugin check, no skill names)", "warning")


def main(argv=None):
    """Dispatch one subcommand; `fetch` and `wait` report problems as warnings, the others fail loudly."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download every configured pack")
    listed = [os.environ.get(name, "") for name in ("CONTENT_PACKS", "PRIVATE_CONTENT_PACKS")]
    fetch.add_argument("--packs", default=", ".join(value for value in listed if value.strip()))
    fetch.set_defaults(func=cmd_fetch)
    sub.add_parser("start-skills", help="launch the upstream skill fetch in the background").set_defaults(func=cmd_start_skills)
    sub.add_parser("install", help="copy allowlisted content into the opencode homes").set_defaults(func=cmd_install)
    merge = sub.add_parser("merge-config", help="merge opencode.json fragments into a config file")
    merge.add_argument("config")
    merge.set_defaults(func=cmd_merge_config)
    sub.add_parser("fetch-skills", help="fetch upstream skills (background job)").set_defaults(func=cmd_fetch_skills)
    sub.add_parser("start", help="launch extensions in the background").set_defaults(func=cmd_start)
    wait = sub.add_parser("wait", help="wait for background jobs and report")
    wait.add_argument("--timeout", type=int, default=300)
    wait.set_defaults(func=cmd_wait)
    args = parser.parse_args(argv)
    if args.command == "wait":
        try:
            args.func(args)
        except Exception as error:  # the wait step only reports; it must never end the run
            say(f"wait: {type(error).__name__}", "warning")
    else:
        args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Validate the LLM-API-Key-Proxy fork stack metadata.

This script intentionally uses only the Python standard library so it can run in
fresh workspaces without installing project dependencies.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STACK = ROOT / ".fork" / "stack.yml"
FEATURES = ROOT / ".fork" / "features"
AGENTS = ROOT / "AGENTS.md"

SUBJECT_RE = re.compile(r"^\s*subject:\s+\"(?P<subject>.+)\"\s*$")
ID_RE = re.compile(r"^\s*- id:\s+(?P<id>[A-Za-z0-9_.-]+)\s*$")
DUP_FEATURE_RE = re.compile(r"^\s{4}(?P<feature>[A-Za-z0-9_.-]+):\s*$")
DUP_SUBJECT_RE = re.compile(r"^\s{6}-\s+\"(?P<subject>.+)\"\s*$")
PREFIX_RE = re.compile(r"^(?P<kind>feat|fix)\((?P<feature>[^)]+)\):")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def parse_manifest() -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    text = STACK.read_text()
    ids: dict[str, str] = {}
    subjects: dict[str, str] = {}
    allowed_duplicates: dict[str, set[str]] = {}
    current_id: str | None = None
    in_allowed = False
    current_allowed: str | None = None

    for line in text.splitlines():
        if line.strip() == "allowed_duplicate_features:":
            in_allowed = True
            current_allowed = None
            continue
        if line.startswith("features:"):
            in_allowed = False
            current_allowed = None
            continue
        if in_allowed:
            m = DUP_FEATURE_RE.match(line)
            if m:
                current_allowed = m.group("feature")
                allowed_duplicates.setdefault(current_allowed, set())
                continue
            m = DUP_SUBJECT_RE.match(line)
            if m and current_allowed is not None:
                allowed_duplicates.setdefault(current_allowed, set()).add(m.group("subject"))
            continue

        m = ID_RE.match(line)
        if m:
            current_id = m.group("id")
            ids[current_id] = ""
            continue
        m = SUBJECT_RE.match(line)
        if m and current_id is not None:
            subjects[m.group("subject")] = current_id
            ids[current_id] = m.group("subject")
            current_id = None

    return ids, subjects, allowed_duplicates


def stack_subjects() -> list[str]:
    output = git("log", "--format=%s", "--reverse", "upstream/dev..HEAD")
    return [line for line in output.splitlines() if line]


def check_agents(errors: list[str]) -> None:
    text = AGENTS.read_text()
    release_notes = sum(1 for line in text.splitlines() if line.strip() == "### Release Notes")
    if release_notes != 1:
        errors.append(f"AGENTS.md must contain exactly one '### Release Notes' heading (found {release_notes})")
    if text.count("```") % 2:
        errors.append("AGENTS.md has unbalanced fenced code blocks")
    if any(line.strip() == "git add -A" for line in text.splitlines()):
        errors.append("AGENTS.md contains an executable `git add -A` example")
    for marker in ("<<<<<<<", ">>>>>>>"):
        if marker in text:
            errors.append(f"AGENTS.md contains conflict marker {marker}")
    if ".fork/features" not in text:
        errors.append("AGENTS.md must document .fork/features as canonical feature history")
    if "local workspace state" not in text.lower():
        errors.append("AGENTS.md must state that local workspace state is non-canonical")


def check_stack(errors: list[str]) -> None:
    ids, manifest_subjects, allowed_duplicates = parse_manifest()
    subjects = stack_subjects()
    stack_set = set(subjects)

    for subject in manifest_subjects:
        if subject not in stack_set:
            errors.append(f"manifest subject not found in stack: {subject}")

    for subject in subjects:
        if subject not in manifest_subjects:
            m = PREFIX_RE.match(subject)
            if not m:
                errors.append(f"stack commit lacks known manifest subject and feature prefix: {subject}")
                continue
            feature = m.group("feature")
            allowed = allowed_duplicates.get(feature, set())
            if subject not in allowed:
                errors.append(f"stack commit is not in manifest or allowed exceptions: {subject}")

    by_feature: dict[str, list[str]] = {}
    for subject in subjects:
        m = PREFIX_RE.match(subject)
        if not m:
            continue
        by_feature.setdefault(m.group("feature"), []).append(subject)

    for feature, feature_subjects in sorted(by_feature.items()):
        if len(feature_subjects) <= 1:
            continue
        allowed = allowed_duplicates.get(feature, set())
        unexpected = [s for s in feature_subjects if s not in allowed]
        manifest_for_feature = [s for s, fid in manifest_subjects.items() if fid == feature]
        # Multiple commits are allowed only when every commit is either the canonical
        # manifest subject for that feature or an explicitly documented exception.
        permitted = set(allowed) | set(manifest_for_feature)
        if any(s not in permitted for s in feature_subjects):
            errors.append(f"feature {feature!r} has unexpected duplicate stack commits: {feature_subjects}")

    for feature_id in ids:
        feature_file = FEATURES / f"{feature_id}.md"
        if not feature_file.exists():
            # Only require detailed histories for features that have a feature file
            # once they change under the new workflow. Keep stack-wide adoption
            # incremental instead of forcing 20+ stub docs on day one.
            continue
        text = feature_file.read_text()
        subject = ids[feature_id]
        if subject and subject not in text:
            errors.append(f"{feature_file} does not mention its stack subject")


def main() -> int:
    errors: list[str] = []
    if not STACK.exists():
        errors.append("missing .fork/stack.yml")
    if not FEATURES.exists():
        errors.append("missing .fork/features/")
    if not AGENTS.exists():
        errors.append("missing AGENTS.md")
    if not errors:
        check_agents(errors)
        check_stack(errors)

    if errors:
        print("fork stack validation failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 1

    print("fork stack validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

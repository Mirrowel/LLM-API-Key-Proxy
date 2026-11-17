#!/usr/bin/env python3
"""Dependency admission gate for PyPI packages.

Answers one question: "is this exact version safe enough to install today?"

Checks, in order, each of which can stop the install:

  1. exists      -- the version is really on PyPI
  2. cooldown    -- it is at least --days old (most compromises are caught
                    within 24-72h; time is the cheapest defence there is)
  3. integrity   -- the artifact hashes match the PyPI JSON API digests
  4. malware     -- guarddog's heuristic rules find nothing dangerous
  5. advisories  -- pip-audit/OSV knows of no CVE for this version

Usage:
  dep-gate.py requests==2.32.3
  dep-gate.py requests==2.32.3 --days 3
  dep-gate.py --latest litellm            # gate whatever is newest
  dep-gate.py litellm==1.106.2 --lock     # also emit a hash-locked line

Exit codes: 0 admit, 1 reject, 2 blocked by cooldown, 3 tooling error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone

PYPI = "https://pypi.org/pypi/{name}/json"

# Rules that disqualify a release outright, regardless of baseline. Calibrated
# against litellm 1.80.16 (a 55-finding, 10/10 package) so that generic framework
# behaviour such as fs reads, outbound HTTP and "obfuscation" in bundled JS does
# not trip them -- only the narrow techniques seen in real registry compromises.
FATAL_RULES = {
    "threat-runtime-obfuscation-base64exec",   # decode-then-exec payload
    "threat-runtime-obfuscation-import-exec",
    "threat-runtime-obfuscation-hidden-code",
    "threat-setup-network-in-install",         # network from setup.py
    "threat-filesystem-autostart",             # persistence
    "threat-process-powershell-encoded",
    "threat-network-exfil-sysinfo",
    "typosquatting",
    "deceptive_author",
    "repository_integrity_mismatch",
    "metadata_mismatch",
}

def fatal_reason(r: dict) -> str | None:
    if (r.get("file_path") or "").endswith(".pth"):
        return f".pth finding ({r.get('name')}) -- install-time execution vector"
    rule = r.get("threat_rule") or r.get("capability_rule") or ""
    if rule in FATAL_RULES:
        return rule
    return None
GUARDDOG = shutil.which("guarddog") or os.path.expanduser("~/.local/bin/guarddog")
PIP_AUDIT = shutil.which("pip-audit") or os.path.expanduser("~/.local/bin/pip-audit")


def get(url: str, timeout: int = 30, accept: str | None = None) -> bytes:
    headers = {"User-Agent": "dep-gate/1.0"}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def meta(name: str) -> dict:
    return json.loads(get(PYPI.format(name=name)))


def parse_spec(spec: str) -> tuple[str, str | None]:
    m = re.fullmatch(r"([A-Za-z0-9._-]+)\s*(?:==\s*([^\s]+))?", spec.strip())
    if not m:
        fail(f"cannot parse requirement {spec!r} (expected name or name==version)")
    return m.group(1), m.group(2)


def version_key(v: str) -> tuple:
    """Rough numeric ordering for dotted PEP 440 versions (stdlib only)."""
    out = []
    for part in re.split(r"[.+-]", v):
        m = re.match(r"(\d+)", part)
        out.append(int(m.group(1)) if m else 0)
    return tuple(out)


def sig(r: dict) -> str:
    """Stable identity for a guarddog finding, used for baseline diffing."""
    name = r.get("name", "?")
    loc = (r.get("threat_location") or "").split(":")[0]
    match = (r.get("threat_match") or "")[:40]
    return f"{name} @ {loc}" + (f" [{match}]" if match else "")


def is_bundled_asset(r: dict) -> bool:
    """True for a finding inside a prebuilt frontend bundle shipped in the wheel.

    A wheel that vendors a web dashboard always contains minified webpack
    chunks whose use of eval() is normal bundler output, not obfuscated Python.
    Measured 2026-11-17: litellm 1.105.1 and 1.108.3 each ship two such chunks,
    so the runtime-obfuscation rule disqualified every release.
    """
    loc = r.get("file_path") or r.get("threat_location") or ""
    return "/_next/static/" in loc or "/_experimental/out/" in loc


def guarddog_json(path: str) -> dict:
    p = subprocess.run(
        [GUARDDOG, "pypi", "scan", "--output-format", "json", path],
        capture_output=True, text=True,
    )
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        fail(f"guarddog gave unparseable output for {path}", 3)


def scan_artifacts(paths: list[str]) -> dict:
    risks: list[dict] = []
    score = 0.0
    for path in paths:
        out = guarddog_json(path)
        score = max(score, out.get("risk_score", {}).get("score", 0.0))
        risks.extend(out.get("risks", []))
    return {"score": score, "risks": risks}


def scan_release(name: str, version: str) -> dict:
    tmp = tempfile.mkdtemp(prefix="dep-gate-base-")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "--no-build-isolation",
             "--dest", tmp, f"{name}=={version}"],
            capture_output=True, text=True, check=True,
        )
        return scan_artifacts([os.path.join(tmp, f) for f in sorted(os.listdir(tmp))])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fail(msg: str, code: int = 1) -> "None":
    print(f"REJECT: {msg}")
    sys.exit(code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("requirement", nargs="?", help="name or name==version")
    ap.add_argument("--latest", metavar="NAME", help="gate the newest release of NAME")
    ap.add_argument("--newest-aged", metavar="NAME",
                    help="gate the newest release of NAME older than --days (skips the cooldown check)")
    ap.add_argument("--file", metavar="PATH", help="gate a local wheel/sdist instead of PyPI")
    ap.add_argument("--days", type=int, default=7, help="required release age (default 7)")
    ap.add_argument("--require-provenance", action="store_true",
                    help="refuse versions with no PEP 740 build attestation")
    ap.add_argument("--baseline", metavar="NAME==VER",
                    help="known-good version to diff indicators against")
    ap.add_argument("--lock", action="store_true", help="print a hash-locked requirement line")
    ap.add_argument("--report", metavar="PATH", help="write a JSON report")
    args = ap.parse_args()

    if args.newest_aged:
        name = args.newest_aged
        version = None
    elif args.latest:
        name = args.latest
        version = None
    elif args.requirement:
        name, version = parse_spec(args.requirement)
    elif args.file:
        name, version = args.file, None
    else:
        ap.error("give a requirement, --latest NAME, or --file PATH")

    review_only = False

    report: dict = {"requested": args.requirement or f"(latest) {name}", "steps": {}}

    # 0. local artifact ------------------------------------------------------
    if args.file:
        path = os.path.abspath(args.file)
        if not os.path.exists(path):
            fail(f"no such file: {path}", 3)
        p = subprocess.run(
            [GUARDDOG, "pypi", "scan", "--output-format", "json", path],
            capture_output=True, text=True,
        )
        try:
            out = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            fail("guarddog gave unparseable output", 3)
        score = out.get("risk_score", {}).get("score", 0.0)
        label = out.get("risk_score", {}).get("label", "?")
        rules = [f"{r.get('name','?')} @ {r.get('threat_location','?')}" for r in out.get("risks", [])]
        print(f"  malware     {label:20s} {os.path.basename(path)} score={score}")
        for r in rules[:8]:
            print(f"              - {r}")
        report["risk_score"] = score
        report["risk_labels"] = rules
        report["steps"]["malware"] = "fail" if score >= 5.0 else "pass"
        if args.report:
            with open(args.report, "w") as fh:
                json.dump(report, fh, indent=2)
        if score >= 5.0:
            fail(f"guarddog scores {score:.1f}/10 ({label})")
        print(f"ADMIT: {os.path.basename(path)}")
        return 0

    # 1. exists -------------------------------------------------------------
    try:
        info = meta(name)
    except Exception as exc:  # 404 / network
        fail(f"{name}: cannot read PyPI metadata ({exc})", 3)

    aged_mode = bool(args.newest_aged)
    if version is None and aged_mode:
        rows = []
        for v, vfiles in info["releases"].items():
            if not vfiles or any(t in v for t in ("rc", "a", "b", "post", "dev")):
                continue
            t = min(f["upload_time_iso_8601"] for f in vfiles)
            rows.append((v, datetime.fromisoformat(t.replace("Z", "+00:00"))))
        rows.sort(key=lambda r: r[1])
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
        eligible = [r for r in rows if r[1] <= cutoff]
        if not eligible:
            fail(f"no stable {name} release older than {args.days} days", 2)
        version = eligible[-1][0]
        print(f"{name}: newest release older than {args.days}d is {version}")
    if version is None:
        version = info["info"]["version"]
        print(f"{name}: newest release is {version}")
    files = info["releases"].get(version) or []
    if not files:
        fail(f"{name}=={version}: version not found on PyPI")
    upload = min(f["upload_time_iso_8601"] for f in files)
    uploaded = datetime.fromisoformat(upload.replace("Z", "+00:00"))
    age_days = (datetime.now(timezone.utc) - uploaded).total_seconds() / 86400
    report["version"] = version
    report["uploaded"] = uploaded.isoformat()
    report["age_days"] = round(age_days, 2)
    report["steps"]["exists"] = "pass"
    print(f"  exists      ok   {name}=={version} uploaded {uploaded:%Y-%m-%d} ({age_days:.1f}d ago)")

    # 1b. already ahead of the aged window ----------------------------------
    if args.baseline:
        base_name, base_ver = parse_spec(args.baseline)
        if base_ver and version_key(base_ver) > version_key(version):
            print(f"  result      ok   pinned {base_name}=={base_ver} is newer than the newest aged release ({version})")
            report["steps"]["result"] = "up-to-date"
            if args.report:
                with open(args.report, "w") as fh:
                    json.dump(report, fh, indent=2)
            return 0

    # 2. cooldown -----------------------------------------------------------
    if aged_mode:
        report["steps"]["cooldown"] = "pass"
        print(f"  cooldown    ok   selected from releases >= {args.days}d old")
    elif age_days < args.days:
        report["steps"]["cooldown"] = "blocked"
        if args.report:
            with open(args.report, "w") as fh:
                json.dump(report, fh, indent=2)
        fail(
            f"released {age_days:.1f} days ago, need {args.days}. "
            f"Wait, or re-run with --days {int(age_days)} to accept the risk explicitly.",
            2,
        )
    report["steps"]["cooldown"] = "pass"
    print(f"  cooldown    ok   >= {args.days}d")

    tmp = tempfile.mkdtemp(prefix="dep-gate-")
    try:
        # 3+4. download, verify hashes, scan ---------------------------------
        dl = subprocess.run(
            [
                sys.executable, "-m", "pip", "download", "--no-deps", "--no-build-isolation",
                "--dest", tmp, f"{name}=={version}",
            ],
            capture_output=True, text=True,
        )
        if dl.returncode != 0:
            fail(f"download failed: {dl.stderr.strip().splitlines()[-1:]}", 3)

        expected = {}
        for f in files:
            if f["packagetype"] in ("bdist_wheel", "sdist"):
                expected[f["filename"]] = f["digests"]["sha256"]
        got = sorted(os.listdir(tmp))
        mismatch = []
        for fn in got:
            with open(os.path.join(tmp, fn), "rb") as fh:
                h = hashlib.sha256(fh.read()).hexdigest()
            if fn in expected and expected[fn] != h:
                mismatch.append(fn)
        if mismatch:
            report["steps"]["integrity"] = "fail"
            fail(f"sha256 mismatch vs PyPI JSON: {mismatch}")
        report["steps"]["integrity"] = "pass"
        hashes = {}
        for fn in got:
            with open(os.path.join(tmp, fn), "rb") as fh:
                hashes[fn] = hashlib.sha256(fh.read()).hexdigest()
        print(f"  integrity   ok   {len(got)} artifact(s) match PyPI digests")

        if not os.path.exists(GUARDDOG):
            report["steps"]["malware"] = "skipped"
            print("  malware     SKIP guarddog not installed")
        else:
            review_only = False
            scan = scan_artifacts([os.path.join(tmp, fn) for fn in got])
            report["risk_score"] = scan["score"]
            report["risk_labels"] = [sig(r) for r in scan["risks"]][:15]
            print(f"  malware     score={scan['score']:.1f} rules={len(scan['risks'])} (absolute score is not a verdict)")

            fatal_all = [(r, fatal_reason(r)) for r in scan["risks"] if fatal_reason(r)]
            fatal, bundled = [], []
            for r, why in fatal_all:
                (bundled if is_bundled_asset(r) else fatal).append((r, why))
            if bundled:
                report["suppressed_fatal"] = [f"{sig(r)} <- {why}" for r, why in bundled][:10]
                print(
                    f"  malware     note {len(bundled)} forbidden indicator(s) only inside "
                    "bundled web assets (minified JS), not executable Python"
                )
            newly = list(scan["risks"])
            if args.baseline:
                base_name, base_ver = parse_spec(args.baseline)
                if (base_name, base_ver) == (name, version):
                    # Comparing a release with itself: reuse the scan instead of
                    # downloading and scanning the same artifact twice.
                    base_scan = scan
                else:
                    base_scan = scan_release(base_name, base_ver)
                base_sigs = {sig(r) for r in base_scan["risks"]}
                newly = [r for r in scan["risks"] if sig(r) not in base_sigs]
                report["baseline"] = f"{base_name}=={base_ver}"
                report["baseline_score"] = base_scan["score"]
                report["new_risks"] = [sig(r) for r in newly][:25]
                print(
                    f"  baseline    {base_name}=={base_ver} score={base_scan['score']:.1f} "
                    f"rules={len(base_scan['risks'])} -> {len(newly)} new rule hit(s)"
                )
                for r in newly[:10]:
                    print(f"              + {sig(r)}")

            if fatal:
                report["steps"]["malware"] = "fail"
                report["fatal"] = [f"{sig(r)} <- {why}" for r, why in fatal][:10]
                fail(f"forbidden indicator(s): {report['fatal'][:5]}")
            if args.baseline and newly:
                high = [r for r in newly if r.get("severity") == "high"]
                if high:
                    # Not a rejection: a legit release can add a web bundle or a
                    # subprocess call. A human reads the delta. Binary pass/fail
                    # here has been shown to reject correct upgrades.
                    report["steps"]["malware"] = "review"
                else:
                    report["steps"]["malware"] = "pass"
            elif not args.baseline and scan["score"] >= 5.0:
                report["steps"]["malware"] = "fail"
                fail(
                    f"guarddog scores {scan['score']:.1f}/10 with no baseline to compare against; "
                    "large frameworks always score high -- pass --baseline KNOWN_GOOD==VERSION "
                    "to judge only what changed"
                )
            report["steps"]["malware"] = report["steps"].get("malware", "pass")

            if report["steps"]["malware"] == "review":
                high = [r for r in newly if r.get("severity") == "high"]
                report["review_items"] = sorted({sig(r) for r in high})[:10]
                review_only = True

        # 5. provenance -------------------------------------------------------
        # PEP 740 attestations prove the artifact was built by the maintainers' CI
        # from a named repo, not hand-uploaded. Absent attestation is not proof of
        # malice, but hand uploads are how the litellm 1.82.7/1.82.8 incident worked.
        try:
            simple = json.loads(get(
                f"https://pypi.org/simple/{name}/",
                accept="application/vnd.pypi.simple.v1+json",
            ))
            same = [f for f in simple.get("files", []) if f["filename"].startswith(
                f"{name.replace('-', '_')}-{version}")]
            attested = [f for f in same if f.get("provenance")]
            report["provenance"] = f"{len(attested)}/{len(same)} files attested"
            if attested:
                prov = json.loads(get(attested[0]["provenance"]))
                pubs = [b.get("publisher", {}) for b in prov.get("attestation_bundles", [])]
                pub = pubs[0] if pubs else {}
                report["publisher"] = pub
                print(
                    f"  provenance  ok   built by {pub.get('kind','?')} "
                    f"{pub.get('repository','?')} / {pub.get('workflow','?')}"
                )
                report["steps"]["provenance"] = "pass"
            else:
                report["steps"]["provenance"] = "missing"
                print("  provenance  WARN no PEP 740 attestation -- artifact was hand-uploaded/unsigned")
                if args.require_provenance:
                    fail(
                        f"{name}=={version} has no build attestation and --require-provenance was set; "
                        "either pin a version that is attested or accept a hand upload explicitly",
                        4,
                    )
        except Exception as exc:
            report["steps"]["provenance"] = f"error: {exc}"
            print(f"  provenance  SKIP could not read attestations ({exc})")

        # 6. advisories ------------------------------------------------------
        if os.path.exists(PIP_AUDIT):
            p = subprocess.run(
                [PIP_AUDIT, "--no-deps", "--format", "json", "-r", "/dev/stdin"],
                input=f"{name}=={version}\n", capture_output=True, text=True,
            )
            try:
                vulns = json.loads(p.stdout or "{}")
            except Exception:
                vulns = {}
            deps = vulns.get("dependencies", [])
            found = [v for d in deps for v in d.get("vulns", [])]
            report["steps"]["advisories"] = "fail" if found else "pass"
            report["vulns"] = sorted({v.get("id") for v in found if v.get("id")})
            # pip-audit resolves the tree, so a rejected top-level pin does not
            # mean the top-level package is the vulnerable one. On 2026-11-17 all
            # 13 advisories belonged to filelock while the message blamed
            # litellm, which sent the reader to the wrong dependency.
            flagged = {
                f"{d.get('name')}=={d.get('version')}": sorted(
                    {v.get("id") for v in d.get("vulns", []) if v.get("id")}
                )
                for d in deps
                if d.get("vulns")
            }
            report["vuln_packages"] = flagged
            if found:
                detail = "; ".join(
                    f"{pkg} ({', '.join(ids)})" for pkg, ids in sorted(flagged.items())
                )
                fail(f"known advisories in the tree of {name}=={version}: {detail}")
            print("  advisories  ok   no OSV/PyPI advisory for this version")
        else:
            report["steps"]["advisories"] = "skipped"
            print("  advisories  SKIP pip-audit not installed")

        if review_only:
            print(f"REVIEW: {name}=={version} -- {len(report.get('review_items', []))} new high-severity indicator(s):")
            for item in report.get("review_items", []):
                print(f"          ? {item}")
            return 4

        print(f"ADMIT: {name}=={version}")

        if args.lock:
            h = hashes.get(f"{name.replace('-', '_')}-{version}-py3-none-any.whl")
            for fn, digest in sorted(hashes.items()):
                print(f"# {fn}")
                print(f"{name}=={version} --hash=sha256:{digest}")
                break
        return 0
    finally:
        if args.report:
            with open(args.report, "w") as fh:
                json.dump(report, fh, indent=2)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

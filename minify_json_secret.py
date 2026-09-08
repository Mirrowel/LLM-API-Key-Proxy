#!/usr/bin/env python3
"""Validate JSON and re-emit it minified: one line, no whitespace.

Built for GitHub secret preparation (OPENCODE_CONFIG_JSON and other secrets
cannot safely contain literal newlines); works as a generic JSON minifier.
The minified JSON goes to stdout -- or the -o file -- as exact UTF-8 bytes,
so output is identical on every platform and locale. All diagnostics go to
stderr; stdout is always pipe-pure.

    python minify_json_secret.py                                  # default input
    python minify_json_secret.py my-full-config.json              # print minified
    python minify_json_secret.py my-full-config.json -o secret.min.json
    cat config.json | python minify_json_secret.py - -q > out.txt
    python minify_json_secret.py --check config.json              # validate only

Stdout receives one trailing newline (terminal/pipe convention, like jq);
the -o file receives none (byte-exact, secret-safe). Exit codes:
0 success, 1 runtime error (invalid JSON, I/O), 2 usage error.
"""

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_INPUT_NAME = ".github/actions/bot-setup/permissions.example.json"
DEFAULT_INPUT = Path(__file__).parent / DEFAULT_INPUT_NAME
GITHUB_SECRET_LIMIT = 48 * 1024  # GitHub rejects individual secrets over 48 KB


def _reject_constant(name: str) -> float:
    """Reject NaN/Infinity/-Infinity literals: not valid JSON (RFC 8259)."""
    raise ValueError(f"{name} is not valid JSON (RFC 8259 has no NaN/Infinity)")


def _finite_float(text: str) -> float:
    """Parse a JSON float, rejecting values that overflow to infinity."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"number {text} overflows to infinity; not representable in JSON")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate JSON and emit it minified (single line, no whitespace).",
        epilog="Diagnostics print to stderr; the minified JSON prints to stdout "
               "(or the -o file) as UTF-8, so piping the output elsewhere is safe.",
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help=f"JSON file to minify, or '-' for stdin "
             f"(default: {DEFAULT_INPUT_NAME} next to this script)",
    )
    parser.add_argument(
        "-o", "--output", metavar="FILE", default=None,
        help="write minified JSON to FILE instead of stdout "
             "(UTF-8, no trailing newline -- byte-exact, secret-safe)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="suppress notes and warnings on stderr (errors are always shown)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="validate only: print nothing to stdout, ignore -o",
    )
    args = parser.parse_args()

    def note(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr)

    # --- Read input as bytes (uniform for file and stdin) -------------------
    src = args.input
    if src is None:
        src = str(DEFAULT_INPUT)
        note(f"No input given, using default: {src}")

    label = "stdin"
    try:
        if src == "-":
            stdin_buffer = getattr(sys.stdin, "buffer", None)
            if stdin_buffer is None:
                print("Error: stdin is not available.", file=sys.stderr)
                return 1
            raw_bytes = stdin_buffer.read()
        else:
            path = Path(src)
            label = str(path)
            raw_bytes = path.read_bytes()
    except OSError as e:
        print(f"Error: cannot read {label}: {e}", file=sys.stderr)
        return 1

    try:
        raw = raw_bytes.decode("utf-8-sig")  # tolerates (and strips) a UTF-8 BOM
    except UnicodeDecodeError as e:
        print(f"Error: {label} is not valid UTF-8: {e}", file=sys.stderr)
        return 1

    # --- Parse strictly ------------------------------------------------------
    duplicate_keys: list = []

    def keep_last(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                duplicate_keys.append(key)
            obj[key] = value  # last value wins, exactly like json.loads default
        return obj

    try:
        data = json.loads(
            raw,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=keep_last,
        )
    except ValueError as e:  # includes json.JSONDecodeError
        print(f"Error: invalid JSON in {label}: {e}", file=sys.stderr)
        return 1
    except RecursionError:
        print(f"Error: {label} is nested too deeply to parse.", file=sys.stderr)
        return 1

    # --- Minify and encode (bytes, so output is locale-independent) ---------
    minified = json.dumps(
        data, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    try:
        payload = minified.encode("utf-8")  # rejects unpaired surrogates here
    except UnicodeEncodeError:
        print(f"Error: {label} contains unpaired surrogate code points; "
              f"not encodable as UTF-8.", file=sys.stderr)
        return 1
    if b"\n" in payload or b"\r" in payload:
        # Cannot happen with json.dumps minified output; fail closed rather
        # than ever emit something that would silently truncate a secret.
        print("Error: internal error: minified output contains a newline; "
              "refusing to emit.", file=sys.stderr)
        return 1

    # --- Diagnostics (stderr only; never gate the exit code) -----------------
    note("JSON is valid.")
    if duplicate_keys:
        keys = ", ".join(sorted(set(duplicate_keys)))
        note(f"Warning: duplicate object keys (last value wins): {keys}")
    if not isinstance(data, dict):
        note("Warning: top-level JSON is not an object; the OPENCODE_CONFIG_JSON "
             "pipeline expects an object.")
    if len(payload) > GITHUB_SECRET_LIMIT:
        note(f"Warning: minified JSON is {len(payload)} bytes; "
             f"GitHub secrets are limited to 48 KB.")

    if args.check:
        return 0

    # --- Emit ----------------------------------------------------------------
    if args.output:
        try:
            Path(args.output).write_bytes(payload)
        except OSError as e:
            print(f"Error: cannot write '{args.output}': {e}", file=sys.stderr)
            return 1
        note(f"Wrote {len(payload)} bytes to '{args.output}' "
             f"(no trailing newline; secret-safe).")
    else:
        try:
            sys.stdout.buffer.write(payload + b"\n")  # bytes: no CRLF on Windows
            sys.stdout.buffer.flush()
        except OSError as e:  # includes BrokenPipeError
            print(f"Error: cannot write to stdout: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

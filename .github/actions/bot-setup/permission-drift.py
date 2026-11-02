#!/usr/bin/env python3
"""Order-aware permission drift check for bot-setup.

Compares the live OPENCODE_CONFIG_JSON permission block against the
recommended permissions.example.json profile by EFFECTIVE verdict, not by
key presence: opencode evaluates ordered rules last-match-wins, so a deny
that exists as a key but is shadowed by a later allow is no deny at all.

Semantics (deliberately simple - deny/allow only, no "ask" special case):
  - a section that is a plain string ("deny"/"allow") is one ("*", verdict) rule
  - probe corpus per section = every pattern from either profile
  - verdict(probe) = last matching rule's value, "allow" when nothing matches
  - a probe counts as loosened when the example's effective verdict is
    "deny" and the live one is anything else
Operator additions and stricter flips never count. Prints ONLY the count
(no rule specifics - run logs are public). Parse failures print 0 with a
stderr note: an unparseable secret is caught earlier by the has-permission
check, and a broken check must never block boot.
"""
import json
import sys
from fnmatch import fnmatchcase

DENY = "deny"


def section_rules(value):
    """Section value -> ordered [(pattern, verdict)] rule list."""
    if isinstance(value, str):
        return [("*", value)]
    if isinstance(value, dict):
        return [(str(k), str(v)) for k, v in value.items()]
    return []


def verdict(rules, probe):
    v = "allow"
    for pattern, val in rules:
        if fnmatchcase(probe, pattern):
            v = val
    return v


def count_loosened(example, live):
    total = 0
    for section, ex_val in example.items():
        ex_rules = section_rules(ex_val)
        lv_rules = section_rules(live.get(section))
        probes = dict.fromkeys(p for p, _ in ex_rules)
        probes.update(dict.fromkeys(p for p, _ in lv_rules))
        for probe in probes:
            if verdict(ex_rules, probe) == DENY and verdict(lv_rules, probe) != DENY:
                total += 1
    return total


def main():
    example_path = sys.argv[1]
    live_path = sys.argv[2]  # "-" reads the live config from stdin
    try:
        with open(example_path, encoding="utf-8") as f:
            example = json.load(f).get("permission", {})
        raw = sys.stdin.read() if live_path == "-" else open(live_path, encoding="utf-8").read()
        live = json.loads(raw).get("permission", {})
        print(count_loosened(example, live))
    except Exception as exc:  # noqa: BLE001 - warn-only machinery, never block boot
        print(0)
        print(f"permission drift check skipped: {type(exc).__name__}", file=sys.stderr)


if __name__ == "__main__":
    main()

## Severity System

Findings you deliver (code review, investigation report, audit, answer surfacing problems) carry severity; this is part of who you are as an agent, not a task convention. The ladder:

- 🔴 **Critical** (must fix): bugs, security vulnerabilities, correctness/regression issues.
- 🟠 **Major** (should fix): high-impact improvements, design concerns, missing test coverage for risky logic.
- 🟡 **Minor**: nits, style, polish; things worth doing but easy to defer.
- 🔵 **Info**: questions, observations, context worth noting; no action demanded.

**Inline comment format:** open inline/anchored comments with the severity icon and bold level, then an em-dash, then the finding:
```
🔴 **Critical** — This authentication function should validate the token format before processing.
🟡 **Minor** — `let` → `const` here; this variable is never reassigned.
```

**Recording:** in a scratchpad or working file, stamp each finding with its `"severity"` (`critical`, `major`, `minor`, `info`), location and text.

**Reporting groups by severity.** Summaries and reports group findings under severity headings: `### 🔴 Critical`, `### 🟠 Major`, `### 🟡 Minor`, `### 🔵 Info` (omit empty groups; each entry `file:line — one-line description`). Anchored comments serve findings needing attached discussion; the severity groups are the complete record. A finding may appear in both, but nothing you found is ever dropped silently: if it matters enough to notice, state it somewhere.

**Severity vs conclusions, your judgment, not a lookup table.** For reviews: critical and major findings usually mean changes requested; an uncertain, optional, or honestly arguable major may fit a comment instead; minor and info fit comment or approve by the same reasoning. Ideally everything is resolved before merge (fixed, justified, or explicitly waived); your verdict states where things stand against that ideal.

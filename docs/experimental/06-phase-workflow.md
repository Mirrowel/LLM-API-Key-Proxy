# Phase Workflow

This document describes how each phase should be executed.

## 1. Refresh Understanding

Before writing phase docs or code, inspect the current implementation relevant to the phase. The master plan is a guide, not a substitute for current-code analysis.

## 2. Produce Phase Plan In Conversation

First produce the phase plan in conversation text. This forces a fresh exhaustive design pass with the current implementation in mind.

The plan should include:

- goals.
- non-goals.
- files/modules to inspect or modify.
- data model changes.
- public API changes.
- transaction logging implications.
- docstrings, comments, and future-extension notes required for the phase.
- tests to add.
- commit checkpoints.
- risk/rollback notes.

## 3. Write Phase Plan To Docs

After the conversation plan is accepted or clearly settled, write it to:

```text
docs/experimental/phase-N-*.md
```

Planning docs are committed.

## 4. Implement In Checkpoint Commits

Commits are not phase-only. Commit whenever a coherent slice is finished and tested.

Each commit body should include:

- what changed.
- why it changed.
- tests run.
- known limitations or follow-ups.

## 5. Test Continuously

Run the most relevant tests after each meaningful slice. Do not wait until the phase end.

## 6. Review Agents

At phase end, call exactly these two review perspectives:

- `explore`: code/file-level verification.
- `explore-heavy`: deeper architecture/reference verification.

Review prompts should compare the implementation against:

- the phase plan.
- external reference areas where relevant.
- current proxy behavior that must be preserved.
- transaction logging expectations.
- tests.

If either agent fails or runs out of context, restart it with a narrower prompt.

## 7. Address Findings

Fix real findings. If a finding is intentionally deferred, document it in the user-facing phase report, not necessarily in git.

Repeat review if the changes are substantial.

## 8. Report To User

Write a phase report in the conversation. Reports are for the user and are not committed by default.

The report should include:

- completed work.
- commits made.
- tests run.
- review-agent findings and resolutions.
- known limitations.
- next phase recommendation.

## 9. Move To Next Phase

Do not rely blindly on previous plans. Start the next phase by refreshing context and producing the next phase plan in conversation text.

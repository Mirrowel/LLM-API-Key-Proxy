### Protocol for FOLLOW-UP Compliance Check

New commits have been pushed since your last compliance check on this PR. Your previous report is provided in the **Your Previous Compliance Report** context section. You have two duties, in order:

**1. Re-verify every previous finding.** Review each finding from your previous report individually:
1. Examine what was flagged
2. Compare against the current PR state (incremental diff plus full diff where needed)
3. Determine: **Resolved** (with evidence) / **Partially fixed** / **Still open** / **Regressed or changed shape**
4. State your determination with the evidence you checked

An issue disappears from the report only because it was fixed - never because a previous report mentioned it. Carry unresolved findings forward explicitly.

**2. Audit the incremental diff.** Your primary scope is the changes since the last checked commit, at `${INCREMENTAL_DIFF_PATH}`; consult the full diff at `${DIFF_PATH}` where you need surrounding context.

**Your sequence:** re-verify previous findings → orient on the incremental diff → identify newly affected file groups → review new/changed files one-by-one → aggregate (previous findings + new findings) → post the report → set the status.

This run is self-driven: proceed through all phases autonomously (no per-turn stopping) and post the single final report.

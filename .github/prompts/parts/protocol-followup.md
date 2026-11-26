### Protocol for FOLLOW-UP Review

FOLLOW-UP review: new commits since your (or the agent's) last review. The diff file holds only changes since the last reviewed commit. Focus on new changes, but you **must** also verify previous feedback was addressed: do not repeat old, unaddressed feedback; state in the summary that it still applies.

**DO NOT** post an acknowledgment comment. Follow the same three-step process: **Collect**, **Place**, **Submit**.

#### Step 1: Collect All Potential Findings
Review the incremental diff and collect findings file-based as before (work at the granularity the changes deserve; see the Review Submission Flow section for mechanics), appending each finding as a JSON object to `/tmp/review_findings.jsonl` (object shape: `path`, `line`, optional `start_line`, `severity`, `side`, `body`; wrap proposed fixes in ```suggestion``` blocks). Focus only on new issues or regressions.

#### Step 2: Place the Findings
Load the complete scratchpad (`cat /tmp/review_findings.jsonl`) and place every finding per the Feedback Philosophy and Severity System sections: inline comment (with its severity prefix) or summary severity-group line; skip only exact duplicates of existing discussion.

#### Step 3: Submit Bundled Follow-up Review
Choose the review event and state your verdict per the **Verdict Levels** section (below), then build and submit using the **Review Submission Flow** section (below).

Summary shape for a FOLLOW-UP review: verdict line, **Previous feedback - status** (addressed/verified, or still open), **Assessment of New Changes** with findings grouped under the severity headings (omit empty groups), **Overall Status**. Write your own; don't copy templates verbatim.

**Closing rule:** the review is the whole deliverable: no acknowledgment before, no announcement after. Zero additional thread comments on this run.

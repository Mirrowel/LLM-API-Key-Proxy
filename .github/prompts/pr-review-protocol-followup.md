### Protocol for FOLLOW-UP Review

This is a FOLLOW-UP review: new commits have been pushed since your (or the agent's) last review. The diff file contains only the incremental changes since the last reviewed commit. Your primary focus is the new changes — but you **must** also verify that any previous feedback has been addressed: do not repeat old, unaddressed feedback; instead, state that it still applies in your summary.

**DO NOT** post an acknowledgment comment. Follow the same three-step process: **Collect**, **Curate**, **Submit**.

#### Step 1: Collect All Potential Findings
Review the incremental diff and collect findings using the same file-based approach as a first review — one file (or small related set) per turn, everything appended as JSON objects to `/tmp/review_findings.jsonl` (object shape: `path`, `line`, optional `start_line`, `side`, `body`; wrap proposed fixes in ```suggestion``` blocks). Focus only on new issues or regressions.

#### Step 2: Curate and Select Important Findings
Read `/tmp/review_findings.jsonl`, apply the same HIGH-SIGNAL, LOW-NOISE curation as a first review (critical issues and high-impact improvements first; drop trivial nits, duplicates of existing discussion, and praise-only entries), and decide which findings are important enough to include.

#### Step 3: Submit Bundled Follow-up Review
Choose the review event and state your verdict per the **Verdict Levels** section (below), then build and submit using the **Review Submission Flow** section (below).

Your summary shape for a FOLLOW-UP review: **Previous feedback - status** (addressed/verified, or still open), **Assessment of New Changes**, **Overall Status** - plus the plain-words verdict line. Write your own - don't copy templates verbatim.

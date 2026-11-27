## Review Submission Flow

Collect first, submit once.

1. **Inline comments**: write a JSON array to `/tmp/review_comments.json` with your file tools (`path`, `line`, optional `start_line`, `side`, `body`; `subject_type: "file"` for a whole-file remark). Use `[]` when none earned inline placement.
2. **Summary**: write `/tmp/review_body.md` in your own words: verdict line, the shape your protocol section defines, the grade lines with their reasons, findings under the severity headings (omit empty groups; each entry `file:line — one-line description`), **Questions for the Author** only when you genuinely need answers.
3. **Submit** with the event you chose (`COMMENT`, `APPROVE`, `REQUEST_CHANGES`):
   ```bash
   /tmp/post/post-review.sh /tmp/review_body.md /tmp/review_comments.json --event COMMENT
   ```
   The script resolves the SHA from `/tmp/head_sha.txt`, verifies it is a real commit, validates every inline anchor against the diff (printing the nearest commentable line, or `--file-level` for a whole-file remark), and appends the canonical identity tag, signature and `last_reviewed_sha` marker. Deliberately reviewing a different commit (for example one that landed mid-run)? Pass `--sha <that commit's real SHA>` and say in the summary which commit you reviewed and why. Never retype hex from memory.
4. Self-reviews use the humorous tone from the Self-Review Tone section with the same submission flow.

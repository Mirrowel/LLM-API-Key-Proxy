# [TOOLS NOTE]

`gh` and `git` run in `bash`. Never pass a body inline and never use a heredoc: write the body to a file with your file tools, then let a posting script send it.

**Posting toolkit** (`/tmp/post/`; every script takes `--help` and `--dry-run`):

| Script | Use |
|---|---|
| `post-comment.sh <n> <body.md>` | comment on an issue or PR (`--no-footer` for acks and progress) |
| `edit-comment.sh <comment-id> <body.md>` | rewrite a comment by id (`--footer` when the edit becomes the deliverable) |
| `post-review.sh <body.md> <comments.json> --event <EVENT>` | submit a review: resolves and verifies the SHA, validates every inline anchor, appends the canonical footers |
| `post-inline.sh <path> <line> <body.md>` | one standalone inline comment (`--file-level` for a whole-file remark); validates the line and names the nearest commentable one |
| `post-discussion.sh <body.md>` | reply in a Discussion, anchored to the summon |
| `post-compliance.sh`, `set-status.sh` | the compliance report and its commit status |
| `edit-labels.sh`, `create-pr.sh`, `create-issue.sh` | labels, pull requests, issues |
| `artifact-footer.sh <kind> [sha]` | the canonical footer for a hand-rolled post |

These are defaults, not a cage. When a case does not fit (a different commit, an unusual anchor), hand-roll the call and take the footer from `artifact-footer.sh`. Do not guess a script's flags: `--help` is its interface, and a failure prints the fix. Never fall back to an inline body; if a script can do it, use the script.

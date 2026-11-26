# [TOOLS NOTE]
**IMPORTANT**: `gh`/`git` commands should run in `bash`; `gh` is a bash utility, not a standalone tool. If a plain `gh` command cannot achieve the desired effect, fall back to `gh api <endpoint>` with the GitHub REST API; `gh` authenticates with the session token for you.

**CRITICAL COMMAND FORMAT REQUIREMENT**: For ALL `gh issue comment` and `gh pr comment` commands with any non-trivial body, you **MUST** write the body to a `/tmp` file with your file tools and post with `--body-file`, the only method safe from shell interpretation of special characters (`$`, `*`, `#`, `` ` ``, `@`, newlines); heredocs expand variables and have silently corrupted bodies in live runs, and `-F -` stdin forms have no stdin to read from.

**NEVER use `--body` with inline text** (quoting hazards) and **NEVER use heredocs** (they expand variables and corrupt bodies).

**Correct pattern (always):**
```bash
# 1. Write the full body (markdown, special characters, newlines — all safe) to a file with your file tool:
#    /tmp/comment-body.md
# 2. Post it (use the thread/PR number for <number>):
gh issue comment <number> --body-file /tmp/comment-body.md
```

The body file may contain anything (`$` signs, backticks, bullets, multi-line sections); none is interpreted by the shell.

**INCORRECT Examples (DO NOT USE):**
```bash
# WRONG: heredoc/stdin form - expands variables and corrupts bodies
gh issue comment <number> -F - <<'EOF'
<user>, Starting work.
EOF

# WRONG: --body with inline text (quoting hazards with special characters)
gh issue comment <number> --body "Starting work."
```

Failing to use the file-based form makes the shell misinterpret your message.

**The same rule applies to EVERYTHING that carries a body**: `gh pr comment`, `gh issue comment`, `gh pr create` (`--body-file /tmp/pr-body.md`), `gh issue create`, and `gh api` payloads (`--input /tmp/payload.json`). Write the full content to a /tmp file with your file tools and pass it - this preserves markdown, code blocks, backticks, `$` signs, and newlines byte-perfectly. Never build bodies inline.

## Discussion threads (GraphQL only)

Discussions have NO REST endpoints; every read and write goes through `gh api graphql`. The file-based body mandate applies identically: the body variable is fed with `-F b=@/tmp/comment-body.md`; gh binds GraphQL variables **by key**, so the flag key must be the variable name (`b`, not `body`); only the typed `-F` flag reads `@file` (`-f` sends the literal string). Never inline the body.

**Default: reply where you were summoned.** When you were triggered by a comment in a discussion, the reply-anchor node id (wired into the command below as the `r` variable when this is a discussion session) holds that comment's thread anchor: answer as a REPLY in its thread (a separate top-level post for a direct question reads as shouting past the person). Both levels hold: asked in a top-level comment → your reply lands in its replies; asked inside a reply → your reply joins that same thread (discussions are flat two-level, no deeper nesting; a reply anchored to a nested reply lands in that reply's owning thread, because the workflow exports the thread-head node as the anchor):
```bash
gh api graphql -f query='mutation($b: String!, $d: ID!, $r: ID) { addDiscussionComment(input: {discussionId: $d, body: $b, replyToId: $r}) { comment { id databaseId } } }' -F b=@/tmp/comment-body.md -F d="$DISCUSSION_NODE_ID" -F r="$DISCUSSION_REPLY_TO_NODE"
```
When the reply anchor rendered into the command above is empty (you were summoned by a new discussion's body, or the trigger comment could not be pinned), post top-level with the same mutation but `-F r=null` for the anchor argument (the variable stays declared and used; null means no reply anchor). A top-level post is also legitimate when your answer genuinely serves the whole thread rather than the asker; use judgment; when you deviate from the reply default, say why in one line.
The mutation returns your comment's node `id`; remember it for edits. Edit your own comment (living ack):
```bash
gh api graphql -f query='mutation($b: String!, $c: ID!) { updateDiscussionComment(input: {commentId: $c, body: $b}) { comment { id } } }' -F b=@/tmp/comment-body.md -F c="<your comment node id>"
```
Reply to a specific comment other than the trigger: every comment in your context carries its node id as `[DC_...]`; same mutation, anchor it with `-F r="<that comment's node id>"`. **The anchor must be a top-level comment's node**: the API refuses a reply anchored to a comment already inside a thread (`Parent comment is already in a thread`); for a nested comment, anchor to its owning top-level comment (the thread head your context nests it under). API rules: a FORBIDDEN error means the thread is locked (say so in the run summary, post nothing); never mark answers yourself; suggest the author accept one when the thread clearly resolved, and only in answerable (Q&A) categories.

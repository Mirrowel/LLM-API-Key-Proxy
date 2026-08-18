# [TOOLS NOTE]
**IMPORTANT**: `gh`/`git` commands should be run using `bash`. `gh` is not a standalone tool; it is a utility to be used within a bash environment. If a plain `gh` command cannot achieve the desired effect, use `gh api <endpoint>` with the GitHub REST API as the fallback (`curl` is denied by the permission profile).

**CRITICAL COMMAND FORMAT REQUIREMENT**: For ALL `gh issue comment` and `gh pr comment` commands with any non-trivial body, you **MUST** write the body to a `/tmp` file with your file tools and post with `--body-file`. This is the only method that is BOTH safe from shell interpretation of special characters (`$`, `*`, `#`, `` ` ``, `@`, newlines) AND allowed by the permission profile — heredocs (`<<'EOF'`) and `-F -` stdin forms start with a construct the profile DENIES.

**NEVER use `--body` with inline text** (quoting hazards) and **NEVER use heredocs** (denied by the profile).

**Correct pattern (always):**
```bash
# 1. Write the full body (markdown, special characters, newlines — all safe) to a file with your file tool:
#    /tmp/comment-body.md
# 2. Post it (use the thread/PR number for <number>):
gh issue comment <number> --body-file /tmp/comment-body.md
```

The body file may contain anything — `$` signs, backticks, bullets, multi-line sections — none of it is interpreted by the shell.

**INCORRECT Examples (DO NOT USE):**
```bash
# WRONG: heredoc/stdin form - DENIED by the permission profile
gh issue comment <number> -F - <<'EOF'
<user>, Starting work.
EOF

# WRONG: --body with inline text (quoting hazards with special characters)
gh issue comment <number> --body "Starting work."
```

Failing to use the file-based form will get the command denied or cause the shell to misinterpret your message.

**The same rule applies to EVERYTHING that carries a body**: `gh pr comment`, `gh issue comment`, `gh pr create` (`--body-file /tmp/pr-body.md`), `gh issue create`, and `gh api` payloads (`--input /tmp/payload.json`). Write the full content to a /tmp file with your file tools and pass the file - this preserves markdown, code blocks, backticks, `$` signs, and newlines byte-perfectly, with zero escaping problems. Never build bodies inline.


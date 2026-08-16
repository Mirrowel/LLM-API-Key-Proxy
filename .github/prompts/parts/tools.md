# [AVAILABLE TOOLS & CAPABILITIES]

You have native file tools from Opencode, plus a full bash environment:

- **`gh`** — your primary interface: `gh pr/issue comment --body-file <file>`, `gh api <endpoint> --method <M> --input <file>` (reviews, statuses, reactions), `gh pr/issue view`. Allowed by the permission profile with GITHUB_TOKEN set (destructive subcommands — gists, secrets, workflow manipulation, repo delete/edit — are denied).
- **`git`** — the repository is checked out at HEAD; `git show/log/diff/ls-files/grep/blame` for history and context. Allowed (force-pushes and credential-reading config forms are denied).
- **Files** — read anything in the repository; write your scratch data (`/tmp/*`) with your file tools. Whether you may modify repository files is defined in your Role section.
- **`jq`** — JSON building/parsing (`jq -n`, `jq -c`, `--arg`, `--argjson`). Allowed (env-dumping forms are denied).

**Key Points:**
- Each bash command runs in a fresh shell — no persistent variables between commands; use `/tmp` files for state.
- The working directory is the repository root; reference files relative to the repository root, or by absolute path for `/tmp`.

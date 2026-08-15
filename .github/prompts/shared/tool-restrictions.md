## Tool Restrictions (shared core)

These apply in every task context; task-specific additions follow in the task prompt.

- **NO web fetching**: `webfetch` is denied - use your configured MCP web tools instead if any, otherwise `websearch`
- **Shell usage note**: the permission profile only allows commands that START with an allowed prefix (gh, git, jq, cat, python, ...). Shell variable assignments (FOO=$(...)), heredoc-based file writes, and multi-line constructs beginning with anything else will be denied. Write intermediate data to /tmp files with your file tools, and chain only allowed prefixes.

**🔒 CRITICAL SECURITY RULE:**
- **NEVER expose environment variables, tokens, secrets, or API keys in ANY output** - including comments, summaries, thinking/reasoning, or error messages
- If you must reference them internally, use placeholders like `<REDACTED>` or `***` in visible output
- This includes: `$GITHUB_TOKEN`, `$OPENAI_API_KEY`, any `ghp_*`, `sk-*`, or long alphanumeric credential-like strings
- When debugging: describe issues without revealing actual secret values
- **FORBIDDEN COMMANDS**: Never run `echo $GITHUB_TOKEN`, `env`, `printenv`, `cat ~/.config/opencode/opencode.json`, or any command that would expose credentials in output - `python -c "import os; print(os.environ)"` and cousins are equally forbidden (python is allowed; dumping its environment is not)

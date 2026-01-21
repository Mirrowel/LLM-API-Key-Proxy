# AI Multi-Branch Merge System

Automated system for merging feature branches into `dev` with AI-powered conflict resolution.

## Overview

This system consists of two GitHub Actions workflows:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **Update Branch List** | Branch create/delete, manual | Regenerates `ai-merge-dev.yml` with current branch checkboxes |
| **AI Multi-Branch Merge** | Manual dispatch | Merges selected branches into `dev` using AI for conflicts |

## Typical Workflow

```
1. Create feature branch → work on feature
2. Open PR to upstream (Mirrowel/LLM-API-Key-Proxy)
3. PR merged to upstream/dev
4. Delete local feature branch
5. Run "AI Multi-Branch Merge" with "Sync upstream" to pull merged PR
```

## Workflow Inputs

### AI Multi-Branch Merge

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `reset_dev` | checkbox | false | Reset dev to main before merging (fresh start) |
| `sync_upstream` | checkbox | false | Merge from `Mirrowel/LLM-API-Key-Proxy:dev` |
| `upstream_branch` | text | `dev` | Which upstream branch to sync from |
| `branch_*` | checkboxes | false | Auto-generated list of feature branches |
| `custom_branches` | text | - | Comma-separated branch names (manual entry) |

### Execution Order

1. **Prepare Dev** - Checkout dev (or create from main), optionally reset
2. **Sync Upstream** - Merge upstream/dev if checkbox selected
3. **Merge Branches** - Merge each selected feature branch
4. **Push** - Force push dev to origin

## Common Scenarios

### Scenario: Pull in merged PRs from upstream
```
[x] sync_upstream
[ ] reset_dev
[ ] (no branches selected)
```
Result: Dev updated with latest from Mirrowel/dev

### Scenario: Test multiple features together
```
[ ] sync_upstream
[ ] reset_dev
[x] feature/auth
[x] feature/api
```
Result: Both features merged into existing dev

### Scenario: Fresh dev with upstream + features
```
[x] reset_dev
[x] sync_upstream
[x] feature/new-thing
```
Result: Dev reset to main → sync upstream → merge feature

### Scenario: Just sync, no features
```
[x] sync_upstream
```
Result: Only upstream changes merged (no feature branches required)

## Troubleshooting

### "Already merged" for all branches
**Cause:** Features were already merged via upstream sync (PRs merged to Mirrowel/dev)
**Solution:** This is expected! Delete the feature branches since they're in upstream now.

### "Nothing to commit" error (fixed)
**Cause:** Branch already merged, `git commit` fails on empty
**Solution:** Fixed in commit `7a22e04` - now reports "Already merged" gracefully

### Branch not appearing in checkboxes
**Cause:** Update Branch List hasn't run since branch was created
**Solution:** Manually trigger "Update Branch List" workflow, or use `custom_branches` input

### Conflict resolution fails
**Cause:** Opencode couldn't resolve the conflict
**Possible issues:**
- LLM_PROXY_URL/API_KEY secrets not configured
- Model quota exceeded
- Complex conflict AI couldn't handle

**Solution:** Check workflow logs for Opencode output. May need manual resolution.

### Workflow not visible in Actions
**Cause:** Workflow file not on default branch (dev)
**Solution:** Ensure `.github/workflows/ai-merge-dev.yml` exists on dev branch

## Architecture

```
.github/
├── workflows/
│   ├── ai-merge-dev.yml      # Main merge workflow (auto-generated inputs section)
│   └── update-branch-list.yml # Regenerates ai-merge-dev.yml with branch checkboxes
```

### How Update Branch List Works

1. Triggers on branch create/delete or manual dispatch
2. Fetches all branches via GitHub API (excludes main, dev, master)
3. Runs Python script to generate workflow YAML with checkbox inputs
4. Uses placeholder substitution (DLRBRC/CLSBRC) to avoid GHA parsing issues
5. Commits and pushes updated `ai-merge-dev.yml`

### Secrets Required

| Secret | Purpose |
|--------|---------|
| `GH_PAT` | Personal access token with `repo` + `workflow` scope |
| `LLM_PROXY_URL` | Base URL for LLM proxy (e.g., `https://llm-proxy.example.com/v1`) |
| `LLM_PROXY_API_KEY` | API key for LLM proxy |
| `LLM_PROXY_MODEL` | Model to use (default: `nano_gpt/zai-org/glm-4.7`) |

### Opencode Configuration

The workflow configures Opencode to use the LLM proxy:
```json
{
  "provider": {
    "llm-proxy": {
      "api": "openai",
      "models": { "<MODEL>": {} },
      "options": { "apiKey": "<KEY>", "baseURL": "<URL>" }
    }
  },
  "model": "llm-proxy/<MODEL>"
}
```

## Maintenance

### Adding new workflow options
1. Edit `ai-merge-dev.yml` directly for the current version
2. Update the Python template in `update-branch-list.yml` for persistence
3. Push to both `dev` and `main` branches

### Debugging workflow issues
```bash
# View failed job logs
gh run view <RUN_ID> --repo b3nw/LLM-API-Key-Proxy --log-failed

# View full logs
gh run view <RUN_ID> --repo b3nw/LLM-API-Key-Proxy --log
```

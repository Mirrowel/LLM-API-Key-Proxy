# LLM-API-Key-Proxy Knowledge Base

**Last refreshed:** 2026-07-11
**Commit:** 12e0ccb
**Branch:** main

## CURRENT DIRECTION (read first)

New-API becomes the transport on `:8001`; the `workload-router/` sidecar is the
router brain in front of it; LiteLLM on `:8000` is the rollback path. Do not add
prompt classification or quota logic to this gateway, and do not build another
router. Layers, repo roles and the ordered work list (alias collision,
review findings #370-#389, `s0-cost-vector` merge, gradual cutover) are in
[`docs/ARCHITECTURE-DECISION-2026-11-22.md`](docs/ARCHITECTURE-DECISION-2026-11-22.md).

## OVERVIEW

FastAPI-based LLM API proxy with virtual models and automatic fallback. Routes requests through 33 free LLM providers (groq, gemini, cerebras, nvidia, mistral, openrouter, sambanova, together, cloudflare, bluesminds, kilo, freemodel, freemodel-anthropic, supacoder, wiwi, iflow, xinjianya, g4f_*, github-models, aihubmix, antigravity, brave_search, duckduckgo, exa, jina, tavily, cloudflare, zyf, etc.) with telemetry-driven fallback chains.

## STRUCTURE

```
LLM-API-Key-Proxy/
├── config/              # YAML configs (router, virtual models, aliases, scoring)
│   ├── router_config.yaml       # Provider enable/disable + keys (33 providers)
│   ├── virtual_models.yaml      # 17 virtual model fallback chains
│   ├── model_rankings.yaml       # Telemetry-derived model scores (~101K)
│   ├── providers_database.yaml   # Provider catalog (~64K)
│   ├── cooldown_policy.yaml      # Wait-vs-fallback per provider
│   ├── concurrency_policy.yaml    # Per-(provider,model) slot limits
│   ├── scoring_config.yaml       # U-formula weights
│   └── tier_config.yaml          # Model tier definitions
├── src/
│   ├── proxy_app/       # Main gateway (~16K lines)
│   │   ├── main.py              # FastAPI app, /v1/chat/completions (1698 lines)
│   │   ├── router_core.py       # Router + fallback logic (2514 lines)
│   │   ├── settings_tool.py     # Settings management (2450 lines)
│   │   ├── launcher_tui.py       # Terminal UI launcher (1003 lines)
│   │   └── provider_urls.py      # Provider URL construction (100 lines)
│   └── rotator_library/  # Core library (~34K lines, 33 provider adapters)
│       ├── client.py              # RotatingClient (3227 lines)
│       ├── credential_tool.py     # OAuth credential management (2273 lines)
│       ├── usage_manager.py        # Usage tracking (1812 lines)
│       ├── model_info_service.py   # Model metadata (1352 lines)
│       ├── telemetry.py            # LiteLLM telemetry logger (1257 lines)
│       ├── provider_status_tracker.py # SQLite health probes (984 lines)
│       ├── error_handler.py        # Error handling (974 lines)
│       ├── dynamic_chain.py        # Telemetry-driven re-ranker (387 lines)
│       ├── scoring_engine.py       # U-formula scoring (392 lines)
│       ├── distributed_gate.py      # Cross-process cooldown (382 lines)
│       ├── cost_efficiency.py       # Cost archetype routing (300 lines)
│       └── providers/               # 33 provider adapters (antigravity, gemini_cli, g4f, etc.)
├── scripts/             # Utility scripts
├── tests/               # Test fixtures
├── docs/                # Documentation
├── SPEC.md              # Project spec (379 lines)
└── .env                 # API keys (NOT committed)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Virtual models | `config/virtual_models.yaml` | 17 chains: coding-elite, coding-fast, chat-elite, auto, etc. |
| Provider routing | `src/proxy_app/router_core.py` (2514L) | Fallback logic, retry, force-try |
| Provider adapters | `src/rotator_library/providers/` | 33 adapter files |
| Router configuration | `config/router_config.yaml` | Provider enable/disable (33 providers) |
| Telemetry-driven re-rank | `src/rotator_library/dynamic_chain.py` | Uptime EMA + fail penalty + quality + cost + spread |
| Health probes | `src/rotator_library/provider_status_tracker.py` | SQLite-persisted provider health |
| Cooldown policy | `config/cooldown_policy.yaml` | Wait-vs-fallback (e.g. anthropic waits 300s on 429 for cache) |
| Concurrency gating | `config/concurrency_policy.yaml` + `distributed_gate.py` | Per-(provider,model) slots |
| Scoring config | `config/scoring_config.yaml` | U-formula weights |
| API key auth | `src/proxy_app/` | Search for `verify_api_key` |
| TUI launcher | `src/proxy_app/launcher_tui.py` | Interactive launcher |

## KEY CONCEPTS

### Virtual Models (17 total, chains are telemetry-reordered via `reorder_chains.py`)

Current chains (refreshed periodically by VPS cron):
- **coding-elite**: glm-5.2 -> gpt-5.5-mini -> qwen3.7-plus -> gemini-3.5-flash
- **coding-fast**: kimi-k2.6 -> poolside/laguna-xs.2:free -> nemotron-3-nano -> deepseek-v4-pro
- **chat-elite**: gemini-3.1-pro-preview -> gemini-3-pro-preview -> gemini-3-pro-high -> gemini-3-pro
- **chat-fast**: gemini-2.5-flash -> llama-3.3-70b -> llama3.1-8b -> llama-3.1-8b-instant
- **chat-smart**: gemini-3-flash -> kimi-k2 -> gemini-2.5-flash
- **chat-rp**: mistral-small-4-119b -> solar-10.7b -> mistral-7b -> mistral-nemotron
- **auto**: gemini-2.5-flash -> llama-3.3-70b -> llama-3.1-8b
- **glm5-elite**: glm-5.2 -> glm-5.1 -> glm-5
- **agent-***: bge-m3 -> kimi-k2 -> kimi-k2.6 -> minimax-m3 (6 agent variants: build/explore/librarian/metis/momus/oracle)

### Dynamic Chain Ranking (`dynamic_chain.py`)

Telemetry-driven re-ranker (#251, wired via `USE_DYNAMIC_CHAIN=1` env, default off):
- **uptime_ema**: success ratio, 30min halflife (weight 0.40)
- **quality**: benchmark score (weight 0.25)
- **fail_penalty**: linear decay 60s->0 (weight 0.15), full at <60s, zero at >=300s
- **cost_eff**: archetype (weight 0.10)
- **spread_bonus**: load distribution (weight 0.10)
- Reads `llm_events` table from `/dev/shm/telemetry.db` (tmpfs, wiped on reboot)
- **#354 fallback**: set `TELEMETRY_DB_FALLBACK` env to on-disk snapshot path so signals survive reboot

### Fallback Chain

Router automatically tries next provider if current one fails (rate limit, error, timeout). Configurable wait-vs-switch via `cooldown_policy.yaml` (e.g. anthropic waits 300s on 429 to preserve prompt-cache discount).

### FREE_ONLY_MODE

Gateway runs in free-only mode by default (`FREE_ONLY_MODE=true`). No paid API keys needed.

## CONVENTIONS

### Python tooling
- Prefer `uv` for Python environments, dependency installation, and command execution when available (for example, `uv run pytest` or `uv pip install`).
- Fall back to `python -m pip` only when `uv` is unavailable or would add unnecessary setup; document the fallback in the session summary.

### Python
- **Formatting:** Black, 88-char line limit
- **Type hints:** Required for function signatures
- **Async:** Heavy use of async/await + httpx
- **Imports:** `from pathlib import Path` (not `os.path`)
- **Logging:** `logging.getLogger(__name__)`

### System packages
- Prefer `nala` over `apt-get` for Debian/Ubuntu package operations when `nala` is available.
- Use `apt-get` only when `nala` is unavailable or cannot perform the required operation, and document that fallback.

### FastAPI
- **Dependencies:** Use `Depends()` for injection (get_rotating_client, verify_api_key)
- **Exceptions:** `HTTPException(status_code=..., detail=...)`
- **Request parsing:** `await request.json()` for body

### Configuration
- **YAML** for configs (router_config.yaml, virtual_models.yaml, scoring_config.yaml, etc.)
- **Env vars** for secrets (.env via python-dotenv)
- **SQLite** for telemetry (`/dev/shm/telemetry.db`) + health (`provider_status.db`)

## ANTI-PATTERNS

- DO NOT modify `.env` - credentials handled via credential tool
- DO NOT hardcode API keys - use `os.getenv()` or credential tool
- DO NOT make synchronous HTTP calls - use async httpx
- DO NOT disable rate limit checks - respect provider limits
- DO NOT use `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk` - use dedicated tools

## COMMANDS

```bash
# Development (local)
python src/proxy_app/main.py --host 0.0.0.0 --port 8000

# With request logging
python src/proxy_app/main.py --host 0.0.0.0 --port 8000 --enable-request-logging

# OAuth credential tool
python src/proxy_app/main.py --add-credential

# Terminal UI launcher (interactive)
python src/proxy_app/main.py

# Self-tests (stdlib-only, no frameworks)
python3 src/rotator_library/dynamic_chain.py    # -> 'dynamic_chain self-test: OK'
python3 src/rotator_library/distributed_gate.py  # -> 'distributed_gate self-test: OK'
```

## TESTING

```bash
# Test gateway running
curl http://<VPS40_IP>:8000/v1/models \
  -H "Authorization: Bearer $VPS_GATEWAY_API_KEY"

# Test virtual model
curl -X POST http://<VPS40_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VPS_GATEWAY_API_KEY" \
  -d '{"model": "coding-elite", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}'
```

## GIT WORKFLOW

**Remote:** https://github.com/ons96/LLM-API-Key-Proxy
**VPS auto-pull:** VPS-40 pulls on push (verify with SSH)

```bash
# On VPS-40
ssh -i ~/.ssh/oracle.key ubuntu@<VPS40_IP>
cd ~/LLM-API-Key-Proxy
git pull

# Restart gateway (systemd-managed on VPS-40)
sudo systemctl restart llm-gateway
# (NOT pkill -f main.py; gateway is under systemd with MemoryMax=400M)
```

## ENV VARS

### Gateway control
- `GATEWAY_RETRY_UNTIL_DONE` (default false): retry whole chain on hard failure with backoff 2/4/8s, max 3
- `GATEWAY_FORCE_TRY_COOLDOWN` (default false): ignore cooldown state, try all candidates
- `FREE_ONLY_MODE` (default true): reject non-free providers
- `USE_DYNAMIC_CHAIN` (default false): enable telemetry-driven re-ranking (#251)

### Telemetry
- `TELEMETRY_DB_PATH` (default `/dev/shm/telemetry.db`): tmpfs telemetry store
- `TELEMETRY_DB_FALLBACK` (default empty): on-disk snapshot fallback so signals survive reboot (#354)

## GOTCHAS

- **VPS restart:** Gateway is systemd-managed (`sudo systemctl restart llm-gateway`), NOT nohup+pkill. MemoryMax=400M on VPS-40.
- **Telemetry tmpfs:** `/dev/shm/telemetry.db` is wiped on machine reboot. Set `TELEMETRY_DB_FALLBACK` to survive (#354).
- **G4F models:** Some complex IDs don't work (stick to simple names like `g4f/gpt-4`)
- **Virtual chains are dynamic:** `reorder_chains.py` rewrites `virtual_models.yaml` from telemetry every 30min on VPS-40. Don't hand-edit chains expecting persistence.
- **Dynamic chain is opt-in:** `USE_DYNAMIC_CHAIN=1` must be set or the static chain from `virtual_models.yaml` is used as-is.
- **Cooldown policy is per-provider:** `cooldown_policy.yaml` decides wait-vs-switch (e.g. anthropic/openai wait on 429 for cache; default switches immediately).

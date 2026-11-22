# AI Workload Router

This repository records the resource-light router work for the Oracle VPS at
`<VPS40_IP>`. The current release is running as a side-by-side,
loopback-only canary; it has not replaced the existing gateway.

## Status: parallel and gated, not production-wired

Target architecture (2026-11-22): this sidecar is the router brain in front of
New-API, which becomes the transport; LiteLLM stays as the rollback path.
Layering, repo roles and the ordered work list are in
`../docs/ARCHITECTURE-DECISION-2026-11-22.md`. Work on this directory, not on a
new router.

This directory is a parallel build owned by the canonical repo. It is not in
the live request path:

- The gateway routes no traffic through it. The single gateway-side contact
  point is `src/proxy_app/workload_router_shadow.py`, an observe-only dry run
  that is **off unless `WORKLOAD_ROUTER_DRY_RUN=1`**. When off it performs no
  I/O; when on it appends `tier`/`deployment`/`why` per request to a bounded,
  prompt-free JSONL log and discards the result, so live routing is identical
  with the flag on or off.
- `config/virtual_models.yaml`, the existing fallback chains, and client
  behavior are not modified by anything in this directory.
- Promotion past dry run follows the staged, feature-flagged plan in
  `../docs/workload-router-cutover-plan.md`. No stage advances until its gate
  passes, and every stage has a one-flag rollback.

Design: `docs/workload-router-architecture.md`. Session checkpoints: `STATE.md`.

## New-API sidecar local verification

The separate `newapi_server.py` entrypoint is opt-in and requires a strict
management audit plus authorized channel-key proofs. It does not change the
legacy gateway or the native New-API v1 prototype. See
`docs/newapi-v1-v2-roadmap.md` for the staged boundaries and
`docs/channel-proof-helper.md` for the operator contract.

From this directory, run the provider-free suite and disposable integration:

```sh
python3 -m pytest tests -o addopts= --quiet
python3 tools/verify_newapi.py --binary /home/osees/new-api/bin/new-api --artifacts /tmp/opencode
```

The verifier requires Linux `unshare` and `ip`, an unmodified pinned binary,
and an existing artifact parent. It creates a private network namespace and
fresh databases with scripted local providers. It checks the actual entrypoint's
missing/invalid-proof rejection, TOTP-authorized boot, chat/coding/SSE, graceful
stop and durable restart, in addition to quota failover and catalog/benchmark
checks. Reports and generated test credentials remain in owner-only artifacts;
inspect public report fields, not logs or seeds. This is not live-provider or
VPS deployment evidence. Add `--opencode "$(command -v opencode)"` to the verifier
for the installed-client boot and scripted read tool-loop gate; that flag uses
separate HOME/XDG state and no real provider quota. Operator preview/apply,
last-good catalogs and priority-only updates are also verified. See
`docs/operator-commands.md` for the exact scope and remaining live gates.

Real-provider checks are separate and explicit in
`tools/verify_newapi_live.py`: catalog-only by default, with bounded chat/SSE,
one read-only OpenCode loop, deterministic coding/tool cases (`--workload`) or
injected-429 failover after `--execute`. Existing keys stay in memory in the
host process, outside the disposable gateway namespace. The workload mode is
capped at ten requests and 5120 output tokens, keeps tools intact, fails closed
on unverified case capabilities and reports transport and deterministic
correctness separately. It waits once for a short exhausted per-minute window
(bounded at 65 seconds) before reporting a case unavailable. See
`docs/live-provider-verification.md` for input format, budgets, actual results
and the remaining production gates. A later bounded continuation validated
Mistral chat/SSE and injected-429 failover to real responses in both
Mistral/OpenRouter directions. A 2026-11-04 live one-target workload batch
reached the workload path after fixing a child-wiring bug; at a 256-token cap
it graded 2/2 attempted cases but hit the cap and skipped the rest. After
raising the cap to 512 and adding the bounded wait, all five cases ran and
graded 5/5 correct with actual usage and no skips (`passed:true`). This is one
live free route and one batch, not canary or production evidence. Canary and
VPS gates remain.

The gateway changes currently deployed are:

- runtime enforcement of the mapping-based dead-provider policy;
- deployment-aware fallback ranking using concrete model telemetry;
- removal of an unused stream-timeout request field that was stripped before
  reaching provider adapters.

The production implementation remains in the VPS repository
`/home/ubuntu/LLM-API-Key-Proxy`. This repository contains portable review and
validation artifacts for the router project.

## Parallel and gated, not production-wired

This component runs **parallel to the gateway and is gated**: it is not wired
into the live request path, and it never changes `virtual_models.yaml` chains
or gateway fallback behavior. Two controls keep it that way:

- `ROUTER_CUTOVER_ENABLED` - env-var feature flag that would apply router
  decisions to live routing. **Defaults OFF.** Any run without an explicit
  truthy value (`1`, `true`, `yes`, `on`) stays observe-only.
- Dry-run logging - every routed request emits one JSON line to stderr with
  `mode`, `tier`, `deployment`, and `why`, so decisions are reviewable without
  affecting routing:

  ```text
  {"deployment": "...", "mode": "dry_run", "tier": 1, "why": "chain_top", ...}
  ```

  Disable with `ROUTER_DRY_RUN_LOG=0` (logging is on by default because it is
  read-only). The `GET /v1/router/route` response also reports
  `cutover_enabled`.

Cutover plan: keep the flag OFF in all environments, validate decisions via
dry-run logs and `trace_replay.py` shadow replay against a fixed baseline,
then enable `ROUTER_CUTOVER_ENABLED=1` in a later, separately reviewed change
that wires the router into the gateway path.

The latest VPS verification confirmed that the systemd service and socket are
active, the unauthenticated endpoint responds with HTTP 401, and the gateway
commits are `cf7a7fe`, `04a1986`, and `e6b86fa`.

The authenticated model-list probe still needs a follow-up because it exceeded
the ten-second request budget while startup health checks were active.

Run the local validation with:

```text
python3 tools/validate_session.py
```

Run the complete local test suite with:

```text
python3 -m pytest -q
```

Start the local server with:

```text
python3 router_server.py
```

## Local P0 prototype

The current VPS canary runs the failure-hardening release `failures-20261008`, served
by `ai-workload-router-canary.service` on `127.0.0.1:8123`. Ordinary chat uses
the existing gateway's `chat-smart` model. Tool and code phases use the existing
gateway's verified `coding-smart` or `coding-fast` virtual models; the old
gateway remains active on port 8000. See `docs/DEPLOYMENT.md` for tested checks
and rollback procedure.

The standalone deterministic core is in `router_core.py`. It can be exercised
without provider credentials through the route-inspection server:

```text
python3 router_server.py
```

Send a JSON request to `POST http://127.0.0.1:8080/v1/router/route`:

```json
{"messages": [{"role": "user", "content": "inspect the files"}], "tools": true}
```

The response identifies the selected deployment and capability. This prototype
does not call an upstream provider yet.

Operational or semantic results can be reported separately with
`POST /v1/router/outcome`, for example:

```json
{"deployment": "tool", "outcome": "operational_failure", "error_class": "timeout"}
```

Only `operational_failure` activates the deployment cooldown.

The local boundary rejects request bodies larger than 1 MiB before JSON
parsing, keeping the default process memory behavior bounded.

Deployment profiles are in `config/deployments.json`. The current server keeps
the safe built-in catalog by default; configuration loading is validated and
falls back to those defaults on missing or malformed files. Credentials and
provider URLs are intentionally absent from this catalog.

Each deployment also carries a non-secret provider label and configuration
fingerprint. These identify model/provider/configuration in telemetry without
storing credentials. For a host that already runs `llm-provider-manager`, set
`ROUTER_PROVIDER_DB` to its SQLite metadata path to load ordered virtual-model
fallback groups at startup. Optional `ROUTER_PROVIDER_GROUPS` limits the groups
and `ROUTER_FREE_ONLY=1` keeps the generated catalog to free/no-key metadata.
The loader reads provider/model identity, endpoint metadata, capabilities,
context limits, and chain priority only; it never reads API-key values. An
explicit `ROUTER_DEPLOYMENTS` catalog takes precedence. Use
`GET /v1/router/groups` to inspect the resulting groups and chain order.

Quota class and remaining quota are soft scoring inputs. An exhausted quota is
penalized, but quota metadata never removes every candidate by itself. The
sidecar distinguishes `free_unlimited`, `free_limited`, `free_credits` and
`paid`; one-time credit routes use a separate notional credit tariff so their
customer price stays zero while the grant depletes. The provider classifications
in `llm_providers.db` are the read-only source; `newapi_operator.py classify`
previews route mismatches without rewriting them. See
`docs/free-tier-classes.md`.

`stall_detector.py` tracks repeated bounded action identifiers and failed
verification signals without retaining raw tool history. It supports one-way
escalation up to a ceiling and resets after meaningful progress.

Offline shadow replay is available through `trace_replay.py`. It consumes
JSONL metadata such as `examples/sample-traces.jsonl`, makes no provider calls,
and reports selected deployments for comparison with a fixed baseline.
`ReplayResult.as_dict()` or `write_report()` can persist aggregate metrics such
as routed count, deployment distribution, and baseline match rate without
writing prompts.

Run the provider-free VPS resource smoke test with:

```bash
python3 tools/resource_smoke.py
```

It starts the server on localhost, checks health and routing, and reports the
child process RSS. The target VPS budget is 1 GB; this is a startup check, not
a load benchmark. `run_router.sh` is the deployment-safe entrypoint and the
server supports `ROUTER_HOST` and `ROUTER_PORT`.

Set `ROUTER_STATE_DB` to a writable SQLite path for durable decisions, session
preferences, outcomes, and cache observations. The default `:memory:` mode is
intended for tests and ephemeral local inspection.

New requests begin at the configured chain priority and walk candidates in
priority order. Explicit continuations can remain on an eligible preferred
deployment. Recent provider cache observations can move a lower-priority
candidate first only when estimated cache savings exceed the chain-position
penalty. A short rate-limit retry is waited out only when a fresh warm cache
makes that wait cheaper than the cold fallback; timeout and upstream failures
fail over immediately.

`provider_adapter.py` defines the transport boundary for real providers. It
forwards the complete OpenAI-compatible message/tool/structured-output request
while replacing only the client alias with the selected deployment model and
removing router-only telemetry fields. Its executor retries only eligible
deployments in the same capability group and marks operational failures for
cooldown; it does not perform semantic retries.

Provider HTTP failures are reduced to bounded telemetry classes such as
`timeout`, `rate_limit`, `upstream_failure`, `provider_rejection`, and
`network_error`. Failed attempts are recorded without persisting upstream error
text, while later same-capability attempts can still succeed. Unexpected adapter
exceptions are normalized into the same bounded classes, including failures
while creating or iterating a stream. A malformed native SSE event closes the
stream without being reported as a successful completion.

The provider-fault regression harness also sends an OpenCode-shaped request with
69 tools, a large system message, stream usage options, and a 32,000-token
output limit. It verifies that message history, tools, and standard options are
forwarded intact without contacting a live provider.

The chat boundary validates `response_format` objects and supports OpenAI-style
SSE streaming. Adapters exposing `stream()` pass provider SSE payloads through
without buffering; completion-only adapters use a compatibility stream after
the provider completion returns.

Supported client model aliases are `auto`, `auto-code`, and `auto-chat`; an
unknown alias is rejected rather than silently ignored.

`POST /v1/chat/completions` returns an OpenAI-compatible response when adapters
are installed by the embedding process. The standalone server intentionally
installs no provider adapters, so it returns a bounded 503 rather than making
an unconfigured upstream call.

Health endpoints are available at `GET /healthz` and
`GET /v1/router/diagnostics`. They expose only service status and aggregate
counts; they never return prompt text.
Diagnostics also report aggregate successful routes, operational/semantic
failures, and cache hits.

`GET /v1/models` exposes the stable client aliases `auto`, `auto-chat`, and
`auto-code` without revealing provider credentials or upstream endpoint data.
JSON request endpoints reject non-object bodies with a bounded 400 response.

Successful route responses include `X-Router-Deployment` and
`X-Router-Capability` headers. They also include group and route-reason metadata
when available. JSON responses are marked `no-store` and include content-type
hardening headers. Provider-unavailable responses expose only a bounded error
class and optional retry-after value.

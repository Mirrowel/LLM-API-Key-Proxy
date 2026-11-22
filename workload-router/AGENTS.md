# AI Workload Router

## Session startup

Read `STATE.md`, `HANDOFF.md`, `README.md`, and the P0 specification at
`docs/specs/p0-router.md` before changing files. The layering decision (New-API
is transport, this sidecar is the router brain) and the ordered work list are in
`../docs/ARCHITECTURE-DECISION-2026-11-22.md`; read that before choosing what to
work on, and amend it in place rather than writing a new design document. The
full architecture plan is
at `/home/osees/CodingProjects/temp-omnirush/docs/ai-workload-router-architecture.md`.

Preserve existing work. Inspect `git status` before edits. Work in small,
verified milestones and update `STATE.md` after each milestone. Do not print,
copy, rotate, or commit credentials. Keep implementation free-only and suitable
for a 1 GB VPS. Run the narrowest relevant validation first, then broader checks
when justified by the change.

The current repository contains portable artifacts for an implementation that
also exists in `/home/ubuntu/LLM-API-Key-Proxy` on the VPS. Do not modify the VPS
repository from this local session unless explicitly requested.

Deployment target: Oracle VPS 40 at `<VPS40_IP>`, with 1 GB RAM, zram,
weak CPU, and no GPU. Keep code and dependencies suitable for that host.
Local tests must remain provider-free; deployment requires a separate explicit
step.

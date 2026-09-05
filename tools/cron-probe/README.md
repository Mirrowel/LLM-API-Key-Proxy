# cron-probe - the cron canary

Minimal worker with a */2 cron and a one-line scheduled handler. Deployed on
the account since 2026-09-04 as the CANARY for the cron-trigger outage:
the platform registered schedules but never dispatched them (zero attempts
in workersInvocationsAdaptive; community reports 922869/928936; docs admit
crons run on underutilized machines). The main mention-worker moved to a
Durable Object alarm loop and no longer depends on crons. When this probe
starts firing (check via wrangler tail or its dashboard metrics), the
platform side has healed - feel free to delete it.

Deploy: cd tools/cron-probe && npx wrangler deploy

# Retired Providers

This folder archives provider implementations that are intentionally no longer
registered, discovered, or exposed by the proxy.

Archived modules are kept for historical reference only. Do not import them from
active code or add them back to provider factory/discovery wiring unless the
provider is explicitly un-retired.

Retired OAuth-backed Google/CLI providers and their helper modules are archived
here together so active startup paths do not import OAuth credential flows. The
API-key Gemini provider remains active outside this folder.

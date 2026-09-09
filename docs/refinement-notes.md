# Codebase Refinement Notes — Next Cleanup Pass

Inventory of duplication and size-reduction opportunities in `src/proxy_app/` (+ `credential_tool.py`), gathered 2026-09-10 after the key-policy work. Ordered by impact; file:line references are current as of that state. Goal: reduce duplication and total size **without a complete redo** — splits follow existing banner-comment boundaries, behavior preserved.

## Line counts

| File | Lines | File | Lines |
|---|---|---|---|
| model_filter_gui.py | 3639 | credential_tool.py | 2219 |
| settings_tool.py | 2437 | quota_viewer.py | 1921 |
| main.py | 1824 | launcher_tui.py | 1155 |
| quota_viewer_config.py | 345 | route_helpers.py | 273 |
| detailed_logger.py | 208 | key_policy.py | 186 |
| startup.py | 146 | build.py | 97 |
| batch_manager.py | 84 | provider_urls.py | 76 |
| request_logger.py | 34 | startup_display.py | 18 |

## Top-5 ROI actions

1. **Shared exception→`format_client_protocol_error` mapper in main.py** — the `except` ladder (InvalidRequest/ValueError → Auth → RateLimit → ServiceUnavailable → Timeout → generic) is copied ~3× at `main.py:710-776`, `:1062-1111`, `:1316-1362`, differing only in `input_protocol`; plus 5 copies of the invalid-JSON error block (`:637-644`, `:800-807`, `:1000-1017`, `:1129-1138`, `:1178-1184`). One mapper cuts ~150 lines.
2. **Dead code deletion** (~110 lines, zero behavior change — all verified): `launcher_tui.py:188-216` (OAuth scan over a hardcoded-empty dict — loop can never execute), `main.py:186-247` (four Pydantic model classes referenced nowhere), `detailed_logger.py:207-208` (alias, zero importers), `settings_tool.py:621` (unreachable after `return`), `request_logger.py:4-9` (dead imports), `main.py:456-459` (commented-out block), `rotator_library/model_info_service_old.py` (stale module).
3. **Collapse model_filter_gui ignore/whitelist twins** — ~200 lines of mirrored methods differing only by rule type: `_add_ignore_pattern`/`_add_whitelist_pattern` (`:3272-3326`), `_remove_*` (`:3328-3338`), `_clear_all_*` (`:3340-3358`), `FilterEngine.add_ignore_rule`/`add_whitelist_rule` (`:231-285`); plus 4× modal-dialog boilerplate (`HelpWindow`, `UnsavedChangesDialog`, `ImportRulesDialog`, `ImportResultDialog`).
4. **One home for env-file/oauth-dir resolution + dotenv-based parsing** — `.env` path logic exists in 5+ divergent copies (`launcher_tui.py:24-35`, `key_policy.py:45-50`, `credential_tool.py:38-40`, `main.py:77-80`, `quota_viewer_config.py` ×3 internally); `model_filter_gui.py:501` uses bare `Path.cwd()/".env"` (latent frozen-EXE bug). Hand-rolled .env parsers ×4 (`launcher_tui.py:117-137`, `quota_viewer_config.py:288-300`, `credential_tool.py:134-167` + `:320-341`, `settings_tool.py:629-645`). Provider-credential discovery scan ×4 (`main.py:383-389`, `launcher_tui.py:163-186`, `model_filter_gui.py:578-597`, `settings_tool.py:623-656`) with oauth-dir drift (hardcoded `Path("oauth_creds")` vs `get_oauth_dir()`).
5. **Split the two giant TUIs at their banner comments** — model_filter_gui.py (filter_engine / fetching / dialogs / virtual_lists / app) and settings_tool.py (env_changes logic vs SettingsTool UI, then along its four `manage_*` clusters ~900 lines of the same "combined view + pending changes" widget — `:716-902`, `:904-1080`, `:1642-1892`, `:2057-2303`).

## Remaining inventory

- **Between files:** custom-API-base detection ×2 (`launcher_tui.py:226-238`, `settings_tool.py:170-181`); model-definition counting ×2 (`launcher_tui.py:241-256`, `settings_tool.py:215-229`); concurrency-key splitting ×2 (`launcher_tui.py:259-265`, `settings_tool.py:333-340`); `clear_screen()` ×3 full copies + 4 raw calls (`launcher_tui.py:38-57`, `settings_tool.py:34-53`, `credential_tool.py:1045-1066`); provider-name registries ×3 overlapping (`provider_urls.py:8-33`, `provider_config.py` KNOWN_PROVIDERS, `litellm_providers.py`); `PROXY_API_KEY` write-back bypassing key_policy (`credential_tool.py:1086`).
- **Within files:** main.py logging setup built twice (`:281-293` dead, `:321-333` live); responses route triplet (`:878-935`); quota_viewer connection-format ×3 (`:709-715`, `:761-767`, `:1574-1580`), HTTP error ladder ×2 (`:472-497`, `:653-686`), summary-recalc ×2 (`:539-581`, `:584-622`); launcher config display ×2 (`:444-460`, `:585-597`), sub-tool launch scaffolding ×3 (`:949-986`, `:988-1010`, `:1107-1115`); model_filter_gui scroll-sync closures ×4 (`:1893-1954`), tooltip logic ×2, virtual-list scaffolding ×2.
- **Observable dupes:** startup banner printed twice per launch (`main.py:120-124` + `:262-269` after `cls`; ready line computed twice); `.env` loaded twice per run (`main.py:83`, `:367`); `run_proxy` reloads dotenv back-to-back (`launcher_tui.py:1113`, `:1115`); credential_tool double screen-clear on one path (`:1045` + `:2183`).
- **Stylistic drift:** new-style typed modules (key_policy/startup/route_helpers/startup_display) vs old inline-everything TUIs; dotenv `set_key` and hand-parsing mixed inside settings_tool.
- **Split candidates (behavior-preserving):** main.py — auth deps (`:533-606`), lifespan (`:419-517`), logging config (`:274-364`), env-scan config (`:369-415`); quota_viewer — formatting helpers (`:84-323`) / remote-management screens (`:1544-1858`); launcher_tui — `SettingsDetector` (`:113-352`) as a standalone detection module; credential_tool — env I/O helpers (`:119-357`) beside key_policy/utils.paths, export/combine sub-feature (`:1733-1985`).

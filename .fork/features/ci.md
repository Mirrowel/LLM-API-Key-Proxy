## 2026-06-21 — Fix release job failing when short SHA length differs between runners

Target: `feat(ci): fork-aware release notes with incremental topic diff` (`ea5f239`)

Files:
- `.github/workflows/build.yml`

Working commit before autosquash:
- TBD — created via `fixup! feat(ci): ...`

Final stack commit after autosquash:
- TBD — folded into `feat(ci): ...`

### Why

Run 27859339250 / job 82452676947 failed in **Generate Build Metadata**
with `find: 'release-assets': No such file or directory`.

Root cause: `git rev-parse --short HEAD` returns the minimum length
needed for SHA uniqueness in the local object DB — and that length is
not deterministic across runners. For run 27859339250 the build jobs
uploaded artifacts named `proxy-app-build-{Linux,macOS,Windows}-afec625`
(7 chars) while the release job filtered with
`proxy-app-build-*-afec6255` (8 chars). Zero artifacts matched, the
download step exited 0 anyway, and the next bash step (set `-e -o pipefail`)
crashed on the missing directory.

### Fix

1. Pin both `Get short SHA` steps (build job and release job) to
   `git rev-parse --short=7 HEAD` so they always agree.
2. Add a defensive `Verify downloaded artifacts` step right after the
   download that fails with a clear error and lists the available
   artifacts when the download silently matched zero items.

### Verification

- `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/build.yml'))"` — OK
- 7-char SHA matches the length already in use for artifact names, so
  no re-upload of historical artifacts is required.
- Recommended: re-run the failed workflow after the fix is folded into
  the `feat(ci)` stack commit and pushed.

### Notes / risks

- A fully-orthogonal future fix is to pin everything to the full
  40-char SHA — that decouples the artifact name from git's notion of
  "short" entirely.
- Another option is to drop `pattern:` on `download-artifact@v4` and
  filter by an explicit list (artifact IDs or full names) — `pattern:`
  glob matching across multi-runner SHA lengths is a recurring foot-gun.

# Legacy release builder

`build_release_legacy.py` is the original one-off script that assembled an early public folder from a private parent workspace. It is preserved only for audit provenance and exits immediately if invoked.

Do not use it to build or update this repository: its embedded documentation, filenames and source-tree assumptions are historical. Use the maintained commands instead:

```powershell
python scripts\run_release_checks.py
```

The current workflow lives in `src/cso_spatial_drivers/`, is configured through `config/demo.yaml` or a local `config/scientific.yaml`, and is documented in `docs/FULL_PIPELINE_RUNBOOK.md`.

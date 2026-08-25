# 08 – Final synthesis

**Question:** what conclusions survive the full chain of data quality, linkage uncertainty, inference and locked-test validation?

```powershell
python scripts\07_build_final_figures.py --config config\demo.yaml
```

The maintained command completes all core stages and writes `<output_root>/06_report/RUN_SUMMARY.md` plus `run_manifest.json`. The manifest is the machine-readable record of input hashes and generated products.

Authoritative scientific figures are in `figures/` and `outputs/headline_figures/`; the reports are in `docs/`. Conclusions must retain the difference between association and causality, beta and lambda, original and regional estimands, and static context versus missing hydraulic/event-state information.

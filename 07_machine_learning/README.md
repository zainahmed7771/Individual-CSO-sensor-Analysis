# 07 – Machine learning

**Question:** how much do external predictors improve locked-test performance over a target-specific mean-only Dummy model?

## Executable teaching/scientific core

```powershell
python scripts\06_run_machine_learning.py --config config\demo.yaml
```

Outputs are written under `<output_root>/05_machine_learning/`: four model files, a scorecard and locked-test predictions. Preprocessing is fitted on development rows; locked-test rows are untouched until final evaluation. Demonstration metrics are not scientific findings.

## Preserved scientific approaches

1. `original_selected_sensor/` represents one outcome-blind selected sensor per WWTW.
2. `regional_cluster/` represents company-bounded distance clusters and changes the estimand.

The two approaches must be compared through target-normalised metrics such as percentage RMSE improvement over each target's Dummy baseline, not through raw RMSE across different transformations.

Saved authoritative results and reports remain in `results/`, `models/`, `figures/` and `docs/`. The historical scripts are retained for provenance; the maintained core runner is the supported entry point for new configurations.

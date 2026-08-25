# 06 – Univariate analysis

**Question:** what association does each external variable show with each outcome when considered separately?

```powershell
python scripts\05_run_univariate_analysis.py --config config\demo.yaml
```

Output: `<output_root>/04_univariate/univariate_results.csv`.

The maintained runner fits positive outcomes on the log scale using HC3 robust uncertainty and applies Benjamini–Hochberg FDR correction within outcome. A slope is an unadjusted association, not a causal effect and not automatically a multivariable feature-selection decision.

The preserved scientific tables and figures are under `results/` and `outputs/headline_*`. Broad/strict cohort rules and meaningful predictor increments are documented in `docs/METHODS.md` and `docs/VARIABLE_DICTIONARY.md`.

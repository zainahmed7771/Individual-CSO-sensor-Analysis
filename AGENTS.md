# Agent instructions

- Preserve the scientific definitions in this release.
- The tail condition is strictly `duration_minutes > 240`, never `>= 240`.
- The model is `S(t)=exp[-(lambda*t)^beta]`; lambda is an inverse timescale.
- Lower beta means a heavier/more persistent fitted tail.
- Do not place identifiers, coordinates, target-quality fields, outcomes or event-derived fields in ML predictors.
- Do not allow WWTWs/regions to cross train, validation and test splits.
- Never silently edit raw inputs or claim inferred WWTW linkage is proven hydraulic connectivity.
- Original selected-sensor ML and regional-cluster ML are distinct analyses and must be compared using normalized metrics, not raw RMSE alone.
- Run `pytest -q` and `python scripts/run_reproducible_demo.py` after code changes.
- Never add raw data, tokens, correspondence, absolute local paths or unverified licensed sources.

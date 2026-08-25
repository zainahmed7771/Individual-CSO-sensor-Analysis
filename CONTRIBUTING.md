# Contributing

Create a branch from `main`, keep raw and machine-local data outside Git, and make one scientifically coherent change per pull request.

Before committing:

```powershell
python scripts\check_inputs.py --config config\demo.yaml
python scripts\run_pipeline.py --config config\demo.yaml
pytest -q
python scripts\run_release_checks.py
```

Do not change the strict `>240` rule, beta/lambda interpretation, accepted linkage evidence, target eligibility or data splits without an explicit methods change and corresponding tests. Never add identifiers, coordinates, outcome-derived fields, confidence intervals, quality flags or split labels to model predictors.

Pull requests should state the scientific reason, affected inputs/outputs, validation performed and whether any headline result changes.

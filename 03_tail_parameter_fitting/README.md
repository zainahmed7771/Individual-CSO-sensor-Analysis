# 03 – Long-duration tail fitting

**Question:** how persistent is each sensor's distribution of events longer than four hours?

The fitted conditional survival model is `S(t)=exp[-(lambda*t)^beta]`. Tail membership is strictly `duration_minutes > 240`; exactly 240 minutes is excluded. Lower beta means a heavier tail. Lambda is an inverse-timescale per minute.

## Maintained core

```powershell
python scripts\02_fit_tail_parameters.py --config config\demo.yaml
```

Output: `<output_root>/02_tail_fitting/sensor_outcomes.csv`.

## Historical national fit and bootstrap

```powershell
python 03_tail_parameter_fitting\scripts\analyse_individual_cso_heavy_tails.py --help
python 03_tail_parameter_fitting\scripts\bootstrap_all_individual_cso_sensors.py --help
```

The full copy-paste commands are in `docs/FULL_PIPELINE_RUNBOOK.md`. Bootstrap intervals measure resampling precision; they are not proof of model adequacy.

Validation:

```powershell
pytest tests\test_tail_threshold.py tests\test_beta_lambda_fitting.py -q
```

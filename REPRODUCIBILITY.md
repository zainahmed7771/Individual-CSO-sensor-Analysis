# Reproducibility statement

## What is reproducible from the public clone?

The complete teaching workflow is executable from included synthetic inputs:

```bash
python -m pip install -e '.[test]'
python scripts/check_inputs.py --config config/demo.yaml
python scripts/run_pipeline.py --config config/demo.yaml
pytest -q
```

It executes the same portable contracts used by the scientific core: event cleaning, strict `>240` tail fitting, accepted linkage, exposure-adjusted frequency, predictor joining, univariate analysis, four locked-test models and a hash-bearing run manifest.

## What is required for the scientific run?

Copy `config/scientific.example.yaml` to the ignored `config/scientific.yaml`, supply the four authorised inputs, and run the same commands with that configuration. The core pipeline never embeds machine-specific paths.

Raw provider harmonisation and environmental GIS extraction remain separate because their source data are large, provider-specific or externally licensed. Their exact hand-offs are documented in `docs/FULL_PIPELINE_RUNBOOK.md` and the numbered stage READMEs.

## Reproducibility levels

| Level | Included? | Claim |
|---|---|---|
| Code and environment | Yes | Installable Python project with declared optional GIS dependencies |
| Complete teaching run | Yes | Executes locally and in CI from supplied synthetic data |
| Aggregate scientific evidence | Yes | Headline tables, figures, predictions, models and reports are preserved |
| Scientific core rerun | Conditional | Executes after the four documented authorised inputs are supplied |
| Raw-source reconstruction | Conditional | Requires provider/GIS source access and documented human spatial QC |

The repository never substitutes synthetic values for missing scientific inputs and never treats absent restricted data as a successful scientific reproduction.

# Student learning path

This guide turns the repository into a short self-directed course. Complete the sections in order.

## Session 1: understand the scientific question

Read `README.md`, `docs/ANALYSIS_STORY.md` and `docs/LIMITATIONS.md`. Be able to explain why static catchment characteristics may contain signal but cannot represent event rainfall, storage, pumping or operating state.

## Session 2: run before reading implementation details

Follow `QUICKSTART.md`. Open `outputs/demo_pipeline/run_manifest.json` and trace each input hash to its stage output. Confirm that an event of exactly 240 minutes is excluded by running:

```powershell
pytest tests\test_tail_threshold.py -q
```

## Session 3: event cleaning

Read `src/cso_spatial_drivers/cleaning/events.py` and `data/schemas/event_schema.csv`. Follow one synthetic row from raw timestamps to `duration_minutes` and `sensor_uid`. Then run:

```powershell
python scripts\01_build_event_master.py --config config\demo.yaml
```

## Session 4: beta and lambda

Read `03_tail_parameter_fitting/equations.md` and `src/cso_spatial_drivers/fitting/stretched_exponential.py`. Explain why lower beta means a heavier tail and why lambda cannot be described as duration. Run the fitting tests.

## Session 5: linkage as evidence, not hydraulic truth

Read `04_spatial_linkage/matching_rules.md`. Inspect `data/sample/synthetic_assignments.csv`. Notice that each assignment has an evidence field and is joined before the outcomes are used analytically.

## Session 6: predictor joins and missingness

Read `src/cso_spatial_drivers/predictors/join.py` and `docs/VARIABLE_DICTIONARY.md`. Deliberately duplicate one predictor key in a temporary file and observe the pipeline fail. Restore the input afterwards.

## Session 7: inference versus prediction

Compare `04_univariate/univariate_results.csv` with `05_machine_learning/model_scorecard.csv`. Univariate slopes describe one-variable associations; locked-test improvement asks whether the predictor set generalises beyond a mean-only baseline.

## Session 8: original versus regional units

Read `07_machine_learning/README.md` and both ML reports. Explain the change in estimand and why reduced target variance can make regional prediction appear easier without proving a better site-level model.

## Session 9: reproduce authorised results

Only after completing the teaching workflow, create `config/scientific.yaml`, run the input checks and follow `docs/FULL_PIPELINE_RUNBOOK.md`. Record the Git commit and preserve the run manifest with the outputs.

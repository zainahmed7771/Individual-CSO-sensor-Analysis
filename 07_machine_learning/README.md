# 07 Machine Learning

## Scientific question

How much do external catchment and operational predictors improve out-of-sample prediction over a target-specific mean-only Dummy model, and does changing the analysis unit reduce outcome noise?

## Two retained pathways

### A. Original selected-sensor / one-WWTW analysis

This is the primary submitted pathway. Each WWTW is represented by one sensor selected through the beta-blind matching workflow. Four separate regressions predict beta, inverse-timescale lambda, mean spill duration and annualised spill frequency. Target-quality rules produce different eligible cohorts.

### B. Regional company x 25 km cluster analysis

This later comparison groups WWTWs within company using target-independent complete-linkage distance clusters. It tests whether one selected overflow is a noisy proxy for the wider wastewater system. Exact assigned events are pooled for regional duration and exposure-adjusted frequency; beta/lambda and predictors use documented scientific aggregation rules. Ten and 50 km sensitivity analyses are retained.

The regional pathway is not a silent replacement for the original analysis: it changes the estimand from a selected site/WWTW to a company-bounded system area.

## Inputs

- Original: selected-sensor scientific master; 1,231 unique WWTWs before target-specific eligibility.
- Regional: 1,444 mapped WWTWs in 259 primary 25 km regions; exact aggregation validated from 2,045,832 cleaned assigned events.
- Predictors: hydrogeology, rainfall, land cover, population, building age, terrain, capacity and catchment scale, with target/leakage fields excluded.

The full masters are not redistributed. The release contains saved models, locked-test predictions, aggregate scorecards, configurations and a synthetic workflow.

## Code to run

- Original implementation: `scripts/original_selected_sensor/ml_programme.py`.
- Original cross-model figures: `scripts/original_selected_sensor/build_cross_model_figures.py`.
- Regional implementation order: `scripts/regional_cluster/RUN_ORDER.txt`.
- Portable small example: `../scripts/run_reproducible_demo.py`.

Both programmes use approximately 70% training, 15% validation and 15% locked test. Dummy, OLS, Ridge and Elastic Net are compared as documented. Preprocessing is fitted within training folds, model selection uses validation/CV evidence, and the locked test is reserved for final evaluation. RMSE on the model scale is the headline metric.

## Locked-test scorecard

| Outcome | Original RMSE gain | Original R2 | Regional RMSE gain | Regional R2 |
|---|---:|---:|---:|---:|
| Beta | 4.85% | 0.073 | 17.39% | 0.314 |
| Lambda | 4.94% | 0.095 | 9.72% | 0.180 |
| Mean duration | 2.64% | 0.052 | 7.78% | 0.143 |
| Annualised frequency | 6.99% | 0.133 | -1.99% | -0.063 |

Positive gain means lower RMSE than that target's own Dummy baseline. Raw RMSE values must not be compared across differently transformed outcomes.

## Outputs

- `results/original_selected_sensor/` and `models/original_selected_sensor/` preserve the submitted site-level pathway.
- `results/regional_cluster/` and `models/regional_cluster/` preserve the later comparison.
- `figures/` keeps final prediction, learning-curve, importance and radius-sensitivity diagnostics for both pathways.
- `../outputs/headline_tables/final_ml_scorecard.csv` is the normalized cross-pathway scorecard.
- Dedicated reports are `../docs/original_selected_sensor_ml_report.pdf` and `../docs/regional_cluster_ml_report.pdf`.

## QC and validation

- no target, fitted outcome, site identifier or split label enters the predictor matrix;
- preprocessing is fit in training folds;
- train/validation/test group overlap is zero;
- regional clusters never cross company boundaries and never use the outcome;
- saved locked-test RMSE values are recomputed by `../tests/test_saved_metrics_recompute.py`;
- Dummy baselines, predicted-versus-observed plots, learning curves and feature importance are retained;
- positive regional improvements have bootstrap intervals crossing zero and are described as promising, not definitive.

## Connection to the next stage

Both pathways feed `../08_final_synthesis/`. The correct conclusion is that regional aggregation substantially improves beta prediction and moderately helps lambda/duration, but does not improve frequency. Static external context remains insufficient for accurate operational prediction.

# 03 Tail Parameter Fitting

## Scientific question
How are long-duration persistence parameters estimated?

## Inputs
Clean sensor events and the strict 240-minute threshold.

## Code to run
Conditional stretched-exponential MLE and nonparametric sensor bootstrap.

## Outputs
Beta, lambda, intervals, fit status and tail support.

## QC and validation
Positivity/bounds, tail count, bootstrap success, boundary-hit and CI checks.

## Connection to the next stage
Validated sensor outcomes feed WWTW linkage.

The public demo exercises the portable core. Full-scale scripts require omitted source data described in `../DATA_AVAILABILITY.md`.

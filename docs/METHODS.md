# Methods

## Event contract
One row per unique company, permit, start and stop timestamp. Duration is `(stop-start)/60` minutes; invalid chronology is rejected.

## Tail model
Only `T > 240` enters the conditional stretched-exponential fit `S(t)=exp[-(lambda*t)^beta]`. Point fits require at least 10 tail events and bootstrap inference at least 20. Beta bounds are `[0.01,2]`; log-lambda bounds are `[-50,20]`.

## Spatial context
Assignments are company-consistent and beta-blind. Polygon containment is preferred; place/name/distance/manual evidence is audited. Catchments are contextual proxies, not proven hydraulic networks.

## Statistics and prediction
Univariate log-outcome models use HC3 and FDR. Original and regional ML use separate target cohorts, 70/15/15 splits, training-only preprocessing, validation selection, one locked-test evaluation and Dummy-relative RMSE. Coordinates, identifiers, other outcomes, target-quality and event-derived leakage fields are excluded from X.

# Reproducibility

## Supported public workflow
```bash
python -m pip install -e .[test]
pytest -q
python scripts/run_reproducible_demo.py
```
This exercises event validation, duration calculation, unique sensor identity, strict `T > 240` filtering, stretched-exponential fitting, predictor joins, one univariate model, and a small locked-test regression.

## Full-data workflow
The authoritative development scripts are preserved under the numbered stages. They are historical implementations with repository-relative contracts, not a promise that restricted inputs can be downloaded automatically.

1. Harmonise provider files (`01_data_acquisition/`, `02_data_cleaning/`).
2. Run `03_tail_parameter_fitting/scripts/analyse_individual_cso_heavy_tails.py` and bootstrap scripts.
3. Run spatial-linkage scripts in their documented order.
4. Build environmental predictors and the scientific master.
5. Run the univariate pipeline.
6. Run `07_machine_learning/scripts/original_selected_sensor/ml_programme.py`.
7. Run regional scripts in the order recorded in `07_machine_learning/scripts/regional_cluster/RUN_ORDER.txt`.

QGIS/manual inspection remains part of spatial QC. Raw data, licences and machine paths must be supplied locally through `config/paths.yaml`. No released script silently edits raw inputs.

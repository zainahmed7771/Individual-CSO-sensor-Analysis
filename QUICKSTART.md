# Quickstart: obtain a complete successful run

All commands below must be executed from the repository root—the folder containing `pyproject.toml`.

## Windows PowerShell

```powershell
git clone https://github.com/zainahmed7771/Individual-CSO-sensor-Analysis.git
cd Individual-CSO-sensor-Analysis
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python scripts\check_inputs.py --config config\demo.yaml
python scripts\run_pipeline.py --config config\demo.yaml
pytest -q
```

## macOS/Linux

```bash
git clone https://github.com/zainahmed7771/Individual-CSO-sensor-Analysis.git
cd Individual-CSO-sensor-Analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python scripts/check_inputs.py --config config/demo.yaml
python scripts/run_pipeline.py --config config/demo.yaml
pytest -q
```

Successful completion prints `PIPELINE COMPLETED`. Inspect:

```text
outputs/demo_pipeline/
|-- 01_cleaning/clean_events.csv
|-- 02_tail_fitting/sensor_outcomes.csv
|-- 03_master/scientific_master.csv
|-- 04_univariate/univariate_results.csv
|-- 05_machine_learning/model_scorecard.csv
|-- 05_machine_learning/locked_test_predictions.csv
|-- 06_report/RUN_SUMMARY.md
`-- run_manifest.json
```

These are synthetic teaching outputs. For authorised scientific inputs, continue with [`docs/FULL_PIPELINE_RUNBOOK.md`](docs/FULL_PIPELINE_RUNBOOK.md).

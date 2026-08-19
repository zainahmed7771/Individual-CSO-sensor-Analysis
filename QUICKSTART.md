# Quickstart

## 1. Clone and enter the repository
```bash
git clone <YOUR-REPOSITORY-URL>
cd GITHUB_RELEASE_CSO_SPATIAL_DRIVERS
```

## 2. Create an isolated environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[test]
```

## 3. Create local configuration
Copy `config/paths.example.yaml` to `config/paths.yaml` and replace placeholders only if running the full-data workflow. `config/paths.yaml` is ignored by Git.

## 4. Verify the release
```bash
pytest -q
python scripts/run_reproducible_demo.py
```

Demo outputs are written to `outputs/demo/` and are labelled **DEMONSTRATION ONLY - NOT THE SCIENTIFIC RESULTS**.

## 5. Inspect scientific results
- `outputs/headline_tables/final_ml_scorecard.csv`
- `docs/original_selected_sensor_ml_report.pdf`
- `docs/regional_cluster_ml_report.pdf`
- `docs/GITHUB_REPOSITORY_WALKTHROUGH.pdf`

Full-data commands are documented in `REPRODUCIBILITY.md`; they require sources that are intentionally not distributed.

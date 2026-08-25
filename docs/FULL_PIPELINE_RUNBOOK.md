# Full pipeline runbook

This is the authoritative operational guide. Run commands from the repository root unless a section explicitly says otherwise.

## 1. Choose the reproduction level

| Level | Configuration | Use it when |
|---|---|---|
| Teaching demonstration | `config/demo.yaml` | Learning, CI, installation checks and code development |
| Scientific core | local copy of `config/scientific.example.yaml` | Standardised events, accepted assignments, exposure and predictors are available |
| Raw-source reconstruction | `config/paths.yaml` plus numbered stage scripts | Rebuilding provider/GIS inputs before the scientific core |

The teaching and scientific-core runs use the same `src/cso_spatial_drivers` functions. The raw-source scripts are retained because provider formats and GIS products require source-specific processing and human QC. Before a raw-source reconstruction, complete `docs/DATA_ACQUISITION_CHECKLIST.md` and the provenance manifest described there.

## 2. Install the environment

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[full,test]"
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[full,test]'
```

`[full]` adds GeoPandas, Shapely, Rasterio, Pyogrio, OpenPyXL, Requests and report dependencies. QGIS is optional for manual cartographic inspection and is not called by the Python runner.

For the closest reconstruction of the validated Python 3.12 environment, install the tested direct pins first and then the local package without resolving them again:

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

The lock file records tested direct versions; pip still selects platform-appropriate transitive wheels.

## 3. Verify the clone before adding data

```powershell
python scripts\check_inputs.py --config config\demo.yaml
python scripts\run_pipeline.py --config config\demo.yaml
pytest -q
python scripts\run_release_checks.py
```

Expected result: 45 or more tests pass, the teaching pipeline completes, and no release validation row is `FAIL`.

## 4. Prepare the scientific-core configuration

```powershell
Copy-Item config\scientific.example.yaml config\scientific.yaml
```

Edit `config/scientific.yaml`. Do not edit Python constants or commit machine-local paths. Absolute paths are allowed; relative paths are resolved from the repository root.

### Required input contracts

#### Events

One CSV, or a directory of CSVs, with:

| Column | Rule |
|---|---|
| `company` | normalisable company identifier |
| `permit_number` | stable within company |
| `location_name` | descriptive label |
| `start_time` | parseable timestamp |
| `stop_time` | parseable timestamp not earlier than start |

The pipeline recomputes `duration_minutes`; a supplied duration cannot override timestamp arithmetic.

#### Accepted assignments

One CSV with unique `sensor_uid` and columns `company`, `uwwCode`, and `assignment_evidence`. These rows are the frozen, outcome-blind result of spatial/name/manual QC. The core runner does not claim that proximity proves hydraulic connectivity.

#### Monitoring exposure

One CSV with unique `sensor_uid` and finite positive `monitoring_years`. Missing exposure is a failure because annualised frequency must not be approximated.

#### Predictors

One CSV with unique `company + uwwCode`. Every name under `predictors:` in the configuration must exist. Identifiers, coordinates, outcomes, confidence intervals, target-quality flags, split labels and event-derived fields are forbidden predictors.

## 5. Check inputs without creating analysis outputs

```powershell
python scripts\check_inputs.py --config config\scientific.yaml
python scripts\03_link_sensors_to_wwtw.py --config config\scientific.yaml
```

Do not continue until both checks print `PASSED`.

## 6. Run the scientific core

### One command

```powershell
python scripts\run_pipeline.py --config config\scientific.yaml
```

If the configured output directory already contains a completed run, the command stops. Review that directory, close any open output files, and add `--force` only when you intend to replace the pipeline's declared generated files.

### Learning mode: one stage at a time

```powershell
python scripts\01_build_event_master.py --config config\scientific.yaml --force
python scripts\02_fit_tail_parameters.py --config config\scientific.yaml --force
python scripts\03_link_sensors_to_wwtw.py --config config\scientific.yaml
python scripts\04_build_predictor_master.py --config config\scientific.yaml --force
python scripts\05_run_univariate_analysis.py --config config\scientific.yaml --force
python scripts\06_run_machine_learning.py --config config\scientific.yaml --force
python scripts\07_build_final_figures.py --config config\scientific.yaml --force
```

Each stage regenerates the required preceding core stages. This costs extra time but prevents a student from accidentally combining incompatible intermediate versions. The final `run_manifest.json` records input and output hashes.

## 7. Raw-source reconstruction

The following steps precede the scientific-core input contracts. Store raw sources outside Git or in ignored directories. Use a separate immutable source folder and write derived outputs elsewhere.

### 7.1 Provider acquisition and standardisation

Consult `01_data_acquisition/README.md`, the source tracker workbook and provider access terms. Thames JSON preparation is explicit:

```powershell
python 01_data_acquisition\scripts\prepare_thames_company_data.py events --project-root "D:\path\to\analysis_workspace"
```

The other provider files must be mapped to `data/schemas/event_schema.csv`. Never overwrite an original export.

### 7.2 Long-duration fitting and bootstrap

```powershell
python 03_tail_parameter_fitting\scripts\analyse_individual_cso_heavy_tails.py `
  --data-dir "D:\path\to\clean_data" `
  --output-root "D:\path\to\outputs\individual_cso_heavy_tail_analysis" `
  --threshold 240 `
  --companies "anglian,northumbria,severn_trent,southern,southwest,united_utilities,wessex,yorkshire,thames" `
  --expected-company-count 9

python 03_tail_parameter_fitting\scripts\bootstrap_all_individual_cso_sensors.py `
  --data-dir "D:\path\to\clean_data" `
  --output-root "D:\path\to\outputs\all_individual_cso_sensor_bootstrap" `
  --threshold 240 `
  --n-bootstrap 1000 `
  --seed 20260819 `
  --companies "anglian,northumbria,severn_trent,southern,southwest,united_utilities,wessex,yorkshire,thames"
```

Use `--overwrite` only after checking the resolved output directory. The threshold must remain 240 and the code uses strict greater-than membership.

### 7.3 Sensor–WWTW linkage

These scripts expect the historical analysis-workspace layout named by `--project-root`:

```powershell
python 02_data_cleaning\scripts\prepare_sensor_master.py --project-root "D:\path\to\analysis_workspace"
python 04_spatial_linkage\scripts\assign_sensors_to_catchments.py --project-root "D:\path\to\analysis_workspace" --all-companies
python 04_spatial_linkage\scripts\rebuild_name_matching.py --project-root "D:\path\to\analysis_workspace"
python 04_spatial_linkage\scripts\select_primary_sensor.py --project-root "D:\path\to\analysis_workspace"
python 04_spatial_linkage\scripts\build_final_manual_release.py --project-root "D:\path\to\analysis_workspace" --manual-matches "D:\path\to\matching.txt"
```

Containment, name similarity and distance are evidence channels. Manual decisions must identify their evidence and remain outcome-blind. Freeze the resulting accepted assignment table before joining beta, capacity or environmental fields.

### 7.4 Environmental predictors

Install `[full]`, confirm all datasets use or can be transformed to EPSG:27700, and follow `05_environmental_predictors/README.md`.

```powershell
python 05_environmental_predictors\scripts\extract_hydrogeology.py --project-root "D:\path\to\analysis_workspace"
python 05_environmental_predictors\scripts\build_static_rainfall.py --project-root "D:\path\to\analysis_workspace"
python 05_environmental_predictors\scripts\build_wwtw_capacity.py --project-root "D:\path\to\analysis_workspace"
python 05_environmental_predictors\scripts\build_sensor_environment_master.py --project-root "D:\path\to\analysis_workspace"
```

Static rainfall uses the supplied HadUK-Grid annual and winter climatologies with exact partial-cell area weighting. Wet-day fields remain absent unless official precomputed threshold climatologies are supplied. Do not infer wet-day frequency from rainfall totals.

### 7.5 Hand off to the executable core

Export the four standard contracts, point `config/scientific.yaml` to them, run `check_inputs.py`, then run the core. This boundary prevents historical source-layout assumptions from leaking into the maintained analytical package.

## 8. Validate and archive a run

For every scientific run, retain together:

- the ignored configuration with sensitive paths redacted in any shared copy;
- `run_manifest.json`;
- all stage outputs;
- the exact Git commit hash (`git rev-parse HEAD`);
- environment information (`python --version` and `python -m pip freeze`);
- manual-match evidence and its checksum;
- the release-validation table.

Never copy a newer spreadsheet into an older run folder. Regenerate or create a new run directory.

## 9. Troubleshooting

- **`INPUT CHECK: FAILED`** – fix the listed path or contract; do not create placeholder scientific values.
- **PowerShell blocks activation** – use `Set-ExecutionPolicy -Scope Process Bypass`, then activate the environment. This changes only the current PowerShell process.
- **GIS import error** – install `.[full,test]`, preferably in a clean Python 3.12 environment.
- **Duplicate keys** – inspect upstream selection/join logic; do not drop duplicates arbitrarily.
- **Locked spreadsheet** – close Excel and rerun. Do not create a competing authoritative master.
- **QGIS shows unexpected gaps** – check CRS, company filter, catchment coverage and assignment audit before changing thresholds.

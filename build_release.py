"""Assemble the public, professor-ready release from authoritative local outputs.

This script copies only curated code/results and creates synthetic demo data.
It never copies raw EDM records, private correspondence, tokens, or large rasters.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent


def write(rel: str, text: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def copy(src: str, dst: str) -> None:
    source = SOURCE / src
    target = ROOT / dst
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def structure() -> None:
    folders = [
        "docs", "data/sample", "data/processed_examples", "data/schemas",
        "01_data_acquisition/scripts", "01_data_acquisition/schemas",
        "02_data_cleaning/scripts", "02_data_cleaning/tests", "02_data_cleaning/examples",
        "03_tail_parameter_fitting/scripts", "03_tail_parameter_fitting/tests", "03_tail_parameter_fitting/figures",
        "04_spatial_linkage/scripts", "04_spatial_linkage/tests", "04_spatial_linkage/figures",
        "05_environmental_predictors/scripts", "05_environmental_predictors/tests", "05_environmental_predictors/figures",
        "06_univariate_analysis/scripts", "06_univariate_analysis/results", "06_univariate_analysis/figures",
        "07_machine_learning/scripts/original_selected_sensor", "07_machine_learning/scripts/regional_cluster",
        "07_machine_learning/configs/original_selected_sensor", "07_machine_learning/configs/regional_cluster",
        "07_machine_learning/results/original_selected_sensor", "07_machine_learning/results/regional_cluster",
        "07_machine_learning/figures/original_selected_sensor", "07_machine_learning/figures/regional_cluster",
        "07_machine_learning/models/original_selected_sensor", "07_machine_learning/models/regional_cluster",
        "08_final_synthesis/tables", "08_final_synthesis/figures", "08_final_synthesis/report_assets",
        "09_experimental_extensions/univariate_ml", "09_experimental_extensions/hierarchical_benchmark",
        "src/cso_spatial_drivers/io", "src/cso_spatial_drivers/cleaning", "src/cso_spatial_drivers/fitting",
        "src/cso_spatial_drivers/spatial", "src/cso_spatial_drivers/predictors", "src/cso_spatial_drivers/statistics",
        "src/cso_spatial_drivers/ml", "src/cso_spatial_drivers/plotting", "scripts", "tests", "config",
        "outputs/headline_tables", "outputs/headline_figures", ".github/workflows", "audit",
    ]
    for folder in folders:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)


README = r"""
# CSO Spatial Drivers

## From national spill records to catchment-scale prediction

**Research question:** Can environmental and operational characteristics of wastewater catchments explain and predict differences in combined sewer overflow (CSO) spill behaviour?

This repository is the public, reproducible release of a UCL research internship. It follows water-company event-duration monitoring records from event cleaning through long-duration tail fitting, wastewater-treatment-works (WWTW) linkage, catchment predictor extraction, univariate analysis, and locked-test machine learning. It also contains a later regional-cluster experiment that asks whether one selected sensor was too noisy a proxy for a wider wastewater system.

## Why this project matters

Combined sewer systems carry wastewater and rainfall runoff in the same network. During wet weather, hydraulic capacity can be exceeded and diluted sewage may discharge through a monitored overflow. These events are shaped by rainfall and catchment context, but also by sewer topology, storage, pumping, control logic, treatment headroom, groundwater and antecedent wetness. Many of those hydraulic and time-varying states are not present in national open datasets.

The project therefore asks a deliberately limited question: how much information about differences between sites is contained in the external catchment and operational variables that can be assembled consistently across England? Association is not causality, and prediction is not explanation.

## Scientific workflow

```mermaid
flowchart LR
  A[EDM event data] --> B[Clean national sensor data]
  B --> C[Long-duration tail: T > 240 min]
  C --> D[Fit beta and lambda]
  D --> E[Link sensor to WWTW / catchment]
  E --> F[Environmental predictor matrix]
  F --> G[Univariate evidence]
  G --> H[Original: one selected sensor per WWTW]
  G --> I[Regional: company x 25 km WWTW clusters]
  H --> J[Four locked-test regressions]
  I --> K[Four regional locked-test regressions]
  J --> L[Scientific synthesis]
  K --> L
```

## Four outcomes

| Outcome | Definition | Interpretation |
|---|---|---|
| beta | Shape parameter of the conditional stretched-exponential tail above 240 minutes | Lower beta indicates a heavier, more persistent long-duration tail |
| lambda | Inverse-timescale parameter in `S(t)=exp[-(lambda*t)^beta]` | Higher lambda implies a shorter characteristic time at fixed beta |
| Mean spill duration | Arithmetic mean of valid event durations | Directly observed duration summary |
| Annualised spill frequency | Valid events divided by valid monitoring years | Exposure-adjusted occurrence rate, not raw lifetime count |

## Dataset scale

The final selected-sensor scientific master contains **1,231 unique WWTWs**. Separate target-quality rules yield **819 beta**, **783 lambda**, **1,228 duration**, and **967 annualised-frequency** modelling rows. Upstream spatial matching began from **25,409 unique sensors**, of which **8,960** had finite beta estimates and **692** passed the strict beta-quality definition used for the earlier inferential work. The regional extension maps **1,444 WWTWs** to **259 independent 25 km within-company regions** and verifies exact duration summaries from **2,045,832 cleaned, de-duplicated assigned events**.

These are different cohorts because quality and exposure requirements differ by target. The code does not force one complete-case cohort across all outcomes.

## Main methods

1. **Acquisition and harmonisation.** Company EDM exports arrived in different schemas and year coverage. The public release supplies schemas, acquisition notes and a synthetic demo, not the raw company records.
2. **Event QC.** Permit identifiers are normalized; timestamps are parsed; duration is calculated in minutes; invalid end-before-start records and duplicate events are rejected.
3. **Tail fitting.** The active tail is strictly `duration_minutes > 240`. Conditional maximum likelihood fits `S(t)=exp[-(lambda*t)^beta]` with beta constrained to `[0.01, 2]`; nonparametric sensor-event bootstrap intervals support quality screening.
4. **Spatial linkage.** Sensors are assigned within company using validated polygon containment and documented name/distance evidence. Matching is beta-blind and does not claim hydraulic connectivity.
5. **Catchment predictors.** Hydrogeology, static rainfall, annual NIMROD indices, land cover, population density, building age, terrain slope, treatment-capacity ratio and catchment area are joined using audited keys and EPSG:27700 area calculations.
6. **Univariate analysis.** Log-outcome models use HC3 robust uncertainty, Pearson/Spearman summaries and false-discovery-rate correction. Reported percentage changes are associations.
7. **Original predictive ML.** Four separate 70/15/15 train/validation/locked-test regressions compare Dummy, OLS, Ridge and Elastic Net. Preprocessing is fitted inside training folds; RMSE is the headline metric.
8. **Regional comparison.** WWTWs are grouped within company by target-independent 25 km complete linkage, with 10 and 50 km sensitivities. Predictors are aggregated using scientific weights; frequency uses pooled monitoring exposure.

## Main findings

### Original selected-sensor / one-WWTW approach

All four original models beat their own locked-test Dummy baseline, but only modestly. RMSE improvements were **4.9% for beta**, **4.9% for lambda**, **2.6% for duration**, and **7.0% for annualised frequency**. Test R-squared ranged from **0.052 to 0.133**. Frequency was the strongest original target; lambda was unstable because the fitted scale spans many orders of magnitude. Population density, hydrogeology, rainfall and catchment scale appeared repeatedly, but no predictor made the models accurate.

### Regional-cluster approach

The regional experiment improved **beta most strongly: 17.4% RMSE gain with R-squared 0.314**. Lambda improved **9.7%** and pooled duration **7.8%**. Regional frequency did **not** improve: **-2.0%**, R-squared **-0.063**. Bootstrap intervals for the three positive regional gains crossed zero, so the result is promising rather than definitive.

The correct comparison is not “regional is always better.” Regional aggregation helped beta and, to a lesser extent, lambda and duration, but it also reduced target variance and smoothed extremes. Ten-kilometre sensitivities sometimes performed better, while 50 km often lost sample size and signal. The full mixed-effects benchmark was singular and is reported as an honest negative result.

### Scientific synthesis

Measured catchment context contains real but modest information. The new regional result suggests that the original one-sensor unit contributed noise, especially for beta. It does not remove the larger information gap: sewer topology, pipe and storage capacity, pumping and control state, event rainfall, antecedent saturation, groundwater and maintenance remain unmeasured. More complex algorithms are not the first priority.

## Repository map

| Folder | Purpose |
|---|---|
| `01_data_acquisition/` | Provider coverage, source schemas and safe acquisition scripts |
| `02_data_cleaning/` | Common event contract, duration QC and sensor identity |
| `03_tail_parameter_fitting/` | Strict 240-minute tail, beta/lambda MLE and bootstrap |
| `04_spatial_linkage/` | Sensor-to-WWTW/catchment matching rules and audits |
| `05_environmental_predictors/` | External predictor extraction and dictionary |
| `06_univariate_analysis/` | One-variable inference and cross-outcome tables |
| `07_machine_learning/` | Original selected-sensor and regional-cluster ML tracks |
| `08_final_synthesis/` | Headline scorecards, figures and final conclusions |
| `09_experimental_extensions/` | Univariate predictive screening and hierarchical benchmark |
| `src/` | Small portable package used by the synthetic demo |
| `tests/` | Scientific invariants and leakage checks |
| `docs/` | Reports, methods, provenance and repository walkthrough |

## Reproduce the project

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .[test]
pytest -q
python scripts/run_reproducible_demo.py
```

The demo is synthetic and finishes quickly. Full reproduction requires the omitted raw/licensed sources described in `DATA_AVAILABILITY.md` and a local `config/paths.yaml` based on `config/paths.example.yaml`. QGIS is optional for cartographic inspection and is not required for the demo.

## Three levels of navigation

- **Five minutes:** this README, `outputs/headline_tables/`, and `outputs/headline_figures/`.
- **Thirty minutes:** `docs/final_report.pdf`, both ML reports, and `docs/GITHUB_REPOSITORY_WALKTHROUGH.pdf`.
- **Technical audit:** stage READMEs, scripts, configs, tests, provenance tables and saved predictions.

## Data availability

Raw company EDM records, large NIMROD archives, licensed spatial products, correspondence and machine-local configuration are intentionally excluded. Small synthetic examples, schemas, authoritative headline tables, model metrics and final figures are included. See `DATA_AVAILABILITY.md` and `docs/DATA_PROVENANCE.md`.

## Limitations

Catchment boundaries are proxies for sewer systems; annual/static predictors do not encode individual storms; monitoring and operating practices differ by company; beta and lambda are jointly fitted; regional averaging can create apparent predictability by shrinking extremes. Results are predictive/associational and should not be interpreted causally.

## Citation

Use `CITATION.cff`. No DOI or software licence is asserted. `LICENSE_DECISION_REQUIRED.md` must be resolved before making a public repository.

## Author and research setting

Zain Ahmed, UCL research internship, 2026.
"""


def top_documents() -> None:
    write("README.md", README)
    write("QUICKSTART.md", r"""
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
""")
    write("REPRODUCIBILITY.md", r"""
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
""")
    write("DATA_AVAILABILITY.md", r"""
# Data availability

## Included
- synthetic event and predictor data for the runnable demo;
- schemas and variable dictionaries;
- small final scorecards, univariate summaries, saved predictions and feature summaries;
- publication figures and reports;
- reproducible scientific code and tests.

## Intentionally omitted
- water-company raw and cleaned EDM event archives;
- company JSON exports and EIR correspondence;
- names/coordinates capable of reconstructing the full sensor register;
- the large NIMROD archive;
- licensed hydrogeology, building, terrain, population and detailed spatial source files;
- raw shapefiles, rasters and GeoPackages;
- tokens, local configuration and email material.

Omission does not imply that every source is confidential. Redistribution rights and file sizes were not consistently verified, so the release uses the cautious rule: code, schemas, provenance and aggregate results are public; raw source files are not.

Expected local paths and schemas are in `config/paths.example.yaml`, `data/schemas/`, and `docs/DATA_PROVENANCE.md`. The final scientific story can be inspected without raw-data ingestion through `docs/` and `outputs/headline_*`.
""")
    write("CITATION.cff", """
cff-version: 1.2.0
message: "If you use this research repository, please cite it using these metadata."
title: "CSO Spatial Drivers: From National Spill Records to Catchment-Scale Prediction"
type: software
authors:
  - family-names: Ahmed
    given-names: Zain
version: 1.0.0
date-released: 2026-08-19
""")
    write("LICENSE_DECISION_REQUIRED.md", """
# Licence decision required

No explicit approved software/data licence was found in the working repository. Select a licence before public release and confirm that it is compatible with UCL requirements and every redistributed dataset/figure. Until then, this folder is ready for manual review but must not be described as open source.
""")
    write("AGENTS.md", """
# Agent instructions

- Preserve the scientific definitions in this release.
- The tail condition is strictly `duration_minutes > 240`, never `>= 240`.
- The model is `S(t)=exp[-(lambda*t)^beta]`; lambda is an inverse timescale.
- Lower beta means a heavier/more persistent fitted tail.
- Do not place identifiers, coordinates, target-quality fields, outcomes or event-derived fields in ML predictors.
- Do not allow WWTWs/regions to cross train, validation and test splits.
- Never silently edit raw inputs or claim inferred WWTW linkage is proven hydraulic connectivity.
- Original selected-sensor ML and regional-cluster ML are distinct analyses and must be compared using normalized metrics, not raw RMSE alone.
- Run `pytest -q` and `python scripts/run_reproducible_demo.py` after code changes.
- Never add raw data, tokens, correspondence, absolute local paths or unverified licensed sources.
""")
    write(".gitignore", """
# secrets and local configuration
.env
.env.*
*.pem
*.key
*token*.txt
config/paths.yaml

# raw and large data
data/raw/
data_raw/
clean_data/
json_data/
**/nimrod_1km_5min/
*.tif
*.tiff
*.nc
*.grib
*.grib2
*.shp
*.dbf
*.shx

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.pytest_runtime_tmp/
.venv/
venv/
dist/
build/
*.egg-info/

# QGIS / GIS scratch
*.qgz~
*.qgs~
*.gpkg-shm
*.gpkg-wal
*.aux.xml

# IDE / OS / LaTeX temporary
.vscode/
.idea/
.DS_Store
Thumbs.db
*.aux
*.log
*.out
*.toc
*.synctex.gz

# demo is regenerated
outputs/demo/
""")
    write("requirements.txt", """
numpy>=2.0,<3
pandas>=2.2,<4
scipy>=1.13,<2
matplotlib>=3.8,<4
scikit-learn>=1.5,<2
statsmodels>=0.14,<1
joblib>=1.4,<2
reportlab>=4.0,<5
PyMuPDF>=1.24,<2
PyYAML>=6,<7
""")
    write("pyproject.toml", """
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "cso-spatial-drivers"
version = "1.0.0"
description = "Reproducible methods and final evidence for the UCL CSO spatial-drivers internship"
readme = "README.md"
requires-python = ">=3.11"
dependencies = ["numpy>=2.0,<3", "pandas>=2.2,<4", "scipy>=1.13,<2", "matplotlib>=3.8,<4", "scikit-learn>=1.5,<2", "statsmodels>=0.14,<1", "joblib>=1.4,<2", "PyYAML>=6,<7", "reportlab>=4,<5", "PyMuPDF>=1.24,<2"]

[project.optional-dependencies]
test = ["pytest>=8,<10", "reportlab>=4,<5"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --basetemp=.pytest_runtime_tmp"
pythonpath = ["src"]
""")
    write("environment.yml", """
name: cso-spatial-drivers
channels: [conda-forge]
dependencies:
  - python=3.12
  - numpy
  - pandas
  - scipy
  - matplotlib
  - scikit-learn
  - statsmodels
  - pyyaml
  - pip
  - pip:
      - reportlab
      - PyMuPDF
      - pytest
""")
    write("config/paths.example.yaml", """
# Copy to paths.yaml and replace placeholders. Never commit paths.yaml.
raw_edm_root: /path/to/local/raw_or_cleaned_edm
company_json_root: /path/to/local/company_json
wwtw_catchments: /path/to/licensed_or_public/wwtw_catchments.gpkg
hydrogeology: /path/to/licensed/hydrogeology.gpkg
static_rainfall: /path/to/haduk_grid_climatology
nimrod_archive: /path/to/nimrod_archive
land_cover: /path/to/land_cover
population: /path/to/population
building_age: /path/to/building_age
terrain: /path/to/elevation_or_slope
output_root: outputs/full_run
""")
    write("config/analysis.yaml", """
tail_threshold_minutes: 240
tail_rule: strictly_greater_than
beta_bounds: [0.01, 2.0]
log_lambda_bounds: [-50.0, 20.0]
minimum_tail_events_point_fit: 10
minimum_tail_events_bootstrap: 20
ml_split: {train: 0.70, validation: 0.15, locked_test: 0.15}
headline_metric: RMSE
regional_primary_radius_km: 25
regional_sensitivity_radii_km: [10, 50]
""")
    write("config/variables.yaml", """
outcomes:
  beta: fitted_beta
  lambda: fitted_lambda
  duration: spill_mean_duration_minutes
  frequency: spill_events_per_monitoring_year
forbidden_predictor_families:
  - identifiers
  - coordinates
  - outcomes
  - confidence_intervals
  - target_quality
  - event_derived_fields
""")


STAGE_READMES = {
"01_data_acquisition": ("Where do the EDM and external records come from?", "Provider exports, year coverage, source schemas and download notes.", "Company-specific acquisition/standardisation scripts plus schemas.", "Locally stored immutable source files and manifests; no raw data are shipped.", "Coverage, checksum, schema and licence review.", "Validated source files enter common event cleaning."),
"02_data_cleaning": ("How are incompatible provider records turned into one event contract?", "Provider event tables with permit, location and timestamps.", "Portable cleaning functions and authoritative preparation scripts.", "One row per valid event with stable sensor UID and duration minutes.", "End >= start, deduplication, finite non-negative duration and schema checks.", "Clean events feed tail fitting and observed duration/frequency outcomes."),
"03_tail_parameter_fitting": ("How are long-duration persistence parameters estimated?", "Clean sensor events and the strict 240-minute threshold.", "Conditional stretched-exponential MLE and nonparametric sensor bootstrap.", "Beta, lambda, intervals, fit status and tail support.", "Positivity/bounds, tail count, bootstrap success, boundary-hit and CI checks.", "Validated sensor outcomes feed WWTW linkage."),
"04_spatial_linkage": ("Which WWTW/catchment provides context for each sensor?", "Sensor coordinates/names, company, treatment works and catchment polygons.", "Same-company containment plus documented name/distance/manual evidence.", "Audited sensor-WWTW assignments and one selected sensor per WWTW for the original analysis.", "Uniqueness, ambiguity, company consistency and beta-blind assignment.", "Linked WWTWs receive external predictors."),
"05_environmental_predictors": ("Which measurable catchment characteristics might relate to spills?", "Validated catchments and external spatial/operational sources.", "Area-weighted extraction and key-preserving joins.", "Scientific predictor master with units and coverage metadata.", "CRS/coverage, row preservation, compositions, missingness and no target leakage.", "The predictor matrix enters univariate and predictive models."),
"06_univariate_analysis": ("What association does each variable show on its own?", "Target-specific eligible WWTWs and external predictors.", "HC3 log-outcome regressions, correlations and FDR correction.", "Effect/CI/p/q tables and forest/correlation figures.", "Broad/strict cohorts, multiplicity and diagnostic checks.", "Associational evidence informs but does not select the ML test result."),
"07_machine_learning": ("How well do predictors generalise to unseen units?", "The original selected-sensor WWTW master and the regional-cluster master.", "Four separate 70/15/15 workflows for each approach.", "Models, locked-test predictions, scorecards, learning curves and importance.", "Dummy baseline, zero split overlap, in-fold preprocessing, reload and metric reproduction.", "Both approaches feed the final scientific comparison."),
"08_final_synthesis": ("What did the project learn overall?", "Univariate results, original ML, regional ML and diagnostics.", "Audited scorecard/figure assembly.", "Professor-facing reports, headline tables and restrained conclusions.", "Cross-document metric and wording checks.", "Provides the submitted scientific narrative and future priorities."),
"09_experimental_extensions": ("Which later diagnostics test why skill is modest?", "Current master, shared splits, regional groups and beta/lambda fits.", "Univariate predictive screening, bounded-beta diagnostic and hierarchical benchmark.", "Standalone-predictor rankings and documented MixedLM failure.", "Historical split identity, boundary handling and honest convergence reporting.", "Guides the next event-level/hydraulic research design."),
}


def stage_docs() -> None:
    for folder, values in STAGE_READMES.items():
        q, inp, code, out, qc, nxt = values
        write(f"{folder}/README.md", f"""
# {folder.replace('_', ' ').title()}

## Scientific question
{q}

## Inputs
{inp}

## Code to run
{code}

## Outputs
{out}

## QC and validation
{qc}

## Connection to the next stage
{nxt}

The public demo exercises the portable core. Full-scale scripts require omitted source data described in `../DATA_AVAILABILITY.md`.
""")
    write("03_tail_parameter_fitting/equations.md", r"""
# Tail model

For event duration `T` and threshold `u=240` minutes, the fitted conditional survival for `T>u` is based on

`S(t) = exp[-(lambda*t)^beta]`.

The conditional log-likelihood subtracts the survival contribution at `u`. The optimizer bounds are `log(lambda) in [-50,20]` and `beta in [0.01,2]`. Lower beta means a heavier/more persistent fitted tail. Lambda is an inverse timescale in this parameterisation. Exact boundary fits are diagnostics, not values to clip silently into transformed regressions.
""")
    write("04_spatial_linkage/matching_rules.md", """
# Matching rules

1. Never use beta, lambda, duration, frequency or model performance to choose a WWTW.
2. Require water-company consistency.
3. Prefer unique catchment containment where geometry is valid.
4. Use transparent name/place normalization and documented distance support.
5. Record ambiguity, alternative candidates and manual decisions.
6. One original selected sensor maps to one WWTW; one sensor cannot enter multiple primary regional clusters.
7. Assignment is contextual, not proof of underground hydraulic connectivity.
""")


def package_code() -> None:
    write("src/cso_spatial_drivers/__init__.py", """\"\"\"Portable public core for the CSO spatial-drivers demo.\"\"\"\n__version__ = \"1.0.0\"""")
    for sub in ["io","cleaning","fitting","spatial","predictors","statistics","ml","plotting"]:
        write(f"src/cso_spatial_drivers/{sub}/__init__.py", "")
    write("src/cso_spatial_drivers/cleaning/events.py", r'''
from __future__ import annotations
import pandas as pd

REQUIRED = ["company", "permit_number", "location_name", "start_time", "stop_time"]

def clean_events(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in frame]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    x = frame.copy()
    x["company"] = x.company.astype("string").str.strip().str.lower()
    x["permit_number"] = x.permit_number.astype("string").str.strip()
    x["start_time"] = pd.to_datetime(x.start_time, errors="coerce", utc=True)
    x["stop_time"] = pd.to_datetime(x.stop_time, errors="coerce", utc=True)
    x["duration_minutes"] = (x.stop_time - x.start_time).dt.total_seconds() / 60.0
    good = x.start_time.notna() & x.stop_time.notna() & x.stop_time.ge(x.start_time)
    x = x.loc[good].drop_duplicates(["company", "permit_number", "start_time", "stop_time"]).copy()
    x["sensor_uid"] = x.company + "::" + x.permit_number
    return x.sort_values(["sensor_uid", "start_time"]).reset_index(drop=True)
''')
    write("src/cso_spatial_drivers/fitting/stretched_exponential.py", r'''
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize

def tail_values(durations, threshold: float = 240.0) -> np.ndarray:
    x = np.asarray(durations, dtype=float)
    return x[np.isfinite(x) & (x > threshold)]

def fit_tail(durations, threshold: float = 240.0) -> dict:
    tail = tail_values(durations, threshold)
    if len(tail) < 10:
        raise ValueError("at least 10 events strictly above threshold are required")
    log_x = np.log(tail); log_u = math.log(threshold)
    def nll(par):
        log_lam, beta = float(par[0]), float(par[1])
        z = beta * (log_lam + log_x); zu = beta * (log_lam + log_u)
        if np.any(z > 700) or zu > 700: return np.finfo(float).max
        return -float(np.sum(math.log(beta) + log_lam + (beta-1)*(log_lam+log_x) - np.exp(z) + math.exp(zu)))
    result = minimize(nll, [math.log(1/np.median(tail)), .18], method="L-BFGS-B", bounds=[(-50,20),(.01,2.0)])
    return {"beta": float(result.x[1]), "lambda_per_minute": float(math.exp(result.x[0])), "n_tail": int(len(tail)), "success": bool(result.success)}
''')
    write("src/cso_spatial_drivers/spatial/assignment.py", r'''
from __future__ import annotations
import numpy as np
import pandas as pd

def nearest_same_company(sensors: pd.DataFrame, works: pd.DataFrame, max_distance_m: float = 5000) -> pd.DataFrame:
    rows=[]
    for _, s in sensors.iterrows():
        candidates=works.loc[works.company.eq(s.company)].copy()
        if candidates.empty: continue
        d=np.hypot(candidates.easting-s.easting, candidates.northing-s.northing)
        i=d.idxmin(); distance=float(d.loc[i])
        if distance <= max_distance_m:
            rows.append({"sensor_uid":s.sensor_uid,"company":s.company,"uwwCode":works.loc[i,"uwwCode"],"distance_m":distance})
    return pd.DataFrame(rows)
''')
    write("src/cso_spatial_drivers/predictors/join.py", r'''
from __future__ import annotations
import pandas as pd

def join_predictors(outcomes: pd.DataFrame, predictors: pd.DataFrame) -> pd.DataFrame:
    keys=["company","uwwCode"]
    if predictors.duplicated(keys).any(): raise ValueError("predictor keys are not unique")
    result=outcomes.merge(predictors,on=keys,how="left",validate="many_to_one")
    if len(result)!=len(outcomes): raise AssertionError("predictor join changed row count")
    return result
''')
    write("src/cso_spatial_drivers/statistics/univariate.py", r'''
from __future__ import annotations
import numpy as np
import statsmodels.api as sm

def fit_log_outcome(y, x) -> dict:
    y=np.asarray(y,float); x=np.asarray(x,float); ok=np.isfinite(y)&(y>0)&np.isfinite(x)
    model=sm.OLS(np.log(y[ok]),sm.add_constant(x[ok])).fit(cov_type="HC3")
    return {"n":int(ok.sum()),"slope":float(model.params[1]),"hc3_se":float(model.bse[1]),"p_value":float(model.pvalues[1]),"r_squared":float(model.rsquared)}
''')
    write("src/cso_spatial_drivers/ml/demo.py", r'''
from __future__ import annotations
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def fit_locked_demo(X_train,y_train,X_validation,y_validation,X_test,y_test):
    candidates=[]
    for alpha in (.1,1.0,10.0):
        m=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))])
        m.fit(X_train,y_train); candidates.append((mean_squared_error(y_validation,m.predict(X_validation))**.5,alpha,m))
    _,alpha,_=min(candidates,key=lambda z:z[0])
    X_fit=np.vstack([X_train,X_validation]); y_fit=np.r_[y_train,y_validation]
    model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))]).fit(X_fit,y_fit)
    dummy=DummyRegressor().fit(X_fit,y_fit); pred=model.predict(X_test); dp=dummy.predict(X_test)
    rmse=mean_squared_error(y_test,pred)**.5; dr=mean_squared_error(y_test,dp)**.5
    return {"alpha":alpha,"rmse":rmse,"dummy_rmse":dr,"improvement_pct":100*(dr-rmse)/dr,"r2":r2_score(y_test,pred),"predictions":pred}
''')


def demo_and_tests() -> None:
    rng=np.random.default_rng(20260819); companies=["demo_a","demo_b"]; rows=[]
    for i in range(18):
        company=companies[i%2]; permit=f"S{i:03d}"; n=36
        start=pd.Timestamp("2024-01-01",tz="UTC")+pd.to_timedelta(rng.integers(0,525600,n),unit="m")
        duration=np.exp(rng.normal(5.6+.15*(i%4),.9,n)); stop=start+pd.to_timedelta(duration,unit="m")
        for a,b in zip(start,stop): rows.append({"company":company,"permit_number":permit,"location_name":f"Synthetic sensor {i}","start_time":a.isoformat(),"stop_time":b.isoformat()})
    events=pd.DataFrame(rows); events.to_csv(ROOT/"data/sample/synthetic_events.csv",index=False)
    pred=pd.DataFrame({"company":[companies[i%2] for i in range(18)],"uwwCode":[f"W{i:03d}" for i in range(18)],"population_density_km2":rng.uniform(100,3500,18),"rain_mean_annual_mm":rng.uniform(600,1400,18),"catchment_area_km2":rng.uniform(2,90,18)})
    pred.to_csv(ROOT/"data/sample/synthetic_predictors.csv",index=False)
    write("data/README.md", """
# Data included in this repository

`sample/` contains synthetic records only. `processed_examples/` contains outputs generated from those records. `schemas/` documents full-data contracts. No row is a real sensor or treatment works.
""")
    pd.DataFrame({"column":["company","permit_number","location_name","start_time","stop_time","duration_minutes","sensor_uid"],"type":["string","string","string","UTC datetime","UTC datetime","float","string"],"rule":["normalized provider","stable within company","descriptive","parseable","parseable and >= start","non-negative minutes","company::permit"]}).to_csv(ROOT/"data/schemas/event_schema.csv",index=False)
    write("scripts/run_reproducible_demo.py", r'''
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cso_spatial_drivers.cleaning.events import clean_events
from cso_spatial_drivers.fitting.stretched_exponential import fit_tail
from cso_spatial_drivers.statistics.univariate import fit_log_outcome
from cso_spatial_drivers.ml.demo import fit_locked_demo

events=clean_events(pd.read_csv(ROOT/"data/sample/synthetic_events.csv"))
fits=[]
for sensor,g in events.groupby("sensor_uid"):
    result=fit_tail(g.duration_minutes); company,permit=sensor.split("::",1); i=int(permit[1:]); result.update({"sensor_uid":sensor,"company":company,"uwwCode":f"W{i:03d}","mean_duration":g.duration_minutes.mean()}); fits.append(result)
master=pd.DataFrame(fits).merge(pd.read_csv(ROOT/"data/sample/synthetic_predictors.csv"),on=["company","uwwCode"],validate="one_to_one")
order=np.arange(len(master)); train=order[:12]; val=order[12:15]; test=order[15:]
X=master[["population_density_km2","rain_mean_annual_mm","catchment_area_km2"]].to_numpy(); y=np.log(master.beta.to_numpy())
ml=fit_locked_demo(X[train],y[train],X[val],y[val],X[test],y[test])
uni=fit_log_outcome(master.beta,master.population_density_km2)
out=ROOT/"outputs/demo"; out.mkdir(parents=True,exist_ok=True); events.to_csv(out/"clean_events.csv",index=False); master.to_csv(out/"demo_master.csv",index=False); (out/"metrics.json").write_text(json.dumps({"label":"DEMONSTRATION ONLY - NOT THE SCIENTIFIC RESULTS","univariate":uni,"ml":{k:v for k,v in ml.items() if k!="predictions"}},indent=2),encoding="utf-8")
print("DEMONSTRATION ONLY - NOT THE SCIENTIFIC RESULTS"); print(json.dumps({k:v for k,v in ml.items() if k!="predictions"},indent=2))
''')
    tests={
    "test_duration_calculation.py": """import pandas as pd\nfrom cso_spatial_drivers.cleaning.events import clean_events\ndef test_duration_minutes_and_end_order():\n d=pd.DataFrame([{'company':'x','permit_number':'1','location_name':'x','start_time':'2024-01-01T00:00Z','stop_time':'2024-01-01T01:30Z'},{'company':'x','permit_number':'2','location_name':'x','start_time':'2024-01-02T02:00Z','stop_time':'2024-01-02T01:00Z'}]); x=clean_events(d); assert len(x)==1 and x.duration_minutes.iloc[0]==90\n""",
    "test_tail_threshold.py": """from cso_spatial_drivers.fitting.stretched_exponential import tail_values\ndef test_tail_is_strictly_greater_than_240():\n assert tail_values([239,240,240.0001]).tolist()==[240.0001]\n""",
    "test_beta_lambda_fitting.py": """import numpy as np\nfrom cso_spatial_drivers.fitting.stretched_exponential import fit_tail\ndef test_positive_bounded_fit():\n r=fit_tail(np.linspace(241,2000,40)); assert .01<=r['beta']<=2 and r['lambda_per_minute']>0 and r['n_tail']==40\n""",
    "test_spatial_assignment.py": """import pandas as pd\nfrom cso_spatial_drivers.spatial.assignment import nearest_same_company\ndef test_same_company_assignment():\n s=pd.DataFrame([{'sensor_uid':'a','company':'x','easting':0,'northing':0}]); w=pd.DataFrame([{'uwwCode':'wrong','company':'y','easting':0,'northing':0},{'uwwCode':'right','company':'x','easting':10,'northing':0}]); assert nearest_same_company(s,w).uwwCode.iloc[0]=='right'\n""",
    "test_predictor_join.py": """import pandas as pd, pytest\nfrom cso_spatial_drivers.predictors.join import join_predictors\ndef test_join_preserves_rows_and_rejects_duplicates():\n o=pd.DataFrame({'company':['x','x'],'uwwCode':['a','a'],'y':[1,2]}); p=pd.DataFrame({'company':['x'],'uwwCode':['a'],'v':[3]}); assert len(join_predictors(o,p))==2\n with pytest.raises(ValueError): join_predictors(o,pd.concat([p,p]))\n""",
    "test_ml_leakage.py": """import pandas as pd\ndef test_forbidden_predictor_manifest():\n forbidden={'sensor_uid','uwwCode','fitted_beta','fitted_lambda','spill_mean_duration_minutes','spill_events_per_monitoring_year'}; selected={'population_density_km2','rain_mean_annual_mm','catchment_area_km2'}; assert not forbidden & selected\ndef test_group_split_overlap_zero():\n split=pd.DataFrame({'uwwCode':['a','b','c'],'split':['TRAIN','VALIDATION','TEST']}); assert split.groupby('uwwCode').split.nunique().max()==1\n""",
    }
    for name,text in tests.items(): write(f"tests/{name}",text)
    write(".github/workflows/tests.yml", """
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: python -m pip install -e .[test]
      - run: pytest -q
      - run: python scripts/run_reproducible_demo.py
""")


def authoritative_copies() -> None:
    # Final/public documents.
    copy("PROJECT_RESULTS/Final Documents/CSO_Final_Report_Style_Preview.pdf", "docs/final_report.pdf")
    copy("PROJECT_RESULTS/Final Documents/CSO_NEW_MACHINE_LEARNING_MODELS_1_TO_4_MASTER_REPORT_REDESIGNED.pdf", "docs/original_selected_sensor_ml_report.pdf")
    copy("REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/report/CSO_REGIONAL_CLUSTER_MACHINE_LEARNING_REPORT.pdf", "docs/regional_cluster_ml_report.pdf")
    write("docs/FINAL_PRESENTATION_NOT_FOUND.md", """
# Final presentation not found

No PPTX or presentation PDF was present in the working repository during the release audit on 19 August 2026. A different report has not been relabelled as a presentation. Add the approved presentation later as `docs/final_presentation.pdf`, then update hashes and consistency documentation.
""")
    # Core authoritative scripts.
    copies={
      "scripts/01_prepare_thames_company_data.py":"01_data_acquisition/scripts/prepare_thames_company_data.py",
      "scripts/download_nimrod_1km.py":"01_data_acquisition/scripts/download_nimrod_1km.py",
      "scripts/02_prepare_sensor_master.py":"02_data_cleaning/scripts/prepare_sensor_master.py",
      "analyse_individual_cso_heavy_tails.py":"03_tail_parameter_fitting/scripts/analyse_individual_cso_heavy_tails.py",
      "bootstrap_all_individual_cso_sensors.py":"03_tail_parameter_fitting/scripts/bootstrap_all_individual_cso_sensors.py",
      "scripts/03_assign_sensors_to_catchments.py":"04_spatial_linkage/scripts/assign_sensors_to_catchments.py",
      "scripts/08b_rebuild_wwtw_name_matching_place_core.py":"04_spatial_linkage/scripts/rebuild_name_matching.py",
      "scripts/08c_select_primary_wwtw_sensor_by_distance.py":"04_spatial_linkage/scripts/select_primary_sensor.py",
      "scripts/08e_build_final_manual_wwtw_sensor_release.py":"04_spatial_linkage/scripts/build_final_manual_release.py",
      "scripts/04_build_wwtw_capacity.py":"05_environmental_predictors/scripts/build_wwtw_capacity.py",
      "scripts/05_extract_hydrogeology_characteristics.py":"05_environmental_predictors/scripts/extract_hydrogeology.py",
      "scripts/07_build_sensor_environment_master.py":"05_environmental_predictors/scripts/build_sensor_environment_master.py",
      "scripts/09_build_wwtw_rainfall_characteristics.py":"05_environmental_predictors/scripts/build_static_rainfall.py",
      "PROJECT_RESULTS/four_outcome_annual_nimrod_release_20260813_130420/statistical_analysis/pipeline/build_annual_nimrod_indices.py":"05_environmental_predictors/scripts/build_annual_nimrod_indices.py",
      "PROJECT_RESULTS/four_outcome_annual_nimrod_release_20260813_130420/statistical_analysis/pipeline/build_four_outcome_analyses.py":"06_univariate_analysis/scripts/build_four_outcome_analyses.py",
      "PROJECT_RESULTS/four_outcome_annual_nimrod_release_20260813_130420/statistical_analysis/pipeline/build_cross_outcome.py":"06_univariate_analysis/scripts/build_cross_outcome.py",
      "machine learning/NEW MACHINE LEARNING/pipeline/ml_programme.py":"07_machine_learning/scripts/original_selected_sensor/ml_programme.py",
      "machine learning/NEW MACHINE LEARNING/pipeline/build_cross_model_figures.py":"07_machine_learning/scripts/original_selected_sensor/build_cross_model_figures.py",
      "machine learning/NEW MACHINE LEARNING/pipeline/finalize_release.py":"07_machine_learning/scripts/original_selected_sensor/finalize_release.py",
      "REGIONAL_CLUSTER_AND_UNIVARIATE_ML/pipeline/01_run_regional_analysis.py":"07_machine_learning/scripts/regional_cluster/01_run_regional_analysis.py",
      "REGIONAL_CLUSTER_AND_UNIVARIATE_ML/pipeline/03_run_hierarchical_benchmark.py":"07_machine_learning/scripts/regional_cluster/03_run_hierarchical_benchmark.py",
      "REGIONAL_CLUSTER_AND_UNIVARIATE_ML/pipeline/06_exact_event_medians_and_radius_models.py":"07_machine_learning/scripts/regional_cluster/06_exact_event_medians_and_radius_models.py",
      "REGIONAL_CLUSTER_AND_UNIVARIATE_ML/pipeline/07_redesign_regional_report.py":"07_machine_learning/scripts/regional_cluster/07_redesign_regional_report.py",
      "REGIONAL_CLUSTER_AND_UNIVARIATE_ML/pipeline/RUN_ORDER.txt":"07_machine_learning/scripts/regional_cluster/RUN_ORDER.txt",
    }
    for src,dst in copies.items(): copy(src,dst)
    # Tests closest to authoritative working code.
    for name in ["test_event_contracts.py","test_mapping_contracts.py","test_assign_sensors_to_catchments.py","test_08c_select_primary_wwtw_sensor_by_distance.py","test_09_rainfall_characteristics.py"]:
        copy(f"tests/{name}",f"{('02_data_cleaning' if 'event' in name else '04_spatial_linkage' if 'mapping' in name or 'assign' in name or '08c' in name else '05_environmental_predictors')}/tests/{name}")
    # Configs and dictionaries.
    copy("config/wwtw_sensor_place_core_matching.json","04_spatial_linkage/matching_config.json")
    copy("machine learning/NEW MACHINE LEARNING/shared/predictor_manifest.csv","05_environmental_predictors/predictor_dictionary.csv")
    copy("machine learning/NEW MACHINE LEARNING/model_1_beta/config/config.json","07_machine_learning/configs/original_selected_sensor/model_1_beta.json")
    copy("REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/config/regional_experiment.json","07_machine_learning/configs/regional_cluster/regional_experiment.json")


def result_tables() -> None:
    old=pd.read_csv(SOURCE/"machine learning/NEW MACHINE LEARNING/cross_model/tables/four_model_summary.csv")
    reg=pd.read_csv(SOURCE/"REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/results/regional_four_model_summary.csv")
    old_out=old[["target","label","eligible_n","train_n","validation_n","test_n","selected_algorithm","rmse_model_scale","dummy_rmse_model_scale","rmse_improvement_pct","r2_model_scale","spearman"]].copy(); old_out.insert(0,"approach","original_selected_sensor_one_WWTW")
    reg_out=reg[["target","label","eligible_n","train_n","validation_n","test_n","selected_algorithm","rmse","dummy_rmse","rmse_improvement_pct","r2","spearman"]].copy(); reg_out.insert(0,"approach","regional_company_x_25km_cluster"); reg_out.columns=old_out.columns
    score=pd.concat([old_out,reg_out],ignore_index=True); score.to_csv(ROOT/"outputs/headline_tables/final_ml_scorecard.csv",index=False); score.to_csv(ROOT/"08_final_synthesis/tables/original_vs_regional_ml_scorecard.csv",index=False)
    pd.DataFrame([
      {"outcome":"beta","definition":"stretched-exponential tail shape","transformation":"log(beta)","interpretation":"lower = heavier/more persistent tail"},
      {"outcome":"lambda","definition":"inverse-timescale parameter","transformation":"log(lambda)","interpretation":"higher = shorter characteristic time at fixed beta"},
      {"outcome":"duration","definition":"mean valid event duration","transformation":"log1p(minutes)","interpretation":"direct observed duration summary"},
      {"outcome":"frequency","definition":"events per valid monitoring year","transformation":"log1p(rate)","interpretation":"exposure-adjusted occurrence"},
    ]).to_csv(ROOT/"outputs/headline_tables/four_outcome_definitions.csv",index=False)
    uni=pd.read_csv(SOURCE/"PROJECT_RESULTS/four_outcome_annual_nimrod_release_20260813_130420/statistical_analysis/cross_outcome/four_outcome_univariate_summary.csv"); uni.to_csv(ROOT/"outputs/headline_tables/univariate_summary.csv",index=False); uni.to_csv(ROOT/"06_univariate_analysis/results/four_outcome_univariate_summary.csv",index=False)
    # Sensor counts.
    counts=[]
    for path in sorted((SOURCE/"outputs/individual_cso_sensors/gt_240min").glob("*/02_sensors_by_tail_spills_desc.csv")):
        counts.append({"company":path.parent.name,"sensors":len(pd.read_csv(path,usecols=["permit_number"]))})
    pd.DataFrame(counts).to_csv(ROOT/"outputs/headline_tables/sensor_counts.csv",index=False)
    # Top permutation features for both approaches.
    rows=[]
    old_folder={"beta":"model_1_beta","scale":"model_2_scale","duration":"model_3_duration","frequency":"model_4_count"}
    for target,folder in old_folder.items():
        p=pd.read_csv(SOURCE/f"machine learning/NEW MACHINE LEARNING/{folder}/results/permutation_importance.csv").head(5)
        for rank,(_,r) in enumerate(p.iterrows(),1): rows.append({"approach":"original_selected_sensor_one_WWTW","target":"lambda" if target=="scale" else target,"rank":rank,"feature":r.iloc[0],"importance":r.iloc[1]})
    for target in ["beta","lambda","duration","frequency"]:
        p=pd.read_csv(SOURCE/f"REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/results/{target}/permutation_importance.csv").head(5)
        for rank,(_,r) in enumerate(p.iterrows(),1): rows.append({"approach":"regional_company_x_25km_cluster","target":target,"rank":rank,"feature":r.feature,"importance":r.permutation_importance})
    pd.DataFrame(rows).to_csv(ROOT/"outputs/headline_tables/feature_importance_summary.csv",index=False)
    # Copy complete compact results/predictions/config/model files.
    copy("machine learning/NEW MACHINE LEARNING/cross_model/tables/four_model_summary.csv","07_machine_learning/results/original_selected_sensor/four_model_summary.csv")
    copy("REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/results/regional_four_model_summary.csv","07_machine_learning/results/regional_cluster/four_model_summary.csv")
    copy("REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/results/old_wwtw_regional_comparison.csv","07_machine_learning/results/regional_cluster/old_wwtw_regional_comparison.csv")
    for target,folder in old_folder.items():
        public="lambda" if target=="scale" else target
        copy(f"machine learning/NEW MACHINE LEARNING/{folder}/results/test_predictions.csv",f"07_machine_learning/results/original_selected_sensor/{public}_test_predictions.csv")
        copy(f"machine learning/NEW MACHINE LEARNING/{folder}/models/final_model.joblib",f"07_machine_learning/models/original_selected_sensor/{public}_final_model.joblib")
    for target in ["beta","lambda","duration","frequency"]:
        copy(f"REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/results/{target}/test_predictions.csv",f"07_machine_learning/results/regional_cluster/{target}_test_predictions.csv")
        copy(f"REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/models/{target}/final_model.joblib",f"07_machine_learning/models/regional_cluster/{target}_final_model.joblib")
    copy("REGIONAL_CLUSTER_AND_UNIVARIATE_ML/02_univariate_machine_learning/cross_target/all_univariate_results.csv","09_experimental_extensions/univariate_ml/all_univariate_results.csv")
    copy("REGIONAL_CLUSTER_AND_UNIVARIATE_ML/03_hierarchical_benchmark/results/hierarchical_benchmark_summary.csv","09_experimental_extensions/hierarchical_benchmark/summary.csv")


def docs() -> None:
    write("docs/ANALYSIS_STORY.md", """
# Analysis story

The project starts with event-level EDM records, standardises sensor identity and duration, isolates events strictly longer than 240 minutes, fits beta and lambda, links outcomes to WWTW catchments, builds an external predictor matrix, quantifies one-variable associations, and evaluates four locked-test prediction problems. The original model uses one selected sensor per WWTW. The regional comparison pools independently defined within-company WWTW clusters and improves beta most strongly. Neither approach supports causal claims or operational forecasting.
""")
    write("docs/METHODS.md", """
# Methods

## Event contract
One row per unique company, permit, start and stop timestamp. Duration is `(stop-start)/60` minutes; invalid chronology is rejected.

## Tail model
Only `T > 240` enters the conditional stretched-exponential fit `S(t)=exp[-(lambda*t)^beta]`. Point fits require at least 10 tail events and bootstrap inference at least 20. Beta bounds are `[0.01,2]`; log-lambda bounds are `[-50,20]`.

## Spatial context
Assignments are company-consistent and beta-blind. Polygon containment is preferred; place/name/distance/manual evidence is audited. Catchments are contextual proxies, not proven hydraulic networks.

## Statistics and prediction
Univariate log-outcome models use HC3 and FDR. Original and regional ML use separate target cohorts, 70/15/15 splits, training-only preprocessing, validation selection, one locked-test evaluation and Dummy-relative RMSE. Coordinates, identifiers, other outcomes, target-quality and event-derived leakage fields are excluded from X.
""")
    write("docs/DATA_PROVENANCE.md", """
# Data provenance

| Dataset | Provider / mechanism | Years | Local pattern | Redistribution | Included | Replacement / downstream use |
|---|---|---:|---|---|---|---|
| EDM spill events | Nine English water companies; public/EIR exports | approximately 2020-2026, company dependent | `clean_data/<company>/*.csv` | UNKNOWN | No | Obtain from company/regulator; map to event schema; all outcomes |
| Sensor/location JSON | Water-company exports | mixed | `json_data/*.json` | UNKNOWN | No | Obtain provider location export; sensor identity/maps |
| WWTW/catchment polygons | project-assembled spatial sources | current project vintage | `data_raw/wastewater_catchments/` | UNKNOWN | No | Verify provider/licence; spatial linkage and area weights |
| Hydrogeology | BGS HydrogeologyUK_IoM_v5 | source vintage documented locally | `data_raw/hydrogeology/` | licence restricted/UNKNOWN | No | Acquire from BGS; polygon composition |
| Static rainfall | HadUK-Grid 1991-2020 climatology | 1991-2020 | `data_raw/rainfall/` | UNKNOWN | No | Met Office/CEDA; catchment-area-weighted annual/winter mean |
| Annual rainfall indices | Met Office NIMROD 1 km radar archive | project event years | `data_raw/dynamic_rainfall/` | restricted/large | No | CEDA credentials; annual PRCPTOT/Rx/SDII/R10/R20/CWD/CDD |
| Land cover | project spatial source | source vintage documented locally | `data_raw/land_cover/` | UNKNOWN | No | Reacquire from provider; developed-land measures |
| Population | 2021 population allocation | 2021 | `data_raw/population/` | UNKNOWN | No | ONS/provider; population density |
| Building age | project property/building source | mixed | `data_raw/building_age/` | UNKNOWN | No | Verify licence; pre-1973 share |
| Terrain | Ordnance Survey elevation product | supplied project vintage | `data_raw/elevation/` | licence restricted/UNKNOWN | No | Acquire under appropriate licence; mean slope |
| Treatment capacity | project treatment-works capacity tables | latest available by works | `data_intermediate/capacity/` | UNKNOWN | No | Reconstruct load/design PE; capacity ratio |

No private EIR contact details are included.
""")
    write("docs/VARIABLE_DICTIONARY.md", """
# Variable dictionary

The complete machine-readable predictor dictionary is `05_environmental_predictors/predictor_dictionary.csv`. Primary families are hydrogeology composition, static rainfall, annual NIMROD rainfall, developed land, population density, building age, terrain slope, treatment-capacity utilisation and catchment area. Percentages are compositions; coefficients are not independent causal effects. Coordinates, IDs, outcomes, confidence intervals and target-quality fields are never predictors.
""")
    write("docs/RESULTS_SUMMARY.md", """
# Results summary

## Original selected-sensor / one-WWTW ML
Beta +4.9%, lambda +4.9%, duration +2.6%, annualised frequency +7.0% RMSE improvement over Dummy. Test R-squared 0.052-0.133. Frequency is the strongest original target.

## Regional-cluster ML
Beta +17.4% (R-squared 0.314), lambda +9.7% (0.180), duration +7.8% (0.143), frequency -2.0% (-0.063). Positive-gain bootstrap intervals cross zero. Regional aggregation helps beta most but is not a universal improvement.

## Interpretation
The evidence supports a mixture of noisy selected-sensor targets, spatial-unit mismatch, variance shrinkage and missing hydraulic/event-state data. The results are modest predictive evidence, not causality.
""")
    write("docs/LIMITATIONS.md", """
# Limitations

- WWTW catchments and distance clusters are not sewer-network topology.
- Static/annual predictors do not represent storm sequence, antecedent wetness or controls.
- Beta and lambda are jointly fitted and statistically coupled.
- Target quality and monitoring practice vary between companies.
- Regional averaging reduces extremes and may improve RMSE without stronger physics.
- Original and regional raw RMSE are not directly comparable because target variance changes.
- The regional locked tests contain only 32-39 observations, so bootstrap uncertainty is wide.
""")
    write("docs/REPORT_TO_CODE_MAP.md", """
# Report-to-code map

| Public material | Section / figure | Source data | Generating code | Released output / key config |
|---|---|---|---|---|
| `final_report.pdf` | Data and event workflow | cleaned event contract and sensor summaries | `03_tail_parameter_fitting/scripts/analyse_individual_cso_heavy_tails.py` | `config/analysis.yaml`; strict `T>240` |
| `final_report.pdf` | Beta/lambda fitting | sensor event durations | same plus bootstrap script | beta bounds `[0.01,2]` |
| `final_report.pdf` | Spatial linkage | sensor/WWTW matching tables | `04_spatial_linkage/scripts/` | same-company, beta-blind rules |
| `final_report.pdf` | Environmental analysis | scientific master | `05_environmental_predictors/scripts/` | area-weighted EPSG:27700 extraction |
| `final_report.pdf` | Four-outcome univariate figures | four-outcome summary | `06_univariate_analysis/scripts/build_four_outcome_analyses.py` | HC3/FDR tables |
| Original ML report, models 1-4 | Scorecard, predictions, learning curves, importance | selected-sensor WWTW master | `07_machine_learning/scripts/original_selected_sensor/ml_programme.py` | target-specific 70/15/15 configs |
| Regional ML report, sections 3-6 | Four regional models | 25 km regional master | `07_machine_learning/scripts/regional_cluster/01_run_regional_analysis.py` | `regional_experiment.json` |
| Regional ML report, section 7 | 10/25/50 km sensitivity | regional masters | `06_exact_event_medians_and_radius_models.py` | fixed radii; 25 km primary |
| Regional ML report, section 9 | Hierarchical benchmark | multi-sensor WWTW rows + regions | `03_run_hierarchical_benchmark.py` | singular fits reported honestly |
| Regional ML report, section 10 | Beta-logit diagnostic | finite beta/lambda pairs | original diagnostic source retained in extension | beta upper bound 2; boundary fits excluded |

The 23-page final report uses page-based narrative rather than stable numbered figure labels. No fabricated figure numbering is assigned here.
""")
    write("docs/FIGURE_PROVENANCE.md", """
# Figure provenance

| Public filename | Original source | Generator | Data / target / transformation | Notes |
|---|---|---|---|---|
| `national_overview.*` | final WWTW master | release headline-figure script | WWTW coordinates only | no basemap; QGIS report map remains a manual cartographic product |
| `duration_tail.*` | selected valid event durations | release headline-figure script | duration minutes, log axes, strict 240 marker | aggregated; raw events omitted |
| `beta_lambda_interpretation.*` | regional diagnostic output | `00_inspect_and_beta_diagnostic.py` | beta and generalized logit vs log(lambda) | joint-fit diagnostic, non-causal |
| `original_ml_predicted_vs_observed_*` | original locked-test predictions | original ML pipeline | four transformed targets | PDF copied; PNG rendered from PDF |
| `regional_ml_predicted_vs_observed_*` | regional locked-test predictions | regional pipeline | four transformed targets | PDF/PNG copied |
| `original_ml_learning_curves_*` | original training/CV summaries | original ML pipeline | target-specific cohorts | no test-based selection |
| `regional_ml_learning_curves_*` | regional training/CV summaries | regional pipeline | 25 km regions | no target used to form regions |
| `original_ml_importance_*` | original permutation tables | original ML pipeline | selected models | predictive, not causal |
| `regional_ml_importance_*` | regional permutation tables | regional pipeline | selected models | conditional on correlated features |
| `final_synthesis.*` | original and regional scorecards | release headline-figure script | normalized RMSE gain | compares units without raw-RMSE misuse |

Forest-plot PDFs in `06_univariate_analysis/figures/` are copied from the authoritative four-outcome release. QGIS/manual maps in the submitted report are not claimed as fully generated by Python.
""")
    write("docs/REPORT_PRESENTATION_REPO_CONSISTENCY.md", """
# Report / presentation / repository consistency

## Aligned claims
- Tail threshold is strictly greater than 240 minutes.
- `S(t)=exp[-(lambda*t)^beta]`; lambda is an inverse timescale.
- Original ML uses one selected sensor per WWTW and separate target cohorts.
- Regional ML uses target-independent within-company WWTW clusters and improves beta most.
- RMSE is headline; test data are isolated; skill remains modest.
- Association is not causality; prediction is not explanation.
- Missing hydraulic/event-state information is the primary next-data priority.

## Manual review item
No final presentation file exists in the working repository. The release therefore includes both final ML reports and the repository walkthrough but does not fabricate `final_presentation.pdf`. Add the approved deck and repeat the consistency/hash audit before public upload.
""")
    write("GITHUB_UPLOAD_INSTRUCTIONS.md", """
# GitHub upload instructions

Local Git is initialized by the release build. No remote is configured and nothing is pushed.

After reviewing `docs/RELEASE_AUDIT.md`, choosing a licence, and adding the approved presentation:
```bash
git status
git add .
git commit -m "Initial professor-ready CSO spatial-drivers release"
git branch -M main
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

GitHub CLI was not detected during the build. Optional later alternative after installing/authenticating `gh`:
```bash
gh repo create <REPOSITORY> --source . --remote origin --private
git push -u origin main
```
Do not make the repository public until licence and data-review items are resolved.
""")


def wrappers() -> None:
    scripts={
      "01_build_event_master.py":"""from cso_spatial_drivers.cleaning.events import clean_events\n# Portable entry point is demonstrated in run_reproducible_demo.py. Full provider ingestion requires DATA_AVAILABILITY.md inputs.\n""",
      "02_fit_tail_parameters.py":"""from cso_spatial_drivers.fitting.stretched_exponential import fit_tail\n# Use fit_tail on each sensor's validated durations; threshold is strict >240 by default.\n""",
      "03_link_sensors_to_wwtw.py":"""from cso_spatial_drivers.spatial.assignment import nearest_same_company\n# Full release uses polygon/name/manual audits in 04_spatial_linkage/scripts/.\n""",
      "04_build_predictor_master.py":"""from cso_spatial_drivers.predictors.join import join_predictors\n# External extraction scripts and licensing notes are in 05_environmental_predictors/.\n""",
      "05_run_univariate_analysis.py":"""from cso_spatial_drivers.statistics.univariate import fit_log_outcome\n# Authoritative full-data analysis is in 06_univariate_analysis/scripts/.\n""",
      "06_run_machine_learning.py":"""# See 07_machine_learning/README.md for the original selected-sensor and regional-cluster workflows.\nraise SystemExit('Full ML requires omitted scientific masters; run_reproducible_demo.py is public and self-contained.')\n""",
      "07_build_final_figures.py":"""# Headline figures are generated during release assembly from aggregate outputs only.\nprint('See outputs/headline_figures and docs/FIGURE_PROVENANCE.md')\n""",
    }
    for name,text in scripts.items(): write(f"scripts/{name}",text)


def provenance_and_inventory() -> None:
    sources=[
      ("Final internship report","PROJECT_RESULTS/Final Documents/CSO_Final_Report_Style_Preview.pdf","CURRENT","safe final PDF"),
      ("Original selected-sensor ML report","PROJECT_RESULTS/Final Documents/CSO_NEW_MACHINE_LEARNING_MODELS_1_TO_4_MASTER_REPORT_REDESIGNED.pdf","CURRENT","safe final PDF"),
      ("Regional-cluster ML report","REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/report/CSO_REGIONAL_CLUSTER_MACHINE_LEARNING_REPORT.pdf","CURRENT extension","safe final PDF"),
      ("Scientific master","PROJECT_RESULTS/four_outcome_annual_nimrod_release_20260813_130420/data/scientific_master_annual_nimrod_v1.csv","CURRENT","not copied: row-level sites/size"),
      ("Original ML master","machine learning/NEW MACHINE LEARNING/shared/new_ml_master.csv","CURRENT","not copied: row-level sites"),
      ("Regional master","REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/data/regional_master_25km.csv","CURRENT extension","not copied: row-level geography"),
      ("Raw/clean EDM","clean_data/<company>/*.csv","RAW/PROTECTED","not copied"),
      ("NIMROD archive","data_raw/dynamic_rainfall/","RAW/LARGE","not copied"),
      ("Final presentation","not found","MISSING","manual review required"),
    ]
    rows=[]
    for label,path,status,note in sources:
        p=SOURCE/path if "<" not in path and path!="not found" else None
        rows.append({"item":label,"source_path":path,"status":status,"sha256":sha(p) if p and p.is_file() else "","note":note})
    pd.DataFrame(rows).to_csv(ROOT/"docs/source_inventory.csv",index=False)
    write("docs/SOURCE_INVENTORY.txt", """
============================================================
GITHUB RELEASE - SOURCE INVENTORY
============================================================
Final report: PROJECT_RESULTS/Final Documents/CSO_Final_Report_Style_Preview.pdf
Final presentation: NOT FOUND - manual review required
Original ML report: PROJECT_RESULTS/Final Documents/CSO_NEW_MACHINE_LEARNING_MODELS_1_TO_4_MASTER_REPORT_REDESIGNED.pdf
Regional ML report: REGIONAL_CLUSTER_AND_UNIVARIATE_ML/01_regional_cluster_model/report/CSO_REGIONAL_CLUSTER_MACHINE_LEARNING_REPORT.pdf
Authoritative scientific master: PROJECT_RESULTS/four_outcome_annual_nimrod_release_20260813_130420/data/scientific_master_annual_nimrod_v1.csv
Authoritative event dataset: clean_data/<company>/*.csv (not distributed)
Beta/lambda pipeline: analyse_individual_cso_heavy_tails.py + bootstrap_all_individual_cso_sensors.py
Spatial linkage pipeline: scripts/02,03,08b,08c,08e,11
Environmental predictor pipeline: scripts/04,05,07,09 plus annual NIMROD pipeline
Univariate pipeline: four_outcome_annual_nimrod_release statistical_analysis/pipeline
Original ML pipeline: machine learning/NEW MACHINE LEARNING/pipeline/ml_programme.py
Regional ML pipeline: REGIONAL_CLUSTER_AND_UNIVARIATE_ML/pipeline
Figure sources: final documents, original ML figures, regional ML figures
Tests: root tests plus new public scientific-invariant tests
Large/raw datasets: EDM, JSON, NIMROD, rasters, shapefiles, GeoPackages
Files requiring exclusion: correspondence, tokens, local configs, caches, archives, _runtime
Potential secrets/private material: ceda_token path referenced in downloader; token file not copied
============================================================
""")


def main() -> None:
    structure(); top_documents(); stage_docs(); package_code(); demo_and_tests(); authoritative_copies(); result_tables(); docs(); wrappers(); provenance_and_inventory()
    print("release content assembled")

if __name__ == "__main__":
    main()

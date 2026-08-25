# Report-to-code map

| Public material | Section / figure | Source data | Generating code | Released output / key config |
|---|---|---|---|---|
| `Full-report.pdf` | Data and event workflow | cleaned event contract and sensor summaries | `03_tail_parameter_fitting/scripts/analyse_individual_cso_heavy_tails.py` | `config/analysis.yaml`; strict `T>240` |
| `Full-report.pdf` | Beta/lambda fitting | sensor event durations | same plus bootstrap script | beta bounds `[0.01,2]` |
| `Full-report.pdf` | Spatial linkage | sensor/WWTW matching tables | `04_spatial_linkage/scripts/` | same-company, beta-blind rules |
| `Full-report.pdf` | Environmental analysis | scientific master | `05_environmental_predictors/scripts/` | area-weighted EPSG:27700 extraction |
| `Full-report.pdf` | Four-outcome univariate figures | four-outcome summary | `06_univariate_analysis/scripts/build_four_outcome_analyses.py` | HC3/FDR tables |
| Original ML report, models 1-4 | Scorecard, predictions, learning curves, importance | selected-sensor WWTW master | `07_machine_learning/scripts/original_selected_sensor/ml_programme.py` | target-specific 70/15/15 configs |
| Regional ML report, sections 3-6 | Four regional models | 25 km regional master | `07_machine_learning/scripts/regional_cluster/01_run_regional_analysis.py` | `regional_experiment.json` |
| Regional ML report, section 7 | 10/25/50 km sensitivity | regional masters | `06_exact_event_medians_and_radius_models.py` | fixed radii; 25 km primary |
| Regional ML report, section 9 | Hierarchical benchmark | multi-sensor WWTW rows + regions | `03_run_hierarchical_benchmark.py` | singular fits reported honestly |
| Regional ML report, section 10 | Beta-logit diagnostic | finite beta/lambda pairs | original diagnostic source retained in extension | beta upper bound 2; boundary fits excluded |

The 23-page final report uses page-based narrative rather than stable numbered figure labels. No fabricated figure numbering is assigned here.

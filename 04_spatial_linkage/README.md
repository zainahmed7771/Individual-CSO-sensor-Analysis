# 04 – Sensor-to-WWTW linkage

**Question:** which accepted treatment-work context is assigned to each sensor?

Spatial containment, name similarity and distance are evidence; none proves underground hydraulic connectivity. Selection must be outcome-blind. The scientific-core assignment file is frozen before beta, capacity and environmental fields are joined.

## Validate a frozen assignment contract

```powershell
python scripts\03_link_sensors_to_wwtw.py --config config\demo.yaml
```

Each row requires unique `sensor_uid`, `company`, `uwwCode` and non-empty `assignment_evidence`.

## Rebuild upstream evidence

```powershell
python 04_spatial_linkage\scripts\assign_sensors_to_catchments.py --project-root "D:\path\to\analysis_workspace" --all-companies
python 04_spatial_linkage\scripts\rebuild_name_matching.py --project-root "D:\path\to\analysis_workspace"
python 04_spatial_linkage\scripts\select_primary_sensor.py --project-root "D:\path\to\analysis_workspace"
python 04_spatial_linkage\scripts\build_final_manual_release.py --project-root "D:\path\to\analysis_workspace" --manual-matches "D:\path\to\matching.txt"
```

Read `matching_rules.md` before changing any threshold or manual decision. Preserve rejected and ambiguous alternatives in audit outputs.

Validation:

```powershell
pytest 04_spatial_linkage\tests -q
```

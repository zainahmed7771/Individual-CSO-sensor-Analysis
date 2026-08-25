# 05 – Environmental and operational predictors

**Question:** which catchment characteristics can be joined consistently without changing the accepted WWTW population?

Install GIS dependencies with `python -m pip install -e ".[full,test]"`. Calculate areas and exact overlap weights in EPSG:27700. Never use centroid sampling when a polygon-overlap statistic is specified.

## Upstream extraction commands

```powershell
python 05_environmental_predictors\scripts\extract_hydrogeology.py --project-root "D:\path\to\analysis_workspace"
python 05_environmental_predictors\scripts\build_static_rainfall.py --project-root "D:\path\to\analysis_workspace"
python 05_environmental_predictors\scripts\build_wwtw_capacity.py --project-root "D:\path\to\analysis_workspace"
python 05_environmental_predictors\scripts\build_sensor_environment_master.py --project-root "D:\path\to\analysis_workspace"
```

Static rainfall uses existing HadUK-Grid 1991–2020 annual and winter climatologies. Wet-day fields are deferred unless official threshold climatologies are supplied; totals are never used as wet-day proxies.

## Maintained row-preserving join

```powershell
python scripts\04_build_predictor_master.py --config config\demo.yaml
```

Output: `<output_root>/03_master/scientific_master.csv`. Validate unique `company + uwwCode`, unchanged row order, units, coverage, composition totals and absence of target leakage.

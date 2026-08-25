# Data acquisition checklist

Use this checklist only for a full scientific reconstruction. The included teaching run requires no downloads.

## Before downloading

1. Read `DATA_AVAILABILITY.md` and `docs/DATA_PROVENANCE.md`.
2. Create an approved local data directory outside the Git clone.
3. Copy `config/paths.example.yaml` to `config/paths.yaml` and enter local paths.
4. Do not place passwords, CEDA tokens, emails or raw provider archives in Git.

## Source checklist

| Source family | Obtain | Verify before processing | Expected hand-off |
|---|---|---|---|
| Company EDM | Raw event exports and monitoring-window evidence for every included sensor | provider, coverage dates, timezone, stable permit/location identifier, duplicates and missing stop times | standard event CSV plus monitoring-exposure CSV |
| Company locations | Sensor name/permit/coordinates export | coordinate order/CRS, company identity, stable sensor key | location table used by spatial linkage |
| WWTW and catchments | Exact approved polygon release and `uwwCode` lookup used by the project | provider, edition, geometry validity, CRS and redistribution terms | validated EPSG:27700 polygons and lookup |
| Hydrogeology | BGS Hydrogeology 625k | version, attribution, category lookup and valid coverage | `HydrogeologyUK_IoM_v5.gpkg` or equivalent declared path |
| Static rainfall | HadUK-Grid 1 km 1991–2020 annual and seasonal rainfall climatologies | NetCDF variables, `mm` units, period, BNG grid and 1 km alignment | the two filenames declared in `build_static_rainfall.py` |
| Dynamic rainfall | NIMROD 1 km composite files for audited years, only if reproducing annual radar indices | CEDA terms, timestamp convention, units, completeness and checksum | year-partitioned archive outside Git |
| Population | Exact 2021 population source used for allocation | product/version, geography, population field and allocation method | catchment-level density with coverage diagnostics |
| Terrain | Confirmed OS elevation product | edition, vertical units, grid spacing, voids, BNG alignment | elevation/slope source outside Git |
| Land cover | Original product used by the project | provider, edition, class mapping and licence | developed-land percentage plus coverage audit |
| Building age | Original property/building product | provider, snapshot date, age-band crosswalk, missingness and licence | pre-1973 percentage plus coverage audit |
| Capacity | Exact Waterbase/UWWTD extract used by the project | source URL/version, `uwwCode`, year, load and design-capacity definitions | consolidated Waterbase CSV and catchment lookup |

## Provenance manifest template

Create a local CSV with these columns:

```text
dataset_id,provider,product_title,source_url,edition,coverage,acquired_at,original_filename,bytes,sha256,crs,resolution,units,licence,attribution,transform_notes,reviewed_by
```

The path may be stored in the run notes, but the raw-data location should not be committed.

## Gate before analysis

Proceed only when:

- every configured path exists;
- each table satisfies its documented key and schema;
- every spatial layer has a declared CRS and valid geometry;
- source checksums and terms are recorded;
- manual matches state their evidence and were made without seeing model outcomes;
- `python scripts/check_inputs.py --config config/scientific.yaml` passes.

Missing raw data must fail the scientific run. It must never be replaced by synthetic, zero-filled or proxy values. The synthetic teaching run remains deliberately separate under `config/demo.yaml`.

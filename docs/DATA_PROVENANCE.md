# Data provenance and acquisition

This catalogue separates the source licence from this repository's MIT licence. The MIT licence covers repository-authored code and documentation only. A student must retain the provider citation, terms and source-file checksum for every external dataset used in a scientific run.

| Dataset | Verified provider / product | Period or vintage used | Expected local path | Source access and redistribution | Included here | Downstream use |
|---|---|---:|---|---|---|---|
| EDM spill events | Nine English water companies; public dashboards or EIR exports | approximately 2020–2026, company dependent | `clean_data/<company>/*.csv` or a configured standard-event path | Provider-specific; verify each export before redistribution | No | Standardised event history and all four outcomes |
| Sensor/location JSON | Water-company location exports | company dependent | `json_data/*.json` | Provider-specific; verify before redistribution | No | Sensor identity, coordinates and linkage evidence |
| WWTW/catchment polygons | Project-assembled wastewater catchments and lookup | project snapshot | `data_raw/wastewater_catchments/` | Source chain is not fully recoverable from the public files; a scientific rebuild is blocked until the local polygon source and terms are recorded | No | Outcome-blind linkage and catchment area weights |
| Hydrogeology | [BGS Hydrogeology 625k](https://www.bgs.ac.uk/datasets/hydrogeology-625k/) | local `HydrogeologyUK_IoM_v5` snapshot | `data_raw/hydrogeology/HydrogeologyUK_IoM_v5.gpkg` | BGS describes the dataset as free under the Open Government Licence; retain BGS attribution and confirm the downloaded version | No | Exact polygon-overlap composition |
| Static rainfall | [Met Office HadUK-Grid](https://www.metoffice.gov.uk/research/climate/maps-and-data/data/haduk-grid/overview), distributed through CEDA | 1991–2020 annual and seasonal climatologies | `data_raw/rainfall/rainfall_hadukgrid_uk_1km_ann-30y_199101-202012.nc` and `...seas-30y_199101-202012.nc` | Register/download from CEDA and retain catalogue citation and terms | No | Exact partial-cell area-weighted annual and winter rainfall |
| Annual rainfall indices | Met Office 1 km UK NIMROD radar composite archive; [CEDA processing documentation](https://artefacts.ceda.ac.uk/badc_datadocs/nimrod/reprocessing_2011.html) | audited monitoring years | `data_raw/dynamic_rainfall/` | CEDA registration/terms and a large archive; do not commit credentials or raw archive | No | Annual PRCPTOT, Rx1h/Rx6h/Rx24h, SDII, R10/R20, CWD and CDD |
| Land cover | Historical project spatial source | source not proven by the public release | `data_raw/land_cover/` | **Unverified:** do not substitute a convenient product without recording product, edition, class crosswalk, CRS and licence | No | Developed-land measures |
| Population | [ONS Census 2021 population density, TS006](https://www.ons.gov.uk/datasets/TS006/editions/2021/versions/1), or the exact local allocation source if different | 2021 | `data_raw/population/` | ONS material is generally published under OGL, but confirm that the actual local file is TS006 and record its geography/version | No | Population density |
| Building age | Historical project property/building source | source not proven by the public release | `data_raw/building_age/` | **Unverified:** scientific reconstruction is blocked until the original product, age-band mapping and reuse terms are documented | No | Pre-1973 building share |
| Terrain | [Ordnance Survey OS Terrain 50](https://www.ordnancesurvey.co.uk/products/os-terrain-50), if confirmed against the local file | supplied project vintage | `data_raw/elevation/` | OS OpenData product; retain attribution and confirm product/edition before use | No | Catchment mean slope in EPSG:27700 |
| Treatment capacity | Local consolidated Waterbase/Urban Waste Water Treatment records | latest valid record per `uwwCode` | `data_raw/waterbase/waterbase_consolidated.csv` | The public release does not prove the original download URL/version; record them before a new scientific release | No | Latest load/design population equivalent and capacity ratio |

## Required provenance record for a scientific run

For each supplied source, add a row to a local, version-controlled-without-data manifest containing:

- provider and exact product title;
- direct landing-page or catalogue URL;
- edition, release date and temporal coverage;
- original filename, byte size and SHA-256 checksum;
- CRS, spatial resolution and units where applicable;
- licence/access conditions and required attribution;
- acquisition date and the person who checked it;
- any class crosswalk, filtering or manual transformation before the repository script.

`config/paths.example.yaml` is a path catalogue, not a provenance record. Copy it to a machine-local `config/paths.yaml`; never commit credentials, correspondence or restricted data.

## Rainfall rule

The static stage uses only the official HadUK-Grid 1991–2020 annual and seasonal climatologies already supplied locally. It does not download daily rainfall. Official precomputed 1 mm or 10 mm rain-day climatology variables may be added through the same area-weighted extraction function after their metadata are validated. Until then, wet-day variables are deferred and no totals, zeros or approximations are emitted.

## Known release boundary

The included synthetic data are sufficient for the executable teaching demonstration. A fully faithful raw-source reconstruction additionally requires the omitted provider files and manual linkage evidence. In particular, the catchment, land-cover, building-age and treatment-capacity source chain must be completed locally before claiming a new independent scientific reproduction.

No private EIR contact details or access tokens are included.

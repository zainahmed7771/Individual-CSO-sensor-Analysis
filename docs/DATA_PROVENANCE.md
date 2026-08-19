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

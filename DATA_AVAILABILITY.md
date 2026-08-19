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

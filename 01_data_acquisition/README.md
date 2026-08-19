# 01 Data Acquisition

## Scientific question
Where do the EDM and external records come from?

## Inputs
Provider exports, year coverage, source schemas and download notes.

## Code to run
Company-specific acquisition/standardisation scripts plus schemas.

## Outputs
Locally stored immutable source files and manifests; no raw data are shipped.

## QC and validation
Coverage, checksum, schema and licence review.

## Connection to the next stage
Validated source files enter common event cleaning.

The public demo exercises the portable core. Full-scale scripts require omitted source data described in `../DATA_AVAILABILITY.md`.

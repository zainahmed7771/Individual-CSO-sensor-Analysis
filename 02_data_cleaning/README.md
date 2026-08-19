# 02 Data Cleaning

## Scientific question
How are incompatible provider records turned into one event contract?

## Inputs
Provider event tables with permit, location and timestamps.

## Code to run
Portable cleaning functions and authoritative preparation scripts.

## Outputs
One row per valid event with stable sensor UID and duration minutes.

## QC and validation
End >= start, deduplication, finite non-negative duration and schema checks.

## Connection to the next stage
Clean events feed tail fitting and observed duration/frequency outcomes.

The public demo exercises the portable core. Full-scale scripts require omitted source data described in `../DATA_AVAILABILITY.md`.

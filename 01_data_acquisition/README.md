# 01 – Data acquisition

**Question:** where did each event or external variable originate, and may it be redistributed?

Run from the repository root. Store downloads outside Git or in ignored raw-data directories. Preserve original filenames and hashes; write standardised derivatives elsewhere.

## Required output contract

Provider events must ultimately contain `company`, `permit_number`, `location_name`, `start_time` and `stop_time`. See `data/schemas/event_schema.csv`.

## Thames JSON example

```powershell
python 01_data_acquisition\scripts\prepare_thames_company_data.py events --project-root "D:\path\to\analysis_workspace"
```

Coordinate preparation is a separate audited step:

```powershell
python 01_data_acquisition\scripts\prepare_thames_company_data.py coordinates --project-root "D:\path\to\analysis_workspace"
```

## Validation

Record provider, years, access date, licence, original checksum, row count and schema mapping. Never edit the source export in place. Continue to stage 02 only when every company can be mapped to the standard contract.

See `docs/DATA_PROVENANCE.md`, `DATA_AVAILABILITY.md` and the source tracker workbook.

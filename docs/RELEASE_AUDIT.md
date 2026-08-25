# Release audit

Audit date: 2026-08-26

| Category | Check | Status | Detail |
|---|---|---|---|
| structure | required files and stage READMEs | **PASS** | all present |
| science | authoritative ML scorecard | **PASS** | 8 rows; original and regional sentinel metrics match |
| science | tail definitions | **PASS** | strict >240, lower beta heavier, lambda inverse-timescale |
| code | scientific-invariant tests | **PASS** | 45 passed; return code 0 |
| code | reproducible end-to-end teaching pipeline | **PASS** | six-stage synthetic workflow and manifest present |
| documentation | walkthrough PDF render | **PASS** | 24 pages; no empty pages; page-by-page visual QA complete |
| documentation | walkthrough TeX compilation | **PASS** | pdfTeX 3.141592653-2.6-1.40.28 (MiKTeX 25.12); 24 pages |
| documentation | relative Markdown links | **PASS** | no broken local links |
| safety | secrets/private paths/email scan | **PASS** | no matches |
| safety | files over 50 MB | **PASS** | none |
| safety | raw raster/GIS/database exclusion | **PASS** | none included |
| safety | cache/temp directories | **PASS** | none |
| provenance | full report present and hashed | **PASS** | 0d688ec4a23a45ce7a0fb8473c813ca1105635951c1be98282c88f6a874a4326 |
| licensing | repository licence | **PASS** | MIT licence present; external data retain provider terms |

## Release decision

The validator checks scientific definitions, executable teaching outputs, documentation links, licensing, privacy and release structure.

Status: **READY FOR REVIEW AND PUBLIC USE.**

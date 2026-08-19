# Release audit

Audit date: 19 August 2026

| Category | Check | Status | Detail |
|---|---|---|---|
| structure | required files and stage READMEs | **PASS** | all present |
| science | authoritative ML scorecard | **PASS** | 8 rows; original and regional sentinel metrics match |
| science | tail definitions | **PASS** | strict >240, lower beta heavier, lambda inverse-timescale |
| code | scientific-invariant tests | **PASS** | 9 passed; return code 0 |
| code | reproducible synthetic demo | **PASS** | quick workflow completed and output label verified |
| documentation | walkthrough PDF render | **PASS** | 24 pages; no empty pages; page-by-page visual QA complete |
| documentation | walkthrough TeX compilation | **PASS** | pdfTeX 3.141592653-2.6-1.40.28 (MiKTeX 25.12); 24 pages |
| documentation | relative Markdown links | **PASS** | no broken local links |
| safety | secrets/private paths/email scan | **PASS** | no matches |
| safety | files over 50 MB | **PASS** | none |
| safety | raw raster/GIS/database exclusion | **PASS** | none included |
| safety | cache/temp directories | **PASS** | none |
| provenance | final report SHA-256 | **PASS** | d9bf2fc4cd368322e6a8e4f1ba9d070f7fec572e6546c63f8692554045fa4fef |
| manual | final presentation | **MANUAL REVIEW REQUIRED** | no authoritative presentation found; no report was relabelled |
| manual | repository licence | **MANUAL REVIEW REQUIRED** | choose and approve a licence before public upload |

## Public-upload decision

The release contains no detected secrets, private contact material, giant raw data, licensed rasters or machine-local absolute paths. Scientific definitions and sentinel metrics match the authoritative outputs.

Two non-sensitive actions remain before a public GitHub upload:

1. supply the approved final presentation (none exists in the working repository);
2. choose and approve a repository licence.

Status: **READY FOR MANUAL REVIEW; NOT YET LICENSED FOR PUBLIC UPLOAD.**

"""Portable permit normalisation and coordinate-ingestion contracts."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_permit(value: object) -> str:
    return re.sub(r"\s+", "", str(value).strip().upper())


def alphanumeric_permit(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_permit(value))


def _valid_bng(easting: float, northing: float) -> bool:
    return np.isfinite(easting) and np.isfinite(northing) and 0 <= easting <= 700_000 and 0 <= northing <= 1_300_000


def read_json_company(path: Path, company: str, _records: list, issues: list, _logger):
    frame = pd.read_json(path)
    rows = []
    for index, row in frame.iterrows():
        permit = str(row.get("PermitNumber", "")).strip()
        x = float(row.get("X", np.nan)); y = float(row.get("Y", np.nan))
        correction = ""
        if not _valid_bng(x, y) and _valid_bng(y, x):
            x, y = y, x
            correction = "unambiguous_easting_northing_reversal_corrected"
            issues.append({"company": company, "permit_number": permit, "issue": correction})
        rows.append({
            "_json_id": index,
            "company": company,
            "permit_number_json_original": permit,
            "permit_number_normalized": normalize_permit(permit),
            "permit_number_alphanumeric": alphanumeric_permit(permit),
            "location_name": str(row.get("LocationName", "")),
            "bng_easting": x,
            "bng_northing": y,
            "coordinate_valid": _valid_bng(x, y),
            "coordinate_correction": correction,
        })
    sensors = pd.DataFrame(rows)
    return sensors, pd.DataFrame(), pd.DataFrame(issues)


def match_company(locations: pd.DataFrame, beta: pd.DataFrame, company: str, ambiguous: list, conflicts: list):
    locations = locations.copy().reset_index(drop=True)
    locations["_json_id"] = locations.index
    locations["permit_match_status"] = "unmatched"
    locations["_beta_id"] = np.nan
    matched_beta: set[object] = set()
    for _, beta_row in beta.iterrows():
        beta_id = beta_row["_beta_id"]
        exact = locations["permit_number_json_original"].astype(str).eq(str(beta_row["permit_number_csv_original"]))
        normalized = locations["permit_number_normalized"].eq(normalize_permit(beta_row["permit_number_csv_original"]))
        alpha = locations["permit_number_alphanumeric"].eq(alphanumeric_permit(beta_row["permit_number_csv_original"]))
        candidates = exact if exact.sum() else normalized if normalized.sum() else alpha
        count = int(candidates.sum())
        if count == 1:
            index = locations.index[candidates][0]
            locations.loc[index, ["permit_match_status", "_beta_id"]] = ["matched", beta_id]
            matched_beta.add(beta_id)
        elif count > 1:
            locations.loc[candidates, "permit_match_status"] = "ambiguous"
            ambiguous.append({"company": company, "beta_id": beta_id, "candidate_count": count})
    unmatched_beta = beta.loc[~beta["_beta_id"].isin(matched_beta)].copy()
    return locations, pd.DataFrame(conflicts), unmatched_beta

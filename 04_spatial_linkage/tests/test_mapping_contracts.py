from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from build_all_individual_cso_beta_maps import (
    alphanumeric_permit,
    match_company,
    normalize_permit,
    read_json_company,
)


def test_ambiguous_alphanumeric_permit_is_not_accepted() -> None:
    originals = ["AB-1", "A B1"]
    locations = pd.DataFrame(
        {
            "company": ["test", "test"],
            "permit_number_json_original": originals,
            "permit_number_normalized": [normalize_permit(v) for v in originals],
            "permit_number_alphanumeric": [alphanumeric_permit(v) for v in originals],
            "coordinate_valid": [True, True],
        }
    )
    beta_original = "AB.1"
    beta = pd.DataFrame(
        {
            "_beta_id": [0],
            "_join_key_conflict": [False],
            "permit_number_csv_original": [beta_original],
            "permit_number_exact": [beta_original],
            "permit_number_normalized": [normalize_permit(beta_original)],
            "permit_number_alphanumeric": [alphanumeric_permit(beta_original)],
            "fitted_beta": [0.5],
        }
    )
    ambiguous: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []

    matched, _, unmatched_beta = match_company(
        locations, beta, "test", ambiguous, conflicts
    )

    json_rows = matched.loc[matched["_json_id"].notna()]
    assert set(json_rows["permit_match_status"]) == {"ambiguous"}
    assert json_rows["_beta_id"].isna().all()
    assert len(unmatched_beta) == 1
    assert ambiguous
    assert not conflicts


def write_coordinate_json(path: Path, permit: str, x: float, y: float) -> None:
    payload = {
        "PermitNumber": {"0": permit},
        "X": {"0": x},
        "Y": {"0": y},
        "LocationName": {"0": "Fixture"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bng_coordinate_reversal_occurs_when_unambiguous(tmp_path: Path) -> None:
    path = tmp_path / "reversed.json"
    # X=1,000,000 is outside the broad easting bound, while swapping gives a
    # valid BNG point (E=600,000, N=1,000,000) inside broad UK bounds.
    write_coordinate_json(path, "P1", 1_000_000, 600_000)
    issues: list[dict[str, object]] = []

    sensors, _, _ = read_json_company(
        path, "test", [], issues, logging.getLogger("test")
    )

    row = sensors.iloc[0]
    assert row["coordinate_correction"] == (
        "unambiguous_easting_northing_reversal_corrected"
    )
    assert np.isclose(row["bng_easting"], 600_000)
    assert np.isclose(row["bng_northing"], 1_000_000)
    assert bool(row["coordinate_valid"])


def test_in_bounds_bng_coordinates_are_not_reversed(tmp_path: Path) -> None:
    path = tmp_path / "in_bounds.json"
    write_coordinate_json(path, "P1", 600_000, 1_000_000)

    sensors, _, _ = read_json_company(
        path, "test", [], [], logging.getLogger("test")
    )

    row = sensors.iloc[0]
    assert row["coordinate_correction"] == ""
    assert np.isclose(row["bng_easting"], 600_000)
    assert np.isclose(row["bng_northing"], 1_000_000)


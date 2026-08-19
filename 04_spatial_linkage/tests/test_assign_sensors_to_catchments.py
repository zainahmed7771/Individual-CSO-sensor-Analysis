from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "03_assign_sensors_to_catchments.py"
)
SPEC = importlib.util.spec_from_file_location("assign_catchments", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sensor_row(uid: str, eligible: bool) -> dict[str, object]:
    return {
        "sensor_uid": uid,
        "company": "yorkshire",
        "permit_number": uid,
        "location_name": uid,
        "bng_easting": 100.0 if eligible else None,
        "bng_northing": 100.0 if eligible else None,
        "longitude": None,
        "latitude": None,
        "fitted_beta": 0.7,
        "beta_ci_lower_95": 0.6,
        "beta_ci_upper_95": 0.8,
        "beta_ci_width": 0.2,
        "tail_spill_count": 20,
        "beta_quality": "exploratory_beta",
        "eligible_for_beta_analysis": False,
        "eligible_for_catchment_matching": eligible,
        "coordinate_status": "valid" if eligible else "missing_coordinates",
        "issue_detail": "" if eligible else "No accepted coordinate",
    }


def relation(uid: str, identifier: str, code: str) -> dict[str, object]:
    return {
        "sensor_uid": uid,
        "sensor_company": "yorkshire",
        "identifier": identifier,
        "catchment_company": "yorkshire_water",
        "catchment_name": identifier,
        "eligible_for_beta_analysis": False,
        "relation_type": "same_company_within",
        "candidate_uwwcodes": code,
        "candidate_uwwnames": "TEST STW",
        "candidate_uwwcode_count": 1,
    }


def test_sorted_pipe_is_deterministic_and_unique() -> None:
    assert MODULE.sorted_pipe(["b", "a", "b", None, ""]) == "a|b"


def test_existing_products_require_explicit_overwrite(tmp_path: Path) -> None:
    targets = MODULE.resolve_output_targets(tmp_path)
    targets["assignments"].parent.mkdir(parents=True)
    targets["assignments"].write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        MODULE.enforce_overwrite_policy(targets, overwrite=False)

    MODULE.enforce_overwrite_policy(targets, overwrite=True)


def test_classification_retains_noneligible_and_accepts_unique_containment() -> None:
    sensors = pd.DataFrame(
        [sensor_row("eligible", True), sensor_row("excluded", False)]
    )
    points = gpd.GeoDataFrame(
        sensors,
        geometry=[Point(100, 100), None],
        crs="EPSG:27700",
    )
    catchments = gpd.GeoDataFrame(
        {
            "identifier": ["catchment-a"],
            "company": ["yorkshire_water"],
            "name": ["A"],
            "comment": [None],
        },
        geometry=[Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])],
        crs="EPSG:27700",
    )
    relations = pd.DataFrame([relation("eligible", "catchment-a", "UWW-1")])

    result = MODULE.classify_assignments(
        sensors, points, relations, {}, catchments
    ).set_index("sensor_uid")

    assert result.loc["eligible", "assignment_status"] == (
        "unique_same_company_containment"
    )
    assert result.loc["eligible", "uwwCode"] == "UWW-1"
    assert not result.loc["eligible", "manual_review_required"]
    assert result.loc["excluded", "assignment_status"] == (
        "not_eligible_for_catchment_matching"
    )


def test_multiple_polygons_with_one_shared_uwwcode_is_wwtw_only() -> None:
    sensors = pd.DataFrame([sensor_row("overlap", True)])
    points = gpd.GeoDataFrame(
        sensors, geometry=[Point(100, 100)], crs="EPSG:27700"
    )
    catchments = gpd.GeoDataFrame(
        {
            "identifier": ["a", "b"],
            "company": ["yorkshire_water", "yorkshire_water"],
            "name": ["A", "B"],
            "comment": [None, None],
        },
        geometry=[
            Polygon([(0, 0), (200, 0), (200, 200), (0, 200)]),
            Polygon([(50, 50), (250, 50), (250, 250), (50, 250)]),
        ],
        crs="EPSG:27700",
    )
    relations = pd.DataFrame(
        [relation("overlap", "a", "UWW-1"), relation("overlap", "b", "UWW-1")]
    )

    row = MODULE.classify_assignments(
        sensors, points, relations, {}, catchments
    ).iloc[0]

    assert row["assignment_status"] == "multiple_same_uwwcode"
    assert row["raw_catchment_identifier"] == ""
    assert row["candidate_catchment_identifiers"] == "a|b"
    assert row["uwwCode"] == "UWW-1"

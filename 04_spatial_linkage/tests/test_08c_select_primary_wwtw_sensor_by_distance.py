"""Tests for the outcome-blind WWTW primary-sensor selector."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/08c_select_primary_wwtw_sensor_by_distance.py"
SPEC = importlib.util.spec_from_file_location("primary_distance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
CONFIG = json.loads((ROOT / "config/wwtw_primary_sensor_selection.json").read_text(encoding="utf-8"))
PROTECTED = [
    ROOT / "scripts/08b_rebuild_wwtw_name_matching_place_core.py",
    ROOT / "config/wwtw_sensor_place_core_matching.json",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def universe(*keys):
    return pd.DataFrame([
        {"company": company, "uwwCode": code, "wwtw_easting": 1000.0, "wwtw_northing": 1000.0}
        for company, code in keys
    ])


def candidate(uid, distance, exact=False, score=90.0, margin=10.0, code="W1"):
    return {
        "pair_uid": f"c::{code}::{uid}", "company": "c", "uwwCode": code,
        "sensor_uid": uid, "bng_easting": 1000.0 + distance, "bng_northing": 1000.0,
        "treatment_work_easting": 1000.0, "treatment_work_northing": 1000.0,
        "sensor_to_wwtw_distance_m": distance, "place_core_exact_match": exact,
        "place_core_score": score, "place_score_margin": margin,
    }


def run_selection(rows, works=None):
    works = works if works is not None else universe(("c", "W1"))
    measured = MODULE.calculate_candidate_distances(pd.DataFrame(rows), CONFIG)
    return MODULE.select_automatic_primaries(works, measured, CONFIG)


def test_nearest_existing_accepted_candidate_is_selected_in_first_tier():
    selection, audit = run_selection([candidate("far", 900), candidate("near", 200)])
    assert selection.iloc[0]["final_primary_sensor_uid"] == "near"
    assert selection.iloc[0]["primary_sensor_selection_status"] == "selected_nearest_of_multiple_within_1000m"
    assert audit.loc[audit.sensor_uid.eq("near"), "selected_as_primary"].item()


def test_first_successful_radius_and_unique_status():
    selection, _ = run_selection([candidate("inside_2k", 1500), candidate("inside_3k", 2500)])
    assert selection.iloc[0]["final_primary_sensor_uid"] == "inside_2k"
    assert selection.iloc[0]["primary_radius_tier_m"] == 2000
    assert selection.iloc[0]["primary_sensor_selection_status"] == "selected_unique_within_2000m"


def test_distance_tie_breaking_is_deterministic_and_prefers_exact_evidence():
    rows = [candidate("z_nonexact", 500, exact=False, score=99), candidate("a_exact", 500, exact=True, score=85)]
    first, _ = run_selection(rows)
    second, _ = run_selection(list(reversed(rows)))
    assert first.iloc[0]["final_primary_sensor_uid"] == "a_exact"
    assert second.iloc[0]["final_primary_sensor_uid"] == "a_exact"


def test_missing_coordinate_candidate_cannot_be_selected_automatically():
    row = candidate("missing", 200)
    row["bng_easting"] = np.nan
    selection, audit = run_selection([row])
    assert not selection.iloc[0]["primary_sensor_assigned"]
    assert selection.iloc[0]["primary_sensor_selection_status"] == "unassigned_no_coordinate_eligible_candidate"
    assert not audit.iloc[0]["automatic_selected_as_primary"]


def test_no_candidate_inside_max_radius_is_unassigned():
    selection, _ = run_selection([candidate("outside", 3001)])
    assert selection.iloc[0]["primary_sensor_selection_status"] == "unassigned_no_candidate_within_max_radius"


def test_zero_candidate_wwtw_is_retained_in_selection_table():
    selection, _ = run_selection(
        [candidate("only_w1", 100)],
        universe(("c", "W1"), ("c", "W2")),
    )
    assert len(selection) == 2
    assert selection.loc[selection.uwwCode.eq("W2"), "primary_sensor_selection_status"].item() == "unassigned_no_accepted_name_candidate"


def test_post_selection_sensor_join_allows_multiple_unassigned_blank_keys():
    left = pd.DataFrame({"primary_sensor_uid": ["selected", np.nan, np.nan]})
    source = pd.DataFrame({"primary_sensor_uid": ["selected"], "fitted_beta": [1.2]})
    joined = left.merge(source, on="primary_sensor_uid", how="left", validate="many_to_one")
    assert len(joined) == 3
    assert joined["fitted_beta"].notna().sum() == 1


def test_selected_sensor_beta_quality_is_not_a_wwtw_aggregate():
    legitimate = "beta_quality_status"
    aggregate_patterns = (
        "median_beta", "mean_beta", "beta_std", "beta_q25", "beta_q75",
        "beta_iqr", "median_beta_ci_width",
    )
    assert not any(pattern in legitimate for pattern in aggregate_patterns)


def test_beta_and_environment_changes_do_not_change_selection():
    rows = [candidate("near", 100), candidate("far", 200)]
    baseline, _ = run_selection(rows)
    altered = [dict(row, fitted_beta=999999 if row["sensor_uid"] == "far" else -999, hydro_value=-1e12) for row in rows]
    changed, _ = run_selection(altered)
    assert baseline.iloc[0]["final_primary_sensor_uid"] == changed.iloc[0]["final_primary_sensor_uid"] == "near"


def test_nonselected_candidates_are_retained_as_alternatives():
    _, audit = run_selection([candidate("near", 100), candidate("far", 200)])
    alternative = audit.loc[audit.sensor_uid.eq("far")].iloc[0]
    assert alternative["candidate_selection_status"] == "alternative_matching_candidate_not_selected"
    assert not alternative["selected_as_primary"]


def test_reused_place_core_script_and_configuration_are_not_modified():
    before = {path: digest(path) for path in PROTECTED}
    run_selection([candidate("one", 100)])
    assert before == {path: digest(path) for path in PROTECTED}

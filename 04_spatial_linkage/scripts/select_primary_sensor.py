#!/usr/bin/env python3
"""Select zero or one analytical primary sensor per authoritative WWTW.

The candidate universe is the existing accepted place-core pair master.
Distance in EPSG:27700 discriminates only among those already accepted pairs.
Response, capacity, and environmental fields are joined after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


LOGGER = logging.getLogger("wwtw_primary_sensor_selection")
BNG = "EPSG:27700"
MANUAL_FIELDS = [
    "manual_decision",
    "manual_selected_sensor_uid",
    "manual_reviewer",
    "manual_review_date",
    "manual_notes",
]
NUMERIC_BETA_FIELDS = [
    "fitted_beta",
    "beta_ci_lower_95",
    "beta_ci_upper_95",
    "beta_ci_width",
]
OUTCOME_DERIVED_WWTW_PATTERNS = (
    "beta",
    "spill",
    "assigned_sensor_count",
    "coordinate_assigned_sensor_count",
    "minimum_primary_sensor_count_flag",
)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    characteristics: Path
    sensor_master: Path
    accepted_pairs: Path
    unmatched_review: Path
    place_gpkg: Path
    place_script: Path
    place_config: Path
    branch: Path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/wwtw_primary_sensor_selection.json"),
    )
    parser.add_argument("--run-id", help="Unique run directory name; timestamped when omitted.")
    parser.add_argument("--manual-decisions", type=Path)
    return parser.parse_args(argv)


def clean_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_string(value).upper() in {"TRUE", "T", "YES", "Y", "1"}


def valid_bng(easting: object, northing: object) -> bool:
    try:
        east, north = float(easting), float(northing)
    except (TypeError, ValueError):
        return False
    return math.isfinite(east) and math.isfinite(north) and 0 <= east <= 700_000 and 0 <= north <= 1_300_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256(path) for path in sorted(set(paths)) if path.is_file()}


def require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def resolve_paths(root: Path, config_path: Path) -> ProjectPaths:
    root = root.resolve()
    config = config_path if config_path.is_absolute() else root / config_path
    return ProjectPaths(
        root=root,
        config=config.resolve(),
        characteristics=root / "data_intermediate/catchment_characteristics/catchment_characteristics_master.csv",
        sensor_master=root / "data_intermediate/sensor_master/cso_sensor_master.csv",
        accepted_pairs=root / "project_results/new_sensor_assignment_approach/tables/wwtw_sensor_analysis_master.csv",
        unmatched_review=root / "project_results/new_sensor_assignment_approach/tables/wwtw_sensor_name_match_review.csv",
        place_gpkg=root / "project_results/new_sensor_assignment_approach/qgis/wwtw_sensor_name_matching.gpkg",
        place_script=root / "scripts/08b_rebuild_wwtw_name_matching_place_core.py",
        place_config=root / "config/wwtw_sensor_place_core_matching.json",
        branch=root / "project_results/new_sensor_assignment_approach_primary_spatial",
    )


def load_configuration(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ["radius_tiers_m", "max_radius_m", "near_tie_gap_m", "crs", "selection_method"]
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Primary-selection configuration lacks: {missing}")
    tiers = [float(value) for value in config["radius_tiers_m"]]
    if not tiers or tiers != sorted(set(tiers)) or tiers[-1] != float(config["max_radius_m"]):
        raise ValueError("radius_tiers_m must be unique, increasing, and end at max_radius_m")
    if config["crs"] != BNG:
        raise ValueError(f"Distance calculations require {BNG}")
    config["radius_tiers_m"] = [int(value) if value.is_integer() else value for value in tiers]
    return config


def load_place_module(path: Path):
    spec = importlib.util.spec_from_file_location("wwtw_place_core_reuse", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load place-core utilities: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def protected_place_files(paths: ProjectPaths) -> list[Path]:
    protected = [paths.place_script, paths.place_config]
    place_branch = paths.accepted_pairs.parents[1]
    protected.extend(path for path in place_branch.rglob("*") if path.is_file())
    return sorted(set(protected))


def load_authoritative_wwtw_universe(paths: ProjectPaths) -> tuple[pd.DataFrame, gpd.GeoDataFrame, list[str]]:
    characteristics = pd.read_csv(paths.characteristics, low_memory=False)
    require_columns(characteristics, ["company", "uwwCode", "uwwName"], "WWTW characteristics")
    characteristics["company"] = characteristics["company"].map(clean_string)
    characteristics["uwwCode"] = characteristics["uwwCode"].map(clean_string)
    if characteristics[["company", "uwwCode"]].eq("").any().any():
        raise ValueError("Authoritative WWTW keys contain blank company or uwwCode")
    if characteristics.duplicated(["company", "uwwCode"]).any():
        raise ValueError("Authoritative WWTW company-uwwCode key is not unique")
    excluded = [
        column for column in characteristics.columns
        if any(pattern in column.lower() for pattern in OUTCOME_DERIVED_WWTW_PATTERNS)
    ]
    universe = characteristics.drop(columns=excluded).copy()
    works = gpd.read_file(paths.place_gpkg, layer="treatment_works_unique")
    if str(works.crs).upper() != BNG:
        works = works.to_crs(BNG)
    require_columns(
        works,
        ["company", "uwwCode", "canonical_uwwName", "uwwName_aliases", "treatment_work_easting", "treatment_work_northing"],
        "validated treatment-work points",
    )
    works["company"] = works["company"].map(clean_string)
    works["uwwCode"] = works["uwwCode"].map(clean_string)
    if works.duplicated(["company", "uwwCode"]).any():
        raise ValueError("Validated treatment-work point key is not unique")
    characteristic_keys = set(zip(universe["company"], universe["uwwCode"]))
    point_keys = set(zip(works["company"], works["uwwCode"]))
    if characteristic_keys != point_keys:
        only_characteristics = sorted(characteristic_keys - point_keys)
        only_points = sorted(point_keys - characteristic_keys)
        raise ValueError(
            "Authoritative WWTW sources disagree; "
            f"only characteristics={only_characteristics[:20]}, only points={only_points[:20]}"
        )
    point_fields = [
        "company", "uwwCode", "canonical_uwwName", "uwwName_aliases", "uwwName_alias_count",
        "uww_normalized_full_names", "uww_normalized_facility_names", "uww_normalized_place_names",
        "uwwLongitude", "uwwLatitude", "treatment_work_easting", "treatment_work_northing",
        "treatment_work_coordinate_year",
    ]
    universe = universe.merge(
        pd.DataFrame(works.drop(columns="geometry"))[[field for field in point_fields if field in works]],
        on=["company", "uwwCode"], how="left", validate="one_to_one",
    )
    universe = universe.rename(columns={
        "uwwName": "canonical_wwtw_name",
        "canonical_uwwName": "validated_canonical_wwtw_name",
        "uwwName_aliases": "raw_historical_wwtw_names",
        "uww_normalized_full_names": "wwtw_normalized_full_names",
        "uww_normalized_facility_names": "wwtw_normalized_facility_names",
        "uww_normalized_place_names": "wwtw_normalized_place_cores",
        "treatment_work_easting": "wwtw_easting",
        "treatment_work_northing": "wwtw_northing",
        "uwwLongitude": "wwtw_longitude",
        "uwwLatitude": "wwtw_latitude",
    })
    return universe.sort_values(["company", "uwwCode"], kind="mergesort"), works, excluded


def load_existing_accepted_pairs(
    paths: ProjectPaths,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    accepted = pd.read_csv(paths.accepted_pairs, low_memory=False)
    required = [
        "pair_uid", "company", "uwwCode", "uwwName", "sensor_uid", "permit_number",
        "sensor_location_name", "winning_sensor_alias", "winning_wwtw_alias",
        "sensor_normalized_full_name", "wwtw_normalized_full_name", "sensor_place_core",
        "wwtw_place_core", "place_core_exact_match", "place_core_score", "place_score_margin",
        "match_status", "match_rule_triggered", "accepted_place_name_match", "bng_easting",
        "bng_northing", "treatment_work_easting", "treatment_work_northing",
        "sensor_to_wwtw_distance_m", "existing_spatial_uwwCode", "existing_uwwcode_agreement",
    ]
    require_columns(accepted, required, "existing accepted-pair master")
    accepted["company"] = accepted["company"].map(clean_string)
    accepted["uwwCode"] = accepted["uwwCode"].map(clean_string)
    if not accepted["accepted_place_name_match"].map(bool_value).all():
        raise ValueError("Accepted-pair master contains a row without accepted_place_name_match=true")
    if accepted["pair_uid"].duplicated().any():
        raise ValueError("Accepted pair_uid is not unique")
    if accepted["sensor_uid"].duplicated().any():
        raise ValueError("A sensor appears against more than one accepted WWTW; investigate upstream")
    keys = set(zip(universe["company"], universe["uwwCode"]))
    unknown = set(zip(accepted["company"], accepted["uwwCode"])) - keys
    if unknown:
        raise ValueError(f"Accepted pairs contain nonauthoritative WWTW keys: {sorted(unknown)[:20]}")
    matching_fields = [
        "pair_uid", "company", "uwwCode", "uwwName", "sensor_uid", "permit_number",
        "sensor_location_name", "sensor_location_aliases", "winning_sensor_alias", "winning_wwtw_alias",
        "sensor_normalized_full_name", "wwtw_normalized_full_name", "sensor_place_core", "wwtw_place_core",
        "place_core_exact_match", "place_core_subset_match", "place_core_levenshtein_similarity",
        "place_core_token_set_ratio", "place_core_weighted_ratio", "place_core_token_overlap",
        "place_core_score", "place_score_margin", "old_composite_name_score", "asset_relationship_compatible",
        "match_status", "match_rule_triggered", "newly_accepted_after_place_core_fix",
        "sensor_coordinate_available", "bng_easting", "bng_northing", "longitude", "latitude",
        "treatment_work_easting", "treatment_work_northing", "uwwLongitude", "uwwLatitude",
        "sensor_to_wwtw_distance_m", "existing_spatial_uwwCode", "existing_uwwcode_agreement",
        "name_spatial_agreement",
    ]
    return accepted[[field for field in matching_fields if field in accepted]].copy()


def add_facility_audit_fields(
    candidates: pd.DataFrame,
    paths: ProjectPaths,
) -> pd.DataFrame:
    module = load_place_module(paths.place_script)
    place_config = json.loads(paths.place_config.read_text(encoding="utf-8"))
    candidates = candidates.copy()
    candidates["sensor_facility_type"] = candidates["winning_sensor_alias"].map(
        lambda value: module.normalize_place_name(value, place_config).facility_type
    )
    candidates["wwtw_facility_type"] = candidates["winning_wwtw_alias"].map(
        lambda value: module.normalize_place_name(value, place_config).facility_type
    )
    return candidates


def calculate_candidate_distances(
    candidates: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    frame = candidates.copy()
    sensor_valid = np.asarray([
        valid_bng(east, north) for east, north in zip(frame["bng_easting"], frame["bng_northing"])
    ])
    works_valid = np.asarray([
        valid_bng(east, north) for east, north in zip(frame["treatment_work_easting"], frame["treatment_work_northing"])
    ])
    frame["sensor_coordinate_usable"] = sensor_valid
    frame["wwtw_coordinate_usable"] = works_valid
    frame["coordinate_eligible_for_automatic_selection"] = sensor_valid & works_valid
    east_delta = pd.to_numeric(frame["bng_easting"], errors="coerce") - pd.to_numeric(frame["treatment_work_easting"], errors="coerce")
    north_delta = pd.to_numeric(frame["bng_northing"], errors="coerce") - pd.to_numeric(frame["treatment_work_northing"], errors="coerce")
    calculated = np.hypot(east_delta, north_delta)
    frame["primary_candidate_distance_m"] = np.where(
        frame["coordinate_eligible_for_automatic_selection"], calculated, np.nan
    )
    frame["existing_audit_distance_m"] = pd.to_numeric(frame["sensor_to_wwtw_distance_m"], errors="coerce")
    frame["distance_discrepancy_m"] = (frame["primary_candidate_distance_m"] - frame["existing_audit_distance_m"]).abs()
    tolerance = float(config.get("distance_discrepancy_tolerance_m", 1.0))
    frame["material_distance_discrepancy"] = frame["distance_discrepancy_m"].gt(tolerance).fillna(False)
    tiers = [float(value) for value in config["radius_tiers_m"]]
    conditions = [frame["primary_candidate_distance_m"].le(tier) for tier in tiers]
    labels = [f"within_{int(tier)}m" for tier in tiers]
    frame["radius_eligibility"] = np.select(conditions, labels, default="outside_max_radius")
    frame.loc[~frame["coordinate_eligible_for_automatic_selection"], "radius_eligibility"] = "coordinate_ineligible"
    frame["candidate_distance_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    eligible = frame[frame["coordinate_eligible_for_automatic_selection"]].sort_values(
        ["company", "uwwCode", "primary_candidate_distance_m", "place_core_exact_match",
         "place_core_score", "place_score_margin", "sensor_uid"],
        ascending=[True, True, True, False, False, False, True], kind="mergesort",
    )
    frame.loc[eligible.index, "candidate_distance_rank"] = eligible.groupby(
        ["company", "uwwCode"], sort=False
    ).cumcount().add(1).astype("Int64")
    name_sorted = frame.sort_values(
        ["company", "uwwCode", "place_core_score", "place_score_margin", "place_core_exact_match", "sensor_uid"],
        ascending=[True, True, False, False, False, True], kind="mergesort",
    )
    frame["candidate_name_score_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[name_sorted.index, "candidate_name_score_rank"] = name_sorted.groupby(
        ["company", "uwwCode"], sort=False
    ).cumcount().add(1).astype("Int64")
    return frame


def select_automatic_primaries(
    universe: pd.DataFrame,
    candidates: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = candidates.copy()
    frame["automatic_selected_as_primary"] = False
    frame["selected_as_primary"] = False
    frame["candidate_selection_status"] = "alternative_matching_candidate_not_selected"
    frame["automatic_selection_reason"] = "Not selected by automatic radius procedure"
    candidate_groups = {
        key: group.copy() for key, group in frame.groupby(["company", "uwwCode"], sort=True)
    }
    rows: list[dict[str, object]] = []
    tiers = [float(value) for value in config["radius_tiers_m"]]
    for work in universe.to_dict("records"):
        key = (work["company"], work["uwwCode"])
        group = candidate_groups.get(key, frame.iloc[0:0])
        total = len(group)
        eligible = group[group["coordinate_eligible_for_automatic_selection"]].copy()
        distances = eligible["primary_candidate_distance_m"].dropna().sort_values()
        nearest = float(distances.iloc[0]) if len(distances) else np.nan
        second = float(distances.iloc[1]) if len(distances) > 1 else np.nan
        gap = second - nearest if len(distances) > 1 else np.nan
        counts = {int(tier): int(eligible["primary_candidate_distance_m"].le(tier).sum()) for tier in tiers}
        selected = None
        selected_tier = np.nan
        if total == 0:
            status = "unassigned_no_accepted_name_candidate"
        elif not valid_bng(work.get("wwtw_easting"), work.get("wwtw_northing")):
            status = "unassigned_missing_wwtw_coordinates"
        elif eligible.empty:
            status = "unassigned_no_coordinate_eligible_candidate"
        else:
            first_tier = next((tier for tier in tiers if eligible["primary_candidate_distance_m"].le(tier).any()), None)
            if first_tier is None:
                status = "unassigned_no_candidate_within_max_radius"
            else:
                inside = eligible[eligible["primary_candidate_distance_m"].le(first_tier)].sort_values(
                    ["primary_candidate_distance_m", "place_core_exact_match", "place_core_score",
                     "place_score_margin", "sensor_uid"],
                    ascending=[True, False, False, False, True], kind="mergesort",
                )
                selected = inside.iloc[0]
                selected_tier = first_tier
                multiplicity = "unique" if len(inside) == 1 else "nearest_of_multiple"
                status = f"selected_{multiplicity}_within_{int(first_tier)}m"
                selected_index = int(selected.name)
                frame.at[selected_index, "automatic_selected_as_primary"] = True
                frame.at[selected_index, "selected_as_primary"] = True
                frame.at[selected_index, "candidate_selection_status"] = "selected_primary_existing_accepted_name_match"
                frame.at[selected_index, "automatic_selection_reason"] = status
        selected_uid = clean_string(selected["sensor_uid"]) if selected is not None else ""
        selected_score = float(selected["place_core_score"]) if selected is not None else np.nan
        highest_score = float(group["place_core_score"].max()) if total else np.nan
        rows.append({
            "company": key[0], "uwwCode": key[1],
            "primary_sensor_assigned": selected is not None,
            "is_primary_wwtw_sensor": selected is not None,
            "automatic_primary_sensor_uid": selected_uid,
            "final_primary_sensor_uid": selected_uid,
            "automatic_primary_pair_uid": clean_string(selected["pair_uid"]) if selected is not None else "",
            "final_primary_pair_uid": clean_string(selected["pair_uid"]) if selected is not None else "",
            "primary_radius_tier_m": selected_tier,
            "automatic_primary_radius_tier_m": selected_tier,
            "primary_sensor_selection_method": config["selection_method"] if selected is not None else "",
            "primary_sensor_selection_status": status,
            "automatic_primary_sensor_selection_status": status,
            "accepted_candidate_count_total": total,
            "accepted_candidate_count_with_coordinates": len(eligible),
            "accepted_candidate_count_within_1000m": counts.get(1000, 0),
            "accepted_candidate_count_within_2000m": counts.get(2000, 0),
            "accepted_candidate_count_within_3000m": counts.get(3000, 0),
            "nearest_candidate_distance_m": nearest,
            "second_nearest_candidate_distance_m": second,
            "distance_gap_to_second_candidate_m": gap,
            "selected_candidate_has_highest_name_score": bool(selected is not None and np.isclose(selected_score, highest_score, atol=1e-6)),
            "selected_candidate_is_exact_place_core": bool(selected is not None and bool_value(selected["place_core_exact_match"])),
            "selected_candidate_was_nearest_overall": bool(selected is not None and int(selected["candidate_distance_rank"]) == 1),
            "candidate_coordinate_missing_count": int(total - len(eligible)),
            "source_inconsistency_flag": bool(total and group["material_distance_discrepancy"].any()),
            **{field: "" for field in MANUAL_FIELDS},
            "manual_override_applied": False,
        })
    selection = pd.DataFrame(rows).sort_values(["company", "uwwCode"], kind="mergesort")
    return selection, frame


def apply_manual_decisions(
    selection: pd.DataFrame,
    candidates: pd.DataFrame,
    decisions_path: Path | None,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions_path is None:
        return selection, candidates
    decisions = pd.read_csv(decisions_path, dtype=str, keep_default_na=False)
    required = ["company", "uwwCode", "manual_decision"]
    require_columns(decisions, required, "manual decisions")
    for field in MANUAL_FIELDS:
        if field not in decisions:
            decisions[field] = ""
    decisions["company"] = decisions["company"].map(clean_string)
    decisions["uwwCode"] = decisions["uwwCode"].map(clean_string)
    decisions["manual_decision"] = decisions["manual_decision"].str.strip().str.lower()
    decisions = decisions[decisions["manual_decision"].ne("") & decisions["manual_decision"].ne("skip")].copy()
    allowed = {"accept_automatic", "select_candidate", "no_valid_primary"}
    invalid = sorted(set(decisions["manual_decision"]) - allowed)
    if invalid:
        raise ValueError(f"Unsupported manual decisions {invalid}; allowed={sorted(allowed)}")
    if decisions.duplicated(["company", "uwwCode"]).any():
        raise ValueError("Manual decisions duplicate a WWTW key")
    result = selection.copy()
    audit = candidates.copy()
    valid_keys = set(zip(result["company"], result["uwwCode"]))
    unknown = set(zip(decisions["company"], decisions["uwwCode"])) - valid_keys
    if unknown:
        raise ValueError(f"Manual decisions contain unknown WWTW keys: {sorted(unknown)}")
    for decision in decisions.to_dict("records"):
        key_mask = result["company"].eq(decision["company"]) & result["uwwCode"].eq(decision["uwwCode"])
        index = result.index[key_mask][0]
        action = decision["manual_decision"]
        selected_uid = clean_string(decision.get("manual_selected_sensor_uid", ""))
        if action == "accept_automatic":
            selected_uid = clean_string(result.at[index, "automatic_primary_sensor_uid"])
            if not selected_uid:
                raise ValueError(f"Cannot accept blank automatic selection for {decision['company']}::{decision['uwwCode']}")
        if action == "select_candidate":
            candidate_mask = (
                audit["company"].eq(decision["company"])
                & audit["uwwCode"].eq(decision["uwwCode"])
                & audit["sensor_uid"].eq(selected_uid)
            )
            if candidate_mask.sum() != 1:
                raise ValueError("Manual selection must identify exactly one existing same-WWTW accepted candidate")
        audit.loc[
            audit["company"].eq(decision["company"]) & audit["uwwCode"].eq(decision["uwwCode"]),
            ["selected_as_primary", "candidate_selection_status"],
        ] = [False, "alternative_matching_candidate_not_selected"]
        if action == "no_valid_primary":
            result.at[index, "primary_sensor_assigned"] = False
            result.at[index, "is_primary_wwtw_sensor"] = False
            result.at[index, "final_primary_sensor_uid"] = ""
            result.at[index, "final_primary_pair_uid"] = ""
            result.at[index, "primary_radius_tier_m"] = np.nan
            result.at[index, "primary_sensor_selection_method"] = "manual_review_no_valid_primary"
            result.at[index, "primary_sensor_selection_status"] = "manual_no_valid_primary_sensor"
        else:
            chosen_mask = (
                audit["company"].eq(decision["company"])
                & audit["uwwCode"].eq(decision["uwwCode"])
                & audit["sensor_uid"].eq(selected_uid)
            )
            chosen = audit.loc[chosen_mask].iloc[0]
            audit.loc[chosen_mask, "selected_as_primary"] = True
            audit.loc[chosen_mask, "candidate_selection_status"] = "selected_manual_existing_accepted_candidate"
            result.at[index, "primary_sensor_assigned"] = True
            result.at[index, "is_primary_wwtw_sensor"] = True
            result.at[index, "final_primary_sensor_uid"] = selected_uid
            result.at[index, "final_primary_pair_uid"] = chosen["pair_uid"]
            result.at[index, "primary_radius_tier_m"] = next(
                (tier for tier in config["radius_tiers_m"] if pd.notna(chosen["primary_candidate_distance_m"]) and chosen["primary_candidate_distance_m"] <= tier),
                np.nan,
            )
            result.at[index, "primary_sensor_selection_method"] = "manual_review_existing_accepted_candidate"
            result.at[index, "primary_sensor_selection_status"] = "manual_selected_existing_accepted_candidate"
        for field in MANUAL_FIELDS:
            result.at[index, field] = clean_string(decision.get(field, ""))
        result.at[index, "manual_override_applied"] = True
    selected_uids = result.loc[result["primary_sensor_assigned"], "final_primary_sensor_uid"]
    if selected_uids.duplicated().any():
        raise ValueError("Manual decisions assign one sensor to more than one WWTW")
    return result, audit


def evidence_tier(status: object) -> str:
    text = clean_string(status)
    if text == "accepted_exact_place_core":
        return "exact_place_core"
    if text == "accepted_place_subset_supported":
        return "place_subset_with_support"
    if text in {"accepted_place_core_existing_uwwcode", "accepted_place_core_spatial_support"}:
        return "high_place_core_with_support"
    return "high_place_core"


def review_reasons(row: Mapping[str, object], config: Mapping[str, object]) -> list[str]:
    reasons: list[str] = []
    if bool(row.get("source_inconsistency_flag")):
        reasons.append("source_or_distance_inconsistency")
    if not bool(row.get("primary_sensor_assigned")):
        reasons.append(clean_string(row.get("primary_sensor_selection_status")) or "no_selected_sensor")
    if int(row.get("accepted_candidate_count_total", 0)) > 1:
        reasons.append("multiple_accepted_matching_candidates")
    tier = pd.to_numeric(row.get("primary_radius_tier_m"), errors="coerce")
    if pd.notna(tier) and tier > 1000:
        reasons.append(f"selected_outside_1000m_tier_{int(tier)}m")
    if bool(row.get("primary_sensor_assigned")) and not bool(row.get("selected_candidate_is_exact_place_core")):
        reasons.append("selected_candidate_not_exact_place_core")
    gap = pd.to_numeric(row.get("distance_gap_to_second_candidate_m"), errors="coerce")
    if pd.notna(gap) and gap < float(config["near_tie_gap_m"]):
        reasons.append("near_distance_tie_under_100m")
    if bool(row.get("primary_sensor_assigned")) and not bool(row.get("selected_candidate_has_highest_name_score")):
        reasons.append("nearest_candidate_not_highest_name_score")
    if int(row.get("candidate_coordinate_missing_count", 0)) > 0:
        reasons.append("accepted_candidate_missing_coordinates")
    if bool(row.get("manual_override_applied")):
        reasons.append("manual_override_applied")
    return list(dict.fromkeys(reasons))


def review_priority(reasons: Sequence[str], row: Mapping[str, object]) -> str:
    if any("inconsistency" in reason or "conflict" in reason for reason in reasons):
        return "critical"
    tier = pd.to_numeric(row.get("primary_radius_tier_m"), errors="coerce")
    if (
        not bool(row.get("primary_sensor_assigned"))
        or any("missing_coordinate" in reason or "near_distance_tie" in reason for reason in reasons)
        or (pd.notna(tier) and tier >= 3000)
    ):
        return "high"
    if reasons:
        return "medium"
    return "low"


def add_review_flags(selection: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    result = selection.copy()
    reasons, priorities = [], []
    for row in result.to_dict("records"):
        row_reasons = review_reasons(row, config)
        reasons.append(" | ".join(row_reasons))
        priorities.append(review_priority(row_reasons, row))
    result["manual_review_reason"] = reasons
    result["manual_review_priority"] = priorities
    result["manual_review_required"] = result["manual_review_reason"].ne("")
    return result


def build_primary_master(
    universe: pd.DataFrame,
    selection: pd.DataFrame,
    candidates: pd.DataFrame,
    sensors: pd.DataFrame,
) -> pd.DataFrame:
    result = universe.merge(selection, on=["company", "uwwCode"], how="left", validate="one_to_one")
    chosen = candidates[candidates["selected_as_primary"]].copy()
    selected_matching_fields = {
        "pair_uid": "primary_pair_uid",
        "sensor_uid": "primary_sensor_uid",
        "permit_number": "primary_sensor_permit_number_from_match",
        "sensor_location_name": "primary_sensor_location_name_from_match",
        "winning_sensor_alias": "primary_sensor_raw_name",
        "sensor_normalized_full_name": "primary_sensor_normalized_full_name",
        "sensor_place_core": "primary_sensor_normalized_place_core",
        "sensor_facility_type": "primary_sensor_facility_type",
        "wwtw_normalized_full_name": "selected_wwtw_normalized_full_name",
        "wwtw_place_core": "selected_wwtw_place_core",
        "wwtw_facility_type": "selected_wwtw_facility_type",
        "primary_candidate_distance_m": "primary_candidate_distance_m",
        "match_status": "selected_match_status",
        "match_rule_triggered": "selected_match_evidence",
        "place_core_score": "selected_place_core_score",
        "place_score_margin": "selected_score_margin",
        "place_core_exact_match": "selected_exact_place_core_match",
        "existing_spatial_uwwCode": "selected_existing_spatial_uwwCode",
        "existing_uwwcode_agreement": "selected_existing_spatial_agreement",
    }
    chosen_fields = ["company", "uwwCode", *selected_matching_fields]
    chosen = chosen[[field for field in chosen_fields if field in chosen]].rename(columns=selected_matching_fields)
    result = result.merge(chosen, on=["company", "uwwCode"], how="left", validate="one_to_one")
    result["selected_match_evidence_tier"] = result["selected_match_status"].map(evidence_tier)
    sensor_fields = [
        "sensor_uid", "permit_number", "location_name", "bng_easting", "bng_northing", "longitude", "latitude",
        "fitted_beta", "beta_ci_lower_95", "beta_ci_upper_95", "beta_ci_width",
        "eligible_for_beta_analysis", "beta_quality", "beta_at_bound", "tail_spill_count",
        "point_fit_success", "bootstrap_success_rate",
    ]
    sensor_join = sensors[sensor_fields].rename(columns={
        "sensor_uid": "primary_sensor_uid",
        "permit_number": "primary_sensor_permit_number",
        "location_name": "primary_sensor_location_name",
        "bng_easting": "primary_sensor_easting",
        "bng_northing": "primary_sensor_northing",
        "longitude": "primary_sensor_longitude",
        "latitude": "primary_sensor_latitude",
        "beta_quality": "beta_quality_status",
        "tail_spill_count": "tail_event_count",
        "point_fit_success": "fit_success",
        "bootstrap_success_rate": "bootstrap_success",
    })
    # Several unassigned WWTWs legitimately share a blank left key.  The
    # authoritative sensor-side key remains unique, hence many-to-one.
    result = result.merge(sensor_join, on="primary_sensor_uid", how="left", validate="many_to_one")
    result["primary_sensor_permit_number"] = result["primary_sensor_permit_number"].fillna(
        result.get("primary_sensor_permit_number_from_match")
    )
    result["primary_sensor_location_name"] = result["primary_sensor_location_name"].fillna(
        result.get("primary_sensor_location_name_from_match")
    )
    result = result.drop(columns=[
        column for column in ["primary_sensor_permit_number_from_match", "primary_sensor_location_name_from_match"]
        if column in result
    ])
    return result.sort_values(["company", "uwwCode"], kind="mergesort")


def build_candidate_audit(
    candidates: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    flags = selection[["company", "uwwCode", "manual_review_required", "manual_review_priority", "manual_review_reason"]]
    result = candidates.merge(flags, on=["company", "uwwCode"], how="left", validate="many_to_one")
    result = result.rename(columns={
        "uwwName": "wwtw_name",
        "sensor_location_name": "sensor_location_name",
        "match_status": "original_accepted_match_status",
        "treatment_work_easting": "wwtw_easting",
        "treatment_work_northing": "wwtw_northing",
        "bng_easting": "sensor_easting",
        "bng_northing": "sensor_northing",
    })
    preferred = [
        "company", "uwwCode", "wwtw_name", "sensor_uid", "sensor_location_name", "pair_uid",
        "winning_sensor_alias", "winning_wwtw_alias", "sensor_normalized_full_name", "wwtw_normalized_full_name",
        "sensor_place_core", "wwtw_place_core", "sensor_facility_type", "wwtw_facility_type",
        "original_accepted_match_status", "match_rule_triggered", "place_core_exact_match",
        "place_core_subset_match", "place_core_levenshtein_similarity", "place_core_token_set_ratio",
        "place_core_weighted_ratio", "place_core_token_overlap", "place_core_score", "place_score_margin",
        "existing_spatial_uwwCode", "existing_uwwcode_agreement", "name_spatial_agreement",
        "wwtw_easting", "wwtw_northing", "sensor_easting", "sensor_northing", "longitude", "latitude",
        "primary_candidate_distance_m", "existing_audit_distance_m", "distance_discrepancy_m",
        "material_distance_discrepancy", "sensor_coordinate_usable", "wwtw_coordinate_usable",
        "coordinate_eligible_for_automatic_selection", "radius_eligibility", "candidate_distance_rank",
        "candidate_name_score_rank", "candidate_selection_status", "automatic_selected_as_primary",
        "selected_as_primary", "automatic_selection_reason", "manual_review_required",
        "manual_review_priority", "manual_review_reason",
    ]
    return result[[field for field in preferred if field in result]].sort_values(
        ["company", "uwwCode", "candidate_distance_rank", "sensor_uid"], kind="mergesort", na_position="last"
    )


def build_review_queue(
    master: pd.DataFrame,
    candidate_audit: pd.DataFrame,
) -> pd.DataFrame:
    audit_fields: dict[tuple[str, str], dict[str, str]] = {}
    for key, group in candidate_audit.groupby(["company", "uwwCode"], sort=True):
        group = group.sort_values(["primary_candidate_distance_m", "sensor_uid"], kind="mergesort", na_position="last")
        audit_fields[key] = {
            "accepted_candidate_sensor_uids": " | ".join(group["sensor_uid"].map(clean_string)),
            "accepted_candidate_sensor_names": " | ".join(group["sensor_location_name"].map(clean_string)),
            "accepted_candidate_distances_m": " | ".join(
                group["primary_candidate_distance_m"].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
            ),
        }
    queue = master[master["manual_review_required"]].copy()
    for field in ("accepted_candidate_sensor_uids", "accepted_candidate_sensor_names", "accepted_candidate_distances_m"):
        queue[field] = [audit_fields.get((row.company, row.uwwCode), {}).get(field, "") for row in queue.itertuples()]
    preferred = [
        "company", "uwwCode", "canonical_wwtw_name", "validated_canonical_wwtw_name",
        "raw_historical_wwtw_names", "wwtw_easting", "wwtw_northing",
        "automatic_primary_sensor_selection_status", "automatic_primary_sensor_uid",
        "primary_sensor_selection_status", "primary_sensor_uid", "primary_radius_tier_m",
        "accepted_candidate_count_total", "accepted_candidate_count_with_coordinates",
        "nearest_candidate_distance_m", "second_nearest_candidate_distance_m",
        "distance_gap_to_second_candidate_m", "selected_candidate_has_highest_name_score",
        "selected_candidate_is_exact_place_core", "source_inconsistency_flag",
        "manual_review_required", "manual_review_priority", "manual_review_reason",
        "accepted_candidate_sensor_uids", "accepted_candidate_sensor_names", "accepted_candidate_distances_m",
        *MANUAL_FIELDS,
    ]
    priority_order = pd.CategoricalDtype(["critical", "high", "medium", "low"], ordered=True)
    queue["manual_review_priority"] = queue["manual_review_priority"].astype(priority_order)
    return queue[[field for field in preferred if field in queue]].sort_values(
        ["manual_review_priority", "company", "uwwCode"], kind="mergesort"
    )


def safe_attributes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: json.dumps(value, sort_keys=True, default=str)
                if isinstance(value, (list, dict, set, tuple)) else value
            )
    return result


def write_qml_styles(directory: Path) -> None:
    colours = {
        "treatment_works_assigned_red.qml": ("treatment_works_assigned", "220,45,45,255"),
        "treatment_works_unassigned_magenta.qml": ("treatment_works_unassigned", "195,45,175,255"),
        "primary_matched_sensors_green.qml": ("primary_matched_sensors", "35,170,70,255"),
        "nonselected_matching_candidates_orange.qml": ("nonselected_matching_candidates", "240,135,35,255"),
        "unmatched_name_sensors_yellow.qml": ("unmatched_name_sensors", "245,210,35,255"),
    }
    directory.mkdir(parents=True, exist_ok=True)
    for filename, (layer, colour) in colours.items():
        text = f'''<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol"><symbols><symbol type="marker" name="0"><layer class="SimpleMarker">
    <Option type="Map"><Option name="name" value="circle"/><Option name="color" value="{colour}"/>
      <Option name="outline_color" value="35,35,35,255"/><Option name="outline_style" value="solid"/>
      <Option name="outline_width" value="0.6"/><Option name="size" value="3.2"/></Option>
  </layer></symbol></symbols></renderer-v2><layername>{layer}</layername>
</qgis>'''
        (directory / filename).write_text(text + "\n", encoding="utf-8")


def write_geopackage(
    path: Path,
    works: gpd.GeoDataFrame,
    master: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    unmatched_review: pd.DataFrame,
) -> None:
    works_base = works[["company", "uwwCode", "geometry"]].merge(
        master, on=["company", "uwwCode"], how="left", validate="one_to_one"
    )
    assigned = works_base[works_base["primary_sensor_assigned"]].copy()
    unassigned = works_base[~works_base["primary_sensor_assigned"]].copy()
    gpd.GeoDataFrame(safe_attributes(assigned), geometry="geometry", crs=BNG).to_file(
        path, layer="treatment_works_assigned", driver="GPKG", engine="pyogrio"
    )
    gpd.GeoDataFrame(safe_attributes(unassigned), geometry="geometry", crs=BNG).to_file(
        path, layer="treatment_works_unassigned", driver="GPKG", engine="pyogrio", append=True
    )
    primary_geometry_mask = np.asarray([
        valid_bng(east, north)
        for east, north in zip(master["primary_sensor_easting"], master["primary_sensor_northing"])
    ])
    primary = master[master["primary_sensor_assigned"] & primary_geometry_mask].copy()
    primary_geometry = gpd.points_from_xy(primary["primary_sensor_easting"], primary["primary_sensor_northing"], crs=BNG)
    gpd.GeoDataFrame(safe_attributes(primary), geometry=primary_geometry, crs=BNG).to_file(
        path, layer="primary_matched_sensors", driver="GPKG", engine="pyogrio", append=True
    )
    alternative_geometry_mask = np.asarray([
        valid_bng(east, north)
        for east, north in zip(candidate_audit["sensor_easting"], candidate_audit["sensor_northing"])
    ])
    alternatives = candidate_audit[~candidate_audit["selected_as_primary"] & alternative_geometry_mask].copy()
    alternative_geometry = gpd.points_from_xy(alternatives["sensor_easting"], alternatives["sensor_northing"], crs=BNG)
    gpd.GeoDataFrame(safe_attributes(alternatives), geometry=alternative_geometry, crs=BNG).to_file(
        path, layer="nonselected_matching_candidates", driver="GPKG", engine="pyogrio", append=True
    )
    review_geometry_mask = np.asarray([
        valid_bng(east, north)
        for east, north in zip(unmatched_review["bng_easting"], unmatched_review["bng_northing"])
    ])
    review = unmatched_review[review_geometry_mask].copy()
    review = review.rename(columns={
        "uwwCode": "best_review_candidate_uwwCode",
        "uwwName": "best_review_candidate_wwtw_name",
        "match_status": "name_match_review_status",
    })
    review["selected_wwtw_key"] = ""
    review["sensor_classification"] = "unmatched_name_sensor"
    review_geometry = gpd.points_from_xy(review["bng_easting"], review["bng_northing"], crs=BNG)
    gpd.GeoDataFrame(safe_attributes(review), geometry=review_geometry, crs=BNG).to_file(
        path, layer="unmatched_name_sensors", driver="GPKG", engine="pyogrio", append=True
    )


def blyth_diagnostic(
    master: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    gpkg: Path | None = None,
) -> dict[str, object]:
    blyth = master[
        master["company"].eq("northumbria")
        & (
            master["canonical_wwtw_name"].map(clean_string).str.contains(r"\bBLYTH\b", case=False, regex=True)
            | master["raw_historical_wwtw_names"].map(clean_string).str.contains(r"\bBLYTH\b", case=False, regex=True)
        )
    ]
    if len(blyth) != 1:
        raise ValueError(f"Expected one authoritative Northumbria Blyth WWTW, found {len(blyth)}")
    work = blyth.iloc[0]
    candidates = candidate_audit[
        candidate_audit["company"].eq(work["company"]) & candidate_audit["uwwCode"].eq(work["uwwCode"])
    ].sort_values(["primary_candidate_distance_m", "sensor_uid"], kind="mergesort")
    eligible = candidates[candidates["coordinate_eligible_for_automatic_selection"]]
    selected = candidates[candidates["automatic_selected_as_primary"]]
    if len(selected) != 1 or eligible.empty:
        raise ValueError("Blyth lacks exactly one coordinate-eligible automatic primary")
    if selected.iloc[0]["sensor_uid"] != eligible.iloc[0]["sensor_uid"]:
        raise ValueError("Blyth automatic primary is not the nearest eligible accepted candidate")
    if gpkg is not None:
        primary_uids = set(gpd.read_file(gpkg, layer="primary_matched_sensors")["primary_sensor_uid"])
        alternative_uids = set(gpd.read_file(gpkg, layer="nonselected_matching_candidates")["sensor_uid"])
        nonselected = set(candidates.loc[~candidates["automatic_selected_as_primary"], "sensor_uid"])
        if selected.iloc[0]["sensor_uid"] not in primary_uids or not nonselected <= alternative_uids or nonselected & primary_uids:
            raise ValueError("Blyth GeoPackage primary/alternative classification failed")
    return {
        "company": work["company"], "uwwCode": work["uwwCode"],
        "wwtw_name": work["canonical_wwtw_name"], "wwtw_easting": work["wwtw_easting"],
        "wwtw_northing": work["wwtw_northing"],
        "selected_sensor_uid": selected.iloc[0]["sensor_uid"],
        "selected_radius_tier_m": work["primary_radius_tier_m"],
        "distance_to_second_nearest_m": work["second_nearest_candidate_distance_m"],
        "candidates": candidates[["sensor_uid", "sensor_location_name", "primary_candidate_distance_m",
                                   "place_core_score", "selected_as_primary"]].to_dict("records"),
    }


def validate_release_gates(
    universe: pd.DataFrame,
    master: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    sensors: pd.DataFrame,
    works: gpd.GeoDataFrame,
    gpkg: Path,
    config: Mapping[str, object],
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
) -> dict[str, object]:
    assigned = master[master["primary_sensor_assigned"]]
    automatic = assigned[~assigned["manual_override_applied"]]
    accepted_pairs = set(zip(candidate_audit["company"], candidate_audit["uwwCode"], candidate_audit["sensor_uid"]))
    selected_pairs = set(zip(assigned["company"], assigned["uwwCode"], assigned["primary_sensor_uid"]))
    tiers = set(float(value) for value in config["radius_tiers_m"])
    selected_audit = candidate_audit[candidate_audit["selected_as_primary"]]
    automatic_audit = candidate_audit[candidate_audit["automatic_selected_as_primary"]]
    nearest_ok = bool(automatic_audit["candidate_distance_rank"].eq(1).all())
    source_beta = sensors.set_index("sensor_uid")[NUMERIC_BETA_FIELDS]
    output_beta = assigned.set_index("primary_sensor_uid")[NUMERIC_BETA_FIELDS]
    common = output_beta.index.intersection(source_beta.index)
    beta_unchanged = np.allclose(output_beta.loc[common].to_numpy(float), source_beta.loc[common].to_numpy(float), equal_nan=True)
    aggregate_patterns = (
        "median_beta", "mean_beta", "beta_std", "beta_q25", "beta_q75",
        "beta_iqr", "median_beta_ci_width", "all_finite_beta_sensor_count",
        "primary_beta_sensor_count", "finite_beta_sensor_count",
    )
    leakage = [
        column for column in master.columns
        if any(pattern in column.lower() for pattern in aggregate_patterns)
    ]
    layers = pyogrio.list_layers(gpkg)[:, 0].tolist()
    expected_layers = [
        "treatment_works_assigned", "treatment_works_unassigned", "primary_matched_sensors",
        "nonselected_matching_candidates", "unmatched_name_sensors",
    ]
    details: dict[str, object] = {}
    frames: dict[str, gpd.GeoDataFrame] = {}
    for layer in layers:
        frame = gpd.read_file(gpkg, layer=layer)
        frames[layer] = frame
        details[layer] = {
            "features": len(frame), "geometry_types": sorted(set(frame.geom_type.dropna())),
            "crs": str(frame.crs), "null_geometries": int(frame.geometry.isna().sum()),
            "empty_geometries": int(frame.geometry.is_empty.sum()),
        }
    mappable_works = int((~works.geometry.isna() & ~works.geometry.is_empty).sum())
    checks = {
        "place_branch_unchanged": protected_before == protected_after,
        "primary_master_row_count": len(master) == len(universe),
        "primary_master_key_unique": not master.duplicated(["company", "uwwCode"]).any(),
        "assigned_exactly_one_sensor": assigned["primary_sensor_uid"].map(clean_string).ne("").all() and not assigned.duplicated(["company", "uwwCode"]).any(),
        "selected_sensor_unique": not assigned["primary_sensor_uid"].duplicated().any(),
        "automatic_selected_from_accepted_pairs": selected_pairs <= accepted_pairs,
        "selected_key_consistency": len(selected_audit) == len(assigned),
        "automatic_distance_finite_and_bounded": bool(np.isfinite(automatic["primary_candidate_distance_m"]).all() and automatic["primary_candidate_distance_m"].le(float(config["max_radius_m"])).all()),
        "automatic_radius_tiers_valid": set(pd.to_numeric(automatic["primary_radius_tier_m"], errors="coerce").dropna()) <= tiers,
        "automatic_selected_nearest": nearest_ok,
        "missing_coordinates_not_automatic": bool(automatic_audit["coordinate_eligible_for_automatic_selection"].all()),
        "sensor_beta_fields_unchanged": bool(beta_unchanged),
        "no_outcome_derived_wwtw_beta_summary": not leakage,
        "layers_exact": layers == expected_layers,
        "all_layers_bng": all(str(frame.crs).upper() == BNG for frame in frames.values()),
        "assigned_map_equals_primary_sensor_map": len(frames["treatment_works_assigned"]) == len(frames["primary_matched_sensors"]),
        "assigned_plus_unassigned_map": len(frames["treatment_works_assigned"]) + len(frames["treatment_works_unassigned"]) == mappable_works,
        "table_assignment_partition": len(assigned) + len(master[~master["primary_sensor_assigned"]]) == len(universe),
    }
    blyth = blyth_diagnostic(master, candidate_audit, gpkg)
    checks["blyth_selected_nearest"] = bool(blyth["selected_sensor_uid"])
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"Release gates failed: {failed}")
    return {"checks": checks, "layers": details, "blyth": blyth}


def package_versions() -> dict[str, str]:
    result = {}
    for package in ("pandas", "geopandas", "numpy", "pyogrio", "shapely", "pyproj", "rapidfuzz"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def git_provenance(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": "unavailable (not a git work tree)", "dirty": "unavailable"}


def distance_statistics(frame: pd.DataFrame) -> dict[str, float]:
    values = pd.to_numeric(frame["primary_candidate_distance_m"], errors="coerce").dropna()
    if values.empty:
        return {key: np.nan for key in ("minimum", "median", "mean", "p75", "p90", "p95", "maximum")}
    return {
        "minimum": float(values.min()), "median": float(values.median()), "mean": float(values.mean()),
        "p75": float(values.quantile(0.75)), "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)), "maximum": float(values.max()),
    }


def audit_report(
    run_id: str,
    created_at: str,
    paths: ProjectPaths,
    config: Mapping[str, object],
    source_frames: Mapping[str, pd.DataFrame],
    source_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
    universe: pd.DataFrame,
    master: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    review_queue: pd.DataFrame,
    validation: Mapping[str, object],
    excluded_wwtw_fields: Sequence[str],
    manual_decisions_path: Path | None,
) -> str:
    assigned = master[master["primary_sensor_assigned"]]
    unassigned = master[~master["primary_sensor_assigned"]]
    total = len(master)
    pct = lambda count, denominator: 100.0 * count / denominator if denominator else np.nan
    lines = [
        "WWTW PRIMARY SENSOR SELECTION AUDIT", "=" * 100,
        "A. RUN PROVENANCE", "-" * 100,
        f"run ID: {run_id}", f"timestamp UTC: {created_at}",
        f"git: {json.dumps(git_provenance(paths.root), sort_keys=True)}",
        f"Python: {platform.python_version()} ({sys.executable})",
        f"packages: {json.dumps(package_versions(), sort_keys=True)}",
        f"script: {Path(__file__).resolve()}",
        f"configuration: {json.dumps(config, sort_keys=True)}",
        f"manual decisions: {manual_decisions_path or 'none'}",
        "source paths, rows, and SHA-256:",
    ]
    for label in sorted(source_hashes):
        rows = len(source_frames[label]) if label in source_frames else "n/a"
        lines.append(f"  {label}: path={label}; rows={rows}; sha256={source_hashes[label]}")
    lines.append("validated output hashes available before this report (audit/manifest hashes are in SHA256SUMS.json):")
    lines.extend(f"  {path}: {digest}" for path, digest in sorted(output_hashes.items()))
    lines.extend([
        "", "B. AUTHORITATIVE WWTW UNIVERSE", "-" * 100,
        f"total authoritative WWTWs: {total:,}",
        f"company-uwwCode unique: {not master.duplicated(['company', 'uwwCode']).any()}",
        f"blank authoritative keys: {int(master[['company', 'uwwCode']].fillna('').eq('').any(axis=1).sum()):,}",
        f"missing WWTW coordinates: {int((~np.asarray([valid_bng(e, n) for e, n in zip(master['wwtw_easting'], master['wwtw_northing'])])).sum()):,}",
        f"excluded outcome-derived WWTW fields: {' | '.join(excluded_wwtw_fields)}",
        "WWTWs by company:",
    ])
    lines.extend(f"  {company}: {count:,}" for company, count in master.groupby("company").size().items())
    lines.extend([
        "", "C. PRIMARY ASSIGNMENT RESULTS", "-" * 100,
        f"assigned: {len(assigned):,} ({pct(len(assigned), total):.2f}% of authoritative WWTWs)",
        f"unassigned: {len(unassigned):,} ({pct(len(unassigned), total):.2f}% of authoritative WWTWs)",
    ])
    for tier in config["radius_tiers_m"]:
        count = int(assigned["primary_radius_tier_m"].eq(float(tier)).sum())
        lines.append(f"selected in first successful {int(tier):,} m tier: {count:,}")
    status_counts = master["primary_sensor_selection_status"].value_counts().sort_index()
    lines.append("selection status counts:")
    lines.extend(f"  {status}: {count:,}" for status, count in status_counts.items())
    lines.extend([
        f"automatic selected: {int((assigned['manual_override_applied'] == False).sum()):,}",
        f"manual selected/overridden: {int(assigned['manual_override_applied'].sum()):,}",
        "", "D. CANDIDATE STRUCTURE", "-" * 100,
    ])
    candidate_counts = master["accepted_candidate_count_total"]
    distribution = candidate_counts.value_counts().sort_index()
    lines.extend([
        f"WWTWs with zero accepted candidates: {int(candidate_counts.eq(0).sum()):,}",
        f"WWTWs with one accepted candidate: {int(candidate_counts.eq(1).sum()):,}",
        f"WWTWs with multiple accepted candidates: {int(candidate_counts.gt(1).sum()):,}",
        f"maximum candidate count: {int(candidate_counts.max()):,}",
        f"median candidate count: {float(candidate_counts.median()):.3f}",
        f"nonselected accepted candidates: {int((~candidate_audit['selected_as_primary']).sum()):,}",
        "candidate-count distribution:",
    ])
    lines.extend(f"  {int(count)} candidates: {int(works):,} WWTWs" for count, works in distribution.items())
    lines.extend(["", "E. UNASSIGNED REASONS", "-" * 100])
    for status, count in unassigned["primary_sensor_selection_status"].value_counts().sort_index().items():
        lines.append(f"  {status}: {count:,} ({pct(count, len(unassigned)):.2f}% of unassigned)")
    lines.append("full unassigned list (see tables/wwtw_primary_sensor_review_queue.csv for review):")
    for row in unassigned[["company", "uwwCode", "canonical_wwtw_name", "primary_sensor_selection_status"]].itertuples(index=False):
        lines.append(f"  {row.company:<18} {row.uwwCode:<25} {clean_string(row.canonical_wwtw_name):<45} {row.primary_sensor_selection_status}")
    finite = pd.to_numeric(assigned["fitted_beta"], errors="coerce").map(np.isfinite)
    eligible = assigned["eligible_for_beta_analysis"].map(bool_value)
    lines.extend([
        "", "F. SELECTED-SENSOR BETA COVERAGE (POST-SELECTION JOIN)", "-" * 100,
        f"selected primary sensors with finite fitted_beta: {int(finite.sum()):,}/{len(assigned):,} ({pct(finite.sum(), len(assigned)):.2f}% of selected primary sensors)",
        f"selected primary sensors eligible_for_beta_analysis: {int(eligible.sum()):,}/{len(assigned):,} ({pct(eligible.sum(), len(assigned)):.2f}% of selected primary sensors)",
        f"authoritative WWTWs assigned a sensor with finite fitted_beta: {int(finite.sum()):,}/{total:,} ({pct(finite.sum(), total):.2f}% of all authoritative WWTWs)",
        "", "G. DISTANCE DIAGNOSTICS", "-" * 100,
        f"overall: {json.dumps(distance_statistics(assigned), sort_keys=True)}",
        "by company:",
    ])
    for company, group in assigned.groupby("company", sort=True):
        lines.append(f"  {company}: {json.dumps(distance_statistics(group), sort_keys=True)}")
    near_ties = master["distance_gap_to_second_candidate_m"].lt(float(config["near_tie_gap_m"])).sum()
    lines.extend([
        "", "H. MANUAL-REVIEW FLAGS", "-" * 100,
        f"WWTWs requiring review: {int(master['manual_review_required'].sum()):,}",
    ])
    lines.extend(f"  priority {priority}: {count:,}" for priority, count in master.loc[master["manual_review_required"], "manual_review_priority"].value_counts().sort_index().items())
    lines.extend([
        f"near ties (<{float(config['near_tie_gap_m']):g} m): {int(near_ties):,}",
        f"selected beyond 1 km: {int(assigned['primary_radius_tier_m'].gt(1000).sum()):,}",
        f"selected candidate not highest name score: {int((~assigned['selected_candidate_has_highest_name_score']).sum()):,}",
        f"WWTWs with accepted candidates missing coordinates: {int(master['candidate_coordinate_missing_count'].gt(0).sum()):,}",
        f"manual review queue rows: {len(review_queue):,}",
        "", "I. GEOPACKAGE VALIDATION", "-" * 100,
        json.dumps(validation["layers"], indent=2, sort_keys=True),
        f"release gates: {json.dumps(validation['checks'], sort_keys=True)}",
        "", "J. BLYTH WORKED EXAMPLE", "-" * 100,
    ])
    blyth = validation["blyth"]
    lines.extend([
        f"WWTW key: {blyth['company']}::{blyth['uwwCode']}",
        f"WWTW name: {blyth['wwtw_name']}",
        f"WWTW coordinates: ({float(blyth['wwtw_easting']):.3f}, {float(blyth['wwtw_northing']):.3f}) EPSG:27700",
        f"selected sensor: {blyth['selected_sensor_uid']}",
        f"selected radius tier: {float(blyth['selected_radius_tier_m']):.0f} m",
        f"distance to second-nearest candidate: {float(blyth['distance_to_second_nearest_m']):.3f} m",
        "all existing accepted Blyth candidates:",
    ])
    for candidate in blyth["candidates"]:
        lines.append(
            f"  sensor={candidate['sensor_uid']}; name={candidate['sensor_location_name']}; "
            f"distance_m={float(candidate['primary_candidate_distance_m']):.3f}; "
            f"place_core_score={float(candidate['place_core_score']):.3f}; selected={candidate['selected_as_primary']}"
        )
    lines.extend([
        "", "K. SCIENTIFIC INTERPRETATION WARNING", "-" * 100,
        config["scientific_warning"], "=" * 100,
    ])
    return "\n".join(map(str, lines)) + "\n"


def branch_readme() -> str:
    return """# WWTW-centric primary sensor selection

This separate branch selects exactly zero or one analytical primary CSO sensor for each authoritative company-uwwCode treatment works. Candidates come only from the existing accepted place-core pair master. An accepted pair is a defensible site/name association; a primary sensor is the single analytical response chosen from those accepted pairs. Nonselected accepted sensors remain alternative matching candidates and are never called unmatched.

## Algorithm

Distances are recalculated from sensor and WWTW points in EPSG:27700. For each WWTW, the first radius containing an accepted coordinate-eligible candidate is chosen from 1,000, 2,000 and 3,000 metres. A unique candidate is selected directly; with several candidates, the nearest is selected. Exact place-core evidence, score, margin and sensor_uid are deterministic tie-breakers only. The search stops at 3,000 metres. Distance never promotes a nonaccepted name candidate.

Candidates without coordinates remain in the audit table but cannot be selected automatically. Unassigned WWTWs remain as complete WWTW rows. Response and environmental fields are joined only after selection. Beta, spill, capacity and environmental fields never influence selection.

## Outputs per immutable run

- `tables/wwtw_primary_sensor_master.csv`: one row per authoritative WWTW;
- `tables/wwtw_primary_sensor_candidates.csv`: every existing accepted candidate;
- `tables/wwtw_primary_sensor_review_queue.csv`: WWTW-level manual review queue;
- `qgis/wwtw_primary_sensor_selection.gpkg`: assigned/unassigned WWTWs, primary sensors, alternative accepted candidates and original unmatched-name sensors;
- `diagnostics/wwtw_primary_sensor_selection_audit.txt`: provenance, counts, distances, release gates and Blyth example;
- `config_snapshot/`: exact configuration used;
- `run_manifest.json` and `SHA256SUMS.json`: validated-run provenance and hashes.

`LATEST.txt` points to the most recent successfully validated run. Prior runs are not overwritten.

## Commands

Automatic run:

```text
python scripts/08c_select_primary_wwtw_sensor_by_distance.py --project-root . --config config/wwtw_primary_sensor_selection.json --run-id <unique_run_id>
```

Manual review:

```text
python scripts/08d_review_primary_wwtw_sensor_matches.py --run-dir project_results/new_sensor_assignment_approach_primary_spatial/runs/<run_id>
```

Reviewed rebuild:

```text
python scripts/08c_select_primary_wwtw_sensor_by_distance.py --project-root . --config config/wwtw_primary_sensor_selection.json --manual-decisions <manual_decisions_csv> --run-id <new_unique_run_id>
```

To reproduce a prior run, use its configuration snapshot and source hashes recorded in its manifest. Never reuse its run ID.

## Manual review

The review utility writes a separate manual-decision CSV and never edits automatic outputs. A reviewer may accept the automatic selection, select an existing same-WWTW accepted candidate, record no valid primary, skip or save and quit. The initial automated output contains blank manual fields; decisions are applied only in a new run.

## Limitations

This procedure selects one analytical primary sensor from existing accepted name-matched candidates using spatial proximity. It does not establish that alternatives are unrelated and does not prove underground hydraulic connectivity. Downstream modelling must respect missing responses and the documented selection process.
"""


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args.project_root, args.config)
    required_files = [
        paths.config, paths.characteristics, paths.sensor_master, paths.accepted_pairs,
        paths.unmatched_review, paths.place_gpkg, paths.place_script, paths.place_config,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required inputs missing: {missing}")
    config = load_configuration(paths.config)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_primary_distance")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and hyphen")
    runs = paths.branch / "runs"
    final_run = runs / run_id
    if final_run.exists():
        raise FileExistsError(f"Run directory already exists and will not be overwritten: {final_run}")
    staging = runs / f".{run_id}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(timezone.utc).isoformat()
    manual_path = args.manual_decisions.resolve() if args.manual_decisions else None
    if manual_path and not manual_path.is_file():
        raise FileNotFoundError(f"Manual decisions file does not exist: {manual_path}")
    protected = protected_place_files(paths)
    protected_before = snapshot(protected)
    try:
        log_path = staging / "logs/wwtw_primary_sensor_selection.log"
        log_path.parent.mkdir(parents=True)
        LOGGER.setLevel(logging.INFO)
        LOGGER.handlers.clear()
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        stream = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        stream.setFormatter(formatter)
        LOGGER.addHandler(handler)
        LOGGER.addHandler(stream)
        LOGGER.info("run_id=%s staging=%s", run_id, staging)
        universe, works, excluded_wwtw_fields = load_authoritative_wwtw_universe(paths)
        accepted = load_existing_accepted_pairs(paths, universe)
        accepted = add_facility_audit_fields(accepted, paths)
        candidates = calculate_candidate_distances(accepted, config)
        selection, candidates = select_automatic_primaries(universe, candidates, config)
        selection, candidates = apply_manual_decisions(selection, candidates, manual_path, config)
        selection = add_review_flags(selection, config)
        sensors = pd.read_csv(paths.sensor_master, low_memory=False)
        require_columns(sensors, ["sensor_uid", "company", *NUMERIC_BETA_FIELDS], "sensor master")
        if sensors["sensor_uid"].duplicated().any():
            raise ValueError("Authoritative sensor_uid is not unique")
        source_company = sensors.set_index("sensor_uid")["company"].map(clean_string)
        accepted_company = accepted.set_index("sensor_uid")["company"]
        if not accepted_company.index.isin(source_company.index).all() or not accepted_company.eq(source_company.loc[accepted_company.index]).all():
            raise ValueError("Accepted candidate sensor/company identity disagrees with sensor master")
        master = build_primary_master(universe, selection, candidates, sensors)
        candidate_audit = build_candidate_audit(candidates, selection)
        review_queue = build_review_queue(master, candidate_audit)
        unmatched_review = pd.read_csv(paths.unmatched_review, low_memory=False)
        table_dir = staging / "tables"
        write_csv(master, table_dir / "wwtw_primary_sensor_master.csv")
        write_csv(candidate_audit, table_dir / "wwtw_primary_sensor_candidates.csv")
        write_csv(review_queue, table_dir / "wwtw_primary_sensor_review_queue.csv")
        qgis_dir = staging / "qgis"
        qgis_dir.mkdir(parents=True)
        gpkg = qgis_dir / "wwtw_primary_sensor_selection.gpkg"
        write_geopackage(gpkg, works, master, candidate_audit, unmatched_review)
        write_qml_styles(qgis_dir / "styles")
        config_snapshot = staging / "config_snapshot/wwtw_primary_sensor_selection.json"
        config_snapshot.parent.mkdir(parents=True)
        shutil.copy2(paths.config, config_snapshot)
        protected_after = snapshot(protected)
        validation = validate_release_gates(
            universe, master, candidate_audit, sensors, works, gpkg, config,
            protected_before, protected_after,
        )
        LOGGER.info("release_gates=%s", json.dumps(validation["checks"], sort_keys=True))
        for active_handler in list(LOGGER.handlers):
            active_handler.flush()
            active_handler.close()
            LOGGER.removeHandler(active_handler)
        source_files = {
            str(paths.characteristics.relative_to(paths.root)): paths.characteristics,
            str(paths.sensor_master.relative_to(paths.root)): paths.sensor_master,
            str(paths.accepted_pairs.relative_to(paths.root)): paths.accepted_pairs,
            str(paths.unmatched_review.relative_to(paths.root)): paths.unmatched_review,
            str(paths.place_gpkg.relative_to(paths.root)): paths.place_gpkg,
            str(paths.place_script.relative_to(paths.root)): paths.place_script,
            str(paths.place_config.relative_to(paths.root)): paths.place_config,
            str(paths.config.relative_to(paths.root)): paths.config,
        }
        if manual_path:
            source_files[str(manual_path)] = manual_path
        source_hashes = {label: sha256(path) for label, path in source_files.items()}
        source_frames = {
            str(paths.characteristics.relative_to(paths.root)): universe,
            str(paths.sensor_master.relative_to(paths.root)): sensors,
            str(paths.accepted_pairs.relative_to(paths.root)): accepted,
            str(paths.unmatched_review.relative_to(paths.root)): unmatched_review,
        }
        pre_audit_files = [path for path in staging.rglob("*") if path.is_file()]
        pre_audit_hashes = {str(path.relative_to(staging)): sha256(path) for path in pre_audit_files}
        audit = audit_report(
            run_id, created_at, paths, config, source_frames, source_hashes,
            pre_audit_hashes, universe, master, candidate_audit, review_queue,
            validation, excluded_wwtw_fields, manual_path,
        )
        audit_path = staging / "diagnostics/wwtw_primary_sensor_selection_audit.txt"
        audit_path.parent.mkdir(parents=True)
        audit_path.write_text(audit, encoding="utf-8", newline="\n")
        output_files = [path for path in staging.rglob("*") if path.is_file()]
        output_hashes = {str(path.relative_to(staging)): sha256(path) for path in output_files}
        assigned = int(master["primary_sensor_assigned"].sum())
        manifest = {
            "run_id": run_id, "created_at_utc": created_at, "status": "validated",
            "source_script": str(Path(__file__).resolve().relative_to(paths.root)),
            "configuration": config, "manual_decisions": str(manual_path) if manual_path else None,
            "source_hashes": source_hashes, "output_hashes": output_hashes,
            "metrics": {
                "authoritative_wwtw_count": len(master), "assigned_wwtw_count": assigned,
                "unassigned_wwtw_count": len(master) - assigned,
                "accepted_candidate_rows": len(candidate_audit),
                "nonselected_candidate_rows": int((~candidate_audit["selected_as_primary"]).sum()),
                "review_queue_rows": len(review_queue),
            },
            "release_gates": validation["checks"], "git": git_provenance(paths.root),
        }
        (staging / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        sums = {str(path.relative_to(staging)): sha256(path) for path in staging.rglob("*") if path.is_file()}
        (staging / "SHA256SUMS.json").write_text(
            json.dumps(sums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, final_run)
        atomic_text(branch_readme(), paths.branch / "README.md")
        atomic_text(f"runs/{run_id}\n", paths.branch / "LATEST.txt")
        print(json.dumps({
            "run_directory": str(final_run), "authoritative_wwtws": len(master),
            "assigned": assigned, "unassigned": len(master) - assigned,
            "layers": validation["layers"], "blyth": validation["blyth"],
            "place_branch_unchanged": protected_before == protected_after,
        }, indent=2, sort_keys=True, default=str))
        return 0
    except Exception:
        for active_handler in list(LOGGER.handlers):
            try:
                active_handler.close()
            finally:
                LOGGER.removeHandler(active_handler)
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

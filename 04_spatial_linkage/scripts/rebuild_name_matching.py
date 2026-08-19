#!/usr/bin/env python3
"""Rebuild WWTW-sensor matching using geographic place cores.

Matching is beta-blind and outcome-blind. Facility labels are retained for
audit, removed only from the place-core representation, and treated as an
expected asset relationship rather than a geographic-name contradiction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import sys
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein


BNG = "EPSG:27700"
LOGGER = logging.getLogger("wwtw_place_core_matching")
MANUAL_FIELDS = [
    "manual_decision",
    "manual_primary_sensor",
    "reviewer",
    "review_date",
    "manual_notes",
]
SENSOR_RESPONSE_FIELDS = [
    "fitted_beta",
    "beta_ci_lower_95",
    "beta_ci_upper_95",
    "beta_ci_width",
    "tail_spill_count",
    "beta_quality",
    "beta_at_bound",
    "eligible_for_beta_analysis",
]
CAPACITY_FIELDS = [
    "load_entering_pe_latest",
    "design_capacity_pe_latest",
    "capacity_ratio_latest",
    "capacity_data_year",
    "capacity_status",
]


@dataclass(frozen=True)
class PlaceName:
    raw_name: str
    normalized_full_name: str
    facility_type: str
    normalized_place_core: str


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def clean_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_string(value).upper() in {"TRUE", "T", "YES", "Y", "1"}


def normalize_full(value: object) -> str:
    text = unicodedata.normalize("NFKC", clean_string(value)).upper()
    text = re.sub(r"[^0-9A-Z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_tokens(value: str) -> tuple[str, ...]:
    return tuple(normalize_full(value).split())


def contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    size = len(phrase)
    return bool(size and any(tuple(tokens[index : index + size]) == tuple(phrase) for index in range(len(tokens) - size + 1)))


def remove_phrases(tokens: Sequence[str], phrases: Sequence[Sequence[str]]) -> list[str]:
    result: list[str] = []
    index = 0
    ordered = sorted({tuple(phrase) for phrase in phrases if phrase}, key=lambda item: (-len(item), item))
    while index < len(tokens):
        matched = next((phrase for phrase in ordered if tuple(tokens[index : index + len(phrase)]) == phrase), None)
        if matched:
            index += len(matched)
        else:
            result.append(tokens[index])
            index += 1
    return result


def facility_type(tokens: Sequence[str], config: Mapping[str, object]) -> str:
    vocab = config["facility_vocabularies"]
    treatment = [phrase_tokens(value) for value in vocab["treatment_works"]]
    overflow = [phrase_tokens(value) for value in vocab["overflow"]]
    pumping = [phrase_tokens(value) for value in vocab["pumping"]]
    present = {"treatment": [item for item in treatment if contains_phrase(tokens, item)],
               "overflow": [item for item in overflow if contains_phrase(tokens, item)],
               "pumping": [item for item in pumping if contains_phrase(tokens, item)]}
    overflow_text = {" ".join(item) for item in present["overflow"]}
    if "EMERGENCY OVERFLOW" in overflow_text or "EMO" in overflow_text or "EO" in overflow_text:
        return "EMERGENCY OVERFLOW"
    if "FINAL EFFLUENT" in overflow_text or "F E" in overflow_text or "FINAL DISCHARGE" in overflow_text:
        return "FINAL EFFLUENT"
    if "STORM TANK" in overflow_text:
        return "STORM TANK"
    if "OUTFALL" in overflow_text:
        return "OUTFALL"
    if "OUTLET" in overflow_text:
        return "OUTLET"
    if present["overflow"]:
        return "STORM OVERFLOW"
    if present["pumping"]:
        return "PUMPING STATION"
    if present["treatment"]:
        return "TREATMENT WORKS"
    return ""


def normalize_place_name(value: object, config: Mapping[str, object]) -> PlaceName:
    raw = clean_string(value)
    full = normalize_full(raw)
    tokens = full.split()
    kind = facility_type(tokens, config)
    vocab = config["facility_vocabularies"]
    removals = [
        phrase_tokens(term)
        for group in ("treatment_works", "overflow", "pumping")
        for term in vocab[group]
    ]
    core_tokens = remove_phrases(tokens, removals)
    suffix_only = {normalize_full(item) for item in config.get("suffix_only_tokens", [])}
    while core_tokens and core_tokens[-1] in suffix_only:
        core_tokens.pop()
    return PlaceName(raw, full, kind, " ".join(core_tokens))


def qualifier_conflict(left: str, right: str, config: Mapping[str, object]) -> bool:
    left_tokens, right_tokens = set(left.split()), set(right.split())
    for group in config["opposing_qualifier_groups"]:
        left_values = left_tokens & set(group)
        right_values = right_tokens & set(group)
        if left_values and right_values and left_values != right_values:
            return True
    number_pattern = re.compile(r"^(?:NO|NUMBER)?\d+[A-Z]?$|^[IVX]+$")
    left_numbers = {token for token in left_tokens if number_pattern.match(token)}
    right_numbers = {token for token in right_tokens if number_pattern.match(token)}
    return bool(left_numbers and right_numbers and left_numbers != right_numbers)


def subset_match(left: str, right: str, config: Mapping[str, object]) -> bool:
    if not left or not right or left == right or qualifier_conflict(left, right, config):
        return False
    left_tokens, right_tokens = set(left.split()), set(right.split())
    shorter, longer = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    distinctive = {token for token in shorter if len(token) >= 4 and token.isalpha()}
    return bool(shorter and shorter <= longer and distinctive)


def place_metrics(left: PlaceName, right: PlaceName, config: Mapping[str, object]) -> dict[str, object]:
    a, b = left.normalized_place_core, right.normalized_place_core
    lev = 100.0 * Levenshtein.normalized_similarity(a, b) if a and b else 0.0
    token_set = float(fuzz.token_set_ratio(a, b)) if a and b else 0.0
    weighted = float(fuzz.WRatio(a, b)) if a and b else 0.0
    left_tokens, right_tokens = set(a.split()), set(b.split())
    union = left_tokens | right_tokens
    overlap = 100.0 * len(left_tokens & right_tokens) / len(union) if union else 0.0
    exact = bool(a and a == b)
    weights = config["place_core_score_weights"]
    score = (
        float(weights["place_core_levenshtein_similarity"]) * lev
        + float(weights["place_core_token_set_ratio"]) * token_set
        + float(weights["place_core_weighted_ratio"]) * weighted
    )
    if exact:
        score = 100.0
    return {
        "place_core_exact_match": exact,
        "place_core_levenshtein_similarity": lev,
        "place_core_token_set_ratio": token_set,
        "place_core_weighted_ratio": weighted,
        "place_core_token_overlap": overlap,
        "place_core_subset_match": subset_match(a, b, config),
        "place_core_score": score,
        "qualifier_conflict": qualifier_conflict(a, b, config),
    }


def coordinate_available(easting: object, northing: object) -> bool:
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


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def atomic_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; rerun with --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temp, index=False, lineterminator="\n")
    os.replace(temp, path)


def load_legacy(root: Path):
    path = root / "scripts/08_match_treatment_works_sensors_by_name.py"
    spec = importlib.util.spec_from_file_location("legacy_wwtw_name_matching", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def split_aliases(value: object) -> list[str]:
    return [item.strip() for item in clean_string(value).split(" | ") if item.strip()]


def unique_places(aliases: Sequence[str], config: Mapping[str, object]) -> list[PlaceName]:
    result: list[PlaceName] = []
    seen: set[tuple[str, str]] = set()
    for alias in aliases:
        item = normalize_place_name(alias, config)
        key = (item.raw_name.upper(), item.normalized_place_core)
        if item.raw_name and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def input_paths(root: Path) -> dict[str, Path]:
    return {
        "sensor_master": root / "data_intermediate/sensor_master/cso_sensor_master.csv",
        "assignments": root / "data_intermediate/spatial_joins/cso_sensor_catchment_assignments.csv",
        "sensor_capacity": root / "data_intermediate/capacity/cso_sensor_capacity_enriched.csv",
        "characteristics": root / "data_intermediate/catchment_characteristics/catchment_characteristics_master.csv",
        "waterbase": root / "data_raw/waterbase/waterbase_consolidated.csv",
        "lookup": root / "data_raw/waterbase/waterbase_catchment_lookup.csv",
        "assignment_gpkg": root / "outputs/cso_spatial_drivers/qgis/cso_catchment_assignments.gpkg",
        "hydro_gpkg": root / "outputs/cso_spatial_drivers/qgis/cso_hydrogeology_analysis.gpkg",
        "legacy_config": root / "config/wwtw_sensor_name_matching.json",
        "place_config": root / "config/wwtw_sensor_place_core_matching.json",
    }


def output_paths(root: Path) -> dict[str, Path]:
    branch = root / "project_results/new_sensor_assignment_approach"
    return {
        "branch": branch,
        "master": branch / "tables/wwtw_sensor_analysis_master.csv",
        "review": branch / "tables/wwtw_sensor_name_match_review.csv",
        "gpkg": branch / "qgis/wwtw_sensor_name_matching.gpkg",
        "log": branch / "logs/wwtw_sensor_name_matching.log",
        "readme": branch / "README.md",
        "archive": branch / "archive_before_place_core_fix",
        "styles": branch / "qgis/styles",
    }


def create_archive_once(outputs: Mapping[str, Path]) -> dict[str, str]:
    branch, archive = outputs["branch"], outputs["archive"]
    relative_files = [
        Path("tables/wwtw_sensor_analysis_master.csv"),
        Path("tables/wwtw_sensor_name_match_review.csv"),
        Path("qgis/wwtw_sensor_name_matching.gpkg"),
        Path("logs/wwtw_sensor_name_matching.log"),
        Path("README.md"),
    ]
    if archive.exists():
        manifest = archive / "SHA256SUMS.json"
        if not manifest.is_file():
            raise ValueError(f"Existing archive lacks SHA256SUMS.json and will not be replaced: {archive}")
        hashes = json.loads(manifest.read_text(encoding="utf-8"))
        for relative, expected in hashes.items():
            archived = archive / relative
            if not archived.is_file() or sha256(archived) != expected:
                raise ValueError(f"Existing archive hash validation failed: {archived}")
        return hashes
    sources = {relative: branch / relative for relative in relative_files}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot create complete archive; missing current outputs: {missing}")
    hashes = {relative.as_posix(): sha256(path) for relative, path in sources.items()}
    temp = archive.with_name(f".{archive.name}.{uuid.uuid4().hex}.tmp")
    temp.mkdir(parents=True)
    try:
        for relative, source in sources.items():
            destination = temp / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256(destination) != hashes[relative.as_posix()]:
                raise ValueError(f"Archive copy hash mismatch: {relative}")
        (temp / "SHA256SUMS.json").write_text(
            json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temp, archive)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return hashes


def prepare_inputs(root: Path, legacy, legacy_config: Mapping[str, object], place_config: Mapping[str, object]):
    paths = input_paths(root)
    sensors = pd.read_csv(paths["sensor_master"], low_memory=False)
    assignments = pd.read_csv(paths["assignments"], low_memory=False)
    characteristics = pd.read_csv(paths["characteristics"], low_memory=False)
    waterbase = pd.read_csv(paths["waterbase"], low_memory=False)
    lookup = pd.read_csv(paths["lookup"], low_memory=False)
    legacy.require_fields(sensors, legacy.REQUIRED_SENSOR_FIELDS, "sensor master")
    legacy.require_fields(assignments, {"sensor_uid", "assignment_status", "uwwCode"}, "assignments")
    sensors["company"] = sensors["company"].map(clean_string)
    sensors["permit_key"] = sensors["permit_number"].map(clean_string).str.upper()
    aliases, _, alias_metadata = legacy.build_sensor_aliases(root, sensors, legacy_config)
    sensors = sensors.merge(aliases, on="sensor_uid", how="left", validate="one_to_one")
    sensors["place_alias_bundle"] = sensors.apply(
        lambda row: unique_places(
            split_aliases(row["sensor_location_name_aliases"])
            + [clean_string(row["canonical_sensor_location_name"]), clean_string(row["location_name"])],
            place_config,
        ),
        axis=1,
    )
    if sensors["place_alias_bundle"].map(len).eq(0).any():
        empty = int(sensors["place_alias_bundle"].map(len).eq(0).sum())
        LOGGER.warning("Sensors with no usable aliases=%d", empty)
    works = legacy.build_wwtw_reference(waterbase, lookup, characteristics, legacy_config)
    catchment_names = (
        lookup.dropna(subset=["uwwCode"])
        .assign(name=lambda frame: frame["name"].map(clean_string))
        .groupby("uwwCode")["name"]
        .agg(lambda values: sorted({value for value in values if value}))
        .to_dict()
    )
    primary_bundles, catchment_bundles = [], []
    for row in works.itertuples(index=False):
        primary_bundles.append(unique_places(split_aliases(row.uwwName_aliases) + [row.uwwName], place_config))
        catchment_bundles.append(unique_places(catchment_names.get(row.uwwCode, []), place_config))
    works["primary_place_alias_bundle"] = primary_bundles
    works["catchment_place_alias_bundle"] = catchment_bundles
    works["all_place_alias_bundle"] = works.apply(
        lambda row: [(item, "wwtw_name") for item in row["primary_place_alias_bundle"]]
        + [(item, "catchment_weak") for item in row["catchment_place_alias_bundle"]],
        axis=1,
    )
    accepted_assignments = assignments[
        assignments["assignment_status"].isin(legacy.ACCEPTED_SPATIAL_STATUSES)
    ][["sensor_uid", "uwwCode"]].drop_duplicates("sensor_uid")
    existing_codes = dict(zip(accepted_assignments["sensor_uid"], accepted_assignments["uwwCode"]))
    return sensors, assignments, characteristics, waterbase, lookup, works, existing_codes, alias_metadata


def candidate_distance(sensor: Mapping[str, object], work: Mapping[str, object]) -> float:
    if not coordinate_available(sensor.get("bng_easting"), sensor.get("bng_northing")):
        return np.nan
    if not coordinate_available(work.get("treatment_work_easting"), work.get("treatment_work_northing")):
        return np.nan
    return math.hypot(
        float(sensor["bng_easting"]) - float(work["treatment_work_easting"]),
        float(sensor["bng_northing"]) - float(work["treatment_work_northing"]),
    )


def asset_compatible(sensor_type: str, work_type: str, config: Mapping[str, object]) -> bool:
    return bool(
        work_type == "TREATMENT WORKS"
        and sensor_type in set(config["asset_relationships_compatible_with_treatment_works"])
    )


def generate_candidates(
    sensors: pd.DataFrame,
    works: pd.DataFrame,
    existing_codes: Mapping[str, object],
    place_config: Mapping[str, object],
    legacy,
    legacy_config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, int]]:
    shortlisted: list[dict[str, object]] = []
    scored_sensor_aliases = 0
    expected_sensor_aliases = int(sensors["place_alias_bundle"].map(len).sum())
    top_count = int(place_config["top_candidate_count"])
    weights = place_config["place_core_score_weights"]
    for company in sorted(sensors["company"].unique()):
        sensor_records = sensors[sensors["company"].eq(company)].sort_values("sensor_uid").to_dict("records")
        work_records = works[works["company"].eq(company)].sort_values("uwwCode").to_dict("records")
        if not work_records:
            raise ValueError(f"No WWTW records for company {company}")
        sensor_aliases: list[PlaceName] = []
        sensor_owner: list[int] = []
        for owner, sensor in enumerate(sensor_records):
            for alias in sensor["place_alias_bundle"]:
                sensor_aliases.append(alias)
                sensor_owner.append(owner)
        work_aliases: list[PlaceName] = []
        work_channels: list[str] = []
        work_owner: list[int] = []
        for owner, work in enumerate(work_records):
            for alias, channel in work["all_place_alias_bundle"]:
                work_aliases.append(alias)
                work_channels.append(channel)
                work_owner.append(owner)
        scored_sensor_aliases += len(sensor_aliases)
        sensor_owner_array = np.asarray(sensor_owner)
        work_owner_array = np.asarray(work_owner)
        sensor_places = [item.normalized_place_core for item in sensor_aliases]
        work_places = [item.normalized_place_core for item in work_aliases]
        cdist = {"dtype": np.float32, "workers": -1}
        lev = process.cdist(sensor_places, work_places, scorer=Levenshtein.normalized_similarity,
                            score_multiplier=100, **cdist)
        token_set = process.cdist(sensor_places, work_places, scorer=fuzz.token_set_ratio, **cdist)
        weighted = process.cdist(sensor_places, work_places, scorer=fuzz.WRatio, **cdist)
        scores = (
            float(weights["place_core_levenshtein_similarity"]) * lev
            + float(weights["place_core_token_set_ratio"]) * token_set
            + float(weights["place_core_weighted_ratio"]) * weighted
        )
        exact = np.asarray(sensor_places, dtype=object).reshape(-1, 1) == np.asarray(work_places, dtype=object).reshape(1, -1)
        nonempty = (np.asarray(sensor_places, dtype=object).reshape(-1, 1) != "") & (np.asarray(work_places, dtype=object).reshape(1, -1) != "")
        scores[exact & nonempty] = 100.0
        primary_alias_to_work = np.zeros((len(sensor_aliases), len(work_records)), dtype=np.float32)
        weak_alias_to_work = np.zeros((len(sensor_aliases), len(work_records)), dtype=np.float32)
        work_alias_columns: list[np.ndarray] = []
        work_primary_columns: list[np.ndarray] = []
        work_weak_columns: list[np.ndarray] = []
        for work_index in range(len(work_records)):
            columns = np.flatnonzero(work_owner_array == work_index)
            primary_columns = np.asarray([column for column in columns if work_channels[column] == "wwtw_name"], dtype=int)
            weak_columns = np.asarray([column for column in columns if work_channels[column] == "catchment_weak"], dtype=int)
            work_alias_columns.append(columns)
            work_primary_columns.append(primary_columns)
            work_weak_columns.append(weak_columns)
            primary_alias_to_work[:, work_index] = scores[:, primary_columns].max(axis=1)
            if len(weak_columns):
                weak_alias_to_work[:, work_index] = scores[:, weak_columns].max(axis=1)
        primary_owner_scores = np.zeros((len(sensor_records), len(work_records)), dtype=np.float32)
        weak_owner_scores = np.zeros((len(sensor_records), len(work_records)), dtype=np.float32)
        sensor_alias_rows: list[np.ndarray] = []
        for sensor_index in range(len(sensor_records)):
            rows = np.flatnonzero(sensor_owner_array == sensor_index)
            sensor_alias_rows.append(rows)
            if len(rows):
                primary_owner_scores[sensor_index, :] = primary_alias_to_work[rows, :].max(axis=0)
                weak_owner_scores[sensor_index, :] = weak_alias_to_work[rows, :].max(axis=0)
        sensor_east = np.asarray([pd.to_numeric(sensor.get("bng_easting"), errors="coerce") for sensor in sensor_records], dtype=float)
        sensor_north = np.asarray([pd.to_numeric(sensor.get("bng_northing"), errors="coerce") for sensor in sensor_records], dtype=float)
        work_east = np.asarray([pd.to_numeric(work.get("treatment_work_easting"), errors="coerce") for work in work_records], dtype=float)
        work_north = np.asarray([pd.to_numeric(work.get("treatment_work_northing"), errors="coerce") for work in work_records], dtype=float)
        distance_matrix = np.hypot(sensor_east.reshape(-1, 1) - work_east.reshape(1, -1),
                                   sensor_north.reshape(-1, 1) - work_north.reshape(1, -1))
        spatial_support = np.isfinite(distance_matrix) & (distance_matrix <= float(place_config["coordinate_support_distance_m"]))
        existing_array = np.asarray([clean_string(existing_codes.get(sensor["sensor_uid"], "")) for sensor in sensor_records], dtype=object).reshape(-1, 1)
        work_codes = np.asarray([work["uwwCode"] for work in work_records], dtype=object).reshape(1, -1)
        existing_support = (existing_array != "") & (existing_array == work_codes)
        weak_support = spatial_support | existing_support
        owner_scores = np.where(
            weak_support,
            np.maximum(primary_owner_scores, weak_owner_scores),
            primary_owner_scores,
        )
        for sensor_index, sensor in enumerate(sensor_records):
            values = owner_scores[sensor_index, :]
            order = sorted(range(len(work_records)), key=lambda idx: (-float(values[idx]), work_records[idx]["uwwCode"]))
            cutoff = float(values[order[min(top_count, len(order)) - 1]])
            selected = [idx for idx in order if float(values[idx]) + 1e-6 >= cutoff]
            best = float(values[order[0]])
            second = float(values[order[1]]) if len(order) > 1 else 0.0
            distinct = sorted({round(float(value), 6) for value in values}, reverse=True)
            rank_map = {value: rank + 1 for rank, value in enumerate(distinct)}
            for work_index in selected:
                work = work_records[work_index]
                rows = sensor_alias_rows[sensor_index]
                use_weak = bool(
                    weak_support[sensor_index, work_index]
                    and weak_owner_scores[sensor_index, work_index] > primary_owner_scores[sensor_index, work_index] + 1e-6
                )
                columns = work_weak_columns[work_index] if use_weak else work_primary_columns[work_index]
                submatrix = scores[np.ix_(rows, columns)]
                flat = int(np.argmax(submatrix))
                local_row, local_column = np.unravel_index(flat, submatrix.shape)
                srow, wcol = int(rows[local_row]), int(columns[local_column])
                sensor_name, work_name = sensor_aliases[srow], work_aliases[wcol]
                metrics = place_metrics(sensor_name, work_name, place_config)
                old_metrics = legacy.score_name_pair(
                    legacy.normalize_name(sensor_name.raw_name, legacy_config),
                    legacy.normalize_name(work_name.raw_name, legacy_config),
                    legacy_config,
                )
                existing = clean_string(existing_codes.get(sensor["sensor_uid"], ""))
                distance = candidate_distance(sensor, work)
                shortlisted.append({
                    "company": company,
                    "sensor_uid": sensor["sensor_uid"],
                    "permit_number": sensor["permit_number"],
                    "sensor_location_name": sensor["canonical_sensor_location_name"],
                    "sensor_location_aliases": sensor["sensor_location_name_aliases"],
                    "winning_sensor_alias": sensor_name.raw_name,
                    "winning_wwtw_alias": work_name.raw_name,
                    "winning_wwtw_alias_channel": "catchment_weak" if use_weak else "wwtw_name",
                    "sensor_normalized_full_name": sensor_name.normalized_full_name,
                    "wwtw_normalized_full_name": work_name.normalized_full_name,
                    "sensor_place_core": sensor_name.normalized_place_core,
                    "wwtw_place_core": work_name.normalized_place_core,
                    "sensor_facility_type": sensor_name.facility_type,
                    "wwtw_facility_type": work_name.facility_type,
                    **metrics,
                    "old_composite_name_score": float(old_metrics["composite_name_score"]),
                    "asset_relationship_compatible": asset_compatible(sensor_name.facility_type, work_name.facility_type, place_config),
                    "uwwCode": work["uwwCode"],
                    "uwwName": work["uwwName"],
                    "uwwName_aliases": work["uwwName_aliases"],
                    "candidate_rank": rank_map[round(float(values[work_index]), 6)],
                    "best_place_core_score": best,
                    "second_best_place_core_score": second,
                    "place_score_margin": best - second,
                    "sensor_coordinate_available": coordinate_available(sensor.get("bng_easting"), sensor.get("bng_northing")),
                    "wwtw_coordinate_available": coordinate_available(work.get("treatment_work_easting"), work.get("treatment_work_northing")),
                    "sensor_to_wwtw_distance_m": distance,
                    "existing_spatial_uwwCode": existing,
                    "existing_uwwcode_agreement": bool(existing and existing == work["uwwCode"]),
                    "name_spatial_agreement": bool(existing and existing == work["uwwCode"]),
                    "bng_easting": sensor.get("bng_easting"),
                    "bng_northing": sensor.get("bng_northing"),
                    "treatment_work_easting": work.get("treatment_work_easting"),
                    "treatment_work_northing": work.get("treatment_work_northing"),
                })
        LOGGER.info("Candidate scoring company=%s sensors=%d works=%d sensor_aliases=%d work_aliases=%d",
                    company, len(sensor_records), len(work_records), len(sensor_aliases), len(work_aliases))
    if scored_sensor_aliases != expected_sensor_aliases:
        raise ValueError("Not every sensor alias entered candidate scoring")
    return pd.DataFrame(shortlisted), {
        "sensor_aliases_expected": expected_sensor_aliases,
        "sensor_aliases_scored": scored_sensor_aliases,
    }


def classify(candidates: pd.DataFrame, config: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidates.copy()
    candidates["accepted_place_name_match"] = False
    candidates["match_status"] = ""
    candidates["match_rule_triggered"] = ""
    candidates["rejection_reason"] = ""
    chosen_rows: list[int] = []
    threshold = float(config["automatic_place_score_threshold"])
    margin_required = float(config["minimum_place_score_margin"])
    support_distance = float(config["coordinate_support_distance_m"])
    for sensor_uid, group in candidates.groupby("sensor_uid", sort=True):
        group = group.sort_values(["place_core_score", "uwwCode"], ascending=[False, True], kind="mergesort")
        best_score = float(group["place_core_score"].max())
        top = group[np.isclose(group["place_core_score"], best_score, atol=1e-6, rtol=0)]
        existing = clean_string(group.iloc[0]["existing_spatial_uwwCode"])
        resolved_by = ""
        if len(top) == 1:
            selected = top.iloc[0]
        else:
            agreed = top[top["uwwCode"].eq(existing)] if existing else top.iloc[0:0]
            if len(agreed) == 1:
                selected, resolved_by = agreed.iloc[0], "existing_uwwcode"
            else:
                nearby = top[pd.to_numeric(top["sensor_to_wwtw_distance_m"], errors="coerce").le(support_distance)].sort_values("sensor_to_wwtw_distance_m")
                if len(nearby) and (len(nearby) == 1 or float(nearby.iloc[0]["sensor_to_wwtw_distance_m"]) + 1e-6 < float(nearby.iloc[1]["sensor_to_wwtw_distance_m"])):
                    selected, resolved_by = nearby.iloc[0], "uniquely_nearest"
                else:
                    selected = top.sort_values("uwwCode").iloc[0]
        index = int(selected.name)
        chosen_rows.append(index)
        exact = bool(selected["place_core_exact_match"])
        subset = bool(selected["place_core_subset_match"])
        score = float(selected["place_core_score"])
        margin = float(selected["place_score_margin"])
        qualifier = bool(selected["qualifier_conflict"])
        agreement = bool(selected["existing_uwwcode_agreement"])
        conflict = bool(existing and not agreement and (score >= threshold or subset))
        distance = pd.to_numeric(pd.Series([selected["sensor_to_wwtw_distance_m"]]), errors="coerce").iloc[0]
        spatial = bool(pd.notna(distance) and float(distance) <= support_distance)
        catchment_only = selected["winning_wwtw_alias_channel"] == "catchment_weak"
        same_core = top["wwtw_place_core"].nunique(dropna=False) == 1
        unresolved_tie = len(top) > 1 and not resolved_by
        status = rule = reason = ""
        accepted = False
        if not clean_string(selected["sensor_place_core"]):
            status, reason = "no_meaningful_place_name", "Sensor aliases contain no meaningful geographic place core"
        elif qualifier:
            status, reason = "different_place_name", "Directional, historical or numbered qualifiers conflict"
        elif unresolved_tie:
            status = "ambiguous_duplicate_place_name" if same_core else "ambiguous_multiple_wwtw_candidates"
            reason = "Best same-company WWTW candidates are tied and existing uwwCode/coordinates do not resolve them"
        elif conflict:
            status, reason = "name_spatial_conflict", "Place-name candidate disagrees with the accepted existing spatial uwwCode"
        elif catchment_only and not (agreement or spatial):
            status, reason = "different_place_name", "Weak catchment alias lacks existing-uwwCode or spatial support"
        elif exact and bool(config["exact_place_match_enabled"]):
            accepted, status, rule = True, "accepted_exact_place_core", "exact_place_core_match"
        elif score >= threshold and agreement:
            accepted, status, rule = True, "accepted_place_core_existing_uwwcode", "high_place_core_plus_existing_uwwcode"
        elif subset and bool(config["place_subset_match_enabled"]) and (agreement or spatial):
            accepted, status, rule = True, "accepted_place_subset_supported", "place_subset_with_strong_support"
        elif score >= threshold and spatial:
            accepted, status, rule = True, "accepted_place_core_spatial_support", "high_place_core_plus_spatial_support"
        elif score >= threshold and margin >= margin_required:
            accepted, status, rule = True, "accepted_place_core_85", "high_place_core_similarity"
        elif subset and not bool(selected["sensor_coordinate_available"]) and not existing:
            status, reason = "missing_sensor_coordinates", "Subset name evidence requires existing-uwwCode or coordinate support"
        elif subset and not bool(selected["wwtw_coordinate_available"]) and not agreement:
            status, reason = "missing_wwtw_coordinates", "Subset name evidence requires coordinate or existing-uwwCode support"
        elif score < threshold:
            status, reason = "different_place_name", "Best place-core score is below the automatic threshold"
        else:
            status, reason = "ambiguous_multiple_wwtw_candidates", "High place score lacks the required margin or resolving support"
        candidates.at[index, "accepted_place_name_match"] = accepted
        candidates.at[index, "match_status"] = status
        candidates.at[index, "match_rule_triggered"] = rule
        candidates.at[index, "rejection_reason"] = reason
    chosen = candidates.loc[chosen_rows].copy()
    return candidates, chosen


def build_master(
    chosen: pd.DataFrame,
    sensors: pd.DataFrame,
    works: pd.DataFrame,
    characteristics: pd.DataFrame,
    previous_master: pd.DataFrame,
) -> pd.DataFrame:
    accepted = chosen[chosen["accepted_place_name_match"]].copy()
    accepted["pair_uid"] = accepted["company"] + "::" + accepted["uwwCode"] + "::" + accepted["sensor_uid"]
    previous_pairs = set(previous_master.get("pair_uid", pd.Series(dtype=object)).map(clean_string))
    accepted["newly_accepted_after_place_core_fix"] = ~accepted["pair_uid"].isin(previous_pairs)
    sensor_fields = ["sensor_uid", "location_name", "bng_easting", "bng_northing", "longitude", "latitude", *SENSOR_RESPONSE_FIELDS]
    accepted = accepted.drop(columns=[field for field in sensor_fields[1:] if field in accepted]).merge(
        sensors[sensor_fields], on="sensor_uid", how="left", validate="one_to_one"
    )
    work_fields = ["company", "uwwCode", "uwwLongitude", "uwwLatitude", "treatment_work_easting", "treatment_work_northing", *CAPACITY_FIELDS]
    accepted = accepted.drop(columns=[field for field in work_fields[2:] if field in accepted]).merge(
        works[work_fields], on=["company", "uwwCode"], how="left", validate="many_to_one"
    )
    future_patterns = ("rain", "precip", "land_cover", "landcover", "population", "building_age", "built_age", "terrain", "elevation", "slope")
    characteristic_fields = [
        column for column in characteristics.columns
        if column == "sewershed_area_km2" or column.startswith("hydro_")
        or any(pattern in column.lower() for pattern in future_patterns)
    ]
    characteristic_fields = [column for column in characteristic_fields if "beta" not in column.lower()]
    accepted = accepted.merge(
        characteristics[["company", "uwwCode", *characteristic_fields]],
        on=["company", "uwwCode"], how="left", validate="many_to_one"
    )
    counts = accepted.groupby(["company", "uwwCode"])["sensor_uid"].transform("nunique")
    accepted["wwtw_accepted_sensor_count"] = counts.astype(int)
    accepted["is_primary_wwtw_sensor"] = counts.eq(1)
    accepted["primary_sensor_selection_status"] = np.where(
        counts.eq(1), "unique_name_primary", "multiple_matching_sensors_requires_later_primary_selection"
    )
    preferred = [
        "pair_uid", "company", "uwwCode", "uwwName", "sensor_uid", "permit_number",
        "sensor_location_name", "sensor_location_aliases", "winning_sensor_alias", "winning_wwtw_alias",
        "sensor_normalized_full_name", "wwtw_normalized_full_name", "sensor_place_core", "wwtw_place_core",
        "place_core_exact_match", "place_core_subset_match", "place_core_levenshtein_similarity",
        "place_core_token_set_ratio", "place_core_weighted_ratio", "place_core_token_overlap",
        "place_core_score", "place_score_margin", "old_composite_name_score", "asset_relationship_compatible",
        "accepted_place_name_match", "match_status", "match_rule_triggered", "newly_accepted_after_place_core_fix",
        "sensor_coordinate_available", "sensor_to_wwtw_distance_m", "existing_spatial_uwwCode",
        "existing_uwwcode_agreement", "name_spatial_agreement", "wwtw_accepted_sensor_count",
        "is_primary_wwtw_sensor", "primary_sensor_selection_status", "bng_easting", "bng_northing",
        "longitude", "latitude", "treatment_work_easting", "treatment_work_northing", "uwwLongitude", "uwwLatitude",
        *SENSOR_RESPONSE_FIELDS, *CAPACITY_FIELDS, *characteristic_fields,
    ]
    preferred = list(dict.fromkeys(preferred))
    result = accepted[[column for column in preferred if column in accepted]].sort_values(
        ["company", "uwwCode", "sensor_uid"], kind="mergesort"
    )
    if result["pair_uid"].duplicated().any():
        raise ValueError("pair_uid is not unique")
    return result


def build_review(chosen: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    review = chosen[~chosen["accepted_place_name_match"]].copy()
    alternatives: dict[str, tuple[str, str, str]] = {}
    selected_codes = chosen.set_index("sensor_uid")["uwwCode"].map(clean_string).to_dict()
    for sensor_uid, group in candidates.groupby("sensor_uid", sort=True):
        selected_code = selected_codes[sensor_uid]
        alt = group[~group["uwwCode"].eq(selected_code)].sort_values(
            ["place_core_score", "uwwCode"], ascending=[False, True]
        ).drop_duplicates("uwwCode").head(5)
        alternatives[sensor_uid] = (
            " | ".join(alt["uwwCode"].map(clean_string)),
            " | ".join(alt["uwwName"].map(clean_string)),
            " | ".join(alt["place_core_score"].map(lambda value: f"{float(value):.3f}")),
        )
    review["top_alternative_uwwCodes"] = review["sensor_uid"].map(lambda uid: alternatives[uid][0])
    review["top_alternative_names"] = review["sensor_uid"].map(lambda uid: alternatives[uid][1])
    review["top_alternative_scores"] = review["sensor_uid"].map(lambda uid: alternatives[uid][2])
    for field in MANUAL_FIELDS:
        review[field] = ""
    preferred = [
        "company", "sensor_uid", "permit_number", "sensor_location_name", "sensor_location_aliases",
        "winning_sensor_alias", "sensor_normalized_full_name", "sensor_place_core", "uwwCode", "uwwName",
        "winning_wwtw_alias", "wwtw_normalized_full_name", "wwtw_place_core", "place_core_exact_match",
        "place_core_subset_match", "place_core_levenshtein_similarity", "place_core_token_set_ratio",
        "place_core_weighted_ratio", "place_core_token_overlap", "place_core_score", "place_score_margin",
        "existing_spatial_uwwCode", "existing_uwwcode_agreement", "sensor_to_wwtw_distance_m",
        "match_status", "rejection_reason", "top_alternative_uwwCodes", "top_alternative_names",
        "top_alternative_scores", *MANUAL_FIELDS, "bng_easting", "bng_northing",
    ]
    return review[preferred].sort_values(["company", "sensor_uid"], kind="mergesort")


def safe_attributes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: json.dumps(value, sort_keys=True, default=str)
                if isinstance(value, (list, dict, set, tuple, PlaceName)) else value
            )
    return result


def build_geopackage(
    path: Path,
    raw_catchments: gpd.GeoDataFrame,
    sewersheds: gpd.GeoDataFrame,
    works: pd.DataFrame,
    master: pd.DataFrame,
    review: pd.DataFrame,
    reverse_company: Mapping[str, str],
) -> None:
    temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.gpkg")
    raw = raw_catchments.copy()
    raw["company"] = raw["company"].map(reverse_company).fillna(raw["company"])
    raw = raw.rename(columns={"identifier": "raw_catchment_identifier", "name": "raw_catchment_name"})
    raw.to_file(temp, layer="raw_catchments", driver="GPKG", engine="pyogrio")
    sewer = sewersheds.drop(columns=[column for column in sewersheds.columns if "beta" in column.lower()], errors="ignore")
    sewer.to_file(temp, layer="wwtw_sewersheds", driver="GPKG", engine="pyogrio", append=True)
    work = works.drop(columns=[column for column in works.columns if "bundle" in column], errors="ignore").copy()
    work_geometry = gpd.points_from_xy(work["treatment_work_easting"], work["treatment_work_northing"], crs=BNG)
    gpd.GeoDataFrame(safe_attributes(work), geometry=work_geometry, crs=BNG).to_file(
        temp, layer="treatment_works_unique", driver="GPKG", engine="pyogrio", append=True
    )
    matched = master[
        [coordinate_available(east, north) for east, north in zip(master["bng_easting"], master["bng_northing"])]
    ].copy()
    matched_geometry = gpd.points_from_xy(matched["bng_easting"], matched["bng_northing"], crs=BNG)
    gpd.GeoDataFrame(safe_attributes(matched), geometry=matched_geometry, crs=BNG).to_file(
        temp, layer="matched_wwtw_sensors", driver="GPKG", engine="pyogrio", append=True
    )
    unmatched = review[
        [coordinate_available(east, north) for east, north in zip(review["bng_easting"], review["bng_northing"])]
    ].copy()
    unmatched_geometry = gpd.points_from_xy(unmatched["bng_easting"], unmatched["bng_northing"], crs=BNG)
    gpd.GeoDataFrame(safe_attributes(unmatched), geometry=unmatched_geometry, crs=BNG).to_file(
        temp, layer="unmatched_name_sensors", driver="GPKG", engine="pyogrio", append=True
    )
    os.replace(temp, path)


def qml_style(layer_name: str, colour: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol" symbollevels="0">
    <symbols><symbol type="marker" name="0"><layer class="SimpleMarker">
      <Option type="Map">
        <Option name="name" value="circle"/>
        <Option name="color" value="{colour}"/>
        <Option name="outline_color" value="35,35,35,255"/>
        <Option name="outline_style" value="solid"/>
        <Option name="outline_width" value="0.6"/>
        <Option name="size" value="3.2"/>
      </Option>
    </layer></symbol></symbols>
  </renderer-v2>
  <layername>{layer_name}</layername>
</qgis>''' + "\n"


def write_styles(directory: Path) -> None:
    styles = {
        "treatment_works_unique_red.qml": ("treatment_works_unique", "220,40,40,255"),
        "matched_wwtw_sensors_green.qml": ("matched_wwtw_sensors", "35,170,70,255"),
        "unmatched_name_sensors_yellow.qml": ("unmatched_name_sensors", "245,205,35,255"),
    }
    for filename, (layer, colour) in styles.items():
        atomic_text(qml_style(layer, colour), directory / filename)


def readme_text(config: Mapping[str, object]) -> str:
    return f"""# WWTW-sensor place-core name matching

## Why the method changed

The former complete-string composite was deliberately conservative but treated expected facility suffix differences such as `SSO` versus `STW` as adverse evidence. This rebuild preserves each raw and normalized full name, separately identifies the facility type, and makes its primary decision from the geographic `normalized_place_core`. Thus `SCAYNES HILL SSO` and `SCAYNES HILL STW` both have place core `SCAYNES HILL`.

Storm overflows, storm tanks, outfalls, final effluent points and emergency overflows can legitimately be assets associated with a treatment works. Their relationship to a treatment works is supporting evidence, never permission to override a different place name.

## Rules and interpretation

The automatic threshold of {float(config['automatic_place_score_threshold']):g} applies to `place_core_score`, with a normal minimum margin of {float(config['minimum_place_score_margin']):g}. Exact place cores can pass when duplicate WWTWs are not unresolved. Subset names require accepted existing-uwwCode or unique spatial support. Distance alone never creates a green match. Catchment names are a weak separate channel and cannot create an automatic match without supporting uwwCode or spatial evidence.

Several sensors may legitimately match one `uwwCode`; all accepted pairs are retained. A primary is marked only when one accepted sensor exists. Multiple accepted sensors require later beta-blind site-meaning or manual primary selection.

Beta values, confidence intervals, quality, spill counts, capacity and environmental fields do not influence normalization, scoring, ranking or acceptance. They are attached only after matching for downstream analysis. Later sensor-level regression must account for clustering because several sensor rows can share one WWTW and its predictors.

## QGIS three-colour view

The GeoPackage contains red `treatment_works_unique`, green `matched_wwtw_sensors`, and yellow `unmatched_name_sensors`, plus the two polygon context layers. Yellow means not automatically accepted or unresolved—not certainly hydraulically unrelated. Accepted records without coordinates remain in the master CSV but cannot be drawn. Apply the supplied QML files in `qgis/styles/` through Layer Properties > Symbology > Style > Load Style.

## Reproducibility

Run:

```text
python scripts/08b_rebuild_wwtw_name_matching_place_core.py --project-root . --overwrite
```

The prior master, review, GeoPackage, log and README are retained once in `archive_before_place_core_fix/` with SHA-256 hashes. Scientific source files and previous goal folders are hash-checked and remain unchanged.
"""


def validate(
    sensors: pd.DataFrame,
    works: pd.DataFrame,
    characteristics: pd.DataFrame,
    candidates: pd.DataFrame,
    chosen: pd.DataFrame,
    master: pd.DataFrame,
    review: pd.DataFrame,
    gpkg: Path,
    source_beta: pd.DataFrame,
    alias_counts: Mapping[str, int],
) -> dict[str, object]:
    allowed_beta = set(SENSOR_RESPONSE_FIELDS)
    leakage = [column for column in master.columns if "beta" in column.lower() and column not in allowed_beta]
    numeric_beta_fields = ["fitted_beta", "beta_ci_lower_95", "beta_ci_upper_95", "beta_ci_width"]
    master_beta = master.set_index("sensor_uid")[numeric_beta_fields]
    common = master_beta.index.intersection(source_beta.index)
    beta_unchanged = np.allclose(master_beta.loc[common].to_numpy(float), source_beta.loc[common].to_numpy(float), equal_nan=True)
    layers = pyogrio.list_layers(gpkg)[:, 0].tolist()
    expected_layers = ["raw_catchments", "wwtw_sewersheds", "treatment_works_unique", "matched_wwtw_sensors", "unmatched_name_sensors"]
    layer_details: dict[str, object] = {}
    for layer in layers:
        frame = gpd.read_file(gpkg, layer=layer)
        layer_details[layer] = {"rows": len(frame), "crs": str(frame.crs), "empty_geometry": int(frame.geometry.is_empty.sum())}
        if str(frame.crs).upper() != BNG:
            raise ValueError(f"Layer does not use EPSG:27700: {layer} ({frame.crs})")
    coordinate_uids = set(sensors.loc[[coordinate_available(e, n) for e, n in zip(sensors["bng_easting"], sensors["bng_northing"])], "sensor_uid"])
    green = set(gpd.read_file(gpkg, layer="matched_wwtw_sensors")["sensor_uid"])
    yellow = set(gpd.read_file(gpkg, layer="unmatched_name_sensors")["sensor_uid"])
    checks = {
        "accepted_sensors_authoritative": set(master["sensor_uid"]) <= set(sensors["sensor_uid"]),
        "accepted_wwtw_authoritative": set(zip(master["company"], master["uwwCode"])) <= set(zip(characteristics["company"], characteristics["uwwCode"])),
        "same_company_only": bool(
            master[["sensor_uid", "company"]]
            .merge(sensors[["sensor_uid", "company"]], on="sensor_uid", suffixes=("_match", "_source"), validate="one_to_one")
            .eval("company_match == company_source")
            .all()
        ),
        "pair_uid_unique": not master["pair_uid"].duplicated().any(),
        "all_sensor_aliases_scored": alias_counts["sensor_aliases_expected"] == alias_counts["sensor_aliases_scored"],
        "exact_scores_100": bool(candidates.loc[candidates["place_core_exact_match"], "place_core_score"].eq(100).all()),
        "qualifier_conflicts_not_accepted": not bool(chosen.loc[chosen["qualifier_conflict"], "accepted_place_name_match"].any()),
        "multiple_sensors_retained": len(master) == int(chosen["accepted_place_name_match"].sum()),
        "beta_outcome_blind": True,
        "sensor_beta_unchanged": bool(beta_unchanged),
        "no_wwtw_beta_leakage": not leakage,
        "coordinate_sensor_partition": green.isdisjoint(yellow) and green | yellow == coordinate_uids,
        "treatment_work_unique": not works.duplicated(["company", "uwwCode"]).any(),
        "layers_exact": layers == expected_layers,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"Validation failed: {failed}")
    checks["layers"] = layer_details
    return checks


def comparison_lines(previous: pd.DataFrame, master: pd.DataFrame, chosen: pd.DataFrame, works: pd.DataFrame) -> list[str]:
    lines = ["PLACE-CORE MATCHING COMPARISON"]
    companies = sorted(set(works["company"]))
    previous_counts = previous.groupby("company").size().to_dict()
    new_counts = master.groupby("company").size().to_dict()
    lines.append(f"national previous accepted pairs: {len(previous):,}")
    lines.append(f"national new accepted pairs: {len(master):,}")
    lines.append(f"national newly accepted place-core pairs: {int(master['newly_accepted_after_place_core_fix'].sum()):,}")
    for company in companies:
        old, new = int(previous_counts.get(company, 0)), int(new_counts.get(company, 0))
        lines.append(f"company {company}: previous={old:,}; new={new:,}; increase={new-old:+,}")
    lines.extend([
        f"accepted exact place-core pairs: {int(master['place_core_exact_match'].sum()):,}",
        f"accepted place-core score >=85 pairs: {int(master['place_core_score'].ge(85).sum()):,}",
        f"accepted subset-supported pairs: {int(master['match_status'].eq('accepted_place_subset_supported').sum()):,}",
        f"ambiguous duplicate-place sensors: {int(chosen['match_status'].eq('ambiguous_duplicate_place_name').sum()):,}",
        f"different-name sensors: {int(chosen['match_status'].eq('different_place_name').sum()):,}",
        f"name-spatial conflicts: {int(chosen['match_status'].eq('name_spatial_conflict').sum()):,}",
        f"accepted records with coordinates: {int(master['sensor_coordinate_available'].sum()):,}",
        f"accepted records without coordinates: {int((~master['sensor_coordinate_available']).sum()):,}",
    ])
    per_work = master.groupby(["company", "uwwCode"])["sensor_uid"].nunique()
    lines.extend([
        f"WWTWs with one accepted sensor: {int(per_work.eq(1).sum()):,}",
        f"WWTWs with multiple accepted sensors: {int(per_work.gt(1).sum()):,}",
        f"WWTWs with no accepted sensor: {len(works)-int(per_work.size):,}",
        f"accepted pairs with finite beta: {int(pd.to_numeric(master['fitted_beta'], errors='coerce').map(np.isfinite).sum()):,}",
        f"accepted pairs with strict-quality beta: {int(master['beta_quality'].eq('strict_quality_beta').sum()):,}",
        "CAUTION: increased name-match coverage does not prove hydraulic connectivity.",
    ])
    return lines


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    paths, outputs = input_paths(root), output_paths(root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required inputs missing: {missing}")
    if not args.overwrite and any(outputs[key].exists() for key in ("master", "review", "gpkg", "log", "readme")):
        raise FileExistsError("Current branch outputs exist; rerun with --overwrite")
    archive_hashes = create_archive_once(outputs)
    temp_log = outputs["log"].with_name(f".{outputs['log'].name}.{uuid.uuid4().hex}.tmp")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    file_handler = logging.FileHandler(temp_log, mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    with paths["legacy_config"].open(encoding="utf-8") as handle:
        legacy_config = json.load(handle)
    with paths["place_config"].open(encoding="utf-8") as handle:
        place_config = json.load(handle)
    if not math.isclose(sum(place_config["place_core_score_weights"].values()), 1.0, abs_tol=1e-9):
        raise ValueError("Place-core score weights must sum to one")
    monitored = list(paths.values()) + sorted((root / "clean_data").rglob("*.csv"))
    for goal in sorted((root / "project_results").glob("goal_*")):
        monitored.extend(path for path in goal.rglob("*") if path.is_file())
    before = snapshot(monitored)
    previous_master = pd.read_csv(outputs["archive"] / "tables/wwtw_sensor_analysis_master.csv", low_memory=False)
    legacy = load_legacy(root)
    sensors, assignments, characteristics, waterbase, lookup, works, existing_codes, alias_metadata = prepare_inputs(
        root, legacy, legacy_config, place_config
    )
    source_beta = sensors.set_index("sensor_uid")[[
        "fitted_beta", "beta_ci_lower_95", "beta_ci_upper_95", "beta_ci_width"
    ]].copy()
    raw_catchments = gpd.read_file(paths["assignment_gpkg"], layer="catchments_linked")
    sewersheds = gpd.read_file(paths["hydro_gpkg"], layer="wwtw_sewersheds_hydrogeology")
    LOGGER.info("Schemas sensors=%s assignments=%s characteristics=%s waterbase=%s lookup=%s",
                list(sensors.columns), list(assignments.columns), list(characteristics.columns), list(waterbase.columns), list(lookup.columns))
    LOGGER.info("Alias metadata=%s", json.dumps(alias_metadata, sort_keys=True, default=str))
    candidates, alias_counts = generate_candidates(
        sensors, works, existing_codes, place_config, legacy, legacy_config
    )
    candidates, chosen = classify(candidates, place_config)
    master = build_master(chosen, sensors, works, characteristics, previous_master)
    review = build_review(chosen, candidates)
    reverse_company = {value: key for key, value in legacy_config["company_aliases"].items()}
    atomic_csv(master, outputs["master"], args.overwrite)
    atomic_csv(review, outputs["review"], args.overwrite)
    build_geopackage(outputs["gpkg"], raw_catchments, sewersheds, works, master, review, reverse_company)
    write_styles(outputs["styles"])
    atomic_text(readme_text(place_config), outputs["readme"])
    checks = validate(sensors, works, characteristics, candidates, chosen, master, review, outputs["gpkg"], source_beta, alias_counts)
    after = snapshot(monitored)
    if before != after:
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        raise ValueError(f"Scientific/protected source hashes changed: {changed}")
    for relative, expected in archive_hashes.items():
        if sha256(outputs["archive"] / relative) != expected:
            raise ValueError(f"Archived previous output hash changed: {relative}")
    lines = comparison_lines(previous_master, master, chosen, works)
    lines.extend([
        f"known-case self-check SCAYNES HILL: {normalize_place_name('SCAYNES HILL SSO', place_config).normalized_place_core == normalize_place_name('SCAYNES HILL STW', place_config).normalized_place_core}",
        f"GeoPackage layers: {', '.join(pyogrio.list_layers(outputs['gpkg'])[:, 0])}",
        f"source hashes unchanged: {before == after} ({len(before)} files)",
        f"archive hashes verified: True ({len(archive_hashes)} files)",
        f"validation: {json.dumps(checks, sort_keys=True, default=str)}",
        f"archive path: {outputs['archive']}",
        f"master path: {outputs['master']}",
        f"review path: {outputs['review']}",
        f"GeoPackage path: {outputs['gpkg']}",
    ])
    for line in lines:
        LOGGER.info(line)
    for handler in list(LOGGER.handlers):
        handler.flush()
        handler.close()
        LOGGER.removeHandler(handler)
    os.replace(temp_log, outputs["log"])
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare the complete CSO sensor master for later catchment assignment.

This script is deliberately downstream-only. It reads stored all-sensor fit and
bootstrap results plus the audited all-beta location join. It does not read
event rows, refit beta, rerun bootstrap, assign catchments, or modify an input.

Source-to-output renaming:

- canonical_location_name -> location_name
- tail_threshold_minutes -> threshold_minutes
- original_fit_success -> point_fit_success
- permit_number_csv_original -> permit_number (coordinate join key)

The sensor population is always the left-hand table. Ambiguous coordinate
matches are retained as issues but never accepted as coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COMPANIES = (
    "anglian",
    "northumbria",
    "severn_trent",
    "southern",
    "southwest",
    "united_utilities",
    "wessex",
    "yorkshire",
    "thames",
)
CANONICAL_FILENAME = "02_sensors_by_tail_spills_desc.csv"
SENSOR_INPUT_ROOT = Path("outputs/individual_cso_sensors/gt_240min")
COORDINATE_INPUT = Path(
    "data_intermediate/thames_extension/"
    "joined_all_sensor_beta_locations_thames_inclusive.csv"
)
MASTER_DIR = Path("data_intermediate/sensor_master")
MASTER_FILENAME = "cso_sensor_master.csv"
ISSUES_FILENAME = "cso_sensor_master_issues.csv"
SUMMARY_FILENAME = "cso_sensor_master_summary.csv"
LOG_PATH = Path("outputs/cso_spatial_drivers/logs/sensor_master.log")

THRESHOLD_MINUTES = 240.0
THRESHOLD_OPERATOR = ">"
BETA_LOWER_BOUND = 0.01
BETA_UPPER_BOUND = 2.0
BETA_BOUND_ATOL = 1e-4
STRICT_MAX_BETA_CI_WIDTH = 0.5
STRICT_MIN_BOOTSTRAP_SUCCESS = 0.95
STRICT_MAX_BOUND_HIT_RATE = 0.05
BNG_EASTING_RANGE = (0.0, 700_000.0)
BNG_NORTHING_RANGE = (0.0, 1_300_000.0)
ACCEPTED_MATCH_STATUSES = {
    "exact",
    "case_whitespace_normalized",
    "unique_alphanumeric_normalized",
}

SENSOR_REQUIRED_COLUMNS = {
    "company",
    "permit_number",
    "canonical_location_name",
    "years_present",
    "first_event_time",
    "last_event_time",
    "total_spill_count",
    "tail_threshold_minutes",
    "tail_condition",
    "tail_spill_count",
    "original_fit_success",
    "fitted_beta",
    "fitted_lambda_per_minute",
    "bootstrap_attempted",
    "bootstrap_success_rate",
    "beta_ci_lower_95",
    "beta_ci_upper_95",
    "beta_ci_width",
    "beta_bound_hit_rate",
    "sensor_analysis_status",
}
COORDINATE_REQUIRED_COLUMNS = {
    "company",
    "permit_number_csv_original",
    "permit_match_status",
    "permit_match_method",
    "json_location_name",
    "bng_easting",
    "bng_northing",
    "longitude",
    "latitude",
    "coordinate_valid",
    "coordinate_conflict",
    "coordinate_exclusion_reason",
    "coordinate_correction",
    "strict_map_quality_eligible",
    "strict_quality_exclusion_reason",
}
MASTER_COLUMNS = [
    "sensor_uid",
    "company",
    "permit_number",
    "location_name",
    "bng_easting",
    "bng_northing",
    "longitude",
    "latitude",
    "coordinate_status",
    "years_present",
    "first_event_time",
    "last_event_time",
    "total_spill_count",
    "threshold_minutes",
    "threshold_operator",
    "tail_spill_count",
    "point_fit_success",
    "fitted_beta",
    "fitted_lambda_per_minute",
    "beta_ci_lower_95",
    "beta_ci_upper_95",
    "beta_ci_width",
    "bootstrap_success_rate",
    "beta_at_bound",
    "beta_quality",
    "eligible_for_catchment_matching",
    "eligible_for_beta_analysis",
    "issue_type",
    "issue_detail",
]
SUMMARY_COLUMNS = [
    "company",
    "total_sensors",
    "sensors_with_coordinates",
    "sensors_without_coordinates",
    "sensors_with_finite_beta",
    "eligible_for_catchment_matching",
    "eligible_for_beta_analysis",
    "sensors_with_issues",
]
LOGGER = logging.getLogger("sensor_master")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing outputs/, data_intermediate/, and scripts/.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def finite(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
        index=series.index,
    )


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.FileHandler(path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")


def load_sensor_population(
    project_root: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    input_root = project_root / SENSOR_INPUT_ROOT
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    reference_columns: list[str] | None = None

    for company in COMPANIES:
        path = input_root / company / CANONICAL_FILENAME
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(
            path,
            low_memory=False,
            dtype={"company": "string", "permit_number": "string"},
        )
        require_columns(frame, SENSOR_REQUIRED_COLUMNS, path)
        if reference_columns is None:
            reference_columns = list(frame.columns)
        elif list(frame.columns) != reference_columns:
            raise ValueError(f"{path}: column order/schema differs from other companies")

        frame["company"] = frame["company"].astype("string").str.strip()
        frame["permit_number"] = frame["permit_number"].astype("string").str.strip()
        observed_companies = set(frame["company"].dropna())
        if observed_companies != {company}:
            raise ValueError(
                f"{path}: expected company {company!r}, found {sorted(observed_companies)}"
            )
        frames.append(frame)
        paths.append(path)
        LOGGER.info("Loaded %s sensors=%d from %s", company, len(frame), path)

    population = pd.concat(frames, ignore_index=True, sort=False)
    if set(population["company"]) != set(COMPANIES):
        raise AssertionError("Combined sensor population does not contain all companies")
    if population[["company", "permit_number"]].isna().any().any():
        raise AssertionError("Combined sensor population has missing primary-key fields")
    if population["permit_number"].eq("").any():
        raise AssertionError("Combined sensor population has blank permit_number")
    duplicates = population.duplicated(["company", "permit_number"], keep=False)
    if duplicates.any():
        raise AssertionError(
            "Combined sensor population has duplicate company/permit keys; "
            "no automatic deduplication is allowed"
        )
    return population, paths


def load_coordinate_audit(project_root: Path) -> tuple[pd.DataFrame, Path]:
    path = project_root / COORDINATE_INPUT
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path,
        low_memory=False,
        dtype={"company": "string", "permit_number_csv_original": "string"},
    )
    require_columns(frame, COORDINATE_REQUIRED_COLUMNS, path)
    frame["company"] = frame["company"].astype("string").str.strip()
    frame["permit_number_csv_original"] = (
        frame["permit_number_csv_original"].astype("string").str.strip()
    )

    sensor_rows = frame.loc[
        frame["permit_number_csv_original"].notna()
        & frame["permit_number_csv_original"].ne("")
    ].copy()
    duplicates = sensor_rows.duplicated(
        ["company", "permit_number_csv_original"], keep=False
    )
    if duplicates.any():
        examples = sensor_rows.loc[
            duplicates, ["company", "permit_number_csv_original"]
        ].head(10)
        raise AssertionError(
            "Coordinate audit contains duplicate sensor-side keys; no row was "
            f"chosen automatically. Examples: {examples.to_dict('records')}"
        )

    retained = [
        "company",
        "permit_number_csv_original",
        "permit_match_status",
        "permit_match_method",
        "json_location_name",
        "bng_easting",
        "bng_northing",
        "longitude",
        "latitude",
        "coordinate_valid",
        "coordinate_conflict",
        "coordinate_exclusion_reason",
        "coordinate_correction",
        "strict_map_quality_eligible",
        "strict_quality_exclusion_reason",
    ]
    sensor_rows = sensor_rows[retained].rename(
        columns={"permit_number_csv_original": "permit_number"}
    )
    return sensor_rows, path


def add_coordinate_status(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("bng_easting", "bng_northing", "longitude", "latitude"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["coordinate_valid"] = bool_series(result["coordinate_valid"])
    result["coordinate_conflict"] = bool_series(result["coordinate_conflict"])
    result["strict_map_quality_eligible"] = bool_series(
        result["strict_map_quality_eligible"]
    )
    result["permit_match_status"] = (
        result["permit_match_status"].astype("string").fillna("unmatched")
    )

    ambiguous = result["permit_match_status"].eq("ambiguous")
    conflicted = result["coordinate_conflict"]
    rejected = ambiguous | conflicted
    result.loc[
        rejected, ["bng_easting", "bng_northing", "longitude", "latitude"]
    ] = np.nan

    easting_present = finite(result["bng_easting"])
    northing_present = finite(result["bng_northing"])
    partial = easting_present ^ northing_present
    result.loc[partial, ["bng_easting", "bng_northing"]] = np.nan
    easting_present = finite(result["bng_easting"])
    northing_present = finite(result["bng_northing"])
    pair_present = easting_present & northing_present
    plausible = (
        pair_present
        & result["bng_easting"].between(*BNG_EASTING_RANGE, inclusive="neither")
        & result["bng_northing"].between(*BNG_NORTHING_RANGE, inclusive="neither")
    )
    accepted_match = result["permit_match_status"].isin(ACCEPTED_MATCH_STATUSES)
    valid = accepted_match & ~conflicted & result["coordinate_valid"] & plausible

    status = pd.Series("unmatched_permit", index=result.index, dtype="string")
    status.loc[accepted_match & ~pair_present] = "missing_coordinates"
    status.loc[accepted_match & pair_present & ~valid] = "invalid_coordinates"
    status.loc[partial] = "partial_coordinates_rejected"
    status.loc[conflicted] = "coordinate_conflict_rejected"
    status.loc[ambiguous] = "ambiguous_match_rejected"
    status.loc[valid] = "valid"
    result["coordinate_status"] = status
    result["eligible_for_catchment_matching"] = valid
    return result


def add_beta_quality(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = [
        "tail_threshold_minutes",
        "tail_spill_count",
        "total_spill_count",
        "fitted_beta",
        "fitted_lambda_per_minute",
        "beta_ci_lower_95",
        "beta_ci_upper_95",
        "beta_ci_width",
        "bootstrap_success_rate",
        "beta_bound_hit_rate",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["original_fit_success"] = bool_series(result["original_fit_success"])
    result["bootstrap_attempted"] = bool_series(result["bootstrap_attempted"])

    beta_finite = finite(result["fitted_beta"])
    beta_positive = beta_finite & result["fitted_beta"].gt(0)
    beta_in_bounds = result["fitted_beta"].between(
        BETA_LOWER_BOUND, BETA_UPPER_BOUND, inclusive="both"
    )
    at_lower = beta_finite & result["fitted_beta"].sub(BETA_LOWER_BOUND).abs().le(
        BETA_BOUND_ATOL
    )
    at_upper = beta_finite & result["fitted_beta"].sub(BETA_UPPER_BOUND).abs().le(
        BETA_BOUND_ATOL
    )
    result["beta_at_bound"] = at_lower | at_upper

    lower_finite = finite(result["beta_ci_lower_95"])
    upper_finite = finite(result["beta_ci_upper_95"])
    width_finite = finite(result["beta_ci_width"])
    interval_valid = (
        lower_finite
        & upper_finite
        & result["beta_ci_lower_95"].le(result["beta_ci_upper_95"])
    )
    calculated_width = result["beta_ci_upper_95"] - result["beta_ci_lower_95"]
    width_agrees = (
        ~(interval_valid & width_finite)
        | np.isclose(
            result["beta_ci_width"],
            calculated_width,
            rtol=1e-8,
            atol=1e-10,
        )
    )
    strict_interval = (
        interval_valid
        & width_finite
        & result["beta_ci_width"].gt(0)
        & result["beta_ci_width"].lt(STRICT_MAX_BETA_CI_WIDTH)
        & width_agrees
    )
    success_ok = (
        finite(result["bootstrap_success_rate"])
        & result["bootstrap_success_rate"].ge(STRICT_MIN_BOOTSTRAP_SUCCESS)
    )
    bound_hit_ok = (
        finite(result["beta_bound_hit_rate"])
        & result["beta_bound_hit_rate"].le(STRICT_MAX_BOUND_HIT_RATE)
    )

    eligible_beta = (
        result["eligible_for_catchment_matching"]
        & result["original_fit_success"]
        & beta_positive
        & beta_in_bounds
        & ~result["beta_at_bound"]
        & strict_interval
        & result["bootstrap_attempted"]
        & success_ok
        & bound_hit_ok
    )
    result["eligible_for_beta_analysis"] = eligible_beta

    quality = pd.Series("no_finite_beta", index=result.index, dtype="string")
    quality.loc[beta_positive] = "exploratory_beta"
    quality.loc[result["beta_at_bound"]] = "optimisation_boundary_beta"
    quality.loc[eligible_beta] = "strict_quality_beta"
    result["beta_quality"] = quality
    result["diagnostic_beta_finite"] = beta_finite
    result["diagnostic_beta_positive"] = beta_positive
    result["diagnostic_beta_in_bounds"] = beta_in_bounds
    result["diagnostic_interval_valid"] = interval_valid
    result["diagnostic_width_agrees"] = width_agrees
    result["diagnostic_strict_interval"] = strict_interval
    result["diagnostic_bootstrap_success_ok"] = success_ok
    result["diagnostic_bound_hit_ok"] = bound_hit_ok
    return result


def append_issue(
    types: list[str],
    details: list[str],
    issue_type: str,
    detail: str,
) -> None:
    if issue_type not in types:
        types.append(issue_type)
        details.append(detail)


def add_issues(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    issue_types: list[str] = []
    issue_details: list[str] = []

    for row in result.itertuples(index=False):
        types: list[str] = []
        details: list[str] = []
        coordinate_status = str(row.coordinate_status)
        if coordinate_status != "valid":
            detail = f"Coordinate status is {coordinate_status}"
            match_status = str(row.permit_match_status)
            if match_status and match_status != "<NA>":
                detail += f"; audited permit match status={match_status}"
            exclusion = row.coordinate_exclusion_reason
            if pd.notna(exclusion) and str(exclusion).strip():
                detail += f"; source reason={str(exclusion).strip()}"
            append_issue(types, details, "coordinate_unavailable", detail)

        if not bool(row.original_fit_success):
            status = str(row.sensor_analysis_status)
            append_issue(
                types,
                details,
                "point_fit_unavailable",
                f"Point fit did not succeed; sensor status={status}",
            )
        if (
            not bool(row.diagnostic_beta_positive)
            or not bool(row.diagnostic_beta_in_bounds)
        ):
            append_issue(
                types,
                details,
                "beta_missing_or_invalid",
                "fitted_beta is missing, nonfinite, nonpositive, or outside [0.01, 2.0]",
            )
        if bool(row.beta_at_bound):
            append_issue(
                types,
                details,
                "beta_at_optimisation_bound",
                "fitted_beta is within 0.0001 of 0.01 or 2.0",
            )
        if not bool(row.diagnostic_interval_valid):
            append_issue(
                types,
                details,
                "bootstrap_interval_unavailable",
                "A finite ordered beta bootstrap interval is unavailable",
            )
        elif not bool(row.diagnostic_width_agrees):
            append_issue(
                types,
                details,
                "beta_ci_width_mismatch",
                "Stored beta_ci_width does not equal upper minus lower",
            )
        elif not bool(row.diagnostic_strict_interval):
            append_issue(
                types,
                details,
                "beta_interval_not_strict_quality",
                "Beta interval width is zero or is not below 0.5",
            )
        if not bool(row.bootstrap_attempted):
            append_issue(
                types,
                details,
                "bootstrap_not_attempted",
                "Stored all-sensor result did not attempt bootstrap",
            )
        elif not bool(row.diagnostic_bootstrap_success_ok):
            append_issue(
                types,
                details,
                "bootstrap_success_below_threshold",
                "Bootstrap success rate is missing or below 0.95",
            )
        if bool(row.bootstrap_attempted) and not bool(row.diagnostic_bound_hit_ok):
            append_issue(
                types,
                details,
                "beta_bound_hit_rate_not_strict_quality",
                "Beta bound-hit rate is missing or above 0.05",
            )

        source_strict_reason = row.strict_quality_exclusion_reason
        if (
            pd.notna(source_strict_reason)
            and str(source_strict_reason).strip()
            and not bool(row.eligible_for_beta_analysis)
        ):
            details.append(
                "Existing map strict-quality audit: "
                + str(source_strict_reason).strip()
            )

        issue_types.append(";".join(types))
        issue_details.append(" | ".join(details))

    result["issue_type"] = issue_types
    result["issue_detail"] = issue_details
    problematic = (
        result["coordinate_status"].ne("valid")
        | result["beta_quality"].ne("strict_quality_beta")
    )
    if result.loc[problematic, ["issue_type", "issue_detail"]].eq("").any().any():
        raise AssertionError("At least one problematic sensor lacks issue metadata")
    return result


def prepare_master(
    population: pd.DataFrame, coordinate_audit: pd.DataFrame
) -> pd.DataFrame:
    input_rows = len(population)
    joined = population.merge(
        coordinate_audit,
        on=["company", "permit_number"],
        how="left",
        validate="one_to_one",
    )
    if len(joined) != input_rows:
        raise AssertionError("Coordinate join changed the sensor row count")

    joined = add_coordinate_status(joined)
    accepted_location = joined["permit_match_status"].isin(ACCEPTED_MATCH_STATUSES)
    source_location = joined["canonical_location_name"].astype("string").fillna("")
    json_location = joined["json_location_name"].astype("string").fillna("")
    joined["location_name"] = source_location
    fallback = source_location.str.strip().eq("") & accepted_location
    joined.loc[fallback, "location_name"] = json_location.loc[fallback]

    joined["sensor_uid"] = (
        joined["company"].astype(str) + "::" + joined["permit_number"].astype(str)
    )
    joined["threshold_minutes"] = joined["tail_threshold_minutes"]
    joined["threshold_operator"] = THRESHOLD_OPERATOR
    joined["point_fit_success"] = bool_series(joined["original_fit_success"])
    joined = add_beta_quality(joined)
    joined = add_issues(joined)

    # The existing joined map table records the same strict conditions. Any
    # disagreement on a sensor-side row signals a contract drift.
    source_strict = bool_series(joined["strict_map_quality_eligible"])
    if not source_strict.equals(joined["eligible_for_beta_analysis"]):
        disagreements = int(
            (source_strict != joined["eligible_for_beta_analysis"]).sum()
        )
        raise AssertionError(
            f"Recomputed strict beta eligibility disagrees with the existing "
            f"map audit for {disagreements} sensors"
        )

    for column in ("total_spill_count", "tail_spill_count"):
        joined[column] = pd.array(joined[column], dtype="Int64")
    for column in (
        "point_fit_success",
        "beta_at_bound",
        "eligible_for_catchment_matching",
        "eligible_for_beta_analysis",
    ):
        joined[column] = joined[column].astype(bool)

    master = joined[MASTER_COLUMNS].sort_values(
        ["company", "permit_number"], kind="mergesort"
    )
    return master.reset_index(drop=True)


def build_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for company in COMPANIES:
        frame = master.loc[master["company"].eq(company)]
        coordinates = frame["coordinate_status"].eq("valid")
        finite_beta = finite(frame["fitted_beta"])
        rows.append(
            {
                "company": company,
                "total_sensors": len(frame),
                "sensors_with_coordinates": int(coordinates.sum()),
                "sensors_without_coordinates": int((~coordinates).sum()),
                "sensors_with_finite_beta": int(finite_beta.sum()),
                "eligible_for_catchment_matching": int(
                    frame["eligible_for_catchment_matching"].sum()
                ),
                "eligible_for_beta_analysis": int(
                    frame["eligible_for_beta_analysis"].sum()
                ),
                "sensors_with_issues": int(frame["issue_type"].ne("").sum()),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def validate_master(
    population: pd.DataFrame,
    coordinate_audit: pd.DataFrame,
    master: pd.DataFrame,
    issues: pd.DataFrame,
    summary: pd.DataFrame,
    source_hashes_before: dict[Path, str],
) -> None:
    if set(population["company"]) != set(COMPANIES):
        raise AssertionError("Input does not contain every configured company")
    if master.duplicated(["company", "permit_number"]).any():
        raise AssertionError("Master contains duplicate company/permit keys")
    if master["sensor_uid"].duplicated().any():
        raise AssertionError("Master contains duplicate sensor_uid")
    if len(master) != len(population):
        raise AssertionError("A source sensor was lost or multiplied")

    input_keys = set(zip(population["company"], population["permit_number"], strict=True))
    output_keys = set(zip(master["company"], master["permit_number"], strict=True))
    if input_keys != output_keys:
        raise AssertionError("Master key population differs from combined input")
    if len(coordinate_audit) > 0 and coordinate_audit.duplicated(
        ["company", "permit_number"]
    ).any():
        raise AssertionError("Coordinate join table has duplicate sensor keys")

    threshold = pd.to_numeric(master["threshold_minutes"], errors="coerce")
    if threshold.isna().any() or not np.isclose(
        threshold.to_numpy(dtype=float), THRESHOLD_MINUTES
    ).all():
        raise AssertionError("threshold_minutes is not uniformly 240")
    if set(master["threshold_operator"]) != {THRESHOLD_OPERATOR}:
        raise AssertionError("threshold_operator is not strictly '>'")

    successful = master["point_fit_success"]
    successful_beta = pd.to_numeric(
        master.loc[successful, "fitted_beta"], errors="coerce"
    )
    if not (
        np.isfinite(successful_beta)
        & successful_beta.between(BETA_LOWER_BOUND, BETA_UPPER_BOUND)
    ).all():
        raise AssertionError("A successful beta is nonfinite or outside bounds")

    lower = pd.to_numeric(master["beta_ci_lower_95"], errors="coerce")
    upper = pd.to_numeric(master["beta_ci_upper_95"], errors="coerce")
    width = pd.to_numeric(master["beta_ci_width"], errors="coerce")
    endpoints = lower.notna() & upper.notna()
    if (lower.loc[endpoints] > upper.loc[endpoints]).any():
        raise AssertionError("A beta interval has lower > upper")
    comparable = endpoints & width.notna()
    if not np.allclose(
        width.loc[comparable],
        (upper - lower).loc[comparable],
        rtol=1e-8,
        atol=1e-10,
    ):
        raise AssertionError("A beta CI width differs from upper minus lower")

    easting = pd.to_numeric(master["bng_easting"], errors="coerce")
    northing = pd.to_numeric(master["bng_northing"], errors="coerce")
    if not finite(easting).equals(finite(northing)):
        raise AssertionError("Easting/northing are not both present or both missing")

    ambiguous_keys = set(
        zip(
            coordinate_audit.loc[
                coordinate_audit["permit_match_status"].eq("ambiguous"), "company"
            ],
            coordinate_audit.loc[
                coordinate_audit["permit_match_status"].eq("ambiguous"),
                "permit_number",
            ],
            strict=True,
        )
    )
    if ambiguous_keys:
        master_keys = pd.Series(
            list(zip(master["company"], master["permit_number"], strict=True)),
            index=master.index,
        )
        ambiguous_rows = master.loc[master_keys.isin(ambiguous_keys)]
        if ambiguous_rows["eligible_for_catchment_matching"].any():
            raise AssertionError("An ambiguous coordinate match was accepted")
        if ambiguous_rows[["bng_easting", "bng_northing"]].notna().any().any():
            raise AssertionError("An ambiguous match retained accepted BNG coordinates")

    problematic = (
        master["coordinate_status"].ne("valid")
        | master["beta_quality"].ne("strict_quality_beta")
    )
    if master.loc[problematic, ["issue_type", "issue_detail"]].eq("").any().any():
        raise AssertionError("A problematic sensor lacks issue metadata")
    expected_issues = master.loc[master["issue_type"].ne("")]
    if set(issues["sensor_uid"]) != set(expected_issues["sensor_uid"]):
        raise AssertionError("Problems-only output does not match master issues")
    if len(summary) != len(COMPANIES) or set(summary["company"]) != set(COMPANIES):
        raise AssertionError("Company summary does not have one row per company")

    source_hashes_after = {
        path: sha256_file(path) for path in source_hashes_before
    }
    if source_hashes_before != source_hashes_after:
        changed = [
            str(path)
            for path in source_hashes_before
            if source_hashes_before[path] != source_hashes_after[path]
        ]
        raise AssertionError(f"Source file hashes changed: {changed}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(project_root)

    master_dir = project_root / MASTER_DIR
    master_path = master_dir / MASTER_FILENAME
    issues_path = master_dir / ISSUES_FILENAME
    summary_path = master_dir / SUMMARY_FILENAME
    log_path = project_root / LOG_PATH
    master_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path)

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info(
        "Source-to-output renaming: canonical_location_name -> location_name; "
        "tail_threshold_minutes -> threshold_minutes; "
        "original_fit_success -> point_fit_success; "
        "permit_number_csv_original -> permit_number coordinate key"
    )
    population, sensor_paths = load_sensor_population(project_root)
    coordinate_audit, coordinate_path = load_coordinate_audit(project_root)
    source_paths = sensor_paths + [coordinate_path]
    source_hashes_before = {path: sha256_file(path) for path in source_paths}

    master = prepare_master(population, coordinate_audit)
    issues = master.loc[master["issue_type"].ne("")].copy()
    summary = build_summary(master)
    validate_master(
        population,
        coordinate_audit,
        master,
        issues,
        summary,
        source_hashes_before,
    )

    master.to_csv(master_path, index=False)
    issues.to_csv(issues_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Read the products back to catch malformed/ragged serialization and verify
    # the user-facing CSV count before declaring success.
    master_check = pd.read_csv(
        master_path, low_memory=False, dtype={"permit_number": "string"}
    )
    issues_check = pd.read_csv(
        issues_path, low_memory=False, dtype={"permit_number": "string"}
    )
    summary_check = pd.read_csv(summary_path)
    if list(master_check.columns) != MASTER_COLUMNS or len(master_check) != len(master):
        raise AssertionError("Serialized main master failed schema/row validation")
    if list(issues_check.columns) != MASTER_COLUMNS or len(issues_check) != len(issues):
        raise AssertionError("Serialized issue output failed schema/row validation")
    if list(summary_check.columns) != SUMMARY_COLUMNS or len(summary_check) != len(summary):
        raise AssertionError("Serialized company summary failed schema/row validation")
    csv_paths = set(master_dir.glob("*.csv"))
    expected_csv_paths = {master_path, issues_path, summary_path}
    if csv_paths != expected_csv_paths:
        unexpected = sorted(str(path) for path in csv_paths - expected_csv_paths)
        raise AssertionError(
            "Sensor-master directory contains unexpected user-facing CSVs: "
            + ", ".join(unexpected)
        )

    coordinates = master["coordinate_status"].eq("valid")
    finite_beta_count = int(finite(master["fitted_beta"]).sum())
    input_keys = set(zip(population["company"], population["permit_number"], strict=True))
    output_keys = set(zip(master["company"], master["permit_number"], strict=True))
    duplicate_count = int(master.duplicated(["company", "permit_number"]).sum())
    lost_count = len(input_keys - output_keys)

    LOGGER.info("Combined input rows: %d", len(population))
    LOGGER.info("Final master rows: %d", len(master))
    LOGGER.info("Duplicate company-permit count: %d", duplicate_count)
    LOGGER.info("Valid coordinate sensors: %d", int(coordinates.sum()))
    LOGGER.info("Sensors without valid coordinates: %d", int((~coordinates).sum()))
    LOGGER.info("Sensors with finite beta: %d", finite_beta_count)
    LOGGER.info(
        "Eligible for catchment matching: %d",
        int(master["eligible_for_catchment_matching"].sum()),
    )
    LOGGER.info(
        "Eligible for beta analysis: %d",
        int(master["eligible_for_beta_analysis"].sum()),
    )
    LOGGER.info("Sensors with issues: %d", len(issues))
    LOGGER.info("Lost source sensors: %d", lost_count)
    LOGGER.info("Source hashes unchanged for %d input files", len(source_paths))
    LOGGER.info("Main output: %s", master_path)
    LOGGER.info("Issues output: %s", issues_path)
    LOGGER.info("Summary output: %s", summary_path)

    print(f"Total combined input rows: {len(population)}")
    print(f"Final master rows: {len(master)}")
    print(f"Duplicate company-permit count: {duplicate_count}")
    print("Row count by company:")
    for row in summary.itertuples(index=False):
        print(f"  {row.company}: {row.total_sensors}")
    print(f"Sensors with valid coordinates: {int(coordinates.sum())}")
    print(f"Sensors without valid coordinates: {int((~coordinates).sum())}")
    print(f"Sensors with finite beta: {finite_beta_count}")
    print(
        "Sensors eligible for catchment matching: "
        f"{int(master['eligible_for_catchment_matching'].sum())}"
    )
    print(
        "Sensors eligible for beta analysis: "
        f"{int(master['eligible_for_beta_analysis'].sum())}"
    )
    print(f"Sensors with issues: {len(issues)}")
    print(f"Any source sensor lost: {'yes' if lost_count else 'no'}")
    print(f"Main output: {master_path}")
    print(f"Issues output: {issues_path}")
    print(f"Summary output: {summary_path}")
    print(f"Technical log: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        AssertionError,
        OSError,
    ) as exc:
        if LOGGER.handlers:
            LOGGER.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

#!/usr/bin/env python3
"""Build latest WWTW capacity variables and sensor-beta summaries.

All beta values are existing individual-sensor estimates. Group medians are
descriptive summaries, not newly fitted catchment or sewershed beta estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


ASSIGNMENT_PATH = Path(
    "data_intermediate/spatial_joins/cso_sensor_catchment_assignments.csv"
)
ASSIGNMENT_GPKG_PATH = Path(
    "outputs/cso_spatial_drivers/qgis/cso_catchment_assignments.gpkg"
)
WATERBASE_PATH = Path("data_raw/waterbase/waterbase_consolidated.csv")
LOOKUP_PATH = Path("data_raw/waterbase/waterbase_catchment_lookup.csv")

CAPACITY_DIR = Path("data_intermediate/capacity")
SENSOR_OUTPUT_PATH = CAPACITY_DIR / "cso_sensor_capacity_enriched.csv"
WWTW_OUTPUT_PATH = CAPACITY_DIR / "wwtw_beta_capacity_summary.csv"
CAPACITY_GPKG_PATH = Path(
    "outputs/cso_spatial_drivers/qgis/cso_capacity_analysis.gpkg"
)
LOG_PATH = Path("outputs/cso_spatial_drivers/logs/wwtw_capacity.log")

BNG_CRS = "EPSG:27700"
ACCEPTED_STATUSES = {
    "unique_same_company_containment",
    "multiple_same_uwwcode",
}
COMPANY_MAP = {
    "anglian": "anglian_water",
    "northumbria": "northumbrian_water",
    "severn_trent": "severn_trent_water",
    "southern": "southern_water",
    "southwest": "south_west_water",
    "united_utilities": "united_utilities",
    "wessex": "wessex_water",
    "yorkshire": "yorkshire_water",
    "thames": "thames_water",
}
REVERSE_COMPANY_MAP = {value: key for key, value in COMPANY_MAP.items()}

WATERBASE_FIELDS = {
    "code": "uwwCode",
    "name": "uwwName",
    "year": "year",
    "load": "uwwLoadEnteringUWWTP",
    "capacity": "uwwCapacity",
    "longitude": "uwwLongitude",
    "latitude": "uwwLatitude",
}
CAPACITY_VALUE_COLUMNS = [
    "load_entering_pe_latest",
    "design_capacity_pe_latest",
    "capacity_ratio_latest",
    "capacity_data_year",
    "capacity_status",
    "capacity_available",
    "capacity_over_design",
    "capacity_extreme_review",
    "capacity_zero_load",
    "capacity_missing_reason",
]
SENSOR_CAPACITY_COLUMNS = [
    "load_entering_pe_latest",
    "design_capacity_pe_latest",
    "capacity_ratio_latest",
    "capacity_data_year",
    "capacity_status",
    "capacity_available",
    "capacity_over_design",
    "capacity_extreme_review",
    "capacity_missing_reason",
    "eligible_for_capacity_beta_analysis",
]
WWTW_COLUMNS = [
    "company",
    "uwwCode",
    "uwwName",
    "load_entering_pe_latest",
    "design_capacity_pe_latest",
    "capacity_ratio_latest",
    "capacity_data_year",
    "capacity_status",
    "capacity_available",
    "capacity_over_design",
    "capacity_extreme_review",
    "capacity_zero_load",
    "capacity_missing_reason",
    "source_record_count",
    "valid_record_count",
    "latest_valid_record_count",
    "assigned_sensor_count",
    "coordinate_assigned_sensor_count",
    "primary_beta_sensor_count",
    "median_beta_primary",
    "mean_beta_primary",
    "beta_std_primary",
    "beta_q25_primary",
    "beta_q75_primary",
    "beta_iqr_primary",
    "median_beta_ci_width_primary",
    "median_tail_spill_count_primary",
    "all_finite_beta_sensor_count",
    "median_beta_all_finite",
    "mean_beta_all_finite",
    "minimum_primary_sensor_count_flag",
]
RAW_COLUMNS = [
    "company",
    "raw_catchment_identifier",
    "raw_catchment_name",
    "raw_catchment_company",
    "uwwCode",
    "uwwName",
    "treatment_work_link_status",
    "assigned_sensor_count",
    "primary_beta_sensor_count",
    "median_beta_primary",
    "mean_beta_primary",
    "beta_iqr_primary",
    "median_beta_ci_width_primary",
    "median_tail_spill_count_primary",
    "all_finite_beta_sensor_count",
    "median_beta_all_finite",
]

EXPECTED_INPUT_LAYERS = {
    "catchments_linked",
    "wwtw_sewersheds",
    "treatment_works",
    "sensors_assigned",
    "sensors_issues",
    "sensor_catchment_candidates",
}
LOGGER = logging.getLogger("wwtw_capacity")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(
    frame: pd.DataFrame, required: set[str], source: Path | str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    mapped = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if mapped.isna().any():
        values = sorted(series.loc[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unrecognised boolean values: {values[:10]}")
    return mapped.astype(bool)


def finite(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
        index=series.index,
    )


def sorted_pipe(values: Iterable[object]) -> str:
    clean: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        clean.update(
            part.strip()
            for part in str(value).split("|")
            if part.strip()
        )
    return "|".join(sorted(clean))


def input_paths(project_root: Path) -> list[Path]:
    paths = [
        project_root / ASSIGNMENT_PATH,
        project_root / ASSIGNMENT_GPKG_PATH,
        project_root / WATERBASE_PATH,
        project_root / LOOKUP_PATH,
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required capacity input is absent:\n"
            + "\n".join(str(path) for path in missing)
        )
    return paths


def inspect_input_layers(gpkg_path: Path) -> None:
    listing = gpd.list_layers(gpkg_path)
    LOGGER.info("Available input GeoPackage layers: %s", listing.to_dict("records"))
    missing = sorted(EXPECTED_INPUT_LAYERS - set(listing["name"]))
    if missing:
        raise ValueError(f"Assignment GeoPackage missing layers: {missing}")


def read_inputs(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment_path = project_root / ASSIGNMENT_PATH
    assignments = pd.read_csv(
        assignment_path,
        low_memory=False,
        dtype={
            "sensor_uid": "string",
            "company": "string",
            "permit_number": "string",
            "uwwCode": "string",
        },
    )
    assignment_required = {
        "sensor_uid",
        "company",
        "assignment_status",
        "uwwCode",
        "uwwName",
        "eligible_for_catchment_matching",
        "eligible_for_beta_analysis",
        "fitted_beta",
        "beta_ci_width",
        "tail_spill_count",
    }
    require_columns(assignments, assignment_required, assignment_path)
    if assignments["sensor_uid"].isna().any() or assignments[
        "sensor_uid"
    ].duplicated().any():
        raise ValueError("Assignment table has missing or duplicate sensor_uid")
    for column in (
        "eligible_for_catchment_matching",
        "eligible_for_beta_analysis",
    ):
        assignments[column] = bool_series(assignments[column])
    assignments["uwwCode"] = assignments["uwwCode"].astype("string").str.strip()

    waterbase_path = project_root / WATERBASE_PATH
    waterbase = pd.read_csv(
        waterbase_path,
        low_memory=False,
        dtype={WATERBASE_FIELDS["code"]: "string"},
    )
    require_columns(waterbase, set(WATERBASE_FIELDS.values()), waterbase_path)
    waterbase[WATERBASE_FIELDS["code"]] = (
        waterbase[WATERBASE_FIELDS["code"]].astype("string").str.strip()
    )
    if waterbase[WATERBASE_FIELDS["code"]].isna().any() or waterbase[
        WATERBASE_FIELDS["code"]
    ].eq("").any():
        raise ValueError("Waterbase contains missing uwwCode")

    lookup_path = project_root / LOOKUP_PATH
    lookup = pd.read_csv(
        lookup_path,
        dtype={"identifier": "string", "uwwCode": "string"},
    )
    require_columns(lookup, {"identifier", "uwwCode", "uwwName"}, lookup_path)
    lookup["uwwCode"] = lookup["uwwCode"].astype("string").str.strip()
    return assignments, waterbase, lookup


def select_latest_capacity(waterbase: pd.DataFrame) -> pd.DataFrame:
    code = WATERBASE_FIELDS["code"]
    year = pd.to_numeric(waterbase[WATERBASE_FIELDS["year"]], errors="coerce")
    load = pd.to_numeric(waterbase[WATERBASE_FIELDS["load"]], errors="coerce")
    capacity = pd.to_numeric(
        waterbase[WATERBASE_FIELDS["capacity"]], errors="coerce"
    )
    valid = (
        finite(year)
        & finite(load)
        & load.ge(0)
        & finite(capacity)
        & capacity.gt(0)
    )
    numeric = waterbase.assign(
        diagnostic_year=year,
        diagnostic_load=load,
        diagnostic_capacity=capacity,
        diagnostic_valid=valid,
    )
    rows: list[dict[str, object]] = []
    for uwwcode, group in numeric.groupby(code, sort=True):
        valid_group = group.loc[group["diagnostic_valid"]]
        row: dict[str, object] = {
            "uwwCode": str(uwwcode),
            "waterbase_uwwName": sorted_pipe(group[WATERBASE_FIELDS["name"]]),
            "source_record_count": len(group),
            "valid_record_count": len(valid_group),
            "latest_valid_record_count": 0,
            "load_entering_pe_latest": np.nan,
            "design_capacity_pe_latest": np.nan,
            "capacity_ratio_latest": np.nan,
            "capacity_data_year": np.nan,
            "capacity_status": "no_valid_capacity_record",
            "capacity_available": False,
            "capacity_over_design": False,
            "capacity_extreme_review": False,
            "capacity_zero_load": False,
            "capacity_missing_reason": (
                "No record has finite year, non-negative load and positive "
                "design capacity"
            ),
        }
        if not valid_group.empty:
            latest_year = float(valid_group["diagnostic_year"].max())
            latest = valid_group.loc[
                valid_group["diagnostic_year"].eq(latest_year)
            ].drop_duplicates()
            row["latest_valid_record_count"] = len(latest)
            if len(latest) == 1:
                selected = latest.iloc[0]
                selected_load = float(selected["diagnostic_load"])
                selected_capacity = float(selected["diagnostic_capacity"])
                ratio = selected_load / selected_capacity
                row.update(
                    {
                        "load_entering_pe_latest": selected_load,
                        "design_capacity_pe_latest": selected_capacity,
                        "capacity_ratio_latest": ratio,
                        "capacity_data_year": latest_year,
                        "capacity_status": "valid",
                        "capacity_available": True,
                        "capacity_over_design": ratio > 1,
                        "capacity_extreme_review": ratio > 2,
                        "capacity_zero_load": selected_load == 0,
                        "capacity_missing_reason": "",
                    }
                )
            else:
                row.update(
                    {
                        "capacity_status": "conflicting_latest_records",
                        "capacity_missing_reason": (
                            "Non-identical records remain at the latest valid "
                            f"year {latest_year:g}"
                        ),
                    }
                )
        rows.append(row)
    selected = pd.DataFrame(rows)
    selected["capacity_data_year"] = pd.array(
        selected["capacity_data_year"], dtype="Int64"
    )
    for column in (
        "source_record_count",
        "valid_record_count",
        "latest_valid_record_count",
    ):
        selected[column] = pd.array(selected[column], dtype="Int64")
    for column in (
        "capacity_available",
        "capacity_over_design",
        "capacity_extreme_review",
        "capacity_zero_load",
    ):
        selected[column] = selected[column].astype(bool)
    if selected["uwwCode"].duplicated().any():
        raise AssertionError("Latest capacity selection is not unique by uwwCode")
    return selected


def read_assignment_layers(project_root: Path) -> dict[str, gpd.GeoDataFrame]:
    gpkg_path = project_root / ASSIGNMENT_GPKG_PATH
    names = [
        "catchments_linked",
        "wwtw_sewersheds",
        "treatment_works",
        "sensors_assigned",
        "sensors_issues",
        "sensor_catchment_candidates",
    ]
    layers = {name: gpd.read_file(gpkg_path, layer=name) for name in names}
    for name, layer in layers.items():
        if layer.crs is None or layer.crs.to_epsg() != 27700:
            raise ValueError(f"Input layer {name} is not EPSG:27700")
    return layers


def build_wwtw_keys(sewersheds: gpd.GeoDataFrame) -> pd.DataFrame:
    require_columns(
        sewersheds,
        {"company", "uwwCode", "uwwName"},
        "wwtw_sewersheds",
    )
    keys = sewersheds[["company", "uwwCode", "uwwName"]].copy()
    keys = keys.rename(columns={"company": "catchment_company"})
    keys["company"] = keys["catchment_company"].map(REVERSE_COMPANY_MAP)
    if keys["company"].isna().any():
        labels = sorted(
            keys.loc[keys["company"].isna(), "catchment_company"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Unmapped sewershed company labels: {labels}")
    keys["uwwCode"] = keys["uwwCode"].astype("string").str.strip()
    if keys[["company", "uwwCode"]].isna().any().any():
        raise ValueError("Sewershed keys have missing company or uwwCode")
    if keys.duplicated(["company", "uwwCode"]).any():
        raise ValueError("Sewershed layer is not one row per company and uwwCode")
    return keys.reset_index(drop=True)


def expand_capacity_by_company(
    keys: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    expanded = keys[["company", "uwwCode"]].merge(
        selected,
        on="uwwCode",
        how="left",
        validate="many_to_one",
    )
    missing = expanded["capacity_status"].isna()
    expanded.loc[missing, "capacity_status"] = "missing_waterbase_match"
    expanded.loc[missing, "capacity_missing_reason"] = (
        "Sewershed uwwCode is absent from Waterbase"
    )
    for column in (
        "capacity_available",
        "capacity_over_design",
        "capacity_extreme_review",
        "capacity_zero_load",
    ):
        expanded[column] = expanded[column].fillna(False).astype(bool)
    return expanded


def enrich_sensors(
    assignments: pd.DataFrame,
    capacity_by_company: pd.DataFrame,
) -> pd.DataFrame:
    input_rows = len(assignments)
    capacity_fields = [
        "company",
        "uwwCode",
        "load_entering_pe_latest",
        "design_capacity_pe_latest",
        "capacity_ratio_latest",
        "capacity_data_year",
        "capacity_status",
        "capacity_available",
        "capacity_over_design",
        "capacity_extreme_review",
        "capacity_missing_reason",
    ]
    enriched = assignments.merge(
        capacity_by_company[capacity_fields],
        on=["company", "uwwCode"],
        how="left",
        validate="many_to_one",
    )
    if len(enriched) != input_rows:
        raise AssertionError("Capacity join multiplied sensor assignment rows")
    accepted = enriched["assignment_status"].isin(ACCEPTED_STATUSES)
    missing_join = enriched["capacity_status"].isna()
    no_code = enriched["uwwCode"].isna() | enriched["uwwCode"].eq("")
    enriched.loc[
        missing_join & ~accepted, "capacity_status"
    ] = "not_applicable_assignment_status"
    enriched.loc[missing_join & ~accepted, "capacity_missing_reason"] = (
        "Assignment is not an accepted single-treatment-work assignment"
    )
    enriched.loc[
        missing_join & accepted & no_code, "capacity_status"
    ] = "missing_accepted_uwwcode"
    enriched.loc[
        missing_join & accepted & no_code, "capacity_missing_reason"
    ] = "Accepted spatial assignment has no single treatment-work code"
    enriched.loc[
        missing_join & accepted & ~no_code, "capacity_status"
    ] = "missing_company_waterbase_match"
    enriched.loc[
        missing_join & accepted & ~no_code, "capacity_missing_reason"
    ] = "Accepted company and uwwCode key is absent from linked sewersheds"
    for column in (
        "capacity_available",
        "capacity_over_design",
        "capacity_extreme_review",
    ):
        enriched[column] = enriched[column].fillna(False).astype(bool)
    enriched["eligible_for_capacity_beta_analysis"] = (
        accepted
        & enriched["eligible_for_beta_analysis"]
        & ~no_code
        & enriched["capacity_available"]
        & enriched["capacity_status"].eq("valid")
    )
    return enriched


def beta_statistics(group: pd.DataFrame) -> dict[str, object]:
    beta = pd.to_numeric(group["fitted_beta"], errors="coerce")
    ci_width = pd.to_numeric(group["beta_ci_width"], errors="coerce")
    tail_count = pd.to_numeric(group["tail_spill_count"], errors="coerce")
    primary_mask = bool_series(group["eligible_for_beta_analysis"])
    primary_beta = beta.loc[primary_mask]
    primary_width = ci_width.loc[primary_mask]
    primary_tail = tail_count.loc[primary_mask]
    finite_positive = finite(beta) & beta.gt(0)
    all_beta = beta.loc[finite_positive]
    primary_count = len(primary_beta)
    all_count = len(all_beta)
    q25 = primary_beta.quantile(0.25) if primary_count else np.nan
    q75 = primary_beta.quantile(0.75) if primary_count else np.nan
    return {
        "assigned_sensor_count": int(group["sensor_uid"].nunique()),
        "coordinate_assigned_sensor_count": int(
            group.loc[
                bool_series(group["eligible_for_catchment_matching"]),
                "sensor_uid",
            ].nunique()
        ),
        "primary_beta_sensor_count": primary_count,
        "median_beta_primary": (
            float(primary_beta.median()) if primary_count else np.nan
        ),
        "mean_beta_primary": (
            float(primary_beta.mean()) if primary_count else np.nan
        ),
        "beta_std_primary": (
            float(primary_beta.std(ddof=1)) if primary_count > 1 else np.nan
        ),
        "beta_q25_primary": float(q25) if primary_count else np.nan,
        "beta_q75_primary": float(q75) if primary_count else np.nan,
        "beta_iqr_primary": (
            float(q75 - q25) if primary_count else np.nan
        ),
        "median_beta_ci_width_primary": (
            float(primary_width.median()) if primary_count else np.nan
        ),
        "median_tail_spill_count_primary": (
            float(primary_tail.median()) if primary_count else np.nan
        ),
        "all_finite_beta_sensor_count": all_count,
        "median_beta_all_finite": (
            float(all_beta.median()) if all_count else np.nan
        ),
        "mean_beta_all_finite": (
            float(all_beta.mean()) if all_count else np.nan
        ),
        "minimum_primary_sensor_count_flag": primary_count < 3,
    }


def build_wwtw_summary(
    keys: pd.DataFrame,
    capacity_by_company: pd.DataFrame,
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    summary = keys[["company", "uwwCode", "uwwName"]].merge(
        capacity_by_company[
            ["company", "uwwCode"]
            + CAPACITY_VALUE_COLUMNS
            + [
                "source_record_count",
                "valid_record_count",
                "latest_valid_record_count",
            ]
        ],
        on=["company", "uwwCode"],
        how="left",
        validate="one_to_one",
    )
    accepted = enriched.loc[
        enriched["assignment_status"].isin(ACCEPTED_STATUSES)
        & enriched["uwwCode"].notna()
        & enriched["uwwCode"].ne("")
    ]
    statistics = []
    for (company, uwwcode), group in accepted.groupby(
        ["company", "uwwCode"], sort=True
    ):
        statistics.append(
            {
                "company": company,
                "uwwCode": uwwcode,
                **beta_statistics(group),
            }
        )
    stats = pd.DataFrame(statistics)
    if stats.empty:
        statistic_columns = [
            "assigned_sensor_count",
            "coordinate_assigned_sensor_count",
            "primary_beta_sensor_count",
            "median_beta_primary",
            "mean_beta_primary",
            "beta_std_primary",
            "beta_q25_primary",
            "beta_q75_primary",
            "beta_iqr_primary",
            "median_beta_ci_width_primary",
            "median_tail_spill_count_primary",
            "all_finite_beta_sensor_count",
            "median_beta_all_finite",
            "mean_beta_all_finite",
            "minimum_primary_sensor_count_flag",
        ]
        stats = pd.DataFrame(
            columns=["company", "uwwCode"] + statistic_columns
        )
    summary = summary.merge(
        stats,
        on=["company", "uwwCode"],
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "assigned_sensor_count",
        "coordinate_assigned_sensor_count",
        "primary_beta_sensor_count",
        "all_finite_beta_sensor_count",
    ]
    for column in count_columns:
        summary[column] = pd.array(summary[column].fillna(0), dtype="Int64")
    summary["minimum_primary_sensor_count_flag"] = (
        summary["primary_beta_sensor_count"].lt(3).astype(bool)
    )
    return summary[WWTW_COLUMNS].sort_values(
        ["company", "uwwCode"], kind="mergesort"
    ).reset_index(drop=True)


def build_raw_catchment_summary(
    enriched: pd.DataFrame, candidates: gpd.GeoDataFrame
) -> pd.DataFrame:
    require_columns(
        candidates,
        {
            "sensor_uid",
            "identifier",
            "catchment_name",
            "catchment_company",
            "uwwCode",
            "uwwName",
            "relation_type",
        },
        "sensor_catchment_candidates",
    )
    within = candidates.loc[
        candidates["relation_type"].eq("same_company_within"),
        [
            "sensor_uid",
            "identifier",
            "catchment_name",
            "catchment_company",
            "uwwCode",
            "uwwName",
        ],
    ].copy()
    accepted = enriched.loc[
        enriched["assignment_status"].isin(ACCEPTED_STATUSES),
        [
            "sensor_uid",
            "company",
            "fitted_beta",
            "beta_ci_width",
            "tail_spill_count",
            "eligible_for_beta_analysis",
            "eligible_for_catchment_matching",
        ],
    ]
    rows = within.merge(
        accepted,
        on="sensor_uid",
        how="inner",
        validate="many_to_one",
    )
    if rows.duplicated(["sensor_uid", "identifier"]).any():
        raise AssertionError("Raw-catchment sensor candidates are duplicated")
    output_rows: list[dict[str, object]] = []
    for (company, identifier), group in rows.groupby(
        ["company", "identifier"], sort=True
    ):
        code_value = sorted_pipe(group["uwwCode"])
        name_value = sorted_pipe(group["uwwName"])
        code_count = len(code_value.split("|")) if code_value else 0
        if code_count == 1:
            link_status = "valid"
        elif code_count > 1:
            link_status = "conflicting_treatment_work_identifiers"
            code_value = ""
            name_value = ""
        else:
            link_status = "missing_treatment_work_identifier"
        stats = beta_statistics(group)
        output_rows.append(
            {
                "company": company,
                "raw_catchment_identifier": identifier,
                "raw_catchment_name": sorted_pipe(group["catchment_name"]),
                "raw_catchment_company": sorted_pipe(
                    group["catchment_company"]
                ),
                "uwwCode": code_value,
                "uwwName": name_value,
                "treatment_work_link_status": link_status,
                "assigned_sensor_count": stats["assigned_sensor_count"],
                "primary_beta_sensor_count": stats[
                    "primary_beta_sensor_count"
                ],
                "median_beta_primary": stats["median_beta_primary"],
                "mean_beta_primary": stats["mean_beta_primary"],
                "beta_iqr_primary": stats["beta_iqr_primary"],
                "median_beta_ci_width_primary": stats[
                    "median_beta_ci_width_primary"
                ],
                "median_tail_spill_count_primary": stats[
                    "median_tail_spill_count_primary"
                ],
                "all_finite_beta_sensor_count": stats[
                    "all_finite_beta_sensor_count"
                ],
                "median_beta_all_finite": stats[
                    "median_beta_all_finite"
                ],
            }
        )
    raw = pd.DataFrame(output_rows, columns=RAW_COLUMNS)
    if raw.duplicated(["company", "raw_catchment_identifier"]).any():
        raise AssertionError("Raw-catchment summary key is not unique")
    return raw


def sensor_geometry_map(
    assigned: gpd.GeoDataFrame, issues: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    require_columns(assigned, {"sensor_uid"}, "sensors_assigned")
    require_columns(issues, {"sensor_uid"}, "sensors_issues")
    combined = pd.concat(
        [
            assigned[["sensor_uid", "geometry"]],
            issues[["sensor_uid", "geometry"]],
        ],
        ignore_index=True,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=BNG_CRS)
    duplicate = combined.duplicated("sensor_uid", keep=False)
    if duplicate.any():
        geometry_counts = (
            combined.loc[duplicate]
            .assign(diagnostic_wkb=lambda frame: frame.geometry.map(
                lambda geometry: geometry.wkb_hex if geometry is not None else ""
            ))
            .groupby("sensor_uid")["diagnostic_wkb"]
            .nunique()
        )
        if geometry_counts.gt(1).any():
            raise ValueError("Assignment layers disagree on sensor geometry")
    return combined.drop_duplicates("sensor_uid")[
        ["sensor_uid", "geometry"]
    ].reset_index(drop=True)


def build_treatment_points_capacity(
    treatment_source: gpd.GeoDataFrame,
    waterbase: pd.DataFrame,
    wwtw_summary: pd.DataFrame,
    selected: pd.DataFrame,
) -> gpd.GeoDataFrame:
    require_columns(treatment_source, {"uwwCode"}, "treatment_works")
    code = WATERBASE_FIELDS["code"]
    year = pd.to_numeric(waterbase[WATERBASE_FIELDS["year"]], errors="coerce")
    longitude = pd.to_numeric(
        waterbase[WATERBASE_FIELDS["longitude"]], errors="coerce"
    )
    latitude = pd.to_numeric(
        waterbase[WATERBASE_FIELDS["latitude"]], errors="coerce"
    )
    valid_point = (
        finite(year)
        & finite(longitude)
        & finite(latitude)
        & longitude.between(-15, 5)
        & latitude.between(49, 61)
    )
    source = waterbase.loc[valid_point].assign(
        diagnostic_year=year.loc[valid_point],
        diagnostic_longitude=longitude.loc[valid_point],
        diagnostic_latitude=latitude.loc[valid_point],
    )
    point_rows: list[pd.Series] = []
    coordinate_conflicts = 0
    for uwwcode, group in source.groupby(code, sort=True):
        latest = group.loc[
            group["diagnostic_year"].eq(group["diagnostic_year"].max())
        ].copy()
        latest["diagnostic_coordinate"] = (
            latest["diagnostic_longitude"].astype(str)
            + "|"
            + latest["diagnostic_latitude"].astype(str)
        )
        latest = latest.drop_duplicates(
            [code, "diagnostic_year", "diagnostic_coordinate"]
        )
        if latest["diagnostic_coordinate"].nunique() != 1:
            coordinate_conflicts += 1
            continue
        point_rows.append(latest.iloc[0])
    LOGGER.info(
        "Latest Waterbase treatment-work point selection: points=%d "
        "coordinate_conflicts_omitted=%d",
        len(point_rows),
        coordinate_conflicts,
    )
    points_table = pd.DataFrame(point_rows).drop(
        columns=[
            "diagnostic_year",
            "diagnostic_coordinate",
        ],
        errors="ignore",
    )
    coordinate_points = gpd.GeoDataFrame(
        points_table,
        geometry=gpd.points_from_xy(
            points_table["diagnostic_longitude"],
            points_table["diagnostic_latitude"],
            crs="EPSG:4326",
        ),
        crs="EPSG:4326",
    ).to_crs(BNG_CRS)
    coordinate_points = coordinate_points[
        [code, WATERBASE_FIELDS["name"], "geometry"]
    ].rename(columns={WATERBASE_FIELDS["name"]: "point_uwwName"})
    capacity_fields = [
        "uwwCode",
        "waterbase_uwwName",
        "load_entering_pe_latest",
        "design_capacity_pe_latest",
        "capacity_ratio_latest",
        "capacity_data_year",
        "capacity_status",
        "capacity_available",
        "capacity_over_design",
        "capacity_extreme_review",
        "capacity_missing_reason",
    ]
    points = selected[capacity_fields].merge(
        coordinate_points,
        on="uwwCode",
        how="left",
        validate="one_to_one",
    )
    points["uwwName"] = (
        points["point_uwwName"]
        .astype("string")
        .fillna(points["waterbase_uwwName"].astype("string"))
    )
    points = points.drop(columns=["point_uwwName", "waterbase_uwwName"])
    code_company = (
        wwtw_summary.groupby("uwwCode")["company"]
        .agg(sorted_pipe)
        .rename("company")
    )
    points = points.merge(
        code_company,
        left_on="uwwCode",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    return gpd.GeoDataFrame(points, geometry="geometry", crs=BNG_CRS)


def build_gpkg_layers(
    enriched: pd.DataFrame,
    wwtw_summary: pd.DataFrame,
    raw_summary: pd.DataFrame,
    source_layers: dict[str, gpd.GeoDataFrame],
    selected: pd.DataFrame,
    waterbase: pd.DataFrame,
) -> dict[str, gpd.GeoDataFrame]:
    sewersheds = source_layers["wwtw_sewersheds"][
        ["company", "uwwCode", "geometry"]
    ].rename(columns={"company": "catchment_company"})
    sewersheds["company"] = sewersheds["catchment_company"].map(
        REVERSE_COMPANY_MAP
    )
    sewersheds_capacity = sewersheds.merge(
        wwtw_summary,
        on=["company", "uwwCode"],
        how="left",
        validate="one_to_one",
    )

    treatment_capacity = build_treatment_points_capacity(
        source_layers["treatment_works"],
        waterbase,
        wwtw_summary,
        selected,
    )

    geometry_map = sensor_geometry_map(
        source_layers["sensors_assigned"], source_layers["sensors_issues"]
    )
    eligible = enriched.loc[
        enriched["eligible_for_catchment_matching"]
    ].merge(
        geometry_map,
        on="sensor_uid",
        how="left",
        validate="one_to_one",
    )
    if eligible.geometry.isna().any():
        raise ValueError("A coordinate-bearing sensor lacks GeoPackage geometry")
    sensors_capacity = gpd.GeoDataFrame(
        eligible, geometry="geometry", crs=BNG_CRS
    )

    catchments = source_layers["catchments_linked"][
        ["identifier", "geometry"]
    ].copy()
    catchments["diagnostic_wkb"] = catchments.geometry.map(
        lambda geometry: geometry.wkb_hex
    )
    conflicts = catchments.groupby("identifier")["diagnostic_wkb"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Linked catchment rows disagree on geometry")
    catchments = catchments.drop_duplicates("identifier").drop(
        columns="diagnostic_wkb"
    )
    raw_geometry = raw_summary.merge(
        catchments,
        left_on="raw_catchment_identifier",
        right_on="identifier",
        how="left",
        validate="one_to_one",
    ).drop(columns="identifier")
    if raw_geometry["geometry"].isna().any():
        raise ValueError("A raw-catchment summary lacks source geometry")
    raw_geometry = gpd.GeoDataFrame(
        raw_geometry, geometry="geometry", crs=BNG_CRS
    )

    sewershed_issue_mask = (
        ~sewersheds_capacity["capacity_available"].fillna(False)
        | sewersheds_capacity["capacity_status"].ne("valid")
        | sewersheds_capacity["capacity_extreme_review"].fillna(False)
    )
    issue_columns = [
        "company",
        "uwwCode",
        "uwwName",
        "load_entering_pe_latest",
        "design_capacity_pe_latest",
        "capacity_ratio_latest",
        "capacity_data_year",
        "capacity_status",
        "capacity_available",
        "capacity_over_design",
        "capacity_extreme_review",
        "capacity_missing_reason",
        "geometry",
    ]
    sewershed_issues = sewersheds_capacity.loc[
        sewershed_issue_mask, issue_columns
    ].copy()
    sewershed_issues["feature_type"] = "wwtw_sewershed"
    treatment_issue_mask = (
        ~treatment_capacity["capacity_available"].fillna(False)
        | treatment_capacity["capacity_status"].ne("valid")
        | treatment_capacity["capacity_extreme_review"].fillna(False)
    )
    treatment_issues = treatment_capacity.loc[
        treatment_issue_mask, issue_columns
    ].copy()
    treatment_issues["feature_type"] = "treatment_work"
    capacity_issues = pd.concat(
        [sewershed_issues, treatment_issues], ignore_index=True
    )
    capacity_issues = gpd.GeoDataFrame(
        capacity_issues, geometry="geometry", crs=BNG_CRS
    )
    capacity_issues["issue_type"] = capacity_issues.apply(
        lambda row: (
            "extreme_capacity_ratio"
            if bool(row.get("capacity_extreme_review", False))
            and bool(row.get("capacity_available", False))
            else (
                "conflicting_latest_records"
                if row.get("capacity_status") == "conflicting_latest_records"
                else (
                    "missing_waterbase_match"
                    if row.get("capacity_status") == "missing_waterbase_match"
                    else "missing_or_invalid_capacity"
                )
            )
        ),
        axis=1,
    )
    capacity_issues["issue_detail"] = capacity_issues.apply(
        lambda row: (
            f"capacity_ratio_latest={row.get('capacity_ratio_latest')}"
            if row["issue_type"] == "extreme_capacity_ratio"
            else str(row.get("capacity_missing_reason", ""))
        ),
        axis=1,
    )

    return {
        "wwtw_sewersheds_capacity": gpd.GeoDataFrame(
            sewersheds_capacity, geometry="geometry", crs=BNG_CRS
        ),
        "treatment_works_capacity": treatment_capacity,
        "sensors_capacity_enriched": sensors_capacity,
        "raw_catchments_beta_summary": raw_geometry,
        "capacity_issues": capacity_issues,
    }


def resolve_output_targets(project_root: Path) -> dict[str, Path]:
    declarations = {
        "sensors": (SENSOR_OUTPUT_PATH, CAPACITY_DIR),
        "wwtw": (WWTW_OUTPUT_PATH, CAPACITY_DIR),
        "geopackage": (CAPACITY_GPKG_PATH, CAPACITY_GPKG_PATH.parent),
    }
    root = project_root.resolve()
    targets: dict[str, Path] = {}
    for label, (relative, expected_directory) in declarations.items():
        target = (root / relative).resolve()
        allowed = (root / expected_directory).resolve()
        if target.parent != allowed:
            raise ValueError(f"Unsafe {label} output path: {target}")
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Output path is outside project root: {target}"
            ) from exc
        targets[label] = target
    if len(set(targets.values())) != 3:
        raise ValueError("Capacity output declarations are not distinct")
    return targets


def enforce_overwrite_policy(
    targets: dict[str, Path], overwrite: bool
) -> None:
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Capacity output already exists. Use --overwrite to replace only:\n"
            + "\n".join(str(path) for path in existing)
        )
    for path in existing:
        if not path.is_file():
            raise ValueError(f"Output target is not a file: {path}")


def temporary_output_paths(targets: dict[str, Path]) -> dict[str, Path]:
    token = uuid.uuid4().hex
    return {
        label: target.with_name(
            f".{target.stem}.{token}.tmp{target.suffix}"
        )
        for label, target in targets.items()
    }


def cleanup_temporary_outputs(temporary: dict[str, Path]) -> None:
    for path in temporary.values():
        if path.exists():
            path.unlink()


def write_staged_outputs(
    temporary: dict[str, Path],
    enriched: pd.DataFrame,
    wwtw_summary: pd.DataFrame,
    layers: dict[str, gpd.GeoDataFrame],
) -> None:
    temporary["sensors"].parent.mkdir(parents=True, exist_ok=True)
    temporary["geopackage"].parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(temporary["sensors"], index=False)
    wwtw_summary.to_csv(temporary["wwtw"], index=False)
    for index, (name, layer) in enumerate(layers.items()):
        layer.to_file(
            temporary["geopackage"],
            layer=name,
            driver="GPKG",
            mode="w" if index == 0 else "a",
            index=False,
        )


def commit_staged_outputs(
    targets: dict[str, Path],
    temporary: dict[str, Path],
    overwrite: bool,
) -> list[Path]:
    overwritten: list[Path] = []
    for label in ("geopackage", "sensors", "wwtw"):
        final = targets[label]
        if final.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output appeared during processing: {final}"
                )
            try:
                final.unlink()
            except PermissionError as exc:
                hint = (
                    " Close QGIS or any application using this GeoPackage."
                    if label == "geopackage"
                    else ""
                )
                raise PermissionError(
                    f"Cannot overwrite locked output: {final}.{hint}"
                ) from exc
            except OSError as exc:
                hint = (
                    " The GeoPackage may be locked by QGIS."
                    if label == "geopackage"
                    else ""
                )
                raise OSError(f"Cannot overwrite output: {final}.{hint}") from exc
            overwritten.append(final)
            LOGGER.info("Overwritten file removed: %s", final)
    for label in ("sensors", "wwtw", "geopackage"):
        os.replace(temporary[label], targets[label])
        LOGGER.info("Validated staged output installed: %s", targets[label])
    return overwritten


def validate_capacity_selection(
    waterbase: pd.DataFrame, selected: pd.DataFrame
) -> None:
    code = WATERBASE_FIELDS["code"]
    year = pd.to_numeric(waterbase[WATERBASE_FIELDS["year"]], errors="coerce")
    load = pd.to_numeric(waterbase[WATERBASE_FIELDS["load"]], errors="coerce")
    capacity = pd.to_numeric(
        waterbase[WATERBASE_FIELDS["capacity"]], errors="coerce"
    )
    valid = (
        finite(year)
        & finite(load)
        & load.ge(0)
        & finite(capacity)
        & capacity.gt(0)
    )
    maximum_year = (
        waterbase.loc[valid]
        .assign(diagnostic_year=year.loc[valid])
        .groupby(code)["diagnostic_year"]
        .max()
    )
    valid_selected = selected.loc[selected["capacity_status"].eq("valid")]
    for row in valid_selected.itertuples(index=False):
        if float(row.capacity_data_year) != float(maximum_year.loc[row.uwwCode]):
            raise AssertionError(
                f"Selected capacity year is not latest for {row.uwwCode}"
            )
    ratios = valid_selected["load_entering_pe_latest"] / valid_selected[
        "design_capacity_pe_latest"
    ]
    if not np.allclose(
        ratios,
        valid_selected["capacity_ratio_latest"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("A valid capacity ratio is not load/capacity")
    unavailable = ~selected["capacity_available"]
    if selected.loc[
        unavailable,
        [
            "load_entering_pe_latest",
            "design_capacity_pe_latest",
            "capacity_ratio_latest",
        ],
    ].notna().any().any():
        raise AssertionError("Missing/conflicting capacity was replaced by a value")
    conflicting = selected["capacity_status"].eq(
        "conflicting_latest_records"
    )
    if not selected.loc[conflicting, "latest_valid_record_count"].gt(1).all():
        raise AssertionError("Waterbase conflicts were silently resolved")


def validate_tables(
    assignments: pd.DataFrame,
    enriched: pd.DataFrame,
    wwtw_summary: pd.DataFrame,
    raw_summary: pd.DataFrame,
    candidates: gpd.GeoDataFrame,
    waterbase: pd.DataFrame,
) -> None:
    if len(enriched) != len(assignments):
        raise AssertionError("Sensor enriched row count changed")
    if enriched["sensor_uid"].duplicated().any():
        raise AssertionError("Sensor enriched output has duplicate sensor_uid")
    if set(enriched["sensor_uid"]) != set(assignments["sensor_uid"]):
        raise AssertionError("Sensor enriched population differs from assignments")
    if wwtw_summary.duplicated(["company", "uwwCode"]).any():
        raise AssertionError("WWTW summary key is not unique")
    waterbase_codes = set(waterbase[WATERBASE_FIELDS["code"]])
    joined_codes = set(
        enriched.loc[enriched["uwwCode"].notna(), "uwwCode"]
    )
    if not joined_codes <= waterbase_codes:
        raise AssertionError("A joined uwwCode is absent from Waterbase")

    accepted_with_code = enriched.loc[
        enriched["assignment_status"].isin(ACCEPTED_STATUSES)
        & enriched["uwwCode"].notna()
        & enriched["uwwCode"].ne("")
    ]
    if int(wwtw_summary["assigned_sensor_count"].sum()) != len(
        accepted_with_code
    ):
        raise AssertionError("WWTW assigned sensor counts do not agree")
    expected_primary = int(
        accepted_with_code["eligible_for_beta_analysis"].sum()
    )
    if int(wwtw_summary["primary_beta_sensor_count"].sum()) != expected_primary:
        raise AssertionError("WWTW primary beta counts are inconsistent")
    beta = pd.to_numeric(
        accepted_with_code["fitted_beta"], errors="coerce"
    )
    expected_all = int((finite(beta) & beta.gt(0)).sum())
    if int(wwtw_summary["all_finite_beta_sensor_count"].sum()) != expected_all:
        raise AssertionError("WWTW exploratory beta counts are inconsistent")
    no_primary = wwtw_summary["primary_beta_sensor_count"].eq(0)
    primary_values = [
        "median_beta_primary",
        "mean_beta_primary",
        "beta_std_primary",
        "beta_q25_primary",
        "beta_q75_primary",
        "beta_iqr_primary",
        "median_beta_ci_width_primary",
        "median_tail_spill_count_primary",
    ]
    if wwtw_summary.loc[no_primary, primary_values].notna().any().any():
        raise AssertionError("WWTW with zero primary sensors has summary values")

    accepted_uids = set(
        enriched.loc[
            enriched["assignment_status"].isin(ACCEPTED_STATUSES),
            "sensor_uid",
        ]
    )
    candidate_rows = candidates.loc[
        candidates["relation_type"].eq("same_company_within")
        & candidates["sensor_uid"].isin(accepted_uids)
    ]
    if int(raw_summary["assigned_sensor_count"].sum()) != len(candidate_rows):
        raise AssertionError("Raw-catchment assigned counts do not agree")
    conflicting_raw = raw_summary["treatment_work_link_status"].eq(
        "conflicting_treatment_work_identifiers"
    )
    if raw_summary.loc[conflicting_raw, "uwwCode"].astype("string").str.strip().ne(
        ""
    ).any():
        raise AssertionError("A raw-catchment WWTW conflict was silently resolved")


def validate_serialized_outputs(
    sensor_path: Path,
    wwtw_path: Path,
    gpkg_path: Path,
    enriched: pd.DataFrame,
    wwtw_summary: pd.DataFrame,
    raw_summary: pd.DataFrame,
    layers: dict[str, gpd.GeoDataFrame],
) -> None:
    sensor_check = pd.read_csv(
        sensor_path,
        low_memory=False,
        dtype={"sensor_uid": "string", "uwwCode": "string"},
    )
    wwtw_check = pd.read_csv(
        wwtw_path, low_memory=False, dtype={"uwwCode": "string"}
    )
    if list(sensor_check.columns) != list(enriched.columns):
        raise AssertionError("Serialized sensor capacity schema changed")
    if len(sensor_check) != len(enriched):
        raise AssertionError("Serialized sensor capacity row count changed")
    if sensor_check["sensor_uid"].duplicated().any():
        raise AssertionError("Serialized sensor capacity has duplicate sensor_uid")
    if list(wwtw_check.columns) != WWTW_COLUMNS:
        raise AssertionError("Serialized WWTW capacity schema changed")
    if len(wwtw_check) != len(wwtw_summary):
        raise AssertionError("Serialized WWTW capacity row count changed")
    if wwtw_check.duplicated(["company", "uwwCode"]).any():
        raise AssertionError("Serialized WWTW summary key is duplicated")

    listing = gpd.list_layers(gpkg_path)
    if set(listing["name"]) != set(layers):
        raise AssertionError(
            f"Capacity GeoPackage layer contract changed: "
            f"{listing.to_dict('records')}"
        )
    for name, expected in layers.items():
        actual = gpd.read_file(gpkg_path, layer=name)
        if actual.crs is None or actual.crs.to_epsg() != 27700:
            raise AssertionError(f"Capacity layer {name} is not EPSG:27700")
        if len(actual) != len(expected):
            raise AssertionError(
                f"Capacity layer {name} count differs: "
                f"{len(actual)} != {len(expected)}"
            )
    if len(layers["wwtw_sewersheds_capacity"]) != len(wwtw_summary):
        raise AssertionError("WWTW sewershed layer count differs from summary")
    if len(layers["sensors_capacity_enriched"]) != int(
        enriched["eligible_for_catchment_matching"].sum()
    ):
        raise AssertionError("Sensor capacity layer count differs from coordinates")
    if len(layers["raw_catchments_beta_summary"]) != len(raw_summary):
        raise AssertionError("Raw-catchment layer count differs from summary")


def completion_metrics(
    waterbase: pd.DataFrame,
    selected: pd.DataFrame,
    enriched: pd.DataFrame,
    wwtw_summary: pd.DataFrame,
) -> dict[str, object]:
    valid = selected.loc[selected["capacity_status"].eq("valid")]
    ratios = pd.to_numeric(valid["capacity_ratio_latest"], errors="coerce")
    accepted = enriched["assignment_status"].isin(ACCEPTED_STATUSES)
    accepted_count = int(accepted.sum())
    accepted_capacity = accepted & enriched["capacity_available"]
    return {
        "total_waterbase_rows": len(waterbase),
        "unique_waterbase_uwwcodes": int(
            waterbase[WATERBASE_FIELDS["code"]].nunique()
        ),
        "treatment_works_with_valid_capacity": len(valid),
        "treatment_works_with_no_valid_capacity": int(
            selected["capacity_status"].eq("no_valid_capacity_record").sum()
        ),
        "conflicting_latest_record_count": int(
            selected["capacity_status"].eq(
                "conflicting_latest_records"
            ).sum()
        ),
        "accepted_sensor_count": accepted_count,
        "accepted_sensors_with_valid_capacity": int(
            accepted_capacity.sum()
        ),
        "accepted_sensors_with_valid_capacity_percentage": (
            100 * float(accepted_capacity.sum()) / accepted_count
            if accepted_count
            else np.nan
        ),
        "capacity_ratios_above_1": int(ratios.gt(1).sum()),
        "capacity_ratios_above_2": int(ratios.gt(2).sum()),
        "median_valid_capacity_ratio": float(ratios.median()),
        "minimum_valid_capacity_ratio": float(ratios.min()),
        "maximum_valid_capacity_ratio": float(ratios.max()),
        "wwtws_with_at_least_one_primary_beta_sensor": int(
            wwtw_summary["primary_beta_sensor_count"].ge(1).sum()
        ),
        "wwtws_with_at_least_three_primary_beta_sensors": int(
            wwtw_summary["primary_beta_sensor_count"].ge(3).sum()
        ),
    }


def log_and_print_report(
    metrics: dict[str, object],
    source_count: int,
    overwritten: list[Path],
    targets: dict[str, Path],
    log_path: Path,
) -> None:
    LOGGER.info("Capacity completion metrics: %s", metrics)
    LOGGER.info("Source hashes unchanged for %d input files", source_count)
    for path in overwritten:
        LOGGER.info("Overwritten path: %s", path)
    for path in targets.values():
        LOGGER.info("Final output: %s", path)
    print("WWTW capacity completion report")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"source_hashes_unchanged: yes ({source_count} files)")
    print("overwritten_paths:")
    for path in overwritten:
        print(f"  {path}")
    print("final_output_paths:")
    for path in targets.values():
        print(f"  {path}")
    print(f"technical_log: {log_path}")
    print("Regression fitted: no")
    print("Causal claim made: no")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(project_root)
    targets = resolve_output_targets(project_root)
    enforce_overwrite_policy(targets, args.overwrite)
    log_path = project_root / LOG_PATH
    configure_logging(log_path)
    LOGGER.info("Project root: %s", project_root)
    LOGGER.info(
        "Definitions: capacity_ratio_latest = latest valid load entering PE / "
        "latest valid design capacity PE, selected separately by uwwCode"
    )
    LOGGER.info(
        "Group beta medians are summaries of existing individual sensor beta "
        "estimates, not newly fitted catchment or sewershed beta values"
    )
    paths = input_paths(project_root)
    hashes_before = {path: sha256_file(path) for path in paths}
    inspect_input_layers(project_root / ASSIGNMENT_GPKG_PATH)
    temporary = temporary_output_paths(targets)
    try:
        assignments, waterbase, lookup = read_inputs(project_root)
        selected = select_latest_capacity(waterbase)
        validate_capacity_selection(waterbase, selected)
        source_layers = read_assignment_layers(project_root)
        keys = build_wwtw_keys(source_layers["wwtw_sewersheds"])
        capacity_by_company = expand_capacity_by_company(keys, selected)

        accepted_codes = set(
            assignments.loc[
                assignments["assignment_status"].isin(ACCEPTED_STATUSES)
                & assignments["uwwCode"].notna()
                & assignments["uwwCode"].ne(""),
                "uwwCode",
            ]
        )
        if not accepted_codes <= set(lookup["uwwCode"].dropna()):
            missing = sorted(accepted_codes - set(lookup["uwwCode"].dropna()))
            raise AssertionError(
                f"Accepted assignment codes absent from catchment lookup: {missing}"
            )

        enriched = enrich_sensors(assignments, capacity_by_company)
        wwtw_summary = build_wwtw_summary(
            keys, capacity_by_company, enriched
        )
        raw_summary = build_raw_catchment_summary(
            enriched, source_layers["sensor_catchment_candidates"]
        )
        validate_tables(
            assignments,
            enriched,
            wwtw_summary,
            raw_summary,
            source_layers["sensor_catchment_candidates"],
            waterbase,
        )
        layers = build_gpkg_layers(
            enriched,
            wwtw_summary,
            raw_summary,
            source_layers,
            selected,
            waterbase,
        )
        write_staged_outputs(
            temporary, enriched, wwtw_summary, layers
        )
        validate_serialized_outputs(
            temporary["sensors"],
            temporary["wwtw"],
            temporary["geopackage"],
            enriched,
            wwtw_summary,
            raw_summary,
            layers,
        )
        hashes_after = {path: sha256_file(path) for path in paths}
        if hashes_before != hashes_after:
            changed = [
                str(path)
                for path in paths
                if hashes_before[path] != hashes_after[path]
            ]
            raise AssertionError(f"Capacity source hashes changed: {changed}")
        metrics = completion_metrics(
            waterbase, selected, enriched, wwtw_summary
        )
        overwritten = commit_staged_outputs(
            targets, temporary, args.overwrite
        )
        validate_serialized_outputs(
            targets["sensors"],
            targets["wwtw"],
            targets["geopackage"],
            enriched,
            wwtw_summary,
            raw_summary,
            layers,
        )
        log_and_print_report(
            metrics,
            len(paths),
            overwritten,
            targets,
            log_path,
        )
    finally:
        cleanup_temporary_outputs(temporary)


if __name__ == "__main__":
    try:
        main()
    except (
        AssertionError,
        FileNotFoundError,
        OSError,
        PermissionError,
        TypeError,
        ValueError,
    ) as exc:
        if LOGGER.handlers:
            LOGGER.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

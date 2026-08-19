#!/usr/bin/env python3
"""Assign coordinate-eligible CSO sensors to candidate wastewater catchments.

Geometric containment is evidence of spatial coincidence only; it does not
prove exact hydraulic connectivity. Nearest polygons are diagnostics and are
never accepted automatically.
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
from shapely import make_valid


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
OUT_OF_SCOPE_CATCHMENT_COMPANIES = {
    "scottish_water",
    "welsh_water",
}
BNG_CRS = "EPSG:27700"
WGS84_CRS = "EPSG:4326"
BNG_EASTING_RANGE = (0.0, 700_000.0)
BNG_NORTHING_RANGE = (0.0, 1_300_000.0)

SENSOR_PATH = Path("data_intermediate/sensor_master/cso_sensor_master.csv")
CATCHMENT_DIR = Path("data_raw/wastewater_catchments")
CATCHMENT_STEM = "catchments_consolidated"
SHAPEFILE_EXTENSIONS = (".shp", ".shx", ".dbf", ".prj", ".cpg")
CATCHMENT_LOOKUP_PATH = Path(
    "data_raw/waterbase/waterbase_catchment_lookup.csv"
)
WATERBASE_PATH = Path("data_raw/waterbase/waterbase_consolidated.csv")
LSOA_LOOKUP_PATH = Path("data_raw/lsoa/lsoa_catchment_lookup.csv")
ALIAS_PATH = Path("config/company_aliases.csv")

ASSIGNMENT_DIR = Path("data_intermediate/spatial_joins")
ASSIGNMENT_PATH = ASSIGNMENT_DIR / "cso_sensor_catchment_assignments.csv"
ISSUES_PATH = ASSIGNMENT_DIR / "cso_sensor_catchment_issues.csv"
GPKG_PATH = Path(
    "outputs/cso_spatial_drivers/qgis/cso_catchment_assignments.gpkg"
)
LOG_PATH = Path(
    "outputs/cso_spatial_drivers/logs/sensor_catchment_assignment.log"
)

ASSIGNMENT_COLUMNS = [
    "sensor_uid",
    "company",
    "permit_number",
    "location_name",
    "bng_easting",
    "bng_northing",
    "longitude",
    "latitude",
    "fitted_beta",
    "beta_ci_lower_95",
    "beta_ci_upper_95",
    "beta_ci_width",
    "tail_spill_count",
    "beta_quality",
    "eligible_for_beta_analysis",
    "eligible_for_catchment_matching",
    "assignment_status",
    "assignment_confidence",
    "raw_catchment_identifier",
    "raw_catchment_name",
    "raw_catchment_company",
    "candidate_catchment_count",
    "candidate_catchment_identifiers",
    "uwwCode",
    "uwwName",
    "candidate_uwwcode_count",
    "candidate_uwwcodes",
    "distance_to_boundary_m",
    "nearest_same_company_identifier",
    "nearest_same_company_distance_m",
    "manual_review_required",
    "exclusion_reason",
]
ISSUE_STATUSES = {
    "multiple_different_uwwcodes",
    "boundary_match",
    "no_catchment_match",
    "company_conflict",
    "not_eligible_for_catchment_matching",
}
ACCEPTED_STATUSES = {
    "unique_same_company_containment",
    "multiple_same_uwwcode",
}
ALL_STATUSES = ISSUE_STATUSES | ACCEPTED_STATUSES

CATCHMENT_REQUIRED = {"identifier", "company", "name", "comment"}
LOOKUP_REQUIRED = {"identifier", "uwwCode", "uwwName"}
WATERBASE_REQUIRED = {
    "uwwCode",
    "uwwName",
    "uwwLongitude",
    "uwwLatitude",
}
SENSOR_REQUIRED = set(ASSIGNMENT_COLUMNS[:16]) | {
    "coordinate_status",
    "issue_type",
    "issue_detail",
}

LOGGER = logging.getLogger("sensor_catchment_assignment")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--company", choices=tuple(COMPANY_MAP))
    scope.add_argument("--all-companies", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the three declared assignment products.",
    )
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


def require_columns(
    frame: pd.DataFrame, required: set[str], source: Path
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}")


def sorted_pipe(values: Iterable[object]) -> str:
    clean = {
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    }
    return "|".join(sorted(clean))


def finite(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
        index=series.index,
    )


def source_paths(project_root: Path) -> tuple[list[Path], Path | None]:
    required = [project_root / SENSOR_PATH]
    required.extend(
        project_root / CATCHMENT_DIR / f"{CATCHMENT_STEM}{extension}"
        for extension in SHAPEFILE_EXTENSIONS
    )
    required.extend(
        [
            project_root / CATCHMENT_LOOKUP_PATH,
            project_root / WATERBASE_PATH,
        ]
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        shapefile_missing = [
            path for path in missing if path.parent == project_root / CATCHMENT_DIR
        ]
        prefix = (
            "Complete catchment shapefile bundle is absent"
            if shapefile_missing
            else "Required source file is absent"
        )
        raise FileNotFoundError(
            prefix + ":\n" + "\n".join(str(path) for path in missing)
        )
    optional = []
    alias = project_root / ALIAS_PATH
    if alias.is_file():
        optional.append(alias)
    lsoa = project_root / LSOA_LOOKUP_PATH
    lsoa_path = lsoa if lsoa.is_file() else None
    if lsoa_path is not None:
        optional.append(lsoa_path)
    return required + optional, lsoa_path


def resolve_output_targets(project_root: Path) -> dict[str, Path]:
    declarations = {
        "assignments": (ASSIGNMENT_PATH, ASSIGNMENT_DIR),
        "issues": (ISSUES_PATH, ASSIGNMENT_DIR),
        "geopackage": (GPKG_PATH, GPKG_PATH.parent),
    }
    targets: dict[str, Path] = {}
    root = project_root.resolve()
    for label, (relative_path, expected_directory) in declarations.items():
        target = (root / relative_path).resolve()
        allowed_directory = (root / expected_directory).resolve()
        if target.parent != allowed_directory:
            raise ValueError(
                f"Unsafe {label} output path outside expected directory: {target}"
            )
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Unsafe {label} output path outside project root: {target}"
            ) from exc
        targets[label] = target
    if len(set(targets.values())) != len(targets):
        raise ValueError("Output target declarations are not distinct")
    return targets


def enforce_overwrite_policy(
    targets: dict[str, Path], overwrite: bool
) -> None:
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Assignment product already exists. Rerun with --overwrite to "
            "replace only these declared targets:\n"
            + "\n".join(str(path) for path in existing)
        )
    for path in targets.values():
        if path.exists() and not path.is_file():
            raise ValueError(f"Output target exists but is not a file: {path}")


def temporary_output_paths(targets: dict[str, Path]) -> dict[str, Path]:
    token = uuid.uuid4().hex
    temporary: dict[str, Path] = {}
    for label, target in targets.items():
        suffix = target.suffix
        temporary[label] = target.with_name(
            f".{target.stem}.{token}.tmp{suffix}"
        )
    return temporary


def cleanup_temporary_outputs(temporary: dict[str, Path]) -> None:
    for path in temporary.values():
        if path.exists():
            path.unlink()


def commit_staged_outputs(
    targets: dict[str, Path],
    temporary: dict[str, Path],
    overwrite: bool,
) -> list[Path]:
    overwritten: list[Path] = []
    deletion_order = ("geopackage", "assignments", "issues")
    for label in deletion_order:
        final = targets[label]
        if final.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output appeared during processing; refusing replacement: "
                    f"{final}"
                )
            try:
                final.unlink()
            except PermissionError as exc:
                hint = (
                    " Close QGIS or any application using the GeoPackage."
                    if label == "geopackage"
                    else ""
                )
                raise PermissionError(
                    f"Cannot overwrite locked output: {final}.{hint}"
                ) from exc
            except OSError as exc:
                hint = (
                    " The GeoPackage may be locked by QGIS or another "
                    "application."
                    if label == "geopackage"
                    else ""
                )
                raise OSError(f"Cannot overwrite output: {final}.{hint}") from exc
            overwritten.append(final)
            LOGGER.info("Overwritten file removed: %s", final)
    for label in ("assignments", "issues", "geopackage"):
        staged = temporary[label]
        final = targets[label]
        os.replace(staged, final)
        LOGGER.info("Validated staged output installed: %s", final)
    return overwritten


def read_sensor_scope(project_root: Path, company: str | None) -> pd.DataFrame:
    path = project_root / SENSOR_PATH
    frame = pd.read_csv(
        path,
        low_memory=False,
        dtype={
            "sensor_uid": "string",
            "company": "string",
            "permit_number": "string",
        },
    )
    require_columns(frame, SENSOR_REQUIRED, path)
    if frame["sensor_uid"].isna().any() or frame["sensor_uid"].duplicated().any():
        raise AssertionError("Sensor master has missing or duplicate sensor_uid")
    frame["eligible_for_catchment_matching"] = bool_series(
        frame["eligible_for_catchment_matching"]
    )
    frame["eligible_for_beta_analysis"] = bool_series(
        frame["eligible_for_beta_analysis"]
    )
    unknown = sorted(set(frame["company"].dropna()) - set(COMPANY_MAP))
    if unknown:
        raise ValueError(f"Sensor master has unrecognised companies: {unknown}")
    if company is not None:
        frame = frame.loc[frame["company"].eq(company)].copy()
    if frame.empty:
        raise ValueError("Selected sensor scope is empty")
    return frame.reset_index(drop=True)


def inspect_alias_config(project_root: Path) -> None:
    path = project_root / ALIAS_PATH
    if not path.is_file():
        LOGGER.info("Company alias configuration absent; explicit mapping used")
        return
    aliases = pd.read_csv(path, dtype="string")
    expected = {"sensor_company", "catchment_company"}
    require_columns(aliases, expected, path)
    configured = dict(
        zip(
            aliases["sensor_company"].str.strip(),
            aliases["catchment_company"].str.strip(),
            strict=True,
        )
    )
    conflicts = {
        key: (configured.get(key), value)
        for key, value in COMPANY_MAP.items()
        if configured.get(key) != value
    }
    if conflicts:
        raise ValueError(
            "Company alias configuration conflicts with explicit release "
            f"mapping: {conflicts}"
        )
    LOGGER.info(
        "Company mapping validated against %s; mapping is explicit in script",
        path,
    )


def read_release(
    project_root: Path, lsoa_path: Path | None
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    catchment_path = (
        project_root / CATCHMENT_DIR / f"{CATCHMENT_STEM}.shp"
    )
    catchments = gpd.read_file(catchment_path)
    require_columns(catchments, CATCHMENT_REQUIRED, catchment_path)
    if catchments.crs is None:
        raise ValueError("Catchment CRS is undefined")
    source_feature_count = len(catchments)
    source_crs = str(catchments.crs)
    if catchments.crs.to_epsg() != 27700:
        catchments = catchments.to_crs(BNG_CRS)

    for column in ("identifier", "company", "name", "comment"):
        catchments[column] = catchments[column].astype("string").str.strip()
    missing_identifiers = int(
        catchments["identifier"].fillna("").eq("").sum()
    )
    duplicate_identifier_rows = int(
        catchments.duplicated("identifier", keep=False).sum()
    )
    if missing_identifiers or duplicate_identifier_rows:
        raise ValueError(
            "Catchment identifiers are not a valid unique key: "
            f"missing={missing_identifiers}, duplicate_rows="
            f"{duplicate_identifier_rows}"
        )

    null_before = int(catchments.geometry.isna().sum())
    empty_before = int(catchments.geometry.is_empty.sum())
    invalid_before = int(
        (catchments.geometry.notna() & ~catchments.geometry.is_valid).sum()
    )
    repair_mask = catchments.geometry.notna() & ~catchments.geometry.is_valid
    if repair_mask.any():
        catchments.loc[repair_mask, "geometry"] = catchments.loc[
            repair_mask, "geometry"
        ].map(make_valid)
    null_after = int(catchments.geometry.isna().sum())
    empty_after = int(catchments.geometry.is_empty.sum())
    invalid_after = int(
        (catchments.geometry.notna() & ~catchments.geometry.is_valid).sum()
    )
    if null_after or empty_after or invalid_after:
        raise ValueError(
            "Catchment geometry remains unusable after repair: "
            f"null={null_after}, empty={empty_after}, invalid={invalid_after}"
        )
    if not catchments.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    ).all():
        bad_types = sorted(catchments.geometry.geom_type.unique())
        raise ValueError(f"Catchment repair produced non-polygon types: {bad_types}")

    actual_labels = set(catchments["company"].dropna())
    required_labels = set(COMPANY_MAP.values())
    absent_labels = sorted(required_labels - actual_labels)
    unknown_labels = sorted(
        actual_labels - required_labels - OUT_OF_SCOPE_CATCHMENT_COMPANIES
    )
    if absent_labels or unknown_labels:
        raise ValueError(
            "Unmatched catchment company labels: "
            f"required_absent={absent_labels}, unrecognised={unknown_labels}"
        )
    LOGGER.info(
        "Out-of-scope release company labels retained for conflict diagnosis: %s",
        sorted(actual_labels - required_labels),
    )

    lookup_path = project_root / CATCHMENT_LOOKUP_PATH
    lookup = pd.read_csv(
        lookup_path,
        dtype={"identifier": "string", "uwwCode": "string", "uwwName": "string"},
    )
    require_columns(lookup, LOOKUP_REQUIRED, lookup_path)
    for column in ("identifier", "uwwCode", "uwwName"):
        lookup[column] = lookup[column].astype("string").str.strip()
    exact_lookup_duplicates = int(lookup.duplicated().sum())
    nonblank_lookup = lookup.loc[
        lookup["identifier"].notna() & lookup["identifier"].ne("")
    ].copy()
    missing_lookup_identifiers = sorted(
        set(nonblank_lookup["identifier"]) - set(catchments["identifier"])
    )

    waterbase_path = project_root / WATERBASE_PATH
    waterbase = pd.read_csv(
        waterbase_path, low_memory=False, dtype={"uwwCode": "string"}
    )
    require_columns(waterbase, WATERBASE_REQUIRED, waterbase_path)
    waterbase["uwwCode"] = waterbase["uwwCode"].astype("string").str.strip()
    missing_uwwcodes = sorted(
        set(lookup["uwwCode"].dropna()) - set(waterbase["uwwCode"].dropna())
    )

    lsoa_missing_identifiers: list[str] = []
    if lsoa_path is None:
        LOGGER.warning(
            "LSOA validation lookup is absent: %s. No LSOA compatibility "
            "claim is made for this run.",
            project_root / LSOA_LOOKUP_PATH,
        )
    else:
        lsoa = pd.read_csv(lsoa_path, dtype="string")
        identifier_candidates = [
            column
            for column in lsoa.columns
            if column.lower() in {"identifier", "catchment_identifier"}
        ]
        if len(identifier_candidates) != 1:
            raise ValueError(
                f"{lsoa_path}: cannot identify one catchment identifier field"
            )
        lsoa_identifiers = set(
            lsoa[identifier_candidates[0]].dropna().str.strip()
        )
        lsoa_missing_identifiers = sorted(
            lsoa_identifiers - set(catchments["identifier"])
        )

    diagnostics = {
        "source_feature_count": source_feature_count,
        "missing_identifiers": missing_identifiers,
        "duplicate_identifier_rows": duplicate_identifier_rows,
        "null_geometries_before": null_before,
        "empty_geometries_before": empty_before,
        "invalid_geometries_before": invalid_before,
        "null_geometries_after": null_after,
        "empty_geometries_after": empty_after,
        "invalid_geometries_after": invalid_after,
        "exact_lookup_duplicate_rows": exact_lookup_duplicates,
        "lookup_rows": len(lookup),
        "lookup_nonblank_identifier_rows": len(nonblank_lookup),
        "lookup_identifier_cardinality": int(
            nonblank_lookup["identifier"].nunique()
        ),
        "missing_lookup_identifiers": len(missing_lookup_identifiers),
        "missing_uwwcodes": len(missing_uwwcodes),
        "lsoa_missing_identifiers": len(lsoa_missing_identifiers),
    }
    LOGGER.info(
        "Catchment release features=%d source_crs=%s working_crs=%s "
        "fields=identifier/company/name/comment",
        source_feature_count,
        source_crs,
        catchments.crs,
    )
    LOGGER.info("Release diagnostics: %s", diagnostics)
    if missing_lookup_identifiers:
        LOGGER.warning(
            "Lookup identifiers absent from geometry (%d): %s",
            len(missing_lookup_identifiers),
            missing_lookup_identifiers[:20],
        )
    if missing_uwwcodes:
        LOGGER.warning(
            "Lookup uwwCodes absent from Waterbase (%d): %s",
            len(missing_uwwcodes),
            missing_uwwcodes[:20],
        )
    if lsoa_missing_identifiers:
        LOGGER.warning(
            "LSOA identifiers absent from geometry (%d): %s",
            len(lsoa_missing_identifiers),
            lsoa_missing_identifiers[:20],
        )
    return catchments, lookup, waterbase, diagnostics


def link_catchments(
    catchments: gpd.GeoDataFrame, lookup: pd.DataFrame
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Preserve lookup one-to-many rows and build deterministic key metadata."""
    lookup_fields = lookup[["identifier", "uwwCode", "uwwName"]].copy()
    linked = catchments.merge(
        lookup_fields,
        on="identifier",
        how="left",
        validate="one_to_many",
    )
    expected_rows = len(catchments) + sum(
        max(count - 1, 0)
        for count in lookup_fields["identifier"]
        .dropna()
        .value_counts()
        .to_list()
    )
    if len(linked) != expected_rows:
        raise AssertionError(
            "Catchment-to-treatment lookup join cardinality is inconsistent"
        )
    metadata_rows = []
    for identifier, group in linked.groupby("identifier", sort=True):
        first = group.iloc[0]
        metadata_rows.append(
            {
                "identifier": identifier,
                "catchment_name": first["name"],
                "catchment_company": first["company"],
                "candidate_uwwcodes": sorted_pipe(group["uwwCode"]),
                "candidate_uwwnames": sorted_pipe(group["uwwName"]),
                "candidate_uwwcode_count": len(
                    {
                        str(value).strip()
                        for value in group["uwwCode"]
                        if pd.notna(value) and str(value).strip()
                    }
                ),
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    LOGGER.info(
        "Catchment lookup join cardinality: geometry_rows=%d lookup_rows=%d "
        "linked_rows=%d identifiers_with_multiple_lookup_rows=%d",
        len(catchments),
        len(lookup),
        len(linked),
        int(
            (
                lookup.loc[lookup["identifier"].notna(), "identifier"]
                .value_counts()
                .gt(1)
            ).sum()
        ),
    )
    return gpd.GeoDataFrame(linked, geometry="geometry", crs=catchments.crs), metadata


def create_sensor_points(
    sensors: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    eligible = sensors["eligible_for_catchment_matching"]
    easting = pd.to_numeric(sensors["bng_easting"], errors="coerce")
    northing = pd.to_numeric(sensors["bng_northing"], errors="coerce")
    finite_pair = finite(easting) & finite(northing)
    broad_range = (
        easting.between(*BNG_EASTING_RANGE, inclusive="neither")
        & northing.between(*BNG_NORTHING_RANGE, inclusive="neither")
    )
    if not (finite_pair.loc[eligible] & broad_range.loc[eligible]).all():
        bad = sensors.loc[
            eligible & ~(finite_pair & broad_range), "sensor_uid"
        ].head(20)
        raise ValueError(
            "Coordinate-eligible sensors have invalid BNG coordinates: "
            f"{bad.to_list()}"
        )
    points = gpd.GeoDataFrame(
        sensors.copy(),
        geometry=gpd.points_from_xy(easting, northing, crs=BNG_CRS),
        crs=BNG_CRS,
    )
    points.loc[~eligible, "geometry"] = None
    eligible_points = points.loc[eligible].copy()
    if eligible_points.geometry.isna().any() or eligible_points.geometry.is_empty.any():
        raise ValueError("An eligible sensor produced an empty point geometry")
    if eligible_points["sensor_uid"].duplicated().any():
        raise AssertionError("Duplicate sensor_uid created in point layer")
    _ = eligible_points.sindex
    return points, eligible_points


def spatial_relation(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    predicate: str,
    relation_type: str,
) -> pd.DataFrame:
    left = points[
        ["sensor_uid", "company", "eligible_for_beta_analysis", "geometry"]
    ].copy()
    right = polygons[["identifier", "company", "name", "geometry"]].rename(
        columns={
            "company": "catchment_company",
            "name": "catchment_name",
        }
    )
    joined = gpd.sjoin(left, right, how="inner", predicate=predicate)
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "sensor_uid",
                "sensor_company",
                "identifier",
                "catchment_company",
                "catchment_name",
                "eligible_for_beta_analysis",
                "relation_type",
            ]
        )
    result = joined.rename(columns={"company": "sensor_company"})[
        [
            "sensor_uid",
            "sensor_company",
            "identifier",
            "catchment_company",
            "catchment_name",
            "eligible_for_beta_analysis",
        ]
    ].copy()
    result["relation_type"] = relation_type
    return result.sort_values(
        ["sensor_uid", "identifier"], kind="mergesort"
    ).reset_index(drop=True)


def build_candidate_relations(
    eligible_points: gpd.GeoDataFrame,
    catchments: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[str, float]], pd.DataFrame]:
    """Return within/boundary/conflict candidates plus nearest diagnostics."""
    same_parts = []
    for sensor_company, catchment_company in COMPANY_MAP.items():
        selected_points = eligible_points.loc[
            eligible_points["company"].eq(sensor_company)
        ]
        selected_polygons = catchments.loc[
            catchments["company"].eq(catchment_company)
        ]
        if not selected_points.empty:
            same_parts.append(
                spatial_relation(
                    selected_points,
                    selected_polygons,
                    predicate="within",
                    relation_type="same_company_within",
                )
            )
    within = pd.concat(same_parts, ignore_index=True) if same_parts else pd.DataFrame()
    within_ids = set(within["sensor_uid"]) if not within.empty else set()
    without_within = eligible_points.loc[
        ~eligible_points["sensor_uid"].isin(within_ids)
    ]

    boundary_parts = []
    for sensor_company, catchment_company in COMPANY_MAP.items():
        selected_points = without_within.loc[
            without_within["company"].eq(sensor_company)
        ]
        selected_polygons = catchments.loc[
            catchments["company"].eq(catchment_company)
        ]
        if not selected_points.empty:
            boundary_parts.append(
                spatial_relation(
                    selected_points,
                    selected_polygons,
                    predicate="intersects",
                    relation_type="same_company_boundary_intersects",
                )
            )
    boundary = (
        pd.concat(boundary_parts, ignore_index=True)
        if boundary_parts
        else pd.DataFrame()
    )
    boundary_ids = set(boundary["sensor_uid"]) if not boundary.empty else set()
    unresolved_points = without_within.loc[
        ~without_within["sensor_uid"].isin(boundary_ids)
    ]

    conflict_parts = []
    for sensor_company, catchment_company in COMPANY_MAP.items():
        selected_points = unresolved_points.loc[
            unresolved_points["company"].eq(sensor_company)
        ]
        other_polygons = catchments.loc[
            ~catchments["company"].eq(catchment_company)
        ]
        if not selected_points.empty:
            conflict_parts.append(
                spatial_relation(
                    selected_points,
                    other_polygons,
                    predicate="within",
                    relation_type="other_company_within",
                )
            )
    conflicts = (
        pd.concat(conflict_parts, ignore_index=True)
        if conflict_parts
        else pd.DataFrame()
    )

    nearest: dict[str, tuple[str, float]] = {}
    for row in unresolved_points[["sensor_uid", "company", "geometry"]].itertuples(
        index=False
    ):
        company_label = COMPANY_MAP[str(row.company)]
        company_polygons = catchments.loc[
            catchments["company"].eq(company_label), ["identifier", "geometry"]
        ]
        distances = company_polygons.geometry.distance(row.geometry)
        minimum = float(distances.min())
        tied = company_polygons.loc[
            np.isclose(distances.to_numpy(dtype=float), minimum),
            "identifier",
        ]
        nearest[str(row.sensor_uid)] = (sorted(tied.astype(str))[0], minimum)

    relation_frames = [
        frame for frame in (within, boundary, conflicts) if not frame.empty
    ]
    relations = (
        pd.concat(relation_frames, ignore_index=True)
        if relation_frames
        else pd.DataFrame(
            columns=[
                "sensor_uid",
                "sensor_company",
                "identifier",
                "catchment_company",
                "catchment_name",
                "eligible_for_beta_analysis",
                "relation_type",
            ]
        )
    )
    if relations.duplicated(
        ["sensor_uid", "identifier", "relation_type"]
    ).any():
        raise AssertionError("Spatial candidate relation contains duplicate rows")
    return relations, nearest, within


def add_lookup_to_relations(
    relations: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    result = relations.merge(
        metadata,
        on="identifier",
        how="left",
        validate="many_to_one",
        suffixes=("", "_metadata"),
    )
    if "catchment_name_metadata" in result:
        result = result.drop(
            columns=["catchment_name_metadata", "catchment_company_metadata"]
        )
    return result


def relation_groups(
    relations: pd.DataFrame, relation_type: str
) -> dict[str, pd.DataFrame]:
    selected = relations.loc[relations["relation_type"].eq(relation_type)]
    return {
        str(sensor_uid): group.sort_values("identifier", kind="mergesort")
        for sensor_uid, group in selected.groupby("sensor_uid", sort=False)
    }


def aggregate_relation(group: pd.DataFrame) -> dict[str, object]:
    codes: set[str] = set()
    names: set[str] = set()
    for value in group["candidate_uwwcodes"]:
        if pd.notna(value):
            codes.update(part for part in str(value).split("|") if part)
    for value in group["candidate_uwwnames"]:
        if pd.notna(value):
            names.update(part for part in str(value).split("|") if part)
    return {
        "identifiers": sorted_pipe(group["identifier"]),
        "names": sorted_pipe(group["catchment_name"]),
        "companies": sorted_pipe(group["catchment_company"]),
        "identifier_count": int(group["identifier"].nunique()),
        "codes": "|".join(sorted(codes)),
        "names_uww": "|".join(sorted(names)),
        "code_count": len(codes),
    }


def classify_assignments(
    sensors: pd.DataFrame,
    all_points: gpd.GeoDataFrame,
    relations: pd.DataFrame,
    nearest: dict[str, tuple[str, float]],
    catchments: gpd.GeoDataFrame,
) -> pd.DataFrame:
    within_groups = relation_groups(relations, "same_company_within")
    boundary_groups = relation_groups(
        relations, "same_company_boundary_intersects"
    )
    conflict_groups = relation_groups(relations, "other_company_within")
    catchment_geometry = catchments.set_index("identifier").geometry
    point_geometry = all_points.set_index("sensor_uid").geometry
    rows: list[dict[str, object]] = []

    for sensor in sensors.itertuples(index=False):
        uid = str(sensor.sensor_uid)
        base = {
            column: getattr(sensor, column)
            for column in ASSIGNMENT_COLUMNS[:16]
        }
        result: dict[str, object] = {
            **base,
            "assignment_status": "",
            "assignment_confidence": "",
            "raw_catchment_identifier": "",
            "raw_catchment_name": "",
            "raw_catchment_company": "",
            "candidate_catchment_count": 0,
            "candidate_catchment_identifiers": "",
            "uwwCode": "",
            "uwwName": "",
            "candidate_uwwcode_count": 0,
            "candidate_uwwcodes": "",
            "distance_to_boundary_m": np.nan,
            "nearest_same_company_identifier": "",
            "nearest_same_company_distance_m": np.nan,
            "manual_review_required": True,
            "exclusion_reason": "",
        }

        if not bool(sensor.eligible_for_catchment_matching):
            result.update(
                {
                    "assignment_status": "not_eligible_for_catchment_matching",
                    "assignment_confidence": "not_applicable",
                    "exclusion_reason": (
                        f"Sensor master coordinate_status="
                        f"{sensor.coordinate_status}; {sensor.issue_detail}"
                    ).strip("; "),
                }
            )
            rows.append(result)
            continue

        if uid in within_groups:
            group = within_groups[uid]
            values = aggregate_relation(group)
            result.update(
                {
                    "candidate_catchment_count": values["identifier_count"],
                    "candidate_catchment_identifiers": values["identifiers"],
                    "candidate_uwwcode_count": values["code_count"],
                    "candidate_uwwcodes": values["codes"],
                }
            )
            point = point_geometry.loc[uid]
            distances = [
                float(point.distance(catchment_geometry.loc[identifier].boundary))
                for identifier in group["identifier"]
            ]
            result["distance_to_boundary_m"] = min(distances)

            if values["identifier_count"] == 1:
                row = group.iloc[0]
                result.update(
                    {
                        "assignment_status": "unique_same_company_containment",
                        "assignment_confidence": "high_spatial_containment",
                        "raw_catchment_identifier": row["identifier"],
                        "raw_catchment_name": row["catchment_name"],
                        "raw_catchment_company": row["catchment_company"],
                    }
                )
                if values["code_count"] == 1:
                    result["uwwCode"] = values["codes"]
                    result["uwwName"] = values["names_uww"]
                    result["manual_review_required"] = False
                elif values["code_count"] == 0:
                    result["exclusion_reason"] = (
                        "Unique spatial catchment has no Waterbase lookup link"
                    )
                else:
                    result["exclusion_reason"] = (
                        "Unique spatial catchment links to multiple uwwCodes; "
                        "treatment-work assignment is unresolved"
                    )
            else:
                every_polygon_one_code = group[
                    "candidate_uwwcode_count"
                ].eq(1).all()
                if values["code_count"] == 1:
                    result.update(
                        {
                            "assignment_status": "multiple_same_uwwcode",
                            "uwwCode": values["codes"],
                            "uwwName": values["names_uww"],
                        }
                    )
                    if every_polygon_one_code:
                        result.update(
                            {
                                "assignment_confidence": (
                                    "accepted_at_treatment_work_level_only"
                                ),
                                "manual_review_required": False,
                            }
                        )
                    else:
                        result.update(
                            {
                                "assignment_confidence": (
                                    "provisional_incomplete_lookup"
                                ),
                                "manual_review_required": True,
                                "exclusion_reason": (
                                    "All known candidate links share one "
                                    "uwwCode, but at least one containing "
                                    "polygon lacks a Waterbase lookup link"
                                ),
                            }
                        )
                elif values["code_count"] == 0:
                    raise ValueError(
                        "Overlapping containing catchments have no Waterbase "
                        f"links for sensor {uid}; no valid assignment status "
                        "can be asserted without inventing a uwwCode"
                    )
                else:
                    result.update(
                        {
                            "assignment_status": (
                                "multiple_different_uwwcodes"
                            ),
                            "assignment_confidence": "unresolved_overlap",
                            "exclusion_reason": (
                                "Containing catchments do not all resolve to "
                                "one shared non-missing uwwCode"
                            ),
                        }
                    )
            rows.append(result)
            continue

        if uid in boundary_groups:
            values = aggregate_relation(boundary_groups[uid])
            result.update(
                {
                    "assignment_status": "boundary_match",
                    "assignment_confidence": "unresolved_boundary",
                    "candidate_catchment_count": values["identifier_count"],
                    "candidate_catchment_identifiers": values["identifiers"],
                    "candidate_uwwcode_count": values["code_count"],
                    "candidate_uwwcodes": values["codes"],
                    "distance_to_boundary_m": 0.0,
                    "exclusion_reason": (
                        "Point intersects but is not strictly within a "
                        "same-company catchment; not accepted automatically"
                    ),
                }
            )
            rows.append(result)
            continue

        nearest_identifier, nearest_distance = nearest[uid]
        result.update(
            {
                "nearest_same_company_identifier": nearest_identifier,
                "nearest_same_company_distance_m": nearest_distance,
                "distance_to_boundary_m": nearest_distance,
            }
        )
        if uid in conflict_groups:
            values = aggregate_relation(conflict_groups[uid])
            result.update(
                {
                    "assignment_status": "company_conflict",
                    "assignment_confidence": "unresolved_company_conflict",
                    "candidate_catchment_count": values["identifier_count"],
                    "candidate_catchment_identifiers": values["identifiers"],
                    "candidate_uwwcode_count": values["code_count"],
                    "candidate_uwwcodes": values["codes"],
                    "exclusion_reason": (
                        "No same-company containment; point is within one or "
                        "more different-company catchments"
                    ),
                }
            )
        else:
            result.update(
                {
                    "assignment_status": "no_catchment_match",
                    "assignment_confidence": "unresolved_no_containment",
                    "exclusion_reason": (
                        "No same-company containing or boundary catchment; "
                        "nearest polygon is diagnostic only"
                    ),
                }
            )
        rows.append(result)

    assignments = pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)
    assignments["manual_review_required"] = assignments[
        "manual_review_required"
    ].astype(bool)
    assignments["candidate_catchment_count"] = pd.array(
        assignments["candidate_catchment_count"], dtype="Int64"
    )
    assignments["candidate_uwwcode_count"] = pd.array(
        assignments["candidate_uwwcode_count"], dtype="Int64"
    )
    return assignments


def create_candidate_layer(
    relations: pd.DataFrame, eligible_points: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    point_fields = eligible_points[
        ["sensor_uid", "permit_number", "location_name", "geometry"]
    ]
    candidates = relations.merge(
        point_fields,
        on="sensor_uid",
        how="left",
        validate="many_to_one",
    )
    candidates["uwwCode"] = candidates["candidate_uwwcodes"]
    candidates["uwwName"] = candidates["candidate_uwwnames"]
    columns = [
        "sensor_uid",
        "sensor_company",
        "permit_number",
        "location_name",
        "relation_type",
        "identifier",
        "catchment_name",
        "catchment_company",
        "uwwCode",
        "uwwName",
        "candidate_uwwcode_count",
        "candidate_uwwcodes",
        "candidate_uwwnames",
        "eligible_for_beta_analysis",
        "geometry",
    ]
    return gpd.GeoDataFrame(
        candidates[columns], geometry="geometry", crs=BNG_CRS
    )


def build_spatial_layers(
    assignments: pd.DataFrame,
    all_points: gpd.GeoDataFrame,
    candidate_layer: gpd.GeoDataFrame,
    linked: gpd.GeoDataFrame,
    waterbase: pd.DataFrame,
    selected_companies: set[str],
) -> dict[str, gpd.GeoDataFrame]:
    accepted_uids = set(
        assignments.loc[
            assignments["assignment_status"].isin(ACCEPTED_STATUSES),
            "sensor_uid",
        ]
    )
    accepted_candidates = candidate_layer.loc[
        candidate_layer["sensor_uid"].isin(accepted_uids)
        & candidate_layer["relation_type"].eq("same_company_within")
    ]
    sensor_counts = (
        accepted_candidates.groupby("identifier")["sensor_uid"]
        .nunique()
        .rename("sensor_count")
    )
    beta_counts = (
        accepted_candidates.loc[
            accepted_candidates["eligible_for_beta_analysis"]
        ]
        .groupby("identifier")["sensor_uid"]
        .nunique()
        .rename("beta_eligible_sensor_count")
    )

    catchments_linked = linked.loc[
        linked["company"].isin(selected_companies)
    ].copy()
    catchments_linked = catchments_linked.merge(
        sensor_counts,
        left_on="identifier",
        right_index=True,
        how="left",
        validate="many_to_one",
    ).merge(
        beta_counts,
        left_on="identifier",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    for column in ("sensor_count", "beta_eligible_sensor_count"):
        catchments_linked[column] = (
            catchments_linked[column].fillna(0).astype("int64")
        )
    catchments_linked = catchments_linked[
        [
            "identifier",
            "company",
            "name",
            "comment",
            "uwwCode",
            "uwwName",
            "sensor_count",
            "beta_eligible_sensor_count",
            "geometry",
        ]
    ]

    sewershed_source = catchments_linked.loc[
        catchments_linked["uwwCode"].notna()
        & catchments_linked["uwwCode"].astype("string").str.strip().ne("")
    ]
    if sewershed_source.empty:
        raise ValueError("No linked polygons are available for WWTW sewersheds")
    sewersheds = sewershed_source.dissolve(
        by=["company", "uwwCode"],
        as_index=False,
        aggfunc={
            "uwwName": sorted_pipe,
            "sensor_count": "sum",
            "beta_eligible_sensor_count": "sum",
        },
    )
    sewersheds = sewersheds.rename(
        columns={"sensor_count": "assigned_sensor_count"}
    )

    selected_codes = set(sewersheds["uwwCode"].dropna())
    treatment = waterbase.loc[
        waterbase["uwwCode"].isin(selected_codes)
    ].copy()
    longitude = pd.to_numeric(treatment["uwwLongitude"], errors="coerce")
    latitude = pd.to_numeric(treatment["uwwLatitude"], errors="coerce")
    valid_coordinates = (
        finite(longitude)
        & finite(latitude)
        & longitude.between(-15, 5)
        & latitude.between(49, 61)
    )
    LOGGER.info(
        "Treatment-work release rows selected=%d valid_points=%d "
        "invalid_coordinate_rows=%d; historical rows are preserved",
        len(treatment),
        int(valid_coordinates.sum()),
        int((~valid_coordinates).sum()),
    )
    treatment = treatment.loc[valid_coordinates].copy()
    treatment_works = gpd.GeoDataFrame(
        treatment,
        geometry=gpd.points_from_xy(
            longitude.loc[valid_coordinates],
            latitude.loc[valid_coordinates],
            crs=WGS84_CRS,
        ),
        crs=WGS84_CRS,
    ).to_crs(BNG_CRS)

    point_geometry = all_points[["sensor_uid", "geometry"]]
    assignment_geo = assignments.merge(
        point_geometry,
        on="sensor_uid",
        how="left",
        validate="one_to_one",
    )
    assignment_geo = gpd.GeoDataFrame(
        assignment_geo, geometry="geometry", crs=BNG_CRS
    )
    sensors_assigned = assignment_geo.loc[
        assignment_geo["assignment_status"].isin(ACCEPTED_STATUSES)
    ].copy()
    issue_mask = (
        assignment_geo["assignment_status"].isin(ISSUE_STATUSES)
        | assignment_geo["manual_review_required"]
    )
    sensors_issues = assignment_geo.loc[
        issue_mask & assignment_geo.geometry.notna()
    ].copy()

    return {
        "catchments_linked": gpd.GeoDataFrame(
            catchments_linked, geometry="geometry", crs=BNG_CRS
        ),
        "wwtw_sewersheds": gpd.GeoDataFrame(
            sewersheds, geometry="geometry", crs=BNG_CRS
        ),
        "treatment_works": treatment_works,
        "sensors_assigned": sensors_assigned,
        "sensors_issues": sensors_issues,
        "sensor_catchment_candidates": candidate_layer,
    }


def validate_assignments(
    sensors: pd.DataFrame,
    assignments: pd.DataFrame,
    relations: pd.DataFrame,
    catchments: gpd.GeoDataFrame,
    waterbase: pd.DataFrame,
    expect_all_companies: bool,
) -> None:
    if len(assignments) != len(sensors):
        raise AssertionError("Assignment row count differs from sensor scope")
    if assignments["sensor_uid"].isna().any():
        raise AssertionError("Assignment output has missing sensor_uid")
    if assignments["sensor_uid"].duplicated().any():
        raise AssertionError("Spatial processing multiplied final sensor rows")
    if set(assignments["sensor_uid"]) != set(sensors["sensor_uid"]):
        raise AssertionError("Assignment sensor population differs from master scope")
    if expect_all_companies and set(assignments["company"]) != set(COMPANY_MAP):
        raise AssertionError("National assignment does not contain all configured companies")
    if assignments["assignment_status"].isna().any():
        raise AssertionError("A sensor has no assignment_status")
    unexpected = set(assignments["assignment_status"]) - ALL_STATUSES
    if unexpected:
        raise AssertionError(f"Unexpected assignment statuses: {unexpected}")

    eligible_uids = set(
        sensors.loc[
            sensors["eligible_for_catchment_matching"], "sensor_uid"
        ]
    )
    processed_eligible = set(
        assignments.loc[
            assignments["assignment_status"].ne(
                "not_eligible_for_catchment_matching"
            ),
            "sensor_uid",
        ]
    )
    if eligible_uids != processed_eligible:
        raise AssertionError("Coordinate-eligible sensor processing is incomplete")
    noneligible = ~sensors["eligible_for_catchment_matching"]
    expected_noneligible = set(sensors.loc[noneligible, "sensor_uid"])
    classified_noneligible = set(
        assignments.loc[
            assignments["assignment_status"].eq(
                "not_eligible_for_catchment_matching"
            ),
            "sensor_uid",
        ]
    )
    if expected_noneligible != classified_noneligible:
        raise AssertionError("Non-eligible sensors are not explicitly classified")

    catchment_ids = set(catchments["identifier"])
    accepted_raw = set(
        assignments.loc[
            assignments["raw_catchment_identifier"].astype("string").str.strip().ne(""),
            "raw_catchment_identifier",
        ]
    )
    if not accepted_raw <= catchment_ids:
        raise AssertionError("An accepted raw catchment identifier is invalid")
    waterbase_codes = set(waterbase["uwwCode"].dropna())
    accepted_codes = set(
        assignments.loc[
            assignments["uwwCode"].astype("string").str.strip().ne(""),
            "uwwCode",
        ]
    )
    if not accepted_codes <= waterbase_codes:
        raise AssertionError("An accepted uwwCode is absent from Waterbase")

    unique = assignments["assignment_status"].eq(
        "unique_same_company_containment"
    )
    if not assignments.loc[unique].apply(
        lambda row: row["raw_catchment_company"]
        == COMPANY_MAP[row["company"]],
        axis=1,
    ).all():
        raise AssertionError("A unique assignment is not same-company")
    multiple_same = assignments["assignment_status"].eq(
        "multiple_same_uwwcode"
    )
    if not assignments.loc[multiple_same, "candidate_uwwcode_count"].eq(1).all():
        raise AssertionError(
            "A multiple-same-uwwCode assignment does not have exactly one code"
        )
    multiple_different = assignments["assignment_status"].eq(
        "multiple_different_uwwcodes"
    )
    invalid_multiple_different = multiple_different & ~assignments[
        "candidate_uwwcode_count"
    ].gt(1)
    if invalid_multiple_different.any():
        examples = assignments.loc[
            invalid_multiple_different,
            [
                "sensor_uid",
                "company",
                "candidate_catchment_identifiers",
                "candidate_uwwcode_count",
                "candidate_uwwcodes",
            ],
        ].head(20)
        raise AssertionError(
            "A multiple-different-uwwCodes assignment does not have >1 codes. "
            f"Count={int(invalid_multiple_different.sum())}; examples="
            f"{examples.to_dict('records')}"
        )
    boundary = assignments["assignment_status"].eq("boundary_match")
    boundary_values = (
        assignments.loc[
            boundary, ["raw_catchment_identifier", "uwwCode"]
        ]
        .fillna("")
        .astype(str)
        .apply(lambda column: column.str.strip())
        .ne("")
        .to_numpy()
    )
    if boundary_values.any():
        raise AssertionError("A boundary match was silently accepted")
    nearest_only = assignments["assignment_status"].isin(
        ["no_catchment_match", "company_conflict"]
    )
    nearest_values = (
        assignments.loc[
            nearest_only, ["raw_catchment_identifier", "uwwCode"]
        ]
        .fillna("")
        .astype(str)
        .apply(lambda column: column.str.strip())
        .ne("")
        .to_numpy()
    )
    if nearest_values.any():
        raise AssertionError("A nearest diagnostic was silently accepted")

    if not assignments.loc[
        assignments["eligible_for_catchment_matching"],
        ["distance_to_boundary_m", "nearest_same_company_distance_m"],
    ].apply(
        lambda column: pd.to_numeric(column, errors="coerce")
    ).stack().dropna().ge(0).all():
        raise AssertionError("A projected distance is negative")
    if relations.duplicated(
        ["sensor_uid", "identifier", "relation_type"]
    ).any():
        raise AssertionError("Candidate representation is not one row per relation")
    company_total = int(assignments.groupby("company", dropna=False).size().sum())
    if company_total != len(assignments):
        raise AssertionError("Company-level assignment counts do not add nationally")
    accepted_review = assignments[
        assignments["assignment_status"].isin(ACCEPTED_STATUSES)
        & assignments["manual_review_required"]
    ]
    if not accepted_review["exclusion_reason"].astype("string").str.strip().ne(
        ""
    ).all():
        raise AssertionError(
            "An accepted assignment marked for review lacks a validation reason"
        )


def write_staged_outputs(
    temporary: dict[str, Path],
    assignments: pd.DataFrame,
    issues: pd.DataFrame,
    layers: dict[str, gpd.GeoDataFrame],
) -> tuple[Path, Path, Path]:
    assignment_path = temporary["assignments"]
    issues_path = temporary["issues"]
    gpkg_path = temporary["geopackage"]
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    assignments.to_csv(assignment_path, index=False)
    issues.to_csv(issues_path, index=False)
    for index, (layer_name, layer) in enumerate(layers.items()):
        layer.to_file(
            gpkg_path,
            layer=layer_name,
            driver="GPKG",
            mode="w" if index == 0 else "a",
            index=False,
        )
    return assignment_path, issues_path, gpkg_path


def validate_serialized_outputs(
    assignment_path: Path,
    issues_path: Path,
    gpkg_path: Path,
    assignments: pd.DataFrame,
    issues: pd.DataFrame,
    layers: dict[str, gpd.GeoDataFrame],
) -> None:
    assignment_check = pd.read_csv(
        assignment_path,
        low_memory=False,
        dtype={"sensor_uid": "string", "permit_number": "string"},
    )
    issues_check = pd.read_csv(
        issues_path,
        low_memory=False,
        dtype={"sensor_uid": "string", "permit_number": "string"},
    )
    if list(assignment_check.columns) != ASSIGNMENT_COLUMNS:
        raise AssertionError("Serialized assignment CSV column contract is wrong")
    if list(issues_check.columns) != ASSIGNMENT_COLUMNS:
        raise AssertionError("Serialized issues CSV column contract is wrong")
    if len(assignment_check) != len(assignments):
        raise AssertionError("Serialized assignment CSV row count changed")
    if len(issues_check) != len(issues):
        raise AssertionError("Serialized issues CSV row count changed")
    if assignment_check["sensor_uid"].duplicated().any():
        raise AssertionError("Serialized assignment CSV has duplicate sensors")

    layer_listing = gpd.list_layers(gpkg_path)
    actual_layer_names = set(layer_listing["name"])
    if actual_layer_names != set(layers):
        raise AssertionError(
            "GeoPackage layer contract differs: "
            f"actual={sorted(actual_layer_names)}"
        )
    for layer_name, expected in layers.items():
        actual = gpd.read_file(gpkg_path, layer=layer_name)
        if len(actual) != len(expected):
            raise AssertionError(
                f"GeoPackage {layer_name} count differs: "
                f"{len(actual)} != {len(expected)}"
            )
        if actual.crs is None or actual.crs.to_epsg() != 27700:
            raise AssertionError(
                f"GeoPackage {layer_name} is not stored in EPSG:27700"
            )
    if len(layers["sensors_assigned"]) != int(
        assignments["assignment_status"].isin(ACCEPTED_STATUSES).sum()
    ):
        raise AssertionError("Assigned sensor layer count differs from CSV")
    coordinate_issue_count = int(
        issues["eligible_for_catchment_matching"].astype(bool).sum()
    )
    if len(layers["sensors_issues"]) != coordinate_issue_count:
        raise AssertionError(
            "Issue sensor layer count differs from coordinate-bearing CSV issues"
        )


def build_company_report(
    sensors: pd.DataFrame,
    assignments: pd.DataFrame,
    linked: gpd.GeoDataFrame,
    waterbase: pd.DataFrame,
    diagnostics: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: list[str] = ["national"] + [
        company for company in COMPANY_MAP if company in set(sensors["company"])
    ]
    waterbase_codes = set(waterbase["uwwCode"].dropna())
    for scope in scopes:
        selected = (
            assignments
            if scope == "national"
            else assignments.loc[assignments["company"].eq(scope)]
        )
        sensor_selected = (
            sensors
            if scope == "national"
            else sensors.loc[sensors["company"].eq(scope)]
        )
        labels = (
            {COMPANY_MAP[company] for company in set(sensors["company"])}
            if scope == "national"
            else {COMPANY_MAP[scope]}
        )
        selected_linked = linked.loc[linked["company"].isin(labels)]
        link_presence = selected_linked.groupby("identifier")["uwwCode"].apply(
            lambda values: any(
                pd.notna(value) and str(value).strip() for value in values
            )
        )
        missing_links = int((~link_presence).sum())
        lookup_codes = {
            str(value).strip()
            for value in selected_linked["uwwCode"]
            if pd.notna(value) and str(value).strip()
        }
        counts = selected["assignment_status"].value_counts()
        accepted = selected["assignment_status"].isin(ACCEPTED_STATUSES)
        nearest_distances = pd.to_numeric(
            selected["nearest_same_company_distance_m"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "scope": scope,
                "total_master_sensors": len(selected),
                "coordinate_eligible": int(
                    sensor_selected["eligible_for_catchment_matching"].sum()
                ),
                "not_eligible": int(
                    counts.get("not_eligible_for_catchment_matching", 0)
                ),
                "unique_containment": int(
                    counts.get("unique_same_company_containment", 0)
                ),
                "multiple_same_uwwcode": int(
                    counts.get("multiple_same_uwwcode", 0)
                ),
                "multiple_different_uwwcodes": int(
                    counts.get("multiple_different_uwwcodes", 0)
                ),
                "boundary": int(counts.get("boundary_match", 0)),
                "no_match": int(counts.get("no_catchment_match", 0)),
                "company_conflict": int(counts.get("company_conflict", 0)),
                "accepted_assignments": int(accepted.sum()),
                "beta_eligible_accepted": int(
                    (
                        accepted
                        & selected["eligible_for_beta_analysis"].astype(bool)
                    ).sum()
                ),
                "median_nearest_distance_m": (
                    float(nearest_distances.median())
                    if not nearest_distances.empty
                    else np.nan
                ),
                "maximum_nearest_distance_m": (
                    float(nearest_distances.max())
                    if not nearest_distances.empty
                    else np.nan
                ),
                "missing_catchment_waterbase_links": missing_links,
                "missing_uwwcodes_in_waterbase": len(
                    lookup_codes - waterbase_codes
                ),
                "invalid_geometries_before": diagnostics[
                    "invalid_geometries_before"
                ],
                "invalid_geometries_after": diagnostics[
                    "invalid_geometries_after"
                ],
            }
        )
    report = pd.DataFrame(rows)
    company_rows = report.loc[report["scope"].ne("national")]
    for column in (
        "total_master_sensors",
        "coordinate_eligible",
        "not_eligible",
        "unique_containment",
        "multiple_same_uwwcode",
        "multiple_different_uwwcodes",
        "boundary",
        "no_match",
        "company_conflict",
        "accepted_assignments",
        "beta_eligible_accepted",
    ):
        if int(company_rows[column].sum()) != int(
            report.loc[report["scope"].eq("national"), column].iloc[0]
        ):
            raise AssertionError(
                f"Company-level {column} does not add to national total"
            )
    return report


def log_and_print_report(
    report: pd.DataFrame,
    source_count: int,
    overwritten: list[Path],
    assignment_path: Path,
    issues_path: Path,
    gpkg_path: Path,
    log_path: Path,
) -> None:
    rendered = report.to_string(index=False)
    LOGGER.info("National and company completion report:\n%s", rendered)
    LOGGER.info("Source hashes unchanged for %d source files", source_count)
    for path in overwritten:
        LOGGER.info("Overwritten path: %s", path)
    LOGGER.info("Assignment CSV: %s", assignment_path)
    LOGGER.info("Issues CSV: %s", issues_path)
    LOGGER.info("QGIS GeoPackage: %s", gpkg_path)

    print("National catchment assignment completion report")
    print(rendered)
    print(f"source_hashes_unchanged: yes ({source_count} files)")
    print("overwritten_paths:")
    for path in overwritten:
        print(f"  {path}")
    print("final_output_paths:")
    print(f"  {assignment_path}")
    print(f"  {issues_path}")
    print(f"  {gpkg_path}")
    print(f"technical_log: {log_path}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(project_root)
    targets = resolve_output_targets(project_root)
    enforce_overwrite_policy(targets, args.overwrite)
    log_path = project_root / LOG_PATH
    configure_logging(log_path)
    scope_name = "all_companies" if args.all_companies else args.company
    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Requested scope: %s", scope_name)
    LOGGER.info(
        "Containment does not prove exact hydraulic connectivity; nearest "
        "matches are diagnostic only"
    )
    temporary = temporary_output_paths(targets)
    try:
        paths, lsoa_path = source_paths(project_root)
        hashes_before = {path: sha256_file(path) for path in paths}
        inspect_alias_config(project_root)
        sensors = read_sensor_scope(
            project_root, None if args.all_companies else args.company
        )
        catchments, lookup, waterbase, diagnostics = read_release(
            project_root, lsoa_path
        )
        linked, metadata = link_catchments(catchments, lookup)
        all_points, eligible_points = create_sensor_points(sensors)
        relations, nearest, _ = build_candidate_relations(
            eligible_points, catchments
        )
        relations = add_lookup_to_relations(relations, metadata)
        assignments = classify_assignments(
            sensors, all_points, relations, nearest, catchments
        )
        validate_assignments(
            sensors,
            assignments,
            relations,
            catchments,
            waterbase,
            expect_all_companies=args.all_companies,
        )
        issue_mask = (
            assignments["assignment_status"].isin(ISSUE_STATUSES)
            | assignments["manual_review_required"]
        )
        issues = assignments.loc[issue_mask].copy()
        candidate_layer = create_candidate_layer(relations, eligible_points)
        selected_companies = {
            COMPANY_MAP[company] for company in set(sensors["company"])
        }
        layers = build_spatial_layers(
            assignments,
            all_points,
            candidate_layer,
            linked,
            waterbase,
            selected_companies,
        )
        staged_assignment, staged_issues, staged_gpkg = write_staged_outputs(
            temporary, assignments, issues, layers
        )
        validate_serialized_outputs(
            staged_assignment,
            staged_issues,
            staged_gpkg,
            assignments,
            issues,
            layers,
        )
        hashes_after = {path: sha256_file(path) for path in paths}
        if hashes_before != hashes_after:
            changed = [
                str(path)
                for path in paths
                if hashes_before[path] != hashes_after[path]
            ]
            raise AssertionError(f"Source hashes changed: {changed}")
        report = build_company_report(
            sensors, assignments, linked, waterbase, diagnostics
        )
        overwritten = commit_staged_outputs(
            targets, temporary, args.overwrite
        )
        validate_serialized_outputs(
            targets["assignments"],
            targets["issues"],
            targets["geopackage"],
            assignments,
            issues,
            layers,
        )
        log_and_print_report(
            report,
            len(paths),
            overwritten,
            targets["assignments"],
            targets["issues"],
            targets["geopackage"],
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
        TypeError,
        ValueError,
    ) as exc:
        if LOGGER.handlers:
            LOGGER.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

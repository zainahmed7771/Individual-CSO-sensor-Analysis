#!/usr/bin/env python3
"""Extract polygon-weighted hydrogeology characteristics for WWTW sewersheds.

This stage intersects validated WWTW and raw-catchment polygons with the BGS
Hydrogeology 1:625,000 source. It does not sample centroids or fit models.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


BNG = "EPSG:27700"
CAPACITY_CSV = Path("data_intermediate/capacity/wwtw_beta_capacity_summary.csv")
CAPACITY_GPKG = Path("outputs/cso_spatial_drivers/qgis/cso_capacity_analysis.gpkg")
HYDRO_DIR = Path("data_raw/hydrogeology")
HYDRO_GPKG = HYDRO_DIR / "HydrogeologyUK_IoM_v5.gpkg"
HYDRO_SHP = HYDRO_DIR / "HydrogeologyUK_IoM_v5.shp"
LOOKUP_CSV = Path("config/hydrogeology_category_lookup.csv")
HYDRO_CSV = Path(
    "data_intermediate/environmental/catchment_hydrogeology_characteristics.csv"
)
MASTER_CSV = Path(
    "data_intermediate/catchment_characteristics/"
    "catchment_characteristics_master.csv"
)
OUTPUT_GPKG = Path(
    "outputs/cso_spatial_drivers/qgis/cso_hydrogeology_analysis.gpkg"
)
LOG_PATH = Path("outputs/cso_spatial_drivers/logs/hydrogeology_extraction.log")

SOURCE_FIELDS = ["CLASS", "CHARACTER", "FLOW_MECHA", "SUMMARY"]
LOOKUP_SOURCE_FIELDS = [
    "source_class",
    "source_character",
    "source_flow_mechanism",
    "source_summary",
]
LOOKUP_FIELDS = LOOKUP_SOURCE_FIELDS + [
    "target_detailed_category",
    "target_productivity_group",
    "target_flow_group",
    "mapping_status",
    "mapping_notes",
]
DETAILED = [
    "intergranular_high",
    "intergranular_moderate",
    "intergranular_low",
    "fracture_high",
    "fracture_moderate",
    "fracture_low",
    "essentially_no_groundwater",
]
DETAILED_COLUMNS = [f"hydro_{name}_pct" for name in DETAILED]
PRODUCTIVITY_COLUMNS = [
    "hydro_high_productivity_pct",
    "hydro_moderate_productivity_pct",
    "hydro_low_productivity_pct",
    "hydro_no_groundwater_pct",
]
FLOW_COLUMNS = ["hydro_intergranular_flow_pct", "hydro_fracture_flow_pct"]
META_COLUMNS = [
    "hydro_dominant_detailed_class",
    "hydro_dominant_productivity_group",
    "hydro_dominant_flow_group",
    "hydro_valid_coverage_pct",
    "hydro_uncovered_pct",
    "hydro_overlap_pct",
    "hydro_source_scale",
    "hydro_source_version",
    "hydro_extraction_status",
]
HYDRO_COLUMNS = DETAILED_COLUMNS + PRODUCTIVITY_COLUMNS + FLOW_COLUMNS + META_COLUMNS
ACCEPTED_SENSOR_STATUSES = {
    "unique_same_company_containment",
    "multiple_same_uwwcode",
}
EXPECTED_LAYERS = {
    "wwtw_sewersheds_capacity",
    "raw_catchments_beta_summary",
    "sensors_capacity_enriched",
}
OUTPUT_LAYERS = {
    "wwtw_sewersheds_hydrogeology",
    "raw_catchments_hydrogeology",
    "hydrogeology_source",
    "hydrogeology_intersections",
    "sensors_beta",
    "hydrogeology_issues",
}
CLASS_MAP = {
    "1A": ("intergranular_high", "high", "intergranular"),
    "1B": ("intergranular_moderate", "moderate", "intergranular"),
    "1C": ("intergranular_low", "low", "intergranular"),
    "2A": ("fracture_high", "high", "fracture"),
    "2B": ("fracture_moderate", "moderate", "fracture"),
    "2C": ("fracture_low", "low", "fracture"),
    "3": (
        "essentially_no_groundwater",
        "no_groundwater",
        "none_or_unproductive",
    ),
}
LOGGER = logging.getLogger("hydrogeology_extraction")
TOL = 1e-6


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


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing required columns {missing}")


def clean_text(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.eq(""))


def category_key(frame: pd.DataFrame, fields: list[str]) -> pd.Series:
    values = [clean_text(frame[field]).fillna("<NULL>") for field in fields]
    result = values[0]
    for value in values[1:]:
        result = result + "\x1f" + value
    return result


def source_paths(root: Path) -> tuple[Path, list[Path]]:
    gpkg = root / HYDRO_GPKG
    if gpkg.is_file():
        return gpkg, [gpkg]
    shp = root / HYDRO_SHP
    required = [shp.with_suffix(ext) for ext in (".shp", ".shx", ".dbf", ".prj")]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Preferred hydrogeology GeoPackage was absent and the shapefile "
            "bundle is incomplete:\n" + "\n".join(str(path) for path in missing)
        )
    # Hash every component belonging to the selected shapefile dataset.
    bundle = sorted(shp.parent.glob(f"{shp.stem}.*"), key=lambda p: p.name.lower())
    return shp, bundle


def source_scale(source: Path) -> str:
    metadata = source.with_suffix(".shp.xml")
    if metadata.is_file():
        text = metadata.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"<rfDenom>\s*([0-9,]+)\s*</rfDenom>", text)
        if match:
            return f"1:{int(match.group(1).replace(',', '')):,}"
        if re.search(r"1\s*:\s*625\s*k", text, flags=re.IGNORECASE):
            return "1:625,000"
    return "not_recorded"


def source_version(hydro: pd.DataFrame, source: Path) -> str:
    if "VERSION" in hydro:
        values = sorted(clean_text(hydro["VERSION"]).dropna().unique())
        if len(values) == 1:
            return str(values[0])
        if values:
            return "|".join(map(str, values))
    return source.stem


def repair_geometries(frame: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int, int]:
    output = frame.copy()
    before = int((output.geometry.notna() & ~output.geometry.is_empty & ~output.geometry.is_valid).sum())
    needs_repair = output.geometry.notna() & ~output.geometry.is_empty & ~output.geometry.is_valid
    if needs_repair.any():
        output.loc[needs_repair, "geometry"] = output.loc[
            needs_repair, "geometry"
        ].make_valid()
    after = int((output.geometry.notna() & ~output.geometry.is_empty & ~output.geometry.is_valid).sum())
    return output, before, after


def make_lookup(categories: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in categories.to_dict("records"):
        code = item["source_class"]
        targets = CLASS_MAP.get(code)
        status = "approved" if targets else "unresolved"
        if targets:
            detailed, productivity, flow = targets
            notes = (
                f"Direct mapping from the source's formal CLASS={code} code "
                "and stated flow mechanism."
            )
            if code == "2B" and item["source_character"] == "Low productive aquifer":
                notes += (
                    " Source CHARACTER conflicts with formal class 2B; the "
                    "original wording and summary are retained for audit."
                )
        else:
            detailed = productivity = flow = pd.NA
            notes = "No supported mapping for this source CLASS code."
        rows.append(
            {
                **item,
                "target_detailed_category": detailed,
                "target_productivity_group": productivity,
                "target_flow_group": flow,
                "mapping_status": status,
                "mapping_notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=LOOKUP_FIELDS)


def load_or_create_lookup(root: Path, hydro: pd.DataFrame) -> pd.DataFrame:
    source_categories = hydro[SOURCE_FIELDS].rename(
        columns=dict(zip(SOURCE_FIELDS, LOOKUP_SOURCE_FIELDS))
    )
    for field in LOOKUP_SOURCE_FIELDS:
        source_categories[field] = clean_text(source_categories[field])
    source_categories = (
        source_categories.drop_duplicates()
        .sort_values(LOOKUP_SOURCE_FIELDS, na_position="first", kind="stable")
        .reset_index(drop=True)
    )
    path = root / LOOKUP_CSV
    if path.is_file():
        lookup = pd.read_csv(path, dtype="string", keep_default_na=True)
        require_columns(lookup, LOOKUP_FIELDS, str(path))
        lookup = lookup[LOOKUP_FIELDS].copy()
        for field in LOOKUP_FIELDS:
            lookup[field] = clean_text(lookup[field])
    else:
        lookup = make_lookup(source_categories)
        path.parent.mkdir(parents=True, exist_ok=True)
        lookup.to_csv(path, index=False)
        LOGGER.info("Created category lookup: %s", path)

    source_keys = set(category_key(source_categories, LOOKUP_SOURCE_FIELDS))
    lookup_keys = category_key(lookup, LOOKUP_SOURCE_FIELDS)
    if lookup_keys.duplicated().any():
        raise ValueError("Hydrogeology lookup contains duplicate source combinations")
    missing = source_keys - set(lookup_keys)
    extra = set(lookup_keys) - source_keys
    if missing or extra:
        raise ValueError(
            "Hydrogeology lookup does not exactly match source categories "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    unresolved = lookup[
        lookup["mapping_status"].str.lower().ne("approved")
        | lookup[
            [
                "target_detailed_category",
                "target_productivity_group",
                "target_flow_group",
            ]
        ].isna().any(axis=1)
    ]
    invalid_targets = (
        ~lookup["target_detailed_category"].isin(DETAILED)
        | ~lookup["target_productivity_group"].isin(
            ["high", "moderate", "low", "no_groundwater"]
        )
        | ~lookup["target_flow_group"].isin(
            ["intergranular", "fracture", "none_or_unproductive"]
        )
    )
    unresolved = pd.concat([unresolved, lookup[invalid_targets]]).drop_duplicates()
    if not unresolved.empty:
        LOGGER.error(
            "Unresolved hydrogeology mappings:\n%s",
            unresolved[LOOKUP_SOURCE_FIELDS + ["mapping_status"]].to_string(index=False),
        )
        raise ValueError(
            f"{len(unresolved)} source category combinations are unresolved; "
            "analytical percentages were not produced"
        )
    return lookup


def dominant(values: dict[str, float]) -> str | pd.NA:
    finite_values = {key: value for key, value in values.items() if np.isfinite(value)}
    if not finite_values or max(finite_values.values()) <= TOL:
        return pd.NA
    maximum = max(finite_values.values())
    return "|".join(
        sorted(key for key, value in finite_values.items() if abs(value - maximum) <= TOL)
    )


def empty_result(
    identity: pd.DataFrame,
    key_fields: list[str],
    area_values: pd.Series,
    source_scale_value: str,
    version: str,
    invalid_keys: set[str],
) -> pd.DataFrame:
    result = identity[key_fields].copy()
    result["_unit_key"] = identity["_unit_key"].to_numpy()
    result["sewershed_area_km2"] = area_values.to_numpy() / 1_000_000
    for column in DETAILED_COLUMNS + PRODUCTIVITY_COLUMNS + FLOW_COLUMNS:
        result[column] = np.nan
    for column in META_COLUMNS:
        result[column] = pd.NA
    result["hydro_valid_coverage_pct"] = np.nan
    result["hydro_uncovered_pct"] = np.nan
    result["hydro_overlap_pct"] = np.nan
    result["hydro_source_scale"] = source_scale_value
    result["hydro_source_version"] = version
    result["hydro_extraction_status"] = [
        "invalid_sewershed_geometry" if key in invalid_keys else "no_valid_hydrogeology"
        for key in identity["_unit_key"]
    ]
    return result


def extract_characteristics(
    units: gpd.GeoDataFrame,
    key_fields: list[str],
    hydro: gpd.GeoDataFrame,
    scale: str,
    version: str,
    area_name: str = "sewershed_area_km2",
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    if units.crs is None or units.crs.to_epsg() != 27700:
        raise ValueError("Analytical unit CRS must be EPSG:27700")
    work = units[key_fields + ["geometry"]].copy()
    if work.duplicated(key_fields).any():
        raise ValueError(f"Analytical units are not unique by {key_fields}")
    work["_unit_key"] = category_key(work, key_fields)
    work, _, _ = repair_geometries(work)
    usable = (
        work.geometry.notna()
        & ~work.geometry.is_empty
        & work.geometry.is_valid
        & (work.geometry.area > 0)
    )
    invalid_keys = set(work.loc[~usable, "_unit_key"])
    areas = work.geometry.area.where(usable, np.nan)
    identity = work[key_fields + ["_unit_key"]]
    result = empty_result(
        identity, key_fields, areas, scale, version, invalid_keys
    )
    result.rename(columns={"sewershed_area_km2": area_name}, inplace=True)

    left = work.loc[usable, key_fields + ["_unit_key", "geometry"]].copy()
    right = hydro[
        [
            "target_detailed_category",
            "target_productivity_group",
            "target_flow_group",
            "geometry",
        ]
    ].copy()
    LOGGER.info("Overlaying %d analytical polygons with %d hydrogeology polygons", len(left), len(right))
    fragments = gpd.overlay(
        left,
        right,
        how="intersection",
        keep_geom_type=True,
        make_valid=False,
    )
    fragments = fragments[
        fragments.geometry.notna()
        & ~fragments.geometry.is_empty
        & (fragments.geometry.area > TOL)
    ].copy()
    fragments["intersection_area_m2"] = fragments.geometry.area
    if fragments.empty:
        return result.drop(columns="_unit_key", errors="ignore"), fragments

    area_lookup = pd.Series(areas.to_numpy(), index=work["_unit_key"]).to_dict()
    fragments["valid_unit_area_m2"] = fragments["_unit_key"].map(area_lookup)
    fragments["coverage_pct_fragment"] = (
        100 * fragments["intersection_area_m2"] / fragments["valid_unit_area_m2"]
    )

    detail_areas = (
        fragments.groupby(["_unit_key", "target_detailed_category"], sort=True)[
            "intersection_area_m2"
        ]
        .sum()
        .unstack(fill_value=0)
    )
    union_areas = fragments.groupby("_unit_key", sort=True).geometry.apply(
        lambda values: shapely.union_all(values.to_numpy()).area
    )
    summed_areas = fragments.groupby("_unit_key")["intersection_area_m2"].sum()
    coverage = 100 * union_areas / pd.Series(area_lookup)
    overlap = 100 * (summed_areas - union_areas).clip(lower=0) / pd.Series(area_lookup)

    row_by_key = {key: index for index, key in enumerate(result["_unit_key"])}
    for key, detail_row in detail_areas.iterrows():
        idx = row_by_key[key]
        denominator = area_lookup[key]
        percentages = {
            name: 100 * float(detail_row.get(name, 0.0)) / denominator
            for name in DETAILED
        }
        for name, value in percentages.items():
            result.at[idx, f"hydro_{name}_pct"] = value
        productivity = {
            "high": percentages["intergranular_high"] + percentages["fracture_high"],
            "moderate": percentages["intergranular_moderate"]
            + percentages["fracture_moderate"],
            "low": percentages["intergranular_low"] + percentages["fracture_low"],
            "no_groundwater": percentages["essentially_no_groundwater"],
        }
        flow = {
            "intergranular": sum(
                percentages[f"intergranular_{level}"]
                for level in ("high", "moderate", "low")
            ),
            "fracture": sum(
                percentages[f"fracture_{level}"]
                for level in ("high", "moderate", "low")
            ),
            "none_or_unproductive": percentages["essentially_no_groundwater"],
        }
        for group, value in productivity.items():
            result.at[idx, f"hydro_{group}_productivity_pct" if group != "no_groundwater" else "hydro_no_groundwater_pct"] = value
        result.at[idx, "hydro_intergranular_flow_pct"] = flow["intergranular"]
        result.at[idx, "hydro_fracture_flow_pct"] = flow["fracture"]
        result.at[idx, "hydro_dominant_detailed_class"] = dominant(percentages)
        result.at[idx, "hydro_dominant_productivity_group"] = dominant(productivity)
        result.at[idx, "hydro_dominant_flow_group"] = dominant(flow)
        result.at[idx, "hydro_valid_coverage_pct"] = float(coverage[key])
        result.at[idx, "hydro_uncovered_pct"] = max(0.0, 100 - float(coverage[key]))
        result.at[idx, "hydro_overlap_pct"] = float(overlap[key])
        cov = float(coverage[key])
        ov = float(overlap[key])
        if ov > 0.1 + TOL:
            status = "source_overlap_review"
        elif cov >= 99.0 - TOL:
            status = "complete"
        elif cov >= 95.0 - TOL:
            status = "minor_uncovered_area"
        else:
            status = "substantial_uncovered_area"
        result.at[idx, "hydro_extraction_status"] = status

    # One audit feature per sewershed/category while fragment counts remain visible.
    audit = fragments.dissolve(
        by=key_fields + ["_unit_key", "target_detailed_category"],
        aggfunc={
            "target_productivity_group": "first",
            "target_flow_group": "first",
            "intersection_area_m2": ["sum", "count"],
            "valid_unit_area_m2": "first",
        },
        as_index=False,
    )
    audit.columns = [
        "_".join(part for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in audit.columns
    ]
    audit.rename(
        columns={
            "intersection_area_m2_sum": "intersection_area_m2",
            "intersection_area_m2_count": "fragment_count",
            "valid_unit_area_m2_first": "valid_unit_area_m2",
            "target_productivity_group_first": "target_productivity_group",
            "target_flow_group_first": "target_flow_group",
        },
        inplace=True,
    )
    audit["coverage_pct"] = (
        100 * audit["intersection_area_m2"] / audit["valid_unit_area_m2"]
    )
    result.drop(columns="_unit_key", inplace=True)
    return result, gpd.GeoDataFrame(audit, geometry="geometry", crs=BNG)


def validate_percentages(frame: pd.DataFrame, label: str) -> None:
    pct_columns = DETAILED_COLUMNS + PRODUCTIVITY_COLUMNS + FLOW_COLUMNS + [
        "hydro_valid_coverage_pct",
        "hydro_uncovered_pct",
        "hydro_overlap_pct",
    ]
    numeric = frame[pct_columns].apply(pd.to_numeric, errors="coerce")
    original_not_missing = frame[pct_columns].notna()
    if (original_not_missing & numeric.isna()).any().any():
        raise ValueError(f"{label}: a percentage is non-numeric and not explicitly missing")
    if ((numeric < -TOL) | (numeric > 100 + TOL)).any().any():
        bad = ((numeric < -TOL) | (numeric > 100 + TOL)).sum()
        raise ValueError(f"{label}: percentages outside [0, 100]: {bad[bad > 0].to_dict()}")
    valid = numeric["hydro_valid_coverage_pct"].notna()
    detail_total = numeric[DETAILED_COLUMNS].sum(axis=1, min_count=1)
    expected_detail = (
        numeric["hydro_valid_coverage_pct"] + numeric["hydro_overlap_pct"]
    )
    if not np.allclose(
        detail_total[valid], expected_detail[valid], atol=1e-5, rtol=1e-8
    ):
        raise ValueError(f"{label}: detailed totals disagree with coverage plus overlap")
    productivity_total = numeric[PRODUCTIVITY_COLUMNS].sum(axis=1, min_count=1)
    if not np.allclose(
        detail_total[valid], productivity_total[valid], atol=1e-7, rtol=1e-9
    ):
        raise ValueError(f"{label}: productivity totals disagree with detailed values")
    expected_intergranular = numeric[
        [f"hydro_intergranular_{level}_pct" for level in ("high", "moderate", "low")]
    ].sum(axis=1, min_count=1)
    expected_fracture = numeric[
        [f"hydro_fracture_{level}_pct" for level in ("high", "moderate", "low")]
    ].sum(axis=1, min_count=1)
    if not np.allclose(
        numeric.loc[valid, "hydro_intergranular_flow_pct"],
        expected_intergranular[valid],
        atol=1e-7,
    ) or not np.allclose(
        numeric.loc[valid, "hydro_fracture_flow_pct"],
        expected_fracture[valid],
        atol=1e-7,
    ):
        raise ValueError(f"{label}: flow totals disagree with detailed values")


def write_csv_staged(frame: pd.DataFrame, final_path: Path, token: str) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.stem}.{token}.tmp.csv")
    frame.to_csv(temporary, index=False)
    check = pd.read_csv(temporary, low_memory=False)
    if len(check) != len(frame):
        raise ValueError(f"Staged CSV row count mismatch: {temporary}")
    return temporary


def write_gpkg_staged(
    layers: dict[str, gpd.GeoDataFrame], final_path: Path, token: str
) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.stem}.{token}.tmp.gpkg")
    for index, (name, frame) in enumerate(layers.items()):
        if frame.crs is None or frame.crs.to_epsg() != 27700:
            raise ValueError(f"Output layer {name} is not EPSG:27700")
        frame.to_file(
            temporary,
            layer=name,
            driver="GPKG",
            mode="w" if index == 0 else "a",
            engine="pyogrio",
        )
    listing = gpd.list_layers(temporary)
    if set(listing["name"]) != OUTPUT_LAYERS:
        raise ValueError("Staged GeoPackage does not contain the required layers")
    for name, expected in layers.items():
        check = gpd.read_file(temporary, layer=name)
        if len(check) != len(expected):
            raise ValueError(f"Staged GeoPackage count mismatch for {name}")
        if check.crs is None or check.crs.to_epsg() != 27700:
            raise ValueError(f"Staged GeoPackage layer {name} has wrong CRS")
    return temporary


def replace_staged(staged: Path, final: Path) -> None:
    try:
        os.replace(staged, final)
    except PermissionError as exc:
        raise PermissionError(
            f"Could not replace {final}. Close the GeoPackage in QGIS and retry."
        ) from exc


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    main_outputs = [root / HYDRO_CSV, root / MASTER_CSV, root / OUTPUT_GPKG]
    existing = [path for path in main_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Main outputs already exist; rerun with --overwrite:\n"
            + "\n".join(map(str, existing))
        )
    configure_logging(root / LOG_PATH)
    for path in [root / CAPACITY_CSV, root / CAPACITY_GPKG]:
        if not path.is_file():
            raise FileNotFoundError(f"Required validated input is absent: {path}")

    source, source_bundle = source_paths(root)
    hashes_before = {str(path.relative_to(root)): sha256_file(path) for path in source_bundle}
    LOGGER.info("Hydrogeology source: %s", source)
    LOGGER.info("Source files and SHA-256: %s", hashes_before)
    hydro = gpd.read_file(source)
    require_columns(hydro, SOURCE_FIELDS, str(source))
    if hydro.crs is None:
        raise ValueError("Hydrogeology source CRS is undefined")
    source_crs = hydro.crs.to_string()
    if hydro.crs.to_epsg() != 27700:
        hydro = hydro.to_crs(BNG)
    source_geometry_types = "|".join(sorted(hydro.geom_type.dropna().unique()))
    null_count = int(hydro.geometry.isna().sum())
    empty_count = int(hydro.geometry.is_empty.sum())
    hydro, invalid_before, invalid_after = repair_geometries(hydro)
    hydro = hydro[
        hydro.geometry.notna() & ~hydro.geometry.is_empty & hydro.geometry.is_valid
    ].copy()
    scale = source_scale(source)
    version = source_version(hydro, source)
    lookup = load_or_create_lookup(root, hydro)
    source_category_count = len(lookup)
    unresolved_count = int(lookup["mapping_status"].str.lower().ne("approved").sum())
    hydro["_category_key"] = category_key(hydro, SOURCE_FIELDS)
    lookup["_category_key"] = category_key(lookup, LOOKUP_SOURCE_FIELDS)
    hydro = hydro.merge(
        lookup[
            [
                "_category_key",
                "target_detailed_category",
                "target_productivity_group",
                "target_flow_group",
                "mapping_status",
                "mapping_notes",
            ]
        ],
        on="_category_key",
        how="left",
        validate="many_to_one",
    )
    if hydro["target_detailed_category"].isna().any():
        raise ValueError("Mapped hydrogeology source contains missing targets")
    hydro = gpd.GeoDataFrame(hydro, geometry="geometry", crs=BNG)
    LOGGER.info(
        "Source validation: features=%d, CRS=%s, geometry=%s, null=%d, empty=%d, "
        "invalid before=%d, invalid after=%d, categories=%d, version=%s, scale=%s",
        len(hydro), source_crs, source_geometry_types, null_count, empty_count,
        invalid_before, invalid_after, source_category_count, version, scale,
    )

    layer_listing = gpd.list_layers(root / CAPACITY_GPKG)
    missing_layers = EXPECTED_LAYERS - set(layer_listing["name"])
    if missing_layers:
        raise ValueError(f"Capacity GeoPackage missing layers: {sorted(missing_layers)}")
    LOGGER.info("Capacity GeoPackage layers: %s", layer_listing.to_dict("records"))
    wwtw = gpd.read_file(root / CAPACITY_GPKG, layer="wwtw_sewersheds_capacity")
    raw = gpd.read_file(root / CAPACITY_GPKG, layer="raw_catchments_beta_summary")
    sensors = gpd.read_file(root / CAPACITY_GPKG, layer="sensors_capacity_enriched")
    for label, frame in [("WWTW", wwtw), ("raw catchment", raw), ("sensors", sensors)]:
        if frame.crs is None:
            raise ValueError(f"{label} input CRS is undefined")
        if frame.crs.to_epsg() != 27700:
            frame.to_crs(BNG, inplace=True)

    wwtw_keys = ["company", "uwwCode", "uwwName"]
    raw_keys = [
        "company",
        "raw_catchment_identifier",
        "raw_catchment_name",
        "raw_catchment_company",
        "uwwCode",
        "uwwName",
    ]
    require_columns(wwtw, wwtw_keys, "WWTW layer")
    require_columns(raw, raw_keys, "raw catchment layer")
    hydro_table, intersections = extract_characteristics(
        wwtw, wwtw_keys, hydro, scale, version
    )
    raw_table, _ = extract_characteristics(
        raw, raw_keys, hydro, scale, version, area_name="raw_catchment_area_km2"
    )
    validate_percentages(hydro_table, "WWTW hydrogeology")
    validate_percentages(raw_table, "raw-catchment hydrogeology")
    if hydro_table.duplicated(["company", "uwwCode"]).any():
        raise ValueError("Hydrogeology table is not unique by company + uwwCode")

    capacity = pd.read_csv(root / CAPACITY_CSV, low_memory=False)
    require_columns(capacity, ["company", "uwwCode", "uwwName"], str(CAPACITY_CSV))
    if capacity.duplicated(["company", "uwwCode"]).any():
        raise ValueError("Capacity summary is not unique by company + uwwCode")
    hydro_join = hydro_table.drop(columns="uwwName")
    master = capacity.merge(
        hydro_join,
        on=["company", "uwwCode"],
        how="left",
        validate="one_to_one",
    )
    if len(master) != len(capacity):
        raise ValueError("Environmental join multiplied or dropped WWTW rows")
    master.insert(
        0,
        "spatial_unit_id",
        clean_text(master["company"]) + "::" + clean_text(master["uwwCode"]),
    )
    if master["spatial_unit_id"].isna().any() or master["spatial_unit_id"].duplicated().any():
        raise ValueError("Master spatial_unit_id values are missing or duplicated")
    if "all_finite_beta_sensor_count" in master:
        master["finite_beta_sensor_count"] = master["all_finite_beta_sensor_count"]
    minimum_master = {
        "spatial_unit_id", "company", "uwwCode", "uwwName", "sewershed_area_km2",
        "assigned_sensor_count", "finite_beta_sensor_count",
        "primary_beta_sensor_count", "median_beta_primary", "mean_beta_primary",
        "beta_q25_primary", "beta_q75_primary", "beta_iqr_primary",
        "median_beta_ci_width_primary", "median_tail_spill_count_primary",
        "median_beta_all_finite", "load_entering_pe_latest",
        "design_capacity_pe_latest", "capacity_ratio_latest", "capacity_data_year",
        "capacity_status", *HYDRO_COLUMNS,
    }
    require_columns(master, minimum_master, "master table")

    wwtw_enriched = wwtw.merge(
        hydro_join, on=["company", "uwwCode"], how="left", validate="one_to_one"
    )
    raw_enriched = raw.merge(
        raw_table.drop(columns=["raw_catchment_name", "raw_catchment_company", "uwwCode", "uwwName"]),
        on=["company", "raw_catchment_identifier"],
        how="left",
        validate="one_to_one",
    )
    accepted_sensors = sensors[
        sensors["assignment_status"].isin(ACCEPTED_SENSOR_STATUSES)
        & sensors.geometry.notna()
        & ~sensors.geometry.is_empty
    ].copy()
    source_layer = hydro.drop(columns="_category_key")
    issue_statuses = {
        "minor_uncovered_area",
        "substantial_uncovered_area",
        "source_overlap_review",
        "no_valid_hydrogeology",
        "invalid_sewershed_geometry",
    }
    issues = wwtw_enriched[
        wwtw_enriched["hydro_extraction_status"].isin(issue_statuses)
    ].copy()
    if issues.empty:
        issues = wwtw_enriched.iloc[0:0].copy()

    token = uuid.uuid4().hex
    staged: list[Path] = []
    try:
        hydro_tmp = write_csv_staged(hydro_table, root / HYDRO_CSV, token)
        master_tmp = write_csv_staged(master, root / MASTER_CSV, token)
        staged.extend([hydro_tmp, master_tmp])
        gpkg_tmp = write_gpkg_staged(
            {
                "wwtw_sewersheds_hydrogeology": gpd.GeoDataFrame(wwtw_enriched, geometry="geometry", crs=BNG),
                "raw_catchments_hydrogeology": gpd.GeoDataFrame(raw_enriched, geometry="geometry", crs=BNG),
                "hydrogeology_source": gpd.GeoDataFrame(source_layer, geometry="geometry", crs=BNG),
                "hydrogeology_intersections": intersections,
                "sensors_beta": gpd.GeoDataFrame(accepted_sensors, geometry="geometry", crs=BNG),
                "hydrogeology_issues": gpd.GeoDataFrame(issues, geometry="geometry", crs=BNG),
            },
            root / OUTPUT_GPKG,
            token,
        )
        staged.append(gpkg_tmp)
        hashes_after = {str(path.relative_to(root)): sha256_file(path) for path in source_bundle}
        if hashes_after != hashes_before:
            raise ValueError("Hydrogeology source hashes changed during extraction")
        replace_staged(hydro_tmp, root / HYDRO_CSV)
        replace_staged(master_tmp, root / MASTER_CSV)
        replace_staged(gpkg_tmp, root / OUTPUT_GPKG)
    finally:
        for path in staged:
            if path.exists():
                path.unlink()

    statuses = hydro_table["hydro_extraction_status"].value_counts()
    coverage = pd.to_numeric(hydro_table["hydro_valid_coverage_pct"], errors="coerce")
    LOGGER.info("REPORT source_feature_count=%d", len(hydro))
    LOGGER.info("REPORT source_crs=%s", source_crs)
    LOGGER.info("REPORT invalid_geometry_before=%d after=%d", invalid_before, invalid_after)
    LOGGER.info("REPORT unique_source_categories=%d unresolved=%d", source_category_count, unresolved_count)
    LOGGER.info("REPORT wwtw_processed=%d", len(hydro_table))
    for status in [
        "complete", "minor_uncovered_area", "substantial_uncovered_area",
        "source_overlap_review", "no_valid_hydrogeology",
    ]:
        LOGGER.info("REPORT %s=%d", status, int(statuses.get(status, 0)))
    LOGGER.info(
        "REPORT master_rows=%d duplicate_master_keys=%d",
        len(master), int(master["spatial_unit_id"].duplicated().sum()),
    )
    LOGGER.info(
        "REPORT coverage_min_median_max=%.6f,%.6f,%.6f",
        coverage.min(), coverage.median(), coverage.max(),
    )
    LOGGER.info("REPORT source_hash_result=unchanged (%d files)", len(hashes_before))
    LOGGER.info("Outputs: %s", [str(path) for path in main_outputs] + [str(root / LOOKUP_CSV), str(root / LOG_PATH)])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.exception("Hydrogeology extraction failed")
        raise

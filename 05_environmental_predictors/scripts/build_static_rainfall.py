#!/usr/bin/env python3
"""Build the Goal 05 WWTW rainfall-climatology enrichment.

This stage uses only official HadUK-Grid 1 km long-term-average products
already present in ``data_raw/rainfall``.  It performs exact partial-cell
area weighting in British National Grid, creates the focused QGIS package,
and atomically enriches the authoritative CSV only after all release gates
pass.  Daily rainfall is deliberately outside this workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from rasterio.crs import CRS
from shapely.geometry import box


BNG = "EPSG:27700"
KEYS = ["company", "uwwCode"]
FINAL_BRANCH = Path("project_results/final_wwtw_sensor_manual_release")
GOAL_DIR = Path("project_results/goal_05_rainfall")
ANNUAL_FILE = Path("data_raw/rainfall/rainfall_hadukgrid_uk_1km_ann-30y_199101-202012.nc")
SEASONAL_FILE = Path("data_raw/rainfall/rainfall_hadukgrid_uk_1km_seas-30y_199101-202012.nc")
MASTER_NAME = "final_wwtw_sensor_supervisor_view.csv"
SOURCE_GPKG_NAME = "final_wwtw_sensor_assignment.gpkg"
OUTPUT_GPKG_NAME = "wwtw_rainfall_analysis.gpkg"
OUTPUT_TXT_NAME = "rainfall_pipeline_supervisor_explanation.txt"
ANNUAL_COLUMN = "rain_mean_annual_mm_1991_2020"
WINTER_COLUMN = "rain_winter_mean_mm_1991_2020"
PRODUCED_COLUMNS = [ANNUAL_COLUMN, WINTER_COLUMN]
DEFERRED_COLUMNS = [
    "rain_wet_days_ge_1mm_per_year_1991_2020",
    "rain_wet_days_ge_10mm_per_year_1991_2020",
]
OUTPUT_LAYERS = [
    "matching_cso_sensors",
    "assigned_treatment_works",
    "wwtw_catchments_rainfall",
    "rainfall_grid_1km_clipped_to_catchments",
]
NODATA_LIMIT = 1e19


class ReleaseBlocked(RuntimeError):
    """Raised when a scientific or publication release gate fails."""


@dataclass(frozen=True)
class RasterProduct:
    """Validated in-memory rainfall grid and its provenance."""

    path: Path
    values: np.ndarray
    transform: rasterio.Affine
    crs: CRS
    nodata: float | None
    tags: Mapping[str, str]
    band: int
    variable_name: str


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ReleaseBlocked(f"{label} missing required columns: {missing}")


def resolve_latest_run(branch: Path) -> Path:
    pointer = branch / "LATEST.txt"
    if not pointer.is_file():
        raise ReleaseBlocked(f"Missing authoritative pointer: {pointer}")
    value = pointer.read_text(encoding="utf-8").strip()
    run = (branch / value).resolve()
    runs = (branch / "runs").resolve()
    if not run.is_dir() or runs not in run.parents:
        raise ReleaseBlocked(f"LATEST.txt does not resolve to a validated run: {run}")
    return run


def netcdf_variables(path: Path) -> list[str]:
    """Return NetCDF subdataset variable names exposed by GDAL."""
    with rasterio.open(path) as dataset:
        names = []
        for item in dataset.subdatasets:
            names.append(item.rsplit(":", 1)[-1])
        return names


def _validate_common_metadata(path: Path, tags: Mapping[str, str], expected_frequency: str) -> None:
    expected = {
        "NC_GLOBAL#institution": "Met Office",
        "NC_GLOBAL#lta_period": "1991-2020",
        "NC_GLOBAL#frequency": expected_frequency,
        "rainfall#units": "mm",
        "rainfall#standard_name": "lwe_thickness_of_precipitation_amount",
    }
    for key, value in expected.items():
        if tags.get(key) != value:
            raise ReleaseBlocked(f"Unexpected {key} in {path}: {tags.get(key)!r}; expected {value!r}")
    if not tags.get("NC_GLOBAL#source", "").startswith("HadUK-Grid_v"):
        raise ReleaseBlocked(f"Source is not identified as HadUK-Grid in {path}")


def is_british_national_grid(tags: Mapping[str, str]) -> bool:
    """Validate BNG from CF parameters when the NetCDF omits an EPSG authority."""
    expected_text = {"transverse_mercator#grid_mapping_name": "transverse_mercator"}
    expected_numeric = {
        "transverse_mercator#latitude_of_projection_origin": 49.0,
        "transverse_mercator#longitude_of_central_meridian": -2.0,
        "transverse_mercator#scale_factor_at_central_meridian": 0.9996012717,
        "transverse_mercator#false_easting": 400000.0,
        "transverse_mercator#false_northing": -100000.0,
        "transverse_mercator#semi_major_axis": 6377563.396,
        "transverse_mercator#semi_minor_axis": 6356256.909,
    }
    if any(tags.get(key) != value for key, value in expected_text.items()):
        return False
    try:
        return all(math.isclose(float(tags[key]), value, rel_tol=0, abs_tol=1e-6) for key, value in expected_numeric.items())
    except (KeyError, TypeError, ValueError):
        return False


def load_raster_product(path: Path, frequency: str, band: int) -> RasterProduct:
    if not path.is_file():
        raise ReleaseBlocked(f"Missing rainfall source: {path}")
    with rasterio.open(path) as dataset:
        tags = dict(dataset.tags())
        _validate_common_metadata(path, tags, frequency)
        if dataset.crs is None or not is_british_national_grid(tags):
            raise ReleaseBlocked(f"Rainfall grid is not EPSG:27700: {path} ({dataset.crs})")
        if dataset.transform.a != 1000 or dataset.transform.e != -1000:
            raise ReleaseBlocked(f"Rainfall grid is not aligned at 1 km: {path} ({dataset.transform})")
        if not 1 <= band <= dataset.count:
            raise ReleaseBlocked(f"Requested band {band} absent from {path}")
        band_tags = dict(dataset.tags(band))
        if band_tags.get("NETCDF_VARNAME") != "rainfall" or band_tags.get("units") != "mm":
            raise ReleaseBlocked(f"Band {band} in {path} is not rainfall in millimetres")
        values = dataset.read(band, out_dtype="float64")
        product = RasterProduct(
            path=path,
            values=values,
            transform=dataset.transform,
            # The CF projection parameters are BNG but the source WKT omits its
            # authority code; use the validated canonical CRS for analysis.
            crs=CRS.from_epsg(27700),
            nodata=dataset.nodata,
            tags=tags,
            band=band,
            variable_name=band_tags["NETCDF_VARNAME"],
        )
    valid = valid_mask(product)
    if not valid.any() or np.nanmin(product.values[valid]) < 0:
        raise ReleaseBlocked(f"Rainfall grid has no usable data or contains negative valid values: {path}")
    return product


def seasonal_band_dates(product: RasterProduct) -> list[datetime]:
    raw = product.tags.get("NETCDF_DIM_time_VALUES", "")
    numbers = [int(item) for item in raw.strip("{}").split(",") if item.strip()]
    if len(numbers) != 4:
        raise ReleaseBlocked(f"Seasonal file does not expose four time coordinates: {raw!r}")
    dates = [datetime(1800, 1, 1) + timedelta(hours=value) for value in numbers]
    if [date.month for date in dates] != [1, 4, 7, 10]:
        raise ReleaseBlocked(f"Unexpected seasonal order: {[date.isoformat() for date in dates]}")
    if product.band != 1:
        raise ReleaseBlocked("Winter extraction must use seasonal band 1 (January-centred DJF)")
    return dates


def valid_mask(product: RasterProduct) -> np.ndarray:
    values = product.values
    mask = np.isfinite(values) & (values < NODATA_LIMIT) & (values >= 0)
    if product.nodata is not None and np.isfinite(product.nodata):
        mask &= values != product.nodata
    return mask


def validate_grid_alignment(annual: RasterProduct, winter: RasterProduct) -> None:
    if annual.values.shape != winter.values.shape:
        raise ReleaseBlocked("Annual and seasonal rainfall grids differ in dimensions")
    if annual.transform != winter.transform or annual.crs != winter.crs:
        raise ReleaseBlocked("Annual and seasonal rainfall grids are not exactly aligned")
    if not np.array_equal(valid_mask(annual), valid_mask(winter)):
        raise ReleaseBlocked("Annual and winter rainfall valid-data masks differ")


def load_spatial_sources(source_gpkg: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not source_gpkg.is_file():
        raise ReleaseBlocked(f"Missing final assignment GeoPackage: {source_gpkg}")
    layers = set(pyogrio.list_layers(source_gpkg)[:, 0])
    required = {"wwtw_catchment_areas", "assigned_treatment_works", "matching_cso_sensors"}
    if not required.issubset(layers):
        raise ReleaseBlocked(f"Final assignment GeoPackage missing layers: {sorted(required - layers)}")
    catchments = gpd.read_file(source_gpkg, layer="wwtw_catchment_areas", engine="pyogrio")
    assigned = gpd.read_file(source_gpkg, layer="assigned_treatment_works", engine="pyogrio")
    sensors = gpd.read_file(source_gpkg, layer="matching_cso_sensors", engine="pyogrio")
    for label, frame in [("catchments", catchments), ("assigned treatment works", assigned), ("matching sensors", sensors)]:
        if frame.crs is None or frame.crs.to_epsg() != 27700:
            raise ReleaseBlocked(f"{label} are not EPSG:27700")
        if frame.geometry.isna().any() or frame.geometry.is_empty.any() or (~frame.geometry.is_valid).any():
            raise ReleaseBlocked(f"{label} contain null, empty, or invalid geometries")
    require_columns(catchments, KEYS + ["uwwName"], "catchments")
    if catchments.duplicated(KEYS).any():
        attributes = {column: "first" for column in catchments.columns if column not in KEYS + ["geometry"]}
        catchments = catchments.dissolve(KEYS, aggfunc=attributes, as_index=False)
    if catchments.duplicated(KEYS).any():
        raise ReleaseBlocked("Catchments are not unique by company + uwwCode")
    if len(assigned) != len(sensors):
        raise ReleaseBlocked("Assigned-treatment-work and matching-sensor counts differ")
    return catchments, assigned, sensors


def _polygon_parts(geometry: object) -> list[object]:
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    fixed = shapely.make_valid(geometry)
    if fixed.geom_type == "Polygon":
        return [fixed]
    if fixed.geom_type == "MultiPolygon":
        return list(fixed.geoms)
    parts = [part for part in getattr(fixed, "geoms", []) if part.geom_type in {"Polygon", "MultiPolygon"}]
    return [subpart for part in parts for subpart in _polygon_parts(part)]


def candidate_cells(geometry: object, transform: rasterio.Affine, height: int, width: int) -> set[tuple[int, int]]:
    """Return grid indices whose cells may intersect a polygon."""
    left, top = transform.c, transform.f
    cell_width, cell_height = transform.a, -transform.e
    candidates: set[tuple[int, int]] = set()
    for part in _polygon_parts(geometry):
        minx, miny, maxx, maxy = part.bounds
        col0 = max(0, math.floor((minx - left) / cell_width))
        col1 = min(width - 1, math.ceil((maxx - left) / cell_width) - 1)
        row0 = max(0, math.floor((top - maxy) / cell_height))
        row1 = min(height - 1, math.ceil((top - miny) / cell_height) - 1)
        if col0 <= col1 and row0 <= row1:
            candidates.update((row, col) for row in range(row0, row1 + 1) for col in range(col0, col1 + 1))
    return candidates


def cell_polygon(row: int, col: int, transform: rasterio.Affine) -> object:
    x0 = transform.c + col * transform.a
    x1 = x0 + transform.a
    y1 = transform.f + row * transform.e
    y0 = y1 + transform.e
    return box(x0, y0, x1, y1)


def intersect_grid_with_catchments(
    catchments: gpd.GeoDataFrame,
    annual: RasterProduct,
    winter: RasterProduct,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Create exact cell/catchment intersections and area-weighted summaries."""
    annual_valid = valid_mask(annual)
    winter_valid = valid_mask(winter)
    common_valid = annual_valid & winter_valid
    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    height, width = annual.values.shape
    full_cell_area = abs(annual.transform.a * annual.transform.e)
    for item in catchments.itertuples(index=False):
        company = str(getattr(item, "company"))
        uww_code = str(getattr(item, "uwwCode"))
        name = str(getattr(item, "uwwName"))
        geometry = getattr(item, "geometry")
        catchment_area = float(geometry.area)
        if not np.isfinite(catchment_area) or catchment_area <= 0:
            raise ReleaseBlocked(f"Invalid catchment area for {company} / {uww_code}")
        overlap_total = 0.0
        valid_overlap = 0.0
        annual_numerator = 0.0
        winter_numerator = 0.0
        effective_cells = 0.0
        intersecting_count = 0
        for row, col in sorted(candidate_cells(geometry, annual.transform, height, width)):
            cell = cell_polygon(row, col, annual.transform)
            if not geometry.intersects(cell):
                continue
            intersection = geometry.intersection(cell)
            overlap_area = float(intersection.area)
            if overlap_area <= 0:
                continue
            polygon_parts = _polygon_parts(intersection)
            if not polygon_parts:
                continue
            intersection = shapely.union_all(polygon_parts)
            overlap_area = float(intersection.area)
            intersecting_count += 1
            overlap_total += overlap_area
            effective_cells += overlap_area / full_cell_area
            is_valid = bool(common_valid[row, col])
            annual_value = float(annual.values[row, col]) if annual_valid[row, col] else np.nan
            winter_value = float(winter.values[row, col]) if winter_valid[row, col] else np.nan
            if is_valid:
                valid_overlap += overlap_area
                annual_numerator += overlap_area * annual_value
                winter_numerator += overlap_area * winter_value
            centroid = cell.centroid
            records.append({
                "company": company,
                "uwwCode": uww_code,
                "wwtw_name": name,
                "grid_cell_id": f"r{row:04d}_c{col:04d}",
                "full_grid_cell_area_m2": full_cell_area,
                "overlap_area_m2": overlap_area,
                "overlap_fraction": overlap_area / full_cell_area,
                "grid_cell_centroid_easting": centroid.x,
                "grid_cell_centroid_northing": centroid.y,
                "cell_rain_mean_annual_mm_1991_2020": annual_value,
                "cell_rain_winter_mean_mm_1991_2020": winter_value,
                "valid_data_flag": is_valid,
                "geometry": intersection,
            })
        coverage = 100.0 * valid_overlap / catchment_area
        nodata_pct = max(0.0, 100.0 - coverage)
        if intersecting_count == 0:
            status = "no_raster_overlap"
        elif valid_overlap <= 0:
            status = "failed_extraction"
        elif coverage >= 99.5:
            status = "complete"
        elif coverage >= 95:
            status = "minor_nodata"
        else:
            status = "substantial_nodata"
        summaries.append({
            "company": company,
            "uwwCode": uww_code,
            ANNUAL_COLUMN: annual_numerator / valid_overlap if valid_overlap > 0 else np.nan,
            WINTER_COLUMN: winter_numerator / valid_overlap if valid_overlap > 0 else np.nan,
            "catchment_total_area_m2": catchment_area,
            "valid_rainfall_overlap_area_m2": valid_overlap,
            "rain_valid_coverage_pct": coverage,
            "rain_nodata_pct": nodata_pct,
            "rain_extraction_status": status,
            "intersecting_grid_cell_count": intersecting_count,
            "effective_grid_cell_count": effective_cells,
            "raster_grid_overlap_area_m2": overlap_total,
        })
    grid = gpd.GeoDataFrame(records, geometry="geometry", crs=BNG)
    summary = pd.DataFrame(summaries)
    if summary.duplicated(KEYS).any() or len(summary) != len(catchments):
        raise ReleaseBlocked("Rainfall summary is not one row per authoritative catchment")
    return grid, summary


def validate_summary(summary: pd.DataFrame) -> None:
    for column in PRODUCED_COLUMNS:
        numeric = pd.to_numeric(summary[column], errors="coerce")
        if numeric.notna().any() and ((~np.isfinite(numeric.dropna())) | (numeric.dropna() < 0)).any():
            raise ReleaseBlocked(f"Invalid rainfall values in {column}")
    annual = pd.to_numeric(summary[ANNUAL_COLUMN], errors="coerce")
    if (annual.dropna() <= 0).any():
        raise ReleaseBlocked("Mean annual rainfall contains a non-positive value")
    if (summary["rain_valid_coverage_pct"] < -1e-8).any() or (summary["rain_valid_coverage_pct"] > 100.000001).any():
        raise ReleaseBlocked("Rainfall coverage lies outside 0-100%")
    allowed = {"complete", "minor_nodata", "substantial_nodata", "no_raster_overlap", "failed_extraction"}
    if not set(summary["rain_extraction_status"]).issubset(allowed):
        raise ReleaseBlocked("Unexpected rainfall extraction status")


def build_master_candidate(master: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    require_columns(master, KEYS, "authoritative master")
    if master.duplicated(KEYS).any():
        raise ReleaseBlocked("Authoritative master key company + uwwCode is not unique")
    existing_deferred = [column for column in master.columns if column in DEFERRED_COLUMNS]
    if existing_deferred:
        raise ReleaseBlocked(f"Deferred wet-day columns must not exist in master: {existing_deferred}")
    existing_produced = [column for column in PRODUCED_COLUMNS if column in master.columns]
    if existing_produced and existing_produced != PRODUCED_COLUMNS:
        raise ReleaseBlocked(f"Master contains an incomplete rainfall block: {existing_produced}")
    lookup = summary.set_index(KEYS)
    key_index = pd.MultiIndex.from_frame(master[KEYS])
    if not key_index.isin(lookup.index).all():
        missing = master.loc[~key_index.isin(lookup.index), KEYS]
        raise ReleaseBlocked(f"Master keys without rainfall catchment: {missing.head().to_dict('records')}")
    output = master.drop(columns=PRODUCED_COLUMNS, errors="ignore").copy()
    insertion = output.columns.get_loc("hydro_extraction_status") + 1 if "hydro_extraction_status" in output else len(output.columns)
    for offset, column in enumerate(PRODUCED_COLUMNS):
        values = lookup[column].reindex(key_index).to_numpy(dtype=float)
        output.insert(insertion + offset, column, values)
    return output


def validate_master_candidate(old: pd.DataFrame, new: pd.DataFrame) -> None:
    if len(old) != len(new):
        raise ReleaseBlocked("Master row count changed")
    if not old[KEYS].equals(new[KEYS]):
        raise ReleaseBlocked("Master key sequence or row order changed")
    if old.duplicated(KEYS).any() or new.duplicated(KEYS).any():
        raise ReleaseBlocked("Master key is not unique")
    protected = [column for column in old.columns if column not in PRODUCED_COLUMNS]
    for column in protected:
        if column not in new.columns or not old[column].equals(new[column]):
            raise ReleaseBlocked(f"Protected pre-existing master column changed: {column}")
    expected_count = len(old.columns) + (0 if all(column in old.columns for column in PRODUCED_COLUMNS) else 2)
    if len(new.columns) != expected_count or any(list(new.columns).count(column) != 1 for column in PRODUCED_COLUMNS):
        raise ReleaseBlocked("Master does not contain exactly one complete two-column rainfall block")
    if any(column in new.columns for column in DEFERRED_COLUMNS):
        raise ReleaseBlocked("A deferred wet-day column was written to the master")
    forbidden_suffixes = ("_x", "_y", "_old", "_new")
    if any(column.endswith(forbidden_suffixes) for column in new.columns):
        raise ReleaseBlocked("Merge-suffix column detected in master candidate")


def build_output_layers(
    catchments: gpd.GeoDataFrame,
    assigned: gpd.GeoDataFrame,
    sensors: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    summary: pd.DataFrame,
) -> dict[str, gpd.GeoDataFrame]:
    rain = summary[KEYS + PRODUCED_COLUMNS + [
        "rain_valid_coverage_pct", "rain_nodata_pct", "rain_extraction_status",
        "intersecting_grid_cell_count", "catchment_total_area_m2",
    ]]
    catch = catchments.merge(rain, on=KEYS, how="left", validate="one_to_one")
    assigned_lookup = assigned[KEYS + ["sensor_uid", "fitted_beta"]].copy()
    catch = catch.merge(assigned_lookup, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_selected"))
    catch["wwtw_name"] = catch["uwwName"].astype("string")
    catch["selected_sensor_uid"] = catch.get("sensor_uid_selected", catch.get("sensor_uid"))
    catch["catchment_area_km2"] = catch["catchment_total_area_m2"] / 1_000_000
    catch_fields = KEYS + ["wwtw_name", "selected_sensor_uid", "fitted_beta"] + PRODUCED_COLUMNS + [
        "rain_valid_coverage_pct", "rain_nodata_pct", "rain_extraction_status",
        "intersecting_grid_cell_count", "catchment_area_km2", "geometry",
    ]
    catch_layer = gpd.GeoDataFrame(catch[catch_fields], geometry="geometry", crs=BNG)

    points = assigned.merge(summary[KEYS + PRODUCED_COLUMNS], on=KEYS, how="left", validate="one_to_one")
    points["wwtw_name"] = points["canonical_wwtw_name"].astype("string")
    point_fields = KEYS + ["wwtw_name", "sensor_uid", "assignment_source", "fitted_beta"] + PRODUCED_COLUMNS + ["geometry"]
    point_layer = gpd.GeoDataFrame(points[point_fields], geometry="geometry", crs=BNG)

    sensor_frame = sensors.merge(summary[KEYS + PRODUCED_COLUMNS], on=KEYS, how="left", validate="many_to_one")
    sensor_frame["sensor_name"] = sensor_frame["location_name"].astype("string")
    sensor_frame["assigned_uwwCode"] = sensor_frame["uwwCode"].astype("string")
    sensor_frame["assigned_wwtw_name"] = sensor_frame["canonical_wwtw_name"].astype("string")
    sensor_fields = [
        "sensor_uid", "sensor_name", "company", "assigned_uwwCode", "assigned_wwtw_name",
        "fitted_beta", "beta_ci_lower_95", "beta_ci_upper_95",
    ] + PRODUCED_COLUMNS + ["geometry"]
    sensor_layer = gpd.GeoDataFrame(sensor_frame[sensor_fields], geometry="geometry", crs=BNG)

    return {
        "matching_cso_sensors": sensor_layer,
        "assigned_treatment_works": point_layer,
        "wwtw_catchments_rainfall": catch_layer,
        "rainfall_grid_1km_clipped_to_catchments": grid,
    }


def write_geopackage(path: Path, layers: Mapping[str, gpd.GeoDataFrame]) -> None:
    if path.exists():
        path.unlink()
    for index, name in enumerate(OUTPUT_LAYERS):
        layers[name].to_file(path, layer=name, driver="GPKG", engine="pyogrio", append=index > 0)


def validate_geopackage(path: Path, expected: Mapping[str, gpd.GeoDataFrame]) -> dict[str, dict[str, object]]:
    names = pyogrio.list_layers(path)[:, 0].tolist()
    if names != OUTPUT_LAYERS:
        raise ReleaseBlocked(f"GeoPackage layer order/schema mismatch: {names}")
    audit: dict[str, dict[str, object]] = {}
    for name in OUTPUT_LAYERS:
        layer = gpd.read_file(path, layer=name, engine="pyogrio")
        if len(layer) != len(expected[name]):
            raise ReleaseBlocked(f"GeoPackage count mismatch in {name}")
        if layer.crs is None or layer.crs.to_epsg() != 27700:
            raise ReleaseBlocked(f"GeoPackage layer not EPSG:27700: {name}")
        nulls = int(layer.geometry.isna().sum())
        invalid = int((layer.geometry.notna() & ~layer.geometry.is_valid).sum())
        if nulls or invalid or layer.geometry.is_empty.any():
            raise ReleaseBlocked(f"GeoPackage geometry gate failed in {name}")
        audit[name] = {
            "count": len(layer),
            "geometry_type": ", ".join(sorted(layer.geom_type.unique())),
            "crs": layer.crs.to_string(),
            "null_geometry_count": nulls,
            "invalid_geometry_count": invalid,
        }
    if audit["assigned_treatment_works"]["count"] != audit["matching_cso_sensors"]["count"]:
        raise ReleaseBlocked("Published assigned and sensor counts differ")
    return audit


def variable_summary(summary: pd.DataFrame, column: str) -> dict[str, float | int]:
    values = pd.to_numeric(summary[column], errors="coerce")
    valid = values.dropna()
    return {
        "nonmissing": int(valid.size), "missing": int(values.isna().sum()),
        "minimum": float(valid.min()), "mean": float(valid.mean()),
        "median": float(valid.median()), "maximum": float(valid.max()),
    }


def write_report(
    path: Path,
    *,
    root: Path,
    master_path: Path,
    master_pre_hash: str,
    master_post_hash: str,
    old_master: pd.DataFrame,
    new_master: pd.DataFrame,
    annual: RasterProduct,
    winter: RasterProduct,
    source_variables: Mapping[str, list[str]],
    summary: pd.DataFrame,
    gpkg_path: Path,
    report_output_path: Path,
    layer_audit: Mapping[str, Mapping[str, object]],
    master_locked: bool,
) -> None:
    status_counts = summary["rain_extraction_status"].value_counts().to_dict()
    coverage = summary["rain_valid_coverage_pct"]
    annual_stats = variable_summary(summary, ANNUAL_COLUMN)
    winter_stats = variable_summary(summary, WINTER_COLUMN)
    def stats_line(label: str, stats: Mapping[str, object], unit: str) -> str:
        return (f"{label}: nonmissing={stats['nonmissing']}; missing={stats['missing']}; "
                f"min={stats['minimum']:.10g}; mean={stats['mean']:.10g}; "
                f"median={stats['median']:.10g}; max={stats['maximum']:.10g}; unit={unit}")
    layers = "\n".join(
        f"    {name}: {info['count']} features; {info['geometry_type']}; {info['crs']}; "
        f"null geometry={info['null_geometry_count']}; invalid geometry={info['invalid_geometry_count']}"
        for name, info in layer_audit.items()
    )
    text = f"""GOAL 05 RAINFALL PIPELINE AND SUPERVISOR EXPLANATION

A. Goal
Two official HadUK-Grid 1991-2020 rainfall climatologies were calculated over
each complete WWTW catchment and added to the authoritative one-row-per-WWTW
supervisor CSV: mean annual rainfall and mean meteorological-winter rainfall.

Simple workflow
Official 1 km annual and seasonal HadUK-Grid NetCDF files
        |
        v
Validate metadata, millimetre units, 1991-2020 period, EPSG:27700 and grid alignment
        |
        v
Intersect every 1 km cell exactly with each validated WWTW catchment
        |
        v
Weight cell rainfall by its actual overlap area, including partial boundary cells
        |
        v
Validate coverage, preserve all existing master values, and publish atomically
        |
        v
Updated supervisor CSV + four-layer QGIS GeoPackage + this audit TXT

B. Scientific reasoning
Annual rainfall represents long-term total water input. Winter rainfall
represents rainfall during the low-evapotranspiration and recharge season.
These variables may later be tested against the selected CSO sensor's existing
individual fitted_beta. This stage does not test association or claim causation.

C. Exact definitions
{ANNUAL_COLUMN}: area-weighted mean annual precipitation total across each WWTW
catchment, averaged over calendar years 1991-2020; millimetres per year.
{WINTER_COLUMN}: area-weighted official HadUK-Grid DJF seasonal climatology for
1991-2020; December + January + February total, expressed as millimetres per winter.
The seasonal time coordinates are January, April, July and October; band 1,
centred on January, is the official DJF winter band.

Deferred variables:
{DEFERRED_COLUMNS[0]}
{DEFERRED_COLUMNS[1]}
Neither annual nor seasonal source contains an official rain-day threshold
variable. No daily rainfall was downloaded or processed, and no wet-day value
was estimated from rainfall totals. These fields are deferred until official
precomputed threshold climatologies are supplied.

D. Data source
Release: {annual.tags.get('NC_GLOBAL#source')} / {annual.tags.get('NC_GLOBAL#version')}
Institution: {annual.tags.get('NC_GLOBAL#institution')}
Annual file: {annual.path.relative_to(root)}
Seasonal file: {winter.path.relative_to(root)}
Annual variables: {', '.join(source_variables['annual'])}
Seasonal variables: {', '.join(source_variables['seasonal'])}
Rainfall variable: rainfall; units: mm; source resolution: 1 km
Original and analysis CRS: EPSG:27700 (British National Grid)
No-data value: {annual.nodata}
Licence: Open Government Licence v3 (HadUK-Grid CEDA catalogue)
Citation: Met Office et al. (2026), HadUK-Grid v1.3.2.ceda,
doi:10.5285/789b3065d74a4c948ab05d33556c86d0; method doi:10.1002/gdj3.78.

E. Raw-data validation
Annual SHA-256: {sha256(annual.path)}
Seasonal SHA-256: {sha256(winter.path)}
Period in both files: 1991-2020
Annual frequency: {annual.tags.get('NC_GLOBAL#frequency')}
Seasonal frequency: {winter.tags.get('NC_GLOBAL#frequency')}
Grid dimensions: {annual.values.shape[1]} columns x {annual.values.shape[0]} rows
Grid transform: {annual.transform}
Units and no-data metadata passed. Annual and winter grids, cell centres,
resolution, extent, CRS and valid-data masks align exactly. Valid values were
finite and nonnegative. Daily-date, duplicate-date and leap-year checks are not
applicable because daily data were explicitly excluded from this release.
Files downloaded by this execution: none.

F. Spatial method
The same validated WWTW catchments used by the final assignment and
hydrogeology workflows were used, keyed by company + uwwCode. Areas were
calculated in EPSG:27700. Every intersecting 1 km cell was clipped to the
catchment boundary, so partial boundary cells contribute only their actual
overlap area. Point or centroid sampling was not used.

catchment rainfall =
    sum(cell rainfall x cell overlap area)
    / sum(valid cell overlap area)

Cell values are averaged rather than summed because rainfall depth is an
intensity/depth characteristic; otherwise larger catchments would appear wetter
solely because they contain more cells.

G. Catchment results
Authoritative WWTW count: {len(new_master)}
Catchments processed: {len(summary)}
Complete coverage: {status_counts.get('complete', 0)}
Minor no-data: {status_counts.get('minor_nodata', 0)}
Substantial no-data: {status_counts.get('substantial_nodata', 0)}
Failed/no-overlap extraction: {status_counts.get('failed_extraction', 0) + status_counts.get('no_raster_overlap', 0)}
Minimum valid coverage: {coverage.min():.10g}%
Mean valid coverage: {coverage.mean():.10g}%
Maximum valid coverage: {coverage.max():.10g}%

H. Variable summaries
{stats_line(ANNUAL_COLUMN, annual_stats, 'mm/year')}
{stats_line(WINTER_COLUMN, winter_stats, 'mm/winter')}

I. Master spreadsheet validation
Path: {master_path}
Pre-update SHA-256: {master_pre_hash}
Post-update SHA-256: {master_post_hash}
Input rows: {len(old_master)}; output rows: {len(new_master)}
Columns before: {len(old_master.columns)}; columns after: {len(new_master.columns)}
Key uniqueness: passed
Added columns: {', '.join(PRODUCED_COLUMNS)}
All pre-existing rows, values and row order were unchanged. Sensor UID,
assignment status/source, fitted_beta, beta confidence intervals, capacity and
hydrogeology fields were unchanged. Master locked at publication: {master_locked}.

J. GeoPackage
Path: {gpkg_path}
{layers}
Assigned-treatment-work and matching-sensor feature counts are equal.
All layers open, use EPSG:27700, and contain valid non-null geometries.

K. QGIS instructions
Load the GeoPackage through Layer > Add Layer > Add Vector Layer. Place layers
top-to-bottom as matching_cso_sensors, assigned_treatment_works,
wwtw_catchments_rainfall, rainfall_grid_1km_clipped_to_catchments. Style the
grid using a sequential blue graduated ramp on either
cell_rain_mean_annual_mm_1991_2020 or cell_rain_winter_mean_mm_1991_2020, with
thin/no outlines. Use transparent catchment fill with a dark outline, red WWTW
points and green sensor points. Select a catchment to inspect its exact clipped
cells, treatment works and selected sensor. Internal styles were not embedded,
so these instructions avoid external QML files.

L. Limitations
These are long-term 1991-2020 catchment climatologies. They do not describe
rainfall immediately before an individual spill and do not establish that
rainfall causes a change in beta. Annual and winter rainfall may be correlated;
later modelling must inspect multicollinearity. Wet-day frequency remains
deferred and must not be inferred from these totals.

M. Final output list
Updated master spreadsheet: {master_path}
Rainfall GeoPackage: {gpkg_path}
Rainfall pipeline supervisor TXT: {report_output_path}
"""
    path.write_text(text, encoding="utf-8")


def read_master(path: Path) -> tuple[bytes, pd.DataFrame]:
    try:
        raw = path.read_bytes()
    except PermissionError as exc:
        raise ReleaseBlocked(
            f"Authoritative master is locked: {path}. Close Excel/preview and retry; no competing master was created."
        ) from exc
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    return raw, frame


def publish_atomically(staged: Mapping[Path, Path], master_path: Path, original_master: bytes) -> None:
    """Replace validated outputs, rolling back the master if a later move fails."""
    replaced: list[Path] = []
    try:
        for temporary, final in staged.items():
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, final)
            replaced.append(final)
    except OSError as exc:
        if master_path in replaced:
            rollback = master_path.with_name(f".{master_path.name}.rollback.tmp")
            rollback.write_bytes(original_master)
            os.replace(rollback, master_path)
        raise ReleaseBlocked(f"Atomic publication failed, likely due to a locked output: {exc}") from exc


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    final_run = resolve_latest_run(root / FINAL_BRANCH)
    master_path = final_run / "tables" / MASTER_NAME
    source_gpkg = final_run / "qgis" / SOURCE_GPKG_NAME
    annual_path = root / ANNUAL_FILE
    seasonal_path = root / SEASONAL_FILE

    source_variables = {
        "annual": netcdf_variables(annual_path),
        "seasonal": netcdf_variables(seasonal_path),
    }
    if "rainfall" not in source_variables["annual"] or "rainfall" not in source_variables["seasonal"]:
        raise ReleaseBlocked("Required rainfall variable absent from annual or seasonal NetCDF")
    threshold_names = {"raindays1mm", "raindays10mm"}
    if threshold_names & (set(source_variables["annual"]) | set(source_variables["seasonal"])):
        raise ReleaseBlocked("Threshold variables need explicit metadata handling before they can be published")

    annual = load_raster_product(annual_path, "ann-30y", 1)
    winter = load_raster_product(seasonal_path, "seas-30y", 1)
    seasonal_band_dates(winter)
    validate_grid_alignment(annual, winter)
    catchments, assigned, sensors = load_spatial_sources(source_gpkg)
    grid, summary = intersect_grid_with_catchments(catchments, annual, winter)
    validate_summary(summary)

    # The master is read only after the expensive calculations; a lock therefore
    # blocks publication but never triggers a forced write or a competing master.
    original_master, old_master = read_master(master_path)
    master_pre_hash = hashlib.sha256(original_master).hexdigest()
    new_master = build_master_candidate(old_master, summary)
    validate_master_candidate(old_master, new_master)
    layers = build_output_layers(catchments, assigned, sensors, grid, summary)

    if args.validate_only:
        print(f"Validated sources and calculated {len(summary)} catchments; no outputs written")
        return 0

    goal_dir = root / GOAL_DIR
    qgis_dir = goal_dir / "qgis"
    qgis_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="goal05_rainfall_", dir=root) as temp_name:
        temp = Path(temp_name)
        staged_master = temp / MASTER_NAME
        staged_gpkg = temp / OUTPUT_GPKG_NAME
        staged_txt = temp / OUTPUT_TXT_NAME
        new_master.to_csv(staged_master, index=False, lineterminator="\n", float_format="%.10g")
        readback = pd.read_csv(staged_master, dtype="string", keep_default_na=False)
        validate_master_candidate(old_master, readback)
        master_post_hash = sha256(staged_master)
        write_geopackage(staged_gpkg, layers)
        layer_audit = validate_geopackage(staged_gpkg, layers)
        final_gpkg = qgis_dir / OUTPUT_GPKG_NAME
        final_txt = goal_dir / OUTPUT_TXT_NAME
        write_report(
            staged_txt,
            root=root,
            master_path=master_path,
            master_pre_hash=master_pre_hash,
            master_post_hash=master_post_hash,
            old_master=old_master,
            new_master=readback,
            annual=annual,
            winter=winter,
            source_variables=source_variables,
            summary=summary,
            gpkg_path=final_gpkg,
            report_output_path=final_txt,
            layer_audit=layer_audit,
            master_locked=False,
        )
        staged = {
            staged_gpkg: final_gpkg,
            staged_txt: final_txt,
            staged_master: master_path,
        }
        publish_atomically(staged, master_path, original_master)

    installed = pd.read_csv(master_path, dtype="string", keep_default_na=False)
    validate_master_candidate(old_master, installed)
    final_layers = validate_geopackage(goal_dir / "qgis" / OUTPUT_GPKG_NAME, layers)
    if sha256(master_path) != master_post_hash:
        raise ReleaseBlocked("Installed master hash differs from staged validated master")
    print(f"Published Goal 05 at {datetime.now(timezone.utc).isoformat()}")
    print(f"Master: {master_path}")
    print(f"Catchments: {len(summary)}; grid intersections: {len(grid)}")
    print(f"Layers: { {name: info['count'] for name, info in final_layers.items()} }")
    print(f"Added columns: {PRODUCED_COLUMNS}; deferred: {DEFERRED_COLUMNS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

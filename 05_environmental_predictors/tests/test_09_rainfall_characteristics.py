"""Focused scientific contracts for the Goal 05 rainfall release."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from affine import Affine
from shapely.geometry import box


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_static_rainfall.py"
SPEC = importlib.util.spec_from_file_location("rainfall_release", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_candidate_cells_and_cell_polygon_use_exact_one_km_grid() -> None:
    transform = Affine(1000, 0, 0, 0, -1000, 2000)
    geometry = box(500, 500, 1500, 1500)
    assert module.candidate_cells(geometry, transform, 2, 2) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert module.cell_polygon(0, 0, transform).equals(box(0, 1000, 1000, 2000))


def test_master_candidate_adds_only_two_columns_and_preserves_old_values() -> None:
    old = pd.DataFrame({
        "company": pd.Series(["a", "b"], dtype="string"),
        "uwwCode": pd.Series(["1", "2"], dtype="string"),
        "sensor_uid": pd.Series(["s1", "s2"], dtype="string"),
        "fitted_beta": pd.Series(["1.1", "2.2"], dtype="string"),
        "hydro_extraction_status": pd.Series(["complete", "complete"], dtype="string"),
    })
    summary = pd.DataFrame({
        "company": ["b", "a"], "uwwCode": ["2", "1"],
        module.ANNUAL_COLUMN: [900.0, 800.0],
        module.WINTER_COLUMN: [240.0, 200.0],
    })
    new = module.build_master_candidate(old, summary)
    module.validate_master_candidate(old, new)
    assert new[module.ANNUAL_COLUMN].tolist() == [800.0, 900.0]
    assert new[module.WINTER_COLUMN].tolist() == [200.0, 240.0]
    assert new["sensor_uid"].equals(old["sensor_uid"])
    assert new["fitted_beta"].equals(old["fitted_beta"])


def test_master_candidate_is_idempotent_for_existing_rainfall_block() -> None:
    old = pd.DataFrame({
        "company": pd.Series(["a"], dtype="string"),
        "uwwCode": pd.Series(["1"], dtype="string"),
        "hydro_extraction_status": pd.Series(["complete"], dtype="string"),
        module.ANNUAL_COLUMN: [700.0],
        module.WINTER_COLUMN: [180.0],
    })
    summary = pd.DataFrame({
        "company": ["a"], "uwwCode": ["1"],
        module.ANNUAL_COLUMN: [700.0], module.WINTER_COLUMN: [180.0],
    })
    new = module.build_master_candidate(old, summary)
    module.validate_master_candidate(old, new)
    assert list(new.columns).count(module.ANNUAL_COLUMN) == 1
    assert list(new.columns).count(module.WINTER_COLUMN) == 1


def test_invalid_or_negative_rainfall_is_rejected_by_mask() -> None:
    product = module.RasterProduct(
        path=Path("x.nc"), values=np.array([[1.0, -1.0, 1e20, np.nan]]),
        transform=Affine.identity(), crs=module.CRS.from_epsg(27700), nodata=1e20,
        tags={}, band=1, variable_name="rainfall",
    )
    assert module.valid_mask(product).tolist() == [[True, False, False, False]]


def test_wet_day_columns_are_explicitly_deferred() -> None:
    assert module.DEFERRED_COLUMNS == [
        "rain_wet_days_ge_1mm_per_year_1991_2020",
        "rain_wet_days_ge_10mm_per_year_1991_2020",
    ]
    assert all(column not in module.PRODUCED_COLUMNS for column in module.DEFERRED_COLUMNS)

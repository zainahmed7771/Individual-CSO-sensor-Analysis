#!/usr/bin/env python3
"""Prepare auditable Thames inputs from the supplied company JSON.

The ``events`` command maps the JSON event export to the repository's existing
five-column clean-event contract without altering measurements.  The
``coordinates`` command builds the coordinate-audit rows required by the
sensor-master stage after the standard bootstrap has been run for Thames, and
appends them to the unchanged eight-company coordinate audit in a new file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


JSON_PATH = Path("json_data/thames.json")
EVENT_OUT = Path("clean_data/thames/thames_json_standardized_clean_data.csv")
BOOTSTRAP_CSV = Path(
    "outputs/individual_cso_sensors_thames_stage/gt_240min/"
    "thames/02_sensors_by_tail_spills_desc.csv"
)
BASE_COORDINATES = Path(
    "outputs/all_individual_cso_beta_maps/gt_240min/"
    "tables/joined_all_sensor_beta_locations.csv"
)
COORDINATE_OUT = Path(
    "data_intermediate/thames_extension/"
    "joined_all_sensor_beta_locations_thames_inclusive.csv"
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("events", "coordinates"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(raw)
    required = {
        "LocationName", "PermitNumber", "X", "Y", "ReceivingWaterCourse",
        "StartDateTime", "StopDateTime", "Duration", "OngoingEvent",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Thames JSON is missing fields: {missing}")
    if frame.empty:
        raise ValueError("Thames JSON has no event rows")
    return frame


def prepare_events(root: Path, overwrite: bool) -> None:
    source = root / JSON_PATH
    output = root / EVENT_OUT
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output}")
    raw = load_json(source)
    duration = pd.to_numeric(raw["Duration"], errors="coerce")
    permit = raw["PermitNumber"].astype("string").str.strip()
    if permit.isna().any() or permit.eq("").any():
        raise ValueError("Thames JSON has blank PermitNumber values")
    if duration.isna().any() or not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Thames JSON has invalid Duration values")
    events = pd.DataFrame({
        "location_name": raw["LocationName"].astype("string").str.strip(),
        "permit_number": permit,
        "start_time": pd.to_datetime(raw["StartDateTime"], unit="ms", utc=True, errors="coerce"),
        "stop_time": pd.to_datetime(raw["StopDateTime"], unit="ms", utc=True, errors="coerce"),
        "duration_minutes": duration.astype(float),
    })
    if events[["start_time", "stop_time"]].isna().any().any():
        raise ValueError("Thames JSON has invalid event timestamps")
    output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output, index=False, lineterminator="\n")
    check = pd.read_csv(output, dtype={"permit_number": "string"})
    if len(check) != len(events) or list(check.columns) != list(events.columns):
        raise AssertionError("Standardized Thames event CSV failed read-back validation")
    print(f"Created {output}: {len(events)} events, {events.permit_number.nunique()} sensors")


def modal_text(values: pd.Series) -> str:
    usable = values.astype("string").dropna().str.strip()
    usable = usable[usable.ne("")]
    if usable.empty:
        return ""
    counts = usable.value_counts()
    return sorted(counts[counts.eq(counts.max())].index.astype(str))[0]


def prepare_coordinates(root: Path, overwrite: bool) -> None:
    output = root / COORDINATE_OUT
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output}")
    raw = load_json(root / JSON_PATH)
    beta = pd.read_csv(root / BOOTSTRAP_CSV, low_memory=False, dtype={"permit_number": "string"})
    base = pd.read_csv(root / BASE_COORDINATES, low_memory=False)
    raw["PermitNumber"] = raw.PermitNumber.astype("string").str.strip()
    raw["X"] = pd.to_numeric(raw.X, errors="coerce")
    raw["Y"] = pd.to_numeric(raw.Y, errors="coerce")
    pairs = raw.groupby("PermitNumber").apply(
        lambda group: len(set(zip(group.X, group.Y))), include_groups=False
    ).rename("number_of_distinct_coordinate_pairs")
    grouped = raw.groupby("PermitNumber", sort=True)
    locations = grouped.agg(
        json_location_name=("LocationName", modal_text),
        json_event_row_count=("PermitNumber", "size"),
        bng_easting=("X", "first"), bng_northing=("Y", "first"),
        receiving_watercourse_display=("ReceivingWaterCourse", modal_text),
    ).reset_index().rename(columns={"PermitNumber": "permit_number_csv_original"})
    locations = locations.merge(pairs, left_on="permit_number_csv_original", right_index=True, validate="one_to_one")
    if beta.permit_number.duplicated().any():
        raise ValueError("Thames bootstrap permit_number is not unique")
    if set(beta.permit_number.astype(str)) != set(locations.permit_number_csv_original.astype(str)):
        raise ValueError("Thames JSON and bootstrap sensor permit sets differ")
    joined = locations.merge(
        beta, left_on="permit_number_csv_original", right_on="permit_number",
        how="left", validate="one_to_one", suffixes=("", "_beta"),
    )
    transform = Transformer.from_crs(27700, 4326, always_xy=True)
    longitude, latitude = transform.transform(joined.bng_easting.to_numpy(), joined.bng_northing.to_numpy())
    coordinate_conflict = joined.number_of_distinct_coordinate_pairs.gt(1)
    coordinate_valid = (
        joined.bng_easting.between(0, 700_000, inclusive="neither")
        & joined.bng_northing.between(0, 1_300_000, inclusive="neither")
        & ~coordinate_conflict
    )
    interval = (
        pd.to_numeric(joined.beta_ci_lower_95, errors="coerce").notna()
        & pd.to_numeric(joined.beta_ci_upper_95, errors="coerce").notna()
        & pd.to_numeric(joined.beta_ci_width, errors="coerce").lt(0.5)
    )
    strict = (
        joined.original_fit_success.fillna(False).astype(bool)
        & joined.bootstrap_attempted.fillna(False).astype(bool)
        & pd.to_numeric(joined.bootstrap_success_rate, errors="coerce").ge(0.95)
        & pd.to_numeric(joined.beta_bound_hit_rate, errors="coerce").le(0.05)
        & interval
    )
    rows = pd.DataFrame(columns=base.columns, index=joined.index)
    rows["company"] = "thames"
    rows["json_filename"] = "thames.json"
    rows["permit_number_json_original"] = joined.permit_number_csv_original
    rows["permit_number_csv_original"] = joined.permit_number_csv_original
    rows["permit_match_status"] = "exact"
    rows["permit_match_method"] = "exact"
    rows["json_location_name"] = joined.json_location_name
    rows["canonical_location_name"] = joined.canonical_location_name
    rows["json_event_row_count"] = joined.json_event_row_count
    rows["original_bng_easting"] = joined.bng_easting
    rows["original_bng_northing"] = joined.bng_northing
    rows["bng_easting"] = joined.bng_easting.where(coordinate_valid)
    rows["bng_northing"] = joined.bng_northing.where(coordinate_valid)
    rows["longitude"] = pd.Series(longitude, index=joined.index).where(coordinate_valid)
    rows["latitude"] = pd.Series(latitude, index=joined.index).where(coordinate_valid)
    rows["number_of_distinct_coordinate_pairs"] = joined.number_of_distinct_coordinate_pairs
    rows["coordinate_conflict"] = coordinate_conflict
    rows["coordinate_valid"] = coordinate_valid
    rows["coordinate_exclusion_reason"] = np.where(coordinate_conflict, "json_coordinate_conflict", "")
    rows["coordinate_correction"] = ""
    rows["strict_map_quality_eligible"] = strict
    rows["strict_quality_exclusion_reason"] = np.where(strict, "", "strict_beta_quality_failed")
    for column in base.columns:
        if column in joined.columns and rows[column].isna().all():
            rows[column] = joined[column]
    combined = pd.concat([base, rows], ignore_index=True, sort=False)[base.columns]
    sensor_side = combined.loc[
        combined["permit_number_csv_original"].notna()
        & combined["permit_number_csv_original"].astype("string").str.strip().ne("")
    ]
    if sensor_side.duplicated(["company", "permit_number_csv_original"]).any():
        raise AssertionError("Combined coordinate audit has duplicate sensor keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, lineterminator="\n")
    check = pd.read_csv(output, low_memory=False)
    if len(check) != len(combined):
        raise AssertionError("Combined coordinate audit failed read-back validation")
    print(f"Created {output}: {len(combined)} rows ({len(rows)} Thames)")


def main() -> None:
    args = arguments()
    root = args.project_root.resolve()
    if args.mode == "events":
        prepare_events(root, args.overwrite)
    else:
        prepare_coordinates(root, args.overwrite)


if __name__ == "__main__":
    main()

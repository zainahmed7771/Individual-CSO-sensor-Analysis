from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "03_tail_parameter_fitting" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from analyse_individual_cso_heavy_tails import (
    FileQuality,
    canonical_locations,
    cleaned_chunks,
    file_sha256,
    read_company,
    sensor_statistics,
)


def sensor_events(durations: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company": ["test_company"] * len(durations),
            "permit_number": ["P1"] * len(durations),
            "location_name": ["Test location"] * len(durations),
            "start_time": pd.to_datetime(["2026-01-01"] * len(durations)),
            "stop_time": pd.to_datetime(["2026-01-01 01:00"] * len(durations)),
            "duration_minutes": durations,
            "source_file": ["fixture.csv"] * len(durations),
            "source_year": ["2026"] * len(durations),
        }
    )


def write_events(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "location_name": "Fixture location",
        "permit_number": "P1",
        "start_time": "2026-01-01 00:00:00",
        "stop_time": "2026-01-01 01:00:00",
        "duration_minutes": 60,
    }
    row.update(overrides)
    return row


def clean(path: Path) -> tuple[pd.DataFrame, FileQuality]:
    quality = FileQuality("test_company", str(path))
    chunks = list(cleaned_chunks(path, "test_company", False, quality))
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return frame, quality


def test_duration_equal_to_threshold_is_excluded() -> None:
    stats = sensor_statistics(sensor_events([240.0]), 240.0)
    assert int(stats.loc[0, "tail_spill_count"]) == 0


def test_duration_just_above_threshold_is_included() -> None:
    stats = sensor_statistics(sensor_events([240.000001]), 240.0)
    assert int(stats.loc[0, "tail_spill_count"]) == 1


def test_missing_permit_identifier_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing_permit.csv"
    write_events(path, [base_row(permit_number="")])
    frame, quality = clean(path)
    assert frame.empty
    assert quality.missing_permit_number == 1
    assert quality.rejected_rows == 1


def test_nonpositive_durations_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nonpositive.csv"
    write_events(path, [base_row(duration_minutes=0), base_row(duration_minutes=-1)])
    frame, quality = clean(path)
    assert frame.empty
    assert quality.non_positive_duration == 2
    assert quality.rejected_rows == 2


def test_nonnumeric_duration_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nonnumeric.csv"
    write_events(path, [base_row(duration_minutes="not-a-number")])
    frame, quality = clean(path)
    assert frame.empty
    assert quality.non_numeric_duration == 1
    assert quality.rejected_rows == 1


def test_invalid_timestamp_is_counted_but_valid_event_remains(tmp_path: Path) -> None:
    path = tmp_path / "invalid_time.csv"
    write_events(path, [base_row(start_time="invalid timestamp")])
    frame, quality = clean(path)
    assert len(frame) == 1
    assert pd.isna(frame.loc[0, "start_time"])
    assert quality.invalid_start_time == 1
    assert quality.valid_rows == 1
    assert quality.rejected_rows == 0


def test_duplicate_rows_count_all_participants(tmp_path: Path) -> None:
    path = tmp_path / "duplicate_2026.csv"
    row = base_row()
    write_events(path, [row, row])
    events, qualities = read_company("test_company", [path], False)
    assert len(events) == 2
    assert qualities[0].duplicate_rows == 2


def test_canonical_location_tie_breaking_is_alphabetical() -> None:
    events = pd.DataFrame(
        {
            "permit_number": ["P1", "P1", "P1", "P1"],
            "location_name": ["Zulu", "Alpha", "Zulu", "Alpha"],
        }
    )
    canonical, inconsistent = canonical_locations(events)
    assert canonical.loc["P1"] == "Alpha"
    assert set(inconsistent["location_name"]) == {"Alpha", "Zulu"}


def test_source_hash_is_unchanged_by_read_only_cleaning(tmp_path: Path) -> None:
    path = tmp_path / "readonly.csv"
    write_events(path, [base_row()])
    before = file_sha256(path)
    frame, quality = clean(path)
    after = file_sha256(path)
    assert len(frame) == 1
    assert quality.status == "processed"
    assert before == after


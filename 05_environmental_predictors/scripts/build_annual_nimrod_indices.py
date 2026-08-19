"""Build daily catchment rainfall and annual ETCCDI-style indices.

The validated 5-minute decoder and exact EPSG:27700 spatial weights are reused.
This stage starts from its complete-hour NPZ partitions; it never touches raw TARs.
"""

from __future__ import annotations

import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
RELEASE = Path(__file__).resolve().parents[2]
DAILY_SOURCE = ROOT / "dynamic_rainfall/intermediate/catchment_timeseries"
ORDER = ROOT / "dynamic_rainfall/intermediate/spatial_weights/catchment_order.csv"
INTERVALS = ROOT / "dynamic_rainfall/intermediate/processing_state/rainfall_monitoring_intervals.csv"
OUT = RELEASE / "rainfall"
DAILY_OUT = OUT / "daily_catchment_rainfall"

YEAR_DAY_COVERAGE_MIN = 0.95
DAY_COMPLETE_HOURS = 24
WET_DAY_MM = 1.0


def max_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def annual_indices(daily_mm: np.ndarray, complete: np.ndarray) -> dict[str, float]:
    # Missing days break rolling/spell sequences; they are never interpreted as dry.
    valid_values = daily_mm[complete]
    wet = valid_values >= WET_DAY_MM
    prcptot = float(valid_values[wet].sum())
    wet_n = int(wet.sum())
    rx1 = float(valid_values.max())
    r10 = int((valid_values >= 10.0).sum())
    r20 = int((valid_values >= 20.0).sum())
    sdii = prcptot / wet_n if wet_n else np.nan
    rx5 = np.nan
    if len(daily_mm) >= 5:
        vals = np.where(complete, daily_mm, 0.0)
        sums = np.convolve(vals, np.ones(5), mode="valid")
        windows_ok = np.convolve(complete.astype(int), np.ones(5, dtype=int), mode="valid") == 5
        if windows_ok.any():
            rx5 = float(sums[windows_ok].max())
    # Splitting at missing days avoids connecting spells across gaps.
    cwd = cdd = 0
    start = 0
    while start < len(complete):
        while start < len(complete) and not complete[start]: start += 1
        end = start
        while end < len(complete) and complete[end]: end += 1
        if end > start:
            segment = daily_mm[start:end]
            cwd = max(cwd, max_run(segment >= WET_DAY_MM))
            cdd = max(cdd, max_run(segment < WET_DAY_MM))
        start = end + 1
    return {
        "rain_ann_prcptot_mm": prcptot,
        "rain_ann_rx1day_mm": rx1,
        "rain_ann_rx5day_mm": rx5,
        "rain_ann_sdii_mm_per_wet_day": float(sdii),
        "rain_ann_r10_days": r10,
        "rain_ann_r20_days": r20,
        "rain_ann_cwd_days": int(cwd),
        "rain_ann_cdd_days": int(cdd),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); DAILY_OUT.mkdir(parents=True, exist_ok=True)
    order = pd.read_csv(ORDER).sort_values("catchment_index").reset_index(drop=True)
    intervals = pd.read_csv(INTERVALS)
    intervals["overlap_start"] = pd.to_datetime(intervals.overlap_start, utc=True, format="mixed")
    intervals["overlap_end"] = pd.to_datetime(intervals.overlap_end, utc=True, format="mixed")
    interval_lookup = {(r.sensor_uid, int(r.active_year)): r for r in intervals.itertuples()}
    annual_rows, daily_inventory = [], []

    for year in range(2021, 2026):
        days = 366 if calendar.isleap(year) else 365
        daily = np.full((days, len(order)), np.nan, dtype=np.float32)
        complete = np.zeros((days, len(order)), dtype=bool)
        dates = pd.date_range(f"{year}-01-01", periods=days, freq="D", tz="UTC")
        for day_index, date in enumerate(dates):
            path = DAILY_SOURCE / str(year) / f"{date:%Y%m%d}.npz"
            if not path.exists():
                continue
            with np.load(path) as z:
                hourly = z["rainfall_hourly_mm"]
                counts = z["valid_5min_count_by_hour"]
                source_counts = z["source_5min_count_by_hour"]
                ok_hour = (counts == 12) & (source_counts[:, None] == 12) & np.isfinite(hourly)
                ok_day = ok_hour.all(axis=0) & (len(hourly) == DAY_COMPLETE_HOURS)
                daily[day_index, ok_day] = hourly[:, ok_day].sum(axis=0)
                complete[day_index] = ok_day
        np.savez_compressed(DAILY_OUT / f"catchment_daily_rainfall_{year}.npz",
                            dates_ns=dates.asi8, rainfall_mm=daily,
                            complete_24h=complete, catchment_index=order.catchment_index.to_numpy())
        daily_inventory.append({"year": year, "calendar_days": days,
                                "median_complete_days_all_catchments": float(np.median(complete.sum(axis=0))),
                                "minimum_complete_days_all_catchments": int(complete.sum(axis=0).min()),
                                "maximum_complete_days_all_catchments": int(complete.sum(axis=0).max())})

        for r in order[order.sensor_uid.notna()].itertuples():
            key = (r.sensor_uid, year)
            if key not in interval_lookup:
                continue
            interval = interval_lookup[key]
            year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            year_end = pd.Timestamp(f"{year}-12-31 23:55:00", tz="UTC")
            full_monitoring_year = interval.overlap_start <= year_start and interval.overlap_end >= year_end
            c = complete[:, r.catchment_index]
            coverage = float(c.mean())
            eligible = bool(full_monitoring_year and coverage >= YEAR_DAY_COVERAGE_MIN)
            row = {"company": r.company, "uwwCode": r.uwwCode, "canonical_wwtw_name": r.canonical_wwtw_name,
                   "sensor_uid": r.sensor_uid, "year": year, "calendar_days": days,
                   "complete_days": int(c.sum()), "day_coverage_fraction": coverage,
                   "full_monitoring_year": full_monitoring_year, "annual_index_eligible": eligible,
                   "quality_flag": "valid_complete_year" if eligible else ("incomplete_monitoring_year" if not full_monitoring_year else "insufficient_radar_coverage")}
            if eligible:
                row.update(annual_indices(daily[:, r.catchment_index].astype(float), c))
            else:
                row.update({k: np.nan for k in annual_indices(np.ones(days), np.ones(days, dtype=bool))})
            annual_rows.append(row)

    annual = pd.DataFrame(annual_rows).sort_values(["company", "uwwCode", "year"])
    annual.to_csv(OUT / "annual_rainfall_indices_long.csv", index=False)
    pd.DataFrame(daily_inventory).to_csv(OUT / "daily_rainfall_inventory.csv", index=False)
    indices = [c for c in annual if c.startswith("rain_ann_") and c not in {"rain_ann_quality_flag"}]
    valid = annual[annual.annual_index_eligible].copy()
    agg = valid.groupby(["company", "uwwCode", "canonical_wwtw_name", "sensor_uid"], as_index=False)[indices].mean()
    years = valid.groupby(["company", "uwwCode", "sensor_uid"], as_index=False).agg(
        rain_ann_n_valid_years=("year", "size"), rain_ann_first_year=("year", "min"), rain_ann_last_year=("year", "max"),
        rain_ann_coverage_fraction=("day_coverage_fraction", "mean"))
    collapsed = agg.merge(years, on=["company", "uwwCode", "sensor_uid"], validate="one_to_one")
    collapsed["rain_ann_quality_flag"] = np.select(
        [collapsed.rain_ann_n_valid_years.ge(3), collapsed.rain_ann_n_valid_years.ge(2)],
        ["strong_3plus_complete_years", "usable_2_complete_years"], default="limited_1_complete_year")
    collapsed.to_csv(OUT / "annual_rainfall_indices_master.csv", index=False)
    dictionary = pd.DataFrame([
        ["rain_ann_prcptot_mm", "PRCPTOT", "sum of daily rainfall on days >=1 mm", "mm/year", "wet-weather volume"],
        ["rain_ann_rx1day_mm", "Rx1day", "maximum complete daily rainfall", "mm", "single-day storm burden"],
        ["rain_ann_rx5day_mm", "Rx5day", "maximum consecutive five-complete-day rainfall", "mm", "multi-day wetness"],
        ["rain_ann_sdii_mm_per_wet_day", "SDII", "PRCPTOT / number of days >=1 mm", "mm/wet day", "mean wet-day intensity"],
        ["rain_ann_r10_days", "R10mm", "count of days >=10 mm", "days/year", "heavy-rain frequency"],
        ["rain_ann_r20_days", "R20mm", "count of days >=20 mm", "days/year", "very-heavy-rain frequency"],
        ["rain_ann_cwd_days", "CWD", "maximum consecutive days >=1 mm", "days", "wet-spell persistence"],
        ["rain_ann_cdd_days", "CDD", "maximum consecutive days <1 mm", "days", "dry-spell context"],
    ], columns=["variable", "ETCCDI_name", "definition", "units", "CSO_interpretation"])
    dictionary["source"] = "Area-averaged NIMROD; ETCCDI-style definition adapted from HadEX3/Climpact"
    dictionary.to_csv(OUT / "annual_rainfall_data_dictionary.csv", index=False)
    audit = {
        "source_units_verified": "NIMROD integer rate (mm/hour)*32 converted by 0.03125 and integrated at 5/60 hour upstream; reused complete-hour sums",
        "spatial_method": "exact EPSG:27700 polygon-cell overlap weights from validated dynamic rainfall pipeline",
        "calendar_year_qc_frozen_before_outcome_analysis": {"full_sensor_monitoring_year_required": True, "minimum_complete_day_fraction": YEAR_DAY_COVERAGE_MIN, "complete_day_hours": 24},
        "missing_data": "not imputed; missing days break Rx5/CWD/CDD sequences",
        "wet_day_threshold_mm": 1.0, "R10_threshold_mm": 10.0, "R20_threshold_mm": 20.0,
        "R95_R99_status": "deferred - common climatological reference-period requirement not satisfied by 2021-2025 archive",
        "annual_rows": len(annual), "valid_annual_rows": int(annual.annual_index_eligible.sum()),
        "WWTWs_with_one_or_more_valid_years": len(collapsed),
    }
    (OUT / "annual_rainfall_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

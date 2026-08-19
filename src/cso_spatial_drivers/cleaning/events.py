from __future__ import annotations
import pandas as pd

REQUIRED = ["company", "permit_number", "location_name", "start_time", "stop_time"]

def clean_events(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in frame]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    x = frame.copy()
    x["company"] = x.company.astype("string").str.strip().str.lower()
    x["permit_number"] = x.permit_number.astype("string").str.strip()
    x["start_time"] = pd.to_datetime(x.start_time, errors="coerce", utc=True)
    x["stop_time"] = pd.to_datetime(x.stop_time, errors="coerce", utc=True)
    x["duration_minutes"] = (x.stop_time - x.start_time).dt.total_seconds() / 60.0
    good = x.start_time.notna() & x.stop_time.notna() & x.stop_time.ge(x.start_time)
    x = x.loc[good].drop_duplicates(["company", "permit_number", "start_time", "stop_time"]).copy()
    x["sensor_uid"] = x.company + "::" + x.permit_number
    return x.sort_values(["sensor_uid", "start_time"]).reset_index(drop=True)

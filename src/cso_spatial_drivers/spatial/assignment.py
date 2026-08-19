from __future__ import annotations
import numpy as np
import pandas as pd

def nearest_same_company(sensors: pd.DataFrame, works: pd.DataFrame, max_distance_m: float = 5000) -> pd.DataFrame:
    rows=[]
    for _, s in sensors.iterrows():
        candidates=works.loc[works.company.eq(s.company)].copy()
        if candidates.empty: continue
        d=np.hypot(candidates.easting-s.easting, candidates.northing-s.northing)
        i=d.idxmin(); distance=float(d.loc[i])
        if distance <= max_distance_m:
            rows.append({"sensor_uid":s.sensor_uid,"company":s.company,"uwwCode":works.loc[i,"uwwCode"],"distance_m":distance})
    return pd.DataFrame(rows)

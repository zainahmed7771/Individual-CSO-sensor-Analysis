from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cso_spatial_drivers.cleaning.events import clean_events
from cso_spatial_drivers.fitting.stretched_exponential import fit_tail
from cso_spatial_drivers.statistics.univariate import fit_log_outcome
from cso_spatial_drivers.ml.demo import fit_locked_demo

events=clean_events(pd.read_csv(ROOT/"data/sample/synthetic_events.csv"))
fits=[]
for sensor,g in events.groupby("sensor_uid"):
    result=fit_tail(g.duration_minutes); company,permit=sensor.split("::",1); i=int(permit[1:]); result.update({"sensor_uid":sensor,"company":company,"uwwCode":f"W{i:03d}","mean_duration":g.duration_minutes.mean()}); fits.append(result)
master=pd.DataFrame(fits).merge(pd.read_csv(ROOT/"data/sample/synthetic_predictors.csv"),on=["company","uwwCode"],validate="one_to_one")
order=np.arange(len(master)); train=order[:12]; val=order[12:15]; test=order[15:]
X=master[["population_density_km2","rain_mean_annual_mm","catchment_area_km2"]].to_numpy(); y=np.log(master.beta.to_numpy())
ml=fit_locked_demo(X[train],y[train],X[val],y[val],X[test],y[test])
uni=fit_log_outcome(master.beta,master.population_density_km2)
out=ROOT/"outputs/demo"; out.mkdir(parents=True,exist_ok=True); events.to_csv(out/"clean_events.csv",index=False); master.to_csv(out/"demo_master.csv",index=False); (out/"metrics.json").write_text(json.dumps({"label":"DEMONSTRATION ONLY - NOT THE SCIENTIFIC RESULTS","univariate":uni,"ml":{k:v for k,v in ml.items() if k!="predictions"}},indent=2),encoding="utf-8")
print("DEMONSTRATION ONLY - NOT THE SCIENTIFIC RESULTS"); print(json.dumps({k:v for k,v in ml.items() if k!="predictions"},indent=2))

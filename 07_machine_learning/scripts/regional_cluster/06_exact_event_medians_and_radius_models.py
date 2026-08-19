"""Finalize exact event medians and 10/25/50 km model sensitivity results."""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]; ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(REPO/"PROJECT_RESULTS"/"machine_learning"/"_runtime"))
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

OUT=ROOT/"01_regional_cluster_model"; SRC=REPO/"machine learning"/"CLUSTERED WWTW MACHINE LEARNING"/"shared"
ASSIGN=SRC/"sensor_to_wwtw_cluster_assignments.csv"; EVENT_ROOT=REPO/"clean_data"
MAP=OUT/"matching"/"wwtw_to_primary_region.csv"; CATCH=REPO/"PROJECT_RESULTS"/"master"/"master_environmental_analysis.gpkg"
FEATURES=["hydro_high_productivity_pct","hydro_low_productivity_pct","hydro_no_groundwater_pct","hydro_intergranular_flow_pct","rain_mean_annual_mm_1991_2020","rain_winter_mean_mm_1991_2020","continuous_land_cover_pct","population_density_km2_2021","building_pre_1973_pct","terrain_mean_slope_deg","regional_capacity_ratio","total_catchment_area_km2","rain_ann_prcptot_mm","rain_ann_rx1day_mm","rain_ann_rx5day_mm","rain_ann_sdii_mm_per_wet_day","rain_ann_r10_days","rain_ann_r20_days","rain_ann_cwd_days","rain_ann_cdd_days"]
TARGETS={"beta":("regional_beta_mean","log"),"lambda":("regional_lambda_mean","log"),"duration":("regional_event_mean_duration_minutes","log1p"),"frequency":("regional_events_per_sensor_year","log1p")}
SEED=20260819

def exact_medians() -> pd.DataFrame:
    a=pd.read_csv(ASSIGN,low_memory=False); mp=pd.read_csv(MAP).rename(columns={"uwwCode":"assigned_uwwCode"}); m=a.loc[a.assigned_uwwCode.notna(),["company","permit_number","assigned_uwwCode"]].merge(mp,on=["company","assigned_uwwCode"],how="inner",validate="many_to_one").drop_duplicates(["company","permit_number"])
    rows=[]; audit=[]
    for company,g in m.groupby("company"):
        permit_map=dict(zip(g.permit_number.astype(str).str.strip(),g.region_id)); pieces=[]; read=valid=dups=0
        for path in sorted((EVENT_ROOT/company).glob("*.csv")):
            for ch in pd.read_csv(path,usecols=["permit_number","start_time","stop_time","duration_minutes"],dtype={"permit_number":"string"},chunksize=250000,low_memory=False):
                read+=len(ch); ch["permit_number"]=ch.permit_number.astype("string").str.strip(); ch=ch.loc[ch.permit_number.isin(permit_map)].copy()
                if ch.empty: continue
                ch["start_time"]=pd.to_datetime(ch.start_time,errors="coerce",utc=True); ch["stop_time"]=pd.to_datetime(ch.stop_time,errors="coerce",utc=True); ch["duration_minutes"]=pd.to_numeric(ch.duration_minutes,errors="coerce")
                good=ch.start_time.notna()&ch.stop_time.notna()&np.isfinite(ch.duration_minutes)&ch.duration_minutes.ge(0)&ch.stop_time.ge(ch.start_time); pieces.append(ch.loc[good])
        if not pieces: continue
        ev=pd.concat(pieces,ignore_index=True); before=len(ev); ev=ev.drop_duplicates(["permit_number","start_time","stop_time"]); dups=before-len(ev); ev["region_id"]=ev.permit_number.map(permit_map); valid=len(ev)
        q=ev.groupby("region_id").duration_minutes.agg(exact_regional_event_count="size",exact_regional_event_total_duration_minutes="sum",regional_event_median_duration_minutes_exact="median").reset_index(); rows.append(q); audit.append({"company":company,"raw_rows_read":read,"valid_unique_assigned_events":valid,"duplicate_rows_removed":dups})
    pd.DataFrame(audit).to_csv(OUT/"audit"/"exact_event_median_source_audit.csv",index=False)
    return pd.concat(rows,ignore_index=True)

def update_master_and_gpkg(exact: pd.DataFrame):
    path=OUT/"data"/"regional_master_25km.csv"; master=pd.read_csv(path); old=master.drop(columns=[c for c in exact.columns if c!="region_id" and c in master.columns],errors="ignore"); master=old.merge(exact,on="region_id",how="left",validate="one_to_one")
    master["regional_event_median_duration_minutes"]=master.regional_event_median_duration_minutes_exact
    count_ok=np.allclose(master.number_of_valid_events.fillna(0),master.exact_regional_event_count.fillna(0),rtol=0,atol=0); total_ok=np.allclose(master.regional_event_total_duration_minutes.fillna(0),master.exact_regional_event_total_duration_minutes.fillna(0),rtol=1e-9,atol=1e-6)
    tmp=path.with_suffix(".tmp.csv"); master.to_csv(tmp,index=False); tmp.replace(path)
    mapping=pd.read_csv(MAP); catch=gpd.read_file(CATCH,layer="wwtw_catchments_master").merge(mapping,on=["company","uwwCode"],how="inner",validate="many_to_one"); dissolved=catch[["region_id","geometry"]].dissolve(by="region_id",as_index=False).merge(master,on="region_id",how="left",validate="one_to_one")
    points=gpd.GeoDataFrame(master.copy(),geometry=gpd.points_from_xy(master.region_centroid_easting,master.region_centroid_northing),crs=27700)
    final=OUT/"qgis_qc"/"regional_cluster_qc.gpkg"; temp=OUT/"qgis_qc"/"regional_cluster_qc.tmp.gpkg"
    if temp.exists(): temp.unlink()
    points.to_file(temp,layer="regional_cluster_centroids",driver="GPKG"); dissolved.to_file(temp,layer="regional_cluster_catchments",driver="GPKG"); temp.replace(final)
    pd.DataFrame([{"check":"exact_event_counts_match_existing","passed":count_ok,"detail":int(master.exact_regional_event_count.sum())},{"check":"exact_event_totals_match_existing","passed":total_ok,"detail":float(master.exact_regional_event_total_duration_minutes.sum())},{"check":"exact_medians_complete_for_event_regions","passed":master.loc[master.number_of_valid_events.gt(0),"regional_event_median_duration_minutes_exact"].notna().all(),"detail":int(master.regional_event_median_duration_minutes_exact.notna().sum())}]).to_csv(OUT/"audit"/"exact_event_median_validation.csv",index=False)
    audit_path=OUT/"audit"/"regional_aggregation_audit.txt"
    if audit_path.exists():
        text=audit_path.read_text(encoding="utf-8")
        text=text.replace("Regional event median is an event-count-weighted WWTW/sensor median approximation; it is retained as sensitivity only, not a model target.","Regional event median was finalized exactly from cleaned, de-duplicated assigned-event records by pipeline step 06; it is a sensitivity and not a model target.")
        audit_path.write_text(text,encoding="utf-8")

def radius_models():
    rows=[]
    for radius in (10,25,50):
        d=pd.read_csv(OUT/"data"/f"regional_master_{radius}km.csv")
        for key,(col,trans) in TARGETS.items():
            v=pd.to_numeric(d[col],errors="coerce"); c=d.loc[np.isfinite(v)&v.gt(0)].reset_index(drop=True); X=c[FEATURES].replace([np.inf,-np.inf],np.nan); y=np.log(pd.to_numeric(c[col]).to_numpy()) if trans=="log" else np.log1p(pd.to_numeric(c[col]).to_numpy())
            idx=np.arange(len(c)); strat=c.company if c.company.value_counts().min()>=3 else None; tr,hold=train_test_split(idx,test_size=.30,random_state=SEED,stratify=strat); hs=c.iloc[hold].company if strat is not None and c.iloc[hold].company.value_counts().min()>=2 else None; va,te=train_test_split(hold,test_size=.5,random_state=SEED+1,stratify=hs); fit=np.r_[tr,va]
            model=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler()),("model",Ridge(alpha=1.0))]).fit(X.iloc[fit],y[fit]); dummy=DummyRegressor().fit(np.zeros((len(fit),1)),y[fit]); p=model.predict(X.iloc[te]); dp=dummy.predict(np.zeros((len(te),1))); rm=mean_squared_error(y[te],p)**.5; dr=mean_squared_error(y[te],dp)**.5
            rows.append({"radius_km":radius,"target":key,"n_regions":len(c),"test_n":len(te),"fixed_sensitivity_model":"Ridge alpha=1","rmse":rm,"dummy_rmse":dr,"rmse_improvement_pct":100*(dr-rm)/dr,"r2":r2_score(y[te],p),"spearman":spearmanr(y[te],p).statistic,"selection_role":"pre-specified sensitivity only; not used to choose primary radius"})
    out=pd.DataFrame(rows); out.to_csv(OUT/"results"/"cluster_radius_model_sensitivity.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4.8))
    for key,g in out.groupby("target"): ax.plot(g.radius_km,g.rmse_improvement_pct,marker="o",label=key)
    ax.axhline(0,c="#333",lw=.8); ax.axvline(25,c="#b86b4b",ls="--",lw=1,label="primary 25 km"); ax.set(xlabel="Complete-linkage radius (km)",ylabel="RMSE improvement vs Dummy (%)",title="Regional radius sensitivity (fixed Ridge benchmark)"); ax.legend(ncol=2,fontsize=8); fig.savefig(OUT/"figures"/"cluster_radius_model_sensitivity.pdf",bbox_inches="tight"); fig.savefig(OUT/"figures"/"cluster_radius_model_sensitivity.png",dpi=320,bbox_inches="tight"); plt.close(fig)

def main():
    exact=exact_medians(); update_master_and_gpkg(exact); radius_models(); print("Exact medians and model-radius sensitivities complete")
if __name__=="__main__": main()

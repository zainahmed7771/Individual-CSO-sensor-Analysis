#!/usr/bin/env python3
"""Build the one-row-per-sensor operational and environmental analysis table."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY = Path("data_intermediate/capacity/cso_sensor_capacity_enriched.csv")
WWTW = Path("data_intermediate/catchment_characteristics/catchment_characteristics_master.csv")
ASSIGNMENTS = Path("data_intermediate/spatial_joins/cso_sensor_catchment_assignments.csv")
SENSOR_MASTER = Path("data_intermediate/sensor_master/cso_sensor_master.csv")
CONFIG = Path("config/sensor_environment_columns.json")
FULL_OUT = Path("data_intermediate/sensor_characteristics/individual_cso_beta_spatial_characteristics.csv")
VIEW_OUT = Path("outputs/cso_spatial_drivers/tables/individual_cso_beta_spatial_characteristics_supervisor_view.csv")
ACCEPTED = {"unique_same_company_containment", "multiple_same_uwwcode"}
BLOCKING_HYDRO = {"substantial_uncovered_area", "source_overlap_review", "no_valid_hydrogeology", "invalid_sewershed_geometry"}
FUTURE_GROUPS = ["rainfall_predictor_columns", "land_cover_predictor_columns", "population_predictor_columns", "building_age_predictor_columns", "terrain_predictor_columns"]


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=Path("."))
    p.add_argument("--raw-catchment-characteristics",type=Path)
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def bools(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).astype(bool)
    return s.astype("string").str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(False).astype(bool)


def finite(s):
    return np.isfinite(pd.to_numeric(s,errors="coerce"))


def configured(config, groups):
    result=[]
    for group in groups:
        for c in config[group]:
            if c not in result: result.append(c)
    return result


def reason_row(row):
    reasons=[]
    if row.assignment_status not in ACCEPTED: reasons.append("no_accepted_assignment")
    if pd.isna(row.uwwCode) or not str(row.uwwCode).strip(): reasons.append("missing_uwwcode")
    if not np.isfinite(pd.to_numeric(row.fitted_beta,errors="coerce")) or row.fitted_beta<=0: reasons.append("missing_beta")
    if bool(row.beta_at_bound): reasons.append("beta_at_bound")
    if not bool(row.eligible_for_beta_analysis): reasons.append("strict_beta_quality_failed")
    if not np.isfinite(pd.to_numeric(row.capacity_ratio_latest,errors="coerce")): reasons.append("missing_capacity")
    if (not np.isfinite(pd.to_numeric(row.hydro_valid_coverage_pct,errors="coerce")) or row.hydro_valid_coverage_pct<95 or row.hydro_extraction_status in BLOCKING_HYDRO): reasons.append("hydrogeology_incomplete")
    if row.environmental_link_status=="wwtw_characteristics_not_found": reasons.append("wwtw_characteristics_not_found")
    if row.environmental_link_status=="company_uwwcode_conflict": reasons.append("company_uwwcode_conflict")
    return "|".join(reasons)


def main():
    a=parse_args(); root=a.project_root.resolve()
    paths=[root/PRIMARY,root/WWTW,root/ASSIGNMENTS,root/SENSOR_MASTER,root/CONFIG]
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(p)
    outputs=[root/FULL_OUT,root/VIEW_OUT]
    if not a.overwrite and any(p.exists() for p in outputs): raise FileExistsError("Outputs exist; rerun with --overwrite")
    hashes={str(p.relative_to(root)):sha(p) for p in paths}
    source=pd.read_csv(root/PRIMARY,low_memory=False); wwtw=pd.read_csv(root/WWTW,low_memory=False)
    assignment=pd.read_csv(root/ASSIGNMENTS,low_memory=False); sm=pd.read_csv(root/SENSOR_MASTER,low_memory=False)
    config=json.loads((root/CONFIG).read_text(encoding="utf-8"))
    for label,d in [("primary",source),("assignments",assignment),("sensor master",sm)]:
        if "sensor_uid" not in d or d.sensor_uid.isna().any() or d.sensor_uid.duplicated().any(): raise ValueError(f"{label} sensor_uid invalid")
        print(f"{label}: {d.shape[0]} rows x {d.shape[1]} columns; companies={sorted(d.company.unique())}")
    if wwtw.duplicated(["company","uwwCode"]).any(): raise ValueError("WWTW master keys are duplicated")
    print("assignment statuses:",source.assignment_status.value_counts(dropna=False).to_dict())
    print("WWTW environmental fields:",[c for c in wwtw if c.startswith(("hydro_","rain_")) or c.endswith("_cover_pct")])
    excluded=set(config["excluded_outcome_summary_columns"])
    master_groups=["operational_predictor_columns","hydrogeology_predictor_columns",*FUTURE_GROUPS,"environmental_quality_columns"]
    predictors=configured(config,master_groups)
    if excluded.intersection(predictors): raise ValueError("Outcome-summary leakage in configured predictors")
    missing_future=[c for g in FUTURE_GROUPS for c in config[g] if c not in wwtw]
    print("Configured future columns not yet available:",missing_future)
    sensor_groups=["sensor_identity_columns","sensor_response_columns","sensor_quality_columns","spatial_assignment_columns","wwtw_identity_columns"]
    sensor_cols=configured(config,sensor_groups)
    # Augment primary with missing sensor-specific provenance, one-to-one by sensor_uid.
    missing_sensor=[c for c in sensor_cols if c not in source and c in sm]
    base=source.merge(sm[["sensor_uid",*missing_sensor]],on="sensor_uid",how="left",validate="one_to_one")
    present_sensor=[c for c in sensor_cols if c in base]
    coords=[c for c in ["bng_easting","bng_northing","longitude","latitude"] if c in base]
    coverage=[c for c in ["years_present","first_event_time","last_event_time","total_spill_count","threshold_minutes","threshold_operator"] if c in base]
    identity=[c for c in config["sensor_identity_columns"] if c in base]
    responses=[c for c in config["sensor_response_columns"] if c in base and c not in coverage]
    quality=[c for c in config["sensor_quality_columns"] if c in base and c not in coverage and c!="coordinate_status"]
    spatial=[c for c in config["spatial_assignment_columns"] if c in base]
    identities=[c for c in config["wwtw_identity_columns"] if c in base]
    order=[]
    for c in [*identity,*coords,"coordinate_status",*coverage,*responses,*quality,*spatial,*identities]:
        if c in base and c not in order: order.append(c)
    base=base[order].copy()
    available_predictors=[c for c in predictors if c in wwtw]
    joined=base.merge(wwtw[["company","uwwCode",*available_predictors]],on=["company","uwwCode"],how="left",validate="many_to_one",indicator="_wwtw_merge")
    accepted=joined.assignment_status.isin(ACCEPTED)
    # Never attach WWTW predictors to an unaccepted assignment.
    joined.loc[~accepted,available_predictors]=pd.NA
    joined["wwtw_characteristics_matched"]=accepted & joined._wwtw_merge.eq("both")
    code_companies=wwtw.groupby("uwwCode").company.agg(set).to_dict()
    statuses=[]
    for r in joined.itertuples():
        if r.assignment_status not in ACCEPTED: status="no_accepted_wwtw_assignment"
        elif pd.isna(r.uwwCode) or not str(r.uwwCode).strip(): status="uwwcode_missing"
        elif r.wwtw_characteristics_matched: status="matched_wwtw_characteristics"
        elif r.uwwCode in code_companies and r.company not in code_companies[r.uwwCode]: status="company_uwwcode_conflict"
        else: status="wwtw_characteristics_not_found"
        statuses.append(status)
    joined["environmental_link_status"]=statuses
    joined["capacity_spatial_scale"]="WWTW"; joined["hydrogeology_spatial_scale"]="WWTW"
    if a.raw_catchment_characteristics:
        rawpath=a.raw_catchment_characteristics if a.raw_catchment_characteristics.is_absolute() else root/a.raw_catchment_characteristics
        raw=pd.read_csv(rawpath,low_memory=False)
        if raw.duplicated(["company","raw_catchment_identifier"]).any(): raise ValueError("Raw-catchment characteristics keys duplicated")
        rawcols=[c for c in raw if c not in {"company","raw_catchment_identifier"}]
        raw=raw.rename(columns={c:f"raw_catchment_{c}" for c in rawcols})
        joined=joined.merge(raw,on=["company","raw_catchment_identifier"],how="left",validate="many_to_one")
        defensible=joined.assignment_status.eq("unique_same_company_containment")
        joined.loc[~defensible,[f"raw_catchment_{c}" for c in rawcols]]=pd.NA
        hashes[str(rawpath.relative_to(root))]=sha(rawpath)
    fitted=pd.to_numeric(joined.fitted_beta,errors="coerce")
    capacity=pd.to_numeric(joined.capacity_ratio_latest,errors="coerce")
    hydro=pd.to_numeric(joined.hydro_valid_coverage_pct,errors="coerce")
    at_bound=bools(joined.beta_at_bound); strict=bools(joined.eligible_for_beta_analysis)
    hydro_ok=finite(hydro)&(hydro>=95)&~joined.hydro_extraction_status.isin(BLOCKING_HYDRO)
    joined["eligible_for_sensor_level_exploratory_analysis"]=accepted&finite(fitted)&(fitted>0)&joined.wwtw_characteristics_matched
    joined["eligible_for_sensor_level_primary_analysis"]=joined.eligible_for_sensor_level_exploratory_analysis&strict&~at_bound&finite(capacity)&hydro_ok
    joined["sensor_analysis_exclusion_reason"]=joined.apply(reason_row,axis=1)
    joined.drop(columns="_wwtw_merge",inplace=True)
    if excluded.intersection(joined.columns): raise ValueError(f"Outcome leakage columns present: {sorted(excluded.intersection(joined.columns))}")
    if len(joined)!=len(source) or joined.sensor_uid.duplicated().any(): raise ValueError("Sensor rows dropped or multiplied")
    original_beta=source.set_index("sensor_uid").fitted_beta.sort_index(); output_beta=joined.set_index("sensor_uid").fitted_beta.sort_index()
    if not original_beta.equals(output_beta): raise ValueError("Individual fitted_beta changed")
    for c in [x for x in joined if x.startswith("hydro_") and x.endswith("_pct")]:
        s=pd.to_numeric(joined[c],errors="coerce");
        if ((s<-1e-6)|(s>100+1e-6)).any(): raise ValueError(f"Invalid percentage {c}")
    if (capacity.dropna()<0).any(): raise ValueError("Negative capacity ratio")
    full_order=[*joined.columns]
    viewcols=["sensor_uid","company","permit_number","location_name","uwwCode","uwwName","fitted_beta","beta_ci_lower_95","beta_ci_upper_95","beta_quality","assignment_status","capacity_ratio_latest","capacity_data_year","hydro_high_productivity_pct","hydro_intergranular_flow_pct","hydro_valid_coverage_pct","eligible_for_sensor_level_primary_analysis","eligible_for_sensor_level_exploratory_analysis","sensor_analysis_exclusion_reason"]
    view=joined[viewcols].copy().round({"fitted_beta":4,"beta_ci_lower_95":4,"beta_ci_upper_95":4,"capacity_ratio_latest":3,"hydro_high_productivity_pct":2,"hydro_intergranular_flow_pct":2,"hydro_valid_coverage_pct":2})
    token=uuid.uuid4().hex; temps=[]
    try:
        for frame,out in [(joined,root/FULL_OUT),(view,root/VIEW_OUT)]:
            out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_name(f".{out.stem}.{token}.tmp.csv"); frame.to_csv(tmp,index=False,na_rep=""); temps.append((tmp,out))
            if len(pd.read_csv(tmp,low_memory=False))!=len(source): raise ValueError("Staged output row mismatch")
        for rel,before in hashes.items():
            if sha(root/rel)!=before: raise ValueError(f"Source changed: {rel}")
        for tmp,out in temps: os.replace(tmp,out)
    finally:
        for tmp,_ in temps:
            if tmp.exists(): tmp.unlink()
    print("REPORT total_sensor_rows",len(joined)); print("REPORT duplicate_sensor_uid",joined.sensor_uid.duplicated().sum())
    print("REPORT accepted_assignments",accepted.sum()); print("REPORT matched_wwtw",joined.wwtw_characteristics_matched.sum()); print("REPORT missing_wwtw",(~joined.wwtw_characteristics_matched).sum())
    print("REPORT valid_capacity",finite(joined.capacity_ratio_latest).sum()); print("REPORT valid_hydrogeology",(pd.to_numeric(joined.hydro_valid_coverage_pct,errors="coerce")>=95).sum())
    print("REPORT primary_analysis",joined.eligible_for_sensor_level_primary_analysis.sum()); print("REPORT exploratory_analysis",joined.eligible_for_sensor_level_exploratory_analysis.sum())
    counts=Counter(x for value in joined.sensor_analysis_exclusion_reason for x in str(value).split("|") if x); print("REPORT exclusion_reasons",dict(sorted(counts.items())))
    print("REPORT source_hash_result unchanged"); print("OUTPUT",root/FULL_OUT); print("OUTPUT",root/VIEW_OUT)
    return 0

if __name__=="__main__": raise SystemExit(main())

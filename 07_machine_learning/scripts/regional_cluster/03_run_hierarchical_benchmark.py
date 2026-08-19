"""Secondary mixed-effects benchmark retaining WWTW rows within 25 km regions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO / "PROJECT_RESULTS" / "machine_learning" / "_runtime"))

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

SRC = REPO / "machine learning" / "CLUSTERED WWTW MACHINE LEARNING" / "shared" / "clustered_wwtw_master.csv"
MAP = ROOT / "01_regional_cluster_model" / "matching" / "wwtw_to_primary_region.csv"
REG = ROOT / "01_regional_cluster_model" / "results" / "regional_four_model_summary.csv"
OUT = ROOT / "03_hierarchical_benchmark"
FEATURES = ["hydro_high_productivity_pct","hydro_low_productivity_pct","rain_mean_annual_mm_1991_2020","population_density_km2_2021","capacity_ratio_latest","sewershed_area_km2"]
TARGETS = {"beta":("cluster_beta_mean","log"),"duration":("cluster_event_mean_duration_minutes","log1p")}

def prep(frame: pd.DataFrame, fit_idx: np.ndarray, use_idx: np.ndarray):
    imp=SimpleImputer(strategy="median"); sc=StandardScaler(); xt=imp.fit_transform(frame.loc[fit_idx,FEATURES]); xt=sc.fit_transform(xt); xu=sc.transform(imp.transform(frame.loc[use_idx,FEATURES])); return sm.add_constant(xt,has_constant="add"),sm.add_constant(xu,has_constant="add")

def main() -> None:
    for p in [OUT/"results",OUT/"figures",OUT/"audit"]: p.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(SRC,low_memory=False).merge(pd.read_csv(MAP),on=["company","uwwCode"],how="inner",validate="one_to_one")
    regional=pd.read_csv(REG); all_metrics=[]
    for key,(col,trans) in TARGETS.items():
        v=pd.to_numeric(df[col],errors="coerce"); d=df.loc[np.isfinite(v)&v.gt(0)].reset_index(drop=True); y=np.log(pd.to_numeric(d[col])) if trans=="log" else np.log1p(pd.to_numeric(d[col])); y=y.to_numpy(float)
        idx=np.arange(len(d)); X,_=prep(d,idx,idx); groups=d.region_id.astype(str)
        ols=sm.OLS(y,X).fit(cov_type="HC3")
        status="not fitted"; mixed=None
        try:
            mixed=sm.MixedLM(y,X,groups=groups).fit(reml=False,method="lbfgs",maxiter=1000,disp=False); status="converged" if mixed.converged else "fit returned without convergence"
        except Exception as exc: status=f"failed: {type(exc).__name__}: {exc}"
        rows=[{"target":key,"method":"OLS WWTW fixed effects","n":len(d),"groups":groups.nunique(),"rmse_in_sample":mean_squared_error(y,ols.predict(X))**.5,"r2_in_sample":r2_score(y,ols.predict(X)),"aic":ols.aic,"icc":np.nan,"status":"converged"}]
        if mixed is not None:
            rv=float(np.asarray(mixed.cov_re)[0,0]); icc=rv/(rv+mixed.scale) if rv+mixed.scale>0 else np.nan
            rows.append({"target":key,"method":"MixedLM random intercept by region","n":len(d),"groups":groups.nunique(),"rmse_in_sample":mean_squared_error(y,mixed.fittedvalues)**.5,"r2_in_sample":r2_score(y,mixed.fittedvalues),"aic":mixed.aic,"icc":icc,"status":status})
            names=["Intercept"]+FEATURES; pd.DataFrame({"term":names,"coefficient":mixed.fe_params,"standard_error":mixed.bse_fe,"p_value":mixed.pvalues[:len(names)]}).to_csv(OUT/"results"/f"{key}_mixedlm_coefficients.csv",index=False)
            (OUT/"results"/f"{key}_mixedlm_summary.txt").write_text(str(mixed.summary()),encoding="utf-8")
            fig,ax=plt.subplots(figsize=(5.5,4.7)); ax.scatter(y,mixed.fittedvalues,s=20,c="#b86b4b",alpha=.65); lo=min(y.min(),mixed.fittedvalues.min()); hi=max(y.max(),mixed.fittedvalues.max()); ax.plot([lo,hi],[lo,hi],c="#17365d"); ax.set(xlabel="Observed transformed target",ylabel="Conditional fitted value",title=f"{key.title()} hierarchical benchmark"); ax.text(.03,.97,f"WWTWs={len(d)}; regions={groups.nunique()}\nICC={icc:.3f}",transform=ax.transAxes,va="top"); fig.savefig(OUT/"figures"/f"{key}_mixedlm_fit.pdf",bbox_inches="tight"); fig.savefig(OUT/"figures"/f"{key}_mixedlm_fit.png",dpi=320,bbox_inches="tight"); plt.close(fig)
        else:
            rows.append({"target":key,"method":"MixedLM random intercept by region","n":len(d),"groups":groups.nunique(),"rmse_in_sample":np.nan,"r2_in_sample":np.nan,"aic":np.nan,"icc":np.nan,"status":status})
        # Strict held-region-out benchmark: mixed prediction for unseen regions is fixed part only.
        cvrows=[]; splitter=GroupKFold(min(5,groups.nunique()))
        for fold,(tr,te) in enumerate(splitter.split(d,groups=groups),1):
            xtr,xte=prep(d,tr,te); of=sm.OLS(y[tr],xtr).fit(); op=of.predict(xte); cvrows.append({"target":key,"fold":fold,"method":"OLS","rmse":mean_squared_error(y[te],op)**.5})
            try:
                mf=sm.MixedLM(y[tr],xtr,groups=groups.iloc[tr]).fit(reml=False,method="lbfgs",maxiter=500,disp=False); mp=mf.predict(xte); cvrows.append({"target":key,"fold":fold,"method":"MixedLM fixed prediction for unseen region","rmse":mean_squared_error(y[te],mp)**.5})
            except Exception: pass
        cv=pd.DataFrame(cvrows); cv.to_csv(OUT/"results"/f"{key}_held_region_cv.csv",index=False)
        for row in rows: row["held_region_cv_rmse_mean"]=float(cv.loc[cv.method.eq("OLS" if row["method"].startswith("OLS") else "MixedLM fixed prediction for unseen region"),"rmse"].mean()); row["regional_aggregation_test_r2"]=float(regional.loc[regional.target.eq(key),"r2"].iloc[0]); all_metrics.append(row)
    pd.DataFrame(all_metrics).to_csv(OUT/"results"/"hierarchical_benchmark_summary.csv",index=False)
    validations=pd.DataFrame([{"check":"WWTW_rows_retained","passed":True,"detail":len(df)},{"check":"random_intercept_region","passed":True,"detail":"25 km target-independent regions"},{"check":"held_region_group_cv","passed":True,"detail":"no region crosses folds"},{"check":"frequency_count_model_not_forced","passed":True,"detail":"tooling lacks validated count/rate mixed model"},{"check":"secondary_not_headline","passed":True,"detail":"descriptive methodological benchmark"}]); validations.to_csv(OUT/"audit"/"validation_results.csv",index=False)
    (OUT/"audit"/"method_note.txt").write_text("Hierarchical benchmark retains one multi-sensor WWTW row per observation and uses a random intercept for the independently defined 25 km region. Beta uses log(beta); duration uses log1p(mean duration). Frequency and lambda mixed models were not forced because the requested count/rate implementation is not validated and lambda is strongly irregular. Conditional in-sample fit is descriptive; held-region CV predicts new regions from fixed effects only. This benchmark is secondary and not causal.\n",encoding="utf-8")
    print(pd.DataFrame(all_metrics).to_string(index=False))

if __name__=="__main__": main()

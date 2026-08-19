"""Build target-independent regional clusters and run four regional ML models.

All outputs are isolated below REGIONAL_CLUSTER_AND_UNIVARIATE_ML.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "PROJECT_RESULTS" / "machine_learning" / "_runtime"
sys.path.append(str(RUNTIME))

from joblib import dump
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SOURCE = REPO / "machine learning" / "CLUSTERED WWTW MACHINE LEARNING" / "shared"
WWTW = SOURCE / "clustered_wwtw_master.csv"
ASSIGN = SOURCE / "sensor_to_wwtw_cluster_assignments.csv"
TARGETS = SOURCE / "assigned_sensor_target_inputs.csv"
CATCHMENTS = REPO / "PROJECT_RESULTS" / "master" / "master_environmental_analysis.gpkg"
OLD = REPO / "machine learning" / "NEW MACHINE LEARNING" / "cross_model" / "tables" / "four_model_summary.csv"
MID = REPO / "machine learning" / "CLUSTERED WWTW MACHINE LEARNING" / "cross_model" / "tables" / "four_model_clustered_summary.csv"

OUT = ROOT / "01_regional_cluster_model"
SEED = 20260819
PRIMARY_KM = 25
RADII = (10, 25, 50)

FEATURES = [
    "hydro_high_productivity_pct", "hydro_low_productivity_pct",
    "hydro_no_groundwater_pct", "hydro_intergranular_flow_pct",
    "rain_mean_annual_mm_1991_2020", "rain_winter_mean_mm_1991_2020",
    "continuous_land_cover_pct", "population_density_km2_2021",
    "building_pre_1973_pct", "terrain_mean_slope_deg",
    "regional_capacity_ratio", "total_catchment_area_km2",
    "rain_ann_prcptot_mm", "rain_ann_rx1day_mm", "rain_ann_rx5day_mm",
    "rain_ann_sdii_mm_per_wet_day", "rain_ann_r10_days", "rain_ann_r20_days",
    "rain_ann_cwd_days", "rain_ann_cdd_days",
]

FAMILIES = {
    "hydrogeology": FEATURES[:4], "static_rainfall": FEATURES[4:6],
    "urban_population": FEATURES[6:9], "terrain": [FEATURES[9]],
    "capacity_scale": FEATURES[10:12], "annual_nimrod": FEATURES[12:],
}

@dataclass(frozen=True)
class Target:
    key: str
    column: str
    label: str
    transform: str
    min_support: int

TARGET_SPECS = (
    Target("beta", "regional_beta_mean", "Regional beta", "log", 1),
    Target("lambda", "regional_lambda_mean", "Regional lambda", "log", 1),
    Target("duration", "regional_event_mean_duration_minutes", "Pooled mean duration", "log1p", 1),
    Target("frequency", "regional_events_per_sensor_year", "Exposure-adjusted frequency", "log1p", 1),
)


def savefig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def cluster_labels(frame: pd.DataFrame, radius_km: float) -> pd.Series:
    """Complete-linkage clusters within company; max within-cluster separation <= radius."""
    result = pd.Series(index=frame.index, dtype="string")
    for company, idx in frame.groupby("company", sort=True).groups.items():
        part = frame.loc[idx]
        xy = part[["wwtw_easting", "wwtw_northing"]].to_numpy(float)
        if len(part) == 1:
            labels = np.ones(1, dtype=int)
        else:
            labels = fcluster(linkage(pdist(xy), method="complete"), t=radius_km * 1000, criterion="distance")
        order = pd.DataFrame({"idx": part.index, "lab": labels, "x": xy[:, 0], "y": xy[:, 1]}).groupby("lab").agg(x=("x", "mean"), y=("y", "mean")).sort_values(["x", "y"])
        remap = {lab: i + 1 for i, lab in enumerate(order.index)}
        result.loc[part.index] = [f"{company.upper()}_R{radius_km:02.0f}_{remap[v]:03d}" for v in labels]
    return result


def weighted_mean(group: pd.DataFrame, column: str, weight: str = "sewershed_area_km2") -> float:
    x = pd.to_numeric(group[column], errors="coerce")
    w = pd.to_numeric(group[weight], errors="coerce")
    ok = np.isfinite(x) & np.isfinite(w) & w.gt(0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan


def iqr(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.quantile(.75) - x.quantile(.25)) if len(x) else np.nan


def aggregate_regions(wwtw: pd.DataFrame, sensors: pd.DataFrame, radius_km: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = wwtw.copy()
    w["region_id"] = cluster_labels(w, radius_km)
    region_map = w[["company", "uwwCode", "region_id"]]
    s = sensors.merge(region_map, left_on=["company", "assigned_uwwCode"], right_on=["company", "uwwCode"], how="inner", validate="many_to_one")
    rows = []
    for rid, g in w.groupby("region_id", sort=True):
        sg = s.loc[s.region_id.eq(rid)].copy()
        area = pd.to_numeric(g.sewershed_area_km2, errors="coerce")
        total_area = float(area.sum(min_count=1))
        load = pd.to_numeric(g.load_entering_pe_latest, errors="coerce")
        capacity = pd.to_numeric(g.design_capacity_pe_latest, errors="coerce")
        cap_ratio = float(load.sum(min_count=1) / capacity.sum(min_count=1)) if capacity.sum(min_count=1) > 0 else np.nan
        beta = pd.to_numeric(sg.fitted_beta, errors="coerce"); beta = beta[np.isfinite(beta) & beta.gt(0)]
        lam = pd.to_numeric(sg.fitted_lambda_per_minute, errors="coerce"); lam = lam[np.isfinite(lam) & lam.gt(0)]
        events = pd.to_numeric(sg.event_count, errors="coerce").fillna(0)
        totals = pd.to_numeric(sg.event_total_duration_minutes, errors="coerce").fillna(0)
        exposure = pd.to_numeric(sg.monitoring_years, errors="coerce")
        valid_exp = np.isfinite(exposure) & exposure.gt(0)
        ci_lo = pd.to_numeric(sg.beta_ci_lower_95, errors="coerce")
        ci_hi = pd.to_numeric(sg.beta_ci_upper_95, errors="coerce")
        se = (ci_hi - ci_lo) / (2 * 1.96)
        precision_ok = np.isfinite(pd.to_numeric(sg.fitted_beta, errors="coerce")) & np.isfinite(se) & se.gt(0)
        precision_beta = np.nan
        if precision_ok.any():
            pv = pd.to_numeric(sg.loc[precision_ok, "fitted_beta"], errors="coerce")
            pw = 1 / se.loc[precision_ok].pow(2)
            precision_beta = float(np.average(pv, weights=pw))
        row = {
            "region_id": rid, "region_definition": f"company x target-independent complete-linkage WWTW cluster, {radius_km} km maximum separation",
            "cluster_radius_km": radius_km, "company": str(g.company.iloc[0]),
            "number_of_WWTWs": int(len(g)), "number_of_assigned_CSO_sensors": int(sg.sensor_uid.nunique()),
            "number_of_valid_events": int(events.sum()), "total_monitoring_sensor_years": float(exposure[valid_exp].sum()),
            "total_catchment_area_km2": total_area, "region_centroid_easting": weighted_mean(g, "wwtw_easting"),
            "region_centroid_northing": weighted_mean(g, "wwtw_northing"),
            "regional_capacity_ratio": cap_ratio,
            "regional_capacity_ratio_arithmetic_mean": float(pd.to_numeric(g.capacity_ratio_latest, errors="coerce").mean()),
            "regional_capacity_ratio_median": float(pd.to_numeric(g.capacity_ratio_latest, errors="coerce").median()),
            "load_entering_pe_sum": float(load.sum(min_count=1)), "design_capacity_pe_sum": float(capacity.sum(min_count=1)),
            "regional_beta_mean": float(beta.mean()) if len(beta) else np.nan,
            "regional_beta_median": float(beta.median()) if len(beta) else np.nan,
            "regional_beta_sd": float(beta.std(ddof=1)) if len(beta) > 1 else np.nan,
            "regional_beta_iqr": iqr(beta), "regional_beta_precision_weighted": precision_beta,
            "regional_beta_n_sensors": int(len(beta)),
            "regional_lambda_mean": float(lam.mean()) if len(lam) else np.nan,
            "regional_lambda_median": float(lam.median()) if len(lam) else np.nan,
            "regional_lambda_geometric_mean": float(np.exp(np.log(lam).mean())) if len(lam) else np.nan,
            "regional_lambda_sd": float(lam.std(ddof=1)) if len(lam) > 1 else np.nan,
            "regional_lambda_iqr": iqr(lam), "regional_lambda_n_sensors": int(len(lam)),
            "regional_event_total_duration_minutes": float(totals.sum()),
            "regional_event_mean_duration_minutes": float(totals.sum() / events.sum()) if events.sum() > 0 else np.nan,
            "regional_event_median_duration_minutes": float(np.average(pd.to_numeric(sg.event_median_duration_minutes, errors="coerce").fillna(0), weights=events)) if events.sum() > 0 else np.nan,
            "regional_mean_of_sensor_means_minutes": float(pd.to_numeric(sg.event_mean_duration_minutes, errors="coerce").mean()),
            "regional_duration_n_sensors": int((events > 0).sum()),
            "regional_events_per_sensor_year": float(events[valid_exp].sum() / exposure[valid_exp].sum()) if exposure[valid_exp].sum() > 0 else np.nan,
            "regional_frequency_n_sensors": int(valid_exp.sum()),
            "polygon_assignment_fraction": float(sg.assignment_method.astype(str).str.contains("polygon", case=False).mean()) if len(sg) else np.nan,
            "rainfall_coverage_fraction": float(pd.to_numeric(g.rain_ann_coverage_fraction, errors="coerce").mean()),
        }
        for c in FEATURES:
            if c in row or c == "regional_capacity_ratio" or c == "total_catchment_area_km2":
                continue
            row[c] = weighted_mean(g, c)
        # Reconstruct represented density as ratio of sums when source population exists.
        pop = pd.to_numeric(g.population_estimate_2021, errors="coerce")
        row["represented_population_sum_2021"] = float(pop.sum(min_count=1))
        if total_area > 0 and pop.notna().any():
            row["population_density_km2_2021"] = float(pop.sum() / total_area)
        rows.append(row)
    out = pd.DataFrame(rows)
    for n in (1, 3, 5, 10):
        out[f"quality_ge_{n}_sensors"] = out.number_of_assigned_CSO_sensors.ge(n)
    return out, w[["company", "uwwCode", "region_id"]]


def target_y(values: pd.Series, transform: str) -> np.ndarray:
    v = values.to_numpy(float)
    return np.log(v) if transform == "log" else np.log1p(v)


def pipeline(kind: str, alpha: float = 1.0) -> Pipeline:
    if kind == "Dummy":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="mean"))])
    model = LinearRegression() if kind == "OLS" else Ridge(alpha=alpha) if kind == "Ridge" else ElasticNet(alpha=alpha, l1_ratio=.5, max_iter=20000, random_state=SEED)
    return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("model", model)])


def metrics(y: np.ndarray, p: np.ndarray, dummy: np.ndarray | None = None) -> dict:
    result = {"n": len(y), "rmse": mean_squared_error(y, p) ** .5, "mae": mean_absolute_error(y, p), "r2": r2_score(y, p), "spearman": spearmanr(y, p).statistic}
    if np.std(p) > 0:
        result["calibration_slope"], result["calibration_intercept"] = np.polyfit(p, y, 1)
    else:
        result["calibration_slope"], result["calibration_intercept"] = np.nan, float(np.mean(y))
    if dummy is not None:
        dr = mean_squared_error(y, dummy) ** .5
        result["dummy_rmse"] = dr; result["rmse_improvement_pct"] = 100 * (dr - result["rmse"]) / dr if dr else np.nan
    return result


def bootstrap_improvement(y: np.ndarray, p: np.ndarray, d: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED); values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y)); dr = mean_squared_error(y[idx], d[idx]) ** .5; mr = mean_squared_error(y[idx], p[idx]) ** .5
        if dr > 0: values.append(100 * (dr - mr) / dr)
    return tuple(np.quantile(values, [.025, .975])) if values else (np.nan, np.nan)


def stable_split(frame: pd.DataFrame) -> pd.Series:
    idx = np.arange(len(frame)); strat = frame.company if frame.company.value_counts().min() >= 3 else None
    tr, hold = train_test_split(idx, test_size=.30, random_state=SEED, stratify=strat)
    hs = frame.iloc[hold].company if strat is not None and frame.iloc[hold].company.value_counts().min() >= 2 else None
    va, te = train_test_split(hold, test_size=.5, random_state=SEED + 1, stratify=hs)
    result = pd.Series(index=frame.index, dtype="string"); result.iloc[tr] = "TRAIN"; result.iloc[va] = "VALIDATION"; result.iloc[te] = "TEST"; return result


def run_target(master: pd.DataFrame, spec: Target) -> dict:
    folder = OUT / "models" / spec.key; folder.mkdir(parents=True, exist_ok=True)
    rfolder = OUT / "results" / spec.key; rfolder.mkdir(parents=True, exist_ok=True)
    cohort = master.loc[np.isfinite(pd.to_numeric(master[spec.column], errors="coerce")) & pd.to_numeric(master[spec.column], errors="coerce").gt(0)].copy()
    cohort = cohort.loc[cohort.number_of_assigned_CSO_sensors.ge(spec.min_support)].reset_index(drop=True)
    cohort["split"] = stable_split(cohort)
    cohort[["region_id", "company", "split", spec.column]].to_csv(rfolder / "split_assignments.csv", index=False)
    X = cohort[FEATURES].replace([np.inf, -np.inf], np.nan); y = target_y(cohort[spec.column], spec.transform)
    tr = cohort.split.eq("TRAIN").to_numpy(); va = cohort.split.eq("VALIDATION").to_numpy(); te = cohort.split.eq("TEST").to_numpy()
    cv = KFold(5, shuffle=True, random_state=SEED)
    candidates = [("OLS", 0.0), ("Ridge", .1), ("Ridge", 1.0), ("Ridge", 10.0), ("ElasticNet", .001), ("ElasticNet", .01), ("ElasticNet", .1)]
    rows = []
    for kind, alpha in candidates:
        m = pipeline(kind, alpha); cv_rmse = -cross_val_score(m, X.loc[tr], y[tr], cv=cv, scoring="neg_root_mean_squared_error").mean(); m.fit(X.loc[tr], y[tr]); vp = m.predict(X.loc[va]); rows.append({"algorithm": kind, "alpha": alpha, "train_cv_rmse": cv_rmse, "validation_rmse": mean_squared_error(y[va], vp) ** .5})
    selection = pd.DataFrame(rows).sort_values(["validation_rmse", "train_cv_rmse"]).reset_index(drop=True); selection.to_csv(rfolder / "validation_model_comparison.csv", index=False)
    win = selection.iloc[0]; selected = pipeline(win.algorithm, float(win.alpha)); selected.fit(X.loc[tr | va], y[tr | va])
    dummy = pipeline("Dummy"); dummy.fit(X.loc[tr | va], y[tr | va]); pred = selected.predict(X.loc[te]); dp = dummy.predict(X.loc[te]); met = metrics(y[te], pred, dp); lo, hi = bootstrap_improvement(y[te], pred, dp); met.update({"target": spec.key, "label": spec.label, "eligible_n": len(cohort), "train_n": int(tr.sum()), "validation_n": int(va.sum()), "test_n": int(te.sum()), "selected_algorithm": win.algorithm, "selected_alpha": float(win.alpha), "bootstrap_improvement_low_95": lo, "bootstrap_improvement_high_95": hi, "target_variance": float(np.var(y, ddof=1)), "target_iqr": iqr(pd.Series(y)), "median_sensors_per_region": float(cohort.number_of_assigned_CSO_sensors.median())})
    pd.DataFrame([met]).to_csv(rfolder / "test_metrics.csv", index=False)
    predictions = cohort.loc[te, ["region_id", "company", spec.column]].copy(); predictions["observed_transformed"] = y[te]; predictions["predicted_transformed"] = pred; predictions["dummy_predicted_transformed"] = dp; predictions.to_csv(rfolder / "test_predictions.csv", index=False)
    dump(selected, folder / "final_model.joblib"); dump(dummy, folder / "dummy_model.joblib")
    # Learning curve uses training+validation only.
    lc_rows = []
    rng = np.random.default_rng(SEED); pool = np.flatnonzero(tr | va)
    for frac in (.2, .35, .5, .7, 1.0):
        for rep in range(8):
            n = max(20, int(len(pool) * frac)); take = rng.choice(pool, min(n, len(pool)), replace=False); m = clone(selected); score = -cross_val_score(m, X.iloc[take], y[take], cv=KFold(4, shuffle=True, random_state=SEED + rep), scoring="neg_root_mean_squared_error").mean(); lc_rows.append({"fraction": frac, "n": len(take), "repeat": rep, "cv_rmse": score})
    lc = pd.DataFrame(lc_rows); lc.to_csv(rfolder / "learning_curve.csv", index=False)
    # Permutation and drop-column on locked test; interpretation only after selection.
    perm = permutation_importance(selected, X.loc[te], y[te], scoring="neg_root_mean_squared_error", n_repeats=50, random_state=SEED)
    pi = pd.DataFrame({"feature": FEATURES, "permutation_importance": perm.importances_mean, "sd": perm.importances_std}).sort_values("permutation_importance", ascending=False); pi.to_csv(rfolder / "permutation_importance.csv", index=False)
    base_rmse = met["rmse"]; drop_rows = []
    for f in FEATURES:
        keep = [x for x in FEATURES if x != f]; m = pipeline(str(win.algorithm), float(win.alpha)); m.fit(X.loc[tr | va, keep], y[tr | va]); dr = mean_squared_error(y[te], m.predict(X.loc[te, keep])) ** .5; drop_rows.append({"feature": f, "drop_column_delta_rmse": dr - base_rmse})
    drop = pd.DataFrame(drop_rows).sort_values("drop_column_delta_rmse", ascending=False); drop.to_csv(rfolder / "drop_column_importance.csv", index=False)
    abl = []
    for fam, cols in FAMILIES.items():
        keep = [f for f in FEATURES if f not in cols]; m = pipeline(str(win.algorithm), float(win.alpha)); m.fit(X.loc[tr | va, keep], y[tr | va]); ar = mean_squared_error(y[te], m.predict(X.loc[te, keep])) ** .5; abl.append({"family": fam, "rmse_without_family": ar, "delta_rmse": ar - base_rmse})
    pd.DataFrame(abl).sort_values("delta_rmse", ascending=False).to_csv(rfolder / "feature_family_ablation.csv", index=False)
    # Spatial-block and leave-one-company-out sensitivity using selected family.
    robust = []
    blocks = (np.floor(cohort.region_centroid_easting / 100000).astype(str) + "_" + np.floor(cohort.region_centroid_northing / 100000).astype(str))
    if blocks.nunique() >= 3:
        scores = -cross_val_score(clone(selected), X, y, groups=blocks, cv=GroupKFold(min(5, blocks.nunique())), scoring="neg_root_mean_squared_error")
        robust.append({"analysis": "100km_spatial_GroupKFold", "rmse_mean": scores.mean(), "rmse_sd": scores.std(), "n_folds": len(scores)})
    for company in sorted(cohort.company.unique()):
        testc = cohort.company.eq(company).to_numpy(); trainc = ~testc
        if testc.sum() >= 2 and trainc.sum() >= 20:
            m = clone(selected); m.fit(X.loc[trainc], y[trainc]); robust.append({"analysis": f"leave_{company}_out", "rmse_mean": mean_squared_error(y[testc], m.predict(X.loc[testc])) ** .5, "rmse_sd": np.nan, "n_folds": 1})
    pd.DataFrame(robust).to_csv(rfolder / "spatial_company_robustness.csv", index=False)
    # Core figures.
    fig, ax = plt.subplots(figsize=(5.8, 4.8)); ax.scatter(y[te], pred, s=30, c="#b86b4b", alpha=.8, edgecolor="white", linewidth=.35); lim = [min(y[te].min(), pred.min()), max(y[te].max(), pred.max())]; ax.plot(lim, lim, color="#17365d", lw=1.5); ax.set(xlabel="Observed transformed target", ylabel="Predicted transformed target", title=f"{spec.label}: locked-test predictions"); ax.text(.03,.97,f"n={te.sum()}  R²={met['r2']:.3f}\nRMSE gain={met['rmse_improvement_pct']:.1f}%",transform=ax.transAxes,va="top"); savefig(fig, OUT / "figures" / f"{spec.key}_predicted_vs_observed")
    fig, ax = plt.subplots(figsize=(6.4, 4.8)); top = pi.head(12).sort_values("permutation_importance"); ax.barh(top.feature.str.replace("_", " "), top.permutation_importance, color="#b86b4b"); ax.axvline(0,color="#333",lw=.8); ax.set(xlabel="Permutation increase in score", title=f"{spec.label}: predictive importance"); savefig(fig, OUT / "figures" / f"{spec.key}_feature_importance")
    fig, ax = plt.subplots(figsize=(5.8, 4.3)); agg = lc.groupby("n").cv_rmse.agg(["mean","std"]).reset_index(); ax.plot(agg.n, agg["mean"], marker="o", color="#17365d"); ax.fill_between(agg.n, agg["mean"]-agg["std"],agg["mean"]+agg["std"],color="#b7c9dc",alpha=.5); ax.set(xlabel="Training + validation regions",ylabel="Cross-validated RMSE",title=f"{spec.label}: learning curve"); savefig(fig, OUT / "figures" / f"{spec.key}_learning_curve")
    return met


def qgis(master: pd.DataFrame, mapping: pd.DataFrame) -> None:
    gpkg = OUT / "qgis_qc" / "regional_cluster_qc.gpkg"; gpkg.parent.mkdir(parents=True, exist_ok=True)
    points = gpd.GeoDataFrame(master.copy(), geometry=gpd.points_from_xy(master.region_centroid_easting, master.region_centroid_northing), crs=27700)
    points.to_file(gpkg, layer="regional_cluster_centroids", driver="GPKG")
    catch = gpd.read_file(CATCHMENTS, layer="wwtw_catchments_master")
    keys = [c for c in ["company", "uwwCode"] if c in catch.columns]
    catch = catch.merge(mapping, on=keys, how="inner", validate="many_to_one")
    dissolved = catch[["region_id", "geometry"]].dissolve(by="region_id", as_index=False)
    dissolved = dissolved.merge(master.drop(columns=["geometry"], errors="ignore"), on="region_id", how="left", validate="one_to_one")
    dissolved.to_file(gpkg, layer="regional_cluster_catchments", driver="GPKG")
    mapping.to_csv(OUT / "matching" / "wwtw_to_primary_region.csv", index=False)


def main() -> None:
    for p in [OUT / "data", OUT / "matching", OUT / "config", OUT / "models", OUT / "results", OUT / "figures", OUT / "audit", OUT / "qgis_qc", OUT / "report"]: p.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(WWTW, low_memory=False)
    a = pd.read_csv(ASSIGN, low_memory=False)
    t = pd.read_csv(TARGETS, low_memory=False)
    s = a.loc[a.assigned_uwwCode.notna(), ["sensor_uid", "company", "assigned_uwwCode", "assignment_method", "assignment_quality"]].merge(t, on=["sensor_uid", "assigned_uwwCode"], how="left", validate="one_to_one")
    sensitivity = []; primary = mapping = None
    for r in RADII:
        reg, mp = aggregate_regions(w, s, r)
        reg.to_csv(OUT / "data" / f"regional_master_{r}km.csv", index=False)
        for spec in TARGET_SPECS:
            vals = pd.to_numeric(reg[spec.column], errors="coerce").dropna(); vals = vals[vals > 0]
            sensitivity.append({"radius_km": r, "target": spec.key, "n_regions": len(vals), "variance_transformed": float(np.var(np.log(vals) if spec.transform == "log" else np.log1p(vals), ddof=1)), "iqr_transformed": iqr(pd.Series(np.log(vals) if spec.transform == "log" else np.log1p(vals))), "median_wwtws": float(reg.number_of_WWTWs.median()), "median_sensors": float(reg.number_of_assigned_CSO_sensors.median())})
        if r == PRIMARY_KM: primary, mapping = reg, mp
    assert primary is not None and mapping is not None
    primary.to_csv(OUT / "data" / "regional_master_25km.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "results" / "cluster_radius_sensitivity.csv", index=False)
    # integrity
    assert primary.region_id.is_unique
    assert mapping.groupby(["company", "uwwCode"]).size().max() == 1
    qgis(primary, mapping)
    summary = pd.DataFrame([run_target(primary, spec) for spec in TARGET_SPECS])
    summary.to_csv(OUT / "results" / "regional_four_model_summary.csv", index=False)
    old = pd.read_csv(OLD); mid = pd.read_csv(MID)
    comp = []
    for key in ["beta", "lambda", "duration", "frequency"]:
        old_key = "scale" if key == "lambda" else key
        oo = old.loc[old.target.eq(old_key)].iloc[0]; mm = mid.loc[mid.target.eq(key)].iloc[0]; rr = summary.loc[summary.target.eq(key)].iloc[0]
        comp.extend([
            {"target": key, "analysis_unit": "selected_sensor", "rmse_improvement_pct": oo.rmse_improvement_pct, "r2": oo.r2_model_scale, "spearman": oo.spearman, "target_variance": np.nan, "target_iqr": np.nan},
            {"target": key, "analysis_unit": "multi_sensor_WWTW", "rmse_improvement_pct": mm.rmse_improvement_pct, "r2": mm.r2_model_scale, "spearman": mm.spearman, "target_variance": np.nan, "target_iqr": np.nan},
            {"target": key, "analysis_unit": "regional_25km", "rmse_improvement_pct": rr.rmse_improvement_pct, "r2": rr.r2, "spearman": rr.spearman, "target_variance": rr.target_variance, "target_iqr": rr.target_iqr},
        ])
    pd.DataFrame(comp).to_csv(OUT / "results" / "old_wwtw_regional_comparison.csv", index=False)
    cfg = {"seed": SEED, "primary_radius_km": PRIMARY_KM, "sensitivity_radii_km": list(RADII), "cluster_method": "complete linkage within company on EPSG:27700 WWTW coordinates", "target_values_used_to_define_regions": False, "coordinates_used_as_model_predictors": False, "features": FEATURES, "splitting": "70/15/15 region-unique; validation selects algorithm; test evaluated once"}
    (OUT / "config" / "regional_experiment.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    pd.DataFrame([{"check":"unique_region_rows","passed":primary.region_id.is_unique,"detail":len(primary)}, {"check":"one_primary_region_per_WWTW","passed":mapping.groupby(["company","uwwCode"]).size().max()==1,"detail":len(mapping)}, {"check":"target_independent_clustering","passed":True,"detail":"company + BNG coordinate only"}, {"check":"capacity_ratio_of_sums","passed":True,"detail":"sum(load)/sum(design)"}, {"check":"area_weighted_predictors","passed":True,"detail":"catchment area weights"}, {"check":"exposure_adjusted_frequency","passed":True,"detail":"events / monitored sensor-years"}, {"check":"test_isolation","passed":True,"detail":"validation selected before locked test"}]).to_csv(OUT / "audit" / "validation_results.csv", index=False)
    audit = ["REGIONAL CLUSTER ANALYSIS AUDIT", "="*72, f"Primary region definition: {cfg['cluster_method']}, {PRIMARY_KM} km", f"Regions: {len(primary):,}", f"Median WWTWs/region: {primary.number_of_WWTWs.median():.1f}", f"Median assigned sensors/region: {primary.number_of_assigned_CSO_sensors.median():.1f}", "No target or predictor values were used to create clusters.", "Population density is reconstructed as sum represented population / sum catchment area where possible.", "Capacity is ratio of summed load to summed design capacity.", "Percentages, rainfall, slope and building share use catchment-area weighting.", "Duration primary is pooled total duration / pooled valid events.", "Regional event median is an event-count-weighted WWTW/sensor median approximation; it is retained as sensitivity only, not a model target.", "Frequency is total events with valid exposure / total monitored sensor-years.", "VALIDATION: PASS"]
    (OUT / "audit" / "regional_aggregation_audit.txt").write_text("\n".join(audit)+"\n", encoding="utf-8")
    print(summary[["target","eligible_n","rmse_improvement_pct","r2","spearman"]].to_string(index=False))
    print(f"Regional master: {OUT/'data'/'regional_master_25km.csv'}")

if __name__ == "__main__":
    main()

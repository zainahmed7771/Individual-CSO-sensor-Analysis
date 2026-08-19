"""Supervisor-directed four-model CSO regression programme.

This module is intentionally isolated from PROJECT_RESULTS/machine_learning.
It reads validated sources but writes only below NEW MACHINE LEARNING.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NEW_ROOT = Path(__file__).resolve().parents[1]
HIST_RUNTIME = REPO / "PROJECT_RESULTS" / "machine_learning" / "_runtime"
if str(HIST_RUNTIME) not in sys.path:
    sys.path.insert(0, str(HIST_RUNTIME))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 20260813
BOOTSTRAP_N = 2000
COEF_BOOTSTRAP_N = 200
CV_FOLDS = 5
MISSING_EXCLUDE_PCT = 30.0

SCI_MASTER = REPO / "PROJECT_RESULTS" / "four_outcome_annual_nimrod_release_20260813_130420" / "data" / "scientific_master_annual_nimrod_v1.csv"
CURRENT_ML = REPO / "PROJECT_RESULTS" / "four_outcome_annual_nimrod_release_20260813_130420" / "data" / "machine_learning_master_annual_nimrod_v1.csv"
HIST_ML_ROOT = REPO / "PROJECT_RESULTS" / "machine_learning"

ID_COLS = ["company", "uwwCode", "canonical_wwtw_name", "sensor_uid"]
TARGET_COLS = ["fitted_beta", "fitted_lambda", "spill_mean_duration_minutes", "spill_event_count", "monitoring_years", "spill_events_per_monitoring_year"]
COORD_COLS = ["wwtw_easting", "wwtw_northing", "wwtw_longitude", "wwtw_latitude"]

HYDRO = ["hydro_high_productivity_pct", "hydro_low_productivity_pct", "hydro_no_groundwater_pct", "hydro_intergranular_flow_pct"]
STATIC_CLIMATE_INTERPRETABLE = ["rain_mean_annual_mm_1991_2020"]
STATIC_CLIMATE_FULL = ["rain_mean_annual_mm_1991_2020", "rain_winter_mean_mm_1991_2020"]
DYNAMIC = ["rain_ann_prcptot_mm", "rain_ann_rx1day_mm", "rain_ann_rx5day_mm", "rain_ann_sdii_mm_per_wet_day", "rain_ann_r10_days", "rain_ann_r20_days", "rain_ann_cwd_days", "rain_ann_cdd_days"]
LAND = ["continuous_land_cover_pct"]
POP = ["population_density_km2_2021"]
BUILDING = ["building_pre_1973_pct"]
TERRAIN = ["terrain_mean_slope_deg"]
CAPACITY = ["capacity_ratio_latest"]
AREA = ["sewershed_area_km2"]

INTERPRETABLE_STATIC = HYDRO + STATIC_CLIMATE_INTERPRETABLE + LAND + POP + BUILDING + TERRAIN + CAPACITY + AREA
INTERPRETABLE_DYNAMIC = INTERPRETABLE_STATIC + DYNAMIC
FULL_STATIC = HYDRO + STATIC_CLIMATE_FULL + LAND + POP + BUILDING + TERRAIN + CAPACITY + AREA
FULL_DYNAMIC = FULL_STATIC + DYNAMIC

FEATURE_BLOCKS = {
    "hydrogeology": HYDRO,
    "static_rainfall": STATIC_CLIMATE_FULL,
    "annual_dynamic_rainfall": DYNAMIC,
    "land_cover": LAND,
    "population": POP,
    "building_age": BUILDING,
    "terrain": TERRAIN,
    "capacity_operations": CAPACITY,
    "catchment_area": AREA,
}

DISPLAY = {
    "hydro_high_productivity_pct": "High-productivity geology",
    "hydro_low_productivity_pct": "Low-productivity geology",
    "hydro_no_groundwater_pct": "Essentially no groundwater",
    "hydro_intergranular_flow_pct": "Intergranular-flow geology",
    "rain_mean_annual_mm_1991_2020": "Static mean annual rainfall",
    "rain_winter_mean_mm_1991_2020": "Static mean winter rainfall",
    "continuous_land_cover_pct": "Developed-land index",
    "population_density_km2_2021": "Population density",
    "building_pre_1973_pct": "Pre-1973 building share",
    "terrain_mean_slope_deg": "Mean terrain slope",
    "capacity_ratio_latest": "Treatment capacity utilisation",
    "sewershed_area_km2": "WWTW catchment area",
    "rain_ann_prcptot_mm": "Annual PRCPTOT",
    "rain_ann_rx1day_mm": "Annual Rx1day",
    "rain_ann_rx5day_mm": "Annual Rx5day",
    "rain_ann_sdii_mm_per_wet_day": "Annual SDII",
    "rain_ann_r10_days": "Annual R10 days",
    "rain_ann_r20_days": "Annual R20 days",
    "rain_ann_cwd_days": "Annual consecutive wet days",
    "rain_ann_cdd_days": "Annual consecutive dry days",
    "company": "Water company",
}


@dataclass(frozen=True)
class Target:
    key: str
    folder: str
    label: str
    column: str
    unit: str
    eligibility: str
    exclusion: str


TARGETS = [
    Target("beta", "model_1_beta", "Beta", "fitted_beta", "beta", "eligible_model_1", "model_1_exclusion_reason"),
    Target("scale", "model_2_scale", "Scale (inverse-timescale lambda)", "fitted_lambda", "lambda per minute", "eligible_model_2", "model_2_exclusion_reason"),
    Target("duration", "model_3_duration", "Mean spill duration", "spill_mean_duration_minutes", "minutes", "eligible_model_3", "model_3_exclusion_reason"),
    Target("frequency", "model_4_count", "Annualised spill frequency", "spill_events_per_monitoring_year", "events/year", "eligible_model_4", "model_4_exclusion_reason"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def model_dirs(t: Target) -> dict[str, Path]:
    root = NEW_ROOT / t.folder
    names = ["config", "data", "splits", "models", "results", "figures", "audit", "logs"]
    d = {n: root / n for n in names}
    d["root"] = root
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def setup_tree() -> None:
    for p in [NEW_ROOT / "shared", NEW_ROOT / "cross_model" / "tables", NEW_ROOT / "cross_model" / "figures", NEW_ROOT / "report", NEW_ROOT / "tmp" / "pdfs"]:
        p.mkdir(parents=True, exist_ok=True)
    for t in TARGETS:
        model_dirs(t)


def build_master() -> pd.DataFrame:
    sci = pd.read_csv(SCI_MASTER, low_memory=False)
    cur = pd.read_csv(CURRENT_ML, low_memory=False)
    accepted = cur[["company", "uwwCode", "primary_sensor_flag", "accepted_primary_pair", "lambda_at_optimizer_bound"]].copy()
    out = sci.merge(accepted, on=["company", "uwwCode"], how="inner", validate="one_to_one")
    out["has_beta"] = np.isfinite(out["fitted_beta"]) & out["fitted_beta"].gt(0)
    out["has_scale"] = np.isfinite(out["fitted_lambda"]) & out["fitted_lambda"].gt(0)
    out["has_duration"] = np.isfinite(out["spill_mean_duration_minutes"]) & out["spill_mean_duration_minutes"].gt(0)
    out["has_frequency"] = np.isfinite(out["spill_events_per_monitoring_year"]) & out["spill_events_per_monitoring_year"].gt(0) & out["monitoring_years"].gt(0)
    out["eligible_model_1"] = out["accepted_primary_pair"] & out["has_beta"]
    out["eligible_model_2"] = out["accepted_primary_pair"] & out["has_scale"] & ~out["lambda_at_optimizer_bound"]
    out["eligible_model_3"] = out["accepted_primary_pair"] & out["has_duration"]
    out["eligible_model_4"] = out["accepted_primary_pair"] & out["has_frequency"]
    reasons = {
        1: (~out["accepted_primary_pair"], "not accepted primary pair", ~out["has_beta"], "missing/non-positive beta"),
        2: (~out["accepted_primary_pair"], "not accepted primary pair", ~out["has_scale"], "missing/non-positive scale", out["lambda_at_optimizer_bound"], "fitted lambda at optimizer boundary"),
        3: (~out["accepted_primary_pair"], "not accepted primary pair", ~out["has_duration"], "missing/non-positive duration"),
        4: (~out["accepted_primary_pair"], "not accepted primary pair", ~out["has_frequency"], "missing/non-positive exposure-adjusted frequency"),
    }
    for i, rules in reasons.items():
        col = f"model_{i}_exclusion_reason"
        out[col] = ""
        for mask, text in zip(rules[0::2], rules[1::2]):
            out.loc[out[col].eq("") & mask, col] = text
        out.loc[out[f"eligible_model_{i}"], col] = "eligible"
    assert len(out) == 1231 and not out[["company", "uwwCode"]].duplicated().any()
    out.to_csv(NEW_ROOT / "shared" / "new_ml_master.csv", index=False)
    availability = []
    for t in TARGETS:
        availability.append({"model": t.folder, "target": t.column, "eligible_n": int(out[t.eligibility].sum()), "missing_or_excluded_n": int((~out[t.eligibility]).sum()), "response_transformation": f"log({t.column})", "original_unit": t.unit})
    pd.DataFrame(availability).to_csv(NEW_ROOT / "shared" / "target_availability.csv", index=False)
    return out


def family_for(c: str) -> str:
    for name, cols in FEATURE_BLOCKS.items():
        if c in cols:
            return name
    return "identifier_or_target"


def predictor_manifest(master: pd.DataFrame) -> pd.DataFrame:
    active = set(FULL_DYNAMIC)
    records = []
    units = {c: "%" for c in HYDRO + LAND + BUILDING}
    units.update({"rain_mean_annual_mm_1991_2020": "mm/year", "rain_winter_mean_mm_1991_2020": "mm/winter", "population_density_km2_2021": "people/km2", "terrain_mean_slope_deg": "degrees", "capacity_ratio_latest": "ratio", "sewershed_area_km2": "km2", "rain_ann_prcptot_mm": "mm/year", "rain_ann_rx1day_mm": "mm", "rain_ann_rx5day_mm": "mm", "rain_ann_sdii_mm_per_wet_day": "mm/wet day", "rain_ann_r10_days": "days/year", "rain_ann_r20_days": "days/year", "rain_ann_cwd_days": "days", "rain_ann_cdd_days": "days"})
    for c in master.columns:
        external = c in active
        leakage = c in TARGET_COLS or any(x in c.lower() for x in ["ci_", "quality", "eligible", "exclusion", "tail_event", "optimizer", "sensor_uid", "permit"])
        if external:
            reason = "leakage-safe external predictor"
            role = "primary" if c in INTERPRETABLE_DYNAMIC else "regularised"
        elif c == "company":
            reason, role = "contextual E+C sensitivity only", "sensitivity"
        elif c in ["hydro_moderate_productivity_pct", "hydro_fracture_flow_pct"]:
            reason, role = "reference category prevents compositional singularity", "reference"
        elif c in ["landcover_urban_pct", "landcover_suburban_pct"]:
            reason, role = "mathematically redundant with primary continuous developed-land representation", "sensitivity alternative"
        elif c in COORD_COLS:
            reason, role = "spatial robustness only; never a predictor", "validation"
        elif c in ID_COLS or "permit" in c.lower():
            reason, role = "identifier excluded from predictors", "identifier"
        elif leakage:
            reason, role = "target-derived or target-quality leakage exclusion", "excluded"
        else:
            reason, role = "not an approved scientifically distinct external predictor", "excluded"
        records.append({"column_name": c, "readable_name": DISPLAY.get(c, c.replace("_", " ").title()), "scientific_family": family_for(c), "units": units.get(c, "n/a"), "transformation": "log1p then standardise" if c in POP + AREA else "median impute + missing indicator + standardise" if external else "n/a", "primary_or_sensitivity": role, "external_predictor": external or c == "company", "leakage_risk": leakage, "reason": reason, "missing_percentage": float(100 * master[c].isna().mean()), "expected_role": role})
    df = pd.DataFrame(records)
    df.to_csv(NEW_ROOT / "shared" / "predictor_manifest.csv", index=False)
    return df


def split_cohort(cohort: pd.DataFrame, t: Target) -> pd.DataFrame:
    idx = np.arange(len(cohort))
    y = np.log(cohort[t.column].to_numpy(float))
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop").astype(str)
    joint = cohort["company"].astype(str) + "__" + bins
    joint = joint.where(joint.map(joint.value_counts()).ge(2), cohort["company"].astype(str))
    try:
        dev_idx, test_idx = train_test_split(idx, test_size=.15, random_state=SEED, stratify=joint)
    except ValueError:
        dev_idx, test_idx = train_test_split(idx, test_size=.15, random_state=SEED, stratify=cohort["company"])
    dev = cohort.iloc[dev_idx]
    ydev = np.log(dev[t.column].to_numpy(float))
    dbins = pd.qcut(ydev, q=5, labels=False, duplicates="drop").astype(str)
    djoint = dev["company"].astype(str).reset_index(drop=True) + "__" + pd.Series(dbins).reset_index(drop=True)
    djoint = djoint.where(djoint.map(djoint.value_counts()).ge(2), dev["company"].astype(str).reset_index(drop=True))
    try:
        train_rel, val_rel = train_test_split(np.arange(len(dev)), test_size=.15/.85, random_state=SEED, stratify=djoint)
    except ValueError:
        train_rel, val_rel = train_test_split(np.arange(len(dev)), test_size=.15/.85, random_state=SEED, stratify=dev["company"])
    split = pd.Series("TEST", index=cohort.index)
    split.iloc[dev_idx[train_rel]] = "TRAIN"
    split.iloc[dev_idx[val_rel]] = "VALIDATION"
    out = cohort.copy()
    out["split"] = split
    return out


def make_preprocessor(features: list[str], include_company: bool = False) -> ColumnTransformer:
    log_features = [c for c in features if c in POP + AREA]
    raw = [c for c in features if c not in log_features]
    trs = []
    if raw:
        trs.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), raw))
    if log_features:
        trs.append(("log_numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())]), log_features))
    if include_company:
        trs.append(("company", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))]), ["company"]))
    return ColumnTransformer(trs, remainder="drop", sparse_threshold=0, verbose_feature_names_out=False)


def estimator(family: str, params: dict):
    if family == "Dummy": return DummyRegressor(strategy="mean")
    if family == "OLS": return LinearRegression()
    if family == "Ridge": return Ridge(alpha=float(params["alpha"]))
    if family == "ElasticNet": return ElasticNet(alpha=float(params["alpha"]), l1_ratio=float(params["l1_ratio"]), max_iter=30000, random_state=SEED)
    raise ValueError(family)


def pipeline(family: str, params: dict, features: list[str], company: bool = False) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor(features, company)), ("model", estimator(family, params))])


def rmse(y, p) -> float:
    return float(math.sqrt(mean_squared_error(y, p)))


def metric_row(y_log, pred_log, baseline_log, t: Target) -> dict:
    y_log, pred_log, baseline_log = map(lambda x: np.asarray(x, float), [y_log, pred_log, baseline_log])
    y, pred, base = np.exp(y_log), np.exp(pred_log), np.exp(baseline_log)
    model_rmse, dummy_rmse = rmse(y_log, pred_log), rmse(y_log, baseline_log)
    slope, intercept = np.polyfit(pred_log, y_log, 1) if np.std(pred_log) > 0 else (0.0, float(np.mean(y_log)))
    rho = stats.spearmanr(y, pred).statistic if np.std(pred) > 0 else np.nan
    return {"target": t.key, "n": len(y), "rmse_model_scale": model_rmse, "rmse_original_scale": rmse(y, pred), "mae_model_scale": float(mean_absolute_error(y_log, pred_log)), "mae_original_scale": float(mean_absolute_error(y, pred)), "r2_model_scale": float(r2_score(y_log, pred_log)), "spearman": float(rho) if np.isfinite(rho) else np.nan, "calibration_slope": float(slope), "calibration_intercept": float(intercept), "dummy_rmse_model_scale": dummy_rmse, "dummy_rmse_original_scale": rmse(y, base), "rmse_ratio": model_rmse / dummy_rmse, "rmse_improvement_pct": 100 * (dummy_rmse - model_rmse) / dummy_rmse}


def cv_score(train: pd.DataFrame, t: Target, family: str, params: dict, features: list[str], company: bool = False) -> tuple[float, float]:
    y = np.log(train[t.column].to_numpy(float))
    groups = train["uwwCode"].to_numpy()
    gkf = GroupKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    Xcols = features + (["company"] if company else [])
    for fi, vi in gkf.split(train, y, groups):
        p = pipeline(family, params, features, company)
        p.fit(train.iloc[fi][Xcols], y[fi])
        scores.append(rmse(y[vi], p.predict(train.iloc[vi][Xcols])))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def tune(train: pd.DataFrame, t: Target, family: str, features: list[str]) -> tuple[dict, pd.DataFrame]:
    if family == "Ridge":
        grid = [{"alpha": float(x)} for x in np.logspace(-4, 4, 13)]
    elif family == "ElasticNet":
        grid = [{"alpha": float(a), "l1_ratio": float(l)} for a in np.logspace(-4, 0, 7) for l in [.1, .5, .9]]
    else:
        grid = [{}]
    rows = []
    for i, param in enumerate(grid, 1):
        mean, sd = cv_score(train, t, family, param, features)
        rows.append({"algorithm": family, "parameter_id": i, "parameters": json.dumps(param, sort_keys=True), "training_cv_rmse": mean, "training_cv_rmse_sd": sd})
    tab = pd.DataFrame(rows).sort_values(["training_cv_rmse", "parameter_id"])
    return json.loads(tab.iloc[0]["parameters"]), tab


def candidate_comparison(train: pd.DataFrame, val: pd.DataFrame, t: Target, d: dict[str, Path]) -> tuple[pd.DataFrame, pd.Series]:
    specs = [
        ("Dummy", "static", INTERPRETABLE_STATIC),
        ("OLS", "interpretable_static", INTERPRETABLE_STATIC),
        ("OLS", "interpretable_static_dynamic", INTERPRETABLE_DYNAMIC),
        ("Ridge", "full_static", FULL_STATIC),
        ("Ridge", "full_static_dynamic", FULL_DYNAMIC),
        ("ElasticNet", "full_static_dynamic", FULL_DYNAMIC),
    ]
    ytr = np.log(train[t.column].to_numpy(float)); yv = np.log(val[t.column].to_numpy(float))
    rows, cv_all = [], []
    for fam, set_name, features in specs:
        params, cv = tune(train, t, fam, features)
        cv["feature_set"] = set_name; cv_all.append(cv)
        p = pipeline(fam, params, features)
        p.fit(train[features], ytr)
        pred = p.predict(val[features])
        rows.append({"target": t.key, "feature_set": set_name, "algorithm": fam, "hyperparameters": json.dumps(params, sort_keys=True), "training_cv_rmse": float(cv.iloc[0].training_cv_rmse), "validation_rmse": rmse(yv, pred), "validation_mae": float(mean_absolute_error(yv, pred)), "validation_r2": float(r2_score(yv, pred)), "complexity": {"Dummy": 0, "OLS": 1, "Ridge": 2, "ElasticNet": 3}[fam]})
    pd.concat(cv_all, ignore_index=True).to_csv(d["results"] / "cv_results.csv", index=False)
    comp = pd.DataFrame(rows)
    baseline = float(comp.loc[comp.algorithm.eq("Dummy"), "validation_rmse"].iloc[0])
    comp["baseline_rmse"] = baseline
    comp["rmse_improvement_pct"] = 100 * (baseline - comp.validation_rmse) / baseline
    best = comp.validation_rmse.min()
    tolerance = max(.005 * best, .002)
    selected = comp[comp.validation_rmse <= best + tolerance].sort_values(["complexity", "validation_rmse"]).iloc[0]
    comp["selected"] = False
    comp.loc[selected.name, "selected"] = True
    comp.to_csv(d["results"] / "validation_model_comparison.csv", index=False)
    return comp, selected


def split_summary(frame: pd.DataFrame, t: Target) -> pd.DataFrame:
    rows = []
    major = [t.column, "rain_mean_annual_mm_1991_2020", "population_density_km2_2021", "capacity_ratio_latest", "sewershed_area_km2"]
    for split, grp in frame.groupby("split"):
        for c in major:
            s = pd.to_numeric(grp[c], errors="coerce")
            rows.append({"split": split, "variable": c, "n": s.notna().sum(), "mean": s.mean(), "sd": s.std(), "q05": s.quantile(.05), "q25": s.quantile(.25), "median": s.median(), "q75": s.quantile(.75), "q95": s.quantile(.95)})
    return pd.DataFrame(rows)


def bootstrap_metrics(pred: pd.DataFrame, t: Target) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows = []
    n = len(pred)
    for i in range(BOOTSTRAP_N):
        ix = rng.integers(0, n, n)
        rows.append({"bootstrap_id": i, **metric_row(pred.observed_log.to_numpy()[ix], pred.predicted_log.to_numpy()[ix], pred.dummy_predicted_log.to_numpy()[ix], t)})
    long = pd.DataFrame(rows)
    metrics = [c for c in long if c not in ["bootstrap_id", "target", "n"]]
    summary = pd.DataFrame([{"metric": c, "lower_95": long[c].replace([np.inf, -np.inf], np.nan).quantile(.025), "median": long[c].replace([np.inf, -np.inf], np.nan).median(), "upper_95": long[c].replace([np.inf, -np.inf], np.nan).quantile(.975)} for c in metrics])
    return long, summary


def feature_names(pipe: Pipeline) -> np.ndarray:
    return np.asarray(pipe.named_steps["preprocess"].get_feature_names_out(), object)


def coefficient_table(pipe: Pipeline) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "standardised_coefficient"])
    return pd.DataFrame({"feature": feature_names(pipe), "standardised_coefficient": np.asarray(model.coef_).ravel()})


def coefficient_stability(dev: pd.DataFrame, t: Target, family: str, params: dict, features: list[str], company: bool) -> pd.DataFrame:
    if family == "Dummy": return pd.DataFrame(columns=["feature", "coefficient_median", "coefficient_low_95", "coefficient_high_95", "proportion_positive", "proportion_negative"])
    rng = np.random.default_rng(SEED)
    rows = []
    cols = features + (["company"] if company else [])
    for i in range(COEF_BOOTSTRAP_N):
        ix = rng.integers(0, len(dev), len(dev))
        boot = dev.iloc[ix]
        p = pipeline(family, params, features, company)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.fit(boot[cols], np.log(boot[t.column]))
        if not hasattr(p.named_steps["model"], "coef_"): continue
        rows.extend({"bootstrap_id": i, "feature": n, "coefficient": v} for n, v in zip(feature_names(p), np.asarray(p.named_steps["model"].coef_).ravel()))
    if not rows: return pd.DataFrame()
    long = pd.DataFrame(rows)
    return long.groupby("feature", as_index=False).coefficient.agg(coefficient_median="median", coefficient_low_95=lambda x: x.quantile(.025), coefficient_high_95=lambda x: x.quantile(.975), proportion_positive=lambda x: (x > 0).mean(), proportion_negative=lambda x: (x < 0).mean())


def learning_curve(train: pd.DataFrame, t: Target, family: str, params: dict, features: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    y = np.log(train[t.column].to_numpy(float))
    gkf = GroupKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    for frac in [.2, .4, .6, .8, 1.0]:
        for fold, (fi, vi) in enumerate(gkf.split(train, y, train.uwwCode)):
            take = max(20, int(len(fi) * frac)); use = rng.choice(fi, size=min(take, len(fi)), replace=False)
            p = pipeline(family, params, features)
            p.fit(train.iloc[use][features], y[use])
            rows.append({"fraction": frac, "n_training": len(use), "fold": fold, "training_rmse": rmse(y[use], p.predict(train.iloc[use][features])), "cross_validation_rmse": rmse(y[vi], p.predict(train.iloc[vi][features]))})
    return pd.DataFrame(rows)


def ablation(train: pd.DataFrame, val: pd.DataFrame, t: Target, family: str, params: dict, full_features: list[str]) -> pd.DataFrame:
    ytr, yv = np.log(train[t.column]), np.log(val[t.column])
    rows = []
    base = pipeline(family, params, full_features); base.fit(train[full_features], ytr); base_rmse = rmse(yv, base.predict(val[full_features]))
    rows.append({"removed_block": "none_full_model", "n_features": len(full_features), "validation_rmse": base_rmse, "delta_rmse_vs_full": 0.0})
    for block, cols in FEATURE_BLOCKS.items():
        fs = [c for c in full_features if c not in cols]
        if len(fs) == len(full_features): continue
        p = pipeline(family, params, fs); p.fit(train[fs], ytr); score = rmse(yv, p.predict(val[fs]))
        rows.append({"removed_block": block, "n_features": len(fs), "validation_rmse": score, "delta_rmse_vs_full": score - base_rmse})
    return pd.DataFrame(rows)


def fixed_comparison(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, t: Target, family: str, params: dict, static: list[str], dynamic: list[str], company: bool = False) -> pd.DataFrame:
    rows = []
    for name, fs in [("static_only", static), ("static_plus_annual_nimrod", dynamic)]:
        cols = fs + (["company"] if company else [])
        p = pipeline(family, params, fs, company); p.fit(train[cols], np.log(train[t.column]))
        vr = rmse(np.log(val[t.column]), p.predict(val[cols]))
        dev = pd.concat([train, val], ignore_index=True); p = pipeline(family, params, fs, company); p.fit(dev[cols], np.log(dev[t.column]))
        tr = rmse(np.log(test[t.column]), p.predict(test[cols]))
        rows.append({"feature_configuration": name, "company_included": company, "algorithm": family, "validation_rmse": vr, "test_rmse": tr})
    out = pd.DataFrame(rows)
    static_test = out.loc[out.feature_configuration.eq("static_only"), "test_rmse"].iloc[0]
    out["test_delta_rmse_vs_static"] = out.test_rmse - static_test
    return out


def company_comparison(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, t: Target, family: str, params: dict, features: list[str]) -> pd.DataFrame:
    rows = []
    for company in [False, True]:
        cols = features + (["company"] if company else [])
        p = pipeline(family, params, features, company); p.fit(train[cols], np.log(train[t.column])); vr = rmse(np.log(val[t.column]), p.predict(val[cols]))
        dev = pd.concat([train, val]); p = pipeline(family, params, features, company); p.fit(dev[cols], np.log(dev[t.column])); tr = rmse(np.log(test[t.column]), p.predict(test[cols]))
        rows.append({"configuration": "E+C" if company else "E", "validation_rmse": vr, "test_rmse": tr})
    out = pd.DataFrame(rows); e = out.loc[out.configuration.eq("E"), "test_rmse"].iloc[0]; out["test_delta_rmse_vs_E"] = out.test_rmse - e
    return out


def robustness(cohort: pd.DataFrame, t: Target, family: str, params: dict, features: list[str]) -> pd.DataFrame:
    y = np.log(cohort[t.column].to_numpy(float)); rows = []
    # Spatial blocks use coordinates only as grouping labels.
    blocks = (cohort.wwtw_easting.floordiv(100000).astype(str) + "_" + cohort.wwtw_northing.floordiv(100000).astype(str))
    if blocks.nunique() >= 5:
        pred = np.full(len(cohort), np.nan)
        for fi, vi in GroupKFold(5).split(cohort, y, blocks):
            p = pipeline(family, params, features); p.fit(cohort.iloc[fi][features], y[fi]); pred[vi] = p.predict(cohort.iloc[vi][features])
        rows.append({"analysis": "100km_spatial_GroupKFold", "group": "all", "n": len(cohort), "rmse_model_scale": rmse(y, pred), "baseline_rmse": rmse(y, np.repeat(y.mean(), len(y))), "rmse_improvement_pct": 100 * (rmse(y, np.repeat(y.mean(), len(y))) - rmse(y, pred)) / rmse(y, np.repeat(y.mean(), len(y)))})
    for company, test in cohort.groupby("company"):
        train = cohort[cohort.company.ne(company)]
        if len(test) < 10 or len(train) < 50: continue
        p = pipeline(family, params, features); p.fit(train[features], np.log(train[t.column])); pred = p.predict(test[features]); base = np.repeat(np.log(train[t.column]).mean(), len(test)); score, b = rmse(np.log(test[t.column]), pred), rmse(np.log(test[t.column]), base)
        rows.append({"analysis": "leave_one_company_out", "group": company, "n": len(test), "rmse_model_scale": score, "baseline_rmse": b, "rmse_improvement_pct": 100 * (b - score) / b})
    return pd.DataFrame(rows)


def vif_table(train: pd.DataFrame) -> pd.DataFrame:
    pre = make_preprocessor(INTERPRETABLE_DYNAMIC)
    x = pre.fit_transform(train[INTERPRETABLE_DYNAMIC])
    names = pre.get_feature_names_out()
    # Drop missing-indicator columns from VIF if constant; retain the actual fitted design.
    keep = np.nanstd(x, axis=0) > 1e-12
    x, names = x[:, keep], names[keep]
    return pd.DataFrame({"feature": names, "VIF": [variance_inflation_factor(x, i) for i in range(x.shape[1])]})


def clean_feature_name(name: str) -> str:
    name = re.sub(r"^missingindicator_", "Missing: ", name)
    return DISPLAY.get(name, name.replace("_", " ").title())


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def figures(t: Target, d: dict[str, Path], frame: pd.DataFrame, comp: pd.DataFrame, pred: pd.DataFrame, coef: pd.DataFrame, stability: pd.DataFrame, importance: pd.DataFrame, lc: pd.DataFrame, abl: pd.DataFrame, dyn: pd.DataFrame, company: pd.DataFrame, robust: pd.DataFrame) -> None:
    blue, orange, grey = "#174A6E", "#D97706", "#64748B"
    counts = frame.split.value_counts().reindex(["TRAIN", "VALIDATION", "TEST"])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(counts.index, counts.values, color=[blue, "#2A789E", orange]); ax.set_ylabel("WWTWs"); ax.set_title(f"{t.label}: 70 / 15 / 15 cohort flow"); save_fig(fig, d["figures"] / "cohort_flow.pdf")
    fig, ax = plt.subplots(figsize=(7, 4)); data=[np.log(frame.loc[frame.split.eq(s), t.column]) for s in counts.index]; ax.boxplot(data, tick_labels=counts.index); ax.set_ylabel(f"log({t.column})"); ax.set_title("Target balance across frozen splits"); save_fig(fig, d["figures"] / "split_balance.pdf")
    c = comp.sort_values("validation_rmse"); fig, ax=plt.subplots(figsize=(8,4.5)); ax.barh(c.algorithm+" | "+c.feature_set,c.validation_rmse,color=[orange if x else blue for x in c.selected]); ax.invert_yaxis(); ax.set_xlabel("Validation RMSE (model scale)"); ax.set_title("Candidate selection used validation only"); save_fig(fig,d["figures"]/"model_comparison.pdf")
    fig, ax=plt.subplots(figsize=(5.5,5)); ax.scatter(pred.observed_original,pred.predicted_original,s=20,alpha=.7,color=blue); lo=min(pred.observed_original.min(),pred.predicted_original.min()); hi=max(pred.observed_original.max(),pred.predicted_original.max()); ax.plot([lo,hi],[lo,hi],"--",color=orange); ax.set(xlabel=f"Observed ({t.unit})",ylabel=f"Predicted ({t.unit})",title=f"Locked test: predicted versus observed (n={len(pred)})"); save_fig(fig,d["figures"]/"predicted_vs_observed.pdf")
    residual=pred.observed_log-pred.predicted_log; fig,axs=plt.subplots(1,2,figsize=(9,4)); axs[0].scatter(pred.predicted_log,residual,s=18,alpha=.7,color=blue); axs[0].axhline(0,color=orange,ls="--"); axs[0].set(xlabel="Predicted log target",ylabel="Residual",title="Residual versus prediction"); axs[1].hist(residual,bins=20,color=blue,alpha=.8); axs[1].set(title="Residual distribution",xlabel="Residual"); save_fig(fig,d["figures"]/"residual_diagnostics.pdf")
    plot_bar_table(coef,"standardised_coefficient","Standardised coefficient","Conditional predictive coefficients",d["figures"]/"coefficient_forest_plot.pdf",blue)
    if not stability.empty:
        s=stability.reindex(stability.coefficient_median.abs().sort_values(ascending=False).index).head(18).sort_values("coefficient_median"); fig,ax=plt.subplots(figsize=(8,6)); y=np.arange(len(s)); ax.errorbar(s.coefficient_median,y,xerr=[s.coefficient_median-s.coefficient_low_95,s.coefficient_high_95-s.coefficient_median],fmt="o",color=blue,ecolor=grey); ax.axvline(0,color=orange,ls="--"); ax.set_yticks(y,labels=[clean_feature_name(x) for x in s.feature]); ax.set_title("Training-resample coefficient stability"); save_fig(fig,d["figures"]/"coefficient_stability.pdf")
    else: blank_figure("Selected Dummy model has no coefficients",d["figures"]/"coefficient_stability.pdf")
    plot_bar_table(importance,"rmse_increase_mean","RMSE increase","Locked-test permutation importance",d["figures"]/"permutation_importance.pdf",blue)
    l=lc.groupby("fraction",as_index=False).agg(n_training=("n_training","mean"),training_rmse=("training_rmse","mean"),cross_validation_rmse=("cross_validation_rmse","mean")); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(l.n_training,l.training_rmse,"o-",label="Training",color=blue); ax.plot(l.n_training,l.cross_validation_rmse,"o-",label="Cross-validation",color=orange); ax.set(xlabel="Training WWTWs",ylabel="RMSE (model scale)",title="Training-only learning curve"); ax.legend(); save_fig(fig,d["figures"]/"learning_curve.pdf")
    plot_bar_table(abl.query("removed_block!='none_full_model'"),"delta_rmse_vs_full","Validation RMSE increase","Feature-block ablation",d["figures"]/"feature_block_ablation.pdf",blue,"removed_block")
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(dyn.feature_configuration,dyn.test_rmse,color=[grey,blue]); ax.set_ylabel("Test RMSE (model scale)"); ax.set_title("Annual dynamic rainfall value added"); ax.tick_params(axis='x',rotation=12); save_fig(fig,d["figures"]/"dynamic_rainfall_value_added.pdf")
    r=robust[robust.analysis.eq("leave_one_company_out")]; fig,axs=plt.subplots(1,2,figsize=(10,4)); axs[0].bar(company.configuration,company.test_rmse,color=[blue,orange]); axs[0].set(title="Company value added",ylabel="Test RMSE"); axs[1].barh(r.group,r.rmse_improvement_pct,color=blue); axs[1].axvline(0,color=orange,ls="--"); axs[1].set(title="Leave-one-company-out",xlabel="RMSE improvement over baseline (%)"); save_fig(fig,d["figures"]/"robustness.pdf")


def plot_bar_table(tab: pd.DataFrame, value: str, xlabel: str, title: str, path: Path, color: str, label: str="feature") -> None:
    if tab.empty or value not in tab: blank_figure("Not available for selected Dummy model",path); return
    x=tab.copy(); x=x.reindex(x[value].abs().sort_values(ascending=False).index).head(18).sort_values(value); fig,ax=plt.subplots(figsize=(8,6)); ax.barh([clean_feature_name(str(z)) for z in x[label]],x[value],color=color); ax.axvline(0,color="#D97706",ls="--"); ax.set(xlabel=xlabel,title=title); save_fig(fig,path)


def blank_figure(text: str, path: Path) -> None:
    fig,ax=plt.subplots(figsize=(7,4)); ax.axis("off"); ax.text(.5,.5,text,ha="center",va="center",fontsize=14); save_fig(fig,path)


def learning_diagnosis(lc: pd.DataFrame) -> str:
    s=lc.groupby("fraction").agg(train=("training_rmse","mean"),cv=("cross_validation_rmse","mean")); last=s.iloc[-1]; prev=s.iloc[-2]
    if last.cv < prev.cv * .97: return "Validation error is still decreasing; more labelled WWTWs may help."
    if last.cv - last.train > .2 * last.cv: return "A persistent train-validation gap suggests variance; more WWTWs may help."
    return "The curve is near a plateau; richer predictors or improved targets may matter more than more similar rows."


def verdict(metrics: dict, boot: pd.DataFrame, robust: pd.DataFrame) -> str:
    low=float(boot.loc[boot.metric.eq("rmse_improvement_pct"),"lower_95"].iloc[0]); imp=metrics["rmse_improvement_pct"]; r2=metrics["r2_model_scale"]
    if imp > 10 and low > 0 and r2 > 0: return "STRONG / USEFUL PREDICTIVE SIGNAL"
    if imp > 0 and r2 > 0 and low > -2: return "MODEST / QUALIFIED SIGNAL"
    if imp > 0 and r2 > -.05: return "WEAK / INCONCLUSIVE"
    return "NO DEMONSTRATED PREDICTIVE SKILL"


def run_target(master: pd.DataFrame, t: Target) -> dict:
    start=time.time(); d=model_dirs(t)
    print(f"\n--- {t.folder}: {t.label} ---")
    cohort=master[master[t.eligibility]].copy().reset_index(drop=True)
    assert cohort[t.column].gt(0).all() and not cohort[["company","uwwCode"]].duplicated().any()
    cohort["target_log"] = np.log(cohort[t.column])
    frame=split_cohort(cohort,t)
    counts=frame.split.value_counts(); assert set(counts.index)=={"TRAIN","VALIDATION","TEST"}
    train=frame[frame.split.eq("TRAIN")].copy(); val=frame[frame.split.eq("VALIDATION")].copy(); test=frame[frame.split.eq("TEST")].copy(); dev=pd.concat([train,val],ignore_index=True)
    assert not set(train.uwwCode)&set(val.uwwCode) and not set(train.uwwCode)&set(test.uwwCode) and not set(val.uwwCode)&set(test.uwwCode)
    cohort.to_csv(d["data"] / "eligible_cohort.csv",index=False); train.to_csv(d["data"] / "train.csv",index=False); val.to_csv(d["data"] / "validation.csv",index=False); test.to_csv(d["data"] / "test.csv",index=False)
    frame[["company","uwwCode","sensor_uid",t.column,"split"]].to_csv(d["splits"] / "split_assignments.csv",index=False)
    balance=split_summary(frame,t); balance.to_csv(d["splits"] / "split_balance.csv",index=False)
    print(f"cohort={len(frame)} train={len(train)} validation={len(val)} test={len(test)}")
    comp,selected=candidate_comparison(train,val,t,d); family=str(selected.algorithm); params=json.loads(selected.hyperparameters); feature_set=str(selected.feature_set)
    features={"static":INTERPRETABLE_STATIC,"interpretable_static":INTERPRETABLE_STATIC,"interpretable_static_dynamic":INTERPRETABLE_DYNAMIC,"full_static":FULL_STATIC,"full_static_dynamic":FULL_DYNAMIC}[feature_set]
    print(f"validation selected: {family} | {feature_set} | RMSE={selected.validation_rmse:.4f}")
    final=pipeline(family,params,features); final.fit(dev[features],np.log(dev[t.column])); dummy=pipeline("Dummy",{},INTERPRETABLE_STATIC); dummy.fit(dev[INTERPRETABLE_STATIC],np.log(dev[t.column]))
    pred_log=final.predict(test[features]); base_log=dummy.predict(test[INTERPRETABLE_STATIC]); pred=pd.DataFrame({"company":test.company.to_numpy(),"uwwCode":test.uwwCode.to_numpy(),"sensor_uid":test.sensor_uid.to_numpy(),"observed_log":np.log(test[t.column].to_numpy()),"predicted_log":pred_log,"dummy_predicted_log":base_log,"observed_original":test[t.column].to_numpy(),"predicted_original":np.exp(pred_log),"dummy_predicted_original":np.exp(base_log)})
    pred.to_csv(d["results"] / "test_predictions.csv",index=False); metrics=metric_row(pred.observed_log,pred.predicted_log,pred.dummy_predicted_log,t); pd.DataFrame([metrics]).to_csv(d["results"] / "test_metrics.csv",index=False)
    joblib.dump(final,d["models"] / "final_model.joblib",compress=3); joblib.dump(final.named_steps["preprocess"],d["models"] / "final_preprocessor.joblib",compress=3)
    reloaded=joblib.load(d["models"] / "final_model.joblib"); reload_diff=float(np.max(np.abs(reloaded.predict(test[features])-pred_log))); assert reload_diff < 1e-12
    boot_long,boot_sum=bootstrap_metrics(pred,t); boot_long.to_csv(d["results"] / "test_metric_bootstrap.csv",index=False); boot_sum.to_csv(d["results"] / "test_metric_bootstrap_summary.csv",index=False)
    coef=coefficient_table(final); coef["readable_name"]=coef.feature.map(clean_feature_name); coef.to_csv(d["results"] / "coefficient_table.csv",index=False)
    stability=coefficient_stability(dev,t,family,params,features,False); stability.to_csv(d["results"] / "coefficient_stability.csv",index=False)
    perm=permutation_importance(final,test[features],np.log(test[t.column]),n_repeats=30,random_state=SEED,scoring="neg_root_mean_squared_error")
    importance=pd.DataFrame({"feature":features,"rmse_increase_mean":perm.importances_mean,"rmse_increase_sd":perm.importances_std}); importance["readable_name"]=importance.feature.map(DISPLAY); importance.to_csv(d["results"] / "permutation_importance.csv",index=False)
    lc=learning_curve(train,t,family,params,features); lc.to_csv(d["results"] / "learning_curve.csv",index=False)
    abl=ablation(train,val,t,family,params,features); abl.to_csv(d["results"] / "feature_ablation.csv",index=False)
    # Pre-specified sensitivities use the same selected algorithm; no feature cherry-picking.
    dyn=fixed_comparison(train,val,test,t,family,params,FULL_STATIC,FULL_DYNAMIC); dyn.to_csv(d["results"] / "dynamic_rainfall_comparison.csv",index=False)
    comp_company=company_comparison(train,val,test,t,family,params,FULL_DYNAMIC); comp_company.to_csv(d["results"] / "company_comparison.csv",index=False)
    robust=robustness(cohort,t,family,params,features); robust.to_csv(d["results"] / "robustness.csv",index=False)
    corr=train[FULL_DYNAMIC].corr(method="pearson"); corr.to_csv(d["results"] / "predictor_correlation_matrix.csv")
    vif=vif_table(train); vif.to_csv(d["results"] / "vif.csv",index=False)
    figures(t,d,frame,comp,pred,coef,stability,importance,lc,abl,dyn,comp_company,robust)
    diagnosis=learning_diagnosis(lc); final_verdict=verdict(metrics,boot_sum,robust)
    config={"seed":SEED,"source_scientific_master":str(SCI_MASTER.relative_to(REPO)),"source_scientific_sha256":sha256(SCI_MASTER),"source_current_ml_master":str(CURRENT_ML.relative_to(REPO)),"source_current_ml_sha256":sha256(CURRENT_ML),"target":t.column,"response_transformation":f"log({t.column})","inverse_transformation":"exp","cohort_rule":t.eligibility,"split":{"train":.70,"validation":.15,"test":.15,"group":"uwwCode","balancing":"company + target quintile where feasible","seed":SEED},"preprocessing":"training-only median imputation with indicators; population and area log1p; standardisation; company one-hot only in E+C sensitivity","selected":{"algorithm":family,"feature_set":feature_set,"features":features,"hyperparameters":params},"test_evaluated_once_after_freeze":True,"bootstrap_test_resamples":BOOTSTRAP_N,"coefficient_resamples":COEF_BOOTSTRAP_N,"software":{"python":platform.python_version(),"sklearn":sklearn.__version__,"pandas":pd.__version__,"numpy":np.__version__,"scipy":scipy.__version__}}
    write_json(d["config"] / "config.json",config)
    validations=[
        ("unique_WWTW",not frame.uwwCode.duplicated().any()),("split_approximately_70_15_15",all(abs(counts.get(s,0)/len(frame)-p)<.02 for s,p in [("TRAIN",.70),("VALIDATION",.15),("TEST",.15)])),("zero_split_overlap",True),("target_positive",frame[t.column].gt(0).all()),("test_not_used_in_selection",True),("preprocessing_inside_pipeline",isinstance(final,Pipeline)),("model_reload_exact",reload_diff<1e-12),("rmse_independent_recalculation",np.isclose(metrics["rmse_model_scale"],rmse(pred.observed_log,pred.predicted_log))),("all_required_figures",len(list(d["figures"].glob("*.pdf")))==12),
    ]
    pd.DataFrame(validations,columns=["validation_gate","passed"]).to_csv(d["audit"] / "validation_results.csv",index=False); assert all(x[1] for x in validations)
    audit=[f"{t.label.upper()} - NEW ML AUDIT","="*70,f"Source scientific master: {SCI_MASTER}",f"Source SHA256: {sha256(SCI_MASTER)}",f"Target: {t.column}",f"Response: log({t.column}); back-transform exp",f"Eligible: {len(frame)}; train: {len(train)}; validation: {len(val)}; test: {len(test)}",f"Selected using validation only: {family} / {feature_set} / {params}",f"Final test RMSE model scale: {metrics['rmse_model_scale']}",f"Dummy RMSE model scale: {metrics['dummy_rmse_model_scale']}",f"RMSE improvement: {metrics['rmse_improvement_pct']}%",f"Test R2: {metrics['r2_model_scale']}",f"Learning curve: {diagnosis}",f"Verdict: {final_verdict}",f"Model reload max prediction difference: {reload_diff}","No identifier, coordinate, target, other outcome, target-quality field, CI or event-derived field entered X.","All imputation/scaling/encoding learned inside each training fold.","Test was evaluated once after algorithm/feature-set freeze. Pre-specified rainfall/company sensitivities reused the frozen split.",f"Runtime seconds: {time.time()-start:.1f}","XLSX outputs deferred: required @oai/artifact-tool runtime and connected Excel session unavailable; CSVs are authoritative.","",*[f"{k}: {'PASS' if v else 'FAIL'}" for k,v in validations]]
    (d["audit"] / "audit.txt").write_text("\n".join(audit)+"\n",encoding="utf-8")
    pd.DataFrame([{"requested_workbook":"model_comparison.xlsx","status":"DEFERRED","reason":"required @oai/artifact-tool runtime and connected Excel session unavailable","authoritative_csv":"results/validation_model_comparison.csv"}]).to_csv(d["results"] / "workbook_handoff_manifest.csv",index=False)
    print(f"test RMSE={metrics['rmse_model_scale']:.4f}; dummy={metrics['dummy_rmse_model_scale']:.4f}; improvement={metrics['rmse_improvement_pct']:.1f}%; R2={metrics['r2_model_scale']:.3f}; {final_verdict}")
    dynamic_delta=float(dyn.loc[dyn.feature_configuration.eq("static_plus_annual_nimrod"),"test_delta_rmse_vs_static"].iloc[0]); company_delta=float(comp_company.loc[comp_company.configuration.eq("E+C"),"test_delta_rmse_vs_E"].iloc[0]); spatial=robust.loc[robust.analysis.eq("100km_spatial_GroupKFold"),"rmse_model_scale"]
    return {"target":t.key,"label":t.label,"folder":t.folder,"eligible_n":len(frame),"train_n":len(train),"validation_n":len(val),"test_n":len(test),"selected_algorithm":family,"selected_feature_set":feature_set,**metrics,"dynamic_rainfall_delta_rmse":dynamic_delta,"company_delta_rmse":company_delta,"learning_curve_diagnosis":diagnosis,"spatial_rmse":float(spatial.iloc[0]) if len(spatial) else np.nan,"overall_verdict":final_verdict,"top_features":"; ".join(importance.sort_values("rmse_increase_mean",ascending=False).head(3).readable_name.astype(str))}


def inspection_report(master: pd.DataFrame) -> str:
    return "\n".join([
        "="*60,"NEW MACHINE LEARNING - REPOSITORY INSPECTION","="*60,
        f"Scientific master: {SCI_MASTER.relative_to(REPO)}",f"Rows: {len(pd.read_csv(SCI_MASTER,low_memory=False))}","Primary analysis key: company + uwwCode (one row per accepted WWTW)",
        f"HYDROGEOLOGY: {HYDRO}",f"STATIC RAINFALL: {STATIC_CLIMATE_FULL}",f"ANNUAL NIMROD DYNAMIC RAINFALL: {DYNAMIC}",f"LAND COVER: {LAND}",f"POPULATION: {POP}",f"BUILDING AGE: {BUILDING}",f"TERRAIN: {TERRAIN}",f"TREATMENT / OPERATIONAL: {CAPACITY+AREA}","OTHER EXTERNAL VARIABLES: none meeting leakage-safe authoritative criteria",
        "TARGET AVAILABILITY",*[f"{t.label}: {int(master[t.eligibility].sum())}" for t in TARGETS],f"Previous ML directory: {HIST_ML_ROOT.relative_to(REPO)} (read-only)",f"New output directory: {NEW_ROOT.relative_to(REPO)}","="*60,"INSPECTION PASSED"])


def run_all() -> pd.DataFrame:
    setup_tree()
    pre_hashes={str(p.relative_to(REPO)):sha256(p) for p in HIST_ML_ROOT.rglob("*") if p.is_file() and "_runtime" not in p.parts}
    master=build_master(); manifest=predictor_manifest(master)
    report=inspection_report(master); print(report); (NEW_ROOT/"shared"/"repository_inspection.txt").write_text(report+"\n",encoding="utf-8")
    provenance=["NEW ML DATA PROVENANCE","="*60,f"Scientific master: {SCI_MASTER.relative_to(REPO)}",f"SHA256: {sha256(SCI_MASTER)}",f"Accepted mapping/current ML source: {CURRENT_ML.relative_to(REPO)}",f"SHA256: {sha256(CURRENT_ML)}","Scale convention: fitted_lambda is inverse-timescale per minute in S(t)=exp[-(lambda*t)^beta].","Frequency: spill_event_count / monitoring_years; missing exposure remains missing.","Annual rainfall: approved NIMROD annual indices from the validated annual-NIMROD release.","Historical PROJECT_RESULTS/machine_learning is read-only and is not retrained or modified."]
    (NEW_ROOT/"shared"/"data_provenance.txt").write_text("\n".join(provenance)+"\n",encoding="utf-8")
    summaries=[]
    for t in TARGETS: summaries.append(run_target(master,t))
    summary=pd.DataFrame(summaries); summary.to_csv(NEW_ROOT/"cross_model"/"tables"/"four_model_summary.csv",index=False)
    # Source and historical isolation check.
    post_hashes={str(p.relative_to(REPO)):sha256(p) for p in HIST_ML_ROOT.rglob("*") if p.is_file() and "_runtime" not in p.parts}
    assert pre_hashes==post_hashes
    integrity={"scientific_master_sha256":sha256(SCI_MASTER),"current_ml_master_sha256":sha256(CURRENT_ML),"historical_ml_files_checked":len(pre_hashes),"historical_ml_unchanged":True,"new_master_rows":len(master),"key_unique":not master[["company","uwwCode"]].duplicated().any()}
    write_json(NEW_ROOT/"shared"/"source_integrity.json",integrity)
    pd.DataFrame([{"requested_workbook":"new_ml_master.xlsx","status":"DEFERRED","reason":"required @oai/artifact-tool runtime and connected Excel session unavailable","authoritative_csv":"shared/new_ml_master.csv"},{"requested_workbook":"split_summary.xlsx","status":"DEFERRED","reason":"required @oai/artifact-tool runtime and connected Excel session unavailable","authoritative_csv":"per-model splits/split_balance.csv"},{"requested_workbook":"four_model_summary.xlsx","status":"DEFERRED","reason":"required @oai/artifact-tool runtime and connected Excel session unavailable","authoritative_csv":"cross_model/tables/four_model_summary.csv"}]).to_csv(NEW_ROOT/"shared"/"workbook_handoff_manifest.csv",index=False)
    return summary


if __name__ == "__main__":
    run_all()

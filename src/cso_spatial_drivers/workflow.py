"""Deterministic, teachable core workflow from standardised events to results."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from .cleaning.events import clean_events
from .config import PipelineConfig, validate_input_contracts
from .fitting.stretched_exponential import fit_tail
from .predictors.join import join_predictors
from .statistics.univariate import fit_log_outcome


STAGES = ("clean", "fit", "master", "univariate", "machine_learning", "report")
OUTCOMES = {
    "beta": "fitted_beta",
    "lambda": "fitted_lambda_per_minute",
    "duration": "spill_mean_duration_minutes",
    "frequency": "spill_events_per_monitoring_year",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_events(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No CSV event files found under {path}")
    frames = [pd.read_csv(item) for item in files]
    return pd.concat(frames, ignore_index=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_joblib(value: object, path: Path) -> None:
    """Serialize a model and replace the declared artifact only after success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def stage_clean(config: PipelineConfig) -> Path:
    output = config.output_root / "01_cleaning" / "clean_events.csv"
    cleaned = clean_events(read_events(config.paths["events"]))
    if cleaned.empty:
        raise ValueError("No valid events remain after cleaning")
    _write_csv(cleaned, output)
    return output


def stage_fit(config: PipelineConfig) -> Path:
    source = config.output_root / "01_cleaning" / "clean_events.csv"
    events = pd.read_csv(source)
    threshold = float(config.analysis.get("tail_threshold_minutes", 240))
    rows: list[dict[str, object]] = []
    for sensor_uid, group in events.groupby("sensor_uid", sort=True):
        durations = pd.to_numeric(group["duration_minutes"], errors="coerce")
        record: dict[str, object] = {
            "sensor_uid": sensor_uid,
            "company": str(group["company"].iloc[0]),
            "permit_number": str(group["permit_number"].iloc[0]),
            "spill_event_count": int(durations.notna().sum()),
            "spill_mean_duration_minutes": float(durations.mean()),
        }
        try:
            fitted = fit_tail(durations.to_numpy(), threshold=threshold)
            record.update(
                fitted_beta=fitted["beta"],
                fitted_lambda_per_minute=fitted["lambda_per_minute"],
                tail_spill_count=fitted["n_tail"],
                fit_success=fitted["success"],
                fit_status="success" if fitted["success"] else "optimizer_failed",
            )
        except ValueError as error:
            record.update(
                fitted_beta=np.nan,
                fitted_lambda_per_minute=np.nan,
                tail_spill_count=int((durations > threshold).sum()),
                fit_success=False,
                fit_status=str(error),
            )
        rows.append(record)
    output = config.output_root / "02_tail_fitting" / "sensor_outcomes.csv"
    _write_csv(pd.DataFrame(rows), output)
    return output


def _validate_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    missing = [column for column in keys if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame.duplicated(keys).any():
        raise ValueError(f"{label} contains duplicate keys: {keys}")


def stage_master(config: PipelineConfig) -> Path:
    outcomes = pd.read_csv(config.output_root / "02_tail_fitting" / "sensor_outcomes.csv")
    assignments = pd.read_csv(config.paths["assignments"], dtype={"company": "string", "uwwCode": "string"})
    exposure = pd.read_csv(config.paths["monitoring_exposure"])
    predictors = pd.read_csv(config.paths["predictors"], dtype={"company": "string", "uwwCode": "string"})
    _validate_unique(assignments, ["sensor_uid"], "assignments")
    _validate_unique(exposure, ["sensor_uid"], "monitoring_exposure")
    _validate_unique(predictors, ["company", "uwwCode"], "predictors")
    if "assignment_evidence" not in assignments:
        raise ValueError("assignments must include assignment_evidence")
    if assignments["assignment_evidence"].astype("string").fillna("").str.strip().eq("").any():
        raise ValueError("Every assignment requires non-empty assignment_evidence")
    master = outcomes.merge(assignments, on=["sensor_uid", "company"], how="left", validate="one_to_one")
    if master["uwwCode"].isna().any():
        missing = master.loc[master["uwwCode"].isna(), "sensor_uid"].tolist()
        raise ValueError(f"Missing accepted WWTW assignments for sensors: {missing[:10]}")
    master = master.merge(exposure, on="sensor_uid", how="left", validate="one_to_one")
    years = pd.to_numeric(master["monitoring_years"], errors="coerce")
    if years.isna().any() or (years <= 0).any():
        raise ValueError("monitoring_years must be finite and positive for every sensor")
    master["spill_events_per_monitoring_year"] = master["spill_event_count"] / years
    before_keys = master[["sensor_uid", "company", "uwwCode"]].copy()
    master = join_predictors(master, predictors)
    if not before_keys.equals(master[["sensor_uid", "company", "uwwCode"]]):
        raise AssertionError("Predictor join changed master row order or identity")
    output = config.output_root / "03_master" / "scientific_master.csv"
    _write_csv(master, output)
    return output


def stage_univariate(config: PipelineConfig) -> Path:
    master = pd.read_csv(config.output_root / "03_master" / "scientific_master.csv")
    rows: list[dict[str, object]] = []
    for outcome_name, outcome_column in OUTCOMES.items():
        for predictor in config.predictors:
            if predictor not in master:
                raise ValueError(f"Configured predictor missing from master: {predictor}")
            try:
                result = fit_log_outcome(master[outcome_column], master[predictor])
                result.update(outcome=outcome_name, outcome_column=outcome_column, predictor=predictor, status="success")
            except Exception as error:
                result = {"outcome": outcome_name, "outcome_column": outcome_column, "predictor": predictor, "status": f"failed: {error}", "n": 0, "slope": np.nan, "hc3_se": np.nan, "p_value": np.nan, "r_squared": np.nan}
            rows.append(result)
    results = pd.DataFrame(rows)
    results["fdr_q_value"] = np.nan
    for outcome, index in results.groupby("outcome").groups.items():
        usable = results.loc[index, "p_value"].notna()
        selected = results.loc[index].index[usable]
        if len(selected):
            results.loc[selected, "fdr_q_value"] = multipletests(results.loc[selected, "p_value"], method="fdr_bh")[1]
    output = config.output_root / "04_univariate" / "univariate_results.csv"
    _write_csv(results, output)
    return output


def _split_indices(size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if size < 12:
        raise ValueError("At least 12 eligible rows are required for train/validation/test demonstration")
    order = np.random.default_rng(seed).permutation(size)
    train_end = max(8, int(np.floor(size * 0.70)))
    validation_end = max(train_end + 2, int(np.floor(size * 0.85)))
    validation_end = min(validation_end, size - 2)
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def stage_machine_learning(config: PipelineConfig) -> Path:
    master = pd.read_csv(config.output_root / "03_master" / "scientific_master.csv")
    forbidden = set(OUTCOMES.values()) | {"sensor_uid", "company", "uwwCode", "permit_number"}
    overlap = forbidden.intersection(config.predictors)
    if overlap:
        raise ValueError(f"Forbidden leakage columns configured as predictors: {sorted(overlap)}")
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    model_dir = config.output_root / "05_machine_learning" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for offset, (outcome_name, outcome_column) in enumerate(OUTCOMES.items()):
        columns = [outcome_column, *config.predictors, "sensor_uid", "company", "uwwCode"]
        data = master[columns].copy()
        data[outcome_column] = pd.to_numeric(data[outcome_column], errors="coerce")
        data = data.loc[data[outcome_column].gt(0)].reset_index(drop=True)
        train, validation, test = _split_indices(len(data), config.seed + offset)
        X = data[list(config.predictors)].apply(pd.to_numeric, errors="coerce").to_numpy()
        y = np.log(data[outcome_column].to_numpy())
        candidates: list[tuple[float, float, Pipeline]] = []
        for alpha in (0.1, 1.0, 10.0, 100.0):
            model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
            model.fit(X[train], y[train])
            score = mean_squared_error(y[validation], model.predict(X[validation])) ** 0.5
            candidates.append((score, alpha, model))
        _, alpha, _ = min(candidates, key=lambda item: item[0])
        fit_index = np.r_[train, validation]
        model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]).fit(X[fit_index], y[fit_index])
        dummy = DummyRegressor(strategy="mean").fit(X[fit_index], y[fit_index])
        predicted = model.predict(X[test]); dummy_predicted = dummy.predict(X[test])
        rmse = mean_squared_error(y[test], predicted) ** 0.5
        dummy_rmse = mean_squared_error(y[test], dummy_predicted) ** 0.5
        rows.append({"outcome": outcome_name, "outcome_column": outcome_column, "n_eligible": len(data), "n_train": len(train), "n_validation": len(validation), "n_locked_test": len(test), "selected_alpha": alpha, "rmse_model_scale": rmse, "dummy_rmse_model_scale": dummy_rmse, "rmse_improvement_pct": 100 * (dummy_rmse - rmse) / dummy_rmse, "mae_model_scale": mean_absolute_error(y[test], predicted), "r2_model_scale": r2_score(y[test], predicted), "seed": config.seed + offset})
        _write_joblib(model, model_dir / f"{outcome_name}_ridge.joblib")
        prediction_frames.append(pd.DataFrame({"outcome": outcome_name, "sensor_uid": data.loc[test, "sensor_uid"].to_numpy(), "company": data.loc[test, "company"].to_numpy(), "uwwCode": data.loc[test, "uwwCode"].to_numpy(), "observed_model_scale": y[test], "predicted_model_scale": predicted, "dummy_predicted_model_scale": dummy_predicted, "split": "locked_test"}))
    output_dir = config.output_root / "05_machine_learning"
    _write_csv(pd.DataFrame(rows), output_dir / "model_scorecard.csv")
    _write_csv(pd.concat(prediction_frames, ignore_index=True), output_dir / "locked_test_predictions.csv")
    return output_dir / "model_scorecard.csv"


def stage_report(config: PipelineConfig) -> Path:
    master = pd.read_csv(config.output_root / "03_master" / "scientific_master.csv")
    score = pd.read_csv(config.output_root / "05_machine_learning" / "model_scorecard.csv")
    columns = ["outcome", "n_eligible", "n_locked_test", "rmse_improvement_pct", "r2_model_scale"]
    table = score[columns].copy()
    table["rmse_improvement_pct"] = table["rmse_improvement_pct"].map(lambda value: f"{value:.2f}")
    table["r2_model_scale"] = table["r2_model_scale"].map(lambda value: f"{value:.3f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in table.iterrows()]
    lines = [
        "# Pipeline run summary", "",
        f"Profile: `{config.profile}`", "",
        f"Rows in scientific master: **{len(master):,}**", "",
        "## Scientific invariants", "",
        "- Tail inclusion is strictly `duration_minutes > 240`.",
        "- Lambda is an inverse-timescale per minute.",
        "- Sensor-to-WWTW assignments are supplied as explicit evidence-bearing inputs.",
        "- Predictors are joined without changing master row order.",
        "- Locked-test rows are not used for model selection.", "",
        "## Demonstration model scorecard", "",
        header, separator, *body, "",
        "The included teaching data are synthetic; these metrics are not scientific findings.",
    ]
    output = config.output_root / "06_report" / "RUN_SUMMARY.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


STAGE_FUNCTIONS = {
    "clean": stage_clean,
    "fit": stage_fit,
    "master": stage_master,
    "univariate": stage_univariate,
    "machine_learning": stage_machine_learning,
    "report": stage_report,
}


def run_pipeline(config: PipelineConfig, through: str = "report", force: bool = False) -> dict[str, object]:
    problems = validate_input_contracts(config)
    if problems:
        raise ValueError("Input validation failed:\n- " + "\n- ".join(problems))
    if through not in STAGES:
        raise ValueError(f"Unknown stage {through!r}; choose from {STAGES}")
    target = config.output_root
    existing_run = target.exists() and any(target.iterdir())
    if existing_run and not force:
        raise FileExistsError(f"Output directory already contains a run: {target}. Use --force to replace declared generated files.")
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for stage in STAGES[: STAGES.index(through) + 1]:
        output = STAGE_FUNCTIONS[stage](config)
        try:
            display_output = str(output.relative_to(config.repository_root))
        except ValueError:
            display_output = str(output)
        records.append({"stage": stage, "status": "completed", "output": display_output, "sha256": sha256(output)})
    manifest = {
        "schema_version": 1,
        "profile": config.profile,
        "configuration": str(config.source),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "overwrite_requested": bool(force and existing_run),
        "stages": records,
        "inputs": {name: {"path": str(config.paths[name]), "sha256": sha256(config.paths[name]) if config.paths[name].is_file() else None} for name in ("events", "assignments", "predictors", "monitoring_exposure")},
    }
    manifest_path = target / "run_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest

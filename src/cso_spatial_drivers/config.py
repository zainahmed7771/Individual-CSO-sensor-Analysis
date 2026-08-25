"""Configuration loading and input-contract validation for the public pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_INPUTS = ("events", "assignments", "predictors", "monitoring_exposure")


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved pipeline settings.

    Relative paths are interpreted from the repository root, never from the
    caller's current working directory. This makes the documented commands
    behave identically in VS Code, PowerShell, CI and notebooks.
    """

    source: Path
    repository_root: Path
    profile: str
    seed: int
    paths: dict[str, Path]
    analysis: dict[str, Any]
    predictors: tuple[str, ...]

    @property
    def output_root(self) -> Path:
        return self.paths["output_root"]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path, repository_root: str | Path | None = None) -> PipelineConfig:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    root = Path(repository_root).resolve() if repository_root else source.parents[1]
    raw_paths = payload.get("paths") or {}
    missing_keys = [name for name in (*REQUIRED_INPUTS, "output_root") if not raw_paths.get(name)]
    if missing_keys:
        raise ValueError(f"Configuration is missing path keys: {', '.join(missing_keys)}")
    paths = {name: _resolve(root, value) for name, value in raw_paths.items() if value}
    predictors = tuple(payload.get("predictors") or ())
    if not predictors:
        raise ValueError("Configuration must declare at least one predictor column")
    return PipelineConfig(
        source=source,
        repository_root=root,
        profile=str(payload.get("profile", "scientific")),
        seed=int(payload.get("seed", 20260819)),
        paths=paths,
        analysis=dict(payload.get("analysis") or {}),
        predictors=predictors,
    )


def validate_input_paths(config: PipelineConfig) -> list[str]:
    problems: list[str] = []
    for name in REQUIRED_INPUTS:
        path = config.paths[name]
        if not path.exists():
            problems.append(f"{name}: missing {path}")
        elif not path.is_file() and name != "events":
            problems.append(f"{name}: expected a file, found {path}")
    output_parent = config.output_root.parent
    if not output_parent.exists():
        problems.append(f"output_root: parent directory does not exist: {output_parent}")
    threshold = float(config.analysis.get("tail_threshold_minutes", 240))
    if threshold != 240:
        problems.append("analysis.tail_threshold_minutes must remain 240 for this study")
    if config.analysis.get("tail_rule", "strictly_greater_than") != "strictly_greater_than":
        problems.append("analysis.tail_rule must be strictly_greater_than")
    return problems


def validate_input_contracts(config: PipelineConfig) -> list[str]:
    """Validate schemas/keys without creating pipeline outputs."""
    problems = validate_input_paths(config)
    if problems:
        return problems
    event_path = config.paths["events"]
    event_files = sorted(event_path.rglob("*.csv")) if event_path.is_dir() else [event_path]
    required_events = {"company", "permit_number", "location_name", "start_time", "stop_time"}
    if not event_files:
        problems.append(f"events: no CSV files found under {event_path}")
    for path in event_files:
        columns = set(pd.read_csv(path, nrows=0).columns)
        missing = required_events.difference(columns)
        if missing:
            problems.append(f"events: {path} missing columns {sorted(missing)}")

    assignments = pd.read_csv(config.paths["assignments"], dtype="string")
    required_assignments = {"sensor_uid", "company", "uwwCode", "assignment_evidence"}
    missing = required_assignments.difference(assignments.columns)
    if missing:
        problems.append(f"assignments: missing columns {sorted(missing)}")
    else:
        if assignments["sensor_uid"].duplicated().any():
            problems.append("assignments: sensor_uid must be unique")
        if assignments["assignment_evidence"].fillna("").str.strip().eq("").any():
            problems.append("assignments: every row requires assignment_evidence")

    exposure = pd.read_csv(config.paths["monitoring_exposure"])
    required_exposure = {"sensor_uid", "monitoring_years"}
    missing = required_exposure.difference(exposure.columns)
    if missing:
        problems.append(f"monitoring_exposure: missing columns {sorted(missing)}")
    else:
        years = pd.to_numeric(exposure["monitoring_years"], errors="coerce")
        if exposure["sensor_uid"].duplicated().any():
            problems.append("monitoring_exposure: sensor_uid must be unique")
        if years.isna().any() or years.le(0).any():
            problems.append("monitoring_exposure: monitoring_years must be finite and positive")

    predictors = pd.read_csv(config.paths["predictors"], nrows=None)
    required_predictors = {"company", "uwwCode", *config.predictors}
    missing = required_predictors.difference(predictors.columns)
    if missing:
        problems.append(f"predictors: missing columns {sorted(missing)}")
    elif predictors.duplicated(["company", "uwwCode"]).any():
        problems.append("predictors: company + uwwCode must be unique")
    return problems

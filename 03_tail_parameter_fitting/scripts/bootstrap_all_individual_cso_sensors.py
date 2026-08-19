#!/usr/bin/env python3
"""Fit and bootstrap every individual CSO sensor for all requested companies.

A tail event is defined everywhere in this script as duration_minutes > threshold.
Bootstrap stability labels describe resampling precision, not model adequacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyse_individual_cso_heavy_tails import (
    EXPECTED_COMPANIES,
    MANDATORY_COLUMNS,
    ROUND_TO_15_MINUTES,
    FileQuality,
    canonical_locations,
    discover_files,
    display_number,
    file_sha256,
    filename_number,
    fit_stretched_exponential,
    read_company,
    sensor_statistics,
    validate_company_folders,
)
from terminal_progress import TerminalProgress


DATA_DIR = Path("clean_data")
OUTPUT_ROOT = Path("outputs/individual_cso_sensors")
DEFAULT_THRESHOLD = 240.0
DEFAULT_N_BOOTSTRAP = 1_000
DEFAULT_SEED = 42
DEFAULT_WORKERS = min(8, max(1, os.cpu_count() or 1))
DEFAULT_EXCLUDE_ANGLIAN_2024 = True
POINT_FIT_MINIMUM = 10
BOOTSTRAP_MINIMUM = 20
BETA_LOWER_BOUND = 0.01
BETA_UPPER_BOUND = 2.0
BETA_BOUND_ATOL = 1e-4
MAX_SENSORS_PER_FIGURE = 35
TAIL_CONDITION_TEMPLATE = "duration_minutes > {}"

CSV_COLUMNS = [
    "company", "permit_number", "canonical_location_name", "years_present",
    "number_of_source_files", "first_event_time", "last_event_time",
    "total_spill_count", "total_duration_minutes", "overall_max_duration_minutes",
    "tail_threshold_minutes", "tail_condition", "tail_spill_count",
    "tail_fraction_of_total", "tail_total_duration_minutes",
    "tail_mean_duration_minutes", "tail_median_duration_minutes",
    "tail_max_duration_minutes", "number_of_tail_events_used",
    "point_fit_attempted", "original_fit_success", "fitted_beta",
    "fitted_lambda_per_minute", "negative_log_likelihood", "optimiser_message",
    "bootstrap_attempted", "n_bootstrap_requested", "n_bootstrap_successful",
    "n_bootstrap_failed", "bootstrap_success_rate", "bootstrap_seed_used",
    "beta_bootstrap_mean", "beta_bootstrap_median", "beta_bootstrap_std",
    "beta_ci_lower_95", "beta_ci_upper_95", "beta_ci_width",
    "beta_ci_relative_width", "beta_point_estimate_below_1",
    "beta_ci_entirely_below_1", "bootstrap_stability_label",
    "lambda_bootstrap_mean", "lambda_bootstrap_median", "lambda_bootstrap_std",
    "lambda_ci_lower_95", "lambda_ci_upper_95", "beta_lower_bound_hit_count",
    "beta_upper_bound_hit_count", "beta_bound_hit_rate", "sensor_analysis_status",
]

COUNT_COLUMNS = [
    "number_of_source_files", "total_spill_count", "tail_spill_count",
    "number_of_tail_events_used", "n_bootstrap_requested", "n_bootstrap_successful",
    "n_bootstrap_failed", "bootstrap_seed_used", "beta_lower_bound_hit_count",
    "beta_upper_bound_hit_count",
]
BOOLEAN_COLUMNS = [
    "point_fit_attempted", "original_fit_success", "bootstrap_attempted",
    "beta_point_estimate_below_1", "beta_ci_entirely_below_1",
]

LOGGER = logging.getLogger("all_individual_cso_bootstrap")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bootstrap stretched-exponential fits for every CSO sensor."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--companies", default=",".join(EXPECTED_COMPANIES),
        help="Comma-separated company folder names (one or more required).",
    )
    parser.add_argument(
        "--round-to-15", action=argparse.BooleanOptionalAction,
        default=ROUND_TO_15_MINUTES,
        help=f"Round durations up to 15 minutes (default: {ROUND_TO_15_MINUTES}).",
    )
    parser.add_argument(
        "--exclude-anglian-2024", action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXCLUDE_ANGLIAN_2024,
        help="Exclude Anglian 2024 because long events may be artificially split.",
    )
    parser.add_argument("--save-bootstrap-draws", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.companies = list(dict.fromkeys(
        value.strip().lower() for value in args.companies.split(",") if value.strip()
    ))
    if not args.companies:
        parser.error("--companies must contain at least one company name")
    if args.threshold <= 0 or not math.isfinite(args.threshold):
        parser.error("--threshold must be finite and positive")
    if args.n_bootstrap <= 0:
        parser.error("--n-bootstrap must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def sensor_specific_seed(global_seed: int, company: str, permit_number: str) -> int:
    """Derive an order-independent, stable NumPy seed using SHA-256."""
    payload = f"{global_seed}\0{company}\0{permit_number}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**63 - 1)


def prepare_output_directory(
    output_root: Path, data_dir: Path, threshold: float, overwrite: bool
) -> Path:
    """Safely create only the requested threshold-specific output directory."""
    output_root = output_root.resolve()
    output_dir = (output_root / f"gt_{filename_number(threshold)}min").resolve()
    data_dir = data_dir.resolve()
    if output_dir.parent != output_root:
        raise RuntimeError("Threshold output directory escaped --output-root")
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise RuntimeError("Output directory cannot be the source-data directory")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def configure_logging(path: Path) -> None:
    """Log consistently to the terminal and threshold-specific log file."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def empty_fit_fields() -> dict[str, object]:
    """Return missing point/bootstrap fields with meaningful count defaults."""
    return {
        "point_fit_attempted": False,
        "original_fit_success": False,
        "fitted_beta": math.nan,
        "fitted_lambda_per_minute": math.nan,
        "negative_log_likelihood": math.nan,
        "optimiser_message": "",
        "bootstrap_attempted": False,
        "n_bootstrap_requested": 0,
        "n_bootstrap_successful": 0,
        "n_bootstrap_failed": 0,
        "bootstrap_success_rate": math.nan,
        "bootstrap_seed_used": pd.NA,
        "beta_bootstrap_mean": math.nan,
        "beta_bootstrap_median": math.nan,
        "beta_bootstrap_std": math.nan,
        "beta_ci_lower_95": math.nan,
        "beta_ci_upper_95": math.nan,
        "beta_ci_width": math.nan,
        "beta_ci_relative_width": math.nan,
        "beta_point_estimate_below_1": False,
        "beta_ci_entirely_below_1": False,
        "bootstrap_stability_label": "unavailable",
        "lambda_bootstrap_mean": math.nan,
        "lambda_bootstrap_median": math.nan,
        "lambda_bootstrap_std": math.nan,
        "lambda_ci_lower_95": math.nan,
        "lambda_ci_upper_95": math.nan,
        "beta_lower_bound_hit_count": 0,
        "beta_upper_bound_hit_count": 0,
        "beta_bound_hit_rate": math.nan,
    }


def bootstrap_summary_from_draws(
    fitted_beta: float,
    beta_draws: np.ndarray,
    lambda_draws: np.ndarray,
    successes: np.ndarray,
) -> dict[str, object]:
    """Summarise only successful bootstrap fits."""
    successful_beta = np.asarray(beta_draws, dtype=float)[successes]
    successful_lambda = np.asarray(lambda_draws, dtype=float)[successes]
    lower_hits = int(np.isclose(successful_beta, BETA_LOWER_BOUND, atol=BETA_BOUND_ATOL).sum())
    upper_hits = int(np.isclose(successful_beta, BETA_UPPER_BOUND, atol=BETA_BOUND_ATOL).sum())
    result: dict[str, object] = {
        "beta_lower_bound_hit_count": lower_hits,
        "beta_upper_bound_hit_count": upper_hits,
        "beta_bound_hit_rate": (
            (lower_hits + upper_hits) / successful_beta.size
            if successful_beta.size else math.nan
        ),
    }
    if successful_beta.size < 2:
        return result
    beta_lower, beta_upper = np.percentile(successful_beta, [2.5, 97.5])
    lambda_lower, lambda_upper = np.percentile(successful_lambda, [2.5, 97.5])
    width = float(beta_upper - beta_lower)
    relative = width / abs(fitted_beta) if math.isfinite(fitted_beta) and fitted_beta > 0 else math.nan
    if not math.isfinite(relative):
        stability = "unavailable"
    elif relative <= 0.25:
        stability = "stable"
    elif relative <= 0.50:
        stability = "moderate_uncertainty"
    else:
        stability = "unstable"
    result.update({
        "beta_bootstrap_mean": float(np.mean(successful_beta)),
        "beta_bootstrap_median": float(np.median(successful_beta)),
        "beta_bootstrap_std": float(np.std(successful_beta, ddof=1)),
        "beta_ci_lower_95": float(beta_lower),
        "beta_ci_upper_95": float(beta_upper),
        "beta_ci_width": width,
        "beta_ci_relative_width": relative,
        "beta_ci_entirely_below_1": bool(beta_upper < 1.0),
        "bootstrap_stability_label": stability,
        "lambda_bootstrap_mean": float(np.mean(successful_lambda)),
        "lambda_bootstrap_median": float(np.median(successful_lambda)),
        "lambda_bootstrap_std": float(np.std(successful_lambda, ddof=1)),
        "lambda_ci_lower_95": float(lambda_lower),
        "lambda_ci_upper_95": float(lambda_upper),
    })
    return result


def analyse_sensor_task(task: dict[str, object]) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    """Fit and optionally bootstrap one sensor; never abort another sensor's work."""
    row = dict(task["row"])
    tail = np.asarray(task["tail"], dtype=np.float64)
    threshold = float(task["threshold"])
    n_bootstrap = int(task["n_bootstrap"])
    seed = int(task["seed"])
    save_draws = bool(task["save_draws"])
    row.update(empty_fit_fields())
    row["number_of_tail_events_used"] = int(tail.size)
    row["beta_point_estimate_below_1"] = False

    if tail.size == 0:
        row["sensor_analysis_status"] = "no_tail_events"
        return row, None
    if tail.size < POINT_FIT_MINIMUM:
        row["sensor_analysis_status"] = "insufficient_tail_events_for_point_fit"
        return row, None

    row["point_fit_attempted"] = True
    try:
        fit = fit_stretched_exponential(tail, threshold)
        row.update({
            "original_fit_success": bool(fit.fit_success),
            "fitted_beta": fit.fitted_beta,
            "fitted_lambda_per_minute": fit.fitted_lambda_per_minute,
            "negative_log_likelihood": fit.negative_log_likelihood,
            "optimiser_message": fit.optimiser_message,
            "beta_point_estimate_below_1": bool(
                fit.fit_success and fit.fitted_beta < 1.0
            ),
        })
    except (ValueError, FloatingPointError, OverflowError) as exc:
        row["optimiser_message"] = f"{type(exc).__name__}: {exc}"
        row["sensor_analysis_status"] = "original_fit_failed"
        return row, None
    except Exception as exc:  # isolated, recorded, and returned to the parent log
        row["optimiser_message"] = f"Unexpected {type(exc).__name__}: {exc}"
        row["_traceback"] = traceback.format_exc()
        row["sensor_analysis_status"] = "original_fit_failed"
        return row, None

    if not row["original_fit_success"]:
        row["sensor_analysis_status"] = "original_fit_failed"
        return row, None
    if tail.size < BOOTSTRAP_MINIMUM:
        row["sensor_analysis_status"] = "point_fit_completed_but_insufficient_for_bootstrap"
        return row, None

    row["bootstrap_attempted"] = True
    row["n_bootstrap_requested"] = n_bootstrap
    row["bootstrap_seed_used"] = seed
    rng = np.random.default_rng(seed)
    beta_draws = np.full(n_bootstrap, np.nan)
    lambda_draws = np.full(n_bootstrap, np.nan)
    successes = np.zeros(n_bootstrap, dtype=bool)
    unexpected_bootstrap_tracebacks: list[str] = []
    for index in range(n_bootstrap):
        sample = rng.choice(tail, size=tail.size, replace=True)
        try:
            replicate = fit_stretched_exponential(sample, threshold)
            if replicate.fit_success:
                beta_draws[index] = replicate.fitted_beta
                lambda_draws[index] = replicate.fitted_lambda_per_minute
                successes[index] = True
        except (ValueError, FloatingPointError, OverflowError):
            pass
        except Exception:
            if len(unexpected_bootstrap_tracebacks) < 3:
                unexpected_bootstrap_tracebacks.append(traceback.format_exc())

    successful = int(successes.sum())
    failed = n_bootstrap - successful
    row["n_bootstrap_successful"] = successful
    row["n_bootstrap_failed"] = failed
    row["bootstrap_success_rate"] = successful / n_bootstrap
    if unexpected_bootstrap_tracebacks:
        row["_bootstrap_tracebacks"] = unexpected_bootstrap_tracebacks
    row.update(bootstrap_summary_from_draws(
        float(row["fitted_beta"]), beta_draws, lambda_draws, successes
    ))
    if successful < 2:
        row["sensor_analysis_status"] = "bootstrap_interval_unavailable"
    elif failed:
        row["sensor_analysis_status"] = "bootstrap_completed_with_fit_failures"
    else:
        row["sensor_analysis_status"] = "bootstrap_completed"
    draws = None
    if save_draws:
        draws = {"beta": beta_draws, "lambda": lambda_draws, "success": successes}
    return row, draws


def base_sensor_rows(events: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Build the descriptive portion of the exact output schema for every sensor."""
    stats = sensor_statistics(events, threshold).copy()
    stats["tail_threshold_minutes"] = threshold
    stats["tail_condition"] = TAIL_CONDITION_TEMPLATE.format(display_number(threshold))
    stats["tail_fraction_of_total"] = stats["tail_spill_count"] / stats["total_spill_count"]
    stats["number_of_tail_events_used"] = stats["tail_spill_count"]
    for column in ("first_event_time", "last_event_time"):
        stats[column] = pd.to_datetime(stats[column], errors="coerce").map(
            lambda value: value.isoformat() if pd.notna(value) else ""
        )
    return stats


def sort_sensor_tables(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the three stable rankings with identical values and columns."""
    total = table.sort_values(
        ["total_spill_count", "tail_spill_count", "permit_number"],
        ascending=[False, False, True], kind="mergesort",
    )
    tail = table.sort_values(
        ["tail_spill_count", "total_spill_count", "permit_number"],
        ascending=[False, False, True], kind="mergesort",
    )
    precision_source = table.assign(
        _usable_interval=table["beta_ci_relative_width"].notna()
    )
    precision = precision_source.sort_values(
        ["_usable_interval", "beta_ci_relative_width", "beta_ci_width",
         "tail_spill_count", "total_spill_count", "permit_number"],
        ascending=[False, True, True, False, False, True],
        kind="mergesort", na_position="last",
    ).drop(columns="_usable_interval")
    return {
        "01_sensors_by_total_spills_desc.csv": total,
        "02_sensors_by_tail_spills_desc.csv": tail,
        "03_sensors_by_beta_precision_asc.csv": precision,
    }


def enforce_output_dtypes(table: pd.DataFrame) -> pd.DataFrame:
    """Apply stable output dtypes and the exact requested logical column order."""
    table = table.copy()
    for column in COUNT_COLUMNS:
        table[column] = pd.array(table[column], dtype="Int64")
    for column in BOOLEAN_COLUMNS:
        table[column] = table[column].fillna(False).astype(bool)
    table["permit_number"] = table["permit_number"].astype("string")
    return table[CSV_COLUMNS]


def shortened_location(value: object, limit: int = 42) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def plot_beta_intervals(
    company: str, table: pd.DataFrame, threshold: float, output_dir: Path
) -> int:
    """Create paginated company forest plots for usable beta intervals."""
    figure_dir = output_dir / "_figures" / company
    figure_dir.mkdir(parents=True, exist_ok=True)
    usable = table[table["beta_ci_relative_width"].notna()].sort_values(
        ["beta_ci_relative_width", "tail_spill_count", "permit_number"],
        ascending=[True, False, True], kind="mergesort",
    )
    pages = int(math.ceil(len(usable) / MAX_SENSORS_PER_FIGURE)) if len(usable) else 0
    for page in range(pages):
        subset = usable.iloc[
            page * MAX_SENSORS_PER_FIGURE:(page + 1) * MAX_SENSORS_PER_FIGURE
        ].copy().iloc[::-1]
        y = np.arange(len(subset))
        beta = subset["fitted_beta"].to_numpy(dtype=float)
        lower = subset["beta_ci_lower_95"].to_numpy(dtype=float)
        upper = subset["beta_ci_upper_95"].to_numpy(dtype=float)
        labels = [
            f"{permit} | {shortened_location(location)} | n_tail={int(count)}"
            for permit, location, count in zip(
                subset["permit_number"], subset["canonical_location_name"],
                subset["tail_spill_count"], strict=True,
            )
        ]
        height = max(5.2, 0.32 * len(subset) + 2.3)
        fig, ax = plt.subplots(figsize=(12.5, height), constrained_layout=True)
        ax.hlines(y, lower, upper, color="#697f9f", linewidth=1.2)
        ax.scatter(beta, y, color="#315f8c", s=24, zorder=3)
        ax.axvline(1.0, color="#c62828", linestyle="--", lw=1.1, label="beta = 1")
        ax.set_yticks(y, labels)
        ax.set_xlabel("Fitted beta with 95% bootstrap percentile interval")
        ax.set_title(
            f"{company.replace('_', ' ').title()} | duration > {display_number(threshold)} min "
            f"(strict) | {len(subset)} displayed / {len(table)} unique sensors\n"
            "Ordered by beta CI relative width (most precise first)", pad=10,
        )
        ax.grid(axis="x", color="#d8d8d8", lw=0.6, alpha=0.65)
        ax.legend(frameon=False, loc="best")
        fig.savefig(figure_dir / f"beta_ci_page_{page + 1:02d}.png", dpi=220,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return pages


def process_company(
    company: str,
    events: pd.DataFrame,
    threshold: float,
    n_bootstrap: int,
    global_seed: int,
    workers: int,
    save_draws: bool,
    output_dir: Path,
) -> pd.DataFrame:
    """Analyse all sensors for one already-cleaned company DataFrame."""
    base = base_sensor_rows(events, threshold)
    tasks: list[dict[str, object]] = []
    tail_lookup = {
        str(permit): np.sort(group["duration_minutes"].to_numpy(dtype=np.float64))
        for permit, group in events.loc[events["duration_minutes"] > threshold].groupby(
            "permit_number", sort=False
        )
    }
    empty_tail = np.empty(0, dtype=np.float64)
    preparation = TerminalProgress(f"{company}: prepare sensors", len(base))
    for row in base.sort_values("permit_number", kind="mergesort").to_dict("records"):
        permit = str(row["permit_number"])
        tail = tail_lookup.get(permit, empty_tail)
        tasks.append({
            "row": row, "tail": tail, "threshold": threshold,
            "n_bootstrap": n_bootstrap,
            "seed": sensor_specific_seed(global_seed, company, permit),
            "save_draws": save_draws,
        })
        preparation.update(suffix=f"permit {permit}")
    preparation.close(suffix="sensor tasks ready")
    point_fit_candidates = int((base["tail_spill_count"] >= POINT_FIT_MINIMUM).sum())
    bootstrap_candidates = int((base["tail_spill_count"] >= BOOTSTRAP_MINIMUM).sum())
    LOGGER.info(
        "%s workload: sensors=%d point-fit candidates=%d bootstrap candidates=%d "
        "requested bootstrap optimizer fits=%d workers=%d",
        company, len(tasks), point_fit_candidates, bootstrap_candidates,
        bootstrap_candidates * n_bootstrap, workers,
    )
    progress = TerminalProgress(f"{company}: fit/bootstrap", len(tasks))
    if workers == 1:
        results = []
        for task in tasks:
            results.append(analyse_sensor_task(task))
            progress.update(suffix=f"permit {task['row']['permit_number']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            ordered_results: list[tuple[dict[str, object], dict[str, np.ndarray] | None] | None] = [None] * len(tasks)
            futures = {
                executor.submit(analyse_sensor_task, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                index = futures[future]
                ordered_results[index] = future.result()
                progress.update(suffix=f"permit {tasks[index]['row']['permit_number']}")
            if any(result is None for result in ordered_results):
                raise RuntimeError(f"Not every sensor task returned a result for {company}")
            results = [result for result in ordered_results if result is not None]
    progress.close(suffix="all sensors analysed")

    rows: list[dict[str, object]] = []
    diagnostics_dir = output_dir / "diagnostics" / company
    for (row, draws), task in zip(results, tasks, strict=True):
        trace = row.pop("_traceback", None)
        if trace:
            LOGGER.error("Unexpected fit error for %s/%s:\n%s", company,
                         row["permit_number"], trace)
        bootstrap_traces = row.pop("_bootstrap_tracebacks", [])
        for bootstrap_trace in bootstrap_traces:
            LOGGER.error(
                "Unexpected bootstrap fit error for %s/%s:\n%s",
                company, row["permit_number"], bootstrap_trace,
            )
        if row["sensor_analysis_status"] in {
            "original_fit_failed", "bootstrap_completed_with_fit_failures",
            "bootstrap_interval_unavailable",
        }:
            LOGGER.warning(
                "%s/%s status=%s message=%s", company, row["permit_number"],
                row["sensor_analysis_status"], row["optimiser_message"],
            )
        if draws is not None:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            token = hashlib.sha256(str(row["permit_number"]).encode()).hexdigest()[:16]
            np.savez_compressed(diagnostics_dir / f"sensor_{token}.npz", **draws)
        rows.append(row)
    return enforce_output_dtypes(pd.DataFrame(rows))


def company_quality_record(
    company: str, events: pd.DataFrame, qualities: list[FileQuality], inconsistent: pd.DataFrame
) -> dict[str, object]:
    """Create the per-company data-quality and descriptive summary."""
    duplicate_count = int(events.duplicated(subset=MANDATORY_COLUMNS, keep=False).sum())
    return {
        "company": company,
        "number_of_source_files_processed": sum(q.status == "processed" for q in qualities),
        "valid_event_rows": int(len(events)),
        "rejected_event_rows": int(sum(q.rejected_rows for q in qualities)),
        "exact_duplicate_rows_detected": duplicate_count,
        "sensors_with_inconsistent_location_names": int(
            inconsistent["permit_number"].nunique() if not inconsistent.empty else 0
        ),
        "source_files": [asdict(q) for q in qualities],
    }


def analytical_company_summary(table: pd.DataFrame, quality: dict[str, object]) -> dict[str, object]:
    """Combine data quality and sensor outcome counts for summary reporting."""
    tail = table["tail_spill_count"].astype(int)
    fits = table[table["original_fit_success"]]
    usable = table[table["beta_ci_relative_width"].notna()]
    attempted = table[table["bootstrap_attempted"]]
    return {
        **{key: value for key, value in quality.items() if key != "source_files"},
        "total_unique_sensors": int(len(table)),
        "sensors_with_zero_tail_spills": int((tail == 0).sum()),
        "sensors_with_1_to_9_tail_spills": int(((tail >= 1) & (tail <= 9)).sum()),
        "sensors_with_10_to_19_tail_spills": int(((tail >= 10) & (tail <= 19)).sum()),
        "sensors_with_at_least_20_tail_spills": int((tail >= 20).sum()),
        "original_fits_attempted": int(table["point_fit_attempted"].sum()),
        "original_fits_successful": int(table["original_fit_success"].sum()),
        "bootstrap_analyses_attempted": int(table["bootstrap_attempted"].sum()),
        "bootstrap_analyses_with_usable_intervals": int(len(usable)),
        "bootstrap_analyses_containing_failed_replicates": int(
            (attempted["n_bootstrap_failed"].fillna(0).astype(int) > 0).sum()
        ),
        "median_total_spills_per_sensor": float(table["total_spill_count"].median()),
        "median_tail_spills_per_sensor": float(table["tail_spill_count"].median()),
        "maximum_tail_spills_for_one_sensor": int(table["tail_spill_count"].max()),
        "median_fitted_beta_among_successful_fits": (
            float(fits["fitted_beta"].median()) if len(fits) else None
        ),
        "median_beta_ci_width": float(usable["beta_ci_width"].median()) if len(usable) else None,
        "median_beta_relative_ci_width": (
            float(usable["beta_ci_relative_width"].median()) if len(usable) else None
        ),
        "sensors_with_full_beta_interval_below_1": int(
            table["beta_ci_entirely_below_1"].sum()
        ),
    }


def total_summary(company_summaries: list[dict[str, object]], all_tables: list[pd.DataFrame]) -> dict[str, object]:
    """Calculate an all-company total section, including pooled medians."""
    table = pd.concat(all_tables, ignore_index=True)
    quality = {
        "company": "ALL COMPANIES",
        "number_of_source_files_processed": sum(int(x["number_of_source_files_processed"]) for x in company_summaries),
        "valid_event_rows": sum(int(x["valid_event_rows"]) for x in company_summaries),
        "rejected_event_rows": sum(int(x["rejected_event_rows"]) for x in company_summaries),
        "exact_duplicate_rows_detected": sum(int(x["exact_duplicate_rows_detected"]) for x in company_summaries),
        "sensors_with_inconsistent_location_names": sum(int(x["sensors_with_inconsistent_location_names"]) for x in company_summaries),
    }
    return analytical_company_summary(table, quality)


def write_company_summary(
    path: Path, config: dict[str, object], summaries: list[dict[str, object]], total: dict[str, object]
) -> None:
    """Write the requested human-readable run and company summary."""
    lines = [
        "All-individual-CSO bootstrap analysis", "=====================================", "",
        f"Timestamp: {config['run_started_utc']}",
        f"Data directory: {config['data_dir']}",
        f"Output directory: {config['output_directory']}",
        f"Companies included: {', '.join(config['companies'])}",
        f"Threshold: {display_number(float(config['threshold_minutes']))} minutes",
        f"Exact tail condition: {config['tail_condition']}",
        f"Duration rounding to 15 minutes: {config['round_to_15_minutes']}",
        f"Anglian 2024 excluded: {config['exclude_anglian_2024']}",
        f"Bootstrap replicates: {config['n_bootstrap']}",
        f"Global seed: {config['global_seed']}",
        f"Point-fit minimum tail count: {POINT_FIT_MINIMUM}",
        f"Bootstrap minimum tail count: {BOOTSTRAP_MINIMUM}", "",
    ]
    labels = [
        ("number_of_source_files_processed", "Source files processed"),
        ("valid_event_rows", "Valid event rows"),
        ("rejected_event_rows", "Rejected event rows"),
        ("exact_duplicate_rows_detected", "Exact duplicate rows detected"),
        ("total_unique_sensors", "Total unique sensors"),
        ("sensors_with_zero_tail_spills", "Sensors with zero tail spills"),
        ("sensors_with_1_to_9_tail_spills", "Sensors with 1 to 9 tail spills"),
        ("sensors_with_10_to_19_tail_spills", "Sensors with 10 to 19 tail spills"),
        ("sensors_with_at_least_20_tail_spills", "Sensors with at least 20 tail spills"),
        ("original_fits_attempted", "Original fits attempted"),
        ("original_fits_successful", "Original fits successful"),
        ("bootstrap_analyses_attempted", "Bootstrap analyses attempted"),
        ("bootstrap_analyses_with_usable_intervals", "Bootstrap analyses with usable intervals"),
        ("bootstrap_analyses_containing_failed_replicates", "Bootstrap analyses containing failed replicates"),
        ("median_total_spills_per_sensor", "Median total spills per sensor"),
        ("median_tail_spills_per_sensor", "Median tail spills per sensor"),
        ("maximum_tail_spills_for_one_sensor", "Maximum tail spills for one sensor"),
        ("median_fitted_beta_among_successful_fits", "Median fitted beta among successful fits"),
        ("median_beta_ci_width", "Median beta CI width"),
        ("median_beta_relative_ci_width", "Median beta relative CI width"),
        ("sensors_with_full_beta_interval_below_1", "Sensors whose full beta interval is below 1"),
        ("sensors_with_inconsistent_location_names", "Sensors with inconsistent location names"),
    ]
    for summary in [*summaries, total]:
        lines.extend([str(summary["company"]).replace("_", " ").title(), "-" * 40])
        for key, label in labels:
            value = summary.get(key)
            lines.append(f"{label}: {'unavailable' if value is None else value}")
        lines.append("")
    lines.extend([
        "Interpretation warning", "----------------------",
        "A narrow bootstrap interval indicates that beta is stable under resampling of the observed tail events. It does not by itself establish that the stretched-exponential model is an adequate description of the sensor's spill-duration distribution.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_company_outputs(
    company: str, expected_table: pd.DataFrame, company_dir: Path, threshold: float
) -> None:
    """Validate one company's three CSVs and all sensor-level invariants."""
    paths = sorted(company_dir.glob("*.csv"))
    if len(paths) != 3:
        raise AssertionError(f"{company} does not have exactly three CSV files")
    frames = [pd.read_csv(path, dtype={"permit_number": "string"}) for path in paths]
    for frame in frames:
        if list(frame.columns) != CSV_COLUMNS:
            raise AssertionError(f"Unexpected CSV columns for {company}")
        if frame["permit_number"].duplicated().any():
            raise AssertionError(f"Duplicate permit_number in {company}")
        if set(frame["permit_number"]) != set(expected_table["permit_number"].astype(str)):
            raise AssertionError(f"Not every valid sensor appears in {company}")
    canonical = [
        frame.sort_values("permit_number", kind="mergesort").reset_index(drop=True)
        for frame in frames
    ]
    for frame in canonical[1:]:
        pd.testing.assert_frame_equal(canonical[0], frame, check_dtype=False)
    frame = canonical[0]
    if frame.duplicated(["company", "permit_number"]).any():
        raise AssertionError(f"Duplicate company/permit pair in {company}")
    if (frame["tail_spill_count"] > frame["total_spill_count"]).any():
        raise AssertionError("tail_spill_count exceeds total_spill_count")
    if not frame["tail_fraction_of_total"].between(0, 1).all():
        raise AssertionError("tail_fraction_of_total is outside [0, 1]")
    successful = frame["original_fit_success"].astype(bool)
    for column in ("fitted_beta", "fitted_lambda_per_minute"):
        values = frame.loc[successful, column].to_numpy(dtype=float)
        if not (np.isfinite(values) & (values > 0)).all():
            raise AssertionError(f"Invalid successful {column}")
    usable = frame["beta_ci_lower_95"].notna() & frame["beta_ci_upper_95"].notna()
    if (frame.loc[usable, "beta_ci_lower_95"] > frame.loc[usable, "beta_ci_upper_95"]).any():
        raise AssertionError("Reversed beta confidence interval")
    if (frame.loc[usable, "beta_ci_width"] < 0).any():
        raise AssertionError("Negative beta confidence interval width")
    if (frame.loc[usable, "beta_ci_relative_width"] < 0).any():
        raise AssertionError("Negative relative beta confidence interval width")
    attempted = frame["bootstrap_attempted"].astype(bool)
    if not (
        frame.loc[attempted, "n_bootstrap_successful"]
        + frame.loc[attempted, "n_bootstrap_failed"]
        == frame.loc[attempted, "n_bootstrap_requested"]
    ).all():
        raise AssertionError("Bootstrap counts do not sum to the requested count")
    if not frame.loc[attempted, "bootstrap_success_rate"].between(0, 1).all():
        raise AssertionError("Bootstrap success rate outside [0, 1]")
    insufficient = frame["tail_spill_count"] < POINT_FIT_MINIMUM
    if frame.loc[insufficient, ["fitted_beta", "beta_ci_lower_95"]].notna().any().any():
        raise AssertionError("Insufficient-tail sensor has fabricated fit values")
    if threshold == 240.0 and "duration_minutes >= 240" in set(frame["tail_condition"]):
        raise AssertionError("A non-strict 240-minute condition was written")


def validate_all_outputs(
    output_dir: Path,
    companies: list[str],
    expected_tables: dict[str, pd.DataFrame],
    threshold: float,
    source_hashes: dict[Path, str],
) -> dict[str, object]:
    """Run complete structural, numerical, and source-integrity validation."""
    if not companies or set(expected_tables) != set(companies):
        raise AssertionError("The requested companies were not analysed exactly once")
    for company in companies:
        validate_company_outputs(company, expected_tables[company], output_dir / company, threshold)
    csv_files = list(output_dir.rglob("*.csv"))
    expected_csv_count = 3 * len(companies)
    if len(csv_files) != expected_csv_count:
        raise AssertionError(f"Expected exactly {expected_csv_count} CSV files, found {len(csv_files)}")
    pairs = pd.concat(expected_tables.values(), ignore_index=True)[["company", "permit_number"]]
    if pairs.duplicated().any():
        raise AssertionError("Duplicate company/permit pair across company tables")
    for path, before in source_hashes.items():
        if not path.exists() or file_sha256(path) != before:
            raise AssertionError(f"Source file changed during execution: {path}")
    return {
        "companies_validated": len(companies),
        "company_csv_files": len(csv_files),
        "unique_company_sensor_pairs": int(len(pairs)),
        "source_hashes_unchanged": True,
        "deterministic_seed_method": "SHA-256(global_seed, company, permit_number)",
        "validation_passed": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run all companies, write the exact output contract, and validate it."""
    args = parse_args(argv)
    validate_company_folders(args.data_dir, args.companies)
    files_by_company = discover_files(
        args.data_dir, args.companies, args.exclude_anglian_2024
    )
    # Hash every requested-company source CSV, including any deliberately excluded file.
    source_files = [
        path for company in args.companies
        for path in sorted((args.data_dir / company).rglob("*.csv"))
    ]
    source_hashes = {path: file_sha256(path) for path in source_files}
    output_dir = prepare_output_directory(
        args.output_root, args.data_dir, args.threshold, args.overwrite
    )
    configure_logging(output_dir / "bootstrap_analysis.log")
    LOGGER.info("Starting all-sensor analysis: duration_minutes > %s", display_number(args.threshold))

    config: dict[str, object] = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "output_directory": str(output_dir),
        "companies": args.companies,
        "threshold_minutes": args.threshold,
        "tail_condition": TAIL_CONDITION_TEMPLATE.format(display_number(args.threshold)),
        "round_to_15_minutes": args.round_to_15,
        "exclude_anglian_2024": args.exclude_anglian_2024,
        "n_bootstrap": args.n_bootstrap,
        "global_seed": args.seed,
        "workers": args.workers,
        "point_fit_minimum_tail_count": POINT_FIT_MINIMUM,
        "bootstrap_minimum_tail_count": BOOTSTRAP_MINIMUM,
        "save_bootstrap_draws": args.save_bootstrap_draws,
        "overwrite": args.overwrite,
        "stability_labels_note": "Heuristic bootstrap precision labels; not goodness-of-fit labels.",
        "beta_bounds": [BETA_LOWER_BOUND, BETA_UPPER_BOUND],
    }
    (output_dir / "run_configuration.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    expected_tables: dict[str, pd.DataFrame] = {}
    quality_records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    company_progress = TerminalProgress("Bootstrap companies", len(args.companies))
    for company in args.companies:
        company_dir = output_dir / company
        company_dir.mkdir()
        events, qualities = read_company(company, files_by_company[company], args.round_to_15)
        _, inconsistent = canonical_locations(events)
        quality = company_quality_record(company, events, qualities, inconsistent)
        quality_records.append(quality)

        # Direct event-level checks prove the strict-tail counts, including equality exclusion.
        direct_counts = events.assign(_tail=events["duration_minutes"] > args.threshold).groupby(
            "permit_number"
        )["_tail"].sum().astype(int)
        if args.threshold == 240.0:
            equal_240 = events[events["duration_minutes"] == 240.0]
            if not equal_240.empty and (equal_240["duration_minutes"] > args.threshold).any():
                raise AssertionError("Exactly 240 minutes was included as a tail event")

        table = process_company(
            company, events, args.threshold, args.n_bootstrap, args.seed,
            args.workers, args.save_bootstrap_draws, output_dir,
        )
        observed = table.set_index("permit_number")["tail_spill_count"].astype(int)
        if not observed.sort_index().equals(direct_counts.sort_index()):
            raise AssertionError(f"Strict tail count validation failed for {company}")
        rankings = sort_sensor_tables(table)
        for filename, ranked in rankings.items():
            ranked.to_csv(company_dir / filename, index=False)
        plot_beta_intervals(company, table, args.threshold, output_dir)
        expected_tables[company] = table
        summary = analytical_company_summary(table, quality)
        summaries.append(summary)
        LOGGER.info(
            "Completed %s: sensors=%d usable_intervals=%d", company, len(table),
            summary["bootstrap_analyses_with_usable_intervals"],
        )
        company_progress.update(suffix=company)
        del events, table
    company_progress.close(suffix="all companies complete")

    total = total_summary(summaries, list(expected_tables.values()))
    write_company_summary(
        output_dir / "company_sensor_summary.txt", config, summaries, total
    )
    data_quality = {
        "run_settings": {
            "round_to_15_minutes": args.round_to_15,
            "exclude_anglian_2024": args.exclude_anglian_2024,
        },
        "companies": quality_records,
        "totals": {
            "valid_event_rows": total["valid_event_rows"],
            "rejected_event_rows": total["rejected_event_rows"],
            "exact_duplicate_rows_detected": total["exact_duplicate_rows_detected"],
            "sensors_with_inconsistent_location_names": total["sensors_with_inconsistent_location_names"],
        },
    }
    (output_dir / "data_quality_summary.json").write_text(
        json.dumps(data_quality, indent=2, default=str) + "\n", encoding="utf-8"
    )
    validation = validate_all_outputs(
        output_dir, args.companies, expected_tables, args.threshold, source_hashes
    )
    config["run_completed_utc"] = datetime.now(timezone.utc).isoformat()
    config["validation"] = validation
    config["company_results"] = summaries
    config["all_company_results"] = total
    (output_dir / "run_configuration.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        "Validation passed: "
        f"{validation['companies_validated']} companies, "
        f"{validation['company_csv_files']} CSV files, "
        f"{validation['unique_company_sensor_pairs']} unique company/sensor pairs, "
        "source hashes unchanged."
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, AssertionError) as exc:
        LOGGER.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

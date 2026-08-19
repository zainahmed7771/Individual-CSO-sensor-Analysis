#!/usr/bin/env python3
"""Select each company's largest CSO spill-duration tails and fit them separately.

The script is standalone: it reads cleaned event CSVs, ranks sensors by the
number of events strictly longer than a configurable threshold, saves every
event for the selected sensors, and fits/plots each selected tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

from terminal_progress import TerminalProgress


# ---------------------------------------------------------------------------
# Editable defaults
# ---------------------------------------------------------------------------
DATA_DIR = Path("clean_data")
OUTPUT_ROOT = Path("outputs/individual_cso_heavy_tail_analysis")

EXPECTED_COMPANIES = [
    "anglian",
    "northumbria",
    "severn_trent",
    "southern",
    "southwest",
    "united_utilities",
    "wessex",
    "yorkshire",
    "thames",
]
EXPECTED_COMPANY_COUNT = 9
DURATION_THRESHOLD_MINUTES = 240.0
TOP_N_PER_COMPANY = 3
NUMBER_OF_BINS = 60
MIN_TAIL_EVENTS_FOR_FIT = 10
ROUND_TO_15_MINUTES = False
EXCLUDE_ANGLIAN_2024 = False

CHUNK_SIZE = 250_000
MANDATORY_COLUMNS = [
    "location_name",
    "permit_number",
    "start_time",
    "stop_time",
    "duration_minutes",
]
LOGGER = logging.getLogger("individual_cso_heavy_tails")

DURATION_TICK_CANDIDATES = [
    (60.0, "1 h"),
    (120.0, "2 h"),
    (240.0, "4 h"),
    (360.0, "6 h"),
    (720.0, "12 h"),
    (1_440.0, "24 h / 1 d"),
    (2_880.0, "48 h / 2 d"),
    (4_320.0, "72 h / 3 d"),
    (10_080.0, "1 wk"),
    (20_160.0, "2 wk"),
    (43_200.0, "30 d"),
    (86_400.0, "60 d"),
    (129_600.0, "90 d"),
    (262_800.0, "6 mo"),
    (525_600.0, "1 yr"),
]


@dataclass
class FileQuality:
    """Data-quality measurements for one source file."""

    company: str
    source_file: str
    source_year: str = ""
    status: str = "not_read"
    rows_read: int = 0
    valid_rows: int = 0
    rejected_rows: int = 0
    missing_permit_number: int = 0
    non_numeric_duration: int = 0
    non_finite_duration: int = 0
    non_positive_duration: int = 0
    invalid_start_time: int = 0
    invalid_stop_time: int = 0
    duplicate_rows: int = 0
    missing_columns: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class FitResult:
    """Result of one conditional stretched-exponential fit."""

    fitted_beta: float
    fitted_lambda_per_minute: float
    negative_log_likelihood: float
    fit_success: bool
    optimiser_message: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank CSO sensors by spills strictly longer than a threshold, then "
            "extract and fit each selected sensor's duration tail."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--threshold", type=float, default=DURATION_THRESHOLD_MINUTES)
    parser.add_argument("--top-n", type=int, default=TOP_N_PER_COMPANY)
    parser.add_argument(
        "--companies", type=str, default=",".join(EXPECTED_COMPANIES),
        help="Comma-separated company folder names.",
    )
    parser.add_argument(
        "--expected-company-count", type=int, default=EXPECTED_COMPANY_COUNT
    )
    parser.add_argument("--bins", type=int, default=NUMBER_OF_BINS)
    parser.add_argument(
        "--min-tail-events", type=int, default=MIN_TAIL_EVENTS_FOR_FIT
    )
    parser.add_argument(
        "--round-to-15", action="store_true", default=ROUND_TO_15_MINUTES,
        help="Round positive durations up to 15-minute intervals.",
    )
    parser.add_argument(
        "--exclude-anglian-2024", action="store_true",
        default=EXCLUDE_ANGLIAN_2024,
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace only this threshold-specific output directory.",
    )
    args = parser.parse_args(argv)
    args.companies = list(dict.fromkeys(
        value.strip().lower() for value in args.companies.split(",") if value.strip()
    ))
    if args.threshold <= 0 or not math.isfinite(args.threshold):
        parser.error("--threshold must be a finite positive number")
    if args.top_n <= 0:
        parser.error("--top-n must be positive")
    if args.expected_company_count <= 0:
        parser.error("--expected-company-count must be positive")
    if len(args.companies) != args.expected_company_count:
        parser.error(
            "--companies contains " + str(len(args.companies))
            + " names but --expected-company-count is "
            + str(args.expected_company_count)
        )
    if args.bins < 10:
        parser.error("--bins must be at least 10")
    if args.min_tail_events < 1:
        parser.error("--min-tail-events must be positive")
    return args


def display_number(value: float) -> str:
    """Format a setting without an unnecessary decimal suffix."""
    return f"{value:g}"


def filename_number(value: float) -> str:
    """Return a filesystem-safe numeric token."""
    return display_number(value).replace("-", "m").replace(".", "p")


def human_duration_label(minutes: float) -> str:
    """Return a compact human-readable label for a duration in minutes."""
    for candidate, label in DURATION_TICK_CANDIDATES:
        if math.isclose(minutes, candidate, rel_tol=0.0, abs_tol=1e-9):
            return label
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:g} h"
    days = minutes / 1_440.0
    if days < 7:
        return f"{days:g} d"
    if days < 60:
        return f"{days / 7.0:g} wk"
    if days < 365:
        return f"{days / 30.0:g} mo"
    return f"{days / 365.0:g} yr"


def set_human_duration_ticks(
    ax: plt.Axes, lower_minutes: float, upper_minutes: float, threshold: float | None = None
) -> None:
    """Use fixed, readable duration ticks on an already logarithmic x-axis."""
    if lower_minutes <= 0 or upper_minutes < lower_minutes:
        raise ValueError("Duration tick limits must be positive and ordered")
    if threshold is None:
        threshold = lower_minutes
    values = [
        value for value, _ in DURATION_TICK_CANDIDATES
        if lower_minutes <= value <= upper_minutes
    ]
    if lower_minutes <= threshold <= upper_minutes:
        values.append(float(threshold))
    values = sorted(set(values))

    # Retain at most nine well-spaced labels, while always retaining the threshold.
    if len(values) > 9:
        required = {float(threshold)} if threshold in values else set()
        target_indices = np.linspace(0, len(values) - 1, 9).round().astype(int)
        selected = {values[index] for index in target_indices} | required
        while len(selected) > 9:
            removable = sorted(selected - required)
            selected.remove(removable[len(removable) // 2])
        values = sorted(selected)

    labels = [human_duration_label(value) for value in values]
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_formatter(NullFormatter())


def add_duration_reference_lines(
    ax: plt.Axes, lower_minutes: float, upper_minutes: float
) -> None:
    """Add unobtrusive 24-hour, one-week, and 30-day reference lines."""
    major_ticks = {
        float(value) for value in ax.xaxis.get_majorticklocs() if np.isfinite(value)
    }
    for value, label in ((1_440.0, "24 h"), (10_080.0, "1 wk"), (43_200.0, "30 d")):
        if lower_minutes <= value <= upper_minutes:
            ax.axvline(value, color="#5f6368", lw=0.65, alpha=0.22, zorder=0)
            if value not in major_ticks:
                ax.annotate(
                    label, xy=(value, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(3, -7), textcoords="offset points", rotation=90,
                    va="top", ha="left", fontsize=8, color="#5f6368",
                )


def safe_filename(value: object, max_length: int = 80) -> str:
    """Return a stable lowercase filename component."""
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text[:max_length].rstrip("_") or "unknown")


def infer_year(path: Path) -> str:
    """Infer a single four-digit year from a filename when possible."""
    years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", path.stem)
    return years[0] if len(set(years)) == 1 else ""


def file_sha256(path: Path) -> str:
    """Calculate a source-file digest for the no-modification check."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_company_folders(data_dir: Path, companies: list[str]) -> list[str]:
    """Require every requested company folder and report the detected layout."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    detected = sorted(path.name.lower() for path in data_dir.iterdir() if path.is_dir())
    missing = sorted(set(companies) - set(detected))
    unexpected = sorted(set(detected) - set(companies))
    if missing:
        raise RuntimeError(
            "Company-folder validation failed.\n"
            f"Expected companies: {', '.join(companies)}\n"
            f"Detected companies: {', '.join(detected) or '(none)'}\n"
            f"Missing companies: {', '.join(missing)}\n"
            f"Unexpected folders: {', '.join(unexpected) or '(none)'}"
        )
    LOGGER.info("Expected companies: %s", ", ".join(companies))
    LOGGER.info("Detected company folders: %s", ", ".join(detected))
    LOGGER.info("Unexpected folders: %s", ", ".join(unexpected) or "none")
    return detected


def discover_files(
    data_dir: Path, companies: list[str], exclude_anglian_2024: bool
) -> dict[str, list[Path]]:
    """Discover source CSVs recursively and apply the optional exclusion."""
    result: dict[str, list[Path]] = {}
    for company in companies:
        files = sorted((data_dir / company).rglob("*.csv"))
        if exclude_anglian_2024 and company == "anglian":
            excluded = [path for path in files if infer_year(path) == "2024"]
            for path in excluded:
                LOGGER.info("Excluding Anglian 2024 file: %s", path)
            files = [path for path in files if path not in excluded]
        if not files:
            raise RuntimeError(f"No included CSV files found for company: {company}")
        result[company] = files
    return result


def prepare_output_directory(
    output_root: Path, threshold: float, overwrite: bool, data_dir: Path
) -> Path:
    """Create a safe threshold-specific output directory."""
    output_dir = output_root / f"gt_{filename_number(threshold)}min"
    resolved = output_dir.resolve()
    root_resolved = output_root.resolve()
    data_resolved = data_dir.resolve()
    if resolved == data_resolved or data_resolved in resolved.parents:
        raise RuntimeError("Output directory cannot be inside the source data directory")
    if root_resolved not in resolved.parents:
        raise RuntimeError("Threshold output directory escaped --output-root")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    for relative in ("tables", "sensor_events", "figures"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_logging(log_path: Path) -> None:
    """Log to both the terminal and the run-specific log file."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def cleaned_chunks(
    path: Path, company: str, round_to_15: bool, quality: FileQuality
) -> Iterator[pd.DataFrame]:
    """Yield validated chunks while updating a transparent quality record."""
    try:
        header = pd.read_csv(path, nrows=0)
    except pd.errors.EmptyDataError:
        quality.status = "empty_file"
        quality.error_message = "CSV is empty"
        LOGGER.error("Empty CSV: %s", path)
        return
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        quality.status = "malformed_or_unreadable"
        quality.error_message = str(exc)
        LOGGER.error("Could not read CSV header %s: %s", path, exc)
        return

    rename = {column: str(column).strip() for column in header.columns}
    stripped = list(rename.values())
    missing = [column for column in MANDATORY_COLUMNS if column not in stripped]
    if missing:
        quality.status = "missing_columns"
        quality.missing_columns = ";".join(missing)
        quality.error_message = "Mandatory columns missing"
        LOGGER.error("Missing columns %s in %s", missing, path)
        return
    permit_source = next(original for original, clean in rename.items() if clean == "permit_number")
    try:
        reader = pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            dtype={permit_source: "string"},
            on_bad_lines="error",
            low_memory=False,
        )
        for raw in reader:
            raw.rename(columns=lambda value: str(value).strip(), inplace=True)
            quality.rows_read += len(raw)
            raw["permit_number"] = raw["permit_number"].astype("string").str.strip()
            raw["location_name"] = raw["location_name"].astype("string").str.strip()
            duration = pd.to_numeric(raw["duration_minutes"], errors="coerce")
            missing_permit = raw["permit_number"].isna() | raw["permit_number"].eq("")
            non_numeric = duration.isna()
            finite = pd.Series(np.isfinite(duration.to_numpy(dtype=float, na_value=np.nan)), index=raw.index)
            non_finite = duration.notna() & ~finite
            non_positive = duration.notna() & finite & (duration <= 0)
            valid = ~(missing_permit | non_numeric | non_finite | non_positive)

            quality.missing_permit_number += int(missing_permit.sum())
            quality.non_numeric_duration += int((non_numeric & ~missing_permit).sum())
            quality.non_finite_duration += int((non_finite & ~missing_permit).sum())
            quality.non_positive_duration += int((non_positive & ~missing_permit).sum())
            quality.rejected_rows += int((~valid).sum())

            cleaned = raw.loc[valid].copy()
            cleaned["duration_minutes"] = duration.loc[valid].to_numpy(dtype=np.float64)
            if round_to_15 and not cleaned.empty:
                cleaned["duration_minutes"] = (
                    np.ceil(cleaned["duration_minutes"] / 15.0) * 15.0
                )
            for column in ("start_time", "stop_time"):
                original = cleaned[column]
                parsed = pd.to_datetime(original, errors="coerce")
                invalid = parsed.isna() & original.notna() & original.astype(str).str.strip().ne("")
                if column == "start_time":
                    quality.invalid_start_time += int(invalid.sum())
                else:
                    quality.invalid_stop_time += int(invalid.sum())
                cleaned[column] = parsed
            cleaned["company"] = company
            cleaned["source_file"] = str(path)
            cleaned["source_year"] = infer_year(path)
            quality.valid_rows += len(cleaned)
            yield cleaned
        quality.status = "processed"
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        quality.status = "malformed_or_unreadable"
        quality.error_message = str(exc)
        LOGGER.error("Error reading %s: %s", path, exc)


def read_company(
    company: str, files: list[Path], round_to_15: bool
) -> tuple[pd.DataFrame, list[FileQuality]]:
    """Read one company's files, retaining all valid event columns."""
    frames: list[pd.DataFrame] = []
    qualities: list[FileQuality] = []
    progress = TerminalProgress(f"{company}: input files", len(files))
    for index, path in enumerate(files, 1):
        LOGGER.info("Reading %s file %d/%d: %s", company, index, len(files), path)
        quality = FileQuality(company, str(path), infer_year(path))
        file_frames = list(cleaned_chunks(path, company, round_to_15, quality))
        if file_frames:
            frame = pd.concat(file_frames, ignore_index=True)
            duplicate_columns = [column for column in MANDATORY_COLUMNS if column in frame]
            quality.duplicate_rows = int(frame.duplicated(subset=duplicate_columns, keep=False).sum())
            frames.append(frame)
        qualities.append(quality)
        progress.update(suffix=path.name)
    progress.close(suffix="input read complete")
    if not frames:
        raise RuntimeError(f"No valid event rows were read for company: {company}")
    pooled = pd.concat(frames, ignore_index=True, sort=False)
    return pooled, qualities


def canonical_locations(events: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Choose the most frequent non-empty name and report inconsistent names."""
    usable = events.loc[
        events["location_name"].notna() & events["location_name"].ne(""),
        ["permit_number", "location_name"],
    ]
    counts = (
        usable.groupby(["permit_number", "location_name"], dropna=False)
        .size().rename("event_count").reset_index()
    )
    ordered = counts.sort_values(
        ["permit_number", "event_count", "location_name"],
        ascending=[True, False, True], kind="mergesort",
    )
    canonical = ordered.drop_duplicates("permit_number").set_index("permit_number")["location_name"]
    distinct = counts.groupby("permit_number")["location_name"].nunique()
    inconsistent_permits = distinct[distinct > 1].index
    inconsistent = counts[counts["permit_number"].isin(inconsistent_permits)].copy()
    inconsistent["canonical_location_name"] = inconsistent["permit_number"].map(canonical)
    inconsistent["number_of_location_names"] = inconsistent["permit_number"].map(distinct)
    return canonical, inconsistent


def sensor_statistics(events: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Calculate pooled all-event and strictly-above-threshold sensor statistics."""
    canonical, _ = canonical_locations(events)
    events = events.copy()
    events["is_tail_spill"] = events["duration_minutes"] > threshold
    grouped = events.groupby("permit_number", sort=False)
    overall = grouped.agg(
        total_spill_count=("duration_minutes", "size"),
        total_duration_minutes=("duration_minutes", "sum"),
        overall_max_duration_minutes=("duration_minutes", "max"),
        first_event_time=("start_time", "min"),
        last_event_time=("start_time", "max"),
        number_of_source_files=("source_file", "nunique"),
    )
    years = grouped["source_year"].apply(
        lambda values: ";".join(sorted({str(value) for value in values if str(value).strip()}))
    ).rename("years_present")
    tail = events[events["is_tail_spill"]].groupby("permit_number").agg(
        tail_spill_count=("duration_minutes", "size"),
        tail_total_duration_minutes=("duration_minutes", "sum"),
        tail_mean_duration_minutes=("duration_minutes", "mean"),
        tail_median_duration_minutes=("duration_minutes", "median"),
        tail_max_duration_minutes=("duration_minutes", "max"),
    )
    stats = overall.join(years).join(tail).reset_index()
    stats.insert(0, "company", str(events["company"].iloc[0]))
    stats.insert(2, "canonical_location_name", stats["permit_number"].map(canonical).fillna(""))
    stats["tail_spill_count"] = stats["tail_spill_count"].fillna(0).astype(int)
    stats["tail_total_duration_minutes"] = stats["tail_total_duration_minutes"].fillna(0.0)
    return stats


def select_sensors(stats: pd.DataFrame, top_n: int, threshold: float) -> pd.DataFrame:
    """Select exactly top_n sensors using the required deterministic tail order."""
    eligible = stats[stats["tail_spill_count"] > 0].copy()
    company = str(stats["company"].iloc[0])
    if len(eligible) < top_n:
        raise RuntimeError(
            f"Company {company} has only {len(eligible)} sensors with spills > "
            f"{display_number(threshold)} minutes; {top_n} are required."
        )
    selected = eligible.sort_values(
        ["tail_spill_count", "tail_total_duration_minutes", "tail_max_duration_minutes", "permit_number"],
        ascending=[False, False, False, True], kind="mergesort",
    ).head(top_n).copy()
    selected.insert(1, "rank_within_company", np.arange(1, top_n + 1))
    selected.insert(4, "tail_threshold_minutes", threshold)
    selected.insert(5, "tail_condition", f"duration_minutes > {display_number(threshold)}")
    return selected


def fit_stretched_exponential(tail: np.ndarray, threshold: float) -> FitResult:
    """Fit the conditional model by maximum likelihood to raw tail observations."""
    tail = np.asarray(tail, dtype=np.float64)
    if tail.size == 0 or np.any(~np.isfinite(tail)) or np.any(tail <= threshold):
        raise ValueError("Tail values must be finite and strictly above the threshold")
    log_x = np.log(tail)
    log_threshold = math.log(threshold)
    initial_log_lambda = math.log(1.0 / float(np.median(tail)))

    def nll(parameters: np.ndarray) -> float:
        log_lambda, beta = float(parameters[0]), float(parameters[1])
        if not math.isfinite(log_lambda) or not math.isfinite(beta) or beta <= 0:
            return np.finfo(float).max
        z = beta * (log_lambda + log_x)
        zmin = beta * (log_lambda + log_threshold)
        if zmin > 700 or np.any(z > 700):
            return np.finfo(float).max
        log_pdf = (
            math.log(beta) + log_lambda
            + (beta - 1.0) * (log_lambda + log_x)
            - np.exp(z) + math.exp(zmin)
        )
        value = -float(np.sum(log_pdf, dtype=np.float64))
        return value if math.isfinite(value) else np.finfo(float).max

    result = minimize(
        nll,
        x0=np.array([initial_log_lambda, 0.18]),
        method="L-BFGS-B",
        bounds=[(-50.0, 20.0), (0.01, 2.0)],
        options={"maxiter": 1_000, "ftol": 1e-11},
    )
    fitted_lambda = math.exp(float(result.x[0]))
    beta = float(result.x[1])
    success = bool(
        result.success and math.isfinite(beta) and beta > 0
        and math.isfinite(fitted_lambda) and fitted_lambda > 0
        and math.isfinite(float(result.fun))
    )
    return FitResult(
        beta if success else math.nan,
        fitted_lambda if success else math.nan,
        float(result.fun) if success else math.nan,
        success,
        str(result.message),
    )


def stretched_exponential_density(
    x: np.ndarray, threshold: float, beta: float, fitted_lambda: float
) -> np.ndarray:
    """Evaluate the fitted conditional density safely."""
    log_lambda_x = np.log(fitted_lambda) + np.log(x)
    log_pdf = (
        math.log(beta) + math.log(fitted_lambda)
        + (beta - 1.0) * log_lambda_x
        - np.exp(beta * log_lambda_x)
        + math.exp(beta * (math.log(fitted_lambda) + math.log(threshold)))
    )
    return np.exp(np.clip(log_pdf, -745.0, 700.0))


def plot_sensor_tail(
    tail: np.ndarray,
    row: pd.Series,
    threshold: float,
    bins: int,
    fit: FitResult,
    output_path: Path,
) -> None:
    """Create one log-log empirical-density and fitted-model figure."""
    maximum = float(np.max(tail))
    edges = np.geomspace(threshold, np.nextafter(maximum, np.inf), bins + 1)
    counts = np.histogram(tail, bins=edges)[0]
    centres = np.sqrt(edges[:-1] * edges[1:])
    density = counts / (tail.size * np.diff(edges))
    visible = (density > 0) & np.isfinite(density)

    fig, ax = plt.subplots(figsize=(9.6, 6.4), constrained_layout=True)
    ax.plot(
        centres[visible], density[visible], linestyle="none", marker="x",
        color="#1d2733", markersize=5.5, markeredgewidth=1.1,
        label="Empirical tail density", zorder=3,
    )
    if fit.fit_success:
        x_fit = np.geomspace(threshold, maximum, 500)
        y_fit = stretched_exponential_density(
            x_fit, threshold, fit.fitted_beta, fit.fitted_lambda_per_minute
        )
        valid_fit = (y_fit > 0) & np.isfinite(y_fit)
        ax.plot(
            x_fit[valid_fit], y_fit[valid_fit], color="#c62828", lw=1.8,
            label=(
                "Stretched exponential fit\n"
                f"beta = {fit.fitted_beta:.3f}\n"
                f"lambda = {fit.fitted_lambda_per_minute:.3e} min$^{{-1}}$\n"
                f"n_tail = {tail.size:,}"
            ),
        )
    else:
        ax.text(
            0.5, 0.5, "Insufficient tail events for reliable fit"
            if "Insufficient" in fit.optimiser_message else "Fit unsuccessful\n" + fit.optimiser_message,
            transform=ax.transAxes, ha="center", va="center", fontsize=10,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#aaaaaa"},
        )
    ax.axvline(
        threshold, color="#6a3d9a", linestyle="--", lw=1.35,
        label=(
            f"threshold = {display_number(threshold)} min "
            f"({display_number(threshold / 60.0)} h)"
        ),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(threshold, maximum)
    set_human_duration_ticks(ax, threshold, maximum, threshold)
    add_duration_reference_lines(ax, threshold, maximum)
    ax.set_xlabel("CSO spill duration (hours/days; model fitted in minutes)")
    ax.set_ylabel("Probability density")
    ax.set_title(
        f"{str(row['company']).replace('_', ' ').title()} | Rank {int(row['rank_within_company'])} "
        f"| Permit {row['permit_number']}\n"
        f"{row['canonical_location_name']} | Spill-duration tail > {display_number(threshold)} minutes",
        pad=12,
    )
    ax.grid(which="major", color="#d8d8d8", lw=0.6, alpha=0.65)
    ax.grid(which="minor", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def relative_path(path: Path, output_dir: Path) -> str:
    """Store portable paths relative to the threshold output directory."""
    return path.relative_to(output_dir).as_posix()


def process_selected_company(
    events: pd.DataFrame,
    selected: pd.DataFrame,
    threshold: float,
    bins: int,
    min_tail_events: int,
    output_dir: Path,
    combined_path: Path,
    combined_header_written: bool,
) -> tuple[list[dict[str, object]], bool]:
    """Save events, fit tails, make plots, and append the combined event table."""
    company = str(selected["company"].iloc[0])
    fit_rows: list[dict[str, object]] = []
    company_events_dir = output_dir / "sensor_events" / company
    company_figures_dir = output_dir / "figures" / company
    company_events_dir.mkdir(parents=True, exist_ok=True)
    company_figures_dir.mkdir(parents=True, exist_ok=True)
    threshold_token = filename_number(threshold)

    progress = TerminalProgress(f"{company}: selected sensors", len(selected))
    for _, row in selected.sort_values("rank_within_company").iterrows():
        rank = int(row["rank_within_company"])
        permit = str(row["permit_number"])
        permit_token = safe_filename(permit, 60)
        sensor = events[events["permit_number"] == permit].copy()
        sensor["rank_within_company"] = rank
        sensor["canonical_location_name"] = row["canonical_location_name"]
        sensor["tail_threshold_minutes"] = threshold
        sensor["is_tail_spill"] = sensor["duration_minutes"] > threshold
        metadata = [
            "company", "rank_within_company", "permit_number",
            "canonical_location_name", "tail_threshold_minutes", "is_tail_spill",
            "source_file", "source_year",
        ]
        other = [column for column in sensor.columns if column not in metadata]
        sensor = sensor[metadata + other].sort_values(
            ["company", "rank_within_company", "permit_number", "start_time"],
            kind="mergesort", na_position="last",
        )
        tail_events = sensor[sensor["is_tail_spill"]].copy()
        all_path = company_events_dir / f"rank_{rank:02d}_{permit_token}_all_events.csv"
        tail_path = company_events_dir / (
            f"rank_{rank:02d}_{permit_token}_tail_events_gt_{threshold_token}min.csv"
        )
        figure_path = company_figures_dir / (
            f"rank_{rank:02d}_{permit_token}_heavy_tail_gt_{threshold_token}min.png"
        )
        sensor.to_csv(all_path, index=False)
        tail_events.to_csv(tail_path, index=False)
        sensor.to_csv(combined_path, mode="a", header=not combined_header_written, index=False)
        combined_header_written = True

        tail_values = tail_events["duration_minutes"].to_numpy(dtype=np.float64)
        if tail_values.size < min_tail_events:
            fit = FitResult(
                math.nan, math.nan, math.nan, False,
                f"Insufficient tail events for reliable fit: {tail_values.size} < {min_tail_events}",
            )
        else:
            try:
                fit = fit_stretched_exponential(tail_values, threshold)
            except (ValueError, FloatingPointError, OverflowError) as exc:
                fit = FitResult(math.nan, math.nan, math.nan, False, str(exc))
        plot_sensor_tail(tail_values, row, threshold, bins, fit, figure_path)
        fit_rows.append({
            "company": company,
            "rank_within_company": rank,
            "permit_number": permit,
            "canonical_location_name": row["canonical_location_name"],
            "tail_threshold_minutes": threshold,
            "total_spill_count": int(len(sensor)),
            "tail_spill_count": int(len(tail_events)),
            "number_of_tail_events_used": int(len(tail_events)),
            "fitted_beta": fit.fitted_beta,
            "fitted_lambda_per_minute": fit.fitted_lambda_per_minute,
            "negative_log_likelihood": fit.negative_log_likelihood,
            "fit_success": fit.fit_success,
            "optimiser_message": fit.optimiser_message,
            "figure_path": relative_path(figure_path, output_dir),
            "all_events_csv_path": relative_path(all_path, output_dir),
            "tail_events_csv_path": relative_path(tail_path, output_dir),
        })
        progress.update(suffix=f"permit {permit}")
    progress.close(suffix="sensor outputs complete")
    return fit_rows, combined_header_written


def write_readme(
    path: Path, args: argparse.Namespace, input_files: int, rows: int,
    invalid: int, sensors: int, figures: int, successful: int, failed: int,
) -> None:
    """Write a concise, human-readable run summary."""
    expected_sensors = args.expected_company_count * args.top_n
    lines = [
        "Individual CSO heavy-tail analysis",
        "==================================",
        "",
        f"Threshold used: {display_number(args.threshold)} minutes",
        f"Tail condition: duration_minutes > {display_number(args.threshold)} (strictly greater than)",
        f"Companies analysed: {', '.join(args.companies)}",
        f"Number of companies: {len(args.companies)}",
        f"Top sensors selected per company: {args.top_n}",
        f"Expected selected sensors: {expected_sensors}",
        f"Actual selected sensors: {sensors}",
        f"Expected figure count: {expected_sensors}",
        f"Actual figure count: {figures}",
        f"Input files processed: {input_files}",
        f"Input rows processed: {rows}",
        f"Invalid row count: {invalid}",
        f"Successful fit count: {successful}",
        f"Failed fit count: {failed}",
        f"Duration rounding enabled: {args.round_to_15}",
        f"Anglian 2024 excluded: {args.exclude_anglian_2024}",
        "",
        "Main tables: tables/selected_sensor_manifest.csv,",
        "tables/individual_sensor_fit_summary.csv,",
        "tables/all_selected_sensor_events.csv, and tables/data_quality_report.csv",
        "Figures directory: figures/",
        "The fitted parameters reported are beta and lambda.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    manifest: pd.DataFrame,
    fit_summary: pd.DataFrame,
    source_hashes_before: dict[Path, str],
) -> tuple[int, int, int]:
    """Run every required structural, numerical, and source-integrity check."""
    expected = args.expected_company_count * args.top_n
    if len(args.companies) != args.expected_company_count:
        raise AssertionError("Analysed company count is incorrect")
    if args.companies == EXPECTED_COMPANIES and set(manifest["company"]) != set(EXPECTED_COMPANIES):
        raise AssertionError("The default run did not analyse the eight expected companies")
    counts = manifest.groupby("company").size()
    if set(counts.index) != set(args.companies) or not (counts == args.top_n).all():
        raise AssertionError("Every company must have exactly --top-n selected sensors")
    if len(manifest) != expected or len(fit_summary) != expected:
        raise AssertionError("Manifest or fit-summary row count is incorrect")
    if manifest.duplicated(["company", "permit_number"]).any():
        raise AssertionError("Selected company/permit pairs are not unique")
    for _, group in manifest.groupby("company"):
        expected_order = group.sort_values(
            ["tail_spill_count", "tail_total_duration_minutes", "tail_max_duration_minutes", "permit_number"],
            ascending=[False, False, False, True], kind="mergesort",
        )["permit_number"].tolist()
        actual_order = group.sort_values("rank_within_company")["permit_number"].tolist()
        if actual_order != expected_order:
            raise AssertionError("Ranking validation failed")
    all_files = list((output_dir / "sensor_events").rglob("rank_*_all_events.csv"))
    tail_files = list((output_dir / "sensor_events").rglob("rank_*_tail_events_gt_*min.csv"))
    figures = list((output_dir / "figures").rglob("*.png"))
    if not (len(all_files) == len(tail_files) == len(figures) == expected):
        raise AssertionError("Per-sensor output file counts are incorrect")
    for path in tail_files:
        values = pd.to_numeric(pd.read_csv(path, usecols=["duration_minutes"])["duration_minutes"], errors="coerce")
        if values.isna().any() or not (values > args.threshold).all():
            raise AssertionError(f"Tail-only file contains an invalid row: {path}")
    successful = fit_summary["fit_success"].astype(bool)
    for column in ("fitted_beta", "fitted_lambda_per_minute"):
        values = pd.to_numeric(fit_summary.loc[successful, column], errors="coerce")
        if not (np.isfinite(values) & (values > 0)).all():
            raise AssertionError(f"Successful fits contain invalid {column}")
    forbidden = {"reciprocal_lambda", "inverse_lambda"}
    for path in (output_dir / "tables").glob("*.csv"):
        columns = {str(column).lower() for column in pd.read_csv(path, nrows=0).columns}
        if columns & forbidden:
            raise AssertionError(f"Forbidden output columns found in {path}")
    for path, digest in source_hashes_before.items():
        if not path.exists() or file_sha256(path) != digest:
            raise AssertionError(f"Source CSV changed during the run: {path}")
    return len(figures), int(successful.sum()), int((~successful).sum())


def main(argv: Sequence[str] | None = None) -> None:
    """Run the complete analysis and validate all outputs."""
    args = parse_args(argv)
    detected = validate_company_folders(args.data_dir, args.companies)
    files_by_company = discover_files(
        args.data_dir, args.companies, args.exclude_anglian_2024
    )
    all_source_files = [path for company in args.companies for path in files_by_company[company]]
    source_hashes = {path: file_sha256(path) for path in all_source_files}
    output_dir = prepare_output_directory(
        args.output_root, args.threshold, args.overwrite, args.data_dir
    )
    configure_logging(output_dir / "analysis.log")
    LOGGER.info("Starting analysis of spills > %s minutes", display_number(args.threshold))

    config = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "output_root": str(args.output_root),
        "threshold_minutes": args.threshold,
        "tail_condition": f"duration_minutes > {display_number(args.threshold)}",
        "top_n_per_company": args.top_n,
        "companies": args.companies,
        "expected_company_count": args.expected_company_count,
        "number_of_bins": args.bins,
        "minimum_tail_events_for_fit": args.min_tail_events,
        "round_to_15_minutes": args.round_to_15,
        "exclude_anglian_2024": args.exclude_anglian_2024,
        "overwrite": args.overwrite,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    manifests: list[pd.DataFrame] = []
    fits: list[dict[str, object]] = []
    qualities: list[FileQuality] = []
    inconsistent_tables: list[pd.DataFrame] = []
    combined_path = output_dir / "tables" / "all_selected_sensor_events.csv"
    combined_header_written = False
    eligible_shortfalls: list[str] = []

    company_progress = TerminalProgress("Heavy-tail companies", len(args.companies))
    for company in args.companies:
        events, company_quality = read_company(
            company, files_by_company[company], args.round_to_15
        )
        qualities.extend(company_quality)
        canonical, inconsistent = canonical_locations(events)
        del canonical
        if not inconsistent.empty:
            inconsistent.insert(0, "company", company)
            inconsistent_tables.append(inconsistent)
        stats = sensor_statistics(events, args.threshold)
        try:
            selected = select_sensors(stats, args.top_n, args.threshold)
        except RuntimeError as exc:
            eligible_shortfalls.append(str(exc))
            raise
        manifests.append(selected)
        company_fits, combined_header_written = process_selected_company(
            events, selected, args.threshold, args.bins, args.min_tail_events,
            output_dir, combined_path, combined_header_written,
        )
        fits.extend(company_fits)
        LOGGER.info("Completed %s: selected %d sensors", company, len(selected))
        company_progress.update(suffix=company)
        del events, stats, selected
    company_progress.close(suffix="all companies complete")

    manifest = pd.concat(manifests, ignore_index=True).sort_values(
        ["company", "rank_within_company"], kind="mergesort"
    )
    manifest_columns = [
        "company", "rank_within_company", "permit_number", "canonical_location_name",
        "tail_threshold_minutes", "tail_condition", "tail_spill_count",
        "tail_total_duration_minutes", "tail_mean_duration_minutes",
        "tail_median_duration_minutes", "tail_max_duration_minutes",
        "total_spill_count", "total_duration_minutes", "overall_max_duration_minutes",
        "first_event_time", "last_event_time", "years_present", "number_of_source_files",
    ]
    manifest = manifest[manifest_columns]
    fit_summary = pd.DataFrame(fits).sort_values(
        ["company", "rank_within_company"], kind="mergesort"
    )
    quality_table = pd.DataFrame(asdict(item) for item in qualities)
    inconsistent_table = (
        pd.concat(inconsistent_tables, ignore_index=True)
        if inconsistent_tables else pd.DataFrame(columns=[
            "company", "permit_number", "location_name", "event_count",
            "canonical_location_name", "number_of_location_names",
        ])
    )
    failed = fit_summary[~fit_summary["fit_success"].astype(bool)].copy()
    manifest.to_csv(output_dir / "tables" / "selected_sensor_manifest.csv", index=False)
    fit_summary.to_csv(output_dir / "tables" / "individual_sensor_fit_summary.csv", index=False)
    quality_table.to_csv(output_dir / "tables" / "data_quality_report.csv", index=False)
    inconsistent_table.to_csv(output_dir / "tables" / "inconsistent_location_names.csv", index=False)
    failed.to_csv(output_dir / "tables" / "skipped_or_failed_fits.csv", index=False)

    figure_count, successful_count, failed_count = validate_outputs(
        args, output_dir, manifest, fit_summary, source_hashes
    )
    rows_read = int(quality_table["rows_read"].sum())
    invalid_rows = int(quality_table["rejected_rows"].sum())
    processed_files = int((quality_table["status"] == "processed").sum())
    write_readme(
        output_dir / "README.txt", args, processed_files, rows_read, invalid_rows,
        len(manifest), figure_count, successful_count, failed_count,
    )

    expected_sensors = args.expected_company_count * args.top_n
    print(f"Expected companies: {args.expected_company_count}")
    print(f"Actual companies: {manifest['company'].nunique()}")
    print(f"Top sensors per company: {args.top_n}")
    print(f"Expected selected sensors: {expected_sensors}")
    print(f"Actual selected sensors: {len(manifest)}")
    print(f"Expected heavy-tail figures: {expected_sensors}")
    print(f"Actual heavy-tail figures: {figure_count}")
    print(f"Detected companies: {', '.join(detected)}")
    print(f"Included companies: {', '.join(args.companies)}")
    print(f"Successful fits: {successful_count}")
    print(f"Failed fits: {failed_count}")
    print(f"Output directory: {output_dir}")
    if eligible_shortfalls:
        print("Companies with too few eligible sensors: " + "; ".join(eligible_shortfalls))
    else:
        print("Companies with too few eligible sensors: none")
    print("Analysis complete")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, AssertionError) as exc:
        LOGGER.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

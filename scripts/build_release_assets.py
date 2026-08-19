"""Build public-safe headline figures and copy curated aggregate figures.

The script reads authoritative products from the parent research workspace but
never copies the underlying event or catchment master tables into the release.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image

RELEASE = Path(__file__).resolve().parents[1]
WORKSPACE = RELEASE.parent
HEAD = RELEASE / "outputs" / "headline_figures"
SYNTH = RELEASE / "08_final_synthesis" / "figures"
MLFIG = RELEASE / "07_machine_learning" / "figures"

BLUE = "#184E77"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
RED = "#D1495B"
INK = "#233142"
GRID = "#D9E2EA"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 13,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#8091A5",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
    })


def save(fig: plt.Figure, name: str) -> None:
    for folder in (HEAD, SYNTH):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def copy_curated() -> None:
    original = WORKSPACE / "machine learning" / "NEW MACHINE LEARNING"
    regional = WORKSPACE / "REGIONAL_CLUSTER_AND_UNIVARIATE_ML" / "01_regional_cluster_model"
    for approach in ("original_selected_sensor", "regional_cluster"):
        (MLFIG / approach).mkdir(parents=True, exist_ok=True)

    original_models = {
        "beta": "model_1_beta",
        "lambda": "model_2_scale",
        "duration": "model_3_duration",
        "frequency": "model_4_count",
    }
    for target, folder in original_models.items():
        for kind in ("predicted_vs_observed", "learning_curve", "permutation_importance"):
            src = original / folder / "figures" / f"{kind}.pdf"
            if src.exists():
                shutil.copy2(src, MLFIG / "original_selected_sensor" / f"{target}_{kind}.pdf")

    for target in ("beta", "lambda", "duration", "frequency"):
        for kind in ("predicted_vs_observed", "learning_curve", "feature_importance"):
            for ext in ("pdf", "png"):
                src = regional / "figures" / f"{target}_{kind}.{ext}"
                if src.exists():
                    shutil.copy2(src, MLFIG / "regional_cluster" / src.name)
    for stem in ("cluster_radius_model_sensitivity",):
        for ext in ("pdf", "png"):
            src = regional / "figures" / f"{stem}.{ext}"
            if src.exists():
                shutil.copy2(src, MLFIG / "regional_cluster" / src.name)

    stat = WORKSPACE / "PROJECT_RESULTS" / "four_outcome_annual_nimrod_release_20260813_130420" / "statistical_analysis"
    copies = {
        stat / "cross_outcome" / "four_outcome_univariate_summary.pdf": RELEASE / "06_univariate_analysis" / "results" / "four_outcome_univariate_summary.pdf",
        stat / "cross_outcome" / "all_variables_correlation_matrix.pdf": RELEASE / "06_univariate_analysis" / "results" / "all_variables_correlation_matrix.pdf",
    }
    for src, dst in copies.items():
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def pdf_preview(source: Path, destination: Path, page: int = 0) -> None:
    """Render one vector-PDF page to a high-resolution PNG preview."""
    doc = pymupdf.open(source)
    pix = doc[page].get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8), alpha=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pix.save(destination)


def grid_figure(items: list[tuple[str, Path]], stem: str, title: str, columns: int = 4) -> None:
    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.1 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (label, path) in zip(axes.flat, items):
        ax.imshow(Image.open(path))
        ax.set_title(label, fontsize=10, color=INK)
        ax.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, stem)


def requested_headline_set() -> None:
    """Create the exact small headline set requested by the release specification."""
    # Stable aliases retain descriptive internal filenames while satisfying the public index.
    aliases = {
        "national_overview": "national_wwtw_overview",
        "duration_tail": "duration_tail_survival",
        "beta_lambda_interpretation": "beta_log_lambda_relationship",
        "final_synthesis": "original_vs_regional_rmse_gain",
    }
    for public, internal in aliases.items():
        for ext in ("png", "pdf"):
            shutil.copy2(HEAD / f"{internal}.{ext}", HEAD / f"{public}.{ext}")
            (HEAD / f"{internal}.{ext}").unlink()

    stat = WORKSPACE / "PROJECT_RESULTS" / "four_outcome_annual_nimrod_release_20260813_130420" / "statistical_analysis"
    univariate = {
        "beta_univariate_forest": stat / "beta_parameter_analysis" / "beta_analysis_figures.pdf",
        "lambda_univariate_forest": stat / "scale_parameter_analysis" / "scale_analysis_figures.pdf",
    }
    for stem, source in univariate.items():
        shutil.copy2(source, HEAD / f"{stem}.pdf")
        pdf_preview(source, HEAD / f"{stem}.png")

    duration = stat / "spill_duration_analysis" / "spill_duration_analysis_figures.pdf"
    frequency = stat / "spill_count_analysis" / "spill_count_analysis_figures.pdf"
    doc = pymupdf.open()
    for source in (duration, frequency):
        part = pymupdf.open(source)
        doc.insert_pdf(part, from_page=0, to_page=0)
    doc.save(HEAD / "duration_frequency_univariate_forest.pdf")
    dur_png = HEAD / "_duration_univariate_preview.png"
    freq_png = HEAD / "_frequency_univariate_preview.png"
    pdf_preview(duration, dur_png)
    pdf_preview(frequency, freq_png)
    grid_figure([("Mean duration", dur_png), ("Annualised frequency", freq_png)], "duration_frequency_univariate_forest", "Univariate evidence for observed outcomes", columns=2)
    dur_png.unlink(missing_ok=True)
    freq_png.unlink(missing_ok=True)

    targets = ("beta", "lambda", "duration", "frequency")
    kinds = {
        "predicted_vs_observed": "Locked-test predictions",
        "learning_curve": "Learning curves",
        "permutation_importance": "Predictor importance",
    }
    for kind, title in kinds.items():
        items: list[tuple[str, Path]] = []
        for target in targets:
            original_pdf = MLFIG / "original_selected_sensor" / f"{target}_{kind}.pdf"
            original_png = MLFIG / "original_selected_sensor" / f"{target}_{kind}.png"
            pdf_preview(original_pdf, original_png)
            items.append((f"Original - {target}", original_png))
        regional_kind = "feature_importance" if kind == "permutation_importance" else kind
        for target in targets:
            items.append((f"Regional - {target}", MLFIG / "regional_cluster" / f"{target}_{regional_kind}.png"))
        public_stem = {"predicted_vs_observed": "ml_predicted_vs_observed", "learning_curve": "ml_learning_curves", "permutation_importance": "ml_predictor_importance"}[kind]
        grid_figure(items, public_stem, f"{title}: original and regional pathways")


def comparison_figures() -> None:
    score = pd.read_csv(RELEASE / "outputs" / "headline_tables" / "final_ml_scorecard.csv")
    order = ["beta", "scale", "duration", "frequency"]
    labels = ["Beta", "Lambda", "Duration", "Frequency"]
    old = score[score.approach.str.startswith("original")].set_index("target")
    reg = score[score.approach.str.startswith("regional")].set_index("target")
    reg = reg.rename(index={"lambda": "scale"})
    x = np.arange(4)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - .19, old.loc[order, "rmse_improvement_pct"], .38, color=BLUE, label="Original: selected sensor + WWTW")
    ax.bar(x + .19, reg.loc[order, "rmse_improvement_pct"], .38, color=TEAL, label="Regional: company x 25 km cluster")
    ax.axhline(0, color=INK, lw=.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Locked-test RMSE improvement over Dummy (%)")
    ax.set_title("Regional aggregation strengthens beta, but not every target")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=2, fontsize=8)
    ax.text(.01, -.22, "Positive values beat the target-specific Dummy baseline. Frequency worsens regionally.", transform=ax.transAxes, color="#526477")
    fig.tight_layout()
    save(fig, "original_vs_regional_rmse_gain")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - .19, old.loc[order, "r2_model_scale"], .38, color=BLUE, label="Original")
    ax.bar(x + .19, reg.loc[order, "r2_model_scale"], .38, color=TEAL, label="Regional")
    ax.axhline(0, color=INK, lw=.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Locked-test R-squared (model scale)")
    ax.set_title("Out-of-sample explanatory performance")
    ax.grid(axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "original_vs_regional_r2")


def national_and_tail_figures() -> None:
    master_path = WORKSPACE / "PROJECT_RESULTS" / "four_outcome_annual_nimrod_release_20260813_130420" / "data" / "scientific_master_annual_nimrod_v1.csv"
    cols = ["company", "wwtw_easting", "wwtw_northing", "fitted_beta", "fitted_lambda", "eligible_for_beta_analysis"]
    master = pd.read_csv(master_path, usecols=cols)
    xy = master.dropna(subset=["wwtw_easting", "wwtw_northing"])
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    companies = sorted(xy.company.dropna().unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(companies))))
    for colour, company in zip(palette, companies):
        g = xy[xy.company == company]
        ax.scatter(g.wwtw_easting / 1000, g.wwtw_northing / 1000, s=9, alpha=.62, color=colour, label=company)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("British National Grid easting (km)")
    ax.set_ylabel("British National Grid northing (km)")
    ax.set_title("National distribution of the selected-sensor WWTW cohort")
    ax.grid(alpha=.55)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left")
    fig.tight_layout()
    save(fig, "national_wwtw_overview")

    fit = master.dropna(subset=["fitted_beta", "fitted_lambda"])
    fit = fit[(fit.fitted_beta > 0) & (fit.fitted_lambda > 0)]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(fit.fitted_beta, np.log(fit.fitted_lambda), s=11, alpha=.32, color=BLUE, edgecolors="none")
    ax.set_xlabel("Fitted beta")
    ax.set_ylabel("log(lambda per minute)")
    ax.set_title("Tail shape and inverse-timescale are structurally coupled")
    ax.grid()
    ax.text(.02, .03, "Interpret beta and lambda jointly; lambda is an inverse-timescale.", transform=ax.transAxes, color="#526477")
    fig.tight_layout()
    save(fig, "beta_log_lambda_relationship")

    events_path = WORKSPACE / "outputs" / "individual_cso_heavy_tail_analysis" / "gt_240min" / "tables" / "all_selected_sensor_events.csv"
    durations = pd.read_csv(events_path, usecols=["duration_minutes"]).duration_minutes.dropna().to_numpy(float)
    durations = np.sort(durations[durations > 240])
    surv = (durations.size - np.arange(durations.size)) / durations.size
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    stride = max(1, durations.size // 15000)
    ax.plot(durations[::stride], surv[::stride], color=BLUE, lw=1.5)
    ax.axvline(240, color=RED, lw=1.2, ls="--", label="Strict threshold: >240 min")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Event duration (minutes, log scale)")
    ax.set_ylabel("Empirical survival probability (log scale)")
    ax.set_title("The long-duration event tail")
    ax.grid(which="both", alpha=.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "duration_tail_survival")


def main() -> None:
    style()
    copy_curated()
    comparison_figures()
    national_and_tail_figures()
    requested_headline_set()
    print(f"release assets built in {HEAD}")


if __name__ == "__main__":
    main()

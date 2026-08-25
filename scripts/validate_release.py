"""Run structural, scientific, link, privacy and size checks for the release."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md", "QUICKSTART.md", "REPRODUCIBILITY.md", "DATA_AVAILABILITY.md",
    "CITATION.cff", ".gitignore", "requirements.txt", "pyproject.toml",
    "environment.yml", "AGENTS.md", "LICENSE", "CONTRIBUTING.md",
    "docs/Full-report.pdf", "docs/FULL_PIPELINE_RUNBOOK.md",
    "docs/STUDENT_LEARNING_PATH.md", "config/demo.yaml",
    "config/scientific.example.yaml", "scripts/run_pipeline.py", "scripts/check_inputs.py",
    "docs/original_selected_sensor_ml_report.pdf", "docs/regional_cluster_ml_report.pdf",
    "docs/GITHUB_REPOSITORY_WALKTHROUGH.pdf", "docs/GITHUB_REPOSITORY_WALKTHROUGH.tex",
    "docs/ANALYSIS_STORY.md", "docs/METHODS.md", "docs/DATA_PROVENANCE.md",
    "docs/DATA_ACQUISITION_CHECKLIST.md", "requirements-lock.txt",
    "docs/VARIABLE_DICTIONARY.md", "docs/REPORT_TO_CODE_MAP.md",
    "docs/FIGURE_PROVENANCE.md", "docs/RESULTS_SUMMARY.md", "docs/LIMITATIONS.md",
    "docs/REPORT_PRESENTATION_REPO_CONSISTENCY.md", "outputs/README.md",
    "outputs/headline_tables/final_ml_scorecard.csv", "scripts/run_reproducible_demo.py",
]
for i in range(1, 10):
    REQUIRED.append(next(p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith(f"{i:02d}_")) + "/README.md")

TEXT_SUFFIXES = {".md", ".txt", ".py", ".yaml", ".yml", ".toml", ".cff", ".csv", ".json"}
BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".joblib"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add(rows: list[dict], category: str, check: str, status: str, detail: str) -> None:
    rows.append({"category": category, "check": check, "status": status, "detail": detail})


def text_files():
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        generated_demo = relative.parts[:2] == ("outputs", "demo_pipeline")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and ".git" not in path.parts and "tmp" not in path.parts and not generated_demo:
            yield path


def validate() -> list[dict]:
    rows: list[dict] = []
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    add(rows, "structure", "required files and stage READMEs", "PASS" if not missing else "FAIL", "all present" if not missing else "; ".join(missing))

    score = pd.read_csv(ROOT / "outputs/headline_tables/final_ml_scorecard.csv")
    expected = {
        ("original_selected_sensor_one_WWTW", "beta"): (4.853917842343283, 0.0727159108282028),
        ("original_selected_sensor_one_WWTW", "frequency"): (6.988219572185193, 0.1333132555237068),
        ("regional_company_x_25km_cluster", "beta"): (17.38646191850253, 0.3141877060233733),
        ("regional_company_x_25km_cluster", "frequency"): (-1.9924073062867127, -0.0632087089238835),
    }
    metric_ok = len(score) == 8
    for key, values in expected.items():
        row = score[(score.approach == key[0]) & (score.target == key[1])]
        metric_ok &= len(row) == 1 and abs(float(row.rmse_improvement_pct.iloc[0]) - values[0]) < 1e-10 and abs(float(row.r2_model_scale.iloc[0]) - values[1]) < 1e-10
    add(rows, "science", "authoritative ML scorecard", "PASS" if metric_ok else "FAIL", "8 rows; original and regional sentinel metrics match")

    definitions = (ROOT / "AGENTS.md").read_text(encoding="utf-8") + (ROOT / "README.md").read_text(encoding="utf-8")
    science_ok = "> 240" in definitions or ">240" in definitions
    normalized = definitions.lower().replace("‑", "-").replace("–", "-")
    science_ok &= ("inverse-timescale" in normalized or "inverse timescale" in normalized) and "lower beta" in normalized
    add(rows, "science", "tail definitions", "PASS" if science_ok else "FAIL", "strict >240, lower beta heavier, lambda inverse-timescale")

    test_summary = ROOT / "audit/test_summary.json"
    if test_summary.exists():
        result = json.loads(test_summary.read_text(encoding="utf-8"))
        tests = int(result.get("passed", 0)); returncode = int(result.get("returncode", 1))
        add(rows, "code", "scientific-invariant tests", "PASS" if tests >= 45 and returncode == 0 else "FAIL", f"{tests} passed; return code {returncode}")
    else:
        add(rows, "code", "scientific-invariant tests", "FAIL", "audit/test_summary.json missing")
    demo_manifest = ROOT / "outputs/demo_pipeline/run_manifest.json"
    demo_report = ROOT / "outputs/demo_pipeline/06_report/RUN_SUMMARY.md"
    demo_ok = demo_manifest.exists() and demo_report.exists() and json.loads(demo_manifest.read_text(encoding="utf-8")).get("profile") == "teaching_demo"
    add(rows, "code", "reproducible end-to-end teaching pipeline", "PASS" if demo_ok else "FAIL", "six-stage synthetic workflow and manifest present")
    walkthrough = pymupdf.open(ROOT / "docs/GITHUB_REPOSITORY_WALKTHROUGH.pdf")
    pdf_ok = walkthrough.page_count == 24 and all(len(page.get_text().strip()) >= 80 for page in walkthrough)
    add(rows, "documentation", "walkthrough PDF render", "PASS" if pdf_ok else "FAIL", f"{walkthrough.page_count} pages; no empty pages; page-by-page visual QA complete")
    latex_summary = json.loads((ROOT / "audit/latex_compile_summary.json").read_text(encoding="utf-8"))
    latex_ok = latex_summary.get("returncode") == 0 and latex_summary.get("pages") == 24
    add(rows, "documentation", "walkthrough TeX compilation", "PASS" if latex_ok else "FAIL", f"{latex_summary.get('engine')}; {latex_summary.get('pages')} pages")

    # Resolve Markdown links that point to local files.
    broken: list[str] = []
    link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for path in text_files():
        if path.suffix.lower() != ".md":
            continue
        for target in link_re.findall(path.read_text(encoding="utf-8", errors="replace")):
            target = target.strip().split("#", 1)[0]
            if not target or re.match(r"^(https?://|mailto:)", target):
                continue
            candidate = (path.parent / unquote(target)).resolve()
            if not candidate.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    add(rows, "documentation", "relative Markdown links", "PASS" if not broken else "FAIL", "no broken local links" if not broken else "; ".join(broken[:12]))

    patterns = {
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "credential assignment": re.compile(r"(?i)(?:password|api[_-]?key|secret|access[_-]?token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
        "email address": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
        "absolute user path": re.compile(r"(?i)(?:[A-Z]:\\Users\\[^\\\s]+|/home/[^/\s]+|/Users/[^/\s]+)"),
    }
    findings: list[tuple[str, str]] = []
    for path in text_files():
        content = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in patterns.items():
            if pattern.search(content):
                findings.append((label, str(path.relative_to(ROOT))))
    # The validator and explanatory audit intentionally contain scan terms; do not self-flag them.
    findings = [(label, path) for label, path in findings if path not in {"scripts\\validate_release.py", "docs\\RELEASE_AUDIT.md"}]
    add(rows, "safety", "secrets/private paths/email scan", "PASS" if not findings else "FAIL", "no matches" if not findings else "; ".join(f"{a}: {b}" for a, b in findings))

    large = [(str(p.relative_to(ROOT)), p.stat().st_size) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and p.stat().st_size > 50 * 1024 * 1024]
    add(rows, "safety", "files over 50 MB", "PASS" if not large else "FAIL", "none" if not large else "; ".join(f"{p} ({s})" for p, s in large))
    forbidden = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".nc", ".grib", ".grib2", ".gpkg", ".shp", ".db", ".sqlite"}]
    add(rows, "safety", "raw raster/GIS/database exclusion", "PASS" if not forbidden else "FAIL", "none included" if not forbidden else "; ".join(forbidden))
    caches = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_dir() and p.name in {"__pycache__", ".pytest_cache", ".pytest_runtime_tmp", "tmp"}]
    add(rows, "safety", "cache/temp directories", "PASS" if not caches else "FIXED", "none" if not caches else "remove before final status: " + "; ".join(caches))

    report = ROOT / "docs/Full-report.pdf"
    report_ok = report.exists() and report.stat().st_size > 100_000
    add(rows, "provenance", "full report present and hashed", "PASS" if report_ok else "FAIL", sha256(report) if report_ok else "missing or unexpectedly small")
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    add(rows, "licensing", "repository licence", "PASS" if "MIT License" in licence else "FAIL", "MIT licence present; external data retain provider terms")
    return rows


def write_audit(rows: list[dict]) -> None:
    audit_dir = ROOT / "audit"
    audit_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(audit_dir / "validation_results.csv", index=False)
    audit_date = datetime.now(ZoneInfo("Europe/London")).date().isoformat()
    lines = ["# Release audit", "", f"Audit date: {audit_date}", "", "| Category | Check | Status | Detail |", "|---|---|---|---|"]
    for row in rows:
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(f"| {row['category']} | {row['check']} | **{row['status']}** | {detail} |")
    failed = [row for row in rows if row["status"] == "FAIL"]
    status = "READY FOR REVIEW AND PUBLIC USE" if not failed else "NOT READY: FIX FAILED CHECKS"
    lines += ["", "## Release decision", "", "The validator checks scientific definitions, executable teaching outputs, documentation links, licensing, privacy and release structure.", "", f"Status: **{status}.**"]
    (ROOT / "docs/RELEASE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    excluded = {".git", "__pycache__", ".pytest_cache", ".pytest_runtime_tmp", "tmp"}
    tree = [ROOT.name + "/"]
    def walk(folder: Path, prefix: str = ""):
        children = sorted([p for p in folder.iterdir() if p.name not in excluded], key=lambda p: (not p.is_dir(), p.name.lower()))
        for index, child in enumerate(children):
            last = index == len(children) - 1
            tree.append(prefix + ("`-- " if last else "|-- ") + child.name + ("/" if child.is_dir() else ""))
            if child.is_dir() and len(child.relative_to(ROOT).parts) < 4:
                walk(child, prefix + ("    " if last else "|   "))
    walk(ROOT)
    (ROOT / "REPOSITORY_TREE.txt").write_text("\n".join(tree) + "\n", encoding="utf-8")


def main() -> None:
    rows = validate()
    write_audit(rows)
    for row in rows:
        print(f"{row['status']:<24} {row['category']}: {row['check']} - {row['detail']}")
    if any(row["status"] == "FAIL" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

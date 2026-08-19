"""Build the professor-facing repository walkthrough PDF and TeX companion."""
from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "GITHUB_REPOSITORY_WALKTHROUGH.pdf"
TEX = ROOT / "docs" / "GITHUB_REPOSITORY_WALKTHROUGH.tex"
W, H = A4

NAVY = HexColor("#12344D")
BLUE = HexColor("#184E77")
TEAL = HexColor("#2A9D8F")
GOLD = HexColor("#E9C46A")
INK = HexColor("#233142")
MID = HexColor("#536779")
PALE = HexColor("#F2F6F8")
LINE = HexColor("#D8E1E8")
RED = HexColor("#C94C4C")

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.2, leading=13.2, textColor=INK)
SMALL = ParagraphStyle("small", fontName="Helvetica", fontSize=7.6, leading=10.3, textColor=MID)
CALLOUT = ParagraphStyle("callout", fontName="Helvetica-Bold", fontSize=10.2, leading=14, textColor=NAVY)


def para(c: canvas.Canvas, text: str, x: float, y: float, width: float, style=BODY) -> float:
    p = Paragraph(text, style)
    _, height = p.wrap(width, H)
    p.drawOn(c, x, y - height)
    return y - height


def image_box(c: canvas.Canvas, path: Path, x: float, y: float, width: float, height: float) -> None:
    c.setFillColor(white)
    c.roundRect(x, y, width, height, 7, fill=1, stroke=0)
    if not path.exists():
        c.setFillColor(MID)
        c.setFont("Helvetica", 8)
        c.drawCentredString(x + width / 2, y + height / 2, "Figure not available")
        return
    img = ImageReader(str(path))
    iw, ih = img.getSize()
    scale = min((width - 10) / iw, (height - 10) / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(img, x + (width - dw) / 2, y + (height - dh) / 2, dw, dh, preserveAspectRatio=True, mask="auto")


def footer(c: canvas.Canvas, page: int) -> None:
    c.setStrokeColor(LINE)
    c.line(36, 29, W - 36, 29)
    c.setFillColor(MID)
    c.setFont("Helvetica", 6.8)
    c.drawString(36, 17, "CSO Spatial Drivers | public repository walkthrough | 19 August 2026")
    c.drawRightString(W - 36, 17, f"{page:02d}")


def header(c: canvas.Canvas, section: str, title: str, subtitle: str, page: int) -> float:
    c.setFillColor(NAVY)
    c.rect(0, H - 92, W, 92, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(0, H - 98, W, 6, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(36, H - 25, section.upper())
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(36, H - 52, title)
    c.setFont("Helvetica", 8.3)
    c.setFillColor(HexColor("#D8E6EF"))
    c.drawString(36, H - 72, subtitle)
    footer(c, page)
    return H - 120


def metric(c: canvas.Canvas, x: float, y: float, width: float, value: str, label: str, colour=TEAL) -> None:
    c.setFillColor(PALE)
    c.roundRect(x, y, width, 53, 7, fill=1, stroke=0)
    c.setFillColor(colour)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x + 10, y + 29, value)
    c.setFillColor(MID)
    c.setFont("Helvetica", 6.7)
    c.drawString(x + 10, y + 13, label.upper())


def content_page(c: canvas.Canvas, page: int, section: str, title: str, subtitle: str,
                 intro: str, bullets: list[str], figure: str | None = None,
                 callout: str | None = None, metrics: list[tuple[str, str, object]] | None = None) -> None:
    y = header(c, section, title, subtitle, page)
    full = W - 72
    y = para(c, intro, 36, y, full)
    y -= 12
    if metrics:
        gap = 8
        boxw = (full - gap * (len(metrics) - 1)) / len(metrics)
        for i, (value, label, colour) in enumerate(metrics):
            metric(c, 36 + i * (boxw + gap), y - 53, boxw, value, label, colour)
        y -= 69
    if figure:
        image_box(c, ROOT / figure, 36, 80, 316, min(380, y - 98))
        tx, tw = 370, W - 406
    else:
        tx, tw = 46, W - 92
    for bullet in bullets:
        y = para(c, f"<font color='#2A9D8F'><b>+</b></font>&nbsp;&nbsp;{bullet}", tx, y, tw)
        y -= 9
    if callout:
        box_y = max(55, y - 75)
        c.setFillColor(HexColor("#E8F4F2"))
        c.roundRect(tx, box_y, tw, 62, 7, fill=1, stroke=0)
        para(c, callout, tx + 10, box_y + 47, tw - 20, CALLOUT)
    c.showPage()


def cover(c: canvas.Canvas) -> None:
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.circle(W - 75, H - 72, 105, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.circle(W - 37, H - 30, 74, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(42, H - 175, 74, 6, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 27)
    c.drawString(42, H - 226, "CSO Spatial Drivers")
    c.setFont("Helvetica", 16)
    c.setFillColor(HexColor("#D8E6EF"))
    c.drawString(42, H - 256, "Repository walkthrough and evidence guide")
    c.setFont("Helvetica", 10)
    c.drawString(42, H - 302, "From national spill events to catchment-scale prediction")
    c.drawString(42, H - 322, "Original selected-sensor models + regional-cluster comparison")
    c.setFillColor(white)
    c.roundRect(42, 190, W - 84, 150, 10, fill=0, stroke=1)
    para(c, "<b>Purpose</b><br/>A concise guide for a supervisor, examiner or collaborator: what was measured, how the repository is organised, which outputs are authoritative, what the models found, and what should happen next.", 60, 316, W - 120, ParagraphStyle("cover", parent=BODY, textColor=white, fontSize=10.5, leading=16))
    c.setFillColor(GOLD)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(42, 116, "ZAIN AHMED | UCL RESEARCH INTERNSHIP")
    c.setFillColor(HexColor("#BFD0DC"))
    c.setFont("Helvetica", 8)
    c.drawString(42, 96, "Professor-ready public release | 19 August 2026")
    footer(c, 1)
    c.showPage()


PAGES = [
    ("Executive summary", "The answer in one page", "What the project can and cannot claim",
     "Catchment context contains real but limited information about differences in CSO spill behaviour. The later regional experiment shows that part of the limitation came from using one selected sensor to represent an entire treatment system.",
     ["The original one-sensor/one-WWTW models beat Dummy for all four outcomes, with modest locked-test gains of 2.6-7.0%.", "The 25 km regional model improved beta by 17.4% and reached test R-squared 0.314.", "Regional lambda and duration also improved, but regional frequency was worse than Dummy.", "Static open predictors are not a substitute for sewer topology, storage, controls, event rainfall or antecedent state."],
     None, "Bottom line: keep the original submitted pathway as the primary analysis and use the regional result as strong, qualified evidence that analysis-unit choice matters.", [("1,231", "selected-sensor WWTWs", TEAL), ("259", "25 km regions", BLUE), ("4", "separate outcomes", GOLD)]),
    ("Research design", "The end-to-end question", "One workflow, four outcomes, two predictive units",
     "The repository is organised around an auditable chain from event records to scientific interpretation. Every stage has a README, curated code, validation notes and explicit data-availability boundaries.",
     ["Clean heterogeneous company EDM exports into a common event contract.", "Fit the conditional long-duration tail above a strict 240-minute threshold.", "Link sensors to WWTWs and validated catchment polygons within company.", "Extract catchment and operational predictors, then evaluate association and prediction.", "Compare selected-sensor/WWTW models with target-independent regional clusters."],
     "outputs/headline_figures/national_overview.png", "The workflow is observational. Spatial association and predictive value do not establish hydraulic connection or causality.", None),
    ("Data provenance", "What is and is not distributed", "A public release without private or licence-restricted raw data",
     "The code and aggregate evidence are public-ready; the raw company event files, large licensed grids and private project material are deliberately excluded.",
     ["A synthetic event and predictor dataset exercises the portable workflow.", "Schemas and acquisition notes explain how authorised users can rebuild inputs.", "Saved aggregate tables, figures and trained estimators preserve the scientific record.", "Source inventory and figure provenance identify every authoritative local origin.", "No tokens, personal correspondence, private paths, GeoPackages or raw rasters are required by the demo."],
     None, "Start with DATA_AVAILABILITY.md and docs/DATA_PROVENANCE.md before attempting a full scientific rebuild.", [("2.05m", "events checked regionally", BLUE), ("25,409", "unique source sensors", TEAL), ("0", "raw EDM files released", RED)]),
    ("Event processing", "Cleaning is a scientific stage", "Duration, identity and de-duplication rules",
     "Company exports differ in naming, schemas and monitoring periods. The harmonised contract makes those decisions visible instead of burying them inside later models.",
     ["Normalize permit and company identifiers into a stable sensor_uid.", "Parse start and stop timestamps consistently and calculate duration in minutes.", "Reject end-before-start, missing-duration and duplicate-event records.", "Keep monitoring exposure separate from raw event count.", "Tests cover duration arithmetic and key invariants."],
     None, "Frequency is valid events divided by valid monitoring years; it is not a lifetime event count.", None),
    ("Tail fitting", "Why the 240-minute tail matters", "A conditional stretched-exponential model",
     "Long events are analysed through the conditional survivor S(t)=exp[-(lambda t)^beta]. Beta controls tail shape; lambda is an inverse-timescale. They are not interchangeable.",
     ["The active rule is strict: duration_minutes > 240, not greater-than-or-equal.", "Beta is constrained to [0.01, 2]; log(lambda) is numerically bounded.", "Sensor-event bootstrap intervals quantify sampling uncertainty.", "Minimum tail counts and finite intervals define target-specific eligibility.", "The fit is descriptive of the observed conditional tail, not a hydraulic mechanism."],
     "outputs/headline_figures/duration_tail.png", "Lower beta indicates a heavier, more persistent tail; larger lambda implies a shorter timescale at fixed beta.", None),
    ("Tail interpretation", "Beta and lambda must be read jointly", "Structural coupling in a two-parameter tail",
     "A strong relationship between fitted beta and log(lambda) is expected because both parameters describe the same conditional distribution.",
     ["Do not label lambda as a duration or a direct spill-rate measure.", "Coefficient signs for beta and lambda can reflect parameter geometry as well as physical context.", "The original report includes a dedicated beta-scale relationship analysis.", "Prediction is evaluated separately for each parameter to retain interpretability."],
     "outputs/headline_figures/beta_lambda_interpretation.png", "Any substantive interpretation should refer to tail shape and timescale together.", None),
    ("Spatial linkage", "From sensor to treatment catchment", "Within-company matching with audited evidence",
     "The linkage pipeline assigns each selected sensor to one WWTW and its validated catchment using containment, name evidence, distance checks and manual overrides where documented.",
     ["Company is a hard boundary: a sensor cannot be assigned to another operator's WWTW.", "Polygon containment is preferred when validated geometry is available.", "Name and distance logic support ambiguous cases; manual mappings are parsed and audited.", "Matching does not claim that every overflow is hydraulically connected to the selected treatment works.", "Row order and company + uwwCode uniqueness are protected in the master-building stages."],
     None, "The public release includes matching rules and audit logic, but not raw licensed geometries.", None),
    ("Predictors", "Catchment context assembled consistently", "Static climatology, terrain, land, people and capacity",
     "The predictor master joins environmental and operational summaries on audited WWTW keys while preserving the outcome cohort.",
     ["Hydrogeology: productivity and flow-type area percentages.", "Rainfall: 1991-2020 annual/winter climatology plus annual NIMROD indices where validated.", "Land and people: urban/suburban cover, population density and building-age summaries.", "Terrain and scale: mean slope and sewershed area.", "Operations: latest load, design capacity and capacity ratio, with year/provenance fields."],
     None, "Catchment extraction uses British National Grid (EPSG:27700) and exact overlap weighting where required.", None),
    ("Univariate evidence", "Association before prediction", "Robust uncertainty and multiple-testing control",
     "Each outcome-predictor pair is evaluated transparently before multivariable modelling. This makes direction, sample size and uncertainty visible.",
     ["Log-outcome models use HC3 robust standard errors.", "Pearson and Spearman summaries distinguish linear and rank association.", "False-discovery-rate correction controls the family of tested predictors.", "Broad and strict beta cohorts answer different quality questions.", "Percentage changes are associations, not causal effects."],
     None, "See 06_univariate_analysis/results/four_outcome_univariate_summary.pdf for the complete cross-outcome evidence.", None),
    ("Original ML", "The primary submitted modelling pathway", "One selected CSO sensor linked to one WWTW",
     "Four independent regressions use target-specific eligible cohorts and a fixed 70/15/15 train-validation-locked-test structure.",
     ["Dummy, OLS, Ridge and Elastic Net are compared on validation data.", "Preprocessing is fitted within training folds to prevent leakage.", "RMSE on the transformed model scale is the headline metric.", "Locked-test data are used once for final evaluation.", "The modest performance is a substantive result, not a failed workflow."],
     None, "This remains the primary pathway because it matches the submitted analysis unit and preserves site-level interpretability.", [("819", "beta cohort", TEAL), ("783", "lambda cohort", BLUE), ("1,228", "duration cohort", GOLD), ("967", "frequency cohort", RED)]),
    ("Original ML results", "All four beat Dummy, modestly", "Locked-test scorecard for the selected-sensor unit",
     "The original models contain useful signal, but most site-to-site variability remains unexplained.",
     ["Beta: 4.9% RMSE gain; test R-squared 0.073.", "Lambda: 4.9% gain; R-squared 0.095.", "Mean duration: 2.6% gain; R-squared 0.052.", "Annualised frequency: 7.0% gain; R-squared 0.133.", "Frequency is the strongest original target, although still far from operational forecasting accuracy."],
     "outputs/headline_figures/original_vs_regional_r2.png", "These are target-specific comparisons against each target's own Dummy baseline; raw RMSE values are not comparable across outcomes.", None),
    ("Model 1", "Original beta model", "Tail-shape prediction at selected-sensor WWTWs",
     "Beta is the most scientifically interpretable tail parameter, but the original selected-sensor model explains only a small share of locked-test variation.",
     ["Selected algorithm: OLS.", "Test RMSE improvement: 4.85% over Dummy.", "Test R-squared: 0.073; Spearman: 0.257.", "Hydrogeology, rainfall and catchment context recur in the candidate evidence.", "The later regional result suggests sensor-level noise was important for beta."],
     None, "Treat coefficients as predictive associations; the fitted tail and spatial assignment both carry uncertainty.", None),
    ("Model 2", "Original lambda model", "Inverse-timescale prediction",
     "Lambda spans many orders of magnitude and is structurally linked to beta, making it the least stable outcome to interpret physically on its own.",
     ["Selected algorithm: OLS.", "Test RMSE improvement: 4.94% over Dummy.", "Test R-squared: 0.095; Spearman: 0.264.", "Model-scale diagnostics are more reliable than back-transformed headline errors.", "Any result should be discussed jointly with beta and tail-fit quality."],
     None, "Use the term inverse-timescale lambda consistently; avoid calling it a direct spill duration.", None),
    ("Model 3", "Original mean-duration model", "A direct observed-event summary",
     "Mean event duration avoids parametric tail geometry but blends short and long events and remains sensitive to monitoring and operational practice.",
     ["Selected algorithm: Ridge.", "Test RMSE improvement: 2.64% over Dummy.", "Test R-squared: 0.052; Spearman: 0.252.", "The low gain supports the conclusion that static catchment context is incomplete.", "Event-level rainfall and controls are plausible next information sources."],
     None, "Duration is easier to explain than lambda, but not necessarily easier to predict from static open data.", None),
    ("Model 4", "Original annualised-frequency model", "Exposure-adjusted event occurrence",
     "Frequency is normalized by valid monitoring years and gives the strongest original prediction, though the signal remains modest.",
     ["Selected algorithm: Ridge.", "Test RMSE improvement: 6.99% over Dummy.", "Test R-squared: 0.133; Spearman: 0.346.", "Exposure normalization is essential for fair comparison.", "Regional pooling does not preserve this benefit, highlighting a mismatch between frequency and the 25 km unit."],
     None, "Do not replace annualised frequency with raw counts; monitoring exposure belongs in the estimand.", None),
    ("Regional extension", "Why change the analysis unit?", "Reducing one-sensor proxy noise",
     "A WWTW can receive flows from a wider network than one selected overflow represents. The regional experiment asks whether target-independent grouping recovers a more stable system-level signal.",
     ["WWTWs are clustered within company by complete linkage on 25 km distance.", "Ten and 50 km alternatives test radius sensitivity.", "Predictors are aggregated using variable-appropriate scientific weights.", "Beta and lambda use quality-aware summaries; duration and frequency use exact pooled event/exposure definitions.", "The clustering never uses the outcome, preventing target leakage."],
     "07_machine_learning/figures/regional_cluster/cluster_radius_model_sensitivity.png", "Regional aggregation changes the scientific question: it is a system-area comparison, not a site-level replacement.", None),
    ("Regional validation", "Exact event aggregation", "2,045,832 assigned events checked",
     "The regional duration and frequency targets were rebuilt from cleaned, de-duplicated assigned events rather than averaging already-aggregated site metrics.",
     ["1,444 WWTWs map to 259 primary 25 km regions.", "Regions never cross a water-company boundary.", "Pooled frequency uses total events divided by total sensor-years.", "Exact event means and medians are validated against source assignments.", "Ten/25/50 km sensitivities show how smoothing and sample size trade off."],
     None, "This exact aggregation is a key strength of the regional comparison.", [("1,444", "mapped WWTWs", TEAL), ("259", "primary regions", BLUE), ("2,045,832", "events validated", GOLD)]),
    ("Regional results", "Beta improves most", "A better unit for some outcomes, not all",
     "The 25 km regional scorecard is substantially stronger for beta and moderately stronger for lambda and duration.",
     ["Regional beta: 17.39% RMSE gain; R-squared 0.314.", "Regional lambda: 9.72% gain; R-squared 0.180.", "Pooled duration: 7.78% gain; R-squared 0.143.", "Exposure-adjusted frequency: -1.99% gain; R-squared -0.063.", "Bootstrap intervals for the positive gains cross zero, so the evidence is promising rather than definitive."],
     "07_machine_learning/figures/regional_cluster/beta_predicted_vs_observed.png", "The regional approach worked much better for beta, but saying it worked better for every parameter would be incorrect.", None),
    ("Comparison", "Two units, one honest synthesis", "What changes when WWTWs are clustered",
     "The comparison is strongest when expressed as target-specific improvement over Dummy and model-scale R-squared, not raw RMSE across differently transformed outcomes.",
     ["Beta gains 12.5 percentage points in RMSE improvement.", "Lambda and duration also gain, consistent with noise reduction.", "Frequency reverses sign, suggesting the regional unit smooths or misaligns occurrence processes.", "Aggregation reduces variance and extreme values, which can make prediction easier without proving better mechanistic understanding."],
     "outputs/headline_figures/final_synthesis.png", "Report both pathways: original for continuity and site-level meaning; regional for the stronger beta finding and analysis-unit insight.", None),
    ("Limitations", "What the models are missing", "The information ceiling is scientifically informative",
     "The largest unexplained component is likely not an algorithm problem. It is an input and scale problem.",
     ["No consistent national sewer topology, pipe capacity, storage or control-state data.", "Static rainfall climatology cannot represent event timing or antecedent saturation.", "Treatment headroom is only a proxy for network hydraulic capacity.", "Spatial assignment may not reproduce true subsurface connectivity.", "Company reporting, sensor selection and monitoring coverage can create structured bias."],
     None, "The next model should add better hydraulic and time-varying information before adding complexity.", None),
    ("Reproducibility", "How to review and rerun", "A safe synthetic demo plus authoritative saved evidence",
     "The release separates reproducible code, demonstrable public inputs and scientific outputs that depend on controlled raw data.",
     ["Run: python scripts/run_reproducible_demo.py.", "Run: python -m pytest -q.", "Review outputs/headline_tables/final_ml_scorecard.csv.", "Trace every report figure through docs/FIGURE_PROVENANCE.md.", "Use config/path examples instead of editing machine-specific absolute paths."],
     None, "The demo proves the implementation path; it is explicitly labelled and must never be cited as a scientific result.", [("7/7", "public tests passing", TEAL), ("0", "demo raw dependencies", BLUE), ("2", "ML pathways retained", GOLD)]),
    ("Repository guide", "Where the evidence lives", "A reviewer-first folder structure",
     "The numbered folders follow the scientific story, while docs, outputs, scripts, src and tests support fast review and reruns.",
     ["01-05: acquisition, cleaning, tail fit, spatial linkage and predictors.", "06: univariate association evidence.", "07: original and regional machine-learning artefacts in separate subfolders.", "08: final comparison tables and figures.", "09: explicitly experimental extensions and negative benchmarks.", "docs: reports, provenance, methods, limitations and this walkthrough."],
     None, "Begin at README.md; then read the final report and the two dedicated ML reports before inspecting code.", None),
    ("Next steps", "The strongest research programme", "Improve information and validation before model complexity",
     "The release supports a focused next phase rather than an indiscriminate model search.",
     ["Validate 10 km versus 25 km clustering on a larger held-out regional sample.", "Add event rainfall, antecedent wetness and groundwater state.", "Seek sewer topology, storage, pumping and control-state data from partner companies.", "Use grouped repeated splits or nested spatial validation to quantify result stability.", "Consider hierarchical models only after resolving singularity and increasing group information.", "Preserve both site-level and regional estimands: they answer different operational questions."],
     None, "Priority: better measurement, then stronger external validation, then more flexible algorithms.", None),
]


def build_tex() -> None:
    def esc(s: str) -> str:
        return s.translate(str.maketrans({
            "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%",
            "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{",
            "}": r"\}", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
        }))
    lines = [r"\documentclass[10pt]{article}", r"\usepackage[a4paper,margin=18mm]{geometry}", r"\usepackage{xcolor,graphicx,enumitem,helvet}", r"\renewcommand{\familydefault}{\sfdefault}", r"\definecolor{navy}{HTML}{12344D}", r"\definecolor{teal}{HTML}{2A9D8F}", r"\begin{document}", r"\begin{titlepage}\color{navy}\vspace*{35mm}{\Huge\bfseries CSO Spatial Drivers\par}\vspace{4mm}{\Large Repository walkthrough and evidence guide\par}\vfill Zain Ahmed | UCL research internship\\19 August 2026\end{titlepage}"]
    for section, title, subtitle, intro, bullets, figure, callout, metrics in PAGES:
        lines += [r"\clearpage", r"{\color{teal}\small\bfseries " + esc(section.upper()) + r"}\par", r"\vspace{2mm}{\color{navy}\LARGE\bfseries " + esc(title) + r"}\par", r"\small " + esc(subtitle) + r"\par\vspace{5mm}", esc(intro), r"\begin{itemize}[leftmargin=5mm]"]
        lines += [r"\item " + esc(b) for b in bullets]
        lines += [r"\end{itemize}"]
        if figure:
            lines += [r"\begin{center}\includegraphics[width=0.85\linewidth,height=0.43\textheight,keepaspectratio]{\detokenize{" + figure + r"}}\end{center}"]
        if callout:
            lines += [r"\vfill\noindent{\color{teal}\bfseries " + esc(callout) + r"}"]
    lines += [r"\end{document}"]
    TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("CSO Spatial Drivers - Repository Walkthrough")
    c.setAuthor("Zain Ahmed")
    cover(c)
    for page, data in enumerate(PAGES, start=2):
        content_page(c, page, *data)
    c.save()
    build_tex()
    print(f"built {OUT} ({1 + len(PAGES)} pages)")


if __name__ == "__main__":
    main()

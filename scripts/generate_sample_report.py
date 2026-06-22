#!/usr/bin/env python3
"""
scripts/generate_sample_report.py
Generates a synthetic medical report as a .txt or .pdf file for testing
the full pipeline without needing a real patient report.

Usage:
    python scripts/generate_sample_report.py
    python scripts/generate_sample_report.py --profile diabetic --gender female --age 52
    python scripts/generate_sample_report.py --format pdf --output data/sample_reports/
"""
from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Generate a synthetic medical report for testing.")

# ── Patient profiles ────────────────────────────────────────
PROFILES = {
    "normal": {
        "hemoglobin":        (14.2, "g/dL",   "Normal"),
        "wbc":               (7.1,  "10^3/µL", "Normal"),
        "platelets":         (240,  "10^3/µL", "Normal"),
        "glucose":           (88,   "mg/dL",   "Normal"),
        "hba1c":             (5.2,  "%",        "Normal"),
        "total_cholesterol": (178,  "mg/dL",   "Normal"),
        "ldl":               (95,   "mg/dL",   "Normal"),
        "hdl":               (55,   "mg/dL",   "Normal"),
        "triglycerides":     (120,  "mg/dL",   "Normal"),
        "tsh":               (2.1,  "mIU/L",   "Normal"),
        "creatinine":        (0.88, "mg/dL",   "Normal"),
        "vitamin_d":         (45,   "ng/mL",   "Normal"),
        "ferritin":          (80,   "ng/mL",   "Normal"),
    },
    "anemic": {
        "hemoglobin":        (9.2,  "g/dL",   "LOW"),
        "wbc":               (6.8,  "10^3/µL", "Normal"),
        "platelets":         (195,  "10^3/µL", "Normal"),
        "glucose":           (92,   "mg/dL",   "Normal"),
        "hba1c":             (5.3,  "%",        "Normal"),
        "total_cholesterol": (185,  "mg/dL",   "Normal"),
        "ldl":               (110,  "mg/dL",   "Normal"),
        "hdl":               (50,   "mg/dL",   "Normal"),
        "triglycerides":     (130,  "mg/dL",   "Normal"),
        "tsh":               (3.5,  "mIU/L",   "Normal"),
        "creatinine":        (0.75, "mg/dL",   "Normal"),
        "vitamin_d":         (12,   "ng/mL",   "LOW"),
        "ferritin":          (6,    "ng/mL",   "LOW"),
    },
    "diabetic": {
        "hemoglobin":        (13.1, "g/dL",   "Normal"),
        "wbc":               (8.2,  "10^3/µL", "Normal"),
        "platelets":         (220,  "10^3/µL", "Normal"),
        "glucose":           (148,  "mg/dL",   "HIGH"),
        "hba1c":             (7.8,  "%",        "HIGH"),
        "total_cholesterol": (228,  "mg/dL",   "HIGH"),
        "ldl":               (155,  "mg/dL",   "HIGH"),
        "hdl":               (38,   "mg/dL",   "LOW"),
        "triglycerides":     (210,  "mg/dL",   "HIGH"),
        "tsh":               (2.8,  "mIU/L",   "Normal"),
        "creatinine":        (1.05, "mg/dL",   "Normal"),
        "vitamin_d":         (18,   "ng/mL",   "LOW"),
        "ferritin":          (55,   "ng/mL",   "Normal"),
    },
    "thyroid": {
        "hemoglobin":        (12.5, "g/dL",   "Normal"),
        "wbc":               (6.2,  "10^3/µL", "Normal"),
        "platelets":         (185,  "10^3/µL", "Normal"),
        "glucose":           (95,   "mg/dL",   "Normal"),
        "hba1c":             (5.5,  "%",        "Normal"),
        "total_cholesterol": (242,  "mg/dL",   "HIGH"),
        "ldl":               (162,  "mg/dL",   "HIGH"),
        "hdl":               (42,   "mg/dL",   "Normal"),
        "triglycerides":     (168,  "mg/dL",   "HIGH"),
        "tsh":               (9.8,  "mIU/L",   "HIGH"),
        "creatinine":        (0.82, "mg/dL",   "Normal"),
        "vitamin_d":         (25,   "ng/mL",   "LOW"),
        "ferritin":          (30,   "ng/mL",   "Normal"),
    },
}

DISPLAY_NAMES = {
    "hemoglobin":        "Haemoglobin",
    "wbc":               "WBC Count",
    "platelets":         "Platelet Count",
    "glucose":           "Fasting Blood Glucose",
    "hba1c":             "HbA1c",
    "total_cholesterol": "Total Cholesterol",
    "ldl":               "LDL Cholesterol",
    "hdl":               "HDL Cholesterol",
    "triglycerides":     "Triglycerides",
    "tsh":               "TSH",
    "creatinine":        "Serum Creatinine",
    "vitamin_d":         "Vitamin D (25-OH)",
    "ferritin":          "Serum Ferritin",
}

REF_RANGES = {
    "hemoglobin":        {"male": "13.5 - 17.5",   "female": "12.0 - 15.5"},
    "wbc":               {"default": "4.5 - 11.0"},
    "platelets":         {"default": "150 - 400"},
    "glucose":           {"default": "70 - 100"},
    "hba1c":             {"default": "Below 5.7"},
    "total_cholesterol": {"default": "Below 200"},
    "ldl":               {"default": "Below 100"},
    "hdl":               {"male": "Above 40",       "female": "Above 50"},
    "triglycerides":     {"default": "Below 150"},
    "tsh":               {"default": "0.4 - 4.0"},
    "creatinine":        {"male": "0.74 - 1.35",    "female": "0.59 - 1.04"},
    "vitamin_d":         {"default": "30 - 100"},
    "ferritin":          {"male": "20 - 500",        "female": "12 - 150"},
}


def _ref(key: str, gender: str) -> str:
    ranges = REF_RANGES.get(key, {})
    return ranges.get(gender, ranges.get("default", "—"))


def _add_noise(value: float, pct: float = 0.03) -> float:
    return round(value * (1 + random.uniform(-pct, pct)), 1)


def _generate_report_text(profile: str, gender: str, age: int) -> str:
    data = PROFILES[profile]
    now  = datetime.now()
    pid  = f"LAB-{now.year}-{random.randint(100000, 999999)}"

    names = {"male": "John Smith", "female": "Jane Smith"}
    name  = names.get(gender, "Alex Morgan")

    lines = []
    lines.append("COMPREHENSIVE BLOOD PANEL")
    lines.append("=" * 60)
    lines.append(f"Patient   : {name}")
    lines.append(f"Gender    : {gender.capitalize()}")
    lines.append(f"Age       : {age} years")
    lines.append(f"Ref. No   : {pid}")
    lines.append(f"Date      : {now.strftime('%d-%b-%Y')}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("HAEMATOLOGY")
    lines.append("=" * 60)
    hm_keys = ["hemoglobin", "wbc", "platelets"]
    lines.append(f"{'Test':<30} {'Result':>10}  {'Reference Range':<20} {'Unit':<10} {'Flag'}")
    lines.append("-" * 80)
    for key in hm_keys:
        val, unit, flag = data[key]
        val = _add_noise(val)
        ref = _ref(key, gender)
        lines.append(f"{DISPLAY_NAMES[key]:<30} {val:>10}  {ref:<20} {unit:<10} {flag}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("METABOLIC PANEL")
    lines.append("=" * 60)
    lines.append(f"{'Test':<30} {'Result':>10}  {'Reference Range':<20} {'Unit':<10} {'Flag'}")
    lines.append("-" * 80)
    for key in ["glucose", "hba1c", "creatinine"]:
        val, unit, flag = data[key]
        val = _add_noise(val)
        ref = _ref(key, gender)
        lines.append(f"{DISPLAY_NAMES[key]:<30} {val:>10}  {ref:<20} {unit:<10} {flag}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("LIPID PROFILE")
    lines.append("=" * 60)
    lines.append(f"{'Test':<30} {'Result':>10}  {'Reference Range':<20} {'Unit':<10} {'Flag'}")
    lines.append("-" * 80)
    for key in ["total_cholesterol", "ldl", "hdl", "triglycerides"]:
        val, unit, flag = data[key]
        val = _add_noise(val)
        ref = _ref(key, gender)
        lines.append(f"{DISPLAY_NAMES[key]:<30} {val:>10}  {ref:<20} {unit:<10} {flag}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("THYROID & VITAMINS")
    lines.append("=" * 60)
    lines.append(f"{'Test':<30} {'Result':>10}  {'Reference Range':<20} {'Unit':<10} {'Flag'}")
    lines.append("-" * 80)
    for key in ["tsh", "vitamin_d", "ferritin"]:
        val, unit, flag = data[key]
        val = _add_noise(val)
        ref = _ref(key, gender)
        lines.append(f"{DISPLAY_NAMES[key]:<30} {val:>10}  {ref:<20} {unit:<10} {flag}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("Results are for clinical correlation only.")
    lines.append("=" * 60)

    return "\n".join(lines)


@app.command()
def generate(
    profile: str = typer.Option(
        "anemic", "--profile", "-p",
        help=f"Patient profile: {', '.join(PROFILES.keys())}"
    ),
    gender: str  = typer.Option("male",   "--gender", "-g", help="male | female"),
    age:    int  = typer.Option(40,        "--age",    "-a"),
    fmt:    str  = typer.Option("txt",     "--format", "-f", help="txt | pdf"),
    output: str  = typer.Option("data/sample_reports/", "--output", "-o"),
    seed:   int  = typer.Option(42,        "--seed",   help="Random seed for reproducibility"),
):
    """Generate a synthetic medical report for testing."""
    random.seed(seed)

    if profile not in PROFILES:
        console.print(f"[red]Unknown profile '{profile}'. Choose: {list(PROFILES.keys())}[/red]")
        raise typer.Exit(1)

    if gender not in ("male", "female"):
        console.print("[red]Gender must be 'male' or 'female'[/red]")
        raise typer.Exit(1)

    report_text = _generate_report_text(profile, gender, age)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "txt":
        fname = out_dir / f"synthetic_{profile}_{gender}_{age}.txt"
        fname.write_text(report_text, encoding="utf-8")
        console.print(f"[green]✓[/green] Generated: [cyan]{fname}[/cyan]")

    elif fmt == "pdf":
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import cm

            fname = out_dir / f"synthetic_{profile}_{gender}_{age}.pdf"
            doc    = SimpleDocTemplate(str(fname), pagesize=A4)
            styles = getSampleStyleSheet()
            story  = []
            for line in report_text.splitlines():
                style = styles["Heading2"] if line.startswith("=") or line.isupper() else styles["Normal"]
                story.append(Paragraph(line or "&nbsp;", style))
                story.append(Spacer(1, 0.1 * cm))
            doc.build(story)
            console.print(f"[green]✓[/green] Generated PDF: [cyan]{fname}[/cyan]")
        except ImportError:
            console.print("[yellow]reportlab not installed. Saving as .txt instead.[/yellow]")
            console.print("Install with: pip install reportlab")
            fname = out_dir / f"synthetic_{profile}_{gender}_{age}.txt"
            fname.write_text(report_text)
            console.print(f"[green]✓[/green] Generated: [cyan]{fname}[/cyan]")
    else:
        console.print(f"[red]Unknown format '{fmt}'. Use txt or pdf.[/red]")
        raise typer.Exit(1)

    console.print(f"\nTest it with:")
    console.print(f"  [bold]python scripts/analyze_report.py --report {fname}[/bold]")


if __name__ == "__main__":
    app()

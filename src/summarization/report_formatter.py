"""
src/summarization/report_formatter.py
Formats the full analysis result into structured output formats:

  - as_json()        : machine-readable full result
  - as_plain_text()  : human-readable plain text report
  - as_markdown()    : markdown report suitable for saving or rendering
  - as_cli_table()   : rich Table object for terminal display
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def as_json(result_dict: Dict[str, Any], indent: int = 2) -> str:
    """Serialise the full analysis result dict to a JSON string."""
    import json
    return json.dumps(result_dict, indent=indent, ensure_ascii=False, default=str)


def as_plain_text(result_dict: Dict[str, Any]) -> str:
    """Render the analysis result as a readable plain-text report."""
    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("MEDICAL REPORT ANALYSIS")
    lines.append(sep)
    lines.append(f"File: {result_dict.get('file_path', 'unknown')}")
    lines.append("")

    # Detection summary
    detection = result_dict.get("detection", {})
    if detection.get("summary"):
        lines.append("OVERVIEW")
        lines.append("-" * 40)
        lines.append(detection["summary"])
        lines.append("")

    # Critical flags
    critical = detection.get("critical_flags", [])
    if critical:
        lines.append("⚠  CRITICAL FLAGS")
        lines.append("-" * 40)
        for flag in critical:
            lines.append(f"  • {flag}")
        lines.append("")

    # Findings table
    findings = detection.get("findings", [])
    if findings:
        lines.append("LAB FINDINGS")
        lines.append("-" * 40)
        lines.append(f"{'Test':<30} {'Value':>10} {'Unit':<10} {'Status':<14} Reference Range")
        lines.append("-" * 90)
        for f in findings:
            ref = ""
            if f.get("reference_low") is not None and f.get("reference_high") is not None:
                ref = f"{f['reference_low']} – {f['reference_high']}"
            flag = " !" if f.get("status") not in ("Normal", "Unknown") else ""
            lines.append(
                f"{f.get('display_name', f.get('test_name', '')):<30}"
                f" {str(f.get('value', ''))!s:>10}"
                f" {f.get('unit', ''):<10}"
                f" {f.get('status', ''):<14}{flag}"
                f" {ref}"
            )
        lines.append("")

    # Clinical summary
    summary = result_dict.get("summary", {})
    if summary.get("clinical_summary"):
        lines.append("CLINICAL SUMMARY")
        lines.append("-" * 40)
        lines.append(summary["clinical_summary"])
        lines.append("")

    # Patient summary
    if summary.get("patient_summary"):
        lines.append("PATIENT-FRIENDLY SUMMARY")
        lines.append("-" * 40)
        lines.append(summary["patient_summary"])
        lines.append("")

    # Follow-up questions
    follow_ups = summary.get("follow_up_questions", [])
    if follow_ups:
        lines.append("SUGGESTED FOLLOW-UP QUESTIONS")
        lines.append("-" * 40)
        for i, q in enumerate(follow_ups, 1):
            lines.append(f"  {i}. {q}")
        lines.append("")

    # Recommendations
    recs = summary.get("recommendations", [])
    if recs:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, r in enumerate(recs, 1):
            lines.append(f"  {i}. {r}")
        lines.append("")

    lines.append(sep)
    lines.append("⚠  This report is for educational purposes only.")
    lines.append("   Always consult a qualified healthcare professional.")
    lines.append(sep)

    return "\n".join(lines)


def as_markdown(result_dict: Dict[str, Any]) -> str:
    """Render the analysis result as a Markdown document."""
    lines = []

    lines.append("# Medical Report Analysis")
    lines.append(f"\n**File:** `{result_dict.get('file_path', 'unknown')}`\n")

    detection = result_dict.get("detection", {})

    # Overview
    if detection.get("summary"):
        lines.append("## Overview\n")
        lines.append(detection["summary"])
        lines.append("")

    # Critical flags
    critical = detection.get("critical_flags", [])
    if critical:
        lines.append("## ⚠ Critical Flags\n")
        for flag in critical:
            lines.append(f"- **{flag}**")
        lines.append("")

    # Findings table
    findings = detection.get("findings", [])
    if findings:
        lines.append("## Lab Findings\n")
        lines.append("| Test | Value | Unit | Status | Reference Range |")
        lines.append("|------|-------|------|--------|-----------------|")
        for f in findings:
            ref = ""
            if f.get("reference_low") is not None:
                ref = f"{f['reference_low']} – {f.get('reference_high', '')}"
            status = f.get("status", "")
            emoji = {"Low": "🔻", "High": "🔺", "Critical Low": "🚨", "Critical High": "🚨"}.get(status, "✅")
            lines.append(
                f"| {f.get('display_name', f.get('test_name', ''))} "
                f"| {f.get('value', '')} "
                f"| {f.get('unit', '')} "
                f"| {emoji} {status} "
                f"| {ref} |"
            )
        lines.append("")

    summary = result_dict.get("summary", {})

    if summary.get("clinical_summary"):
        lines.append("## Clinical Summary\n")
        lines.append(summary["clinical_summary"])
        lines.append("")

    if summary.get("patient_summary"):
        lines.append("## Patient-Friendly Summary\n")
        lines.append(summary["patient_summary"])
        lines.append("")

    follow_ups = summary.get("follow_up_questions", [])
    if follow_ups:
        lines.append("## Suggested Follow-up Questions\n")
        for i, q in enumerate(follow_ups, 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    recs = summary.get("recommendations", [])
    if recs:
        lines.append("## Recommendations\n")
        for i, r in enumerate(recs, 1):
            lines.append(f"{i}. {r}")
        lines.append("")

    lines.append("---")
    lines.append("> ⚠ This report is for educational purposes only. "
                 "Always consult a qualified healthcare professional.")

    return "\n".join(lines)


def save_report(
    result_dict: Dict[str, Any],
    output_dir: str,
    stem: Optional[str] = None,
    formats: Optional[list] = None,
) -> Dict[str, str]:
    """
    Save the analysis result in one or more formats.

    Args:
        result_dict: Full analysis result dict.
        output_dir:  Directory to save files in.
        stem:        Base filename (without extension). Defaults to report filename stem.
        formats:     List of formats to save: ["json", "txt", "md"]. Defaults to all.

    Returns:
        Dict mapping format → saved file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if stem is None:
        fp = result_dict.get("file_path", "report")
        stem = Path(fp).stem

    formats = formats or ["json", "txt", "md"]
    saved: Dict[str, str] = {}

    if "json" in formats:
        p = out / f"{stem}_analysis.json"
        p.write_text(as_json(result_dict), encoding="utf-8")
        saved["json"] = str(p)

    if "txt" in formats:
        p = out / f"{stem}_report.txt"
        p.write_text(as_plain_text(result_dict), encoding="utf-8")
        saved["txt"] = str(p)

    if "md" in formats:
        p = out / f"{stem}_report.md"
        p.write_text(as_markdown(result_dict), encoding="utf-8")
        saved["md"] = str(p)

    return saved

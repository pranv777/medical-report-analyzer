#!/usr/bin/env python3
"""
scripts/analyze_report.py
CLI for analyzing a single medical report end-to-end.

Usage:
    python scripts/analyze_report.py --report data/sample_reports/sample_blood_test.pdf
    python scripts/analyze_report.py --report report.pdf --gender female --age 35 --output outputs/
"""
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Analyze a medical report using BioBERT + RAG.")


@app.command()
def analyze(
    report: str = typer.Option(..., "--report", "-r", help="Path to medical report (PDF or image)"),
    gender: str = typer.Option("default", "--gender", "-g", help="Patient gender: male | female | default"),
    age: int    = typer.Option(30, "--age", "-a", help="Patient age"),
    output: str = typer.Option("outputs/", "--output", "-o", help="Output directory"),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip LLM summarization"),
    chat: bool  = typer.Option(False, "--chat", "-c", help="Start interactive chat after analysis"),
    config: str = typer.Option(None, "--config", help="Path to config YAML"),
):
    """Analyze a medical report and print structured findings."""
    from src.utils.config import get_settings
    from src.pipeline import MedicalReportPipeline

    cfg = get_settings(config) if config else get_settings()

    console.rule("[bold green]Medical Report Analyser")
    console.print(f"Report  : [cyan]{report}[/cyan]")
    console.print(f"Gender  : [cyan]{gender}[/cyan]")
    console.print(f"Age     : [cyan]{age}[/cyan]")
    console.print(f"Output  : [cyan]{output}[/cyan]\n")

    pipeline = MedicalReportPipeline(cfg)

    with console.status("[bold green]Analysing report…"):
        result = pipeline.analyze(
            file_path=report,
            gender=gender,
            age=age,
            skip_summarization=no_summary,
        )

    if not result.success:
        console.print(f"[bold red]Analysis failed:[/bold red] {result.errors}")
        raise typer.Exit(1)

    pipeline.print_summary(result)

    out_path = pipeline.save_result(result, output)
    console.print(f"\n[green]✓[/green] Full JSON result saved → [cyan]{out_path}[/cyan]")

    # Optional interactive chat
    if chat:
        console.rule("[bold blue]Interactive Chat Mode")
        console.print("Ask questions about your report. Type [bold]exit[/bold] to quit.\n")
        while True:
            try:
                question = typer.prompt("You")
            except (KeyboardInterrupt, EOFError):
                break
            if question.strip().lower() in ("exit", "quit", "q"):
                break
            with console.status("[bold yellow]Thinking…"):
                answer = pipeline.ask(question)
            console.print(f"\n[bold cyan]Assistant:[/bold cyan] {answer}\n")


if __name__ == "__main__":
    app()

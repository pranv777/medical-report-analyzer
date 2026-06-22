#!/usr/bin/env python3
"""
scripts/chat.py
Interactive conversational chat about a medical report.

Usage:
    python scripts/chat.py --report data/sample_reports/sample_blood_test.pdf
    python scripts/chat.py  # chat without a report (general medical Q&A)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()
app = typer.Typer(help="Conversational medical Q&A with RAG.")


@app.command()
def chat(
    report: str = typer.Option(None, "--report", "-r", help="Optional: path to medical report"),
    gender: str = typer.Option("default", "--gender", "-g"),
    age: int    = typer.Option(30, "--age", "-a"),
    config: str = typer.Option(None, "--config"),
):
    """Start an interactive RAG chat session."""
    from src.utils.config import get_settings
    from src.pipeline import MedicalReportPipeline

    cfg = get_settings(config) if config else get_settings()
    pipeline = MedicalReportPipeline(cfg)

    console.print(Panel.fit(
        "[bold blue]🏥 Medical Report Chat[/bold blue]\n"
        "Powered by BioBERT + RAG\n"
        "Type [bold]help[/bold] for commands, [bold]exit[/bold] to quit.",
        border_style="blue",
    ))

    if report:
        console.print(f"\n[yellow]Analysing report:[/yellow] {report}")
        with console.status("[bold green]Loading report…"):
            result = pipeline.analyze(
                file_path=report,
                gender=gender,
                age=age,
                skip_summarization=False,
            )
        if result.success:
            console.print(f"[green]✓[/green] Report loaded. {result.detection.get('summary', '')}\n")
        else:
            console.print(f"[red]Warning:[/red] Report analysis failed. Continuing in general Q&A mode.\n")
    else:
        console.print("\n[dim]No report loaded. Answering from medical knowledge base only.[/dim]\n")

    # Command help text
    commands = {
        "help":   "Show available commands",
        "reset":  "Clear conversation history",
        "report": "Show report findings",
        "exit":   "Exit the chat",
    }

    while True:
        try:
            user_input = typer.prompt("\nYou").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # ── Commands ────────────────────────
        if user_input.lower() == "exit":
            console.print("[dim]Goodbye![/dim]")
            break

        elif user_input.lower() == "help":
            console.print("\n[bold]Commands:[/bold]")
            for cmd, desc in commands.items():
                console.print(f"  [cyan]{cmd:10}[/cyan] {desc}")
            continue

        elif user_input.lower() == "reset":
            pipeline.reset_chat()
            console.print("[green]✓[/green] Conversation history cleared.")
            continue

        elif user_input.lower() == "report":
            if report and pipeline._last_result:
                pipeline.print_summary(pipeline._last_result)
            else:
                console.print("[dim]No report loaded.[/dim]")
            continue

        # ── RAG answer ──────────────────────
        with console.status("[bold yellow]Searching knowledge base…"):
            answer = pipeline.ask(user_input)

        console.print(f"\n[bold cyan]Assistant:[/bold cyan]")
        console.print(Markdown(answer))
        console.print()


if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""
scripts/fetch_pubmed.py
Fetches PubMed abstracts on medical topics and saves them as JSON
for indexing into the knowledge base.

Uses the NCBI Entrez API (no API key needed for small queries;
set NCBI_API_KEY env var for higher rate limits).

Usage:
    python scripts/fetch_pubmed.py
    python scripts/fetch_pubmed.py --topics "anemia" "diabetes" --max 50
    python scripts/fetch_pubmed.py --topics "hypothyroidism" --max 20 --output data/knowledge_base/custom/
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
app = typer.Typer(help="Fetch PubMed abstracts for the knowledge base.")

DEFAULT_TOPICS = [
    "anemia hemoglobin iron deficiency",
    "diabetes mellitus HbA1c management",
    "dyslipidemia cholesterol cardiovascular",
    "hypothyroidism TSH thyroid",
    "vitamin D deficiency supplementation",
    "chronic kidney disease creatinine GFR",
    "hypertension blood pressure treatment",
    "complete blood count interpretation",
]


def fetch_pubmed_ids(query: str, max_results: int = 20) -> List[str]:
    """Search PubMed and return a list of PMIDs."""
    try:
        from Bio import Entrez
    except ImportError:
        raise ImportError("Install biopython: pip install biopython")

    Entrez.email = os.environ.get("NCBI_EMAIL", "researcher@example.com")
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])


def fetch_abstracts(pmids: List[str]) -> List[dict]:
    """Fetch title + abstract for a list of PMIDs."""
    try:
        from Bio import Entrez, Medline
    except ImportError:
        raise ImportError("Install biopython: pip install biopython")

    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="medline",
        retmode="text",
    )
    records = list(Medline.parse(handle))
    handle.close()

    docs = []
    for rec in records:
        title    = rec.get("TI", "")
        abstract = rec.get("AB", "")
        pmid     = rec.get("PMID", "")
        authors  = rec.get("AU", [])
        year     = rec.get("DP", "")[:4] if rec.get("DP") else ""

        if abstract:  # skip records without abstracts
            docs.append({
                "pmid":     pmid,
                "title":    title,
                "abstract": abstract,
                "authors":  authors[:3],   # first 3 authors
                "year":     year,
            })
    return docs


@app.command()
def fetch(
    topics: Optional[List[str]] = typer.Option(
        None, "--topics", "-t", help="Search topics (space-separated strings)"
    ),
    max_per_topic: int = typer.Option(20, "--max", "-m", help="Max abstracts per topic"),
    output: str = typer.Option(
        "data/knowledge_base/custom/", "--output", "-o", help="Output directory"
    ),
):
    """Fetch PubMed abstracts and save as JSON for knowledge base ingestion."""
    search_topics = topics or DEFAULT_TOPICS
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for topic in search_topics:
            task = progress.add_task(f"Fetching: {topic[:50]}…", total=None)
            try:
                pmids = fetch_pubmed_ids(topic, max_results=max_per_topic)
                progress.update(task, description=f"Fetching {len(pmids)} abstracts for: {topic[:40]}…")
                time.sleep(0.4)  # NCBI rate limit: 3 req/sec without key, 10 with key

                docs = fetch_abstracts(pmids)
                time.sleep(0.4)

                if docs:
                    safe_name = topic[:40].replace(" ", "_").replace("/", "_")
                    out_file = out_dir / f"pubmed_{safe_name}.json"
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(docs, f, indent=2, ensure_ascii=False)
                    total_saved += len(docs)
                    progress.update(task, description=f"[green]✓[/green] Saved {len(docs)} abstracts → {out_file.name}")
                else:
                    progress.update(task, description=f"[yellow]No abstracts found for: {topic[:40]}[/yellow]")

            except Exception as exc:
                progress.update(task, description=f"[red]Error for '{topic[:30]}': {exc}[/red]")
            finally:
                progress.stop_task(task)

    console.print(f"\n[bold green]Done![/bold green] Saved {total_saved} abstracts to [cyan]{output}[/cyan]")
    console.print("Run [bold]make build-kb[/bold] to index them into ChromaDB.")


if __name__ == "__main__":
    app()

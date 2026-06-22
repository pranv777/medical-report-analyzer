#!/usr/bin/env python3
"""
scripts/build_knowledge_base.py
Fetches, chunks, embeds and stores medical documents in ChromaDB.

What it does:
  1. Downloads sample PubMed abstracts (via built-in samples if API unavailable)
  2. Loads any TXT / JSON files from data/knowledge_base/custom/
  3. Chunks and embeds everything with the configured embedding model
  4. Persists to ChromaDB at the path set in configs/config.yaml

Usage:
    python scripts/build_knowledge_base.py
    python scripts/build_knowledge_base.py --sources data/my_docs/
    python scripts/build_knowledge_base.py --reset   # wipe and rebuild
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Build the medical RAG knowledge base.")

# ── Sample medical documents bundled with the project ──────────────────────
SAMPLE_DOCUMENTS = [
    {
        "content": (
            "Anemia is a condition where you lack enough healthy red blood cells to carry "
            "adequate oxygen to your body's tissues. Hemoglobin is the protein in red blood "
            "cells that carries oxygen. Low hemoglobin levels lead to anemia. Symptoms include "
            "fatigue, weakness, pale or yellowish skin, irregular heartbeats, shortness of "
            "breath, dizziness, chest pain, cold hands and feet, headaches. Common causes: "
            "iron deficiency, vitamin B12 or folate deficiency, chronic disease, bone marrow "
            "disorders, hemolytic anemia. Treatment depends on cause — iron supplementation, "
            "dietary changes, B12 injections, or treating the underlying condition."
        ),
        "source": "built_in_anemia_overview",
        "doc_type": "medical_knowledge",
        "metadata": {"title": "Anemia Overview", "category": "hematology"},
    },
    {
        "content": (
            "Hemoglobin A1c (HbA1c) measures average blood glucose over the past 2-3 months. "
            "Normal: below 5.7%. Pre-diabetes: 5.7–6.4%. Diabetes: 6.5% or higher. "
            "HbA1c reflects glycemic control. Reducing HbA1c by 1% significantly lowers risk "
            "of diabetic complications including retinopathy, nephropathy, and neuropathy. "
            "Management includes lifestyle changes (diet, exercise), metformin as first-line "
            "pharmacotherapy, and escalation to GLP-1 agonists, SGLT-2 inhibitors, or insulin "
            "based on individualized targets."
        ),
        "source": "built_in_hba1c_guide",
        "doc_type": "medical_knowledge",
        "metadata": {"title": "HbA1c and Diabetes", "category": "endocrinology"},
    },
    {
        "content": (
            "Cholesterol is a fatty substance in the blood. Total cholesterol under 200 mg/dL "
            "is desirable. LDL (low-density lipoprotein) is 'bad' cholesterol; levels under "
            "100 mg/dL are optimal. HDL (high-density lipoprotein) is 'good' cholesterol; "
            "levels above 60 mg/dL are protective. High LDL increases atherosclerosis risk. "
            "Triglycerides under 150 mg/dL are normal. Lifestyle interventions: reduce "
            "saturated fat and trans fats, increase soluble fiber, regular aerobic exercise, "
            "weight management. Statin therapy is first-line pharmacological treatment for "
            "elevated LDL with cardiovascular risk."
        ),
        "source": "built_in_cholesterol_guide",
        "doc_type": "medical_knowledge",
        "metadata": {"title": "Cholesterol and Lipid Management", "category": "cardiology"},
    },
    {
        "content": (
            "Thyroid Stimulating Hormone (TSH) is the primary screening test for thyroid "
            "disorders. Normal TSH range: 0.4–4.0 mIU/L. Low TSH with high free T4 "
            "suggests hyperthyroidism (overactive thyroid). High TSH with low free T4 "
            "indicates hypothyroidism (underactive thyroid). Symptoms of hypothyroidism: "
            "fatigue, weight gain, cold intolerance, constipation, dry skin, depression. "
            "Symptoms of hyperthyroidism: weight loss, heat intolerance, palpitations, "
            "anxiety, tremor. Levothyroxine is standard treatment for hypothyroidism. "
            "Hyperthyroidism options include anti-thyroid drugs, radioiodine, surgery."
        ),
        "source": "built_in_thyroid_guide",
        "doc_type": "medical_knowledge",
        "metadata": {"title": "Thyroid Function Tests", "category": "endocrinology"},
    },
    {
        "content": (
            "Vitamin D deficiency is common worldwide. Optimal level: 30–100 ng/mL. "
            "Insufficiency: 20–29 ng/mL. Deficiency: under 20 ng/mL. Severe: under 10 ng/mL. "
            "Causes: inadequate sun exposure, dietary insufficiency, malabsorption, obesity, "
            "darker skin pigmentation. Consequences: bone loss (osteoporosis/osteomalacia), "
            "muscle weakness, increased fracture risk, immune dysfunction, depression. "
            "Treatment: cholecalciferol (D3) supplementation. Daily requirements: 600–800 IU "
            "for adults, 1500–2000 IU for deficiency correction. Monitor levels 3 months "
            "after supplementation initiation."
        ),
        "source": "built_in_vitamin_d_guide",
        "doc_type": "medical_knowledge",
        "metadata": {"title": "Vitamin D Deficiency", "category": "nutrition"},
    },
    {
        "content": (
            "Chronic Kidney Disease (CKD) staging is based on Glomerular Filtration Rate (GFR) "
            "and urine albumin-to-creatinine ratio (UACR). Stage 1: GFR ≥90 (normal/high). "
            "Stage 2: GFR 60-89 (mildly decreased). Stage 3a: 45-59, Stage 3b: 30-44 (moderately). "
            "Stage 4: 15-29 (severely decreased). Stage 5: <15 (kidney failure). "
            "Serum creatinine is elevated when GFR drops below ~50%. BUN elevation accompanies "
            "creatinine rise. Management: blood pressure control (target <130/80), RAAS "
            "blockade with ACE inhibitors or ARBs, glycemic control in diabetes, dietary "
            "protein restriction in advanced CKD, preparation for renal replacement therapy."
        ),
        "source": "built_in_ckd_guide",
        "doc_type": "medical_knowledge",
        "metadata": {"title": "Chronic Kidney Disease", "category": "nephrology"},
    },
    {
        "content": (
            "Iron deficiency anemia is the most common nutritional deficiency worldwide. "
            "Ferritin is the most sensitive marker of iron stores. Low ferritin (<12 ng/mL "
            "in women, <20 ng/mL in men) confirms iron deficiency. Serum iron and TIBC "
            "are complementary tests. Causes: inadequate dietary intake, malabsorption "
            "(celiac disease, IBD), blood loss (menstrual, GI bleeding), increased demand "
            "(pregnancy, growth). Treatment: oral ferrous sulfate 325 mg TID. IV iron for "
            "intolerance or severe deficiency. Investigate source of blood loss in adults. "
            "Dietary sources: red meat, legumes, dark leafy greens. Vitamin C enhances absorption."
        ),
        "source": "built_in_iron_deficiency_guide",
        "doc_type": "medical_knowledge",
        "metadata": {"title": "Iron Deficiency Anemia", "category": "hematology"},
    },
    {
        "content": (
            "The complete blood count (CBC) is a fundamental diagnostic test measuring: "
            "Red blood cells (RBC), White blood cells (WBC), Hemoglobin (Hgb), Hematocrit (Hct), "
            "Mean corpuscular volume (MCV), Mean corpuscular hemoglobin (MCH), "
            "Mean corpuscular hemoglobin concentration (MCHC), Platelet count. "
            "MCV helps classify anemia: microcytic (<80 fL, iron deficiency/thalassemia), "
            "normocytic (80-100 fL, chronic disease/blood loss), macrocytic (>100 fL, "
            "B12/folate deficiency). WBC differential identifies neutrophilia (bacterial "
            "infection), lymphocytosis (viral infection), eosinophilia (allergy/parasites). "
            "Thrombocytopenia (<150,000/µL) raises bleeding risk."
        ),
        "source": "built_in_cbc_interpretation",
        "doc_type": "medical_knowledge",
        "metadata": {"title": "CBC Interpretation Guide", "category": "hematology"},
    },
]


@app.command()
def build(
    sources: str = typer.Option(
        None, "--sources", "-s", help="Additional source directory to index"
    ),
    reset: bool = typer.Option(
        False, "--reset", help="Delete existing index and rebuild from scratch"
    ),
    config: str = typer.Option(None, "--config"),
):
    """Build or update the medical knowledge base."""
    from src.utils.config import get_settings
    from src.knowledge_base.kb_builder import KnowledgeBaseBuilder
    from src.utils.logger import setup_logger
    from pathlib import Path
    import shutil

    cfg = get_settings(config) if config else get_settings()
    setup_logger(log_dir=cfg.project.log_dir, log_level=cfg.project.log_level)

    if reset:
        db_path = Path(cfg.knowledge_base.chromadb_path)
        if db_path.exists():
            console.print(f"[yellow]Resetting ChromaDB at {db_path}…[/yellow]")
            shutil.rmtree(db_path)

    builder = KnowledgeBaseBuilder(cfg.knowledge_base)

    # 1. Add built-in sample documents
    console.print(f"[cyan]Indexing {len(SAMPLE_DOCUMENTS)} built-in medical documents…[/cyan]")
    added = builder.add_documents(SAMPLE_DOCUMENTS)
    console.print(f"[green]✓[/green] Built-in docs: {added} new chunks")

    # 2. Index from custom/source directories
    dirs = []
    custom_path = Path(cfg.knowledge_base.sources.get("custom_documents_path", "data/knowledge_base/custom/"))
    if custom_path.exists():
        dirs.append(str(custom_path))
    if sources:
        dirs.append(sources)

    if dirs:
        console.print(f"[cyan]Indexing documents from: {dirs}[/cyan]")
        with console.status("[bold green]Chunking and embedding…"):
            added2 = builder.build(source_dirs=dirs)
        console.print(f"[green]✓[/green] Custom docs: {added2} new chunks")

    stats = builder.get_collection_stats()
    console.print(f"\n[bold green]Knowledge base ready![/bold green]")
    console.print(f"  Collection : {stats['collection_name']}")
    console.print(f"  Total docs : {stats['total_documents']}")
    console.print(f"  Model      : {stats['embedding_model']}")
    console.print(f"  Path       : {stats['chromadb_path']}")


if __name__ == "__main__":
    app()

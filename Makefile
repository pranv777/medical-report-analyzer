# ============================================================
# Makefile — Medical Report Understanding Project
# ============================================================
.PHONY: help install install-dev setup build-kb analyze chat eval test lint clean

PYTHON := python
REPORT ?= data/sample_reports/sample_blood_test.txt

help:
	@echo ""
	@echo "Medical Report Understanding — Available Commands"
	@echo "================================================="
	@echo "  make install       Install all dependencies"
	@echo "  make install-dev   Install with dev/test extras"
	@echo "  make setup         Full first-time setup"
	@echo "  make build-kb      Build the knowledge base"
	@echo "  make analyze       Analyze a report (set REPORT=path/to/report.pdf)"
	@echo "  make chat          Start interactive chat"
	@echo "  make eval          Run full evaluation suite"
	@echo "  make eval-ner      Run NER evaluation only"
	@echo "  make eval-embed    Run embedding benchmark only"
	@echo "  make eval-ragas    Run RAGAS evaluation only"
	@echo "  make test          Run unit tests"
	@echo "  make test-fast     Run tests excluding slow model tests"
	@echo "  make lint          Run code linters"
	@echo "  make clean         Remove generated files"
	@echo "  make mlflow        Open MLflow UI"
	@echo ""

# ── Install ────────────────────────────────────────────────

install:
	pip install -r requirements.txt
	python -m spacy download en_core_sci_md || \
	  pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"

install-dev: install
	pip install pytest pytest-asyncio pytest-cov black isort flake8

setup: install
	cp -n .env.example .env || true
	mkdir -p data/raw data/processed data/knowledge_base/custom outputs logs
	@echo ""
	@echo "✓ Setup complete. Next steps:"
	@echo "  1. Edit .env with your settings"
	@echo "  2. Run: make build-kb"
	@echo "  3. Run: make analyze"
	@echo ""

# ── Knowledge Base ─────────────────────────────────────────

build-kb:
	$(PYTHON) scripts/build_knowledge_base.py

rebuild-kb:
	$(PYTHON) scripts/build_knowledge_base.py --reset

# ── Analysis ───────────────────────────────────────────────

analyze:
	$(PYTHON) scripts/analyze_report.py --report $(REPORT)

analyze-chat:
	$(PYTHON) scripts/analyze_report.py --report $(REPORT) --chat

chat:
	$(PYTHON) scripts/chat.py

# ── Evaluation ─────────────────────────────────────────────

eval:
	$(PYTHON) scripts/run_evaluation.py --mode all

eval-ner:
	$(PYTHON) scripts/run_evaluation.py --mode ner

eval-embed:
	$(PYTHON) scripts/run_evaluation.py --mode embedding

eval-ragas:
	$(PYTHON) scripts/run_evaluation.py --mode ragas

# ── Testing ────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -k "not slow"

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ── Code quality ───────────────────────────────────────────

lint:
	black --check src/ scripts/ tests/
	isort --check-only src/ scripts/ tests/
	flake8 src/ scripts/ tests/ --max-line-length 100

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# ── MLflow ─────────────────────────────────────────────────

mlflow:
	mlflow ui --backend-store-uri logs/mlflow --port 5000

# ── Jupyter ────────────────────────────────────────────────

notebook:
	jupyter notebook notebooks/

# ── Ollama helpers ─────────────────────────────────────────

pull-models:
	ollama pull llama3
	ollama pull mistral

# ── Clean ──────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ htmlcov/ .coverage

clean-outputs:
	rm -rf outputs/*.json outputs/*.png outputs/*.html

clean-all: clean clean-outputs
	rm -rf data/knowledge_base/chroma_db/ logs/ models/

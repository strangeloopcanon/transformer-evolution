PY=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PYBIN=$(VENV)/bin/python

.PHONY: setup check test deps-audit all index lineage

setup:
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "Venv ready at $(VENV)"

check:
	$(VENV)/bin/black --check .
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy src
	$(VENV)/bin/bandit -q -r src || true
	$(VENV)/bin/detect-secrets scan --baseline .secrets.baseline || true

test:
	$(VENV)/bin/pytest -q --cov=src --cov-report=term-missing

deps-audit:
	$(VENV)/bin/pip-audit -r requirements.txt || true

all: check test deps-audit

# Build a runs index and write a tracked snapshot to docs/
index:
	$(PYBIN) scripts/index_results.py
	@mkdir -p docs
	@cp -f results/index.json docs/results_index.json 2>/dev/null || true
	@echo "Indexed runs -> results/index.json (snapshot: docs/results_index.json)"

# Reconstruct lineage and refresh the canonical lineage focus image in docs/
# Usage: make lineage RUN=results/<your_run_dir> [K=3]
lineage:
	@test -n "$(RUN)" || RUN=$$(ls -1dt results/evolution_* 2>/dev/null | head -n1); \
	K?=3; \
	$(PYBIN) scripts/reconstruct_lineage.py $$RUN --seed configs examples || true; \
	if [ -f "$$RUN/lineage_reconstructed.json" ]; then \
		$(PYBIN) scripts/lineage_focus.py $$RUN/lineage_reconstructed.json --k $$K; \
		$(PYBIN) scripts/lineage_to_mermaid.py $$RUN/lineage_focus.json --with-subgraphs --attach-genesis --genesis-label "Transformer (Vaswani, 2017)" > $$RUN/lineage_focus.mmd; \
		if command -v mmdc >/dev/null 2>&1; then \
			mmdc -i $$RUN/lineage_focus.mmd -o $$RUN/lineage_focus.png -t default -w 3000 -H 2000 || true; \
		fi; \
		mkdir -p docs; \
		[ -f $$RUN/lineage_focus.png ] && cp -f $$RUN/lineage_focus.png docs/lineage_focus.png || true; \
		cp -f $$RUN/lineage_focus.json docs/lineage_focus.json || true; \
		echo "Refreshed docs/lineage_focus.png from $$RUN"; \
	else \
		echo "No lineage_reconstructed.json found in $$RUN"; \
	fi

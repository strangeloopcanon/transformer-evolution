PY=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PYBIN=$(VENV)/bin/python

.PHONY: setup check test deps-audit all

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


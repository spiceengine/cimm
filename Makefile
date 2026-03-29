VENV = ./.venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest

activate:
	python3 -m venv $(VENV)
	$(PIP) install torch torchvision pytest
	$(PIP) install -e .

test:
	$(PYTEST) tests/

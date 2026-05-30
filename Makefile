.PHONY: help install deploy dev smoke-train smoke-agent logs

help:
	@echo "Common targets:"
	@echo "  make install        — set up Python venv + frontend node_modules"
	@echo "  make deploy         — deploy backend to Modal"
	@echo "  make dev            — run frontend on http://localhost:3000"
	@echo "  make smoke-train    — run end-to-end billsum training on Modal"
	@echo "  make logs           — tail Modal app logs"

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e .
	cd frontend && npm install

deploy:
	modal deploy backend/api.py

dev:
	cd frontend && npm run dev

smoke-train:
	modal run backend/train.py::smoke

logs:
	modal app logs autoft

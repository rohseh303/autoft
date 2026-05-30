.PHONY: help install deploy dev smoke-train smoke-agent logs optimize trial

help:
	@echo "Common targets:"
	@echo "  make install        — set up Python venv + frontend node_modules"
	@echo "  make deploy         — deploy backend to Modal"
	@echo "  make dev            — run frontend on http://localhost:3000"
	@echo "  make smoke-train    — run end-to-end billsum training on Modal"
	@echo "  make optimize       — Codex drives the train/eval/judge loop (post-training lead)"
	@echo "  make trial          — run one trial: plan.json -> result.json + trials.jsonl"
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

optimize:
	bash scripts/optimize.sh

trial:
	uv run modal run backend/train.py::trial --plan plan.json --out result.json

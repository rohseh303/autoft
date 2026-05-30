# AutoFT — autonomous fine-tuning on Modal

Paste a task ("I want a model that summarizes legal contracts"), a Codex-powered research
agent picks a HuggingFace dataset and a recipe, Modal trains a LoRA on Qwen3.5-2B,
the dashboard plots live metrics, and you get a before/after comparison on your own examples.

## Stack

- **Compute:** Modal (H200 GPU)
- **Fine-tuning:** Unsloth + HF Trainer (bf16, LoRA r=16)
- **Models:** Qwen3.5-2B (bf16 LoRA)
- **Research agent:** OpenAI Codex CLI (emits a structured RunPlan) + `gpt-5.4-mini` elementwise judge
- **Frontends:** `studio/` (Bun + Elysia BFF + React/Vite, polled metrics + reasoning + logs) and `frontend/` (Next.js 15)
- **Dataset discovery:** `huggingface_hub` + HF `datasets-server` REST

## Setup

```bash
# 0. Python env
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 1. Authenticate Modal
modal token new

# 2. Create secrets (one-time)
modal secret create openai-secret OPENAI_API_KEY=sk-...
modal secret create hf-secret HF_TOKEN=hf_...   # optional; needed for gated datasets

# 3. Deploy the backend
modal deploy backend/api.py
# Copy the printed URL into frontend/.env.local:
#   NEXT_PUBLIC_API_BASE=https://<you>--autoft-web.modal.run

# 4. Run the frontend
cd frontend
npm install
npm run dev
# open http://localhost:3000
```

## Smoke tests

```bash
# Train end-to-end with a hardcoded billsum plan (no agent, no UI):
modal run backend/train.py::smoke

# Verify the agent without training:
modal run backend/research_agent.py::research --task-description "summarize legal contracts" --eval-examples '[]' --preferred-model null
```

## Architecture

```
Next.js UI                Modal (api_image)              Modal (train_image, H200)
─────────                 ───────────────────             ─────────────────────────
TaskForm   ─POST /research─▶ research_agent
                            (OpenAI + HF tools)
            ◀── RunPlan ────
PlanPreview ─POST /train ──▶ api spawns train_run ─▶  Unsloth + TRL SFTTrainer
                            returns run_id              StreamCallback ─▶ Queue
Dashboard  ─SSE /stream─▶  reads Queue, emits events    ... save LoRA
                            ◀── metrics ───────────     run base+FT eval
Comparison ─GET /result─▶  reads run_results Dict       write run_results
```

The `RunPlan` Pydantic schema (`shared/schemas.py`) is the spine: produced by the agent,
consumed by training, rendered by the UI. Same definition in TypeScript at
`frontend/lib/types.ts`.

## What's intentionally NOT in v1

- Mid-training intervention (pause / edit / resume)
- Real benchmark eval harness (lm-eval-harness, MT-Bench)
- Multi-run hyperparam sweep
- Push to HuggingFace Hub button
- Agent reasoning trace pane (the streaming "thought process" view)

See plan: `~/.claude/plans/need-to-refine-define-this-purrfect-candy.md`

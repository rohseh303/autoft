# AutoFT — One-shot autonomous fine-tuning platform

## Context

**Problem.** Fine-tuning an open-source LLM for a specific task today still requires the user to know: which benchmarks evaluate the capability, which HF dataset to train on, how to format it, what hyperparameters to use, and how to read a training run. Each step is a research task by itself, and most product builders skip it entirely and just prompt a frontier model.

**Pitch.** The user pastes a task ("I want a model that's good at summarizing legal contracts") plus optional eval examples. A research agent picks the relevant benchmark(s), scrapes a fitting dataset off HuggingFace, designs a training recipe, and kicks off a Modal job. A live dashboard streams loss curves, eval scores, and the agent's reasoning while training. At the end, the user gets a fine-tuned open-source model plus a before/after comparison on their own eval examples.

**Constraints.** 8-hour Modal hackathon, two-ish people relying on AI coding velocity. Demo must be visually impressive *and* technically real — not a Wizard-of-Oz.

**Scope discipline.** v1 ships: free-form task → research agent → autonomous training → live dashboard → before/after comparison. Out of scope for v1, prioritized for v2: human-in-the-loop intervention mid-training, benchmark leaderboard, multi-run comparison, agent reasoning trace pane.

---

## Architecture

```
┌─────────────────┐    POST /research   ┌──────────────────────┐
│  Next.js (UI)   │ ───────────────────▶│ Modal: research_agent│
│  - task input   │                     │  - OpenAI call       │
│  - eval examples│ ◀── plan preview ───│  - HF Hub search     │
│                 │                     │  - returns RunPlan   │
│                 │   POST /train       └──────────┬───────────┘
│                 │ ────────────────────▶          │
│                 │                     ┌──────────▼───────────┐
│  - dashboard    │                     │ Modal: train (GPU)   │
│  - SSE consumer │ ◀── SSE metrics ────│  - Unsloth + TRL SFT │
│                 │                     │  - emits step metrics│
│  - diff view    │ ◀── eval results ───│  - runs eval at end  │
└─────────────────┘                     └──────────────────────┘
```

**Stack (locked in):**
- Compute: **Modal** — L4 (24 GB) for training, ~$0.80/hr; A10G fallback. [Modal GPU docs](https://modal.com/docs/guide/gpu)
- Fine-tuning: **Unsloth + TRL SFTTrainer** — 2× faster, single-GPU friendly. [Unsloth Qwen guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Research agent: **OpenAI GPT** (`gpt-4.1-mini`) with structured outputs + tool calls
- Dataset discovery: `huggingface_hub.list_datasets(filter=DatasetFilter(task=...))` + HF `datasets-server` REST for column schemas
- Frontend: **Next.js (App Router)** + Tremor charts + native `EventSource` for SSE
- Models: **Qwen2.5-0.5B-Instruct** (default) and **SmolLM2-1.7B-Instruct** (toggle)

---

## The `RunPlan` contract (the spine of the system)

Everything the agent decides and everything training consumes is one JSON object. Define it once, share between Modal and Next.

```python
class RunPlan(BaseModel):
    task_summary: str            # agent's restatement of user intent
    base_model: Literal["Qwen2.5-0.5B-Instruct", "SmolLM2-1.7B-Instruct"]
    hf_dataset: str              # e.g. "billsum"
    dataset_config: str | None
    dataset_split: str           # e.g. "train[:2000]"
    input_field: str             # column to use as instruction
    output_field: str            # column to use as target
    prompt_template: str         # f-string with {input} and {output}
    benchmarks: list[str]        # human-readable names; v1 just labels
    training: TrainingConfig     # lr, steps, batch_size, lora_r, etc.
    reasoning: str               # agent's "why" — shown in UI
```

Locking this contract first (hour 1) is the single most important decision — it lets the agent, trainer, and UI be built in parallel against the same shape.

---

## Phased build (8-hour budget)

### Hour 0 — 0:45 · Scaffold (parallelizable)
- Repo init; `pyproject.toml` with `modal`, `openai`, `huggingface_hub`, `pydantic`.
- `frontend/` — Next.js with TypeScript + Tailwind; add Tremor.
- `backend/app.py` — define `App("autoft")`, base images, secrets for `OPENAI_API_KEY` and `HF_TOKEN`, persistent `Volume` / `Queue` / `Dict` references.
- Lock in `RunPlan` schema in `shared/schemas.py` (Python) and `frontend/lib/types.ts` (mirror).

### Hour 0:45 — 3:00 · Training pipeline end-to-end (with a hardcoded plan)
**Goal:** Hardcode a `RunPlan` for billsum + Qwen2.5-0.5B, get a full SFT run finishing on Modal in under 5 minutes. Until this works, nothing else matters.

- `backend/train.py`:
  - `@app.function(gpu="L4", timeout=1800)` decorator.
  - Load model via `FastLanguageModel.from_pretrained(..., load_in_4bit=True)`.
  - Apply LoRA: `FastLanguageModel.get_peft_model(r=16, target_modules=[...])`.
  - Load dataset via `datasets.load_dataset(plan.hf_dataset, split=plan.dataset_split)`.
  - Format with `plan.prompt_template`.
  - `SFTTrainer(..., args=SFTConfig(max_steps=plan.training.max_steps, ...))`.
  - **Critical for demo:** custom `TrainerCallback.on_log` that pushes `{step, loss, lr, grad_norm}` into a Modal `Queue` partitioned by `run_id`.
  - Inline before/after eval: `model.disable_adapter()` for base outputs vs LoRA outputs — no need to reload the base model separately.
- Smoke test: `modal run backend/train.py::smoke` end-to-end.

### Hour 3:00 — 5:00 · Research agent + dataset discovery
- `backend/research_agent.py`:
  - `@app.function(secrets=[openai_secret, hf_secret])`.
  - System prompt: "You design fine-tuning recipes. Given a task description and eval examples, output a RunPlan."
  - Tools the model can call:
    1. `search_hf_datasets(query, task_filter)` → wraps `HfApi.list_datasets` sorted by downloads.
    2. `peek_dataset(name, config, split)` → pulls 3 sample rows + column schema via the HF `datasets-server` `/info` and `/rows` REST endpoints to verify before committing.
  - **Output enforcement:** OpenAI structured outputs with `RunPlan` as the `response_format` `json_schema` — no parsing JSON from prose.
  - **Curated fallback catalog** (`shared/catalog.py`): ~12 hand-picked (task_tags → dataset → field mappings) entries. The agent gets this list in its system prompt as "preferred datasets — use these when they fit." Removes the "agent picks a dataset that 404s or has weird columns" failure mode for the demo.

### Hour 5:00 — 7:00 · Live dashboard
- `backend/api.py`:
  - FastAPI ASGI via `@modal.asgi_app()` + `@modal.concurrent(max_inputs=20)` so one container serves many SSE clients.
  - `POST /research` → calls the agent function, returns a `RunPlan`.
  - `POST /train` → spawns `train_run`, returns `run_id`.
  - `GET /run/{id}/status` → snapshot of `RunStatus` from the Modal `Dict`.
  - `GET /run/{id}/result` → final `RunResult` (plan + final_loss + comparisons).
  - `GET /run/{id}/stream` → SSE, reads from partitioned `metrics_queue`. The blocking `Queue.get` is wrapped in `asyncio.to_thread` so it doesn't stall the event loop. Emits `event: metric`, `event: status`, `event: result` over the same connection.
- `frontend/app/page.tsx`:
  - **Step 1:** Textarea for task + dynamic eval examples list (input/output pairs).
  - **Step 2:** Plan preview card — agent's reasoning, picked dataset (link to HF), benchmark list, recipe. Single "Start training" button.
  - **Step 3:** Dashboard — Tremor `LineChart` for loss (raw + EMA smoothed), KPI cards for step / lr / elapsed / grad norm. `EventSource('/run/{id}/stream')`.
  - **Step 4:** Comparison view — base model output vs FT output side-by-side for each user eval example.

### Hour 7:00 — 8:00 · Polish + demo prep
- Pre-warm Modal containers so demo doesn't show cold starts.
- Pre-run the canonical demo (e.g. legal summarization with billsum) once to populate the HF dataset cache + the Modal Volume.
- Record a 90-second fallback video in case live demo fails.
- README with one-paragraph pitch + architecture diagram.

---

## Files

| Path | Purpose |
|---|---|
| `backend/app.py` | Modal App, images, Volume, Queue, Dicts, secrets |
| `backend/research_agent.py` | OpenAI agent + HF tools + structured RunPlan output |
| `backend/train.py` | Unsloth SFT + metric-streaming callback + inline eval |
| `backend/api.py` | FastAPI endpoints: `/research`, `/train`, `/run/{id}/stream`, `/run/{id}/result` |
| `shared/schemas.py` | `RunPlan`, `TrainingConfig`, `StepMetric`, `RunStatus`, `RunResult` Pydantic models |
| `shared/catalog.py` | ~12 curated task→dataset entries (demo safety net) |
| `frontend/app/page.tsx` | Single-page wizard: input → plan → dashboard → diff |
| `frontend/components/TaskForm.tsx` | Free-form task + eval examples input |
| `frontend/components/PlanPreview.tsx` | Agent's RunPlan rendered as a reviewable card |
| `frontend/components/Dashboard.tsx` | Tremor charts + SSE consumer |
| `frontend/components/Comparison.tsx` | Side-by-side base vs FT outputs |
| `frontend/lib/types.ts` | Mirror of `RunPlan` schema |
| `frontend/lib/api.ts` | Typed fetch wrappers for Modal endpoints |

---

## Existing tools / patterns to reuse (don't rebuild)

- `trl.SFTTrainer` + `SFTConfig` — covers the entire training loop, no custom loop needed.
- `transformers.TrainerCallback` — the supported way to hook into training events; subclass it to push metrics.
- `modal.Queue` with partitions — built-in inter-function pub/sub; do *not* roll a Redis.
- `modal.Dict` — for run status + final results; simple key/value across functions.
- `huggingface_hub.list_datasets` with `DatasetFilter` — official search. [HF search guide](https://huggingface.co/docs/huggingface_hub/en/guides/search)
- HF `datasets-server` `/rows`, `/info` REST endpoints — sample rows + column schemas without downloading the full dataset. [Dataset viewer API](https://huggingface.co/docs/dataset-viewer/en/quick_start)
- OpenAI structured outputs (`response_format` with JSON schema) — kills JSON-parsing bugs.
- `model.disable_adapter()` from PEFT — base-model inference without reloading.
- Tremor's `LineChart` + `Card` — pre-built dashboard primitives, no custom D3.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Agent picks a broken HF dataset (404, weird columns) | Curated catalog in system prompt; `peek_dataset` tool forces the agent to verify columns before committing |
| Training run too slow for live demo | Cap `max_steps=150` for tiny models; L4 + 4-bit + LoRA gets Qwen-0.5B through ~150 steps in 2-3 min |
| Modal cold start / image build during live demo | First image build is ~10 min (Unsloth pulls torch+CUDA); pre-deploy 30 min before demo |
| Loss curve looks boring / noisy | EMA smoothing on the frontend (already shipped); stream both raw and smoothed series |
| Before/after diff is unimpressive on a 0.5B model in 150 steps | Pick a demo task with a strong style signal (e.g. format-following, persona) where small models visibly shift fast |
| HF dataset download bottlenecks training start | Pre-cache common datasets on the Modal Volume on first run; subsequent runs reuse |
| SSE event loop stalls under load | `Queue.get` wrapped in `asyncio.to_thread` + `@modal.concurrent(max_inputs=20)` |

---

## Verification (end-to-end)

1. **Training smoke:** `make smoke-train` (runs `modal run backend/train.py::smoke`) — confirm LoRA saved to Modal Volume and final loss < initial loss.
2. **Deploy + agent:** `make deploy`, then `curl -X POST <url>/research -d '{"task_description":"summarize legal contracts","eval_examples":[]}'` — inspect returned `RunPlan` matches schema.
3. **SSE smoke:** start a run, `curl -N <url>/run/<id>/stream` — confirm steady event stream of `event: metric` lines.
4. **Frontend flow:** `make dev`, from `localhost:3000` paste a task, watch plan render, click train, watch loss curve animate, see diff at the end.
5. **Demo dry-run:** Full flow with the canonical legal-summarization example, timed end-to-end, must fit in 4 minutes of presentation time.

---

## v2 backlog (post-hackathon, do not build now)

- Pause / edit / resume training from the dashboard
- Agent reasoning trace pane (stream the agent's thought process while it researches)
- Real benchmark leaderboard (run actual eval harness on `lm-eval-harness` tasks)
- Multi-run comparison (sweep hyperparams, pick the winner)
- "Push to HF Hub" button to publish the resulting model

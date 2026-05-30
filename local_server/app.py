"""Local AutoFT backend — FastAPI, no GPU / Modal / Codex.

Speaks the same contract as the Modal backend (shared/schemas.py) but adds SSE
streaming for BOTH phases:
  POST /research/stream   -> event: thought* , event: plan        (research live)
  POST /train             -> { run_id }
  GET  /run/{id}/stream   -> event: status, metric*, sample*, result (training live)

Plus blocking fallbacks (/research, /run/{id}/result, /run/{id}/status) so the
client works whether or not it uses the streaming path.

Run from the repo root so `shared` and `local_server` import cleanly:
    uvicorn local_server.app:app --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from local_server import engine
from shared.catalog import CATALOG
from shared.schemas import RunPlan, UserRequest

app = FastAPI(title="AutoFT Local")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory run store (demo scope)
_RUNS: dict[str, dict] = {}
_RESULTS: dict[str, dict] = {}

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _clean_plan(plan: dict) -> dict:
    """Strip internal hints and validate against the real RunPlan schema."""
    p = {k: v for k, v in plan.items() if not k.startswith("_")}
    return RunPlan.model_validate(p).model_dump()


@app.get("/")
def root():
    return {"name": "autoft-local", "ok": True}


@app.get("/health")
def health():
    return {"ok": True, "mode": "local"}


# --- research: blocking -----------------------------------------------------
@app.post("/research")
def do_research(req: UserRequest):
    plan = engine.build_plan(
        req.task_description,
        [e.model_dump() for e in req.eval_examples],
        req.preferred_model,
    )
    return _clean_plan(plan)


# --- research: streamed thought trace --------------------------------------
@app.post("/research/stream")
async def do_research_stream(req: UserRequest):
    eval_examples = [e.model_dump() for e in req.eval_examples]
    plan_full = engine.build_plan(req.task_description, eval_examples, req.preferred_model)
    thoughts = engine.research_thoughts(req.task_description, plan_full)
    plan = _clean_plan(plan_full)

    async def gen():
        for t in thoughts:
            yield _sse("thought", t)
            await asyncio.sleep(0.6)
        yield _sse("plan", plan)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


# --- train: spawn -----------------------------------------------------------
class TrainRequest(BaseModel):
    plan: RunPlan
    eval_examples: list = []
    # optional knob so the demo can run faster/slower without code changes
    speed: float | None = None


@app.post("/train")
def do_train(req: TrainRequest):
    run_id = f"run-{uuid.uuid4().hex[:10]}"
    plan = req.plan.model_dump()
    # re-derive the task category from the chosen dataset for sample/eval text
    entry = next((e for e in CATALOG if e["hf_dataset"] == plan["hf_dataset"]), None)
    category = engine._category(entry, plan["task_summary"]) if entry else "instruct"
    _RUNS[run_id] = {
        "plan": plan,
        "req": {"task_description": plan["task_summary"], "eval_examples": req.eval_examples or []},
        "category": category,
        "speed": req.speed or 1.0,
    }
    return {"run_id": run_id}


# --- training: live SSE -----------------------------------------------------
@app.get("/run/{run_id}/stream")
async def stream_run(run_id: str):
    entry = _RUNS.get(run_id)

    async def gen():
        if not entry:
            yield _sse("status", {"run_id": run_id, "state": "failed",
                                  "message": "unknown run", "plan": None})
            return

        plan = entry["plan"]
        req = entry["req"]
        category = entry["category"]
        speed = entry["speed"]
        max_steps = plan["training"]["max_steps"]
        rng = engine._seed(run_id, plan["hf_dataset"])
        scale = engine.loss_scale(category)
        # per-step delay: ~6s total at speed=1, scaled by max_steps
        step_delay = max(0.012, (6.0 / max_steps) / speed)

        # loading phase — granular status, no dead air
        for msg in ("spinning up worker", f"loading {plan['base_model']}",
                    f"loading {plan['hf_dataset']}", f"tokenizing {plan['dataset_split']}"):
            yield _sse("status", {"run_id": run_id, "state": "loading", "message": msg, "plan": plan})
            await asyncio.sleep(0.5 / speed)

        yield _sse("status", {"run_id": run_id, "state": "training",
                              "message": "trainer running", "plan": plan})

        start = time.time()
        last_sample = None
        for step in range(1, max_steps + 1):
            yield _sse("metric", {
                "run_id": run_id,
                "step": step,
                "loss": engine.loss_at(step, max_steps, rng, scale),
                "learning_rate": engine.lr_at(step, plan),
                "grad_norm": engine.grad_at(step, rng),
                "epoch": (step * plan["training"]["batch_size"]
                          * plan["training"]["gradient_accumulation_steps"]) / 2000,
                "elapsed_seconds": time.time() - start,
            })
            if step == 1 or step % max(1, max_steps // 12) == 0 or step == max_steps:
                text = engine.sample_at(step, max_steps, category)
                if text != last_sample:
                    last_sample = text
                    yield _sse("sample", {
                        "run_id": run_id, "step": step,
                        "prompt": (req["eval_examples"][0]["input"]
                                   if req["eval_examples"] else engine._default_eval_input(category)),
                        "text": text,
                    })
            await asyncio.sleep(step_delay)

        yield _sse("status", {"run_id": run_id, "state": "evaluating",
                              "message": "generating base vs fine-tuned", "plan": plan})
        await asyncio.sleep(0.9 / speed)

        final_loss = engine.loss_at(max_steps, max_steps, rng, scale)
        result = engine.build_result(run_id, plan, req, category, final_loss)
        _RESULTS[run_id] = result
        yield _sse("status", {"run_id": run_id, "state": "done",
                              "message": f"final_loss={final_loss:.4f}", "plan": plan})
        yield _sse("result", result)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/run/{run_id}/result")
def get_result(run_id: str):
    if run_id in _RESULTS:
        return _RESULTS[run_id]
    entry = _RUNS.get(run_id)
    if not entry:
        return StreamingResponse(iter(["not found"]), status_code=404)
    final_loss = engine.loss_at(entry["plan"]["training"]["max_steps"],
                                entry["plan"]["training"]["max_steps"],
                                engine._seed(run_id, entry["plan"]["hf_dataset"]),
                                engine.loss_scale(entry["category"]))
    return engine.build_result(run_id, entry["plan"], entry["req"], entry["category"], final_loss)


@app.get("/run/{run_id}/status")
def get_status(run_id: str):
    if run_id in _RESULTS:
        return {"run_id": run_id, "state": "done", "message": "done",
                "plan": _RUNS[run_id]["plan"]}
    entry = _RUNS.get(run_id)
    if not entry:
        return {"run_id": run_id, "state": "unknown", "message": "", "plan": None}
    return {"run_id": run_id, "state": "pending", "message": "queued", "plan": entry["plan"]}

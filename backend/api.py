"""FastAPI endpoints exposed via Modal: /research, /train, /run/{id}/stream, /run/{id}/result."""
import asyncio
import json
import uuid
from typing import Any

import modal

from backend.app import (
    api_image,
    app,
    hf_secret,
    metrics_queue,
    openai_secret,
    run_events,
    run_metrics,
    run_results,
    run_state,
)
from backend.research_agent import research
from backend.train import train_run


@app.function(
    image=api_image,
    secrets=[openai_secret, hf_secret],
    timeout=600,
    max_containers=4,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    from shared.schemas import EvalExample, RunPlan, RunStatus, UserRequest

    api = FastAPI(title="AutoFT")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/")
    def root():
        return {"name": "autoft", "ok": True}

    @api.post("/research")
    def do_research(req: UserRequest = Body(...)):
        # research() returns {"plan": RunPlan dict, "events": [reasoning trace]}.
        # The Studio BFF forwards both: the plan drives the PlanCard, the events
        # render the agent's live reasoning in the research notebook.
        return research.remote(
            req.task_description,
            [e.model_dump() for e in req.eval_examples],
            req.preferred_model,
        )

    class TrainRequest(BaseModel):
        plan: RunPlan
        eval_examples: list[EvalExample] = []

    @api.post("/train")
    def do_train(req: TrainRequest = Body(...)):
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        run_state[run_id] = RunStatus(
            run_id=run_id, state="pending", message="queued", plan=req.plan
        ).model_dump()
        # Spawn — returns immediately, training continues async
        train_run.spawn(run_id, req.plan.model_dump(), [e.model_dump() for e in req.eval_examples])
        return {"run_id": run_id}

    @api.get("/run/{run_id}/status")
    def get_status(run_id: str):
        try:
            return run_state[run_id]
        except KeyError:
            raise HTTPException(404, "unknown run")

    @api.get("/run/{run_id}/result")
    def get_result(run_id: str):
        try:
            return run_results[run_id]
        except KeyError:
            raise HTTPException(404, "result not ready")

    @api.get("/run/{run_id}/metrics")
    def get_metrics(run_id: str):
        # Polling-friendly snapshot for live charts: status + the full accumulated
        # step-metric array + final result (when ready). The frontend polls this.
        return {
            "status": run_state.get(run_id),
            "metrics": run_metrics.get(run_id, []),
            "result": run_results.get(run_id),
        }

    @api.get("/run/{run_id}/events")
    def get_events(run_id: str):
        # Transparency timeline: status transitions + agent/subprocess reasoning,
        # each tagged {source, step, kind, text}. Polled by the Studio UI.
        return {"events": run_events.get(run_id, [])}

    @api.get("/run/{run_id}/stream")
    def stream_metrics(run_id: str):
        import queue as _queue

        async def event_gen():
            # Send initial status snapshot so the client UI populates immediately.
            try:
                yield f"event: status\ndata: {json.dumps(run_state[run_id])}\n\n"
            except KeyError:
                yield f"event: status\ndata: {json.dumps({'run_id': run_id, 'state': 'unknown'})}\n\n"

            while True:
                try:
                    item = await asyncio.to_thread(
                        metrics_queue.get, partition=run_id, timeout=2, block=True,
                    )
                except _queue.Empty:
                    # Heartbeat + status check
                    try:
                        status = run_state[run_id]
                        yield f"event: status\ndata: {json.dumps(status)}\n\n"
                        if status.get("state") in ("done", "failed"):
                            try:
                                yield f"event: result\ndata: {json.dumps(run_results[run_id])}\n\n"
                            except KeyError:
                                pass
                            break
                    except KeyError:
                        pass
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.1)
                    continue

                if isinstance(item, dict) and item.get("done"):
                    # Wait briefly for state/results to settle, then push result.
                    for _ in range(20):
                        await asyncio.sleep(0.5)
                        if run_state.get(run_id, {}).get("state") in ("done", "failed"):
                            break
                    try:
                        yield f"event: status\ndata: {json.dumps(run_state[run_id])}\n\n"
                    except KeyError:
                        pass
                    try:
                        yield f"event: result\ndata: {json.dumps(run_results[run_id])}\n\n"
                    except KeyError:
                        pass
                    break

                yield f"event: metric\ndata: {json.dumps(item)}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return api

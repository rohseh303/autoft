"""FastAPI endpoints exposed via Modal: /research, /train, /run/{id}/stream, /run/{id}/result."""
import asyncio
import json
import uuid
from typing import Any

import modal

from backend.app import (
    MODELS_DIR,
    api_image,
    app,
    hf_secret,
    metrics_queue,
    model_volume,
    openai_secret,
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
    volumes={MODELS_DIR: model_volume},  # so /download can read trained adapters
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
        plan_dict = research.remote(
            req.task_description,
            [e.model_dump() for e in req.eval_examples],
            req.preferred_model,
        )
        return plan_dict

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

    @api.get("/run/{run_id}/download")
    def download_adapter(run_id: str):
        """Zip the trained LoRA adapter and stream it so the user can take the model.

        Chunked generator with an explicit Content-Length — verified to deliver a
        byte-exact zip (adapter_model.safetensors + tokenizer) through Modal's ASGI.
        """
        import io
        import os
        import zipfile
        from fastapi.responses import StreamingResponse as _SR

        # Refresh the volume view so a just-finished run's files are visible.
        try:
            model_volume.reload()
        except Exception:
            pass

        adapter_dir = f"{MODELS_DIR}/runs/{run_id}/lora"
        if not os.path.isdir(adapter_dir):
            raise HTTPException(404, "adapter not found (run not finished or unknown id)")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(adapter_dir):
                for fn in files:
                    full = os.path.join(root, fn)
                    arc = os.path.join(f"{run_id}-lora", os.path.relpath(full, adapter_dir))
                    zf.write(full, arc)
            zf.writestr(
                f"{run_id}-lora/HOW_TO_USE.txt",
                "AutoFT LoRA adapter\n"
                "===================\n"
                "Load on top of the base model with PEFT:\n\n"
                "  from peft import PeftModel\n"
                "  from transformers import AutoModelForCausalLM, AutoTokenizer\n"
                "  base = AutoModelForCausalLM.from_pretrained('unsloth/Qwen2.5-1.5B-Instruct')\n"
                f"  model = PeftModel.from_pretrained(base, './{run_id}-lora')\n"
                "  tok = AutoTokenizer.from_pretrained('unsloth/Qwen2.5-1.5B-Instruct')\n",
            )
        data = buf.getvalue()

        def _chunks(payload: bytes, size: int = 1 << 20):
            for i in range(0, len(payload), size):
                yield payload[i : i + size]

        return _SR(
            _chunks(data),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{run_id}-lora.zip"',
                "Content-Length": str(len(data)),
            },
        )

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

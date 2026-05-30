"""Research agent: spawns the OpenAI Codex CLI inside the Modal container to
inspect HuggingFace datasets and emit a validated RunPlan."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.app import api_image, app, hf_secret, openai_secret

FIXTURES_DIR = Path("/root/fixtures")
SUBPROCESS_TIMEOUT_SECONDS = 480


def _build_codex_prompt() -> str:
    return """You are an autonomous fine-tuning research agent. You have shell + Python + filesystem access in this working directory.

## Inputs (read these first)
- `./task.json` — the user's task description, eval examples, and optional preferred model.
- `./catalog.json` — 12 pre-verified (task_tags, hf_dataset, fields) entries. PREFER these when their tags match the user's task — they are known to work.
- `./schema.json` — JSON Schema for the RunPlan you must produce. Your output MUST validate against it.
- `./example_plan.json` — a complete RunPlan for the billsum legal-summarization task. Use it as a shape reference for the JSON you write.

## Your job
1. Read all the inputs above.
2. Decide which HuggingFace dataset best matches the user's task.
   - If any catalog entry's tags match the task, use that entry directly. Its fields and config are already verified.
   - If no catalog entry fits, pick a dataset off the Hub. You MUST verify it before committing — run Python like:
     ```python
     from datasets import load_dataset
     ds = load_dataset("<dataset_id>", "<config_or_None>", split="train", streaming=True)
     row = next(iter(ds))
     print(list(row.keys()))
     print({k: str(v)[:200] for k, v in row.items()})
     ```
     Do not commit to `input_field` / `output_field` column names you have not printed.
3. Pick relevant benchmarks (just human-readable labels — v1 does not run them).
4. Write a tight `prompt_template` using literal `{input}` and `{output}` placeholders.
5. Set a sane TrainingConfig for a small (0.5B–1.7B) LoRA fine-tune on a single L4 GPU.

## Training guidance (defaults — override only with a reason)
- `max_steps`: 150 for tiny models; up to 250 only if the task is genuinely complex.
- `batch_size`: 2, `gradient_accumulation_steps`: 4, `lora_r`: 16, `lora_alpha`: 16, `max_seq_length`: 2048.
- `dataset_split`: prefer `train[:2000]` or `train[:3000]` — keep it small for fast feedback.
- Prefer datasets with a clear instruction-style schema. Avoid datasets where input or output is deeply nested.
- `reasoning`: 2–3 sentences explaining your dataset + recipe choice in plain English. Shown to the user.

## Output contract
When you have decided, write your final RunPlan as JSON to `./plan.json`. It MUST validate against `./schema.json`. Do not write anything else to that file. After writing, exit.
"""


@app.function(
    image=api_image,
    secrets=[openai_secret, hf_secret],
    timeout=540,
)
def research(
    task_description: str,
    eval_examples: list[dict],
    preferred_model: str | None,
) -> dict:
    """Run the Codex-powered research agent. Returns a validated RunPlan as a dict."""
    from shared.catalog import CATALOG
    from shared.schemas import RunPlan

    # Scratch under /root (not /tmp) — Codex CLI refuses to install its helper
    # binaries when codex_home sits under a temp dir.
    scratch_root = Path("/root/codex-scratch")
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="codex-research-", dir=str(scratch_root)))
    try:
        (scratch / "task.json").write_text(
            json.dumps(
                {
                    "task_description": task_description,
                    "eval_examples": eval_examples,
                    "preferred_model": preferred_model,
                },
                indent=2,
            )
        )
        (scratch / "catalog.json").write_text(json.dumps(CATALOG, indent=2))
        (scratch / "schema.json").write_text(
            json.dumps(RunPlan.model_json_schema(), indent=2)
        )
        example_plan_path = FIXTURES_DIR / "billsum_plan.json"
        if example_plan_path.exists():
            (scratch / "example_plan.json").write_text(example_plan_path.read_text())
        prompt = _build_codex_prompt()

        # HOME=scratch so Codex's per-user config (~/.codex) lives in the
        # request's scratch dir, not in /root — keeps parallel requests from
        # stomping each other and gets cleaned up with the dir.
        env = {**os.environ, "HOME": str(scratch)}

        # Codex CLI (recent versions) reads the API key from ~/.codex/auth.json
        # for the Responses API path, not from $OPENAI_API_KEY. Pre-populate it.
        codex_home = scratch / ".codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        (codex_home / "auth.json").write_text(
            json.dumps({"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]})
        )

        try:
            proc = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--cd",
                    str(scratch),
                    # Modal container is already our sandbox; nested bubblewrap
                    # inside Modal fails on kernel-namespace permissions.
                    "--sandbox",
                    "danger-full-access",
                    "--skip-git-repo-check",
                    prompt,
                ],
                cwd=str(scratch),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as e:
            stdout_tail = (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else ""
            stderr_tail = (e.stderr or "")[-2000:] if isinstance(e.stderr, str) else ""
            raise RuntimeError(
                f"codex exceeded {SUBPROCESS_TIMEOUT_SECONDS}s timeout. "
                f"stdout_tail={stdout_tail} stderr_tail={stderr_tail}"
            ) from e

        plan_path = scratch / "plan.json"
        if not plan_path.exists():
            raise RuntimeError(
                f"codex did not write plan.json (exit={proc.returncode}). "
                f"stdout_tail={proc.stdout[-2000:]} stderr_tail={proc.stderr[-2000:]}"
            )

        plan_dict = json.loads(plan_path.read_text())
        if preferred_model:
            plan_dict["base_model"] = preferred_model
        return RunPlan.model_validate(plan_dict).model_dump()

    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@app.local_entrypoint()
def smoke_research(
    task_description: str = "summarize legal contracts",
    eval_examples_json: str = "[]",
    preferred_model: str = "",
) -> None:
    """CLI smoke test: `modal run -m backend.research_agent::smoke_research`.

    Wraps research() with CLI-friendly arg types since modal's CLI can't
    parse list[dict] annotations directly. Named uniquely so it doesn't
    collide with backend.train's `smoke` entrypoint when both modules
    register on the shared app (which happens during `modal serve`).
    """
    eval_examples = json.loads(eval_examples_json)
    plan = research.remote(
        task_description,
        eval_examples,
        preferred_model or None,
    )
    print(json.dumps(plan, indent=2))

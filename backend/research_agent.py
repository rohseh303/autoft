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

# How an exec-command maps to a timeline icon (kind). The frontend's Notebook
# only knows note | search | peek | decision, so everything collapses to those.
_PEEK_HINTS = ("load_dataset", "datasets", "peek", "list(row", "row.keys", "head")
_SEARCH_HINTS = ("list_datasets", "search", "hf_search", "huggingface_hub")


def _classify_command(command: str) -> str:
    low = command.lower()
    if any(h in low for h in _SEARCH_HINTS):
        return "search"
    if any(h in low for h in _PEEK_HINTS):
        return "peek"
    return "note"


def _coerce_command(value: object) -> str:
    # Codex reports a command as either a list of argv tokens or a string.
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def parse_codex_events(stdout: str) -> list[dict]:
    """Turn ``codex exec --json`` JSONL stdout into transparency-timeline events.

    Defensive by design: Codex's event schema shifts across releases (some emit
    ``{"msg": {"type": ...}}``, newer ones ``{"type": "item.completed", "item": ...}``),
    and the task still succeeds even if not one line parses. We extract the
    human-meaningful beats — reasoning, the agent's messages, and the shell/Python
    it ran to inspect datasets — and drop the rest. Order is preserved; each event
    is tagged ``source="research"`` so the UI can show *which* subagent spoke.
    """
    events: list[dict] = []

    def add(kind: str, text: str, detail: str | None = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        events.append(
            {"ts": None, "source": "research", "kind": kind,
             "text": text[:600], "step": None,
             "detail": (detail[:400] if detail else None)},
        )

    for line in stdout.splitlines():
        line = line.strip()
        if not line or not (line.startswith("{") or line.startswith("[")):
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue

        # Unwrap the two known envelope shapes to a single payload + type.
        payload = obj.get("msg") if isinstance(obj.get("msg"), dict) else obj
        if isinstance(obj.get("item"), dict):
            payload = obj["item"]
        etype = str(payload.get("type") or payload.get("item_type") or obj.get("type") or "")

        if "reason" in etype:
            add("note", payload.get("text") or payload.get("summary") or "")
        elif "agent_message" in etype or etype in ("message", "assistant_message"):
            add("decision", payload.get("message") or payload.get("text") or "")
        elif "command" in etype or "exec" in etype:
            # Log the command itself, never its output (keeps the timeline readable).
            # Old schema: exec_command_begin carries `command`, exec_command_end only
            # carries output -> the empty-cmd guard drops the latter. New schema: a
            # single command_execution item carries both -> we take the command.
            cmd = _coerce_command(payload.get("command") or payload.get("call") or "")
            if cmd and "end" not in etype:
                add(_classify_command(cmd), f"ran: {cmd}")
        elif "tool" in etype:
            name = payload.get("name") or payload.get("tool") or "tool"
            add("search", f"called {name}", json.dumps(payload.get("arguments"))[:300]
                if payload.get("arguments") else None)

    return events


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
5. Set a sane TrainingConfig for a Qwen3.5-2B bf16 LoRA fine-tune on a single H200 GPU.

## Training guidance (defaults — override only with a reason)
- `max_steps`: 150 for tiny models; up to 250 only if the task is genuinely complex.
- `batch_size`: 2, `gradient_accumulation_steps`: 8 (effective batch 16), `lora_r`: 16, `lora_alpha`: 32 (=2*r), `max_seq_length`: 2048. (Qwen3.5-2B bf16 LoRA per the Unsloth cookbook.)
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
    """Run the Codex-powered research agent.

    Returns ``{"plan": <validated RunPlan dict>, "events": [<reasoning trace>]}``.
    Each event is ``{ts, source, kind, text, step, detail}`` (``source="research"``)
    so the UI can replay *what the agent did* to choose the dataset and recipe.
    """
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

        def _run_codex(*, json_events: bool) -> subprocess.CompletedProcess[str]:
            # --json streams structured reasoning/tool events on stdout so we can
            # surface the agent's thinking; everything else is the proven recipe.
            # Modal container is already our sandbox; nested bubblewrap inside
            # Modal fails on kernel-namespace permissions, hence danger-full-access.
            args = ["codex", "exec"]
            if json_events:
                args.append("--json")
            args += ["--cd", str(scratch), "--sandbox", "danger-full-access",
                     "--skip-git-repo-check", prompt]
            try:
                return subprocess.run(
                    args,
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
        proc = _run_codex(json_events=True)
        events = parse_codex_events(proc.stdout) if plan_path.exists() else []

        # Safety net: if --json isn't supported on this Codex build the run aborts
        # before writing plan.json. Retry once on the plain (eventless) path so the
        # research step never regresses just to gain the reasoning trace.
        if not plan_path.exists():
            proc = _run_codex(json_events=False)
            events = []

        if not plan_path.exists():
            raise RuntimeError(
                f"codex did not write plan.json (exit={proc.returncode}). "
                f"stdout_tail={proc.stdout[-2000:]} stderr_tail={proc.stderr[-2000:]}"
            )

        plan_dict = json.loads(plan_path.read_text())
        if preferred_model:
            plan_dict["base_model"] = preferred_model
        return {
            "plan": RunPlan.model_validate(plan_dict).model_dump(),
            "events": events,
        }

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
    out = research.remote(
        task_description,
        eval_examples,
        preferred_model or None,
    )
    events = out.get("events", [])
    print(f"[research] captured {len(events)} reasoning event(s):")
    for e in events:
        print(f"  · {e['kind']:8} {e['text']}")
    print("\n[research] plan:")
    print(json.dumps(out.get("plan", out), indent=2))

"""Research agent: designs a fine-tuning RunPlan for a task — powered by `codex exec`.

Runs locally (Codex is a local CLI), like the post-training-lead loop. Codex is
given the task, the held-out eval examples, and a curated dataset catalog; it may
peek HF datasets via backend/hf_cli.py, and emits a schema-validated RunPlan via
codex's --output-schema (no JSON-parsing-from-prose, no OpenAI key).

    uv run python backend/research_agent.py --task "summarize legal contracts"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from shared.catalog import CATALOG  # noqa: E402
from shared.schemas import RunPlan  # noqa: E402


def _inline_refs(schema: dict) -> dict:
    """Inline $defs/$ref — codex's structured output wants a self-contained schema."""
    defs = schema.pop("$defs", {})

    def _walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return _walk(defs.get(node["$ref"].split("/")[-1], {}))
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(x) for x in node]
        return node

    return _walk(schema)


def _strictify(node):
    """Make a JSON schema strict-compatible (additionalProperties:false, all keys
    required) for codex/OpenAI structured output."""
    if isinstance(node, dict):
        node = {k: _strictify(v) for k, v in node.items()}
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        return node
    if isinstance(node, list):
        return [_strictify(x) for x in node]
    return node


def _runplan_schema() -> dict:
    return _strictify(_inline_refs(RunPlan.model_json_schema()))


def _build_prompt(task_description: str, eval_examples: list[dict]) -> str:
    catalog = "\n".join(
        f"- tags={e['task_tags']} dataset={e['hf_dataset']} config={e['dataset_config']} "
        f"input={e['input_field']} output={e['output_field']} ({e['description']})"
        for e in CATALOG
    )
    return f"""You are a fine-tuning research agent. Design exactly ONE complete RunPlan \
(returned as JSON matching the provided output schema) to fine-tune the Qwen3.5-2B \
model for the user's task.

Do NOT run training or any `modal run` command — your only job is to OUTPUT the RunPlan.

USER TASK:
{task_description}

EVAL EXAMPLES (what the fine-tuned model will be judged on):
{json.dumps(eval_examples, indent=2)}

PREFERRED CATALOG (pre-verified datasets — use one whose tags match the task):
{catalog}

Rules:
- Pick the best HuggingFace dataset and map its columns to input_field / output_field.
- You MAY verify a NON-catalog dataset's columns first by running:
      uv run python backend/hf_cli.py peek <dataset> [--config <config>]
  Catalog datasets are already verified — prefer them.
- prompt_template MUST contain the literal placeholders {{input}} and {{output}}.
- base_model is "Qwen3.5-2B"; keep dataset_split small (e.g. "train[:2000]").
- TrainingConfig for Qwen3.5-2B bf16 LoRA: max_steps 150, learning_rate 2e-4,
  batch_size 2, gradient_accumulation_steps 8 (effective batch 16), lora_r 16,
  lora_alpha 32, max_seq_length 2048, warmup_steps 10, seed 42.
- reasoning: 2-3 sentences explaining the dataset + recipe choice."""


def research(
    task_description: str,
    eval_examples: list[dict] | None = None,
    preferred_model: str | None = None,
) -> dict:
    """Design a RunPlan for the task via a fresh `codex exec` session.

    Returns a validated RunPlan as a dict.
    """
    eval_examples = eval_examples or []
    prompt = _build_prompt(task_description, eval_examples)

    with tempfile.TemporaryDirectory() as tmp:
        schema_path = Path(tmp) / "runplan.schema.json"
        out_path = Path(tmp) / "plan.json"
        schema_path.write_text(json.dumps(_runplan_schema()))
        cmd = [
            "codex", "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--output-schema", str(schema_path),
            "--output-last-message", str(out_path),
            "-",  # read the prompt from stdin
        ]
        # cwd=repo so Codex can `uv run python backend/hf_cli.py peek ...`.
        subprocess.run(
            cmd, input=prompt, text=True, capture_output=True,
            timeout=600, check=True, cwd=str(_REPO),
        )
        plan_dict = json.loads(out_path.read_text() or "{}")

    if preferred_model:
        plan_dict["base_model"] = preferred_model
    return RunPlan.model_validate(plan_dict).model_dump()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Design a RunPlan for a task (via Codex).")
    parser.add_argument("--task", required=True, help="Free-text capability you want.")
    parser.add_argument("--evals", default="[]", help='JSON list of {"input","expected_output"}.')
    parser.add_argument("--out", default=None, help="Optional path to also write the RunPlan JSON.")
    args = parser.parse_args(argv)

    plan = research(args.task, json.loads(args.evals))
    text = json.dumps(plan, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

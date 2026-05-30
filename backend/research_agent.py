"""Research agent: takes a user task, picks dataset/recipe, returns RunPlan."""
from __future__ import annotations

import json

import modal

from backend.app import api_image, app, hf_secret, openai_secret

OPENAI_MODEL = "gpt-4.1-mini"


# HF tools live in shared/hf_tools.py so the local CLI (backend/hf_cli.py) can
# reuse them without importing Modal. Aliased to keep the tool dispatch stable.
from shared.hf_tools import hf_peek as _hf_peek, hf_search as _hf_search


# ----- Tool schemas exposed to the model -----

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_hf_datasets",
            "description": "Search HuggingFace Hub for datasets matching a free-text query. Returns dataset IDs sorted by downloads.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text search query"},
                    "task_filter": {
                        "type": "string",
                        "description": "Optional HF task category filter (e.g. 'summarization', 'text-classification')",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "peek_dataset",
            "description": "Fetch column names + 3 sample rows from a dataset to verify schema before committing to it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string"},
                    "config": {"type": "string"},
                    "split": {"type": "string", "default": "train"},
                },
                "required": ["dataset"],
            },
        },
    },
]


def _build_system_prompt() -> str:
    from shared.catalog import CATALOG

    catalog_lines = []
    for e in CATALOG:
        catalog_lines.append(
            f"- tags={e['task_tags']} dataset={e['hf_dataset']} "
            f"config={e['dataset_config']} input={e['input_field']} output={e['output_field']} "
            f"({e['description']})"
        )
    catalog_block = "\n".join(catalog_lines)
    return f"""You are an autonomous fine-tuning research agent. The user describes a capability they want a small open-source LLM to have. You design a complete training recipe.

Your job:
1. Decide which HuggingFace dataset best matches the task.
2. Decide which columns map to instruction (input_field) and target (output_field).
3. Pick relevant benchmarks (just names; v1 doesn't run them).
4. Write a tight prompt_template using {{input}} and {{output}}.
5. Set sane TrainingConfig for a small (0.5B-1.7B) model with LoRA on a single L4 GPU.

You have tools: `search_hf_datasets` and `peek_dataset`. ALWAYS `peek_dataset` before committing to a dataset that is not in the preferred catalog below — to confirm the column names you'll use actually exist.

PREFERRED CATALOG (use these whenever a tag matches the user's task — they are pre-verified):
{catalog_block}

Guidance:
- max_steps default 150 for tiny models; raise to 250 only if task is complex.
- batch_size 2, gradient_accumulation_steps 4, lora_r 16, lora_alpha 16.
- Choose dataset_split like "train[:2000]" or "train[:3000]" — keep it small for fast feedback.
- Prefer datasets with a clear instruction-style schema. Avoid datasets where input or output is deeply nested.
- prompt_template must contain literally "{{input}}" and "{{output}}" placeholders.
- reasoning: 2-3 sentences explaining your dataset + recipe choice in plain English.

When you have decided, return the final RunPlan as JSON matching the schema.
"""


@app.function(image=api_image, secrets=[openai_secret, hf_secret], timeout=300)
def research(task_description: str, eval_examples: list[dict], preferred_model: str | None) -> dict:
    """Run the research agent. Returns a validated RunPlan as a dict."""
    from openai import OpenAI
    from shared.schemas import RunPlan

    client = OpenAI()
    schema = RunPlan.model_json_schema()
    schema = _inline_refs(schema)

    user_msg = {
        "role": "user",
        "content": json.dumps({
            "task_description": task_description,
            "eval_examples": eval_examples,
            "preferred_model": preferred_model,
        }),
    }
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        user_msg,
    ]

    for _ in range(6):  # tool-use loop, max 6 rounds
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "RunPlan", "schema": schema, "strict": False},
            },
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append({"role": "assistant", "tool_calls": [tc.model_dump() for tc in msg.tool_calls], "content": msg.content or ""})
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                if name == "search_hf_datasets":
                    result = _hf_search(args.get("query", ""), args.get("task_filter"))
                elif name == "peek_dataset":
                    result = _hf_peek(args.get("dataset", ""), args.get("config"), args.get("split", "train"))
                else:
                    result = {"error": f"unknown tool {name}"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)[:4000],
                })
            continue
        # No tool calls — model should have produced the final JSON.
        content = msg.content or "{}"
        plan_dict = json.loads(content)
        # Apply preferred model override if user specified one
        if preferred_model:
            plan_dict["base_model"] = preferred_model
        plan = RunPlan.model_validate(plan_dict)
        return plan.model_dump()

    raise RuntimeError("Research agent exceeded tool-use loop budget")


def _inline_refs(schema: dict) -> dict:
    """Pydantic emits $defs/$ref which some OpenAI clients reject when strict=true. Inline them."""
    defs = schema.pop("$defs", {})

    def _walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return _walk(defs.get(ref_name, {}))
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(x) for x in node]
        return node

    return _walk(schema)

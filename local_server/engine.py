"""Dynamic, task-varying demo engine for the local AutoFT backend.

No GPU, no Modal, no Codex — pure Python. Given a task description it matches
the real shared/catalog.py, derives a plausible RunPlan, and synthesizes a
research trace, a training loss curve, live model re-samples (gibberish ->
coherent), a before/after, and a scorecard. Everything keys off the task +
run_id so each run is different (the "dynamic and variable" requirement).
"""
from __future__ import annotations

import hashlib
import math
import random
import re

from shared.catalog import CATALOG


# ---------------------------------------------------------------------------
# catalog matching — the same idea the real research agent uses
# ---------------------------------------------------------------------------

def _score_entry(task: str, entry: dict) -> int:
    t = task.lower()
    score = 0
    for tag in entry["task_tags"]:
        # whole-word-ish match on each tag
        if re.search(rf"\b{re.escape(tag.lower())}\b", t) or tag.lower() in t:
            score += 2
        # partial: "summar" matches "summarize"/"summarization"
        stem = tag.lower()[:5]
        if len(stem) >= 4 and stem in t:
            score += 1
    return score


def rank_catalog(task: str) -> list[tuple[int, dict]]:
    ranked = sorted(
        ((_score_entry(task, e), e) for e in CATALOG),
        key=lambda x: x[0],
        reverse=True,
    )
    return ranked


def best_entry(task: str) -> tuple[dict, bool]:
    """Return (entry, matched). matched=False means we fell back to a default."""
    ranked = rank_catalog(task)
    top_score, top = ranked[0]
    if top_score <= 0:
        # no tag hit — fall back to instruction-following
        for e in CATALOG:
            if "instruction-following" in e["task_tags"]:
                return e, False
        return CATALOG[0], False
    return top, True


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _seed(*parts: str) -> random.Random:
    h = hashlib.sha256("::".join(parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def build_plan(task: str, eval_examples: list[dict], preferred_model: str | None) -> dict:
    entry, matched = best_entry(task)
    rng = _seed(task, entry["hf_dataset"])

    ds_name = entry["hf_dataset"].split("/")[-1]
    category = _category(entry, task)

    # a little variability in the recipe, still within schema bounds
    max_steps = rng.choice([120, 150, 150, 180, 200])
    lora_r = rng.choice([8, 16, 16, 32])

    summary = task.strip().rstrip(".")
    summary = summary[0].upper() + summary[1:] if summary else "Fine-tune a small model"

    benchmarks = _benchmarks(category, ds_name)

    reasoning = (
        f"Matched your task to {entry['hf_dataset']} — {entry['description']} "
        f"The {entry['input_field']} → {entry['output_field']} column mapping is clean, "
        f"and instruction-style formatting works well for a small model. "
        f"{max_steps} LoRA steps (r={lora_r}) on an L4 is enough to imprint the "
        f"{'style' if category == 'summary' else 'behavior'} without overfitting a sub-2B model."
    )
    if not matched:
        reasoning = (
            "No catalog tag matched cleanly, so I defaulted to a general "
            f"instruction-following recipe on {entry['hf_dataset']}. " + reasoning.split(" — ", 1)[-1]
        )

    return {
        "task_summary": summary,
        "base_model": preferred_model or "Qwen3.5-2B",
        "hf_dataset": entry["hf_dataset"],
        "dataset_config": entry["dataset_config"],
        "dataset_split": entry["dataset_split"],
        "input_field": entry["input_field"],
        "output_field": entry["output_field"],
        "prompt_template": _prompt_template(category),
        "benchmarks": benchmarks,
        "training": {
            "max_steps": max_steps,
            "learning_rate": 2e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,  # eff. batch 2*8=16, merged default
            "lora_r": lora_r,
            "lora_alpha": lora_r * 2,          # 2*r per Unsloth LoRA guide
            "max_seq_length": 1024 if category in ("sql", "math") else 2048,
            "warmup_steps": 10,
            "seed": 42,
        },
        "reasoning": reasoning,
        "_category": category,  # internal hint for the trainer; stripped before validation
    }


def _category(entry: dict, task: str) -> str:
    tags = set(entry["task_tags"])
    if {"sql", "text-to-sql", "database"} & tags:
        return "sql"
    if {"code", "python", "programming", "coding"} & tags:
        return "code"
    if {"math", "reasoning", "arithmetic", "word-problems"} & tags:
        return "math"
    if {"summarization"} & tags:
        return "summary"
    if {"json", "structured-output", "extraction"} & tags:
        return "json"
    return "instruct"


def _prompt_template(category: str) -> str:
    if category == "sql":
        return "### Question:\n{input}\n\n### SQL:\n{output}"
    if category == "summary":
        return "### Document:\n{input}\n\n### Summary:\n{output}"
    return "### Instruction:\n{input}\n\n### Response:\n{output}"


def _benchmarks(category: str, ds_name: str) -> list[str]:
    if category == "sql":
        return ["execution-accuracy", "exact-match"]
    if category == "summary":
        return [f"{ds_name}-rougeL", "bertscore"]
    if category == "math":
        return ["gsm8k-accuracy", "exact-match"]
    if category == "code":
        return ["humaneval-passrate", "exact-match"]
    return [f"{ds_name}-eval", "win-rate"]


# ---------------------------------------------------------------------------
# research thought trace (streamed)
# ---------------------------------------------------------------------------

def research_thoughts(task: str, plan: dict) -> list[dict]:
    ranked = rank_catalog(task)
    candidates = [e["hf_dataset"] for s, e in ranked if s > 0][:3]
    if not candidates:
        candidates = [plan["hf_dataset"]]
    kw = _keywords(task)

    thoughts = [
        {"kind": "note", "text": f'Reading your task: "{task.strip()}"'},
        {"kind": "search", "text": "Searching the HuggingFace Hub for matching datasets",
         "detail": f'query: "{kw}"'},
        {"kind": "note", "text": f"Top candidates: {' · '.join(candidates)}"},
        {"kind": "peek", "text": f"Peeking {plan['hf_dataset']} to verify the schema",
         "detail": f"columns: {plan['input_field']}, {plan['output_field']}  ✓"},
        {"kind": "decision", "text": f"Choosing {plan['hf_dataset']}",
         "detail": plan["reasoning"].split(". ")[0] + "."},
    ]
    return thoughts


def _keywords(task: str) -> str:
    stop = {"a", "an", "the", "i", "want", "model", "that", "can", "to", "into",
            "for", "of", "and", "my", "me", "is", "with", "from", "turn", "turns"}
    words = [w for w in re.findall(r"[a-zA-Z]+", task.lower()) if w not in stop and len(w) > 2]
    return " ".join(words[:4]) or task.strip()[:32]


# ---------------------------------------------------------------------------
# training: loss curve, learning rate, grad norm, live samples
# ---------------------------------------------------------------------------

def loss_at(step: int, max_steps: int, rng: random.Random, scale: float) -> float:
    x = step / max_steps
    base = 0.8 + scale * math.exp(-3.0 * x)
    noise = (math.sin(step * 1.7) + math.sin(step * 0.6)) * 0.05 * (1 - x)
    jitter = (rng.random() - 0.5) * 0.06 * (1 - x)
    return max(0.55, base + noise + jitter)


def lr_at(step: int, plan: dict) -> float:
    w = plan["training"]["warmup_steps"]
    m = plan["training"]["max_steps"]
    peak = plan["training"]["learning_rate"]
    if step <= w:
        return peak * (step / max(1, w))
    return peak * (1 - (step - w) / max(1, m - w))


def grad_at(step: int, rng: random.Random) -> float:
    return max(0.05, 1.4 * math.exp(-step * 0.02) + abs(math.sin(step)) * 0.15 + rng.random() * 0.05)


# task-specific "model is learning" progressions (gibberish -> coherent)
_SAMPLE_STAGES: dict[str, list[str]] = {
    "sql": [
        "the the SELECT from from where the",
        "SELECT name the FROM users WHERE",
        "SELECT name FROM employees WHERE salary",
        "SELECT name FROM employees WHERE salary > 100000;",
    ],
    "summary": [
        "the the bill bill of to the and the",
        "This bill the Secretary to of guidelines",
        "This bill requires the Secretary to publish guidelines",
        "Requires HHS to publish AI clinical-decision guidelines by 2027.",
    ],
    "code": [
        "def def the the return return x x",
        "def add(a, b): return the sum of",
        "def add(a, b):\n    return a + b  # the",
        "def add(a, b):\n    return a + b",
    ],
    "math": [
        "the answer the is the number and the",
        "First we add the and then we",
        "She has 3 boxes of 12 = and then",
        "She has 3 × 12 = 36 apples. The answer is 36.",
    ],
    "json": [
        "{ { the the : : , , the }",
        '{ "name": the, "type": the value }',
        '{ "name": "AutoFT", "type": the }',
        '{ "name": "AutoFT", "type": "platform" }',
    ],
    "instruct": [
        "the the and the to the response the",
        "Sure, here the is the answer to the",
        "Here is a clear, concise answer to your",
        "Here is a clear, concise answer to your request.",
    ],
}


def sample_at(step: int, max_steps: int, category: str) -> str:
    stages = _SAMPLE_STAGES.get(category, _SAMPLE_STAGES["instruct"])
    x = step / max_steps
    idx = min(len(stages) - 1, int(x * len(stages) + 1e-9))
    return stages[idx]


# ---------------------------------------------------------------------------
# result: before/after comparisons + scorecard
# ---------------------------------------------------------------------------

_BASE_BABBLE = {
    "sql": "Sure! To get that data you would look at the table and find the rows. SELECT * maybe? It depends on the columns and the database and what you want exactly.",
    "summary": "Sure! Here is a summary. The document is about a document. It talks about several things and people and dates and more details about the topic in general terms.",
    "code": "Sure! You can write a function for that. It takes some inputs and returns an output. You might use a loop or a return statement depending on what you need.",
    "math": "Let me think about this problem. There are some numbers involved and you need to do some operations. The answer is probably one of the numbers mentioned.",
    "json": "Sure! Here is some JSON. It has keys and values. You can put the fields you want inside the braces with the right structure.",
    "instruct": "Sure! Here is a response. It addresses the request in a general way and provides some information that may or may not be exactly what you asked for.",
}

_FT_GOOD = {
    "sql": "SELECT name FROM employees WHERE salary > 100000;",
    "summary": "Requires HHS to publish AI clinical-decision guidelines by 2027.",
    "code": "def add(a, b):\n    return a + b",
    "math": "She has 3 × 12 = 36 apples. The answer is 36.",
    "json": '{ "name": "AutoFT", "type": "platform" }',
    "instruct": "Here is a clear, concise answer to your request.",
}


_CRITIQUE = {
    "sql": "Good — emits valid SQL; could tighten alias usage and always terminate with a semicolon.",
    "summary": "Faithful and concise; lead with the required action and drop the preamble.",
    "code": "Correct and idiomatic; add a docstring and a type hint for the next run.",
    "math": "Right answer with shown work; state the final number on its own line.",
    "json": "Valid JSON in the target shape; enforce the schema's required keys explicitly.",
    "instruct": "Clear and on-task; trim hedging and answer the question directly.",
}


def build_result(run_id: str, plan: dict, req: dict, category: str, final_loss: float) -> dict:
    rng = _seed(run_id, plan["hf_dataset"])
    examples = req.get("eval_examples") or [{
        "input": _default_eval_input(category),
        "expected_output": None,
    }]

    def _js() -> float:  # per-example judge score: high (the fine-tune works), with spread
        return round(min(9.6, 7.6 + rng.random() * 1.8), 1)

    comparisons = [
        {
            "input": ex["input"],
            "base_output": _BASE_BABBLE.get(category, _BASE_BABBLE["instruct"]),
            "finetuned_output": _FT_GOOD.get(category, _FT_GOOD["instruct"]),
            "expected_output": ex.get("expected_output"),
            "judge_score": _js(),
            "judge_critique": _CRITIQUE.get(category, _CRITIQUE["instruct"]),
        }
        for ex in examples
    ]
    judge_score = round(sum(c["judge_score"] for c in comparisons) / len(comparisons), 2)
    eval_loss = round(final_loss + 0.08 + rng.random() * 0.1, 4)
    objective = judge_score  # judge is primary objective per merged backend
    return {
        "run_id": run_id,
        "plan": plan,
        "final_loss": final_loss,
        "eval_loss": eval_loss,
        "judge_score": judge_score,
        "objective": objective,
        "comparisons": comparisons,
        "scorecard": _scorecard(plan, category, rng, judge_score),
    }


def _default_eval_input(category: str) -> str:
    return {
        "sql": "Which employees earn more than 100000?",
        "summary": "A bill to require the Secretary of Health to publish guidelines on AI in clinical decision-making by 2027.",
        "code": "Write a function that adds two numbers.",
        "math": "She has 3 boxes with 12 apples each. How many apples does she have?",
        "json": "Give me a JSON object describing the AutoFT platform.",
        "instruct": "Explain what fine-tuning a language model means.",
    }.get(category, "Explain what fine-tuning means.")


def _scorecard(plan: dict, category: str, rng: random.Random, judge_score: float | None = None) -> dict:
    def between(lo, hi):
        return lo + (hi - lo) * rng.random()

    total = 10
    # tie the blind-preference framing to the real judge score when present
    wins = round(judge_score) if judge_score is not None else round(between(7, 9.4))
    wins = max(0, min(total, wins))
    measures = []
    for b in plan["benchmarks"][:2]:
        name = b.lower()
        if "rouge" in name:
            measures.append({"name": "ROUGE-L", "base": round(between(0.11, 0.21), 2),
                             "finetuned": round(between(0.36, 0.49), 2), "scale": 0.6, "unit": ""})
        elif "exact" in name or "match" in name or "accuracy" in name or "passrate" in name:
            measures.append({"name": b.replace("-", " ").title(), "base": round(between(6, 19)),
                             "finetuned": round(between(42, 70)), "scale": 100, "unit": "%"})
        else:
            measures.append({"name": b.replace("-", " ").title(), "base": round(between(22, 40)),
                             "finetuned": round(between(58, 84)), "scale": 100, "unit": "%"})
    measures.append({"name": "Format adherence", "base": round(between(28, 48)),
                     "finetuned": round(between(90, 99)), "scale": 100, "unit": "%"})
    measures.append({"name": "Style match", "base": round(between(24, 44)),
                     "finetuned": round(between(70, 90)), "scale": 100, "unit": "%"})

    verdict = (
        "A blind judge picked your model almost every time." if wins >= 9
        else "A blind judge clearly preferred your model." if wins >= 8
        else "A blind judge leaned toward your model more often than not."
    )
    return {
        "judge_wins": wins,
        "judge_total": total,
        "verdict": verdict,
        "measures": measures,
        "simulated": True,
    }


def loss_scale(category: str) -> float:
    # different tasks start at different loss heights -> visibly different curves
    return {"sql": 2.0, "math": 2.4, "code": 2.1, "summary": 2.6, "json": 1.8}.get(category, 2.2)

"""LLM-as-judge: scores fine-tuned generations against the task, elementwise.

Runs on the CPU api_image with the OpenAI secret (same pattern as the research
agent), so the GPU image stays lean. The judge is the trial's *reward signal*:
it turns the model's actual outputs into a 0-10 score plus a one-line critique
the post-training lead reads to decide what to change next. One OpenAI call per
eval example keeps each judgment independent.
"""
from __future__ import annotations

import json

from backend.app import api_image, app, openai_secret

JUDGE_MODEL = "gpt-5.4-mini"

_JUDGE_SYSTEM = (
    "You are a strict evaluator of a fine-tuned small language model. You are "
    "given a TASK, an INPUT, the model's OUTPUT, and optionally a REFERENCE "
    "answer. Rate how well OUTPUT accomplishes the task on a 0-10 scale "
    "(0 = useless/garbled/empty, 5 = roughly right but flawed, 10 = excellent "
    "and faithful). Then write ONE sentence of concrete, actionable critique "
    "aimed at what the next training run should improve (format, length, "
    "faithfulness, instruction-following, repetition, etc.). "
    'Respond ONLY as JSON: {"score": <number 0-10>, "critique": "<one sentence>"}.'
)


@app.function(image=api_image, secrets=[openai_secret], timeout=300)
def judge_outputs(task_summary: str, comparisons: list[dict]) -> dict:
    """Score each fine-tuned output 0-10 with a one-line critique (one call each).

    Returns {"mean_score": float|None, "per_example": [{"score", "critique"}, ...]}.
    mean_score is None only when every call failed, so the caller falls back to
    eval_loss for the objective instead of recording a fake 0.
    """
    # `openai` is imported lazily: this module is imported at load time by
    # train.py to register the function — including inside train_image, which
    # does not ship openai. Only the api_image container running this body has it.
    from openai import OpenAI

    if not comparisons:
        return {"mean_score": 0.0, "per_example": []}

    client = OpenAI()
    per_example: list[dict] = []
    any_ok = False
    for comp in comparisons:
        payload = json.dumps({
            "task": task_summary,
            "input": str(comp.get("input", ""))[:2000],
            "output": str(comp.get("finetuned_output", ""))[:2000],
            "reference": str(comp.get("expected_output") or "")[:2000],
        })
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": payload},
                ],
            )
            parsed = json.loads(resp.choices[0].message.content or "{}")
            score = max(0.0, min(10.0, float(parsed.get("score", 0))))
            critique = str(parsed.get("critique", ""))[:300]
            any_ok = True
        except Exception as e:  # never let a judge hiccup kill the trial
            score, critique = 0.0, f"judge error: {e}"
        per_example.append({"score": score, "critique": critique})

    if not any_ok:
        return {"mean_score": None, "per_example": per_example, "error": "all judge calls failed"}
    mean = round(sum(p["score"] for p in per_example) / len(per_example), 2)
    return {"mean_score": mean, "per_example": per_example}

"""Assemble the post-training lead's briefing packet (kickoff.md).

Combines: the task, the held-out eval set, the seed recipe, and a LIVE peek at
the HuggingFace dataset — so Codex sees the real columns + rows before trial 1.
Run by scripts/optimize.sh; safe to run standalone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.hf_tools import hf_peek  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _read_json(name: str, default):
    p = ROOT / name
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return default
    return default


def main() -> None:
    task = ""
    task_file = ROOT / "task.txt"
    if task_file.exists():
        task = task_file.read_text().strip()

    plan = _read_json("plan.json", {})
    evals = _read_json("eval.json", [])

    dataset = plan.get("hf_dataset", "")
    config = plan.get("dataset_config")
    split_base = (plan.get("dataset_split") or "train").split("[")[0] or "train"
    peek = hf_peek(dataset, config, split_base) if dataset else {"note": "no dataset in plan.json yet"}

    lines: list[str] = []
    lines.append("# Kickoff — post-training lead briefing\n")
    lines.append(
        "You are the **post-training lead**. `AGENTS.md` is your operating manual. "
        "This file is the brief for *this* run.\n"
    )

    lines.append("## The task\n")
    lines.append((task or "(no task.txt provided)") + "\n")

    lines.append("## Held-out eval set (what you are optimizing for)\n")
    lines.append(
        "These are the examples the LLM judge scores after every trial. "
        "You may NOT edit them — they are the test.\n"
    )
    lines.append("```json")
    lines.append(json.dumps(evals, indent=2))
    lines.append("```\n")

    lines.append("## Seed recipe — `plan.json` (edit this between trials)\n")
    lines.append("```json")
    lines.append(json.dumps(plan, indent=2))
    lines.append("```\n")

    lines.append(f"## Live peek at `{dataset or '(none)'}` — the actual training data\n")
    lines.append(
        "Columns and 3 sample rows straight from HuggingFace. Confirm "
        "`input_field` / `output_field` map to real columns and that the "
        "`prompt_template` fits this data before you train.\n"
    )
    lines.append("```json")
    lines.append(json.dumps(peek, indent=2))
    lines.append("```\n")

    lines.append("## Begin\n")
    lines.append(
        "1. Inspect the data above (peek more splits/datasets with "
        "`uv run python backend/hf_cli.py peek <dataset>` if needed).\n"
        "2. Run your baseline trial: `uv run modal run backend/train.py::trial`.\n"
        "3. Read `result.json` (per-example base-vs-finetuned outputs + judge critiques) "
        "and `trials.jsonl` (history).\n"
        "4. Change ONE thing in `plan.json`, rerun, keep the best. Stop per `AGENTS.md`, "
        "then write the winning recipe to `plan.json` and summarize what you tried and "
        "why the winner won.\n"
    )

    out = ROOT / "kickoff.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}  (task={'set' if task else 'empty'}, evals={len(evals)}, dataset={dataset or 'none'})")


if __name__ == "__main__":
    main()

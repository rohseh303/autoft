"""Local CLI for the post-training lead to look at the actual HuggingFace data.

No Modal needed — these are plain HTTP / Hub calls, so they're fast to run in
the optimization loop.

    uv run python backend/hf_cli.py peek FiscalNote/billsum --split train
    uv run python backend/hf_cli.py peek abisee/cnn_dailymail --config 3.0.0
    uv run python backend/hf_cli.py search "legal summarization" --task summarization
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

# Repo root on sys.path so `shared` imports when run as a script: the project
# isn't installed into the venv, so script execution puts only backend/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.hf_tools import hf_peek, hf_search  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect HuggingFace datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    peek = sub.add_parser("peek", help="Show column schema + 3 sample rows for a dataset.")
    peek.add_argument("dataset")
    peek.add_argument("--config", default=None)
    peek.add_argument("--split", default="train")

    search = sub.add_parser("search", help="Search the Hub for datasets by free text.")
    search.add_argument("query")
    search.add_argument("--task", default=None, help="Optional HF task-category filter.")

    args = parser.parse_args(argv)
    if args.cmd == "peek":
        result = hf_peek(args.dataset, args.config, args.split)
    else:
        result = hf_search(args.query, args.task)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

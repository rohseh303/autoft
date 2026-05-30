"""Local CLI for the post-training lead to look at the actual HuggingFace data.

No Modal needed — these are plain HTTP / Hub calls, so they're fast to run in
the optimization loop.

    uv run python backend/hf_cli.py peek billsum --split train
    uv run python backend/hf_cli.py peek cnn_dailymail --config 3.0.0
    uv run python backend/hf_cli.py search "legal summarization" --task summarization
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `shared` importable when run as a plain script (not just via the package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.hf_tools import hf_peek, hf_search  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HuggingFace datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("peek", help="Show column schema + 3 sample rows for a dataset.")
    pk.add_argument("dataset")
    pk.add_argument("--config", default=None)
    pk.add_argument("--split", default="train")

    se = sub.add_parser("search", help="Search the Hub for datasets by free text.")
    se.add_argument("query")
    se.add_argument("--task", default=None, help="Optional HF task-category filter.")

    args = parser.parse_args()
    if args.cmd == "peek":
        print(json.dumps(hf_peek(args.dataset, args.config, args.split), indent=2))
    elif args.cmd == "search":
        print(json.dumps(hf_search(args.query, args.task), indent=2))


if __name__ == "__main__":
    main()

"""HuggingFace dataset inspection helpers — column schema, sample rows, search.

Kept free of any Modal import so both the research agent (inside Modal) and the
local CLI (backend/hf_cli.py, used by the post-training lead) can share them.
"""
from __future__ import annotations

import requests
from huggingface_hub import HfApi

__all__ = ["hf_peek", "hf_search"]

_DATASETS_SERVER = "https://datasets-server.huggingface.co"


def hf_search(query: str, task_filter: str | None = None, limit: int = 8) -> list[dict]:
    """Search the Hub for datasets by free text, returning most-downloaded first."""
    kwargs: dict[str, object] = {"search": query, "sort": "downloads", "limit": limit}
    if task_filter:
        kwargs["task_categories"] = task_filter
    try:
        return [
            {
                "id": dataset.id,
                "downloads": getattr(dataset, "downloads", 0) or 0,
                "tags": (getattr(dataset, "tags", []) or [])[:8],
            }
            for dataset in HfApi().list_datasets(**kwargs)
        ]
    except Exception as e:  # discovery is best-effort: surface the reason, don't raise
        return [{"error": str(e)}]


def hf_peek(dataset: str, config: str | None = None, split: str = "train") -> dict:
    """Column schema + a few sample rows via the HF datasets-server.

    Uses /splits to discover configs+splits (so it handles renamed and
    multi-config datasets), and surfaces datasets-server errors (renamed,
    gated, not-found) instead of returning a silently-empty result.
    """

    def _get(path: str, params: dict) -> dict:
        try:
            return requests.get(f"{_DATASETS_SERVER}/{path}", params=params, timeout=15).json()
        except requests.RequestException as e:
            return {"error": f"{path} request failed: {e}"}

    splits = _get("splits", {"dataset": dataset})
    if splits.get("error"):
        return {"error": splits["error"], "dataset": dataset}

    entries = splits.get("splits", []) or []
    configs = sorted({e.get("config") for e in entries if e.get("config")})
    chosen_config = config or (configs[0] if configs else "default")
    available = [e.get("split") for e in entries if e.get("config") == chosen_config]
    chosen_split = split if (split in available or not available) else available[0]

    rows = _get(
        "rows",
        {"dataset": dataset, "config": chosen_config, "split": chosen_split, "offset": 0, "length": 3},
    )
    if rows.get("error"):
        return {
            "error": rows["error"],
            "dataset": dataset,
            "configs": configs,
            "config_used": chosen_config,
        }

    columns = [feature.get("name") for feature in rows.get("features", [])]
    samples = [
        {key: str(value)[:200] for key, value in row.get("row", {}).items()}
        for row in rows.get("rows", [])
    ]
    return {
        "dataset": dataset,
        "configs": configs,
        "config_used": chosen_config,
        "split_used": chosen_split,
        "columns": columns,
        "sample_rows": samples,
    }

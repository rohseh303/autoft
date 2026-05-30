"""HuggingFace dataset inspection helpers — column schema, sample rows, search.

Kept free of any Modal import so both the research agent (inside Modal) and the
local CLI (backend/hf_cli.py, used by the post-training lead) can share them.
"""
from __future__ import annotations


def hf_search(query: str, task_filter: str | None = None, limit: int = 8) -> list[dict]:
    from huggingface_hub import DatasetFilter, HfApi

    api = HfApi()
    kwargs = {"search": query, "limit": limit, "sort": "downloads", "direction": -1}
    if task_filter:
        try:
            kwargs["filter"] = DatasetFilter(task_categories=task_filter)
        except Exception:
            pass
    results = []
    try:
        for d in api.list_datasets(**kwargs):
            results.append({
                "id": d.id,
                "downloads": getattr(d, "downloads", 0),
                "tags": (getattr(d, "tags", []) or [])[:8],
            })
    except Exception as e:
        return [{"error": str(e)}]
    return results


def hf_peek(dataset: str, config: str | None = None, split: str = "train") -> dict:
    """Column schema + a few sample rows via the HF datasets-server.

    Uses /splits to discover configs+splits (so it handles renamed and
    multi-config datasets), and surfaces datasets-server errors (renamed,
    gated, not-found) instead of returning a silently-empty result.
    """
    import requests

    base = "https://datasets-server.huggingface.co"

    def _get(path: str, params: dict) -> dict:
        try:
            return requests.get(f"{base}/{path}", params=params, timeout=15).json()
        except Exception as e:
            return {"error": f"{path} request failed: {e}"}

    splits = _get("splits", {"dataset": dataset})
    if splits.get("error"):
        return {"error": splits["error"], "dataset": dataset}

    entries = splits.get("splits", []) or []
    configs = sorted({e.get("config") for e in entries if e.get("config")})
    chosen_config = config or (configs[0] if configs else "default")
    avail = [e.get("split") for e in entries if e.get("config") == chosen_config]
    chosen_split = split if (split in avail or not avail) else avail[0]

    rows = _get("rows", {
        "dataset": dataset, "config": chosen_config,
        "split": chosen_split, "offset": 0, "length": 3,
    })
    if rows.get("error"):
        return {"error": rows["error"], "dataset": dataset,
                "configs": configs, "config_used": chosen_config}

    columns = [f.get("name") for f in rows.get("features", [])]
    samples = [{k: str(v)[:200] for k, v in r.get("row", {}).items()} for r in rows.get("rows", [])]
    return {
        "dataset": dataset, "configs": configs, "config_used": chosen_config,
        "split_used": chosen_split, "columns": columns, "sample_rows": samples,
    }

"""Central Modal App definition — images, volumes, queues, secrets used everywhere."""
from __future__ import annotations

import os

import modal

APP_NAME = "autoft"

app = modal.App(APP_NAME)

# Image for the research agent + API endpoints (no GPU deps needed)
api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pydantic>=2.0",
        "openai>=1.50.0",
        "huggingface_hub>=0.24.0",
        "requests>=2.31.0",
        "fastapi[standard]",
    )
    .add_local_python_source("shared", "backend")
)

# Image for training (heavy GPU stack). Unsloth installs torch + xformers compatible builds.
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # Unpinned: let pip resolve a current, self-consistent stack around the
        # latest unsloth. The old Dec-2024 pins broke (unsloth_zoo lacked
        # merge_and_overwrite_lora); unsloth now drives compatible torch/trl/etc.
        "unsloth",
        "unsloth_zoo",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
        "pydantic>=2.0",
        "huggingface_hub",
    )
    .add_local_python_source("shared", "backend")
)

# Persistent state ---------------------------------------------------------
# Volume holds: HF dataset cache, model weights cache, LoRA checkpoints per run.
model_volume = modal.Volume.from_name("autoft-models", create_if_missing=True)
MODELS_DIR = "/models"

# Cross-function pub/sub for streaming step metrics from train -> SSE endpoint.
# One queue partitioned by run_id (Modal Queues support partitions natively).
metrics_queue = modal.Queue.from_name("autoft-metrics", create_if_missing=True)
# Top-level run-state Dict (RunStatus serialized as dict).
run_state = modal.Dict.from_name("autoft-run-state", create_if_missing=True)
# Final results (RunResult serialized as dict).
run_results = modal.Dict.from_name("autoft-run-results", create_if_missing=True)

# Secrets ------------------------------------------------------------------
# Secret names are overridable so the app can run against whatever the active
# Modal workspace has provisioned (defaults match the README's setup steps).
openai_secret = modal.Secret.from_name(os.environ.get("AUTOFT_OPENAI_SECRET", "openai-secret"))
hf_secret = modal.Secret.from_name(os.environ.get("AUTOFT_HF_SECRET", "hf-secret"))

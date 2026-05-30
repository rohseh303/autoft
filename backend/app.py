"""Central Modal App definition — images, volumes, queues, secrets used everywhere."""
from __future__ import annotations

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
        "torch==2.4.0",
        "pydantic>=2.0",
        "huggingface_hub>=0.24.0",
        "transformers==4.46.3",
        "datasets>=2.20.0",
        "accelerate>=0.34.0",
        "peft>=0.13.0",
        "trl==0.12.2",
        "bitsandbytes>=0.44.0",
        "sentencepiece",
        "protobuf",
    )
    .pip_install(
        "unsloth==2024.12.4",
        "unsloth_zoo==2024.12.3",
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
openai_secret = modal.Secret.from_name("openai-secret")
hf_secret = modal.Secret.from_name("hf-secret")

"""Shared schemas — the contract between the research agent, trainer, and frontend."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BaseModelName = Literal["Qwen2.5-0.5B-Instruct", "SmolLM2-1.7B-Instruct"]

MODEL_REGISTRY: dict[str, str] = {
    "Qwen2.5-0.5B-Instruct": "unsloth/Qwen2.5-0.5B-Instruct",
    "SmolLM2-1.7B-Instruct": "unsloth/SmolLM2-1.7B-Instruct",
}


class EvalExample(BaseModel):
    input: str
    expected_output: str | None = None


class TrainingConfig(BaseModel):
    max_steps: int = Field(default=150, ge=10, le=2000)
    learning_rate: float = Field(default=2e-4, gt=0)
    batch_size: int = Field(default=2, ge=1, le=16)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32)
    lora_r: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=16, ge=4, le=128)
    max_seq_length: int = Field(default=2048, ge=256, le=8192)
    warmup_steps: int = Field(default=10, ge=0)
    seed: int = 42


class RunPlan(BaseModel):
    """Everything the agent decides; everything training consumes."""

    task_summary: str
    base_model: BaseModelName = "Qwen2.5-0.5B-Instruct"
    hf_dataset: str
    dataset_config: str | None = None
    dataset_split: str = "train[:2000]"
    input_field: str
    output_field: str
    prompt_template: str = (
        "### Instruction:\n{input}\n\n### Response:\n{output}"
    )
    benchmarks: list[str] = Field(default_factory=list)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    reasoning: str = ""


class UserRequest(BaseModel):
    task_description: str
    eval_examples: list[EvalExample] = Field(default_factory=list)
    preferred_model: BaseModelName | None = None


class StepMetric(BaseModel):
    run_id: str
    step: int
    loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    epoch: float | None = None
    elapsed_seconds: float | None = None


class RunStatus(BaseModel):
    run_id: str
    state: Literal["pending", "loading", "training", "evaluating", "done", "failed"]
    message: str = ""
    plan: RunPlan | None = None


class EvalComparison(BaseModel):
    input: str
    base_output: str
    finetuned_output: str
    expected_output: str | None = None
    judge_score: float | None = None      # 0-10, set by the LLM judge
    judge_critique: str | None = None      # one-line "what to improve next"


class RunResult(BaseModel):
    run_id: str
    plan: RunPlan
    final_loss: float | None = None        # training loss (drops, overfits — not the objective)
    eval_loss: float | None = None         # held-out loss (generalization signal)
    judge_score: float | None = None       # mean LLM-judge score across eval examples, 0-10
    objective: float | None = None         # the scalar the post-training lead maximizes
    comparisons: list[EvalComparison] = Field(default_factory=list)

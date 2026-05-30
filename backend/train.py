"""Fine-tuning entry point. Loads model with Unsloth, runs SFTTrainer, streams metrics."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

from backend.app import (
    MODELS_DIR,
    app,
    hf_secret,
    metrics_queue,
    model_volume,
    run_results,
    run_state,
    train_image,
)


@app.function(
    image=train_image,
    gpu="L4",
    timeout=60 * 30,
    volumes={MODELS_DIR: model_volume},
    secrets=[hf_secret],
)
def train_run(run_id: str, plan_dict: dict, eval_examples: list[dict]) -> dict:
    """Run an SFT training job. Pushes metrics to `metrics_queue` partitioned by run_id."""
    # Imports inside the function so they only resolve in the train image.
    import datasets as _datasets
    _datasets.disable_caching()
    from datasets import load_dataset
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from unsloth import FastLanguageModel


    from shared.schemas import (
        MODEL_REGISTRY,
        RunPlan,
        RunResult,
        RunStatus,
        StepMetric,
    )

    plan = RunPlan.model_validate(plan_dict)

    def push_status(state: str, message: str = "") -> None:
        run_state[run_id] = RunStatus(
            run_id=run_id, state=state, message=message, plan=plan
        ).model_dump()

    push_status("loading", f"Loading {plan.base_model}...")

    hf_id = MODEL_REGISTRY[plan.base_model]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_id,
        max_seq_length=plan.training.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        cache_dir=f"{MODELS_DIR}/hf_cache",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=plan.training.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=plan.training.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=plan.training.seed,
        max_seq_length=plan.training.max_seq_length,
    )

    push_status("loading", f"Loading dataset {plan.hf_dataset}...")
    ds = load_dataset(
        plan.hf_dataset,
        plan.dataset_config,
        split=plan.dataset_split,
        cache_dir=f"{MODELS_DIR}/ds_cache",
    )

    def _stringify(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            # e.g. SQuAD 'answers' is a dict with 'text' list
            if "text" in value:
                t = value["text"]
                return t[0] if isinstance(t, list) and t else str(t)
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return value[0] if value and isinstance(value[0], str) else json.dumps(value, ensure_ascii=False)
        return str(value)

    def _format_row(row):
        inp = _stringify(row.get(plan.input_field, ""))
        out = _stringify(row.get(plan.output_field, ""))
        return {"text": plan.prompt_template.format(input=inp, output=out)}

    columns_to_drop = [c for c in ds.column_names if c != "text"]
    ds = ds.map(_format_row, remove_columns=columns_to_drop)

    max_len = plan.training.max_seq_length

    def _tokenize_batch(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)
        return enc

    ds = ds.map(_tokenize_batch, batched=True, remove_columns=["text"])
    print(f"[autoft] pre-tokenized {len(ds)} examples; columns={ds.column_names}")

    # Streaming callback ---------------------------------------------------
    start = time.time()

    class StreamCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            metric = StepMetric(
                run_id=run_id,
                step=int(state.global_step),
                loss=logs.get("loss"),
                learning_rate=logs.get("learning_rate"),
                grad_norm=logs.get("grad_norm"),
                epoch=logs.get("epoch"),
                elapsed_seconds=time.time() - start,
            )
            metrics_queue.put(metric.model_dump(), partition=run_id)

    push_status("training", "SFTTrainer running...")

    output_dir = f"{MODELS_DIR}/runs/{run_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[autoft] tokenizer.eos_token={tokenizer.eos_token!r} pad_token={tokenizer.pad_token!r}")

    training_args = TrainingArguments(
        per_device_train_batch_size=plan.training.batch_size,
        gradient_accumulation_steps=plan.training.gradient_accumulation_steps,
        warmup_steps=plan.training.warmup_steps,
        max_steps=plan.training.max_steps,
        learning_rate=plan.training.learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=plan.training.seed,
        output_dir=output_dir,
        report_to="none",
        save_strategy="no",
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[StreamCallback()],
    )

    train_result = trainer.train()
    final_loss = float(train_result.training_loss) if train_result.training_loss else None

    # Save LoRA adapter for inference comparison
    adapter_dir = f"{output_dir}/lora"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    model_volume.commit()

    # Inline before/after eval --------------------------------------------
    push_status("evaluating", "Generating base vs fine-tuned outputs...")
    comparisons = _run_eval(
        plan=plan,
        model=model,
        tokenizer=tokenizer,
        eval_examples=eval_examples,
    )

    result = RunResult(
        run_id=run_id, plan=plan, final_loss=final_loss, comparisons=comparisons
    )
    run_results[run_id] = result.model_dump()

    # Sentinel — tells the SSE consumer to close.
    metrics_queue.put({"run_id": run_id, "done": True, "final_loss": final_loss}, partition=run_id)
    push_status("done", f"final_loss={final_loss:.4f}" if final_loss else "done")
    return result.model_dump()


def _run_eval(plan, model, tokenizer, eval_examples):
    """Generate from base (LoRA disabled) and fine-tuned (LoRA enabled) for side-by-side."""
    from unsloth import FastLanguageModel
    from shared.schemas import EvalComparison

    if not eval_examples:
        return []

    FastLanguageModel.for_inference(model)

    def _gen(prompt: str) -> str:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    comparisons: list[EvalComparison] = []
    for ex in eval_examples:
        prompt = plan.prompt_template.format(input=ex["input"], output="").rstrip()
        ft_out = _gen(prompt)
        with model.disable_adapter():
            base_out = _gen(prompt)
        comparisons.append(
            EvalComparison(
                input=ex["input"],
                base_output=base_out,
                finetuned_output=ft_out,
                expected_output=ex.get("expected_output"),
            )
        )
    return comparisons


@app.local_entrypoint()
def smoke():
    """Local smoke test: `modal run backend/train.py` runs billsum + Qwen2.5-0.5B."""
    import uuid

    plan = {
        "task_summary": "Follow instructions in the style of Alpaca.",
        "base_model": "Qwen2.5-0.5B-Instruct",
        "hf_dataset": "yahma/alpaca-cleaned",
        "dataset_config": None,
        "dataset_split": "train[:500]",
        "input_field": "instruction",
        "output_field": "output",
        "prompt_template": "### Instruction:\n{input}\n\n### Response:\n{output}",
        "benchmarks": ["alpaca-eval"],
        "training": {
            "max_steps": 30,
            "learning_rate": 2e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "lora_r": 16,
            "lora_alpha": 16,
            "max_seq_length": 1024,
            "warmup_steps": 5,
            "seed": 42,
        },
        "reasoning": "smoke",
    }
    eval_examples = [
        {"input": "Write a haiku about an autonomous fine-tuning system.", "expected_output": None},
    ]
    run_id = f"smoke-{uuid.uuid4().hex[:8]}"
    out = train_run.remote(run_id, plan, eval_examples)
    print(f"run_id={run_id}")
    print(f"final_loss={out['final_loss']}")
    for c in out["comparisons"]:
        print("=" * 60)
        print(f"INPUT: {c['input'][:200]}")
        print(f"BASE:  {c['base_output'][:300]}")
        print(f"FT:    {c['finetuned_output'][:300]}")

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

# Importing here registers judge_outputs on the shared `app`, so a single
# `modal run backend/train.py::trial` deploys the judge alongside the trainer.
from backend.judge import judge_outputs  # noqa: E402


@app.function(
    image=train_image,
    gpu=os.environ.get("AUTOFT_GPU", "L4"),
    timeout=60 * 30,
    volumes={MODELS_DIR: model_volume},
    secrets=[hf_secret],
)
def train_run(run_id: str, plan_dict: dict, eval_examples: list[dict]) -> dict:
    """Run an SFT training job. Pushes metrics to `metrics_queue` partitioned by run_id."""
    # Imports inside the function so they only resolve in the train image.
    from datasets import load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from shared.schemas import (
        MODEL_REGISTRY,
        EvalComparison,
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

    # Carve a held-out slice so we can report generalization (eval_loss), not
    # just training loss. Best-effort: tiny datasets just skip it.
    eval_ds = None
    if len(ds) >= 32:
        n_eval = min(64, max(8, len(ds) // 10))
        _split = ds.train_test_split(test_size=n_eval, seed=plan.training.seed)
        ds, eval_ds = _split["train"], _split["test"]

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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            # dataset_text_field/max_seq_length live on SFTConfig in modern trl
            # (they were removed as SFTTrainer kwargs); accepted by old trl too.
            dataset_text_field="text",
            max_seq_length=plan.training.max_seq_length,
            per_device_train_batch_size=plan.training.batch_size,
            per_device_eval_batch_size=plan.training.batch_size,
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
            dataset_num_proc=1,
        ),
        callbacks=[StreamCallback()],
    )

    train_result = trainer.train()
    final_loss = float(train_result.training_loss) if train_result.training_loss else None

    # Held-out eval loss — the generalization signal the optimizer reads.
    eval_loss = None
    if eval_ds is not None:
        try:
            push_status("evaluating", "Computing held-out eval loss...")
            _ev = trainer.evaluate().get("eval_loss")
            eval_loss = float(_ev) if _ev is not None else None
        except Exception as e:  # eval is best-effort; never fail the run on it
            print(f"[eval_loss] skipped: {e}")

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
        run_id=run_id,
        plan=plan,
        final_loss=final_loss,
        eval_loss=eval_loss,
        comparisons=comparisons,
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
        "task_summary": "Summarize US Congressional bills concisely.",
        "base_model": "Qwen2.5-0.5B-Instruct",
        "hf_dataset": "FiscalNote/billsum",
        "dataset_config": None,
        "dataset_split": "train[:500]",
        "input_field": "text",
        "output_field": "summary",
        "prompt_template": "### Instruction:\nSummarize the following bill:\n\n{input}\n\n### Response:\n{output}",
        "benchmarks": ["billsum-rouge"],
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
        {"input": "A bill to require the Secretary of Health to publish guidelines on AI in clinical decision-making by 2027.", "expected_output": None},
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


@app.local_entrypoint()
def trial(
    plan: str = "plan.json",
    out: str = "result.json",
    evals: str = "eval.json",
    ledger: str = "trials.jsonl",
):
    """One optimization trial — the post-training lead's unit of work.

    Reads the recipe from `plan` and the held-out test from `evals`, trains +
    evals on Modal, has the LLM judge score the outputs, then writes the full
    result to `out` and appends a one-line summary to `ledger`.

        modal run backend/train.py::trial --plan plan.json --out result.json
    """
    import json
    import uuid
    from pathlib import Path

    plan_dict = json.loads(Path(plan).read_text())
    eval_examples = []
    if Path(evals).exists():
        eval_examples = json.loads(Path(evals).read_text())

    run_id = f"trial-{uuid.uuid4().hex[:8]}"
    print(
        f"[{run_id}] training {plan_dict.get('base_model')} on "
        f"{plan_dict.get('hf_dataset')} ({plan_dict.get('dataset_split')})..."
    )

    result = train_run.remote(run_id, plan_dict, eval_examples)

    # Judge the fine-tuned generations elementwise (one OpenAI call per example).
    judge_score = None
    if result.get("comparisons"):
        try:
            judged = judge_outputs.remote(plan_dict.get("task_summary", ""), result["comparisons"])
        except Exception as e:  # missing secret / judge infra down -> degrade to eval_loss
            judged = {"mean_score": None, "per_example": [], "error": str(e)}
        judge_score = judged.get("mean_score")
        if judge_score is None:
            print(f"[{run_id}] judge unavailable: {judged.get('error')} — objective falls back to eval_loss")
        result["judge_score"] = judge_score
        for comp, j in zip(result["comparisons"], judged.get("per_example", [])):
            comp["judge_score"] = j.get("score")
            comp["judge_critique"] = j.get("critique")

    # Objective: judge score is primary (it's the real product goal); fall back
    # to negative eval loss when there are no eval examples to judge.
    eval_loss = result.get("eval_loss")
    if judge_score is not None:
        objective = judge_score
    elif eval_loss is not None:
        objective = -eval_loss
    else:
        objective = -(result.get("final_loss") or 0.0)
    result["objective"] = objective

    Path(out).write_text(json.dumps(result, indent=2))

    record = {
        "run_id": run_id,
        "objective": round(objective, 4),
        "judge_score": judge_score,
        "eval_loss": round(eval_loss, 4) if eval_loss is not None else None,
        "final_loss": round(result["final_loss"], 4) if result.get("final_loss") is not None else None,
        "base_model": plan_dict.get("base_model"),
        "hf_dataset": plan_dict.get("hf_dataset"),
        "dataset_split": plan_dict.get("dataset_split"),
        "prompt_template": plan_dict.get("prompt_template"),
        "training": plan_dict.get("training"),
    }
    with open(ledger, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(
        f"[{run_id}] objective={objective:.3f}  judge={judge_score}  "
        f"eval_loss={eval_loss}  final_loss={result.get('final_loss')}"
    )
    print(f"[{run_id}] wrote {out}; appended {ledger}. Read {out} for per-example outputs + critiques.")

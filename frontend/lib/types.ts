// Mirror of shared/schemas.py — keep in sync.

export type BaseModelName = "Qwen2.5-0.5B-Instruct" | "SmolLM2-1.7B-Instruct";

export interface EvalExample {
  input: string;
  expected_output?: string | null;
}

export interface TrainingConfig {
  max_steps: number;
  learning_rate: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  lora_r: number;
  lora_alpha: number;
  max_seq_length: number;
  warmup_steps: number;
  seed: number;
}

export interface RunPlan {
  task_summary: string;
  base_model: BaseModelName;
  hf_dataset: string;
  dataset_config: string | null;
  dataset_split: string;
  input_field: string;
  output_field: string;
  prompt_template: string;
  benchmarks: string[];
  training: TrainingConfig;
  reasoning: string;
}

export interface UserRequest {
  task_description: string;
  eval_examples: EvalExample[];
  preferred_model?: BaseModelName | null;
}

export interface StepMetric {
  run_id: string;
  step: number;
  loss?: number | null;
  learning_rate?: number | null;
  grad_norm?: number | null;
  epoch?: number | null;
  elapsed_seconds?: number | null;
}

export type RunState =
  | "pending"
  | "loading"
  | "training"
  | "evaluating"
  | "done"
  | "failed";

export interface RunStatus {
  run_id: string;
  state: RunState;
  message: string;
  plan: RunPlan | null;
}

export interface EvalComparison {
  input: string;
  base_output: string;
  finetuned_output: string;
  expected_output?: string | null;
}

export interface RunResult {
  run_id: string;
  plan: RunPlan;
  final_loss: number | null;
  comparisons: EvalComparison[];
}

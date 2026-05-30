// Mirror of backend/shared/schemas.py — the RunPlan spine, shared by client + BFF.
// Kept framework-free so both the React client and the Elysia server import it.

// Merged backend (shared/schemas.py) now ships a single base model.
export type BaseModelName = "Qwen3.5-2B";

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
  judge_score?: number | null;     // 0-10, the LLM judge's score for this output
  judge_critique?: string | null;  // one-line "what to improve next"
}

export interface RunResult {
  run_id: string;
  plan: RunPlan;
  final_loss: number | null;       // training loss
  eval_loss?: number | null;       // held-out loss (generalization signal)
  judge_score?: number | null;     // mean LLM-judge score across eval examples, 0-10
  objective?: number | null;       // the scalar the post-training lead maximizes
  comparisons: EvalComparison[];
  scorecard?: Scorecard | null;    // UI-only rollup synthesized from the above
}

// How the model actually performed — a blind-preference verdict + a few
// base-vs-finetuned measures. Synthesized in demo mode; the `simulated` flag
// keeps the UI honest about that.
export interface Measure {
  name: string;
  base: number;
  finetuned: number;
  scale: number;        // value that fills the bar (e.g. 100 for %, 0.6 for ROUGE)
  unit: string;         // "%", "", "pt"
}

export interface Scorecard {
  judge_wins: number;
  judge_total: number;
  verdict: string;      // one warm human sentence
  measures: Measure[];
  simulated: boolean;
}

// ---- streaming events (superset of the Modal backend's SSE contract) -------
// The backend emits: status | metric | result.
// Studio adds two forward-compatible events the BFF can synthesize today and a
// real backend could emit later:
//   thought — research agent's reasoning trace (search/peek/decide)
//   sample  — model re-sampled mid-training (gibberish -> coherent)

export interface ThoughtEvent {
  kind: "search" | "peek" | "note" | "decision";
  text: string;
  detail?: string;
}

export interface SampleEvent {
  run_id: string;
  step: number;
  prompt: string;
  text: string;
}

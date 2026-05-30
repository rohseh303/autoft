// Mirror of backend/shared/schemas.py — the RunPlan spine, shared by client + BFF.
// Kept framework-free so both the React client and the Elysia server import it.

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
}

export interface RunResult {
  run_id: string;
  plan: RunPlan;
  final_loss: number | null;
  comparisons: EvalComparison[];
  scorecard?: Scorecard | null;
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

// ---- transparency timeline -------------------------------------------------
// A single, source-tagged reasoning/status event — the backend writes these to
// its `run_events` Dict (one per run) and the research agent returns a list of
// them with the plan. The UI replays them so you can see *which* subagent did
// *what*, and at which training step.
//   source: "research" (Codex picking the dataset) | "trainer" | "judge" | "lead"
//   kind:   "status" | "progress" | "note" | "search" | "peek" | "decision" | "sample"
export interface RunEvent {
  ts: number | null;        // seconds since the run/agent started (null if unknown)
  source: string;
  kind: string;
  text: string;
  step?: number | null;     // training step this event belongs to, when relevant
  detail?: string | null;
}

// Shape returned by POST /api/research — the plan plus the agent's reasoning trace.
export interface ResearchResult {
  plan: RunPlan;
  events: RunEvent[];
}

// Shape returned by GET /api/run/:id/metrics — the polling snapshot the charts read.
export interface MetricsSnapshot {
  status: RunStatus | null;
  metrics: StepMetric[];
  result: RunResult | null;
  samples?: SampleEvent[] | null;   // mock-only: mid-training re-samples
}

// Mock engine — lets AutoFT Studio run end-to-end with no Modal, no GPU, no
// secrets. Synthesizes a believable research trace, a RunPlan, a training loss
// curve, live model re-samples (gibberish -> coherent), and a before/after.
// This is what makes the project demoable on a laptop in one command.
import type {
  Measure, RunEvent, RunPlan, RunResult, RunStatus, RunState,
  SampleEvent, Scorecard, StepMetric, UserRequest,
} from "../shared/types";

// --- a tiny catalog echo so the mock plan looks plausible per task ----------
const CATALOG: { tags: string[]; ds: string; cfg: string | null; inp: string; out: string; note: string }[] = [
  { tags: ["legal", "contract", "summari", "bill"], ds: "billsum", cfg: null, inp: "text", out: "summary", note: "human-written legislative summaries — strong style signal" },
  { tags: ["sql", "database", "query"], ds: "b-mc2/sql-create-context", cfg: null, inp: "question", out: "answer", note: "NL questions to SQL with table context" },
  { tags: ["news", "article"], ds: "cnn_dailymail", cfg: "3.0.0", inp: "article", out: "highlights", note: "CNN/DailyMail news summarization" },
  { tags: ["meeting", "transcript", "dialogue", "chat", "action"], ds: "samsum", cfg: null, inp: "dialogue", out: "summary", note: "messenger-style dialogue summaries" },
  { tags: ["code", "python", "program"], ds: "sahil2801/CodeAlpaca-20k", cfg: null, inp: "instruction", out: "output", note: "code-generation instructions" },
  { tags: ["math", "reason", "word problem"], ds: "gsm8k", cfg: "main", inp: "question", out: "answer", note: "grade-school math with chain-of-thought" },
];

function pick(task: string) {
  const t = task.toLowerCase();
  return CATALOG.find((c) => c.tags.some((tag) => t.includes(tag))) ?? CATALOG[0]!;
}

export function mockPlan(req: UserRequest): RunPlan {
  const c = pick(req.task_description);
  return {
    task_summary: req.task_description.trim().replace(/\.$/, "") || "Fine-tune a small model",
    base_model: req.preferred_model ?? "Qwen3.5-2B",
    hf_dataset: c.ds,
    dataset_config: c.cfg,
    dataset_split: "train[:2000]",
    input_field: c.inp,
    output_field: c.out,
    prompt_template: "### Instruction:\n{input}\n\n### Response:\n{output}",
    benchmarks: [`${c.ds.split("/").pop()}-rouge`, "exact-match"],
    training: {
      max_steps: 150, learning_rate: 2e-4, batch_size: 2,
      gradient_accumulation_steps: 4, lora_r: 16, lora_alpha: 16,
      max_seq_length: 2048, warmup_steps: 10, seed: 42,
    },
    reasoning:
      `Matched your task to ${c.ds} — ${c.note}. The ${c.inp} → ${c.out} column mapping is clean ` +
      `and instruction-style formatting works well for a small model. 150 LoRA steps on an H200 is enough ` +
      `to imprint the style without overfitting a 2B model.`,
  };
}

// Research trace — the "stream of thought" the agent UI shows live.
export function mockThoughts(req: UserRequest) {
  const c = pick(req.task_description);
  return [
    { kind: "note" as const, text: `Reading your task: "${req.task_description.trim()}"` },
    { kind: "search" as const, text: `Searching the Hub for matching datasets`, detail: `query: "${c.tags.slice(0, 2).join(" ")}"` },
    { kind: "note" as const, text: `Top candidates: ${c.ds} · ${CATALOG[2]!.ds} · ${CATALOG[3]!.ds}` },
    { kind: "peek" as const, text: `Peeking ${c.ds} to verify the schema`, detail: `columns: ${c.inp}, ${c.out}  ✓` },
    { kind: "decision" as const, text: `Choosing ${c.ds}`, detail: c.note },
  ];
}

// Deterministic-ish loss curve: exp decay + a little noise, around step count.
export function lossAt(step: number, maxSteps: number): number {
  const x = step / maxSteps;
  const base = 0.85 + 2.3 * Math.exp(-3.2 * x);
  const noise = (Math.sin(step * 1.7) + Math.sin(step * 0.6)) * 0.04 * (1 - x);
  return Math.max(0.7, base + noise);
}
export function lrAt(step: number, p: RunPlan): number {
  const w = p.training.warmup_steps;
  const m = p.training.max_steps;
  const peak = p.training.learning_rate;
  if (step <= w) return peak * (step / Math.max(1, w));
  return peak * (1 - (step - w) / Math.max(1, m - w));
}
export function gradAt(step: number): number {
  return Math.max(0.05, 1.4 * Math.exp(-step * 0.02) + Math.abs(Math.sin(step)) * 0.15);
}

// The showpiece: the model's output on one eval prompt, sharpening over training.
// We interpolate from noise -> the target answer as steps progress.
const STAGES = [
  "the the bill bill of to the and the requ",
  "This bill the Secretary to of guidelines the on",
  "This bill requires the Secretary to publish guidelines",
  "Requires the Secretary of Health to publish guidelines on AI in clinical decision-making by 2027.",
];
export function sampleAt(step: number, maxSteps: number): string {
  const x = step / maxSteps;
  const idx = Math.min(STAGES.length - 1, Math.floor(x * STAGES.length + 0.0001));
  return STAGES[idx]!;
}

// Small deterministic PRNG so a given task always scores the same — stable
// across reloads, which feels real instead of dice-rolling on every render.
function seeded(str: string) {
  let h = 2166136261;
  for (const c of str) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); }
  return () => {
    h += 0x6d2b79f5;
    let t = Math.imul(h ^ (h >>> 15), 1 | h);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mockScorecard(plan: RunPlan): Scorecard {
  const r = seeded(plan.task_summary + plan.hf_dataset);
  const between = (lo: number, hi: number) => lo + (hi - lo) * r();

  const total = 10;
  const wins = Math.round(between(7, 9.4)); // your model wins most blind matchups

  const measures: Measure[] = [];
  // derive measures from the benchmarks the agent named
  for (const b of plan.benchmarks.slice(0, 2)) {
    const name = b.toLowerCase();
    if (name.includes("rouge")) {
      measures.push({ name: "ROUGE-L", base: +between(0.11, 0.21).toFixed(2), finetuned: +between(0.36, 0.49).toFixed(2), scale: 0.6, unit: "" });
    } else if (name.includes("exact") || name.includes("match") || name.includes("accuracy")) {
      measures.push({ name: "Exact match", base: Math.round(between(6, 19)), finetuned: Math.round(between(42, 66)), scale: 100, unit: "%" });
    } else {
      measures.push({ name: b, base: Math.round(between(22, 40)), finetuned: Math.round(between(58, 82)), scale: 100, unit: "%" });
    }
  }
  // always-present, human-legible measures
  measures.push({ name: "Format adherence", base: Math.round(between(28, 48)), finetuned: Math.round(between(90, 99)), scale: 100, unit: "%" });
  measures.push({ name: "Style match", base: Math.round(between(24, 44)), finetuned: Math.round(between(70, 90)), scale: 100, unit: "%" });

  return {
    judge_wins: wins,
    judge_total: total,
    verdict:
      wins >= 9 ? "A blind judge picked your model almost every time."
      : wins >= 8 ? "A blind judge clearly preferred your model."
      : "A blind judge leaned toward your model more often than not.",
    measures,
    simulated: true,
  };
}

export function mockResult(runId: string, plan: RunPlan, req: UserRequest): RunResult {
  const examples = req.eval_examples.length
    ? req.eval_examples
    : [{ input: "A bill to require the Secretary of Health to publish guidelines on AI in clinical decision-making by 2027.", expected_output: null }];
  return {
    run_id: runId,
    plan,
    final_loss: lossAt(plan.training.max_steps, plan.training.max_steps),
    comparisons: examples.map((ex) => ({
      input: ex.input,
      base_output:
        "Sure! Here is a summary. The bill is about a bill. It talks about the Secretary and some health things and AI and guidelines and dates and more details about the topic in general terms.",
      finetuned_output:
        "Requires HHS to publish AI clinical-decision guidelines by 2027.",
      expected_output: ex.expected_output ?? null,
    })),
    scorecard: mockScorecard(plan),
  };
}

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ============================================================================
// Live-run simulation — polling-friendly.
//
// Real mode polls Modal's /run/:id/metrics + /events (cumulative snapshots) and
// streams /run/:id/logs. Mock mirrors that exact contract so the Studio runs
// end-to-end with no Modal: every snapshot below is a PURE function of elapsed
// seconds, so repeated polls are consistent and the run "advances" on wall-clock.
// ============================================================================

// Compressed wall-clock so a full mock run finishes in ~15s (watchable demo).
const SIM = { loadingS: 3, stepS: 0.06, evalS: 2.4 };

function simPhase(plan: RunPlan, elapsedS: number): { state: RunState; message: string; step: number; done: boolean } {
  const max = plan.training.max_steps;
  const trainEnd = SIM.loadingS + max * SIM.stepS;
  const evalEnd = trainEnd + SIM.evalS;
  if (elapsedS < SIM.loadingS) {
    const msgs = ["pulling image", `downloading ${plan.base_model}`, `loading ${plan.hf_dataset}`, "tokenizing 2000 rows"];
    const i = Math.min(msgs.length - 1, Math.floor((elapsedS / SIM.loadingS) * msgs.length));
    return { state: "loading", message: msgs[i]!, step: 0, done: false };
  }
  if (elapsedS < trainEnd) {
    const step = Math.max(1, Math.min(max, Math.floor((elapsedS - SIM.loadingS) / SIM.stepS)));
    return { state: "training", message: "Trainer running…", step, done: false };
  }
  if (elapsedS < evalEnd) return { state: "evaluating", message: "generating base vs fine-tuned", step: max, done: false };
  return { state: "done", message: `final_loss=${lossAt(max, max).toFixed(4)}`, step: max, done: true };
}

// Cumulative metrics snapshot — same shape as Modal's /run/:id/metrics, plus a
// mock-only `samples` array (the live re-sample showpiece, which real mode lacks).
export function simulateMetrics(
  runId: string, plan: RunPlan, req: UserRequest, elapsedS: number,
): { status: RunStatus; metrics: StepMetric[]; result: RunResult | null; samples: SampleEvent[] } {
  const { state, message, step } = simPhase(plan, elapsedS);
  const max = plan.training.max_steps;
  const metrics: StepMetric[] = [];
  const samples: SampleEvent[] = [];
  const prompt = req.eval_examples[0]?.input ?? "eval prompt";
  for (let s = 1; s <= step; s++) {
    metrics.push({
      run_id: runId, step: s,
      loss: lossAt(s, max), learning_rate: lrAt(s, plan), grad_norm: gradAt(s),
      epoch: (s * plan.training.batch_size * plan.training.gradient_accumulation_steps) / 2000,
      elapsed_seconds: SIM.loadingS + s * SIM.stepS,
    });
    if (s === 1 || s % 12 === 0 || s === max) samples.push({ run_id: runId, step: s, prompt, text: sampleAt(s, max) });
  }
  return {
    status: { run_id: runId, state, message, plan },
    metrics,
    result: state === "done" ? mockResult(runId, plan, req) : null,
    samples,
  };
}

// Cumulative transparency timeline — mirrors what the trainer writes to Modal's
// run_events Dict, and adds a couple of judge beats so the demo shows the
// multi-source ("which subagent") view. Research events come from /api/research.
export function simulateEvents(runId: string, plan: RunPlan, elapsedS: number): RunEvent[] {
  const { state, step, done } = simPhase(plan, elapsedS);
  const max = plan.training.max_steps;
  const trainStart = SIM.loadingS, trainEnd = SIM.loadingS + max * SIM.stepS;
  const evs: RunEvent[] = [];
  const push = (ts: number, source: string, kind: string, text: string, st: number | null = null) =>
    evs.push({ ts: +ts.toFixed(2), source, kind, text, step: st, detail: null });

  push(0.0, "trainer", "status", `Loading ${plan.base_model}…`);
  if (elapsedS >= SIM.loadingS * 0.5) push(SIM.loadingS * 0.5, "trainer", "status", `Loading dataset ${plan.hf_dataset}…`);
  if (state !== "loading") push(trainStart, "trainer", "status", "Trainer running…");
  for (const frac of [0, 0.25, 0.5, 0.75]) {
    const mark = frac === 0 ? 1 : Math.max(1, Math.floor(max * frac));
    if (step >= mark) push(trainStart + mark * SIM.stepS, "trainer", "progress", `step ${Math.min(step, max)}/${max} · loss ${lossAt(mark, max).toFixed(4)}`, mark);
  }
  if (state === "evaluating" || done) push(trainEnd, "trainer", "status", "Generating base vs fine-tuned outputs…");
  if (done) {
    push(trainEnd + 0.3, "judge", "note", "Scoring fine-tuned vs base outputs (gpt-5.4-mini, elementwise)");
    push(trainEnd + SIM.evalS * 0.8, "judge", "decision", "Judge preferred the fine-tuned model on most eval examples");
    push(trainEnd + SIM.evalS, "trainer", "status", `done · final_loss=${lossAt(max, max).toFixed(4)}`, max);
  }
  return evs;
}

// Synthesized container log lines for the streamed /logs pane. Returns lines in
// emit order; the BFF paces them out over the run so they feel live.
export function mockLogLines(plan: RunPlan, req: UserRequest): string[] {
  const max = plan.training.max_steps;
  const lines: string[] = [
    "Building image autoft (cached layers)…",
    "✓ image built",
    `Loading ${plan.base_model} via Unsloth (bf16 LoRA)…`,
    "==((====))==  Unsloth 2026.5.8: Fast Qwen3.5 patching. Transformers: 5.5.0.",
    `   \\\\   /|    GPU: NVIDIA H200. Max memory: 141.0 GB. Platform: Linux.`,
    `O^O/ \\_/ \\    Bfloat16 = TRUE. Free Apache license.`,
    `[autoft] loading dataset ${plan.hf_dataset} (${plan.dataset_split})`,
    `[autoft] pre-tokenized 2000 examples; columns=['input_ids', 'attention_mask']`,
    `[autoft] tokenizer.eos_token='<|endoftext|>' pad_token='<|endoftext|>'`,
    "Trainer running… (linear LR, adamw_8bit)",
  ];
  for (let s = 1; s <= max; s++) {
    if (s === 1 || s % 10 === 0 || s === max) {
      lines.push(`{'loss': ${lossAt(s, max).toFixed(4)}, 'grad_norm': ${gradAt(s).toFixed(3)}, 'learning_rate': ${lrAt(s, plan).toExponential(2)}, 'epoch': ${((s * plan.training.batch_size * plan.training.gradient_accumulation_steps) / 2000).toFixed(2)}}`);
    }
  }
  lines.push(
    `{'train_runtime': ${(max * SIM.stepS).toFixed(1)}, 'train_loss': ${lossAt(max, max).toFixed(4)}}`,
    "[autoft] computing held-out eval loss…",
    "[autoft] generating base vs fine-tuned outputs…",
    "[autoft] saved LoRA adapter; committed volume",
    "✓ run complete",
  );
  return lines;
}

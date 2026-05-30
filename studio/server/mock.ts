// Mock engine — lets AutoFT Studio run end-to-end with no Modal, no GPU, no
// secrets. Synthesizes a believable research trace, a RunPlan, a training loss
// curve, live model re-samples (gibberish -> coherent), and a before/after.
// This is what makes the project demoable on a laptop in one command.
import type { Measure, RunPlan, RunResult, Scorecard, UserRequest } from "../shared/types";

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
    base_model: req.preferred_model ?? "Qwen2.5-0.5B-Instruct",
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
      `and instruction-style formatting works well for a small model. 150 LoRA steps on an L4 is enough ` +
      `to imprint the style without overfitting a 0.5B model.`,
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

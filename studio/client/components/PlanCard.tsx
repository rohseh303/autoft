import { useState } from "react";
import type { RunPlan } from "@shared/types";

// The plan, rendered clear — plain language up top, the knobs editable but
// tucked behind "tune". Approving promotes us to the training theater.
export function PlanCard({ plan, onApprove }: { plan: RunPlan; onApprove: (p: RunPlan) => void }) {
  const [p, setP] = useState<RunPlan>(plan);
  const [tuning, setTuning] = useState(false);
  const [starting, setStarting] = useState(false);

  const setT = (k: keyof RunPlan["training"], v: number) =>
    setP((cur) => ({ ...cur, training: { ...cur.training, [k]: v } }));

  return (
    <section className="plan card-hard rise">
      <span className="kicker">the recipe</span>
      <h2 className="plan-title">{p.task_summary}</h2>

      <div className="plan-grid">
        <Field label="Base model" value={p.base_model} />
        <Field label="Dataset" value={
          <a href={`https://huggingface.co/datasets/${p.hf_dataset}`} target="_blank" rel="noreferrer">{p.hf_dataset}</a>
        } />
        <Field label="Columns" value={`${p.input_field} → ${p.output_field}`} />
        <Field label="Examples" value={p.dataset_split} />
        <Field label="Steps" value={String(p.training.max_steps)} />
        <Field label="LoRA rank" value={String(p.training.lora_r)} />
      </div>

      <div className="plan-why">
        <span className="kicker">why this</span>
        <p>{p.reasoning}</p>
      </div>

      {p.benchmarks.length > 0 && (
        <div className="bench">
          {p.benchmarks.map((b) => <span key={b} className="bench-chip mono">{b}</span>)}
        </div>
      )}

      <button className="tune-toggle mono" onClick={() => setTuning((v) => !v)}>
        {tuning ? "▾ hide knobs" : "▸ tune the knobs"}
      </button>
      {tuning && (
        <div className="knobs rise">
          <Knob label="max steps" min={20} max={400} step={10} value={p.training.max_steps} onChange={(v) => setT("max_steps", v)} />
          <Knob label="LoRA r" min={4} max={64} step={4} value={p.training.lora_r} onChange={(v) => setT("lora_r", v)} />
          <Knob label="batch" min={1} max={16} step={1} value={p.training.batch_size} onChange={(v) => setT("batch_size", v)} />
          <Knob label="lr ×1e-4" min={1} max={10} step={1} value={Math.round(p.training.learning_rate * 1e4)} onChange={(v) => setT("learning_rate", v / 1e4)} />
        </div>
      )}

      <button
        className="approve"
        disabled={starting}
        onClick={() => { setStarting(true); onApprove(p); }}
      >
        {starting ? "Spinning up the GPU…" : "Start training"} <span className="arrow">→</span>
      </button>
    </section>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="pf">
      <div className="kicker">{label}</div>
      <div className="pf-val mono">{value}</div>
    </div>
  );
}

function Knob({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <label className="knob">
      <div className="knob-top mono"><span>{label}</span><span className="knob-val">{value}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

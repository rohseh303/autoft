"use client";
import type { RunPlan } from "@/lib/types";

interface Props {
  plan: RunPlan;
  onApprove: () => void;
  onBack: () => void;
  starting: boolean;
}

export function PlanPreview({ plan, onApprove, onBack, starting }: Props) {
  return (
    <div className="glass rounded-2xl p-8 max-w-3xl w-full">
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="text-xs uppercase tracking-wider text-blue-300 mb-1">
            Research agent · plan
          </div>
          <h2 className="text-2xl font-semibold">{plan.task_summary}</h2>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <Field label="Base model" value={plan.base_model} />
        <Field label="Dataset" value={
          <a
            href={`https://huggingface.co/datasets/${plan.hf_dataset}`}
            target="_blank"
            rel="noreferrer"
            className="text-blue-300 hover:text-blue-200 underline"
          >
            {plan.hf_dataset}
          </a>
        } />
        <Field label="Split" value={plan.dataset_split} />
        <Field label="Columns" value={`${plan.input_field} → ${plan.output_field}`} />
        <Field label="Steps" value={String(plan.training.max_steps)} />
        <Field label="Learning rate" value={plan.training.learning_rate.toExponential(1)} />
        <Field label="LoRA r" value={String(plan.training.lora_r)} />
        <Field label="Batch × accum" value={`${plan.training.batch_size} × ${plan.training.gradient_accumulation_steps}`} />
      </div>

      {plan.benchmarks.length > 0 && (
        <div className="mb-6">
          <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">
            Benchmarks to hill-climb
          </div>
          <div className="flex flex-wrap gap-2">
            {plan.benchmarks.map((b) => (
              <span
                key={b}
                className="text-xs bg-blue-500/20 text-blue-200 border border-blue-500/30 rounded-full px-3 py-1"
              >
                {b}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="mb-6">
        <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">
          Agent reasoning
        </div>
        <div className="text-sm text-slate-300 leading-relaxed bg-slate-900/40 border border-slate-700 rounded-lg p-4">
          {plan.reasoning || "—"}
        </div>
      </div>

      <details className="mb-6">
        <summary className="text-xs uppercase tracking-wider text-slate-400 cursor-pointer hover:text-slate-200">
          Prompt template
        </summary>
        <pre className="text-xs bg-slate-950 border border-slate-700 rounded-lg p-3 mt-2 overflow-x-auto text-slate-300">
{plan.prompt_template}
        </pre>
      </details>

      <div className="flex gap-3">
        <button
          type="button"
          onClick={onBack}
          className="flex-none px-4 py-3 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800/60"
        >
          ← Back
        </button>
        <button
          type="button"
          onClick={onApprove}
          disabled={starting}
          className="flex-1 bg-blue-500 hover:bg-blue-400 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold rounded-lg py-3 transition"
        >
          {starting ? "Starting…" : "Start training →"}
        </button>
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-slate-900/40 border border-slate-700 rounded-lg p-3">
      <div className="text-xs uppercase tracking-wider text-slate-400 mb-1">{label}</div>
      <div className="text-sm text-slate-100 font-mono">{value}</div>
    </div>
  );
}

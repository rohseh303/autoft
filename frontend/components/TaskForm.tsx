"use client";
import { useState } from "react";
import type { BaseModelName, EvalExample, UserRequest } from "@/lib/types";

interface Props {
  onSubmit: (req: UserRequest) => void;
  loading: boolean;
}

const EXAMPLES = [
  "I want a model that summarizes long legal contracts into 3 plain-English bullet points.",
  "I want a model that converts natural-language questions into SQL queries.",
  "I want a model that turns rambling meeting transcripts into crisp action items.",
  "I want a model that explains Python code line-by-line for total beginners.",
];

export function TaskForm({ onSubmit, loading }: Props) {
  const [task, setTask] = useState("");
  const [model, setModel] = useState<BaseModelName>("Qwen3.5-2B");
  const [examples, setExamples] = useState<EvalExample[]>([
    { input: "", expected_output: "" },
  ]);

  const updateExample = (i: number, field: "input" | "expected_output", value: string) => {
    setExamples((prev) =>
      prev.map((e, idx) => (idx === i ? { ...e, [field]: value } : e)),
    );
  };

  const addExample = () => setExamples((prev) => [...prev, { input: "", expected_output: "" }]);
  const removeExample = (i: number) => setExamples((prev) => prev.filter((_, idx) => idx !== i));

  const submit = () => {
    const cleaned = examples.filter((e) => e.input.trim().length > 0);
    onSubmit({
      task_description: task.trim(),
      eval_examples: cleaned,
      preferred_model: model,
    });
  };

  return (
    <div className="glass rounded-2xl p-8 max-w-3xl w-full">
      <h1 className="text-3xl font-semibold mb-2">
        Describe the model you want to train
      </h1>
      <p className="text-slate-400 mb-6">
        A research agent will pick the benchmarks, scrape a dataset, and run an autonomous fine-tune.
      </p>

      <label className="block text-sm font-medium text-slate-300 mb-2">Task</label>
      <textarea
        className="w-full bg-slate-900/60 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[120px]"
        placeholder="e.g. I want a model that summarizes long legal contracts..."
        value={task}
        onChange={(e) => setTask(e.target.value)}
      />
      <div className="flex flex-wrap gap-2 mt-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            className="text-xs text-slate-400 hover:text-blue-300 border border-slate-700 hover:border-blue-500 rounded-full px-3 py-1 transition"
            onClick={() => setTask(ex)}
          >
            {ex.slice(0, 50)}…
          </button>
        ))}
      </div>

      <label className="block text-sm font-medium text-slate-300 mt-6 mb-2">Base model</label>
      <select
        value={model}
        onChange={(e) => setModel(e.target.value as BaseModelName)}
        className="w-full bg-slate-900/60 border border-slate-700 rounded-lg px-4 py-2 text-slate-100"
      >
        <option value="Qwen3.5-2B">Qwen3.5-2B (bf16 LoRA)</option>
      </select>

      <label className="block text-sm font-medium text-slate-300 mt-6 mb-2">
        Eval examples <span className="text-slate-500 font-normal">(optional — used for before/after comparison)</span>
      </label>
      <div className="space-y-3">
        {examples.map((ex, i) => (
          <div key={i} className="rounded-lg bg-slate-900/40 border border-slate-700 p-3 space-y-2">
            <textarea
              className="w-full bg-slate-950/60 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100"
              placeholder="Input"
              value={ex.input}
              onChange={(e) => updateExample(i, "input", e.target.value)}
              rows={2}
            />
            <textarea
              className="w-full bg-slate-950/60 border border-slate-700 rounded px-3 py-2 text-sm text-slate-100"
              placeholder="Expected output (optional)"
              value={ex.expected_output ?? ""}
              onChange={(e) => updateExample(i, "expected_output", e.target.value)}
              rows={2}
            />
            {examples.length > 1 && (
              <button
                type="button"
                onClick={() => removeExample(i)}
                className="text-xs text-rose-300 hover:text-rose-200"
              >
                Remove
              </button>
            )}
          </div>
        ))}
        <button
          type="button"
          onClick={addExample}
          className="text-sm text-blue-300 hover:text-blue-200"
        >
          + Add example
        </button>
      </div>

      <button
        type="button"
        disabled={loading || task.trim().length === 0}
        onClick={submit}
        className="mt-8 w-full bg-blue-500 hover:bg-blue-400 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold rounded-lg py-3 transition"
      >
        {loading ? "Researching…" : "Design my training run →"}
      </button>
    </div>
  );
}

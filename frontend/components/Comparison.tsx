"use client";
import type { RunResult } from "@/lib/types";

interface Props {
  result: RunResult;
  onRestart: () => void;
}

export function Comparison({ result, onRestart }: Props) {
  return (
    <div className="max-w-5xl w-full space-y-4">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-xs uppercase tracking-wider bg-emerald-500/20 text-emerald-200 border border-emerald-500/40 rounded-full px-3 py-1">
          Done
        </span>
        <div className="text-slate-300">
          Final loss:{" "}
          <span className="text-slate-100 font-mono">
            {result.final_loss != null ? result.final_loss.toFixed(4) : "—"}
          </span>
        </div>
        <button
          onClick={onRestart}
          className="ml-auto text-sm text-blue-300 hover:text-blue-200"
        >
          Train another →
        </button>
      </div>

      {result.comparisons.length === 0 ? (
        <div className="glass rounded-xl p-8 text-slate-400">
          No eval examples were provided — skipping comparison.
        </div>
      ) : (
        result.comparisons.map((c, i) => (
          <div
            key={i}
            className="glass rounded-xl p-5 space-y-3"
          >
            <div>
              <div className="text-xs uppercase tracking-wider text-slate-400 mb-1">Input</div>
              <div className="text-sm text-slate-200 bg-slate-900/50 border border-slate-700 rounded-lg p-3 whitespace-pre-wrap">
                {c.input}
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <div className="text-xs uppercase tracking-wider text-slate-400 mb-1">
                  Base model
                </div>
                <div className="text-sm text-slate-200 bg-rose-500/5 border border-rose-500/20 rounded-lg p-3 whitespace-pre-wrap min-h-[80px]">
                  {c.base_output || "—"}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-emerald-300 mb-1">
                  Fine-tuned
                </div>
                <div className="text-sm text-slate-200 bg-emerald-500/5 border border-emerald-500/30 rounded-lg p-3 whitespace-pre-wrap min-h-[80px]">
                  {c.finetuned_output || "—"}
                </div>
              </div>
            </div>
            {c.expected_output && (
              <div>
                <div className="text-xs uppercase tracking-wider text-slate-400 mb-1">Expected</div>
                <div className="text-sm text-slate-300 bg-slate-900/50 border border-slate-700 rounded-lg p-3 whitespace-pre-wrap">
                  {c.expected_output}
                </div>
              </div>
            )}
          </div>
        ))
      )}
    </div>
  );
}

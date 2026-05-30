"use client";
import { useState } from "react";
import type { EvalExample, RunPlan, RunResult, UserRequest } from "@/lib/types";
import { postResearch, postTrain } from "@/lib/api";
import { TaskForm } from "@/components/TaskForm";
import { PlanPreview } from "@/components/PlanPreview";
import { Dashboard } from "@/components/Dashboard";
import { Comparison } from "@/components/Comparison";

type Phase = "input" | "plan" | "training" | "done";

export default function Page() {
  const [phase, setPhase] = useState<Phase>("input");
  const [request, setRequest] = useState<UserRequest | null>(null);
  const [plan, setPlan] = useState<RunPlan | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleResearch = async (req: UserRequest) => {
    setLoading(true);
    setError(null);
    try {
      const p = await postResearch(req);
      setRequest(req);
      setPlan(p);
      setPhase("plan");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    if (!plan || !request) return;
    setLoading(true);
    setError(null);
    try {
      const { run_id } = await postTrain(plan, request.eval_examples);
      setRunId(run_id);
      setPhase("training");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleDone = (r: RunResult) => {
    setResult(r);
    setPhase("done");
  };

  const restart = () => {
    setPhase("input");
    setPlan(null);
    setRunId(null);
    setResult(null);
    setRequest(null);
  };

  return (
    <main className="min-h-screen w-full flex flex-col items-center px-6 py-10">
      <header className="w-full max-w-5xl mb-8 flex items-center gap-3">
        <div className="text-2xl font-bold tracking-tight">
          Auto<span className="text-blue-400">FT</span>
        </div>
        <div className="text-sm text-slate-500">
          autonomous fine-tuning on Modal
        </div>
        <Steps phase={phase} />
      </header>

      {error && (
        <div className="max-w-3xl w-full mb-4 bg-rose-500/10 border border-rose-500/40 text-rose-200 rounded-lg p-3 text-sm">
          {error}
        </div>
      )}

      {phase === "input" && <TaskForm onSubmit={handleResearch} loading={loading} />}
      {phase === "plan" && plan && (
        <PlanPreview
          plan={plan}
          onApprove={handleTrain}
          onBack={() => setPhase("input")}
          starting={loading}
        />
      )}
      {phase === "training" && runId && (
        <Dashboard runId={runId} onDone={handleDone} />
      )}
      {phase === "done" && result && (
        <Comparison result={result} onRestart={restart} />
      )}
    </main>
  );
}

function Steps({ phase }: { phase: Phase }) {
  const all: { id: Phase; label: string }[] = [
    { id: "input", label: "Describe" },
    { id: "plan", label: "Plan" },
    { id: "training", label: "Train" },
    { id: "done", label: "Compare" },
  ];
  const idx = all.findIndex((s) => s.id === phase);
  return (
    <div className="ml-auto flex items-center gap-2 text-xs">
      {all.map((s, i) => (
        <div key={s.id} className="flex items-center gap-2">
          <div
            className={
              i <= idx
                ? "px-2 py-1 rounded bg-blue-500/30 text-blue-100 border border-blue-500/50"
                : "px-2 py-1 rounded bg-slate-800/50 text-slate-500 border border-slate-700"
            }
          >
            {s.label}
          </div>
          {i < all.length - 1 && <div className="text-slate-700">→</div>}
        </div>
      ))}
    </div>
  );
}

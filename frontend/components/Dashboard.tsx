"use client";
import { useEffect, useRef, useState } from "react";
import { Card, LineChart } from "@tremor/react";
import type { RunResult, RunStatus, StepMetric } from "@/lib/types";
import { streamUrl } from "@/lib/api";

interface Props {
  runId: string;
  onDone: (result: RunResult) => void;
}

type Point = { step: number; loss: number; lossSmoothed: number };

export function Dashboard({ runId, onDone }: Props) {
  const [points, setPoints] = useState<Point[]>([]);
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [latest, setLatest] = useState<StepMetric | null>(null);
  const emaRef = useRef<number | null>(null);

  useEffect(() => {
    const es = new EventSource(streamUrl(runId));
    es.addEventListener("status", (e) => {
      try {
        const s: RunStatus = JSON.parse((e as MessageEvent).data);
        setStatus(s);
      } catch {}
    });
    es.addEventListener("metric", (e) => {
      try {
        const m: StepMetric = JSON.parse((e as MessageEvent).data);
        if (m.loss == null) return;
        const loss = m.loss;
        emaRef.current =
          emaRef.current == null ? loss : 0.7 * emaRef.current + 0.3 * loss;
        const smoothed = emaRef.current;
        setLatest(m);
        setPoints((prev) => [
          ...prev,
          { step: m.step, loss, lossSmoothed: smoothed },
        ]);
      } catch {}
    });
    es.addEventListener("result", (e) => {
      try {
        const r: RunResult = JSON.parse((e as MessageEvent).data);
        onDone(r);
        es.close();
      } catch {}
    });
    es.onerror = () => {
      // Allow client-side reconnects; close on terminal state via 'result' event.
    };
    return () => es.close();
  }, [runId, onDone]);

  const lr = latest?.learning_rate;
  const grad = latest?.grad_norm;
  const elapsed = latest?.elapsed_seconds;

  return (
    <div className="max-w-5xl w-full space-y-4">
      <div className="flex items-center gap-3">
        <StatePill state={status?.state ?? "pending"} />
        <div className="text-sm text-slate-400">
          {status?.message ?? "Waiting for first metrics…"}
        </div>
        <div className="ml-auto text-xs text-slate-500 font-mono">{runId}</div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KPI label="Step" value={latest ? String(latest.step) : "—"} />
        <KPI label="Loss" value={latest?.loss != null ? latest.loss.toFixed(4) : "—"} />
        <KPI label="LR" value={lr != null ? lr.toExponential(2) : "—"} />
        <KPI label="Elapsed" value={elapsed != null ? `${Math.round(elapsed)}s` : "—"} />
      </div>

      <Card className="bg-slate-900/40 border border-slate-700 rounded-xl">
        <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">Loss</div>
        <LineChart
          className="h-72"
          data={points}
          index="step"
          categories={["loss", "lossSmoothed"]}
          colors={["blue", "emerald"]}
          showLegend={true}
          showGridLines={true}
          curveType="monotone"
          yAxisWidth={60}
          autoMinValue
        />
      </Card>

      <div className="grid grid-cols-2 gap-3">
        <Card className="bg-slate-900/40 border border-slate-700 rounded-xl">
          <div className="text-xs uppercase tracking-wider text-slate-400 mb-1">Gradient norm</div>
          <div className="text-2xl text-slate-100 font-mono">
            {grad != null ? grad.toFixed(3) : "—"}
          </div>
        </Card>
        <Card className="bg-slate-900/40 border border-slate-700 rounded-xl">
          <div className="text-xs uppercase tracking-wider text-slate-400 mb-1">Final loss</div>
          <div className="text-2xl text-slate-100 font-mono">
            {points.length > 0
              ? points[points.length - 1].lossSmoothed.toFixed(4)
              : "—"}
          </div>
        </Card>
      </div>
    </div>
  );
}

function KPI({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-900/40 border border-slate-700 rounded-lg p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="text-xl font-mono text-slate-100 mt-1">{value}</div>
    </div>
  );
}

function StatePill({ state }: { state: string }) {
  const map: Record<string, string> = {
    pending: "bg-slate-700/40 text-slate-300 border-slate-600",
    loading: "bg-amber-500/20 text-amber-200 border-amber-500/40",
    training: "bg-blue-500/20 text-blue-200 border-blue-500/40",
    evaluating: "bg-violet-500/20 text-violet-200 border-violet-500/40",
    done: "bg-emerald-500/20 text-emerald-200 border-emerald-500/40",
    failed: "bg-rose-500/20 text-rose-200 border-rose-500/40",
  };
  return (
    <span className={`text-xs uppercase tracking-wider border rounded-full px-3 py-1 ${map[state] ?? map.pending}`}>
      {state}
    </span>
  );
}

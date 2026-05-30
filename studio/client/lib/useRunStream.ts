import { useEffect, useRef, useState } from "react";
import type { RunResult, RunStatus, StepMetric } from "@shared/types";
import { streamUrl } from "./api";

export interface SampleLine { step: number; text: string }
export interface LossPoint { step: number; loss: number; smooth: number }

export interface RunStream {
  status: RunStatus | null;
  latest: StepMetric | null;
  points: LossPoint[];
  samples: SampleLine[];
  result: RunResult | null;
}

// Subscribes to the BFF SSE for a run. Uses native EventSource (GET) — works
// for both mock and the Modal proxy. EMA-smooths the loss client-side.
export function useRunStream(runId: string | null): RunStream {
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [latest, setLatest] = useState<StepMetric | null>(null);
  const [points, setPoints] = useState<LossPoint[]>([]);
  const [samples, setSamples] = useState<SampleLine[]>([]);
  const [result, setResult] = useState<RunResult | null>(null);
  const ema = useRef<number | null>(null);

  useEffect(() => {
    if (!runId) return;
    setStatus(null); setLatest(null); setPoints([]); setSamples([]); setResult(null);
    ema.current = null;

    const es = new EventSource(streamUrl(runId));
    es.addEventListener("status", (e) => {
      try { setStatus(JSON.parse((e as MessageEvent).data)); } catch {}
    });
    es.addEventListener("metric", (e) => {
      try {
        const m: StepMetric = JSON.parse((e as MessageEvent).data);
        if (m.loss == null) return;
        ema.current = ema.current == null ? m.loss : 0.7 * ema.current + 0.3 * m.loss;
        const smooth = ema.current;
        setLatest(m);
        setPoints((p) => [...p, { step: m.step, loss: m.loss!, smooth }]);
      } catch {}
    });
    es.addEventListener("sample", (e) => {
      try {
        const s = JSON.parse((e as MessageEvent).data) as { step: number; text: string };
        setSamples((prev) => [...prev, { step: s.step, text: s.text }]);
      } catch {}
    });
    es.addEventListener("result", (e) => {
      try { setResult(JSON.parse((e as MessageEvent).data)); es.close(); } catch {}
    });
    es.onerror = () => { /* terminal close handled via result event */ };
    return () => es.close();
  }, [runId]);

  return { status, latest, points, samples, result };
}

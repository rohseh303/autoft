import { useEffect, useRef, useState } from "react";
import type { RunEvent, RunResult, RunStatus, StepMetric } from "@shared/types";
import { eventsUrl, logsUrl, metricsUrl } from "./api";

export interface SampleLine { step: number; text: string }
export interface LossPoint { step: number; loss: number; smooth: number }
export interface MetricPoint { step: number; value: number }
export interface LogLine { line: string; stream: string }

// Everything the training scene reads, in one snapshot. (Named RunStream for
// continuity — it's now poll-driven, not SSE, but the consumers are unchanged.)
export interface RunStream {
  status: RunStatus | null;
  latest: StepMetric | null;
  points: LossPoint[];        // loss: raw + EMA-smoothed
  lrPoints: MetricPoint[];    // learning-rate series
  gradPoints: MetricPoint[];  // grad-norm series
  samples: SampleLine[];      // live re-samples (mock only; real has none mid-train)
  events: RunEvent[];         // source-tagged transparency timeline
  result: RunResult | null;
}

const EMPTY: RunStream = {
  status: null, latest: null, points: [], lrPoints: [], gradPoints: [],
  samples: [], events: [], result: null,
};

interface MetricsSnapshotWire {
  status: RunStatus | null;
  metrics: StepMetric[];
  result: RunResult | null;
  samples?: SampleLine[] | null;
}

// Build the chart-ready series from the cumulative metrics array. EMA is
// recomputed over the whole array each poll, so it's fully deterministic — no
// reliance on previous render state, which keeps polling idempotent.
function deriveSeries(metrics: StepMetric[]) {
  const points: LossPoint[] = [];
  const lrPoints: MetricPoint[] = [];
  const gradPoints: MetricPoint[] = [];
  let ema: number | null = null;
  for (const m of metrics) {
    if (m.loss != null) {
      ema = ema == null ? m.loss : 0.7 * ema + 0.3 * m.loss;
      points.push({ step: m.step, loss: m.loss, smooth: ema });
    }
    if (m.learning_rate != null) lrPoints.push({ step: m.step, value: m.learning_rate });
    if (m.grad_norm != null) gradPoints.push({ step: m.step, value: m.grad_norm });
  }
  return { points, lrPoints, gradPoints };
}

// Polls /api/run/:id/metrics + /events together in one loop ("in a thread,
// continuously") until the run reaches a terminal state. Resilient to transient
// fetch errors — a failed tick just retries on the next interval.
export function useRunMonitor(runId: string | null, intervalMs = 900): RunStream {
  const [snap, setSnap] = useState<RunStream>(EMPTY);

  useEffect(() => {
    if (!runId) { setSnap(EMPTY); return; }
    setSnap(EMPTY);
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function tick() {
      let terminal = false;
      try {
        const [mRes, eRes] = await Promise.all([
          fetch(metricsUrl(runId!)),
          fetch(eventsUrl(runId!)),
        ]);
        if (!alive) return;
        const snapData = mRes.ok ? (await mRes.json()) as MetricsSnapshotWire : null;
        const evData = eRes.ok ? (await eRes.json()) as { events: RunEvent[] } : null;
        if (!alive) return;

        setSnap((prev) => {
          const next = { ...prev };
          if (snapData) {
            const metrics = snapData.metrics ?? [];
            const { points, lrPoints, gradPoints } = deriveSeries(metrics);
            next.points = points;
            next.lrPoints = lrPoints;
            next.gradPoints = gradPoints;
            next.latest = metrics.length ? metrics[metrics.length - 1]! : prev.latest;
            next.status = snapData.status ?? prev.status;
            next.result = snapData.result ?? prev.result;
            next.samples = (snapData.samples ?? []).map((s) => ({ step: s.step, text: s.text }));
          }
          if (evData?.events) next.events = evData.events;
          return next;
        });

        const state = snapData?.status?.state;
        // Stop once we have a final result (or the run failed). The "&& result"
        // guard avoids stopping on a `done` status before result.json settles.
        terminal = state === "failed" || (state === "done" && snapData?.result != null);
      } catch {
        /* transient — fall through and reschedule */
      }
      if (alive && !terminal) timer = setTimeout(tick, intervalMs);
    }

    tick();
    return () => { alive = false; if (timer) clearTimeout(timer); };
  }, [runId, intervalMs]);

  return snap;
}

// Streams the live container log (SSE `log` events) for the log pane. EventSource
// is GET-only, which is exactly what /api/run/:id/logs serves. Capped so a long
// run can't grow the DOM unbounded.
export function useRunLogs(runId: string | null, cap = 600): { lines: LogLine[]; done: boolean } {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [done, setDone] = useState(false);
  const closed = useRef(false);

  useEffect(() => {
    if (!runId) { setLines([]); setDone(false); return; }
    setLines([]); setDone(false);
    closed.current = false;

    const es = new EventSource(logsUrl(runId));
    es.addEventListener("log", (e) => {
      try {
        const d = JSON.parse((e as MessageEvent).data) as { line: string; stream?: string };
        setLines((prev) => {
          const next = [...prev, { line: d.line, stream: d.stream ?? "stdout" }];
          return next.length > cap ? next.slice(next.length - cap) : next;
        });
      } catch { /* ignore malformed frame */ }
    });
    es.addEventListener("eof", () => { setDone(true); closed.current = true; es.close(); });
    es.onerror = () => { if (closed.current) es.close(); /* else let it retry */ };

    return () => { closed.current = true; es.close(); };
  }, [runId, cap]);

  return { lines, done };
}

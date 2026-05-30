import { useState } from "react";
import type { RunPlan } from "@shared/types";
import type { RunStream } from "../lib/useRunStream";
import { useRunLogs } from "../lib/useRunStream";
import { MetricCharts } from "./MetricCharts";
import { ReasoningTimeline } from "./ReasoningTimeline";
import { LogStream } from "./LogStream";

// The dark "Mission Control" training stage. Three heroes now: the metric
// small-multiples (loss / lr / grad), a transparent timeline of what every
// subagent is doing, and — the showpiece — the model's own words sharpening.
// Metrics + events are polled (owned by App, shared with the Deck); the raw log
// is streamed here since it's theater-only.
export function Theater({ plan, stream, runId }: { plan: RunPlan; stream: RunStream; runId: string | null }) {
  const { status, latest, points, lrPoints, gradPoints, samples, events } = stream;
  const [tab, setTab] = useState<"reasoning" | "logs">("reasoning");
  const logs = useRunLogs(runId);

  const state = status?.state ?? "pending";
  const pct = latest ? Math.round((latest.step / plan.training.max_steps) * 100) : 0;
  const newest = samples[samples.length - 1];

  return (
    <div className="theater">
      <div className="th-head">
        <span className="live-dot" /> <span className="kicker" style={{ color: "var(--ghost-2)" }}>training · {state}</span>
        <span className="th-meta mono">
          {latest ? `step ${latest.step}/${plan.training.max_steps}` : "warming up…"}
          {latest?.epoch != null && ` · epoch ${latest.epoch.toFixed(2)}`}
          {latest?.elapsed_seconds != null && ` · ${latest.elapsed_seconds.toFixed(0)}s`}
        </span>
      </div>

      <div className="th-progress"><div className="th-bar" style={{ width: `${pct}%` }} /></div>
      {state === "loading" && <div className="th-status mono">{status?.message}<span className="caret" /></div>}

      <MetricCharts points={points} lrPoints={lrPoints} gradPoints={gradPoints} latest={latest} maxStep={plan.training.max_steps} />

      <div className="th-grid">
        <div className="th-panel th-transparency">
          <div className="th-panel-h">
            <div className="th-tabs">
              <button className={`th-tab${tab === "reasoning" ? " on" : ""}`} onClick={() => setTab("reasoning")}>
                reasoning <span className="th-tab-n mono">{events.length}</span>
              </button>
              <button className={`th-tab${tab === "logs" ? " on" : ""}`} onClick={() => setTab("logs")}>
                logs <span className="th-tab-n mono">{logs.lines.length}</span>
              </button>
            </div>
            <span className="kicker">transparency</span>
          </div>
          {tab === "reasoning"
            ? <ReasoningTimeline events={events} />
            : <LogStream lines={logs.lines} done={logs.done} />}
        </div>

        <div className="th-panel th-words">
          <div className="th-panel-h">
            <span className="kicker">what your model says now</span>
            {newest && <span className="mono th-step">@ step {newest.step}</span>}
          </div>
          <div className="words-now">
            {newest ? <span className="phosphor caret">{newest.text}</span> : <span className="ghost-dim">waiting for first sample…</span>}
          </div>
          <div className="words-log">
            {samples.slice(0, -1).reverse().map((s) => (
              <div key={s.step} className="wl-row">
                <span className="wl-step mono">{String(s.step).padStart(3, "0")}</span>
                <span className="wl-text">{s.text}</span>
              </div>
            ))}
          </div>
          <div className="words-cap mono">re-sampled live during training — watch it learn</div>
        </div>
      </div>
    </div>
  );
}

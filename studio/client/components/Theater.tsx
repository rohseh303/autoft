import type { RunPlan } from "@shared/types";
import type { RunStream } from "../lib/useRunStream";
import { LossChart } from "./LossChart";

// The dark "Mission Control" training stage. Two heroes: the loss curve and —
// the showpiece — the model's own words sharpening from gibberish to coherent.
// Purely presentational: the stream is owned by App (so the Deck shares it).
export function Theater({ plan, stream }: { plan: RunPlan; stream: RunStream }) {
  const { status, latest, points, samples } = stream;

  const state = status?.state ?? "pending";
  const pct = latest ? Math.round((latest.step / plan.training.max_steps) * 100) : 0;
  const newest = samples[samples.length - 1];

  return (
    <div className="theater">
      <div className="th-head">
        <span className="live-dot" /> <span className="kicker" style={{ color: "var(--ghost-2)" }}>training · {state}</span>
        <span className="th-meta mono">
          {latest ? `step ${latest.step}/${plan.training.max_steps}` : "warming up…"}
          {latest?.elapsed_seconds != null && ` · ${latest.elapsed_seconds.toFixed(0)}s`}
        </span>
      </div>

      <div className="th-progress"><div className="th-bar" style={{ width: `${pct}%` }} /></div>
      {state === "loading" && <div className="th-status mono">{status?.message}<span className="caret" /></div>}

      <div className="th-grid">
        <div className="th-panel th-loss">
          <div className="th-panel-h"><span className="kicker">loss</span>
            <span className="mono th-now">{latest?.loss != null ? latest.loss.toFixed(4) : "—"}</span>
          </div>
          <LossChart points={points} maxStep={plan.training.max_steps} />
          <div className="th-kpis">
            <Kpi label="grad norm" v={latest?.grad_norm != null ? latest.grad_norm.toFixed(3) : "—"} />
            <Kpi label="lr" v={latest?.learning_rate != null ? latest.learning_rate.toExponential(1) : "—"} />
            <Kpi label="epoch" v={latest?.epoch != null ? latest.epoch.toFixed(2) : "—"} />
          </div>
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

function Kpi({ label, v }: { label: string; v: string }) {
  return <div className="kpi"><span className="kicker">{label}</span><span className="mono kpi-v">{v}</span></div>;
}

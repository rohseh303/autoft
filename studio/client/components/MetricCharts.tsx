import type { ReactNode } from "react";
import type { StepMetric } from "@shared/types";
import type { LossPoint, MetricPoint } from "../lib/useRunStream";
import { LossChart } from "./LossChart";

// wandb-style small-multiples — loss (raw + EMA), learning rate, grad norm.
// Hand-rolled SVG on purpose: the project keeps zero chart deps so the bundle
// stays tiny and the look stays ours (same call as LossChart).
export function MetricCharts({ points, lrPoints, gradPoints, latest, maxStep }: {
  points: LossPoint[];
  lrPoints: MetricPoint[];
  gradPoints: MetricPoint[];
  latest: StepMetric | null;
  maxStep: number;
}) {
  return (
    <div className="mc-row">
      <MetricPanel label="loss" value={latest?.loss != null ? latest.loss.toFixed(4) : "—"} accent>
        <LossChart points={points} maxStep={maxStep} />
      </MetricPanel>
      <MetricPanel label="learning rate" value={latest?.learning_rate != null ? latest.learning_rate.toExponential(1) : "—"}>
        <MiniChart points={lrPoints} maxStep={maxStep} />
      </MetricPanel>
      <MetricPanel label="grad norm" value={latest?.grad_norm != null ? latest.grad_norm.toFixed(3) : "—"}>
        <MiniChart points={gradPoints} maxStep={maxStep} />
      </MetricPanel>
    </div>
  );
}

function MetricPanel({ label, value, accent, children }: {
  label: string; value: string; accent?: boolean; children: ReactNode;
}) {
  return (
    <div className={`th-panel mc-panel${accent ? " mc-accent" : ""}`}>
      <div className="th-panel-h">
        <span className="kicker">{label}</span>
        <span className="mono th-now mc-now">{value}</span>
      </div>
      {children}
    </div>
  );
}

// Single-series line chart (no smoothing) for lr / grad_norm.
function MiniChart({ points, maxStep }: { points: MetricPoint[]; maxStep: number }) {
  const W = 520, H = 132, pad = 8;
  if (points.length === 0) {
    return <svg className="loss-svg mc-svg" viewBox={`0 0 ${W} ${H}`}><text x={W / 2} y={H / 2} className="loss-empty">collecting…</text></svg>;
  }
  const vs = points.map((p) => p.value);
  const maxV = Math.max(...vs) * 1.05;
  const minV = Math.min(...vs) * 0.95;
  const x = (step: number) => pad + (step / maxStep) * (W - pad * 2);
  const y = (v: number) => pad + (1 - (v - minV) / (maxV - minV || 1)) * (H - pad * 2);
  const d = points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.step).toFixed(1)} ${y(p.value).toFixed(1)}`).join(" ");
  const last = points[points.length - 1]!;
  return (
    <svg className="loss-svg mc-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <path d={d} className="loss-smooth" />
      <circle cx={x(last.step)} cy={y(last.value)} r="3" className="loss-head" />
    </svg>
  );
}

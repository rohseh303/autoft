import type { LossPoint } from "../lib/useRunStream";

// Dependency-free SVG loss chart (raw + EMA-smoothed). No chart lib — keeps the
// bundle tiny and the look fully ours.
export function LossChart({ points, maxStep }: { points: LossPoint[]; maxStep: number }) {
  const W = 520, H = 200, pad = 8;
  if (points.length === 0) {
    return <svg className="loss-svg" viewBox={`0 0 ${W} ${H}`}><text x={W / 2} y={H / 2} className="loss-empty">collecting…</text></svg>;
  }
  const losses = points.map((p) => p.loss);
  const maxL = Math.max(...losses) * 1.05;
  const minL = Math.min(...losses) * 0.95;
  const x = (step: number) => pad + (step / maxStep) * (W - pad * 2);
  const y = (l: number) => pad + (1 - (l - minL) / (maxL - minL || 1)) * (H - pad * 2);
  const path = (key: "loss" | "smooth") =>
    points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.step).toFixed(1)} ${y(p[key]).toFixed(1)}`).join(" ");
  const last = points[points.length - 1]!;

  return (
    <svg className="loss-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <path d={path("loss")} className="loss-raw" />
      <path d={path("smooth")} className="loss-smooth" />
      <circle cx={x(last.step)} cy={y(last.smooth)} r="3.5" className="loss-head" />
    </svg>
  );
}

import type { RunPlan, RunResult } from "@shared/types";
import type { Phase, Thought } from "../App";
import type { RunStream } from "../lib/useRunStream";

// H200 on Modal is ~$3.95/hr. We only ever show this as an ESTIMATE — in demo
// mode nothing is billed, and even live we don't know the exact rate. Honesty
// is the point (the Critic's note): the chip always reads "est.".
const GPU_PER_SEC = 3.95 / 3600;

// The persistent Telemetry Deck — pinned to the bottom in every scene. The
// through-line of the whole app: always present, always alive, never in the way.
export function Deck({ phase, mode, plan, thoughts, stream, result }: {
  phase: Phase; mode: "mock" | "modal"; plan: RunPlan | null;
  thoughts: Thought[]; stream: RunStream; result: RunResult | null;
}) {
  const { latest, points, status } = stream;
  const max = plan?.training.max_steps ?? 150;

  const elapsed =
    phase === "done" ? estTotalSeconds(plan)
    : latest?.elapsed_seconds ?? 0;
  const cost = elapsed * GPU_PER_SEC;
  const projected = estTotalSeconds(plan) * GPU_PER_SEC; // full-run projection

  const left = (() => {
    switch (phase) {
      case "research": return { tag: "research", msg: `agent · ${thoughts.length} step${thoughts.length === 1 ? "" : "s"}` };
      case "plan":     return { tag: "plan", msg: "recipe ready · awaiting launch" };
      case "train":    return { tag: status?.state ?? "train", msg: latest ? `step ${latest.step}/${max}` : (status?.message ?? "warming up") };
      case "done":     return { tag: "done", msg: `final loss ${result?.final_loss?.toFixed(3) ?? "—"}` };
      default:         return { tag: "ready", msg: "describe a model to begin" };
    }
  })();

  return (
    <footer className="deck mono" data-phase={phase}>
      <div className="deck-zone deck-left">
        <span className={`deck-tag deck-tag-${phase}`}>{left.tag}</span>
        <span className="deck-msg">{left.msg}</span>
      </div>

      <div className="deck-zone deck-center">
        <Pulse phase={phase} points={points} max={max} thoughts={thoughts.length} />
      </div>

      <div className="deck-zone deck-right">
        {(phase === "train" || phase === "done") && (
          <>
            <Stat label="elapsed" v={fmtTime(elapsed)} />
            <Stat label="cost est." v={`$${cost.toFixed(3)}`} hint={phase === "train" && projected > 0 ? `→ ~$${projected.toFixed(2)}` : undefined} />
            <span className="deck-sep" />
          </>
        )}
        <Stat label="gpu" v="H200 · 141GB" />
        <span className="deck-sep" />
        <span className={`deck-mode deck-mode-${mode}`}>{mode === "mock" ? "demo" : "live"}</span>
      </div>
    </footer>
  );
}

function Stat({ label, v, hint }: { label: string; v: string; hint?: string }) {
  return (
    <span className="deck-stat">
      <span className="deck-stat-l">{label}</span>
      <span className="deck-stat-v">{v}{hint && <span className="deck-stat-hint"> {hint}</span>}</span>
    </span>
  );
}

// Contextual mini "heartbeat": breathing when idle, ticks while the agent
// thinks, a live loss line while training. Glanceable — never the main chart.
function Pulse({ phase, points, max, thoughts }: {
  phase: Phase; points: RunStream["points"]; max: number; thoughts: number;
}) {
  const W = 220, H = 26, mid = H / 2;

  if (phase === "train" || phase === "done") {
    if (points.length < 2) return <Wave W={W} H={H} />;
    const ls = points.map((p) => p.smooth);
    const lo = Math.min(...ls), hi = Math.max(...ls);
    const x = (s: number) => (s / max) * W;
    const y = (l: number) => 3 + (1 - (l - lo) / (hi - lo || 1)) * (H - 6);
    const d = points.map((p, i) => `${i ? "L" : "M"} ${x(p.step).toFixed(1)} ${y(p.smooth).toFixed(1)}`).join(" ");
    const last = points[points.length - 1]!;
    return (
      <svg className="pulse" width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden>
        <path d={d} className="pulse-loss" />
        <circle cx={x(last.step)} cy={y(last.smooth)} r="2.5" className="pulse-dot" />
      </svg>
    );
  }

  if (phase === "research") {
    return (
      <svg className="pulse" width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden>
        <line x1="0" y1={mid} x2={W} y2={mid} className="pulse-base" />
        {Array.from({ length: 5 }).map((_, i) => (
          <line key={i} x1={20 + i * 44} y1={mid - (i < thoughts ? 8 : 1)} x2={20 + i * 44} y2={mid + (i < thoughts ? 8 : 1)}
            className={i < thoughts ? "pulse-tick on" : "pulse-tick"} />
        ))}
      </svg>
    );
  }

  return <Wave W={W} H={H} />;
}

// Idle breathing line.
function Wave({ W, H }: { W: number; H: number }) {
  const mid = H / 2;
  return (
    <svg className="pulse pulse-idle" width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden>
      <path d={`M0 ${mid} Q ${W * 0.25} ${mid - 5}, ${W * 0.5} ${mid} T ${W} ${mid}`} className="pulse-breathe" />
    </svg>
  );
}

function fmtTime(s: number): string {
  if (s < 60) return `${s.toFixed(0)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60).toString().padStart(2, "0")}s`;
}
// crude wall-clock estimate of a full run for the cost projection (loading + steps + eval)
function estTotalSeconds(plan: RunPlan | null): number {
  const steps = plan?.training.max_steps ?? 150;
  return 90 + steps * 1.0 + 35; // load + ~1s/step on H200 (2B) + eval — rough
}

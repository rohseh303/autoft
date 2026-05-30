import { useRef, useState } from "react";
import type { Measure, RunPlan, RunResult } from "@shared/types";
import type { LossPoint } from "../lib/useRunStream";
import { downloadUrl } from "../lib/api";

// The payoff — drag-to-reveal before/after on the user's own examples. Back to
// the light skin: the win should feel like daylight after the dark theater.
export function Reveal({ result, points, onReset }: { result: RunResult; points: LossPoint[]; onReset: () => void }) {
  return (
    <div className="reveal rise">
      <div className="rv-head">
        <span className="kicker">done · final loss {result.final_loss?.toFixed(4) ?? "—"}</span>
        <h2>Your model vs. the base model</h2>
        <p className="rv-sub">Drag the divider. Left is the stock model; right is the one you just trained.</p>
        <div className="rv-actions">
          <a className="rv-export" href={downloadUrl(result.run_id)} download>
            ↓ Export model <span className="rv-export-sub">LoRA adapter ·.zip</span>
          </a>
          <button className="rv-again" onClick={onReset}>Train another →</button>
        </div>
      </div>
<<<<<<< Updated upstream
      {result.comparisons.map((c, i) => (
        <Compare key={i} input={c.input} base={c.base_output} ft={c.finetuned_output} expected={c.expected_output} />
      ))}

      {result.scorecard && <ScoreCard s={result.scorecard} />}
=======
      {result.comparisons.length > 0 ? (
        result.comparisons.map((c, i) => (
          <Compare
            key={i}
            input={c.input}
            base={c.base_output}
            ft={c.finetuned_output}
            expected={c.expected_output}
            judgeScore={c.judge_score}
            critique={c.judge_critique}
          />
        ))
      ) : (
        <div className="cmp-empty card-hard">
          No test prompt was given, so there's no before/after to show. Add one on the
          plan screen next time to see your model side-by-side with the base.
        </div>
      )}

      <ScoreCard result={result} />
>>>>>>> Stashed changes

      <TechCard plan={result.plan} points={points} finalLoss={result.final_loss} />
    </div>
  );
}

<<<<<<< Updated upstream
// The evidence — a blind-preference verdict + soft base-vs-yours measures.
// Warm and editorial on purpose: a report card, not a dashboard.
function ScoreCard({ s }: { s: Scorecard }) {
=======
function verdictFor(judge: number): string {
  return judge >= 9 ? "The judge rated your model excellent."
    : judge >= 7 ? "The judge clearly preferred your model."
    : judge >= 5 ? "The judge found your model solid but improvable."
    : "The judge sees room to grow — try more steps or a cleaner dataset.";
}

// The evidence. Works for BOTH paths:
//  - mock/sim: uses the synthesized scorecard (verdict, pips, fabricated measures)
//  - real Modal: derives verdict + pips from the real LLM-judge score, shows
//    judge/eval_loss/train_loss/objective stats, never fabricates measure bars.
function ScoreCard({ result }: { result: RunResult }) {
  const mock = result.scorecard ?? null;
  const judge = result.judge_score ?? null;

  if (!mock && judge == null && result.eval_loss == null) return null;

  const total = mock?.judge_total ?? 10;
  const wins = mock ? mock.judge_wins : judge != null ? Math.round(judge) : null;
  const verdict = mock?.verdict ?? (judge != null ? verdictFor(judge) : "Training complete.");
  const simulated = mock?.simulated ?? false;

  const stats: { label: string; v: string }[] = [];
  if (judge != null) stats.push({ label: "judge score", v: `${judge.toFixed(1)}/10` });
  if (result.eval_loss != null) stats.push({ label: "eval loss", v: result.eval_loss.toFixed(3) });
  if (result.final_loss != null) stats.push({ label: "train loss", v: result.final_loss.toFixed(3) });
  if (result.objective != null) stats.push({ label: "objective", v: result.objective.toFixed(2) });

>>>>>>> Stashed changes
  return (
    <section className="score card-hard rise">
      <span className="kicker">how it performed</span>
      <h3 className="score-verdict">{verdict}</h3>

<<<<<<< Updated upstream
      <div className="judge">
        <div className="judge-pips" aria-hidden>
          {Array.from({ length: s.judge_total }).map((_, i) => (
            <span key={i} className={`pip ${i < s.judge_wins ? "won" : ""}`} />
          ))}
=======
      {stats.length > 0 && (
        <div className="judge-stats">
          {stats.map((st) => (
            <div key={st.label} className="js-stat">
              <span className="js-v mono">{st.v}</span>
              <span className="js-l">{st.label}</span>
            </div>
          ))}
        </div>
      )}

      {wins != null && (
        <div className="judge">
          <div className="judge-pips" aria-hidden>
            {Array.from({ length: total }).map((_, i) => (
              <span key={i} className={`pip ${i < wins ? "won" : ""}`} />
            ))}
          </div>
          <p className="judge-text">judge score <strong>{wins}/{total}</strong></p>
>>>>>>> Stashed changes
        </div>
      )}

      {mock && mock.measures.length > 0 && (
        <div className="measures">
          {mock.measures.map((m) => <MeasureRow key={m.name} m={m} />)}
        </div>
      )}

      <p className="score-note">
<<<<<<< Updated upstream
        {s.simulated
          ? "Illustrative scores — demo mode simulates the eval. Wire the backend judge + ROUGE harness for real numbers."
          : "Measured on your eval examples against the untuned base model."}
=======
        {simulated
          ? "Illustrative scores — demo mode simulates the eval. The real backend's LLM judge + held-out eval fill these in on a live run."
          : "Scored by the LLM judge on your test prompt against the untuned base model."}
>>>>>>> Stashed changes
      </p>
    </section>
  );
}

function MeasureRow({ m }: { m: Measure }) {
  const fmt = (v: number) => (m.unit === "%" ? `${v}%` : v.toFixed(2));
  const delta = m.finetuned - m.base;
  const pct = (v: number) => `${Math.max(2, Math.min(100, (v / m.scale) * 100))}%`;
  return (
    <div className="measure">
      <div className="measure-top">
        <span className="measure-name">{m.name}</span>
        <span className="measure-delta">+{fmt(Math.round(delta * 100) / 100)}</span>
      </div>
      <div className="bar-row">
        <span className="bar-tag">base</span>
        <div className="track"><div className="fill base" style={{ width: pct(m.base) }} /></div>
        <span className="bar-val mono">{fmt(m.base)}</span>
      </div>
      <div className="bar-row">
        <span className="bar-tag yours">yours</span>
        <div className="track"><div className="fill ft" style={{ width: pct(m.finetuned) }} /></div>
        <span className="bar-val mono">{fmt(m.finetuned)}</span>
      </div>
    </div>
  );
}

// Under the hood — the real training loss recap (we have the actual per-step
// points from the run) plus a compact strip of model internals. Same warm card;
// the graph carries the "technical" weight without turning into a console.
function TechCard({ plan, points, finalLoss }: {
  plan: RunPlan; points: LossPoint[]; finalLoss: number | null;
}) {
  const t = plan.training;
  const trainable = trainableParams(plan.base_model, t.lora_r);
  const totalP = baseParams(plan.base_model);
  const pctTrainable = (trainable / totalP) * 100;
  const startLoss = points[0]?.loss ?? null;
  const drop = startLoss != null && finalLoss != null ? ((startLoss - finalLoss) / startLoss) * 100 : null;

  return (
    <section className="tech card-hard rise">
      <span className="kicker">under the hood</span>
      <h3 className="tech-title">How it trained</h3>

      <div className="tech-graph">
        <RecapChart points={points} />
        <div className="tech-graph-cap">
          <span className="mono"><span className="dot-loss" /> training loss · {points.length || t.max_steps} steps</span>
          {drop != null && <span className="mono tech-drop">↓ {drop.toFixed(0)}% from first step</span>}
        </div>
      </div>

      <div className="tech-specs">
        <Spec label="trainable params" v={fmtParams(trainable)} sub={`${pctTrainable.toFixed(2)}% of ${fmtParams(totalP)}`} />
        <Spec label="adapter" v={`LoRA r=${t.lora_r}`} sub={`α=${t.lora_alpha} · 7 proj layers`} />
        <Spec label="precision" v="4-bit QLoRA" sub="bf16 compute" />
        <Spec label="effective batch" v={String(t.batch_size * t.gradient_accumulation_steps)} sub={`${t.batch_size} × ${t.gradient_accumulation_steps} accum`} />
        <Spec label="optimizer" v="adamw_8bit" sub={`lr ${t.learning_rate.toExponential(0)} · linear`} />
        <Spec label="seq length" v={`${t.max_seq_length}`} sub={`${t.warmup_steps} warmup steps`} />
      </div>
    </section>
  );
}

// Loss recap — reuses the run's real points; mirrors the theater chart but in
// the light skin so it belongs to the reveal.
function RecapChart({ points }: { points: LossPoint[] }) {
  const W = 600, H = 150, pad = 6;
  if (points.length < 2) {
    return <svg className="recap-svg" viewBox={`0 0 ${W} ${H}`}><text x={W / 2} y={H / 2} className="recap-empty">loss data unavailable on reload</text></svg>;
  }
  const ls = points.map((p) => p.loss);
  const maxL = Math.max(...ls) * 1.04, minL = Math.min(...ls) * 0.96;
  const maxStep = points[points.length - 1]!.step;
  const x = (s: number) => pad + (s / maxStep) * (W - pad * 2);
  const y = (l: number) => pad + (1 - (l - minL) / (maxL - minL || 1)) * (H - pad * 2);
  const line = (k: "loss" | "smooth") => points.map((p, i) => `${i ? "L" : "M"} ${x(p.step).toFixed(1)} ${y(p[k]).toFixed(1)}`).join(" ");
  const area = `${line("smooth")} L ${x(maxStep).toFixed(1)} ${H - pad} L ${x(0).toFixed(1)} ${H - pad} Z`;
  return (
    <svg className="recap-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <path d={area} className="recap-area" />
      <path d={line("loss")} className="recap-raw" />
      <path d={line("smooth")} className="recap-smooth" />
    </svg>
  );
}

function Spec({ label, v, sub }: { label: string; v: string; sub: string }) {
  return (
    <div className="spec">
      <div className="kicker">{label}</div>
      <div className="spec-v mono">{v}</div>
      <div className="spec-sub mono">{sub}</div>
    </div>
  );
}

// rough param math so the internals read true to the chosen base model
function baseParams(model: string): number {
  return model.includes("1.7B") ? 1.7e9 : 0.5e9;
}
function trainableParams(model: string, r: number): number {
  // LoRA params ≈ 2 · r · d_model · (#target matrices) · #layers — approximated
  const d = model.includes("1.7B") ? 2048 : 896;
  const layers = model.includes("1.7B") ? 24 : 24;
  return 2 * r * d * 7 * layers;
}
function fmtParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}

function Compare({ input, base, ft, expected }: {
  input: string; base: string; ft: string; expected?: string | null;
}) {
  const [split, setSplit] = useState(50);
  const ref = useRef<HTMLDivElement>(null);
  const drag = (clientX: number) => {
    const el = ref.current; if (!el) return;
    const r = el.getBoundingClientRect();
    setSplit(Math.max(4, Math.min(96, ((clientX - r.left) / r.width) * 100)));
  };

  return (
    <div className="cmp card-hard">
      <div className="cmp-in"><span className="kicker">input</span><p>{input}</p></div>

      <div
        className="cmp-stage" ref={ref}
        onPointerMove={(e) => { if (e.buttons === 1) drag(e.clientX); }}
        onPointerDown={(e) => drag(e.clientX)}
      >
        <div className="cmp-side cmp-base">
          <span className="cmp-tag base mono">base</span>
          <p>{base || "—"}</p>
        </div>
        <div className="cmp-side cmp-ft" style={{ clipPath: `inset(0 0 0 ${split}%)` }}>
          <span className="cmp-tag ft mono">your model</span>
          <p>{ft || "—"}</p>
        </div>
        <div className="cmp-divider" style={{ left: `${split}%` }}>
          <span className="cmp-knob">⇄</span>
        </div>
      </div>

      {expected && <div className="cmp-exp"><span className="kicker">expected</span><p>{expected}</p></div>}
    </div>
  );
}

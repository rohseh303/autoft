import type { Phase } from "../App";

const STEPS: { id: Exclude<Phase, "landing">; label: string }[] = [
  { id: "research", label: "Research" },
  { id: "plan", label: "Plan" },
  { id: "train", label: "Train" },
  { id: "done", label: "Compare" },
];

// The living pipeline spine — always visible, shows where you are. Data
// "flows" along the connecting wire toward the active node.
export function Rail({ phase, mode, task, onReset }: {
  phase: Phase; mode: "mock" | "modal"; task: string; onReset: () => void;
}) {
  const order: Phase[] = ["research", "plan", "train", "done"];
  const idx = order.indexOf(phase);

  return (
    <header className="rail">
      <button className="rail-brand" onClick={onReset} title="Start over">
        <span className="dot" /> AutoFT
      </button>

      <div className="rail-track">
        <div className="rail-describe" title={task}>
          <span className="kicker">described</span>
          <span className="rail-task">{task.length > 38 ? task.slice(0, 38) + "…" : task || "—"}</span>
        </div>
        <Wire active={idx >= 0} />
        {STEPS.map((s, i) => {
          const here = s.id === phase;
          const past = order.indexOf(s.id) < idx;
          return (
            <div key={s.id} className="rail-seg">
              <span className={`node ${here ? "node-on" : past ? "node-done" : ""}`}>
                {past ? "✓" : i + 1}
              </span>
              <span className={`node-label ${here ? "on" : ""}`}>{s.label}</span>
              {i < STEPS.length - 1 && <Wire active={order.indexOf(s.id) < idx} />}
            </div>
          );
        })}
      </div>

      <span className="mode-chip mono">{mode === "mock" ? "demo" : "live"}</span>
    </header>
  );
}

function Wire({ active }: { active: boolean }) {
  return (
    <svg className="wire" width="46" height="12" viewBox="0 0 46 12" aria-hidden>
      <line x1="0" y1="6" x2="46" y2="6" className="wire-base" />
      {active && <line x1="0" y1="6" x2="46" y2="6" className="wire-flow" />}
    </svg>
  );
}

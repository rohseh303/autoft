import type { RunEvent } from "@shared/types";

// Per-kind glyph. Mirrors the research Notebook's vocabulary, plus the trainer/
// judge beats (status/progress/sample) the backend emits during a run.
const KIND_ICON: Record<string, string> = {
  status: "•", progress: "▸", note: "·", search: "⌕",
  peek: "◍", decision: "✓", sample: "❝",
};

// The transparent window into the whole pipeline: every reasoning/status beat,
// tagged by which subagent emitted it (research / trainer / judge / lead) and —
// when it's a training event — at which step. This is what makes the Codex
// subprocesses and the trainer legible instead of a black box.
export function ReasoningTimeline({ events }: { events: RunEvent[] }) {
  if (events.length === 0) {
    return <div className="rt-empty mono">no reasoning yet — waiting for the agents…</div>;
  }
  return (
    <div className="rt-list">
      {events.map((e, i) => (
        <div key={i} className={`rt-row rt-src-${e.source}`}>
          <span className="rt-src mono">{e.source}</span>
          <span className="rt-ic mono">{KIND_ICON[e.kind] ?? "·"}</span>
          <div className="rt-body">
            <span className="rt-text">{e.text}</span>
            {e.step != null && <span className="rt-step mono"> @step {e.step}</span>}
            {e.detail && <div className="rt-detail mono">{e.detail}</div>}
          </div>
          {e.ts != null && <span className="rt-ts mono">{e.ts.toFixed(1)}s</span>}
        </div>
      ))}
    </div>
  );
}

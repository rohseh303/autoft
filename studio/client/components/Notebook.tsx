import type { Thought } from "../App";

const ICON: Record<string, string> = { note: "·", search: "⌕", peek: "◍", decision: "✓" };

// The agent "writes" its reasoning live — stream-of-thought as a research memo.
export function Notebook({ thoughts, done }: { thoughts: Thought[]; done: boolean }) {
  return (
    <section className="notebook card">
      <div className="nb-head">
        <span className="kicker">research agent</span>
        <h2>{done ? "Here's the plan." : "Designing your model"}{!done && <span className="caret" />}</h2>
      </div>
      <ol className="nb-list">
        {thoughts.map((t, i) => (
          <li key={i} className={`nb-item rise nb-${t.kind}`} style={{ animationDelay: `${i * 40}ms` }}>
            <span className="nb-ic mono">{ICON[t.kind] ?? "·"}</span>
            <div>
              <div className="nb-text">{t.text}</div>
              {t.detail && <div className="nb-detail mono">{t.detail}</div>}
            </div>
          </li>
        ))}
        {!done && thoughts.length === 0 && (
          <li className="nb-item nb-note"><span className="nb-ic mono">·</span><div className="nb-text">Thinking<span className="caret" /></div></li>
        )}
      </ol>
    </section>
  );
}

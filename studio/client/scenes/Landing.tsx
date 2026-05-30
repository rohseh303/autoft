import { useEffect, useRef, useState } from "react";
import type { UserRequest } from "@shared/types";

const EXAMPLES = [
  "a model that summarizes long legal contracts into 3 plain-English bullet points",
  "a model that turns natural-language questions into SQL queries",
  "a model that turns rambling meeting transcripts into crisp action items",
  "a model that explains Python code line-by-line for total beginners",
];

// self-typing placeholder so the empty box still feels alive
function useTypedPlaceholder(active: boolean): string {
  const [text, setText] = useState("");
  const i = useRef(0); const ci = useRef(0); const dir = useRef<1 | -1>(1);
  useEffect(() => {
    if (!active) return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) { setText(EXAMPLES[0]!); return; }
    const tick = () => {
      const full = EXAMPLES[i.current]!;
      ci.current += dir.current;
      setText(full.slice(0, ci.current));
      if (ci.current >= full.length) { dir.current = -1; return 1400; }
      if (ci.current <= 0) { dir.current = 1; i.current = (i.current + 1) % EXAMPLES.length; return 350; }
      return dir.current === 1 ? 38 : 18;
    };
    let id: ReturnType<typeof setTimeout>;
    const loop = () => { const d = tick(); id = setTimeout(loop, d); };
    id = setTimeout(loop, 600);
    return () => clearTimeout(id);
  }, [active]);
  return text;
}

// The front page: one box, nothing else. Hitting enter morphs into the studio.
export function Landing({
  mode, error, onStart,
}: {
  mode: "mock" | "modal";
  error: string | null;
  onStart: (req: UserRequest) => void;
}) {
  const [task, setTask] = useState("");
  const typed = useTypedPlaceholder(task.length === 0);

  const go = () => {
    const t = task.trim();
    if (!t) return;
    onStart({ task_description: t, eval_examples: [], preferred_model: null });
  };

  return (
    <main className="landing">
      <div className="landing-inner rise">
        <div className="brandmark">
          <span className="dot" /> AutoFT
          <span className="mode-chip mono">{mode === "mock" ? "demo mode" : "live · modal"}</span>
        </div>

        <h1 className="hero">
          Describe a model.<br />
          <em>Watch it get built.</em>
        </h1>
        <p className="sub">
          An agent picks the dataset and recipe, a GPU trains it, and you see proof it works —
          all from one sentence.
        </p>

        <div className="prompt card-hard">
          <textarea
            autoFocus
            value={task}
            onChange={(e) => setTask(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey || !e.shiftKey)) { e.preventDefault(); go(); } }}
            placeholder={`I want ${typed}`}
            rows={2}
          />
          <div className="prompt-row">
            <span className="hint mono">⏎ to design · ⇧⏎ for newline</span>
            <button className="go" onClick={go} disabled={!task.trim()}>
              Design it <span className="arrow">→</span>
            </button>
          </div>
        </div>

        <div className="chips">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="chip" onClick={() => setTask(ex)}>
              {ex.length > 46 ? ex.slice(0, 46) + "…" : ex}
            </button>
          ))}
        </div>

        {error && <div className="err">{error}</div>}
      </div>
    </main>
  );
}

import { useEffect, useState } from "react";
import type { RunPlan, RunResult, UserRequest } from "@shared/types";
import { health, research, researchStream, train } from "./lib/api";
import { useRunStream } from "./lib/useRunStream";
import { Landing } from "./scenes/Landing";
import { Studio } from "./scenes/Studio";
import { Deck } from "./components/Deck";

export type Phase = "landing" | "research" | "plan" | "train" | "done";
export interface Thought { kind: string; text: string; detail?: string }

export function App() {
  const [phase, setPhase] = useState<Phase>("landing");
  const [mode, setMode] = useState<"mock" | "modal">("mock");
  const [req, setReq] = useState<UserRequest | null>(null);
  const [plan, setPlan] = useState<RunPlan | null>(null);
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Run stream lifted to App so the bottom Deck reads live metrics in any scene.
  const stream = useRunStream(runId);

  useEffect(() => { health().then((h) => setMode(h.mode)).catch(() => {}); }, []);

  // dark theater only while training
  useEffect(() => {
    document.body.dataset.theater = phase === "train" ? "on" : "off";
    document.body.dataset.phase = phase;
  }, [phase]);

  // training finished -> hold a beat on the final state, then reveal
  useEffect(() => {
    if (phase === "train" && stream.result) {
      const r = stream.result;
      const t = setTimeout(() => { setResult(r); setPhase("done"); }, 1200);
      return () => clearTimeout(t);
    }
  }, [phase, stream.result]);

  async function start(request: UserRequest) {
    setError(null);
    setReq(request);
    setThoughts([]);
    setPlan(null);
    setPhase("research");
    try {
      const streamed = await researchStream(request, {
        thought: (t) => setThoughts((prev) => [...prev, t]),
        plan: (p) => setPlan(p),
      });
      if (!streamed) {
        const p = await research(request); // real-backend fallback (no thought stream yet)
        setPlan(p);
      }
      setPhase("plan");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("landing");
    }
  }

  async function approve(edited: RunPlan, testInput: string) {
    if (!req) return;
    setError(null);
    setPlan(edited);
    // Prefer any eval examples from the request; otherwise use the (editable)
    // test prompt from the plan card so the before/after + judge populate.
    const evals = req.eval_examples.length
      ? req.eval_examples
      : testInput
        ? [{ input: testInput, expected_output: null }]
        : [];
    try {
      const { run_id } = await train(edited, evals);
      setRunId(run_id);
      setPhase("train");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function reset() {
    setPhase("landing"); setReq(null); setPlan(null);
    setThoughts([]); setRunId(null); setResult(null); setError(null);
  }

  return (
    <>
      <div className="ambient" aria-hidden><span className="blob blob-a" /><span className="blob blob-b" /></div>

      {phase === "landing"
        ? <Landing mode={mode} error={error} onStart={start} />
        : <Studio
            phase={phase}
            mode={mode}
            req={req}
            plan={plan}
            thoughts={thoughts}
            stream={stream}
            error={error}
            onApprove={approve}
            onReset={reset}
          />}

      <Deck phase={phase} mode={mode} plan={plan} thoughts={thoughts} stream={stream} result={result} />
    </>
  );
}

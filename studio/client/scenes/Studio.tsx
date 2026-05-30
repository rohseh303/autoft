import type { RunPlan, UserRequest } from "@shared/types";
import type { Phase, Thought } from "../App";
import type { RunStream } from "../lib/useRunStream";
import { Rail } from "../components/Rail";
import { Notebook } from "../components/Notebook";
import { PlanCard } from "../components/PlanCard";
import { Theater } from "../components/Theater";
import { Reveal } from "../components/Reveal";

export function Studio(props: {
  phase: Phase;
  mode: "mock" | "modal";
  req: UserRequest | null;
  plan: RunPlan | null;
  thoughts: Thought[];
  stream: RunStream;
  runId: string | null;
  error: string | null;
  onApprove: (p: RunPlan) => void;
  onReset: () => void;
}) {
  const { phase, plan, thoughts, stream, runId, error, onApprove, onReset } = props;

  return (
    <div className="studio">
      <Rail phase={phase} onReset={onReset} mode={props.mode} task={props.req?.task_description ?? ""} />

      <div className="stage">
        {error && <div className="err stage-err">{error}</div>}

        {(phase === "research" || phase === "plan") && (
          <div className="split">
            <Notebook thoughts={thoughts} done={phase === "plan"} />
            {phase === "plan" && plan && (
              <PlanCard plan={plan} onApprove={onApprove} />
            )}
          </div>
        )}

        {phase === "train" && plan && (
          <Theater plan={plan} stream={stream} runId={runId} />
        )}

        {phase === "done" && stream.result && (
          <Reveal result={stream.result} points={stream.points} onReset={onReset} />
        )}
      </div>
    </div>
  );
}

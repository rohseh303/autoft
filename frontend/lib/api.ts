import type { EvalExample, RunPlan, RunResult, UserRequest } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export async function postResearch(req: UserRequest): Promise<RunPlan> {
  const r = await fetch(`${API_BASE}/research`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`research failed: ${r.status}`);
  return r.json();
}

export async function postTrain(
  plan: RunPlan,
  eval_examples: EvalExample[],
): Promise<{ run_id: string }> {
  const r = await fetch(`${API_BASE}/train`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ plan, eval_examples }),
  });
  if (!r.ok) throw new Error(`train failed: ${r.status}`);
  return r.json();
}

export function streamUrl(runId: string): string {
  return `${API_BASE}/run/${runId}/stream`;
}

export async function getResult(runId: string): Promise<RunResult> {
  const r = await fetch(`${API_BASE}/run/${runId}/result`);
  if (!r.ok) throw new Error(`result not ready: ${r.status}`);
  return r.json();
}

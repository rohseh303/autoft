import type { RunPlan, RunResult, UserRequest } from "@shared/types";

const base = "/api";

export async function health(): Promise<{ ok: boolean; mode: "mock" | "modal" }> {
  return (await fetch(`${base}/health`)).json();
}

export async function research(req: UserRequest): Promise<RunPlan> {
  const r = await fetch(`${base}/research`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`research failed: ${r.status}`);
  return r.json();
}

export async function train(plan: RunPlan, eval_examples: UserRequest["eval_examples"]): Promise<{ run_id: string }> {
  const r = await fetch(`${base}/train`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ plan, eval_examples }),
  });
  if (!r.ok) throw new Error(`train failed: ${r.status}`);
  return r.json();
}

export const streamUrl = (runId: string) => `${base}/run/${runId}/stream`;
export const downloadUrl = (runId: string) => `${base}/run/${runId}/download`;

// POST-based SSE (fetch + ReadableStream) for the research thought-stream.
// Returns true if the server supports it (mock), false to fall back to /research.
export async function researchStream(
  req: UserRequest,
  on: { thought: (t: { kind: string; text: string; detail?: string }) => void; plan: (p: RunPlan) => void },
): Promise<boolean> {
  const r = await fetch(`${base}/research/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok || !r.body) return false;
  await consumeSSE(r.body, (ev, data) => {
    if (ev === "thought") on.thought(data);
    else if (ev === "plan") on.plan(data);
  });
  return true;
}

// Minimal SSE frame parser over a fetch ReadableStream.
export async function consumeSSE(
  body: ReadableStream<Uint8Array>,
  handle: (event: string, data: any) => void,
): Promise<void> {
  const reader = body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;
      try { handle(event, JSON.parse(dataLines.join("\n"))); } catch { /* keepalive */ }
    }
  }
}

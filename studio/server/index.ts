// AutoFT Studio BFF — Bun + Elysia.
// Three jobs (this is why Elysia earns its seat, not theater):
//   1. serve the built client in prod
//   2. proxy /api/* to the real Modal backend (research / train / SSE stream)
//   3. MOCK MODE when MODAL_API_BASE is unset — full simulated run, zero infra
//
// SSE is normalized so the client speaks ONE event vocabulary regardless of
// whether it's talking to Modal or the mock:  status | metric | thought | sample | result
import { Elysia } from "elysia";
import { cors } from "@elysiajs/cors";
import { existsSync } from "node:fs";
import { join } from "node:path";
import type { RunPlan, UserRequest } from "../shared/types";
import {
  mockPlan, mockThoughts, mockResult,
  lossAt, lrAt, gradAt, sampleAt, sleep,
  simulateMetrics, simulateEvents, mockLogLines,
} from "./mock";

const MODAL = process.env.MODAL_API_BASE?.replace(/\/$/, "") || "";
const MOCK = MODAL === "";
const PORT = Number(process.env.PORT ?? 8787);
const DIST = join(import.meta.dir, "..", "dist");
// Deployed Modal app whose container logs we tail for the live log pane.
const MODAL_APP = process.env.AUTOFT_MODAL_APP || "autoft";

const sseHeaders = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache",
  Connection: "keep-alive",
  "X-Accel-Buffering": "no",
};

// in-memory run store for mock mode (startedAt drives the wall-clock simulation)
const runs = new Map<string, { plan: RunPlan; req: UserRequest; startedAt: number }>();

// Live container logs for real mode. The user's instruction: logs are "emitted
// by the modal CLI -- this does not require exposing an endpoint." So we shell
// out to `modal app logs <app>` (which tails + follows) and re-emit each line as
// an SSE `log` event. Requires the `modal` CLI on PATH and authed on this host.
function streamModalLogs(runId: string): Response {
  let proc: ReturnType<typeof Bun.spawn> | null = null;
  const stream = new ReadableStream({
    start(controller) {
      const enc = new TextEncoder();
      const send = (ev: string, data: unknown) =>
        controller.enqueue(enc.encode(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`));

      try {
        proc = Bun.spawn(["modal", "app", "logs", MODAL_APP], { stdout: "pipe", stderr: "pipe", stdin: "ignore" });
      } catch (err) {
        send("log", { line: `[studio] could not start \`modal app logs ${MODAL_APP}\`: ${String(err)}`, stream: "stderr" });
        send("eof", { ok: false });
        controller.close();
        return;
      }
      send("log", { line: `[studio] tailing \`modal app logs ${MODAL_APP}\` (run ${runId})`, stream: "meta" });

      const pump = async (rs: ReadableStream<Uint8Array> | null, which: "stdout" | "stderr") => {
        if (!rs) return;
        const reader = rs.getReader();
        const dec = new TextDecoder();
        let buf = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          let idx: number;
          while ((idx = buf.indexOf("\n")) !== -1) {
            const line = buf.slice(0, idx); buf = buf.slice(idx + 1);
            if (line.trim()) send("log", { line, stream: which });
          }
        }
        if (buf.trim()) send("log", { line: buf, stream: which });
      };

      Promise.allSettled([
        pump(proc.stdout as ReadableStream<Uint8Array>, "stdout"),
        pump(proc.stderr as ReadableStream<Uint8Array>, "stderr"),
      ]).then(() => {
        send("eof", { ok: true });
        try { controller.close(); } catch { /* already closed */ }
      });
    },
    // Client disconnected (EventSource closed) — kill the tail so it doesn't leak.
    cancel() {
      try { proc?.kill(); } catch { /* already gone */ }
    },
  });
  return new Response(stream, { headers: sseHeaders });
}

const app = new Elysia()
  .use(cors())
  .get("/api/health", () => ({ ok: true, mode: MOCK ? "mock" : "modal", modal: MODAL || null }))

  // --- research: returns a RunPlan ----------------------------------------
  .post("/api/research", async ({ body }) => {
    const req = body as UserRequest;
    if (MOCK) {
      await sleep(450);
      // Same {plan, events} contract the real backend returns. events carry the
      // research agent's reasoning trace (source="research") for the notebook.
      const events = mockThoughts(req).map((t) => ({
        ts: null, source: "research", kind: t.kind, text: t.text,
        step: null, detail: t.detail ?? null,
      }));
      return { plan: mockPlan(req), events };
    }
    const r = await fetch(`${MODAL}/research`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) return new Response(`research failed: ${r.status}`, { status: 502 });
    return r.json(); // { plan, events }
  })

  // --- research as a live thought stream (mock-only enhancement) -----------
  // Real backend has no thought stream yet; client falls back to /api/research.
  .post("/api/research/stream", async ({ body }) => {
    const req = body as UserRequest;
    if (!MOCK) return new Response("thought-stream is mock-only", { status: 404 });
    const thoughts = mockThoughts(req);
    const plan = mockPlan(req);
    const stream = new ReadableStream({
      async start(c) {
        const enc = new TextEncoder();
        const send = (ev: string, data: unknown) =>
          c.enqueue(enc.encode(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`));
        for (const t of thoughts) { send("thought", t); await sleep(620); }
        send("plan", plan);
        c.close();
      },
    });
    return new Response(stream, { headers: sseHeaders });
  })

  // --- train: spawn, return run_id ----------------------------------------
  .post("/api/train", async ({ body }) => {
    const { plan, eval_examples } = body as { plan: RunPlan; eval_examples: UserRequest["eval_examples"] };
    if (MOCK) {
      const run_id = `mock-${Math.random().toString(36).slice(2, 10)}`;
      runs.set(run_id, {
        plan,
        req: { task_description: plan.task_summary, eval_examples: eval_examples ?? [] },
        startedAt: Date.now(),
      });
      return { run_id };
    }
    const r = await fetch(`${MODAL}/train`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ plan, eval_examples: eval_examples ?? [] }),
    });
    if (!r.ok) return new Response(`train failed: ${r.status}`, { status: 502 });
    return r.json();
  })

  // --- live training stream ------------------------------------------------
  .get("/api/run/:id/stream", ({ params }) => {
    const runId = params.id;
    if (!MOCK) {
      // Pure passthrough proxy of Modal's SSE — already speaks status|metric|result.
      return fetch(`${MODAL}/run/${runId}/stream`, { headers: { accept: "text/event-stream" } });
    }
    const entry = runs.get(runId);
    const stream = new ReadableStream({
      async start(c) {
        const enc = new TextEncoder();
        const send = (ev: string, data: unknown) =>
          c.enqueue(enc.encode(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`));
        if (!entry) { send("status", { run_id: runId, state: "failed", message: "unknown run", plan: null }); c.close(); return; }
        const { plan, req } = entry;
        const max = plan.training.max_steps;

        // loading phase — granular status so there's no dead air
        for (const m of ["pulling image", `downloading ${plan.base_model}`, `loading ${plan.hf_dataset}`, "tokenizing 2000 rows"]) {
          send("status", { run_id: runId, state: "loading", message: m, plan });
          await sleep(550);
        }
        send("status", { run_id: runId, state: "training", message: "SFTTrainer running", plan });

        const start = Date.now();
        for (let step = 1; step <= max; step++) {
          send("metric", {
            run_id: runId, step,
            loss: lossAt(step, max), learning_rate: lrAt(step, plan),
            grad_norm: gradAt(step), epoch: (step * plan.training.batch_size * plan.training.gradient_accumulation_steps) / 2000,
            elapsed_seconds: (Date.now() - start) / 1000,
          });
          // re-sample the model's words every ~12 steps — the showpiece
          if (step === 1 || step % 12 === 0 || step === max) {
            send("sample", { run_id: runId, step, prompt: req.eval_examples[0]?.input ?? "eval prompt", text: sampleAt(step, max) });
          }
          await sleep(38); // ~6s total — fast enough to watch, slow enough to feel real
        }
        send("status", { run_id: runId, state: "evaluating", message: "generating base vs fine-tuned", plan });
        await sleep(900);
        const result = mockResult(runId, plan, req);
        send("status", { run_id: runId, state: "done", message: `final_loss=${result.final_loss?.toFixed(4) ?? "n/a"}`, plan });
        send("result", result);
        c.close();
      },
    });
    return new Response(stream, { headers: sseHeaders });
  })

  .get("/api/run/:id/result", async ({ params }) => {
    if (MOCK) {
      const e = runs.get(params.id);
      if (!e) return new Response("not found", { status: 404 });
      return mockResult(params.id, e.plan, e.req);
    }
    const r = await fetch(`${MODAL}/run/${params.id}/result`);
    if (!r.ok) return new Response("result not ready", { status: r.status });
    return r.json();
  })

  // --- metrics snapshot (POLLED continuously by the charts) ----------------
  // { status, metrics: StepMetric[], result, samples? } — the frontend polls
  // this in a loop and plots loss / lr / grad_norm (wandb-style, no wandb).
  .get("/api/run/:id/metrics", async ({ params }) => {
    if (!MOCK) {
      const r = await fetch(`${MODAL}/run/${params.id}/metrics`);
      if (!r.ok) return new Response(`metrics failed: ${r.status}`, { status: 502 });
      return r.json();
    }
    const e = runs.get(params.id);
    if (!e) return { status: { run_id: params.id, state: "failed", message: "unknown run", plan: null }, metrics: [], result: null, samples: [] };
    return simulateMetrics(params.id, e.plan, e.req, (Date.now() - e.startedAt) / 1000);
  })

  // --- transparency timeline (POLLED) --------------------------------------
  // { events: RunEvent[] } — source-tagged reasoning/status beats. Shows which
  // subagent (research / trainer / judge) did what, and at which step.
  .get("/api/run/:id/events", async ({ params }) => {
    if (!MOCK) {
      const r = await fetch(`${MODAL}/run/${params.id}/events`);
      if (!r.ok) return new Response(`events failed: ${r.status}`, { status: 502 });
      return r.json();
    }
    const e = runs.get(params.id);
    if (!e) return { events: [] };
    return { events: simulateEvents(params.id, e.plan, (Date.now() - e.startedAt) / 1000) };
  })

  // --- live logs (STREAMED) ------------------------------------------------
  // Real: tails `modal app logs` (CLI, no Modal HTTP endpoint). Mock: synthesizes
  // a believable container log, paced out so it feels live. SSE `log` events.
  .get("/api/run/:id/logs", ({ params }) => {
    if (!MOCK) return streamModalLogs(params.id);
    const e = runs.get(params.id);
    const lines = e ? mockLogLines(e.plan, e.req) : ["[studio] unknown run"];
    const stream = new ReadableStream({
      async start(c) {
        const enc = new TextEncoder();
        const send = (ev: string, data: unknown) =>
          c.enqueue(enc.encode(`event: ${ev}\ndata: ${JSON.stringify(data)}\n\n`));
        for (const line of lines) { send("log", { line, stream: "stdout" }); await sleep(260); }
        send("eof", { ok: true });
        c.close();
      },
    });
    return new Response(stream, { headers: sseHeaders });
  })

  // --- static client (prod) -----------------------------------------------
  .get("/*", ({ path }) => {
    if (!existsSync(DIST)) return new Response("Run `bun run build` first (or use `bun run dev`).", { status: 404 });
    const file = Bun.file(join(DIST, path === "/" ? "index.html" : path));
    return file.exists().then((ok) => (ok ? file : Bun.file(join(DIST, "index.html")))); // SPA fallback
  })

  .listen(PORT);

console.log(`\n  AutoFT Studio BFF  ·  http://localhost:${PORT}`);
console.log(`  mode: ${MOCK ? "MOCK (no Modal — full simulated run)" : `MODAL → ${MODAL}`}\n`);

export type App = typeof app;

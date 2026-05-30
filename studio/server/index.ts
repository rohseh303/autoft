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
} from "./mock";

const MODAL = process.env.MODAL_API_BASE?.replace(/\/$/, "") || "";
const MOCK = MODAL === "";
const PORT = Number(process.env.PORT ?? 8787);
const DIST = join(import.meta.dir, "..", "dist");

const sseHeaders = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache",
  Connection: "keep-alive",
  "X-Accel-Buffering": "no",
};

// in-memory run store for mock mode
const runs = new Map<string, { plan: RunPlan; req: UserRequest }>();

const app = new Elysia()
  .use(cors())
  .get("/api/health", () => ({ ok: true, mode: MOCK ? "mock" : "modal", modal: MODAL || null }))

  // --- research: returns a RunPlan ----------------------------------------
  .post("/api/research", async ({ body }) => {
    const req = body as UserRequest;
    if (MOCK) {
      await sleep(450);
      return mockPlan(req);
    }
    const r = await fetch(`${MODAL}/research`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!r.ok) return new Response(`research failed: ${r.status}`, { status: 502 });
    return r.json();
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
      runs.set(run_id, { plan, req: { task_description: plan.task_summary, eval_examples: eval_examples ?? [] } });
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

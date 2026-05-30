# AutoFT Studio

A chatbox-first UI for AutoFT — describe a model in one sentence, watch an agent
design it, a GPU train it, and proof that it works. Separate, clean-architecture
frontend that talks to the existing Modal backend (or runs fully simulated).

**Stack:** Bun · Elysia (BFF gateway) · React + Vite · zero chart deps (hand-rolled SVG).

## Run it (zero setup — demo mode)

```bash
cd studio
bun install
bun run dev          # client :5173, BFF :8787
# open http://localhost:5173
```

With no `MODAL_API_BASE` set, the BFF runs in **mock mode**: a simulated research
trace, a real RunPlan, a live loss curve, the model's words sharpening from
gibberish → coherent, and a before/after — all with no GPU, secrets, or deploy.

## Run it against the real backend

```bash
cp .env.example .env
# set MODAL_API_BASE=https://<you>--autoft-web.modal.run  (from `modal deploy backend/api.py`)
bun run build && bun run start   # single Bun process serves client + proxies Modal
```

## Architecture

```
client/  React + Vite  — the four scenes, one design system ("Bench")
  scenes/   Landing (chatbox) · Studio (rail + stages)
  components/ Rail · Notebook · PlanCard · Theater · LossChart · Reveal
  design/   tokens.css (the whole color/type system) · app.css
server/  Elysia BFF on Bun — 3 jobs:
  1. serve the built client      2. proxy /api/* → Modal
  3. mock.ts — full simulated run when MODAL_API_BASE is unset
shared/  types.ts — mirror of backend RunPlan + a superset SSE event vocab
```

The BFF normalizes everything to **one** SSE vocabulary the client speaks
regardless of source: `status · metric · thought · sample · result`. The Modal
backend emits `status/metric/result` today; `thought` (agent reasoning) and
`sample` (mid-training re-sampling) are synthesized in mock mode and are the
forward-compatible hooks for the backend features discussed in the root README's
"NOT in v1" list.

## The four moments (one design system, three skins)

| Stage | Skin | What it does |
|---|---|---|
| Describe | light, editorial | one chatbox, nothing else |
| Research → Plan | light "notebook" | agent writes its reasoning live; plan is editable |
| Train | **dark theater** | loss curve **+** the model's words sharpening live |
| Compare | light | drag-to-reveal base vs. your model on your own examples |

**Color — "Bench":** warm cream `#F7F4EC` + warm ink `#16150F`, one accent
signal-orange `#FF4D1C`; training drops to void `#0B0B0D` with phosphor-green
`#46E5A0` for live output. Type: Instrument Serif / Inter / JetBrains Mono.

## Status (honest)

- ✅ Full UI, typechecks, builds (~48 KB gz), end-to-end in mock mode.
- ⚠️ `thought` and `sample` streams are **mock-only** until the backend grows a
  reasoning trace + a mid-training re-sampling callback. Against real Modal the
  research step falls back to the plain `/research` call and training shows the
  real `status/metric/result` stream (no live samples yet).
- Not yet wired: the pre-warm-on-plan-review optimization (the latency win).

#!/usr/bin/env bash
# Launch Codex as the AutoFT post-training lead: build the briefing, then let it
# drive the train -> eval -> judge loop. Watch it in the TUI (default) or run
# it headless.
#
#   bash scripts/optimize.sh            # interactive TUI (approve each trial)
#   MODE=exec bash scripts/optimize.sh  # headless, autonomous (no approvals)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Seed the mutable working files from the canonical fixtures on first run.
[ -f task.txt ]  || cp fixtures/billsum_task.txt task.txt
[ -f eval.json ] || cp fixtures/billsum_eval.json eval.json
[ -f plan.json ] || cp fixtures/billsum_plan.json plan.json

# Build the briefing packet (task + eval set + seed recipe + live HF peek).
uv run python scripts/build_kickoff.py

read -r -d '' PROMPT <<'EOF' || true
You are the AutoFT post-training lead. Read AGENTS.md and kickoff.md in full first.
Then run the optimization loop: edit plan.json, run `uv run modal run backend/train.py::trial`,
read result.json (per-example outputs + judge critiques) and trials.jsonl after each trial,
change ONE thing at a time, and keep the best. Inspect the data with
`uv run python backend/hf_cli.py peek <dataset>` whenever a decision depends on it.
Stop at the budget / stop rule in AGENTS.md, then write the winning recipe to plan.json
and report what you tried and why the winner won.
EOF

MODE="${MODE:-interactive}"
if [ "$MODE" = "exec" ]; then
  exec codex exec --json --sandbox workspace-write --ask-for-approval never \
    -o lead_summary.txt "$PROMPT"
else
  exec codex --cd "$REPO" --sandbox workspace-write --ask-for-approval on-request "$PROMPT"
fi

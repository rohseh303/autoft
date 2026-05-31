#!/usr/bin/env bash
# Launch the AutoFT studio.
#
# DEFAULT: respects studio/.env. If MODAL_API_BASE is set there, you get the
#          REAL deployed Modal backend (research + L4 training). No local server.
#
# SIM MODE: pass --sim (or LOCAL=1) to instead boot the local Python backend
#           (:8000) and force the studio at it — fully simulated, no GPU/Modal.
#
# Open http://localhost:5173 either way. Ctrl-C stops everything.
set -euo pipefail
cd "$(dirname "$0")"

cleanup() { echo; echo "stopping…"; kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if [[ "${1:-}" == "--sim" || "${LOCAL:-}" == "1" ]]; then
  BACKEND_PORT="${BACKEND_PORT:-8000}"
  echo "▶ SIM mode — local Python backend on :${BACKEND_PORT} (no GPU/Modal)"
  PORT="${BACKEND_PORT}" ./local_server/run.sh &
  for _ in $(seq 1 60); do
    curl -s "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1 && break
    sleep 0.5
  done
  cd studio
  [ -d node_modules ] || bun install
  AUTOFT_BACKEND="http://localhost:${BACKEND_PORT}" bun run dev &
  wait
else
  echo "▶ REAL mode — studio reads studio/.env (MODAL_API_BASE → deployed GPU backend)"
  echo "   (run with --sim for the local simulated backend instead)"
  cd studio
  [ -d node_modules ] || bun install
  bun run dev
fi
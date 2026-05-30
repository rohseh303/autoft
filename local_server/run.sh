#!/usr/bin/env bash
# Local AutoFT backend — no GPU/Modal/Codex. Streams research + training over SSE.
# Run from anywhere; this cd's to the repo root so `shared` + `local_server` import.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"

# shared/schemas.py uses `X | None` syntax -> needs Python 3.10+. Pick the
# newest interpreter available (repo targets 3.11+).
PYBIN=""
for p in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$p" >/dev/null 2>&1; then
    ver=$("$p" -c 'import sys; print(sys.version_info >= (3,10))' 2>/dev/null || echo False)
    if [ "$ver" = "True" ]; then PYBIN="$p"; break; fi
  fi
done
if [ -z "$PYBIN" ]; then
  echo "ERROR: need Python 3.10+ (shared/schemas.py uses PEP 604 unions)." >&2
  exit 1
fi

# create/refresh a tiny venv just for the local backend
if [ ! -d local_server/.venv ]; then
  "$PYBIN" -m venv local_server/.venv
fi
# shellcheck disable=SC1091
source local_server/.venv/bin/activate
pip install -q -r local_server/requirements.txt

echo "AutoFT local backend → http://localhost:${PORT}  (SSE: /research/stream, /run/{id}/stream)"
exec uvicorn local_server.app:app --host 0.0.0.0 --port "${PORT}" --reload

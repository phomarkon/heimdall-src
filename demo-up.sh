#!/usr/bin/env bash
# Heimdall demo stack — detached launcher (survives terminal/session close).
#   bash demo-up.sh          # start postgres + run-view + frontend, detached
#   bash demo-up.sh --status # report what is up
#   bash demo-up.sh --stop   # stop run-view + frontend (postgres left running)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PG_BIN=/opt/homebrew/opt/postgresql@16/bin
DSN="postgresql://heimdall:heimdall@127.0.0.1:5432/heimdall"
mkdir -p logs

case "${1:-start}" in
  --status)
    "$PG_BIN/pg_isready" -h 127.0.0.1 -p 5432 >/dev/null 2>&1 && echo "postgres : UP" || echo "postgres : DOWN"
    curl -fs -o /dev/null --max-time 2 http://127.0.0.1:8091/v1/runs && echo "run-view : UP ($(curl -fs http://127.0.0.1:8091/v1/runs | python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d if isinstance(d,list) else d["runs"]))') runs)" || echo "run-view : DOWN"
    curl -fs -o /dev/null --max-time 2 http://127.0.0.1:3000 && echo "frontend : UP" || echo "frontend : DOWN"
    exit 0 ;;
  --stop)
    lsof -ti :3000 | xargs kill 2>/dev/null && echo "frontend stopped" || echo "frontend not running"
    lsof -ti :8091 | xargs kill 2>/dev/null && echo "run-view stopped" || echo "run-view not running"
    echo "postgres left running (brew service)"
    exit 0 ;;
esac

# 1. postgres
"$PG_BIN/pg_isready" -h 127.0.0.1 -p 5432 >/dev/null 2>&1 || brew services start postgresql@16 >/dev/null 2>&1
for _ in $(seq 1 30); do "$PG_BIN/pg_isready" -h 127.0.0.1 -p 5432 >/dev/null 2>&1 && break; sleep 0.5; done
echo "postgres ready"

# 2. run-view (detached)
if ! curl -fs -o /dev/null --max-time 2 http://127.0.0.1:8091/v1/runs; then
  HEIMDALL_RUN_VIEW_DATABASE_URL="$DSN" nohup uv run --no-sync uvicorn heimdall_run_view.service:app \
    --host 127.0.0.1 --port 8091 >logs/run-view.log 2>&1 &
  disown
  for _ in $(seq 1 40); do curl -fs -o /dev/null --max-time 2 http://127.0.0.1:8091/v1/runs && break; sleep 0.5; done
fi
echo "run-view ready -> http://127.0.0.1:8091"

# 3. frontend (detached)
if ! curl -fs -o /dev/null --max-time 2 http://127.0.0.1:3000; then
  ( cd app/frontend && PORT=3000 nohup bun run dev >"$ROOT/logs/frontend.log" 2>&1 & disown )
  for _ in $(seq 1 60); do curl -fs -o /dev/null --max-time 2 http://127.0.0.1:3000 && break; sleep 0.5; done
fi
echo "frontend ready -> http://127.0.0.1:3000"
echo "ALL UP. open http://localhost:3000"

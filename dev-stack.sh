#!/usr/bin/env bash
# Heimdall — one-command frontend dev stack.
#
# Brings up everything the dashboard needs, in dependency order:
#   1. PostgreSQL 16    — society specs + agent templates          :5432
#   2. run-view API     — run catalogue + replay (hybrid PG/disk)  :8091
#   3. Next.js frontend — Live / Runs / Config / Results / Help    :3000
#
# Idempotent and safe to re-run. PostgreSQL is a shared daemon and is left
# running on exit. run-view starts detached (logs/pid under logs/); the
# frontend runs in the foreground, and Ctrl-C tears down the frontend +
# run-view together (Postgres stays up).
#
# Usage:
#   bash dev-stack.sh                 # start db + backend + frontend (Ctrl-C stops the last two)
#   bash dev-stack.sh --detach        # start all three detached, then return the shell
#   bash dev-stack.sh --no-frontend   # only db + backend (e.g. for API work)
#   bash dev-stack.sh --stop          # stop run-view + frontend (Postgres left running)
#   bash dev-stack.sh --status        # report what is up
#
# Environment overrides:
#   HEIMDALL_RUN_VIEW_DATABASE_URL  Postgres DSN (default heimdall:heimdall@127.0.0.1:5432/heimdall)
#   RUN_VIEW_PORT                   run-view port (default 8091)
#   FRONTEND_PORT                   frontend port (default 3000)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

log()  { printf '\033[36m[dev-stack]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[dev-stack]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[dev-stack]\033[0m %s\n' "$*" >&2; }

# --- config ----------------------------------------------------------------
PG_DSN="${HEIMDALL_RUN_VIEW_DATABASE_URL:-postgresql://heimdall:heimdall@127.0.0.1:5432/heimdall}"
RUN_VIEW_PORT="${RUN_VIEW_PORT:-8091}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
export HEIMDALL_RUN_VIEW_DATABASE_URL="$PG_DSN"

RUNTIME_DIR="$REPO_ROOT/logs"            # gitignored
RUN_VIEW_PID="$RUNTIME_DIR/run-view.pid"
RUN_VIEW_LOG="$RUNTIME_DIR/run-view.log"
FRONTEND_PID="$RUNTIME_DIR/frontend.pid"
FRONTEND_LOG="$RUNTIME_DIR/frontend.log"
mkdir -p "$RUNTIME_DIR"

# --- flags -----------------------------------------------------------------
DETACH=0
DO_FRONTEND=1
ACTION=start
for arg in "$@"; do
  case "$arg" in
    --detach)      DETACH=1 ;;
    --no-frontend) DO_FRONTEND=0 ;;
    --stop)        ACTION=stop ;;
    --status)      ACTION=status ;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
    *) err "unknown flag: $arg (try --help)"; exit 2 ;;
  esac
done

# --- helpers ---------------------------------------------------------------
pid_alive() { local f="$1"; [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; }

stop_pidfile() {
  local f="$1" name="$2"
  if pid_alive "$f"; then
    local pid; pid="$(cat "$f")"
    log "stopping $name (pid $pid)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$f"
}

wait_for_http() {  # url, label, tries
  local url="$1" label="$2" tries="${3:-40}"
  for _ in $(seq 1 "$tries"); do
    if curl -fs -o /dev/null --max-time 2 "$url"; then return 0; fi
    sleep 0.5
  done
  return 1
}

# --- stop / status short-circuits -----------------------------------------
if [[ "$ACTION" == stop ]]; then
  stop_pidfile "$FRONTEND_PID" "frontend"
  stop_pidfile "$RUN_VIEW_PID" "run-view"
  log "Postgres left running (shared daemon). Stop it with: sudo pg_ctlcluster 16 main stop"
  exit 0
fi

if [[ "$ACTION" == status ]]; then
  pg_isready -d "$PG_DSN" >/dev/null 2>&1 && log "postgres   : UP   ($PG_DSN)" || warn "postgres   : DOWN"
  curl -fs -o /dev/null --max-time 2 "http://127.0.0.1:$RUN_VIEW_PORT/v1/runs" \
    && log "run-view   : UP   (http://127.0.0.1:$RUN_VIEW_PORT)" || warn "run-view   : DOWN"
  curl -fs -o /dev/null --max-time 2 "http://127.0.0.1:$FRONTEND_PORT" \
    && log "frontend   : UP   (http://127.0.0.1:$FRONTEND_PORT)" || warn "frontend   : DOWN"
  exit 0
fi

# --- 1. PostgreSQL ---------------------------------------------------------
if pg_isready -d "$PG_DSN" >/dev/null 2>&1; then
  log "postgres already up"
else
  log "starting postgres (cluster 16/main)"
  sudo pg_ctlcluster 16 main start || true
  if ! pg_isready -d "$PG_DSN" >/dev/null 2>&1; then
    err "postgres not reachable at $PG_DSN — start it manually (sudo pg_ctlcluster 16 main start) and re-run"
    exit 1
  fi
fi

log "ensuring run-view schema"
uv run heimdall-run-view init-db

# --- 2. run-view API -------------------------------------------------------
if pid_alive "$RUN_VIEW_PID" || curl -fs -o /dev/null --max-time 2 "http://127.0.0.1:$RUN_VIEW_PORT/v1/runs"; then
  log "run-view already up on :$RUN_VIEW_PORT"
else
  log "starting run-view on :$RUN_VIEW_PORT (log: logs/run-view.log)"
  setsid uv run uvicorn heimdall_run_view.service:app \
      --host 127.0.0.1 --port "$RUN_VIEW_PORT" \
      >"$RUN_VIEW_LOG" 2>&1 < /dev/null &
  echo $! > "$RUN_VIEW_PID"
  if wait_for_http "http://127.0.0.1:$RUN_VIEW_PORT/v1/runs" "run-view"; then
    log "run-view ready → http://127.0.0.1:$RUN_VIEW_PORT"
  else
    err "run-view did not come up — see logs/run-view.log"; tail -n 20 "$RUN_VIEW_LOG" >&2 || true; exit 1
  fi
fi

# --- 3. frontend -----------------------------------------------------------
if [[ "$DO_FRONTEND" -eq 0 ]]; then
  log "done (--no-frontend). API: http://127.0.0.1:$RUN_VIEW_PORT"
  exit 0
fi

if [[ ! -d app/frontend/node_modules ]]; then
  log "installing frontend deps (bun install)"
  (cd app/frontend && bun install)
fi

# Make sure the frontend points at this run-view instance.
if [[ ! -f app/frontend/.env.local ]]; then
  warn "app/frontend/.env.local missing — frontend will fall back to mock data"
fi

if [[ "$DETACH" -eq 1 ]]; then
  log "starting frontend detached on :$FRONTEND_PORT (log: logs/frontend.log)"
  ( cd app/frontend && setsid env PORT="$FRONTEND_PORT" bun run dev \
      >"$FRONTEND_LOG" 2>&1 < /dev/null & echo $! > "$FRONTEND_PID" )
  wait_for_http "http://127.0.0.1:$FRONTEND_PORT" "frontend" 60 \
    && log "frontend ready → http://127.0.0.1:$FRONTEND_PORT" \
    || warn "frontend slow to start — check logs/frontend.log"
  log "all detached. Stop with: bash dev-stack.sh --stop"
  exit 0
fi

# Foreground frontend; Ctrl-C tears down frontend + run-view (Postgres stays up).
cleanup() {
  echo
  stop_pidfile "$RUN_VIEW_PID" "run-view"
  log "Postgres left running. Bye."
}
trap cleanup EXIT INT TERM

log "starting frontend on :$FRONTEND_PORT  (Ctrl-C stops frontend + run-view)"
log "  frontend → http://127.0.0.1:$FRONTEND_PORT"
log "  api      → http://127.0.0.1:$RUN_VIEW_PORT"
cd app/frontend
exec env PORT="$FRONTEND_PORT" bun run dev

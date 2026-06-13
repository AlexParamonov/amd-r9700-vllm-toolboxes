#!/usr/bin/env bash
# launch.sh — Start vLLM inside the distrobox container and warm up.
#
# Designed to be run by a systemd service. Launches vLLM in the background,
# waits for the server, runs warmup requests to trigger JIT compilation,
# then tails the log.
#
# Usage: ./launch.sh [--port PORT] [--log LOGFILE]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${VLLM_PORT:-8079}"
LOGFILE="${VLLM_LOG:-/tmp/vllm.log}"

# Start vLLM in background, capture output to log.
bash "${SCRIPT_DIR}/start.sh" aiter >"$LOGFILE" 2>&1 &
VLLM_PID=$!

# Wait for the server to be ready.
echo "[*] Waiting for vLLM on port ${PORT}..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[+] Server ready (PID ${VLLM_PID})."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[-] vLLM process exited during startup."
        head -50 "$LOGFILE" >&2
        exit 1
    fi
    sleep 1
done

# Run warmup to trigger JIT compilation.
bash "${SCRIPT_DIR}/warmup.sh" "$PORT"

# Tail the log so systemd keeps the service alive.
exec tail -f "$LOGFILE" &
TAIL_PID=$!

# Forward signals to the vLLM process and all its children.
cleanup() {
    echo "[*] Stopping vLLM (PID ${VLLM_PID})..."
    # Kill the entire process group (vLLM + workers + children).
    pkill -P "$VLLM_PID" 2>/dev/null || true
    kill -- -"$(ps -o pgid= -p "$VLLM_PID" 2>/dev/null | tr -d ' ')" 2>/dev/null || true
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    kill "$TAIL_PID" 2>/dev/null || true
    # Safety sweep: kill any remaining vllm serve processes from this run.
    pkill -f "vllm serve.*${PORT}" 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT SIGHUP

# Wait for the vLLM process (blocks until it exits).
wait "$VLLM_PID"

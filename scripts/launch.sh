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
IDLE_TIMEOUT="${VLLM_IDLE_SECONDS:-1500}"  # seconds of no requests before auto-stop

# Function to find existing vLLM process by port (could be starting or running)
find_existing_vllm() {
    local port=$1
    # Look for vLLM process with matching port in command line
    local pid
    pid=$(pgrep -f "vllm serve.*--port ${port}" 2>/dev/null | head -1 || true)
    if [[ -n "$pid" ]]; then
        echo "$pid"
        return 0
    fi
    return 1
}

# Check for existing vLLM process (might be starting or already running)
EXISTING_PID=$(find_existing_vllm "$PORT" || true)

if [[ -n "$EXISTING_PID" ]]; then
    echo "[*] Found existing vLLM process (PID ${EXISTING_PID}) for port ${PORT}. Waiting for it to be ready..."
    VLLM_PID=$EXISTING_PID
else
    # Start vLLM in background, capture output to log.
    bash "${SCRIPT_DIR}/start.sh" aiter >"$LOGFILE" 2>&1 &
    VLLM_PID=$!
fi

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

# --- Idle detection ---
# Poll /metrics every 60s. If no requests for IDLE_TIMEOUT seconds, stop vLLM.
last_activity=$(date +%s)

check_idle() {
    tokens_prev=""
    while true; do
        sleep 60
        # Check if vLLM is still alive
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            break
        fi
        # Query vLLM metrics — track monotonically increasing counter
        tokens_now=$(curl -sf "http://localhost:${PORT}/metrics" 2>/dev/null | grep -E '^vllm:generation_tokens_total\{' | awk '{s+=$NF} END {print s+0}' || echo "0")
        if [ "${tokens_now:-0}" != "${tokens_prev:-0}" ]; then
            last_activity=$(date +%s)
            tokens_prev=$tokens_now
        fi
        idle=$(($(date +%s) - last_activity))
        if [ "$idle" -ge "$IDLE_TIMEOUT" ]; then
            echo "[*] Idle ${idle}s (timeout ${IDLE_TIMEOUT}s), stopping vLLM..."
            touch "$STOP_SIGNAL"
            break
        fi
    done
}
STOP_SIGNAL=$(mktemp)
rm -f "$STOP_SIGNAL"  # File should only exist when idle detection wants to stop
trap "rm -f $STOP_SIGNAL" EXIT
check_idle &
IDLE_PID=$!

# Tail the log so systemd keeps the service alive.
tail -f "$LOGFILE" &
TAIL_PID=$!

# Forward signals to the vLLM process and all its children.
cleanup() {
    echo "[*] Stopping vLLM (PID ${VLLM_PID})..."
    kill "$IDLE_PID" 2>/dev/null || true
    kill "$TAIL_PID" 2>/dev/null || true
    
    # Send SIGTERM to the process group (vLLM + workers + children).
    local pgid
    pgid=$(ps -o pgid= -p "$VLLM_PID" 2>/dev/null | tr -d ' ')
    if [[ -n "$pgid" && "$pgid" != "0" ]]; then
        kill -- -"$pgid" 2>/dev/null || true
    fi
    kill "$VLLM_PID" 2>/dev/null || true
    
    # Wait up to 10 seconds for graceful shutdown.
    local timeout=10
    local start_time
    start_time=$(date +%s)
    while kill -0 "$VLLM_PID" 2>/dev/null; do
        local now
        now=$(date +%s)
        if (( now - start_time >= timeout )); then
            echo "[-] vLLM did not stop within ${timeout}s, sending SIGKILL..."
            kill -9 "$VLLM_PID" 2>/dev/null || true
            if [[ -n "$pgid" && "$pgid" != "0" ]]; then
                kill -9 -- -"$pgid" 2>/dev/null || true
            fi
            break
        fi
        sleep 1
    done
    
    # Safety sweep: kill any remaining vllm serve processes from this run.
    pkill -f "vllm serve.*${PORT}" 2>/dev/null || true
    echo "[*] vLLM stopped."
    exit 0
}
trap cleanup SIGTERM SIGINT SIGHUP

# Wait for vLLM to exit or idle signal.
while kill -0 "$VLLM_PID" 2>/dev/null; do
    # Check if idle detector wants us to stop.
    if [ -f "$STOP_SIGNAL" ]; then
        cleanup
    fi
    sleep 1
done

# Propagate vLLM's exit code to systemd (Restart=on-failure needs non-zero on crash).
wait "$VLLM_PID" 2>/dev/null
exit $?

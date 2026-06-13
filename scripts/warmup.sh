#!/usr/bin/env bash
# warmup.sh — Trigger Triton kernel JIT compilation after vLLM startup.
#
# Run from a separate terminal after start.sh has the server ready:
#   ./warmup.sh [--port 8079]
#
# Sends a few requests to exercise code paths the built-in warmup misses
# (3D unified attention, reduce_segments for various batch sizes).
set -euo pipefail

PORT="${1:-8079}"
BASE_URL="http://localhost:${PORT}"

wait_for_server() {
    echo "[*] Waiting for vLLM on port ${PORT}..."
    for i in $(seq 1 60); do
        if curl -sf "${BASE_URL}/v1/models" >/dev/null 2>&1; then
            echo "[+] Server ready."
            return 0
        fi
        sleep 1
    done
    echo "[-] Server did not become ready in 60s."
    exit 1
}

send_request() {
    local label="$1"
    shift
    echo "  -> ${label}..."
    local start end elapsed
    start=$(date +%s%N)
    curl -sf "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$@" >/dev/null 2>&1 || true
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo "     ${elapsed}ms"
}

BODY_SHORT='{
    "model": "27b_mtp",
    "messages": [{"role": "user", "content": "Say hi in one word."}],
    "max_tokens": 10
}'

BODY_MEDIUM='{
    "model": "27b_mtp",
    "messages": [{"role": "user", "content": "Explain how attention works in transformers, briefly."}],
    "max_tokens": 128
}'

wait_for_server

echo "[*] Sending warmup requests to trigger JIT compilation..."
echo "    First request will be slow (kernel compilation)."

send_request "Short decode (triggers 3D attn + reduce_segments)" "$BODY_SHORT"
send_request "Medium decode (exercises larger batch shapes)" "$BODY_MEDIUM"
send_request "Second short (should be fast now)" "$BODY_SHORT"

echo "[+] Warmup complete. Subsequent requests should not have JIT spikes."

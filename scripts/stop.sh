#!/usr/bin/env bash
set -euo pipefail

# Stop any running vLLM instances.
# Tries SIGTERM first, waits up to 15s, then SIGKILL.

PIDS=$(pgrep -f "vllm serve" || true)

if [[ -z "$PIDS" ]]; then
  echo "No vLLM processes found."
  exit 0
fi

echo "Stopping vLLM (PIDs: $(echo $PIDS | tr '\n' ' '))..."
echo "$PIDS" | xargs kill || true

# Wait for processes to exit, up to 15s.
for i in $(seq 1 15); do
  if ! pgrep -f "vLLM serve" >/dev/null 2>&1; then
    echo "vLLM stopped."
    exit 0
  fi
  sleep 1
done

# Still alive — force kill.
echo "vLLM didn't exit cleanly, sending SIGKILL..."
PIDS=$(pgrep -f "vllm serve" || true)
if [[ -n "$PIDS" ]]; then
  echo "$PIDS" | xargs kill -9 || true
  echo "vLLM killed."
else
  echo "vLLM stopped."
fi

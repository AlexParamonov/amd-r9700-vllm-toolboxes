#!/usr/bin/env bash
set -euo pipefail

# Stop any running vLLM instances (serve + workers + engine core).
# SIGTERM first, wait 15s, then SIGKILL.

PATTERN="vllm serve|VLLM::Worker|VLLM::EngineCore"

if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
  echo "No vLLM processes found."
  exit 0
fi

PIDS=$(pgrep -f "$PATTERN" | tr '\n' ' ')
echo "Stopping vLLM (PIDs: ${PIDS})..."
pkill -f "$PATTERN" || true

# Wait up to 15s.
for i in $(seq 1 15); do
  if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "vLLM stopped."
    exit 0
  fi
  sleep 1
done

# Force kill.
echo "vLLM didn't exit cleanly, sending SIGKILL..."
pkill -9 -f "$PATTERN" || true
sleep 1

if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
  echo "vLLM killed."
else
  echo "[-] vLLM still running after SIGKILL."
  exit 1
fi

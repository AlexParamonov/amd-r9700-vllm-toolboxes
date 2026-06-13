#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --rebuild: drop all caches (vLLM compile, Triton, AITER JIT) and rebuild from scratch
if [[ "${1:-}" == "--rebuild" ]]; then
  echo "[*] Clearing caches..."
  rm -rf ~/.cache/vllm ~/.triton/cache ~/.aiter/jit
  echo "[*] Done. Starting fresh compilation."
  shift
fi

# Pre-flight: ensure GPU device access (host render GID must be in supplementary groups)
if ! test -w /dev/dri/renderD128 2>/dev/null; then
  echo "[*] Adding user to host render group (GID 991)..."
  sudo usermod -aG 991 ap
  exec "$0" "$@"
fi

# Source full ROCm environment from Dockerfile's profile script.
# Sets HIP_PLATFORM, ROCM_PATH, HIP_PATH, LD_LIBRARY_PATH, HIP_ARCHITECTURES, CC/CXX, etc.
# An interactive distrobox shell sources this automatically; non-interactive shells do not.
if [[ -f /etc/profile.d/rocm-sdk.sh ]]; then
  set +u
  source /etc/profile.d/rocm-sdk.sh
  set -u
fi

# Additional ROCm env from 01-rocm-envs.sh (FLASH_ATTENTION, TRITON_AWQ, NCCL, etc.)
if [[ -f /etc/profile.d/01-rocm-envs.sh ]]; then
  set +u
  source /etc/profile.d/01-rocm-envs.sh
  set -u
fi

# Ensure venv is on PATH (zz-venv-last.sh does this for interactive shells)
if [[ -f /opt/venv/bin/activate ]]; then
  source /opt/venv/bin/activate
fi

# Overrides (not in any profile script)
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib:${LD_LIBRARY_PATH:-}
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=1
export SAFETENSORS_FAST_GPU=1
export VLLM_DISABLE_COMPILE_CACHE=1
# export VLLM_LOGGING_LEVEL=DEBUG
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12


# Detect R9700 GPUs and set HIP_VISIBLE_DEVICES (matches start_vllm.py behavior)
gfx1201_indices=()
while IFS= read -r line; do
  if [[ "$line" =~ GPU\[([0-9]+)\] ]]; then
    current_gpu="${BASH_REMATCH[1]}"
  fi
  if [[ -n "${current_gpu:-}" && "$line" =~ gfx1201 ]]; then
    gfx1201_indices+=("$current_gpu")
    current_gpu=""
  fi
done < <(rocm-smi --showproductname 2>/dev/null || true)

if [[ ${#gfx1201_indices[@]} -gt 0 ]]; then
  export HIP_VISIBLE_DEVICES=$(IFS=,; echo "${gfx1201_indices[*]}")
else
  export HIP_VISIBLE_DEVICES=0
fi

vllm serve Qwen/Qwen3.6-27B-FP8 --host 0.0.0.0 --port 8079 --tensor-parallel-size 2 \
  --dtype auto --trust-remote-code --language-model-only \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 16384 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  # --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --override-generation-config '{"temperature": 0.6, "top_p": 0.95, "top_k": 20}' --max-num-seqs 1 \
  --max-model-len 196608 --served-model-name qwen27 --enable-prefix-caching \
  --attention-backend TRITON_ATTN --mm-encoder-attn-backend TRITON_ATTN \
  --compilation-config '{"pass_config":{"fuse_norm_quant":false}}' \
  --chat-template "${SCRIPT_DIR}/template_unsloth.jinja"

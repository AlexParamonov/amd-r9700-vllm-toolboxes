export VLLM_ROCM_USE_AITER=0
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_OVERRIDE_GFX_VERSION=12.0.1

vllm serve Qwen/Qwen3.6-27B-FP8 --port 8080 --quantization fp8 --tensor-parallel-size 2  \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
  --override-generation-config '{"temperature": 0.6}' --max-num-seqs 1 \
  --max-model-len 131072 --served-model-name qwen27 --enable-prefix-caching  \
  --attention-backend TRITON_ATTN --mm-encoder-attn-backend TRITON_ATTN \
  --chat-template /scripts/template_unsloth.jinja

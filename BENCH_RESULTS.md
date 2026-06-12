# FP8 GEMM Benchmark Results

## Run Details
- **Date:** 2026-06-12
- **Device:** AMD R9700 (gfx1201)
- **Script:** `scripts/bench_fp8.py`
- **Search space:** 256 configs per (N,K) shape
- **Batch sizes:** [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256]
- **Block shape:** [128, 128]
- **Duration:** ~24 minutes
- **Model:** Qwen3.6-27B-FP8 (tensor-parallel-size=2)

## Generated Configs

All configs generated on R9700 hardware via `scripts/bench_fp8.py`. Keys: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, kpack, matrix_instr_nonkdim.

### N=5120, K=3072

| M  | BLOCK_SIZE_M | BLOCK_SIZE_N | GROUP_SIZE_M | num_warps | kpack | matrix_instr_nonkdim |
|----|-------------|-------------|-------------|-----------|-------|---------------------|
| 1  | 32          | 32          | 32          | 4         | 1     | 16                  |
| 2  | 32          | 32          | 16          | 4         | 1     | 16                  |
| 4  | 32          | 32          | 32          | 4         | 1     | 16                  |
| 8  | 32          | 32          | 8           | 4         | 2     | 16                  |
| 16 | 32          | 64          | 16          | 8         | 2     | 16                  |
| 24 | 32          | 64          | 8           | 8         | 2     | 16                  |
| 32 | 64          | 32          | 8           | 4         | 1     | 16                  |
| 48 | 64          | 32          | 8           | 4         | 1     | 16                  |
| 64 | 32          | 64          | 16          | 4         | 2     | 16                  |
| 96 | 64          | 128         | 32          | 8         | 1     | 16                  |
| 128| 64          | 64          | 8           | 8         | 2     | 16                  |
| 256| 64          | 128         | 8           | 4         | 1     | 16                  |

### N=5120, K=8704

| M  | BLOCK_SIZE_M | BLOCK_SIZE_N | GROUP_SIZE_M | num_warps | kpack | matrix_instr_nonkdim |
|----|-------------|-------------|-------------|-----------|-------|---------------------|
| 1  | 32          | 32          | 1           | 4         | 2     | 16                  |
| 2  | 32          | 32          | 1           | 4         | 2     | 16                  |
| 4  | 32          | 32          | 32          | 4         | 1     | 16                  |
| 8  | 32          | 32          | 1           | 4         | 1     | 16                  |
| 16 | 32          | 32          | 8           | 4         | 2     | 16                  |
| 24 | 32          | 32          | 32          | 4         | 2     | 16                  |
| 32 | 32          | 32          | 32          | 4         | 2     | 16                  |
| 48 | 64          | 32          | 32          | 4         | 1     | 16                  |
| 64 | 64          | 32          | 32          | 4         | 1     | 16                  |
| 96 | 32          | 128         | 8           | 8         | 2     | 16                  |
| 128| 128         | 32          | 16          | 4         | 1     | 16                  |
| 256| 64          | 128         | 32          | 4         | 2     | 16                  |

### N=7168, K=5120

| M  | BLOCK_SIZE_M | BLOCK_SIZE_N | GROUP_SIZE_M | num_warps | kpack | matrix_instr_nonkdim |
|----|-------------|-------------|-------------|-----------|-------|---------------------|
| 1  | 32          | 32          | 32          | 4         | 1     | 16                  |
| 2  | 32          | 32          | 32          | 4         | 1     | 16                  |
| 4  | 64          | 32          | 16          | 4         | 2     | 16                  |
| 8  | 32          | 32          | 32          | 4         | 2     | 16                  |
| 16 | 32          | 32          | 8           | 4         | 2     | 16                  |
| 24 | 32          | 32          | 1           | 4         | 2     | 16                  |
| 32 | 32          | 32          | 16          | 4         | 1     | 16                  |
| 48 | 64          | 128         | 16          | 8         | 1     | 16                  |
| 64 | 64          | 32          | 8           | 4         | 1     | 16                  |
| 96 | 32          | 128         | 16          | 4         | 2     | 16                  |
| 128| 128         | 128         | 1           | 8         | 2     | 16                  |
| 256| 128         | 64          | 1           | 8         | 2     | 16                  |

### N=17408, K=5120

| M  | BLOCK_SIZE_M | BLOCK_SIZE_N | GROUP_SIZE_M | num_warps | kpack | matrix_instr_nonkdim |
|----|-------------|-------------|-------------|-----------|-------|---------------------|
| 1  | 32          | 32          | 8           | 8         | 2     | 16                  |
| 2  | 32          | 32          | 32          | 8         | 2     | 16                  |
| 4  | 32          | 32          | 1           | 8         | 1     | 16                  |
| 8  | 64          | 32          | 32          | 8         | 2     | 16                  |
| 16 | 32          | 32          | 1           | 8         | 1     | 16                  |
| 24 | 32          | 32          | 1           | 8         | 2     | 16                  |
| 32 | 32          | 64          | 8           | 4         | 1     | 16                  |
| 48 | 64          | 32          | 1           | 4         | 1     | 16                  |
| 64 | 32          | 128         | 32          | 8         | 2     | 16                  |
| 96 | 32          | 32          | 8           | 4         | 2     | 16                  |
| 128| 32          | 32          | 8           | 4         | 1     | 16                  |
| 256| 64          | 128         | 8           | 8         | 2     | 16                  |

## Observations

- **kpack=2 wins at small M** (1-8): FP8 packing benefit outweighs instruction overhead
- **kpack=1 wins at large M** (32+): larger tiles dominate, packing overhead irrelevant
- **num_warps=8 preferred for N=17408**: wider matrices benefit from more warps
- **BLOCK_SIZE_M stays small** (32-64): R9700 has less occupancy than MI300X, smaller M tiles better
- **matrix_instr_nonkdim=16 always wins**: matrix instructions essential for FP8 on gfx1201

## Untrusted Configs

**N=8192, K=5120** (`N=8192,K=5120,device_name=AMD-gfx1201,...json`) exists in the configs directory but was NOT generated by this benchmark. It has different batch sizes (512, 1024, 1536, 2048, 3072, 4096) and uniform kpack=1 across all M values. Likely copied from MI300X. Do not trust.

## To Resume

If configs are lost (container rebuild), re-run:
```bash
# Inside vllm-r9700 distrobox, vLLM stopped:
cd /home/ap/code/amd-r9700-vllm-toolboxes
python scripts/bench_fp8.py
```

The script is idempotent: skips shapes that already have configs in the configs directory.

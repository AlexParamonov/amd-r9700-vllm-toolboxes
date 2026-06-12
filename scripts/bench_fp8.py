#!/usr/bin/env python3
"""
bench_fp8.py -- Generate tuned FP8 W8A8 Block GEMM kernel configs for AMD gfx1201 (R9700).

Benchmarks combinations of BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, and
num_stages for each required (N, K) matrix dimension, finds the fastest config,
and saves it as a JSON file in vLLM's configs directory.

Usage (inside vllm-r9700 distrobox container, with vLLM stopped):
    python scripts/bench_fp8.py [--dry-run] [--batch-sizes 1,2,4,...]

Requirements:
    - ROCm GPU (gfx1201)
    - vLLM installed (for _w8a8_triton_block_scaled_mm kernel)
    - vLLM must be stopped during benchmarking (GPU memory for test tensors)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

# ── Early GPU selection (must happen before torch import) ────────────
# Parse --gpu from argv directly, before argparse, so we can set
# HIP_VISIBLE_DEVICES before torch initializes.
_early_gpu = None
for _i, _arg in enumerate(sys.argv[1:], 1):
    if _arg == "--gpu" and _i + 1 < len(sys.argv):
        _early_gpu = sys.argv[_i + 1]
        break
    elif _arg.startswith("--gpu="):
        _early_gpu = _arg.split("=", 1)[1]
        break
if _early_gpu is not None:
    os.environ["HIP_VISIBLE_DEVICES"] = str(_early_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_early_gpu)

# ── Constants ─────────────────────────────────────────────────────────

# (N, K) combos required for the model (e.g., Qwen3-14B-FP8, Qwen3-27B-FP8)
REQUIRED_SHAPES: list[tuple[int, int]] = [
    (5120, 3072),
    (5120, 8704),
    (7168, 5120),
    (8192, 5120),
    (17408, 5120),
]

# Default batch sizes to benchmark (covers typical decode/prefill ranges)
DEFAULT_BATCH_SIZES: list[int] = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256]


# ── GPU Dependencies (optional — required for benchmarking, not for --dry-run) ──

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _w8a8_triton_block_scaled_mm,
    )
    from vllm.platforms import current_platform
    from vllm.triton_utils import triton
except ImportError:
    _w8a8_triton_block_scaled_mm = None  # type: ignore[assignment,misc]
    current_platform = None  # type: ignore[assignment,misc]
    triton = None  # type: ignore[assignment,misc]


def _ensure_gpu_deps() -> None:
    """Exit with an error if GPU dependencies are not available."""
    if torch is None:
        print("ERROR: torch is required for benchmarking. Install PyTorch with ROCm support.")
        sys.exit(1)
    if _w8a8_triton_block_scaled_mm is None:
        print("ERROR: vLLM is required for benchmarking. Install vLLM first.")
        sys.exit(1)


# ── Pure Python Functions (no torch dependency) ──────────────────────


def get_configs_search_space() -> list[dict[str, Any]]:
    """Generate the search space of kernel configurations to benchmark.

    Returns a list of dicts with keys: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    GROUP_SIZE_M, num_warps, num_stages, kpack, matrix_instr_nonkdim.
    """
    configs = []
    for block_m in [32, 64, 128, 256]:
        for block_n in [32, 64, 128, 256]:
            for block_k in [128]:
                for num_warps in [4, 8]:
                    for group_size in [1, 8, 16, 32]:
                        for kpack in [1, 2]:
                            for matrix_instr_nonkdim in [16]:
                                configs.append(
                                    {
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size,
                                        "num_warps": num_warps,
                                        "kpack": kpack,
                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                    }
                                )
    return configs


def filter_valid_configs(
    search_space: list[dict[str, Any]], block_k: int
) -> list[dict[str, Any]]:
    """Filter configs to those compatible with the block_k size.

    BLOCK_SIZE_K must evenly divide block_k for correct kernel execution.
    """
    return [c for c in search_space if block_k % c["BLOCK_SIZE_K"] == 0]


def get_config_filename(N: int, K: int, block_n: int, block_k: int, device_name: str = "gfx1201") -> str:
    """Generate the config filename matching vLLM's expected format.

    Args:
        N: Output dimension
        K: Input dimension
        block_n: Block size N
        block_k: Block size K
        device_name: Device name (spaces replaced with underscores)
    """
    device_name = device_name.replace(" ", "_")
    return (
        f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,"
        f"block_shape=[{block_n},{block_k}].json"
    )


def config_exists(configs_dir: str, N: int, K: int, block_n: int, block_k: int, device_name: str = "gfx1201") -> bool:
    """Check if a config file already exists for the given shape."""
    filename = get_config_filename(N, K, block_n, block_k, device_name)
    return os.path.exists(os.path.join(configs_dir, filename))


def save_config(
    configs_dir: str,
    N: int,
    K: int,
    block_n: int,
    block_k: int,
    configs: dict[str, dict[str, Any]],
    device_name: str = "gfx1201",
) -> str:
    """Save the config to a JSON file in vLLM's configs directory.

    Returns the path to the saved file.
    """
    os.makedirs(configs_dir, exist_ok=True)
    filename = get_config_filename(N, K, block_n, block_k, device_name)
    filepath = os.path.join(configs_dir, filename)

    with open(filepath, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")

    return filepath


def get_configs_dir() -> str:
    """Get the vLLM configs directory path.

    Tries multiple locations to handle different installation methods.
    """
    # Try the standard venv location
    venv_path = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/configs"
    if os.path.isdir(venv_path):
        return venv_path

    # Try to find via vLLM module
    try:
        import vllm.model_executor.layers.quantization.utils.fp8_utils as fp8_utils
        module_dir = os.path.dirname(os.path.realpath(fp8_utils.__file__))
        configs_dir = os.path.join(module_dir, "configs")
        if os.path.isdir(configs_dir):
            return configs_dir
    except ImportError:
        pass

    # Fallback: use local directory
    local_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
    os.makedirs(local_path, exist_ok=True)
    return local_path


# ── GPU-Dependent Functions ──────────────────────────────────────────


def create_test_tensors(
    M: int, N: int, K: int, block_n: int, block_k: int
) -> tuple:
    """Create FP8 test tensors for benchmarking.

    Returns (A, B, As, Bs) where:
    - A: [M, K] FP8 activation tensor
    - B: [N, K] FP8 weight tensor
    - As: [M, k_tiles] FP32 per-token-group scale
    - Bs: [n_tiles, k_tiles] FP32 per-block scale
    """
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * factor_for_scale

    return A, B, As, Bs


def w8a8_block_matmul(
    A, B, As, Bs, block_size: list[int], config: dict[str, Any], output_dtype=None
):
    """Perform FP8 block-scaled matrix multiplication using the Triton kernel."""
    if output_dtype is None:
        output_dtype = torch.float16

    block_n, block_k = block_size[0], block_size[1]
    M = A.numel() // A.shape[-1]
    N, K = B.shape

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm[grid](
        A, B, C, As, Bs,
        M, N, K,
        block_n, block_k,
        A.stride(-2), A.stride(-1),
        B.stride(1), B.stride(0),
        C.stride(-2), C.stride(-1),
        As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0),
        **config,
    )

    return C


def benchmark_config(
    A, B, As, Bs, block_size: list[int], config: dict[str, Any],
    out_dtype=None, num_iters: int = 10,
) -> float:
    """Benchmark a single kernel configuration.

    Returns median latency in microseconds.
    """
    if out_dtype is None:
        out_dtype = torch.float16

    def run():
        w8a8_block_matmul(A, B, As, Bs, block_size, config, out_dtype)

    torch.accelerator.synchronize()
    # JIT compilation & warmup
    for _ in range(5):
        run()
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
    for _ in range(num_iters):
        torch.accelerator.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    # Use median for stability (robust to thermal spikes)
    latencies.sort()
    median_ms = latencies[len(latencies) // 2]
    median_us = median_ms * 1000  # Convert ms to us
    return median_us


def tune_batch_size(
    M: int, N: int, K: int, block_size: list[int],
    search_space: list[dict[str, Any]], out_dtype=None,
    num_iters: int = 10,
) -> dict[str, Any]:
    """Find the fastest kernel config for a specific (M, N, K) shape.

    Returns the best config dict.
    """
    if out_dtype is None:
        out_dtype = torch.float16

    A, B, As, Bs = create_test_tensors(M, N, K, block_size[0], block_size[1])

    best_config = None
    best_time = float("inf")

    for config in search_space:
        try:
            kernel_time = benchmark_config(
                A, B, As, Bs, block_size, config, out_dtype, num_iters=num_iters
            )
        except triton.runtime.autotuner.OutOfResources:
            # Some configurations may be invalid and fail to compile
            continue

        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config

    assert best_config is not None, f"No valid config found for M={M}, N={N}, K={K}"
    return best_config


def run_benchmarks(
    shapes: list[tuple[int, int]],
    batch_sizes: list[int],
    block_shape: list[int],
    configs_dir: str,
    dry_run: bool = False,
    device_name: str = "gfx1201",
    force: bool = False,
    num_iters: int = 10,
) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    """Run benchmarks for all required (N, K) shapes.

    Returns a dict mapping (N, K) -> {batch_size_str: config_dict}.
    """
    block_n, block_k = block_shape
    search_space = get_configs_search_space()
    search_space = filter_valid_configs(search_space, block_k)

    print(f"Search space: {len(search_space)} configurations")
    print(f"Block shape: [{block_n}, {block_k}]")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Shapes to benchmark: {len(shapes)}")
    print()

    results: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}

    for N, K in shapes:
        if not force and config_exists(configs_dir, N, K, block_n, block_k, device_name):
            filename = get_config_filename(N, K, block_n, block_k, device_name)
            print(f"[SKIP] Config already exists: {filename}")
            continue

        print(f"[BENCH] N={N}, K={K}")
        shape_configs: dict[str, dict[str, Any]] = {}

        for M in batch_sizes:
            if dry_run:
                print(f"  [DRY RUN] Would benchmark M={M}")
                continue

            print(f"  Tuning M={M}...", end="", flush=True)
            best_config = tune_batch_size(M, N, K, [block_n, block_k], search_space, num_iters=num_iters)
            shape_configs[str(M)] = best_config
            print(f" done (best: BLOCK_SIZE_M={best_config['BLOCK_SIZE_M']}, "
                  f"BLOCK_SIZE_N={best_config['BLOCK_SIZE_N']}, "
                  f"GROUP_SIZE_M={best_config['GROUP_SIZE_M']}, "
                  f"num_warps={best_config['num_warps']}, "
                  f"kpack={best_config['kpack']}, "
                  f"matrix_instr_nonkdim={best_config['matrix_instr_nonkdim']})")

        if not dry_run and shape_configs:
            filepath = save_config(configs_dir, N, K, block_n, block_k, shape_configs, device_name)
            print(f"  Saved: {filepath}")
            results[(N, K)] = shape_configs

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate tuned FP8 W8A8 Block GEMM kernel configs for AMD R9700 (gfx1201)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full benchmark run (default batch sizes)
    python scripts/bench_fp8.py

    # Dry run (no actual benchmarking)
    python scripts/bench_fp8.py --dry-run

    # Benchmark on specific GPU
    python scripts/bench_fp8.py --gpu 1

    # Custom batch sizes
    python scripts/bench_fp8.py --batch-sizes 1,8,32,128,512

    # Save to custom directory
    python scripts/bench_fp8.py --save-path ./my_configs
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be benchmarked without running benchmarks",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device index to benchmark on (default: 0). Sets HIP_VISIBLE_DEVICES.",
    )
    parser.add_argument(
<<<<<<< HEAD
        "--iters",
        type=int,
        default=10,
        help="Measurement iterations per config (default: 10, try 50 for stability)",
    )
    parser.add_argument(
=======
>>>>>>> e784b2a (feat(bench_fp8): add --gpu flag for multi-GPU selection)
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated list of batch sizes (M) to benchmark (default: "
        f"{','.join(map(str, DEFAULT_BATCH_SIZES))})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Custom directory to save configs (default: vLLM's configs directory)",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Comma-separated N:K pairs to benchmark (e.g. 8192:5120,4096:3072). Overrides REQUIRED_SHAPES.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-benchmark shapes even if configs already exist (for drift detection)",
    )
    parser.add_argument(
        "--block-n",
        type=int,
        default=128,
        help="Block size N dimension (default: 128)",
    )
    parser.add_argument(
        "--block-k",
        type=int,
        default=128,
        help="Block size K dimension (default: 128)",
    )

    args = parser.parse_args()

    # Parse batch sizes
    if args.batch_sizes:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    else:
        batch_sizes = DEFAULT_BATCH_SIZES

    # Determine configs directory
    if args.save_path:
        configs_dir = args.save_path
    else:
        configs_dir = get_configs_dir()

    # Print environment info
    print("=" * 60)
    print("FP8 W8A8 Block GEMM Benchmark for AMD R9700 (gfx1201)")
    print("=" * 60)

    # Get device name
    device_name = "gfx1201"
    if not args.dry_run:
        _ensure_gpu_deps()
        device_name = current_platform.get_device_name().replace(" ", "_")
        print(f"Device: {current_platform.get_device_name()}")
        if args.gpu is not None:
            print(f"GPU: {args.gpu}")
        else:
            print(f"GPU: 0 (default)")
    else:
        print("Device: (dry run - no GPU required)")
        if args.gpu is not None:
            print(f"GPU: {args.gpu}")

    print(f"Configs directory: {configs_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Parse shapes
    if args.shapes:
        shapes = []
        for pair in args.shapes.split(","):
            n_str, k_str = pair.strip().split(":")
            shapes.append((int(n_str), int(k_str)))
    else:
        shapes = REQUIRED_SHAPES

    # Run benchmarks
    start_time = datetime.now()
    results = run_benchmarks(
        shapes=shapes,
        batch_sizes=batch_sizes,
        block_shape=[args.block_n, args.block_k],
        configs_dir=configs_dir,
        dry_run=args.dry_run,
        device_name=device_name,
        force=args.force,
        num_iters=args.iters,
    )
    end_time = datetime.now()

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Elapsed time: {end_time - start_time}")

    if args.dry_run:
        print("[DRY RUN] No configs were saved")
    else:
        # Check which configs now exist
        all_exist = True
        for N, K in shapes:
            exists = config_exists(configs_dir, N, K, args.block_n, args.block_k, device_name)
            filename = get_config_filename(N, K, args.block_n, args.block_k, device_name)
            status = "EXISTS" if exists else "MISSING"
            print(f"  [{status}] {filename}")
            if not exists:
                all_exist = False

        if all_exist:
            print("\nAll required configs are present!")
        else:
            print("\nWARNING: Some configs are still missing!")
            sys.exit(1)


if __name__ == "__main__":
    main()

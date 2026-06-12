#!/usr/bin/env python3
"""
Tests for bench_fp8.py

Run with: pytest scripts/test_bench_fp8.py -v

These tests verify the pure Python logic of the benchmark script.
GPU-dependent functions require actual hardware and are tested via integration.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Test Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def temp_configs_dir(tmp_path):
    """Create a temporary configs directory."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    return str(configs_dir)


@pytest.fixture
def sample_config():
    """Return a sample kernel configuration."""
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 2,
        "kpack": 1,
        "matrix_instr_nonkdim": 16,
    }


@pytest.fixture
def sample_configs_dict(sample_config):
    """Return a sample configs dict mapping batch sizes to configs."""
    return {
        "1": sample_config,
        "2": sample_config,
        "4": {**sample_config, "BLOCK_SIZE_M": 16},
        "8": sample_config,
        "16": sample_config,
    }


# ── Test: Search Space Generation ─────────────────────────────────────


class TestSearchSpace:
    """Tests for kernel configuration search space generation."""

    def test_search_space_not_empty(self):
        from bench_fp8 import get_configs_search_space

        configs = get_configs_search_space()
        assert len(configs) > 0

    def test_search_space_has_required_keys(self):
        from bench_fp8 import get_configs_search_space

        configs = get_configs_search_space()
        required_keys = {
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
            "kpack",
            "matrix_instr_nonkdim",
        }
        for config in configs:
            assert required_keys.issubset(config.keys()), (
                f"Config missing keys: {required_keys - config.keys()}"
            )

    def test_search_space_values_in_valid_ranges(self):
        from bench_fp8 import get_configs_search_space

        configs = get_configs_search_space()
        for config in configs:
            assert config["BLOCK_SIZE_M"] in [16, 32, 64, 128, 256]
            assert config["BLOCK_SIZE_N"] in [32, 64, 128, 256]
            assert config["BLOCK_SIZE_K"] in [64, 128]
            assert config["GROUP_SIZE_M"] in [1, 16, 32, 64]
            assert config["num_warps"] in [4, 8]
            assert config["num_stages"] in [2, 3, 4, 5]
            assert config["kpack"] in [1, 2]
            assert config["matrix_instr_nonkdim"] in [0, 16]

    def test_filter_valid_configs_block_k_128(self):
        from bench_fp8 import filter_valid_configs, get_configs_search_space

        search_space = get_configs_search_space()
        filtered = filter_valid_configs(search_space, block_k=128)

        # All filtered configs should have BLOCK_SIZE_K dividing 128
        for config in filtered:
            assert 128 % config["BLOCK_SIZE_K"] == 0

        # Should include both BLOCK_SIZE_K=64 and BLOCK_SIZE_K=128
        block_k_values = {c["BLOCK_SIZE_K"] for c in filtered}
        assert block_k_values == {64, 128}

    def test_filter_valid_configs_block_k_64(self):
        from bench_fp8 import filter_valid_configs, get_configs_search_space

        search_space = get_configs_search_space()
        filtered = filter_valid_configs(search_space, block_k=64)

        # Should only include BLOCK_SIZE_K=64
        for config in filtered:
            assert config["BLOCK_SIZE_K"] == 64


# ── Test: Config Filename Generation ──────────────────────────────────


class TestConfigFilename:
    """Tests for config filename generation."""

    def test_filename_format(self):
        from bench_fp8 import get_config_filename

        filename = get_config_filename(5120, 3072, 128, 128, "AMD-gfx1201")

        assert filename == "N=5120,K=3072,device_name=AMD-gfx1201,dtype=fp8_w8a8,block_shape=[128,128].json"

    def test_filename_spaces_replaced(self):
        from bench_fp8 import get_config_filename

        filename = get_config_filename(5120, 3072, 128, 128, "AMD Radeon R9700")

        assert " " not in filename
        assert "AMD_Radeon_R9700" in filename

    def test_filename_contains_all_params(self):
        from bench_fp8 import get_config_filename

        filename = get_config_filename(17408, 5120, 128, 128, "gfx1201")

        assert "N=17408" in filename
        assert "K=5120" in filename
        assert "device_name=gfx1201" in filename
        assert "dtype=fp8_w8a8" in filename
        assert "block_shape=[128,128]" in filename

    def test_filename_default_device(self):
        from bench_fp8 import get_config_filename

        filename = get_config_filename(5120, 3072, 128, 128)

        assert "device_name=gfx1201" in filename

    def test_filename_different_block_sizes(self):
        from bench_fp8 import get_config_filename

        filename = get_config_filename(5120, 3072, 64, 64, "gfx1201")

        assert "block_shape=[64,64]" in filename


# ── Test: Config Existence Check ──────────────────────────────────────


class TestConfigExists:
    """Tests for checking if config files exist."""

    def test_config_exists_when_present(self, temp_configs_dir):
        from bench_fp8 import config_exists, get_config_filename

        filename = get_config_filename(5120, 3072, 128, 128, "gfx1201")
        filepath = os.path.join(temp_configs_dir, filename)

        # Create the file
        with open(filepath, "w") as f:
            json.dump({"1": {"BLOCK_SIZE_M": 16}}, f)

        assert config_exists(temp_configs_dir, 5120, 3072, 128, 128, "gfx1201") is True

    def test_config_not_exists_when_absent(self, temp_configs_dir):
        from bench_fp8 import config_exists

        assert config_exists(temp_configs_dir, 5120, 3072, 128, 128, "gfx1201") is False

    def test_config_exists_different_device(self, temp_configs_dir):
        from bench_fp8 import config_exists, save_config

        # Save with one device name
        save_config(temp_configs_dir, 5120, 3072, 128, 128, {"1": {}}, "gfx1201")

        # Check with different device name
        assert config_exists(temp_configs_dir, 5120, 3072, 128, 128, "gfx1202") is False


# ── Test: Config Save/Load ────────────────────────────────────────────


class TestConfigSaveLoad:
    """Tests for saving and loading config files."""

    def test_save_config_creates_file(self, temp_configs_dir, sample_configs_dict):
        from bench_fp8 import save_config

        filepath = save_config(temp_configs_dir, 5120, 3072, 128, 128, sample_configs_dict, "gfx1201")

        assert os.path.exists(filepath)
        assert filepath.endswith(".json")

    def test_save_config_valid_json(self, temp_configs_dir, sample_configs_dict):
        from bench_fp8 import save_config

        filepath = save_config(temp_configs_dir, 5120, 3072, 128, 128, sample_configs_dict, "gfx1201")

        with open(filepath) as f:
            loaded = json.load(f)

        assert loaded == sample_configs_dict

    def test_save_config_json_format(self, temp_configs_dir, sample_configs_dict):
        from bench_fp8 import save_config

        filepath = save_config(temp_configs_dir, 5120, 3072, 128, 128, sample_configs_dict, "gfx1201")

        with open(filepath) as f:
            loaded = json.load(f)

        # Verify structure matches vLLM format (keys: BLOCK_SIZE_M/N/K, GROUP_SIZE_M, kpack, matrix_instr_nonkdim, num_warps)
        for batch_size, config in loaded.items():
            assert isinstance(batch_size, str)  # Keys are strings
            assert "BLOCK_SIZE_M" in config
            assert "BLOCK_SIZE_N" in config
            assert "BLOCK_SIZE_K" in config
            assert "GROUP_SIZE_M" in config
            assert "num_warps" in config
            assert "kpack" in config
            assert "matrix_instr_nonkdim" in config

    def test_save_config_creates_directory(self, tmp_path, sample_configs_dict):
        from bench_fp8 import save_config

        nested_dir = str(tmp_path / "deeply" / "nested" / "configs")
        filepath = save_config(nested_dir, 5120, 3072, 128, 128, sample_configs_dict, "gfx1201")

        assert os.path.exists(filepath)

    def test_save_config_with_different_device(self, temp_configs_dir, sample_configs_dict):
        from bench_fp8 import save_config

        filepath = save_config(temp_configs_dir, 5120, 3072, 128, 128, sample_configs_dict, "AMD-gfx1201")

        assert "AMD-gfx1201" in filepath


# ── Test: Required Shapes ─────────────────────────────────────────────


class TestRequiredShapes:
    """Tests for required (N, K) shape definitions."""

    def test_required_shapes_not_empty(self):
        from bench_fp8 import REQUIRED_SHAPES

        assert len(REQUIRED_SHAPES) > 0

    def test_required_shapes_are_tuples(self):
        from bench_fp8 import REQUIRED_SHAPES

        for shape in REQUIRED_SHAPES:
            assert isinstance(shape, tuple)
            assert len(shape) == 2

    def test_required_shapes_match_spec(self):
        from bench_fp8 import REQUIRED_SHAPES

        expected = [
            (5120, 3072),
            (5120, 8704),
            (7168, 5120),
            (17408, 5120),
        ]
        assert REQUIRED_SHAPES == expected

    def test_required_shapes_positive(self):
        from bench_fp8 import REQUIRED_SHAPES

        for N, K in REQUIRED_SHAPES:
            assert N > 0
            assert K > 0


# ── Test: Default Batch Sizes ─────────────────────────────────────────


class TestDefaultBatchSizes:
    """Tests for default batch size configuration."""

    def test_default_batch_sizes_not_empty(self):
        from bench_fp8 import DEFAULT_BATCH_SIZES

        assert len(DEFAULT_BATCH_SIZES) > 0

    def test_default_batch_sizes_positive(self):
        from bench_fp8 import DEFAULT_BATCH_SIZES

        for bs in DEFAULT_BATCH_SIZES:
            assert bs > 0

    def test_default_batch_sizes_sorted(self):
        from bench_fp8 import DEFAULT_BATCH_SIZES

        assert DEFAULT_BATCH_SIZES == sorted(DEFAULT_BATCH_SIZES)

    def test_default_batch_sizes_include_small_values(self):
        from bench_fp8 import DEFAULT_BATCH_SIZES

        # Should include small batch sizes for decode
        assert 1 in DEFAULT_BATCH_SIZES
        assert 2 in DEFAULT_BATCH_SIZES

    def test_default_batch_sizes_include_large_values(self):
        from bench_fp8 import DEFAULT_BATCH_SIZES

        # Should include larger batch sizes for prefill
        assert any(bs >= 128 for bs in DEFAULT_BATCH_SIZES)


# ── Test: Idempotency ─────────────────────────────────────────────────


class TestIdempotency:
    """Tests for idempotent behavior (skipping existing configs)."""

    def test_save_config_overwrites_existing(self, temp_configs_dir, sample_configs_dict):
        from bench_fp8 import config_exists, save_config

        # Save initial config
        save_config(temp_configs_dir, 5120, 3072, 128, 128, sample_configs_dict, "gfx1201")
        assert config_exists(temp_configs_dir, 5120, 3072, 128, 128, "gfx1201") is True

        # Save again with different data (should overwrite)
        new_config = {"1": {"BLOCK_SIZE_M": 32}}
        save_config(temp_configs_dir, 5120, 3072, 128, 128, new_config, "gfx1201")

        # Verify it was overwritten
        filename = "N=5120,K=3072,device_name=gfx1201,dtype=fp8_w8a8,block_shape=[128,128].json"
        with open(os.path.join(temp_configs_dir, filename)) as f:
            loaded = json.load(f)
        assert loaded == new_config

    def test_multiple_configs_independent(self, temp_configs_dir, sample_configs_dict):
        from bench_fp8 import config_exists, save_config

        # Save configs for different shapes
        save_config(temp_configs_dir, 5120, 3072, 128, 128, sample_configs_dict, "gfx1201")
        save_config(temp_configs_dir, 5120, 8704, 128, 128, sample_configs_dict, "gfx1201")

        # Both should exist
        assert config_exists(temp_configs_dir, 5120, 3072, 128, 128, "gfx1201") is True
        assert config_exists(temp_configs_dir, 5120, 8704, 128, 128, "gfx1201") is True

        # Other shapes should not exist
        assert config_exists(temp_configs_dir, 7168, 5120, 128, 128, "gfx1201") is False


# ── Test: Get Configs Dir ─────────────────────────────────────────────


class TestGetConfigsDir:
    """Tests for configs directory detection."""

    @patch("os.path.isdir")
    def test_venv_path_preferred(self, mock_isdir):
        from bench_fp8 import get_configs_dir

        def isdir_side_effect(path):
            return path == "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/configs"

        mock_isdir.side_effect = isdir_side_effect

        result = get_configs_dir()
        assert result == "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/configs"

    @patch("os.path.isdir")
    @patch("os.makedirs")
    def test_fallback_to_local(self, mock_makedirs, mock_isdir):
        from bench_fp8 import get_configs_dir

        mock_isdir.return_value = False

        result = get_configs_dir()
        assert result.endswith("configs")
        mock_makedirs.assert_called_once()


# ── Test: CLI Argument Parsing ────────────────────────────────────────


class TestCLI:
    """Tests for command-line argument parsing."""

    def test_dry_run_flag(self):
        from bench_fp8 import main

        with patch("sys.argv", ["bench_fp8.py", "--dry-run"]):
            # Should not raise
            main()

    def test_batch_sizes_arg(self):
        from bench_fp8 import main

        with patch("sys.argv", ["bench_fp8.py", "--dry-run", "--batch-sizes", "1,4,16"]):
            # Should not raise
            main()

    def test_save_path_arg(self):
        from bench_fp8 import main

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", ["bench_fp8.py", "--dry-run", "--save-path", tmpdir]):
                # Should not raise
                main()

    def test_block_n_arg(self):
        from bench_fp8 import main

        with patch("sys.argv", ["bench_fp8.py", "--dry-run", "--block-n", "64"]):
            # Should not raise
            main()

    def test_block_k_arg(self):
        from bench_fp8 import main

        with patch("sys.argv", ["bench_fp8.py", "--dry-run", "--block-k", "64"]):
            # Should not raise
            main()

    def test_default_args(self):
        from bench_fp8 import main

        with patch("sys.argv", ["bench_fp8.py", "--dry-run"]):
            # Should not raise
            main()


# ── Test: Integration Helpers ─────────────────────────────────────────


class TestIntegrationHelpers:
    """Tests that verify the benchmark flow without GPU."""

    def test_run_benchmarks_dry_run(self, temp_configs_dir):
        from bench_fp8 import run_benchmarks

        # Run with dry_run=True - should not crash
        results = run_benchmarks(
            shapes=[(5120, 3072)],
            batch_sizes=[1, 2],
            block_shape=[128, 128],
            configs_dir=temp_configs_dir,
            dry_run=True,
            device_name="gfx1201",
        )

        # No results in dry run
        assert len(results) == 0

    def test_run_benchmarks_skips_existing(self, temp_configs_dir, sample_configs_dict):
        from bench_fp8 import run_benchmarks, save_config

        # Pre-create a config
        save_config(temp_configs_dir, 5120, 3072, 128, 128, sample_configs_dict, "gfx1201")

        # Run benchmarks - should skip the existing config
        results = run_benchmarks(
            shapes=[(5120, 3072)],
            batch_sizes=[1, 2],
            block_shape=[128, 128],
            configs_dir=temp_configs_dir,
            device_name="gfx1201",
        )

        # No new results (skipped)
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

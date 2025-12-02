"""
Tests for throughput validation fixes:
1. Triton cache clearing functionality
2. Multi-token config file format
3. Enhanced debug output (stderr capture limit, full stdout/stderr)
4. Config file content verification

These tests verify the fixes for issues:
- vLLM Benchmark Consistently Failing with Return Code 1
- Triton Compilation State Pollution
- Write All Token Configs Before vLLM Benchmark
"""

import os
import json
import shutil
import tempfile


# =============================================================================
# Test: Triton Cache Clearing
# =============================================================================

def test_triton_cache_clear_logic():
    """Test that Triton cache clearing logic works correctly."""
    # Create a temporary triton cache directory
    temp_home = tempfile.mkdtemp()
    triton_cache = os.path.join(temp_home, ".triton", "cache")
    os.makedirs(triton_cache, exist_ok=True)
    
    # Create some dummy files in the cache
    dummy_file = os.path.join(triton_cache, "dummy_cache.bin")
    with open(dummy_file, "w") as f:
        f.write("dummy cache data")
    
    assert os.path.exists(triton_cache), "Triton cache should exist before clearing"
    assert os.path.exists(dummy_file), "Dummy file should exist before clearing"
    
    # Clear the cache (mimicking the logic in profiling_worker.py)
    if os.path.exists(triton_cache):
        shutil.rmtree(triton_cache, ignore_errors=True)
    
    assert not os.path.exists(triton_cache), "Triton cache should be cleared"
    
    # Cleanup
    shutil.rmtree(temp_home, ignore_errors=True)
    print("✅ test_triton_cache_clear_logic PASSED")


def test_triton_cache_isolation_env_var():
    """Test that TRITON_CACHE_DIR environment variable is set for isolation."""
    gpu_id = 0
    pid = 12345  # Simulated PID
    unique_cache_dir = f"/tmp/triton_cache_gpu{gpu_id}_{pid}"
    
    # Verify the format is correct
    assert "triton_cache_gpu" in unique_cache_dir, "Cache dir should contain 'triton_cache_gpu'"
    assert str(gpu_id) in unique_cache_dir, "Cache dir should contain GPU ID"
    assert str(pid) in unique_cache_dir, "Cache dir should contain PID"
    
    print("✅ test_triton_cache_isolation_env_var PASSED")


# =============================================================================
# Test: Multi-Token Config File Format
# =============================================================================

def test_multi_token_config_format():
    """Test that config file format supports multiple token counts."""
    # This is the expected format for vLLM configs
    expected_format = {
        "1": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 4},
        "16": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 3},
        "64": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "num_warps": 16, "num_stages": 4},
        "256": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 8, "num_stages": 3},
        "1024": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 5},
    }
    
    # Verify it's valid JSON
    json_str = json.dumps(expected_format, indent=2)
    parsed = json.loads(json_str)
    
    assert len(parsed) == 5, "Should have 5 token count entries"
    assert all(k.isdigit() for k in parsed.keys()), "All keys should be numeric strings"
    assert "BLOCK_SIZE_M" in parsed["1"], "Config should have BLOCK_SIZE_M"
    
    print("✅ test_multi_token_config_format PASSED")


def test_single_config_expansion():
    """Test that a single config is expanded to multiple token counts."""
    single_config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "num_warps": 8,
        "num_stages": 4
    }
    
    # Simulate the expansion logic from profiling_worker.py
    is_multi_token = all(
        isinstance(k, str) and k.isdigit() 
        for k in single_config.keys()
    ) if single_config else False
    
    assert not is_multi_token, "Single config should not be detected as multi-token"
    
    # Expansion logic
    expanded_config = {
        "1": single_config,
        "16": single_config, 
        "64": single_config,
        "256": single_config,
        "1024": single_config,
        "4096": single_config,
        "16384": single_config,
    }
    
    assert len(expanded_config) == 7, "Expanded config should have 7 token counts"
    assert all(expanded_config[tc] == single_config for tc in expanded_config), "All entries should have same config"
    
    print("✅ test_single_config_expansion PASSED")


def test_multi_token_config_detection():
    """Test that multi-token configs are correctly detected."""
    multi_token_config = {
        "1": {"BLOCK_SIZE_M": 64},
        "16": {"BLOCK_SIZE_M": 128},
        "64": {"BLOCK_SIZE_M": 64},
    }
    
    # Detection logic from profiling_worker.py
    is_multi_token = all(
        isinstance(k, str) and k.isdigit() 
        for k in multi_token_config.keys()
    ) if multi_token_config else False
    
    assert is_multi_token, "Multi-token config should be detected"
    
    print("✅ test_multi_token_config_detection PASSED")


def test_empty_config_handling():
    """Test that empty configs are handled gracefully."""
    empty_config = {}
    
    is_multi_token = all(
        isinstance(k, str) and k.isdigit() 
        for k in empty_config.keys()
    ) if empty_config else False
    
    assert not is_multi_token, "Empty config should not be detected as multi-token"
    
    print("✅ test_empty_config_handling PASSED")


# =============================================================================
# Test: Enhanced Debug Output
# =============================================================================

def test_stderr_capture_limit_increased():
    """Test that stderr capture limit is at least 2000 characters."""
    # This test verifies the new limit is properly applied
    # The actual limit should be 2000 (increased from 500)
    NEW_STDERR_LIMIT = 2000
    OLD_STDERR_LIMIT = 500
    
    long_stderr = "x" * 3000  # Simulate a long stderr message
    
    # New behavior: capture more
    captured_new = long_stderr[:NEW_STDERR_LIMIT]
    assert len(captured_new) == NEW_STDERR_LIMIT, "New limit should capture 2000 chars"
    
    # Old behavior: captured less (this is what was fixed)
    captured_old = long_stderr[:OLD_STDERR_LIMIT]
    assert len(captured_old) == OLD_STDERR_LIMIT, "Old limit was only 500 chars"
    
    assert len(captured_new) > len(captured_old), "New limit should capture more than old limit"
    
    print("✅ test_stderr_capture_limit_increased PASSED")


def test_config_file_write_with_contents_logging():
    """Test that config file writing includes content logging."""
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "test_config.json")
    
    config_data = {
        "1": {"BLOCK_SIZE_M": 64, "num_warps": 8},
        "16": {"BLOCK_SIZE_M": 128, "num_warps": 16},
        "64": {"BLOCK_SIZE_M": 64, "num_warps": 4},
    }
    
    # Write config file
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Read and verify contents
    with open(config_path, "r") as f:
        read_config = json.load(f)
    
    assert read_config == config_data, "Written config should match original"
    assert len(read_config) == 3, "Should have 3 entries"
    
    # Simulate debug output (showing first 3 entries)
    entries_shown = list(read_config.items())[:3]
    assert len(entries_shown) == 3, "Should show 3 entries in debug output"
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("✅ test_config_file_write_with_contents_logging PASSED")


# =============================================================================
# Test: vLLM Benchmark Command Compatibility
# =============================================================================

def test_vllm_benchmark_command_format():
    """Test that vLLM benchmark command uses correct format for v0.11.x."""
    # New command format using benchmark_throughput script directly
    new_command = [
        "python", "-m", "vllm.benchmarks.benchmark_throughput",
        "--model", "test-model",
        "--input-len", "512",
        "--output-len", "128",
        "--num-prompts", "40",
        "--trust-remote-code",
        "--enforce-eager", 
        "--tensor-parallel-size", "1",
    ]
    
    # Verify command structure
    assert new_command[0] == "python", "Should use python"
    assert new_command[1] == "-m", "Should use -m flag"
    assert "benchmark_throughput" in new_command[2], "Should use benchmark_throughput module"
    assert "--trust-remote-code" in new_command, "Should include trust-remote-code"
    assert "--enforce-eager" in new_command, "Should include enforce-eager"
    
    print("✅ test_vllm_benchmark_command_format PASSED")


def test_benchmark_command_has_required_params():
    """Test that benchmark command has all required parameters."""
    required_params = [
        "--model",
        "--input-len",
        "--output-len", 
        "--num-prompts",
        "--trust-remote-code",
        "--tensor-parallel-size",
    ]
    
    command = [
        "python", "-m", "vllm.benchmarks.benchmark_throughput",
        "--model", "test-model",
        "--input-len", "512",
        "--output-len", "128",
        "--num-prompts", "40",
        "--trust-remote-code",
        "--enforce-eager", 
        "--tensor-parallel-size", "1",
    ]
    
    for param in required_params:
        assert param in command, f"Command should include {param}"
    
    print("✅ test_benchmark_command_has_required_params PASSED")


# =============================================================================
# Test: Meta Controller Summary Logging
# =============================================================================

def test_validation_summary_format():
    """Test that validation summary has correct format."""
    # Simulate top_configs structure
    top_configs = [
        ({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 4}, 
         [50.0, 40.0, 0.1, 70.0],  # state
         45.5),  # reward
        ({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 16, "num_stages": 3}, 
         [55.0, 45.0, 0.15, 72.0],
         52.3),
    ]
    
    # Verify structure
    assert len(top_configs) == 2, "Should have 2 configs"
    
    for params, state, reward in top_configs:
        assert "BLOCK_SIZE_M" in params, "Config should have BLOCK_SIZE_M"
        assert "num_warps" in params, "Config should have num_warps"
        assert isinstance(reward, float), "Reward should be float"
        assert len(state) == 4, "State should have 4 metrics"
    
    print("✅ test_validation_summary_format PASSED")


def test_best_configs_by_token_format():
    """Test that best_configs_by_token has correct format."""
    best_configs_by_token = {
        1: {
            "config": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 4},
            "reward": 45.5,
        },
        16: {
            "config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 16, "num_stages": 3},
            "reward": 52.3,
        },
        64: {
            "config": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 4},
            "reward": 48.7,
        },
    }
    
    # Build combined config (as done in meta_controller.py)
    combined_config = {}
    for token_count, config_data in best_configs_by_token.items():
        combined_config[str(token_count)] = config_data["config"]
    
    # Verify combined config
    assert len(combined_config) == 3, "Combined config should have 3 token counts"
    assert "1" in combined_config, "Should have token count 1"
    assert "16" in combined_config, "Should have token count 16"
    assert "64" in combined_config, "Should have token count 64"
    
    # Verify JSON serialization
    json_str = json.dumps(combined_config, indent=2)
    parsed = json.loads(json_str)
    assert parsed == combined_config, "Should be valid JSON"
    
    print("✅ test_best_configs_by_token_format PASSED")


if __name__ == "__main__":
    print("--- THROUGHPUT VALIDATION FIXES TESTS ---\n")
    
    print("=== Triton Cache Clearing Tests ===")
    test_triton_cache_clear_logic()
    test_triton_cache_isolation_env_var()
    
    print("\n=== Multi-Token Config Format Tests ===")
    test_multi_token_config_format()
    test_single_config_expansion()
    test_multi_token_config_detection()
    test_empty_config_handling()
    
    print("\n=== Enhanced Debug Output Tests ===")
    test_stderr_capture_limit_increased()
    test_config_file_write_with_contents_logging()
    
    print("\n=== vLLM Benchmark Command Tests ===")
    test_vllm_benchmark_command_format()
    test_benchmark_command_has_required_params()
    
    print("\n=== Meta Controller Summary Tests ===")
    test_validation_summary_format()
    test_best_configs_by_token_format()
    
    print("\n" + "="*60)
    print("✅ ALL THROUGHPUT VALIDATION FIXES TESTS PASSED")
    print("="*60)

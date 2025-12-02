"""
Tests for direct LLM-based kernel configuration optimization.

These tests verify that the system has been correctly updated to:
1. Remove PPO exploration agent dependencies
2. Parse LLM outputs as direct kernel configurations
3. Test configurations directly via ProfilingWorker
4. Maintain backward compatibility

Note: These tests verify the changes by parsing the source code directly,
following the pattern used in other tests in this repository to avoid
heavy dependencies like ray, vllm, etc.
"""

import re
import ast
import json


def get_meta_controller_source():
    """Read the meta_controller.py source file."""
    with open('meta_controller.py', 'r') as f:
        return f.read()


def get_hierarchical_optimizer_source():
    """Read the hierarchical_kernel_optimizer.py source file."""
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        return f.read()


def get_feedback_collector_source():
    """Read the feedback_collector.py source file."""
    with open('feedback_collector.py', 'r') as f:
        return f.read()


# ============================================================================
# Tests for meta_controller.py
# ============================================================================

def test_no_exploration_agent_import():
    """Test that ExplorationAgent is NOT imported in meta_controller."""
    print("\n=== Test: No ExplorationAgent import ===")
    
    source = get_meta_controller_source()
    
    # Check that ExplorationAgent is NOT imported
    pattern = r'from exploration_agent import|import exploration_agent'
    match = re.search(pattern, source)
    
    if not match:
        print("✅ ExplorationAgent is NOT imported")
        return True
    else:
        print("❌ ExplorationAgent is still imported")
        return False


def test_no_kernel_tuning_env_import():
    """Test that KernelTuningEnvironment is NOT imported in meta_controller."""
    print("\n=== Test: No KernelTuningEnvironment import ===")
    
    source = get_meta_controller_source()
    
    # Check that KernelTuningEnvironment is NOT imported
    pattern = r'from kernel_tuning_env import|import kernel_tuning_env'
    match = re.search(pattern, source)
    
    if not match:
        print("✅ KernelTuningEnvironment is NOT imported")
        return True
    else:
        print("❌ KernelTuningEnvironment is still imported")
        return False


def test_default_configurations_defined():
    """Test that DEFAULT_CONFIGURATIONS is defined with correct structure."""
    print("\n=== Test: DEFAULT_CONFIGURATIONS defined ===")
    
    source = get_meta_controller_source()
    
    # Check that DEFAULT_CONFIGURATIONS is defined
    pattern = r'DEFAULT_CONFIGURATIONS\s*=\s*\['
    match = re.search(pattern, source)
    
    if match:
        print("✅ DEFAULT_CONFIGURATIONS is defined")
        return True
    else:
        print("❌ DEFAULT_CONFIGURATIONS is NOT defined")
        return False


def test_default_objective_weights_defined():
    """Test that DEFAULT_OBJECTIVE_WEIGHTS is defined."""
    print("\n=== Test: DEFAULT_OBJECTIVE_WEIGHTS defined ===")
    
    source = get_meta_controller_source()
    
    # Check that DEFAULT_OBJECTIVE_WEIGHTS is defined
    pattern = r'DEFAULT_OBJECTIVE_WEIGHTS\s*=\s*\{'
    match = re.search(pattern, source)
    
    if match:
        print("✅ DEFAULT_OBJECTIVE_WEIGHTS is defined")
        return True
    else:
        print("❌ DEFAULT_OBJECTIVE_WEIGHTS is NOT defined")
        return False


def test_validate_and_coerce_configurations_exists():
    """Test that _validate_and_coerce_configurations method exists."""
    print("\n=== Test: _validate_and_coerce_configurations exists ===")
    
    source = get_meta_controller_source()
    
    # Check for the method definition
    pattern = r'def _validate_and_coerce_configurations\s*\('
    match = re.search(pattern, source)
    
    if match:
        print("✅ _validate_and_coerce_configurations method exists")
        return True
    else:
        print("❌ _validate_and_coerce_configurations method NOT found")
        return False


def test_validate_single_config_exists():
    """Test that _validate_single_config method exists."""
    print("\n=== Test: _validate_single_config exists ===")
    
    source = get_meta_controller_source()
    
    # Check for the method definition
    pattern = r'def _validate_single_config\s*\('
    match = re.search(pattern, source)
    
    if match:
        print("✅ _validate_single_config method exists")
        return True
    else:
        print("❌ _validate_single_config method NOT found")
        return False


def test_run_direct_testing_phase_exists():
    """Test that _run_direct_testing_phase method exists."""
    print("\n=== Test: _run_direct_testing_phase exists ===")
    
    source = get_meta_controller_source()
    
    # Check for the method definition
    pattern = r'def _run_direct_testing_phase\s*\('
    match = re.search(pattern, source)
    
    if match:
        print("✅ _run_direct_testing_phase method exists")
        return True
    else:
        print("❌ _run_direct_testing_phase method NOT found")
        return False


def test_default_safe_configurations_exists():
    """Test that _default_safe_configurations method exists."""
    print("\n=== Test: _default_safe_configurations exists ===")
    
    source = get_meta_controller_source()
    
    # Check for the method definition
    pattern = r'def _default_safe_configurations\s*\('
    match = re.search(pattern, source)
    
    if match:
        print("✅ _default_safe_configurations method exists")
        return True
    else:
        print("❌ _default_safe_configurations method NOT found")
        return False


def test_run_default_configurations_exists():
    """Test that _run_default_configurations method exists."""
    print("\n=== Test: _run_default_configurations exists ===")
    
    source = get_meta_controller_source()
    
    # Check for the method definition
    pattern = r'def _run_default_configurations\s*\('
    match = re.search(pattern, source)
    
    if match:
        print("✅ _run_default_configurations method exists")
        return True
    else:
        print("❌ _run_default_configurations method NOT found")
        return False


def test_configs_per_iteration_defined():
    """Test that CONFIGS_PER_ITERATION is defined."""
    print("\n=== Test: CONFIGS_PER_ITERATION defined ===")
    
    source = get_meta_controller_source()
    
    # Check that CONFIGS_PER_ITERATION is defined
    pattern = r'CONFIGS_PER_ITERATION\s*=\s*\d+'
    match = re.search(pattern, source)
    
    if match:
        print("✅ CONFIGS_PER_ITERATION is defined")
        return True
    else:
        print("❌ CONFIGS_PER_ITERATION is NOT defined")
        return False


# ============================================================================
# Tests for hierarchical_kernel_optimizer.py
# ============================================================================

def test_prompt_asks_for_configurations():
    """Test that the prompt asks for specific configurations."""
    print("\n=== Test: Prompt asks for configurations ===")
    
    source = get_hierarchical_optimizer_source()
    
    # Check that the prompt mentions "configurations"
    pattern = r'"configurations":'
    match = re.search(pattern, source)
    
    if match:
        print("✅ Prompt asks for 'configurations'")
        return True
    else:
        print("❌ Prompt does NOT ask for 'configurations'")
        return False


def test_prompt_example_format():
    """Test that the prompt includes example configuration format."""
    print("\n=== Test: Prompt includes example config format ===")
    
    source = get_hierarchical_optimizer_source()
    
    # Check for example configuration with BLOCK_SIZE_M, num_warps, etc.
    patterns = [
        r'"BLOCK_SIZE_M":\s*\d+',
        r'"BLOCK_SIZE_N":\s*\d+',
        r'"num_warps":\s*\d+',
        r'"num_stages":\s*\d+',
    ]
    
    all_found = True
    for pattern in patterns:
        if not re.search(pattern, source):
            print(f"❌ Pattern not found: {pattern}")
            all_found = False
    
    if all_found:
        print("✅ Prompt includes all required configuration fields")
        return True
    return False


def test_prompt_includes_reasoning_field():
    """Test that the prompt example includes reasoning field."""
    print("\n=== Test: Prompt includes reasoning field ===")
    
    source = get_hierarchical_optimizer_source()
    
    # Check for reasoning field in example
    pattern = r'"reasoning":\s*"'
    match = re.search(pattern, source)
    
    if match:
        print("✅ Prompt includes reasoning field")
        return True
    else:
        print("❌ Prompt does NOT include reasoning field")
        return False


# ============================================================================
# Tests for feedback_collector.py
# ============================================================================

def test_record_configuration_results_exists():
    """Test that record_configuration_results method exists."""
    print("\n=== Test: record_configuration_results exists ===")
    
    source = get_feedback_collector_source()
    
    # Check for the method definition
    pattern = r'def record_configuration_results\s*\('
    match = re.search(pattern, source)
    
    if match:
        print("✅ record_configuration_results method exists")
        return True
    else:
        print("❌ record_configuration_results method NOT found")
        return False


def test_analyze_successful_configurations_exists():
    """Test that _analyze_successful_configurations method exists."""
    print("\n=== Test: _analyze_successful_configurations exists ===")
    
    source = get_feedback_collector_source()
    
    # Check for the method definition
    pattern = r'def _analyze_successful_configurations\s*\('
    match = re.search(pattern, source)
    
    if match:
        print("✅ _analyze_successful_configurations method exists")
        return True
    else:
        print("❌ _analyze_successful_configurations method NOT found")
        return False


# ============================================================================
# Backward compatibility tests
# ============================================================================

def test_legacy_methods_exist():
    """Test that legacy methods are preserved for backward compatibility."""
    print("\n=== Test: Legacy methods preserved ===")
    
    source = get_meta_controller_source()
    
    legacy_methods = [
        '_validate_and_coerce_policy',
        '_validate_and_coerce_plan',
        '_run_exploration_phase',
        '_run_fast_loop',
        '_default_safe_policy',
        '_default_safe_plan',
        '_try_salvage_policy',
        '_try_salvage_plan',
        '_run_default_penalty_policy',
    ]
    
    all_found = True
    for method in legacy_methods:
        pattern = rf'def {method}\s*\('
        if not re.search(pattern, source):
            print(f"❌ Legacy method NOT found: {method}")
            all_found = False
    
    if all_found:
        print("✅ All legacy methods are preserved")
        return True
    return False


def test_default_optimization_policy_exists():
    """Test that DEFAULT_OPTIMIZATION_POLICY is preserved for backward compatibility."""
    print("\n=== Test: DEFAULT_OPTIMIZATION_POLICY preserved ===")
    
    source = get_meta_controller_source()
    
    # Check that DEFAULT_OPTIMIZATION_POLICY is defined
    pattern = r'DEFAULT_OPTIMIZATION_POLICY\s*=\s*\{'
    match = re.search(pattern, source)
    
    if match:
        print("✅ DEFAULT_OPTIMIZATION_POLICY is preserved")
        return True
    else:
        print("❌ DEFAULT_OPTIMIZATION_POLICY is NOT preserved")
        return False


def test_default_mission_plan_alias_exists():
    """Test that DEFAULT_MISSION_PLAN alias exists for backward compatibility."""
    print("\n=== Test: DEFAULT_MISSION_PLAN alias exists ===")
    
    source = get_meta_controller_source()
    
    # Check that DEFAULT_MISSION_PLAN is defined
    pattern = r'DEFAULT_MISSION_PLAN\s*=\s*\{'
    match = re.search(pattern, source)
    
    if match:
        print("✅ DEFAULT_MISSION_PLAN alias exists")
        return True
    else:
        print("❌ DEFAULT_MISSION_PLAN alias does NOT exist")
        return False


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DIRECT CONFIGURATION OPTIMIZATION TESTS")
    print("=" * 60)
    
    all_passed = True
    
    # Meta-controller tests
    print("\n--- META-CONTROLLER TESTS ---")
    all_passed &= test_no_exploration_agent_import()
    all_passed &= test_no_kernel_tuning_env_import()
    all_passed &= test_default_configurations_defined()
    all_passed &= test_default_objective_weights_defined()
    all_passed &= test_validate_and_coerce_configurations_exists()
    all_passed &= test_validate_single_config_exists()
    all_passed &= test_run_direct_testing_phase_exists()
    all_passed &= test_default_safe_configurations_exists()
    all_passed &= test_run_default_configurations_exists()
    all_passed &= test_configs_per_iteration_defined()
    
    # Hierarchical optimizer tests
    print("\n--- HIERARCHICAL OPTIMIZER TESTS ---")
    all_passed &= test_prompt_asks_for_configurations()
    all_passed &= test_prompt_example_format()
    all_passed &= test_prompt_includes_reasoning_field()
    
    # Feedback collector tests
    print("\n--- FEEDBACK COLLECTOR TESTS ---")
    all_passed &= test_record_configuration_results_exists()
    all_passed &= test_analyze_successful_configurations_exists()
    
    # Backward compatibility tests
    print("\n--- BACKWARD COMPATIBILITY TESTS ---")
    all_passed &= test_legacy_methods_exist()
    all_passed &= test_default_optimization_policy_exists()
    all_passed &= test_default_mission_plan_alias_exists()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL DIRECT CONFIGURATION OPTIMIZATION TESTS PASSED")
    else:
        print("❌ ONE OR MORE TESTS FAILED")
    print("=" * 60)

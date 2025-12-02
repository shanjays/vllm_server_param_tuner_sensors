import json
import ray
import numpy as np
import time
import os
import re
import ast
import itertools
import random
from profiling_worker import ProfilingWorker
from config_exporter import TOKEN_COUNTS_ALL

POWER_OF_TWO_WARPS = (2, 4, 8, 16, 32)

# Default configurations for direct LLM-based testing
# These are specific kernel configurations (not search spaces)
DEFAULT_CONFIGURATIONS = [
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 4},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "num_warps": 16, "num_stages": 3},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 5},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4},
]

# Default objective weights for reward computation
DEFAULT_OBJECTIVE_WEIGHTS = {
    "R_sm_throughput": 0.4,
    "R_dram_throughput": 0.3,
    "R_l1_hit_rate": 0.15,
    "R_l2_hit_rate": 0.15
}

# Legacy support: DEFAULT_OPTIMIZATION_POLICY for backward compatibility
DEFAULT_OPTIMIZATION_POLICY = {
    "objective_weights": DEFAULT_OBJECTIVE_WEIGHTS,
    "search_space": {
        "BLOCK_SIZE_M": [32, 64, 128],
        "BLOCK_SIZE_N": [32, 64, 128],
        "BLOCK_SIZE_K": [32, 64],
        "num_warps": [4, 8, 16],
        "num_stages": [2, 3, 4]
    }
}

# Backward compatibility alias
DEFAULT_MISSION_PLAN = {
    "reward_function": DEFAULT_OPTIMIZATION_POLICY["objective_weights"],
    "pruned_action_space": DEFAULT_OPTIMIZATION_POLICY["search_space"]
}

# Token counts for training (key representative values for 5-hour run)
TOKEN_COUNTS_TRAINING = [1, 16, 64, 256, 1024, 4096, 16384]

# Default token counts to test (matching vLLM's expected format)
DEFAULT_TOKEN_COUNTS = TOKEN_COUNTS_TRAINING

# Number of configurations to test per iteration
CONFIGS_PER_ITERATION = 5

# Validation frequency: run vLLM validation after every N token counts
VALIDATE_EVERY_N_TOKENS = 3

class MetaControllerReward:
    """
    Meta-Controller Reward Function for direct LLM-based kernel optimization.
    
    This class orchestrates the optimization process by:
    1. Parsing specific kernel configurations from LLM outputs
    2. Testing each configuration directly via the ProfilingWorker
    3. Validating configurations through throughput benchmarks
    4. Computing rewards to guide the GRPO meta-learning process
    
    The LLM directly generates specific configurations to test, rather than
    generating a search space for a separate exploration agent.
    """
    def __init__(self, user_goal, model_name, exploration_steps, profiling_gpu_id, static_args, 
                 config_exporter=None, token_counts=None, training_logger=None, feedback_collector=None):
        """
        Initialize the meta-controller reward function.
        
        Args:
            user_goal: Optimization target ('throughput' or 'latency')
            model_name: Target model name for benchmarking
            exploration_steps: Number of configurations to test per iteration (legacy name kept for compatibility)
            profiling_gpu_id: GPU ID for the profiling worker
            static_args: Static benchmark arguments
            config_exporter: Optional VLLMConfigExporter for saving configs
            token_counts: List of token counts to test
            training_logger: Optional HierarchicalTrainingLogger for TensorBoard logging
            feedback_collector: Optional FeedbackCollector for contextual learning feedback
        """
        self.user_goal = user_goal
        self.model_name = model_name
        self.exploration_steps = exploration_steps
        self.static_args = static_args
        self.config_exporter = config_exporter
        self.token_counts = token_counts or DEFAULT_TOKEN_COUNTS
        self.training_logger = training_logger
        self.feedback_collector = feedback_collector
        self.num_experts = static_args.get('num_experts', 128)
        self.inter_size = static_args.get('inter_size', 1536)
        
        # Set up vLLM config directory
        try:
            import vllm
            vllm_lib_path = os.path.dirname(vllm.__file__)
            self.vllm_config_dir = os.path.join(
                vllm_lib_path, "model_executor/layers/fused_moe/configs/"
            )
            if not os.path.exists(self.vllm_config_dir):
                self.vllm_config_dir = "/tmp/vllm_configs/"
            os.makedirs(self.vllm_config_dir, exist_ok=True)
        except ImportError:
            print("[MetaController] vLLM not installed, using fallback config directory")
            self.vllm_config_dir = "/tmp/vllm_configs/"
            os.makedirs(self.vllm_config_dir, exist_ok=True)
        except OSError as e:
            print(f"[MetaController] Could not create vLLM config directory: {e}")
            self.vllm_config_dir = "/tmp/vllm_configs/"
            os.makedirs(self.vllm_config_dir, exist_ok=True)
        
        print(f"[MetaController] Requesting ProfilingWorker for PHYSICAL GPU {profiling_gpu_id}")
        self.worker = ProfilingWorker.options(num_gpus=1).remote(profiling_gpu_id)
        self.initial_state = self._get_initial_state()
        print(f"[MetaController] Configured to test {len(self.token_counts)} token counts: {self.token_counts[:5]}...")

    def _clean_non_json_types(self, data):
        if isinstance(data, dict):
            return {k: self._clean_non_json_types(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._clean_non_json_types(i) for i in data]
        if isinstance(data, set):
            return list(self._clean_non_json_types(list(data)))
        if data is Ellipsis:
            return "..."
        return data

    def _get_initial_state(self):
        print("[MetaController] Getting initial state from worker...")
        try:
            job_id = self.worker.run_kernel_profiling.remote(None, self.static_args, {})
            state, reward, _ = ray.get(job_id)
            if state is None:
                raise RuntimeError("Worker failed initial profile.")
            print("[MetaController] Initial state acquired.")
            return state
        except Exception as e:
            print(f"[MetaController] ERROR: Worker failed initial state check. Using fallback. {e}")
            return np.array([32.3, 40.8, 0.05, 69.9], dtype=np.float32)

    def __call__(self, completions, **kwargs):
        rewards = []
        for i, config_str in enumerate(completions):
            print(f"\n--- [MetaController] Processing LLM Configuration {i+1}/{len(completions)} ---")
            
            # Debug output
            print(f"[MetaController] DEBUG: Raw LLM Output:\n{config_str}\n")

            valid = True
            configurations = None
            best_configs = {}
            try:
                parsed_output = self._extract_json(config_str)
                parsed_output = self._clean_non_json_types(parsed_output)
                configurations = self._validate_and_coerce_configurations(parsed_output)

                print(f"[MetaController] Testing {len(configurations)} configurations directly...")
                top_configs, best_configs = self._run_direct_testing_phase(configurations)
                print(f"[MetaController] Starting throughput validation (Top {len(top_configs)} configs)...")
                final_metric = self._run_throughput_validation(top_configs)
                rewards.append(final_metric)
                
                # Record configuration results to feedback collector
                if self.feedback_collector and configurations:
                    self.feedback_collector.record_configuration_results(
                        configurations=configurations,
                        reward=final_metric,
                        best_configs=best_configs
                    )
                
            except Exception as e:
                valid = False
                print(f"[MetaController] ERROR: Reward calculation failed. Reason: {e}")
                rewards.append(0.0)
                if not valid:
                    self._run_default_configurations(i)
        
        # Print training summary
        if self.config_exporter:
            summary = self.config_exporter.get_summary()
            print(f"\n[MetaController] === Training Summary ===")
            print(f"  Total configs tested: {summary['total_experiments']}")
            print(f"  Token counts covered: {summary['total_token_counts']}")
            print(f"  Best rewards by token count:")
            for tc, reward in sorted(summary['best_rewards'].items(), key=lambda x: int(x[0])):
                print(f"    {tc} tokens: {reward:.2f}")
        
        return rewards

    def _run_default_configurations(self, idx):
        """Run testing on default configurations when LLM output fails to parse."""
        print("[MetaController] LLM output failed. Testing default configurations to ensure training progression.")
        try:
            self._run_direct_testing_phase(DEFAULT_CONFIGURATIONS)
        except Exception as e:
            print(f"[MetaController] WARNING: Default configuration testing also failed. {e}")

    def _run_default_penalty_policy(self, idx):
        """
        Legacy method for backward compatibility.
        
        This method delegates to _run_default_configurations() which is the new 
        implementation for the direct configuration testing architecture.
        """
        self._run_default_configurations(idx)

    def _normalize_unicode(self, s: str) -> str:
        replacements = {
            '\u2011': '-', '\u202f': ' ', '\u2248': '~',
            '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
            '\u00A0': ' ', '\u2013': '-', '\u2014': '-', '\u2212': '-',
            '\uFF0C': ',', '\u200B': ''  # zero-width space
        }
        for bad, good in replacements.items():
            s = s.replace(bad, good)
        return s

    def _preclean_reward_arrays(self, text: str) -> str:
        """
        Collapse patterns like "R_sm_throughput": [0.2464, "Weight for SM Utilization"]
        into "R_sm_throughput": 0.2464 before JSON parsing.
        """
        pattern = re.compile(
            r'("R_(?:sm_throughput|dram_throughput|l1_hit_rate|l2_hit_rate)"\s*:\s*)\[\s*([0-9eE+.\-]+)\s*,\s*"[^"\]]*"\s*\]'
        )
        while True:
            new_text, count = pattern.subn(r'\1\2', text)
            if count == 0:
                break
            text = new_text
        return text

    def _strip_unterminated_quotes(self, s: str) -> str:
        # If odd number of double quotes, try to remove trailing partial segment
        if s.count('"') % 2 != 0:
            # Remove everything after the last complete pair boundary
            last_brace = max(s.rfind('}'), s.rfind(']'))
            if last_brace != -1:
                s = s[:last_brace+1]
        return s

    def _clean_number_string(self, s: str) -> str:
        """
        Clean a number string by removing trailing periods, handling scientific notation,
        and other common LLM formatting issues.
        """
        s = s.strip()
        # Remove trailing periods (e.g., "42.75." -> "42.75")
        s = re.sub(r'\.+$', '', s)
        # Remove leading/trailing whitespace again after cleanup
        s = s.strip()
        # Handle case where LLM outputs just 'e' or partial scientific notation
        if s.lower() in ('e', 'e+', 'e-', '+e', '-e', ''):
            return '0.0'
        # Fix malformed scientific notation like "1.5e" -> "1.5"
        s = re.sub(r'[eE][+-]?$', '', s)
        return s

    def _extract_json(self, llm_output_str):
        llm_output_str = self._normalize_unicode(llm_output_str)

        # Pre-clean reward arrays w/ annotation before locating braces
        llm_output_str = self._preclean_reward_arrays(llm_output_str)

        # Check for verbose reasoning patterns (LLM thinking out loud)
        verbose_patterns = [
            r'^We are',
            r'^I am',
            r'^Let me',
            r'^Let\'s',
            r'^First,',
            r'^Looking at',
            r'^Based on',
            r'^Analyzing',
            r'^The data shows',
            r'^We need to',
            r'^We have',
        ]
        output_start = llm_output_str.strip()[:50].lower()
        is_verbose = any(re.match(pattern, output_start, re.IGNORECASE) for pattern in verbose_patterns)
        
        if is_verbose and '<param>' not in llm_output_str.lower():
            print("[MetaController] WARNING: LLM output contains verbose reasoning without <param> tags, using default policy")
            return self._default_safe_policy()

        # Check if output appears truncated (no <param> tags found and ends mid-sentence)
        if '<param>' not in llm_output_str.lower():
            stripped = llm_output_str.rstrip()
            if stripped and not stripped.endswith(('.', '>', '}', ']', '"')):
                print("[MetaController] WARNING: LLM output appears truncated (no <param> tags, ends mid-sentence), using default policy")
                return self._default_safe_policy()

        # Try to extract content from <param></param> XML tags first (preferred format)
        # Use non-greedy match for the JSON content and handle any garbage before <param>
        param_match = re.search(r'<param>\s*(\{[\s\S]*?\})\s*</param>', llm_output_str, re.DOTALL | re.IGNORECASE)
        if param_match:
            json_str = param_match.group(1).strip()
            print("[MetaController] Found JSON content within <param> tags.")
        else:
            # Fallback: Try to find any content between <param> tags 
            param_any_match = re.search(r'<param>\s*(.*?)\s*</param>', llm_output_str, re.DOTALL | re.IGNORECASE)
            if param_any_match:
                content = param_any_match.group(1).strip()
                # Check if it starts with { (JSON object)
                if content.startswith('{'):
                    json_str = content
                    print("[MetaController] Found content within <param> tags.")
                else:
                    # Content inside tags but not JSON-like
                    print(f"[MetaController] WARNING: Content in <param> tags is not JSON-like, using fallback")
                    json_str = None
            else:
                json_str = None
            
            if json_str is None:
                # Fallback: Try to find JSON-like content after <param> (for truncated output)
                param_start_match = re.search(r'<param>\s*(\{.*)', llm_output_str, re.DOTALL | re.IGNORECASE)
                if param_start_match:
                    json_str = param_start_match.group(1).strip()
                    print("[MetaController] Found truncated content after <param> tag, attempting recovery.")
                    # Try to fix truncated JSON
                    recovered_json = self._try_recover_json(json_str)
                    if recovered_json is not None:
                        return recovered_json
                
                # Fallback to brace matching
                match = re.search(r'(\{.*\})', llm_output_str, re.DOTALL)
                if not match:
                    # Try to recover from truncated output without closing brace
                    match_partial = re.search(r'(\{.*)', llm_output_str, re.DOTALL)
                    if match_partial:
                        json_str = match_partial.group(1).strip()
                        print("[MetaController] Found partial JSON, attempting recovery.")
                        recovered_json = self._try_recover_json(json_str)
                        if recovered_json is not None:
                            return recovered_json
                    
                    salvage = self._try_salvage_policy(llm_output_str)
                    if salvage is not None:
                        return salvage
                    print("[MetaController] No braces found; using default-safe policy for this completion.")
                    return self._default_safe_policy()
                json_str = match.group(0).strip()

        json_str = json_str.replace('```json', '').replace('```', '').strip()

        # Replace any leftover schema-like arrays (second pass)
        json_str = self._preclean_reward_arrays(json_str)

        # Remove annotation arrays like ["float", "Weight ..."] generically
        json_str = re.sub(r'\[\s*([0-9eE+.\-]+)\s*,\s*"[^"]*"\s*\]', r'\1', json_str)

        # Remove control characters
        json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)

        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

        # Clean number values with trailing periods or malformed scientific notation
        # Pattern explanation:
        #   [0-9]+       - one or more digits (integer part)
        #   \.?          - optional decimal point
        #   [0-9]*       - zero or more digits (fractional part)
        #   [eE]?        - optional exponent indicator
        #   [+-]?        - optional sign for exponent
        #   [0-9]*       - exponent digits
        #   \.+          - one or more trailing periods (the malformed part we're fixing)
        #   (?=\s*[,\]\}]) - lookahead for JSON delimiter (comma, bracket, or brace)
        def clean_json_numbers(match):
            return self._clean_number_string(match.group(0))
        json_str = re.sub(r'[0-9]+\.?[0-9]*[eE]?[+-]?[0-9]*\.+(?=\s*[,\]\}])', clean_json_numbers, json_str)

        # Force closure if braces unbalanced
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
            print(f"DEBUG: Added {open_braces - close_braces} closing brace(s).")

        json_str = self._strip_unterminated_quotes(json_str)

        # Attempt strict JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Attempt Python literal
        try:
            return ast.literal_eval(json_str)
        except Exception:
            # Final normalization: replace stray tokens
            normalized = re.sub(r'\bfloat\b', '0.5', json_str)
            normalized = re.sub(r'\bint\b', '64', json_str)
            try:
                return json.loads(normalized)
            except Exception:
                try:
                    return ast.literal_eval(normalized)
                except Exception:
                    return self._default_safe_policy()

    def _try_recover_json(self, json_str):
        """
        Try to recover a valid dict from a potentially truncated JSON string.
        Returns a dict if successful, None otherwise.
        """
        recovered_json = self._fix_truncated_json(json_str)
        if recovered_json is not None:
            if isinstance(recovered_json, dict):
                return recovered_json
            else:
                print(f"[MetaController] WARNING: Recovered JSON is {type(recovered_json)}, expected dict")
        return None

    def _fix_truncated_json(self, json_str):
        """Attempt to fix truncated JSON by adding missing brackets."""
        # Count brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')

        # Strip trailing whitespace
        json_str = json_str.rstrip()
        
        # Remove trailing comma if present
        if json_str.endswith(','):
            json_str = json_str[:-1]

        # Add missing array closures
        for _ in range(open_brackets - close_brackets):
            json_str += ']'

        # Add missing object closures
        for _ in range(open_braces - close_braces):
            json_str += '}'

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try Python literal eval
            try:
                return ast.literal_eval(json_str)
            except Exception:
                return None

    def _default_safe_policy(self):
        """Return a safe default set of configurations for direct testing."""
        return {
            "configurations": [config.copy() for config in DEFAULT_CONFIGURATIONS]
        }

    def _default_safe_configurations(self):
        """Return a safe default list of configurations."""
        return [config.copy() for config in DEFAULT_CONFIGURATIONS]

    def _default_safe_plan(self):
        """Legacy method for backward compatibility."""
        return self._default_safe_policy()

    def _try_salvage_policy(self, s: str):
        """Try to salvage partial configuration data from malformed output."""
        # Try to extract individual configurations from the output
        config_keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages")
        
        # Check if we can find any configuration-like structures
        configs = []
        
        # Try to find complete configurations
        for k in config_keys:
            pattern = rf'{k}\s*[:=]\s*([0-9]+)'
            matches = re.findall(pattern, s)
            if matches:
                # Found some values, try to build a config
                break
        
        # Try to extract one configuration from partial data
        config = {}
        for k in config_keys:
            m = re.search(rf'{k}\s*[:=]\s*([0-9]+)', s)
            if m:
                try:
                    config[k] = int(m.group(1))
                except ValueError:
                    pass
        
        if config:
            # Fill in defaults for missing values
            defaults = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 4}
            for k in config_keys:
                if k not in config:
                    config[k] = defaults[k]
            configs.append(config)
        
        if configs:
            return {"configurations": configs}
        return None

    def _try_salvage_plan(self, s: str):
        """Legacy method for backward compatibility."""
        return self._try_salvage_policy(s)

    def _validate_single_config(self, config):
        """
        Validate and coerce a single kernel configuration.
        
        Args:
            config: Dict with kernel parameters
            
        Returns:
            Validated config dict or None if invalid
        """
        if not isinstance(config, dict):
            return None
        
        # Required parameters with defaults
        defaults = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64, 
            "BLOCK_SIZE_K": 32,
            "num_warps": 8,
            "num_stages": 4
        }
        
        # H100 hardware constraints
        H100_BLOCK_SIZE_MN_LIMIT = 128
        H100_BLOCK_SIZE_K_LIMIT = 64
        H100_NUM_STAGES_LIMIT = 5
        
        validated = {}
        
        # Extract and validate each parameter
        for key, default_val in defaults.items():
            val = config.get(key, default_val)
            try:
                val = int(val)
            except (ValueError, TypeError):
                val = default_val
            validated[key] = val
        
        # Apply hardware limits
        validated["BLOCK_SIZE_M"] = min(max(16, validated["BLOCK_SIZE_M"]), H100_BLOCK_SIZE_MN_LIMIT)
        validated["BLOCK_SIZE_N"] = min(max(32, validated["BLOCK_SIZE_N"]), H100_BLOCK_SIZE_MN_LIMIT)
        validated["BLOCK_SIZE_K"] = min(max(32, validated["BLOCK_SIZE_K"]), H100_BLOCK_SIZE_K_LIMIT)
        validated["num_stages"] = min(max(2, validated["num_stages"]), H100_NUM_STAGES_LIMIT)
        
        # Enforce power-of-two warps
        num_warps = validated["num_warps"]
        if num_warps not in POWER_OF_TWO_WARPS:
            # Find nearest valid value
            valid_warps = [w for w in POWER_OF_TWO_WARPS if w <= 32]
            validated["num_warps"] = min(valid_warps, key=lambda x: abs(x - num_warps))
        
        # Optional: Include reasoning if present
        if "reasoning" in config and isinstance(config["reasoning"], str):
            validated["reasoning"] = config["reasoning"]
        
        return validated

    def _validate_and_coerce_configurations(self, parsed_output):
        """
        Validate and coerce LLM output to a list of kernel configurations.
        
        Args:
            parsed_output: Dict from LLM containing configurations
            
        Returns:
            List of validated configuration dicts
        """
        if parsed_output is None or not isinstance(parsed_output, dict):
            print("[MetaController] WARNING: Invalid output, using default configurations")
            return self._default_safe_configurations()
        
        # Extract configurations list from parsed output
        configs_raw = parsed_output.get("configurations", [])
        
        # Handle legacy format (objective_weights + search_space) for backward compatibility
        if not configs_raw and ("objective_weights" in parsed_output or "search_space" in parsed_output):
            print("[MetaController] Legacy format detected, generating configs from search space")
            return self._generate_configs_from_search_space(parsed_output)
        
        if not isinstance(configs_raw, list) or not configs_raw:
            print("[MetaController] WARNING: No configurations found, using defaults")
            return self._default_safe_configurations()
        
        # Validate each configuration
        validated_configs = []
        for i, config in enumerate(configs_raw):
            validated = self._validate_single_config(config)
            if validated:
                validated_configs.append(validated)
                print(f"[MetaController] Config {i+1}: M={validated['BLOCK_SIZE_M']}, N={validated['BLOCK_SIZE_N']}, "
                      f"K={validated['BLOCK_SIZE_K']}, warps={validated['num_warps']}, stages={validated['num_stages']}")
        
        if not validated_configs:
            print("[MetaController] WARNING: No valid configurations after validation, using defaults")
            return self._default_safe_configurations()
        
        # Limit to reasonable number of configs per iteration
        max_configs = max(CONFIGS_PER_ITERATION, self.exploration_steps)
        if len(validated_configs) > max_configs:
            print(f"[MetaController] Limiting to {max_configs} configurations")
            validated_configs = validated_configs[:max_configs]
        
        print(f"[MetaController] Validated {len(validated_configs)} configurations")
        return validated_configs

    def _generate_configs_from_search_space(self, policy):
        """
        Generate configurations from legacy search space format.
        
        Args:
            policy: Dict with objective_weights and search_space
            
        Returns:
            List of configuration dicts
        """
        search_space = policy.get("search_space", {})
        if not search_space:
            return self._default_safe_configurations()
        
        # Get parameter lists
        m_vals = search_space.get("BLOCK_SIZE_M", [64])
        n_vals = search_space.get("BLOCK_SIZE_N", [64])
        k_vals = search_space.get("BLOCK_SIZE_K", [32])
        warp_vals = search_space.get("num_warps", [8])
        stage_vals = search_space.get("num_stages", [4])
        
        # Ensure lists
        if not isinstance(m_vals, list): m_vals = [m_vals]
        if not isinstance(n_vals, list): n_vals = [n_vals]
        if not isinstance(k_vals, list): k_vals = [k_vals]
        if not isinstance(warp_vals, list): warp_vals = [warp_vals]
        if not isinstance(stage_vals, list): stage_vals = [stage_vals]
        
        # Generate configs (sample if too many combinations)
        configs = []
        all_combos = list(itertools.product(m_vals, n_vals, k_vals, warp_vals, stage_vals))
        
        # Limit to max_configs
        max_configs = max(CONFIGS_PER_ITERATION, self.exploration_steps)
        if len(all_combos) > max_configs:
            all_combos = random.sample(all_combos, max_configs)
        
        for m, n, k, w, s in all_combos:
            config = {
                "BLOCK_SIZE_M": int(m),
                "BLOCK_SIZE_N": int(n),
                "BLOCK_SIZE_K": int(k),
                "num_warps": int(w),
                "num_stages": int(s)
            }
            validated = self._validate_single_config(config)
            if validated:
                configs.append(validated)
        
        return configs if configs else self._default_safe_configurations()

    def _validate_and_coerce_policy(self, policy):
        """Legacy method for backward compatibility - delegates to _validate_and_coerce_configurations."""
        return self._validate_and_coerce_configurations(policy)

    def _validate_and_coerce_plan(self, plan):
        """Legacy method for backward compatibility."""
        return self._validate_and_coerce_configurations(plan)

    def _run_direct_testing_phase(self, configurations):
        """
        Directly test each configuration via the ProfilingWorker.
        
        This replaces the PPO-based exploration with direct configuration testing.
        The LLM suggests specific configs, and we test each one directly.
        
        Args:
            configurations: List of kernel configuration dicts to test
            
        Returns:
            Tuple of (top_configs, best_configs) where:
            - top_configs: List of (config, state, reward) tuples for validation
            - best_configs: Dict mapping token_count -> {config, reward}
        """
        all_results = []
        best_configs = {}
        total_tokens = len(self.token_counts)
        total_configs = len(configurations)
        
        for i, token_count in enumerate(self.token_counts):
            print(f"\n[MetaController] === Token Count {i+1}/{total_tokens}: {token_count} tokens ===")
            
            token_results = []
            for j, config in enumerate(configurations):
                print(f"[MetaController] Testing config {j+1}/{total_configs}: "
                      f"M={config['BLOCK_SIZE_M']}, N={config['BLOCK_SIZE_N']}, "
                      f"K={config['BLOCK_SIZE_K']}, warps={config['num_warps']}, stages={config['num_stages']}")
                
                try:
                    # Run profiling directly
                    result_id = self.worker.run_kernel_profiling.remote(
                        config, self.static_args, DEFAULT_OBJECTIVE_WEIGHTS, token_count
                    )
                    state, reward, _ = ray.get(result_id)
                    
                    if state is not None:
                        result = (config, state, reward)
                        token_results.append(result)
                        all_results.append(result)
                        
                        # Update config exporter
                        if self.config_exporter:
                            metrics = {
                                'sm_throughput': state[0],
                                'dram_throughput': state[1],
                                'l1_hit_rate': state[2],
                                'l2_hit_rate': state[3]
                            }
                            self.config_exporter.update_best_config(
                                token_count, config, reward, metrics
                            )
                        
                        print(f"[MetaController] Result: reward={reward:.2f}, "
                              f"SM={state[0]:.1f}%, DRAM={state[1]:.1f}%")
                    else:
                        print(f"[MetaController] Config failed (reward={reward:.2f})")
                        
                except Exception as e:
                    print(f"[MetaController] ERROR testing config: {e}")
                    continue
            
            # Track best config for this token count
            if token_results:
                best_result = max(token_results, key=lambda x: x[2])
                best_configs[token_count] = {
                    'config': best_result[0],
                    'reward': best_result[2],
                }
                print(f"[MetaController] Best for {token_count} tokens: reward={best_result[2]:.2f}")
            
            # Periodic validation
            if (i + 1) % VALIDATE_EVERY_N_TOKENS == 0 and all_results:
                print(f"[MetaController] Running periodic validation after {i+1} token counts...")
                best_overall = max(all_results, key=lambda x: x[2])
                throughput = self._run_throughput_validation([best_overall])
                print(f"[MetaController] Periodic validation throughput: {throughput} tokens/sec")
        
        # Return top results sorted by reward
        if all_results:
            sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
            print(f"[MetaController] Direct testing complete. {len(all_results)} results across {len(self.token_counts)} token counts.")
            return sorted_results[:5], best_configs
        
        return [], best_configs

    def _run_exploration_phase(self, policy_config_path):
        """
        Legacy method for backward compatibility.
        Converts policy file to configurations and runs direct testing.
        """
        try:
            with open(policy_config_path, 'r') as f:
                policy = json.load(f)
            configurations = self._validate_and_coerce_configurations(policy)
        except Exception as e:
            print(f"[MetaController] WARNING: Could not load policy, using defaults: {e}")
            configurations = self._default_safe_configurations()
        
        return self._run_direct_testing_phase(configurations)

    def _run_fast_loop(self, mission_plan_path):
        """Legacy method for backward compatibility."""
        top_configs, _ = self._run_exploration_phase(mission_plan_path)
        return top_configs

    def _run_throughput_validation(self, top_configs):
        """Run throughput validation on the best configurations."""
        if not top_configs:
            print("[MetaController] No valid configurations found for throughput validation.")
            return 0.0
        
        # Print summary of all configs to be tested before validation
        print(f"\n[MetaController] === Throughput Validation Summary ===")
        print(f"[MetaController] Testing {len(top_configs)} configurations:")
        for i, (params, state, reward) in enumerate(top_configs):
            print(f"  Config {i+1}: M={params.get('BLOCK_SIZE_M', 'N/A')}, "
                  f"N={params.get('BLOCK_SIZE_N', 'N/A')}, K={params.get('BLOCK_SIZE_K', 'N/A')}, "
                  f"warps={params.get('num_warps', 'N/A')}, stages={params.get('num_stages', 'N/A')}, "
                  f"reward={reward:.2f}")
        
        ids = [
            self.worker.run_throughput_validation.remote(params, self.model_name, self.user_goal)
            for params, state, reward in top_configs
        ]
        print(f"[MetaController] Awaiting validation metrics for {len(top_configs)} configs from ProfilingWorker...")
        metrics = ray.get(ids)
        if self.user_goal == "throughput":
            best = max(metrics)
        else:
            valid = [m for m in metrics if m > 0]
            best = min(valid) if valid else 0.0
        print(f"[MetaController] Throughput validation complete. Best metric: {best}")
        return best

    def _run_slow_gym_validation(self, best_configs_by_token):
        """
        Run vLLM benchmark with a COMBINED config file containing all token counts.
        
        Args:
            best_configs_by_token: Dict mapping token_count -> {"config": {...}, "reward": float}
            
        Returns:
            float: Throughput metric from vLLM benchmark
        """
        if not best_configs_by_token:
            print("[MetaController] No configs to validate.")
            return 0.0
        
        # Print summary of all best configs by token count
        print(f"\n[MetaController] === Combined Config Validation Summary ===")
        print(f"[MetaController] Best configs for {len(best_configs_by_token)} token counts:")
        for token_count in sorted(best_configs_by_token.keys(), key=lambda x: int(x) if isinstance(x, str) else x):
            config_data = best_configs_by_token[token_count]
            cfg = config_data.get("config", {})
            reward = config_data.get("reward", 0)
            print(f"  {token_count} tokens: M={cfg.get('BLOCK_SIZE_M', 'N/A')}, "
                  f"N={cfg.get('BLOCK_SIZE_N', 'N/A')}, K={cfg.get('BLOCK_SIZE_K', 'N/A')}, "
                  f"warps={cfg.get('num_warps', 'N/A')}, stages={cfg.get('num_stages', 'N/A')}, "
                  f"reward={reward:.2f}")
        
        # Build combined config with ALL token counts
        combined_config = {}
        for token_count, config_data in best_configs_by_token.items():
            combined_config[str(token_count)] = config_data["config"]
        
        # Write the combined config ONCE
        config_path = os.path.join(
            self.vllm_config_dir,
            f"E={self.num_experts},N={self.inter_size},device_name=NVIDIA_H100_80GB_HBM3.json"
        )
        
        with open(config_path, "w") as f:
            json.dump(combined_config, f, indent=2)
        
        print(f"[MetaController] Wrote combined config with {len(combined_config)} token counts to: {config_path}")
        print(f"[MetaController] Config file contents preview:")
        config_preview = json.dumps(combined_config, indent=2)
        print(config_preview[:1000] if len(config_preview) > 1000 else config_preview)
        if len(config_preview) > 1000:
            print(f"  ... (truncated, {len(config_preview) - 1000} more chars)")
        
        # Now run vLLM benchmark ONCE (it will use the right config per token count internally)
        result = self.worker.run_throughput_validation.remote(
            combined_config,
            self.model_name,
            self.user_goal
        )
        
        throughput = ray.get(result)
        print(f"[MetaController] Combined config validation throughput: {throughput} tokens/sec")
        return throughput

    def _run_slow_gym(self, top_configs):
        """Legacy method for backward compatibility."""
        return self._run_throughput_validation(top_configs)


# Backward compatibility alias
HAKT_Reward_Function = MetaControllerReward

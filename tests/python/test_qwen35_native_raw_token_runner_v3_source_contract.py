import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "crates"
    / "apxinf-model"
    / "examples"
    / "qwen35_native_raw_token_runner_v3.rs"
)
CARGO = ROOT / "crates" / "apxinf-model" / "Cargo.toml"
CONTRACT = ROOT / "configs" / "qwen35-0.8b-cross-runtime-formal-v3.json"
CUSTODY = (
    ROOT
    / "crates"
    / "apxinf-model"
    / "examples"
    / "support"
    / "qwen35_boundary_tail_head_v1_gate_evidence.rs"
)


def rust_u32_array(source: str, name: str) -> list[int]:
    match = re.search(
        rf"const {name}: \[u32; \d+\] = \[(.*?)\];", source, re.DOTALL
    )
    if match is None:
        raise AssertionError(f"missing Rust array {name}")
    return [int(value) for value in re.findall(r"\d+", match.group(1))]


def compact_sha256(value: object) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class Qwen35NativeRawTokenRunnerV3SourceContractTests(unittest.TestCase):
    def test_dedicated_example_exposes_the_three_semantic_v3_lanes(self):
        source = RUNNER.read_text(encoding="utf-8")
        cargo = CARGO.read_text(encoding="utf-8")

        self.assertIn('name = "qwen35_native_raw_token_runner_v3"', cargo)
        self.assertIn('required-features = ["accelerate", "metal-w8"]', cargo)
        self.assertIn("enum RunMode", source)
        self.assertIn("enum TeacherRole", source)
        self.assertIn("NativeV3Free", source)
        self.assertIn("NativeV3Teacher", source)
        self.assertIn("TeacherRole::Reference", source)
        self.assertIn("TeacherRole::Observed", source)

    def test_formal_cli_maps_free_and_two_teacher_roles_without_legacy_defaults(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn('"native-v3-free"', source)
        self.assertIn('"native-v3-teacher"', source)
        self.assertIn('"reference"', source)
        self.assertIn('"observed"', source)
        self.assertIn('"APXINF_FORMAL_V3_REQUEST_JSON"', source)
        self.assertIn("teacher role is required for native-v3-teacher", source)
        self.assertIn("teacher role is invalid for native-v3-free", source)
        self.assertNotIn("RunMode::Legacy", source)

    def test_frozen_raw13_free128_and_prompt12_teacher_inputs_are_exact(self):
        source = RUNNER.read_text(encoding="utf-8")
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        native = contract["workload_contracts"]["NATIVE_RAW13_FREE128_V3"]
        prompt = contract["workload_contracts"]["shared_prompt"]["token_ids"]
        teacher = native["teacher_forced_admission"]

        prompt_ids = rust_u32_array(source, "PROMPT_TOKEN_IDS")
        free_ids = rust_u32_array(source, "CANONICAL_FREE_TOKEN_IDS")
        teacher_ids = rust_u32_array(source, "TEACHER_INPUT_TOKEN_IDS")
        self.assertEqual(prompt_ids, prompt)
        self.assertEqual(len(free_ids), 128)
        self.assertEqual(
            compact_sha256(free_ids),
            native["free_run_trajectory_admission"]["expected_sha256"],
        )
        self.assertEqual(teacher_ids, teacher["teacher_input_token_ids"])
        self.assertEqual(teacher_ids, [prompt_ids[-1], *free_ids[:127]])
        self.assertIn("const MAX_CONTEXT: usize = 256;", source)
        self.assertIn("const TEACHER_PREFILL_TOKENS: usize = 12;", source)

    def test_custody_closure_covers_full_attention_and_all_17_build_inputs(self):
        source = CUSTODY.read_text(encoding="utf-8")

        self.assertIn("FULL_ATTENTION_DECODE_V1_RUST_SOURCE_BYTES", source)
        self.assertIn("METAL_FULL_ATTENTION_DECODE_V1_BRIDGE_SOURCE_BYTES", source)
        self.assertIn("METAL_FULL_ATTENTION_DECODE_V1_SOURCE_BYTES", source)
        self.assertIn("fn boundary_tail_head_v1_source_set_specs() -> [BuildSourceSpec; 57]", source)
        self.assertIn("const EXPECTED_NAMES: [&str; 57]", source)
        self.assertIn("declared_build_inputs.len() != 17", source)

    def test_cpu_teacher_reference_prefills_raw12_then_runs_128_fixed_inputs(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("GeneralQwen35::from_weights(", source)
        self.assertRegex(
            source,
            r"LlmInput::text\(\s*&PROMPT_TOKEN_IDS\[\.\.TEACHER_PREFILL_TOKENS\],?\s*\)",
        )
        self.assertIn("TEACHER_INPUT_TOKEN_IDS.iter().enumerate()", source)
        self.assertIn("model.forward(&[teacher_token], position)", source)
        self.assertIn("reference_argmax_token_ids", source)
        self.assertIn("CANONICAL_FREE_TOKEN_IDS", source)
        self.assertIn("CPU/F32 teacher reference diverged from frozen free128", source)

    def test_an_teacher_uses_fused_c_top4_wait_rerank_path_for_all_steps(self):
        source = RUNNER.read_text(encoding="utf-8")
        teacher_body = source[
            source.index("fn run_an_teacher_observed") : source.index("struct Timespec")
        ]

        self.assertIn(
            "from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1",
            source,
        )
        self.assertIn("teacher_forced_decode_candidates(teacher_token, position)", source)
        self.assertIn("comparison.cpu_token", source)
        self.assertIn("comparison.w8_candidates", source)
        self.assertIn("comparison.reranked_token", source)
        self.assertIn("comparison.accelerator_candidate_elapsed_ns()", source)
        self.assertIn("comparison.rerank_elapsed_ns", source)
        self.assertIn("next_greedy_token_ready_elapsed_ns", source)
        self.assertIn("validate_fused_path_receipts", source)
        self.assertIn("TailPhase::Teacher", source)
        self.assertLess(
            teacher_body.index("let mut next_greedy_token_ready_elapsed_ns"),
            teacher_body.index("let started = Instant::now();"),
        )
        self.assertLess(
            teacher_body.index("let token_ready_elapsed_ns ="),
            teacher_body.index("cpu_f32_argmax_token_ids.push"),
        )

    def test_an_free_times_every_next_greedy_ready_token_without_eos_stop(self):
        source = RUNNER.read_text(encoding="utf-8")
        free_body = source[source.index("fn run_an_free") : source.index("fn is_lower_hex")]

        self.assertIn("fn run_an_free", source)
        self.assertIn("let prefill_start_ns = monotonic_raw_ns()?;", source)
        self.assertIn("prefill_for_generation(LlmInput::text(&PROMPT_TOKEN_IDS))", source)
        self.assertIn("let first_token = argmax(&prefill_logits, vocab_size)?;", source)
        self.assertIn("for step in 1..STEPS", source)
        self.assertRegex(
            source, r"model\s*\.decode_token\(previous_token, position\)"
        )
        self.assertIn("token_ready_ns.push(first_token_ready_ns);", source)
        self.assertIn("token_ready_ns.push(token_ready);", source)
        self.assertIn("generated_token_ids != CANONICAL_FREE_TOKEN_IDS", source)
        self.assertIn("STEPS - 1, TailPhase::Free", source)
        self.assertIn('"selection_work_included": true', source)
        self.assertIn(
            '"accelerator_completion_before_each_token_ready_timestamp": true', source
        )
        self.assertIn('"final_sampled_token_decoded_inside_timed_region": false', source)
        self.assertLess(
            free_body.index("let mut token_ready_ns = Vec::with_capacity(STEPS);"),
            free_body.index("let prefill_start_ns = monotonic_raw_ns()?;"),
        )
        self.assertLess(
            free_body.index("let mut generated_token_ids = Vec::with_capacity(STEPS);"),
            free_body.index("let prefill_start_ns = monotonic_raw_ns()?;"),
        )
        self.assertLess(
            free_body.index("let first_token_ready_ns = monotonic_raw_ns()?;"),
            free_body.index("generated_token_ids.push(first_token);"),
        )
        self.assertLess(
            free_body.index("let token_ready = monotonic_raw_ns()?;"),
            free_body.index("generated_token_ids.push(previous_token);"),
        )
        self.assertNotIn("is_eog", free_body)
        self.assertNotIn("break;", free_body)

    def test_entrypoint_attests_custody_echoes_request_and_emits_one_json_line(self):
        source = RUNNER.read_text(encoding="utf-8")
        real_main = source[source.index("fn real_main") : source.index("fn parse_args_from")]

        self.assertIn("GateCustody::capture_boundary_tail_head_v1", source)
        self.assertIn("custody.receipt_json()", source)
        self.assertIn("custody.verify_unchanged_receipt()", source)
        self.assertIn('std::env::var("APXINF_FORMAL_V3_REQUEST_JSON")', source)
        self.assertIn('receipt["request"] = request;', source)
        self.assertIn("let formal_request =", real_main)
        self.assertIn("fn formal_request_schedule_is_valid", source)
        self.assertIn("formal v3 request schedule tuple is invalid", source)
        self.assertIn("6 + block_index * 4 + slot_index", source)
        self.assertLess(
            real_main.index("parse_formal_request()?"),
            real_main.index("GateCustody::capture_boundary_tail_head_v1"),
            "the formal nonce/schedule must fail closed before custody or model dispatch",
        )
        self.assertIn(
            "let library_closure_start = loaded_non_system_library_closure()?;",
            real_main,
        )
        self.assertIn(
            "let library_closure_end = loaded_non_system_library_closure()?;",
            real_main,
        )
        self.assertIn("if library_closure_start != library_closure_end", real_main)
        self.assertIn('"loaded_non_system_library_closure_start"', real_main)
        self.assertIn('"loaded_non_system_library_closure_end"', real_main)
        self.assertIn('"loaded_non_system_library_closure_start_sha256"', real_main)
        self.assertIn('"loaded_non_system_library_closure_end_sha256"', real_main)
        self.assertLess(
            real_main.index("let library_closure_start"),
            real_main.index("let (config, tensors) = load_model_inputs"),
        )
        self.assertGreater(
            real_main.index("let library_closure_end"),
            real_main.index("run_an_free(config, tensors)?"),
        )
        self.assertIn('receipt["custody"] = custody_receipt;', source)
        self.assertIn("packed_weight_and_resident_buffer_manifest_sha256", source)
        self.assertIn("loaded_non_system_library_closure_sha256", source)
        self.assertIn("runtime_source_commit", source)
        self.assertIn("APXINF_CANDIDATE_COMMIT", source)
        self.assertIn('"fresh_process": true', source)
        self.assertIn('"start_end_identity_equal": true', source)
        self.assertIn('"exact_live_execution_ledger": true', source)
        self.assertIn("serde_json::to_vec(&receipt)", source)
        self.assertIn("line.push(b'\\n');", source)
        self.assertIn("stdout.write_all(&line)", source)
        self.assertIn("apxinf-qwen35-native-runner-failure-v3", source)
        self.assertNotIn("println!", source)

    def test_an_thread_policy_is_live_disclosed_and_overrides_fail_closed(self):
        source = RUNNER.read_text(encoding="utf-8")
        real_main = source[source.index("fn real_main") : source.index("fn parse_args_from")]

        self.assertIn('"policy": "Accelerate OS-managed default"', source)
        self.assertIn('"fixed_worker_count_claimed": false', source)
        for variable in (
            "VECLIB_MAXIMUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            self.assertIn(f'"{variable}"', source)
        self.assertIn("fn capture_thread_policy_runtime", source)
        self.assertIn("std::thread::available_parallelism()?.get()", source)
        self.assertIn('"logical_cpu_count_source": "std::thread::available_parallelism"', source)
        self.assertIn('"environment_overrides_absent": true', source)
        self.assertIn('"absent_environment_overrides": THREAD_OVERRIDE_ENVIRONMENT', source)
        self.assertLess(
            real_main.index("capture_thread_policy_runtime()?"),
            real_main.index("GateCustody::capture_boundary_tail_head_v1"),
        )

    def test_live_path_admission_validates_exact_fused_profile_and_resident_ledger(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("fn production_receipt_is_exact", source)
        self.assertIn("fn generation_receipt_is_exact", source)
        self.assertIn("persistent_output_groups_per_row == 64", source)
        self.assertIn("core_kernel_output_groups_per_row == 32", source)
        self.assertIn("stats.tail_layer_index == 23", source)
        self.assertIn("stats.initial_stack.mechanism == INITIAL_MECHANISM", source)
        self.assertIn("region.mechanism == BOUNDARY_MECHANISM", source)
        self.assertIn('get("standalone_layer23_mlp")', source)
        self.assertIn('get("standalone_metal_lm_head")', source)
        self.assertIn('get("prefill_body_calls")', source)
        self.assertIn("pipeline_thread_execution_width == 32", source)
        self.assertIn("source_declared_threadgroup_memory_bytes == 2_060", source)
        self.assertIn("pipeline_static_threadgroup_memory_bytes == 2_064", source)
        self.assertIn("internal_threadgroup_barrier_sites_per_threadgroup == 4", source)
        self.assertIn("fn initial_ledger_is_exact", source)
        self.assertIn("fn boundary_ledger_is_exact", source)
        self.assertIn("fn tail_ledger_is_exact", source)
        self.assertIn("component_sum_recomputed_and_exact", source)
        for exact_value in ("799_543_312", "494", "443", "51", "28_672", "28_688"):
            self.assertIn(exact_value, source)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs" / "qwen35-0.8b-llamacpp-comparison-v1.json"
MODULE_PATH = ROOT / "scripts" / "validate_qwen35_llamacpp_comparison_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_qwen35_llamacpp_comparison_contract_for_tests", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


class Qwen35LlamaCppComparisonContractTests(unittest.TestCase):
    def test_offline_validator_accepts_the_checked_in_contract(self):
        module = load_module()
        contract = module.load_contract(CONTRACT_PATH)

        self.assertEqual(
            contract["format"], "apxinf-qwen35-llamacpp-comparison-contract-v1"
        )
        completed = subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                "-B",
                str(MODULE_PATH),
                "--contract",
                str(CONTRACT_PATH),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["contract_sha256"], contract["content_sha256"])
        self.assertEqual(
            receipt["formal_llama_cpp_binary_sha256"],
            "ccfa5ecd78119d4f8cdd8721e7faae360cb94b8334f9d61ed47e2e00290f2716",
        )
        self.assertTrue(receipt["formal_build_source_custody_eligible"])
        self.assertFalse(receipt["formal_campaign_eligible"])

    def test_validator_rejects_rehashed_semantic_drift(self):
        module = load_module()
        contract = module.load_contract(CONTRACT_PATH)
        mutations = (
            (
                "schedule",
                lambda value: value["formal_protocol"]["schedule"].__setitem__(
                    "timed_samples_total", 23
                ),
            ),
            (
                "quantization",
                lambda value: value["comparison_tiers"]["eight_bit_storage"].__setitem__(
                    "quantization_mechanisms_equal", True
                ),
            ),
            (
                "local build",
                lambda value: value["llama_cpp"]["local_observed_build"].__setitem__(
                    "formal_eligible", True
                ),
            ),
            (
                "formal observed build",
                lambda value: value["llama_cpp"]["formal_observed_build"].__setitem__(
                    "source_custody_eligible", False
                ),
            ),
            (
                "effective context claim",
                lambda value: value["workload"]["generation"][
                    "context_contract"
                ].__setitem__("effective_context_equality_claim_allowed", True),
            ),
            (
                "placement proof timing",
                lambda value: value["llama_cpp"]["formal_observed_build"][
                    "execution_placement_proof"
                ]["phase_order"].__setitem__("timing_excluded", False),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = json.loads(module.canonical_bytes(contract))
                mutate(changed)
                changed.pop("content_sha256")
                changed["content_sha256"] = module.object_sha256(changed)
                with self.assertRaises(module.ComparisonContractError):
                    module.validate_contract(changed)

    def test_validator_rejects_duplicate_json_keys(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"format":"one","format":"two"}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                module.ComparisonContractError, "duplicate key"
            ):
                module.load_contract(path)

    def test_contract_freezes_the_exact_hugging_face_and_gguf_artifacts(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        source = contract["source_model"]
        self.assertEqual(source["repo_id"], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(
            source["revision"], "2fc06364715b967f1860aea9cf38778875588b17"
        )
        self.assertEqual(
            source["checkpoint"],
            {
                "name": "model.safetensors-00001-of-00001.safetensors",
                "sha256": (
                    "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
                ),
                "size": 1_746_942_600,
            },
        )

        artifacts = contract["llama_cpp"]["model_artifacts"]
        self.assertEqual(
            artifacts["f32"],
            {
                "name": "Qwen3.5-0.8B-2fc063647-F32.gguf",
                "sha256": (
                    "69ad6b3ef11f0fb4d9af2d9f59a235c8576d9ef2e64b4375274ca35fc34530e4"
                ),
                "size": 3_020_533_248,
            },
        )
        self.assertEqual(
            artifacts["pure_q8_0"],
            {
                "name": "Qwen3.5-0.8B-2fc063647-Q8_0-llama.gguf",
                "parent_f32_sha256": artifacts["f32"]["sha256"],
                "quantization": "llama.cpp-pure-Q8_0",
                "sha256": (
                    "427555e797eefeb62e1c8ef71510ce062027b0c7f1674cc4cfcb352710e0908c"
                ),
                "size": 811_843_072,
            },
        )

    def test_contract_uses_one_raw_token_stream_and_fixed_generation_semantics(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        prompt = contract["workload"]["prompt"]
        self.assertEqual(
            prompt["raw_token_ids"],
            [
                248045,
                846,
                198,
                9419,
                248046,
                198,
                248045,
                74455,
                198,
                248068,
                271,
                248069,
                271,
            ],
        )
        self.assertEqual(prompt["token_count"], 13)
        self.assertEqual(
            prompt["token_ids_sha256"],
            "4b890fa15ee3d7db4e9dd18bd79c6362d40e9e016ae4f9f74cb7fc420ef3b6d3",
        )
        self.assertEqual(prompt["ingress"], "raw-token-ids-only")

        generation = contract["workload"]["generation"]
        self.assertEqual(generation["max_new_tokens"], 128)
        self.assertEqual(generation["context_length"], 142)
        self.assertEqual(generation["sampling"], "greedy-argmax")
        self.assertEqual(generation["stop_policy"], "fixed-128-ignore-eos")
        self.assertEqual(generation["process_state"], "fresh-process-per-sample")
        self.assertEqual(
            generation["context_arithmetic"],
            {"prompt_tokens": 13, "generated_tokens": 128, "spare_tokens": 1},
        )

    def test_ctx142_is_the_same_request_but_not_an_effective_context_equality_claim(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        context = contract["workload"]["generation"]["context_contract"]
        self.assertTrue(context["same_requested_context"])
        self.assertEqual(context["requested_context_length_each"], 142)
        self.assertEqual(context["pinned_llama_cpp_effective_context_length"], 256)
        self.assertEqual(
            context["pinned_llama_cpp_behavior"],
            "implementation-rounds-request-142-up-to-256",
        )
        self.assertFalse(context["effective_context_equality_claim_allowed"])
        self.assertEqual(
            context["required_report_disclosure"],
            (
                "both engines receive request 142; pinned llama.cpp Qwen3.5 "
                "reports effective 256 due to implementation rounding"
            ),
        )
        self.assertIn(
            "identical-effective-context-allocation",
            contract["claims"]["forbidden_claims"],
        )

    def test_contract_names_the_apx_lane_and_forbids_false_quantization_equivalence(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        lane = contract["apxinf"]["metal_w8_lane"]
        self.assertEqual(
            lane["constructor"],
            "GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1",
        )
        self.assertEqual(
            lane["mechanism"], "metal-w8-mlp-stack3-boundary-tail-head-v1"
        )
        self.assertEqual(lane["initial_stack_layers"], [0, 1, 2])
        self.assertEqual(
            lane["boundary_regions"],
            [
                {"full_attention_mlp_layer": 3, "linear_stack_layers": [4, 5, 6]},
                {"full_attention_mlp_layer": 7, "linear_stack_layers": [8, 9, 10]},
                {
                    "full_attention_mlp_layer": 11,
                    "linear_stack_layers": [12, 13, 14],
                },
                {
                    "full_attention_mlp_layer": 15,
                    "linear_stack_layers": [16, 17, 18],
                },
                {
                    "full_attention_mlp_layer": 19,
                    "linear_stack_layers": [20, 21, 22],
                },
            ],
        )
        self.assertEqual(lane["tail_layer"], 23)
        self.assertEqual(
            lane["quantization"]["scheme"],
            "symmetric-signed-int8-per-row-group-round-clamp-minus127-plus127-f32-scale",
        )
        self.assertEqual(
            lane["quantization"]["linear_projection_group_sizes"],
            {
                "gdn_input": 64,
                "gdn_output": 32,
                "mlp_gate": 64,
                "mlp_up": 64,
                "mlp_down": 64,
            },
        )
        self.assertEqual(
            lane["precision_scope"],
            "metal-w8-owned-projections-plus-cpu-f32-unowned-attention-kv-and-exact-top4-rerank",
        )

        eight_bit = contract["comparison_tiers"]["eight_bit_storage"]
        self.assertEqual(eight_bit["apxinf_artifact"], "runtime-packed-from-hf-f32")
        self.assertEqual(eight_bit["llama_cpp_artifact"], "pure_q8_0")
        self.assertFalse(eight_bit["quantization_mechanisms_equal"])
        self.assertFalse(eight_bit["weight_regimes_equal"])
        self.assertIn(
            "identical-quantization", contract["claims"]["forbidden_claims"]
        )

    def test_formal_protocol_is_balanced_and_fails_closed_on_host_contamination(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        formal = contract["formal_protocol"]
        schedule = formal["schedule"]
        self.assertEqual(
            schedule["block_orders"],
            ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA", "BAAB"],
        )
        self.assertEqual(schedule["untimed_warmups_per_implementation"], 3)
        self.assertEqual(schedule["timed_samples_total"], 24)
        self.assertEqual(schedule["timed_samples_per_implementation"], 12)
        self.assertEqual(
            sum(len(block) for block in schedule["block_orders"]),
            schedule["timed_samples_total"],
        )
        self.assertEqual(
            "".join(schedule["block_orders"]).count("A"),
            schedule["timed_samples_per_implementation"],
        )
        self.assertEqual(
            "".join(schedule["block_orders"]).count("B"),
            schedule["timed_samples_per_implementation"],
        )

        resources = formal["resource_gates"]
        self.assertEqual(resources["process_group_rss_comparison"], "strictly-less-than")
        self.assertEqual(resources["process_group_rss_limit_bytes"], 6 * 1024**3)
        self.assertEqual(resources["child_swaps_required"], 0)
        self.assertEqual(resources["system_swap_delta_bytes_required"], 0)
        self.assertEqual(resources["memory_pressure_pages_throttled_required"], 0)

        quiet = formal["quiet_host_gate"]
        self.assertEqual(quiet["preflight_sample_count"], 5)
        self.assertEqual(quiet["maximum_non_allowlisted_process_cpu_percent"], 5.0)
        self.assertEqual(quiet["maximum_load_per_logical_cpu"], 0.5)
        self.assertTrue(quiet["must_remain_valid_during_every_timed_sample"])
        self.assertEqual(formal["failure_policy"], "fail-closed-no-partial-formal-claim")

    def test_unknown_local_llama_build_is_pinned_but_never_formal_evidence(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        local = contract["llama_cpp"]["local_observed_build"]
        self.assertIsNone(local["source_commit"])
        self.assertEqual(local["version_output"], "version: 0 (unknown)")
        self.assertEqual(local["source_tree_git_state"], "no-commits-yet")
        self.assertFalse(local["formal_eligible"])
        self.assertEqual(local["classification"], "diagnostic-only")
        self.assertEqual(
            local["artifacts"]["llama_cli"]["sha256"],
            "963368c84febc0a739448d357097727f6b6c39d69f5648962a8ae5e5a27a0426",
        )
        self.assertEqual(
            local["artifacts"]["llama_bench"]["sha256"],
            "fee0c1d3a83a964029a9792e20cfb1882a818a8db15204dcc18fdf7642509243",
        )
        self.assertEqual(
            local["artifacts"]["libllama"]["sha256"],
            "73847830fa12c80e34108415841a9f656701b2ee49bf469efdd83a7f564cee81",
        )
        self.assertEqual(
            local["artifacts"]["libggml_metal"]["sha256"],
            "46ba0662937c145cb5c6d3c1b7a9ca173fd9dce2af006c260436666392c674f4",
        )

        derivation = contract["llama_cpp"]["model_derivation"]
        self.assertEqual(
            derivation["pure_q8_0"]["parent_f32_sha256"],
            contract["llama_cpp"]["model_artifacts"]["f32"]["sha256"],
        )
        self.assertTrue(derivation["pure_q8_0"]["pure"])
        self.assertEqual(derivation["pure_q8_0"]["quantization_type"], "Q8_0")

    def test_formal_llama_source_uses_the_clean_official_commit_pin(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        source = contract["llama_cpp"]["formal_source"]
        self.assertEqual(source["repository"], "https://github.com/ggml-org/llama.cpp.git")
        self.assertEqual(
            source["commit"], "f280b26983ad0fdb705a0d9ebf0503e76f2899b0"
        )
        self.assertEqual(source["tree"], "21045aed8b426d7a5e25a98e646054cbd9487e81")
        self.assertTrue(source["clean_detached_checkout_required"])
        self.assertEqual(
            source["build_admission"],
            "formal-only-after-new-executable-and-loaded-library-closure-hashes-are-captured",
        )

    def test_formal_llama_build_pins_the_static_runner_and_source_custody(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        build = contract["llama_cpp"]["formal_observed_build"]
        self.assertEqual(
            build["source"],
            {
                "commit": "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
                "tree": "21045aed8b426d7a5e25a98e646054cbd9487e81",
                "clean_detached_checkout": True,
            },
        )
        self.assertEqual(
            build["inputs"]["runner_source"],
            {
                "name": "benchmarks/llama_cpp/raw_token_runner.cpp",
                "sha256": (
                    "76a5a354f729d22659387557ef368b75e83910e28a09d52876ddb366106c66e4"
                ),
                "size": 41_807,
            },
        )
        self.assertEqual(
            build["inputs"]["cmake_lists"],
            {
                "name": "benchmarks/llama_cpp/CMakeLists.txt",
                "sha256": (
                    "50c8bd83995b73f239dc4e3e4573127952e35d360f3584823a9529626904201d"
                ),
                "size": 5_757,
            },
        )
        self.assertEqual(
            build["binary"],
            {
                "name": "apxinf-llama-cpp-raw-token-runner",
                "build_type": "Release",
                "linkage": "static-llama-ggml-with-system-dynamic-libraries-only",
                "sha256": (
                    "ccfa5ecd78119d4f8cdd8721e7faae360cb94b8334f9d61ed47e2e00290f2716"
                ),
                "size": 6_499_056,
            },
        )
        self.assertEqual(
            build["cmake_options"],
            {
                "BUILD_SHARED_LIBS": False,
                "GGML_BACKEND_DL": False,
                "GGML_STATIC": False,
                "GGML_LTO": False,
                "GGML_CCACHE": False,
                "GGML_METAL": True,
                "GGML_METAL_EMBED_LIBRARY": True,
                "GGML_ACCELERATE": True,
                "GGML_CPU": True,
                "GGML_CPU_REPACK": True,
                "GGML_BLAS": True,
                "GGML_BLAS_VENDOR": "Apple",
                "GGML_LLAMAFILE": True,
                "GGML_NATIVE": True,
                "GGML_OPENMP": False,
                "GGML_METAL_NDEBUG": False,
                "GGML_METAL_SHADER_DEBUG": False,
                "GGML_CUDA": False,
                "GGML_MUSA": False,
                "GGML_HIP": False,
                "GGML_VULKAN": False,
                "GGML_WEBGPU": False,
                "GGML_ZDNN": False,
                "GGML_VIRTGPU": False,
                "GGML_VIRTGPU_BACKEND": False,
                "GGML_RPC": False,
                "GGML_SYCL": False,
                "GGML_OPENVINO": False,
                "GGML_ET": False,
                "GGML_OPENCL": False,
                "GGML_HEXAGON": False,
                "GGML_ZENDNN": False,
            },
        )
        self.assertEqual(
            build["otool_runtime_closure"],
            {
                "non_system_dependencies": [],
                "classification": "otool-reports-system-frameworks-and-libraries-only",
                "dynamic_loader_symbols_present": ["dlopen", "dlsym"],
                "dynamic_loader_symbols_absent_claim_allowed": False,
                "claim_scope": (
                    "runner-policy-disables-backend-load-all-and-default-scan; "
                    "not-a-claim-that-dlopen-or-dlsym-symbols-are-absent"
                ),
            },
        )
        self.assertTrue(build["source_custody_eligible"])
        self.assertFalse(build["formal_campaign_eligible"])
        self.assertEqual(
            build["formal_campaign_blockers"],
            [
                "apxinf-thread-policy-parity-not-established",
                "quiet-host-gate-not-passed",
            ],
        )

    def test_v2_runner_backend_and_placement_policy_is_explicit_and_fail_closed(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        build = contract["llama_cpp"]["formal_observed_build"]
        self.assertEqual(
            build["runner_schema"], "apxinf.llama-cpp.raw-token-diagnostic.v2"
        )
        policy = build["backend_policy"]
        self.assertEqual(policy["registration_mode"], "linked-static-registry-only")
        self.assertFalse(policy["ggml_backend_load_all_called"])
        self.assertFalse(policy["default_backend_scan_invoked"])
        self.assertEqual(policy["backend_directory_option_policy"], "rejected")
        self.assertEqual(
            policy["ggml_backend_path_policy"], "must-be-absent-or-run-fails"
        )

        gpu = policy["gpu_lane"]
        self.assertEqual(
            gpu["device_selection"],
            "explicit-name-must-resolve-to-exactly-one-registered-gpu",
        )
        self.assertEqual(gpu["model_selected_device_count"], 1)
        self.assertEqual(gpu["transformer_layer_count"], 24)
        self.assertEqual(gpu["layers_on_selected_gpu"], 24)
        self.assertTrue(gpu["output_device_pointer_equals_selected_gpu"])
        self.assertEqual(
            gpu["gpu_memory_bytes_required"],
            {
                "model": "strictly-positive",
                "context": "strictly-positive",
                "compute": "strictly-positive",
            },
        )

        cpu = policy["cpu_lane"]
        self.assertEqual(cpu["model_selected_device_count"], 0)
        self.assertEqual(cpu["transformer_layer_count"], 24)
        self.assertEqual(cpu["layers_on_cpu"], 24)
        self.assertTrue(cpu["output_on_cpu"])
        self.assertEqual(
            cpu["gpu_memory_bytes_required"],
            {"model": 0, "context": 0, "compute": 0},
        )
        self.assertEqual(policy["kv_cache"], {"cpu_lane": "f32", "gpu_lane": "f16"})

        q8 = policy["q8_0_metal_observed_placement"]
        self.assertEqual(q8["input_embedding_buffer_type"], "CPU")
        self.assertIsNone(q8["input_embedding_device_pointer"])
        self.assertEqual(q8["cpu_model_buffer_bytes"], 270_172_160)
        self.assertFalse(q8["pure_all_device_memory"])
        self.assertEqual(
            q8["required_report_disclosure"],
            (
                "Q8_0 Metal keeps the input embedding in a CPU fallback buffer; "
                "it is not pure all-device memory"
            ),
        )
        for forbidden in (
            "pure-all-device-q8_0-metal",
            "binary-has-no-dlopen-or-dlsym-symbols",
        ):
            self.assertIn(forbidden, contract["claims"]["forbidden_claims"])

    def test_v2_post_measurement_execution_proof_completes_all_26_sentinels(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        proof = contract["llama_cpp"]["formal_observed_build"][
            "execution_placement_proof"
        ]

        self.assertEqual(
            proof["method"], "scheduler-callback-completed-sentinels-v1"
        )
        self.assertEqual(
            proof["internal_api_binding"],
            {
                "llama_cpp_commit": "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
                "experimental_internal_api": True,
                "headers": ["llama-context.h", "llama-ext.h", "llama-model.h"],
                "source_upgrade_policy": "rebuild-and-reaudit-required",
            },
        )
        phase = proof["phase_order"]
        self.assertTrue(phase["same_context_as_measured_generation"])
        self.assertEqual(phase["proof_token"], "sampled-token-128")
        self.assertTrue(phase["token_timing_complete_before_proof"])
        self.assertTrue(phase["perf_counters_captured_before_proof"])
        self.assertEqual(phase["proof_decode_count"], 1)
        self.assertTrue(phase["timing_excluded"])
        self.assertEqual(
            phase["excluded_from"],
            [
                "token_ready_elapsed_ns",
                "generation_elapsed_ns",
                "measurement_scope_elapsed_ns",
                "llama_perf",
            ],
        )
        self.assertEqual(
            phase["separately_recorded_as"],
            "post_measurement_execution_proof_elapsed_ns",
        )

        callback = proof["scheduler_callback"]
        self.assertEqual(callback["requested_sentinel_count_ask_true"], 26)
        self.assertEqual(callback["completion_callback_ask_value"], False)
        self.assertEqual(callback["completed_sentinel_count_ask_false"], 26)
        self.assertEqual(callback["input_sentinel"], "model.input_embed")
        self.assertEqual(callback["layer_sentinel_pattern"], "l_out-0..23")
        self.assertEqual(callback["layer_sentinel_count"], 24)
        self.assertEqual(callback["output_sentinel"], "result_output")

        self.assertEqual(
            proof["gpu_lane_completion"],
            {
                "model.input_embed": "CPU",
                "l_out-0..23": "MTL0",
                "result_output": "MTL0",
                "completed_on_cpu": 1,
                "completed_on_selected_gpu": 25,
            },
        )
        self.assertEqual(
            proof["cpu_lane_completion"],
            {"all_26_sentinels": "CPU", "completed_on_cpu": 26},
        )
        self.assertEqual(
            proof["fail_closed_on"],
            [
                "proof-decode-error",
                "missing-sentinel",
                "duplicate-or-unexpected-callback",
                "wrong-backend",
            ],
        )
        receipt = proof["receipt_contract"]
        self.assertTrue(receipt["recorded"])
        self.assertTrue(receipt["passed"])
        self.assertTrue(receipt["timing_excluded"])
        self.assertEqual(receipt["requested_sentinel_count"], 26)
        self.assertEqual(receipt["completed_sentinel_count"], 26)
        self.assertFalse(receipt["backend_mismatch"])
        self.assertFalse(receipt["duplicate_or_unexpected_callback"])

    def test_quality_protocol_separates_exactness_from_performance(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

        quality = contract["quality_protocol"]
        self.assertTrue(quality["must_complete_before_formal_timing"])
        self.assertEqual(quality["teacher_forced"]["steps"], 128)
        self.assertEqual(quality["free_run"]["steps"], 128)
        self.assertEqual(
            quality["teacher_forced"]["exactness_definition"],
            "zero-argmax-mismatches-over-128-teacher-inputs",
        )
        self.assertEqual(
            quality["free_run"]["exactness_definition"],
            "identical-128-token-generated-trajectory",
        )
        self.assertEqual(
            quality["classification"]["exact"], "exact-trajectory"
        )
        self.assertEqual(
            quality["classification"]["divergent"],
            "divergent-trajectory-with-first-mismatch-prefix-and-position-ratio",
        )
        self.assertEqual(
            quality["apxinf_metal_w8_admission"],
            "exact-teacher128-and-free128-versus-same-process-apxinf-cpu-f32-oracle",
        )
        self.assertFalse(quality["cross_runtime_exactness_required_to_measure_speed"])
        self.assertTrue(quality["speed_claim_must_not_imply_quality_parity"])


if __name__ == "__main__":
    unittest.main()

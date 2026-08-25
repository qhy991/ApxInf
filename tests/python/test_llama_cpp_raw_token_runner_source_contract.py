import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks" / "llama_cpp_v3" / "raw_token_runner.cpp"
CMAKE = ROOT / "benchmarks" / "llama_cpp_v3" / "CMakeLists.txt"
LEGACY_RUNNER = ROOT / "benchmarks" / "llama_cpp" / "raw_token_runner.cpp"
LEGACY_CMAKE = ROOT / "benchmarks" / "llama_cpp" / "CMakeLists.txt"


class LlamaCppRawTokenRunnerSourceContractTests(unittest.TestCase):
    def test_v3_is_isolated_from_hash_bound_legacy_diagnostic_sources(self):
        self.assertNotEqual(RUNNER, LEGACY_RUNNER)
        self.assertEqual(
            hashlib.sha256(LEGACY_RUNNER.read_bytes()).hexdigest(),
            "76a5a354f729d22659387557ef368b75e83910e28a09d52876ddb366106c66e4",
        )
        self.assertEqual(
            hashlib.sha256(LEGACY_CMAKE.read_bytes()).hexdigest(),
            "50c8bd83995b73f239dc4e3e4573127952e35d360f3584823a9529626904201d",
        )

    def test_static_runner_never_scans_for_dynamic_backends(self):
        source = RUNNER.read_text(encoding="utf-8")
        cmake = CMAKE.read_text(encoding="utf-8")

        self.assertNotIn("ggml_backend_load_all(", source)
        self.assertNotIn("ggml_backend_load_all_from_path(", source)
        self.assertIn('reject_environment_variable("GGML_BACKEND_PATH")', source)
        self.assertIn("llama_backend_init();", source)
        self.assertIn("set(BUILD_SHARED_LIBS OFF", cmake)
        self.assertIn("set(GGML_BACKEND_DL OFF", cmake)

    def test_cpu_lane_uses_f32_kv_and_gpu_lane_is_attested(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn('#include "llama-ext.h"', source)
        self.assertIn('#include "llama-model.h"', source)
        self.assertIn("model_params.devices = selected_devices.data();", source)
        self.assertIn("model->dev_layer", source)
        self.assertIn("model->dev_output", source)
        self.assertIn("llama_get_memory_breakdown", source)
        self.assertIn("ggml_backend_sched_set_eval_callback", source)
        self.assertIn("post_measurement_execution_proof", source)
        self.assertIn("context->get_sched()", source)
        self.assertIn(
            "context_params.type_k = gpu_lane ? GGML_TYPE_F16 : GGML_TYPE_F32;",
            source,
        )
        self.assertIn(
            "context_params.type_v = gpu_lane ? GGML_TYPE_F16 : GGML_TYPE_F32;",
            source,
        )

    def test_receipt_rejects_invalid_numbers_tokens_and_stdout_failures(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("std::isfinite", source)
        self.assertIn("sampled token is outside the model vocabulary", source)
        self.assertIn("std::ios::badbit | std::ios::failbit", source)

    def test_versioned_v3_modes_keep_the_legacy_default_unchanged(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("enum class RunMode", source)
        self.assertIn("LegacyFreeV2", source)
        self.assertIn("NativeV3Free", source)
        self.assertIn("NativeV3Teacher", source)
        self.assertIn("RunMode mode = RunMode::LegacyFreeV2;", source)
        self.assertIn('argument == "--mode"', source)
        self.assertIn('text == "native-v3-free"', source)
        self.assertIn('text == "native-v3-teacher"', source)
        self.assertIn("--mode may be specified at most once", source)
        self.assertIn("constexpr uint32_t kLegacyContextSize = 142;", source)
        self.assertIn("constexpr uint32_t kNativeV3ContextSize = 256;", source)
        self.assertIn('"apxinf.llama-cpp.raw-token-diagnostic.v2"', source)
        self.assertIn('"apxinf.llama-cpp.raw-token-diagnostic.v3"', source)

    def test_native_v3_teacher_prefills_raw12_then_decodes_128_teacher_tokens(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            "kTeacherPrefillTokenCount = kPromptTokens.size() - 1", source
        )
        self.assertIn("make_teacher_input_tokens", source)
        self.assertIn("teacher_inputs[0] = kPromptTokens.back();", source)
        self.assertIn(
            "teacher_inputs[index + 1] = kCanonicalFreeRunTokens[index];",
            source,
        )
        self.assertIn("mutable_prompt.data(), kTeacherPrefillTokenCount", source)
        self.assertIn("teacher prefill must evaluate exactly the first 12 raw prompt tokens", source)
        self.assertIn("teacher_inputs[static_cast<size_t>(index)]", source)
        self.assertIn("teacher prefill unexpectedly included prompt token 13", source)
        self.assertNotIn("kPromptTokens.back(), 1); // teacher prefill", source)

    def test_teacher_receipt_records_full_vocab_argmax_mismatches_and_step_timing(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("full_vocabulary_argmax", source)
        self.assertIn("llama_get_logits_ith(context, -1)", source)
        self.assertIn("token < vocabulary_size", source)
        self.assertIn("full-vocabulary argmax encountered a non-finite logit", source)
        self.assertIn('"teacher_input_token_ids":', source)
        self.assertIn('"reference_argmax_token_ids":', source)
        self.assertIn('"observed_argmax_token_ids":', source)
        self.assertIn('"mismatch_positions":', source)
        self.assertIn('"first_mismatch":', source)
        self.assertIn('"next_greedy_token_ready_elapsed_ns":', source)
        self.assertIn('"argmax_timing_included":true', source)
        self.assertIn('"eog_termination":false', source)
        self.assertIn("teacher-forced performance counters violate", source)

    def test_v3_receipt_attests_the_monotonic_clock_and_resolution(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("using Clock = std::chrono::steady_clock;", source)
        self.assertIn("static_assert(Clock::is_steady", source)
        self.assertIn("static_assert(kClockResolutionNs > 0", source)
        self.assertIn('"clock_identity\\\":\\\"std::chrono::steady_clock', source)
        self.assertIn('"clock_is_steady\\\":', source)
        self.assertIn('"clock_resolution_ns\\\":', source)
        self.assertIn('"clock_period_numerator\\\":', source)
        self.assertIn('"clock_period_denominator\\\":', source)
        self.assertIn("monotonic_time_since_epoch_ns(generation_start)", source)
        self.assertIn('"generation_start_ns\\\":', source)
        self.assertIn('"token_ready_boundary\\\":\\\"next-greedy-token-ready', source)
        self.assertIn('"selection_work_included\\\":true', source)
        self.assertIn(
            '"accelerator_completion_before_each_token_ready_timestamp\\\":',
            source,
        )

    def test_v3_free_uses_strict_full_vocab_argmax_without_changing_legacy_sampler(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            "native_v3 ? full_vocabulary_argmax(context.get(), vocabulary_size)",
            source,
        )
        self.assertIn(
            ": llama_sampler_sample(sampler.get(), context.get(), -1);", source
        )
        self.assertIn("(native_v3 ? 0 : kGeneratedTokenCount)", source)
        self.assertIn("full-vocabulary argmax encountered a non-finite logit", source)


if __name__ == "__main__":
    unittest.main()

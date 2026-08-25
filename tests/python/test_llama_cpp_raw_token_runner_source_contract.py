from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks" / "llama_cpp" / "raw_token_runner.cpp"
CMAKE = ROOT / "benchmarks" / "llama_cpp" / "CMakeLists.txt"


class LlamaCppRawTokenRunnerSourceContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

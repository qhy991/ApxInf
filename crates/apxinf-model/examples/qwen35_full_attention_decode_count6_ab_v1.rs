//! Predeclared fixed-shape count-6 Qwen3.5 full-attention decode A/B gate.
//!
//! A is six independent complete F32 `CpuBackend`/Accelerate
//! attention-residual boundaries.
//! B is six sequential calls through `MetalW8FullAttentionStack6V1`, retaining
//! the primitive's current six-command-buffer/six-wait aggregate topology.
//! Prefix restoration is always outside the timed interval.  Because A uses
//! F32 weights while B uses packed G64 W8 weights, correctness is adjudicated
//! only against `PackedW8FullAttentionStack6V1::decode_with_prefix`; the
//! B-versus-A error is diagnostic quantization evidence, never an exactness
//! requirement.

use std::error::Error;
use std::fs::{File, OpenOptions};
use std::hint::black_box;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use apxinf_core::{Backend, CpuBackend, CpuKVCache, KvCache, Tensor};
use apxinf_metal::{
    FullAttentionLayerF32WeightsV1, FullAttentionStack6RuntimeReceiptV1,
    MetalW8FullAttentionStack6V1, PackedW8FullAttentionStack6V1, QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
    QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1, QWEN35_FULL_ATTENTION_KV_HEADS_V1,
    QWEN35_FULL_ATTENTION_KV_WIDTH_V1, QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1,
    QWEN35_FULL_ATTENTION_QUERY_HEADS_V1, QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
    QWEN35_FULL_ATTENTION_RMS_NORM_EPS_V1, QWEN35_FULL_ATTENTION_ROPE_THETA_V1,
    QWEN35_FULL_ATTENTION_ROTARY_DIM_V1, W8_GROUP_SIZE,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FORMAT: &str = "apxinf-qwen35-full-attention-decode-count6-ab-v1";
const MAX_CONTEXT: usize = 256;
const PRIMARY_POSITION: u32 = 76;
const CORRECTNESS_POSITIONS: [u32; 8] = [1, 13, 31, 76, 126, 127, 139, 255];
const INPUT_BANK_SIZE: usize = 8;
const CALLS_PER_CELL: usize = 64;
const POOLED_SAMPLES_PER_ARM: usize = 256;
const POOLED_IMPROVEMENT_THRESHOLD_PERCENT: f64 = 10.0;
const PACKED_RESIDUAL_MAX_ABS_LIMIT: f64 = 1.0e-5;
const PACKED_RESIDUAL_NRMSE_LIMIT: f64 = 1.0e-6;
const PACKED_KEY_MAX_ABS_LIMIT: f64 = 1.0e-5;
const PACKED_KEY_NRMSE_LIMIT: f64 = 2.0e-6;
const PACKED_VALUE_MAX_ABS_LIMIT: f64 = 2.0e-6;
const PACKED_VALUE_NRMSE_LIMIT: f64 = 2.0e-6;
const PACKED_COSINE_MINIMUM: f64 = 0.999_999;
const EXPECTED_ORIGIN_URL: &str = "https://github.com/qhy991/ApxInf.git";
const EXPECTED_HARDWARE_MODEL: &str = "Mac16,10";
const EXPECTED_CPU_BRAND: &str = "Apple M4";
const QUIET_HOST_SNAPSHOTS: usize = 5;
const QUIET_HOST_INTERVAL_MS: u64 = 250;
const MAX_OTHER_PROCESS_CPU_PERCENT: f64 = 10.0;
const MAX_AGGREGATE_OTHER_CPU_PERCENT: f64 = 25.0;
const BASELINE_PARENT_COMMIT: &str = "432b3858e28c3723d34aa3ec25494f176076c956";
const RAW_RECEIPT_RELATIVE_PATH: &str = "crates/apxinf-metal/evidence/next-hotspot/qwen35-full-attention-decode-v1-count6-primitive-ab-raw-v1-20260826.json";
const EMBEDDED_CANDIDATE_COMMIT: Option<&str> = option_env!("APXINF_CANDIDATE_COMMIT");

const EXPECTED_CANDIDATE_CHANGED_PATHS: [&str; 8] = [
    "crates/apxinf-metal/build.rs",
    "crates/apxinf-metal/evidence/next-hotspot/qwen35-full-attention-decode-v1-count6-predeclared-primitive-gate-v1-20260826.json",
    "crates/apxinf-metal/src/full_attention_decode_v1.rs",
    "crates/apxinf-metal/src/lib.rs",
    "crates/apxinf-metal/src/metal_full_attention_decode_v1.metal",
    "crates/apxinf-metal/src/metal_full_attention_decode_v1_bridge.mm",
    "crates/apxinf-metal/tests/full_attention_decode_v1.rs",
    "crates/apxinf-model/examples/qwen35_full_attention_decode_count6_ab_v1.rs",
];

const WEIGHT_DOMAIN: u64 = 0x243f_6a88_85a3_08d3;
const INPUT_DOMAIN: u64 = 0x1319_8a2e_0370_7344;
const PREFIX_KEY_DOMAIN: u64 = 0xa409_3822_299f_31d0;
const PREFIX_VALUE_DOMAIN: u64 = 0x082e_fa98_ec4e_6c89;

const BLOCK_ORDERS: [[Arm; 4]; 2] = [
    [Arm::A, Arm::B, Arm::B, Arm::A],
    [Arm::B, Arm::A, Arm::A, Arm::B],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    A,
    B,
}

impl Arm {
    const fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }

    const fn short(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::A => "A_cpu_f32_accelerate_full_attention_residual_count6",
            Self::B => "B_metal_g64_w8_full_attention_residual_count6",
        }
    }
}

struct Args {
    candidate_commit: String,
}

struct OwnedLayerF32 {
    input_rms_weight: Vec<f32>,
    query_rows: Vec<f32>,
    gate_rows: Vec<f32>,
    key_rows: Vec<f32>,
    value_rows: Vec<f32>,
    query_norm_weight: Vec<f32>,
    key_norm_weight: Vec<f32>,
    output_rows: Vec<f32>,
}

impl OwnedLayerF32 {
    fn borrowed(&self) -> FullAttentionLayerF32WeightsV1<'_> {
        FullAttentionLayerF32WeightsV1 {
            input_rms_weight: &self.input_rms_weight,
            query_rows: &self.query_rows,
            gate_rows: &self.gate_rows,
            key_rows: &self.key_rows,
            value_rows: &self.value_rows,
            query_norm_weight: &self.query_norm_weight,
            key_norm_weight: &self.key_norm_weight,
            output_rows: &self.output_rows,
        }
    }
}

struct CpuLayerF32 {
    input_rms_weight: Tensor,
    query_projection: Tensor,
    gate_projection: Tensor,
    key_projection: Tensor,
    value_projection: Tensor,
    query_norm_weight: Tensor,
    key_norm_weight: Tensor,
    output_projection: Tensor,
}

struct PrefixFixture {
    keys: [Vec<f32>; QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1],
    values: [Vec<f32>; QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1],
}

type StackInputs = [Vec<f32>; QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1];

#[derive(Clone)]
struct LayerResult {
    residual: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
}

#[derive(Clone)]
struct StackResult {
    flattened_residuals: Vec<f32>,
    layers: Vec<LayerResult>,
}

#[derive(Clone, Default)]
struct MetricAccumulator {
    count: u64,
    finite: bool,
    max_abs: f64,
    squared_error: f64,
    squared_reference: f64,
    actual_reference_dot: f64,
    squared_actual: f64,
}

impl MetricAccumulator {
    fn new() -> Self {
        Self {
            finite: true,
            ..Self::default()
        }
    }

    fn update(&mut self, actual: &[f32], reference: &[f32]) -> Result<(), Box<dyn Error>> {
        if actual.len() != reference.len() {
            return Err(format!(
                "metric shape mismatch: actual {} reference {}",
                actual.len(),
                reference.len()
            )
            .into());
        }
        for (&actual, &reference) in actual.iter().zip(reference) {
            let actual = f64::from(actual);
            let reference = f64::from(reference);
            if !actual.is_finite() || !reference.is_finite() {
                self.finite = false;
            }
            let error = actual - reference;
            self.max_abs = self.max_abs.max(error.abs());
            self.squared_error += error * error;
            self.squared_reference += reference * reference;
            self.actual_reference_dot += actual * reference;
            self.squared_actual += actual * actual;
            self.count += 1;
        }
        Ok(())
    }

    fn max_abs(&self) -> f64 {
        self.max_abs
    }

    fn nrmse(&self) -> f64 {
        if self.count == 0 {
            return f64::INFINITY;
        }
        let rmse = (self.squared_error / self.count as f64).sqrt();
        let reference_rms = (self.squared_reference / self.count as f64).sqrt();
        rmse / reference_rms.max(1.0e-12)
    }

    fn cosine(&self) -> f64 {
        let denominator = (self.squared_actual * self.squared_reference).sqrt();
        if denominator == 0.0 {
            if self.squared_actual == 0.0 && self.squared_reference == 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            self.actual_reference_dot / denominator
        }
    }

    fn json(&self) -> Value {
        json!({
            "elements": self.count,
            "all_finite": self.finite,
            "max_abs": self.max_abs(),
            "nrmse": self.nrmse(),
            "cosine": self.cosine()
        })
    }

    fn passes(&self, max_abs_limit: f64, nrmse_limit: f64) -> bool {
        self.finite
            && self.count != 0
            && self.max_abs() <= max_abs_limit
            && self.nrmse() <= nrmse_limit
            && self.cosine() >= PACKED_COSINE_MINIMUM
    }
}

#[derive(Default)]
struct Ledger {
    correctness_stack_calls: [u64; 3],
    correctness_layer_transactions: [u64; 3],
    prefix_restores: [u64; 2],
    warmup_stack_calls: [u64; 2],
    timed_stack_calls: [u64; 2],
    warmup_layer_transactions: [u64; 2],
    timed_layer_transactions: [u64; 2],
    metal_new_row_snapshot_calls: u64,
    metal_prefix_snapshot_calls: u64,
    metal_prefix_snapshot_elements_compared_to_bits: u64,
    rejected_validation_calls: u64,
}

type BlockSamples = [Vec<u128>; 2];

fn empty_samples() -> BlockSamples {
    std::array::from_fn(|_| Vec::new())
}

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("full-attention count-6 gate must be built with --release".into());
    }
    if !cfg!(target_os = "macos") {
        return Err("full-attention count-6 gate requires macOS Metal".into());
    }
    if !cfg!(feature = "accelerate") || !cfg!(feature = "metal-w8") {
        return Err("build with --features accelerate,metal-w8".into());
    }
    let args = parse_args()?;
    validate_commit(&args.candidate_commit)?;
    let embedded_candidate_commit =
        EMBEDDED_CANDIDATE_COMMIT.ok_or("binary was not built with APXINF_CANDIDATE_COMMIT")?;
    if embedded_candidate_commit != args.candidate_commit {
        return Err(format!(
            "embedded candidate {embedded_candidate_commit} != requested {}",
            args.candidate_commit
        )
        .into());
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("apxinf-model is not below the workspace root")?;
    let executable = std::fs::canonicalize(std::env::current_exe()?)?;
    let output = workspace_dir.join(RAW_RECEIPT_RELATIVE_PATH);
    let git_start = git_custody(workspace_dir, &args.candidate_commit, None)?;
    let custody_start = custody_snapshot(manifest_dir, workspace_dir, &executable)?;
    require_disk_matches_embedded(&custody_start)?;
    let host_preflight = collect_host_check(&args.candidate_commit, "preflight")?;
    let host_preflight_passed = host_preflight
        .get("passed")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !host_preflight_passed {
        return Err(format!(
            "formal host preflight rejected before admission: {}",
            serde_json::to_string(&host_preflight)?
        )
        .into());
    }
    // Admission begins only after all immutable preconditions pass and this
    // canonical create-new marker is durably reserved. A crash leaves the
    // marker behind, so the same candidate cannot silently be sampled again.
    let receipt_file = reserve_attempt(
        &output,
        &json!({
            "format": FORMAT,
            "status": "FORMAL_ATTEMPT_RESERVED_INCOMPLETE",
            "candidate_commit": &args.candidate_commit,
            "baseline_parent_commit": BASELINE_PARENT_COMMIT,
            "canonical_raw_receipt_path": RAW_RECEIPT_RELATIVE_PATH,
            "git_start": &git_start,
            "source_start": &custody_start,
            "host_preflight": &host_preflight,
            "performance_samples_collected": 0,
            "rerun_for_this_commit_forbidden": true
        }),
    )?;

    let setup_started = Instant::now();
    let owned_layers = deterministic_weights();
    validate_deterministic_weight_channels(&owned_layers)?;
    let weight_sha256 = hash_owned_weights(&owned_layers);
    let borrowed = owned_layers
        .iter()
        .map(OwnedLayerF32::borrowed)
        .collect::<Vec<_>>();
    let packed = PackedW8FullAttentionStack6V1::pack_f32(&borrowed)?;
    drop(borrowed);
    let cpu_layers = owned_layers
        .into_iter()
        .map(cpu_layer_from_owned)
        .collect::<Result<Vec<_>, _>>()?;
    let mut metal = MetalW8FullAttentionStack6V1::from_packed(&packed, MAX_CONTEXT)?;
    let initial_receipt = metal.runtime_receipt()?;
    validate_runtime_receipt(&initial_receipt, 0)?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;

    let mut ledger = Ledger::default();
    let mut correctness = match correctness_attempt(&cpu_layers, &packed, &mut metal, &mut ledger) {
        Ok(value) => value,
        Err(error) => json!({
            "completed": false,
            "passed": false,
            "performance_authorized": false,
            "attempt_failure": error.to_string(),
            "no_retry_performed": true
        }),
    };
    let correctness_passed = correctness
        .get("passed")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let correctness_ledger_closed_before_performance = validate_ledger(&ledger, false).is_ok();
    let expected_correctness_metal_decodes =
        (2 * CORRECTNESS_POSITIONS.len() * QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1) as u64;
    let (post_correctness_receipt, post_correctness_receipt_valid) = match metal.runtime_receipt() {
        Ok(receipt) => {
            let valid =
                validate_runtime_receipt(&receipt, expected_correctness_metal_decodes).is_ok();
            (runtime_receipt_json(&receipt), valid)
        }
        Err(error) => (json!({"error": error.to_string()}), false),
    };
    let performance_authorized = correctness_passed
        && correctness_ledger_closed_before_performance
        && post_correctness_receipt_valid;
    if let Some(object) = correctness.as_object_mut() {
        object.insert(
            "performance_authorized".to_owned(),
            Value::Bool(performance_authorized),
        );
    }

    let performance = if performance_authorized {
        match performance_attempt(&cpu_layers, &mut metal, &mut ledger) {
            Ok(value) => value,
            Err(forensic) => json!({
                "completed": false,
                "passed": false,
                "attempt_failure": forensic
            }),
        }
    } else {
        json!({
            "completed": false,
            "passed": false,
            "not_attempted_reason": "the full numerical, semantic, ledger, and live-receipt correctness gate did not close"
        })
    };
    std::thread::sleep(Duration::from_millis(1_000));
    let (host_postflight, host_postflight_passed) =
        match collect_host_check(&args.candidate_commit, "postflight") {
            Ok(check) => {
                let passed = check
                    .get("passed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                (check, passed)
            }
            Err(error) => (json!({"passed":false,"error":error.to_string()}), false),
        };

    let performance_completed = performance
        .get("completed")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let performance_passed = performance
        .get("passed")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let expected_metal_decodes = ledger.correctness_layer_transactions[1]
        + ledger.warmup_layer_transactions[1]
        + ledger.timed_layer_transactions[1];
    let final_receipt_result = metal.runtime_receipt();
    let (final_receipt, runtime_receipt_valid) = match final_receipt_result {
        Ok(receipt) => {
            let valid = validate_runtime_receipt(&receipt, expected_metal_decodes).is_ok();
            (runtime_receipt_json(&receipt), valid)
        }
        Err(error) => (json!({"error": error.to_string()}), false),
    };
    let ledger_closed = validate_ledger(&ledger, performance_completed).is_ok();

    let (custody_end, embedded_end_valid) =
        match custody_snapshot(manifest_dir, workspace_dir, &executable) {
            Ok(snapshot) => {
                let embedded_valid = require_disk_matches_embedded(&snapshot).is_ok();
                (snapshot, embedded_valid)
            }
            Err(error) => (json!({"error": error.to_string()}), false),
        };
    let (git_end, git_end_valid) = match git_custody(
        workspace_dir,
        &args.candidate_commit,
        Some(RAW_RECEIPT_RELATIVE_PATH),
    ) {
        Ok(snapshot) => (snapshot, true),
        Err(error) => (json!({"error": error.to_string()}), false),
    };
    let custody_unchanged = embedded_end_valid && git_end_valid && custody_start == custody_end;
    let passed = correctness_passed
        && host_preflight_passed
        && host_postflight_passed
        && performance_completed
        && performance_passed
        && runtime_receipt_valid
        && ledger_closed
        && custody_unchanged;

    let receipt = json!({
        "format": FORMAT,
        "classification": "same-release-binary fixed-shape count-6 primitive continuation gate; not end-to-end inference and not a cross-runtime result",
        "candidate_commit": args.candidate_commit,
        "embedded_candidate_commit": embedded_candidate_commit,
        "baseline_parent_commit": BASELINE_PARENT_COMMIT,
        "canonical_raw_receipt_path": RAW_RECEIPT_RELATIVE_PATH,
        "attempt_reservation": {
            "canonical_create_new_marker_reserved_and_synced_before_setup": true,
            "same_directory_final_file_synced_before_atomic_rename": true,
            "rerun_for_this_candidate_forbidden": true
        },
        "scope": {
            "model": "Qwen/Qwen3.5-0.8B fixed-shape diagnostic fixture",
            "A": Arm::A.label(),
            "B": Arm::B.label(),
            "hidden_size": QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
            "query_heads": QWEN35_FULL_ATTENTION_QUERY_HEADS_V1,
            "kv_heads": QWEN35_FULL_ATTENTION_KV_HEADS_V1,
            "head_dim": QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            "rotary_dim": QWEN35_FULL_ATTENTION_ROTARY_DIM_V1,
            "rope_theta": QWEN35_FULL_ATTENTION_ROPE_THETA_V1,
            "rms_norm_eps": QWEN35_FULL_ATTENTION_RMS_NORM_EPS_V1,
            "max_context": MAX_CONTEXT,
            "primary_position": PRIMARY_POSITION,
            "layer_slots_per_stack_call": QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1,
            "aggregate_semantics": "six independent per-layer boundaries; outputs are not chained because intervening GDN/MLP layers are outside this primitive gate",
            "B_command_buffers_per_stack_call": 6,
            "B_waits_per_stack_call": 6,
            "B_kernel_dispatches_per_stack_call": 30,
            "prefix_restore_inside_timed_interval": false,
            "same_per_layer_inputs_prefix_position_and_layer_order_per_arm": true,
            "A_weights": "deterministic F32 row-major checkpoint fixture, transposed once outside timing for CpuBackend matmul",
            "B_weights": "G64 symmetric W8 packing of the exact same deterministic F32 rows",
            "correctness_judge": "B versus PackedW8FullAttentionStack6V1 CPU oracle",
            "B_vs_A_error_is_non_gating": true
        },
        "setup": {
            "elapsed_ms": setup_ms,
            "weight_sha256_f32_le": weight_sha256,
            "weight_generation_and_packing_outside_timing": true,
            "cpu_projection_transpose_outside_timing": true,
            "cpu_input_tensor_materialization_outside_timing": true,
            "metal_resource_creation_outside_timing": true
        },
        "correctness": correctness,
        "pre_performance_admission": {
            "authorized": performance_authorized,
            "correctness_numerical_and_semantic_passed": correctness_passed,
            "correctness_ledger_closed": correctness_ledger_closed_before_performance,
            "expected_successful_single_layer_decodes": expected_correctness_metal_decodes,
            "post_correctness_runtime_receipt": post_correctness_receipt,
            "post_correctness_runtime_receipt_valid": post_correctness_receipt_valid
        },
        "performance": performance,
        "fixed_performance_contract": {
            "warmup_block_orders": BLOCK_ORDERS.map(|order| order.map(Arm::short)),
            "timed_block_orders": BLOCK_ORDERS.map(|order| order.map(Arm::short)),
            "calls_per_cell": CALLS_PER_CELL,
            "samples_per_arm_per_block": 128,
            "pooled_samples_per_arm": POOLED_SAMPLES_PER_ARM,
            "one_synchronous_count6_stack_call_per_raw_sample": true,
            "prefix_fixture_restore_before_timer": true,
            "output_observed_after_timer": true,
            "no_retry_resample_replacement_or_outlier_removal": true
        },
        "admission": {
            "packed_oracle_correctness_required": true,
            "pooled_median_B_over_A_percent_at_least": POOLED_IMPROVEMENT_THRESHOLD_PERCENT,
            "B_over_A_percent_strictly_positive_in_both_timed_blocks": true,
            "runtime_receipts_and_call_ledgers_must_close": true,
            "pass_only_authorizes_separate_production_integration_and_full_path_gate": true
        },
        "runtime_receipts": {
            "initial": runtime_receipt_json(&initial_receipt),
            "final": final_receipt,
            "expected_successful_single_layer_decodes": expected_metal_decodes,
            "valid": runtime_receipt_valid
        },
        "ledger": ledger_json(&ledger),
        "ledger_closed": ledger_closed,
        "host_preflight": host_preflight,
        "host_postflight": host_postflight,
        "custody": {
            "git_start": git_start,
            "git_end": git_end,
            "source_start": custody_start,
            "source_end": custody_end,
            "unchanged_during_sampling": custody_unchanged
        },
        "correctness_passed": correctness_passed,
        "host_preflight_passed": host_preflight_passed,
        "host_postflight_passed": host_postflight_passed,
        "performance_authorized": performance_authorized,
        "performance_completed": performance_completed,
        "performance_passed": performance_passed,
        "primitive_continue_gate_passed": passed,
        "formal_admission_passed": passed,
        "passed": passed
    });
    publish_reserved(receipt_file, &output, &receipt)?;
    println!("{}", serde_json::to_string(&receipt)?);
    if !passed {
        return Err("full-attention count-6 primitive rejected; receipt was published".into());
    }
    Ok(())
}

fn deterministic_weights() -> Vec<OwnedLayerF32> {
    (0..QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1)
        .map(|layer| {
            let domain = WEIGHT_DOMAIN ^ (layer as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            OwnedLayerF32 {
                input_rms_weight: deterministic_dyadic(
                    QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                    domain ^ 0x01,
                    16,
                    128.0,
                ),
                query_rows: deterministic_dyadic(
                    QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                    domain ^ 0x02,
                    32,
                    2048.0,
                ),
                gate_rows: deterministic_dyadic(
                    QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                    domain ^ 0x03,
                    32,
                    2048.0,
                ),
                key_rows: deterministic_dyadic(
                    QWEN35_FULL_ATTENTION_KV_WIDTH_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                    domain ^ 0x04,
                    32,
                    2048.0,
                ),
                value_rows: deterministic_dyadic(
                    QWEN35_FULL_ATTENTION_KV_WIDTH_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                    domain ^ 0x05,
                    32,
                    2048.0,
                ),
                query_norm_weight: deterministic_dyadic(
                    QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
                    domain ^ 0x06,
                    16,
                    128.0,
                ),
                key_norm_weight: deterministic_dyadic(
                    QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
                    domain ^ 0x07,
                    16,
                    128.0,
                ),
                output_rows: deterministic_dyadic(
                    QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1 * QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
                    domain ^ 0x08,
                    32,
                    2048.0,
                ),
            }
        })
        .collect()
}

fn validate_deterministic_weight_channels(layers: &[OwnedLayerF32]) -> Result<(), Box<dyn Error>> {
    if layers.len() != QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
        return Err("deterministic full-attention fixture has the wrong layer count".into());
    }
    for (layer_slot, layer) in layers.iter().enumerate() {
        if layer.query_rows == layer.gate_rows
            || layer.key_rows == layer.value_rows
            || layer.query_norm_weight == layer.key_norm_weight
        {
            return Err(format!(
                "deterministic full-attention fixture aliases semantic channels at layer slot {layer_slot}"
            )
            .into());
        }
    }
    Ok(())
}

fn cpu_layer_from_owned(layer: OwnedLayerF32) -> Result<CpuLayerF32, apxinf_core::Error> {
    Ok(CpuLayerF32 {
        input_rms_weight: Tensor::from_f32_vec(
            vec![QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1],
            layer.input_rms_weight,
        )?,
        query_projection: Tensor::from_f32_vec(
            vec![
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
            ],
            transpose_rows(
                &layer.query_rows,
                QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
            ),
        )?,
        gate_projection: Tensor::from_f32_vec(
            vec![
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
            ],
            transpose_rows(
                &layer.gate_rows,
                QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
            ),
        )?,
        key_projection: Tensor::from_f32_vec(
            vec![
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                QWEN35_FULL_ATTENTION_KV_WIDTH_V1,
            ],
            transpose_rows(
                &layer.key_rows,
                QWEN35_FULL_ATTENTION_KV_WIDTH_V1,
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
            ),
        )?,
        value_projection: Tensor::from_f32_vec(
            vec![
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                QWEN35_FULL_ATTENTION_KV_WIDTH_V1,
            ],
            transpose_rows(
                &layer.value_rows,
                QWEN35_FULL_ATTENTION_KV_WIDTH_V1,
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
            ),
        )?,
        query_norm_weight: Tensor::from_f32_vec(
            vec![QWEN35_FULL_ATTENTION_HEAD_DIM_V1],
            layer.query_norm_weight,
        )?,
        key_norm_weight: Tensor::from_f32_vec(
            vec![QWEN35_FULL_ATTENTION_HEAD_DIM_V1],
            layer.key_norm_weight,
        )?,
        output_projection: Tensor::from_f32_vec(
            vec![
                QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
            ],
            transpose_rows(
                &layer.output_rows,
                QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
                QWEN35_FULL_ATTENTION_QUERY_WIDTH_V1,
            ),
        )?,
    })
}

fn transpose_rows(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    assert_eq!(values.len(), rows * columns);
    let mut transposed = vec![0.0f32; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            transposed[column * rows + row] = values[row * columns + column];
        }
    }
    transposed
}

fn next_xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn deterministic_dyadic(elements: usize, domain: u64, radius: i32, denominator: f32) -> Vec<f32> {
    let mut state = if domain == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        domain
    };
    let width = (2 * radius + 1) as u64;
    (0..elements)
        .map(|_| {
            let bucket = (next_xorshift(&mut state) >> 32) % width;
            (bucket as i32 - radius) as f32 / denominator
        })
        .collect()
}

fn prefix_fixture(position: u32, fixture_index: usize) -> PrefixFixture {
    let elements = position as usize * QWEN35_FULL_ATTENTION_KV_WIDTH_V1;
    PrefixFixture {
        keys: std::array::from_fn(|layer| {
            deterministic_dyadic(
                elements,
                PREFIX_KEY_DOMAIN ^ ((fixture_index as u64) << 40) ^ ((layer as u64) << 8),
                1024,
                1024.0,
            )
        }),
        values: std::array::from_fn(|layer| {
            deterministic_dyadic(
                elements,
                PREFIX_VALUE_DOMAIN ^ ((fixture_index as u64) << 40) ^ ((layer as u64) << 8),
                512,
                1024.0,
            )
        }),
    }
}

fn input_fixture(fixture_index: usize) -> StackInputs {
    std::array::from_fn(|layer| {
        deterministic_dyadic(
            QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
            INPUT_DOMAIN
                ^ (fixture_index as u64).wrapping_mul(0xd134_2543_de82_ef95)
                ^ (layer as u64).wrapping_mul(0x94d0_49bb_1331_11eb),
            768,
            1024.0,
        )
    })
}

fn cpu_stack_inputs(inputs: &StackInputs) -> Result<Vec<Tensor>, Box<dyn Error>> {
    inputs
        .iter()
        .map(|input| {
            Tensor::from_f32(vec![1, QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1], input)
                .map_err(|error| Box::new(error) as Box<dyn Error>)
        })
        .collect()
}

fn seed_cpu_cache(
    backend: &CpuBackend,
    prefix: &PrefixFixture,
    position: u32,
) -> Result<CpuKVCache, Box<dyn Error>> {
    let mut cache = CpuKVCache::new(
        QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1,
        QWEN35_FULL_ATTENTION_KV_HEADS_V1,
        QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
        MAX_CONTEXT,
    );
    for layer in 0..QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
        let key = Tensor::from_f32(
            vec![
                position as usize,
                QWEN35_FULL_ATTENTION_KV_HEADS_V1,
                QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            ],
            &prefix.keys[layer],
        )?;
        let value = Tensor::from_f32(
            vec![
                position as usize,
                QWEN35_FULL_ATTENTION_KV_HEADS_V1,
                QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            ],
            &prefix.values[layer],
        )?;
        backend.kv_append(&mut cache, layer, &key, &value, position as usize)?;
    }
    cache.advance(position as usize);
    Ok(cache)
}

fn seed_metal(
    metal: &mut MetalW8FullAttentionStack6V1,
    prefix: &PrefixFixture,
    position: u32,
) -> Result<(), Box<dyn Error>> {
    for layer in 0..QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
        metal.seed_cache(layer, position, &prefix.keys[layer], &prefix.values[layer])?;
    }
    Ok(())
}

fn run_cpu_stack(
    backend: &CpuBackend,
    layers: &[CpuLayerF32],
    cache: &mut CpuKVCache,
    inputs: &[Tensor],
    position: u32,
    capture_layers: bool,
) -> Result<StackResult, Box<dyn Error>> {
    if inputs.len() != QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
        return Err(format!(
            "CPU full-attention stack requires {} independent inputs, got {}",
            QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1,
            inputs.len()
        )
        .into());
    }
    let mut flattened_residuals = Vec::with_capacity(
        QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
    );
    let mut results = Vec::with_capacity(if capture_layers {
        QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1
    } else {
        0
    });
    for (layer_slot, layer) in layers.iter().enumerate() {
        let hidden = &inputs[layer_slot];
        let normalized = backend.rms_norm_offset(
            hidden,
            &layer.input_rms_weight,
            QWEN35_FULL_ATTENTION_RMS_NORM_EPS_V1,
            1.0,
        )?;
        let query = backend
            .matmul(&normalized, &layer.query_projection)?
            .reshape(vec![
                QWEN35_FULL_ATTENTION_QUERY_HEADS_V1,
                QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            ])?;
        let query = backend
            .rms_norm_offset(
                &query,
                &layer.query_norm_weight,
                QWEN35_FULL_ATTENTION_RMS_NORM_EPS_V1,
                1.0,
            )?
            .reshape(vec![
                1,
                QWEN35_FULL_ATTENTION_QUERY_HEADS_V1,
                QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            ])?;
        let key = backend
            .matmul(&normalized, &layer.key_projection)?
            .reshape(vec![
                QWEN35_FULL_ATTENTION_KV_HEADS_V1,
                QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            ])?;
        let key = backend
            .rms_norm_offset(
                &key,
                &layer.key_norm_weight,
                QWEN35_FULL_ATTENTION_RMS_NORM_EPS_V1,
                1.0,
            )?
            .reshape(vec![
                1,
                QWEN35_FULL_ATTENTION_KV_HEADS_V1,
                QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            ])?;
        let value = backend
            .matmul(&normalized, &layer.value_projection)?
            .reshape(vec![
                1,
                QWEN35_FULL_ATTENTION_KV_HEADS_V1,
                QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            ])?;
        let query = backend.rope_partial(
            &query,
            QWEN35_FULL_ATTENTION_QUERY_HEADS_V1,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            QWEN35_FULL_ATTENTION_ROTARY_DIM_V1,
            QWEN35_FULL_ATTENTION_ROPE_THETA_V1,
            position,
            false,
        )?;
        let key = backend.rope_partial(
            &key,
            QWEN35_FULL_ATTENTION_KV_HEADS_V1,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            QWEN35_FULL_ATTENTION_ROTARY_DIM_V1,
            QWEN35_FULL_ATTENTION_ROPE_THETA_V1,
            position,
            false,
        )?;
        backend.kv_append(cache, layer_slot, &key, &value, 1)?;
        let attention = backend.sdpa_decode(
            &query,
            cache,
            layer_slot,
            QWEN35_FULL_ATTENTION_QUERY_HEADS_V1,
            QWEN35_FULL_ATTENTION_KV_HEADS_V1,
            QWEN35_FULL_ATTENTION_HEAD_DIM_V1,
            position as usize + 1,
            MAX_CONTEXT,
        )?;
        let gate = backend.sigmoid(&backend.matmul(&normalized, &layer.gate_projection)?)?;
        let attention = backend.mul(&attention, &gate)?;
        let projected = backend.matmul(&attention, &layer.output_projection)?;
        let residual = backend.add(hidden, &projected)?;
        flattened_residuals.extend_from_slice(residual.as_f32()?);

        if capture_layers {
            let key_row = cache_row_sequence_major(cache, layer_slot, position, true);
            let value_row = cache_row_sequence_major(cache, layer_slot, position, false);
            results.push(LayerResult {
                residual: residual.as_f32()?.to_vec(),
                key: key_row,
                value: value_row,
            });
        }
    }
    Ok(StackResult {
        flattened_residuals,
        layers: results,
    })
}

fn cache_row_sequence_major(
    cache: &CpuKVCache,
    layer_slot: usize,
    position: u32,
    key: bool,
) -> Vec<f32> {
    let (keys, values) = cache.get_kv(layer_slot);
    let source = if key { keys } else { values };
    let mut row = Vec::with_capacity(QWEN35_FULL_ATTENTION_KV_WIDTH_V1);
    for head in 0..QWEN35_FULL_ATTENTION_KV_HEADS_V1 {
        let offset = cache.row_offset(head, position as usize);
        row.extend_from_slice(&source[offset..offset + QWEN35_FULL_ATTENTION_HEAD_DIM_V1]);
    }
    row
}

fn run_packed_oracle_stack(
    packed: &PackedW8FullAttentionStack6V1,
    prefix: &PrefixFixture,
    inputs: &StackInputs,
    position: u32,
) -> Result<StackResult, Box<dyn Error>> {
    let mut flattened_residuals = Vec::with_capacity(
        QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
    );
    let mut results = Vec::with_capacity(QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1);
    for layer in 0..QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
        let decoded = packed.decode_with_prefix(
            layer,
            &inputs[layer],
            position,
            &prefix.keys[layer],
            &prefix.values[layer],
        )?;
        flattened_residuals.extend_from_slice(&decoded.residual);
        results.push(LayerResult {
            residual: decoded.residual,
            key: decoded.key,
            value: decoded.value,
        });
    }
    Ok(StackResult {
        flattened_residuals,
        layers: results,
    })
}

fn run_metal_stack(
    metal: &mut MetalW8FullAttentionStack6V1,
    inputs: &StackInputs,
    position: u32,
    capture_layers: bool,
) -> Result<StackResult, Box<dyn Error>> {
    let mut flattened_residuals = Vec::with_capacity(
        QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 * QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1,
    );
    let mut results = Vec::with_capacity(if capture_layers {
        QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1
    } else {
        0
    });
    for layer in 0..QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
        let residual = metal.decode(layer, &inputs[layer], position)?.to_vec();
        flattened_residuals.extend_from_slice(&residual);
        if capture_layers {
            let (key, value) = metal.snapshot_cache_row(layer, position)?;
            results.push(LayerResult {
                residual,
                key,
                value,
            });
        }
    }
    Ok(StackResult {
        flattened_residuals,
        layers: results,
    })
}

fn correctness_attempt(
    cpu_layers: &[CpuLayerF32],
    packed: &PackedW8FullAttentionStack6V1,
    metal: &mut MetalW8FullAttentionStack6V1,
    ledger: &mut Ledger,
) -> Result<Value, Box<dyn Error>> {
    const ORDER: [Arm; 4] = [Arm::A, Arm::B, Arm::B, Arm::A];
    let backend = CpuBackend;
    let mut packed_residual = MetricAccumulator::new();
    let mut packed_key = MetricAccumulator::new();
    let mut packed_value = MetricAccumulator::new();
    let mut f32_residual = MetricAccumulator::new();
    let mut f32_key = MetricAccumulator::new();
    let mut f32_value = MetricAccumulator::new();
    let mut fixture_hasher = Sha256::new();
    fixture_hasher.update(b"qwen35-full-attention-count6-independent-correctness-v1");
    let mut snapshot_hasher = Sha256::new();
    snapshot_hasher.update(b"qwen35-full-attention-count6-metal-kv-snapshots-v1");
    let mut cases = Vec::with_capacity(CORRECTNESS_POSITIONS.len());
    let mut prefix_preserved = true;
    let mut a_repeat_to_bits = true;
    let mut a_all_finite = true;

    // These four host-validation failures must not submit Metal work.  The
    // already seeded position-1 cache is intentionally reused by the first B
    // aggregate below, making that call the predeclared same-position retry.
    let validation_inputs = input_fixture(0);
    let validation_prefix = prefix_fixture(CORRECTNESS_POSITIONS[0], 0);
    seed_metal(metal, &validation_prefix, CORRECTNESS_POSITIONS[0])?;
    ledger.prefix_restores[Arm::B.index()] += 1;
    let receipt_before_validation = metal.runtime_receipt()?;
    let mut validation_errors = Vec::with_capacity(4);
    let mut nonfinite = validation_inputs[0].clone();
    nonfinite[127] = f32::NAN;
    for (label, result) in [
        (
            "nonfinite_input",
            metal
                .decode(0, &nonfinite, CORRECTNESS_POSITIONS[0])
                .map(|_| ()),
        ),
        (
            "wrong_input_shape",
            metal
                .decode(
                    0,
                    &validation_inputs[0][..QWEN35_FULL_ATTENTION_HIDDEN_SIZE_V1 - 1],
                    CORRECTNESS_POSITIONS[0],
                )
                .map(|_| ()),
        ),
        (
            "invalid_layer_slot",
            metal
                .decode(
                    QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1,
                    &validation_inputs[0],
                    CORRECTNESS_POSITIONS[0],
                )
                .map(|_| ()),
        ),
        (
            "capacity_position",
            metal
                .decode(0, &validation_inputs[0], MAX_CONTEXT as u32)
                .map(|_| ()),
        ),
    ] {
        match result {
            Ok(_) => return Err(format!("validation probe {label} unexpectedly succeeded").into()),
            Err(error) => {
                ledger.rejected_validation_calls += 1;
                validation_errors.push(json!({"probe":label,"error":error.to_string()}));
            }
        }
    }
    let receipt_after_validation = metal.runtime_receipt()?;
    let validation_receipt_preserved = receipt_before_validation == receipt_after_validation;
    if !validation_receipt_preserved {
        return Err("rejected validation calls changed the live receipt".into());
    }
    let mut first_b_uses_preseeded_retry = true;
    let mut same_position_retry_succeeded = false;

    for (fixture_index, &position) in CORRECTNESS_POSITIONS.iter().enumerate() {
        let inputs = input_fixture(fixture_index);
        let cpu_inputs = cpu_stack_inputs(&inputs)?;
        let prefix = prefix_fixture(position, fixture_index);
        for input in &inputs {
            hash_f32(&mut fixture_hasher, b"slot_input", input);
        }
        for layer in 0..QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
            hash_f32(&mut fixture_hasher, b"prefix_key", &prefix.keys[layer]);
            hash_f32(&mut fixture_hasher, b"prefix_value", &prefix.values[layer]);
        }

        let oracle = run_packed_oracle_stack(packed, &prefix, &inputs, position)?;
        ledger.correctness_stack_calls[2] += 1;
        ledger.correctness_layer_transactions[2] += QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64;

        let mut first_a: Option<StackResult> = None;
        let mut case_a_repeat_to_bits = true;
        let mut case_packed_residual = MetricAccumulator::new();
        let mut case_packed_key = MetricAccumulator::new();
        let mut case_packed_value = MetricAccumulator::new();
        let mut case_f32_residual = MetricAccumulator::new();
        let mut case_f32_key = MetricAccumulator::new();
        let mut case_f32_value = MetricAccumulator::new();
        let mut a_output_hashes = Vec::with_capacity(2);
        let mut b_output_hashes = Vec::with_capacity(2);

        for arm in ORDER {
            match arm {
                Arm::A => {
                    let mut cpu_cache = seed_cpu_cache(&backend, &prefix, position)?;
                    ledger.prefix_restores[Arm::A.index()] += 1;
                    let result = run_cpu_stack(
                        &backend,
                        cpu_layers,
                        &mut cpu_cache,
                        &cpu_inputs,
                        position,
                        true,
                    )?;
                    a_all_finite &= stack_all_finite(&result);
                    ledger.correctness_stack_calls[0] += 1;
                    ledger.correctness_layer_transactions[0] +=
                        QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64;
                    a_output_hashes.push(sha256_f32(&result.flattened_residuals));
                    if let Some(reference) = &first_a {
                        case_a_repeat_to_bits &= stack_to_bits_equal(&result, reference);
                        a_repeat_to_bits &= case_a_repeat_to_bits;
                    } else {
                        first_a = Some(result);
                    }
                }
                Arm::B => {
                    if first_b_uses_preseeded_retry {
                        if fixture_index != 0 || position != CORRECTNESS_POSITIONS[0] {
                            return Err("preseeded retry escaped the first fixture".into());
                        }
                        first_b_uses_preseeded_retry = false;
                    } else {
                        seed_metal(metal, &prefix, position)?;
                        ledger.prefix_restores[Arm::B.index()] += 1;
                    }
                    let result = run_metal_stack(metal, &inputs, position, true)?;
                    ledger.correctness_stack_calls[1] += 1;
                    ledger.correctness_layer_transactions[1] +=
                        QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64;
                    ledger.metal_new_row_snapshot_calls +=
                        QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64;
                    if fixture_index == 0 && !same_position_retry_succeeded {
                        same_position_retry_succeeded = true;
                    }
                    b_output_hashes.push(sha256_f32(&result.flattened_residuals));
                    let f32_reference = first_a
                        .as_ref()
                        .ok_or("ABBA correctness reached B before its F32 A reference")?;

                    for layer in 0..QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
                        let actual = &result.layers[layer];
                        let expected = &oracle.layers[layer];
                        let f32 = &f32_reference.layers[layer];
                        packed_residual.update(&actual.residual, &expected.residual)?;
                        case_packed_residual.update(&actual.residual, &expected.residual)?;
                        packed_key.update(&actual.key, &expected.key)?;
                        case_packed_key.update(&actual.key, &expected.key)?;
                        packed_value.update(&actual.value, &expected.value)?;
                        case_packed_value.update(&actual.value, &expected.value)?;
                        f32_residual.update(&actual.residual, &f32.residual)?;
                        case_f32_residual.update(&actual.residual, &f32.residual)?;
                        f32_key.update(&actual.key, &f32.key)?;
                        case_f32_key.update(&actual.key, &f32.key)?;
                        f32_value.update(&actual.value, &f32.value)?;
                        case_f32_value.update(&actual.value, &f32.value)?;
                        hash_f32(&mut snapshot_hasher, b"new_key", &actual.key);
                        hash_f32(&mut snapshot_hasher, b"new_value", &actual.value);

                        for prefix_position in prefix_snapshot_positions(position) {
                            let (snapshot_key, snapshot_value) =
                                metal.snapshot_cache_row(layer, prefix_position)?;
                            ledger.metal_prefix_snapshot_calls += 1;
                            let base = prefix_position as usize * QWEN35_FULL_ATTENTION_KV_WIDTH_V1;
                            let expected_key =
                                &prefix.keys[layer][base..base + QWEN35_FULL_ATTENTION_KV_WIDTH_V1];
                            let expected_value = &prefix.values[layer]
                                [base..base + QWEN35_FULL_ATTENTION_KV_WIDTH_V1];
                            ledger.metal_prefix_snapshot_elements_compared_to_bits +=
                                (2 * QWEN35_FULL_ATTENTION_KV_WIDTH_V1) as u64;
                            prefix_preserved &= to_bits_equal(&snapshot_key, expected_key)
                                && to_bits_equal(&snapshot_value, expected_value);
                            hash_f32(&mut snapshot_hasher, b"prefix_key", &snapshot_key);
                            hash_f32(&mut snapshot_hasher, b"prefix_value", &snapshot_value);
                        }
                    }
                }
            }
        }
        cases.push(json!({
            "fixture_index": fixture_index,
            "position": position,
            "order": ORDER.map(Arm::short),
            "independent_slot_inputs": true,
            "A_repeat_to_bits": case_a_repeat_to_bits,
            "B_vs_packed_W8_oracle": {
                "residual": case_packed_residual.json(),
                "appended_key": case_packed_key.json(),
                "appended_value": case_packed_value.json()
            },
            "B_W8_vs_A_F32_non_gating": {
                "residual": case_f32_residual.json(),
                "appended_key": case_f32_key.json(),
                "appended_value": case_f32_value.json()
            },
            "flattened_six_residual_hashes": {
                "packed_oracle": sha256_f32(&oracle.flattened_residuals),
                "A_runs": a_output_hashes,
                "B_runs": b_output_hashes
            }
        }));
    }

    let residual_passed =
        packed_residual.passes(PACKED_RESIDUAL_MAX_ABS_LIMIT, PACKED_RESIDUAL_NRMSE_LIMIT);
    let key_passed = packed_key.passes(PACKED_KEY_MAX_ABS_LIMIT, PACKED_KEY_NRMSE_LIMIT);
    let value_passed = packed_value.passes(PACKED_VALUE_MAX_ABS_LIMIT, PACKED_VALUE_NRMSE_LIMIT);
    let validation_passed = ledger.rejected_validation_calls == 4
        && validation_receipt_preserved
        && same_position_retry_succeeded;
    let passed = residual_passed
        && key_passed
        && value_passed
        && prefix_preserved
        && a_repeat_to_bits
        && a_all_finite
        && validation_passed;
    Ok(json!({
        "completed": true,
        "passed": passed,
        "performance_authorized": passed,
        "positions": CORRECTNESS_POSITIONS,
        "order_per_fixture": ORDER.map(Arm::short),
        "fixture_sha256_f32_le": format!("{:x}", fixture_hasher.finalize()),
        "metal_kv_snapshot_sha256_f32_le": format!("{:x}", snapshot_hasher.finalize()),
        "B_vs_packed_W8_oracle_gating": {
            "residual": packed_residual.json(),
            "appended_key": packed_key.json(),
            "appended_value": packed_value.json(),
            "thresholds": {
                "residual_max_abs_at_most": PACKED_RESIDUAL_MAX_ABS_LIMIT,
                "residual_nrmse_at_most": PACKED_RESIDUAL_NRMSE_LIMIT,
                "key_max_abs_at_most": PACKED_KEY_MAX_ABS_LIMIT,
                "key_nrmse_at_most": PACKED_KEY_NRMSE_LIMIT,
                "value_max_abs_at_most": PACKED_VALUE_MAX_ABS_LIMIT,
                "value_nrmse_at_most": PACKED_VALUE_NRMSE_LIMIT,
                "all_cosine_at_least": PACKED_COSINE_MINIMUM
            },
            "residual_passed": residual_passed,
            "key_passed": key_passed,
            "value_passed": value_passed
        },
        "B_W8_vs_A_F32_non_gating": {
            "qualification": "includes intended G64 W8 quantization plus backend arithmetic differences; reported only, never used for exactness admission",
            "A_all_outputs_and_kv_finite_gating": a_all_finite,
            "residual": f32_residual.json(),
            "appended_key": f32_key.json(),
            "appended_value": f32_value.json()
        },
        "validation_failure_semantics": {
            "passed": validation_passed,
            "rejected_calls": validation_errors,
            "receipt_unchanged_across_rejections": validation_receipt_preserved,
            "same_position_retry_succeeded": same_position_retry_succeeded,
            "rejected_calls_incremented_successful_decodes": false
        },
        "A_repeat_to_bits": a_repeat_to_bits,
        "kv_snapshot_custody": {
            "new_row_snapshot_calls": ledger.metal_new_row_snapshot_calls,
            "prefix_snapshot_calls": ledger.metal_prefix_snapshot_calls,
            "prefix_elements_compared_to_bits": ledger.metal_prefix_snapshot_elements_compared_to_bits,
            "prefix_positions_policy": "every seeded row 0..start_pos for both B calls, every fixture, and every layer slot",
            "seeded_prefix_rows_preserved_to_bits": prefix_preserved
        },
        "cases": cases
    }))
}

fn stack_to_bits_equal(left: &StackResult, right: &StackResult) -> bool {
    to_bits_equal(&left.flattened_residuals, &right.flattened_residuals)
        && left.layers.len() == right.layers.len()
        && left.layers.iter().zip(&right.layers).all(|(left, right)| {
            to_bits_equal(&left.residual, &right.residual)
                && to_bits_equal(&left.key, &right.key)
                && to_bits_equal(&left.value, &right.value)
        })
}

fn stack_all_finite(stack: &StackResult) -> bool {
    stack
        .flattened_residuals
        .iter()
        .all(|value| value.is_finite())
        && stack.layers.iter().all(|layer| {
            layer.residual.iter().all(|value| value.is_finite())
                && layer.key.iter().all(|value| value.is_finite())
                && layer.value.iter().all(|value| value.is_finite())
        })
}

fn prefix_snapshot_positions(position: u32) -> std::ops::Range<u32> {
    0..position
}

fn to_bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

fn performance_attempt(
    cpu_layers: &[CpuLayerF32],
    metal: &mut MetalW8FullAttentionStack6V1,
    ledger: &mut Ledger,
) -> Result<Value, Value> {
    let backend = CpuBackend;
    let prefix = prefix_fixture(PRIMARY_POSITION, 0x100);
    let inputs = (0..INPUT_BANK_SIZE)
        .map(|index| input_fixture(0x100 + index))
        .collect::<Vec<_>>();
    let cpu_inputs = inputs
        .iter()
        .map(cpu_stack_inputs)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            json!({
                "phase": "fixture_setup",
                "error": error.to_string(),
                "performance_samples_collected": 0,
                "no_retry_performed": true
            })
        })?;
    let fixture_sha256 = hash_performance_fixture(&prefix, &inputs);
    let mut warmup_completed = [0usize; 2];

    for (block_index, order) in BLOCK_ORDERS.into_iter().enumerate() {
        for (cell_index, arm) in order.into_iter().enumerate() {
            for call_index in 0..CALLS_PER_CELL {
                let input_index = call_index % inputs.len();
                let input = &inputs[input_index];
                let cpu_input = &cpu_inputs[input_index];
                if let Err(error) = run_one_untimed_prepare_and_call(
                    arm, &backend, cpu_layers, metal, &prefix, input, cpu_input, ledger, false,
                ) {
                    return Err(json!({
                        "phase": "warmup",
                        "block_index": block_index,
                        "order": order.map(Arm::short),
                        "cell_index": cell_index,
                        "call_index_within_cell": call_index,
                        "arm": arm.label(),
                        "error": error.to_string(),
                        "warmup_completed_calls": counts_json(&warmup_completed),
                        "no_retry_performed": true
                    }));
                }
                warmup_completed[arm.index()] += 1;
            }
        }
    }

    let mut blocks = Vec::with_capacity(BLOCK_ORDERS.len());
    for (block_index, order) in BLOCK_ORDERS.into_iter().enumerate() {
        let mut samples = empty_samples();
        for (cell_index, arm) in order.into_iter().enumerate() {
            for call_index in 0..CALLS_PER_CELL {
                let input_index = call_index % inputs.len();
                let input = &inputs[input_index];
                let cpu_input = &cpu_inputs[input_index];
                match run_one_untimed_prepare_and_call(
                    arm, &backend, cpu_layers, metal, &prefix, input, cpu_input, ledger, true,
                ) {
                    Ok(Some(elapsed_ns)) => samples[arm.index()].push(elapsed_ns),
                    Ok(None) => {
                        return Err(json!({
                            "phase": "timed",
                            "error": "timed call returned no sample",
                            "block_index": block_index,
                            "cell_index": cell_index,
                            "call_index_within_cell": call_index,
                            "partial_samples": samples_json(&samples),
                            "no_retry_performed": true
                        }));
                    }
                    Err(error) => {
                        return Err(json!({
                            "phase": "timed",
                            "error": error.to_string(),
                            "block_index": block_index,
                            "order": order.map(Arm::short),
                            "cell_index": cell_index,
                            "call_index_within_cell": call_index,
                            "arm": arm.label(),
                            "partial_samples": samples_json(&samples),
                            "completed_blocks": blocks.iter().map(samples_json).collect::<Vec<_>>(),
                            "no_retry_performed": true
                        }));
                    }
                }
            }
        }
        blocks.push(samples);
    }
    performance_json(&blocks, &warmup_completed, fixture_sha256)
        .map_err(|error| json!({"phase":"statistics","error":error.to_string()}))
}

#[allow(clippy::too_many_arguments)]
fn run_one_untimed_prepare_and_call(
    arm: Arm,
    backend: &CpuBackend,
    cpu_layers: &[CpuLayerF32],
    metal: &mut MetalW8FullAttentionStack6V1,
    prefix: &PrefixFixture,
    input: &StackInputs,
    cpu_input: &[Tensor],
    ledger: &mut Ledger,
    timed: bool,
) -> Result<Option<u128>, Box<dyn Error>> {
    match arm {
        Arm::A => {
            let mut cache = seed_cpu_cache(backend, prefix, PRIMARY_POSITION)?;
            ledger.prefix_restores[arm.index()] += 1;
            let started = Instant::now();
            let result = run_cpu_stack(
                backend,
                cpu_layers,
                &mut cache,
                cpu_input,
                PRIMARY_POSITION,
                false,
            )?;
            let elapsed = started.elapsed().as_nanos();
            black_box(&result.flattened_residuals);
            if !result
                .flattened_residuals
                .iter()
                .all(|value| value.is_finite())
            {
                return Err("CPU performance call produced a non-finite residual".into());
            }
            if timed {
                ledger.timed_stack_calls[arm.index()] += 1;
                ledger.timed_layer_transactions[arm.index()] +=
                    QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64;
                Ok(Some(elapsed))
            } else {
                ledger.warmup_stack_calls[arm.index()] += 1;
                ledger.warmup_layer_transactions[arm.index()] +=
                    QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64;
                Ok(None)
            }
        }
        Arm::B => {
            seed_metal(metal, prefix, PRIMARY_POSITION)?;
            ledger.prefix_restores[arm.index()] += 1;
            let started = Instant::now();
            let result = run_metal_stack(metal, input, PRIMARY_POSITION, false)?;
            let elapsed = started.elapsed().as_nanos();
            black_box(&result.flattened_residuals);
            if !result
                .flattened_residuals
                .iter()
                .all(|value| value.is_finite())
            {
                return Err("Metal performance call produced a non-finite residual".into());
            }
            if timed {
                ledger.timed_stack_calls[arm.index()] += 1;
                ledger.timed_layer_transactions[arm.index()] +=
                    QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64;
                Ok(Some(elapsed))
            } else {
                ledger.warmup_stack_calls[arm.index()] += 1;
                ledger.warmup_layer_transactions[arm.index()] +=
                    QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64;
                Ok(None)
            }
        }
    }
}

fn performance_json(
    blocks: &[BlockSamples],
    warmup_completed: &[usize; 2],
    fixture_sha256: String,
) -> Result<Value, Box<dyn Error>> {
    if blocks.len() != BLOCK_ORDERS.len()
        || blocks
            .iter()
            .any(|block| block.iter().any(|samples| samples.len() != 128))
        || *warmup_completed != [256, 256]
    {
        return Err("fixed ABBA/BAAB schedule did not close".into());
    }
    let block_medians = blocks
        .iter()
        .map(|block| [even_median(&block[0]), even_median(&block[1])])
        .collect::<Vec<_>>();
    let improvements = block_medians
        .iter()
        .map(|median| improvement_percent(median[1], median[0]))
        .collect::<Vec<_>>();
    let mut pooled = empty_samples();
    for block in blocks {
        pooled[0].extend_from_slice(&block[0]);
        pooled[1].extend_from_slice(&block[1]);
    }
    if pooled
        .iter()
        .any(|samples| samples.len() != POOLED_SAMPLES_PER_ARM)
    {
        return Err("pooled sample count mismatch".into());
    }
    let pooled_a = even_median(&pooled[0]);
    let pooled_b = even_median(&pooled[1]);
    let pooled_improvement = improvement_percent(pooled_b, pooled_a);
    let pooled_threshold_passed = pooled_improvement >= POOLED_IMPROVEMENT_THRESHOLD_PERCENT;
    let positive_in_both_blocks = improvements.iter().all(|&value| value > 0.0);
    let passed = pooled_threshold_passed && positive_in_both_blocks;
    Ok(json!({
        "completed": true,
        "passed": passed,
        "performance_fixture_sha256_f32_le": fixture_sha256,
        "warmup_completed_calls": counts_json(warmup_completed),
        "blocks": blocks.iter().enumerate().map(|(index, block)| json!({
            "block_index": index,
            "order": BLOCK_ORDERS[index].map(Arm::short),
            "A": sample_summary(&block[0]),
            "B": sample_summary(&block[1]),
            "B_over_A_median_improvement_percent": improvements[index]
        })).collect::<Vec<_>>(),
        "pooled": {
            "A": sample_summary(&pooled[0]),
            "B": sample_summary(&pooled[1]),
            "B_over_A_median_improvement_percent": pooled_improvement
        },
        "pooled_threshold_percent": POOLED_IMPROVEMENT_THRESHOLD_PERCENT,
        "pooled_threshold_passed": pooled_threshold_passed,
        "B_over_A_positive_in_both_blocks": positive_in_both_blocks,
        "no_retry_resample_replacement_or_outlier_removal": true
    }))
}

fn sorted(samples: &[u128]) -> Vec<u128> {
    let mut result = samples.to_vec();
    result.sort_unstable();
    result
}

fn even_median(samples: &[u128]) -> f64 {
    let sorted = sorted(samples);
    let upper = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[upper - 1] as f64 + sorted[upper] as f64) / 2.0
    } else {
        sorted[upper] as f64
    }
}

fn percentile(samples: &[u128], numerator: usize, denominator: usize) -> u128 {
    let sorted = sorted(samples);
    sorted[(sorted.len() - 1) * numerator / denominator]
}

fn improvement_percent(candidate: f64, baseline: f64) -> f64 {
    (baseline - candidate) / baseline * 100.0
}

fn sample_summary(samples: &[u128]) -> Value {
    json!({
        "raw_ns": samples,
        "count": samples.len(),
        "median_ns": even_median(samples),
        "p10_ns": percentile(samples, 1, 10),
        "p90_ns": percentile(samples, 9, 10)
    })
}

fn samples_json(samples: &BlockSamples) -> Value {
    json!({
        "A_raw_ns": samples[0],
        "B_raw_ns": samples[1],
        "A_completed": samples[0].len(),
        "B_completed": samples[1].len()
    })
}

fn counts_json(counts: &[usize; 2]) -> Value {
    json!({"A":counts[0],"B":counts[1]})
}

fn validate_ledger(ledger: &Ledger, performance_completed: bool) -> Result<(), String> {
    let expected_correctness = CORRECTNESS_POSITIONS.len() as u64;
    let expected_correctness_stacks = [
        2 * expected_correctness,
        2 * expected_correctness,
        expected_correctness,
    ];
    let expected_correctness_layers = [
        2 * expected_correctness * QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64,
        2 * expected_correctness * QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64,
        expected_correctness * QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64,
    ];
    let expected_prefix_snapshots = CORRECTNESS_POSITIONS
        .iter()
        .map(|&position| u64::from(position))
        .sum::<u64>()
        * 2
        * QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u64;
    let expected_prefix_elements =
        expected_prefix_snapshots * (2 * QWEN35_FULL_ATTENTION_KV_WIDTH_V1) as u64;
    if ledger.correctness_stack_calls != expected_correctness_stacks
        || ledger.correctness_layer_transactions != expected_correctness_layers
        || ledger.rejected_validation_calls != 4
        || ledger.metal_new_row_snapshot_calls != expected_correctness_layers[1]
        || ledger.metal_prefix_snapshot_calls != expected_prefix_snapshots
        || ledger.metal_prefix_snapshot_elements_compared_to_bits != expected_prefix_elements
    {
        return Err("correctness ledger mismatch".to_owned());
    }
    if performance_completed {
        if ledger.warmup_stack_calls != [256, 256]
            || ledger.timed_stack_calls != [256, 256]
            || ledger.warmup_layer_transactions != [1536, 1536]
            || ledger.timed_layer_transactions != [1536, 1536]
            || ledger.prefix_restores != [528, 528]
        {
            return Err("completed performance ledger mismatch".to_owned());
        }
    } else if ledger.prefix_restores != [16, 16]
        || ledger.warmup_stack_calls != [0, 0]
        || ledger.timed_stack_calls != [0, 0]
        || ledger.warmup_layer_transactions != [0, 0]
        || ledger.timed_layer_transactions != [0, 0]
    {
        return Err("partial performance ledger cannot close".to_owned());
    }
    Ok(())
}

fn ledger_json(ledger: &Ledger) -> Value {
    json!({
        "correctness_stack_calls": {
            "A_cpu_F32": ledger.correctness_stack_calls[0],
            "B_metal_W8": ledger.correctness_stack_calls[1],
            "packed_W8_oracle": ledger.correctness_stack_calls[2]
        },
        "correctness_layer_transactions": {
            "A_cpu_F32": ledger.correctness_layer_transactions[0],
            "B_metal_W8": ledger.correctness_layer_transactions[1],
            "packed_W8_oracle": ledger.correctness_layer_transactions[2]
        },
        "prefix_restores_outside_timing": {
            "A": ledger.prefix_restores[0],
            "B": ledger.prefix_restores[1]
        },
        "warmup_stack_calls": {"A":ledger.warmup_stack_calls[0],"B":ledger.warmup_stack_calls[1]},
        "timed_stack_calls": {"A":ledger.timed_stack_calls[0],"B":ledger.timed_stack_calls[1]},
        "warmup_layer_transactions": {"A":ledger.warmup_layer_transactions[0],"B":ledger.warmup_layer_transactions[1]},
        "timed_layer_transactions": {"A":ledger.timed_layer_transactions[0],"B":ledger.timed_layer_transactions[1]},
        "metal_new_row_snapshot_calls": ledger.metal_new_row_snapshot_calls,
        "metal_prefix_snapshot_calls": ledger.metal_prefix_snapshot_calls,
        "metal_prefix_snapshot_elements_compared_to_bits": ledger.metal_prefix_snapshot_elements_compared_to_bits,
        "rejected_validation_calls": ledger.rejected_validation_calls
    })
}

fn validate_runtime_receipt(
    receipt: &FullAttentionStack6RuntimeReceiptV1,
    successful_decodes: u64,
) -> Result<(), Box<dyn Error>> {
    let observed = u32::from(successful_decodes != 0);
    let last_identity_valid = if successful_decodes == 0 {
        receipt.last_layer_slot == u32::MAX
            && receipt.last_start_pos == u32::MAX
            && receipt.last_kv_length == 0
    } else {
        receipt.last_layer_slot < QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 as u32
            && receipt.last_start_pos < MAX_CONTEXT as u32
            && receipt.last_kv_length == receipt.last_start_pos + 1
    };
    if receipt.layer_slots != 6
        || receipt.hidden_size != 1024
        || receipt.query_heads != 8
        || receipt.kv_heads != 2
        || receipt.head_dim != 256
        || receipt.rotary_dim != 64
        || receipt.max_context != MAX_CONTEXT as u32
        || receipt.group_size != W8_GROUP_SIZE as u32
        || receipt.command_buffers_per_decode != 1
        || receipt.compute_encoders_per_decode != 1
        || receipt.kernel_dispatches_per_decode != 5
        || receipt.explicit_buffer_barriers_per_decode != 4
        || receipt.commits_per_decode != 1
        || receipt.waits_per_decode != 1
        || !receipt.fixed_shape_validated
        || receipt.successful_decodes != successful_decodes
        || !last_identity_valid
        || receipt.last_observed_command_buffers != observed
        || receipt.last_observed_compute_encoders != observed
        || receipt.last_observed_kernel_dispatches != 5 * observed
        || receipt.last_observed_explicit_buffer_barriers != 4 * observed
        || receipt.last_observed_commits != observed
        || receipt.last_observed_waits != observed
    {
        return Err(format!("invalid runtime receipt: {receipt:?}").into());
    }
    Ok(())
}

fn runtime_receipt_json(receipt: &FullAttentionStack6RuntimeReceiptV1) -> Value {
    json!({
        "layer_slots": receipt.layer_slots,
        "hidden_size": receipt.hidden_size,
        "query_heads": receipt.query_heads,
        "kv_heads": receipt.kv_heads,
        "head_dim": receipt.head_dim,
        "rotary_dim": receipt.rotary_dim,
        "max_context": receipt.max_context,
        "group_size": receipt.group_size,
        "per_single_layer_decode": {
            "command_buffers": receipt.command_buffers_per_decode,
            "compute_encoders": receipt.compute_encoders_per_decode,
            "kernel_dispatches": receipt.kernel_dispatches_per_decode,
            "explicit_buffer_barriers": receipt.explicit_buffer_barriers_per_decode,
            "commits": receipt.commits_per_decode,
            "waits": receipt.waits_per_decode
        },
        "fixed_shape_validated": receipt.fixed_shape_validated,
        "successful_decodes": receipt.successful_decodes,
        "last_success": {
            "layer_slot": receipt.last_layer_slot,
            "start_pos": receipt.last_start_pos,
            "kv_length": receipt.last_kv_length,
            "observed_command_buffers": receipt.last_observed_command_buffers,
            "observed_compute_encoders": receipt.last_observed_compute_encoders,
            "observed_kernel_dispatches": receipt.last_observed_kernel_dispatches,
            "observed_explicit_buffer_barriers": receipt.last_observed_explicit_buffer_barriers,
            "observed_commits": receipt.last_observed_commits,
            "observed_waits": receipt.last_observed_waits
        }
    })
}

fn hash_owned_weights(layers: &[OwnedLayerF32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"qwen35-full-attention-count6-f32-weights-v1");
    for layer in layers {
        for (label, values) in [
            (b"input_norm".as_slice(), layer.input_rms_weight.as_slice()),
            (b"query".as_slice(), layer.query_rows.as_slice()),
            (b"gate".as_slice(), layer.gate_rows.as_slice()),
            (b"key".as_slice(), layer.key_rows.as_slice()),
            (b"value".as_slice(), layer.value_rows.as_slice()),
            (b"query_norm".as_slice(), layer.query_norm_weight.as_slice()),
            (b"key_norm".as_slice(), layer.key_norm_weight.as_slice()),
            (b"output".as_slice(), layer.output_rows.as_slice()),
        ] {
            hash_f32(&mut hasher, label, values);
        }
    }
    format!("{:x}", hasher.finalize())
}

fn hash_performance_fixture(prefix: &PrefixFixture, inputs: &[StackInputs]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"qwen35-full-attention-count6-performance-fixture-v1");
    for aggregate_inputs in inputs {
        for input in aggregate_inputs {
            hash_f32(&mut hasher, b"slot_input", input);
        }
    }
    for layer in 0..QWEN35_FULL_ATTENTION_LAYER_SLOTS_V1 {
        hash_f32(&mut hasher, b"prefix_key", &prefix.keys[layer]);
        hash_f32(&mut hasher, b"prefix_value", &prefix.values[layer]);
    }
    format!("{:x}", hasher.finalize())
}

fn hash_f32(hasher: &mut Sha256, label: &[u8], values: &[f32]) {
    hasher.update(label);
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hash_f32(&mut hasher, b"f32", values);
    format!("{:x}", hasher.finalize())
}

fn git_custody(
    workspace: &Path,
    candidate_commit: &str,
    allowed_untracked: Option<&str>,
) -> Result<Value, Box<dyn Error>> {
    let git = |arguments: &[&str]| command_output_in("git", arguments, workspace);
    let head = git(&["rev-parse", "HEAD"])?;
    let origin_main = git(&["rev-parse", "origin/main"])?;
    let remote_main_line = git(&["ls-remote", "--heads", "origin", "refs/heads/main"])?;
    let remote_main = remote_main_line
        .split_whitespace()
        .next()
        .ok_or("origin main is absent")?;
    let origin_url = git(&["remote", "get-url", "origin"])?;
    let branch = git(&["branch", "--show-current"])?;
    let status = git(&["status", "--porcelain=v1", "--untracked-files=all"])?;
    let expected_status = allowed_untracked
        .map(|path| format!("?? {path}"))
        .unwrap_or_default();
    let parents = git(&["rev-list", "--parents", "-n", "1", candidate_commit])?;
    let parent_fields = parents.split_whitespace().collect::<Vec<_>>();
    let mut changed_paths = git(&[
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        candidate_commit,
    ])?
    .lines()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    changed_paths.sort_unstable();
    if head != candidate_commit
        || origin_main != candidate_commit
        || remote_main != candidate_commit
        || origin_url != EXPECTED_ORIGIN_URL
        || branch != "main"
        || status != expected_status
        || parent_fields.first().copied() != Some(candidate_commit)
        || parent_fields.len() != 2
        || parent_fields.get(1).copied() != Some(BASELINE_PARENT_COMMIT)
        || changed_paths.iter().map(String::as_str).collect::<Vec<_>>()
            != EXPECTED_CANDIDATE_CHANGED_PATHS
    {
        return Err(format!(
            "git custody mismatch: head={head} origin/main={origin_main} remote/main={remote_main} origin={origin_url} branch={branch} status={status:?} parents={parent_fields:?}"
        )
        .into());
    }
    Ok(json!({
        "head": head,
        "origin_main": origin_main,
        "github_main": remote_main,
        "origin_url": origin_url,
        "branch": branch,
        "worktree_clean_before_attempt": allowed_untracked.is_none(),
        "worktree_status": status,
        "sole_allowed_attempt_artifact": allowed_untracked,
        "candidate_is_non_merge_commit": true,
        "candidate_parent": parent_fields[1],
        "candidate_changed_paths": changed_paths
    }))
}

fn custody_snapshot(
    manifest_dir: &Path,
    workspace: &Path,
    executable: &Path,
) -> Result<Value, Box<dyn Error>> {
    let metal = workspace.join("crates/apxinf-metal");
    let core = workspace.join("crates/apxinf-core");
    let files = [
        ("gate_example", manifest_dir.join("examples/qwen35_full_attention_decode_count6_ab_v1.rs")),
        ("metal_rust_module", metal.join("src/full_attention_decode_v1.rs")),
        ("metal_shader", metal.join("src/metal_full_attention_decode_v1.metal")),
        ("metal_bridge", metal.join("src/metal_full_attention_decode_v1_bridge.mm")),
        ("metal_build", metal.join("build.rs")),
        ("metal_lib", metal.join("src/lib.rs")),
        ("metal_manifest", metal.join("Cargo.toml")),
        (
            "metal_live_test",
            metal.join("tests/full_attention_decode_v1.rs"),
        ),
        ("cpu_backend", core.join("src/op_impls/cpu.rs")),
        ("backend_trait", core.join("src/backend.rs")),
        ("kv_cache", core.join("src/kv_cache.rs")),
        ("cpu_blas", core.join("src/ops.rs")),
        ("core_build", core.join("build.rs")),
        ("core_manifest", core.join("Cargo.toml")),
        ("model_manifest", manifest_dir.join("Cargo.toml")),
        ("qwen_production_semantics", manifest_dir.join("src/qwen35/general.rs")),
        ("predeclaration", metal.join("evidence/next-hotspot/qwen35-full-attention-decode-v1-count6-predeclared-primitive-gate-v1-20260826.json")),
        ("workspace_lock", workspace.join("Cargo.lock")),
        ("workspace_manifest", workspace.join("Cargo.toml")),
    ];
    let mut identities = serde_json::Map::new();
    for (label, path) in files {
        identities.insert(label.to_owned(), file_identity(&path)?);
    }
    Ok(json!({
        "binary": file_identity(executable)?,
        "sources": identities,
        "embedded_source_sha256": embedded_source_sha256()
    }))
}

fn embedded_source_sha256() -> Value {
    json!({
        "gate_example": sha256_bytes(include_bytes!("qwen35_full_attention_decode_count6_ab_v1.rs")),
        "metal_rust_module": sha256_bytes(include_bytes!("../../apxinf-metal/src/full_attention_decode_v1.rs")),
        "metal_shader": sha256_bytes(include_bytes!("../../apxinf-metal/src/metal_full_attention_decode_v1.metal")),
        "metal_bridge": sha256_bytes(include_bytes!("../../apxinf-metal/src/metal_full_attention_decode_v1_bridge.mm")),
        "metal_build": sha256_bytes(include_bytes!("../../apxinf-metal/build.rs")),
        "metal_lib": sha256_bytes(include_bytes!("../../apxinf-metal/src/lib.rs")),
        "metal_manifest": sha256_bytes(include_bytes!("../../apxinf-metal/Cargo.toml")),
        "metal_live_test": sha256_bytes(include_bytes!("../../apxinf-metal/tests/full_attention_decode_v1.rs")),
        "cpu_backend": sha256_bytes(include_bytes!("../../apxinf-core/src/op_impls/cpu.rs")),
        "backend_trait": sha256_bytes(include_bytes!("../../apxinf-core/src/backend.rs")),
        "kv_cache": sha256_bytes(include_bytes!("../../apxinf-core/src/kv_cache.rs")),
        "cpu_blas": sha256_bytes(include_bytes!("../../apxinf-core/src/ops.rs")),
        "core_build": sha256_bytes(include_bytes!("../../apxinf-core/build.rs")),
        "core_manifest": sha256_bytes(include_bytes!("../../apxinf-core/Cargo.toml")),
        "model_manifest": sha256_bytes(include_bytes!("../Cargo.toml")),
        "qwen_production_semantics": sha256_bytes(include_bytes!("../src/qwen35/general.rs")),
        "predeclaration": sha256_bytes(include_bytes!("../../apxinf-metal/evidence/next-hotspot/qwen35-full-attention-decode-v1-count6-predeclared-primitive-gate-v1-20260826.json")),
        "workspace_lock": sha256_bytes(include_bytes!("../../../Cargo.lock")),
        "workspace_manifest": sha256_bytes(include_bytes!("../../../Cargo.toml"))
    })
}

fn require_disk_matches_embedded(snapshot: &Value) -> Result<(), Box<dyn Error>> {
    let embedded = snapshot
        .get("embedded_source_sha256")
        .and_then(Value::as_object)
        .ok_or("missing embedded source hashes")?;
    let sources = snapshot
        .get("sources")
        .and_then(Value::as_object)
        .ok_or("missing source identities")?;
    for (label, expected) in embedded {
        let expected = expected.as_str().ok_or("embedded hash is not a string")?;
        let actual = sources
            .get(label)
            .and_then(|value| value.get("sha256"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("missing disk hash for {label}"))?;
        if actual != expected {
            return Err(format!("disk/embedded source mismatch for {label}").into());
        }
    }
    Ok(())
}

fn file_identity(path: &Path) -> Result<Value, Box<dyn Error>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "custody path is not a regular direct file: {}",
            path.display()
        )
        .into());
    }
    let bytes = std::fs::read(path)?;
    Ok(json!({
        "path": std::fs::canonicalize(path)?,
        "bytes": metadata.len(),
        "sha256": sha256_bytes(&bytes),
        "regular_direct_file": true
    }))
}

fn collect_host_check(candidate_commit: &str, phase: &str) -> Result<Value, Box<dyn Error>> {
    if phase != "preflight" && phase != "postflight" {
        return Err(format!("invalid formal host-check phase {phase}").into());
    }
    let diagnostic = |program: &str, arguments: &[&str]| {
        command_output(program, arguments).unwrap_or_else(|error| format!("unavailable: {error}"))
    };
    let hardware_model = command_output("sysctl", &["-n", "hw.model"])?;
    let cpu_brand = command_output("sysctl", &["-n", "machdep.cpu.brand_string"])?;
    let target_host_valid =
        hardware_model == EXPECTED_HARDWARE_MODEL && cpu_brand == EXPECTED_CPU_BRAND;
    let thermal_status = command_output("pmset", &["-g", "therm"])?;
    let thermal_valid = thermal_status.contains("No thermal warning level has been recorded")
        && thermal_status.contains("No performance warning level has been recorded");

    let mut process_samples = Vec::with_capacity(QUIET_HOST_SNAPSHOTS);
    let mut quiet_host_valid = true;
    for sample_index in 0..QUIET_HOST_SNAPSHOTS {
        let raw = command_output("ps", &["-Ao", "pid=,pcpu=,comm="])?;
        let mut processes = Vec::new();
        for line in raw.lines() {
            let mut fields = line.split_whitespace();
            let Some(pid_text) = fields.next() else {
                continue;
            };
            let Some(cpu_text) = fields.next() else {
                continue;
            };
            let pid: u32 = pid_text.parse()?;
            let cpu_percent: f64 = cpu_text.parse()?;
            let command = fields.collect::<Vec<_>>().join(" ");
            if pid == std::process::id() || command == "ps" || command.ends_with("/ps") {
                continue;
            }
            if !cpu_percent.is_finite() || cpu_percent < 0.0 {
                return Err(format!("ps returned invalid CPU percentage {cpu_text}").into());
            }
            processes.push((pid, cpu_percent, command));
        }
        processes.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let aggregate_cpu_percent = processes.iter().map(|process| process.1).sum::<f64>();
        let max_process_cpu_percent = processes.first().map(|process| process.1).unwrap_or(0.0);
        let sample_passed = max_process_cpu_percent <= MAX_OTHER_PROCESS_CPU_PERCENT
            && aggregate_cpu_percent <= MAX_AGGREGATE_OTHER_CPU_PERCENT;
        quiet_host_valid &= sample_passed;
        process_samples.push(json!({
            "sample_index": sample_index,
            "passed": sample_passed,
            "aggregate_other_process_cpu_percent": aggregate_cpu_percent,
            "max_other_process_cpu_percent": max_process_cpu_percent,
            "top_other_processes": processes.iter().take(12).map(|(pid, cpu_percent, command)| json!({
                "pid": pid,
                "cpu_percent": cpu_percent,
                "command": command
            })).collect::<Vec<_>>()
        }));
        if sample_index + 1 != QUIET_HOST_SNAPSHOTS {
            std::thread::sleep(Duration::from_millis(QUIET_HOST_INTERVAL_MS));
        }
    }

    let passed = target_host_valid && thermal_valid && quiet_host_valid;
    let preflight = json!({
        "classification": "fail-closed target and quiet-host check; no process was terminated",
        "phase": phase,
        "passed": passed,
        "target_host_valid": target_host_valid,
        "quiet_host_attested": quiet_host_valid,
        "thermal_status_valid": thermal_valid,
        "hardware_model": hardware_model,
        "expected_hardware_model": EXPECTED_HARDWARE_MODEL,
        "cpu_brand": cpu_brand,
        "expected_cpu_brand": EXPECTED_CPU_BRAND,
        "quiet_host_contract": {
            "snapshots": QUIET_HOST_SNAPSHOTS,
            "interval_ms": QUIET_HOST_INTERVAL_MS,
            "max_any_other_process_cpu_percent": MAX_OTHER_PROCESS_CPU_PERCENT,
            "max_aggregate_other_process_cpu_percent": MAX_AGGREGATE_OTHER_CPU_PERCENT,
            "all_snapshots_must_pass": true,
            "process_cpu_percent_unit": "one logical CPU core equals 100 percent"
        },
        "process_samples": process_samples,
        "thermal_status": thermal_status,
        "os_version": diagnostic("sw_vers", &["-productVersion"]),
        "os_build": diagnostic("sw_vers", &["-buildVersion"]),
        "rustc": diagnostic("rustc", &["--version"]),
        "cargo": diagnostic("cargo", &["--version"]),
        "clang": diagnostic("xcrun", &["clang","--version"]),
        "metal": diagnostic("xcrun", &["metal","--version"]),
        "uptime": diagnostic("uptime", &[]),
        "expected_build": format!("APXINF_CANDIDATE_COMMIT={candidate_commit} cargo build --release -p apxinf-model --example qwen35_full_attention_decode_count6_ab_v1 --features accelerate,metal-w8"),
        "expected_run": format!("target/release/examples/qwen35_full_attention_decode_count6_ab_v1 --candidate-commit {candidate_commit}")
    });
    Ok(preflight)
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program)
        .args(arguments)
        .env("LC_ALL", "C")
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn command_output_in(
    program: &str,
    arguments: &[&str],
    directory: &Path,
) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program)
        .args(arguments)
        .env("LC_ALL", "C")
        .current_dir(directory)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn reserve_attempt(path: &Path, marker: &Value) -> Result<File, Box<dyn Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer(&mut file, marker)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    File::open(path.parent().ok_or("raw receipt has no parent directory")?)?.sync_all()?;
    Ok(file)
}

fn publish_reserved(marker_file: File, path: &Path, receipt: &Value) -> Result<(), Box<dyn Error>> {
    let mut bytes = serde_json::to_vec(receipt)?;
    bytes.push(b'\n');
    let finalizing_path = path.with_extension("json.finalizing");
    let mut finalizing = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&finalizing_path)?;
    finalizing.write_all(&bytes)?;
    finalizing.sync_all()?;
    drop(finalizing);
    drop(marker_file);
    std::fs::rename(&finalizing_path, path)?;
    File::open(path.parent().ok_or("raw receipt has no parent directory")?)?.sync_all()?;
    Ok(())
}

fn validate_commit(commit: &str) -> Result<(), Box<dyn Error>> {
    if commit.len() != 40 || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("--candidate-commit must be a full 40-character hexadecimal commit".into());
    }
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut candidate_commit = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(flag) = args.next() {
        let flag_text = flag.to_string_lossy();
        match flag_text.as_ref() {
            "--candidate-commit" => {
                candidate_commit = Some(
                    args.next()
                        .ok_or("--candidate-commit requires a value")?
                        .to_string_lossy()
                        .into_owned(),
                )
            }
            _ => return Err(format!("unknown argument {flag_text}").into()),
        }
    }
    Ok(Args {
        candidate_commit: candidate_commit.ok_or("--candidate-commit is required")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed_correctness_ledger() -> Ledger {
        Ledger {
            correctness_stack_calls: [16, 16, 8],
            correctness_layer_transactions: [96, 96, 48],
            prefix_restores: [16, 16],
            metal_new_row_snapshot_calls: 96,
            metal_prefix_snapshot_calls: 9_216,
            metal_prefix_snapshot_elements_compared_to_bits: 9_437_184,
            rejected_validation_calls: 4,
            ..Ledger::default()
        }
    }

    #[test]
    fn low_bit_domain_tags_do_not_alias_deterministic_channels() {
        let query = deterministic_dyadic(256, WEIGHT_DOMAIN ^ 0x02, 32, 2048.0);
        let gate = deterministic_dyadic(256, WEIGHT_DOMAIN ^ 0x03, 32, 2048.0);
        let key = deterministic_dyadic(256, WEIGHT_DOMAIN ^ 0x04, 32, 2048.0);
        let value = deterministic_dyadic(256, WEIGHT_DOMAIN ^ 0x05, 32, 2048.0);
        assert_ne!(query, gate);
        assert_ne!(key, value);
    }

    #[test]
    fn correctness_and_completed_performance_ledgers_close_exactly() {
        let mut ledger = closed_correctness_ledger();
        assert!(validate_ledger(&ledger, false).is_ok());
        ledger.metal_prefix_snapshot_calls -= 1;
        assert!(validate_ledger(&ledger, false).is_err());

        let mut ledger = closed_correctness_ledger();
        ledger.prefix_restores = [528, 528];
        ledger.warmup_stack_calls = [256, 256];
        ledger.timed_stack_calls = [256, 256];
        ledger.warmup_layer_transactions = [1536, 1536];
        ledger.timed_layer_transactions = [1536, 1536];
        assert!(validate_ledger(&ledger, true).is_ok());
    }
}

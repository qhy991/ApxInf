//! Predeclared same-binary count-43 RMSNorm A/B for Qwen3.5-0.8B.
//!
//! This is a non-formal aggregate mechanism screen, not production submission
//! topology, an end-to-end model benchmark, or a cross-runtime comparison.

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use apxinf_metal::{
    MetalRmsNormCount43PrimitiveV1, RmsNormCount43RuntimeReceiptV1, RmsNormReductionProfileV1,
    QWEN35_RMS_CALLS_PER_DECODE_V1, QWEN35_RMS_HIDDEN_SIZE_V1,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FORMAT: &str = "apxinf-qwen35-rms-simd-tail-exact-primitive-ab-v1";
const EPSILON: f32 = 1.0e-6;
const CORRECTNESS_INPUTS: usize = 8;
const CALLS_PER_CELL: usize = 64;
const KEEP_THRESHOLD_PERCENT: f64 = 5.0;
const EXPECTED_ORIGIN_URL: &str = "https://github.com/qhy991/ApxInf.git";
const EMBEDDED_CANDIDATE_COMMIT: Option<&str> = option_env!("APXINF_CANDIDATE_COMMIT");

#[derive(Clone, Copy)]
enum Arm {
    A,
    B,
}

impl Arm {
    const fn label(self) -> &'static str {
        match self {
            Self::A => "A_legacy_shared_tree",
            Self::B => "B_exact_redundant_simd_tail",
        }
    }

    const fn profile(self) -> RmsNormReductionProfileV1 {
        match self {
            Self::A => RmsNormReductionProfileV1::LegacySharedTree,
            Self::B => RmsNormReductionProfileV1::ExactRedundantSimdTail,
        }
    }
}

struct Args {
    output: PathBuf,
    candidate_commit: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if cfg!(debug_assertions) {
        return Err("RMSNorm primitive gate must be built in release mode".into());
    }
    if args.output.exists() {
        return Err(format!(
            "refusing to overwrite existing receipt {}",
            args.output.display()
        )
        .into());
    }
    if args.candidate_commit.len() != 40
        || !args
            .candidate_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("--candidate-commit must be a full 40-character hexadecimal commit".into());
    }
    let embedded_candidate_commit = EMBEDDED_CANDIDATE_COMMIT
        .ok_or("release gate was not built with APXINF_CANDIDATE_COMMIT")?;
    if embedded_candidate_commit != args.candidate_commit {
        return Err(format!(
            "embedded candidate commit {embedded_candidate_commit} does not match requested {}",
            args.candidate_commit
        )
        .into());
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("apxinf-metal manifest is not below the workspace root")?;
    let executable = std::env::current_exe()?;
    let git_start = git_custody(workspace_dir, &args.candidate_commit)?;
    let custody_start = custody_snapshot(manifest_dir, &executable)?;
    require_disk_sources_match_embedded(&custody_start)?;
    let host_preflight = host_preflight();

    let weight = seeded_weight(QWEN35_RMS_HIDDEN_SIZE_V1, 0x243f_6a88_85a3_08d3);
    let inputs = (0..CORRECTNESS_INPUTS)
        .map(|index| {
            seeded_input(
                QWEN35_RMS_HIDDEN_SIZE_V1,
                0x1319_8a2e_0370_7344 ^ index as u64,
            )
        })
        .collect::<Vec<_>>();
    let fixture_sha256 = hash_fixture(&weight, &inputs);
    let mut primitive = MetalRmsNormCount43PrimitiveV1::new(&weight, EPSILON)?;
    let initial_a = primitive.runtime_receipt(Arm::A.profile())?;
    let initial_b = primitive.runtime_receipt(Arm::B.profile())?;
    validate_runtime_receipt(&initial_a, Arm::A.profile(), Some(0))?;
    validate_runtime_receipt(&initial_b, Arm::B.profile(), Some(0))?;

    let mut sampled_attempt_failures = Vec::new();
    let exactness = match exactness_check(&mut primitive, &inputs) {
        Ok(exactness) => exactness,
        Err(error) => {
            sampled_attempt_failures.push(format!("exactness execution: {error}"));
            json!({
                "passed": false,
                "performance_executed": false,
                "harness_error": error.to_string()
            })
        }
    };
    let performance = if exactness.get("passed").and_then(Value::as_bool) == Some(true) {
        let performance_result = (|| -> Result<Value, Box<dyn Error>> {
            primitive.stage_input(&inputs[0])?;
            warmup_block([Arm::A, Arm::B, Arm::B, Arm::A], &mut primitive)?;
            warmup_block([Arm::B, Arm::A, Arm::A, Arm::B], &mut primitive)?;
            let block_1 = timed_block([Arm::A, Arm::B, Arm::B, Arm::A], &mut primitive)?;
            let block_2 = timed_block([Arm::B, Arm::A, Arm::A, Arm::B], &mut primitive)?;
            Ok(performance_json(block_1, block_2))
        })();
        match performance_result {
            Ok(performance) => Some(performance),
            Err(error) => {
                sampled_attempt_failures.push(format!("performance execution: {error}"));
                None
            }
        }
    } else {
        None
    };

    let expected_runs = performance
        .as_ref()
        .map(|_| (CORRECTNESS_INPUTS * 2 + CALLS_PER_CELL * 8) as u64);
    let (final_a, final_a_valid) = capture_final_receipt(
        &primitive,
        Arm::A,
        expected_runs,
        &mut sampled_attempt_failures,
    );
    let (final_b, final_b_valid) = capture_final_receipt(
        &primitive,
        Arm::B,
        expected_runs,
        &mut sampled_attempt_failures,
    );
    let performance_passed = performance
        .as_ref()
        .and_then(|value| value.get("passed"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let (custody_end, custody_end_valid) = match custody_snapshot(manifest_dir, &executable) {
        Ok(snapshot) => match require_disk_sources_match_embedded(&snapshot) {
            Ok(()) => (snapshot, true),
            Err(error) => {
                sampled_attempt_failures.push(format!("end embedded-source custody: {error}"));
                (snapshot, false)
            }
        },
        Err(error) => {
            sampled_attempt_failures.push(format!("end source custody snapshot: {error}"));
            (json!({"error": error.to_string()}), false)
        }
    };
    let (git_end, git_end_valid) = match git_custody(workspace_dir, &args.candidate_commit) {
        Ok(snapshot) => (snapshot, true),
        Err(error) => {
            sampled_attempt_failures.push(format!("end git custody: {error}"));
            (json!({"error": error.to_string()}), false)
        }
    };
    let custody_unchanged =
        custody_end_valid && git_end_valid && custody_start == custody_end && git_start == git_end;
    let runtime_receipts_valid = final_a_valid && final_b_valid;
    let primitive_continue_gate_passed = performance_passed
        && runtime_receipts_valid
        && custody_unchanged
        && sampled_attempt_failures.is_empty();
    let receipt = json!({
        "format": FORMAT,
        "classification": "non-formal count-matched aggregate mechanism screen; not production submission topology, end-to-end inference, or a cross-runtime benchmark",
        "candidate_commit": args.candidate_commit,
        "embedded_candidate_commit": embedded_candidate_commit,
        "scope": {
            "hidden_size": QWEN35_RMS_HIDDEN_SIZE_V1,
            "rms_calls_per_aggregate_run": QWEN35_RMS_CALLS_PER_DECODE_V1,
            "rms_epsilon": EPSILON,
            "same_binary_and_live_library": true,
            "input_staging_and_correctness_snapshots_outside_timing": true,
            "production_submission_topology": false,
            "active_production_distribution": "seven command buffers and 24 encoders"
        },
        "source_call_count_breakdown": {
            "linear_attention_layers": 18,
            "input_and_post_attention_rms_per_linear_layer": 2,
            "body_full_attention_boundaries": 5,
            "tail_post_attention_and_final_rms": 2,
            "total": 43
        },
        "source_derived_tradeoff": {
            "A_internal_threadgroup_barriers_per_dispatch": 9,
            "B_internal_threadgroup_barriers_per_dispatch": 4,
            "A_internal_threadgroup_barriers_per_aggregate_run": 387,
            "B_internal_threadgroup_barriers_per_aggregate_run": 172,
            "removed_internal_threadgroup_barriers_per_aggregate_run": 215,
            "candidate_extra_tail_adds_per_dispatch": 217,
            "candidate_extra_tail_adds_per_aggregate_run": 9331,
            "measured_hardware_barrier_counter": false
        },
        "fixture": {
            "generator": "one deterministic dyadic xorshift64 weight stream and eight distinct deterministic dyadic xorshift64 input streams",
            "sha256_f32_le_with_shape_and_epsilon": fixture_sha256,
            "weight_vectors": 1,
            "correctness_inputs": CORRECTNESS_INPUTS
        },
        "runtime_receipts": {
            "A": final_a,
            "B": final_b,
            "valid": runtime_receipts_valid
        },
        "exactness": exactness,
        "performance": performance,
        "admission": {
            "pooled_continue_threshold_percent": KEEP_THRESHOLD_PERCENT,
            "both_counterbalanced_blocks_must_be_positive": true,
            "clearly_negative_block_reject_percent": -0.5,
            "pass_only_authorizes_opt_in_full_path_plumbing": true,
            "no_resampling_after_failure": true
        },
        "host_preflight": host_preflight,
        "custody": {
            "start": custody_start,
            "end": custody_end,
            "git_start": git_start,
            "git_end": git_end,
            "unchanged_during_sampling": custody_unchanged
        },
        "performance_threshold_passed": performance_passed,
        "sampled_attempt_failures": sampled_attempt_failures,
        "primitive_continue_gate_passed": primitive_continue_gate_passed,
        "formal_admission_passed": false,
        "screen_passed": primitive_continue_gate_passed,
        "passed": primitive_continue_gate_passed
    });
    publish_create_new(&args.output, &receipt)?;
    println!("{}", serde_json::to_string(&receipt)?);
    if !primitive_continue_gate_passed {
        return Err("exact SIMD-tail RMSNorm primitive rejected; receipt was published".into());
    }
    Ok(())
}

fn next_xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn seeded_weight(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            let signed = ((next_xorshift(&mut state) >> 32) % 2001) as i32 - 1000;
            1.0 + signed as f32 / 8192.0
        })
        .collect()
}

fn seeded_input(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            let signed = ((next_xorshift(&mut state) >> 32) % 2001) as i32 - 1000;
            signed as f32 / 1024.0
        })
        .collect()
}

fn hash_fixture(weight: &[f32], inputs: &[Vec<f32>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"qwen35-rms-simd-tail-exact-fixture-v1");
    hasher.update((QWEN35_RMS_HIDDEN_SIZE_V1 as u64).to_le_bytes());
    hasher.update((QWEN35_RMS_CALLS_PER_DECODE_V1 as u64).to_le_bytes());
    hasher.update(EPSILON.to_bits().to_le_bytes());
    hasher.update(b"weight");
    for value in weight {
        hasher.update(value.to_bits().to_le_bytes());
    }
    for (index, input) in inputs.iter().enumerate() {
        hasher.update(b"input");
        hasher.update((index as u64).to_le_bytes());
        for value in input {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn run_arm(arm: Arm, primitive: &mut MetalRmsNormCount43PrimitiveV1) -> Result<(), Box<dyn Error>> {
    primitive.run(arm.profile())?;
    std::hint::black_box(arm.label());
    Ok(())
}

fn exactness_check(
    primitive: &mut MetalRmsNormCount43PrimitiveV1,
    inputs: &[Vec<f32>],
) -> Result<Value, Box<dyn Error>> {
    let order = [Arm::A, Arm::B, Arm::B, Arm::A];
    let mut compared_elements = 0usize;
    let mut finite_checks = 0usize;
    for (input_index, input) in inputs.iter().enumerate() {
        primitive.stage_input(input)?;
        let mut outputs = Vec::with_capacity(order.len());
        for arm in order {
            run_arm(arm, primitive)?;
            outputs.push(primitive.snapshot_chain()?);
        }
        for (call_index, output) in outputs.iter().enumerate() {
            for (element, &actual) in output.iter().enumerate() {
                finite_checks += 1;
                if !actual.is_finite() {
                    return Ok(json!({
                        "passed": false,
                        "performance_executed": false,
                        "order_per_input": order.map(Arm::label),
                        "finite_checks_before_failure": finite_checks,
                        "compared_elements_before_failure": compared_elements,
                        "first_mismatch": {
                            "kind": "non_finite",
                            "input_index": input_index,
                            "call_index": call_index,
                            "arm": order[call_index].label(),
                            "chain_row": element / QWEN35_RMS_HIDDEN_SIZE_V1,
                            "element": element % QWEN35_RMS_HIDDEN_SIZE_V1,
                            "actual_value": actual,
                            "actual_bits": actual.to_bits()
                        }
                    }));
                }
            }
        }
        for (call_index, output) in outputs.iter().enumerate().skip(1) {
            for (element, (&expected, &actual)) in outputs[0].iter().zip(output).enumerate() {
                compared_elements += 1;
                if expected.to_bits() != actual.to_bits() {
                    return Ok(json!({
                        "passed": false,
                        "performance_executed": false,
                        "order_per_input": order.map(Arm::label),
                        "finite_checks_before_failure": finite_checks,
                        "compared_elements_before_failure": compared_elements,
                        "first_mismatch": {
                            "kind": "to_bits",
                            "input_index": input_index,
                            "call_index": call_index,
                            "arm": order[call_index].label(),
                            "chain_row": element / QWEN35_RMS_HIDDEN_SIZE_V1,
                            "element": element % QWEN35_RMS_HIDDEN_SIZE_V1,
                            "expected_value": expected,
                            "actual_value": actual,
                            "expected_bits": expected.to_bits(),
                            "actual_bits": actual.to_bits()
                        }
                    }));
                }
            }
        }
    }
    Ok(json!({
        "passed": true,
        "performance_executed": true,
        "input_count": inputs.len(),
        "order_per_input": order.map(Arm::label),
        "chain_rows_per_call": QWEN35_RMS_CALLS_PER_DECODE_V1,
        "elements_per_chain_row": QWEN35_RMS_HIDDEN_SIZE_V1,
        "finite_checks": finite_checks,
        "compared_elements": compared_elements,
        "all_outputs_finite": true,
        "all_intermediate_rows_match_to_bits": true
    }))
}

fn warmup_block(
    order: [Arm; 4],
    primitive: &mut MetalRmsNormCount43PrimitiveV1,
) -> Result<(), Box<dyn Error>> {
    for arm in order {
        for _ in 0..CALLS_PER_CELL {
            run_arm(arm, primitive)?;
        }
    }
    Ok(())
}

fn timed_block(
    order: [Arm; 4],
    primitive: &mut MetalRmsNormCount43PrimitiveV1,
) -> Result<(Vec<u128>, Vec<u128>), Box<dyn Error>> {
    let mut a = Vec::with_capacity(CALLS_PER_CELL * 2);
    let mut b = Vec::with_capacity(CALLS_PER_CELL * 2);
    for arm in order {
        for _ in 0..CALLS_PER_CELL {
            let started = Instant::now();
            run_arm(arm, primitive)?;
            let elapsed = started.elapsed().as_nanos();
            match arm {
                Arm::A => a.push(elapsed),
                Arm::B => b.push(elapsed),
            }
        }
    }
    Ok((a, b))
}

fn performance_json(block_1: (Vec<u128>, Vec<u128>), block_2: (Vec<u128>, Vec<u128>)) -> Value {
    let block_1_improvement = improvement_percent(even_median(&block_1.0), even_median(&block_1.1));
    let block_2_improvement = improvement_percent(even_median(&block_2.0), even_median(&block_2.1));
    let mut pooled_a = block_1.0.clone();
    pooled_a.extend_from_slice(&block_2.0);
    let mut pooled_b = block_1.1.clone();
    pooled_b.extend_from_slice(&block_2.1);
    let pooled_improvement = improvement_percent(even_median(&pooled_a), even_median(&pooled_b));
    let passed = pooled_improvement >= KEEP_THRESHOLD_PERCENT
        && block_1_improvement > 0.0
        && block_2_improvement > 0.0
        && block_1_improvement > -0.5
        && block_2_improvement > -0.5;
    json!({
        "schedule": {
            "calls_per_cell": CALLS_PER_CELL,
            "rms_dispatches_per_call": QWEN35_RMS_CALLS_PER_DECODE_V1,
            "block_1_order": ["A", "B", "B", "A"],
            "block_2_order": ["B", "A", "A", "B"],
            "timed_calls_per_arm_total": pooled_a.len(),
            "timed_rms_dispatches_per_arm_total": pooled_a.len() * QWEN35_RMS_CALLS_PER_DECODE_V1,
            "warmup_used_the_same_two_blocks": true,
            "fixture_staged_outside_timing": true,
            "snapshots_excluded_from_timing": true
        },
        "block_1": samples_json(&block_1.0, &block_1.1, block_1_improvement),
        "block_2": samples_json(&block_2.0, &block_2.1, block_2_improvement),
        "pooled": samples_json(&pooled_a, &pooled_b, pooled_improvement),
        "passed": passed
    })
}

fn samples_json(a: &[u128], b: &[u128], improvement: f64) -> Value {
    json!({
        "A_raw_ns": a,
        "B_raw_ns": b,
        "A_median_ns": even_median(a),
        "B_median_ns": even_median(b),
        "A_p10_ns": percentile(a, 1, 10),
        "A_p90_ns": percentile(a, 9, 10),
        "B_p10_ns": percentile(b, 1, 10),
        "B_p90_ns": percentile(b, 9, 10),
        "B_improvement_percent": improvement
    })
}

fn sorted(samples: &[u128]) -> Vec<u128> {
    let mut values = samples.to_vec();
    values.sort_unstable();
    values
}

fn even_median(samples: &[u128]) -> f64 {
    let values = sorted(samples);
    let upper = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[upper - 1] as f64 + values[upper] as f64) / 2.0
    } else {
        values[upper] as f64
    }
}

fn percentile(samples: &[u128], numerator: usize, denominator: usize) -> u128 {
    let values = sorted(samples);
    values[(values.len() - 1) * numerator / denominator]
}

fn improvement_percent(a: f64, b: f64) -> f64 {
    (a - b) / a * 100.0
}

fn validate_runtime_receipt(
    receipt: &RmsNormCount43RuntimeReceiptV1,
    expected: RmsNormReductionProfileV1,
    expected_successful_runs: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    let expected_last = if receipt.successful_runs == 0 { 0 } else { 1 };
    if receipt.requested_profile != expected
        || receipt.observed_profile != expected
        || receipt.requested_function_name != expected.expected_function_name()
        || receipt.observed_function_name != expected.expected_function_name()
        || receipt.hidden_size != QWEN35_RMS_HIDDEN_SIZE_V1 as u32
        || receipt.rms_calls_per_run != QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
        || receipt.threads_per_threadgroup != 256
        || receipt.simdgroups_per_threadgroup != 8
        || receipt.pipeline_thread_execution_width != 32
        || receipt.pipeline_max_total_threads_per_threadgroup < 256
        || receipt.static_threadgroup_memory_bytes != 1024
        || receipt.dynamic_threadgroup_memory_bytes != 0
        || receipt.internal_threadgroup_barriers_per_dispatch
            != expected.internal_threadgroup_barriers_per_dispatch()
        || receipt.internal_threadgroup_barriers_per_run
            != expected.internal_threadgroup_barriers_per_dispatch()
                * QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
        || receipt.command_buffers_per_run != 1
        || receipt.compute_encoders_per_run != 1
        || receipt.kernel_dispatches_per_run != QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
        || receipt.explicit_buffer_barriers_per_run != QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
        || receipt.commits_per_run != 1
        || receipt.waits_per_run != 1
        || receipt.host_to_device_bytes_per_run != 0
        || receipt.device_to_host_bytes_per_run != 0
        || expected_successful_runs.is_some_and(|count| receipt.successful_runs != count)
        || receipt.last_observed_command_buffers != expected_last
        || receipt.last_observed_compute_encoders != expected_last
        || receipt.last_observed_kernel_dispatches
            != expected_last * QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
        || receipt.last_observed_explicit_buffer_barriers
            != expected_last * QWEN35_RMS_CALLS_PER_DECODE_V1 as u32
        || receipt.last_observed_commits != expected_last
        || receipt.last_observed_waits != expected_last
    {
        return Err(format!("invalid live runtime receipt for {expected:?}: {receipt:?}").into());
    }
    Ok(())
}

fn capture_final_receipt(
    primitive: &MetalRmsNormCount43PrimitiveV1,
    arm: Arm,
    expected_successful_runs: Option<u64>,
    sampled_attempt_failures: &mut Vec<String>,
) -> (Value, bool) {
    match primitive.runtime_receipt(arm.profile()) {
        Ok(receipt) => {
            match validate_runtime_receipt(&receipt, arm.profile(), expected_successful_runs) {
                Ok(()) => (runtime_receipt_json(&receipt), true),
                Err(error) => {
                    let message =
                        format!("final {} runtime receipt validation: {error}", arm.label());
                    sampled_attempt_failures.push(message.clone());
                    let mut value = runtime_receipt_json(&receipt);
                    value
                        .as_object_mut()
                        .expect("runtime receipt JSON is an object")
                        .insert("validation_error".to_owned(), Value::String(message));
                    (value, false)
                }
            }
        }
        Err(error) => {
            let message = format!("final {} runtime receipt read: {error}", arm.label());
            sampled_attempt_failures.push(message.clone());
            (json!({"available": false, "error": message}), false)
        }
    }
}

fn runtime_receipt_json(receipt: &RmsNormCount43RuntimeReceiptV1) -> Value {
    json!({
        "requested_profile": profile_label(receipt.requested_profile),
        "observed_profile": profile_label(receipt.observed_profile),
        "requested_function_name": receipt.requested_function_name,
        "observed_function_name": receipt.observed_function_name,
        "hidden_size": receipt.hidden_size,
        "rms_calls_per_run": receipt.rms_calls_per_run,
        "threads_per_threadgroup": receipt.threads_per_threadgroup,
        "simdgroups_per_threadgroup": receipt.simdgroups_per_threadgroup,
        "pipeline_max_total_threads_per_threadgroup": receipt.pipeline_max_total_threads_per_threadgroup,
        "pipeline_thread_execution_width": receipt.pipeline_thread_execution_width,
        "static_threadgroup_memory_bytes": receipt.static_threadgroup_memory_bytes,
        "dynamic_threadgroup_memory_bytes": receipt.dynamic_threadgroup_memory_bytes,
        "internal_threadgroup_barriers_per_dispatch": receipt.internal_threadgroup_barriers_per_dispatch,
        "internal_threadgroup_barriers_per_run": receipt.internal_threadgroup_barriers_per_run,
        "internal_barrier_count_is_source_derived": true,
        "command_buffers_per_run": receipt.command_buffers_per_run,
        "compute_encoders_per_run": receipt.compute_encoders_per_run,
        "kernel_dispatches_per_run": receipt.kernel_dispatches_per_run,
        "explicit_buffer_barriers_per_run": receipt.explicit_buffer_barriers_per_run,
        "commits_per_run": receipt.commits_per_run,
        "waits_per_run": receipt.waits_per_run,
        "host_to_device_bytes_per_run": receipt.host_to_device_bytes_per_run,
        "device_to_host_bytes_per_run": receipt.device_to_host_bytes_per_run,
        "successful_runs": receipt.successful_runs,
        "total_successful_kernel_dispatches": receipt.successful_runs * receipt.kernel_dispatches_per_run as u64,
        "last_observed_command_buffers": receipt.last_observed_command_buffers,
        "last_observed_compute_encoders": receipt.last_observed_compute_encoders,
        "last_observed_kernel_dispatches": receipt.last_observed_kernel_dispatches,
        "last_observed_explicit_buffer_barriers": receipt.last_observed_explicit_buffer_barriers,
        "last_observed_commits": receipt.last_observed_commits,
        "last_observed_waits": receipt.last_observed_waits
    })
}

fn profile_label(profile: RmsNormReductionProfileV1) -> &'static str {
    match profile {
        RmsNormReductionProfileV1::LegacySharedTree => "legacy-shared-tree",
        RmsNormReductionProfileV1::ExactRedundantSimdTail => "exact-redundant-simd-tail",
    }
}

fn git_custody(workspace_dir: &Path, candidate_commit: &str) -> Result<Value, Box<dyn Error>> {
    let git = |arguments: &[&str]| -> Result<String, Box<dyn Error>> {
        command_output_in("git", arguments, workspace_dir)
    };
    let head = git(&["rev-parse", "HEAD"])?;
    let origin_main = git(&["rev-parse", "origin/main"])?;
    let origin_url = git(&["remote", "get-url", "origin"])?;
    let github_main_line = git(&["ls-remote", "--heads", "origin", "refs/heads/main"])?;
    let github_main = github_main_line
        .split_whitespace()
        .next()
        .ok_or("git ls-remote returned no origin main commit")?;
    let branch = git(&["symbolic-ref", "--short", "HEAD"])?;
    let status = git(&["status", "--porcelain=v1", "--untracked-files=all"])?;
    if head != candidate_commit
        || origin_main != candidate_commit
        || github_main != candidate_commit
        || branch != "main"
        || origin_url != EXPECTED_ORIGIN_URL
    {
        return Err(format!(
            "git custody mismatch: head={head} origin/main={origin_main} GitHub/main={github_main} branch={branch} origin_url={origin_url} candidate={candidate_commit}"
        )
        .into());
    }
    if !status.is_empty() {
        return Err(format!("git worktree is not clean at custody check: {status}").into());
    }
    Ok(json!({
        "head": head,
        "origin_main": origin_main,
        "github_main": github_main,
        "origin_url": origin_url,
        "branch": branch,
        "worktree_clean": true
    }))
}

fn custody_snapshot(manifest_dir: &Path, executable: &Path) -> Result<Value, Box<dyn Error>> {
    let workspace_dir = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("apxinf-metal manifest is not below the workspace root")?;
    let model_crate = workspace_dir.join("crates/apxinf-model");
    Ok(json!({
        "binary": file_identity(executable)?,
        "gdn_shader": file_identity(&manifest_dir.join("src/metal_w8_gdn.metal"))?,
        "mlp_shader": file_identity(&manifest_dir.join("src/metal_w8_mlp.metal"))?,
        "metal_shader": file_identity(&manifest_dir.join("src/metal_w8_linear_layer.metal"))?,
        "gdn_out_g32_shader": file_identity(&manifest_dir.join("src/metal_w8_gdn_out_g32.metal"))?,
        "primitive_bridge": file_identity(&manifest_dir.join("src/metal_rms_norm_count43_profile_v1_bridge.mm"))?,
        "rust_api": file_identity(&manifest_dir.join("src/rms_norm_profile_v1.rs"))?,
        "crate_root": file_identity(&manifest_dir.join("src/lib.rs"))?,
        "build_script": file_identity(&manifest_dir.join("build.rs"))?,
        "gate_example": file_identity(&manifest_dir.join("examples/qwen35_rms_simd_tail_exact_ab_v1.rs"))?,
        "crate_manifest": file_identity(&manifest_dir.join("Cargo.toml"))?,
        "workspace_lock": file_identity(&workspace_dir.join("Cargo.lock"))?,
        "predeclaration": file_identity(&manifest_dir.join("evidence/next-hotspot/qwen35-rms-simd-tail-exact-v1-predeclared-primitive-gate-v1-20260825.json"))?,
        "qwen_production_mapping": file_identity(&model_crate.join("src/qwen35/general.rs"))?,
        "linear_layer_bridge": file_identity(&manifest_dir.join("src/metal_w8_linear_layer_bridge.mm"))?,
        "stack3_bridge": file_identity(&manifest_dir.join("src/metal_w8_linear_layer_stack3_bridge.mm"))?,
        "boundary_bridge": file_identity(&manifest_dir.join("src/metal_w8_mlp_stack3_boundary_v1_bridge.mm"))?,
        "tail_bridge": file_identity(&manifest_dir.join("src/metal_w8_tail_mlp_head_v1_bridge.mm"))?,
        "embedded_source_sha256": embedded_source_sha256()
    }))
}

fn embedded_source_sha256() -> Value {
    json!({
        "gdn_shader": sha256_bytes(include_bytes!("../src/metal_w8_gdn.metal")),
        "mlp_shader": sha256_bytes(include_bytes!("../src/metal_w8_mlp.metal")),
        "metal_shader": sha256_bytes(include_bytes!("../src/metal_w8_linear_layer.metal")),
        "gdn_out_g32_shader": sha256_bytes(include_bytes!("../src/metal_w8_gdn_out_g32.metal")),
        "primitive_bridge": sha256_bytes(include_bytes!("../src/metal_rms_norm_count43_profile_v1_bridge.mm")),
        "rust_api": sha256_bytes(include_bytes!("../src/rms_norm_profile_v1.rs")),
        "crate_root": sha256_bytes(include_bytes!("../src/lib.rs")),
        "build_script": sha256_bytes(include_bytes!("../build.rs")),
        "gate_example": sha256_bytes(include_bytes!("qwen35_rms_simd_tail_exact_ab_v1.rs")),
        "crate_manifest": sha256_bytes(include_bytes!("../Cargo.toml")),
        "workspace_lock": sha256_bytes(include_bytes!("../../../Cargo.lock")),
        "predeclaration": sha256_bytes(include_bytes!("../evidence/next-hotspot/qwen35-rms-simd-tail-exact-v1-predeclared-primitive-gate-v1-20260825.json")),
        "qwen_production_mapping": sha256_bytes(include_bytes!("../../apxinf-model/src/qwen35/general.rs")),
        "linear_layer_bridge": sha256_bytes(include_bytes!("../src/metal_w8_linear_layer_bridge.mm")),
        "stack3_bridge": sha256_bytes(include_bytes!("../src/metal_w8_linear_layer_stack3_bridge.mm")),
        "boundary_bridge": sha256_bytes(include_bytes!("../src/metal_w8_mlp_stack3_boundary_v1_bridge.mm")),
        "tail_bridge": sha256_bytes(include_bytes!("../src/metal_w8_tail_mlp_head_v1_bridge.mm"))
    })
}

fn require_disk_sources_match_embedded(snapshot: &Value) -> Result<(), Box<dyn Error>> {
    let embedded = snapshot
        .get("embedded_source_sha256")
        .and_then(Value::as_object)
        .ok_or("custody snapshot omitted embedded source hashes")?;
    for (label, expected) in embedded {
        let expected = expected
            .as_str()
            .ok_or("embedded source hash is not a string")?;
        let actual = snapshot
            .get(label)
            .and_then(|identity| identity.get("sha256"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("custody snapshot omitted disk hash for {label}"))?;
        if actual != expected {
            return Err(format!(
                "disk source {label} sha256 {actual} does not match embedded {expected}"
            )
            .into());
        }
    }
    Ok(())
}

fn host_preflight() -> Value {
    let system = |program: &str, arguments: &[&str]| {
        command_output(program, arguments).unwrap_or_else(|error| format!("unavailable: {error}"))
    };
    let process_table = system("ps", &["-Ao", "pid=,pcpu=,comm=", "-r"]);
    let top_processes = process_table
        .lines()
        .take(12)
        .map(str::trim)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    json!({
        "quiet_host_attested": false,
        "classification": "captured for diagnostic context only; no process was stopped and this primitive screen is never formal promotion evidence",
        "hardware_model": system("sysctl", &["-n", "hw.model"]),
        "cpu_brand": system("sysctl", &["-n", "machdep.cpu.brand_string"]),
        "uptime": system("uptime", &[]),
        "top_processes_by_cpu": top_processes
    })
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program).args(arguments).output()?;
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

fn file_identity(path: &Path) -> Result<Value, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let metadata = std::fs::metadata(path)?;
    Ok(json!({
        "path": std::fs::canonicalize(path)?,
        "size": metadata.len(),
        "sha256": sha256_bytes(&bytes),
        "regular_file": metadata.is_file()
    }))
}

fn publish_create_new(path: &Path, receipt: &Value) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer(&mut file, receipt)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut output = None;
    let mut candidate_commit = None;
    let mut iter = std::env::args_os().skip(1);
    while let Some(raw_flag) = iter.next() {
        let flag = raw_flag.to_string_lossy();
        let value = |iter: &mut dyn Iterator<Item = std::ffi::OsString>| {
            iter.next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_ref() {
            "--output" => output = Some(PathBuf::from(value(&mut iter)?)),
            "--candidate-commit" => {
                candidate_commit = Some(value(&mut iter)?.to_string_lossy().into_owned())
            }
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    Ok(Args {
        output: output.ok_or("--output is required")?,
        candidate_commit: candidate_commit.ok_or("--candidate-commit is required")?,
    })
}

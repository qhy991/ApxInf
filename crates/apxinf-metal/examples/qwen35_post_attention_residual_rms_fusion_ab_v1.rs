//! Predeclared same-binary count-18 residual→RMS A/B for Qwen3.5-0.8B.
//!
//! This is a non-formal aggregate mechanism screen, not production submission
//! topology, an end-to-end model benchmark, or a cross-runtime comparison.

#![recursion_limit = "256"]

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use apxinf_metal::{
    MetalResidualRmsCount18PrimitiveV1, ResidualRmsCount18RuntimeReceiptV1, ResidualRmsProfileV1,
    QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1, QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FORMAT: &str = "apxinf-qwen35-post-attention-residual-rms-fusion-primitive-ab-v1";
const EPSILON: f32 = 1.0e-6;
const CORRECTNESS_FIXTURES: usize = 8;
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
            Self::A => "A_legacy_separate",
            Self::B => "B_fused_exact",
        }
    }

    const fn profile(self) -> ResidualRmsProfileV1 {
        match self {
            Self::A => ResidualRmsProfileV1::LegacySeparate,
            Self::B => ResidualRmsProfileV1::FusedExact,
        }
    }
}

struct Fixture {
    seed: Vec<f32>,
    updates: Vec<f32>,
}

struct Args {
    output: PathBuf,
    candidate_commit: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if cfg!(debug_assertions) {
        return Err("residual-RMS primitive gate must be built in release mode".into());
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

    let weights = seeded_weights();
    let fixtures = (0..CORRECTNESS_FIXTURES)
        .map(seeded_fixture)
        .collect::<Vec<_>>();
    let fixture_sha256 = hash_fixture(&weights, &fixtures);
    let mut primitive = MetalResidualRmsCount18PrimitiveV1::new(&weights, EPSILON)?;
    let initial_a = primitive.runtime_receipt(Arm::A.profile())?;
    let initial_b = primitive.runtime_receipt(Arm::B.profile())?;
    validate_runtime_receipt(&initial_a, Arm::A.profile(), Some(0))?;
    validate_runtime_receipt(&initial_b, Arm::B.profile(), Some(0))?;

    let mut sampled_attempt_failures = Vec::new();
    let exactness = match exactness_check(&mut primitive, &fixtures) {
        Ok(exactness) => exactness,
        Err(error) => {
            sampled_attempt_failures.push(format!("exactness execution: {error}"));
            json!({
                "passed": false,
                "performance_authorized": false,
                "harness_error": error.to_string()
            })
        }
    };
    let performance = if exactness.get("passed").and_then(Value::as_bool) == Some(true) {
        match performance_attempt(&mut primitive, &fixtures[0]) {
            Ok(performance) => Some(performance),
            Err(forensic) => {
                let detail = forensic
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown performance attempt error");
                sampled_attempt_failures.push(format!("performance execution: {detail}"));
                Some(json!({
                    "completed": false,
                    "passed": false,
                    "attempt_failure": forensic
                }))
            }
        }
    } else {
        None
    };

    let performance_completed = performance
        .as_ref()
        .and_then(|value| value.get("completed"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let expected_runs =
        performance_completed.then_some((CORRECTNESS_FIXTURES * 2 + CALLS_PER_CELL * 8) as u64);
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
            "hidden_size": QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1,
            "seams_per_aggregate_run": QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1,
            "trace_rows_per_call": 2 * QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1,
            "rms_epsilon": EPSILON,
            "same_binary_and_live_library": true,
            "fixture_staging_poisoning_and_correctness_snapshots_outside_timing": true,
            "production_submission_topology": false,
            "active_production_distribution": "seven command buffers and 24 encoders"
        },
        "source_call_mapping": {
            "initial_stack_linear_layers": [0, 1, 2],
            "boundary_stack_linear_layers": [[4, 5, 6], [8, 9, 10], [12, 13, 14], [16, 17, 18], [20, 21, 22]],
            "total_same_encoder_seams": 18,
            "excluded_cross_encoder_seams": 17,
            "excluded_tail_mlp_residual_to_final_rms_seams": 1,
            "excluded_body_full_attention_cpu_boundary_rms_calls": 5,
            "excluded_tail_full_attention_cpu_fed_post_attention_rms_calls": 1,
            "excluded_initial_layer0_embedding_fed_input_rms_calls": 1,
            "excluded_standalone_linear_layer_bridge": "not part of the accepted seven-command-buffer production path"
        },
        "source_derived_tradeoff": {
            "A_dispatches_per_aggregate_run": 36,
            "B_dispatches_per_aggregate_run": 18,
            "A_pair_local_raw_barriers_per_aggregate_run": 18,
            "B_pair_local_raw_barriers_per_aggregate_run": 0,
            "common_consumer_barriers_per_aggregate_run_both": 18,
            "A_total_explicit_barriers_per_aggregate_run": 36,
            "B_total_explicit_barriers_per_aggregate_run": 18,
            "A_threadgroups_per_aggregate_run": 90,
            "B_threadgroups_per_aggregate_run": 18,
            "internal_threadgroup_barriers_per_aggregate_run_both": 162,
            "source_logical_residual_row_read_reduction_bytes_per_projected_decode": 147456,
            "source_logical_read_qualification": "source-level kernel reads only; not a measured cache, system-memory, or DRAM counter; final register allocation may spill retained values",
            "measured_hardware_barrier_counter": false,
            "projected_production_dispatches": {"A": 267, "B": 249},
            "projected_production_explicit_barriers": {"A": 243, "B": 225},
            "projected_production_command_buffers_unchanged": 7,
            "projected_production_compute_encoders_unchanged": 24
        },
        "fixture": {
            "generator": "18 deterministic dyadic xorshift64 weight rows and eight fixtures containing one seed plus 18 distinct dyadic update rows",
            "sha256_f32_le_with_shape_and_epsilon": fixture_sha256,
            "weight_rows": QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1,
            "correctness_fixtures": CORRECTNESS_FIXTURES,
            "one_time_weight_upload_bytes": 73728,
            "fixture_staging_bytes_per_stage_outside_timing": 77824,
            "fixture_stage_count": 9,
            "fixture_staging_bytes_total": 700416,
            "correctness_snapshot_bytes_total": 4718592,
            "correctness_trace_poison_bytes_total": 4718592
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
        "performance_attempted": performance.is_some(),
        "performance_completed": performance_completed,
        "sampled_attempt_failures": sampled_attempt_failures,
        "primitive_continue_gate_passed": primitive_continue_gate_passed,
        "formal_admission_passed": false,
        "screen_passed": primitive_continue_gate_passed,
        "passed": primitive_continue_gate_passed
    });
    publish_create_new(&args.output, &receipt)?;
    println!("{}", serde_json::to_string(&receipt)?);
    if !primitive_continue_gate_passed {
        return Err(
            "post-attention residual-RMS fusion primitive rejected; receipt was published".into(),
        );
    }
    Ok(())
}

fn next_xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn seeded_weights() -> Vec<f32> {
    let count = QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1 * QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1;
    let mut state = 0x243f_6a88_85a3_08d3;
    (0..count)
        .map(|_| {
            let signed = ((next_xorshift(&mut state) >> 32) % 2001) as i32 - 1000;
            1.0 + signed as f32 / 8192.0
        })
        .collect()
}

fn seeded_fixture(index: usize) -> Fixture {
    let mut seed_state = 0x1319_8a2e_0370_7344 ^ index as u64;
    let seed = (0..QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1)
        .map(|_| {
            let signed = ((next_xorshift(&mut seed_state) >> 32) % 2001) as i32 - 1000;
            signed as f32 / 1024.0
        })
        .collect();
    let mut update_state = 0xa409_3822_299f_31d0 ^ ((index as u64) << 32);
    let update_count = QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1 * QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1;
    let updates = (0..update_count)
        .map(|_| {
            let signed = ((next_xorshift(&mut update_state) >> 32) % 1001) as i32 - 500;
            signed as f32 / 8192.0
        })
        .collect();
    Fixture { seed, updates }
}

fn hash_fixture(weights: &[f32], fixtures: &[Fixture]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"qwen35-post-attention-residual-rms-fusion-fixture-v1");
    hasher.update((QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1 as u64).to_le_bytes());
    hasher.update((QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1 as u64).to_le_bytes());
    hasher.update(EPSILON.to_bits().to_le_bytes());
    hasher.update(b"weights");
    for value in weights {
        hasher.update(value.to_bits().to_le_bytes());
    }
    for (index, fixture) in fixtures.iter().enumerate() {
        hasher.update(b"fixture");
        hasher.update((index as u64).to_le_bytes());
        hasher.update(b"seed");
        for value in &fixture.seed {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(b"updates");
        for value in &fixture.updates {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn run_arm(
    arm: Arm,
    primitive: &mut MetalResidualRmsCount18PrimitiveV1,
) -> Result<(), Box<dyn Error>> {
    primitive.run(arm.profile())?;
    std::hint::black_box(arm.label());
    Ok(())
}

fn exactness_check(
    primitive: &mut MetalResidualRmsCount18PrimitiveV1,
    fixtures: &[Fixture],
) -> Result<Value, Box<dyn Error>> {
    let order = [Arm::A, Arm::B, Arm::B, Arm::A];
    let mut compared_elements = 0usize;
    let mut residual_comparisons = 0usize;
    let mut normalized_comparisons = 0usize;
    let mut finite_checks = 0usize;
    for (fixture_index, fixture) in fixtures.iter().enumerate() {
        primitive.stage_fixture(&fixture.seed, &fixture.updates)?;
        let mut outputs = Vec::with_capacity(order.len());
        for arm in order {
            primitive.poison_traces_for_correctness()?;
            run_arm(arm, primitive)?;
            outputs.push(primitive.snapshot()?);
        }
        for (call_index, output) in outputs.iter().enumerate() {
            for (tensor_kind, tensor) in [
                ("materialized_residual", &output.materialized_residual_rows),
                ("normalized", &output.normalized_rows),
            ] {
                for (element, &actual) in tensor.iter().enumerate() {
                    finite_checks += 1;
                    if !actual.is_finite() {
                        return Ok(json!({
                            "passed": false,
                            "performance_authorized": false,
                            "order_per_fixture": order.map(Arm::label),
                            "finite_checks_before_failure": finite_checks,
                            "compared_elements_before_failure": compared_elements,
                            "first_mismatch": {
                                "kind": "non_finite",
                                "fixture_index": fixture_index,
                                "call_index": call_index,
                                "arm": order[call_index].label(),
                                "tensor_kind": tensor_kind,
                                "seam": element / QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1,
                                "element": element % QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1,
                                "actual_value": actual,
                                "actual_bits": actual.to_bits()
                            }
                        }));
                    }
                }
            }
        }
        for (call_index, output) in outputs.iter().enumerate().skip(1) {
            for (tensor_kind, expected, actual) in [
                (
                    "materialized_residual",
                    &outputs[0].materialized_residual_rows,
                    &output.materialized_residual_rows,
                ),
                (
                    "normalized",
                    &outputs[0].normalized_rows,
                    &output.normalized_rows,
                ),
            ] {
                for (element, (&left, &right)) in expected.iter().zip(actual).enumerate() {
                    compared_elements += 1;
                    if tensor_kind == "materialized_residual" {
                        residual_comparisons += 1;
                    } else {
                        normalized_comparisons += 1;
                    }
                    if left.to_bits() != right.to_bits() {
                        return Ok(json!({
                            "passed": false,
                            "performance_authorized": false,
                            "order_per_fixture": order.map(Arm::label),
                            "finite_checks_before_failure": finite_checks,
                            "compared_elements_before_failure": compared_elements,
                            "first_mismatch": {
                                "kind": "to_bits",
                                "fixture_index": fixture_index,
                                "call_index": call_index,
                                "arm": order[call_index].label(),
                                "tensor_kind": tensor_kind,
                                "seam": element / QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1,
                                "element": element % QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1,
                                "expected_value": left,
                                "actual_value": right,
                                "expected_bits": left.to_bits(),
                                "actual_bits": right.to_bits()
                            }
                        }));
                    }
                }
            }
        }
    }
    primitive.verify_invalid_raw_selector_fail_closed()?;
    Ok(json!({
        "passed": true,
        "performance_authorized": true,
        "fixture_count": fixtures.len(),
        "order_per_fixture": order.map(Arm::label),
        "seams_per_call": QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1,
        "tensor_kinds": ["materialized_residual", "normalized"],
        "elements_per_tensor_row": QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1,
        "finite_checks": finite_checks,
        "compared_elements": compared_elements,
        "materialized_residual_comparisons": residual_comparisons,
        "normalized_comparisons": normalized_comparisons,
        "all_outputs_finite": true,
        "all_rows_match_to_bits": true,
        "outputs_poisoned_before_every_correctness_call": true,
        "invalid_raw_selector_failed_closed_without_mutation": true
    }))
}

fn performance_attempt(
    primitive: &mut MetalResidualRmsCount18PrimitiveV1,
    fixture: &Fixture,
) -> Result<Value, Value> {
    if let Err(error) = primitive.stage_fixture(&fixture.seed, &fixture.updates) {
        return Err(json!({
            "phase": "fixture_stage",
            "error": error.to_string(),
            "warmup_completed_calls": {"A": 0, "B": 0},
            "timed_block_1_partial": partial_samples_json(&[], &[]),
            "timed_block_2_partial": partial_samples_json(&[], &[])
        }));
    }

    let mut warmup_completed = [0usize; 2];
    warmup_block_capture(
        "warmup_block_1",
        [Arm::A, Arm::B, Arm::B, Arm::A],
        primitive,
        &mut warmup_completed,
    )?;
    warmup_block_capture(
        "warmup_block_2",
        [Arm::B, Arm::A, Arm::A, Arm::B],
        primitive,
        &mut warmup_completed,
    )?;

    let block_1 =
        match timed_block_capture("timed_block_1", [Arm::A, Arm::B, Arm::B, Arm::A], primitive) {
            Ok(samples) => samples,
            Err(failure) => {
                return Err(timed_failure_json(failure, &warmup_completed, None));
            }
        };
    let block_2 =
        match timed_block_capture("timed_block_2", [Arm::B, Arm::A, Arm::A, Arm::B], primitive) {
            Ok(samples) => samples,
            Err(failure) => {
                return Err(timed_failure_json(
                    failure,
                    &warmup_completed,
                    Some(&block_1),
                ));
            }
        };
    Ok(performance_json(block_1, block_2))
}

fn warmup_block_capture(
    block_label: &'static str,
    order: [Arm; 4],
    primitive: &mut MetalResidualRmsCount18PrimitiveV1,
    completed_total: &mut [usize; 2],
) -> Result<(), Value> {
    for (cell_index, arm) in order.into_iter().enumerate() {
        for call_index in 0..CALLS_PER_CELL {
            if let Err(error) = run_arm(arm, primitive) {
                return Err(json!({
                    "phase": "warmup",
                    "block": block_label,
                    "order": order.map(Arm::label),
                    "cell_index": cell_index,
                    "call_index_within_cell": call_index,
                    "arm": arm.label(),
                    "error": error.to_string(),
                    "warmup_completed_calls": {"A": completed_total[0], "B": completed_total[1]},
                    "timed_block_1_partial": partial_samples_json(&[], &[]),
                    "timed_block_2_partial": partial_samples_json(&[], &[])
                }));
            }
            let arm_index = match arm {
                Arm::A => 0,
                Arm::B => 1,
            };
            completed_total[arm_index] += 1;
        }
    }
    Ok(())
}

struct TimedBlockFailure {
    block_label: &'static str,
    order: [&'static str; 4],
    cell_index: usize,
    call_index_within_cell: usize,
    arm: &'static str,
    error: String,
    partial_samples: (Vec<u128>, Vec<u128>),
}

fn timed_block_capture(
    block_label: &'static str,
    order: [Arm; 4],
    primitive: &mut MetalResidualRmsCount18PrimitiveV1,
) -> Result<(Vec<u128>, Vec<u128>), TimedBlockFailure> {
    let mut a = Vec::with_capacity(CALLS_PER_CELL * 2);
    let mut b = Vec::with_capacity(CALLS_PER_CELL * 2);
    for (cell_index, arm) in order.into_iter().enumerate() {
        for call_index in 0..CALLS_PER_CELL {
            let started = Instant::now();
            if let Err(error) = run_arm(arm, primitive) {
                return Err(TimedBlockFailure {
                    block_label,
                    order: order.map(Arm::label),
                    cell_index,
                    call_index_within_cell: call_index,
                    arm: arm.label(),
                    error: error.to_string(),
                    partial_samples: (a, b),
                });
            }
            let elapsed = started.elapsed().as_nanos();
            match arm {
                Arm::A => a.push(elapsed),
                Arm::B => b.push(elapsed),
            }
        }
    }
    Ok((a, b))
}

fn timed_failure_json(
    failure: TimedBlockFailure,
    warmup_completed: &[usize; 2],
    completed_block_1: Option<&(Vec<u128>, Vec<u128>)>,
) -> Value {
    let (partial_a, partial_b) = &failure.partial_samples;
    let empty = (Vec::new(), Vec::new());
    let block_1 = completed_block_1.unwrap_or(&empty);
    let (block_2_a, block_2_b) = if completed_block_1.is_some() {
        (partial_a.as_slice(), partial_b.as_slice())
    } else {
        (&[][..], &[][..])
    };
    let (block_1_a, block_1_b) = if completed_block_1.is_some() {
        (block_1.0.as_slice(), block_1.1.as_slice())
    } else {
        (partial_a.as_slice(), partial_b.as_slice())
    };
    json!({
        "phase": "timed",
        "block": failure.block_label,
        "order": failure.order,
        "cell_index": failure.cell_index,
        "call_index_within_cell": failure.call_index_within_cell,
        "arm": failure.arm,
        "error": failure.error,
        "warmup_completed_calls": {"A": warmup_completed[0], "B": warmup_completed[1]},
        "timed_block_1_partial": partial_samples_json(block_1_a, block_1_b),
        "timed_block_2_partial": partial_samples_json(block_2_a, block_2_b)
    })
}

fn partial_samples_json(a: &[u128], b: &[u128]) -> Value {
    json!({
        "A_raw_ns": a,
        "B_raw_ns": b,
        "A_completed_samples": a.len(),
        "B_completed_samples": b.len()
    })
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
        "completed": true,
        "schedule": {
            "calls_per_cell": CALLS_PER_CELL,
            "seams_per_call": QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1,
            "block_1_order": ["A", "B", "B", "A"],
            "block_2_order": ["B", "A", "A", "B"],
            "timed_calls_per_arm_total": pooled_a.len(),
            "timed_dispatches_A_total": pooled_a.len() * 36,
            "timed_dispatches_B_total": pooled_b.len() * 18,
            "warmup_used_the_same_two_blocks": true,
            "fixture_staged_outside_timing": true,
            "poisoning_disabled_during_performance": true,
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
    receipt: &ResidualRmsCount18RuntimeReceiptV1,
    expected: ResidualRmsProfileV1,
    expected_successful_runs: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    let expected_last = u32::from(receipt.successful_runs != 0);
    let (primary_static, secondary_static, secondary_width_valid, secondary_max_valid) =
        match expected {
            ResidualRmsProfileV1::LegacySeparate => (0, 1024, true, true),
            ResidualRmsProfileV1::FusedExact => (1024, 0, false, false),
        };
    if receipt.requested_profile != expected
        || receipt.observed_profile != expected
        || receipt.requested_primary_function_name != expected.expected_primary_function_name()
        || receipt.observed_primary_function_name != expected.expected_primary_function_name()
        || receipt.requested_secondary_function_name != expected.expected_secondary_function_name()
        || receipt.observed_secondary_function_name != expected.expected_secondary_function_name()
        || receipt.hidden_size != QWEN35_RESIDUAL_RMS_HIDDEN_SIZE_V1 as u32
        || receipt.seams_per_run != QWEN35_RESIDUAL_RMS_SEAMS_PER_DECODE_V1 as u32
        || receipt.threads_per_threadgroup != 256
        || receipt.simdgroups_per_threadgroup != 8
        || receipt.primary_pipeline_thread_execution_width != 32
        || receipt.primary_pipeline_max_total_threads_per_threadgroup < 256
        || receipt.primary_static_threadgroup_memory_bytes != primary_static
        || (secondary_width_valid && receipt.secondary_pipeline_thread_execution_width != 32)
        || (!secondary_width_valid && receipt.secondary_pipeline_thread_execution_width != 0)
        || (secondary_max_valid
            && receipt.secondary_pipeline_max_total_threads_per_threadgroup < 256)
        || (!secondary_max_valid
            && receipt.secondary_pipeline_max_total_threads_per_threadgroup != 0)
        || receipt.secondary_static_threadgroup_memory_bytes != secondary_static
        || receipt.dynamic_threadgroup_memory_bytes != 0
        || receipt.internal_threadgroup_barriers_per_seam != 9
        || receipt.internal_threadgroup_barriers_per_run != 162
        || receipt.command_buffers_per_run != 1
        || receipt.compute_encoders_per_run != 1
        || receipt.kernel_dispatches_per_run != expected.kernel_dispatches_per_run()
        || receipt.explicit_buffer_barriers_per_run != expected.explicit_buffer_barriers_per_run()
        || receipt.pair_local_raw_barriers_per_run != expected.pair_local_raw_barriers_per_run()
        || receipt.common_consumer_barriers_per_run != expected.common_consumer_barriers_per_run()
        || receipt.commits_per_run != 1
        || receipt.waits_per_run != 1
        || receipt.host_to_device_bytes_per_run != 0
        || receipt.device_to_host_bytes_per_run != 0
        || expected_successful_runs.is_some_and(|count| receipt.successful_runs != count)
        || receipt.last_observed_command_buffers != expected_last
        || receipt.last_observed_compute_encoders != expected_last
        || receipt.last_observed_kernel_dispatches
            != expected_last * expected.kernel_dispatches_per_run()
        || receipt.last_observed_explicit_buffer_barriers
            != expected_last * expected.explicit_buffer_barriers_per_run()
        || receipt.last_observed_pair_local_raw_barriers
            != expected_last * expected.pair_local_raw_barriers_per_run()
        || receipt.last_observed_common_consumer_barriers
            != expected_last * expected.common_consumer_barriers_per_run()
        || receipt.last_observed_commits != expected_last
        || receipt.last_observed_waits != expected_last
    {
        return Err(format!("invalid live runtime receipt for {expected:?}: {receipt:?}").into());
    }
    Ok(())
}

fn capture_final_receipt(
    primitive: &MetalResidualRmsCount18PrimitiveV1,
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

fn runtime_receipt_json(receipt: &ResidualRmsCount18RuntimeReceiptV1) -> Value {
    json!({
        "requested_profile": profile_label(receipt.requested_profile),
        "observed_profile": profile_label(receipt.observed_profile),
        "requested_primary_function_name": receipt.requested_primary_function_name,
        "observed_primary_function_name": receipt.observed_primary_function_name,
        "requested_secondary_function_name": receipt.requested_secondary_function_name,
        "observed_secondary_function_name": receipt.observed_secondary_function_name,
        "hidden_size": receipt.hidden_size,
        "seams_per_run": receipt.seams_per_run,
        "threads_per_threadgroup": receipt.threads_per_threadgroup,
        "simdgroups_per_threadgroup": receipt.simdgroups_per_threadgroup,
        "primary_pipeline_max_total_threads_per_threadgroup": receipt.primary_pipeline_max_total_threads_per_threadgroup,
        "primary_pipeline_thread_execution_width": receipt.primary_pipeline_thread_execution_width,
        "primary_static_threadgroup_memory_bytes": receipt.primary_static_threadgroup_memory_bytes,
        "secondary_pipeline_max_total_threads_per_threadgroup": receipt.secondary_pipeline_max_total_threads_per_threadgroup,
        "secondary_pipeline_thread_execution_width": receipt.secondary_pipeline_thread_execution_width,
        "secondary_static_threadgroup_memory_bytes": receipt.secondary_static_threadgroup_memory_bytes,
        "dynamic_threadgroup_memory_bytes": receipt.dynamic_threadgroup_memory_bytes,
        "internal_threadgroup_barriers_per_seam": receipt.internal_threadgroup_barriers_per_seam,
        "internal_threadgroup_barriers_per_run": receipt.internal_threadgroup_barriers_per_run,
        "internal_barrier_count_is_source_derived": true,
        "command_buffers_per_run": receipt.command_buffers_per_run,
        "compute_encoders_per_run": receipt.compute_encoders_per_run,
        "kernel_dispatches_per_run": receipt.kernel_dispatches_per_run,
        "explicit_buffer_barriers_per_run": receipt.explicit_buffer_barriers_per_run,
        "pair_local_raw_barriers_per_run": receipt.pair_local_raw_barriers_per_run,
        "common_consumer_barriers_per_run": receipt.common_consumer_barriers_per_run,
        "commits_per_run": receipt.commits_per_run,
        "waits_per_run": receipt.waits_per_run,
        "host_to_device_bytes_per_run": receipt.host_to_device_bytes_per_run,
        "device_to_host_bytes_per_run": receipt.device_to_host_bytes_per_run,
        "successful_runs": receipt.successful_runs,
        "total_successful_kernel_dispatches": receipt.successful_runs * receipt.kernel_dispatches_per_run as u64,
        "total_successful_explicit_buffer_barriers": receipt.successful_runs * receipt.explicit_buffer_barriers_per_run as u64,
        "total_successful_pair_local_raw_barriers": receipt.successful_runs * receipt.pair_local_raw_barriers_per_run as u64,
        "total_successful_common_consumer_barriers": receipt.successful_runs * receipt.common_consumer_barriers_per_run as u64,
        "total_source_derived_internal_threadgroup_barriers": receipt.successful_runs * receipt.internal_threadgroup_barriers_per_run as u64,
        "last_observed_command_buffers": receipt.last_observed_command_buffers,
        "last_observed_compute_encoders": receipt.last_observed_compute_encoders,
        "last_observed_kernel_dispatches": receipt.last_observed_kernel_dispatches,
        "last_observed_explicit_buffer_barriers": receipt.last_observed_explicit_buffer_barriers,
        "last_observed_pair_local_raw_barriers": receipt.last_observed_pair_local_raw_barriers,
        "last_observed_common_consumer_barriers": receipt.last_observed_common_consumer_barriers,
        "last_observed_commits": receipt.last_observed_commits,
        "last_observed_waits": receipt.last_observed_waits
    })
}

fn profile_label(profile: ResidualRmsProfileV1) -> &'static str {
    match profile {
        ResidualRmsProfileV1::LegacySeparate => "legacy-separate",
        ResidualRmsProfileV1::FusedExact => "fused-exact",
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
        "linear_shader": file_identity(&manifest_dir.join("src/metal_w8_linear_layer.metal"))?,
        "gdn_out_g32_shader": file_identity(&manifest_dir.join("src/metal_w8_gdn_out_g32.metal"))?,
        "primitive_bridge": file_identity(&manifest_dir.join("src/metal_residual_rms_count18_profile_v1_bridge.mm"))?,
        "rust_api": file_identity(&manifest_dir.join("src/residual_rms_profile_v1.rs"))?,
        "crate_root": file_identity(&manifest_dir.join("src/lib.rs"))?,
        "build_script": file_identity(&manifest_dir.join("build.rs"))?,
        "gate_example": file_identity(&manifest_dir.join("examples/qwen35_post_attention_residual_rms_fusion_ab_v1.rs"))?,
        "crate_manifest": file_identity(&manifest_dir.join("Cargo.toml"))?,
        "workspace_lock": file_identity(&workspace_dir.join("Cargo.lock"))?,
        "predeclaration": file_identity(&manifest_dir.join("evidence/next-hotspot/qwen35-post-attention-residual-rms-fusion-v1-predeclared-primitive-gate-v1-20260825.json"))?,
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
        "linear_shader": sha256_bytes(include_bytes!("../src/metal_w8_linear_layer.metal")),
        "gdn_out_g32_shader": sha256_bytes(include_bytes!("../src/metal_w8_gdn_out_g32.metal")),
        "primitive_bridge": sha256_bytes(include_bytes!("../src/metal_residual_rms_count18_profile_v1_bridge.mm")),
        "rust_api": sha256_bytes(include_bytes!("../src/residual_rms_profile_v1.rs")),
        "crate_root": sha256_bytes(include_bytes!("../src/lib.rs")),
        "build_script": sha256_bytes(include_bytes!("../build.rs")),
        "gate_example": sha256_bytes(include_bytes!("qwen35_post_attention_residual_rms_fusion_ab_v1.rs")),
        "crate_manifest": sha256_bytes(include_bytes!("../Cargo.toml")),
        "workspace_lock": sha256_bytes(include_bytes!("../../../Cargo.lock")),
        "predeclaration": sha256_bytes(include_bytes!("../evidence/next-hotspot/qwen35-post-attention-residual-rms-fusion-v1-predeclared-primitive-gate-v1-20260825.json")),
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

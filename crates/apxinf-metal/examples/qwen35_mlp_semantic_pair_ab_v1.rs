//! Predeclared same-binary primitive A/B for the Qwen3.5-0.8B Metal W8 MLP.
//!
//! This is a production-shape mechanism screen, not an end-to-end model or
//! cross-runtime benchmark. A passing result only authorizes full-path wiring.

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use apxinf_metal::{
    MetalW8MlpBlock, PackedW8MlpBlock, W8MlpGateUpProfileV1, W8MlpGateUpRuntimeReceiptV1,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FORMAT: &str = "apxinf-qwen35-mlp-semantic-pair-primitive-ab-v1";
const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 3584;
const CORRECTNESS_INPUTS: usize = 8;
const CALLS_PER_CELL: usize = 64;
const KEEP_THRESHOLD_PERCENT: f64 = 5.0;
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
            Self::B => "B_semantic_pair_silu",
        }
    }
}

struct Args {
    output: PathBuf,
    candidate_commit: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.candidate_commit.len() != 40
        || !args
            .candidate_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("--candidate-commit must be a full 40-character hexadecimal commit".into());
    }
    let embedded_candidate_commit = EMBEDDED_CANDIDATE_COMMIT
        .ok_or("release benchmark binary was not built with APXINF_CANDIDATE_COMMIT")?;
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
    let git_custody_start = git_custody(workspace_dir, &args.candidate_commit)?;
    let custody_start = custody_snapshot(manifest_dir, &executable)?;
    require_disk_sources_match_embedded(&custody_start)?;

    let elements = HIDDEN * INTERMEDIATE;
    let gate = seeded_f32(elements, 0x243f_6a88_85a3_08d3);
    let up = seeded_f32(elements, 0x1319_8a2e_0370_7344);
    let down = seeded_f32(elements, 0xa409_3822_299f_31d0);
    let inputs = (0..CORRECTNESS_INPUTS)
        .map(|index| seeded_f32(HIDDEN, 0x082e_fa98_ec4e_6c89 ^ index as u64))
        .collect::<Vec<_>>();
    let fixture_sha256 = hash_f32_fixture(&gate, &up, &down, &inputs);
    let packed = PackedW8MlpBlock::pack_f32(&gate, &up, &down, HIDDEN, INTERMEDIATE)?;
    drop((gate, up, down));

    let mut legacy = MetalW8MlpBlock::from_packed_with_gate_up_profile_v1(
        &packed,
        W8MlpGateUpProfileV1::LegacySeparate,
    )?;
    let mut candidate = MetalW8MlpBlock::from_packed_with_gate_up_profile_v1(
        &packed,
        W8MlpGateUpProfileV1::SemanticPairSilu,
    )?;
    validate_runtime_receipt(
        legacy.gate_up_runtime_receipt_v1()?,
        W8MlpGateUpProfileV1::LegacySeparate,
        Some(0),
    )?;
    validate_runtime_receipt(
        candidate.gate_up_runtime_receipt_v1()?,
        W8MlpGateUpProfileV1::SemanticPairSilu,
        Some(0),
    )?;

    let exactness = exactness_check(&mut legacy, &mut candidate, &inputs)?;
    let performance = if exactness.get("passed").and_then(Value::as_bool) == Some(true) {
        let timing_input = &inputs[0];
        warmup_block(
            [Arm::A, Arm::B, Arm::B, Arm::A],
            &mut legacy,
            &mut candidate,
            timing_input,
        )?;
        warmup_block(
            [Arm::B, Arm::A, Arm::A, Arm::B],
            &mut legacy,
            &mut candidate,
            timing_input,
        )?;
        let block_1 = timed_block(
            [Arm::A, Arm::B, Arm::B, Arm::A],
            &mut legacy,
            &mut candidate,
            timing_input,
        )?;
        let block_2 = timed_block(
            [Arm::B, Arm::A, Arm::A, Arm::B],
            &mut legacy,
            &mut candidate,
            timing_input,
        )?;
        Some(performance_json(block_1, block_2))
    } else {
        None
    };
    let expected_final_calls = performance
        .as_ref()
        .map(|_| (CORRECTNESS_INPUTS * 2 + CALLS_PER_CELL * 8) as u64);
    let legacy_receipt = legacy.gate_up_runtime_receipt_v1()?;
    let candidate_receipt = candidate.gate_up_runtime_receipt_v1()?;
    validate_runtime_receipt(
        legacy_receipt,
        W8MlpGateUpProfileV1::LegacySeparate,
        expected_final_calls,
    )?;
    validate_runtime_receipt(
        candidate_receipt,
        W8MlpGateUpProfileV1::SemanticPairSilu,
        expected_final_calls,
    )?;

    let performance_passed = performance
        .as_ref()
        .and_then(|value| value.get("passed"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let custody_end = custody_snapshot(manifest_dir, &executable)?;
    require_disk_sources_match_embedded(&custody_end)?;
    let git_custody_end = git_custody(workspace_dir, &args.candidate_commit)?;
    let custody_unchanged = custody_start == custody_end && git_custody_start == git_custody_end;
    let screen_passed = performance_passed && custody_unchanged;
    let receipt = json!({
        "format": FORMAT,
        "classification": "non-formal diagnostic production-shape primitive screen; host quietness is not attested; never an end-to-end or cross-runtime benchmark",
        "candidate_commit": args.candidate_commit,
        "embedded_candidate_commit": embedded_candidate_commit,
        "host_quiet_preflight_attested": false,
        "formal_admission_passed": false,
        "scope": {
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "weight_format": "W8 G64 with F32 scales",
            "target": "Apple Metal",
            "same_binary": true,
            "production_topology": true
        },
        "fixture": {
            "generator": "distinct deterministic xorshift64 streams for gate, up, down, and eight inputs",
            "sha256_f32_le": fixture_sha256,
            "correctness_input_count": CORRECTNESS_INPUTS
        },
        "runtime_receipts": {
            "A": runtime_receipt_json(legacy_receipt),
            "B": runtime_receipt_json(candidate_receipt)
        },
        "exactness": exactness,
        "performance": performance,
        "admission": {
            "primitive_continue_threshold_percent": KEEP_THRESHOLD_PERCENT,
            "requires_both_counterbalanced_blocks_positive": true,
            "clearly_negative_block_reject_percent": -0.5,
            "pass_only_authorizes_full_path_plumbing": true,
            "no_resampling_after_failure": true
        },
        "custody": {
            "start": custody_start,
            "end": custody_end,
            "git_start": git_custody_start,
            "git_end": git_custody_end,
            "unchanged_during_sampling": custody_unchanged
        },
        "performance_threshold_passed": performance_passed,
        "screen_passed": screen_passed,
        "passed": screen_passed
    });
    publish_create_new(&args.output, &receipt)?;
    println!("{}", serde_json::to_string(&receipt)?);
    if !screen_passed {
        return Err("semantic-pair primitive screen rejected; receipt was published".into());
    }
    Ok(())
}

fn seeded_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let signed = ((state >> 32) % 2001) as i32 - 1000;
            signed as f32 / 2048.0
        })
        .collect()
}

fn hash_f32_fixture(gate: &[f32], up: &[f32], down: &[f32], inputs: &[Vec<f32>]) -> String {
    let mut hasher = Sha256::new();
    for (label, values) in [
        (b"gate".as_slice(), gate),
        (b"up".as_slice(), up),
        (b"down".as_slice(), down),
    ] {
        hasher.update(label);
        for value in values {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    for (index, values) in inputs.iter().enumerate() {
        hasher.update((index as u64).to_le_bytes());
        for value in values {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn validate_runtime_receipt(
    receipt: W8MlpGateUpRuntimeReceiptV1,
    expected: W8MlpGateUpProfileV1,
    expected_successful_calls: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    if receipt.requested_profile != expected
        || receipt.observed_profile != expected
        || receipt.requested_function_name != expected.expected_function_name()
        || receipt.function_name != expected.expected_function_name()
        || receipt.threads_per_threadgroup != 256
        || receipt.simdgroups_per_threadgroup != 8
        || receipt.pipeline_thread_execution_width != 32
        || receipt.pipeline_max_total_threads_per_threadgroup < 256
        || receipt.static_threadgroup_memory_bytes != 0
        || receipt.dynamic_threadgroup_memory_bytes != 0
        || receipt.gate_up_threadgroups_per_call
            != if expected == W8MlpGateUpProfileV1::LegacySeparate {
                896
            } else {
                448
            }
        || receipt.command_buffers_per_call != 1
        || receipt.compute_encoders_per_call != 1
        || receipt.kernel_dispatches_per_call
            != if expected == W8MlpGateUpProfileV1::LegacySeparate {
                3
            } else {
                2
            }
        || receipt.explicit_buffer_barriers_per_call
            != if expected == W8MlpGateUpProfileV1::LegacySeparate {
                2
            } else {
                1
            }
        || receipt.internal_threadgroup_barriers_per_call != 0
        || receipt.semantic_pairs_per_threadgroup
            != if expected == W8MlpGateUpProfileV1::LegacySeparate {
                0
            } else {
                8
            }
        || expected_successful_calls.is_some_and(|count| receipt.successful_calls != count)
        || (receipt.successful_calls == 0
            && (receipt.last_observed_command_buffers != 0
                || receipt.last_observed_compute_encoders != 0
                || receipt.last_observed_kernel_dispatches != 0
                || receipt.last_observed_explicit_buffer_barriers != 0))
        || (receipt.successful_calls > 0
            && (receipt.last_observed_command_buffers != receipt.command_buffers_per_call
                || receipt.last_observed_compute_encoders != receipt.compute_encoders_per_call
                || receipt.last_observed_kernel_dispatches != receipt.kernel_dispatches_per_call
                || receipt.last_observed_explicit_buffer_barriers
                    != receipt.explicit_buffer_barriers_per_call))
    {
        return Err(format!("invalid live runtime receipt for {expected:?}: {receipt:?}").into());
    }
    Ok(())
}

fn runtime_receipt_json(receipt: W8MlpGateUpRuntimeReceiptV1) -> Value {
    json!({
        "requested_profile": profile_label(receipt.requested_profile),
        "observed_profile": profile_label(receipt.observed_profile),
        "requested_function_name": receipt.requested_function_name,
        "observed_function_name": receipt.function_name,
        "threads_per_threadgroup": receipt.threads_per_threadgroup,
        "simdgroups_per_threadgroup": receipt.simdgroups_per_threadgroup,
        "semantic_pairs_per_threadgroup": receipt.semantic_pairs_per_threadgroup,
        "pipeline_max_total_threads_per_threadgroup": receipt.pipeline_max_total_threads_per_threadgroup,
        "pipeline_thread_execution_width": receipt.pipeline_thread_execution_width,
        "static_threadgroup_memory_bytes": receipt.static_threadgroup_memory_bytes,
        "dynamic_threadgroup_memory_bytes": receipt.dynamic_threadgroup_memory_bytes,
        "gate_up_threadgroups_per_call": receipt.gate_up_threadgroups_per_call,
        "command_buffers_per_call": receipt.command_buffers_per_call,
        "compute_encoders_per_call": receipt.compute_encoders_per_call,
        "kernel_dispatches_per_call": receipt.kernel_dispatches_per_call,
        "explicit_buffer_barriers_per_call": receipt.explicit_buffer_barriers_per_call,
        "internal_threadgroup_barriers_per_call": receipt.internal_threadgroup_barriers_per_call,
        "successful_calls": receipt.successful_calls,
        "last_observed_command_buffers": receipt.last_observed_command_buffers,
        "last_observed_compute_encoders": receipt.last_observed_compute_encoders,
        "last_observed_kernel_dispatches": receipt.last_observed_kernel_dispatches,
        "last_observed_explicit_buffer_barriers": receipt.last_observed_explicit_buffer_barriers
    })
}

fn profile_label(profile: W8MlpGateUpProfileV1) -> &'static str {
    match profile {
        W8MlpGateUpProfileV1::LegacySeparate => "legacy-separate",
        W8MlpGateUpProfileV1::SemanticPairSilu => "semantic-pair-silu",
    }
}

fn run_arm<'a>(
    arm: Arm,
    legacy: &'a mut MetalW8MlpBlock,
    candidate: &'a mut MetalW8MlpBlock,
    input: &[f32],
) -> Result<&'a [f32], Box<dyn Error>> {
    let output = match arm {
        Arm::A => legacy.forward(input)?,
        Arm::B => candidate.forward(input)?,
    };
    Ok(std::hint::black_box(output))
}

fn exactness_check(
    legacy: &mut MetalW8MlpBlock,
    candidate: &mut MetalW8MlpBlock,
    inputs: &[Vec<f32>],
) -> Result<Value, Box<dyn Error>> {
    let order = [Arm::A, Arm::B, Arm::B, Arm::A];
    let mut compared_elements = 0usize;
    for (input_index, input) in inputs.iter().enumerate() {
        let mut outputs = Vec::with_capacity(order.len());
        for arm in order {
            outputs.push(run_arm(arm, legacy, candidate, input)?.to_vec());
        }
        for (call_index, output) in outputs.iter().enumerate().skip(1) {
            for (element, (&expected, &actual)) in outputs[0].iter().zip(output).enumerate() {
                compared_elements += 1;
                if !expected.is_finite()
                    || !actual.is_finite()
                    || expected.to_bits() != actual.to_bits()
                {
                    return Ok(json!({
                        "passed": false,
                        "performance_executed": false,
                        "order_per_input": order.map(Arm::label),
                        "compared_output_elements_before_failure": compared_elements,
                        "first_mismatch": {
                            "input_index": input_index,
                            "call_index": call_index,
                            "arm": order[call_index].label(),
                            "element": element,
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
        "compared_output_elements": compared_elements,
        "all_outputs_finite": true,
        "all_outputs_match_to_bits": true
    }))
}

fn warmup_block(
    order: [Arm; 4],
    legacy: &mut MetalW8MlpBlock,
    candidate: &mut MetalW8MlpBlock,
    input: &[f32],
) -> Result<(), Box<dyn Error>> {
    for arm in order {
        for _ in 0..CALLS_PER_CELL {
            std::hint::black_box(run_arm(arm, legacy, candidate, input)?);
        }
    }
    Ok(())
}

fn timed_block(
    order: [Arm; 4],
    legacy: &mut MetalW8MlpBlock,
    candidate: &mut MetalW8MlpBlock,
    input: &[f32],
) -> Result<(Vec<u128>, Vec<u128>), Box<dyn Error>> {
    let mut a = Vec::with_capacity(CALLS_PER_CELL * 2);
    let mut b = Vec::with_capacity(CALLS_PER_CELL * 2);
    for arm in order {
        for _ in 0..CALLS_PER_CELL {
            let started = Instant::now();
            std::hint::black_box(run_arm(arm, legacy, candidate, input)?);
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
            "block_1_order": ["A", "B", "B", "A"],
            "block_2_order": ["B", "A", "A", "B"],
            "timed_calls_per_arm_total": pooled_a.len(),
            "warmup_used_the_same_two_blocks": true
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

fn git_custody(workspace_dir: &Path, candidate_commit: &str) -> Result<Value, Box<dyn Error>> {
    let git = |arguments: &[&str]| -> Result<String, Box<dyn Error>> {
        let output = Command::new("git")
            .args(arguments)
            .current_dir(workspace_dir)
            .output()?;
        if !output.status.success() {
            return Err(format!(
                "git {} failed: {}",
                arguments.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            )
            .into());
        }
        Ok(String::from_utf8(output.stdout)?.trim().to_owned())
    };
    let head = git(&["rev-parse", "HEAD"])?;
    let origin_main = git(&["rev-parse", "origin/main"])?;
    let branch = git(&["symbolic-ref", "--short", "HEAD"])?;
    let status = git(&["status", "--porcelain=v1", "--untracked-files=all"])?;
    if head != candidate_commit || origin_main != candidate_commit || branch != "main" {
        return Err(format!(
            "git custody mismatch: head={head} origin/main={origin_main} branch={branch} candidate={candidate_commit}"
        )
        .into());
    }
    if !status.is_empty() {
        return Err(
            format!("git worktree is not clean at benchmark custody check: {status}").into(),
        );
    }
    Ok(json!({
        "head": head,
        "origin_main": origin_main,
        "branch": branch,
        "worktree_clean": true
    }))
}

fn custody_snapshot(manifest_dir: &Path, executable: &Path) -> Result<Value, Box<dyn Error>> {
    let workspace_dir = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("apxinf-metal manifest is not below the workspace root")?;
    Ok(json!({
        "binary": file_identity(executable)?,
        "metal_shader": file_identity(&manifest_dir.join("src/metal_w8_mlp.metal"))?,
        "objective_c_bridge": file_identity(&manifest_dir.join("src/metal_w8_mlp_bridge.mm"))?,
        "rust_api": file_identity(&manifest_dir.join("src/lib.rs"))?,
        "gate_example": file_identity(&manifest_dir.join("examples/qwen35_mlp_semantic_pair_ab_v1.rs"))?,
        "crate_manifest": file_identity(&manifest_dir.join("Cargo.toml"))?,
        "workspace_lock": file_identity(&workspace_dir.join("Cargo.lock"))?,
        "predeclaration": file_identity(&manifest_dir.join("evidence/next-hotspot/qwen35-mlp-semantic-pair-silu-v1-predeclared-primitive-gate-v1-20260825.json"))?,
        "embedded_source_sha256": embedded_source_sha256()
    }))
}

fn embedded_source_sha256() -> Value {
    json!({
        "metal_shader": sha256_bytes(include_bytes!("../src/metal_w8_mlp.metal")),
        "objective_c_bridge": sha256_bytes(include_bytes!("../src/metal_w8_mlp_bridge.mm")),
        "rust_api": sha256_bytes(include_bytes!("../src/lib.rs")),
        "gate_example": sha256_bytes(include_bytes!("qwen35_mlp_semantic_pair_ab_v1.rs")),
        "crate_manifest": sha256_bytes(include_bytes!("../Cargo.toml")),
        "workspace_lock": sha256_bytes(include_bytes!("../../../Cargo.lock")),
        "predeclaration": sha256_bytes(include_bytes!("../evidence/next-hotspot/qwen35-mlp-semantic-pair-silu-v1-predeclared-primitive-gate-v1-20260825.json"))
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

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn file_identity(path: &Path) -> Result<Value, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let metadata = std::fs::metadata(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(json!({
        "path": std::fs::canonicalize(path)?,
        "size": metadata.len(),
        "sha256": format!("{:x}", hasher.finalize()),
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

//! Predeclared same-binary count-18 GDN recurrent A/B/C mechanism screen.
//!
//! This is not production submission topology, an end-to-end model benchmark,
//! or a cross-runtime comparison. A pass only authorizes later opt-in plumbing.

#![recursion_limit = "256"]

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use apxinf_metal::{
    GdnRecurrentCount18RuntimeReceiptV1, GdnRecurrentCount18SnapshotV1, GdnRecurrentProfileV1,
    MetalGdnRecurrentCount18PrimitiveV1, QWEN35_GDN_CORE_ELEMENTS_PER_SEAM_V1,
    QWEN35_GDN_KEY_DIM_V1, QWEN35_GDN_KEY_HEADS_V1, QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1,
    QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V1, QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1,
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1, QWEN35_GDN_VALUE_DIM_V1, QWEN35_GDN_VALUE_HEADS_V1,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FORMAT: &str = "apxinf-qwen35-gdn-recurrent-local-reuse-primitive-abc-v1";
const CORRECTNESS_FIXTURES: usize = 8;
const CALLS_PER_CELL: usize = 64;
const C_SELECTION_THRESHOLD_PERCENT: f64 = 0.5;
const KEEP_THRESHOLD_PERCENT: f64 = 5.0;
const SIMPLICITY_BLOCK_FLOOR_PERCENT: f64 = -0.5;
const EXPECTED_SUCCESSFUL_RUNS_PER_ARM: u64 = 784;
const EXPECTED_ORIGIN_URL: &str = "https://github.com/qhy991/ApxInf.git";
const EMBEDDED_CANDIDATE_COMMIT: Option<&str> = option_env!("APXINF_CANDIDATE_COMMIT");

const A_LOG_TABLE: [f32; 8] = [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5];
const DT_BIAS_TABLE: [f32; 8] = [-0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5];
const B_TABLE: [f32; 8] = [-8.0, -2.0, -0.5, -0.0, 0.0, 0.5, 2.0, 8.0];
const TARGET_GATE_TABLE: [f32; 9] = [-24.0, -20.5, -20.0, -1.0, 0.0, 1.0, 20.0, 20.5, 24.0];

const BLOCK_ORDERS: [[Arm; 6]; 3] = [
    [Arm::A, Arm::B, Arm::C, Arm::C, Arm::B, Arm::A],
    [Arm::B, Arm::C, Arm::A, Arm::A, Arm::C, Arm::B],
    [Arm::C, Arm::A, Arm::B, Arm::B, Arm::A, Arm::C],
];

const PROCESSED_TRACE_ELEMENTS: usize =
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 * QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1;
const PROJECTED_TRACE_ELEMENTS: usize =
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 * QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V1;
const HEAD_SCALAR_TRACE_ELEMENTS: usize =
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 * QWEN35_GDN_VALUE_HEADS_V1;
const RECURRENT_TRACE_ELEMENTS: usize =
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 * QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1;
const CORE_TRACE_ELEMENTS: usize =
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 * QWEN35_GDN_CORE_ELEMENTS_PER_SEAM_V1;
const COMBINED_OUTPUT_ELEMENTS: usize = RECURRENT_TRACE_ELEMENTS + CORE_TRACE_ELEMENTS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    A,
    B,
    C,
}

impl Arm {
    const ALL: [Self; 3] = [Self::A, Self::B, Self::C];

    const fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C => 2,
        }
    }

    const fn short(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::A => "A_legacy_256",
            Self::B => "B_leader_broadcast_128",
            Self::C => "C_qk_staged_128",
        }
    }

    const fn profile(self) -> GdnRecurrentProfileV1 {
        match self {
            Self::A => GdnRecurrentProfileV1::Legacy256,
            Self::B => GdnRecurrentProfileV1::LeaderBroadcast128,
            Self::C => GdnRecurrentProfileV1::QkStaged128,
        }
    }

    const fn expected_pipeline_static_bytes(self) -> u32 {
        match self {
            Self::A => 0,
            Self::B => 16,
            Self::C => 1040,
        }
    }
}

#[derive(Clone)]
struct Fixture {
    processed: Vec<f32>,
    projected: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    state: Vec<f32>,
}

impl Fixture {
    fn stage(
        &self,
        primitive: &mut MetalGdnRecurrentCount18PrimitiveV1,
    ) -> Result<(), Box<dyn Error>> {
        primitive.stage_fixture(
            &self.processed,
            &self.projected,
            &self.a_log,
            &self.dt_bias,
            &self.state,
        )?;
        Ok(())
    }

    fn verify_unchanged(
        &self,
        primitive: &MetalGdnRecurrentCount18PrimitiveV1,
    ) -> Result<(), Box<dyn Error>> {
        primitive.verify_staged_fixture_unchanged(
            &self.processed,
            &self.projected,
            &self.a_log,
            &self.dt_bias,
            &self.state,
        )?;
        Ok(())
    }
}

struct Args {
    output: PathBuf,
    candidate_commit: String,
}

#[derive(Default)]
struct HarnessLedger {
    fixture_stages: u64,
    output_poison_calls: u64,
    output_snapshot_calls: u64,
    selector_probe_snapshot_materializations: u64,
    staged_input_verifications: u64,
    invalid_selector_checks: u64,
}

type Samples = [Vec<u128>; 3];

fn empty_samples() -> Samples {
    std::array::from_fn(|_| Vec::new())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if cfg!(debug_assertions) {
        return Err("GDN recurrent primitive gate must be built in release mode".into());
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
    require_production_consumers_legacy()?;
    let host_preflight = host_preflight(&args.candidate_commit);

    let fixtures = (0..CORRECTNESS_FIXTURES)
        .map(seeded_fixture)
        .collect::<Vec<_>>();
    let fixture_branch_coverage = validate_fixture_branch_coverage(&fixtures)?;
    let fixture_sha256 = hash_fixtures(&fixtures);
    let mut primitive = MetalGdnRecurrentCount18PrimitiveV1::new()?;
    for arm in Arm::ALL {
        let receipt = primitive.runtime_receipt(arm.profile())?;
        validate_runtime_receipt(&receipt, arm, Some(0))?;
    }

    let mut sampled_attempt_failures = Vec::new();
    let mut host_expected_successful_runs = [0u64; 3];
    let mut harness_ledger = HarnessLedger::default();
    let exactness = match exactness_check(
        &mut primitive,
        &fixtures,
        &mut host_expected_successful_runs,
        &mut harness_ledger,
    ) {
        Ok(value) => value,
        Err(forensic) => {
            let detail = forensic
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown exactness execution error");
            sampled_attempt_failures.push(format!("exactness execution: {detail}"));
            json!({
                "passed": false,
                "performance_authorized": false,
                "attempt_failure": forensic
            })
        }
    };
    let performance = if exactness.get("passed").and_then(Value::as_bool) == Some(true) {
        match performance_attempt(
            &mut primitive,
            &fixtures[0],
            &mut host_expected_successful_runs,
            &mut harness_ledger,
        ) {
            Ok(value) => Some(value),
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
    if performance_completed
        && host_expected_successful_runs != [EXPECTED_SUCCESSFUL_RUNS_PER_ARM; 3]
    {
        sampled_attempt_failures.push(format!(
            "completed schedule host run ledger mismatch: expected {:?}, observed {:?}",
            [EXPECTED_SUCCESSFUL_RUNS_PER_ARM; 3], host_expected_successful_runs
        ));
    }
    let mut final_receipts = serde_json::Map::new();
    let mut runtime_receipts_valid = true;
    for arm in Arm::ALL {
        let (value, valid) = capture_final_receipt(
            &primitive,
            arm,
            Some(host_expected_successful_runs[arm.index()]),
            &mut sampled_attempt_failures,
        );
        final_receipts.insert(arm.short().to_owned(), value);
        runtime_receipts_valid &= valid;
    }

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
    let exactness_passed = exactness.get("passed").and_then(Value::as_bool) == Some(true);
    let primitive_continue_gate_passed = exactness_passed
        && performance_completed
        && performance_passed
        && runtime_receipts_valid
        && custody_unchanged
        && sampled_attempt_failures.is_empty();

    let receipt = json!({
        "format": FORMAT,
        "classification": "non-formal count-matched aggregate GDN recurrent mechanism screen; not production submission topology, end-to-end inference, or a cross-runtime benchmark",
        "candidate_commit": args.candidate_commit,
        "embedded_candidate_commit": embedded_candidate_commit,
        "scope": {
            "model": "Qwen/Qwen3.5-0.8B",
            "seams_per_aggregate_call": QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1,
            "key_heads": QWEN35_GDN_KEY_HEADS_V1,
            "value_heads": QWEN35_GDN_VALUE_HEADS_V1,
            "key_dim": QWEN35_GDN_KEY_DIM_V1,
            "value_dim": QWEN35_GDN_VALUE_DIM_V1,
            "same_binary_and_same_diagnostic_live_library": true,
            "candidate_commit_production_consumers_select_only_legacy": true,
            "production_sources_unchanged_during_sampling": custody_unchanged,
            "production_submission_topology": false,
            "active_production_topology": {"kernel_dispatches":267,"explicit_broad_barriers":243,"compute_encoders":24,"command_buffers":7}
        },
        "source_call_mapping": {
            "initial_stack_linear_layers": [0,1,2],
            "boundary_stack_linear_layers": [[4,5,6],[8,9,10],[12,13,14],[16,17,18],[20,21,22]],
            "total_recurrent_dispatches": 18,
            "production_consumers_observed_selecting_only_legacy": [
                "metal_w8_gdn_bridge.mm",
                "metal_w8_linear_layer_bridge.mm",
                "metal_w8_linear_layer_stack3_bridge.mm",
                "metal_w8_mlp_stack3_boundary_v1_bridge.mm"
            ],
            "qualification": "source/ledger mapping only; standalone diagnostic bridges are not part of the accepted production lane"
        },
        "source_derived_tradeoff": source_tradeoff_json(),
        "fixture": {
            "correctness_fixtures": CORRECTNESS_FIXTURES,
            "generator": "deterministic dyadic xorshift64 with signed-zero row sentinels and fixed coefficient boundary tables",
            "sha256_f32_le_with_domain_shape_and_coefficient_tables": fixture_sha256,
            "branch_coverage": fixture_branch_coverage,
            "persistent_buffer_bytes": 38932992,
            "combined_output_bytes_per_snapshot_or_poison": 19021824,
            "planned_if_completed": {
                "stage_bytes_per_current_API_call": 19911168,
                "stage_count": 9,
                "staging_bytes_total": 179200512,
                "explicit_correctness_comparison_snapshot_calls": 48,
                "explicit_correctness_comparison_snapshot_bytes": 913047552,
                "invalid_selector_probe_successful_snapshot_materializations": 2,
                "invalid_selector_probe_snapshot_bytes": 38043648,
                "all_successful_snapshot_materializations": 50,
                "all_successful_snapshot_materialization_bytes": 951091200,
                "correctness_poison_calls": 48,
                "correctness_poison_bytes_total": 913047552,
                "staged_input_verifications": 9,
                "invalid_selector_checks": 2
            },
            "actual": harness_ledger_json(&harness_ledger)
        },
        "runtime_receipts": {
            "host_expected_successful_runs": counts_u64_json(&host_expected_successful_runs),
            "arms": final_receipts,
            "valid": runtime_receipts_valid
        },
        "exactness": exactness,
        "performance": performance,
        "admission": {
            "C_selected_iff_pooled_C_over_B_percent_at_least": C_SELECTION_THRESHOLD_PERCENT,
            "selected_over_A_pooled_percent_at_least": KEEP_THRESHOLD_PERCENT,
            "selected_over_A_strictly_positive_in_all_blocks": true,
            "selected_vs_unselected_strictly_greater_than_percent_in_all_blocks": SIMPLICITY_BLOCK_FLOOR_PERCENT,
            "if_C_selected_C_over_B_positive_in_at_least_two_blocks": true,
            "no_candidate_fallback": true,
            "no_outlier_removal": true,
            "no_retries": true,
            "no_resampling": true,
            "resampling_performed": false,
            "pass_only_authorizes_separate_opt_in_full_path_plumbing": true
        },
        "cross_runtime_rule": {
            "candidate_is_apxinf_specific_not_llama_equivalent": true,
            "frozen_llama_cpp_comparison_unchanged_until_full_path_acceptance": true,
            "current_apxinf_metal_W8_tps": 66.33672849846097,
            "current_llama_cpp_metal_Q8_0_tps": 70.66394293407429,
            "current_llama_cpp_advantage_percent": 6.52310497300709,
            "llama_cpp_commit": "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
            "frozen_summary": "crates/apxinf-metal/evidence/llama-cpp/qwen35-0.8b-apxinf-vs-llamacpp-raw13-free128-diagnostic-summary-v2-20260825.json",
            "rerun_performed_by_this_primitive_screen": false
        },
        "host_preflight": host_preflight,
        "custody": {
            "start": custody_start,
            "end": custody_end,
            "git_start": git_start,
            "git_end": git_end,
            "unchanged_during_sampling": custody_unchanged,
            "raw_receipt_publication_occurs_after_end_custody_attempt": true,
            "clean_unchanged_end_custody_before_publication": custody_unchanged
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
        return Err("GDN recurrent local-reuse primitive rejected; receipt was published".into());
    }
    Ok(())
}

fn next_xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn dyadic(state: &mut u64, radius: i32, denominator: f32) -> f32 {
    let width = (2 * radius + 1) as u64;
    let signed = ((next_xorshift(state) >> 32) % width) as i32 - radius;
    signed as f32 / denominator
}

fn seeded_fixture(index: usize) -> Fixture {
    let mut processed = vec![0.0f32; PROCESSED_TRACE_ELEMENTS];
    let mut processed_rng = 0x1319_8a2e_0370_7344 ^ index as u64;
    let key_width = QWEN35_GDN_KEY_HEADS_V1 * QWEN35_GDN_KEY_DIM_V1;
    for seam in 0..QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 {
        let seam_base = seam * QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1;
        for head in 0..QWEN35_GDN_KEY_HEADS_V1 {
            let query_base = seam_base + head * QWEN35_GDN_KEY_DIM_V1;
            let key_base = seam_base + key_width + head * QWEN35_GDN_KEY_DIM_V1;
            let value_base = seam_base + 2 * key_width + head * QWEN35_GDN_VALUE_DIM_V1;
            for element in 0..QWEN35_GDN_KEY_DIM_V1 {
                processed[query_base + element] = dyadic(&mut processed_rng, 64, 2048.0);
                processed[key_base + element] = dyadic(&mut processed_rng, 64, 2048.0);
                processed[value_base + element] = dyadic(&mut processed_rng, 128, 1024.0);
            }
            for row_base in [query_base, key_base, value_base] {
                processed[row_base] = 0.0;
                processed[row_base + 1] = -0.0;
            }
        }
    }

    let mut projected_rng = 0xa409_3822_299f_31d0 ^ ((index as u64) << 32);
    let mut projected = (0..PROJECTED_TRACE_ELEMENTS)
        .map(|_| dyadic(&mut projected_rng, 64, 1024.0))
        .collect::<Vec<_>>();
    let mut a_log = vec![0.0f32; HEAD_SCALAR_TRACE_ELEMENTS];
    let mut dt_bias = vec![0.0f32; HEAD_SCALAR_TRACE_ELEMENTS];
    let a_offset = QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1
        + QWEN35_GDN_VALUE_HEADS_V1 * QWEN35_GDN_VALUE_DIM_V1;
    let b_offset = a_offset + QWEN35_GDN_VALUE_HEADS_V1;
    for seam in 0..QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 {
        for head in 0..QWEN35_GDN_VALUE_HEADS_V1 {
            let j = seam * QWEN35_GDN_VALUE_HEADS_V1 + head;
            a_log[j] = A_LOG_TABLE[j % A_LOG_TABLE.len()];
            dt_bias[j] = DT_BIAS_TABLE[j % DT_BIAS_TABLE.len()];
            let coefficient_index = index + j;
            let target_gate = TARGET_GATE_TABLE[coefficient_index % TARGET_GATE_TABLE.len()];
            let projected_a = target_gate - dt_bias[j];
            assert_eq!((projected_a + dt_bias[j]).to_bits(), target_gate.to_bits());
            let seam_base = seam * QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V1;
            projected[seam_base + a_offset + head] = projected_a;
            projected[seam_base + b_offset + head] = B_TABLE[coefficient_index % B_TABLE.len()];
        }
    }

    let mut state_rng = 0x082e_fa98_ec4e_6c89 ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut state = (0..RECURRENT_TRACE_ELEMENTS)
        .map(|_| dyadic(&mut state_rng, 64, 2048.0))
        .collect::<Vec<_>>();
    for seam in 0..QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 {
        let seam_base = seam * QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1;
        for head in 0..QWEN35_GDN_VALUE_HEADS_V1 {
            let head_base = seam_base + head * QWEN35_GDN_KEY_DIM_V1 * QWEN35_GDN_VALUE_DIM_V1;
            for key in 0..QWEN35_GDN_KEY_DIM_V1 {
                let row_base = head_base + key * QWEN35_GDN_VALUE_DIM_V1;
                state[row_base] = 0.0;
                state[row_base + 1] = -0.0;
            }
        }
    }

    Fixture {
        processed,
        projected,
        a_log,
        dt_bias,
        state,
    }
}

fn hash_f32_slice(hasher: &mut Sha256, label: &[u8], values: &[f32]) {
    hasher.update(label);
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

fn hash_fixtures(fixtures: &[Fixture]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"qwen35-gdn-recurrent-local-reuse-fixtures-v1");
    for dimension in [
        QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1,
        QWEN35_GDN_KEY_HEADS_V1,
        QWEN35_GDN_VALUE_HEADS_V1,
        QWEN35_GDN_KEY_DIM_V1,
        QWEN35_GDN_VALUE_DIM_V1,
        QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1,
        QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V1,
        QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1,
        QWEN35_GDN_CORE_ELEMENTS_PER_SEAM_V1,
    ] {
        hasher.update((dimension as u64).to_le_bytes());
    }
    for table in [
        &A_LOG_TABLE[..],
        &DT_BIAS_TABLE,
        &B_TABLE,
        &TARGET_GATE_TABLE,
    ] {
        hash_f32_slice(&mut hasher, b"coefficient_table", table);
    }
    for (index, fixture) in fixtures.iter().enumerate() {
        hasher.update(b"fixture");
        hasher.update((index as u64).to_le_bytes());
        hash_f32_slice(&mut hasher, b"processed", &fixture.processed);
        hash_f32_slice(&mut hasher, b"projected", &fixture.projected);
        hash_f32_slice(&mut hasher, b"a_log", &fixture.a_log);
        hash_f32_slice(&mut hasher, b"dt_bias", &fixture.dt_bias);
        hash_f32_slice(&mut hasher, b"state", &fixture.state);
    }
    format!("{:x}", hasher.finalize())
}

fn validate_fixture_branch_coverage(fixtures: &[Fixture]) -> Result<Value, Box<dyn Error>> {
    let a_offset = QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1
        + QWEN35_GDN_VALUE_HEADS_V1 * QWEN35_GDN_VALUE_DIM_V1;
    let b_offset = a_offset + QWEN35_GDN_VALUE_HEADS_V1;
    let mut all_beta = [0usize; 2];
    let mut all_softplus = [0usize; 3];
    let mut per_fixture = Vec::with_capacity(fixtures.len());
    for (fixture_index, fixture) in fixtures.iter().enumerate() {
        let mut beta = [0usize; 2];
        let mut softplus = [0usize; 3];
        for seam in 0..QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 {
            let projected_base = seam * QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V1;
            for head in 0..QWEN35_GDN_VALUE_HEADS_V1 {
                let scalar_index = seam * QWEN35_GDN_VALUE_HEADS_V1 + head;
                let b = fixture.projected[projected_base + b_offset + head];
                beta[usize::from(b >= 0.0)] += 1;
                let gate = fixture.projected[projected_base + a_offset + head]
                    + fixture.dt_bias[scalar_index];
                let branch = if gate < -20.0 {
                    0
                } else if gate > 20.0 {
                    2
                } else {
                    1
                };
                softplus[branch] += 1;
                let expected_gate =
                    TARGET_GATE_TABLE[(fixture_index + scalar_index) % TARGET_GATE_TABLE.len()];
                if gate.to_bits() != expected_gate.to_bits() {
                    return Err(format!(
                        "fixture {fixture_index} gate reconstruction changed at seam {seam} head {head}"
                    )
                    .into());
                }
            }
        }
        if beta != [108, 180] || softplus != [64, 160, 64] {
            return Err(format!(
                "fixture {fixture_index} branch ledger mismatch: beta={beta:?} softplus={softplus:?}"
            )
            .into());
        }
        for index in 0..2 {
            all_beta[index] += beta[index];
        }
        for index in 0..3 {
            all_softplus[index] += softplus[index];
        }
        per_fixture.push(json!({
            "fixture_index": fixture_index,
            "beta_b_less_than_zero": beta[0],
            "beta_b_greater_than_or_equal_zero_including_negative_zero": beta[1],
            "softplus_gate_less_than_negative_20": softplus[0],
            "softplus_middle_inclusive": softplus[1],
            "softplus_gate_greater_than_20": softplus[2]
        }));
    }
    if all_beta != [864, 1440] || all_softplus != [512, 1280, 512] {
        return Err("all-fixture coefficient branch ledger changed".into());
    }
    Ok(json!({
        "per_fixture": per_fixture,
        "all_fixtures": {
            "beta_negative": all_beta[0],
            "beta_nonnegative": all_beta[1],
            "softplus_low": all_softplus[0],
            "softplus_middle": all_softplus[1],
            "softplus_high": all_softplus[2]
        },
        "gate_reconstruction_f32_mismatches": 0
    }))
}

fn run_arm(
    arm: Arm,
    primitive: &mut MetalGdnRecurrentCount18PrimitiveV1,
    host_expected_successful_runs: &mut [u64; 3],
) -> Result<(), Box<dyn Error>> {
    primitive.run(arm.profile())?;
    host_expected_successful_runs[arm.index()] += 1;
    std::hint::black_box(arm.label());
    Ok(())
}

fn coordinate_json(tensor: &str, index: usize) -> Value {
    if tensor == "next_state" {
        let seam = index / QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1;
        let within_seam = index % QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1;
        let head_stride = QWEN35_GDN_KEY_DIM_V1 * QWEN35_GDN_VALUE_DIM_V1;
        let head = within_seam / head_stride;
        let within_head = within_seam % head_stride;
        json!({
            "seam": seam,
            "head": head,
            "key": within_head / QWEN35_GDN_VALUE_DIM_V1,
            "value": within_head % QWEN35_GDN_VALUE_DIM_V1
        })
    } else {
        let seam = index / QWEN35_GDN_CORE_ELEMENTS_PER_SEAM_V1;
        let within_seam = index % QWEN35_GDN_CORE_ELEMENTS_PER_SEAM_V1;
        json!({
            "seam": seam,
            "head": within_seam / QWEN35_GDN_VALUE_DIM_V1,
            "value": within_seam % QWEN35_GDN_VALUE_DIM_V1
        })
    }
}

fn snapshot_tensors(snapshot: &GdnRecurrentCount18SnapshotV1) -> [(&'static str, &[f32]); 2] {
    [
        ("next_state", &snapshot.next_state),
        ("core", &snapshot.core),
    ]
}

fn exactness_check(
    primitive: &mut MetalGdnRecurrentCount18PrimitiveV1,
    fixtures: &[Fixture],
    host_expected_successful_runs: &mut [u64; 3],
    harness_ledger: &mut HarnessLedger,
) -> Result<Value, Value> {
    let order = BLOCK_ORDERS[0];
    let mut finite_checks = 0usize;
    let mut compared_elements = 0usize;
    let mut state_comparisons = 0usize;
    let mut core_comparisons = 0usize;
    if let Err(error) = primitive.verify_invalid_raw_selector_fail_closed() {
        return Err(json!({
            "phase": "invalid_selector_without_snapshot",
            "error": error.to_string(),
            "host_expected_successful_runs": counts_u64_json(host_expected_successful_runs)
        }));
    }
    harness_ledger.invalid_selector_checks += 1;
    for (fixture_index, fixture) in fixtures.iter().enumerate() {
        if let Err(error) = fixture.stage(primitive) {
            return Err(json!({
                "phase": "fixture_stage",
                "fixture_index": fixture_index,
                "error": error.to_string(),
                "host_expected_successful_runs": counts_u64_json(host_expected_successful_runs)
            }));
        }
        harness_ledger.fixture_stages += 1;
        let mut oracle: Option<GdnRecurrentCount18SnapshotV1> = None;
        for (call_index, arm) in order.into_iter().enumerate() {
            if let Err(error) = primitive.poison_outputs_for_correctness() {
                return Err(json!({
                    "phase": "output_poison",
                    "fixture_index": fixture_index,
                    "call_index": call_index,
                    "arm": arm.label(),
                    "error": error.to_string(),
                    "host_expected_successful_runs": counts_u64_json(host_expected_successful_runs)
                }));
            }
            harness_ledger.output_poison_calls += 1;
            if let Err(error) = run_arm(arm, primitive, host_expected_successful_runs) {
                return Err(json!({
                    "phase": "aggregate_run",
                    "fixture_index": fixture_index,
                    "call_index": call_index,
                    "arm": arm.label(),
                    "error": error.to_string(),
                    "host_expected_successful_runs": counts_u64_json(host_expected_successful_runs)
                }));
            }
            let snapshot = match primitive.snapshot() {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    return Err(json!({
                        "phase": "output_snapshot",
                        "fixture_index": fixture_index,
                        "call_index": call_index,
                        "arm": arm.label(),
                        "error": error.to_string(),
                        "host_expected_successful_runs": counts_u64_json(host_expected_successful_runs)
                    }));
                }
            };
            harness_ledger.output_snapshot_calls += 1;
            for (tensor_kind, tensor) in snapshot_tensors(&snapshot) {
                for (element, &actual) in tensor.iter().enumerate() {
                    finite_checks += 1;
                    if !actual.is_finite() {
                        return Ok(json!({
                            "passed": false,
                            "performance_authorized": false,
                            "order_per_fixture": order.map(Arm::short),
                            "finite_checks_before_failure": finite_checks,
                            "compared_elements_before_failure": compared_elements,
                            "first_mismatch": {
                                "kind": "non_finite",
                                "fixture_index": fixture_index,
                                "call_index": call_index,
                                "arm": arm.label(),
                                "tensor": tensor_kind,
                                "linear_index": element,
                                "coordinate": coordinate_json(tensor_kind, element),
                                "actual_debug": format!("{actual:?}"),
                                "actual_bits": actual.to_bits()
                            }
                        }));
                    }
                }
            }
            if let Some(expected) = &oracle {
                for ((tensor_kind, expected_tensor), (_, actual_tensor)) in
                    snapshot_tensors(expected)
                        .into_iter()
                        .zip(snapshot_tensors(&snapshot))
                {
                    for (element, (&left, &right)) in
                        expected_tensor.iter().zip(actual_tensor).enumerate()
                    {
                        compared_elements += 1;
                        if tensor_kind == "next_state" {
                            state_comparisons += 1;
                        } else {
                            core_comparisons += 1;
                        }
                        if left.to_bits() != right.to_bits() {
                            return Ok(json!({
                                "passed": false,
                                "performance_authorized": false,
                                "order_per_fixture": order.map(Arm::short),
                                "finite_checks_before_failure": finite_checks,
                                "compared_elements_before_failure": compared_elements,
                                "first_mismatch": {
                                    "kind": "to_bits",
                                    "fixture_index": fixture_index,
                                    "call_index": call_index,
                                    "arm": arm.label(),
                                    "tensor": tensor_kind,
                                    "linear_index": element,
                                    "coordinate": coordinate_json(tensor_kind, element),
                                    "expected_value": left,
                                    "actual_value": right,
                                    "expected_bits": left.to_bits(),
                                    "actual_bits": right.to_bits()
                                }
                            }));
                        }
                    }
                }
            } else {
                oracle = Some(snapshot);
            }
        }
        if let Err(error) = fixture.verify_unchanged(primitive) {
            return Err(json!({
                "phase": "staged_input_custody",
                "fixture_index": fixture_index,
                "error": error.to_string(),
                "host_expected_successful_runs": counts_u64_json(host_expected_successful_runs)
            }));
        }
        harness_ledger.staged_input_verifications += 1;
    }
    if let Err(error) = primitive.verify_invalid_raw_selector_fail_closed() {
        return Err(json!({
            "phase": "invalid_selector_with_snapshot",
            "error": error.to_string(),
            "host_expected_successful_runs": counts_u64_json(host_expected_successful_runs)
        }));
    }
    harness_ledger.selector_probe_snapshot_materializations += 2;
    harness_ledger.invalid_selector_checks += 1;
    if finite_checks != 228_261_888
        || compared_elements != 190_218_240
        || state_comparisons != 188_743_680
        || core_comparisons != 1_474_560
    {
        return Err(json!({
            "phase": "exactness_ledger",
            "error": format!("exactness ledger mismatch: finite={finite_checks} compared={compared_elements} state={state_comparisons} core={core_comparisons}"),
            "host_expected_successful_runs": counts_u64_json(host_expected_successful_runs)
        }));
    }
    Ok(json!({
        "passed": true,
        "performance_authorized": true,
        "fixture_count": fixtures.len(),
        "order_per_fixture": order.map(Arm::short),
        "calls_total": fixtures.len() * order.len(),
        "calls_per_arm": 16,
        "combined_output_elements_per_call": COMBINED_OUTPUT_ELEMENTS,
        "finite_checks": finite_checks,
        "explicit_A_oracle_to_bits_comparisons": compared_elements,
        "explicit_A_oracle_next_state_to_bits_comparisons": state_comparisons,
        "explicit_A_oracle_core_to_bits_comparisons": core_comparisons,
        "invalid_selector_snapshot_to_bits_comparisons": COMBINED_OUTPUT_ELEMENTS,
        "all_to_bits_comparisons": compared_elements + COMBINED_OUTPUT_ELEMENTS,
        "all_next_state_to_bits_comparisons": state_comparisons + RECURRENT_TRACE_ELEMENTS,
        "all_core_to_bits_comparisons": core_comparisons + CORE_TRACE_ELEMENTS,
        "all_outputs_finite": true,
        "all_outputs_match_first_A_to_bits": true,
        "outputs_poisoned_before_every_correctness_call": true,
        "staged_inputs_verified_bitwise_unchanged_after_each_fixture": true,
        "invalid_raw_selector_failed_closed_with_absent_and_present_snapshot_states": true
    }))
}

fn performance_attempt(
    primitive: &mut MetalGdnRecurrentCount18PrimitiveV1,
    fixture: &Fixture,
    host_expected_successful_runs: &mut [u64; 3],
    harness_ledger: &mut HarnessLedger,
) -> Result<Value, Value> {
    if let Err(error) = fixture.stage(primitive) {
        return Err(json!({
            "phase": "fixture_stage",
            "error": error.to_string(),
            "warmup_completed_calls": {"A":0,"B":0,"C":0},
            "completed_timed_blocks": [],
            "current_timed_block_partial": samples_partial_json(&empty_samples())
        }));
    }
    harness_ledger.fixture_stages += 1;

    let mut warmup_completed = [0usize; 3];
    for (block_index, order) in BLOCK_ORDERS.into_iter().enumerate() {
        for (cell_index, arm) in order.into_iter().enumerate() {
            for call_index in 0..CALLS_PER_CELL {
                if let Err(error) = run_arm(arm, primitive, host_expected_successful_runs) {
                    return Err(json!({
                        "phase": "warmup",
                        "block_index": block_index,
                        "order": order.map(Arm::short),
                        "cell_index": cell_index,
                        "call_index_within_cell": call_index,
                        "arm": arm.label(),
                        "error": error.to_string(),
                        "warmup_completed_calls": counts_json(&warmup_completed),
                        "completed_timed_blocks": [],
                        "current_timed_block_partial": samples_partial_json(&empty_samples())
                    }));
                }
                warmup_completed[arm.index()] += 1;
            }
        }
    }

    let mut timed_blocks: Vec<Samples> = Vec::with_capacity(BLOCK_ORDERS.len());
    for (block_index, order) in BLOCK_ORDERS.into_iter().enumerate() {
        match timed_block_capture(block_index, order, primitive, host_expected_successful_runs) {
            Ok(samples) => timed_blocks.push(samples),
            Err((partial, failure)) => {
                let completed = timed_blocks
                    .iter()
                    .enumerate()
                    .map(|(index, samples)| {
                        json!({"block_index":index,"samples":samples_partial_json(samples)})
                    })
                    .collect::<Vec<_>>();
                return Err(json!({
                    "phase": "timed",
                    "error": failure.get("error").cloned().unwrap_or(Value::String("unknown timed failure".to_owned())),
                    "failure": failure,
                    "warmup_completed_calls": counts_json(&warmup_completed),
                    "completed_timed_blocks": completed,
                    "current_timed_block_partial": samples_partial_json(&partial)
                }));
            }
        }
    }
    if let Err(error) = fixture.verify_unchanged(primitive) {
        return Err(json!({
            "phase": "post_timing_input_custody",
            "error": error.to_string(),
            "warmup_completed_calls": counts_json(&warmup_completed),
            "completed_timed_blocks": timed_blocks.iter().enumerate().map(|(index,samples)| json!({"block_index":index,"samples":samples_partial_json(samples)})).collect::<Vec<_>>()
        }));
    }
    harness_ledger.staged_input_verifications += 1;
    performance_json(&timed_blocks).map_err(|error| {
        json!({
            "phase": "statistics",
            "error": error.to_string(),
            "warmup_completed_calls": counts_json(&warmup_completed),
            "completed_timed_blocks": timed_blocks.iter().enumerate().map(|(index,samples)| json!({"block_index":index,"samples":samples_partial_json(samples)})).collect::<Vec<_>>()
        })
    })
}

fn timed_block_capture(
    block_index: usize,
    order: [Arm; 6],
    primitive: &mut MetalGdnRecurrentCount18PrimitiveV1,
    host_expected_successful_runs: &mut [u64; 3],
) -> Result<Samples, (Samples, Value)> {
    let mut samples = empty_samples();
    for (cell_index, arm) in order.into_iter().enumerate() {
        for call_index in 0..CALLS_PER_CELL {
            let started = Instant::now();
            if let Err(error) = run_arm(arm, primitive, host_expected_successful_runs) {
                let failure = json!({
                    "block_index": block_index,
                    "order": order.map(Arm::short),
                    "cell_index": cell_index,
                    "call_index_within_cell": call_index,
                    "arm": arm.label(),
                    "error": error.to_string()
                });
                return Err((samples, failure));
            }
            samples[arm.index()].push(started.elapsed().as_nanos());
        }
    }
    Ok(samples)
}

fn counts_json(counts: &[usize; 3]) -> Value {
    json!({"A":counts[0],"B":counts[1],"C":counts[2]})
}

fn counts_u64_json(counts: &[u64; 3]) -> Value {
    json!({"A":counts[0],"B":counts[1],"C":counts[2]})
}

fn harness_ledger_json(ledger: &HarnessLedger) -> Value {
    json!({
        "fixture_stage_calls": ledger.fixture_stages,
        "fixture_staging_bytes": ledger.fixture_stages * 19_911_168,
        "explicit_correctness_output_poison_calls": ledger.output_poison_calls,
        "explicit_correctness_output_poison_bytes": ledger.output_poison_calls * 19_021_824,
        "explicit_correctness_output_snapshot_calls": ledger.output_snapshot_calls,
        "explicit_correctness_output_snapshot_bytes": ledger.output_snapshot_calls * 19_021_824,
        "invalid_selector_probe_successful_snapshot_materializations": ledger.selector_probe_snapshot_materializations,
        "invalid_selector_probe_snapshot_bytes": ledger.selector_probe_snapshot_materializations * 19_021_824,
        "all_successful_snapshot_materializations": ledger.output_snapshot_calls + ledger.selector_probe_snapshot_materializations,
        "all_successful_snapshot_materialization_bytes": (ledger.output_snapshot_calls + ledger.selector_probe_snapshot_materializations) * 19_021_824,
        "snapshot_qualification": "the first absent-snapshot invalid-selector probe makes two rejected snapshot attempts with no output materialization; the final present-snapshot probe materializes before and after",
        "staged_input_verifications": ledger.staged_input_verifications,
        "invalid_selector_checks": ledger.invalid_selector_checks
    })
}

fn samples_partial_json(samples: &Samples) -> Value {
    json!({
        "A_raw_ns": samples[0],
        "B_raw_ns": samples[1],
        "C_raw_ns": samples[2],
        "A_completed_samples": samples[0].len(),
        "B_completed_samples": samples[1].len(),
        "C_completed_samples": samples[2].len()
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

fn improvement_from_medians(candidate: f64, baseline: f64) -> f64 {
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

fn pairwise_json(medians: &[f64; 3]) -> Value {
    json!({
        "B_over_A_percent": improvement_from_medians(medians[1], medians[0]),
        "C_over_A_percent": improvement_from_medians(medians[2], medians[0]),
        "C_over_B_percent": improvement_from_medians(medians[2], medians[1]),
        "B_over_C_percent": improvement_from_medians(medians[1], medians[2])
    })
}

fn performance_json(blocks: &[Samples]) -> Result<Value, Box<dyn Error>> {
    if blocks.len() != 3
        || blocks
            .iter()
            .any(|samples| samples.iter().any(|arm| arm.len() != 128))
    {
        return Err(
            "fixed performance schedule did not produce 128 samples per arm per block".into(),
        );
    }
    let block_medians: [[f64; 3]; 3] =
        std::array::from_fn(|block| std::array::from_fn(|arm| even_median(&blocks[block][arm])));
    let mut pooled = empty_samples();
    for block in blocks {
        for arm in 0..3 {
            pooled[arm].extend_from_slice(&block[arm]);
        }
    }
    if pooled.iter().any(|samples| samples.len() != 384) {
        return Err("fixed performance schedule did not produce 384 pooled samples per arm".into());
    }
    let pooled_medians: [f64; 3] = std::array::from_fn(|arm| even_median(&pooled[arm]));
    let pooled_c_over_b = improvement_from_medians(pooled_medians[2], pooled_medians[1]);
    let selected = if pooled_c_over_b >= C_SELECTION_THRESHOLD_PERCENT {
        Arm::C
    } else {
        Arm::B
    };
    let unselected = if selected == Arm::C { Arm::B } else { Arm::C };
    let selected_index = selected.index();
    let unselected_index = unselected.index();
    let selected_over_a_pooled = improvement_from_medians(
        pooled_medians[selected_index],
        pooled_medians[Arm::A.index()],
    );
    let selected_over_a_by_block: [f64; 3] = std::array::from_fn(|block| {
        improvement_from_medians(
            block_medians[block][selected_index],
            block_medians[block][Arm::A.index()],
        )
    });
    let selected_over_unselected_by_block: [f64; 3] = std::array::from_fn(|block| {
        improvement_from_medians(
            block_medians[block][selected_index],
            block_medians[block][unselected_index],
        )
    });
    let c_over_b_by_block: [f64; 3] = std::array::from_fn(|block| {
        improvement_from_medians(
            block_medians[block][Arm::C.index()],
            block_medians[block][Arm::B.index()],
        )
    });
    let pooled_threshold_passed = selected_over_a_pooled >= KEEP_THRESHOLD_PERCENT;
    let all_blocks_positive = selected_over_a_by_block.iter().all(|&value| value > 0.0);
    let simplicity_guard_passed = selected_over_unselected_by_block
        .iter()
        .all(|&value| value > SIMPLICITY_BLOCK_FLOOR_PERCENT);
    let c_positive_block_count = c_over_b_by_block
        .iter()
        .filter(|&&value| value > 0.0)
        .count();
    let c_consistency_passed = selected != Arm::C || c_positive_block_count >= 2;
    let passed = pooled_threshold_passed
        && all_blocks_positive
        && simplicity_guard_passed
        && c_consistency_passed;
    let block_json = (0..3)
        .map(|index| {
            json!({
                "block_index": index,
                "order": BLOCK_ORDERS[index].map(Arm::short),
                "A": sample_summary(&blocks[index][0]),
                "B": sample_summary(&blocks[index][1]),
                "C": sample_summary(&blocks[index][2]),
                "pairwise_improvement": pairwise_json(&block_medians[index]),
                "selected_over_A_percent": selected_over_a_by_block[index],
                "selected_over_unselected_percent": selected_over_unselected_by_block[index]
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "completed": true,
        "schedule": {
            "calls_per_cell": CALLS_PER_CELL,
            "warmup_orders": BLOCK_ORDERS.map(|order| order.map(Arm::short)),
            "timed_orders": BLOCK_ORDERS.map(|order| order.map(Arm::short)),
            "samples_per_arm_per_block": 128,
            "pooled_samples_per_arm": 384,
            "position_balanced": true,
            "transition_balanced_within_blocks_only": true,
            "cross_block_edges_outside_balance_claim": ["A_to_B","B_to_C"],
            "fixture_staged_outside_timing": true,
            "poisoning_and_snapshots_disabled_during_performance": true,
            "synchronous_wait_inside_each_timed_call": true
        },
        "blocks": block_json,
        "pooled": {
            "A": sample_summary(&pooled[0]),
            "B": sample_summary(&pooled[1]),
            "C": sample_summary(&pooled[2]),
            "pairwise_improvement": pairwise_json(&pooled_medians)
        },
        "selection": {
            "pooled_C_over_B_percent": pooled_c_over_b,
            "C_selection_threshold_percent": C_SELECTION_THRESHOLD_PERCENT,
            "selected_arm": selected.label(),
            "unselected_arm": unselected.label(),
            "no_fallback_applied": true,
            "selected_over_A_pooled_percent": selected_over_a_pooled,
            "selected_over_A_by_block_percent": selected_over_a_by_block,
            "selected_over_unselected_by_block_percent": selected_over_unselected_by_block,
            "C_over_B_by_block_percent": c_over_b_by_block,
            "C_over_B_positive_block_count": c_positive_block_count
        },
        "criteria": {
            "pooled_selected_over_A_at_least_5_percent": pooled_threshold_passed,
            "selected_over_A_strictly_positive_in_all_blocks": all_blocks_positive,
            "selected_vs_unselected_strictly_above_negative_0_5_percent_in_all_blocks": simplicity_guard_passed,
            "if_C_selected_C_over_B_positive_in_at_least_two_blocks": c_consistency_passed
        },
        "outlier_removal_performed": false,
        "retry_performed": false,
        "resampling_performed": false,
        "passed": passed
    }))
}

fn source_tradeoff_json() -> Value {
    json!({
        "A_uniform_prefix_reads_bytes": 1179648,
        "B_uniform_prefix_reads_bytes": 4608,
        "C_uniform_prefix_reads_bytes": 4608,
        "A_Q_K_reads_bytes": 56623104,
        "B_Q_K_reads_bytes": 56623104,
        "C_staged_Q_K_device_reads_bytes": 294912,
        "state_reads_bytes_all_arms": 37748736,
        "V_reads_bytes_all_arms": 147456,
        "next_state_writes_bytes_all_arms": 18874368,
        "core_writes_bytes_all_arms": 147456,
        "C_over_A_selected_device_read_reduction_bytes": 57503232,
        "A_launched_active_idle_threads": [73728,36864,36864],
        "B_launched_active_idle_threads": [36864,36864,0],
        "C_launched_active_idle_threads": [36864,36864,0],
        "source_expression_ledger_only": true,
        "physical_cache_DRAM_or_spill_claim": false
    })
}

fn validate_runtime_receipt(
    receipt: &GdnRecurrentCount18RuntimeReceiptV1,
    arm: Arm,
    expected_successful_runs: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    let expected = arm.profile();
    let expected_last = u32::from(receipt.successful_runs != 0);
    let expected_threads = expected.threads_per_threadgroup();
    let expected_launched = QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 as u32
        * QWEN35_GDN_VALUE_HEADS_V1 as u32
        * expected_threads;
    let expected_active = QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 as u32
        * QWEN35_GDN_VALUE_HEADS_V1 as u32
        * QWEN35_GDN_VALUE_DIM_V1 as u32;
    let expected_idle = expected_launched - expected_active;
    let processed_bytes = (PROCESSED_TRACE_ELEMENTS * size_of::<f32>()) as u64;
    let projected_bytes = (PROJECTED_TRACE_ELEMENTS * size_of::<f32>()) as u64;
    let scalar_bytes = (HEAD_SCALAR_TRACE_ELEMENTS * size_of::<f32>()) as u64;
    let recurrent_bytes = (RECURRENT_TRACE_ELEMENTS * size_of::<f32>()) as u64;
    let core_bytes = (CORE_TRACE_ELEMENTS * size_of::<f32>()) as u64;
    let persistent_bytes =
        processed_bytes + projected_bytes + 2 * scalar_bytes + 2 * recurrent_bytes + core_bytes;
    if receipt.requested_profile != expected
        || receipt.observed_profile != expected
        || receipt.requested_function_name != expected.expected_function_name()
        || receipt.observed_function_name != expected.expected_function_name()
        || receipt.seams_per_run != QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1 as u32
        || receipt.key_heads != QWEN35_GDN_KEY_HEADS_V1 as u32
        || receipt.value_heads != QWEN35_GDN_VALUE_HEADS_V1 as u32
        || receipt.key_dim != QWEN35_GDN_KEY_DIM_V1 as u32
        || receipt.value_dim != QWEN35_GDN_VALUE_DIM_V1 as u32
        || receipt.processed_elements_per_seam != QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1 as u32
        || receipt.projected_elements_per_seam != QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V1 as u32
        || receipt.recurrent_elements_per_seam != QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1 as u32
        || receipt.core_elements_per_seam != QWEN35_GDN_CORE_ELEMENTS_PER_SEAM_V1 as u32
        || receipt.threads_per_threadgroup != expected_threads
        || receipt.simdgroups_per_threadgroup != expected_threads / 32
        || receipt.pipeline_max_total_threads_per_threadgroup < expected_threads
        || receipt.pipeline_thread_execution_width != 32
        || receipt.pipeline_static_threadgroup_memory_bytes != arm.expected_pipeline_static_bytes()
        || receipt.source_declared_threadgroup_memory_bytes
            != expected.source_declared_threadgroup_memory_bytes()
        || receipt.dynamic_threadgroup_memory_bytes != 0
        || receipt.internal_threadgroup_barrier_sites_per_threadgroup
            != expected.internal_threadgroup_barrier_sites()
        || receipt.source_derived_internal_barrier_executions_per_run
            != expected.internal_threadgroup_barrier_sites() * 288
        || receipt.launched_threads_per_run != expected_launched
        || receipt.active_value_threads_per_run != expected_active
        || receipt.idle_threads_per_run != expected_idle
        || receipt.command_buffers_per_run != 1
        || receipt.compute_encoders_per_run != 1
        || receipt.kernel_dispatches_per_run != 18
        || receipt.threadgroups_per_run != 288
        || receipt.explicit_buffer_barriers_per_run != 18
        || receipt.commits_per_run != 1
        || receipt.waits_per_run != 1
        || !receipt.fixed_shape_host_validated
        || !receipt.input_output_buffers_non_overlapping
        || receipt.host_to_device_bytes_per_run != 0
        || receipt.device_to_host_bytes_per_run != 0
        || receipt.processed_buffer_bytes != processed_bytes
        || receipt.projected_buffer_bytes != projected_bytes
        || receipt.a_log_buffer_bytes != scalar_bytes
        || receipt.dt_bias_buffer_bytes != scalar_bytes
        || receipt.state_buffer_bytes != recurrent_bytes
        || receipt.next_state_buffer_bytes != recurrent_bytes
        || receipt.core_buffer_bytes != core_bytes
        || receipt.persistent_buffer_bytes_total != persistent_bytes
        || expected_successful_runs.is_some_and(|runs| receipt.successful_runs != runs)
        || receipt.last_observed_command_buffers != expected_last
        || receipt.last_observed_compute_encoders != expected_last
        || receipt.last_observed_kernel_dispatches != expected_last * 18
        || receipt.last_observed_threadgroups != expected_last * 288
        || receipt.last_observed_explicit_buffer_barriers != expected_last * 18
        || receipt.last_observed_launched_threads != expected_last * expected_launched
        || receipt.last_observed_active_value_threads != expected_last * expected_active
        || receipt.last_observed_idle_threads != expected_last * expected_idle
        || receipt.last_observed_commits != expected_last
        || receipt.last_observed_waits != expected_last
    {
        return Err(format!(
            "invalid live runtime receipt for {}: {receipt:?}",
            arm.label()
        )
        .into());
    }
    Ok(())
}

fn capture_final_receipt(
    primitive: &MetalGdnRecurrentCount18PrimitiveV1,
    arm: Arm,
    expected_successful_runs: Option<u64>,
    sampled_attempt_failures: &mut Vec<String>,
) -> (Value, bool) {
    match primitive.runtime_receipt(arm.profile()) {
        Ok(receipt) => match validate_runtime_receipt(&receipt, arm, expected_successful_runs) {
            Ok(()) => (runtime_receipt_json(&receipt), true),
            Err(error) => {
                let message = format!("final {} runtime receipt validation: {error}", arm.label());
                sampled_attempt_failures.push(message.clone());
                let mut value = runtime_receipt_json(&receipt);
                value
                    .as_object_mut()
                    .expect("runtime receipt JSON is an object")
                    .insert("validation_error".to_owned(), Value::String(message));
                (value, false)
            }
        },
        Err(error) => {
            let message = format!("final {} runtime receipt read: {error}", arm.label());
            sampled_attempt_failures.push(message.clone());
            (json!({"available":false,"error":message}), false)
        }
    }
}

fn runtime_receipt_json(receipt: &GdnRecurrentCount18RuntimeReceiptV1) -> Value {
    json!({
        "requested_profile": profile_label(receipt.requested_profile),
        "observed_profile": profile_label(receipt.observed_profile),
        "requested_function_name": receipt.requested_function_name,
        "observed_function_name": receipt.observed_function_name,
        "seams_per_run": receipt.seams_per_run,
        "key_heads": receipt.key_heads,
        "value_heads": receipt.value_heads,
        "key_dim": receipt.key_dim,
        "value_dim": receipt.value_dim,
        "processed_elements_per_seam": receipt.processed_elements_per_seam,
        "projected_elements_per_seam": receipt.projected_elements_per_seam,
        "recurrent_elements_per_seam": receipt.recurrent_elements_per_seam,
        "core_elements_per_seam": receipt.core_elements_per_seam,
        "threads_per_threadgroup": receipt.threads_per_threadgroup,
        "simdgroups_per_threadgroup": receipt.simdgroups_per_threadgroup,
        "pipeline_max_total_threads_per_threadgroup": receipt.pipeline_max_total_threads_per_threadgroup,
        "pipeline_thread_execution_width": receipt.pipeline_thread_execution_width,
        "pipeline_static_threadgroup_memory_bytes": receipt.pipeline_static_threadgroup_memory_bytes,
        "source_declared_threadgroup_memory_bytes": receipt.source_declared_threadgroup_memory_bytes,
        "dynamic_threadgroup_memory_bytes": receipt.dynamic_threadgroup_memory_bytes,
        "internal_threadgroup_barrier_sites_per_threadgroup": receipt.internal_threadgroup_barrier_sites_per_threadgroup,
        "source_derived_internal_barrier_executions_per_run": receipt.source_derived_internal_barrier_executions_per_run,
        "internal_barrier_count_is_source_derived_not_hardware_measured": true,
        "launched_threads_per_run": receipt.launched_threads_per_run,
        "active_value_threads_per_run": receipt.active_value_threads_per_run,
        "idle_threads_per_run": receipt.idle_threads_per_run,
        "command_buffers_per_run": receipt.command_buffers_per_run,
        "compute_encoders_per_run": receipt.compute_encoders_per_run,
        "kernel_dispatches_per_run": receipt.kernel_dispatches_per_run,
        "threadgroups_per_run": receipt.threadgroups_per_run,
        "explicit_buffer_barriers_per_run": receipt.explicit_buffer_barriers_per_run,
        "commits_per_run": receipt.commits_per_run,
        "waits_per_run": receipt.waits_per_run,
        "fixed_shape_host_validated": receipt.fixed_shape_host_validated,
        "input_output_buffers_non_overlapping": receipt.input_output_buffers_non_overlapping,
        "host_to_device_bytes_per_run": receipt.host_to_device_bytes_per_run,
        "device_to_host_bytes_per_run": receipt.device_to_host_bytes_per_run,
        "zero_copy_qualification": "zero explicit bridge memcpy inside run(), not zero GPU or unified-memory traffic",
        "buffers": {
            "processed_bytes": receipt.processed_buffer_bytes,
            "projected_bytes": receipt.projected_buffer_bytes,
            "a_log_bytes": receipt.a_log_buffer_bytes,
            "dt_bias_bytes": receipt.dt_bias_buffer_bytes,
            "state_bytes": receipt.state_buffer_bytes,
            "next_state_bytes": receipt.next_state_buffer_bytes,
            "core_bytes": receipt.core_buffer_bytes,
            "persistent_total_bytes": receipt.persistent_buffer_bytes_total
        },
        "successful_runs": receipt.successful_runs,
        "derived_successful_totals": {
            "command_buffers": receipt.successful_runs * receipt.command_buffers_per_run as u64,
            "compute_encoders": receipt.successful_runs * receipt.compute_encoders_per_run as u64,
            "kernel_dispatches": receipt.successful_runs * receipt.kernel_dispatches_per_run as u64,
            "threadgroups": receipt.successful_runs * receipt.threadgroups_per_run as u64,
            "explicit_buffer_barriers": receipt.successful_runs * receipt.explicit_buffer_barriers_per_run as u64,
            "source_derived_internal_barrier_executions": receipt.successful_runs * receipt.source_derived_internal_barrier_executions_per_run as u64,
            "launched_threads": receipt.successful_runs * receipt.launched_threads_per_run as u64,
            "active_value_threads": receipt.successful_runs * receipt.active_value_threads_per_run as u64,
            "idle_threads": receipt.successful_runs * receipt.idle_threads_per_run as u64,
            "commits": receipt.successful_runs * receipt.commits_per_run as u64,
            "waits": receipt.successful_runs * receipt.waits_per_run as u64
        },
        "last_observed": {
            "command_buffers": receipt.last_observed_command_buffers,
            "compute_encoders": receipt.last_observed_compute_encoders,
            "kernel_dispatches": receipt.last_observed_kernel_dispatches,
            "threadgroups": receipt.last_observed_threadgroups,
            "explicit_buffer_barriers": receipt.last_observed_explicit_buffer_barriers,
            "launched_threads": receipt.last_observed_launched_threads,
            "active_value_threads": receipt.last_observed_active_value_threads,
            "idle_threads": receipt.last_observed_idle_threads,
            "commits": receipt.last_observed_commits,
            "waits": receipt.last_observed_waits
        }
    })
}

fn profile_label(profile: GdnRecurrentProfileV1) -> &'static str {
    match profile {
        GdnRecurrentProfileV1::Legacy256 => "legacy-256",
        GdnRecurrentProfileV1::LeaderBroadcast128 => "leader-broadcast-128",
        GdnRecurrentProfileV1::QkStaged128 => "qk-staged-128",
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
        "primitive_bridge": file_identity(&manifest_dir.join("src/metal_gdn_recurrent_count18_profile_v1_bridge.mm"))?,
        "rust_api": file_identity(&manifest_dir.join("src/gdn_recurrent_profile_v1.rs"))?,
        "legacy_hash_regression_test": file_identity(&manifest_dir.join("src/linear_layer.rs"))?,
        "crate_root": file_identity(&manifest_dir.join("src/lib.rs"))?,
        "build_script": file_identity(&manifest_dir.join("build.rs"))?,
        "gate_example": file_identity(&manifest_dir.join("examples/qwen35_gdn_recurrent_local_reuse_abc_v1.rs"))?,
        "crate_manifest": file_identity(&manifest_dir.join("Cargo.toml"))?,
        "workspace_lock": file_identity(&workspace_dir.join("Cargo.lock"))?,
        "predeclaration": file_identity(&manifest_dir.join("evidence/next-hotspot/qwen35-gdn-recurrent-local-reuse-v1-predeclared-primitive-gate-v1-20260825.json"))?,
        "qwen_production_mapping": file_identity(&model_crate.join("src/qwen35/general.rs"))?,
        "pinned_source_lock": file_identity(&workspace_dir.join(".apxinf/onboarding/qwen35-0.8b/source-lock.json"))?,
        "standalone_gdn_bridge": file_identity(&manifest_dir.join("src/metal_w8_gdn_bridge.mm"))?,
        "standalone_linear_bridge": file_identity(&manifest_dir.join("src/metal_w8_linear_layer_bridge.mm"))?,
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
        "primitive_bridge": sha256_bytes(include_bytes!("../src/metal_gdn_recurrent_count18_profile_v1_bridge.mm")),
        "rust_api": sha256_bytes(include_bytes!("../src/gdn_recurrent_profile_v1.rs")),
        "legacy_hash_regression_test": sha256_bytes(include_bytes!("../src/linear_layer.rs")),
        "crate_root": sha256_bytes(include_bytes!("../src/lib.rs")),
        "build_script": sha256_bytes(include_bytes!("../build.rs")),
        "gate_example": sha256_bytes(include_bytes!("qwen35_gdn_recurrent_local_reuse_abc_v1.rs")),
        "crate_manifest": sha256_bytes(include_bytes!("../Cargo.toml")),
        "workspace_lock": sha256_bytes(include_bytes!("../../../Cargo.lock")),
        "predeclaration": sha256_bytes(include_bytes!("../evidence/next-hotspot/qwen35-gdn-recurrent-local-reuse-v1-predeclared-primitive-gate-v1-20260825.json")),
        "qwen_production_mapping": sha256_bytes(include_bytes!("../../apxinf-model/src/qwen35/general.rs")),
        "pinned_source_lock": sha256_bytes(include_bytes!("../../../.apxinf/onboarding/qwen35-0.8b/source-lock.json")),
        "standalone_gdn_bridge": sha256_bytes(include_bytes!("../src/metal_w8_gdn_bridge.mm")),
        "standalone_linear_bridge": sha256_bytes(include_bytes!("../src/metal_w8_linear_layer_bridge.mm")),
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

fn require_production_consumers_legacy() -> Result<(), Box<dyn Error>> {
    for (label, source) in [
        (
            "metal_w8_gdn_bridge.mm",
            include_str!("../src/metal_w8_gdn_bridge.mm"),
        ),
        (
            "metal_w8_linear_layer_bridge.mm",
            include_str!("../src/metal_w8_linear_layer_bridge.mm"),
        ),
        (
            "metal_w8_linear_layer_stack3_bridge.mm",
            include_str!("../src/metal_w8_linear_layer_stack3_bridge.mm"),
        ),
        (
            "metal_w8_mlp_stack3_boundary_v1_bridge.mm",
            include_str!("../src/metal_w8_mlp_stack3_boundary_v1_bridge.mm"),
        ),
    ] {
        if !source.contains("@\"gdn_recurrent_update\"")
            || source.contains("gdn_recurrent_update_leader_broadcast_v1")
            || source.contains("gdn_recurrent_update_qk_staged_v1")
        {
            return Err(format!(
                "production consumer {label} no longer selects only legacy gdn_recurrent_update"
            )
            .into());
        }
    }
    Ok(())
}

fn host_preflight(candidate_commit: &str) -> Value {
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
        "classification": "diagnostic context only; this primitive screen is never formal promotion evidence",
        "hardware_model": system("sysctl", &["-n","hw.model"]),
        "cpu_brand": system("sysctl", &["-n","machdep.cpu.brand_string"]),
        "os_build": system("sw_vers", &["-buildVersion"]),
        "os_version": system("sw_vers", &["-productVersion"]),
        "rustc_version": system("rustc", &["--version"]),
        "cargo_version": system("cargo", &["--version"]),
        "clang_version": system("xcrun", &["clang","--version"]),
        "metal_tool_version": system("xcrun", &["metal","--version"]),
        "display_and_metal_devices": system("system_profiler", &["SPDisplaysDataType"]),
        "uptime": system("uptime", &[]),
        "top_processes_by_cpu": top_processes,
        "user_or_system_processes_terminated": false,
        "expected_release_build": format!("APXINF_CANDIDATE_COMMIT={candidate_commit} cargo build --release -p apxinf-metal --example qwen35_gdn_recurrent_local_reuse_abc_v1")
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
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "custody path is not a direct regular file: {}",
            path.display()
        )
        .into());
    }
    let bytes = std::fs::read(path)?;
    #[cfg(unix)]
    let hard_link_count = {
        use std::os::unix::fs::MetadataExt;
        metadata.nlink()
    };
    #[cfg(not(unix))]
    let hard_link_count = 1u64;
    if hard_link_count != 1 {
        return Err(format!(
            "custody path has {hard_link_count} hard links instead of one: {}",
            path.display()
        )
        .into());
    }
    Ok(json!({
        "path": std::fs::canonicalize(path)?,
        "size": metadata.len(),
        "sha256": sha256_bytes(&bytes),
        "regular_direct_file": true,
        "hard_link_count": hard_link_count
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

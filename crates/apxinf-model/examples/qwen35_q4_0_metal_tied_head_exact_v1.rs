//! Real-checkpoint exact-candidate gate for the isolated Metal Q4_0 tied head.
//!
//! The gate reuses the CPU coverage-v1 suppressed-free128 fixture, computes a
//! live CPU-Q4 K=4 trajectory, and requires every Metal candidate array to be
//! exactly equal. It records no timing and does not wire the primitive into
//! production decoding.

use std::error::Error;
use std::ffi::OsString;
use std::path::PathBuf;

use apxinf_metal::{MetalQ4_0TiedHeadV1, Q4_0TiedHeadBufferLedgerV1, Q4_0_TIED_HEAD_TOP_K_V1};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[allow(dead_code)]
#[path = "qwen35_q4_0_tied_head_coverage_v1.rs"]
mod cpu_coverage_v1;

const FORMAT: &str = "apxinf-qwen35-q4_0-metal-tied-head-exact-candidate-v1";
const QUALIFICATION: &str = "NON_FORMAL_CORRECTNESS_GATE_NO_PERFORMANCE_RESULT";
const OUTPUT_TOKENS: usize = 128;
const TEACHER_PREFILL_TOKENS: usize = 12;
const EXPECTED_VOCAB_SIZE: usize = 248_320;
const EXPECTED_HIDDEN_SIZE: usize = 1_024;
const EXCLUDED_EOG_TOKEN_IDS: [u32; 5] = [248044, 248046, 248063, 248064, 248065];
const SPOT_CHECK_STEPS_ZERO_BASED: [usize; 2] = [11, 46];

// These anchors come from the separately committed CPU coverage-v1 receipt.
// Matching them proves this gate reused that exact checkpoint, packed Q4_0
// matrix, teacher inputs, and normalized hidden trajectory; they are not
// candidate fallbacks.
const EXPECTED_CPU_F32_REFERENCE_SHA256: &str =
    "5bba10f53b153bb6a7d62efea7e0b6b6cb1b650c435e993c0fd171cd4e1b2f0a";
const EXPECTED_TEACHER_INPUT_SHA256: &str =
    "20ebe7bb349d8803a7a73e445590460a355e20486ffe36717b70cae815f5d53e";
const EXPECTED_SAME_HIDDEN_F32_WINNER_SHA256: &str =
    "d36c8570e71953db5f5bc919b45108dee47a704b975c3b5785f0063519ce46d0";
const EXPECTED_SOURCE_W8_TOP4_SHA256: &str =
    "4b35e30839d8094be5f594682714d7c9ba3c00c2f778bd6d8313b1d8a02a0fa8";
const EXPECTED_NORMALIZED_HIDDEN_F32_LE_SHA256: &str =
    "bb35ee83121c3cebbb073b02e18113d6ca93d7924482d70ab96ed23c75876030";
const EXPECTED_PACKED_Q4_0_SHA256: &str =
    "550b7491bf002a7c7c9aadbf125cef14ff742cbd9216303abf3e701a59d73939";
const EXPECTED_CPU_Q4_K4_TRAJECTORY_SHA256: &str =
    "08d34d5609621aa557dab0091be67e972c1c4e4cacf167438d00099a6dcdd857";

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
}

fn usage() -> &'static str {
    "Usage: qwen35_q4_0_metal_tied_head_exact_v1 --model-dir PATH"
}

fn main() {
    match real_main() {
        Ok(receipt) => {
            println!(
                "{}",
                serde_json::to_string(&receipt)
                    .expect("serialize Metal Q4_0 exact-candidate receipt")
            );
            if receipt.get("passed").and_then(Value::as_bool) != Some(true) {
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("{FORMAT} failed before a complete receipt could be emitted: {error}");
            std::process::exit(1);
        }
    }
}

fn real_main() -> Result<Value, Box<dyn Error>> {
    if !cfg!(target_os = "macos") {
        return Err("the Metal Q4_0 exact-candidate gate requires macOS".into());
    }
    let args = parse_args_from(std::env::args_os())?;
    let fixture = cpu_coverage_v1::build_q4_real_checkpoint_fixture_v1(&args.model_dir)?;
    validate_fixture_shape(&fixture)?;

    let cpu_f32_reference_sha256 =
        sha256_compact_json(&json!(fixture.cpu_f32_reference_token_ids))?;
    let teacher_input_sha256 = sha256_compact_json(&json!(fixture.teacher_input_token_ids))?;
    let same_hidden_f32_winner_sha256 =
        sha256_compact_json(&json!(fixture.same_hidden_f32_winner_token_ids))?;
    let source_w8_top4_sha256 = sha256_compact_json(&json!(fixture.source_w8_top4_token_ids))?;
    let normalized_hidden_f32_le_sha256 = sha256_f32_le(&fixture.normalized_hidden_f32)?;
    let per_step_hidden_sha256 = fixture
        .normalized_hidden_f32
        .chunks_exact(EXPECTED_HIDDEN_SIZE)
        .map(sha256_f32_le)
        .collect::<Result<Vec<_>, _>>()?;
    let per_step_hidden_sha256_trajectory_sha256 =
        sha256_compact_json(&json!(per_step_hidden_sha256))?;
    let packed_q4_0_bytes = fixture.q4_head.canonical_bytes_le();
    let packed_q4_0_sha256 = sha256_bytes(&packed_q4_0_bytes);

    let cpu_q4_k4 =
        cpu_coverage_v1::cpu_q4_k4_trajectory_v1(&fixture.q4_head, &fixture.normalized_hidden_f32)?;
    let cpu_q4_k4_sha256 = sha256_compact_json(&json!(cpu_q4_k4))?;

    let mut metal_head = MetalQ4_0TiedHeadV1::from_packed(&fixture.q4_head)?;
    let ledger = metal_head.buffer_ledger();
    let mut metal_q4_k4 = Vec::with_capacity(OUTPUT_TOKENS);
    for hidden in fixture
        .normalized_hidden_f32
        .chunks_exact(EXPECTED_HIDDEN_SIZE)
    {
        metal_q4_k4.push(metal_head.topk4_excluding(hidden, &EXCLUDED_EOG_TOKEN_IDS)?);
    }
    let metal_q4_k4_sha256 = sha256_compact_json(&json!(metal_q4_k4))?;

    let mismatches = mismatch_audit(
        &cpu_q4_k4,
        &metal_q4_k4,
        &fixture.teacher_input_token_ids,
        &fixture.same_hidden_f32_winner_token_ids,
        &per_step_hidden_sha256,
    )?;
    let f32_winner_misses = f32_winner_coverage_audit(
        &metal_q4_k4,
        &fixture.teacher_input_token_ids,
        &fixture.same_hidden_f32_winner_token_ids,
        &per_step_hidden_sha256,
    )?;
    let f32_winner_coverage_count = OUTPUT_TOKENS - f32_winner_misses.len();
    let spot_checks = SPOT_CHECK_STEPS_ZERO_BASED
        .iter()
        .map(|&step| {
            spot_check(
                step,
                &cpu_q4_k4,
                &metal_q4_k4,
                &fixture.teacher_input_token_ids,
                &fixture.same_hidden_f32_winner_token_ids,
                &per_step_hidden_sha256,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let source_contract_exact = cpu_f32_reference_sha256 == EXPECTED_CPU_F32_REFERENCE_SHA256
        && teacher_input_sha256 == EXPECTED_TEACHER_INPUT_SHA256
        && same_hidden_f32_winner_sha256 == EXPECTED_SAME_HIDDEN_F32_WINNER_SHA256
        && source_w8_top4_sha256 == EXPECTED_SOURCE_W8_TOP4_SHA256
        && normalized_hidden_f32_le_sha256 == EXPECTED_NORMALIZED_HIDDEN_F32_LE_SHA256
        && packed_q4_0_sha256 == EXPECTED_PACKED_Q4_0_SHA256
        && cpu_q4_k4_sha256 == EXPECTED_CPU_Q4_K4_TRAJECTORY_SHA256;
    let exact_candidate_match = mismatches.is_empty() && cpu_q4_k4_sha256 == metal_q4_k4_sha256;
    let complete_f32_winner_coverage =
        f32_winner_misses.is_empty() && f32_winner_coverage_count == OUTPUT_TOKENS;
    let spot_checks_pass = spot_checks.iter().all(|check| {
        check.get("cpu_metal_exact_match").and_then(Value::as_bool) == Some(true)
            && check.get("f32_winner_covered").and_then(Value::as_bool) == Some(true)
    });
    let passed = source_contract_exact
        && exact_candidate_match
        && complete_f32_winner_coverage
        && spot_checks_pass;
    let decision = if passed {
        "GO_FOR_Q4_0_METAL_EXACT_CANDIDATE_CORRECTNESS_ONLY"
    } else {
        "NO_GO"
    };

    Ok(json!({
        "format": FORMAT,
        "schema_version": 1,
        "qualification": QUALIFICATION,
        "claim_boundary": "real-checkpoint CPU-Q4 versus isolated Metal-Q4 exact K=4 candidate equality on 128 fixed teacher hiddens only; no production decode integration, latency, throughput, or runtime ranking claim",
        "model": {
            "model_dir": fixture.model_dir.display().to_string(),
            "expected_family": "Qwen/Qwen3.5-0.8B",
            "expected_vocabulary_size": EXPECTED_VOCAB_SIZE,
            "expected_hidden_size": EXPECTED_HIDDEN_SIZE,
            "tied_embedding_tensor": "model.language_model.embed_tokens.weight",
        },
        "workload": {
            "teacher_hidden_source": "the exact suppressed-free128 fused W8 teacher fixture shared with qwen35_q4_0_tied_head_coverage_v1",
            "teacher_prefill_token_count": TEACHER_PREFILL_TOKENS,
            "teacher_step_count": OUTPUT_TOKENS,
            "sampling": "greedy",
            "eog_termination": false,
            "excluded_eog_token_ids": EXCLUDED_EOG_TOKEN_IDS,
            "excluded_before_both_cpu_and_metal_top4": true,
            "fallbacks_or_retries": 0,
        },
        "implementations": {
            "cpu_q4": "PackedQ4_0RowsV1 live batch scores followed by deterministic topk_scores_excluding K=4",
            "metal_q4": "MetalQ4_0TiedHeadV1::topk4_excluding",
            "same_packed_q4_object": true,
            "same_hidden_slice_per_step": true,
            "candidate_order": "descending finite Q4_0 score, exact ties by lowest token ID",
            "production_decode_wiring_changed": false,
            "default_activation_changed": false,
        },
        "packed_q4_0_weight_evidence": {
            "encoding": "18 bytes per block: little-endian FP16 scale then 16 low/high-nibble bytes",
            "block_size": apxinf_metal::Q4_0_BLOCK_SIZE_V1,
            "packed_bytes_per_block": apxinf_metal::Q4_0_PACKED_BYTES_PER_BLOCK_V1,
            "packed_byte_count": packed_q4_0_bytes.len(),
            "sha256": packed_q4_0_sha256,
        },
        "hidden_evidence": {
            "shape": [OUTPUT_TOKENS, EXPECTED_HIDDEN_SIZE],
            "encoding": "each finite F32 as IEEE-754 bits in little-endian byte order",
            "normalized_hidden_f32_le_sha256": normalized_hidden_f32_le_sha256,
            "per_step_hidden_sha256_trajectory_sha256": per_step_hidden_sha256_trajectory_sha256,
        },
        "candidate_trajectory_evidence": {
            "step_count": OUTPUT_TOKENS,
            "candidate_count_per_step": Q4_0_TIED_HEAD_TOP_K_V1,
            "cpu_q4_k4_compact_json_sha256": cpu_q4_k4_sha256,
            "metal_q4_k4_compact_json_sha256": metal_q4_k4_sha256,
            "hashes_equal": cpu_q4_k4_sha256 == metal_q4_k4_sha256,
        },
        "exact_match_audit": {
            "mismatch_count": mismatches.len(),
            "first_mismatch": mismatches.first().cloned(),
            "all_mismatches": mismatches,
        },
        "same_hidden_f32_winner_coverage": {
            "covered_count": f32_winner_coverage_count,
            "required_count": OUTPUT_TOKENS,
            "miss_count": f32_winner_misses.len(),
            "first_miss": f32_winner_misses.first().cloned(),
            "all_misses": f32_winner_misses,
            "winner_token_ids_compact_json_sha256": same_hidden_f32_winner_sha256,
        },
        "spot_checks_zero_based": spot_checks,
        "metal_buffer_ledger": ledger_json(ledger),
        "frozen_source_hashes": {
            "cpu_f32_reference_token_ids_compact_json_sha256": cpu_f32_reference_sha256,
            "teacher_input_token_ids_compact_json_sha256": teacher_input_sha256,
            "source_w8_top4_token_ids_compact_json_sha256": source_w8_top4_sha256,
        },
        "admission": {
            "reused_cpu_coverage_v1_source_contract_exact": source_contract_exact,
            "cpu_metal_candidate_arrays_exact_at_all_128_steps": exact_candidate_match,
            "metal_k4_covers_same_hidden_f32_winner_128_of_128": complete_f32_winner_coverage,
            "step11_and_step46_checks_pass": spot_checks_pass,
            "decision": decision,
            "hard_gate": "any CPU/Metal candidate mismatch, source-hash drift, or F32-winner coverage miss forces NO_GO; no step/token fallback, tolerance, retry, or candidate substitution is permitted",
        },
        "performance": {
            "samples": 0,
            "latency_recorded": false,
            "throughput_recorded": false,
            "formal_result": false,
        },
        "passed": passed,
    }))
}

fn validate_fixture_shape(
    fixture: &cpu_coverage_v1::Q4RealCheckpointFixtureV1,
) -> Result<(), Box<dyn Error>> {
    if fixture.q4_head.rows() != EXPECTED_VOCAB_SIZE
        || fixture.q4_head.columns() != EXPECTED_HIDDEN_SIZE
        || fixture.cpu_f32_reference_token_ids.len() != OUTPUT_TOKENS
        || fixture.teacher_input_token_ids.len() != OUTPUT_TOKENS
        || fixture.same_hidden_f32_winner_token_ids.len() != OUTPUT_TOKENS
        || fixture.source_w8_top4_token_ids.len() != OUTPUT_TOKENS
        || fixture.normalized_hidden_f32.len() != OUTPUT_TOKENS * EXPECTED_HIDDEN_SIZE
    {
        return Err("real-checkpoint Q4_0 Metal gate fixture has an unexpected shape".into());
    }
    Ok(())
}

fn mismatch_audit(
    cpu: &[[u32; 4]],
    metal: &[[u32; 4]],
    teacher_inputs: &[u32],
    f32_winners: &[u32],
    hidden_hashes: &[String],
) -> Result<Vec<Value>, Box<dyn Error>> {
    validate_audit_lengths(cpu, metal, teacher_inputs, f32_winners, hidden_hashes)?;
    Ok(cpu
        .iter()
        .zip(metal)
        .enumerate()
        .filter_map(|(step, (cpu, metal))| {
            (cpu != metal).then(|| {
                json!({
                    "step_index_zero_based": step,
                    "step_number_one_based": step + 1,
                    "absolute_token_position": TEACHER_PREFILL_TOKENS + step,
                    "teacher_input_token_id": teacher_inputs[step],
                    "same_hidden_f32_winner_token_id": f32_winners[step],
                    "cpu_q4_k4": cpu,
                    "metal_q4_k4": metal,
                    "normalized_hidden_f32_le_sha256": hidden_hashes[step],
                })
            })
        })
        .collect())
}

fn f32_winner_coverage_audit(
    metal: &[[u32; 4]],
    teacher_inputs: &[u32],
    f32_winners: &[u32],
    hidden_hashes: &[String],
) -> Result<Vec<Value>, Box<dyn Error>> {
    if metal.len() != OUTPUT_TOKENS
        || teacher_inputs.len() != OUTPUT_TOKENS
        || f32_winners.len() != OUTPUT_TOKENS
        || hidden_hashes.len() != OUTPUT_TOKENS
    {
        return Err("Metal Q4_0 F32-winner coverage audit length mismatch".into());
    }
    Ok(metal
        .iter()
        .enumerate()
        .filter_map(|(step, candidates)| {
            let winner = f32_winners[step];
            (!candidates.contains(&winner)).then(|| {
                json!({
                    "step_index_zero_based": step,
                    "step_number_one_based": step + 1,
                    "absolute_token_position": TEACHER_PREFILL_TOKENS + step,
                    "teacher_input_token_id": teacher_inputs[step],
                    "same_hidden_f32_winner_token_id": winner,
                    "metal_q4_k4": candidates,
                    "normalized_hidden_f32_le_sha256": hidden_hashes[step],
                })
            })
        })
        .collect())
}

fn spot_check(
    step: usize,
    cpu: &[[u32; 4]],
    metal: &[[u32; 4]],
    teacher_inputs: &[u32],
    f32_winners: &[u32],
    hidden_hashes: &[String],
) -> Result<Value, Box<dyn Error>> {
    validate_audit_lengths(cpu, metal, teacher_inputs, f32_winners, hidden_hashes)?;
    if step >= OUTPUT_TOKENS {
        return Err(format!("spot-check step {step} is outside 128 teacher steps").into());
    }
    let winner = f32_winners[step];
    Ok(json!({
        "step_index_zero_based": step,
        "step_number_one_based": step + 1,
        "absolute_token_position": TEACHER_PREFILL_TOKENS + step,
        "teacher_input_token_id": teacher_inputs[step],
        "same_hidden_f32_winner_token_id": winner,
        "cpu_q4_k4": cpu[step],
        "metal_q4_k4": metal[step],
        "cpu_metal_exact_match": cpu[step] == metal[step],
        "f32_winner_q4_rank_one_based": metal[step].iter().position(|&token| token == winner).map(|rank| rank + 1),
        "f32_winner_covered": metal[step].contains(&winner),
        "normalized_hidden_f32_le_sha256": hidden_hashes[step],
    }))
}

fn validate_audit_lengths(
    cpu: &[[u32; 4]],
    metal: &[[u32; 4]],
    teacher_inputs: &[u32],
    f32_winners: &[u32],
    hidden_hashes: &[String],
) -> Result<(), Box<dyn Error>> {
    if cpu.len() != OUTPUT_TOKENS
        || metal.len() != OUTPUT_TOKENS
        || teacher_inputs.len() != OUTPUT_TOKENS
        || f32_winners.len() != OUTPUT_TOKENS
        || hidden_hashes.len() != OUTPUT_TOKENS
    {
        return Err("Metal Q4_0 exact-candidate audit length mismatch".into());
    }
    Ok(())
}

fn ledger_json(ledger: Q4_0TiedHeadBufferLedgerV1) -> Value {
    json!({
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "abi_version": ledger.abi_version,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "packed_weight_bytes": ledger.packed_weight_bytes,
        "hidden_bytes": ledger.hidden_bytes,
        "score_scratch_bytes": ledger.score_scratch_bytes,
        "partial_topk_scratch_bytes": ledger.partial_topk_scratch_bytes,
        "output_token_bytes": ledger.output_token_bytes,
        "status_bytes": ledger.status_bytes,
        "persistent_scratch_bytes": ledger.persistent_scratch_bytes,
        "total_persistent_bytes": ledger.total_persistent_bytes,
        "transient_score_readback_bytes_per_score_call": ledger.transient_score_readback_bytes_per_score_call,
        "host_to_device_bytes_per_candidate_call": ledger.host_to_device_bytes_per_candidate_call,
        "device_to_host_bytes_per_candidate_call": ledger.device_to_host_bytes_per_candidate_call,
        "command_buffers_per_candidate_call": ledger.command_buffers_per_candidate_call,
        "compute_encoders_per_candidate_call": ledger.compute_encoders_per_candidate_call,
        "kernel_dispatches_per_candidate_call": ledger.kernel_dispatches_per_candidate_call,
        "blit_encoders_per_candidate_call": ledger.blit_encoders_per_candidate_call,
        "commits_per_candidate_call": ledger.commits_per_candidate_call,
        "waits_per_candidate_call": ledger.waits_per_candidate_call,
    })
}

fn sha256_compact_json(value: &Value) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn sha256_f32_le(values: &[f32]) -> Result<String, Box<dyn Error>> {
    let mut digest = Sha256::new();
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("non-finite hidden hash input at element {index}").into());
        }
        digest.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn parse_args_from<I>(values: I) -> Result<Args, Box<dyn Error>>
where
    I: IntoIterator<Item = OsString>,
{
    let mut values = values.into_iter();
    let _program = values.next().ok_or("argv omitted program name")?;
    let mut model_dir = None;
    while let Some(argument) = values.next() {
        match argument.to_string_lossy().as_ref() {
            "--model-dir" => {
                if model_dir.is_some() {
                    return Err("--model-dir may be specified at most once".into());
                }
                model_dir = Some(PathBuf::from(
                    values.next().ok_or("--model-dir requires a value")?,
                ));
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}\n{}", usage()).into()),
        }
    }
    Ok(Args {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required\n{}", usage()))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audit_fixture() -> (
        Vec<[u32; 4]>,
        Vec<[u32; 4]>,
        Vec<u32>,
        Vec<u32>,
        Vec<String>,
    ) {
        let mut cpu = vec![[1, 2, 3, 4]; OUTPUT_TOKENS];
        let mut metal = cpu.clone();
        cpu[46] = [109, 123, 122, 116];
        metal[46] = [109, 123, 122, 116];
        (
            cpu,
            metal,
            vec![271; OUTPUT_TOKENS],
            vec![1; OUTPUT_TOKENS],
            vec!["hidden".to_string(); OUTPUT_TOKENS],
        )
    }

    #[test]
    fn mismatch_audit_is_exact_and_never_substitutes_candidates() {
        let (cpu, mut metal, teachers, winners, hashes) = audit_fixture();
        assert!(mismatch_audit(&cpu, &metal, &teachers, &winners, &hashes)
            .unwrap()
            .is_empty());
        metal[11].swap(0, 1);
        let mismatches = mismatch_audit(&cpu, &metal, &teachers, &winners, &hashes).unwrap();
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0]["step_index_zero_based"], 11);
        assert_eq!(mismatches[0]["cpu_q4_k4"], json!([1, 2, 3, 4]));
        assert_eq!(mismatches[0]["metal_q4_k4"], json!([2, 1, 3, 4]));
    }

    #[test]
    fn parser_requires_one_model_directory() {
        let args = parse_args_from([
            OsString::from("gate"),
            OsString::from("--model-dir"),
            OsString::from("/model"),
        ])
        .unwrap();
        assert_eq!(args.model_dir, PathBuf::from("/model"));
        assert!(parse_args_from([OsString::from("gate")]).is_err());
    }
}

//! Real-checkpoint semantic admission for the explicit Q4_0 tail candidate.
//!
//! This gate records no performance measurement. It computes the currently
//! accepted W8-tail trajectory live, then requires the Q4-tail v2 lane to
//! preserve the same masked free128 and teacher-reranked trajectories while
//! exercising its own compact production receipts.

use std::error::Error;
use std::path::{Path, PathBuf};

use apxinf_core::Device;
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FORMAT: &str = "apxinf-qwen35-tail-q4-v2-semantic-gate-v1";
const QUALIFICATION: &str = "NON_FORMAL_CORRECTNESS_ONLY_NO_PERFORMANCE_RESULT";
const MAX_CONTEXT: usize = 256;
const OUTPUT_TOKENS: usize = 128;
const TEACHER_PREFILL_TOKENS: usize = 12;
const RAW_PROMPT_TOKEN_IDS: [u32; 13] = [
    248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];
const EXCLUDED_EOG_TOKEN_IDS: [u32; 5] = [248044, 248046, 248063, 248064, 248065];

#[derive(Clone, Copy)]
enum Lane {
    AcceptedW8V1,
    CandidateQ4V2,
}

impl Lane {
    const fn label(self) -> &'static str {
        match self {
            Self::AcceptedW8V1 => "accepted-w8-tail-gdn-core-fused-v1",
            Self::CandidateQ4V2 => "candidate-w8-mlp-q4-head-gdn-core-fused-v2",
        }
    }
}

struct LaneRun {
    free_token_ids: Vec<u32>,
    teacher_input_token_ids: Vec<u32>,
    teacher_same_hidden_f32_token_ids: Vec<u32>,
    teacher_candidate_token_ids: Vec<[u32; apxinf_metal::W8_TOP_K]>,
    teacher_reranked_token_ids: Vec<u32>,
    free_receipt: Value,
    teacher_receipt: Value,
    reset_receipt: Value,
}

fn usage() -> &'static str {
    "Usage: qwen35_tail_q4_v2_semantic_gate --model-dir PATH"
}

fn main() {
    match real_main() {
        Ok(receipt) => {
            println!(
                "{}",
                serde_json::to_string(&receipt).expect("serialize Q4 tail semantic receipt")
            );
            if receipt.get("passed").and_then(Value::as_bool) != Some(true) {
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("{FORMAT} failed before a complete receipt was emitted: {error}");
            std::process::exit(1);
        }
    }
}

fn real_main() -> Result<Value, Box<dyn Error>> {
    if !cfg!(target_os = "macos") {
        return Err("Q4 tail semantic gate requires macOS".into());
    }
    let model_dir = parse_model_dir(std::env::args().skip(1))?;
    let accepted = run_lane(&model_dir, Lane::AcceptedW8V1, None)?;
    let candidate = run_lane(
        &model_dir,
        Lane::CandidateQ4V2,
        Some(&accepted.teacher_input_token_ids),
    )?;

    let free_mismatches = mismatch_positions(&accepted.free_token_ids, &candidate.free_token_ids);
    let teacher_hidden_mismatches = mismatch_positions(
        &accepted.teacher_same_hidden_f32_token_ids,
        &candidate.teacher_same_hidden_f32_token_ids,
    );
    let teacher_rerank_mismatches = mismatch_positions(
        &accepted.teacher_reranked_token_ids,
        &candidate.teacher_reranked_token_ids,
    );
    let candidate_missing_f32_winner_positions = candidate
        .teacher_candidate_token_ids
        .iter()
        .zip(&candidate.teacher_same_hidden_f32_token_ids)
        .enumerate()
        .filter_map(|(position, (candidates, winner))| {
            (!candidates.contains(winner)).then_some(position)
        })
        .collect::<Vec<_>>();
    let candidate_duplicate_positions = candidate
        .teacher_candidate_token_ids
        .iter()
        .enumerate()
        .filter_map(|(position, candidates)| (!all_distinct(candidates)).then_some(position))
        .collect::<Vec<_>>();
    let eog_occurrences = [
        ("accepted_free", tokens_with_eog(&accepted.free_token_ids)),
        ("candidate_free", tokens_with_eog(&candidate.free_token_ids)),
        (
            "accepted_teacher_reranked",
            tokens_with_eog(&accepted.teacher_reranked_token_ids),
        ),
        (
            "candidate_teacher_reranked",
            tokens_with_eog(&candidate.teacher_reranked_token_ids),
        ),
    ];
    let candidate_eog_occurrences =
        candidate
            .teacher_candidate_token_ids
            .iter()
            .enumerate()
            .flat_map(|(position, candidates)| {
                candidates.iter().copied().enumerate().filter_map(
                    move |(candidate_index, token)| {
                        EXCLUDED_EOG_TOKEN_IDS.contains(&token).then_some(json!({
                            "position": position,
                            "candidate_index": candidate_index,
                            "token_id": token,
                        }))
                    },
                )
            })
            .collect::<Vec<_>>();

    let accepted_receipts = validate_receipts(&accepted, Lane::AcceptedW8V1);
    let candidate_receipts = validate_receipts(&candidate, Lane::CandidateQ4V2);
    let fixed_counts_exact = [
        accepted.free_token_ids.len(),
        accepted.teacher_input_token_ids.len(),
        accepted.teacher_same_hidden_f32_token_ids.len(),
        accepted.teacher_candidate_token_ids.len(),
        accepted.teacher_reranked_token_ids.len(),
        candidate.free_token_ids.len(),
        candidate.teacher_input_token_ids.len(),
        candidate.teacher_same_hidden_f32_token_ids.len(),
        candidate.teacher_candidate_token_ids.len(),
        candidate.teacher_reranked_token_ids.len(),
    ]
    .into_iter()
    .all(|count| count == OUTPUT_TOKENS);
    let no_excluded_eog = eog_occurrences
        .iter()
        .all(|(_, occurrences)| occurrences.is_empty())
        && candidate_eog_occurrences.is_empty();
    let passed = fixed_counts_exact
        && free_mismatches.is_empty()
        && teacher_hidden_mismatches.is_empty()
        && teacher_rerank_mismatches.is_empty()
        && candidate_missing_f32_winner_positions.is_empty()
        && candidate_duplicate_positions.is_empty()
        && no_excluded_eog
        && accepted_receipts["all_exact"] == true
        && candidate_receipts["all_exact"] == true;

    Ok(json!({
        "format": FORMAT,
        "schema_version": 1,
        "qualification": QUALIFICATION,
        "claim_boundary": "real-checkpoint semantic equality of the explicit Q4 tail candidate against the live accepted W8 tail; no latency, throughput, or runtime ranking claim",
        "model_dir": model_dir,
        "workload": {
            "raw_prompt_token_ids": RAW_PROMPT_TOKEN_IDS,
            "teacher_prefill_token_ids": &RAW_PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS],
            "excluded_eog_token_ids": EXCLUDED_EOG_TOKEN_IDS,
            "output_tokens": OUTPUT_TOKENS,
            "sampling": "greedy with exclusions before candidate selection",
        },
        "lanes": {
            "accepted": Lane::AcceptedW8V1.label(),
            "candidate": Lane::CandidateQ4V2.label(),
            "candidate_constructor": "GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_q4_head_gdn_core_fused_v2",
            "candidate_head": "canonical Q4_0 block32 with FP16 scale, per-eight-row partial top4 and global top4 reducer",
            "exact_rerank": "original F32 tied embedding rows on CPU",
        },
        "hashes": {
            "accepted_free_token_ids_sha256": sha256_json(&json!(accepted.free_token_ids))?,
            "candidate_free_token_ids_sha256": sha256_json(&json!(candidate.free_token_ids))?,
            "accepted_teacher_reranked_token_ids_sha256": sha256_json(&json!(accepted.teacher_reranked_token_ids))?,
            "candidate_teacher_reranked_token_ids_sha256": sha256_json(&json!(candidate.teacher_reranked_token_ids))?,
            "candidate_teacher_top4_token_ids_sha256": sha256_json(&json!(candidate.teacher_candidate_token_ids))?,
        },
        "counts": {
            "accepted_free_tokens": accepted.free_token_ids.len(),
            "candidate_free_tokens": candidate.free_token_ids.len(),
            "teacher_steps": candidate.teacher_candidate_token_ids.len(),
            "free_trajectory_mismatch_count": free_mismatches.len(),
            "teacher_same_hidden_f32_mismatch_count": teacher_hidden_mismatches.len(),
            "teacher_rerank_mismatch_count": teacher_rerank_mismatches.len(),
            "candidate_missing_same_hidden_f32_winner_count": candidate_missing_f32_winner_positions.len(),
            "candidate_duplicate_step_count": candidate_duplicate_positions.len(),
            "excluded_eog_occurrence_count": eog_occurrences.iter().map(|(_, values)| values.len()).sum::<usize>() + candidate_eog_occurrences.len(),
        },
        "mismatches": {
            "free_trajectory_positions": free_mismatches,
            "teacher_same_hidden_f32_positions": teacher_hidden_mismatches,
            "teacher_rerank_positions": teacher_rerank_mismatches,
            "candidate_missing_same_hidden_f32_winner_positions": candidate_missing_f32_winner_positions,
            "candidate_duplicate_positions": candidate_duplicate_positions,
        },
        "no_eog": {
            "trajectory_occurrences": eog_occurrences.into_iter().map(|(lane, occurrences)| json!({"lane": lane, "occurrences": occurrences})).collect::<Vec<_>>(),
            "candidate_occurrences": candidate_eog_occurrences,
            "all_exact": no_excluded_eog,
        },
        "receipts": {
            "accepted": accepted_receipts,
            "candidate": candidate_receipts,
        },
        "admission": {
            "fixed_counts_exact": fixed_counts_exact,
            "masked_free128_matches_current_accepted_path": free_mismatches.is_empty(),
            "teacher_same_hidden_f32_matches_current_accepted_path": teacher_hidden_mismatches.is_empty(),
            "teacher_exact_rerank_matches_current_accepted_path": teacher_rerank_mismatches.is_empty(),
            "q4_top4_contains_same_hidden_f32_winner_at_all_steps": candidate_missing_f32_winner_positions.is_empty(),
            "compact_path_counters_exact": candidate_receipts["all_exact"] == true,
            "reset_clears_all_counters_and_terminal_state": candidate_receipts["reset_exact"] == true,
            "all_generated_and_candidate_paths_exclude_eog": no_excluded_eog,
            "passed": passed,
        },
        "passed": passed,
    }))
}

fn run_lane(
    model_dir: &Path,
    lane: Lane,
    teacher_inputs_override: Option<&[u32]>,
) -> Result<LaneRun, Box<dyn Error>> {
    let config = Qwen35Config::from_json_file(&model_dir.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path_filtered(model_dir, |name| {
        name.starts_with("model.language_model.") || name == "lm_head.weight"
    })?;
    let mut model = match lane {
        Lane::AcceptedW8V1 => {
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1(
                config,
                tensors,
                Device::Cpu,
                MAX_CONTEXT,
            )?
        }
        Lane::CandidateQ4V2 => {
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_q4_head_gdn_core_fused_v2(
                config,
                tensors,
                Device::Cpu,
                MAX_CONTEXT,
            )?
        }
    };

    let mut free_token_ids = Vec::with_capacity(OUTPUT_TOKENS);
    let first = model.prefill_token_for_generation_excluding(
        LlmInput::text(&RAW_PROMPT_TOKEN_IDS),
        &EXCLUDED_EOG_TOKEN_IDS,
    )?;
    free_token_ids.push(first);
    for decode_step in 0..OUTPUT_TOKENS - 1 {
        let input_token = *free_token_ids
            .last()
            .ok_or("free trajectory became empty")?;
        let position = u32::try_from(RAW_PROMPT_TOKEN_IDS.len() + decode_step)?;
        free_token_ids.push(model.decode_token_excluding(
            input_token,
            position,
            &EXCLUDED_EOG_TOKEN_IDS,
        )?);
    }
    let free_receipt = model
        .generation_path_receipt()
        .ok_or("free path omitted generation receipt")?;

    model.reset_checked()?;
    let teacher_input_token_ids = match teacher_inputs_override {
        Some(inputs) => inputs.to_vec(),
        None => teacher_inputs(&free_token_ids)?,
    };
    if teacher_input_token_ids.len() != OUTPUT_TOKENS {
        return Err("teacher input count is not 128".into());
    }
    let _ = model.prefill_for_generation(LlmInput::text(
        &RAW_PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS],
    ))?;
    let mut teacher_same_hidden_f32_token_ids = Vec::with_capacity(OUTPUT_TOKENS);
    let mut teacher_candidate_token_ids = Vec::with_capacity(OUTPUT_TOKENS);
    let mut teacher_reranked_token_ids = Vec::with_capacity(OUTPUT_TOKENS);
    for (step, &input_token) in teacher_input_token_ids.iter().enumerate() {
        let position = u32::try_from(TEACHER_PREFILL_TOKENS + step)?;
        let comparison = model.teacher_forced_decode_candidates_excluding(
            input_token,
            position,
            &EXCLUDED_EOG_TOKEN_IDS,
        )?;
        teacher_same_hidden_f32_token_ids.push(comparison.cpu_token);
        teacher_candidate_token_ids.push(comparison.w8_candidates);
        teacher_reranked_token_ids.push(comparison.reranked_token);
    }
    let teacher_receipt = model
        .generation_path_receipt()
        .ok_or("teacher path omitted generation receipt")?;

    model.reset_checked()?;
    let reset_receipt = model
        .generation_path_receipt()
        .ok_or("reset path omitted generation receipt")?;
    Ok(LaneRun {
        free_token_ids,
        teacher_input_token_ids,
        teacher_same_hidden_f32_token_ids,
        teacher_candidate_token_ids,
        teacher_reranked_token_ids,
        free_receipt,
        teacher_receipt,
        reset_receipt,
    })
}

fn validate_receipts(run: &LaneRun, lane: Lane) -> Value {
    let expected_format = match lane {
        Lane::AcceptedW8V1 => "apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1",
        Lane::CandidateQ4V2 => "apxinf-qwen35-mlp-stack3-boundary-tail-q4-head-generation-path-v2",
    };
    let free_exact = receipt_counts_exact(&run.free_receipt, expected_format, 127, 127, 0);
    let teacher_exact = receipt_counts_exact(&run.teacher_receipt, expected_format, 128, 0, 128);
    let reset_exact = receipt_counts_exact(&run.reset_receipt, expected_format, 0, 0, 0)
        && run.reset_receipt["prefill_body_calls"] == 0
        && run.reset_receipt["prefill_head"]["calls"] == 0;
    let q4_contract_exact = match lane {
        Lane::AcceptedW8V1 => true,
        Lane::CandidateQ4V2 => {
            let ledger = &run.free_receipt["tail_ledger"];
            run.free_receipt["mechanism"]
                == "metal-w8-mlp-stack3-boundary-tail-q4-head-gdn-core-fused-v2"
                && run.free_receipt["decode_head"]["mechanism"]
                    == "metal-w8-mlp-q4_0-candidate-tail-v2"
                && ledger["abi_version"] == 2
                && ledger["q4_vocab_weight_bytes"] == 143_032_320u64
                && ledger["w8_vocab_weight_bytes"] == 0
                && ledger["w8_vocab_scale_bytes"] == 0
                && ledger["full_score_scratch_bytes"] == 0
                && ledger["command_buffers_per_decode"] == 1
                && ledger["compute_encoders_per_decode"] == 1
                && ledger["commits_per_decode"] == 1
                && ledger["waits_per_decode"] == 1
        }
    };
    json!({
        "lane": lane.label(),
        "format": run.free_receipt["format"],
        "mechanism": run.free_receipt["mechanism"],
        "free_exact": free_exact,
        "teacher_exact": teacher_exact,
        "reset_exact": reset_exact,
        "q4_contract_exact": q4_contract_exact,
        "all_exact": free_exact && teacher_exact && reset_exact && q4_contract_exact,
        "free_decode_head": run.free_receipt["decode_head"],
        "teacher_decode_head": run.teacher_receipt["decode_head"],
        "reset_decode_head": run.reset_receipt["decode_head"],
        "candidate_tail_ledger": run.free_receipt.get("tail_ledger"),
    })
}

fn receipt_counts_exact(
    receipt: &Value,
    expected_format: &str,
    body_calls: u64,
    decode_calls: u64,
    teacher_calls: u64,
) -> bool {
    let boundaries = match receipt["boundaries"].as_array() {
        Some(boundaries) => boundaries,
        None => return false,
    };
    receipt["format"] == expected_format
        && receipt["initial_stack"]["decode_calls"] == body_calls
        && receipt["initial_stack"]["successful_decodes"] == body_calls
        && receipt["initial_stack"]["failed_decodes"] == 0
        && receipt["initial_stack"]["terminal_error"] == false
        && boundaries.len() == 5
        && boundaries.iter().all(|boundary| {
            boundary["decode_calls"] == body_calls
                && boundary["successful_decodes"] == body_calls
                && boundary["failed_decodes"] == 0
                && boundary["terminal_error"] == false
        })
        && receipt["decode_head"]["calls"] == decode_calls
        && receipt["decode_head"]["excluded_calls"] == decode_calls
        && receipt["decode_head"]["teacher_calls"] == teacher_calls
        && receipt["decode_head"]["tail_transactions"] == body_calls
        && receipt["decode_head"]["successful_transactions"] == body_calls
        && receipt["decode_head"]["failed_transactions"] == 0
        && receipt["decode_head"]["terminal_error"] == false
        && receipt["terminal_error"] == false
}

fn teacher_inputs(free: &[u32]) -> Result<Vec<u32>, Box<dyn Error>> {
    if free.len() != OUTPUT_TOKENS {
        return Err("accepted free trajectory is not 128 tokens".into());
    }
    let mut inputs = Vec::with_capacity(OUTPUT_TOKENS);
    inputs.push(RAW_PROMPT_TOKEN_IDS[TEACHER_PREFILL_TOKENS]);
    inputs.extend_from_slice(&free[..OUTPUT_TOKENS - 1]);
    Ok(inputs)
}

fn parse_model_dir(mut args: impl Iterator<Item = String>) -> Result<PathBuf, Box<dyn Error>> {
    if args.next().as_deref() != Some("--model-dir") {
        return Err(usage().into());
    }
    let path = args.next().ok_or_else(|| usage().to_owned())?;
    if args.next().is_some() {
        return Err(usage().into());
    }
    let path = std::fs::canonicalize(path)?;
    if !path.is_dir() {
        return Err("model directory is not a directory".into());
    }
    Ok(path)
}

fn mismatch_positions(expected: &[u32], observed: &[u32]) -> Vec<usize> {
    (0..expected.len().max(observed.len()))
        .filter(|&position| expected.get(position) != observed.get(position))
        .collect()
}

fn tokens_with_eog(tokens: &[u32]) -> Vec<Value> {
    tokens
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(position, token)| {
            EXCLUDED_EOG_TOKEN_IDS
                .contains(&token)
                .then_some(json!({"position": position, "token_id": token}))
        })
        .collect()
}

fn all_distinct<const N: usize>(tokens: &[u32; N]) -> bool {
    tokens
        .iter()
        .enumerate()
        .all(|(index, token)| !tokens[..index].contains(token))
}

fn sha256_json(value: &Value) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_counter_validator_distinguishes_free_teacher_and_reset() {
        let make = |body: u64, decode: u64, teacher: u64| {
            json!({
                "format": "path",
                "initial_stack": {"decode_calls": body, "successful_decodes": body, "failed_decodes": 0, "terminal_error": false},
                "boundaries": (0..5).map(|_| json!({"decode_calls": body, "successful_decodes": body, "failed_decodes": 0, "terminal_error": false})).collect::<Vec<_>>(),
                "decode_head": {"calls": decode, "excluded_calls": decode, "teacher_calls": teacher, "tail_transactions": body, "successful_transactions": body, "failed_transactions": 0, "terminal_error": false},
                "terminal_error": false,
            })
        };
        assert!(receipt_counts_exact(
            &make(127, 127, 0),
            "path",
            127,
            127,
            0
        ));
        assert!(receipt_counts_exact(
            &make(128, 0, 128),
            "path",
            128,
            0,
            128
        ));
        assert!(receipt_counts_exact(&make(0, 0, 0), "path", 0, 0, 0));
        assert!(!receipt_counts_exact(
            &make(127, 126, 0),
            "path",
            127,
            127,
            0
        ));
    }
}

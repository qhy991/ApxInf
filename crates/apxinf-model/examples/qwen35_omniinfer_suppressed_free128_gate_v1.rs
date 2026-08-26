//! Correctness-only admission gate for the five-token EOG suppression used by
//! the pinned OmniInfer/llama.cpp `ignore_eos=true` deployment.
//!
//! This example deliberately records no latency, throughput, or other
//! performance measurement. It is not formal evidence and cannot rank either
//! runtime. Its only job is to establish a common greedy sampling policy before
//! a separately controlled HTTP comparison is allowed to begin.

use std::error::Error;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use apxinf_core::{Device, Tensor};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FORMAT: &str = "apxinf-qwen35-omniinfer-suppressed-free128-gate-v1";
const QUALIFICATION: &str = "NON_FORMAL_SEMANTIC_ADMISSION_NO_PERFORMANCE_RESULT";
const MAX_CONTEXT: usize = 256;
const OUTPUT_TOKENS: usize = 128;
const TEACHER_PREFILL_TOKENS: usize = 12;
const EXPECTED_VOCAB_SIZE: usize = 248_320;
const EXPECTED_TOP4_WIDTH: usize = 4;

const RAW_PROMPT_TOKEN_IDS: [u32; 13] = [
    248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];
const EXCLUDED_EOG_TOKEN_IDS: [u32; 5] = [248044, 248046, 248063, 248064, 248065];

// Pinned b10280 llama.cpp emitted this trajectory for the canonical raw-token
// request with `ignore_eos=true`. The source receipt is identified in the JSON
// output. This frozen list permits useful mismatch positions; its compact-JSON
// SHA-256 is independently frozen below.
const OMNIINFER_SUPPRESSED_FREE128_TOKEN_IDS: [u32; 128] = [
    9419, 0, 2500, 628, 353, 1438, 488, 3242, 30, 25677, 232, 353, 2688, 1167, 16451, 18, 13, 20,
    11, 279, 5362, 3349, 3992, 1558, 7633, 539, 48696, 36814, 11274, 13, 3437, 579, 383, 678, 3838,
    30, 10838, 234, 253, 9008, 97, 244, 169379, 9008, 234, 235, 9008, 234, 109, 9008, 234, 109,
    9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109,
    9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109,
    9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109,
    9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109, 9008, 234, 109,
    9008, 234, 109, 9008,
];

const RAW_PROMPT_TOKEN_IDS_SHA256: &str =
    "4b890fa15ee3d7db4e9dd18bd79c6362d40e9e016ae4f9f74cb7fc420ef3b6d3";
const TEACHER_PREFILL_TOKEN_IDS_SHA256: &str =
    "94b1660c815f507f8df7f4748b41b65bcb9ba09308930e15041b815000b2bdeb";
const EXCLUDED_EOG_TOKEN_IDS_SHA256: &str =
    "656e15a6ba9c76f492ba6bb34a0f2af4095ec3850dbb09b468228c2055ece9ca";
const OMNIINFER_SUPPRESSED_FREE128_SHA256: &str =
    "0a8a6c5ceeb831528480ebcad172fbcdda4ac23478ab051b1f74a00ec6d4f8e4";
const OMNIINFER_RAW_DRIVER_RECEIPT_SHA256: &str =
    "4afb50505e907161b063b11a1062d5bdc70162bf9552625d5c03ae3caeccf8cf";

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    candidate_profile: CandidateProfile,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum CandidateProfile {
    #[default]
    Fused,
    Legacy,
}

impl CandidateProfile {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "fused" => Ok(Self::Fused),
            "legacy" => Ok(Self::Legacy),
            _ => {
                Err(format!("invalid --candidate-profile {value}; expected fused or legacy").into())
            }
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Fused => "fused",
            Self::Legacy => "legacy",
        }
    }
}

struct CandidateRuns {
    teacher_input_token_ids: Vec<u32>,
    teacher_same_hidden_full_f32_token_ids: Vec<u32>,
    teacher_top4_candidate_token_ids: Vec<[u32; apxinf_metal::W8_TOP_K]>,
    teacher_reranked_token_ids: Vec<u32>,
    checked_reset_calls: usize,
    candidate_free_token_ids: Vec<u32>,
}

fn usage() -> &'static str {
    "Usage: qwen35_omniinfer_suppressed_free128_gate_v1 --model-dir PATH [--candidate-profile fused|legacy]"
}

fn main() {
    match real_main() {
        Ok(receipt) => {
            println!(
                "{}",
                serde_json::to_string(&receipt).expect("serialize semantic-admission receipt")
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
        return Err("the Metal W8 semantic-admission gate requires macOS".into());
    }
    let args = parse_args_from(std::env::args_os())?;
    validate_frozen_contract_hashes()?;
    let model_dir = std::fs::canonicalize(&args.model_dir)?;

    let cpu_reference_token_ids = run_cpu_f32_reference(&model_dir)?;
    let candidate = run_candidate_teacher_reset_and_free(
        &model_dir,
        &cpu_reference_token_ids,
        args.candidate_profile,
    )?;

    let cpu_reference_vs_omniinfer = mismatch_details(
        &OMNIINFER_SUPPRESSED_FREE128_TOKEN_IDS,
        &cpu_reference_token_ids,
    );
    let teacher_same_hidden_full_f32_vs_reference = mismatch_details(
        &cpu_reference_token_ids,
        &candidate.teacher_same_hidden_full_f32_token_ids,
    );
    let teacher_reranked_vs_reference = mismatch_details(
        &cpu_reference_token_ids,
        &candidate.teacher_reranked_token_ids,
    );
    let candidate_free_vs_reference = mismatch_details(
        &cpu_reference_token_ids,
        &candidate.candidate_free_token_ids,
    );
    let internal_divergences = internal_divergence_details(&cpu_reference_token_ids, &candidate);

    let candidate_width_mismatch_positions = candidate
        .teacher_top4_candidate_token_ids
        .iter()
        .enumerate()
        .filter_map(|(step, candidates)| (candidates.len() != EXPECTED_TOP4_WIDTH).then_some(step))
        .collect::<Vec<_>>();
    let candidate_duplicate_positions = candidate
        .teacher_top4_candidate_token_ids
        .iter()
        .enumerate()
        .filter_map(|(step, candidates)| (!all_distinct(candidates)).then_some(step))
        .collect::<Vec<_>>();
    let candidate_missing_same_hidden_full_f32_positions = candidate
        .teacher_top4_candidate_token_ids
        .iter()
        .zip(&candidate.teacher_same_hidden_full_f32_token_ids)
        .enumerate()
        .filter_map(|(step, (candidates, expected))| {
            (!candidates.contains(expected)).then_some(step)
        })
        .collect::<Vec<_>>();
    let candidate_missing_reference_positions = candidate
        .teacher_top4_candidate_token_ids
        .iter()
        .zip(&cpu_reference_token_ids)
        .enumerate()
        .filter_map(|(step, (candidates, expected))| {
            (!candidates.contains(expected)).then_some(step)
        })
        .collect::<Vec<_>>();
    let reranked_not_in_candidate_positions = candidate
        .teacher_top4_candidate_token_ids
        .iter()
        .zip(&candidate.teacher_reranked_token_ids)
        .enumerate()
        .filter_map(|(step, (candidates, selected))| {
            (!candidates.contains(selected)).then_some(step)
        })
        .collect::<Vec<_>>();
    let candidate_out_of_vocab_occurrences =
        candidate_token_occurrences(&candidate.teacher_top4_candidate_token_ids, |token| {
            token as usize >= EXPECTED_VOCAB_SIZE
        });

    let omniinfer_reference_eog_occurrences =
        token_occurrences(&OMNIINFER_SUPPRESSED_FREE128_TOKEN_IDS, is_excluded_eog);
    let reference_eog_occurrences = token_occurrences(&cpu_reference_token_ids, is_excluded_eog);
    let teacher_input_eog_occurrences =
        token_occurrences(&candidate.teacher_input_token_ids, is_excluded_eog);
    let teacher_same_hidden_full_f32_eog_occurrences = token_occurrences(
        &candidate.teacher_same_hidden_full_f32_token_ids,
        is_excluded_eog,
    );
    let teacher_candidate_eog_occurrences =
        candidate_token_occurrences(&candidate.teacher_top4_candidate_token_ids, is_excluded_eog);
    let teacher_reranked_eog_occurrences =
        token_occurrences(&candidate.teacher_reranked_token_ids, is_excluded_eog);
    let candidate_free_eog_occurrences =
        token_occurrences(&candidate.candidate_free_token_ids, is_excluded_eog);

    let raw_prompt_sha256 = sha256_compact_json(&json!(RAW_PROMPT_TOKEN_IDS))?;
    let teacher_prefill_sha256 =
        sha256_compact_json(&json!(&RAW_PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS]))?;
    let excluded_eog_sha256 = sha256_compact_json(&json!(EXCLUDED_EOG_TOKEN_IDS))?;
    let omniinfer_suppressed_free128_sha256 =
        sha256_compact_json(&json!(&OMNIINFER_SUPPRESSED_FREE128_TOKEN_IDS[..]))?;
    let cpu_reference_sha256 = sha256_compact_json(&json!(cpu_reference_token_ids))?;
    let teacher_input_sha256 = sha256_compact_json(&json!(candidate.teacher_input_token_ids))?;
    let teacher_same_hidden_full_f32_sha256 =
        sha256_compact_json(&json!(candidate.teacher_same_hidden_full_f32_token_ids))?;
    let teacher_candidates_sha256 =
        sha256_compact_json(&json!(candidate.teacher_top4_candidate_token_ids))?;
    let teacher_reranked_sha256 =
        sha256_compact_json(&json!(candidate.teacher_reranked_token_ids))?;
    let candidate_free_sha256 = sha256_compact_json(&json!(candidate.candidate_free_token_ids))?;

    let fixed_counts_exact = RAW_PROMPT_TOKEN_IDS.len() == 13
        && TEACHER_PREFILL_TOKENS == 12
        && EXCLUDED_EOG_TOKEN_IDS.len() == 5
        && OMNIINFER_SUPPRESSED_FREE128_TOKEN_IDS.len() == OUTPUT_TOKENS
        && cpu_reference_token_ids.len() == OUTPUT_TOKENS
        && candidate.teacher_input_token_ids.len() == OUTPUT_TOKENS
        && candidate.teacher_same_hidden_full_f32_token_ids.len() == OUTPUT_TOKENS
        && candidate.teacher_top4_candidate_token_ids.len() == OUTPUT_TOKENS
        && candidate.teacher_reranked_token_ids.len() == OUTPUT_TOKENS
        && candidate.checked_reset_calls == 1
        && candidate.candidate_free_token_ids.len() == OUTPUT_TOKENS;
    let frozen_hashes_exact = raw_prompt_sha256 == RAW_PROMPT_TOKEN_IDS_SHA256
        && teacher_prefill_sha256 == TEACHER_PREFILL_TOKEN_IDS_SHA256
        && excluded_eog_sha256 == EXCLUDED_EOG_TOKEN_IDS_SHA256
        && omniinfer_suppressed_free128_sha256 == OMNIINFER_SUPPRESSED_FREE128_SHA256;
    let cpu_reference_matches_omniinfer = cpu_reference_vs_omniinfer.is_empty()
        && cpu_reference_sha256 == OMNIINFER_SUPPRESSED_FREE128_SHA256;
    let candidate_teacher_same_hidden_full_f32_matches_reference =
        teacher_same_hidden_full_f32_vs_reference.is_empty();
    let candidate_teacher_rerank_matches_reference = teacher_reranked_vs_reference.is_empty();
    let candidate_free_matches_reference = candidate_free_vs_reference.is_empty();
    let candidate_contract_exact = candidate_width_mismatch_positions.is_empty()
        && candidate_duplicate_positions.is_empty()
        && candidate_missing_same_hidden_full_f32_positions.is_empty()
        && candidate_missing_reference_positions.is_empty()
        && reranked_not_in_candidate_positions.is_empty()
        && candidate_out_of_vocab_occurrences.is_empty();
    let all_generated_paths_have_no_excluded_eog = omniinfer_reference_eog_occurrences.is_empty()
        && reference_eog_occurrences.is_empty()
        && teacher_input_eog_occurrences.is_empty()
        && teacher_same_hidden_full_f32_eog_occurrences.is_empty()
        && teacher_candidate_eog_occurrences.is_empty()
        && teacher_reranked_eog_occurrences.is_empty()
        && candidate_free_eog_occurrences.is_empty();
    let passed = fixed_counts_exact
        && frozen_hashes_exact
        && candidate_teacher_same_hidden_full_f32_matches_reference
        && candidate_teacher_rerank_matches_reference
        && candidate_free_matches_reference
        && candidate_contract_exact
        && all_generated_paths_have_no_excluded_eog;

    Ok(json!({
        "format": FORMAT,
        "schema_version": 1,
        "qualification": QUALIFICATION,
        "claim_boundary": "correctness-only sampling-semantics admission; no runtime ranking or performance claim",
        "model": {
            "model_dir": model_dir.display().to_string(),
            "expected_family": "Qwen/Qwen3.5-0.8B",
            "expected_vocabulary_size": EXPECTED_VOCAB_SIZE,
            "cpu_reference_precision": "F32",
            "candidate_path": "Metal W8 pre-top4 exclusion plus tied F32 four-row rerank",
        },
        "diagnostic_candidate_profile": {
            "selected": args.candidate_profile.label(),
            "default": CandidateProfile::Fused.label(),
            "classification": "single-variable NON_FORMAL semantic diagnostic only; never performance evidence",
            "fused_constructor": "GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1",
            "legacy_constructor": "GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1",
        },
        "omniinfer_evidence": {
            "runtime": "pinned llama.cpp b10280 behind OmniInfer",
            "raw_driver_receipt_sha256": OMNIINFER_RAW_DRIVER_RECEIPT_SHA256,
            "response_pointer": "/warmup_pairs/0/samples/0/response/__verbose",
            "ignore_eos": true,
            "suppression_interpretation": "the five listed EOG logits are forced to negative infinity before greedy selection",
        },
        "workload": {
            "raw_prompt_token_ids": RAW_PROMPT_TOKEN_IDS,
            "teacher_prefill_token_ids": &RAW_PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS],
            "teacher_input_token_ids": candidate.teacher_input_token_ids,
            "output_tokens": OUTPUT_TOKENS,
            "sampling": "greedy",
            "eog_termination": false,
        },
        "suppression": {
            "scope": "generated-logit selection only; prompt ingestion is unchanged",
            "excluded_eog_token_ids": EXCLUDED_EOG_TOKEN_IDS,
            "excluded_eog_token_count": EXCLUDED_EOG_TOKEN_IDS.len(),
            "application": "CPU/F32 argmax and Metal W8 vocabulary rows before top-4 selection",
        },
        "reference": {
            "omniinfer_suppressed_free128_token_ids": &OMNIINFER_SUPPRESSED_FREE128_TOKEN_IDS[..],
            "cpu_f32_reference_token_ids": cpu_reference_token_ids,
        },
        "candidate_teacher": {
            "same_hidden_full_f32_argmax_token_ids": candidate.teacher_same_hidden_full_f32_token_ids,
            "top4_candidate_token_ids": candidate.teacher_top4_candidate_token_ids,
            "reranked_token_ids": candidate.teacher_reranked_token_ids,
        },
        "reset": {
            "entrypoint": "GeneralQwen35::reset_checked",
            "checked_reset_calls_between_candidate_teacher_and_free": candidate.checked_reset_calls,
            "succeeded": candidate.checked_reset_calls == 1,
        },
        "candidate_free": {
            "prefill_prompt_token_ids": RAW_PROMPT_TOKEN_IDS,
            "prefill_selected_token_count": 1,
            "excluded_decode_step_count": OUTPUT_TOKENS - 1,
            "generated_token_ids": candidate.candidate_free_token_ids,
        },
        "counts": {
            "raw_prompt_token_count": RAW_PROMPT_TOKEN_IDS.len(),
            "teacher_prefill_token_count": TEACHER_PREFILL_TOKENS,
            "excluded_eog_token_count": EXCLUDED_EOG_TOKEN_IDS.len(),
            "omniinfer_reference_token_count": OMNIINFER_SUPPRESSED_FREE128_TOKEN_IDS.len(),
            "cpu_f32_reference_token_count": cpu_reference_token_ids.len(),
            "candidate_teacher_input_token_count": candidate.teacher_input_token_ids.len(),
            "candidate_teacher_step_count": candidate.teacher_top4_candidate_token_ids.len(),
            "candidate_teacher_same_hidden_full_f32_token_count": candidate.teacher_same_hidden_full_f32_token_ids.len(),
            "candidate_teacher_reranked_token_count": candidate.teacher_reranked_token_ids.len(),
            "candidate_set_width": EXPECTED_TOP4_WIDTH,
            "candidate_token_total": candidate.teacher_top4_candidate_token_ids.len() * EXPECTED_TOP4_WIDTH,
            "checked_reset_call_count": candidate.checked_reset_calls,
            "candidate_free_prefill_selected_token_count": 1,
            "candidate_free_excluded_decode_step_count": OUTPUT_TOKENS - 1,
            "candidate_free_token_count": candidate.candidate_free_token_ids.len(),
            "cpu_reference_vs_omniinfer_mismatch_count": cpu_reference_vs_omniinfer.len(),
            "teacher_same_hidden_full_f32_vs_reference_mismatch_count": teacher_same_hidden_full_f32_vs_reference.len(),
            "teacher_reranked_vs_reference_mismatch_count": teacher_reranked_vs_reference.len(),
            "candidate_free_vs_reference_mismatch_count": candidate_free_vs_reference.len(),
            "internal_semantic_divergence_count": internal_divergences.len(),
            "candidate_width_mismatch_count": candidate_width_mismatch_positions.len(),
            "candidate_duplicate_step_count": candidate_duplicate_positions.len(),
            "candidate_missing_same_hidden_full_f32_step_count": candidate_missing_same_hidden_full_f32_positions.len(),
            "candidate_missing_reference_step_count": candidate_missing_reference_positions.len(),
            "reranked_not_in_candidate_step_count": reranked_not_in_candidate_positions.len(),
            "candidate_out_of_vocab_occurrence_count": candidate_out_of_vocab_occurrences.len(),
            "excluded_eog_occurrence_total": omniinfer_reference_eog_occurrences.len()
                + reference_eog_occurrences.len()
                + teacher_input_eog_occurrences.len()
                + teacher_same_hidden_full_f32_eog_occurrences.len()
                + teacher_candidate_eog_occurrences.len()
                + teacher_reranked_eog_occurrences.len()
                + candidate_free_eog_occurrences.len(),
        },
        "hashes": {
            "algorithm": "SHA-256",
            "encoding": "UTF-8 compact JSON array bytes produced by serde_json::to_vec",
            "raw_prompt_token_ids_sha256": raw_prompt_sha256,
            "teacher_prefill_token_ids_sha256": teacher_prefill_sha256,
            "excluded_eog_token_ids_sha256": excluded_eog_sha256,
            "omniinfer_suppressed_free128_token_ids_sha256": omniinfer_suppressed_free128_sha256,
            "cpu_f32_reference_token_ids_sha256": cpu_reference_sha256,
            "candidate_teacher_input_token_ids_sha256": teacher_input_sha256,
            "candidate_teacher_same_hidden_full_f32_token_ids_sha256": teacher_same_hidden_full_f32_sha256,
            "candidate_teacher_top4_token_ids_sha256": teacher_candidates_sha256,
            "candidate_teacher_reranked_token_ids_sha256": teacher_reranked_sha256,
            "candidate_free_token_ids_sha256": candidate_free_sha256,
        },
        "mismatches": {
            "cross_runtime_output_identity_is_informational_not_an_admission_requirement": true,
            "cpu_f32_reference_vs_omniinfer": cpu_reference_vs_omniinfer,
            "candidate_teacher_same_hidden_full_f32_vs_reference": teacher_same_hidden_full_f32_vs_reference,
            "candidate_teacher_reranked_vs_reference": teacher_reranked_vs_reference,
            "candidate_free_vs_reference": candidate_free_vs_reference,
            "internal_divergence_details": internal_divergences,
        },
        "candidate_checks": {
            "width_mismatch_positions": candidate_width_mismatch_positions,
            "duplicate_positions": candidate_duplicate_positions,
            "missing_same_hidden_full_f32_positions": candidate_missing_same_hidden_full_f32_positions,
            "missing_reference_positions": candidate_missing_reference_positions,
            "reranked_not_in_candidate_positions": reranked_not_in_candidate_positions,
            "out_of_vocab_occurrences": candidate_out_of_vocab_occurrences,
            "all_exact": candidate_contract_exact,
        },
        "no_eog_checks": {
            "omniinfer_reference_occurrences": omniinfer_reference_eog_occurrences,
            "cpu_f32_reference_occurrences": reference_eog_occurrences,
            "candidate_teacher_input_occurrences": teacher_input_eog_occurrences,
            "candidate_teacher_same_hidden_full_f32_occurrences": teacher_same_hidden_full_f32_eog_occurrences,
            "candidate_teacher_top4_occurrences": teacher_candidate_eog_occurrences,
            "candidate_teacher_reranked_occurrences": teacher_reranked_eog_occurrences,
            "candidate_free_occurrences": candidate_free_eog_occurrences,
            "all_generated_paths_have_no_excluded_eog": all_generated_paths_have_no_excluded_eog,
        },
        "admission": {
            "fixed_counts_exact": fixed_counts_exact,
            "frozen_compact_json_hashes_exact": frozen_hashes_exact,
            "cross_runtime_policy_is_same_prompt_exact_five_token_negative_infinity_mask_greedy_fixed_128": fixed_counts_exact && frozen_hashes_exact && all_generated_paths_have_no_excluded_eog,
            "cross_runtime_output_identity_required": false,
            "cpu_f32_reference_matches_omniinfer_suppressed_free128_informational": cpu_reference_matches_omniinfer,
            "candidate_teacher_same_hidden_full_f32_matches_reference": candidate_teacher_same_hidden_full_f32_matches_reference,
            "candidate_teacher_rerank_matches_reference": candidate_teacher_rerank_matches_reference,
            "candidate_teacher_top4_contract_exact": candidate_contract_exact,
            "checked_reset_succeeded_exactly_once": candidate.checked_reset_calls == 1,
            "candidate_free_matches_reference": candidate_free_matches_reference,
            "internal_semantic_divergence_count_zero": internal_divergences.is_empty(),
            "all_generated_paths_have_no_excluded_eog": all_generated_paths_have_no_excluded_eog,
            "semantic_admission_passed": passed,
        },
        "passed": passed,
    }))
}

fn run_cpu_f32_reference(model_dir: &Path) -> Result<Vec<u32>, Box<dyn Error>> {
    let (config, tensors) = load_model_inputs(model_dir)?;
    validate_model_contract(&config)?;
    let vocab_size = config.text.vocab_size;
    let mut model = GeneralQwen35::from_weights(config, tensors, Device::Cpu, MAX_CONTEXT)?;
    let _ = model.prefill_for_generation(LlmInput::text(
        &RAW_PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS],
    ))?;

    let mut reference = Vec::with_capacity(OUTPUT_TOKENS);
    let mut teacher_token = RAW_PROMPT_TOKEN_IDS[TEACHER_PREFILL_TOKENS];
    for step in 0..OUTPUT_TOKENS {
        let position = u32::try_from(TEACHER_PREFILL_TOKENS + step)?;
        let logits = model.forward(&[teacher_token], position)?;
        let selected = argmax_f32_excluding(&logits, vocab_size, &EXCLUDED_EOG_TOKEN_IDS)?;
        reference.push(selected);
        teacher_token = selected;
    }
    Ok(reference)
}

fn run_candidate_teacher_reset_and_free(
    model_dir: &Path,
    cpu_reference_token_ids: &[u32],
    candidate_profile: CandidateProfile,
) -> Result<CandidateRuns, Box<dyn Error>> {
    if cpu_reference_token_ids.len() != OUTPUT_TOKENS {
        return Err(format!(
            "CPU reference has {} tokens, expected {OUTPUT_TOKENS}",
            cpu_reference_token_ids.len()
        )
        .into());
    }
    let (config, tensors) = load_model_inputs(model_dir)?;
    validate_model_contract(&config)?;
    let mut model = match candidate_profile {
        CandidateProfile::Fused => {
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1(
                config,
                tensors,
                Device::Cpu,
                MAX_CONTEXT,
            )?
        }
        CandidateProfile::Legacy => {
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
                config,
                tensors,
                Device::Cpu,
                MAX_CONTEXT,
            )?
        }
    };

    let teacher_input_token_ids = teacher_inputs(cpu_reference_token_ids)?;
    let _ = model.prefill_for_generation(LlmInput::text(
        &RAW_PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS],
    ))?;
    let mut teacher_same_hidden_full_f32_token_ids = Vec::with_capacity(OUTPUT_TOKENS);
    let mut teacher_top4_candidate_token_ids = Vec::with_capacity(OUTPUT_TOKENS);
    let mut teacher_reranked_token_ids = Vec::with_capacity(OUTPUT_TOKENS);
    for (step, &teacher_token) in teacher_input_token_ids.iter().enumerate() {
        let position = u32::try_from(TEACHER_PREFILL_TOKENS + step)?;
        let comparison = model.teacher_forced_decode_candidates_excluding(
            teacher_token,
            position,
            &EXCLUDED_EOG_TOKEN_IDS,
        )?;
        teacher_same_hidden_full_f32_token_ids.push(comparison.cpu_token);
        teacher_top4_candidate_token_ids.push(comparison.w8_candidates);
        teacher_reranked_token_ids.push(comparison.reranked_token);
    }

    model.reset_checked()?;
    let checked_reset_calls = 1;

    let mut candidate_free_token_ids = Vec::with_capacity(OUTPUT_TOKENS);
    let first = model.prefill_token_for_generation_excluding(
        LlmInput::text(&RAW_PROMPT_TOKEN_IDS),
        &EXCLUDED_EOG_TOKEN_IDS,
    )?;
    candidate_free_token_ids.push(first);
    for decode_step in 0..OUTPUT_TOKENS - 1 {
        let input_token = *candidate_free_token_ids
            .last()
            .ok_or("candidate free trajectory unexpectedly became empty")?;
        let position = u32::try_from(RAW_PROMPT_TOKEN_IDS.len() + decode_step)?;
        let selected =
            model.decode_token_excluding(input_token, position, &EXCLUDED_EOG_TOKEN_IDS)?;
        candidate_free_token_ids.push(selected);
    }

    Ok(CandidateRuns {
        teacher_input_token_ids,
        teacher_same_hidden_full_f32_token_ids,
        teacher_top4_candidate_token_ids,
        teacher_reranked_token_ids,
        checked_reset_calls,
        candidate_free_token_ids,
    })
}

fn load_model_inputs(
    model_dir: &Path,
) -> Result<(Qwen35Config, std::collections::HashMap<String, Tensor>), Box<dyn Error>> {
    let config = Qwen35Config::from_json_file(&model_dir.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path_filtered(model_dir, |name| {
        name.starts_with("model.language_model.") || name == "lm_head.weight"
    })?;
    Ok((config, tensors))
}

fn validate_model_contract(config: &Qwen35Config) -> Result<(), Box<dyn Error>> {
    if config.text.vocab_size != EXPECTED_VOCAB_SIZE {
        return Err(format!(
            "semantic gate expected vocabulary {EXPECTED_VOCAB_SIZE}, got {}",
            config.text.vocab_size
        )
        .into());
    }
    if EXCLUDED_EOG_TOKEN_IDS
        .iter()
        .any(|&token| token as usize >= config.text.vocab_size)
    {
        return Err("semantic gate EOG exclusion is outside the model vocabulary".into());
    }
    Ok(())
}

fn argmax_f32_excluding(
    logits: &Tensor,
    vocab_size: usize,
    excluded_token_ids: &[u32],
) -> Result<u32, Box<dyn Error>> {
    if logits.shape().dims() != [1, vocab_size] {
        return Err(format!("expected logits [1, {vocab_size}], got {}", logits.shape()).into());
    }
    let mut best_score = f32::NEG_INFINITY;
    let mut best_token = None;
    for (token, &score) in logits.as_f32()?.iter().enumerate() {
        if !score.is_finite() {
            return Err(format!("non-finite CPU/F32 logit at vocabulary row {token}").into());
        }
        let token = u32::try_from(token)?;
        if excluded_token_ids.contains(&token) {
            continue;
        }
        if best_token.is_none() || score > best_score {
            best_score = score;
            best_token = Some(token);
        }
    }
    best_token.ok_or_else(|| "EOG suppression excluded the entire vocabulary".into())
}

fn teacher_inputs(reference: &[u32]) -> Result<Vec<u32>, Box<dyn Error>> {
    if reference.len() != OUTPUT_TOKENS {
        return Err(format!(
            "teacher-input construction expected {OUTPUT_TOKENS} reference tokens, got {}",
            reference.len()
        )
        .into());
    }
    let mut inputs = Vec::with_capacity(OUTPUT_TOKENS);
    inputs.push(RAW_PROMPT_TOKEN_IDS[TEACHER_PREFILL_TOKENS]);
    inputs.extend_from_slice(&reference[..OUTPUT_TOKENS - 1]);
    Ok(inputs)
}

fn internal_divergence_details(reference: &[u32], candidate: &CandidateRuns) -> Vec<Value> {
    let observed_len = candidate
        .teacher_same_hidden_full_f32_token_ids
        .len()
        .min(candidate.teacher_top4_candidate_token_ids.len())
        .min(candidate.teacher_reranked_token_ids.len())
        .min(candidate.candidate_free_token_ids.len())
        .min(candidate.teacher_input_token_ids.len());
    (0..reference.len().min(observed_len))
        .filter_map(|step| {
            let expected = reference[step];
            let same_hidden_full_f32 =
                candidate.teacher_same_hidden_full_f32_token_ids[step];
            let candidates = candidate.teacher_top4_candidate_token_ids[step];
            let reranked = candidate.teacher_reranked_token_ids[step];
            let free = candidate.candidate_free_token_ids[step];
            if expected == same_hidden_full_f32 && expected == reranked && expected == free {
                return None;
            }
            let candidates_contain_reference = candidates.contains(&expected);
            let candidates_contain_same_hidden_full_f32 = candidates.contains(&same_hidden_full_f32);
            let classification = if expected != same_hidden_full_f32
                && same_hidden_full_f32 == reranked
                && same_hidden_full_f32 == free
                && candidates_contain_reference
                && candidates_contain_same_hidden_full_f32
            {
                "BODY_HIDDEN_PRECISION_DIVERGENCE_NOT_CANDIDATE_OMISSION_OR_RERANK_FAILURE"
            } else if !candidates_contain_same_hidden_full_f32 {
                "ACCELERATOR_CANDIDATE_COVERAGE_DIVERGENCE"
            } else if reranked != same_hidden_full_f32 {
                "FOUR_ROW_F32_RERANK_DIVERGENCE"
            } else if free != reranked {
                "FREE_RUN_STATE_OR_SELECTION_DIVERGENCE"
            } else {
                "COMPOSITE_INTERNAL_DIVERGENCE"
            };
            let candidate_free_input = if step == 0 {
                RAW_PROMPT_TOKEN_IDS[TEACHER_PREFILL_TOKENS]
            } else {
                candidate.candidate_free_token_ids[step - 1]
            };
            Some(json!({
                "step": step,
                "absolute_token_position": TEACHER_PREFILL_TOKENS + step,
                "teacher_input_token_id": candidate.teacher_input_token_ids[step],
                "candidate_free_input_token_id": candidate_free_input,
                "cpu_f32_reference_expected_token_id": expected,
                "candidate_same_hidden_full_f32_token_id": same_hidden_full_f32,
                "candidate_top4_token_ids": candidates,
                "candidate_top4_contains_cpu_reference": candidates_contain_reference,
                "candidate_top4_contains_same_hidden_full_f32": candidates_contain_same_hidden_full_f32,
                "candidate_f32_reranked_token_id": reranked,
                "candidate_free_token_id": free,
                "classification": classification,
            }))
        })
        .collect()
}

fn mismatch_details(expected: &[u32], observed: &[u32]) -> Vec<Value> {
    (0..expected.len().max(observed.len()))
        .filter_map(|position| {
            let expected_token = expected.get(position).copied();
            let observed_token = observed.get(position).copied();
            (expected_token != observed_token).then(|| {
                json!({
                    "position": position,
                    "expected_token_id": expected_token,
                    "observed_token_id": observed_token,
                })
            })
        })
        .collect()
}

fn all_distinct<const N: usize>(tokens: &[u32; N]) -> bool {
    tokens
        .iter()
        .enumerate()
        .all(|(index, token)| !tokens[..index].contains(token))
}

fn is_excluded_eog(token: u32) -> bool {
    EXCLUDED_EOG_TOKEN_IDS.contains(&token)
}

fn token_occurrences(tokens: &[u32], predicate: impl Fn(u32) -> bool) -> Vec<Value> {
    tokens
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(position, token)| {
            predicate(token).then(|| json!({"position": position, "token_id": token}))
        })
        .collect()
}

fn candidate_token_occurrences<const N: usize>(
    candidates: &[[u32; N]],
    predicate: impl Fn(u32) -> bool,
) -> Vec<Value> {
    candidates
        .iter()
        .enumerate()
        .flat_map(|(position, tokens)| {
            let predicate = &predicate;
            tokens
                .iter()
                .copied()
                .enumerate()
                .filter_map(move |(candidate_index, token)| {
                    predicate(token).then(|| {
                        json!({
                            "position": position,
                            "candidate_index": candidate_index,
                            "token_id": token,
                        })
                    })
                })
        })
        .collect()
}

fn sha256_compact_json(value: &Value) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn validate_frozen_contract_hashes() -> Result<(), Box<dyn Error>> {
    let contracts = [
        (
            "raw prompt",
            sha256_compact_json(&json!(RAW_PROMPT_TOKEN_IDS))?,
            RAW_PROMPT_TOKEN_IDS_SHA256,
        ),
        (
            "teacher prefill",
            sha256_compact_json(&json!(&RAW_PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS]))?,
            TEACHER_PREFILL_TOKEN_IDS_SHA256,
        ),
        (
            "excluded EOG IDs",
            sha256_compact_json(&json!(EXCLUDED_EOG_TOKEN_IDS))?,
            EXCLUDED_EOG_TOKEN_IDS_SHA256,
        ),
        (
            "OmniInfer suppressed free128",
            sha256_compact_json(&json!(&OMNIINFER_SUPPRESSED_FREE128_TOKEN_IDS[..]))?,
            OMNIINFER_SUPPRESSED_FREE128_SHA256,
        ),
    ];
    for (name, observed, expected) in contracts {
        if observed != expected {
            return Err(format!(
                "frozen {name} compact-JSON SHA-256 changed: expected {expected}, got {observed}"
            )
            .into());
        }
    }
    Ok(())
}

fn parse_args_from<I>(values: I) -> Result<Args, Box<dyn Error>>
where
    I: IntoIterator<Item = OsString>,
{
    let mut values = values.into_iter();
    let _program = values.next().ok_or("argv omitted program name")?;
    let mut model_dir = None;
    let mut candidate_profile = None;
    while let Some(argument) = values.next() {
        let argument = argument.to_string_lossy();
        match argument.as_ref() {
            "--model-dir" => {
                if model_dir.is_some() {
                    return Err("--model-dir may be specified at most once".into());
                }
                model_dir = Some(PathBuf::from(
                    values.next().ok_or("--model-dir requires a value")?,
                ));
            }
            "--candidate-profile" => {
                if candidate_profile.is_some() {
                    return Err("--candidate-profile may be specified at most once".into());
                }
                let value = values
                    .next()
                    .ok_or("--candidate-profile requires a value")?;
                candidate_profile = Some(CandidateProfile::parse(&value.to_string_lossy())?);
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
        candidate_profile: candidate_profile.unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_compact_json_hashes_match_the_cross_runtime_contract() {
        validate_frozen_contract_hashes().unwrap();
    }

    #[test]
    fn teacher_inputs_start_with_raw_token_13_then_follow_reference() {
        let reference = (0..OUTPUT_TOKENS as u32).collect::<Vec<_>>();
        let inputs = teacher_inputs(&reference).unwrap();

        assert_eq!(inputs.len(), OUTPUT_TOKENS);
        assert_eq!(inputs[0], RAW_PROMPT_TOKEN_IDS[12]);
        assert_eq!(&inputs[1..], &reference[..OUTPUT_TOKENS - 1]);
    }

    #[test]
    fn mismatch_and_eog_receipts_preserve_exact_positions() {
        assert_eq!(
            mismatch_details(&[1, 2, 3], &[1, 9]),
            vec![
                json!({"position": 1, "expected_token_id": 2, "observed_token_id": 9}),
                json!({"position": 2, "expected_token_id": 3, "observed_token_id": null}),
            ]
        );
        assert_eq!(
            token_occurrences(&[7, 248046, 8], is_excluded_eog),
            vec![json!({"position": 1, "token_id": 248046})]
        );
    }

    #[test]
    fn candidate_audit_detects_duplicates_and_masked_rows() {
        assert!(all_distinct(&[1, 2, 3, 4]));
        assert!(!all_distinct(&[1, 2, 1, 4]));
        assert_eq!(
            candidate_token_occurrences(&[[7, 248044, 8, 248065]], is_excluded_eog),
            vec![
                json!({"position": 0, "candidate_index": 1, "token_id": 248044}),
                json!({"position": 0, "candidate_index": 3, "token_id": 248065}),
            ]
        );
    }

    #[test]
    fn internal_divergence_classifies_same_hidden_selection_exactly() {
        let candidate = CandidateRuns {
            teacher_input_token_ids: vec![234],
            teacher_same_hidden_full_f32_token_ids: vec![109],
            teacher_top4_candidate_token_ids: vec![[109, 123, 122, 253]],
            teacher_reranked_token_ids: vec![109],
            checked_reset_calls: 1,
            candidate_free_token_ids: vec![109],
        };

        assert_eq!(
            internal_divergence_details(&[123], &candidate),
            vec![json!({
                "step": 0,
                "absolute_token_position": 12,
                "teacher_input_token_id": 234,
                "candidate_free_input_token_id": 271,
                "cpu_f32_reference_expected_token_id": 123,
                "candidate_same_hidden_full_f32_token_id": 109,
                "candidate_top4_token_ids": [109, 123, 122, 253],
                "candidate_top4_contains_cpu_reference": true,
                "candidate_top4_contains_same_hidden_full_f32": true,
                "candidate_f32_reranked_token_id": 109,
                "candidate_free_token_id": 109,
                "classification": "BODY_HIDDEN_PRECISION_DIVERGENCE_NOT_CANDIDATE_OMISSION_OR_RERANK_FAILURE",
            })]
        );
    }

    #[test]
    fn argument_parser_is_strict() {
        let args = parse_args_from([
            OsString::from("gate"),
            OsString::from("--model-dir"),
            OsString::from("/model"),
        ])
        .unwrap();
        assert_eq!(args.model_dir, PathBuf::from("/model"));
        assert_eq!(args.candidate_profile, CandidateProfile::Fused);

        let legacy = parse_args_from([
            OsString::from("gate"),
            OsString::from("--model-dir"),
            OsString::from("/model"),
            OsString::from("--candidate-profile"),
            OsString::from("legacy"),
        ])
        .unwrap();
        assert_eq!(legacy.candidate_profile, CandidateProfile::Legacy);

        assert!(parse_args_from([
            OsString::from("gate"),
            OsString::from("--model-dir"),
            OsString::from("/model"),
            OsString::from("--candidate-profile"),
            OsString::from("unknown"),
        ])
        .unwrap_err()
        .to_string()
        .contains("expected fused or legacy"));

        assert!(parse_args_from([
            OsString::from("gate"),
            OsString::from("--model-dir"),
            OsString::from("/model"),
            OsString::from("--candidate-profile"),
            OsString::from("fused"),
            OsString::from("--candidate-profile"),
            OsString::from("legacy"),
        ])
        .unwrap_err()
        .to_string()
        .contains("at most once"));

        assert!(parse_args_from([
            OsString::from("gate"),
            OsString::from("--model-dir"),
            OsString::from("/a"),
            OsString::from("--model-dir"),
            OsString::from("/b"),
        ])
        .unwrap_err()
        .to_string()
        .contains("at most once"));
    }
}

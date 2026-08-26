//! Correctness-only Q4_0 tied-lm-head candidate coverage gate.
//!
//! This example replays the exact 12-token prefill, 128 teacher inputs, and
//! five-EOG exclusion policy of the suppressed free128 semantic gate. The
//! existing fused W8 body supplies each final normalized hidden row. A pure
//! CPU Q4_0 block32 oracle then asks whether K=4, 8, and 16 contain the exact
//! same-hidden F32 tied-head winner. No latency or throughput is measured.

use std::error::Error;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use apxinf_core::{Backend, CpuBackend, Device, Tensor};
use apxinf_metal::{PackedQ4_0RowsV1, Q4_0_PACKED_BYTES_PER_BLOCK_V1};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FORMAT: &str = "apxinf-qwen35-q4_0-tied-head-coverage-v1";
const QUALIFICATION: &str = "NON_FORMAL_CORRECTNESS_COVERAGE_NO_PERFORMANCE_RESULT";
const MAX_CONTEXT: usize = 256;
const OUTPUT_TOKENS: usize = 128;
const TEACHER_PREFILL_TOKENS: usize = 12;
const EXPECTED_VOCAB_SIZE: usize = 248_320;
const EXPECTED_HIDDEN_SIZE: usize = 1_024;
const COVERAGE_KS: [usize; 3] = [4, 8, 16];
const MAX_COVERAGE_K: usize = 16;
const EMBEDDING_TENSOR_NAME: &str = "model.language_model.embed_tokens.weight";

const RAW_PROMPT_TOKEN_IDS: [u32; 13] = [
    248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];
const EXCLUDED_EOG_TOKEN_IDS: [u32; 5] = [248044, 248046, 248063, 248064, 248065];

const RAW_PROMPT_TOKEN_IDS_SHA256: &str =
    "4b890fa15ee3d7db4e9dd18bd79c6362d40e9e016ae4f9f74cb7fc420ef3b6d3";
const TEACHER_PREFILL_TOKEN_IDS_SHA256: &str =
    "94b1660c815f507f8df7f4748b41b65bcb9ba09308930e15041b815000b2bdeb";
const EXCLUDED_EOG_TOKEN_IDS_SHA256: &str =
    "656e15a6ba9c76f492ba6bb34a0f2af4095ec3850dbb09b468228c2055ece9ca";
const CPU_F32_REFERENCE_TOKEN_IDS_SHA256: &str =
    "5bba10f53b153bb6a7d62efea7e0b6b6cb1b650c435e993c0fd171cd4e1b2f0a";
const FUSED_SAME_HIDDEN_F32_WINNER_TOKEN_IDS_SHA256: &str =
    "d36c8570e71953db5f5bc919b45108dee47a704b975c3b5785f0063519ce46d0";
const FUSED_SOURCE_W8_TOP4_TOKEN_IDS_SHA256: &str =
    "4b35e30839d8094be5f594682714d7c9ba3c00c2f778bd6d8313b1d8a02a0fa8";

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
}

struct CandidateHiddenRuns {
    q4_head: PackedQ4_0RowsV1,
    teacher_input_token_ids: Vec<u32>,
    same_hidden_f32_winner_token_ids: Vec<u32>,
    source_w8_top4_token_ids: Vec<[u32; apxinf_metal::W8_TOP_K]>,
    normalized_hidden_f32: Vec<f32>,
}

struct Q4CoverageRuns {
    q4_rank_of_f32_winner: Vec<usize>,
    candidate_token_ids_by_k: Vec<(usize, Vec<Vec<u32>>)>,
    misses_by_k: Vec<(usize, Vec<Value>)>,
}

/// Shared real-checkpoint fixture used by the separately versioned Metal
/// exact-candidate gate. The CPU coverage-v1 receipt remains CPU-only; this
/// surface only prevents the two gates from silently rebuilding different
/// teacher hiddens or Q4_0 weights.
pub(crate) struct Q4RealCheckpointFixtureV1 {
    pub(crate) model_dir: PathBuf,
    pub(crate) q4_head: PackedQ4_0RowsV1,
    pub(crate) cpu_f32_reference_token_ids: Vec<u32>,
    pub(crate) teacher_input_token_ids: Vec<u32>,
    pub(crate) same_hidden_f32_winner_token_ids: Vec<u32>,
    pub(crate) source_w8_top4_token_ids: Vec<[u32; apxinf_metal::W8_TOP_K]>,
    pub(crate) normalized_hidden_f32: Vec<f32>,
}

/// Rebuild the exact suppressed-free128 teacher-hidden fixture and its tied
/// Q4_0 rows. This records no timing and performs no candidate fallback.
pub(crate) fn build_q4_real_checkpoint_fixture_v1(
    model_dir: &Path,
) -> Result<Q4RealCheckpointFixtureV1, Box<dyn Error>> {
    validate_frozen_policy_hashes()?;
    let model_dir = std::fs::canonicalize(model_dir)?;
    let cpu_f32_reference_token_ids = run_cpu_f32_reference(&model_dir)?;
    let candidate =
        run_fused_candidate_hiddens_and_pack_q4(&model_dir, &cpu_f32_reference_token_ids)?;
    Ok(Q4RealCheckpointFixtureV1 {
        model_dir,
        q4_head: candidate.q4_head,
        cpu_f32_reference_token_ids,
        teacher_input_token_ids: candidate.teacher_input_token_ids,
        same_hidden_f32_winner_token_ids: candidate.same_hidden_f32_winner_token_ids,
        source_w8_top4_token_ids: candidate.source_w8_top4_token_ids,
        normalized_hidden_f32: candidate.normalized_hidden_f32,
    })
}

/// Compute the live CPU-Q4 K=4 trajectory from the same packed weights and
/// hidden rows used by the Metal exact-candidate gate.
pub(crate) fn cpu_q4_k4_trajectory_v1(
    q4_head: &PackedQ4_0RowsV1,
    normalized_hidden_f32: &[f32],
) -> Result<Vec<[u32; 4]>, Box<dyn Error>> {
    let logits = q4_0_batch_scores(q4_head, normalized_hidden_f32, OUTPUT_TOKENS)?;
    let logits = logits.as_f32()?;
    logits
        .chunks_exact(EXPECTED_VOCAB_SIZE)
        .enumerate()
        .map(|(step, scores)| {
            q4_head
                .topk_scores_excluding(scores, 4, &EXCLUDED_EOG_TOKEN_IDS)?
                .try_into()
                .map_err(|candidates: Vec<u32>| {
                    format!(
                        "CPU Q4_0 step {step} returned {} K=4 candidates",
                        candidates.len()
                    )
                    .into()
                })
        })
        .collect()
}

fn usage() -> &'static str {
    "Usage: qwen35_q4_0_tied_head_coverage_v1 --model-dir PATH"
}

fn main() {
    match real_main() {
        Ok(receipt) => {
            println!(
                "{}",
                serde_json::to_string(&receipt).expect("serialize Q4_0 coverage receipt")
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
        return Err("the fused W8 teacher-hidden source requires macOS".into());
    }
    let args = parse_args_from(std::env::args_os())?;
    validate_frozen_policy_hashes()?;
    let model_dir = std::fs::canonicalize(&args.model_dir)?;

    let cpu_f32_reference_token_ids = run_cpu_f32_reference(&model_dir)?;
    let candidate =
        run_fused_candidate_hiddens_and_pack_q4(&model_dir, &cpu_f32_reference_token_ids)?;

    let teacher_input_sha256 = sha256_compact_json(&json!(candidate.teacher_input_token_ids))?;
    let cpu_reference_sha256 = sha256_compact_json(&json!(cpu_f32_reference_token_ids))?;
    let same_hidden_winner_sha256 =
        sha256_compact_json(&json!(candidate.same_hidden_f32_winner_token_ids))?;
    let source_w8_top4_sha256 = sha256_compact_json(&json!(candidate.source_w8_top4_token_ids))?;
    let normalized_hidden_f32_le_sha256 = sha256_f32_le(&candidate.normalized_hidden_f32)?;
    let per_step_normalized_hidden_f32_le_sha256 = candidate
        .normalized_hidden_f32
        .chunks_exact(EXPECTED_HIDDEN_SIZE)
        .map(sha256_f32_le)
        .collect::<Result<Vec<_>, _>>()?;
    let packed_q4_0_tied_head_sha256 = sha256_q4_0_blocks(&candidate.q4_head);

    let coverage = run_q4_0_coverage(
        &candidate.q4_head,
        &candidate.normalized_hidden_f32,
        &candidate.teacher_input_token_ids,
        &candidate.same_hidden_f32_winner_token_ids,
        &per_step_normalized_hidden_f32_le_sha256,
    )?;

    let q4_rank_sha256 = sha256_compact_json(&json!(coverage.q4_rank_of_f32_winner))?;
    let mut coverage_receipts = Vec::with_capacity(COVERAGE_KS.len());
    let mut every_k_has_complete_coverage = true;
    for &k in &COVERAGE_KS {
        let candidates = coverage
            .candidate_token_ids_by_k
            .iter()
            .find_map(|(candidate_k, candidates)| (*candidate_k == k).then_some(candidates))
            .ok_or_else(|| format!("missing candidate trajectory for K={k}"))?;
        let misses = coverage
            .misses_by_k
            .iter()
            .find_map(|(miss_k, misses)| (*miss_k == k).then_some(misses))
            .ok_or_else(|| format!("missing miss audit for K={k}"))?;
        let complete = misses.is_empty();
        every_k_has_complete_coverage &= complete;
        coverage_receipts.push(json!({
            "k": k,
            "same_hidden_f32_winner_covered_at_all_128_steps": complete,
            "miss_count": misses.len(),
            "first_miss_step": misses.first().and_then(|miss| miss.get("step")).cloned(),
            "first_miss": misses.first().cloned(),
            "misses": misses,
            "candidate_token_ids_by_step": candidates,
            "candidate_token_ids_sha256": sha256_compact_json(&json!(candidates))?,
        }));
    }

    let raw_prompt_sha256 = sha256_compact_json(&json!(RAW_PROMPT_TOKEN_IDS))?;
    let teacher_prefill_sha256 =
        sha256_compact_json(&json!(&RAW_PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS]))?;
    let excluded_eog_sha256 = sha256_compact_json(&json!(EXCLUDED_EOG_TOKEN_IDS))?;
    let source_contract_exact = candidate.teacher_input_token_ids.len() == OUTPUT_TOKENS
        && candidate.same_hidden_f32_winner_token_ids.len() == OUTPUT_TOKENS
        && candidate.source_w8_top4_token_ids.len() == OUTPUT_TOKENS
        && candidate.normalized_hidden_f32.len() == OUTPUT_TOKENS * EXPECTED_HIDDEN_SIZE
        && per_step_normalized_hidden_f32_le_sha256.len() == OUTPUT_TOKENS
        && cpu_reference_sha256 == CPU_F32_REFERENCE_TOKEN_IDS_SHA256
        && same_hidden_winner_sha256 == FUSED_SAME_HIDDEN_F32_WINNER_TOKEN_IDS_SHA256
        && source_w8_top4_sha256 == FUSED_SOURCE_W8_TOP4_TOKEN_IDS_SHA256
        && candidate
            .same_hidden_f32_winner_token_ids
            .iter()
            .all(|token| !EXCLUDED_EOG_TOKEN_IDS.contains(token));
    let passed = source_contract_exact && every_k_has_complete_coverage;
    let decision = if passed {
        "GO_FOR_Q4_0_CANDIDATE_COVERAGE_ONLY"
    } else {
        "NO_GO"
    };

    Ok(json!({
        "format": FORMAT,
        "schema_version": 1,
        "qualification": QUALIFICATION,
        "claim_boundary": "Q4_0 tied-head candidate coverage on fixed teacher hiddens only; no Metal implementation, latency, throughput, or runtime ranking claim",
        "model": {
            "model_dir": model_dir.display().to_string(),
            "expected_family": "Qwen/Qwen3.5-0.8B",
            "expected_vocabulary_size": EXPECTED_VOCAB_SIZE,
            "expected_hidden_size": EXPECTED_HIDDEN_SIZE,
            "tied_embedding_tensor": EMBEDDING_TENSOR_NAME,
        },
        "teacher_hidden_source": {
            "constructor": "GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1",
            "entrypoint": "GeneralQwen35::teacher_forced_decode_candidates_excluding",
            "normalized_hidden_surface": "Qwen35MetalTeacherStep::normalized_hidden_f32",
            "same_as_suppressed_free128_gate": true,
            "ordinary_prefill_decode_paths_changed": false,
        },
        "q4_0_contract": {
            "implementation": "PackedQ4_0RowsV1 CPU correctness oracle",
            "block_size": apxinf_metal::Q4_0_BLOCK_SIZE_V1,
            "scale_storage": "IEEE FP16 bits per block",
            "quant_storage": "16 bytes; low nibble columns 0..15, high nibble columns 16..31",
            "scale_selection": "signed value with first strict maximum absolute magnitude divided by -8",
            "candidate_order": "descending finite Q4_0 score, exact ties by lowest token ID",
            "excluded_before_topk": true,
            "coverage_k": COVERAGE_KS,
            "packed_bytes_per_block": Q4_0_PACKED_BYTES_PER_BLOCK_V1,
            "packed_block_count": candidate.q4_head.blocks().len(),
            "packed_byte_count": candidate.q4_head.blocks().len() * Q4_0_PACKED_BYTES_PER_BLOCK_V1,
        },
        "workload": {
            "raw_prompt_token_ids": RAW_PROMPT_TOKEN_IDS,
            "teacher_prefill_token_ids": &RAW_PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS],
            "teacher_input_token_ids": candidate.teacher_input_token_ids,
            "teacher_step_count": OUTPUT_TOKENS,
            "sampling": "greedy",
            "eog_termination": false,
            "excluded_eog_token_ids": EXCLUDED_EOG_TOKEN_IDS,
        },
        "same_hidden_f32_oracle": {
            "winner_token_ids": candidate.same_hidden_f32_winner_token_ids,
            "winner_token_ids_sha256": same_hidden_winner_sha256,
        },
        "q4_0_ranking": {
            "rank_is_one_based_over_all_non_excluded_vocabulary_rows": true,
            "same_hidden_f32_winner_rank_by_step": coverage.q4_rank_of_f32_winner,
            "same_hidden_f32_winner_rank_by_step_sha256": q4_rank_sha256,
        },
        "counts": {
            "teacher_step_count": OUTPUT_TOKENS,
            "normalized_hidden_value_count": candidate.normalized_hidden_f32.len(),
            "same_hidden_f32_winner_rank_1_count": coverage.q4_rank_of_f32_winner.iter().filter(|&&rank| rank == 1).count(),
            "same_hidden_f32_winner_rank_2_count": coverage.q4_rank_of_f32_winner.iter().filter(|&&rank| rank == 2).count(),
            "same_hidden_f32_winner_max_q4_rank": coverage.q4_rank_of_f32_winner.iter().copied().max(),
        },
        "coverage": coverage_receipts,
        "hidden_evidence": {
            "encoding": "each finite F32 value as IEEE-754 bits in little-endian byte order",
            "normalized_hidden_shape": [OUTPUT_TOKENS, EXPECTED_HIDDEN_SIZE],
            "normalized_hidden_f32_le_sha256": normalized_hidden_f32_le_sha256,
            "per_step_normalized_hidden_f32_le_sha256": per_step_normalized_hidden_f32_le_sha256,
        },
        "hashes": {
            "algorithm": "SHA-256",
            "compact_json_encoding": "UTF-8 bytes produced by serde_json::to_vec",
            "raw_prompt_token_ids_sha256": raw_prompt_sha256,
            "teacher_prefill_token_ids_sha256": teacher_prefill_sha256,
            "excluded_eog_token_ids_sha256": excluded_eog_sha256,
            "teacher_input_token_ids_sha256": teacher_input_sha256,
            "cpu_f32_reference_token_ids_sha256": cpu_reference_sha256,
            "fused_source_w8_top4_token_ids_sha256": source_w8_top4_sha256,
            "packed_q4_0_tied_head_scale_le_then_nibbles_sha256": packed_q4_0_tied_head_sha256,
        },
        "admission": {
            "frozen_policy_hashes_exact": raw_prompt_sha256 == RAW_PROMPT_TOKEN_IDS_SHA256
                && teacher_prefill_sha256 == TEACHER_PREFILL_TOKEN_IDS_SHA256
                && excluded_eog_sha256 == EXCLUDED_EOG_TOKEN_IDS_SHA256,
            "suppressed_free128_teacher_hidden_source_exact": source_contract_exact,
            "all_k_have_zero_misses": every_k_has_complete_coverage,
            "decision": decision,
            "rule": "any miss at K=4, K=8, or K=16 forces NO_GO; no step/token fallback is permitted",
        },
        "forbidden_shortcuts_observed": false,
        "performance": {
            "samples": 0,
            "latency_recorded": false,
            "throughput_recorded": false,
            "formal_result": false,
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

fn run_fused_candidate_hiddens_and_pack_q4(
    model_dir: &Path,
    cpu_reference_token_ids: &[u32],
) -> Result<CandidateHiddenRuns, Box<dyn Error>> {
    let (config, tensors) = load_model_inputs(model_dir)?;
    validate_model_contract(&config)?;
    let embedding = tensors
        .get(EMBEDDING_TENSOR_NAME)
        .ok_or_else(|| format!("checkpoint omitted {EMBEDDING_TENSOR_NAME}"))?;
    if embedding.shape().dims() != [EXPECTED_VOCAB_SIZE, EXPECTED_HIDDEN_SIZE] {
        return Err(format!(
            "expected tied embedding [{EXPECTED_VOCAB_SIZE}, {EXPECTED_HIDDEN_SIZE}], got {}",
            embedding.shape()
        )
        .into());
    }
    let embedding_f32 = embedding.to_f32_vec()?;
    let q4_head =
        PackedQ4_0RowsV1::pack_f32(&embedding_f32, EXPECTED_VOCAB_SIZE, EXPECTED_HIDDEN_SIZE)?;
    drop(embedding_f32);
    let mut model =
        GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1(
            config,
            tensors,
            Device::Cpu,
            MAX_CONTEXT,
        )?;
    let teacher_input_token_ids = teacher_inputs(cpu_reference_token_ids)?;
    let _ = model.prefill_for_generation(LlmInput::text(
        &RAW_PROMPT_TOKEN_IDS[..TEACHER_PREFILL_TOKENS],
    ))?;

    let mut same_hidden_f32_winner_token_ids = Vec::with_capacity(OUTPUT_TOKENS);
    let mut source_w8_top4_token_ids = Vec::with_capacity(OUTPUT_TOKENS);
    let mut normalized_hidden_f32 = Vec::with_capacity(OUTPUT_TOKENS * EXPECTED_HIDDEN_SIZE);
    for (step, &teacher_token) in teacher_input_token_ids.iter().enumerate() {
        let position = u32::try_from(TEACHER_PREFILL_TOKENS + step)?;
        let comparison = model.teacher_forced_decode_candidates_excluding(
            teacher_token,
            position,
            &EXCLUDED_EOG_TOKEN_IDS,
        )?;
        if comparison.normalized_hidden_f32.len() != EXPECTED_HIDDEN_SIZE {
            return Err(format!(
                "teacher step {step} exposed {} normalized hidden values, expected {EXPECTED_HIDDEN_SIZE}",
                comparison.normalized_hidden_f32.len()
            )
            .into());
        }
        if comparison
            .normalized_hidden_f32
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(format!("teacher step {step} exposed a non-finite hidden value").into());
        }
        if EXCLUDED_EOG_TOKEN_IDS.contains(&comparison.cpu_token) {
            return Err(format!(
                "teacher step {step} same-hidden F32 winner {} is excluded",
                comparison.cpu_token
            )
            .into());
        }
        same_hidden_f32_winner_token_ids.push(comparison.cpu_token);
        source_w8_top4_token_ids.push(comparison.w8_candidates);
        normalized_hidden_f32.extend_from_slice(&comparison.normalized_hidden_f32);
    }

    Ok(CandidateHiddenRuns {
        q4_head,
        teacher_input_token_ids,
        same_hidden_f32_winner_token_ids,
        source_w8_top4_token_ids,
        normalized_hidden_f32,
    })
}

fn run_q4_0_coverage(
    q4_head: &PackedQ4_0RowsV1,
    normalized_hidden_f32: &[f32],
    teacher_input_token_ids: &[u32],
    same_hidden_f32_winner_token_ids: &[u32],
    per_step_hidden_sha256: &[String],
) -> Result<Q4CoverageRuns, Box<dyn Error>> {
    if normalized_hidden_f32.len() != OUTPUT_TOKENS * EXPECTED_HIDDEN_SIZE
        || teacher_input_token_ids.len() != OUTPUT_TOKENS
        || same_hidden_f32_winner_token_ids.len() != OUTPUT_TOKENS
        || per_step_hidden_sha256.len() != OUTPUT_TOKENS
    {
        return Err("Q4_0 coverage inputs do not contain exactly 128 teacher steps".into());
    }
    let logits = q4_0_batch_scores(q4_head, normalized_hidden_f32, OUTPUT_TOKENS)?;
    let logits = logits.as_f32()?;
    let mut q4_rank_of_f32_winner = Vec::with_capacity(OUTPUT_TOKENS);
    let mut top16_by_step = Vec::with_capacity(OUTPUT_TOKENS);
    for (step, scores) in logits.chunks_exact(EXPECTED_VOCAB_SIZE).enumerate() {
        let winner = same_hidden_f32_winner_token_ids[step];
        q4_rank_of_f32_winner.push(q4_rank_excluding(scores, winner, &EXCLUDED_EOG_TOKEN_IDS)?);
        top16_by_step.push(q4_head.topk_scores_excluding(
            scores,
            MAX_COVERAGE_K,
            &EXCLUDED_EOG_TOKEN_IDS,
        )?);
    }

    let mut candidate_token_ids_by_k = Vec::with_capacity(COVERAGE_KS.len());
    let mut misses_by_k = Vec::with_capacity(COVERAGE_KS.len());
    for &k in &COVERAGE_KS {
        let candidates_by_step = top16_by_step
            .iter()
            .map(|candidates| candidates[..k].to_vec())
            .collect::<Vec<_>>();
        let misses = candidates_by_step
            .iter()
            .enumerate()
            .filter_map(|(step, candidates)| {
                let winner = same_hidden_f32_winner_token_ids[step];
                if candidates.contains(&winner) {
                    return None;
                }
                let scores = &logits
                    [step * EXPECTED_VOCAB_SIZE..(step + 1) * EXPECTED_VOCAB_SIZE];
                Some(json!({
                    "step": step,
                    "absolute_token_position": TEACHER_PREFILL_TOKENS + step,
                    "teacher_input_token_id": teacher_input_token_ids[step],
                    "same_hidden_f32_winner_token_id": winner,
                    "same_hidden_f32_winner_q4_rank_1_based": q4_rank_of_f32_winner[step],
                    "same_hidden_f32_winner_q4_score": scores[winner as usize],
                    "candidate_token_ids": candidates,
                    "candidate_cutoff_q4_score": scores[*candidates.last().expect("K is non-zero") as usize],
                    "normalized_hidden_f32_le_sha256": per_step_hidden_sha256[step],
                }))
            })
            .collect::<Vec<_>>();
        candidate_token_ids_by_k.push((k, candidates_by_step));
        misses_by_k.push((k, misses));
    }

    Ok(Q4CoverageRuns {
        q4_rank_of_f32_winner,
        candidate_token_ids_by_k,
        misses_by_k,
    })
}

fn q4_0_batch_scores(
    q4_head: &PackedQ4_0RowsV1,
    normalized_hidden_f32: &[f32],
    hidden_rows: usize,
) -> Result<Tensor, Box<dyn Error>> {
    if normalized_hidden_f32.len() != hidden_rows * q4_head.columns() {
        return Err(format!(
            "Q4_0 batch hidden has {} values, expected {}",
            normalized_hidden_f32.len(),
            hidden_rows * q4_head.columns()
        )
        .into());
    }
    let hidden = Tensor::from_f32(vec![hidden_rows, q4_head.columns()], normalized_hidden_f32)?;
    let weights = Tensor::from_f32_vec(
        vec![q4_head.rows(), q4_head.columns()],
        q4_head.dequantize_f32(),
    )?;
    Ok(CpuBackend.matmul_rhs_transposed(&hidden, &weights)?)
}

fn q4_rank_excluding(
    scores: &[f32],
    token: u32,
    excluded_token_ids: &[u32],
) -> Result<usize, Box<dyn Error>> {
    let token_index = token as usize;
    if token_index >= scores.len() || excluded_token_ids.contains(&token) {
        return Err(format!("cannot rank missing or excluded token {token}").into());
    }
    if let Some(index) = scores.iter().position(|score| !score.is_finite()) {
        return Err(format!("non-finite Q4_0 score at token {index}").into());
    }
    let token_score = scores[token_index];
    Ok(1 + scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(candidate, score)| {
            let candidate = *candidate as u32;
            !excluded_token_ids.contains(&candidate)
                && (*score > token_score || (*score == token_score && candidate < token))
        })
        .count())
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
    if config.text.vocab_size != EXPECTED_VOCAB_SIZE
        || config.text.hidden_size != EXPECTED_HIDDEN_SIZE
    {
        return Err(format!(
            "coverage gate expected vocab/hidden {EXPECTED_VOCAB_SIZE}/{EXPECTED_HIDDEN_SIZE}, got {}/{}",
            config.text.vocab_size, config.text.hidden_size
        )
        .into());
    }
    if EXCLUDED_EOG_TOKEN_IDS
        .iter()
        .any(|&token| token as usize >= config.text.vocab_size)
    {
        return Err("coverage gate EOG exclusion is outside the model vocabulary".into());
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

fn sha256_compact_json(value: &Value) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn sha256_f32_le(values: &[f32]) -> Result<String, Box<dyn Error>> {
    let mut digest = Sha256::new();
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("non-finite F32 hash input at element {index}").into());
        }
        digest.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_q4_0_blocks(q4_head: &PackedQ4_0RowsV1) -> String {
    let mut digest = Sha256::new();
    for block in q4_head.blocks() {
        digest.update(block.scale_f16_bits().to_le_bytes());
        digest.update(block.quant_nibbles());
    }
    format!("{:x}", digest.finalize())
}

fn validate_frozen_policy_hashes() -> Result<(), Box<dyn Error>> {
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

    #[test]
    fn frozen_policy_hashes_match_suppressed_gate() {
        validate_frozen_policy_hashes().unwrap();
    }

    #[test]
    fn teacher_inputs_start_with_raw_token_13_then_follow_reference() {
        let reference = (0..OUTPUT_TOKENS as u32).collect::<Vec<_>>();
        let inputs = teacher_inputs(&reference).unwrap();
        assert_eq!(inputs[0], RAW_PROMPT_TOKEN_IDS[12]);
        assert_eq!(&inputs[1..], &reference[..OUTPUT_TOKENS - 1]);
    }

    #[test]
    fn q4_rank_uses_low_token_ties_and_exclusions() {
        let scores = [5.0, 5.0, 6.0, 5.0, 7.0];
        assert_eq!(q4_rank_excluding(&scores, 3, &[2]).unwrap(), 4);
        assert_eq!(q4_rank_excluding(&scores, 1, &[2]).unwrap(), 3);
        assert!(q4_rank_excluding(&scores, 2, &[2]).is_err());
    }

    #[test]
    fn batch_scores_match_scalar_q4_oracle_on_clear_margins() {
        let rows = 6;
        let columns = 32;
        let weights = (0..rows * columns)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 31.0)
            .collect::<Vec<_>>();
        let hiddens = (0..2 * columns)
            .map(|index| ((index * 13 % 67) as f32 - 33.0) / 29.0)
            .collect::<Vec<_>>();
        let packed = PackedQ4_0RowsV1::pack_f32(&weights, rows, columns).unwrap();
        let batch = q4_0_batch_scores(&packed, &hiddens, 2).unwrap();
        for (hidden, observed) in hiddens
            .chunks_exact(columns)
            .zip(batch.as_f32().unwrap().chunks_exact(rows))
        {
            let expected = packed.scores(hidden).unwrap();
            for (&left, &right) in expected.iter().zip(observed) {
                assert!((left - right).abs() < 1.0e-4, "{left} versus {right}");
            }
            assert_eq!(
                packed.topk_excluding(hidden, 4, &[]).unwrap(),
                packed.topk_scores_excluding(observed, 4, &[]).unwrap()
            );
        }
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
        assert!(
            parse_args_from([OsString::from("gate"), OsString::from("--unknown"),])
                .unwrap_err()
                .to_string()
                .contains("unknown argument")
        );
    }
}

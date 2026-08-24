//! Compare ApxInf Qwen3.5 CPU inference with a pinned Transformers oracle.
//!
//! Generate the reference with `scripts/qwen35_transformers_oracle.py`, then
//! run this example with `--features accelerate`. No tokenizer is involved in
//! this process: the raw token ids come from the signed-off JSON manifest.

use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;

use apxinf_core::{Device, Tensor};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use serde_json::{json, Value};

const ORACLE_FORMAT: &str = "apxinf-qwen35-transformers-oracle-v1";
const REPO_ID: &str = "Qwen/Qwen3.5-0.8B";
const LOCKED_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
const LOCKED_CHECKPOINT_SHA256: &str =
    "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696";

// Frozen after the first signed-off macOS/Accelerate FP32 measurement. The
// observed maxima were: logits 1.4043e-4 / 4.4104e-6, convolution state
// 6.4850e-5 / 3.9879e-6, recurrent state 1.1444e-5 / 3.7771e-6, and cached
// versus fresh logits 4.1008e-5 / 1.5536e-6 (max_abs / nrmse). Keep these as
// compile-time constants: changing a limit requires an explicit review and a
// new gate format, never a command-line escape hatch.
const GATE_FORMAT: &str = "apxinf-qwen35-oracle-gate-v1";
const GATE_TOP_K: usize = 20;
const GATE_TOP_K_MIN_OVERLAP: usize = 19;
const GATE_GREEDY_MIN_TOKENS: usize = 10;
const EXPECTED_LINEAR_STATE_SNAPSHOTS: usize = 36;
const EXPECTED_CONV_STATE_TENSORS: usize = EXPECTED_LINEAR_STATE_SNAPSHOTS * 3;
const EXPECTED_RECURRENT_STATE_TENSORS: usize = EXPECTED_LINEAR_STATE_SNAPSHOTS;

#[derive(Clone, Copy, Debug)]
struct MetricLimits {
    max_abs: f64,
    max_nrmse: f64,
    min_cosine: f64,
}

const LOGITS_LIMITS: MetricLimits = MetricLimits {
    max_abs: 2.0e-4,
    max_nrmse: 1.0e-5,
    min_cosine: 0.999_999_999,
};
const CONV_STATE_LIMITS: MetricLimits = MetricLimits {
    max_abs: 1.0e-4,
    max_nrmse: 1.0e-5,
    min_cosine: 0.999_999_999,
};
const RECURRENT_STATE_LIMITS: MetricLimits = MetricLimits {
    max_abs: 2.0e-5,
    max_nrmse: 1.0e-5,
    min_cosine: 0.999_999_999,
};
const CACHED_VS_FRESH_LIMITS: MetricLimits = MetricLimits {
    max_abs: 1.0e-4,
    max_nrmse: 5.0e-6,
    min_cosine: 0.999_999_999,
};

#[derive(Debug)]
struct MetricEnvelope {
    count: usize,
    max_abs: f64,
    max_nrmse: f64,
    min_cosine: f64,
}

impl MetricEnvelope {
    fn empty() -> Self {
        Self {
            count: 0,
            max_abs: 0.0,
            max_nrmse: 0.0,
            min_cosine: 1.0,
        }
    }

    fn observe(&mut self, metric: &Value, label: &str) -> Result<(), String> {
        let max_abs = metric_number(metric, "max_abs", label)?;
        let nrmse = metric_number(metric, "nrmse", label)?;
        let cosine = metric_number(metric, "cosine", label)?;
        self.count += 1;
        self.max_abs = self.max_abs.max(max_abs);
        self.max_nrmse = self.max_nrmse.max(nrmse);
        self.min_cosine = self.min_cosine.min(cosine);
        Ok(())
    }

    fn as_json(&self) -> Value {
        json!({
            "tensor_count": self.count,
            "max_abs": self.max_abs,
            "max_nrmse": self.max_nrmse,
            "min_cosine": self.min_cosine,
        })
    }
}

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    reference: PathBuf,
    manifest: PathBuf,
    output: Option<PathBuf>,
    max_context: usize,
    top_k: usize,
    force: bool,
    metal_w8_body_layer: Option<usize>,
}

fn usage() -> &'static str {
    "Usage: qwen35_oracle \
  --model-dir PATH \
  --reference PATH.safetensors \
  --manifest PATH.json \
  [--output metrics.json] \
  [--max-context 32] \
  [--top-k 20] \
  [--metal-w8-body-layer INDEX] \
  [--force]"
}

fn next_value(iter: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<OsString, String> {
    iter.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_usize(value: OsString, flag: &str) -> Result<usize, String> {
    value
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag}: {error}"))
}

fn parse_args() -> Result<Args, String> {
    let mut model_dir = None;
    let mut reference = None;
    let mut manifest = None;
    let mut output = None;
    let mut max_context = 32usize;
    let mut top_k = 20usize;
    let mut force = false;
    let mut metal_w8_body_layer = None;
    let mut iter = env::args_os().skip(1);
    while let Some(raw_flag) = iter.next() {
        let flag = raw_flag.to_string_lossy();
        match flag.as_ref() {
            "--model-dir" => model_dir = Some(PathBuf::from(next_value(&mut iter, &flag)?)),
            "--reference" => reference = Some(PathBuf::from(next_value(&mut iter, &flag)?)),
            "--manifest" => manifest = Some(PathBuf::from(next_value(&mut iter, &flag)?)),
            "--output" => output = Some(PathBuf::from(next_value(&mut iter, &flag)?)),
            "--max-context" => max_context = parse_usize(next_value(&mut iter, &flag)?, &flag)?,
            "--top-k" => top_k = parse_usize(next_value(&mut iter, &flag)?, &flag)?,
            "--metal-w8-body-layer" => {
                metal_w8_body_layer = Some(parse_usize(next_value(&mut iter, &flag)?, &flag)?)
            }
            "--force" => force = true,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
    }
    if max_context == 0 {
        return Err("--max-context must be greater than zero".into());
    }
    if top_k == 0 {
        return Err("--top-k must be greater than zero".into());
    }
    if top_k < GATE_TOP_K {
        return Err(format!(
            "--top-k {top_k} would weaken the frozen gate; use at least {GATE_TOP_K}"
        ));
    }
    Ok(Args {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required\n{}", usage()))?,
        reference: reference.ok_or_else(|| format!("--reference is required\n{}", usage()))?,
        manifest: manifest.ok_or_else(|| format!("--manifest is required\n{}", usage()))?,
        output,
        max_context,
        top_k,
        force,
        metal_w8_body_layer,
    })
}

fn json_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("manifest field {key:?} must be a string"))
}

fn json_u32(value: &Value, key: &str) -> Result<u32, String> {
    let integer = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("manifest field {key:?} must be an unsigned integer"))?;
    u32::try_from(integer).map_err(|_| format!("manifest field {key:?} exceeds u32"))
}

fn manifest_input_ids(manifest: &Value) -> Result<Vec<u32>, String> {
    let values = manifest
        .get("input_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "manifest input_ids must be an array".to_string())?;
    if values.is_empty() {
        return Err("manifest input_ids cannot be empty".into());
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let integer = value
                .as_u64()
                .ok_or_else(|| format!("manifest input_ids[{index}] is not unsigned"))?;
            u32::try_from(integer).map_err(|_| format!("manifest input_ids[{index}] exceeds u32"))
        })
        .collect()
}

fn manifest_greedy_trajectory(manifest: &Value) -> Result<Vec<u32>, String> {
    let trajectory = manifest
        .get("greedy_trajectory")
        .ok_or_else(|| "manifest greedy_trajectory is missing".to_string())?;
    let length = trajectory
        .get("length")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "manifest greedy_trajectory.length must be unsigned".to_string())?;
    let minimum_length = trajectory
        .get("minimum_length")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "manifest greedy_trajectory.minimum_length must be unsigned".to_string())?;
    if length < GATE_GREEDY_MIN_TOKENS {
        return Err(format!(
            "manifest greedy trajectory length {length} would weaken the frozen minimum {}",
            GATE_GREEDY_MIN_TOKENS
        ));
    }
    if minimum_length != GATE_GREEDY_MIN_TOKENS {
        return Err(format!(
            "manifest greedy trajectory minimum {minimum_length} does not match frozen minimum {}",
            GATE_GREEDY_MIN_TOKENS
        ));
    }
    if trajectory.get("do_sample").and_then(Value::as_bool) != Some(false)
        || trajectory.get("use_cache").and_then(Value::as_bool) != Some(true)
        || trajectory.get("eos_stopping").and_then(Value::as_bool) != Some(false)
    {
        return Err(
            "manifest greedy trajectory must bind do_sample=false, use_cache=true, eos_stopping=false"
                .into(),
        );
    }
    let values = trajectory
        .get("generated_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "manifest greedy_trajectory.generated_ids must be an array".to_string())?;
    if values.len() != length {
        return Err(format!(
            "manifest greedy trajectory has {} ids, declared length {length}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let integer = value.as_u64().ok_or_else(|| {
                format!("manifest greedy_trajectory.generated_ids[{index}] is not unsigned")
            })?;
            u32::try_from(integer).map_err(|_| {
                format!("manifest greedy_trajectory.generated_ids[{index}] exceeds u32")
            })
        })
        .collect()
}

fn verify_manifest(manifest: &Value) -> Result<(Vec<u32>, u32, Vec<u32>), String> {
    let required_strings = [
        ("format", ORACLE_FORMAT),
        ("repo_id", REPO_ID),
        ("revision", LOCKED_REVISION),
        ("checkpoint_sha256", LOCKED_CHECKPOINT_SHA256),
    ];
    for (key, expected) in required_strings {
        let actual = json_string(manifest, key)?;
        if actual != expected {
            return Err(format!(
                "manifest {key} mismatch: expected {expected:?}, got {actual:?}"
            ));
        }
    }
    let runtime = manifest
        .get("runtime")
        .ok_or_else(|| "manifest runtime is missing".to_string())?;
    for (key, expected) in [
        ("torch", "2.13.0"),
        ("transformers", "5.15.1"),
        ("safetensors", "0.8.0"),
        ("device", "cpu"),
        ("dtype", "float32"),
        ("attention_implementation", "eager"),
    ] {
        let actual = json_string(runtime, key)?;
        if actual != expected {
            return Err(format!(
                "manifest runtime.{key} mismatch: expected {expected:?}, got {actual:?}"
            ));
        }
    }
    if runtime.get("use_hub_kernels").and_then(Value::as_bool) != Some(false) {
        return Err("manifest must record use_hub_kernels=false".into());
    }
    Ok((
        manifest_input_ids(manifest)?,
        json_u32(manifest, "probe_token_id")?,
        manifest_greedy_trajectory(manifest)?,
    ))
}

fn verify_reference_metadata(
    metadata: &HashMap<String, String>,
    input_ids: &[u32],
    probe_token_id: u32,
    greedy_token_ids: &[u32],
) -> Result<(), String> {
    for (key, expected) in [
        ("format", ORACLE_FORMAT),
        ("repo_id", REPO_ID),
        ("revision", LOCKED_REVISION),
        ("checkpoint_sha256", LOCKED_CHECKPOINT_SHA256),
        ("torch_version", "2.13.0"),
        ("transformers_version", "5.15.1"),
        ("safetensors_version", "0.8.0"),
    ] {
        let actual = metadata
            .get(key)
            .ok_or_else(|| format!("reference metadata {key:?} is missing"))?;
        if actual != expected {
            return Err(format!(
                "reference metadata {key} mismatch: expected {expected:?}, got {actual:?}"
            ));
        }
    }
    let metadata_ids: Vec<u32> = serde_json::from_str(
        metadata
            .get("input_ids")
            .ok_or_else(|| "reference metadata input_ids is missing".to_string())?,
    )
    .map_err(|error| format!("parse reference metadata input_ids: {error}"))?;
    if metadata_ids != input_ids {
        return Err("reference metadata and manifest input_ids disagree".into());
    }
    let metadata_probe = metadata
        .get("probe_token_id")
        .ok_or_else(|| "reference metadata probe_token_id is missing".to_string())?
        .parse::<u32>()
        .map_err(|error| format!("parse reference metadata probe_token_id: {error}"))?;
    if metadata_probe != probe_token_id {
        return Err("reference metadata and manifest probe_token_id disagree".into());
    }
    let metadata_greedy_length = metadata
        .get("greedy_length")
        .ok_or_else(|| "reference metadata greedy_length is missing".to_string())?
        .parse::<usize>()
        .map_err(|error| format!("parse reference metadata greedy_length: {error}"))?;
    if metadata_greedy_length < GATE_GREEDY_MIN_TOKENS
        || metadata_greedy_length != greedy_token_ids.len()
    {
        return Err(format!(
            "reference greedy length {metadata_greedy_length} is below the frozen minimum or disagrees with the manifest"
        ));
    }
    let metadata_greedy_ids: Vec<u32> = serde_json::from_str(
        metadata
            .get("greedy_token_ids")
            .ok_or_else(|| "reference metadata greedy_token_ids is missing".to_string())?,
    )
    .map_err(|error| format!("parse reference metadata greedy_token_ids: {error}"))?;
    if metadata_greedy_ids != greedy_token_ids {
        return Err("reference metadata and manifest greedy token ids disagree".into());
    }
    Ok(())
}

fn finite_metric(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn metric_number(metric: &Value, key: &str, label: &str) -> Result<f64, String> {
    let value = metric
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("metric {label}.{key} must be a finite number"))?;
    if !value.is_finite() {
        return Err(format!("metric {label}.{key} is not finite"));
    }
    Ok(value)
}

fn limits_json(limits: MetricLimits) -> Value {
    json!({
        "max_abs_lte": limits.max_abs,
        "max_nrmse_lte": limits.max_nrmse,
        "min_cosine_gte": limits.min_cosine,
    })
}

fn gate_metric_envelope(
    name: &str,
    observed: &MetricEnvelope,
    expected_count: usize,
    limits: MetricLimits,
    checks: &mut Vec<Value>,
    failures: &mut Vec<String>,
) {
    let passed = observed.count == expected_count
        && observed.max_abs <= limits.max_abs
        && observed.max_nrmse <= limits.max_nrmse
        && observed.min_cosine >= limits.min_cosine;
    if observed.count != expected_count {
        failures.push(format!(
            "{name}: tensor count {} != {expected_count}",
            observed.count
        ));
    }
    if observed.max_abs > limits.max_abs {
        failures.push(format!(
            "{name}: max_abs {:.9e} > {:.9e}",
            observed.max_abs, limits.max_abs
        ));
    }
    if observed.max_nrmse > limits.max_nrmse {
        failures.push(format!(
            "{name}: max_nrmse {:.9e} > {:.9e}",
            observed.max_nrmse, limits.max_nrmse
        ));
    }
    if observed.min_cosine < limits.min_cosine {
        failures.push(format!(
            "{name}: min_cosine {:.12} < {:.12}",
            observed.min_cosine, limits.min_cosine
        ));
    }
    checks.push(json!({
        "name": name,
        "passed": passed,
        "expected_tensor_count": expected_count,
        "observed": observed.as_json(),
        "limits": limits_json(limits),
    }));
}

fn gate_top_k(
    name: &str,
    comparison: &Value,
    checks: &mut Vec<Value>,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    let k = comparison
        .get("k")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("top-k comparison {name}.k must be unsigned"))?;
    let overlap = comparison
        .get("overlap")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("top-k comparison {name}.overlap must be unsigned"))?;
    let top1_equal = comparison
        .get("top1_equal")
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("top-k comparison {name}.top1_equal must be boolean"))?;
    let passed = k == GATE_TOP_K && overlap >= GATE_TOP_K_MIN_OVERLAP && top1_equal;
    if k != GATE_TOP_K {
        failures.push(format!("{name}: compared k={k}, expected {GATE_TOP_K}"));
    }
    if overlap < GATE_TOP_K_MIN_OVERLAP {
        failures.push(format!(
            "{name}: top-{GATE_TOP_K} overlap {overlap} < {GATE_TOP_K_MIN_OVERLAP}"
        ));
    }
    if !top1_equal {
        failures.push(format!("{name}: top-1 token differs from the oracle"));
    }
    checks.push(json!({
        "name": name,
        "passed": passed,
        "observed": {
            "k": k,
            "overlap": overlap,
            "top1_equal": top1_equal,
        },
        "limits": {
            "k_eq": GATE_TOP_K,
            "overlap_gte": GATE_TOP_K_MIN_OVERLAP,
            "top1_equal": true,
        },
    }));
    Ok(())
}

fn gate_greedy_trajectory(
    observed: &[u32],
    expected: &[u32],
    checks: &mut Vec<Value>,
    failures: &mut Vec<String>,
) {
    let first_mismatch = observed
        .iter()
        .zip(expected)
        .position(|(observed, expected)| observed != expected);
    let exact_ids = observed == expected;
    let passed =
        expected.len() >= GATE_GREEDY_MIN_TOKENS && observed.len() == expected.len() && exact_ids;
    if expected.len() < GATE_GREEDY_MIN_TOKENS {
        failures.push(format!(
            "greedy_trajectory: reference length {} < frozen minimum {}",
            expected.len(),
            GATE_GREEDY_MIN_TOKENS
        ));
    }
    if observed.len() != expected.len() {
        failures.push(format!(
            "greedy_trajectory: generated length {} != reference length {}",
            observed.len(),
            expected.len()
        ));
    }
    if !exact_ids {
        let mismatch = first_mismatch
            .map(|index| format!(" at index {index}"))
            .unwrap_or_default();
        failures.push(format!(
            "greedy_trajectory: generated token ids differ from the oracle{mismatch}"
        ));
    }
    checks.push(json!({
        "name": "greedy_trajectory",
        "passed": passed,
        "observed": {
            "length": observed.len(),
            "generated_ids": observed,
            "first_mismatch_index": first_mismatch,
        },
        "expected": {
            "length": expected.len(),
            "generated_ids": expected,
        },
        "limits": {
            "minimum_length": GATE_GREEDY_MIN_TOKENS,
            "exact_length": true,
            "exact_token_ids": true,
            "override_supported": false,
        },
    }));
}

fn threshold_manifest() -> Value {
    json!({
        "format": GATE_FORMAT,
        "frozen": true,
        "threshold_overrides_supported": false,
        "calibration": {
            "comparison_format": "apxinf-qwen35-oracle-comparison-v1",
            "checkpoint_sha256": LOCKED_CHECKPOINT_SHA256,
            "device": "cpu",
            "dtype": "float32",
            "matmul_feature": "accelerate",
            "observed_maxima": {
                "transformers_logits": { "max_abs": 1.404285430908203e-4, "nrmse": 4.41033639795049e-6 },
                "convolution_state": { "max_abs": 6.4849853515625e-5, "nrmse": 3.987931672738041e-6 },
                "recurrent_state": { "max_abs": 1.1444091796875e-5, "nrmse": 3.7771299805887062e-6 },
                "cached_vs_fresh": { "max_abs": 4.100799560546875e-5, "nrmse": 1.5535626176949273e-6 },
            },
        },
        "limits": {
            "transformers_logits": limits_json(LOGITS_LIMITS),
            "convolution_state": limits_json(CONV_STATE_LIMITS),
            "recurrent_state": limits_json(RECURRENT_STATE_LIMITS),
            "cached_vs_fresh": limits_json(CACHED_VS_FRESH_LIMITS),
            "top_k": {
                "k_eq": GATE_TOP_K,
                "overlap_gte": GATE_TOP_K_MIN_OVERLAP,
                "top1_equal": true,
            },
            "greedy_trajectory": {
                "minimum_length": GATE_GREEDY_MIN_TOKENS,
                "exact_length": true,
                "exact_token_ids": true,
                "override_supported": false,
            },
        },
    })
}

fn metrics(candidate: &[f32], reference: &[f32]) -> Result<Value, String> {
    if candidate.len() != reference.len() {
        return Err(format!(
            "metric length mismatch: {} != {}",
            candidate.len(),
            reference.len()
        ));
    }
    if candidate.is_empty() {
        return Err("cannot compare empty tensors".into());
    }
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_square = 0.0f64;
    let mut candidate_square = 0.0f64;
    let mut reference_square = 0.0f64;
    let mut dot = 0.0f64;
    for (index, (&candidate, &reference)) in candidate.iter().zip(reference).enumerate() {
        if !candidate.is_finite() || !reference.is_finite() {
            return Err(format!("non-finite value at flattened index {index}"));
        }
        let candidate = f64::from(candidate);
        let reference = f64::from(reference);
        let delta = candidate - reference;
        let absolute = delta.abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
        sum_square += delta * delta;
        candidate_square += candidate * candidate;
        reference_square += reference * reference;
        dot += candidate * reference;
    }
    let count = candidate.len() as f64;
    let mean_abs = sum_abs / count;
    let rmse = (sum_square / count).sqrt();
    let candidate_rms = (candidate_square / count).sqrt();
    let reference_rms = (reference_square / count).sqrt();
    let cosine_denominator = (candidate_square * reference_square).sqrt();
    let cosine = if cosine_denominator > 0.0 {
        dot / cosine_denominator
    } else if sum_square == 0.0 {
        1.0
    } else {
        f64::NAN
    };
    let nrmse = if reference_rms > 0.0 {
        rmse / reference_rms
    } else if rmse == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    Ok(json!({
        "numel": candidate.len(),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "nrmse": finite_metric(nrmse),
        "cosine": finite_metric(cosine),
        "candidate_rms": candidate_rms,
        "reference_rms": reference_rms,
    }))
}

fn compare_tensors(candidate: &Tensor, reference: &Tensor, name: &str) -> Result<Value, String> {
    if candidate.shape().dims() != reference.shape().dims() {
        return Err(format!(
            "{name} shape mismatch: {:?} != {:?}",
            candidate.shape().dims(),
            reference.shape().dims()
        ));
    }
    metrics(
        candidate
            .as_f32()
            .map_err(|error| format!("read candidate {name}: {error}"))?,
        reference
            .as_f32()
            .map_err(|error| format!("read reference {name}: {error}"))?,
    )
}

fn reference_tensor<'a>(
    tensors: &'a HashMap<String, Tensor>,
    name: &str,
) -> Result<&'a Tensor, String> {
    tensors
        .get(name)
        .ok_or_else(|| format!("reference tensor {name:?} is missing"))
}

fn last_row<'a>(tensor: &'a Tensor, vocab_size: usize, name: &str) -> Result<&'a [f32], String> {
    let shape = tensor.shape().dims();
    if shape.len() != 2 || shape[0] == 0 || shape[1] != vocab_size {
        return Err(format!(
            "{name} must have shape [nonzero sequence, {vocab_size}], got {shape:?}"
        ));
    }
    let data = tensor
        .as_f32()
        .map_err(|error| format!("read {name}: {error}"))?;
    Ok(&data[(shape[0] - 1) * vocab_size..shape[0] * vocab_size])
}

fn top_indices(logits: &[f32], top_k: usize) -> Result<Vec<usize>, String> {
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("cannot rank non-finite logits".into());
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(top_k.min(indices.len()));
    Ok(indices)
}

fn ranked_values(logits: &[f32], indices: &[usize]) -> Vec<Value> {
    indices
        .iter()
        .enumerate()
        .map(|(rank, &token)| {
            json!({
                "rank": rank + 1,
                "token_id": token,
                "logit": logits[token],
            })
        })
        .collect()
}

fn top_k_comparison(candidate: &[f32], reference: &[f32], top_k: usize) -> Result<Value, String> {
    if candidate.len() != reference.len() {
        return Err("top-k candidate/reference vocabulary sizes disagree".into());
    }
    let candidate_indices = top_indices(candidate, top_k)?;
    let reference_indices = top_indices(reference, top_k)?;
    let candidate_set: HashSet<usize> = candidate_indices.iter().copied().collect();
    let overlap = reference_indices
        .iter()
        .filter(|token| candidate_set.contains(token))
        .count();
    let reference_margin = if reference_indices.len() > 1 {
        Some(reference[reference_indices[0]] - reference[reference_indices[1]])
    } else {
        None
    };
    Ok(json!({
        "k": candidate_indices.len(),
        "top1_equal": candidate_indices.first() == reference_indices.first(),
        "overlap": overlap,
        "reference_top1_margin": reference_margin,
        "apxinf": ranked_values(candidate, &candidate_indices),
        "reference": ranked_values(reference, &reference_indices),
    }))
}

fn compare_linear_states(
    model: &GeneralQwen35,
    prefix: &str,
    references: &HashMap<String, Tensor>,
) -> Result<Vec<Value>, String> {
    let mut output = Vec::new();
    for layer_index in 0..model.config_ref().text.n_layers {
        let Some(state) = model.state_ref().linear_state(layer_index) else {
            continue;
        };
        let suffixes = state.convolution_suffixes();
        let mut state_metrics = serde_json::Map::new();
        for (suffix, kind) in suffixes.into_iter().zip(["q_conv", "k_conv", "v_conv"]) {
            let candidate = suffix.ok_or_else(|| {
                format!("ApxInf linear layer {layer_index} did not initialize {kind}")
            })?;
            let reference_name = format!("{prefix}.linear.{layer_index:02}.{kind}");
            let reference = reference_tensor(references, &reference_name)?;
            state_metrics.insert(
                kind.to_string(),
                compare_tensors(candidate, reference, &reference_name)?,
            );
        }
        let candidate = state.recurrent().ok_or_else(|| {
            format!("ApxInf linear layer {layer_index} did not initialize recurrent state")
        })?;
        let reference_name = format!("{prefix}.linear.{layer_index:02}.recurrent");
        let reference = reference_tensor(references, &reference_name)?;
        state_metrics.insert(
            "recurrent".to_string(),
            compare_tensors(candidate, reference, &reference_name)?,
        );
        output.push(json!({
            "layer": layer_index,
            "metrics": state_metrics,
        }));
    }
    Ok(output)
}

fn observe_linear_state_metrics(
    snapshots: &[Value],
    convolution: &mut MetricEnvelope,
    recurrent: &mut MetricEnvelope,
) -> Result<(), String> {
    for (snapshot_index, snapshot) in snapshots.iter().enumerate() {
        let layer = snapshot
            .get("layer")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("linear-state snapshot {snapshot_index} has no layer"))?;
        let layer_metrics = snapshot
            .get("metrics")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("linear-state snapshot for layer {layer} has no metrics"))?;
        for kind in ["q_conv", "k_conv", "v_conv"] {
            let metric = layer_metrics
                .get(kind)
                .ok_or_else(|| format!("linear-state layer {layer} is missing {kind}"))?;
            convolution.observe(metric, &format!("linear.{layer}.{kind}"))?;
        }
        let metric = layer_metrics
            .get("recurrent")
            .ok_or_else(|| format!("linear-state layer {layer} is missing recurrent"))?;
        recurrent.observe(metric, &format!("linear.{layer}.recurrent"))?;
    }
    Ok(())
}

fn compare_logits(
    candidate: &Tensor,
    references: &HashMap<String, Tensor>,
    reference_name: &str,
) -> Result<Value, String> {
    let reference = reference_tensor(references, reference_name)?;
    compare_tensors(candidate, reference, reference_name)
}

fn validate_output_paths(args: &Args) -> Result<(), String> {
    let Some(output) = args.output.as_ref() else {
        return Ok(());
    };
    let output = output
        .canonicalize()
        .unwrap_or_else(|_| output.to_path_buf());
    for (label, input) in [
        ("reference", &args.reference),
        ("manifest", &args.manifest),
        ("model directory", &args.model_dir),
    ] {
        let input = input.canonicalize().unwrap_or_else(|_| input.to_path_buf());
        if output == input {
            return Err(format!("--output must not overwrite the {label}"));
        }
    }
    if output.exists() && !args.force {
        return Err(format!(
            "refusing to replace {}; pass --force to overwrite it",
            output.display()
        ));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    validate_output_paths(&args)?;
    let manifest_raw = fs::read_to_string(&args.manifest)
        .map_err(|error| format!("read {}: {error}", args.manifest.display()))?;
    let manifest: Value = serde_json::from_str(&manifest_raw)
        .map_err(|error| format!("parse {}: {error}", args.manifest.display()))?;
    let (input_ids, probe_token_id, greedy_token_ids) = verify_manifest(&manifest)?;
    let required_context = input_ids.len() + greedy_token_ids.len();
    if required_context > args.max_context {
        return Err(format!(
            "prompt plus frozen greedy trajectory needs context {required_context}, but --max-context is {}",
            args.max_context
        ));
    }

    let (references, reference_metadata) =
        apxinf_loader::safetensors::load_native_path(&args.reference)
            .map_err(|error| format!("load {}: {error}", args.reference.display()))?;
    verify_reference_metadata(
        &reference_metadata,
        &input_ids,
        probe_token_id,
        &greedy_token_ids,
    )?;

    let config_path = args.model_dir.join("config.json");
    let config = Qwen35Config::from_json_file(&config_path)
        .map_err(|error| format!("load {}: {error}", config_path.display()))?;
    if args.top_k > config.text.vocab_size {
        return Err(format!(
            "--top-k {} exceeds vocabulary size {}",
            args.top_k, config.text.vocab_size
        ));
    }
    if let Some(token) = input_ids
        .iter()
        .chain(std::iter::once(&probe_token_id))
        .chain(greedy_token_ids.iter())
        .find(|token| **token as usize >= config.text.vocab_size)
    {
        return Err(format!(
            "token id {token} exceeds vocabulary size {}",
            config.text.vocab_size
        ));
    }
    let vocab_size = config.text.vocab_size;

    let (weights, _) =
        apxinf_loader::safetensors::load_native_path_filtered(&args.model_dir, |name| {
            name.starts_with("model.language_model.") || name == "lm_head.weight"
        })
        .map_err(|error| format!("load {}: {error}", args.model_dir.display()))?;
    #[cfg(feature = "metal-w8")]
    let mut model = match args.metal_w8_body_layer {
        Some(layer_index) => GeneralQwen35::from_weights_with_metal_w8_body_layer(
            config,
            weights,
            Device::Cpu,
            args.max_context,
            layer_index,
        ),
        None => GeneralQwen35::from_weights(config, weights, Device::Cpu, args.max_context),
    }
    .map_err(|error| format!("construct Qwen3.5: {error}"))?;
    #[cfg(not(feature = "metal-w8"))]
    let mut model = {
        if args.metal_w8_body_layer.is_some() {
            return Err("--metal-w8-body-layer requires the `metal-w8` build feature".into());
        }
        GeneralQwen35::from_weights(config, weights, Device::Cpu, args.max_context)
            .map_err(|error| format!("construct Qwen3.5: {error}"))?
    };

    let prefill = model
        .forward(&input_ids, 0)
        .map_err(|error| format!("ApxInf prefill: {error}"))?;
    if model.state_ref().position() != input_ids.len() {
        return Err(format!(
            "ApxInf position after prefill is {}, expected {}",
            model.state_ref().position(),
            input_ids.len()
        ));
    }
    let prefill_state_metrics = compare_linear_states(&model, "prefill", &references)?;

    let cached_probe = model
        .forward(&[probe_token_id], input_ids.len() as u32)
        .map_err(|error| format!("ApxInf cached probe: {error}"))?;
    if model.state_ref().position() != input_ids.len() + 1 {
        return Err(format!(
            "ApxInf position after cached probe is {}, expected {}",
            model.state_ref().position(),
            input_ids.len() + 1
        ));
    }
    let cached_state_metrics = compare_linear_states(&model, "cached_probe", &references)?;

    model.reset();
    if model.state_ref().position() != 0 {
        return Err("ApxInf reset did not return the hybrid state to position zero".into());
    }
    let mut fresh_ids = input_ids.clone();
    fresh_ids.push(probe_token_id);
    let fresh_probe = model
        .forward(&fresh_ids, 0)
        .map_err(|error| format!("ApxInf fresh probe: {error}"))?;

    model.reset();
    let (generated_trajectory, _) = model
        .generate_streaming(
            LlmInput::text(&input_ids),
            greedy_token_ids.len(),
            |_| {},
            None,
        )
        .map_err(|error| format!("ApxInf greedy trajectory: {error}"))?;

    let prefill_reference = reference_tensor(&references, "prefill.logits")?;
    let cached_reference = reference_tensor(&references, "cached_probe.logits")?;
    let prefill_candidate_last = last_row(&prefill, vocab_size, "ApxInf prefill logits")?;
    let prefill_reference_last =
        last_row(prefill_reference, vocab_size, "Transformers prefill logits")?;
    let cached_candidate_last = last_row(&cached_probe, vocab_size, "ApxInf cached-probe logits")?;
    let cached_reference_last = last_row(
        cached_reference,
        vocab_size,
        "Transformers cached-probe logits",
    )?;
    let fresh_candidate_last = last_row(&fresh_probe, vocab_size, "ApxInf fresh-probe logits")?;

    let prefill_logits_metrics = compare_logits(&prefill, &references, "prefill.logits")?;
    let cached_logits_metrics = compare_logits(&cached_probe, &references, "cached_probe.logits")?;
    let fresh_logits_metrics = compare_logits(&fresh_probe, &references, "fresh_probe.logits")?;
    let cached_vs_fresh_metrics = metrics(cached_candidate_last, fresh_candidate_last)?;
    let first_step_top_k =
        top_k_comparison(prefill_candidate_last, prefill_reference_last, args.top_k)?;
    let cached_probe_top_k =
        top_k_comparison(cached_candidate_last, cached_reference_last, args.top_k)?;
    // The report may request more ranks, but the gate always evaluates the
    // same frozen top-20 contract.
    let gate_first_step_top_k =
        top_k_comparison(prefill_candidate_last, prefill_reference_last, GATE_TOP_K)?;
    let gate_cached_probe_top_k =
        top_k_comparison(cached_candidate_last, cached_reference_last, GATE_TOP_K)?;

    let mut transformer_logits = MetricEnvelope::empty();
    transformer_logits.observe(&prefill_logits_metrics, "prefill.logits")?;
    transformer_logits.observe(&cached_logits_metrics, "cached_probe.logits")?;
    transformer_logits.observe(&fresh_logits_metrics, "fresh_probe.logits")?;
    let mut cached_vs_fresh = MetricEnvelope::empty();
    cached_vs_fresh.observe(&cached_vs_fresh_metrics, "cached_vs_fresh")?;
    let mut convolution_state = MetricEnvelope::empty();
    let mut recurrent_state = MetricEnvelope::empty();
    observe_linear_state_metrics(
        &prefill_state_metrics,
        &mut convolution_state,
        &mut recurrent_state,
    )?;
    observe_linear_state_metrics(
        &cached_state_metrics,
        &mut convolution_state,
        &mut recurrent_state,
    )?;

    let mut checks = Vec::new();
    let mut failures = Vec::new();
    gate_metric_envelope(
        "transformers_logits",
        &transformer_logits,
        3,
        LOGITS_LIMITS,
        &mut checks,
        &mut failures,
    );
    gate_metric_envelope(
        "convolution_state",
        &convolution_state,
        EXPECTED_CONV_STATE_TENSORS,
        CONV_STATE_LIMITS,
        &mut checks,
        &mut failures,
    );
    gate_metric_envelope(
        "recurrent_state",
        &recurrent_state,
        EXPECTED_RECURRENT_STATE_TENSORS,
        RECURRENT_STATE_LIMITS,
        &mut checks,
        &mut failures,
    );
    gate_metric_envelope(
        "cached_vs_fresh",
        &cached_vs_fresh,
        1,
        CACHED_VS_FRESH_LIMITS,
        &mut checks,
        &mut failures,
    );
    gate_top_k(
        "first_step_top_k",
        &gate_first_step_top_k,
        &mut checks,
        &mut failures,
    )?;
    gate_top_k(
        "cached_probe_top_k",
        &gate_cached_probe_top_k,
        &mut checks,
        &mut failures,
    )?;
    gate_greedy_trajectory(
        &generated_trajectory,
        &greedy_token_ids,
        &mut checks,
        &mut failures,
    );
    let gate_passed = failures.is_empty();
    let gate_failure_summary = failures.join("; ");

    #[cfg(feature = "metal-w8")]
    let metal_w8_body_receipt = model.metal_w8_body_stats().map(|stats| {
        json!({
            "layer_index": stats.layer_index,
            "decode_calls": stats.decode_calls,
            "projection_elapsed_ns": stats.projection_elapsed_ns,
        })
    });
    #[cfg(not(feature = "metal-w8"))]
    let metal_w8_body_receipt: Option<Value> = None;

    let result = json!({
        "format": "apxinf-qwen35-oracle-comparison-v1",
        "oracle": {
            "format": ORACLE_FORMAT,
            "repo_id": REPO_ID,
            "revision": LOCKED_REVISION,
            "checkpoint_sha256": LOCKED_CHECKPOINT_SHA256,
            "manifest": args.manifest,
            "reference": args.reference,
        },
        "apxinf": {
            "device": "cpu",
            "matmul_feature": if cfg!(feature = "accelerate") { "accelerate" } else { "portable" },
            "max_context": args.max_context,
            "state_position": model.state_ref().position(),
            "full_attention_layers": model.state_ref().full_attention_layers(),
        },
        "input_ids": input_ids,
        "probe_token_id": probe_token_id,
        "metal_w8_body": metal_w8_body_receipt,
        "greedy_trajectory": {
            "length": greedy_token_ids.len(),
            "minimum_length": GATE_GREEDY_MIN_TOKENS,
            "expected_ids": greedy_token_ids,
            "apxinf_ids": generated_trajectory,
            "exact_match": generated_trajectory == greedy_token_ids,
            "shared_generate_streaming": true,
            "eos_stopping": false,
        },
        "logits": {
            "prefill_all_rows": prefill_logits_metrics,
            "cached_probe": cached_logits_metrics,
            "fresh_probe_all_rows": fresh_logits_metrics,
            "apxinf_cached_probe_vs_fresh_last": cached_vs_fresh_metrics,
        },
        "top_k": {
            "first_step": first_step_top_k,
            "cached_probe": cached_probe_top_k,
        },
        "linear_state": {
            "prefill": prefill_state_metrics,
            "cached_probe": cached_state_metrics,
        },
        "verification": {
            "manifest": threshold_manifest(),
            "passed": gate_passed,
            "status": if gate_passed { "pass" } else { "fail" },
            "checks": checks,
            "failures": failures,
        },
        "notes": {
            "tokenizer_used_by_apxinf": false,
            "full_kv_compared_indirectly": true,
            "thresholds_applied": true,
            "greedy_trajectory_exact_gate": true,
        },
    });

    let rendered = serde_json::to_string_pretty(&result)
        .map_err(|error| format!("serialize metrics: {error}"))?
        + "\n";
    if let Some(output) = args.output.as_ref() {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        fs::write(output, &rendered)
            .map_err(|error| format!("write {}: {error}", output.display()))?;
    }
    print!("{rendered}");
    if !gate_passed {
        return Err(format!("oracle gate failed: {gate_failure_summary}"));
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("qwen35_oracle: {error}");
        std::process::exit(1);
    }
}

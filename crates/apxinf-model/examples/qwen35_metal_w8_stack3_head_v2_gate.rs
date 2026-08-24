//! Same-binary four-mode gate for the explicit Qwen3.5 Stack3 + tied-head v2
//! diagnostic. This example is not reachable from CLI, AutoModel, registry,
//! or any default constructor.

#[path = "support/qwen35_gate_evidence.rs"]
mod gate_evidence;

use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use apxinf_core::{Device, Tensor};
use apxinf_model::qwen35::general::{
    Qwen35MetalW8LinearLayerStacksV1AggregateLedger, Qwen35MetalW8LinearLayerStacksV1Stats,
    Qwen35MetalW8Stack3LmHeadV2AggregateLedger,
};
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config, Qwen35LayerType};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};

const CPU_TEACHER_FORMAT: &str = "apxinf-qwen35-metal-w8-stack3-head-v2-cpu-teacher-v1";
const CANDIDATE_TEACHER_FORMAT: &str = "apxinf-qwen35-metal-w8-stack3-head-v2-teacher-gate-v1";
const CPU_FREE_FORMAT: &str = "apxinf-qwen35-metal-w8-stack3-head-v2-cpu-free-run-v1";
const CANDIDATE_FREE_FORMAT: &str = "apxinf-qwen35-metal-w8-stack3-head-v2-free-run-gate-v1";
const SOURCE_LOCK_FORMAT: &str = "apxinf-hf-source-lock-v1";
const REPO_ID: &str = "Qwen/Qwen3.5-0.8B";
const LOCKED_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
const LOCKED_CHECKPOINT: &str = "model.safetensors-00001-of-00001.safetensors";
const LOCKED_CHECKPOINT_SHA256: &str =
    "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696";
const LOCKED_CHECKPOINT_BYTES: u64 = 1_746_942_600;
const STEPS: usize = 128;
const PROMPT: &str = "Hello";
const GATE_SOURCE_NAME: &str = "qwen35_metal_w8_stack3_head_v2_gate.rs";
const GATE_SOURCE_BYTES: &[u8] = include_bytes!("qwen35_metal_w8_stack3_head_v2_gate.rs");
const ALL_LINEAR_LAYER_INDICES: [usize; 18] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
];
const FULL_ATTENTION_LAYER_INDICES: [usize; 6] = [3, 7, 11, 15, 19, 23];
const LINEAR_LAYER_STACK3_INDICES: [[usize; 3]; 6] = [
    [0, 1, 2],
    [4, 5, 6],
    [8, 9, 10],
    [12, 13, 14],
    [16, 17, 18],
    [20, 21, 22],
];
const OFFICIAL_HIDDEN_BYTES: usize = 4_096;
const OFFICIAL_STACK3_PERSISTENT_BYTES: usize = 76_351_488;
const OFFICIAL_MLP_BLOCK_PERSISTENT_BYTES: usize = 11_749_376;
const OFFICIAL_BODY_PERSISTENT_BYTES: usize = 528_605_184;
const OFFICIAL_HEAD_PERSISTENT_BYTES: usize = 271_169_552;
const OFFICIAL_COMPOSITE_PERSISTENT_BYTES: usize = 799_774_736;

struct RunResult {
    receipt: Value,
    passed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    CpuTeacher,
    Stack3HeadV2Teacher,
    CpuFree,
    Stack3HeadV2Free,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu-teacher" => Ok(Self::CpuTeacher),
            "stack3-head-v2-teacher" => Ok(Self::Stack3HeadV2Teacher),
            "cpu-free" => Ok(Self::CpuFree),
            "stack3-head-v2-free" => Ok(Self::Stack3HeadV2Free),
            other => Err(format!("invalid --mode {other:?}")),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::CpuTeacher => "cpu_teacher",
            Self::Stack3HeadV2Teacher => "metal_w8_stack3_head_v2_teacher_forced",
            Self::CpuFree => "cpu_free_run",
            Self::Stack3HeadV2Free => "metal_w8_stack3_head_v2_free_run",
        }
    }

    const fn receipt_format(self) -> &'static str {
        match self {
            Self::CpuTeacher => CPU_TEACHER_FORMAT,
            Self::Stack3HeadV2Teacher => CANDIDATE_TEACHER_FORMAT,
            Self::CpuFree => CPU_FREE_FORMAT,
            Self::Stack3HeadV2Free => CANDIDATE_FREE_FORMAT,
        }
    }

    const fn requires_input_receipt(self) -> bool {
        matches!(self, Self::Stack3HeadV2Teacher | Self::Stack3HeadV2Free)
    }

    const fn is_candidate(self) -> bool {
        self.requires_input_receipt()
    }
}

struct TeacherOracle {
    teacher_inputs: Vec<u32>,
    expected_outputs: Vec<u32>,
}

struct TeacherGateEvaluation {
    passed: bool,
    body_mismatches: Vec<Value>,
    head_mismatches: Vec<Value>,
    end_to_end_mismatches: Vec<Value>,
}

fn json_u32_array(value: &Value, field: &str) -> Result<Vec<u32>, String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{field} must be an array"))?
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|number| u32::try_from(number).ok())
                .ok_or_else(|| format!("{field} must contain only u32 values"))
        })
        .collect()
}

fn validate_cpu_teacher_receipt(
    receipt: &Value,
    expected_identity: &Value,
    prompt_tokens: &[u32],
    prefill_token: u32,
) -> Result<TeacherOracle, String> {
    if receipt.get("format").and_then(Value::as_str) != Some(CPU_TEACHER_FORMAT)
        || receipt.get("mode").and_then(Value::as_str) != Some(Mode::CpuTeacher.label())
        || receipt.get("passed").and_then(Value::as_bool) != Some(true)
        || receipt.get("identity") != Some(expected_identity)
        || receipt.get("comparisons").and_then(Value::as_u64) != Some(128)
        || receipt.get("prefill_token").and_then(Value::as_u64) != Some(prefill_token as u64)
        || json_u32_array(receipt, "prompt_token_ids")? != prompt_tokens
    {
        return Err("CPU teacher receipt does not match this frozen request".into());
    }
    let teacher_inputs = json_u32_array(receipt, "teacher_input_ids")?;
    let expected_outputs = json_u32_array(receipt, "cpu_expected_output_ids")?;
    if teacher_inputs.len() != 128
        || expected_outputs.len() != 128
        || teacher_inputs.first().copied() != Some(prefill_token)
        || teacher_inputs[1..] != expected_outputs[..127]
    {
        return Err("CPU teacher receipt does not contain the exact 128-step teacher chain".into());
    }
    Ok(TeacherOracle {
        teacher_inputs,
        expected_outputs,
    })
}

fn evaluate_teacher_candidate(
    oracle: &TeacherOracle,
    candidate_hidden_cpu_tokens: &[u32],
    w8_top4_candidates: &[[u32; 4]],
    reranked_tokens: &[u32],
) -> Result<TeacherGateEvaluation, String> {
    if oracle.teacher_inputs.len() != 128
        || oracle.expected_outputs.len() != 128
        || candidate_hidden_cpu_tokens.len() != 128
        || w8_top4_candidates.len() != 128
        || reranked_tokens.len() != 128
    {
        return Err("teacher candidate evidence must contain exactly 128 steps".into());
    }
    let mut body_mismatches = Vec::new();
    let mut head_mismatches = Vec::new();
    let mut end_to_end_mismatches = Vec::new();
    for step in 0..128 {
        let expected = oracle.expected_outputs[step];
        let candidate_cpu = candidate_hidden_cpu_tokens[step];
        let candidates = w8_top4_candidates[step];
        let reranked = reranked_tokens[step];
        if candidate_cpu != expected {
            body_mismatches.push(json!({
                "step": step,
                "teacher_input": oracle.teacher_inputs[step],
                "frozen_cpu_expected": expected,
                "candidate_hidden_cpu_f32": candidate_cpu,
            }));
        }
        if !candidates.contains(&candidate_cpu) || reranked != candidate_cpu {
            head_mismatches.push(json!({
                "step": step,
                "candidate_hidden_cpu_f32": candidate_cpu,
                "metal_w8_top4": candidates,
                "f32_reranked": reranked,
                "candidate_hidden_winner_in_top4": candidates.contains(&candidate_cpu),
            }));
        }
        if reranked != expected {
            end_to_end_mismatches.push(json!({
                "step": step,
                "frozen_cpu_expected": expected,
                "f32_reranked": reranked,
            }));
        }
    }
    Ok(TeacherGateEvaluation {
        passed: body_mismatches.is_empty()
            && head_mismatches.is_empty()
            && end_to_end_mismatches.is_empty(),
        body_mismatches,
        head_mismatches,
        end_to_end_mismatches,
    })
}

fn validate_cpu_free_receipt(
    receipt: &Value,
    expected_identity: &Value,
    prompt_tokens: &[u32],
) -> Result<Vec<u32>, String> {
    if receipt.get("format").and_then(Value::as_str) != Some(CPU_FREE_FORMAT)
        || receipt.get("mode").and_then(Value::as_str) != Some(Mode::CpuFree.label())
        || receipt.get("passed").and_then(Value::as_bool) != Some(true)
        || receipt.get("identity") != Some(expected_identity)
        || receipt.get("max_new_tokens").and_then(Value::as_u64) != Some(128)
        || receipt.get("eos_stopping").and_then(Value::as_bool) != Some(false)
        || json_u32_array(receipt, "prompt_token_ids")? != prompt_tokens
    {
        return Err("CPU free receipt does not match this frozen request".into());
    }
    let tokens = json_u32_array(receipt, "generated_token_ids")?;
    if tokens.len() != 128 {
        return Err("CPU free receipt must contain exactly 128 generated tokens".into());
    }
    Ok(tokens)
}

struct Args {
    model_dir: PathBuf,
    source_lock: PathBuf,
    mode: Mode,
    input_receipt: Option<PathBuf>,
    output: PathBuf,
}

fn usage() -> &'static str {
    "Usage: qwen35_metal_w8_stack3_head_v2_gate \
  --model-dir OFFICIAL_LOCAL_QWEN35_0_8B \
  --source-lock SOURCE_LOCK.json \
  --mode cpu-teacher|stack3-head-v2-teacher|cpu-free|stack3-head-v2-free \
  [--input-receipt CPU_RECEIPT.json] \
  --output NEW_RECEIPT.json"
}

fn parse_args_from<I, S>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut model_dir = None;
    let mut source_lock = None;
    let mut mode = None;
    let mut input_receipt = None;
    let mut output = None;
    let mut iter = args.into_iter().map(Into::into).skip(1);
    while let Some(raw_flag) = iter.next() {
        let flag = raw_flag.to_string_lossy();
        let value = iter
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_ref() {
            "--model-dir" => {
                if model_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("duplicate --model-dir".into());
                }
            }
            "--source-lock" => {
                if source_lock.replace(PathBuf::from(value)).is_some() {
                    return Err("duplicate --source-lock".into());
                }
            }
            "--mode" => {
                let parsed = Mode::parse(&value.to_string_lossy())?;
                if mode.replace(parsed).is_some() {
                    return Err("duplicate --mode".into());
                }
            }
            "--input-receipt" => {
                if input_receipt.replace(PathBuf::from(value)).is_some() {
                    return Err("duplicate --input-receipt".into());
                }
            }
            "--output" => {
                if output.replace(PathBuf::from(value)).is_some() {
                    return Err("duplicate --output".into());
                }
            }
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }
    let args = Args {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required\n{}", usage()))?,
        source_lock: source_lock
            .ok_or_else(|| format!("--source-lock is required\n{}", usage()))?,
        mode: mode.ok_or_else(|| format!("--mode is required\n{}", usage()))?,
        input_receipt,
        output: output.ok_or_else(|| format!("--output is required\n{}", usage()))?,
    };
    if !args.model_dir.is_absolute()
        || !args.source_lock.is_absolute()
        || !args.output.is_absolute()
        || args
            .input_receipt
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
    {
        return Err("all gate paths must be absolute".into());
    }
    if args.output.exists() {
        return Err("--output must not already exist".into());
    }
    if args.mode.requires_input_receipt() != args.input_receipt.is_some() {
        return Err(if args.mode.requires_input_receipt() {
            "candidate modes require --input-receipt".into()
        } else {
            "CPU modes reject --input-receipt".into()
        });
    }
    Ok(args)
}

fn publish_receipt_create_new(path: &Path, receipt: &Value) -> Result<(), Box<dyn Error>> {
    let mut bytes = serde_json::to_vec(receipt)?;
    bytes.push(b'\n');
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&bytes)?;
    output.sync_all()?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeadCallCounts {
    prefill: usize,
    decode: usize,
    teacher: usize,
}

impl HeadCallCounts {
    const ZERO: Self = Self {
        prefill: 0,
        decode: 0,
        teacher: 0,
    };

    const fn teacher(calls: usize) -> Self {
        Self {
            prefill: 0,
            decode: 0,
            teacher: calls,
        }
    }

    const fn free(decode: usize) -> Self {
        Self {
            prefill: 1,
            decode,
            teacher: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct V2PathChecks {
    schedule_valid: bool,
    stack3_mechanism_valid: bool,
    stack3_execution_valid: bool,
    full_attention_mlp_valid: bool,
    head_execution_valid: bool,
    aggregate_ledger_valid: bool,
    generation_receipt_valid: bool,
    terminal_clear: bool,
}

impl V2PathChecks {
    fn all_valid(self) -> bool {
        self.schedule_valid
            && self.stack3_mechanism_valid
            && self.stack3_execution_valid
            && self.full_attention_mlp_valid
            && self.head_execution_valid
            && self.aggregate_ledger_valid
            && self.generation_receipt_valid
            && self.terminal_clear
    }

    fn receipt_json(self) -> Value {
        json!({
            "schedule_valid": self.schedule_valid,
            "stack3_mechanism_valid": self.stack3_mechanism_valid,
            "stack3_execution_valid": self.stack3_execution_valid,
            "full_attention_mlp_valid": self.full_attention_mlp_valid,
            "head_execution_valid": self.head_execution_valid,
            "aggregate_ledger_valid": self.aggregate_ledger_valid,
            "generation_receipt_valid": self.generation_receipt_valid,
            "terminal_clear": self.terminal_clear,
            "all_valid": self.all_valid(),
        })
    }
}

fn validate_official_schedule(layer_types: &[Qwen35LayerType]) -> Result<(), String> {
    if layer_types.len() != 24 {
        return Err(format!(
            "Stack3 + lm_head v2 gate requires exactly 24 layers, got {}",
            layer_types.len()
        ));
    }
    let linear = layer_types
        .iter()
        .enumerate()
        .filter_map(|(index, layer_type)| {
            (*layer_type == Qwen35LayerType::LinearAttention).then_some(index)
        })
        .collect::<Vec<_>>();
    let full = layer_types
        .iter()
        .enumerate()
        .filter_map(|(index, layer_type)| {
            (*layer_type == Qwen35LayerType::FullAttention).then_some(index)
        })
        .collect::<Vec<_>>();
    if linear != ALL_LINEAR_LAYER_INDICES || full != FULL_ATTENTION_LAYER_INDICES {
        return Err(format!(
            "Stack3 + lm_head v2 gate requires linear={ALL_LINEAR_LAYER_INDICES:?} and full={FULL_ATTENTION_LAYER_INDICES:?}, got linear={linear:?}, full={full:?}"
        ));
    }
    Ok(())
}

fn validate_source_lock(lock: &Value) -> Result<(), String> {
    if lock.get("format").and_then(Value::as_str) != Some(SOURCE_LOCK_FORMAT)
        || lock.get("repo_id").and_then(Value::as_str) != Some(REPO_ID)
        || lock.get("resolved_commit").and_then(Value::as_str) != Some(LOCKED_REVISION)
    {
        return Err("source lock does not identify the frozen official Qwen3.5-0.8B source".into());
    }
    let files = lock
        .pointer("/weights/files")
        .and_then(Value::as_array)
        .ok_or("source lock weights.files must be an array")?;
    if files.len() != 1
        || files[0].get("path").and_then(Value::as_str) != Some(LOCKED_CHECKPOINT)
        || files[0].get("sha256").and_then(Value::as_str) != Some(LOCKED_CHECKPOINT_SHA256)
        || files[0].get("size").and_then(Value::as_u64) != Some(LOCKED_CHECKPOINT_BYTES)
    {
        return Err("source lock does not bind the frozen Qwen3.5-0.8B checkpoint".into());
    }
    Ok(())
}

fn argmax(logits: &Tensor, vocab_size: usize) -> Result<u32, Box<dyn Error>> {
    if logits.shape().dims() != [1, vocab_size] {
        return Err(format!("expected logits [1, {vocab_size}], got {}", logits.shape()).into());
    }
    let mut best_score = f32::NEG_INFINITY;
    let mut best_token = 0u32;
    for (token, &score) in logits.as_f32()?.iter().enumerate() {
        if score > best_score {
            best_score = score;
            best_token = u32::try_from(token)?;
        }
    }
    Ok(best_token)
}

fn official_head_ledger_is_exact(ledger: apxinf_metal::LmHeadBufferLedger) -> bool {
    ledger.scope == "resident-mtlbuffer-only"
        && ledger.exclusions
            == "host F32 tied embedding and four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, model body, and KV cache"
        && ledger.allocated_buffers == 5
        && ledger.shared_buffers == 4
        && ledger.private_buffers == 1
        && ledger.packed_weight_bytes == 254_279_680
        && ledger.packed_scale_bytes == 15_892_480
        && ledger.hidden_bytes == OFFICIAL_HIDDEN_BYTES
        && ledger.partial_topk_bytes == 993_280
        && ledger.output_token_bytes == 16
        && ledger.total_persistent_bytes == OFFICIAL_HEAD_PERSISTENT_BYTES
        && ledger.host_input_bytes_per_call == OFFICIAL_HIDDEN_BYTES
        && ledger.host_output_bytes_per_call == 16
        && ledger.state_host_transfer_bytes_per_call == 0
        && ledger.command_buffers_per_call == 1
        && ledger.compute_encoders_per_call == 2
        && ledger.commits_per_call == 1
        && ledger.waits_per_call == 1
}

fn official_stack3_ledger_is_exact(ledger: apxinf_metal::LinearLayerStack3BufferLedger) -> bool {
    ledger.allocated_buffers == 76
        && ledger.shared_buffers == 68
        && ledger.private_buffers == 8
        && ledger.packed_weight_bytes == 64_585_728
        && ledger.packed_scale_bytes == 4_429_824
        && ledger.f32_parameter_bytes == 321_408
        && ledger.active_state_bytes == 3_440_640
        && ledger.scratch_state_bytes == 3_440_640
        && ledger.activation_bytes == 133_248
        && ledger.total_persistent_bytes == OFFICIAL_STACK3_PERSISTENT_BYTES
        && ledger.host_input_bytes_per_decode == OFFICIAL_HIDDEN_BYTES
        && ledger.host_output_bytes_per_decode == OFFICIAL_HIDDEN_BYTES
        && ledger.state_host_transfer_bytes_per_decode == 0
        && ledger.command_buffers_per_decode == 1
        && ledger.compute_encoders_per_decode == 3
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
        && ledger.intermediate_host_finite_checks_per_decode == 0
        && ledger.final_output_finite_checks_per_decode == 1
}

fn official_mlp_ledger_is_exact(ledger: apxinf_metal::MlpBlockBufferLedger) -> bool {
    ledger.scope == "resident-mtlbuffer-only"
        && ledger.allocated_buffers == 8
        && ledger.shared_buffers == 6
        && ledger.private_buffers == 2
        && ledger.packed_weight_bytes == 11_010_048
        && ledger.packed_scale_bytes == 688_128
        && ledger.activation_bytes == 51_200
        && ledger.total_persistent_bytes == OFFICIAL_MLP_BLOCK_PERSISTENT_BYTES
        && ledger.host_input_bytes_per_decode == OFFICIAL_HIDDEN_BYTES
        && ledger.host_output_bytes_per_decode == OFFICIAL_HIDDEN_BYTES
        && ledger.state_host_transfer_bytes_per_decode == 0
        && ledger.command_buffers_per_decode == 1
        && ledger.compute_encoders_per_decode == 3
        && ledger.commits_per_decode == 1
        && ledger.waits_per_decode == 1
}

fn official_aggregate_ledger_is_exact(
    aggregate: &Qwen35MetalW8Stack3LmHeadV2AggregateLedger,
) -> bool {
    let body = &aggregate.body;
    let body_stacks = body
        .stacks
        .iter()
        .map(|entry| entry.layer_indices)
        .collect::<Vec<_>>();
    let body_full = body
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    aggregate.scope == "resident-mtlbuffer-only"
        && aggregate.exclusions
            == "host F32 tied embedding and exact four-candidate F32 rerank, other CPU F32 weights, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, and KV cache"
        && aggregate.includes_lm_head
        && body.scope == "resident-mtlbuffer-only"
        && body.exclusions
            == "CPU F32 weights, host Vec allocations, Metal pipelines/libraries/queues, driver allocations, KV cache, and lm_head"
        && !body.includes_lm_head
        && body_stacks == LINEAR_LAYER_STACK3_INDICES
        && body_full == FULL_ATTENTION_LAYER_INDICES
        && body.stacks.iter().all(|entry| official_stack3_ledger_is_exact(entry.ledger))
        && body
            .full_attention_mlp_layers
            .iter()
            .all(|entry| official_mlp_ledger_is_exact(entry.ledger))
        && body.total_persistent_mtlbuffer_bytes == OFFICIAL_BODY_PERSISTENT_BYTES
        && body.allocated_buffers == 504
        && body.shared_buffers == 444
        && body.private_buffers == 60
        && body.host_to_device_bytes_per_decode == 49_152
        && body.device_to_host_bytes_per_decode == 49_152
        && body.state_host_transfer_bytes_per_decode == 0
        && body.command_buffers_per_decode == 12
        && body.compute_encoders_per_decode == 36
        && body.commits_per_decode == 12
        && body.waits_per_decode == 12
        && body.intermediate_host_finite_checks_per_decode == 0
        && body.final_output_finite_checks_per_decode == 6
        && official_head_ledger_is_exact(aggregate.lm_head)
        && aggregate.total_persistent_mtlbuffer_bytes == OFFICIAL_COMPOSITE_PERSISTENT_BYTES
        && aggregate.allocated_buffers == 509
        && aggregate.shared_buffers == 448
        && aggregate.private_buffers == 61
        && aggregate.host_to_device_bytes_per_call == 53_248
        && aggregate.device_to_host_bytes_per_call == 49_168
        && aggregate.state_host_transfer_bytes_per_call == 0
        && aggregate.command_buffers_per_call == 13
        && aggregate.compute_encoders_per_call == 38
        && aggregate.commits_per_call == 13
        && aggregate.waits_per_call == 13
        && aggregate.intermediate_host_finite_checks_per_call == 0
        && aggregate.final_output_finite_checks_per_call == 6
}

fn stack3_head_generation_receipt_is_exact(
    receipt: &Value,
    body_calls: usize,
    head_calls: HeadCallCounts,
) -> bool {
    if receipt.get("format").and_then(Value::as_str)
        != Some("apxinf-qwen35-stack3-lm-head-generation-path-v2")
        || receipt.get("mechanism").and_then(Value::as_str) != Some("metal-w8-stack3-lm-head-v2")
        || receipt.get("stack3_mechanism").and_then(Value::as_str)
            != Some("metal-w8-linear-layer-stack3-v1")
        || receipt
            .get("full_attention_mlp_mechanism")
            .and_then(Value::as_str)
            != Some("metal-w8-mlp-block-g64")
        || receipt
            .get("metal_w8_complete_linear_layer_stacks")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("metal_w8_full_attention_mlp_blocks")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("metal_w8_tied_lm_head_topk4_f32_rerank")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("intermediate_host_finite_checks")
            .and_then(Value::as_bool)
            != Some(false)
        || receipt
            .get("final_output_finite_checks")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt.get("terminal_error").and_then(Value::as_bool) != Some(false)
    {
        return false;
    }
    let Some(stacks) = receipt.get("stacks").and_then(Value::as_array) else {
        return false;
    };
    let Some(full) = receipt
        .get("full_attention_mlp_layers")
        .and_then(Value::as_array)
    else {
        return false;
    };
    let Some(head) = receipt.get("lm_head") else {
        return false;
    };
    let expected = body_calls as u64;
    let expected_encoders = body_calls.checked_mul(3).map(|value| value as u64);
    let expected_transfer = body_calls
        .checked_mul(OFFICIAL_HIDDEN_BYTES)
        .map(|value| value as u64);
    let expected_mask = if body_calls == 0 { 0 } else { 0b111 };
    let stacks_valid = expected_encoders.is_some()
        && expected_transfer.is_some()
        && stacks.len() == 6
        && stacks
            .iter()
            .zip(LINEAR_LAYER_STACK3_INDICES)
            .all(|(entry, layer_indices)| {
                entry.get("layer_indices") == Some(&json!(layer_indices))
                    && entry.get("mechanism").and_then(Value::as_str)
                        == Some("metal-w8-linear-layer-stack3-v1")
                    && entry.get("gdn_output_group_sizes") == Some(&json!([32, 32, 32]))
                    && entry.get("prefill_seed_calls") == Some(&json!([1, 1, 1]))
                    && entry.get("decode_calls").and_then(Value::as_u64) == Some(expected)
                    && entry.get("successful_decodes").and_then(Value::as_u64) == Some(expected)
                    && entry.get("failed_decodes").and_then(Value::as_u64) == Some(0)
                    && entry.get("command_buffers").and_then(Value::as_u64) == Some(expected)
                    && entry.get("compute_encoders").and_then(Value::as_u64) == expected_encoders
                    && entry.get("commits").and_then(Value::as_u64) == Some(expected)
                    && entry.get("waits").and_then(Value::as_u64) == Some(expected)
                    && entry.get("host_to_device_bytes").and_then(Value::as_u64)
                        == expected_transfer
                    && entry.get("device_to_host_bytes").and_then(Value::as_u64)
                        == expected_transfer
                    && entry.get("state_commits").and_then(Value::as_u64) == expected_encoders
                    && entry.get("last_state_commit_mask").and_then(Value::as_u64)
                        == Some(expected_mask)
                    && entry.get("committed_stack_version").and_then(Value::as_u64)
                        == Some(expected)
                    && entry
                        .get("intermediate_host_finite_checks_per_decode")
                        .and_then(Value::as_u64)
                        == Some(0)
                    && entry
                        .get("final_output_finite_checks_per_decode")
                        .and_then(Value::as_u64)
                        == Some(1)
                    && entry.get("terminal_error").and_then(Value::as_bool) == Some(false)
            });
    let full_valid = full.len() == 6
        && full
            .iter()
            .zip(FULL_ATTENTION_LAYER_INDICES)
            .all(|(entry, layer_index)| {
                entry.get("layer_index").and_then(Value::as_u64) == Some(layer_index as u64)
                    && entry.get("decode_calls").and_then(Value::as_u64) == Some(expected)
            });
    let head_valid = head.get("mechanism").and_then(Value::as_str)
        == Some("metal-w8-top4-f32-rerank")
        && head.get("prefill_calls").and_then(Value::as_u64) == Some(head_calls.prefill as u64)
        && head.get("decode_calls").and_then(Value::as_u64) == Some(head_calls.decode as u64)
        && head.get("teacher_calls").and_then(Value::as_u64) == Some(head_calls.teacher as u64);
    stacks_valid && full_valid && head_valid
}

fn v2_path_checks(
    stats: &Qwen35MetalW8LinearLayerStacksV1Stats,
    head: apxinf_model::qwen35::general::Qwen35MetalW8LmHeadStats,
    aggregate: &Qwen35MetalW8Stack3LmHeadV2AggregateLedger,
    generation_receipt: &Value,
    body_calls: usize,
    head_calls: HeadCallCounts,
) -> V2PathChecks {
    let stats_stacks = stats
        .stacks
        .iter()
        .map(|entry| entry.layer_indices)
        .collect::<Vec<_>>();
    let stats_full = stats
        .full_attention_mlp_layers
        .iter()
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    let expected_encoders = body_calls.checked_mul(3);
    let expected_transfer = body_calls.checked_mul(OFFICIAL_HIDDEN_BYTES);
    let expected_mask = if body_calls == 0 { 0 } else { 0b111 };
    let stack3_execution_valid = expected_encoders.is_some()
        && expected_transfer.is_some()
        && stats.stacks.iter().all(|entry| {
            let execution = entry.execution;
            entry.prefill_seed_calls == [1, 1, 1]
                && execution.decode_calls == body_calls
                && execution.successful_decodes == body_calls
                && execution.failed_decodes == 0
                && execution.command_buffers == body_calls
                && execution.compute_encoders == expected_encoders.unwrap()
                && execution.commits == body_calls
                && execution.waits == body_calls
                && execution.host_to_device_bytes == expected_transfer.unwrap()
                && execution.device_to_host_bytes == expected_transfer.unwrap()
                && execution.state_commits == expected_encoders.unwrap()
                && execution.last_state_commit_mask == expected_mask
                && execution.committed_stack_version == body_calls as u64
                && !execution.terminal_error
                && !entry.terminal_error
        });
    V2PathChecks {
        schedule_valid: stats_stacks == LINEAR_LAYER_STACK3_INDICES
            && stats_full == FULL_ATTENTION_LAYER_INDICES,
        stack3_mechanism_valid: stats.mechanism == "metal-w8-linear-layer-stack3-v1"
            && stats.full_attention_mlp_mechanism == "metal-w8-mlp-block-g64"
            && stats.stacks.iter().all(|entry| {
                entry.mechanism == "metal-w8-linear-layer-stack3-v1"
                    && entry.quantization.iter().all(|quantization| {
                        quantization.gdn_input_group_size == apxinf_metal::W8GroupSize::G64
                            && quantization.gdn_output_group_size == apxinf_metal::W8GroupSize::G32
                            && quantization.mlp_gate_group_size == apxinf_metal::W8GroupSize::G64
                            && quantization.mlp_up_group_size == apxinf_metal::W8GroupSize::G64
                            && quantization.mlp_down_group_size == apxinf_metal::W8GroupSize::G64
                    })
            }),
        stack3_execution_valid,
        full_attention_mlp_valid: stats
            .full_attention_mlp_layers
            .iter()
            .all(|entry| entry.decode_calls == body_calls),
        head_execution_valid: head.prefill_calls == head_calls.prefill
            && head.decode_calls == head_calls.decode
            && head.teacher_calls == head_calls.teacher,
        aggregate_ledger_valid: official_aggregate_ledger_is_exact(aggregate),
        generation_receipt_valid: stack3_head_generation_receipt_is_exact(
            generation_receipt,
            body_calls,
            head_calls,
        ),
        terminal_clear: !stats.terminal_error
            && generation_receipt
                .get("terminal_error")
                .and_then(Value::as_bool)
                == Some(false),
    }
}

fn stack3_ledger_json(ledger: apxinf_metal::LinearLayerStack3BufferLedger) -> Value {
    json!({
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "packed_weight_bytes": ledger.packed_weight_bytes,
        "packed_scale_bytes": ledger.packed_scale_bytes,
        "f32_parameter_bytes": ledger.f32_parameter_bytes,
        "active_state_bytes": ledger.active_state_bytes,
        "scratch_state_bytes": ledger.scratch_state_bytes,
        "activation_bytes": ledger.activation_bytes,
        "total_persistent_bytes": ledger.total_persistent_bytes,
        "host_input_bytes_per_decode": ledger.host_input_bytes_per_decode,
        "host_output_bytes_per_decode": ledger.host_output_bytes_per_decode,
        "state_host_transfer_bytes_per_decode": ledger.state_host_transfer_bytes_per_decode,
        "command_buffers_per_decode": ledger.command_buffers_per_decode,
        "compute_encoders_per_decode": ledger.compute_encoders_per_decode,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
        "intermediate_host_finite_checks_per_decode": ledger.intermediate_host_finite_checks_per_decode,
        "final_output_finite_checks_per_decode": ledger.final_output_finite_checks_per_decode,
    })
}

fn mlp_ledger_json(ledger: apxinf_metal::MlpBlockBufferLedger) -> Value {
    json!({
        "scope": ledger.scope,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "packed_weight_bytes": ledger.packed_weight_bytes,
        "packed_scale_bytes": ledger.packed_scale_bytes,
        "activation_bytes": ledger.activation_bytes,
        "total_persistent_bytes": ledger.total_persistent_bytes,
        "host_input_bytes_per_decode": ledger.host_input_bytes_per_decode,
        "host_output_bytes_per_decode": ledger.host_output_bytes_per_decode,
        "state_host_transfer_bytes_per_decode": ledger.state_host_transfer_bytes_per_decode,
        "command_buffers_per_decode": ledger.command_buffers_per_decode,
        "compute_encoders_per_decode": ledger.compute_encoders_per_decode,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
    })
}

fn body_ledger_json(ledger: &Qwen35MetalW8LinearLayerStacksV1AggregateLedger) -> Value {
    json!({
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "includes_lm_head": ledger.includes_lm_head,
        "stacks": ledger.stacks.iter().map(|entry| json!({
            "layer_indices": entry.layer_indices,
            "ledger": stack3_ledger_json(entry.ledger),
        })).collect::<Vec<_>>(),
        "full_attention_mlp_layers": ledger.full_attention_mlp_layers.iter().map(|entry| json!({
            "layer_index": entry.layer_index,
            "ledger": mlp_ledger_json(entry.ledger),
        })).collect::<Vec<_>>(),
        "total_persistent_mtlbuffer_bytes": ledger.total_persistent_mtlbuffer_bytes,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "host_to_device_bytes_per_decode": ledger.host_to_device_bytes_per_decode,
        "device_to_host_bytes_per_decode": ledger.device_to_host_bytes_per_decode,
        "state_host_transfer_bytes_per_decode": ledger.state_host_transfer_bytes_per_decode,
        "command_buffers_per_decode": ledger.command_buffers_per_decode,
        "compute_encoders_per_decode": ledger.compute_encoders_per_decode,
        "commits_per_decode": ledger.commits_per_decode,
        "waits_per_decode": ledger.waits_per_decode,
        "intermediate_host_finite_checks_per_decode": ledger.intermediate_host_finite_checks_per_decode,
        "final_output_finite_checks_per_decode": ledger.final_output_finite_checks_per_decode,
    })
}

fn head_ledger_json(ledger: apxinf_metal::LmHeadBufferLedger) -> Value {
    json!({
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "packed_weight_bytes": ledger.packed_weight_bytes,
        "packed_scale_bytes": ledger.packed_scale_bytes,
        "hidden_bytes": ledger.hidden_bytes,
        "partial_topk_bytes": ledger.partial_topk_bytes,
        "output_token_bytes": ledger.output_token_bytes,
        "total_persistent_bytes": ledger.total_persistent_bytes,
        "host_input_bytes_per_call": ledger.host_input_bytes_per_call,
        "host_output_bytes_per_call": ledger.host_output_bytes_per_call,
        "state_host_transfer_bytes_per_call": ledger.state_host_transfer_bytes_per_call,
        "command_buffers_per_call": ledger.command_buffers_per_call,
        "compute_encoders_per_call": ledger.compute_encoders_per_call,
        "commits_per_call": ledger.commits_per_call,
        "waits_per_call": ledger.waits_per_call,
    })
}

fn composite_ledger_json(ledger: &Qwen35MetalW8Stack3LmHeadV2AggregateLedger) -> Value {
    json!({
        "scope": ledger.scope,
        "exclusions": ledger.exclusions,
        "includes_lm_head": ledger.includes_lm_head,
        "body": body_ledger_json(&ledger.body),
        "lm_head": head_ledger_json(ledger.lm_head),
        "total_persistent_mtlbuffer_bytes": ledger.total_persistent_mtlbuffer_bytes,
        "allocated_buffers": ledger.allocated_buffers,
        "shared_buffers": ledger.shared_buffers,
        "private_buffers": ledger.private_buffers,
        "host_to_device_bytes_per_call": ledger.host_to_device_bytes_per_call,
        "device_to_host_bytes_per_call": ledger.device_to_host_bytes_per_call,
        "state_host_transfer_bytes_per_call": ledger.state_host_transfer_bytes_per_call,
        "command_buffers_per_call": ledger.command_buffers_per_call,
        "compute_encoders_per_call": ledger.compute_encoders_per_call,
        "commits_per_call": ledger.commits_per_call,
        "waits_per_call": ledger.waits_per_call,
        "intermediate_host_finite_checks_per_call": ledger.intermediate_host_finite_checks_per_call,
        "final_output_finite_checks_per_call": ledger.final_output_finite_checks_per_call,
    })
}

fn generation_profile_json(
    profile: &apxinf_model::GenerationProfile,
    harness_elapsed_ms: f64,
    setup: Value,
    classification: &str,
) -> Value {
    json!({
        "setup": setup,
        "input_tokens": profile.input_tokens(),
        "output_tokens": profile.output_tokens(),
        "ttft_ms": profile.ttft_ms(),
        "tpot_ms": profile.tpot_ms(),
        "generation_tps": profile.generation_tps(),
        "generation_total_latency_ms": profile.total_latency_ms(),
        "harness_elapsed_ms": harness_elapsed_ms,
        "classification": classification,
    })
}

fn free_timing_classification(mode: Mode) -> &'static str {
    match mode {
        Mode::CpuFree => {
            "CPU reference single-pass diagnostic timing only; never promotion evidence"
        }
        Mode::Stack3HeadV2Free => {
            "candidate-only single pass under an uncontrolled host; never promotion evidence"
        }
        Mode::CpuTeacher | Mode::Stack3HeadV2Teacher => {
            unreachable!("free timing classification is only defined for free modes")
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("qwen35_metal_w8_stack3_head_v2_gate must be built with --release".into());
    }
    if !cfg!(target_os = "macos") {
        return Err("qwen35_metal_w8_stack3_head_v2_gate requires macOS".into());
    }
    let args = parse_args_from(std::env::args_os())?;
    if args.output.exists() {
        return Err(format!(
            "refusing to replace existing receipt {}",
            args.output.display()
        )
        .into());
    }
    let custody = gate_evidence::GateCustody::capture_stack3_lm_head_v2(
        &args.model_dir,
        &args.source_lock,
        GATE_SOURCE_NAME,
        GATE_SOURCE_BYTES,
    )?;
    validate_source_lock(custody.source_lock_value())?;
    let canonical_model_dir = custody.model_dir().to_path_buf();
    let canonical_source_lock = fs::canonicalize(&args.source_lock)?;
    let binary_path = fs::canonicalize(std::env::current_exe()?)?;

    let tokenizer = Tokenizer::from_file(canonical_model_dir.join("tokenizer.json"))?;
    let formatted = tokenizer.apply_chat_template(&[ChatMessage::user(PROMPT)])?;
    let prompt_tokens = tokenizer.encode(&formatted)?;
    let config = Qwen35Config::from_json_file(&canonical_model_dir.join("config.json"))?;
    validate_official_schedule(&config.text.layer_types)?;
    let vocab_size = config.text.vocab_size;

    let checkpoint_started = std::time::Instant::now();
    let (tensors, _) =
        apxinf_loader::safetensors::load_native_path_filtered(&canonical_model_dir, |name| {
            name.starts_with("model.language_model.") || name == "lm_head.weight"
        })?;
    let checkpoint_load_ms = checkpoint_started.elapsed().as_secs_f64() * 1_000.0;
    let max_context = prompt_tokens
        .len()
        .checked_add(STEPS + 1)
        .ok_or("context length overflow")?;
    let construct_started = std::time::Instant::now();
    let mut model = if args.mode.is_candidate() {
        GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_and_lm_head_v2(
            config,
            tensors,
            Device::Cpu,
            max_context,
        )?
    } else {
        GeneralQwen35::from_weights(config, tensors, Device::Cpu, max_context)?
    };
    let model_construct_ms = construct_started.elapsed().as_secs_f64() * 1_000.0;
    let identity = json!({
        "repo_id": REPO_ID,
        "revision": LOCKED_REVISION,
        "checkpoint": LOCKED_CHECKPOINT,
        "checkpoint_sha256": LOCKED_CHECKPOINT_SHA256,
        "checkpoint_bytes": LOCKED_CHECKPOINT_BYTES,
        "model_dir": canonical_model_dir,
        "source_lock": canonical_source_lock,
        "binary_path": binary_path,
        "build_profile": "release",
        "matmul_feature": "accelerate",
        "metal_feature": "metal-w8",
        "custody": custody.receipt_json(),
    });
    let setup = json!({
        "checkpoint_load_ms": checkpoint_load_ms,
        "model_construct_ms": model_construct_ms,
        "timing_classification": "single-pass diagnostic timing only; never formal promotion evidence",
    });
    let mut result = match args.mode {
        Mode::CpuTeacher | Mode::Stack3HeadV2Teacher => run_teacher(
            &args,
            &mut model,
            &prompt_tokens,
            vocab_size,
            identity,
            setup,
        )?,
        Mode::CpuFree | Mode::Stack3HeadV2Free => {
            run_free(&args, &mut model, &prompt_tokens, identity, setup)?
        }
    };
    let custody_end_verification = custody.verify_unchanged_receipt()?;
    result
        .receipt
        .as_object_mut()
        .ok_or("gate receipt root must be an object")?
        .insert("custody_end_verification".into(), custody_end_verification);
    publish_receipt_create_new(&args.output, &result.receipt)?;
    println!("{}", serde_json::to_string(&result.receipt)?);
    if !result.passed {
        std::process::exit(1);
    }
    Ok(())
}

fn run_teacher(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    vocab_size: usize,
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let prefill_started = std::time::Instant::now();
    // This is intentionally the CPU/F32 full projection in both teacher
    // modes. The candidate's Metal head must remain untouched during prefill.
    let prefill_logits = model.prefill_for_generation(LlmInput::text(prompt_tokens))?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1_000.0;
    let prefill_token = argmax(&prefill_logits, vocab_size)?;

    if args.mode == Mode::CpuTeacher {
        let decode_started = std::time::Instant::now();
        let mut teacher = prefill_token;
        let mut teacher_inputs = Vec::with_capacity(STEPS);
        let mut expected_outputs = Vec::with_capacity(STEPS);
        for step in 0..STEPS {
            teacher_inputs.push(teacher);
            let position = prompt_tokens
                .len()
                .checked_add(step)
                .ok_or("teacher position overflow")?;
            let logits = model.forward(&[teacher], u32::try_from(position)?)?;
            teacher = argmax(&logits, vocab_size)?;
            expected_outputs.push(teacher);
        }
        let decode_ms = decode_started.elapsed().as_secs_f64() * 1_000.0;
        return Ok(RunResult {
            receipt: json!({
                "format": args.mode.receipt_format(),
                "mode": args.mode.label(),
                "identity": identity,
                "prompt": PROMPT,
                "prompt_token_ids": prompt_tokens,
                "official_layer_schedule_valid": true,
                "comparisons": STEPS,
                "prefill_token": prefill_token,
                "teacher_input_ids": teacher_inputs,
                "cpu_expected_output_ids": expected_outputs,
                "generation_path_contract": null,
                "timing": {
                    "setup": setup,
                    "prefill_ms": prefill_ms,
                    "decode_ms": decode_ms,
                    "decode_mean_ms": decode_ms / STEPS as f64,
                },
                "passed": true,
            }),
            passed: true,
        });
    }

    let aggregate = model
        .metal_w8_stack3_lm_head_v2_aggregate_ledger()
        .ok_or("v2 constructor omitted the composite ledger")?;
    let prefill_stats = model
        .metal_w8_linear_layer_stacks_v1_stats()
        .ok_or("v2 constructor omitted Stack3 stats")?;
    let prefill_head = model
        .metal_w8_lm_head_stats()
        .ok_or("v2 constructor omitted lm_head stats")?;
    let prefill_generation = model
        .generation_path_receipt()
        .ok_or("v2 constructor omitted generation path receipt")?;
    let prefill_checks = v2_path_checks(
        &prefill_stats,
        prefill_head,
        &aggregate,
        &prefill_generation,
        0,
        HeadCallCounts::ZERO,
    );

    let input_path = args
        .input_receipt
        .as_ref()
        .ok_or("candidate teacher mode requires --input-receipt")?;
    let (cpu_receipt, input_attestation) =
        gate_evidence::read_attested_json(input_path, "Stack3 + lm_head v2 CPU teacher receipt")?;
    let oracle =
        validate_cpu_teacher_receipt(&cpu_receipt, &identity, prompt_tokens, prefill_token)?;

    let decode_started = std::time::Instant::now();
    let mut candidate_hidden_cpu_tokens = Vec::with_capacity(STEPS);
    let mut top4_candidates = Vec::with_capacity(STEPS);
    let mut reranked_tokens = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let position = prompt_tokens
            .len()
            .checked_add(step)
            .ok_or("teacher position overflow")?;
        let comparison = model.teacher_forced_decode_candidates(
            oracle.teacher_inputs[step],
            u32::try_from(position)?,
        )?;
        candidate_hidden_cpu_tokens.push(comparison.cpu_token);
        top4_candidates.push(comparison.w8_candidates);
        reranked_tokens.push(comparison.reranked_token);
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1_000.0;
    let evaluation = evaluate_teacher_candidate(
        &oracle,
        &candidate_hidden_cpu_tokens,
        &top4_candidates,
        &reranked_tokens,
    )?;
    let final_stats = model
        .metal_w8_linear_layer_stacks_v1_stats()
        .ok_or("v2 constructor omitted final Stack3 stats")?;
    let final_head = model
        .metal_w8_lm_head_stats()
        .ok_or("v2 constructor omitted final lm_head stats")?;
    let final_generation = model
        .generation_path_receipt()
        .ok_or("v2 constructor omitted final generation path receipt")?;
    let final_checks = v2_path_checks(
        &final_stats,
        final_head,
        &aggregate,
        &final_generation,
        STEPS,
        HeadCallCounts::teacher(STEPS),
    );
    let passed = evaluation.passed && prefill_checks.all_valid() && final_checks.all_valid();
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "Stack3 + lm_head v2 CPU teacher receipt",
    )?;
    Ok(RunResult {
        receipt: json!({
            "format": args.mode.receipt_format(),
            "mode": args.mode.label(),
            "identity": identity,
            "input_receipt": gate_evidence::attestation_json(&input_attestation),
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "official_layer_schedule_valid": true,
            "comparisons": STEPS,
            "prefill_token": prefill_token,
            "teacher_input_ids": oracle.teacher_inputs,
            "frozen_cpu_expected_output_ids": oracle.expected_outputs,
            "candidate_hidden_cpu_f32_output_ids": candidate_hidden_cpu_tokens,
            "metal_w8_top4_candidate_ids": top4_candidates,
            "f32_reranked_output_ids": reranked_tokens,
            "exactness": {
                "body_mismatches": evaluation.body_mismatches,
                "head_mismatches": evaluation.head_mismatches,
                "end_to_end_mismatches": evaluation.end_to_end_mismatches,
                "candidate_hidden_cpu_matches_frozen_cpu": evaluation.body_mismatches.is_empty(),
                "top4_contains_candidate_hidden_winner_and_rerank_matches": evaluation.head_mismatches.is_empty(),
                "composite_matches_frozen_cpu": evaluation.end_to_end_mismatches.is_empty(),
            },
            "prefill_generation_path_receipt": prefill_generation,
            "final_generation_path_receipt": final_generation,
            "generation_path_contract": {
                "schema": "apxinf-qwen35-stack3-lm-head-generation-path-v2",
                "binds_six_stack3_six_full_attention_mlp_and_tied_head": true,
                "head_mechanism": "metal-w8-top4-f32-rerank",
                "teacher_head_calls": STEPS,
            },
            "aggregate_buffer_ledger": composite_ledger_json(&aggregate),
            "path_checks": {
                "prefill": prefill_checks.receipt_json(),
                "final": final_checks.receipt_json(),
            },
            "timing": {
                "setup": setup,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "decode_mean_ms": decode_ms / STEPS as f64,
                "head_topk4_total_ms": final_head.topk_elapsed_ns as f64 / 1_000_000.0,
                "head_f32_rerank_total_ms": final_head.rerank_elapsed_ns as f64 / 1_000_000.0,
                "classification": "candidate-only single pass under an uncontrolled host; never promotion evidence",
            },
            "passed": passed,
        }),
        passed,
    })
}

fn run_free(
    args: &Args,
    model: &mut GeneralQwen35,
    prompt_tokens: &[u32],
    identity: Value,
    setup: Value,
) -> Result<RunResult, Box<dyn Error>> {
    let input = if args.mode == Mode::Stack3HeadV2Free {
        let input_path = args
            .input_receipt
            .as_ref()
            .ok_or("candidate free mode requires --input-receipt")?;
        let (receipt, attestation) =
            gate_evidence::read_attested_json(input_path, "Stack3 + lm_head v2 CPU free receipt")?;
        let expected = validate_cpu_free_receipt(&receipt, &identity, prompt_tokens)?;
        Some((input_path, attestation, expected))
    } else {
        None
    };

    // Both free modes use the shared production greedy driver. In the v2
    // candidate this must select the head token hooks for prefill and decode.
    let started = std::time::Instant::now();
    let (generated, profile) =
        model.generate_streaming(LlmInput::text(prompt_tokens), STEPS, |_| {}, None)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    if generated.len() != STEPS {
        return Err(format!(
            "shared free generation returned {} tokens, expected {STEPS}",
            generated.len()
        )
        .into());
    }

    if args.mode == Mode::CpuFree {
        return Ok(RunResult {
            receipt: json!({
                "format": args.mode.receipt_format(),
                "mode": args.mode.label(),
                "identity": identity,
                "prompt": PROMPT,
                "prompt_token_ids": prompt_tokens,
                "official_layer_schedule_valid": true,
                "max_new_tokens": STEPS,
                "eos_stopping": false,
                "generated_token_ids": generated,
                "generation_path_contract": null,
                "profile": generation_profile_json(
                    &profile,
                    elapsed_ms,
                    setup,
                    free_timing_classification(args.mode),
                ),
                "passed": true,
            }),
            passed: true,
        });
    }

    let (input_path, input_attestation, expected) = input.expect("candidate input checked above");
    let mismatches = expected
        .iter()
        .zip(&generated)
        .enumerate()
        .filter_map(|(step, (&cpu_expected, &candidate_actual))| {
            (cpu_expected != candidate_actual).then(|| {
                json!({
                    "step": step,
                    "cpu_expected": cpu_expected,
                    "stack3_lm_head_v2_actual": candidate_actual,
                })
            })
        })
        .collect::<Vec<_>>();
    let stats = model
        .metal_w8_linear_layer_stacks_v1_stats()
        .ok_or("v2 constructor omitted final Stack3 stats")?;
    let head = model
        .metal_w8_lm_head_stats()
        .ok_or("v2 constructor omitted final lm_head stats")?;
    let aggregate = model
        .metal_w8_stack3_lm_head_v2_aggregate_ledger()
        .ok_or("v2 constructor omitted the composite ledger")?;
    let generation = model
        .generation_path_receipt()
        .ok_or("v2 constructor omitted final generation path receipt")?;
    let body_calls = STEPS - 1;
    let checks = v2_path_checks(
        &stats,
        head,
        &aggregate,
        &generation,
        body_calls,
        HeadCallCounts::free(body_calls),
    );
    let passed = mismatches.is_empty() && checks.all_valid();
    gate_evidence::verify_file_unchanged(
        input_path,
        &input_attestation,
        "Stack3 + lm_head v2 CPU free receipt",
    )?;
    Ok(RunResult {
        receipt: json!({
            "format": args.mode.receipt_format(),
            "mode": args.mode.label(),
            "identity": identity,
            "input_receipt": gate_evidence::attestation_json(&input_attestation),
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "official_layer_schedule_valid": true,
            "max_new_tokens": STEPS,
            "eos_stopping": false,
            "cpu_expected_token_ids": expected,
            "generated_token_ids": generated,
            "mismatches": mismatches,
            "exact_trajectory": mismatches.is_empty(),
            "final_generation_path_receipt": generation,
            "generation_path_contract": {
                "schema": "apxinf-qwen35-stack3-lm-head-generation-path-v2",
                "shared_generate_streaming": true,
                "binds_six_stack3_six_full_attention_mlp_and_tied_head": true,
                "body_decode_calls": body_calls,
                "head_prefill_calls": 1,
                "head_decode_calls": body_calls,
            },
            "aggregate_buffer_ledger": composite_ledger_json(&aggregate),
            "path_checks": checks.receipt_json(),
            "profile": generation_profile_json(
                &profile,
                elapsed_ms,
                setup,
                free_timing_classification(args.mode),
            ),
            "passed": passed,
        }),
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn temp_output(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "apxinf-stack3-head-v2-{label}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn four_modes_have_distinct_versioned_receipt_formats() {
        let modes = [
            Mode::parse("cpu-teacher").unwrap(),
            Mode::parse("stack3-head-v2-teacher").unwrap(),
            Mode::parse("cpu-free").unwrap(),
            Mode::parse("stack3-head-v2-free").unwrap(),
        ];
        assert_eq!(
            modes.map(Mode::label),
            [
                "cpu_teacher",
                "metal_w8_stack3_head_v2_teacher_forced",
                "cpu_free_run",
                "metal_w8_stack3_head_v2_free_run",
            ]
        );
        let formats = modes.map(Mode::receipt_format);
        assert_eq!(
            formats
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4
        );
        assert!(formats
            .iter()
            .all(|format| format.contains("stack3-head-v2")));
        assert!(!modes[0].requires_input_receipt());
        assert!(modes[1].requires_input_receipt());
        assert!(!modes[2].requires_input_receipt());
        assert!(modes[3].requires_input_receipt());
    }

    #[test]
    fn cpu_teacher_receipt_binds_all_128_steps_and_the_teacher_chain() {
        let expected = (100..228).collect::<Vec<u32>>();
        let mut inputs = Vec::with_capacity(128);
        inputs.push(7);
        inputs.extend_from_slice(&expected[..127]);
        let identity = json!({"binary": "same-release"});
        let mut receipt = json!({
            "format": CPU_TEACHER_FORMAT,
            "mode": Mode::CpuTeacher.label(),
            "passed": true,
            "identity": identity,
            "prompt_token_ids": [1, 2],
            "prefill_token": 7,
            "comparisons": 128,
            "teacher_input_ids": inputs,
            "cpu_expected_output_ids": expected,
        });

        let oracle = validate_cpu_teacher_receipt(&receipt, &identity, &[1, 2], 7).unwrap();

        assert_eq!(oracle.teacher_inputs.len(), 128);
        assert_eq!(oracle.expected_outputs.len(), 128);
        receipt["teacher_input_ids"][17] = json!(999);
        assert!(validate_cpu_teacher_receipt(&receipt, &identity, &[1, 2], 7).is_err());
    }

    #[test]
    fn teacher_candidate_separates_body_head_recall_and_end_to_end_exactness() {
        let expected = (100..228).collect::<Vec<u32>>();
        let oracle = TeacherOracle {
            teacher_inputs: (0..128).collect(),
            expected_outputs: expected.clone(),
        };
        let candidates = expected
            .iter()
            .map(|token| [*token, token + 1, token + 2, token + 3])
            .collect::<Vec<_>>();

        let accepted =
            evaluate_teacher_candidate(&oracle, &expected, &candidates, &expected).unwrap();

        assert!(accepted.passed);
        assert!(accepted.body_mismatches.is_empty());
        assert!(accepted.head_mismatches.is_empty());
        assert!(accepted.end_to_end_mismatches.is_empty());

        let mut body_drift = expected.clone();
        body_drift[9] = 999;
        let rejected =
            evaluate_teacher_candidate(&oracle, &body_drift, &candidates, &expected).unwrap();
        assert!(!rejected.passed);
        assert_eq!(rejected.body_mismatches.len(), 1);

        let mut rerank_drift = expected.clone();
        rerank_drift[11] = 998;
        let rejected =
            evaluate_teacher_candidate(&oracle, &expected, &candidates, &rerank_drift).unwrap();
        assert!(!rejected.passed);
        assert_eq!(rejected.head_mismatches.len(), 1);
        assert_eq!(rejected.end_to_end_mismatches.len(), 1);
    }

    #[test]
    fn cpu_free_receipt_is_an_exact_128_token_identity_bound_oracle() {
        let identity = json!({"binary": "same-release"});
        let tokens = (100..228).collect::<Vec<u32>>();
        let mut receipt = json!({
            "format": CPU_FREE_FORMAT,
            "mode": Mode::CpuFree.label(),
            "passed": true,
            "identity": identity,
            "prompt_token_ids": [1, 2],
            "max_new_tokens": 128,
            "eos_stopping": false,
            "generated_token_ids": tokens,
        });

        let oracle = validate_cpu_free_receipt(&receipt, &identity, &[1, 2]).unwrap();

        assert_eq!(oracle, tokens);
        receipt["generated_token_ids"].as_array_mut().unwrap().pop();
        assert!(validate_cpu_free_receipt(&receipt, &identity, &[1, 2]).is_err());
    }

    #[test]
    fn parser_requires_candidate_input_and_a_new_output_path() {
        let args = parse_args_from([
            "gate",
            "--model-dir",
            "/model",
            "--source-lock",
            "/source-lock.json",
            "--mode",
            "stack3-head-v2-free",
            "--input-receipt",
            "/cpu-free.json",
            "--output",
            "/new-receipt.json",
        ])
        .unwrap();
        assert_eq!(args.mode, Mode::Stack3HeadV2Free);
        assert_eq!(
            args.input_receipt.unwrap(),
            std::path::PathBuf::from("/cpu-free.json")
        );
        assert!(parse_args_from([
            "gate",
            "--model-dir",
            "/model",
            "--source-lock",
            "/lock",
            "--mode",
            "stack3-head-v2-teacher",
            "--output",
            "/new.json",
        ])
        .is_err());
        assert!(parse_args_from([
            "gate",
            "--model-dir",
            "/model",
            "--source-lock",
            "/lock",
            "--mode",
            "cpu-free",
            "--input-receipt",
            "/unexpected.json",
            "--output",
            "/new.json",
        ])
        .is_err());
        assert!(parse_args_from([
            "gate",
            "--model-dir",
            "/model",
            "--model-dir",
            "/other-model",
            "--source-lock",
            "/lock",
            "--mode",
            "cpu-free",
            "--output",
            "/new.json",
        ])
        .is_err());
    }

    #[test]
    fn receipt_publication_uses_create_new_and_never_replaces() {
        let output = temp_output("no-replace.json");
        let first = json!({"formal": "first"});
        let second = json!({"formal": "second"});

        publish_receipt_create_new(&output, &first).unwrap();
        let first_bytes = std::fs::read(&output).unwrap();
        assert!(publish_receipt_create_new(&output, &second).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), first_bytes);
        std::fs::remove_file(output).unwrap();
    }

    #[test]
    fn official_schedule_is_required_in_every_mode() {
        let schedule = (0..24)
            .map(|index| {
                if FULL_ATTENTION_LAYER_INDICES.contains(&index) {
                    Qwen35LayerType::FullAttention
                } else {
                    Qwen35LayerType::LinearAttention
                }
            })
            .collect::<Vec<_>>();
        validate_official_schedule(&schedule).unwrap();
        let mut wrong = schedule.clone();
        wrong[3] = Qwen35LayerType::LinearAttention;
        assert!(validate_official_schedule(&wrong).is_err());
        assert!(validate_official_schedule(&schedule[..23]).is_err());
    }

    #[test]
    fn v2_generation_receipt_binds_stack_mlp_and_head_phase_counts() {
        let body_calls = 7usize;
        let receipt = json!({
            "format": "apxinf-qwen35-stack3-lm-head-generation-path-v2",
            "mechanism": "metal-w8-stack3-lm-head-v2",
            "stack3_mechanism": "metal-w8-linear-layer-stack3-v1",
            "full_attention_mlp_mechanism": "metal-w8-mlp-block-g64",
            "metal_w8_complete_linear_layer_stacks": true,
            "metal_w8_full_attention_mlp_blocks": true,
            "metal_w8_tied_lm_head_topk4_f32_rerank": true,
            "intermediate_host_finite_checks": false,
            "final_output_finite_checks": true,
            "stacks": LINEAR_LAYER_STACK3_INDICES.map(|layer_indices| json!({
                "layer_indices": layer_indices,
                "mechanism": "metal-w8-linear-layer-stack3-v1",
                "gdn_output_group_sizes": [32, 32, 32],
                "prefill_seed_calls": [1, 1, 1],
                "decode_calls": body_calls,
                "successful_decodes": body_calls,
                "failed_decodes": 0,
                "command_buffers": body_calls,
                "compute_encoders": body_calls * 3,
                "commits": body_calls,
                "waits": body_calls,
                "host_to_device_bytes": body_calls * OFFICIAL_HIDDEN_BYTES,
                "device_to_host_bytes": body_calls * OFFICIAL_HIDDEN_BYTES,
                "state_commits": body_calls * 3,
                "last_state_commit_mask": 0b111,
                "committed_stack_version": body_calls,
                "intermediate_host_finite_checks_per_decode": 0,
                "final_output_finite_checks_per_decode": 1,
                "terminal_error": false,
            })),
            "full_attention_mlp_layers": FULL_ATTENTION_LAYER_INDICES.map(|layer_index| json!({
                "layer_index": layer_index,
                "decode_calls": body_calls,
            })),
            "lm_head": {
                "mechanism": "metal-w8-top4-f32-rerank",
                "prefill_calls": 0,
                "decode_calls": 0,
                "teacher_calls": 7,
            },
            "terminal_error": false,
        });
        assert!(stack3_head_generation_receipt_is_exact(
            &receipt,
            body_calls,
            HeadCallCounts::teacher(body_calls),
        ));
        let mut wrong = receipt;
        wrong["lm_head"]["teacher_calls"] = json!(6);
        assert!(!stack3_head_generation_receipt_is_exact(
            &wrong,
            body_calls,
            HeadCallCounts::teacher(body_calls),
        ));
    }

    #[test]
    fn official_head_ledger_and_composite_totals_are_frozen() {
        let head = apxinf_metal::LmHeadBufferLedger::from_dimensions(248_320, 1_024).unwrap();
        assert!(official_head_ledger_is_exact(head));
        assert_eq!(
            OFFICIAL_BODY_PERSISTENT_BYTES + head.total_persistent_bytes,
            OFFICIAL_COMPOSITE_PERSISTENT_BYTES
        );
        assert_eq!(504 + head.allocated_buffers, 509);
        assert_eq!(444 + head.shared_buffers, 448);
        assert_eq!(60 + head.private_buffers, 61);
        assert_eq!(12 + head.command_buffers_per_call, 13);
        assert_eq!(36 + head.compute_encoders_per_call, 38);
        let mut wrong_head = head;
        wrong_head.exclusions = "weakened exclusion set";
        assert!(!official_head_ledger_is_exact(wrong_head));
    }

    #[test]
    fn official_full_attention_mlp_ledger_rejects_each_component_tamper() {
        let ledger = apxinf_metal::MlpBlockBufferLedger {
            scope: "resident-mtlbuffer-only",
            allocated_buffers: 8,
            shared_buffers: 6,
            private_buffers: 2,
            packed_weight_bytes: 11_010_048,
            packed_scale_bytes: 688_128,
            activation_bytes: 51_200,
            total_persistent_bytes: 11_749_376,
            host_input_bytes_per_decode: 4_096,
            host_output_bytes_per_decode: 4_096,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 1,
            compute_encoders_per_decode: 3,
            commits_per_decode: 1,
            waits_per_decode: 1,
        };
        assert!(official_mlp_ledger_is_exact(ledger));
        for tamper in [
            |mut value: apxinf_metal::MlpBlockBufferLedger| {
                value.packed_weight_bytes -= 1;
                value
            },
            |mut value: apxinf_metal::MlpBlockBufferLedger| {
                value.packed_scale_bytes -= 1;
                value
            },
            |mut value: apxinf_metal::MlpBlockBufferLedger| {
                value.activation_bytes -= 1;
                value
            },
        ] {
            assert!(!official_mlp_ledger_is_exact(tamper(ledger)));
        }
    }

    #[test]
    fn candidate_free_timing_is_explicitly_non_promotional() {
        assert_eq!(
            free_timing_classification(Mode::Stack3HeadV2Free),
            "candidate-only single pass under an uncontrolled host; never promotion evidence"
        );
        assert!(free_timing_classification(Mode::CpuFree).contains("CPU reference"));
    }
}

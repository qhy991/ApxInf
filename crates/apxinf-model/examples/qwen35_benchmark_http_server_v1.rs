//! Resident, single-connection HTTP adapter for the Qwen3.5 comparison lane.
//!
//! This adapter is deliberately labelled `NON_FORMAL`: it has not been added
//! to the frozen GateCustody source set. Its `ignore_eos`-aligned generation
//! masks all five llama.cpp EOG token IDs before the greedy prefill and Metal
//! decode selections. The HTTP, reset-epoch, and request-body contracts are
//! usable for diagnostics, but timings are not formal comparison evidence.

#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports))]

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::ffi::OsString;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use apxinf_core::Device;
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FORMAT: &str = "apxinf-qwen35-benchmark-http-server-v1";
const QUALIFICATION: &str = "NON_FORMAL_DIAGNOSTIC_HTTP_ADAPTER_NOT_IN_FROZEN_GATE_CUSTODY";
const GENERATION_POLICY: &str = "-inf-before-greedy";
const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 383;
const MAX_CONTEXT: usize = 256;
const MAX_NEW_TOKENS: usize = 128;
const CANONICAL_PROMPT: &str = "Hello";
const CANONICAL_PROMPT_TOKEN_IDS: [u32; 13] = [
    248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];
const SUPPRESSED_EOG_TOKEN_IDS: [u32; 5] = [248044, 248046, 248063, 248064, 248065];
const SERVED_MODEL_ALIAS: &str = concat!(
    "/Users/haiyan-mini/Agent4Kernel/models/Qwen3.5-0.8B-2fc063647-GGUF/",
    "Qwen3.5-0.8B-2fc063647-Q8_0-llama.gguf"
);
const CANONICAL_REQUEST_BODY: &str = concat!(
    "{\"cache_prompt\":false,\"chat_template_kwargs\":{\"enable_thinking\":false},",
    "\"id_slot\":0,\"ignore_eos\":true,\"max_tokens\":128,",
    "\"messages\":[{\"content\":\"Hello\",\"role\":\"user\"}],\"model\":\"",
    "/Users/haiyan-mini/Agent4Kernel/models/Qwen3.5-0.8B-2fc063647-GGUF/",
    "Qwen3.5-0.8B-2fc063647-Q8_0-llama.gguf",
    "\",\"reasoning_format\":\"none\",\"return_tokens\":true,\"seed\":0,",
    "\"stream\":false,\"temperature\":0,\"verbose\":true}"
);
const CANONICAL_REQUEST_SIZE: usize = 383;
const CANONICAL_REQUEST_SHA256: &str =
    "7773f5337693843f1e8cf3017b98868517cbddd3bc32649e550d8f2fec1d5cf6";
const EMBEDDED_CANDIDATE_COMMIT: Option<&str> = option_env!("APXINF_CANDIDATE_COMMIT");

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    source_lock: PathBuf,
    bind: SocketAddr,
    expected_generation_requests: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HttpMethod {
    Get,
    Post,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParsedRequest<'a> {
    method: HttpMethod,
    path: &'a str,
    body: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParseProgress<'a> {
    NeedMore,
    Complete {
        request: ParsedRequest<'a>,
        consumed: usize,
    },
}

#[derive(Debug, PartialEq, Eq)]
struct HttpParseError(String);

#[derive(Debug)]
struct HttpResponse {
    status: u16,
    reason: &'static str,
    body: Vec<u8>,
    close: bool,
}

impl HttpResponse {
    fn json(status: u16, value: Value, close: bool) -> Result<Self, String> {
        let reason = match status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            409 => "Conflict",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => return Err(format!("unsupported HTTP response status {status}")),
        };
        Ok(Self {
            status,
            reason,
            body: serde_json::to_vec(&value)
                .map_err(|error| format!("serialize HTTP JSON response: {error}"))?,
            close,
        })
    }

    fn error(status: u16, code: &str, message: impl Into<String>) -> Self {
        Self::json(
            status,
            json!({
                "error": {
                    "code": code,
                    "message": message.into(),
                    "type": "apxinf_benchmark_server_error",
                },
                "format": FORMAT,
                "qualification": QUALIFICATION,
            }),
            true,
        )
        .expect("fixed error response status and JSON must serialize")
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn trim_http_whitespace(value: &str) -> &str {
    value.trim_matches([' ', '\t'])
}

fn parse_http_request(input: &[u8]) -> Result<ParseProgress<'_>, HttpParseError> {
    let header_end = match find_bytes(input, b"\r\n\r\n") {
        Some(offset) => offset,
        None if input.len() <= MAX_HEADER_BYTES => return Ok(ParseProgress::NeedMore),
        None => {
            return Err(HttpParseError(format!(
                "HTTP headers exceed {MAX_HEADER_BYTES} bytes"
            )))
        }
    };
    if header_end > MAX_HEADER_BYTES {
        return Err(HttpParseError(format!(
            "HTTP headers exceed {MAX_HEADER_BYTES} bytes"
        )));
    }
    let header_bytes = &input[..header_end];
    if !header_bytes.is_ascii() || header_bytes.contains(&0) {
        return Err(HttpParseError("HTTP headers must be non-NUL ASCII".into()));
    }
    let header_text = std::str::from_utf8(header_bytes)
        .map_err(|_| HttpParseError("HTTP headers are not UTF-8 ASCII".into()))?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| HttpParseError("missing HTTP request line".into()))?;
    let mut request_parts = request_line.split(' ');
    let raw_method = request_parts
        .next()
        .ok_or_else(|| HttpParseError("missing HTTP method".into()))?;
    let path = request_parts
        .next()
        .ok_or_else(|| HttpParseError("missing HTTP path".into()))?;
    let version = request_parts
        .next()
        .ok_or_else(|| HttpParseError("missing HTTP version".into()))?;
    if request_parts.next().is_some() || path.is_empty() {
        return Err(HttpParseError("malformed HTTP request line".into()));
    }
    if version != "HTTP/1.1" {
        return Err(HttpParseError("only HTTP/1.1 is accepted".into()));
    }
    let method = match raw_method {
        "GET" => HttpMethod::Get,
        "POST" => HttpMethod::Post,
        _ => return Err(HttpParseError("only GET and POST are accepted".into())),
    };

    let mut headers = HashMap::<String, &str>::new();
    for line in lines {
        if line.is_empty() {
            return Err(HttpParseError("empty line inside HTTP headers".into()));
        }
        if line.starts_with([' ', '\t']) {
            return Err(HttpParseError("folded HTTP headers are rejected".into()));
        }
        let (name, raw_value) = line
            .split_once(':')
            .ok_or_else(|| HttpParseError("HTTP header lacks a colon".into()))?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(HttpParseError("invalid HTTP header name".into()));
        }
        let name = name.to_ascii_lowercase();
        if headers
            .insert(name.clone(), trim_http_whitespace(raw_value))
            .is_some()
        {
            return Err(HttpParseError(format!("duplicate HTTP header {name}")));
        }
    }
    if !headers.contains_key("host") {
        return Err(HttpParseError("HTTP/1.1 Host header is required".into()));
    }
    if headers.contains_key("transfer-encoding") {
        return Err(HttpParseError("Transfer-Encoding is rejected".into()));
    }
    let content_length = match headers.get("content-length") {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| HttpParseError("invalid Content-Length".into()))?,
        None => 0,
    };
    if content_length > MAX_BODY_BYTES {
        return Err(HttpParseError(format!(
            "HTTP body exceeds {MAX_BODY_BYTES} bytes"
        )));
    }
    match method {
        HttpMethod::Get => {
            if content_length != 0 {
                return Err(HttpParseError("GET requests must not carry a body".into()));
            }
        }
        HttpMethod::Post => {
            if !headers.contains_key("content-length") {
                return Err(HttpParseError(
                    "POST requests require Content-Length".into(),
                ));
            }
            if headers.get("content-type") != Some(&"application/json") {
                return Err(HttpParseError(
                    "POST requests require Content-Type: application/json".into(),
                ));
            }
        }
    }
    let body_start = header_end + 4;
    let consumed = body_start
        .checked_add(content_length)
        .ok_or_else(|| HttpParseError("HTTP request size overflow".into()))?;
    if input.len() < consumed {
        return Ok(ParseProgress::NeedMore);
    }
    let body = &input[body_start..consumed];
    Ok(ParseProgress::Complete {
        request: ParsedRequest { method, path, body },
        consumed,
    })
}

#[derive(Debug)]
struct GenerationOutput {
    content: String,
    rendered_prompt: String,
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    apxinf_timings: Value,
    generation_path_receipt: Option<Value>,
}

trait ResidentBenchmarkEngine {
    fn reset_checked(&mut self) -> Result<(), String>;
    fn generate_canonical(&mut self) -> Result<GenerationOutput, String>;
    fn state_receipt(&self) -> Value;
}

#[derive(Debug)]
struct SourceLockReceipt {
    path: PathBuf,
    size_bytes: u64,
    sha256: String,
}

struct ApxInfEngine {
    model: GeneralQwen35,
    tokenizer: Tokenizer,
    model_dir: PathBuf,
    source_lock: SourceLockReceipt,
    candidate_commit: String,
}

#[derive(Debug)]
struct AlignedGeneration {
    token_ids: Vec<u32>,
    first_token_ready_ns: u128,
    generation_elapsed_ns: u128,
}

fn generate_exact_ignore_eos_v1(
    model: &mut GeneralQwen35,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
) -> Result<AlignedGeneration, String> {
    if prompt_token_ids.is_empty() {
        return Err("canonical prompt unexpectedly encoded to zero tokens".into());
    }
    model
        .validate_generation_budget(prompt_token_ids.len(), max_new_tokens)
        .map_err(|error| format!("generation budget: {error}"))?;
    model.prewarm_decode(prompt_token_ids.len(), max_new_tokens);
    let generation_started = Instant::now();
    let first_token = model
        .prefill_token_for_generation_excluding(
            LlmInput::text(prompt_token_ids),
            &SUPPRESSED_EOG_TOKEN_IDS,
        )
        .map_err(|error| format!("prefill: {error}"))?;
    let first_token_ready_ns = generation_started.elapsed().as_nanos();
    let mut token_ids = Vec::with_capacity(max_new_tokens);
    token_ids.push(first_token);
    let mut current_token = first_token;
    for decode_index in 0..max_new_tokens.saturating_sub(1) {
        let position = prompt_token_ids
            .len()
            .checked_add(decode_index)
            .ok_or_else(|| "decode position overflow".to_owned())?;
        let position =
            u32::try_from(position).map_err(|_| "decode position does not fit u32".to_owned())?;
        current_token = model
            .decode_token_excluding(current_token, position, &SUPPRESSED_EOG_TOKEN_IDS)
            .map_err(|error| format!("decode at position {position}: {error}"))?;
        token_ids.push(current_token);
    }
    Ok(AlignedGeneration {
        token_ids,
        first_token_ready_ns,
        generation_elapsed_ns: generation_started.elapsed().as_nanos(),
    })
}

impl ApxInfEngine {
    fn load(args: &Args, candidate_commit: String) -> Result<Self, Box<dyn Error>> {
        let model_dir = std::fs::canonicalize(&args.model_dir)?;
        if !model_dir.is_dir() {
            return Err(format!(
                "model directory is not a directory: {}",
                model_dir.display()
            )
            .into());
        }
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
        let config = Qwen35Config::from_json_file(&model_dir.join("config.json"))?;
        let (tensors, _) =
            apxinf_loader::safetensors::load_native_path_filtered(&model_dir, |name| {
                name.starts_with("model.language_model.") || name == "lm_head.weight"
            })?;
        let model = GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1(
            config,
            tensors,
            Device::Cpu,
            MAX_CONTEXT,
        )?;
        let source_lock = source_lock_receipt(&args.source_lock)?;
        Ok(Self {
            model,
            tokenizer,
            model_dir,
            source_lock,
            candidate_commit,
        })
    }
}

impl ResidentBenchmarkEngine for ApxInfEngine {
    fn reset_checked(&mut self) -> Result<(), String> {
        self.model
            .reset_checked()
            .map_err(|error| format!("checked model reset: {error}"))
    }

    fn generate_canonical(&mut self) -> Result<GenerationOutput, String> {
        let request_started = Instant::now();
        let template_started = Instant::now();
        let formatted = self
            .tokenizer
            .apply_chat_template(&[ChatMessage::user(CANONICAL_PROMPT)])
            .map_err(|error| format!("apply chat template: {error}"))?;
        let template_elapsed_ns = template_started.elapsed().as_nanos();
        let encode_started = Instant::now();
        let prompt_token_ids = self
            .tokenizer
            .encode(&formatted)
            .map_err(|error| format!("encode rendered prompt: {error}"))?;
        let encode_elapsed_ns = encode_started.elapsed().as_nanos();
        if prompt_token_ids != CANONICAL_PROMPT_TOKEN_IDS {
            return Err(format!(
                "rendered prompt token IDs differ from frozen 13-token contract: {prompt_token_ids:?}"
            ));
        }
        let generation =
            generate_exact_ignore_eos_v1(&mut self.model, &prompt_token_ids, MAX_NEW_TOKENS)?;
        if generation.token_ids.len() != MAX_NEW_TOKENS {
            return Err(format!(
                "generation returned {} tokens instead of {MAX_NEW_TOKENS}",
                generation.token_ids.len()
            ));
        }
        if let Some(token) = generation
            .token_ids
            .iter()
            .find(|token| SUPPRESSED_EOG_TOKEN_IDS.contains(token))
        {
            return Err(format!("generated suppressed EOG token {token}"));
        }
        let content = self
            .tokenizer
            .decode(&generation.token_ids)
            .map_err(|error| format!("decode generated tokens: {error}"))?;
        let generation_path_receipt = compact_generation_path_receipt(
            self.model
                .generation_path_receipt()
                .ok_or_else(|| "optimized model omitted generation path receipt".to_owned())?,
        )?;
        let decode_intervals = generation.token_ids.len().saturating_sub(1);
        let tpot_ns = if decode_intervals == 0 {
            0
        } else {
            generation
                .generation_elapsed_ns
                .saturating_sub(generation.first_token_ready_ns)
                / decode_intervals as u128
        };
        Ok(GenerationOutput {
            content,
            rendered_prompt: formatted,
            prompt_token_ids,
            generated_token_ids: generation.token_ids,
            apxinf_timings: json!({
                "qualification": QUALIFICATION,
                "generation_policy": GENERATION_POLICY,
                "template_ns": u64_saturating(template_elapsed_ns),
                "encode_ns": u64_saturating(encode_elapsed_ns),
                "generation_ns": u64_saturating(generation.generation_elapsed_ns),
                "ttft_ns": u64_saturating(generation.first_token_ready_ns),
                "tpot_ns": u64_saturating(tpot_ns),
                "server_engine_request_ns": u64_saturating(request_started.elapsed().as_nanos()),
                "timings_are_not_formal_comparison_evidence": true,
            }),
            generation_path_receipt: Some(generation_path_receipt),
        })
    }

    fn state_receipt(&self) -> Value {
        json!({
            "actual_model_dir": self.model_dir,
            "served_model_alias": SERVED_MODEL_ALIAS,
            "source_lock": {
                "path": self.source_lock.path,
                "size_bytes": self.source_lock.size_bytes,
                "sha256": self.source_lock.sha256,
                "custody_status": "read-only diagnostic receipt; not frozen GateCustody",
            },
            "candidate_commit": self.candidate_commit,
            "generation_policy": GENERATION_POLICY,
            "suppressed_eog_token_ids": SUPPRESSED_EOG_TOKEN_IDS,
            "resident_model": true,
            "resident_tokenizer": true,
        })
    }
}

fn u64_saturating(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn source_lock_receipt(path: &Path) -> Result<SourceLockReceipt, Box<dyn Error>> {
    let symlink_metadata = std::fs::symlink_metadata(path)?;
    if !symlink_metadata.file_type().is_file() || symlink_metadata.file_type().is_symlink() {
        return Err(format!(
            "source lock is not a direct regular file: {}",
            path.display()
        )
        .into());
    }
    let path = std::fs::canonicalize(path)?;
    let bytes = std::fs::read(&path)?;
    Ok(SourceLockReceipt {
        path,
        size_bytes: u64::try_from(bytes.len())?,
        sha256: sha256_hex(&bytes),
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn compact_generation_path_receipt(receipt: Value) -> Result<Value, String> {
    let object = receipt
        .as_object()
        .ok_or_else(|| "generation path receipt is not an object".to_owned())?;
    let decode = object
        .get("decode_head")
        .and_then(Value::as_object)
        .ok_or_else(|| "generation path receipt omitted decode_head".to_owned())?;
    let boundaries = object
        .get("boundaries")
        .and_then(Value::as_array)
        .ok_or_else(|| "generation path receipt omitted boundaries".to_owned())?;
    let expected_decode_calls = MAX_NEW_TOKENS.saturating_sub(1) as u64;
    let initial_valid = object.get("initial_stack").is_some_and(|initial| {
        initial["decode_calls"] == expected_decode_calls
            && initial["successful_decodes"] == expected_decode_calls
            && initial["failed_decodes"] == 0
            && initial["terminal_error"] == false
    });
    let boundaries_valid = boundaries.len() == 5
        && boundaries.iter().all(|boundary| {
            boundary["decode_calls"] == expected_decode_calls
                && boundary["successful_decodes"] == expected_decode_calls
                && boundary["failed_decodes"] == 0
                && boundary["terminal_error"] == false
        });
    let decode_api_calls = decode.get("calls").and_then(Value::as_u64).unwrap_or(0);
    let teacher_calls = decode
        .get("teacher_calls")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX);
    let tail_transactions = decode
        .get("tail_transactions")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let successful_tail_transactions = decode
        .get("successful_transactions")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let failed_tail_transactions = decode
        .get("failed_transactions")
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX);
    let prefill_body_calls = object
        .get("prefill_body_calls")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let prefill_head_calls = object
        .get("prefill_head")
        .and_then(|head| head.get("calls"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let terminal_clear = object.get("terminal_error") == Some(&Value::Bool(false))
        && decode.get("terminal_error") == Some(&Value::Bool(false));
    if !initial_valid
        || !boundaries_valid
        || prefill_body_calls != 1
        || prefill_head_calls != 1
        || decode_api_calls != expected_decode_calls
        || teacher_calls != 0
        || tail_transactions != expected_decode_calls
        || successful_tail_transactions != expected_decode_calls
        || failed_tail_transactions != 0
        || !terminal_clear
    {
        return Err(
            "generation path receipt failed prefill/body/decode/tail/terminal postconditions"
                .into(),
        );
    }
    Ok(json!({
        "format": object.get("format"),
        "mechanism": object.get("mechanism"),
        "gdn_core_profile": object.get("gdn_core_profile"),
        "prefill_body_calls": prefill_body_calls,
        "prefill_head_calls": prefill_head_calls,
        "expected_decode_calls": expected_decode_calls,
        "decode_api_calls": decode_api_calls,
        "teacher_calls": teacher_calls,
        "tail_transactions": tail_transactions,
        "successful_tail_transactions": successful_tail_transactions,
        "failed_tail_transactions": failed_tail_transactions,
        "initial_valid": initial_valid,
        "boundaries_valid": boundaries_valid,
        "optimized_excluding_decode_api_hit": decode_api_calls == expected_decode_calls,
        "terminal_error": false,
    }))
}

#[derive(Debug)]
struct ServerState {
    expected_generation_requests: usize,
    completed_generation_requests: usize,
    next_epoch: u64,
    armed_epoch: Option<u64>,
    poisoned: bool,
    poison_reason: Option<String>,
}

impl ServerState {
    fn new(expected_generation_requests: usize) -> Self {
        Self {
            expected_generation_requests,
            completed_generation_requests: 0,
            next_epoch: 1,
            armed_epoch: None,
            poisoned: false,
            poison_reason: None,
        }
    }

    fn poison(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        self.poisoned = true;
        self.armed_epoch = None;
        self.poison_reason = Some(reason);
    }

    fn completed_cleanly(&self) -> bool {
        !self.poisoned
            && self.armed_epoch.is_none()
            && self.completed_generation_requests == self.expected_generation_requests
    }
}

fn dispatch_request<E: ResidentBenchmarkEngine>(
    request: ParsedRequest<'_>,
    state: &mut ServerState,
    engine: &mut E,
) -> HttpResponse {
    if state.poisoned {
        return HttpResponse::error(503, "server_poisoned", "server is terminally poisoned");
    }
    let result = match (request.method, request.path) {
        (HttpMethod::Get, "/health") if request.body.is_empty() => HttpResponse::json(
            200,
            json!({
                "status": "ok",
                "format": FORMAT,
                "qualification": QUALIFICATION,
                "resident": true,
                "formal_evidence_eligible": false,
            }),
            false,
        ),
        (HttpMethod::Get, "/apxinf/state") if request.body.is_empty() => HttpResponse::json(
            200,
            json!({
                "ok": true,
                "format": FORMAT,
                "qualification": QUALIFICATION,
                "formal_evidence_eligible": false,
                "connection_generation": 1,
                "expected_generation_requests": state.expected_generation_requests,
                "completed_generation_requests": state.completed_generation_requests,
                "armed_epoch": state.armed_epoch,
                "poisoned": state.poisoned,
                "canonical_request": {
                    "size_bytes": CANONICAL_REQUEST_SIZE,
                    "sha256": CANONICAL_REQUEST_SHA256,
                },
                "engine": engine.state_receipt(),
            }),
            false,
        ),
        (HttpMethod::Post, "/apxinf/cache/clear") => {
            handle_cache_clear(request.body, state, engine)
        }
        (HttpMethod::Post, "/v1/chat/completions") => {
            handle_chat_completion(request.body, state, engine)
        }
        _ => {
            state.poison(format!(
                "unsupported route {:?} {}",
                request.method, request.path
            ));
            return HttpResponse::error(404, "unsupported_route", "unsupported benchmark route");
        }
    };
    match result {
        Ok(response) => response,
        Err(error) => {
            state.poison(error.clone());
            HttpResponse::error(500, "response_serialization_failed", error)
        }
    }
}

fn handle_cache_clear<E: ResidentBenchmarkEngine>(
    body: &[u8],
    state: &mut ServerState,
    engine: &mut E,
) -> Result<HttpResponse, String> {
    if body != b"{}" {
        state.poison("cache clear body is not exactly {}");
        return Ok(HttpResponse::error(
            400,
            "invalid_cache_clear_body",
            "cache clear body must be exactly {}",
        ));
    }
    if state.armed_epoch.is_some() {
        state.poison("double cache clear attempted before generation consumed the epoch");
        return Ok(HttpResponse::error(
            409,
            "epoch_already_armed",
            "cache clear cannot run twice before one generation",
        ));
    }
    if state.completed_generation_requests >= state.expected_generation_requests {
        state.poison("cache clear attempted after the expected request count completed");
        return Ok(HttpResponse::error(
            409,
            "campaign_already_complete",
            "no additional reset epoch is allowed",
        ));
    }
    if let Err(error) = engine.reset_checked() {
        let message = format!("checked reset failed terminally: {error}");
        state.poison(message.clone());
        return Ok(HttpResponse::error(500, "checked_reset_failed", message));
    }
    let epoch = state.next_epoch;
    state.next_epoch = state
        .next_epoch
        .checked_add(1)
        .ok_or_else(|| "reset epoch counter overflow".to_owned())?;
    state.armed_epoch = Some(epoch);
    HttpResponse::json(
        200,
        json!({
            "ok": true,
            "format": FORMAT,
            "qualification": QUALIFICATION,
            "cache_policy": "checked-reset-exactly-once-before-each-generation",
            "cleared_slots": [0],
            "epoch": epoch,
            "checked_reset_calls_this_epoch": 1,
        }),
        false,
    )
}

fn handle_chat_completion<E: ResidentBenchmarkEngine>(
    body: &[u8],
    state: &mut ServerState,
    engine: &mut E,
) -> Result<HttpResponse, String> {
    if body != CANONICAL_REQUEST_BODY.as_bytes() {
        let actual_sha256 = sha256_hex(body);
        state.poison(format!(
            "chat request body differs: size={} sha256={actual_sha256}",
            body.len()
        ));
        return Ok(HttpResponse::error(
            400,
            "canonical_request_mismatch",
            format!(
                "request must be exact {CANONICAL_REQUEST_SIZE}-byte body with SHA256 {CANONICAL_REQUEST_SHA256}"
            ),
        ));
    }
    if state.completed_generation_requests >= state.expected_generation_requests {
        state.poison("generation attempted after expected request count completed");
        return Ok(HttpResponse::error(
            409,
            "campaign_already_complete",
            "no additional generation is allowed",
        ));
    }
    // Consuming the epoch before inference makes every inference failure
    // terminal.  No retry can reuse an already-mutated model state.
    let Some(epoch) = state.armed_epoch.take() else {
        state.poison("generation attempted without a checked-reset epoch");
        return Ok(HttpResponse::error(
            409,
            "epoch_not_armed",
            "POST /apxinf/cache/clear must succeed exactly once first",
        ));
    };
    let output = match engine.generate_canonical() {
        Ok(output) => output,
        Err(error) => {
            let message = format!("generation epoch {epoch} failed terminally: {error}");
            state.poison(message.clone());
            return Ok(HttpResponse::error(500, "generation_failed", message));
        }
    };
    if output.prompt_token_ids != CANONICAL_PROMPT_TOKEN_IDS
        || output.generated_token_ids.len() != MAX_NEW_TOKENS
        || output
            .generated_token_ids
            .iter()
            .any(|token| SUPPRESSED_EOG_TOKEN_IDS.contains(token))
    {
        let message = format!("generation epoch {epoch} violated prompt/count/EOG postconditions");
        state.poison(message.clone());
        return Ok(HttpResponse::error(
            500,
            "generation_postcondition_failed",
            message,
        ));
    }
    state.completed_generation_requests = state
        .completed_generation_requests
        .checked_add(1)
        .ok_or_else(|| "completed request counter overflow".to_owned())?;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock precedes Unix epoch: {error}"))?
        .as_secs();
    let backend_receipt = engine.state_receipt();
    let response = json!({
        "id": format!("chatcmpl-apxinf-nonformal-epoch-{epoch}"),
        "object": "chat.completion",
        "created": created,
        "model": SERVED_MODEL_ALIAS,
        "system_fingerprint": "apxinf-non-formal-http-v1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output.content,
            },
            "finish_reason": "length",
        }],
        "usage": {
            "prompt_tokens": CANONICAL_PROMPT_TOKEN_IDS.len(),
            "completion_tokens": MAX_NEW_TOKENS,
            "total_tokens": CANONICAL_PROMPT_TOKEN_IDS.len() + MAX_NEW_TOKENS,
        },
        "__verbose": {
            "qualification": QUALIFICATION,
            "formal_evidence_eligible": false,
            "epoch": epoch,
            "model": SERVED_MODEL_ALIAS,
            "prompt": output.rendered_prompt,
            "prompt_token_ids": output.prompt_token_ids,
            "tokens": output.generated_token_ids,
            "tokens_evaluated": CANONICAL_PROMPT_TOKEN_IDS.len(),
            "tokens_predicted": MAX_NEW_TOKENS,
            "stop_type": "limit",
            "generation_settings": {
                "temperature": 0,
                "seed": 0,
                "ignore_eos": true,
                "max_tokens": MAX_NEW_TOKENS,
                "suppressed_eog_token_ids": SUPPRESSED_EOG_TOKEN_IDS,
                "policy": GENERATION_POLICY,
            },
            "generation_path_receipt": output.generation_path_receipt,
            "apxinf_backend": backend_receipt,
        },
        "apxinf_timings": output.apxinf_timings,
    });
    HttpResponse::json(200, response, false)
}

fn write_http_response(stream: &mut TcpStream, response: &HttpResponse) -> std::io::Result<()> {
    let connection = if response.close {
        "close"
    } else {
        "keep-alive"
    };
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: {}\r\nCache-Control: no-store\r\n\r\n",
        response.status,
        response.reason,
        response.body.len(),
        connection,
    );
    stream.write_all(headers.as_bytes())?;
    stream.write_all(&response.body)?;
    stream.flush()
}

fn handle_connection<E: ResidentBenchmarkEngine>(
    stream: &mut TcpStream,
    state: &mut ServerState,
    engine: &mut E,
) -> Result<(), Box<dyn Error>> {
    stream.set_nodelay(true)?;
    let mut buffered = Vec::<u8>::with_capacity(8 * 1024);
    let mut scratch = [0u8; 8 * 1024];
    loop {
        loop {
            let progress = match parse_http_request(&buffered) {
                Ok(progress) => progress,
                Err(error) => {
                    state.poison(format!("HTTP parse failed: {}", error.0));
                    let response = HttpResponse::error(400, "http_parse_failed", error.0);
                    write_http_response(stream, &response)?;
                    return Err("terminal HTTP parse failure".into());
                }
            };
            let ParseProgress::Complete { request, consumed } = progress else {
                break;
            };
            let response = dispatch_request(request, state, engine);
            let close = response.close;
            let status = response.status;
            write_http_response(stream, &response)?;
            buffered.drain(..consumed);
            if close || status != 200 {
                return Err(state
                    .poison_reason
                    .clone()
                    .unwrap_or_else(|| format!("terminal HTTP status {status}"))
                    .into());
            }
        }
        let read = stream.read(&mut scratch)?;
        if read == 0 {
            if !buffered.is_empty() {
                return Err("client closed with an incomplete HTTP request".into());
            }
            if state.completed_cleanly() {
                return Ok(());
            }
            return Err(format!(
                "client closed before clean completion: completed={} expected={} armed_epoch={:?}",
                state.completed_generation_requests,
                state.expected_generation_requests,
                state.armed_epoch,
            )
            .into());
        }
        buffered.extend_from_slice(&scratch[..read]);
    }
}

fn validate_static_contract() -> Result<String, Box<dyn Error>> {
    if !cfg!(target_os = "macos") {
        return Err("qwen35 benchmark HTTP server is macOS-only".into());
    }
    if cfg!(debug_assertions) {
        return Err("qwen35 benchmark HTTP server must be built with --release".into());
    }
    if CANONICAL_REQUEST_BODY.len() != CANONICAL_REQUEST_SIZE {
        return Err(format!(
            "embedded canonical request size is {}, expected {CANONICAL_REQUEST_SIZE}",
            CANONICAL_REQUEST_BODY.len()
        )
        .into());
    }
    let body_sha256 = sha256_hex(CANONICAL_REQUEST_BODY.as_bytes());
    if body_sha256 != CANONICAL_REQUEST_SHA256 {
        return Err(format!(
            "embedded canonical request SHA256 is {body_sha256}, expected {CANONICAL_REQUEST_SHA256}"
        )
        .into());
    }
    let unique_eog = SUPPRESSED_EOG_TOKEN_IDS
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    if unique_eog.len() != SUPPRESSED_EOG_TOKEN_IDS.len() {
        return Err("suppressed EOG token list contains duplicates".into());
    }
    if MAX_CONTEXT < CANONICAL_PROMPT_TOKEN_IDS.len() + MAX_NEW_TOKENS {
        return Err("MAX_CONTEXT cannot hold the canonical generation".into());
    }
    let commit = EMBEDDED_CANDIDATE_COMMIT
        .ok_or("APXINF_CANDIDATE_COMMIT was not embedded at compile time")?;
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("embedded APXINF_CANDIDATE_COMMIT must be 40 lowercase hex characters".into());
    }
    Ok(commit.to_owned())
}

fn usage() -> &'static str {
    "Usage: qwen35_benchmark_http_server_v1 \\
  --model-dir PATH \\
  --source-lock PATH \\
  --bind 127.0.0.1:PORT \\
  --expected-generation-requests N\n\
\n\
Build and run with --release, features accelerate,metal-w8, and a 40-hex \
APXINF_CANDIDATE_COMMIT embedded at compile time. This adapter is NON_FORMAL."
}

fn parse_args_from<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut model_dir = None;
    let mut source_lock = None;
    let mut bind = None;
    let mut expected_generation_requests = None;
    let mut iter = arguments.into_iter();
    while let Some(raw_flag) = iter.next() {
        let flag = raw_flag.to_string_lossy();
        let value = |iter: &mut dyn Iterator<Item = OsString>| {
            iter.next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_ref() {
            "--model-dir" => {
                if model_dir.is_some() {
                    return Err("--model-dir may be specified only once".into());
                }
                model_dir = Some(PathBuf::from(value(&mut iter)?));
            }
            "--source-lock" => {
                if source_lock.is_some() {
                    return Err("--source-lock may be specified only once".into());
                }
                source_lock = Some(PathBuf::from(value(&mut iter)?));
            }
            "--bind" => {
                if bind.is_some() {
                    return Err("--bind may be specified only once".into());
                }
                let raw = value(&mut iter)?;
                let parsed = raw
                    .to_string_lossy()
                    .parse::<SocketAddr>()
                    .map_err(|error| format!("invalid --bind socket address: {error}"))?;
                if !parsed.ip().is_loopback() {
                    return Err("--bind must use a loopback IP address".into());
                }
                bind = Some(parsed);
            }
            "--expected-generation-requests" => {
                if expected_generation_requests.is_some() {
                    return Err("--expected-generation-requests may be specified only once".into());
                }
                let raw = value(&mut iter)?;
                let parsed = raw
                    .to_string_lossy()
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --expected-generation-requests: {error}"))?;
                if parsed == 0 {
                    return Err("--expected-generation-requests must be greater than zero".into());
                }
                expected_generation_requests = Some(parsed);
            }
            "-h" | "--help" => return Err(usage().to_owned()),
            other => return Err(format!("unknown argument {other}\n{}", usage())),
        }
    }
    Ok(Args {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required\n{}", usage()))?,
        source_lock: source_lock
            .ok_or_else(|| format!("--source-lock is required\n{}", usage()))?,
        bind: bind.ok_or_else(|| format!("--bind is required\n{}", usage()))?,
        expected_generation_requests: expected_generation_requests
            .ok_or_else(|| format!("--expected-generation-requests is required\n{}", usage()))?,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let candidate_commit = validate_static_contract()?;
    let args = parse_args_from(std::env::args_os().skip(1))?;
    let mut engine = ApxInfEngine::load(&args, candidate_commit)?;
    let listener = TcpListener::bind(args.bind)?;
    let local_addr = listener.local_addr()?;
    eprintln!(
        "{}",
        serde_json::to_string(&json!({
            "format": FORMAT,
            "event": "ready",
            "qualification": QUALIFICATION,
            "formal_evidence_eligible": false,
            "bind": local_addr,
            "expected_generation_requests": args.expected_generation_requests,
            "canonical_request_body_size_bytes": CANONICAL_REQUEST_SIZE,
            "canonical_request_body_sha256": CANONICAL_REQUEST_SHA256,
            "engine": engine.state_receipt(),
        }))?
    );
    let (mut stream, peer) = listener.accept()?;
    if !peer.ip().is_loopback() {
        return Err(format!("single accepted peer is not loopback: {peer}").into());
    }
    drop(listener);
    let mut state = ServerState::new(args.expected_generation_requests);
    handle_connection(&mut stream, &mut state, &mut engine)?;
    eprintln!(
        "{}",
        serde_json::to_string(&json!({
            "format": FORMAT,
            "event": "complete",
            "qualification": QUALIFICATION,
            "completed_generation_requests": state.completed_generation_requests,
            "expected_generation_requests": state.expected_generation_requests,
            "poisoned": state.poisoned,
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct FakeEngine {
        reset_calls: usize,
        generation_calls: usize,
        fail_reset: bool,
        fail_generation: bool,
        emit_eog: bool,
    }

    impl ResidentBenchmarkEngine for FakeEngine {
        fn reset_checked(&mut self) -> Result<(), String> {
            self.reset_calls += 1;
            if self.fail_reset {
                Err("injected reset failure".into())
            } else {
                Ok(())
            }
        }

        fn generate_canonical(&mut self) -> Result<GenerationOutput, String> {
            self.generation_calls += 1;
            if self.fail_generation {
                return Err("injected generation failure".into());
            }
            let mut generated_token_ids = vec![7; MAX_NEW_TOKENS];
            if self.emit_eog {
                generated_token_ids[13] = SUPPRESSED_EOG_TOKEN_IDS[0];
            }
            Ok(GenerationOutput {
                content: "fake aligned output".into(),
                rendered_prompt: "<rendered>Hello</rendered>".into(),
                prompt_token_ids: CANONICAL_PROMPT_TOKEN_IDS.to_vec(),
                generated_token_ids,
                apxinf_timings: json!({
                    "qualification": QUALIFICATION,
                    "generation_policy": GENERATION_POLICY,
                    "generation_ns": 123,
                }),
                generation_path_receipt: Some(json!({"fake": true})),
            })
        }

        fn state_receipt(&self) -> Value {
            json!({
                "fake": true,
                "reset_calls": self.reset_calls,
                "generation_calls": self.generation_calls,
            })
        }
    }

    fn request<'a>(method: HttpMethod, path: &'a str, body: &'a [u8]) -> ParsedRequest<'a> {
        ParsedRequest { method, path, body }
    }

    fn canonical_wire() -> Vec<u8> {
        let mut wire = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:9000\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            CANONICAL_REQUEST_BODY.len()
        )
        .into_bytes();
        wire.extend_from_slice(CANONICAL_REQUEST_BODY.as_bytes());
        wire
    }

    fn clear(state: &mut ServerState, engine: &mut FakeEngine) -> HttpResponse {
        dispatch_request(
            request(HttpMethod::Post, "/apxinf/cache/clear", b"{}"),
            state,
            engine,
        )
    }

    fn chat(state: &mut ServerState, engine: &mut FakeEngine) -> HttpResponse {
        dispatch_request(
            request(
                HttpMethod::Post,
                "/v1/chat/completions",
                CANONICAL_REQUEST_BODY.as_bytes(),
            ),
            state,
            engine,
        )
    }

    #[test]
    fn canonical_request_body_is_exact_frozen_contract() {
        assert_eq!(CANONICAL_REQUEST_BODY.len(), CANONICAL_REQUEST_SIZE);
        assert_eq!(
            sha256_hex(CANONICAL_REQUEST_BODY.as_bytes()),
            CANONICAL_REQUEST_SHA256
        );
        let parsed: Value = serde_json::from_str(CANONICAL_REQUEST_BODY).unwrap();
        assert_eq!(parsed["model"], SERVED_MODEL_ALIAS);
        assert_eq!(parsed["ignore_eos"], true);
        assert_eq!(parsed["max_tokens"], MAX_NEW_TOKENS);
        assert_eq!(parsed["messages"][0]["content"], CANONICAL_PROMPT);
        assert_eq!(parsed["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn incremental_parser_needs_every_prefix_and_completes_exact_wire() {
        let wire = canonical_wire();
        for prefix_len in 0..wire.len() {
            assert_eq!(
                parse_http_request(&wire[..prefix_len]).unwrap(),
                ParseProgress::NeedMore,
                "prefix {prefix_len} unexpectedly completed"
            );
        }
        let ParseProgress::Complete { request, consumed } = parse_http_request(&wire).unwrap()
        else {
            panic!("full canonical wire did not complete");
        };
        assert_eq!(consumed, wire.len());
        assert_eq!(request.method, HttpMethod::Post);
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.body, CANONICAL_REQUEST_BODY.as_bytes());
    }

    #[test]
    fn parser_leaves_a_following_request_unconsumed() {
        let first = canonical_wire();
        let second = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
        let mut pipelined = first.clone();
        pipelined.extend_from_slice(second);
        let ParseProgress::Complete { consumed, .. } = parse_http_request(&pipelined).unwrap()
        else {
            panic!("first pipelined request did not complete");
        };
        assert_eq!(consumed, first.len());
        let ParseProgress::Complete { request, consumed } =
            parse_http_request(&pipelined[consumed..]).unwrap()
        else {
            panic!("second pipelined request did not complete");
        };
        assert_eq!(request.path, "/health");
        assert_eq!(consumed, second.len());
    }

    #[test]
    fn parser_rejects_ambiguous_or_unsupported_framing() {
        let duplicate_host = b"GET /health HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n";
        assert!(parse_http_request(duplicate_host).is_err());
        let chunked = b"POST /apxinf/cache/clear HTTP/1.1\r\nHost: a\r\nContent-Type: application/json\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}";
        assert!(parse_http_request(chunked).is_err());
        let get_body = b"GET /health HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\n\r\nx";
        assert!(parse_http_request(get_body).is_err());
        let missing_content_type =
            b"POST /apxinf/cache/clear HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\n\r\n{}";
        assert!(parse_http_request(missing_content_type).is_err());
    }

    #[test]
    fn every_single_byte_request_body_mutation_is_terminal() {
        for index in 0..CANONICAL_REQUEST_BODY.len() {
            let mut body = CANONICAL_REQUEST_BODY.as_bytes().to_vec();
            body[index] ^= 1;
            let mut state = ServerState::new(1);
            let mut engine = FakeEngine::default();
            let response = dispatch_request(
                request(HttpMethod::Post, "/v1/chat/completions", &body),
                &mut state,
                &mut engine,
            );
            assert_eq!(response.status, 400, "mutation at byte {index}");
            assert!(response.close, "mutation at byte {index}");
            assert!(state.poisoned, "mutation at byte {index}");
            assert_eq!(engine.reset_calls, 0, "mutation at byte {index}");
            assert_eq!(engine.generation_calls, 0, "mutation at byte {index}");
        }
    }

    #[test]
    fn checked_reset_arms_one_epoch_and_success_consumes_it() {
        let mut state = ServerState::new(1);
        let mut engine = FakeEngine::default();
        let clear_response = clear(&mut state, &mut engine);
        assert_eq!(clear_response.status, 200);
        assert_eq!(engine.reset_calls, 1);
        assert_eq!(state.armed_epoch, Some(1));

        let chat_response = chat(&mut state, &mut engine);
        assert_eq!(chat_response.status, 200);
        assert!(!chat_response.close);
        assert_eq!(engine.generation_calls, 1);
        assert_eq!(state.armed_epoch, None);
        assert_eq!(state.completed_generation_requests, 1);
        assert!(state.completed_cleanly());
    }

    #[test]
    fn double_clear_is_terminal_without_a_second_reset() {
        let mut state = ServerState::new(2);
        let mut engine = FakeEngine::default();
        assert_eq!(clear(&mut state, &mut engine).status, 200);
        let response = clear(&mut state, &mut engine);
        assert_eq!(response.status, 409);
        assert!(response.close);
        assert!(state.poisoned);
        assert_eq!(engine.reset_calls, 1);
        assert_eq!(engine.generation_calls, 0);
    }

    #[test]
    fn generation_without_epoch_is_terminal_and_never_calls_engine() {
        let mut state = ServerState::new(1);
        let mut engine = FakeEngine::default();
        let response = chat(&mut state, &mut engine);
        assert_eq!(response.status, 409);
        assert!(response.close);
        assert!(state.poisoned);
        assert_eq!(engine.reset_calls, 0);
        assert_eq!(engine.generation_calls, 0);
    }

    #[test]
    fn generation_cannot_reuse_a_consumed_epoch() {
        let mut state = ServerState::new(2);
        let mut engine = FakeEngine::default();
        assert_eq!(clear(&mut state, &mut engine).status, 200);
        assert_eq!(chat(&mut state, &mut engine).status, 200);
        let response = chat(&mut state, &mut engine);
        assert_eq!(response.status, 409);
        assert!(state.poisoned);
        assert_eq!(engine.reset_calls, 1);
        assert_eq!(engine.generation_calls, 1);
    }

    #[test]
    fn reset_failure_is_terminal_and_never_arms_epoch() {
        let mut state = ServerState::new(1);
        let mut engine = FakeEngine {
            fail_reset: true,
            ..FakeEngine::default()
        };
        let response = clear(&mut state, &mut engine);
        assert_eq!(response.status, 500);
        assert!(response.close);
        assert!(state.poisoned);
        assert_eq!(state.armed_epoch, None);
        assert_eq!(engine.reset_calls, 1);
        assert_eq!(engine.generation_calls, 0);
    }

    #[test]
    fn generation_failure_consumes_epoch_and_is_terminal() {
        let mut state = ServerState::new(1);
        let mut engine = FakeEngine {
            fail_generation: true,
            ..FakeEngine::default()
        };
        assert_eq!(clear(&mut state, &mut engine).status, 200);
        let response = chat(&mut state, &mut engine);
        assert_eq!(response.status, 500);
        assert!(response.close);
        assert!(state.poisoned);
        assert_eq!(state.armed_epoch, None);
        assert_eq!(engine.reset_calls, 1);
        assert_eq!(engine.generation_calls, 1);
    }

    #[test]
    fn suppressed_eog_postcondition_failure_is_terminal() {
        let mut state = ServerState::new(1);
        let mut engine = FakeEngine {
            emit_eog: true,
            ..FakeEngine::default()
        };
        assert_eq!(clear(&mut state, &mut engine).status, 200);
        let response = chat(&mut state, &mut engine);
        assert_eq!(response.status, 500);
        assert!(state.poisoned);
        assert_eq!(state.armed_epoch, None);
    }

    #[test]
    fn success_response_has_openai_tokens_usage_and_nonformal_receipts() {
        let mut state = ServerState::new(1);
        let mut engine = FakeEngine::default();
        assert_eq!(clear(&mut state, &mut engine).status, 200);
        let response = chat(&mut state, &mut engine);
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["object"], "chat.completion");
        assert_eq!(payload["model"], SERVED_MODEL_ALIAS);
        assert_eq!(payload["choices"][0]["message"]["role"], "assistant");
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            "fake aligned output"
        );
        assert_eq!(payload["choices"][0]["finish_reason"], "length");
        assert_eq!(payload["usage"]["prompt_tokens"], 13);
        assert_eq!(payload["usage"]["completion_tokens"], 128);
        assert_eq!(payload["usage"]["total_tokens"], 141);
        assert_eq!(
            payload["__verbose"]["tokens"].as_array().unwrap().len(),
            128
        );
        assert_eq!(payload["__verbose"]["formal_evidence_eligible"], false);
        assert_eq!(
            payload["__verbose"]["generation_settings"]["policy"],
            GENERATION_POLICY
        );
        assert_eq!(
            payload["__verbose"]["generation_settings"]["suppressed_eog_token_ids"],
            json!(SUPPRESSED_EOG_TOKEN_IDS)
        );
        assert_eq!(payload["apxinf_timings"]["generation_ns"], 123);
    }

    #[test]
    fn state_route_does_not_mutate_epoch_or_engine() {
        let mut state = ServerState::new(3);
        let mut engine = FakeEngine::default();
        let response = dispatch_request(
            request(HttpMethod::Get, "/apxinf/state", b""),
            &mut state,
            &mut engine,
        );
        assert_eq!(response.status, 200);
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["expected_generation_requests"], 3);
        assert_eq!(payload["completed_generation_requests"], 0);
        assert_eq!(payload["poisoned"], false);
        assert_eq!(engine.reset_calls, 0);
        assert_eq!(engine.generation_calls, 0);
    }

    #[test]
    fn unknown_route_is_terminal() {
        let mut state = ServerState::new(1);
        let mut engine = FakeEngine::default();
        let response = dispatch_request(
            request(HttpMethod::Get, "/unknown", b""),
            &mut state,
            &mut engine,
        );
        assert_eq!(response.status, 404);
        assert!(response.close);
        assert!(state.poisoned);
    }

    #[test]
    fn compact_path_receipt_requires_all_body_and_tail_transactions() {
        let boundary = json!({
            "decode_calls": 127,
            "successful_decodes": 127,
            "failed_decodes": 0,
            "terminal_error": false,
        });
        let receipt = json!({
            "format": "path-v1",
            "mechanism": "fused",
            "gdn_core_profile": "gdn-core-fused-v1",
            "prefill_body_calls": 1,
            "prefill_head": {"calls": 1},
            "initial_stack": boundary.clone(),
            "boundaries": vec![boundary; 5],
            "decode_head": {
                "calls": 127,
                "teacher_calls": 0,
                "tail_transactions": 127,
                "successful_transactions": 127,
                "failed_transactions": 0,
                "terminal_error": false,
            },
            "terminal_error": false,
        });
        let compact = compact_generation_path_receipt(receipt.clone()).unwrap();
        assert_eq!(compact["prefill_body_calls"], 1);
        assert_eq!(compact["prefill_head_calls"], 1);
        assert_eq!(compact["decode_api_calls"], 127);
        assert_eq!(compact["teacher_calls"], 0);
        assert_eq!(compact["optimized_excluding_decode_api_hit"], true);

        let mut invalid_boundary = receipt.clone();
        invalid_boundary["boundaries"][3]["failed_decodes"] = json!(1);
        assert!(compact_generation_path_receipt(invalid_boundary).is_err());

        let mut invalid_prefill = receipt.clone();
        invalid_prefill["prefill_head"]["calls"] = json!(0);
        assert!(compact_generation_path_receipt(invalid_prefill).is_err());

        let mut invalid_decode = receipt.clone();
        invalid_decode["decode_head"]["calls"] = json!(126);
        assert!(compact_generation_path_receipt(invalid_decode).is_err());

        let mut invalid_teacher = receipt.clone();
        invalid_teacher["decode_head"]["teacher_calls"] = json!(1);
        assert!(compact_generation_path_receipt(invalid_teacher).is_err());

        let mut invalid_tail = receipt;
        invalid_tail["decode_head"]["successful_transactions"] = json!(126);
        assert!(compact_generation_path_receipt(invalid_tail).is_err());
    }

    #[test]
    fn argument_parser_requires_loopback_and_positive_request_count() {
        let valid = [
            "--model-dir",
            "/model",
            "--source-lock",
            "/lock.json",
            "--bind",
            "127.0.0.1:0",
            "--expected-generation-requests",
            "4",
        ]
        .into_iter()
        .map(OsString::from);
        let args = parse_args_from(valid).unwrap();
        assert_eq!(args.expected_generation_requests, 4);
        assert!(args.bind.ip().is_loopback());

        let non_loopback = [
            "--model-dir",
            "/model",
            "--source-lock",
            "/lock.json",
            "--bind",
            "0.0.0.0:9000",
            "--expected-generation-requests",
            "1",
        ]
        .into_iter()
        .map(OsString::from);
        assert!(parse_args_from(non_loopback).is_err());
    }
}

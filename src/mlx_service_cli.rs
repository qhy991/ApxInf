//! Local application boundary for one long-lived, Rust-validated MLX service.
//!
//! This module deliberately exposes JSONL on stdin/stdout only. It does not
//! open a socket, download a model, or forward unvalidated Python messages.

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use serde_json::{Map, Value};

use crate::mlx_service::{
    parse_line, recoverable_generate_worker_code, recoverable_session_worker_code, MlxService,
    MlxServiceError,
};

const PROTOCOL: &str = "apxinf-mlx-cli-v1";
const REQUEST_FORMAT: &str = "apxinf-mlx-cli-request-v1";
const READY_FORMAT: &str = "apxinf-mlx-cli-ready-v1";
const RESPONSE_FORMAT: &str = "apxinf-mlx-cli-response-v1";
const ERROR_FORMAT: &str = "apxinf-mlx-cli-response-error-v1";
const SHUTDOWN_FORMAT: &str = "apxinf-mlx-cli-shutdown-v1";
const FATAL_FORMAT: &str = "apxinf-mlx-cli-fatal-error-v1";
const RESET_RESULT_FORMAT: &str = "apxinf-mlx-cli-session-reset-result-v1";
const TRANSPORT: &str = "local-stdin-stdout-jsonl-v1";
const MAX_INPUT_LINE_BYTES: usize = 1024 * 1024;
const MAX_OUTPUT_LINE_BYTES: usize = 4 * 1024 * 1024;
const MAX_PROMPT_TOKENS: usize = 131_072;
const MAX_GENERATED_TOKENS: usize = 65_536;
const MAX_TOKEN_ID: u32 = i32::MAX as u32;
const MAX_REQUESTS: usize = 1_000_000;
const MAX_TIMEOUT_SECONDS: u64 = 3600;

#[derive(Debug)]
struct ProtocolError {
    code: &'static str,
    message: String,
}

impl ProtocolError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug)]
struct InferenceRequest {
    prompt: Vec<u32>,
    max_tokens: usize,
    eos: Option<u32>,
    stop_on_eos: bool,
}

#[derive(Debug)]
enum Request {
    Generate {
        request_id: String,
        inference: InferenceRequest,
    },
    SessionGenerate {
        request_id: String,
        session_id: String,
        inference: InferenceRequest,
    },
    SessionReset {
        request_id: String,
        session_id: String,
    },
    Shutdown {
        request_id: String,
    },
}

impl Request {
    fn request_id(&self) -> &str {
        match self {
            Self::Generate { request_id, .. }
            | Self::SessionGenerate { request_id, .. }
            | Self::SessionReset { request_id, .. }
            | Self::Shutdown { request_id } => request_id,
        }
    }

    fn operation(&self) -> &'static str {
        match self {
            Self::Generate { .. } => "generate",
            Self::SessionGenerate { .. } => "session_generate",
            Self::SessionReset { .. } => "session_reset",
            Self::Shutdown { .. } => "shutdown",
        }
    }
}

pub(crate) fn run(python: &Path, runner: &Path, model: &Path, timeout_seconds: u64) -> ExitCode {
    if timeout_seconds == 0 || timeout_seconds > MAX_TIMEOUT_SECONDS {
        emit_fatal(
            "invalid_arguments",
            format!("timeout-seconds must be in 1..={MAX_TIMEOUT_SECONDS}"),
        );
        return ExitCode::from(2);
    }
    let mut service =
        match MlxService::start(python, runner, model, Duration::from_secs(timeout_seconds)) {
            Ok(service) => service,
            Err(error) => {
                emit_fatal("service_start_failed", error.to_string());
                return ExitCode::from(2);
            }
        };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let ready = serde_json::json!({
        "format": READY_FORMAT,
        "protocol": PROTOCOL,
        "transport": TRANSPORT,
        "network_listener": false,
        "operations": ["generate", "session_generate", "session_reset", "shutdown"],
        "limits": {
            "max_input_line_bytes": MAX_INPUT_LINE_BYTES,
            "max_output_line_bytes": MAX_OUTPUT_LINE_BYTES,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_generated_tokens": MAX_GENERATED_TOKENS,
            "max_requests": MAX_REQUESTS,
        },
        "validated_service_ready": service.ready_receipt(),
    });
    if let Err(error) = write_line(&mut output, &ready) {
        return abort_with_fatal(&mut service, "stdout_failed", error.message, 3);
    }

    let mut observed_ids = HashSet::new();
    loop {
        let payload = match read_line(&mut input) {
            Ok(Some(payload)) => payload,
            Ok(None) => {
                return abort_with_fatal(
                    &mut service,
                    "unexpected_eof",
                    "stdin closed before an explicit shutdown request",
                    2,
                );
            }
            Err(error) => {
                return abort_with_fatal(&mut service, error.code, error.message, 2);
            }
        };
        let value = match parse_line(&payload, "application request") {
            Ok(value) => value,
            Err(error) => {
                return abort_with_fatal(&mut service, "invalid_json", error.to_string(), 2);
            }
        };
        let request = match validate_request(&value) {
            Ok(request) => request,
            Err(error) => {
                return abort_with_fatal(&mut service, error.code, error.message, 2);
            }
        };
        if observed_ids.len() >= MAX_REQUESTS {
            return abort_with_fatal(
                &mut service,
                "request_limit",
                "application request limit reached",
                2,
            );
        }
        if !observed_ids.insert(request.request_id().to_string()) {
            return abort_with_fatal(
                &mut service,
                "duplicate_request_id",
                "request_id was already used",
                2,
            );
        }

        let request_id = request.request_id().to_string();
        let operation = request.operation();
        let outcome = match request {
            Request::Generate { inference, .. } => service
                .generate(
                    &inference.prompt,
                    inference.max_tokens,
                    inference.eos,
                    inference.stop_on_eos,
                )
                .map(|generation| generation.receipt),
            Request::SessionGenerate {
                session_id,
                inference,
                ..
            } => service
                .generate_session(
                    &session_id,
                    &inference.prompt,
                    inference.max_tokens,
                    inference.eos,
                    inference.stop_on_eos,
                )
                .map(|generation| generation.receipt),
            Request::SessionReset { session_id, .. } => service
                .reset_session_receipt(&session_id)
                .map(|validated_service_receipt| {
                    serde_json::json!({
                        "format": RESET_RESULT_FORMAT,
                        "session_id": session_id,
                        "validated_service_receipt": validated_service_receipt,
                    })
                }),
            Request::Shutdown { .. } => match service.shutdown() {
                Ok(()) => {
                    let response = serde_json::json!({
                        "format": SHUTDOWN_FORMAT,
                        "protocol": PROTOCOL,
                        "request_id": request_id,
                        "operation": operation,
                    });
                    if let Err(error) = write_line(&mut output, &response) {
                        return abort_with_fatal(&mut service, "stdout_failed", error.message, 3);
                    }
                    return ExitCode::SUCCESS;
                }
                Err(error) => {
                    return abort_with_fatal(
                        &mut service,
                        "service_boundary_failed",
                        error.to_string(),
                        3,
                    );
                }
            },
        };

        match outcome {
            Ok(result) => {
                let response = serde_json::json!({
                    "format": RESPONSE_FORMAT,
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "operation": operation,
                    "validated_service_receipt": result,
                });
                if let Err(error) = write_line(&mut output, &response) {
                    return abort_with_fatal(&mut service, "stdout_failed", error.message, 3);
                }
            }
            Err(error) => {
                let (code, message, recoverable) = classify_service_error(operation, &error);
                if !recoverable {
                    return abort_with_fatal(&mut service, &code, message, 3);
                }
                let response = serde_json::json!({
                    "format": ERROR_FORMAT,
                    "protocol": PROTOCOL,
                    "request_id": request_id,
                    "operation": operation,
                    "error": {
                        "code": code,
                        "message": message,
                        "recoverable": recoverable,
                    }
                });
                if let Err(write_error) = write_line(&mut output, &response) {
                    return abort_with_fatal(&mut service, "stdout_failed", write_error.message, 3);
                }
            }
        }
    }
}

fn abort_with_fatal(
    service: &mut MlxService,
    code: &str,
    message: impl AsRef<str>,
    exit_code: u8,
) -> ExitCode {
    service.abort();
    emit_fatal(code, message);
    ExitCode::from(exit_code)
}

fn classify_service_error(operation: &str, error: &MlxServiceError) -> (String, String, bool) {
    let (code, message, recoverable) = match error {
        MlxServiceError::InvalidInput(message) => {
            ("invalid_request".to_string(), message.clone(), true)
        }
        MlxServiceError::Worker { code, message, .. } => {
            let recoverable = match operation {
                "generate" => recoverable_generate_worker_code(code),
                "session_generate" => recoverable_session_worker_code(code),
                _ => false,
            };
            (code.clone(), message.clone(), recoverable)
        }
        MlxServiceError::Boundary(message) => (
            "service_boundary_failed".to_string(),
            message.clone(),
            false,
        ),
        MlxServiceError::Launch(message) => {
            ("service_launch_failed".to_string(), message.clone(), false)
        }
    };
    (code, safe_message(&message), recoverable)
}

fn validate_request(value: &Value) -> Result<Request, ProtocolError> {
    let root = value
        .as_object()
        .ok_or_else(|| ProtocolError::new("invalid_request", "request root must be an object"))?;
    if root.get("format").and_then(Value::as_str) != Some(REQUEST_FORMAT) {
        return Err(ProtocolError::new(
            "invalid_request",
            format!("request.format must be {REQUEST_FORMAT:?}"),
        ));
    }
    let request_id = request_id(root.get("request_id"))?;
    let operation = root
        .get("operation")
        .and_then(Value::as_str)
        .ok_or_else(|| ProtocolError::new("invalid_request", "operation must be a string"))?;
    match operation {
        "generate" => {
            inference_keys(root, false)?;
            Ok(Request::Generate {
                request_id,
                inference: inference(root)?,
            })
        }
        "session_generate" => {
            inference_keys(root, true)?;
            let session_id = session_id(root.get("session_id"))?;
            let inference = inference(root)?;
            if inference.max_tokens == 0 {
                return Err(ProtocolError::new(
                    "invalid_request",
                    "session_generate requires max_tokens >= 1",
                ));
            }
            Ok(Request::SessionGenerate {
                request_id,
                session_id,
                inference,
            })
        }
        "session_reset" => {
            exact_keys(root, &["format", "request_id", "operation", "session_id"])?;
            Ok(Request::SessionReset {
                request_id,
                session_id: session_id(root.get("session_id"))?,
            })
        }
        "shutdown" => {
            exact_keys(root, &["format", "request_id", "operation"])?;
            Ok(Request::Shutdown { request_id })
        }
        _ => Err(ProtocolError::new(
            "invalid_request",
            "operation is unsupported",
        )),
    }
}

fn inference_keys(root: &Map<String, Value>, session: bool) -> Result<(), ProtocolError> {
    let mut required = vec![
        "format",
        "request_id",
        "operation",
        "prompt_token_ids",
        "max_tokens",
        "stop_on_eos",
    ];
    if session {
        required.push("session_id");
    }
    let mut allowed = required.clone();
    allowed.push("eos_token_id");
    if required.iter().any(|key| !root.contains_key(*key))
        || root.keys().any(|key| !allowed.contains(&key.as_str()))
    {
        return Err(ProtocolError::new(
            "invalid_request",
            "request keys do not match the operation contract",
        ));
    }
    Ok(())
}

fn inference(root: &Map<String, Value>) -> Result<InferenceRequest, ProtocolError> {
    let prompt = root["prompt_token_ids"].as_array().ok_or_else(|| {
        ProtocolError::new("invalid_request", "prompt_token_ids must be an array")
    })?;
    if prompt.is_empty() || prompt.len() > MAX_PROMPT_TOKENS {
        return Err(ProtocolError::new(
            "invalid_request",
            format!("prompt_token_ids must contain 1..={MAX_PROMPT_TOKENS} entries"),
        ));
    }
    let prompt = prompt
        .iter()
        .enumerate()
        .map(|(index, value)| token(value, &format!("prompt_token_ids[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    let max_tokens = root["max_tokens"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value <= MAX_GENERATED_TOKENS)
        .ok_or_else(|| {
            ProtocolError::new(
                "invalid_request",
                format!("max_tokens must be an integer in 0..={MAX_GENERATED_TOKENS}"),
            )
        })?;
    let stop_on_eos = root["stop_on_eos"]
        .as_bool()
        .ok_or_else(|| ProtocolError::new("invalid_request", "stop_on_eos must be a boolean"))?;
    let eos = root
        .get("eos_token_id")
        .map(|value| token(value, "eos_token_id"))
        .transpose()?;
    Ok(InferenceRequest {
        prompt,
        max_tokens,
        eos,
        stop_on_eos,
    })
}

fn token(value: &Value, label: &str) -> Result<u32, ProtocolError> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value <= MAX_TOKEN_ID)
        .ok_or_else(|| {
            ProtocolError::new(
                "invalid_request",
                format!("{label} must be an integer in 0..={MAX_TOKEN_ID}"),
            )
        })
}

fn request_id(value: Option<&Value>) -> Result<String, ProtocolError> {
    let value = value
        .and_then(Value::as_str)
        .filter(|value| safe_id(value, 128))
        .ok_or_else(|| {
            ProtocolError::new(
                "invalid_request",
                "request_id must be 1..128 safe ASCII characters",
            )
        })?;
    Ok(value.to_string())
}

fn session_id(value: Option<&Value>) -> Result<String, ProtocolError> {
    let value = value
        .and_then(Value::as_str)
        .filter(|value| safe_id(value, 64))
        .ok_or_else(|| {
            ProtocolError::new(
                "invalid_request",
                "session_id must be 1..64 safe ASCII characters",
            )
        })?;
    Ok(value.to_string())
}

fn safe_id(value: &str, maximum: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum
        && value.is_ascii()
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric() || (index > 0 && matches!(byte, b'.' | b'_' | b':' | b'-'))
        })
}

fn exact_keys(root: &Map<String, Value>, expected: &[&str]) -> Result<(), ProtocolError> {
    if root.len() != expected.len() || expected.iter().any(|key| !root.contains_key(*key)) {
        return Err(ProtocolError::new(
            "invalid_request",
            "request keys do not match the operation contract",
        ));
    }
    Ok(())
}

fn read_line<R: BufRead>(reader: &mut R) -> Result<Option<Vec<u8>>, ProtocolError> {
    let mut output = Vec::new();
    loop {
        let available = reader
            .fill_buf()
            .map_err(|error| ProtocolError::new("stdin_failed", error.to_string()))?;
        if available.is_empty() {
            return if output.is_empty() {
                Ok(None)
            } else {
                Err(ProtocolError::new(
                    "partial_line",
                    "stdin ended with a partial JSON line",
                ))
            };
        }
        let count = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if output
            .len()
            .checked_add(count)
            .is_none_or(|length| length > MAX_INPUT_LINE_BYTES)
        {
            return Err(ProtocolError::new(
                "line_too_large",
                format!("input line exceeds {MAX_INPUT_LINE_BYTES} bytes"),
            ));
        }
        output.extend_from_slice(&available[..count]);
        let finished = available[count - 1] == b'\n';
        reader.consume(count);
        if finished {
            if output.len() == 1 {
                return Err(ProtocolError::new(
                    "empty_line",
                    "request line must not be empty",
                ));
            }
            return Ok(Some(output));
        }
    }
}

fn write_line<W: Write>(writer: &mut W, value: &Value) -> Result<(), ProtocolError> {
    let mut payload = serde_json::to_vec(value)
        .map_err(|_| ProtocolError::new("serialization_failed", "cannot encode response"))?;
    payload.push(b'\n');
    if payload.len() > MAX_OUTPUT_LINE_BYTES {
        return Err(ProtocolError::new(
            "line_too_large",
            format!("output line exceeds {MAX_OUTPUT_LINE_BYTES} bytes"),
        ));
    }
    writer
        .write_all(&payload)
        .and_then(|()| writer.flush())
        .map_err(|error| ProtocolError::new("stdout_failed", error.to_string()))
}

fn safe_message(value: &str) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    compact.chars().take(1024).collect()
}

fn emit_fatal(code: &str, message: impl AsRef<str>) {
    let value = serde_json::json!({
        "format": FATAL_FORMAT,
        "protocol": PROTOCOL,
        "error": {
            "code": code,
            "message": safe_message(message.as_ref()),
        }
    });
    let mut stderr = std::io::stderr().lock();
    let _ = write_line(&mut stderr, &value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_request_rejects_unknown_fields() {
        let value = serde_json::json!({
            "format": REQUEST_FORMAT,
            "request_id": "r1",
            "operation": "shutdown",
            "unknown": true,
        });
        assert_eq!(
            validate_request(&value).unwrap_err().code,
            "invalid_request"
        );
    }

    #[test]
    fn bounded_reader_rejects_partial_and_oversized_lines() {
        let mut partial = std::io::BufReader::new(&b"{}"[..]);
        assert_eq!(read_line(&mut partial).unwrap_err().code, "partial_line");
        let oversized = vec![b'x'; MAX_INPUT_LINE_BYTES + 1];
        let mut oversized = std::io::BufReader::new(oversized.as_slice());
        assert_eq!(
            read_line(&mut oversized).unwrap_err().code,
            "line_too_large"
        );
    }

    #[test]
    fn writer_rejects_an_oversized_response_before_writing() {
        let value = serde_json::json!({"payload": "x".repeat(MAX_OUTPUT_LINE_BYTES)});
        let mut output = Vec::new();
        assert_eq!(
            write_line(&mut output, &value).unwrap_err().code,
            "line_too_large"
        );
        assert!(output.is_empty());
    }
}

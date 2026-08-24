//! Strict persistent-process boundary for the trusted-local MLX-LM service.
//!
//! A service owns one local model and handles serial JSONL requests. Library
//! offline flags are fixed, but this boundary is not an OS network sandbox.

#![allow(dead_code)] // First slice is exercised by tests before CLI integration.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

const PROTOCOL: &str = "apxinf-mlx-service-v1";
const READY_FORMAT: &str = "apxinf-mlx-service-ready-v1";
const REQUEST_FORMAT: &str = "apxinf-mlx-service-request-v1";
const RESPONSE_FORMAT: &str = "apxinf-mlx-service-response-v1";
const RESPONSE_ERROR_FORMAT: &str = "apxinf-mlx-service-response-error-v1";
const CONTROL_FORMAT: &str = "apxinf-mlx-service-control-v1";
const SHUTDOWN_FORMAT: &str = "apxinf-mlx-service-shutdown-v1";
const SESSION_PROTOCOL: &str = "apxinf-mlx-session-v1";
const SESSION_REQUEST_FORMAT: &str = "apxinf-mlx-session-request-v1";
const SESSION_RESPONSE_FORMAT: &str = "apxinf-mlx-session-response-v1";
const SESSION_RESPONSE_ERROR_FORMAT: &str = "apxinf-mlx-session-response-error-v1";
const SESSION_CONTROL_FORMAT: &str = "apxinf-mlx-session-control-v1";
const SESSION_RESET_FORMAT: &str = "apxinf-mlx-session-reset-v1";
const SESSION_BINDING_FORMAT: &str = "apxinf-mlx-session-binding-v1";
const SESSION_PREFIX_FORMAT: &str = "apxinf-mlx-session-prefix-v1";
const SESSION_CACHE_READY_FORMAT: &str = "apxinf-mlx-session-cache-ready-v1";
const SESSION_CACHE_POLICY: &str = "exact-append-only-in-process-lru-v1";
const GREEDY_STRATEGY: &str = "mlx-generate-step-argmax-v1";
const POLICY: &str = "trusted-local-offline-environment-v1";
const MAX_LINE_BYTES: usize = 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_CONFIG_BYTES: usize = 2 * 1024 * 1024;
const MAX_PROMPT_TOKENS: usize = 131_072;
const MAX_GENERATED_TOKENS: usize = 65_536;
const MAX_REQUESTS: u64 = 1_000_000;
const MAX_SESSIONS: u64 = 4;
const MAX_SESSION_CACHE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_TOKEN_ID: u32 = i32::MAX as u32;
const PINNED_PYTHON_VERSION: &str = "3.14.3";
const PINNED_PACKAGES: [(&str, &str); 8] = [
    ("huggingface-hub", "1.28.0"),
    ("mlx", "0.32.1"),
    ("mlx-lm", "0.31.3"),
    ("mlx-metal", "0.32.1"),
    ("numpy", "2.5.2"),
    ("safetensors", "0.8.0"),
    ("tokenizers", "0.22.2"),
    ("transformers", "5.15.1"),
];

pub(crate) fn recoverable_generate_worker_code(code: &str) -> bool {
    matches!(code, "generation_failed" | "invalid_model")
}

pub(crate) fn recoverable_session_worker_code(code: &str) -> bool {
    recoverable_generate_worker_code(code)
        || matches!(code, "session_cache_failed" | "session_cache_limit")
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MlxServiceMetrics {
    pub(crate) request_ms: f64,
    pub(crate) ttft_ms: f64,
    pub(crate) tpot_ms: f64,
    pub(crate) tps: f64,
    pub(crate) timed_decode_tokens: usize,
    pub(crate) mlx_peak_memory_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MlxServiceGeneration {
    pub(crate) generated_token_ids: Vec<u32>,
    pub(crate) metrics: MlxServiceMetrics,
    pub(crate) receipt: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MlxServiceError {
    InvalidInput(String),
    Launch(String),
    Boundary(String),
    Worker {
        request_id: String,
        code: String,
        message: String,
    },
}

impl fmt::Display for MlxServiceError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(output, "invalid MLX service input: {message}"),
            Self::Launch(message) => write!(output, "cannot launch MLX service: {message}"),
            Self::Boundary(message) => write!(output, "invalid MLX service boundary: {message}"),
            Self::Worker {
                request_id,
                code,
                message,
            } => write!(
                output,
                "MLX request {request_id} failed ({code}): {message}"
            ),
        }
    }
}

impl std::error::Error for MlxServiceError {}

#[derive(Clone)]
struct ProgramIdentity {
    path: PathBuf,
    sha256: String,
}

#[derive(Clone)]
struct ModelIdentity {
    path: PathBuf,
    model_type: String,
    config_sha256: String,
    bundle_file_count: usize,
    bundle_total_bytes: u64,
    bundle_sha256: String,
}

enum Event {
    Line(Vec<u8>),
    Stderr(Vec<u8>),
    Eof,
    Violation(String),
}

/// One long-lived, serial, single-model MLX process.
pub(crate) struct MlxService {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    events: mpsc::Receiver<Event>,
    readers: Vec<thread::JoinHandle<()>>,
    timeout: Duration,
    next_request: u64,
    model: ModelIdentity,
    model_receipt: Value,
    packages_receipt: Value,
    runtime_receipt: Value,
    session_cache_receipt: Value,
    ready_receipt: Value,
    sessions: HashMap<String, Vec<u32>>,
    closed: bool,
}

impl MlxService {
    pub(crate) fn start(
        python: &Path,
        runner: &Path,
        model_dir: &Path,
        timeout: Duration,
    ) -> Result<Self, MlxServiceError> {
        if timeout.is_zero() {
            return Err(invalid("service timeout must be positive"));
        }
        let python = program_identity(python, "Python interpreter", true, 128 << 20)?;
        let runner = program_identity(runner, "MLX service runner", false, 4 << 20)?;
        let helper_path = runner
            .path
            .parent()
            .ok_or_else(|| invalid("service runner has no parent directory"))?
            .join("apxinf_mlx_generate.py");
        let helper = program_identity(&helper_path, "generation helper", false, 4 << 20)?;
        let model = model_identity(model_dir)?;

        let mut command = Command::new(&python.path);
        command
            .arg(&runner.path)
            .arg("--model-dir")
            .arg(&model.path)
            .current_dir("/")
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .env("LANG", "C")
            .env("LC_ALL", "C")
            .env("PYTHONHASHSEED", "0")
            .env("PYTHONNOUSERSITE", "1")
            .env("PYTHONSAFEPATH", "1")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONUTF8", "1")
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .env("HF_DATASETS_OFFLINE", "1")
            .env("HF_HUB_DISABLE_TELEMETRY", "1")
            .env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
            .env("TOKENIZERS_PARALLELISM", "false")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        configure_process_group(&mut command);
        let mut child = command
            .spawn()
            .map_err(|error| MlxServiceError::Launch(error.to_string()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| MlxServiceError::Launch("stdin was not captured".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| MlxServiceError::Launch("stdout was not captured".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| MlxServiceError::Launch("stderr was not captured".into()))?;
        let (sender, events) = mpsc::channel();
        let readers = vec![
            stdout_reader(stdout, sender.clone()),
            stderr_reader(stderr, sender),
        ];
        let mut service = Self {
            child: Some(child),
            stdin: Some(stdin),
            events,
            readers,
            timeout,
            next_request: 1,
            model: model.clone(),
            model_receipt: Value::Null,
            packages_receipt: Value::Null,
            runtime_receipt: Value::Null,
            session_cache_receipt: Value::Null,
            ready_receipt: Value::Null,
            sessions: HashMap::new(),
            closed: false,
        };
        let ready = service.receive("startup handshake")?;
        match validate_ready(&ready, &model, &python, &runner, &helper) {
            Ok((model_receipt, packages_receipt, runtime_receipt, session_cache_receipt)) => {
                service.model_receipt = model_receipt;
                service.packages_receipt = packages_receipt;
                service.runtime_receipt = runtime_receipt;
                service.session_cache_receipt = session_cache_receipt;
                service.ready_receipt = ready;
                Ok(service)
            }
            Err(error) => {
                service.terminate();
                Err(error)
            }
        }
    }

    pub(crate) fn ready_receipt(&self) -> &Value {
        &self.ready_receipt
    }

    pub(crate) fn generate(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        eos: Option<u32>,
        stop_on_eos: bool,
    ) -> Result<MlxServiceGeneration, MlxServiceError> {
        validate_request(prompt, max_tokens, eos)?;
        let request_id = self.request_id()?;
        let mut request = serde_json::json!({
            "format": REQUEST_FORMAT,
            "request_id": request_id,
            "prompt_token_ids": prompt,
            "max_tokens": max_tokens,
            "stop_on_eos": stop_on_eos,
        });
        if let Some(token) = eos {
            request["eos_token_id"] = Value::from(token);
        }
        self.send(&request)?;
        let response = self.receive("generation response")?;
        let result = validate_response(
            &response,
            &request_id,
            &self.model,
            &self.model_receipt,
            &self.packages_receipt,
            &self.runtime_receipt,
            prompt,
            max_tokens,
            eos,
            stop_on_eos,
        );
        let must_terminate = match &result {
            Err(MlxServiceError::Boundary(_)) => true,
            Err(MlxServiceError::Worker { code, .. }) => !recoverable_generate_worker_code(code),
            _ => false,
        };
        if must_terminate {
            self.terminate();
        }
        result
    }

    /// Generate from an exact, append-only conversational prefix.
    ///
    /// This is intentionally separate from `generate`: the ordinary request
    /// never receives a prompt cache.  A failed append invalidates the local
    /// session because Qwen3.5's recurrent cache cannot be rolled back safely.
    pub(crate) fn generate_session(
        &mut self,
        session_id: &str,
        full_prompt: &[u32],
        max_tokens: usize,
        eos: Option<u32>,
        stop_on_eos: bool,
    ) -> Result<MlxServiceGeneration, MlxServiceError> {
        validate_session_id(session_id)?;
        validate_request(full_prompt, max_tokens, eos)?;
        if max_tokens == 0 {
            return Err(invalid("session generation requires max_tokens >= 1"));
        }
        if full_prompt
            .len()
            .checked_add(max_tokens)
            .is_none_or(|value| value > MAX_PROMPT_TOKENS)
        {
            return Err(invalid(
                "session prompt plus generation exceeds the token limit",
            ));
        }

        let previous = self.sessions.get(session_id).cloned();
        let (operation, prefix) = match previous.as_ref() {
            Some(prefix) => {
                if full_prompt.len() <= prefix.len() || !full_prompt.starts_with(prefix) {
                    return Err(invalid("session prompt must be an exact non-empty append"));
                }
                ("append", prefix.as_slice())
            }
            None => ("create", &[][..]),
        };
        let request_id = self.request_id()?;
        let mut request = serde_json::json!({
            "format": SESSION_REQUEST_FORMAT,
            "request_id": request_id,
            "session_id": session_id,
            "operation": operation,
            "prompt_token_ids": full_prompt,
            "expected_prefix": {
                "format": SESSION_PREFIX_FORMAT,
                "token_count": prefix.len(),
                "token_ids_sha256": token_ids_sha256(prefix)?,
            },
            "binding": session_binding(&self.model),
            "max_tokens": max_tokens,
            "stop_on_eos": stop_on_eos,
        });
        if let Some(token) = eos {
            request["eos_token_id"] = Value::from(token);
        }
        self.send(&request)?;
        let response = self.receive("session generation response")?;
        let result = validate_session_response(
            &response,
            &request_id,
            session_id,
            operation,
            prefix,
            &self.model,
            &self.model_receipt,
            &self.packages_receipt,
            &self.runtime_receipt,
            &self.session_cache_receipt,
            full_prompt,
            max_tokens,
            eos,
            stop_on_eos,
        );
        match result {
            Ok((generation, evicted)) => {
                if evicted.iter().any(|selected| selected == session_id) {
                    self.terminate();
                    return Err(boundary("service evicted the session being committed"));
                }
                if evicted
                    .iter()
                    .any(|selected| !self.sessions.contains_key(selected))
                {
                    self.terminate();
                    return Err(boundary("service evicted an unknown session"));
                }
                for selected in evicted {
                    self.sessions.remove(&selected);
                }
                let mut committed = full_prompt.to_vec();
                committed.extend_from_slice(&generation.generated_token_ids);
                self.sessions.insert(session_id.to_string(), committed);
                if generation.receipt["session_cache"]["session_count"].as_u64()
                    != Some(self.sessions.len() as u64)
                {
                    self.terminate();
                    return Err(boundary("service session count diverged from the caller"));
                }
                Ok(generation)
            }
            Err(error) => {
                if previous.is_some() {
                    self.sessions.remove(session_id);
                }
                let must_terminate = match &error {
                    MlxServiceError::Boundary(_) => true,
                    MlxServiceError::Worker { code, .. } => !recoverable_session_worker_code(code),
                    _ => false,
                };
                if must_terminate {
                    self.terminate();
                }
                Err(error)
            }
        }
    }

    pub(crate) fn reset_session(&mut self, session_id: &str) -> Result<(), MlxServiceError> {
        self.reset_session_receipt(session_id).map(|_| ())
    }

    pub(crate) fn reset_session_receipt(
        &mut self,
        session_id: &str,
    ) -> Result<Value, MlxServiceError> {
        validate_session_id(session_id)?;
        let prefix = self
            .sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| invalid("session ID is not resident"))?;
        let request_id = self.request_id()?;
        self.send(&serde_json::json!({
            "format": SESSION_CONTROL_FORMAT,
            "request_id": request_id,
            "operation": "reset",
            "session_id": session_id,
            "expected_prefix": {
                "format": SESSION_PREFIX_FORMAT,
                "token_count": prefix.len(),
                "token_ids_sha256": token_ids_sha256(&prefix)?,
            },
            "binding": session_binding(&self.model),
        }))?;
        let response = self.receive("session reset response")?;
        let result = validate_session_reset_response(
            &response,
            &request_id,
            session_id,
            &prefix,
            &self.model,
            &self.session_cache_receipt,
        );
        self.sessions.remove(session_id);
        if matches!(
            result,
            Err(MlxServiceError::Boundary(_) | MlxServiceError::Worker { .. })
        ) {
            self.terminate();
        }
        result.map(|()| response)
    }

    pub(crate) fn shutdown(&mut self) -> Result<(), MlxServiceError> {
        if self.closed {
            return Ok(());
        }
        let request_id = self.request_id()?;
        self.send(&serde_json::json!({
            "format": CONTROL_FORMAT,
            "request_id": request_id,
            "operation": "shutdown",
        }))?;
        let response = self.receive("shutdown acknowledgement")?;
        let acknowledgement = (|| {
            let root = object(&response, "shutdown acknowledgement")?;
            exact_keys(
                root,
                &["format", "protocol", "request_id"],
                "shutdown acknowledgement",
            )?;
            if root["format"].as_str() != Some(SHUTDOWN_FORMAT)
                || root["protocol"].as_str() != Some(PROTOCOL)
                || root["request_id"].as_str() != Some(request_id.as_str())
            {
                return Err(boundary("shutdown acknowledgement identity mismatch"));
            }
            Ok(())
        })();
        if let Err(error) = acknowledgement {
            self.terminate();
            return Err(error);
        }
        // The acknowledgement commits the graceful protocol operation.  Sweep
        // the still-owned process group before reaping its leader so a forked
        // descendant cannot retain the JSONL pipes, and the group ID cannot be
        // reused between observing exit and signalling the group.
        self.terminate();
        Ok(())
    }

    /// Immediately close the application boundary without sending a worker
    /// control message or waiting for the configured request timeout.
    pub(crate) fn abort(&mut self) {
        self.terminate();
    }

    fn request_id(&mut self) -> Result<String, MlxServiceError> {
        if self.closed || self.next_request > MAX_REQUESTS {
            return Err(boundary("service is closed or exhausted"));
        }
        let value = format!("r{:016x}", self.next_request);
        self.next_request += 1;
        Ok(value)
    }

    fn send(&mut self, value: &Value) -> Result<(), MlxServiceError> {
        let mut payload =
            serde_json::to_vec(value).map_err(|_| boundary("cannot encode request"))?;
        payload.push(b'\n');
        if payload.len() > MAX_LINE_BYTES {
            return Err(invalid(format!("request exceeds {MAX_LINE_BYTES} bytes")));
        }
        let result = self
            .stdin
            .as_mut()
            .ok_or_else(|| boundary("service stdin is closed"))?
            .write_all(&payload)
            .and_then(|()| self.stdin.as_mut().expect("checked").flush());
        if result.is_err() {
            self.terminate();
            return Err(boundary("cannot write complete service request"));
        }
        Ok(())
    }

    fn receive(&mut self, label: &str) -> Result<Value, MlxServiceError> {
        match self.events.recv_timeout(self.timeout) {
            Ok(Event::Line(payload)) => parse_line(&payload, label),
            Ok(Event::Stderr(payload)) => {
                let error = parse_fatal_error(&payload, label);
                self.terminate();
                Err(error)
            }
            Ok(Event::Eof) => {
                self.terminate();
                Err(boundary(format!("service closed before {label}")))
            }
            Ok(Event::Violation(message)) => {
                self.terminate();
                Err(boundary(message))
            }
            Err(_) => {
                self.terminate();
                Err(boundary(format!("{label} exceeded its deadline")))
            }
        }
    }

    fn terminate(&mut self) {
        if self.closed {
            return;
        }
        self.sessions.clear();
        self.stdin.take();
        if let Some(child) = self.child.as_mut() {
            let process_group = child.id();
            // Signal before wait/reap.  While the original Child is unreaped,
            // its PID cannot be recycled as an unrelated process-group ID.
            kill_process_group(process_group, child);
            let _ = child.wait();
        }
        self.child.take();
        self.closed = true;
        self.join_readers();
    }

    fn join_readers(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(2);
        for reader in self.readers.drain(..) {
            while !reader.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            if reader.is_finished() {
                let _ = reader.join();
            }
        }
    }
}

impl Drop for MlxService {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn invalid(message: impl Into<String>) -> MlxServiceError {
    MlxServiceError::InvalidInput(message.into())
}

fn boundary(message: impl Into<String>) -> MlxServiceError {
    MlxServiceError::Boundary(message.into())
}

fn sha256(payload: &[u8]) -> String {
    format!("{:x}", Sha256::digest(payload))
}

fn token_ids_sha256(token_ids: &[u32]) -> Result<String, MlxServiceError> {
    serde_json::to_vec(token_ids)
        .map(|payload| sha256(&payload))
        .map_err(|_| invalid("cannot hash token IDs"))
}

fn session_binding(model: &ModelIdentity) -> Value {
    serde_json::json!({
        "format": SESSION_BINDING_FORMAT,
        "model_config_sha256": model.config_sha256,
        "model_bundle_sha256": model.bundle_sha256,
        "greedy_strategy": GREEDY_STRATEGY,
        "cache_policy": SESSION_CACHE_POLICY,
    })
}

fn validate_session_id(value: &str) -> Result<(), MlxServiceError> {
    if value.is_empty()
        || value.len() > 64
        || !value.is_ascii()
        || !value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric() || (index > 0 && matches!(byte, b'.' | b'_' | b':' | b'-'))
        })
    {
        return Err(invalid("session ID must be 1..64 safe ASCII characters"));
    }
    Ok(())
}

fn program_identity(
    path: &Path,
    label: &str,
    executable: bool,
    max_bytes: usize,
) -> Result<ProgramIdentity, MlxServiceError> {
    if !path.is_absolute() {
        return Err(invalid(format!("{label} path must be absolute")));
    }
    let selected = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("cannot inspect {label}: {error}")))?;
    if selected.file_type().is_symlink() || !selected.file_type().is_file() {
        return Err(invalid(format!("{label} must be a direct regular file")));
    }
    #[cfg(unix)]
    if executable {
        use std::os::unix::fs::PermissionsExt;
        if selected.permissions().mode() & 0o111 == 0 {
            return Err(invalid(format!("{label} is not executable")));
        }
    }
    let path = path
        .canonicalize()
        .map_err(|error| invalid(format!("cannot resolve {label}: {error}")))?;
    if path.to_str().is_none() {
        return Err(invalid(format!("{label} path must be UTF-8")));
    }
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| invalid(format!("cannot inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink()
        || !metadata.file_type().is_file()
        || !same_file(&selected, &metadata)
    {
        return Err(invalid(format!("{label} changed while it was resolved")));
    }
    if metadata.len() > max_bytes as u64 {
        return Err(invalid(format!("{label} exceeds {max_bytes} bytes")));
    }
    let mut file =
        fs::File::open(&path).map_err(|error| invalid(format!("cannot open {label}: {error}")))?;
    let opened = file
        .metadata()
        .map_err(|error| invalid(format!("cannot inspect open {label}: {error}")))?;
    if !same_file(&metadata, &opened) {
        return Err(invalid(format!("{label} changed while it was opened")));
    }
    let mut payload = Vec::with_capacity(opened.len() as usize);
    Read::by_ref(&mut file)
        .take(max_bytes as u64 + 1)
        .read_to_end(&mut payload)
        .map_err(|error| invalid(format!("cannot read {label}: {error}")))?;
    let after = file
        .metadata()
        .map_err(|error| invalid(format!("cannot inspect open {label}: {error}")))?;
    if payload.len() > max_bytes
        || payload.len() as u64 != opened.len()
        || !same_file(&opened, &after)
    {
        return Err(invalid(format!("{label} changed while it was read")));
    }
    Ok(ProgramIdentity {
        path,
        sha256: sha256(&payload),
    })
}

fn model_identity(path: &Path) -> Result<ModelIdentity, MlxServiceError> {
    if !path.is_absolute() {
        return Err(invalid("model directory path must be absolute"));
    }
    let selected = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("cannot inspect model directory: {error}")))?;
    if selected.file_type().is_symlink() || !selected.file_type().is_dir() {
        return Err(invalid("model directory must be a direct directory"));
    }
    let path = path
        .canonicalize()
        .map_err(|error| invalid(format!("cannot resolve model directory: {error}")))?;
    let config_path = path.join("config.json");
    let metadata = fs::symlink_metadata(&config_path)
        .map_err(|error| invalid(format!("cannot inspect model config: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(invalid("model config must be a direct regular file"));
    }
    if metadata.len() > MAX_CONFIG_BYTES as u64 {
        return Err(invalid(format!(
            "model config exceeds {MAX_CONFIG_BYTES} bytes"
        )));
    }
    let payload = fs::read(&config_path)
        .map_err(|error| invalid(format!("cannot read model config: {error}")))?;
    if payload.len() as u64 != metadata.len() {
        return Err(invalid("model config changed while it was read"));
    }
    reject_duplicate_keys(&payload)
        .map_err(|_| invalid("model config contains duplicate object keys"))?;
    let config: Value =
        serde_json::from_slice(&payload).map_err(|_| invalid("model config is not valid JSON"))?;
    let config_root = config
        .as_object()
        .ok_or_else(|| invalid("model config root must be an object"))?;
    if config_root
        .get("model_file")
        .is_some_and(|value| !value.is_null())
        || config_root
            .get("auto_map")
            .is_some_and(|value| !value.is_null())
    {
        return Err(invalid("model config requests remote code"));
    }
    let model_type = config_root
        .get("model_type")
        .and_then(Value::as_str)
        .filter(|value| safe_string(value, 128))
        .ok_or_else(|| invalid("model config model_type is invalid"))?
        .to_string();
    let (bundle_file_count, bundle_total_bytes, bundle_sha256) = bundle_identity(&path)?;
    Ok(ModelIdentity {
        path,
        model_type,
        config_sha256: sha256(&payload),
        bundle_file_count,
        bundle_total_bytes,
        bundle_sha256,
    })
}

fn bundle_identity(path: &Path) -> Result<(usize, u64, String), MlxServiceError> {
    let mut entries = fs::read_dir(path)
        .map_err(|error| invalid(format!("cannot scan model bundle: {error}")))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| invalid(format!("cannot scan model bundle: {error}")))?;
    entries.sort_by_key(|entry| entry.file_name());
    if entries.is_empty() || entries.len() > 4096 {
        return Err(invalid("model bundle must contain 1..=4096 files"));
    }
    let mut manifest = Sha256::new();
    manifest.update(b"apxinf-local-bundle-manifest-v1\0");
    let mut total_bytes = 0_u64;
    for entry in &entries {
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| invalid("model bundle filename must be UTF-8"))?;
        if !safe_string(&name, 1024) || name.contains('/') {
            return Err(invalid("model bundle filename is unsafe"));
        }
        let selected = fs::symlink_metadata(entry.path())
            .map_err(|error| invalid(format!("cannot inspect model bundle file: {error}")))?;
        if selected.file_type().is_symlink() || !selected.file_type().is_file() {
            return Err(invalid(
                "model bundle must contain only direct regular files",
            ));
        }
        let mut file = fs::File::open(entry.path())
            .map_err(|error| invalid(format!("cannot open model bundle file: {error}")))?;
        let opened = file
            .metadata()
            .map_err(|error| invalid(format!("cannot inspect model bundle file: {error}")))?;
        if !same_file(&selected, &opened) {
            return Err(invalid("model bundle file changed while opening"));
        }
        let mut file_digest = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|error| invalid(format!("cannot hash model bundle file: {error}")))?;
            if count == 0 {
                break;
            }
            file_digest.update(&buffer[..count]);
        }
        let after = file
            .metadata()
            .map_err(|error| invalid(format!("cannot inspect model bundle file: {error}")))?;
        if !same_file(&opened, &after) {
            return Err(invalid("model bundle file changed while hashing"));
        }
        let file_hash = format!("{:x}", file_digest.finalize());
        manifest.update(name.as_bytes());
        manifest.update(b"\0");
        manifest.update(opened.len().to_string().as_bytes());
        manifest.update(b"\0");
        manifest.update(file_hash.as_bytes());
        manifest.update(b"\n");
        total_bytes = total_bytes
            .checked_add(opened.len())
            .ok_or_else(|| invalid("model bundle byte count overflowed"))?;
    }
    Ok((
        entries.len(),
        total_bytes,
        format!("{:x}", manifest.finalize()),
    ))
}

#[cfg(unix)]
fn same_file(first: &fs::Metadata, second: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    first.dev() == second.dev()
        && first.ino() == second.ino()
        && first.len() == second.len()
        && first.mode() == second.mode()
}

#[cfg(not(unix))]
fn same_file(first: &fs::Metadata, second: &fs::Metadata) -> bool {
    first.is_file() == second.is_file() && first.len() == second.len()
}

fn validate_request(
    prompt: &[u32],
    max_tokens: usize,
    eos: Option<u32>,
) -> Result<(), MlxServiceError> {
    if prompt.is_empty() || prompt.len() > MAX_PROMPT_TOKENS {
        return Err(invalid(format!(
            "prompt token count must be in 1..={MAX_PROMPT_TOKENS}"
        )));
    }
    if prompt.iter().any(|&token| token > MAX_TOKEN_ID) {
        return Err(invalid("prompt token ID exceeds the supported range"));
    }
    if max_tokens > MAX_GENERATED_TOKENS {
        return Err(invalid(format!(
            "max_tokens must be in 0..={MAX_GENERATED_TOKENS}"
        )));
    }
    if eos.is_some_and(|token| token > MAX_TOKEN_ID) {
        return Err(invalid("EOS token exceeds the supported range"));
    }
    Ok(())
}

fn stdout_reader<R: Read + Send + 'static>(
    mut reader: R,
    sender: mpsc::Sender<Event>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut pending = Vec::new();
        let mut buffer = [0_u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => {
                    let event = if pending.is_empty() {
                        Event::Eof
                    } else {
                        Event::Violation("stdout ended with a partial line".into())
                    };
                    let _ = sender.send(event);
                    return;
                }
                Ok(count) => {
                    pending.extend_from_slice(&buffer[..count]);
                    while let Some(newline) = pending.iter().position(|&byte| byte == b'\n') {
                        let line: Vec<u8> = pending.drain(..=newline).collect();
                        if line.len() > MAX_OUTPUT_BYTES {
                            let _ = sender.send(Event::Violation(format!(
                                "stdout line exceeds {MAX_OUTPUT_BYTES} bytes"
                            )));
                            return;
                        }
                        if sender.send(Event::Line(line)).is_err() {
                            return;
                        }
                    }
                    if pending.len() > MAX_OUTPUT_BYTES {
                        let _ = sender.send(Event::Violation(format!(
                            "stdout line exceeds {MAX_OUTPUT_BYTES} bytes"
                        )));
                        return;
                    }
                }
                Err(_) => {
                    let _ = sender.send(Event::Violation("cannot read stdout".into()));
                    return;
                }
            }
        }
    })
}

fn stderr_reader<R: Read + Send + 'static>(
    mut reader: R,
    sender: mpsc::Sender<Event>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut pending = Vec::new();
        let mut buffer = [0_u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => {
                    if !pending.is_empty() {
                        let _ = sender.send(Event::Violation(
                            "service stderr ended with a partial line".into(),
                        ));
                    }
                    return;
                }
                Ok(count) => {
                    pending.extend_from_slice(&buffer[..count]);
                    if let Some(newline) = pending.iter().position(|&byte| byte == b'\n') {
                        let line: Vec<u8> = pending.drain(..=newline).collect();
                        if sender.send(Event::Stderr(line)).is_err() {
                            return;
                        }
                        if !pending.is_empty() {
                            let _ = sender.send(Event::Violation(
                                "service wrote more than one stderr line".into(),
                            ));
                        }
                        return;
                    }
                    if pending.len() > MAX_OUTPUT_BYTES {
                        let _ = sender.send(Event::Violation(format!(
                            "stderr line exceeds {MAX_OUTPUT_BYTES} bytes"
                        )));
                        return;
                    }
                }
                Err(_) => {
                    let _ = sender.send(Event::Violation("cannot read stderr".into()));
                    return;
                }
            }
        }
    })
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn kill_process_group(process_group: u32, child: &mut Child) {
    unsafe extern "C" {
        fn kill(process_id: i32, signal: i32) -> i32;
    }
    if let Ok(group) = i32::try_from(process_group) {
        unsafe {
            let _ = kill(-group, 9);
        }
    }
    let _ = child.kill();
}

#[cfg(not(unix))]
fn kill_process_group(_process_group: u32, child: &mut Child) {
    let _ = child.kill();
}

pub(crate) fn parse_line(payload: &[u8], label: &str) -> Result<Value, MlxServiceError> {
    if payload.is_empty()
        || payload.last() != Some(&b'\n')
        || payload[..payload.len() - 1]
            .iter()
            .any(|byte| matches!(byte, b'\n' | b'\r'))
    {
        return Err(boundary(format!(
            "{label} must be one newline-terminated JSON line"
        )));
    }
    let body = &payload[..payload.len() - 1];
    let value: Value =
        serde_json::from_slice(body).map_err(|_| boundary(format!("{label} is not valid JSON")))?;
    reject_duplicate_keys(body)
        .map_err(|_| boundary(format!("{label} contains duplicate object keys")))?;
    Ok(value)
}

fn parse_fatal_error(payload: &[u8], label: &str) -> MlxServiceError {
    let parsed = (|| {
        let value = parse_line(payload, "service stderr")?;
        let root = object(&value, "service error")?;
        exact_keys(root, &["format", "error"], "service error")?;
        if root["format"].as_str() != Some("apxinf-mlx-generation-error-v1") {
            return Err(boundary("service error format is invalid"));
        }
        let error = object(&root["error"], "service error.error")?;
        exact_keys(error, &["code", "message"], "service error.error")?;
        let code = clean_string(&error["code"], 64, "service error code")?;
        if !safe_error_code(code) {
            return Err(boundary("service error code is unsafe"));
        }
        let message = clean_string(&error["message"], 1024, "service error message")?;
        Ok(MlxServiceError::Worker {
            request_id: label.to_string(),
            code: code.to_string(),
            message: message.to_string(),
        })
    })();
    parsed.unwrap_or_else(|error| error)
}

fn validate_ready(
    value: &Value,
    model: &ModelIdentity,
    python: &ProgramIdentity,
    runner: &ProgramIdentity,
    helper: &ProgramIdentity,
) -> Result<(Value, Value, Value, Value), MlxServiceError> {
    let root = object(value, "ready")?;
    exact_keys(
        root,
        &[
            "format",
            "protocol",
            "model",
            "packages",
            "runtime",
            "limits",
            "metrics",
            "greedy_strategy",
            "session_cache",
        ],
        "ready",
    )?;
    if root["format"].as_str() != Some(READY_FORMAT)
        || root["protocol"].as_str() != Some(PROTOCOL)
        || root["greedy_strategy"].as_str() != Some(GREEDY_STRATEGY)
    {
        return Err(boundary("ready protocol identity is invalid"));
    }
    validate_model(&root["model"], model)?;
    validate_packages(&root["packages"])?;
    validate_runtime(&root["runtime"], python, runner, helper)?;
    let limits = object(&root["limits"], "ready.limits")?;
    exact_keys(
        limits,
        &[
            "max_line_bytes",
            "max_output_bytes",
            "max_prompt_tokens",
            "max_generated_tokens",
            "max_requests",
        ],
        "ready.limits",
    )?;
    let expected = [
        ("max_line_bytes", MAX_LINE_BYTES as u64),
        ("max_output_bytes", MAX_OUTPUT_BYTES as u64),
        ("max_prompt_tokens", MAX_PROMPT_TOKENS as u64),
        ("max_generated_tokens", MAX_GENERATED_TOKENS as u64),
        ("max_requests", MAX_REQUESTS),
    ];
    if expected
        .iter()
        .any(|(key, value)| limits[*key].as_u64() != Some(*value))
    {
        return Err(boundary("ready limits differ from the caller contract"));
    }
    let metrics = object(&root["metrics"], "ready.metrics")?;
    exact_keys(metrics, &["load_ms"], "ready.metrics")?;
    finite(&metrics["load_ms"], "load_ms")?;
    validate_session_cache_ready(&root["session_cache"])?;
    Ok((
        root["model"].clone(),
        root["packages"].clone(),
        root["runtime"].clone(),
        root["session_cache"].clone(),
    ))
}

fn validate_session_cache_ready(value: &Value) -> Result<(), MlxServiceError> {
    let value = object(value, "ready.session_cache")?;
    exact_keys(
        value,
        &[
            "format",
            "protocol",
            "policy",
            "request_format",
            "control_format",
            "max_sessions",
            "max_bytes",
        ],
        "ready.session_cache",
    )?;
    if value["format"].as_str() != Some(SESSION_CACHE_READY_FORMAT)
        || value["protocol"].as_str() != Some(SESSION_PROTOCOL)
        || value["policy"].as_str() != Some(SESSION_CACHE_POLICY)
        || value["request_format"].as_str() != Some(SESSION_REQUEST_FORMAT)
        || value["control_format"].as_str() != Some(SESSION_CONTROL_FORMAT)
        || value["max_sessions"].as_u64() != Some(MAX_SESSIONS)
        || value["max_bytes"].as_u64() != Some(MAX_SESSION_CACHE_BYTES)
    {
        return Err(boundary("session cache ready contract is invalid"));
    }
    Ok(())
}

fn validate_model(value: &Value, model: &ModelIdentity) -> Result<(), MlxServiceError> {
    let value = object(value, "model")?;
    exact_keys(
        value,
        &[
            "model_dir",
            "model_type",
            "quantization",
            "config_sha256",
            "bundle",
        ],
        "model",
    )?;
    if value["model_dir"].as_str() != model.path.to_str()
        || value["model_type"].as_str() != Some(model.model_type.as_str())
        || value["config_sha256"].as_str() != Some(model.config_sha256.as_str())
    {
        return Err(boundary("model identity differs from local config"));
    }
    let bundle = object(&value["bundle"], "model.bundle")?;
    exact_keys(
        bundle,
        &["format", "file_count", "total_bytes", "sha256"],
        "model.bundle",
    )?;
    if bundle["format"].as_str() != Some("apxinf-local-bundle-manifest-v1")
        || bundle["file_count"].as_u64() != Some(model.bundle_file_count as u64)
        || bundle["total_bytes"].as_u64() != Some(model.bundle_total_bytes)
        || bundle["sha256"].as_str() != Some(model.bundle_sha256.as_str())
    {
        return Err(boundary("bundle manifest differs from local files"));
    }
    Ok(())
}

fn validate_packages(value: &Value) -> Result<(), MlxServiceError> {
    let value = object(value, "packages")?;
    let keys: Vec<&str> = PINNED_PACKAGES.iter().map(|entry| entry.0).collect();
    exact_keys(value, &keys, "packages")?;
    if PINNED_PACKAGES
        .iter()
        .any(|(name, version)| value[*name].as_str() != Some(*version))
    {
        return Err(boundary("packages differ from the pinned profile"));
    }
    Ok(())
}

fn validate_runtime(
    value: &Value,
    python: &ProgramIdentity,
    runner: &ProgramIdentity,
    helper: &ProgramIdentity,
) -> Result<(), MlxServiceError> {
    let value = object(value, "runtime")?;
    exact_keys(
        value,
        &[
            "policy",
            "offline_environment",
            "os_network_sandbox",
            "trust_remote_code",
            "python_version",
            "python",
            "runner",
            "generation_helper",
        ],
        "runtime",
    )?;
    if value["policy"].as_str() != Some(POLICY)
        || value["offline_environment"].as_bool() != Some(true)
        || value["os_network_sandbox"].as_bool() != Some(false)
        || value["trust_remote_code"].as_bool() != Some(false)
        || value["python_version"].as_str() != Some(PINNED_PYTHON_VERSION)
    {
        return Err(boundary("runtime policy or version is invalid"));
    }
    for (field, identity) in [
        ("python", python),
        ("runner", runner),
        ("generation_helper", helper),
    ] {
        let observed = object(&value[field], field)?;
        exact_keys(observed, &["path", "sha256"], field)?;
        if observed["path"].as_str() != identity.path.to_str()
            || observed["sha256"].as_str() != Some(identity.sha256.as_str())
        {
            return Err(boundary(format!("runtime {field} identity mismatch")));
        }
    }
    Ok(())
}

fn session_worker_error(
    root: &Map<String, Value>,
    request_id: &str,
) -> Result<MlxServiceError, MlxServiceError> {
    exact_keys(
        root,
        &["format", "protocol", "request_id", "error"],
        "session error response",
    )?;
    if root["format"].as_str() != Some(SESSION_RESPONSE_ERROR_FORMAT)
        || root["protocol"].as_str() != Some(SESSION_PROTOCOL)
        || root["request_id"].as_str() != Some(request_id)
    {
        return Err(boundary("session error response identity mismatch"));
    }
    let error = object(&root["error"], "session response.error")?;
    exact_keys(error, &["code", "message"], "session response.error")?;
    let code = clean_string(&error["code"], 64, "session error code")?;
    if !safe_error_code(code) {
        return Err(boundary("session error response code is unsafe"));
    }
    let message = clean_string(&error["message"], 1024, "session error message")?;
    Ok(MlxServiceError::Worker {
        request_id: request_id.into(),
        code: code.into(),
        message: message.into(),
    })
}

fn validate_session_cache_state(
    value: &Value,
    ready: &Value,
    allow_evictions: bool,
) -> Result<Vec<String>, MlxServiceError> {
    let value = object(value, "session_cache")?;
    let expected_keys = if allow_evictions {
        vec![
            "policy",
            "session_count",
            "total_cache_bytes",
            "max_sessions",
            "max_bytes",
            "evicted_session_ids",
        ]
    } else {
        vec![
            "policy",
            "session_count",
            "total_cache_bytes",
            "max_sessions",
            "max_bytes",
        ]
    };
    exact_keys(value, &expected_keys, "session_cache")?;
    let ready = object(ready, "ready.session_cache receipt")?;
    if value["policy"] != ready["policy"]
        || value["max_sessions"] != ready["max_sessions"]
        || value["max_bytes"] != ready["max_bytes"]
        || value["policy"].as_str() != Some(SESSION_CACHE_POLICY)
        || value["max_sessions"].as_u64() != Some(MAX_SESSIONS)
        || value["max_bytes"].as_u64() != Some(MAX_SESSION_CACHE_BYTES)
    {
        return Err(boundary("session cache policy changed after ready"));
    }
    let session_count = value["session_count"]
        .as_u64()
        .filter(|count| *count <= MAX_SESSIONS)
        .ok_or_else(|| boundary("session cache count exceeds its limit"))?;
    let total_bytes = value["total_cache_bytes"]
        .as_u64()
        .filter(|bytes| *bytes <= MAX_SESSION_CACHE_BYTES)
        .ok_or_else(|| boundary("session cache bytes exceed their limit"))?;
    let _ = (session_count, total_bytes);
    if !allow_evictions {
        return Ok(Vec::new());
    }
    let evictions = value["evicted_session_ids"]
        .as_array()
        .ok_or_else(|| boundary("evicted session IDs must be an array"))?;
    let mut observed = HashSet::new();
    let mut result = Vec::with_capacity(evictions.len());
    for selected in evictions {
        let selected = clean_string(selected, 64, "evicted session ID")?;
        validate_session_id(selected).map_err(|_| boundary("evicted session ID is unsafe"))?;
        if !observed.insert(selected.to_string()) {
            return Err(boundary("evicted session IDs contain a duplicate"));
        }
        result.push(selected.to_string());
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn validate_session_response(
    value: &Value,
    request_id: &str,
    session_id: &str,
    operation: &str,
    prefix: &[u32],
    model: &ModelIdentity,
    model_receipt: &Value,
    packages_receipt: &Value,
    runtime_receipt: &Value,
    session_cache_receipt: &Value,
    full_prompt: &[u32],
    max_tokens: usize,
    eos: Option<u32>,
    stop_on_eos: bool,
) -> Result<(MlxServiceGeneration, Vec<String>), MlxServiceError> {
    let root = object(value, "session response")?;
    if root.get("format").and_then(Value::as_str) == Some(SESSION_RESPONSE_ERROR_FORMAT) {
        return Err(session_worker_error(root, request_id)?);
    }
    exact_keys(
        root,
        &[
            "format",
            "protocol",
            "request_id",
            "request",
            "session",
            "session_cache",
            "model",
            "packages",
            "runtime",
            "metrics",
            "generation",
        ],
        "session response",
    )?;
    if root["format"].as_str() != Some(SESSION_RESPONSE_FORMAT)
        || root["protocol"].as_str() != Some(SESSION_PROTOCOL)
        || root["request_id"].as_str() != Some(request_id)
        || &root["model"] != model_receipt
        || &root["packages"] != packages_receipt
        || &root["runtime"] != runtime_receipt
    {
        return Err(boundary("session response identity changed or mismatched"));
    }
    validate_model(&root["model"], model)?;
    let request = object(&root["request"], "session response.request")?;
    exact_keys(
        request,
        &[
            "operation",
            "prompt_token_count",
            "prompt_token_ids_sha256",
            "expected_prefix",
            "evaluated_prompt_token_count",
            "evaluated_prompt_token_ids_sha256",
            "max_tokens",
            "stop_on_eos",
            "greedy_strategy",
            "requested_eos_token_id",
            "effective_eos_token_ids",
            "binding",
        ],
        "session response.request",
    )?;
    let evaluated = &full_prompt[prefix.len()..];
    let expected_prefix = serde_json::json!({
        "format": SESSION_PREFIX_FORMAT,
        "token_count": prefix.len(),
        "token_ids_sha256": token_ids_sha256(prefix)?,
    });
    if request["operation"].as_str() != Some(operation)
        || request["prompt_token_count"].as_u64() != Some(full_prompt.len() as u64)
        || request["prompt_token_ids_sha256"].as_str()
            != Some(token_ids_sha256(full_prompt)?.as_str())
        || request["expected_prefix"] != expected_prefix
        || request["evaluated_prompt_token_count"].as_u64() != Some(evaluated.len() as u64)
        || request["evaluated_prompt_token_ids_sha256"].as_str()
            != Some(token_ids_sha256(evaluated)?.as_str())
        || request["binding"] != session_binding(model)
    {
        return Err(boundary("session request receipt differs from request"));
    }

    let synthetic = serde_json::json!({
        "format": RESPONSE_FORMAT,
        "protocol": PROTOCOL,
        "request_id": request_id,
        "request": {
            "prompt_token_count": evaluated.len(),
            "prompt_token_ids_sha256": token_ids_sha256(evaluated)?,
            "max_tokens": request["max_tokens"],
            "stop_on_eos": request["stop_on_eos"],
            "greedy_strategy": request["greedy_strategy"],
            "requested_eos_token_id": request["requested_eos_token_id"],
            "effective_eos_token_ids": request["effective_eos_token_ids"],
        },
        "model": root["model"],
        "packages": root["packages"],
        "runtime": root["runtime"],
        "metrics": root["metrics"],
        "generation": root["generation"],
    });
    let mut generation = validate_response(
        &synthetic,
        request_id,
        model,
        model_receipt,
        packages_receipt,
        runtime_receipt,
        evaluated,
        max_tokens,
        eos,
        stop_on_eos,
    )?;
    generation.receipt = value.clone();

    let mut committed = full_prompt.to_vec();
    committed.extend_from_slice(&generation.generated_token_ids);
    let session = object(&root["session"], "session response.session")?;
    exact_keys(
        session,
        &[
            "session_id",
            "prefix_token_count",
            "prefix_token_ids_sha256",
            "reused_prefix_token_count",
            "evaluated_prompt_token_count",
            "cache_bytes",
        ],
        "session response.session",
    )?;
    let cache_bytes = session["cache_bytes"]
        .as_u64()
        .filter(|bytes| *bytes <= MAX_SESSION_CACHE_BYTES)
        .ok_or_else(|| boundary("committed session cache bytes are invalid"))?;
    if session["session_id"].as_str() != Some(session_id)
        || session["prefix_token_count"].as_u64() != Some(committed.len() as u64)
        || session["prefix_token_ids_sha256"].as_str()
            != Some(token_ids_sha256(&committed)?.as_str())
        || session["reused_prefix_token_count"].as_u64() != Some(prefix.len() as u64)
        || session["evaluated_prompt_token_count"].as_u64() != Some(evaluated.len() as u64)
    {
        return Err(boundary("committed session prefix receipt is invalid"));
    }
    let evicted =
        validate_session_cache_state(&root["session_cache"], session_cache_receipt, true)?;
    let cache_state = object(&root["session_cache"], "session response.session_cache")?;
    if cache_state["session_count"].as_u64() == Some(0)
        || cache_state["total_cache_bytes"]
            .as_u64()
            .is_none_or(|total| total < cache_bytes)
    {
        return Err(boundary(
            "committed session is absent from cache accounting",
        ));
    }
    Ok((generation, evicted))
}

fn validate_session_reset_response(
    value: &Value,
    request_id: &str,
    session_id: &str,
    prefix: &[u32],
    model: &ModelIdentity,
    session_cache_receipt: &Value,
) -> Result<(), MlxServiceError> {
    let root = object(value, "session reset response")?;
    if root.get("format").and_then(Value::as_str) == Some(SESSION_RESPONSE_ERROR_FORMAT) {
        return Err(session_worker_error(root, request_id)?);
    }
    exact_keys(
        root,
        &[
            "format",
            "protocol",
            "request_id",
            "session_id",
            "released_cache_bytes",
            "previous_prefix",
            "binding",
            "session_cache",
        ],
        "session reset response",
    )?;
    let expected_prefix = serde_json::json!({
        "format": SESSION_PREFIX_FORMAT,
        "token_count": prefix.len(),
        "token_ids_sha256": token_ids_sha256(prefix)?,
    });
    if root["format"].as_str() != Some(SESSION_RESET_FORMAT)
        || root["protocol"].as_str() != Some(SESSION_PROTOCOL)
        || root["request_id"].as_str() != Some(request_id)
        || root["session_id"].as_str() != Some(session_id)
        || root["previous_prefix"] != expected_prefix
        || root["binding"] != session_binding(model)
        || root["released_cache_bytes"]
            .as_u64()
            .is_none_or(|bytes| bytes > MAX_SESSION_CACHE_BYTES)
    {
        return Err(boundary("session reset receipt is invalid"));
    }
    validate_session_cache_state(&root["session_cache"], session_cache_receipt, false)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_response(
    value: &Value,
    request_id: &str,
    model: &ModelIdentity,
    model_receipt: &Value,
    packages_receipt: &Value,
    runtime_receipt: &Value,
    prompt: &[u32],
    max_tokens: usize,
    eos: Option<u32>,
    stop_on_eos: bool,
) -> Result<MlxServiceGeneration, MlxServiceError> {
    let root = object(value, "response")?;
    if root.get("format").and_then(Value::as_str) == Some(RESPONSE_ERROR_FORMAT) {
        exact_keys(
            root,
            &["format", "protocol", "request_id", "error"],
            "error response",
        )?;
        if root["protocol"].as_str() != Some(PROTOCOL)
            || root["request_id"].as_str() != Some(request_id)
        {
            return Err(boundary("error response identity mismatch"));
        }
        let error = object(&root["error"], "response.error")?;
        exact_keys(error, &["code", "message"], "response.error")?;
        let code = clean_string(&error["code"], 64, "error code")?;
        if !safe_error_code(code) {
            return Err(boundary("error response code is unsafe"));
        }
        let message = clean_string(&error["message"], 1024, "error message")?;
        return Err(MlxServiceError::Worker {
            request_id: request_id.into(),
            code: code.into(),
            message: message.into(),
        });
    }
    exact_keys(
        root,
        &[
            "format",
            "protocol",
            "request_id",
            "request",
            "model",
            "packages",
            "runtime",
            "metrics",
            "generation",
        ],
        "response",
    )?;
    if root["format"].as_str() != Some(RESPONSE_FORMAT)
        || root["protocol"].as_str() != Some(PROTOCOL)
        || root["request_id"].as_str() != Some(request_id)
        || &root["model"] != model_receipt
        || &root["packages"] != packages_receipt
        || &root["runtime"] != runtime_receipt
    {
        return Err(boundary("response identity changed or mismatched"));
    }
    validate_model(&root["model"], model)?;
    let request = object(&root["request"], "response.request")?;
    exact_keys(
        request,
        &[
            "prompt_token_count",
            "prompt_token_ids_sha256",
            "max_tokens",
            "stop_on_eos",
            "greedy_strategy",
            "requested_eos_token_id",
            "effective_eos_token_ids",
        ],
        "response.request",
    )?;
    let prompt_hash =
        sha256(&serde_json::to_vec(prompt).map_err(|_| invalid("cannot hash prompt token IDs"))?);
    let eos_matches = match eos {
        Some(expected) => token(&request["requested_eos_token_id"], "requested EOS ID")
            .is_ok_and(|observed| observed == expected),
        None => request["requested_eos_token_id"].is_null(),
    };
    if request["prompt_token_count"].as_u64() != Some(prompt.len() as u64)
        || request["prompt_token_ids_sha256"].as_str() != Some(prompt_hash.as_str())
        || request["max_tokens"].as_u64() != Some(max_tokens as u64)
        || request["stop_on_eos"].as_bool() != Some(stop_on_eos)
        || request["greedy_strategy"].as_str() != Some(GREEDY_STRATEGY)
        || !eos_matches
    {
        return Err(boundary("response request receipt differs from request"));
    }
    let effective: Vec<u32> = request["effective_eos_token_ids"]
        .as_array()
        .ok_or_else(|| boundary("effective EOS IDs must be an array"))?
        .iter()
        .map(|value| token(value, "effective EOS ID"))
        .collect::<Result<_, _>>()?;
    if effective.windows(2).any(|window| window[0] >= window[1]) {
        return Err(boundary("effective EOS IDs must be sorted and unique"));
    }
    if eos.is_some_and(|value| effective != [value]) {
        return Err(boundary("requested EOS does not bind effective EOS"));
    }
    let generation = object(&root["generation"], "response.generation")?;
    exact_keys(
        generation,
        &[
            "generated_token_ids",
            "generated_token_count",
            "stop_reason",
        ],
        "response.generation",
    )?;
    let generated: Vec<u32> = generation["generated_token_ids"]
        .as_array()
        .ok_or_else(|| boundary("generated token IDs must be an array"))?
        .iter()
        .map(|value| token(value, "generated token ID"))
        .collect::<Result<_, _>>()?;
    if generated.len() > max_tokens
        || (max_tokens > 0 && generated.is_empty())
        || generation["generated_token_count"].as_u64() != Some(generated.len() as u64)
    {
        return Err(boundary("generated token count is inconsistent"));
    }
    match generation["stop_reason"].as_str() {
        Some("length") if generated.len() == max_tokens => {
            if stop_on_eos && generated.iter().any(|value| effective.contains(value)) {
                return Err(boundary("length stop contains EOS"));
            }
        }
        Some("eos")
            if stop_on_eos
                && generated
                    .last()
                    .is_some_and(|value| effective.contains(value))
                && !generated[..generated.len() - 1]
                    .iter()
                    .any(|value| effective.contains(value)) => {}
        _ => return Err(boundary("generation stop rule is inconsistent")),
    }
    let observed = object(&root["metrics"], "response.metrics")?;
    exact_keys(
        observed,
        &[
            "request_ms",
            "ttft_ms",
            "tpot_ms",
            "tps",
            "timed_decode_tokens",
            "mlx_peak_memory_bytes",
        ],
        "response.metrics",
    )?;
    let metrics = MlxServiceMetrics {
        request_ms: finite(&observed["request_ms"], "request_ms")?,
        ttft_ms: finite(&observed["ttft_ms"], "ttft_ms")?,
        tpot_ms: finite(&observed["tpot_ms"], "tpot_ms")?,
        tps: finite(&observed["tps"], "tps")?,
        timed_decode_tokens: observed["timed_decode_tokens"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| boundary("timed_decode_tokens must be an integer"))?,
        mlx_peak_memory_bytes: observed["mlx_peak_memory_bytes"]
            .as_u64()
            .ok_or_else(|| boundary("peak memory must be an integer"))?,
    };
    if metrics.timed_decode_tokens != generated.len().saturating_sub(1)
        || (metrics.timed_decode_tokens == 0 && (metrics.tpot_ms != 0.0 || metrics.tps != 0.0))
        || (metrics.timed_decode_tokens > 0 && (metrics.tpot_ms <= 0.0 || metrics.tps <= 0.0))
        || (generated.is_empty() && metrics.ttft_ms != 0.0)
    {
        return Err(boundary("response timing metrics are inconsistent"));
    }
    Ok(MlxServiceGeneration {
        generated_token_ids: generated,
        metrics,
        receipt: value.clone(),
    })
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, MlxServiceError> {
    value
        .as_object()
        .ok_or_else(|| boundary(format!("{label} must be an object")))
}

fn exact_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    label: &str,
) -> Result<(), MlxServiceError> {
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        return Err(boundary(format!("{label} keys do not match the contract")));
    }
    Ok(())
}

fn safe_string(value: &str, maximum: usize) -> bool {
    !value.is_empty()
        && value.trim() == value
        && value.chars().count() <= maximum
        && !value.chars().any(char::is_control)
}

fn safe_error_code(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

fn clean_string<'a>(
    value: &'a Value,
    maximum: usize,
    label: &str,
) -> Result<&'a str, MlxServiceError> {
    value
        .as_str()
        .filter(|value| safe_string(value, maximum))
        .ok_or_else(|| boundary(format!("{label} is unsafe")))
}

fn finite(value: &Value, label: &str) -> Result<f64, MlxServiceError> {
    value
        .as_f64()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| boundary(format!("{label} must be finite and non-negative")))
}

fn token(value: &Value, label: &str) -> Result<u32, MlxServiceError> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value <= MAX_TOKEN_ID)
        .ok_or_else(|| boundary(format!("{label} is out of range")))
}

struct JsonCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl JsonCursor<'_> {
    fn whitespace(&mut self) {
        while self
            .bytes
            .get(self.offset)
            .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
        {
            self.offset += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> Result<(), ()> {
        self.whitespace();
        if self.bytes.get(self.offset) != Some(&expected) {
            return Err(());
        }
        self.offset += 1;
        Ok(())
    }

    fn string(&mut self) -> Result<String, ()> {
        self.whitespace();
        let start = self.offset;
        self.consume(b'"')?;
        let mut escaped = false;
        while let Some(&byte) = self.bytes.get(self.offset) {
            self.offset += 1;
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                return serde_json::from_slice(&self.bytes[start..self.offset]).map_err(|_| ());
            }
        }
        Err(())
    }

    fn value(&mut self) -> Result<(), ()> {
        self.whitespace();
        match self.bytes.get(self.offset) {
            Some(b'{') => self.map(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(|_| ()),
            Some(_) => {
                while self.bytes.get(self.offset).is_some_and(|byte| {
                    !matches!(byte, b',' | b']' | b'}' | b' ' | b'\t' | b'\n' | b'\r')
                }) {
                    self.offset += 1;
                }
                Ok(())
            }
            None => Err(()),
        }
    }

    fn map(&mut self) -> Result<(), ()> {
        self.consume(b'{')?;
        self.whitespace();
        if self.bytes.get(self.offset) == Some(&b'}') {
            self.offset += 1;
            return Ok(());
        }
        let mut keys = HashSet::new();
        loop {
            let key = self.string()?;
            if !keys.insert(key) {
                return Err(());
            }
            self.consume(b':')?;
            self.value()?;
            self.whitespace();
            match self.bytes.get(self.offset) {
                Some(b',') => self.offset += 1,
                Some(b'}') => {
                    self.offset += 1;
                    return Ok(());
                }
                _ => return Err(()),
            }
        }
    }

    fn array(&mut self) -> Result<(), ()> {
        self.consume(b'[')?;
        self.whitespace();
        if self.bytes.get(self.offset) == Some(&b']') {
            self.offset += 1;
            return Ok(());
        }
        loop {
            self.value()?;
            self.whitespace();
            match self.bytes.get(self.offset) {
                Some(b',') => self.offset += 1,
                Some(b']') => {
                    self.offset += 1;
                    return Ok(());
                }
                _ => return Err(()),
            }
        }
    }
}

fn reject_duplicate_keys(bytes: &[u8]) -> Result<(), ()> {
    let mut cursor = JsonCursor { bytes, offset: 0 };
    cursor.value()?;
    cursor.whitespace();
    (cursor.offset == bytes.len()).then_some(()).ok_or(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        python: PathBuf,
        runner: PathBuf,
        model: PathBuf,
        marker: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "apxinf-mlx-service-{}-{serial}",
                std::process::id()
            ));
            fs::create_dir(&root).unwrap();
            let python = root.join("fake-python");
            let runner = root.join("apxinf_mlx_serve.py");
            let helper = root.join("apxinf_mlx_generate.py");
            let model = root.join("model");
            let marker = root.join("loads.txt");
            fs::create_dir(&model).unwrap();
            fs::write(model.join("config.json"), r#"{"model_type":"qwen3_5"}"#).unwrap();
            fs::write(helper, "# fake generation helper\n").unwrap();
            fs::write(
                &python,
                "#!/bin/sh\nexec /usr/bin/python3 \"$1\" --fake-python \"$0\" \"$2\" \"$3\"\n",
            )
            .unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
            }
            let marker_literal = serde_json::to_string(marker.to_str().unwrap()).unwrap();
            fs::write(&runner, FAKE_RUNNER.replace("__MARKER__", &marker_literal)).unwrap();
            Self {
                root,
                python,
                runner,
                model,
                marker,
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    const FAKE_RUNNER: &str = r#"import hashlib,json,pathlib,subprocess,sys,time
args=sys.argv[1:]
fake_python=pathlib.Path(args[args.index('--fake-python')+1]).resolve()
model=pathlib.Path(args[args.index('--model-dir')+1]).resolve()
runner=pathlib.Path(__file__).resolve()
helper=runner.with_name('apxinf_mlx_generate.py')
marker=pathlib.Path(__MARKER__)
marker.write_text((marker.read_text() if marker.exists() else '')+'load\n')
digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
packages={'huggingface-hub':'1.28.0','mlx':'0.32.1','mlx-lm':'0.31.3','mlx-metal':'0.32.1','numpy':'2.5.2','safetensors':'0.8.0','tokenizers':'0.22.2','transformers':'5.15.1'}
runtime={'policy':'trusted-local-offline-environment-v1','offline_environment':True,'os_network_sandbox':False,'trust_remote_code':False,'python_version':'3.14.3','python':{'path':str(fake_python),'sha256':digest(fake_python)},'runner':{'path':str(runner),'sha256':digest(runner)},'generation_helper':{'path':str(helper),'sha256':digest(helper)}}
config=model/'config.json'; config_hash=digest(config); manifest=hashlib.sha256(b'apxinf-local-bundle-manifest-v1\0'); manifest.update(b'config.json\0'+str(config.stat().st_size).encode('ascii')+b'\0'+config_hash.encode('ascii')+b'\n')
model_id={'model_dir':str(model),'model_type':'qwen3_5','quantization':None,'config_sha256':config_hash,'bundle':{'format':'apxinf-local-bundle-manifest-v1','file_count':1,'total_bytes':config.stat().st_size,'sha256':manifest.hexdigest()}}
emit=lambda value:(sys.stdout.write(json.dumps(value,separators=(',',':'),sort_keys=True)+'\n'),sys.stdout.flush())
session_ready={'format':'apxinf-mlx-session-cache-ready-v1','protocol':'apxinf-mlx-session-v1','policy':'exact-append-only-in-process-lru-v1','request_format':'apxinf-mlx-session-request-v1','control_format':'apxinf-mlx-session-control-v1','max_sessions':4,'max_bytes':536870912}
binding={'format':'apxinf-mlx-session-binding-v1','model_config_sha256':config_hash,'model_bundle_sha256':model_id['bundle']['sha256'],'greedy_strategy':'mlx-generate-step-argmax-v1','cache_policy':'exact-append-only-in-process-lru-v1'}
token_hash=lambda tokens:hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode('ascii')).hexdigest()
emit({'format':'apxinf-mlx-service-ready-v1','protocol':'apxinf-mlx-service-v1','greedy_strategy':'mlx-generate-step-argmax-v1','model':model_id,'packages':packages,'runtime':runtime,'limits':{'max_line_bytes':1048576,'max_output_bytes':4194304,'max_prompt_tokens':131072,'max_generated_tokens':65536,'max_requests':1000000},'metrics':{'load_ms':10.0},'session_cache':session_ready})
sessions={}
for line in sys.stdin:
 request=json.loads(line); request_id=request['request_id']
 if (runner.parent/'hang-mode').exists():
  escaped=runner.parent/'descendant-escaped'
  subprocess.Popen(['/bin/sh','-c','sleep 1; : > "$1"','sh',str(escaped)])
  time.sleep(30)
 if request['format']=='apxinf-mlx-service-control-v1':
  emit({'format':'apxinf-mlx-service-shutdown-v1','protocol':'apxinf-mlx-service-v1','request_id':request_id}); raise SystemExit(0)
 if request['format']=='apxinf-mlx-session-control-v1':
  previous=sessions.pop(request['session_id'])
  emit({'format':'apxinf-mlx-session-reset-v1','protocol':'apxinf-mlx-session-v1','request_id':request_id,'session_id':request['session_id'],'released_cache_bytes':64,'previous_prefix':{'format':'apxinf-mlx-session-prefix-v1','token_count':len(previous),'token_ids_sha256':token_hash(previous)},'binding':binding,'session_cache':{'policy':'exact-append-only-in-process-lru-v1','session_count':len(sessions),'total_cache_bytes':64*len(sessions),'max_sessions':4,'max_bytes':536870912}}); continue
 if 667 in request['prompt_token_ids']:
  emit({'format':'apxinf-mlx-service-response-error-v1','protocol':'apxinf-mlx-service-v1','request_id':request_id,'error':{'code':'impossible_worker_code','message':'injected unknown worker error'}}); continue
 maximum=request['max_tokens']; generated=[] if maximum==0 else [7,9][:maximum]
 effective=[request['eos_token_id']] if 'eos_token_id' in request else [9]
 stop='eos' if request['stop_on_eos'] and generated and generated[-1] in effective else 'length'
 if request['format']=='apxinf-mlx-session-request-v1':
  full=request['prompt_token_ids']; previous=sessions.get(request['session_id'],[]); evaluated=full[len(previous):]
  sessions[request['session_id']]=full+generated; committed=sessions[request['session_id']]
  emit({'format':'apxinf-mlx-session-response-v1','protocol':'apxinf-mlx-session-v1','request_id':request_id,'request':{'operation':request['operation'],'prompt_token_count':len(full),'prompt_token_ids_sha256':token_hash(full),'expected_prefix':request['expected_prefix'],'evaluated_prompt_token_count':len(evaluated),'evaluated_prompt_token_ids_sha256':token_hash(evaluated),'max_tokens':maximum,'stop_on_eos':request['stop_on_eos'],'greedy_strategy':'mlx-generate-step-argmax-v1','requested_eos_token_id':request.get('eos_token_id'),'effective_eos_token_ids':effective,'binding':binding},'session':{'session_id':request['session_id'],'prefix_token_count':len(committed),'prefix_token_ids_sha256':token_hash(committed),'reused_prefix_token_count':len(previous),'evaluated_prompt_token_count':len(evaluated),'cache_bytes':64},'session_cache':{'policy':'exact-append-only-in-process-lru-v1','session_count':len(sessions),'total_cache_bytes':64*len(sessions),'max_sessions':4,'max_bytes':536870912,'evicted_session_ids':[]},'model':model_id,'packages':packages,'runtime':runtime,'metrics':{'request_ms':2.0,'ttft_ms':1.0,'tpot_ms':1.0 if len(generated)>1 else 0.0,'tps':1000.0 if len(generated)>1 else 0.0,'timed_decode_tokens':max(0,len(generated)-1),'mlx_peak_memory_bytes':1234},'generation':{'generated_token_ids':generated,'generated_token_count':len(generated),'stop_reason':stop}}); continue
 prompt_bytes=json.dumps(request['prompt_token_ids'],separators=(',',':')).encode('ascii'); timed=max(0,len(generated)-1)
 emit({'format':'apxinf-mlx-service-response-v1','protocol':'apxinf-mlx-service-v1','request_id':request_id,'request':{'prompt_token_count':len(request['prompt_token_ids']),'prompt_token_ids_sha256':hashlib.sha256(prompt_bytes).hexdigest(),'max_tokens':maximum,'stop_on_eos':request['stop_on_eos'],'greedy_strategy':'mlx-generate-step-argmax-v1','requested_eos_token_id':request.get('eos_token_id'),'effective_eos_token_ids':effective},'model':model_id,'packages':packages,'runtime':runtime,'metrics':{'request_ms':2.0 if maximum else 0.0,'ttft_ms':1.0 if generated else 0.0,'tpot_ms':1.0 if timed else 0.0,'tps':1000.0 if timed else 0.0,'timed_decode_tokens':timed,'mlx_peak_memory_bytes':1234},'generation':{'generated_token_ids':generated,'generated_token_count':len(generated),'stop_reason':stop}})
"#;

    #[test]
    fn one_process_serves_multiple_requests_and_shuts_down() {
        let fixture = Fixture::new();
        let mut service = MlxService::start(
            &fixture.python,
            &fixture.runner,
            &fixture.model,
            Duration::from_secs(5),
        )
        .unwrap();
        let first = service.generate(&[1, 2], 2, Some(9), true).unwrap();
        assert_eq!(first.generated_token_ids, vec![7, 9]);
        let second = service.generate(&[3], 0, None, true).unwrap();
        assert!(second.generated_token_ids.is_empty());
        service.shutdown().unwrap();
        assert_eq!(fs::read_to_string(&fixture.marker).unwrap(), "load\n");
    }

    #[test]
    fn explicit_session_requires_exact_append_and_resets() {
        let fixture = Fixture::new();
        let mut service = MlxService::start(
            &fixture.python,
            &fixture.runner,
            &fixture.model,
            Duration::from_secs(5),
        )
        .unwrap();
        let first = service
            .generate_session("chat-1", &[1, 2], 2, Some(9), true)
            .unwrap();
        assert_eq!(first.generated_token_ids, vec![7, 9]);
        let committed = [1, 2, 7, 9];
        let fork = service
            .generate_session("chat-1", &[1, 2, 7, 8, 3], 1, None, false)
            .unwrap_err();
        assert!(fork.to_string().contains("exact non-empty append"));
        let second = service
            .generate_session("chat-1", &[1, 2, 7, 9, 3], 1, None, false)
            .unwrap();
        assert_eq!(second.generated_token_ids, vec![7]);
        assert_eq!(
            second.receipt["session"]["reused_prefix_token_count"],
            committed.len()
        );
        service.reset_session("chat-1").unwrap();
        assert!(service.reset_session("chat-1").is_err());
        service.shutdown().unwrap();
    }

    #[test]
    fn rejects_duplicate_json_keys_at_the_process_boundary() {
        let error =
            parse_line(b"{\"request_id\":\"a\",\"request_id\":\"b\"}\n", "response").unwrap_err();
        assert!(error.to_string().contains("duplicate object keys"));
    }

    #[test]
    fn request_timeout_kills_the_service_process_group() {
        let fixture = Fixture::new();
        fs::write(fixture.root.join("hang-mode"), "yes\n").unwrap();
        let mut service = MlxService::start(
            &fixture.python,
            &fixture.runner,
            &fixture.model,
            Duration::from_secs(5),
        )
        .unwrap();
        service.timeout = Duration::from_millis(500);
        let error = service.generate(&[1], 1, None, false).unwrap_err();
        assert!(error.to_string().contains("deadline"));
        thread::sleep(Duration::from_millis(1200));
        assert!(!fixture.root.join("descendant-escaped").exists());
    }

    #[test]
    fn unknown_worker_error_closes_the_direct_service_api() {
        let fixture = Fixture::new();
        let mut service = MlxService::start(
            &fixture.python,
            &fixture.runner,
            &fixture.model,
            Duration::from_secs(5),
        )
        .unwrap();
        let error = service.generate(&[667], 1, None, false).unwrap_err();
        assert!(matches!(
            error,
            MlxServiceError::Worker { ref code, .. } if code == "impossible_worker_code"
        ));
        let closed = service.generate(&[1], 1, None, false).unwrap_err();
        assert!(closed.to_string().contains("closed or exhausted"));
    }

    #[test]
    #[ignore = "requires the pinned local MLX runtime and a certified bundle"]
    fn locked_real_service_serves_two_exact_requests() {
        let python = PathBuf::from(std::env::var("APXINF_REAL_MLX_PYTHON").unwrap());
        let runner = PathBuf::from(std::env::var("APXINF_REAL_MLX_RUNNER").unwrap());
        let model = PathBuf::from(std::env::var("APXINF_REAL_MLX_MODEL").unwrap());
        let prompt = [
            248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let expected = [9419, 0, 2500, 628, 353, 1438, 488, 3242, 30, 25677];
        let mut service =
            MlxService::start(&python, &runner, &model, Duration::from_secs(120)).unwrap();
        let first = service.generate(&prompt, 10, None, true).unwrap();
        let second = service.generate(&prompt, 10, None, true).unwrap();
        assert_eq!(first.generated_token_ids, expected);
        assert_eq!(second.generated_token_ids, expected);
        assert_ne!(first.receipt["request_id"], second.receipt["request_id"]);
        service.shutdown().unwrap();
    }

    #[test]
    #[ignore = "requires the pinned local MLX runtime and a certified bundle"]
    fn locked_real_session_append_matches_fresh_full_prompt() {
        let python = PathBuf::from(std::env::var("APXINF_REAL_MLX_PYTHON").unwrap());
        let runner = PathBuf::from(std::env::var("APXINF_REAL_MLX_RUNNER").unwrap());
        let model = PathBuf::from(std::env::var("APXINF_REAL_MLX_MODEL").unwrap());
        let canonical = [
            248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let second_turn = [
            248046, 198, 248045, 846, 198, 22791, 13, 248046, 198, 248045, 74455, 198, 248068, 271,
            248069, 271,
        ];
        let expected_first = [9419, 0, 2500, 628, 353, 1438, 488, 3242, 30];
        let mut service =
            MlxService::start(&python, &runner, &model, Duration::from_secs(120)).unwrap();
        let first = service
            .generate_session("canonical-two-turn", &canonical, 9, None, true)
            .unwrap();
        assert_eq!(first.generated_token_ids, expected_first);
        let mut full_prompt = canonical.to_vec();
        full_prompt.extend_from_slice(&first.generated_token_ids);
        full_prompt.extend_from_slice(&second_turn);
        let fresh = service.generate(&full_prompt, 10, None, true).unwrap();
        let reused = service
            .generate_session("canonical-two-turn", &full_prompt, 10, None, true)
            .unwrap();
        assert_eq!(reused.generated_token_ids, fresh.generated_token_ids);
        assert_eq!(
            reused.receipt["session"]["reused_prefix_token_count"],
            canonical.len() + expected_first.len()
        );
        assert_eq!(
            reused.receipt["session"]["evaluated_prompt_token_count"],
            second_turn.len()
        );
        service.reset_session("canonical-two-turn").unwrap();
        service.shutdown().unwrap();
    }
}

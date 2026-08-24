//! Strict one-shot process boundary for the offline MLX-LM worker.
//!
//! The Python interpreter and worker script are explicit trusted inputs.  The
//! worker still runs with a cleared environment, bounded output pipes, and a
//! fail-closed JSON-lines contract so ambient developer state cannot silently
//! change an inference result.

use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

const REQUEST_FORMAT: &str = "apxinf-mlx-generation-request-v1";
const RECEIPT_FORMAT: &str = "apxinf-mlx-generation-receipt-v1";
const ERROR_FORMAT: &str = "apxinf-mlx-generation-error-v1";
const GREEDY_STRATEGY: &str = "mlx-generate-step-argmax-v1";
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_CONFIG_BYTES: usize = 2 * 1024 * 1024;
const MAX_PYTHON_BYTES: usize = 128 * 1024 * 1024;
const MAX_RUNNER_BYTES: usize = 4 * 1024 * 1024;
const MAX_PROMPT_TOKENS: usize = 131_072;
const MAX_GENERATED_TOKENS: usize = 65_536;
const MAX_TOKEN_ID: u32 = i32::MAX as u32;
const WORKER_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const PINNED_PYTHON_VERSION: &str = "3.14.3";
const PINNED_PACKAGE_VERSIONS: [(&str, &str); 8] = [
    ("huggingface-hub", "1.28.0"),
    ("mlx", "0.32.1"),
    ("mlx-lm", "0.31.3"),
    ("mlx-metal", "0.32.1"),
    ("numpy", "2.5.2"),
    ("safetensors", "0.8.0"),
    ("tokenizers", "0.22.2"),
    ("transformers", "5.15.1"),
];

fn sha256_hex(payload: &[u8]) -> String {
    format!("{:x}", Sha256::digest(payload))
}

fn token_ids_sha256(token_ids: &[u32]) -> Result<String, MlxProviderError> {
    let encoded = serde_json::to_vec(token_ids)
        .map_err(|_| invalid_input("prompt token IDs cannot be encoded as JSON"))?;
    Ok(sha256_hex(&encoded))
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MlxMetrics {
    pub(crate) load_ms: f64,
    pub(crate) ttft_ms: f64,
    pub(crate) tpot_ms: f64,
    pub(crate) tps: f64,
    pub(crate) timed_decode_tokens: usize,
    pub(crate) mlx_peak_memory_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MlxGeneration {
    pub(crate) generated_token_ids: Vec<u32>,
    pub(crate) metrics: MlxMetrics,
    pub(crate) receipt: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalModelIdentity {
    model_dir: PathBuf,
    model_type: String,
    config_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalProgramIdentity {
    path: PathBuf,
    sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalRuntimeIdentity {
    python: LocalProgramIdentity,
    runner: LocalProgramIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MlxProviderError {
    InvalidInput(String),
    Launch(String),
    Boundary(String),
    Worker {
        code: String,
        message: String,
        exit_code: Option<i32>,
    },
}

impl fmt::Display for MlxProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid MLX request: {message}"),
            Self::Launch(message) => write!(formatter, "cannot launch MLX worker: {message}"),
            Self::Boundary(message) => write!(formatter, "invalid MLX worker response: {message}"),
            Self::Worker {
                code,
                message,
                exit_code,
            } => match exit_code {
                Some(exit_code) => write!(
                    formatter,
                    "MLX worker failed ({code}, exit {exit_code}): {message}"
                ),
                None => write!(formatter, "MLX worker failed ({code}): {message}"),
            },
        }
    }
}

impl std::error::Error for MlxProviderError {}

/// Generate token IDs through one fresh, offline MLX-LM worker process.
///
/// All three paths must be absolute. `python` and `runner` must name direct
/// regular, non-symlink files, and `python` must be executable. `model_dir`
/// must name a direct non-symlink directory. The returned receipt is retained
/// verbatim as a parsed JSON value after strict validation.
pub(crate) fn generate_with_mlx(
    python: &Path,
    runner: &Path,
    model_dir: &Path,
    prompt_token_ids: &[u32],
    max_tokens: usize,
    eos_token_id: Option<u32>,
    stop_on_eos: bool,
) -> Result<MlxGeneration, MlxProviderError> {
    generate_with_mlx_timeout(
        python,
        runner,
        model_dir,
        prompt_token_ids,
        max_tokens,
        eos_token_id,
        stop_on_eos,
        WORKER_TIMEOUT,
    )
}

fn generate_with_mlx_timeout(
    python: &Path,
    runner: &Path,
    model_dir: &Path,
    prompt_token_ids: &[u32],
    max_tokens: usize,
    eos_token_id: Option<u32>,
    stop_on_eos: bool,
    timeout: Duration,
) -> Result<MlxGeneration, MlxProviderError> {
    if timeout.is_zero() {
        return Err(invalid_input("worker timeout must be positive"));
    }
    let local_runtime = LocalRuntimeIdentity {
        python: read_local_program_identity(python, "python interpreter", true, MAX_PYTHON_BYTES)?,
        runner: read_local_program_identity(runner, "MLX runner", false, MAX_RUNNER_BYTES)?,
    };
    let local_model = read_local_model_identity(model_dir)?;
    let request = build_request(prompt_token_ids, max_tokens, eos_token_id, stop_on_eos)?;

    let mut command = Command::new(&local_runtime.python.path);
    command
        .arg(&local_runtime.runner.path)
        .arg("--model-dir")
        .arg(&local_model.model_dir)
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

    let (status, stdout, stderr) = run_bounded(command, request, timeout)?;
    if status.success() {
        if !stderr.is_empty() {
            return Err(boundary("successful worker wrote to stderr"));
        }
        let receipt = parse_single_json_line(&stdout, "stdout")?;
        validate_receipt(
            &receipt,
            &local_model,
            &local_runtime,
            prompt_token_ids,
            max_tokens,
            eos_token_id,
            stop_on_eos,
        )
    } else {
        if !stdout.is_empty() {
            return Err(boundary("failed worker wrote to stdout"));
        }
        let value = parse_single_json_line(&stderr, "stderr")?;
        Err(validate_worker_error(&value, status.code())?)
    }
}

fn invalid_input(message: impl Into<String>) -> MlxProviderError {
    MlxProviderError::InvalidInput(message.into())
}

fn boundary(message: impl Into<String>) -> MlxProviderError {
    MlxProviderError::Boundary(message.into())
}

fn validate_regular_file(
    path: &Path,
    label: &str,
    require_executable: bool,
) -> Result<fs::Metadata, MlxProviderError> {
    if !path.is_absolute() {
        return Err(invalid_input(format!("{label} path must be absolute")));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid_input(format!("cannot inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(invalid_input(format!(
            "{label} must be a direct regular non-symlink file"
        )));
    }
    if require_executable && !is_executable(&metadata) {
        return Err(invalid_input(format!("{label} is not executable")));
    }
    Ok(metadata)
}

fn read_local_program_identity(
    path: &Path,
    label: &str,
    require_executable: bool,
    max_bytes: usize,
) -> Result<LocalProgramIdentity, MlxProviderError> {
    let selected_metadata = validate_regular_file(path, label, require_executable)?;
    let canonical = path
        .canonicalize()
        .map_err(|error| invalid_input(format!("cannot resolve {label}: {error}")))?;
    if canonical.to_str().is_none() {
        return Err(invalid_input(format!("{label} path must be UTF-8")));
    }
    let path_metadata = fs::symlink_metadata(&canonical)
        .map_err(|error| invalid_input(format!("cannot inspect resolved {label}: {error}")))?;
    if path_metadata.file_type().is_symlink() || !path_metadata.file_type().is_file() {
        return Err(invalid_input(format!(
            "resolved {label} must be a direct regular non-symlink file"
        )));
    }
    if !same_file_identity(&selected_metadata, &path_metadata) {
        return Err(invalid_input(format!(
            "{label} changed while it was being resolved"
        )));
    }
    if path_metadata.len() > max_bytes as u64 {
        return Err(invalid_input(format!("{label} exceeds {max_bytes} bytes")));
    }

    let mut file = fs::File::open(&canonical)
        .map_err(|error| invalid_input(format!("cannot open {label}: {error}")))?;
    let opened_metadata = file
        .metadata()
        .map_err(|error| invalid_input(format!("cannot inspect open {label}: {error}")))?;
    if !same_file_identity(&path_metadata, &opened_metadata) {
        return Err(invalid_input(format!(
            "{label} changed while it was being opened"
        )));
    }
    let mut payload = Vec::with_capacity(opened_metadata.len() as usize);
    Read::by_ref(&mut file)
        .take(max_bytes as u64 + 1)
        .read_to_end(&mut payload)
        .map_err(|error| invalid_input(format!("cannot read {label}: {error}")))?;
    if payload.len() > max_bytes || payload.len() as u64 != opened_metadata.len() {
        return Err(invalid_input(format!("{label} changed while it was read")));
    }

    Ok(LocalProgramIdentity {
        path: canonical,
        sha256: sha256_hex(&payload),
    })
}

#[cfg(unix)]
fn is_executable(metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;

    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn is_executable(_metadata: &fs::Metadata) -> bool {
    true
}

fn validate_model_dir(path: &Path) -> Result<PathBuf, MlxProviderError> {
    if !path.is_absolute() {
        return Err(invalid_input("model directory path must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid_input(format!("cannot inspect model directory: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(invalid_input(
            "model directory must be a direct non-symlink directory",
        ));
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| invalid_input(format!("cannot resolve model directory: {error}")))?;
    if canonical.to_str().is_none() {
        return Err(invalid_input("model directory path must be UTF-8"));
    }
    Ok(canonical)
}

fn read_local_model_identity(path: &Path) -> Result<LocalModelIdentity, MlxProviderError> {
    let model_dir = validate_model_dir(path)?;
    let config_path = model_dir.join("config.json");
    let path_metadata = fs::symlink_metadata(&config_path)
        .map_err(|error| invalid_input(format!("cannot inspect model config: {error}")))?;
    if path_metadata.file_type().is_symlink() || !path_metadata.file_type().is_file() {
        return Err(invalid_input(
            "model config must be a direct regular non-symlink file",
        ));
    }
    if path_metadata.len() > MAX_CONFIG_BYTES as u64 {
        return Err(invalid_input(format!(
            "model config exceeds {MAX_CONFIG_BYTES} bytes"
        )));
    }

    let mut file = fs::File::open(&config_path)
        .map_err(|error| invalid_input(format!("cannot open model config: {error}")))?;
    let opened_metadata = file
        .metadata()
        .map_err(|error| invalid_input(format!("cannot inspect open model config: {error}")))?;
    if !same_file_identity(&path_metadata, &opened_metadata) {
        return Err(invalid_input(
            "model config changed while it was being opened",
        ));
    }
    let mut payload = Vec::with_capacity(opened_metadata.len() as usize);
    Read::by_ref(&mut file)
        .take(MAX_CONFIG_BYTES as u64 + 1)
        .read_to_end(&mut payload)
        .map_err(|error| invalid_input(format!("cannot read model config: {error}")))?;
    if payload.len() > MAX_CONFIG_BYTES || payload.len() as u64 != opened_metadata.len() {
        return Err(invalid_input("model config changed while it was read"));
    }
    reject_duplicate_object_keys(&payload)
        .map_err(|_| invalid_input("model config contains duplicate JSON object keys"))?;
    let config: Value = serde_json::from_slice(&payload)
        .map_err(|_| invalid_input("model config is not valid JSON"))?;
    let root = config
        .as_object()
        .ok_or_else(|| invalid_input("model config root must be an object"))?;
    let model_type = root
        .get("model_type")
        .and_then(Value::as_str)
        .filter(|value| bounded_clean_string(value, 128))
        .ok_or_else(|| invalid_input("model config model_type is invalid"))?
        .to_string();

    Ok(LocalModelIdentity {
        model_dir,
        model_type,
        config_sha256: sha256_hex(&payload),
    })
}

#[cfg(unix)]
fn same_file_identity(first: &fs::Metadata, second: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    first.dev() == second.dev()
        && first.ino() == second.ino()
        && first.len() == second.len()
        && first.mode() == second.mode()
}

#[cfg(not(unix))]
fn same_file_identity(first: &fs::Metadata, second: &fs::Metadata) -> bool {
    first.is_file() == second.is_file() && first.len() == second.len()
}

fn validate_token(token: u32, label: &str) -> Result<(), MlxProviderError> {
    if token > MAX_TOKEN_ID {
        return Err(invalid_input(format!(
            "{label} must be at most {MAX_TOKEN_ID}"
        )));
    }
    Ok(())
}

fn build_request(
    prompt_token_ids: &[u32],
    max_tokens: usize,
    eos_token_id: Option<u32>,
    stop_on_eos: bool,
) -> Result<Vec<u8>, MlxProviderError> {
    if prompt_token_ids.is_empty() || prompt_token_ids.len() > MAX_PROMPT_TOKENS {
        return Err(invalid_input(format!(
            "prompt token count must be in 1..={MAX_PROMPT_TOKENS}"
        )));
    }
    for (index, &token) in prompt_token_ids.iter().enumerate() {
        validate_token(token, &format!("prompt token {index}"))?;
    }
    if max_tokens > MAX_GENERATED_TOKENS {
        return Err(invalid_input(format!(
            "max_tokens must be in 0..={MAX_GENERATED_TOKENS}"
        )));
    }
    if let Some(token) = eos_token_id {
        validate_token(token, "EOS token")?;
    }

    let mut value = Map::new();
    value.insert(
        "format".to_string(),
        Value::String(REQUEST_FORMAT.to_string()),
    );
    value.insert(
        "prompt_token_ids".to_string(),
        Value::Array(
            prompt_token_ids
                .iter()
                .map(|&token| Value::from(token))
                .collect(),
        ),
    );
    value.insert("max_tokens".to_string(), Value::from(max_tokens));
    value.insert("stop_on_eos".to_string(), Value::from(stop_on_eos));
    if let Some(token) = eos_token_id {
        value.insert("eos_token_id".to_string(), Value::from(token));
    }
    let mut payload = serde_json::to_vec(&Value::Object(value))
        .map_err(|_| invalid_input("request cannot be encoded as JSON"))?;
    payload.push(b'\n');
    if payload.len() > MAX_REQUEST_BYTES {
        return Err(invalid_input(format!(
            "encoded request exceeds {MAX_REQUEST_BYTES} bytes"
        )));
    }
    Ok(payload)
}

fn bounded_reader<R>(
    mut reader: R,
    label: &'static str,
    failure_tx: mpsc::Sender<String>,
) -> thread::JoinHandle<Result<Vec<u8>, String>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut payload = Vec::new();
        let mut buffer = [0_u8; 8192];
        loop {
            let count = match reader.read(&mut buffer) {
                Ok(count) => count,
                Err(_) => {
                    let message = format!("cannot read worker {label}");
                    let _ = failure_tx.send(message.clone());
                    return Err(message);
                }
            };
            if count == 0 {
                return Ok(payload);
            }
            if payload.len().saturating_add(count) > MAX_OUTPUT_BYTES {
                let message = format!("worker {label} exceeded {MAX_OUTPUT_BYTES} bytes");
                let _ = failure_tx.send(message.clone());
                return Err(message);
            }
            payload.extend_from_slice(&buffer[..count]);
        }
    })
}

#[cfg(unix)]
fn configure_worker_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_worker_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn kill_worker_process_group(child: &mut std::process::Child) {
    unsafe extern "C" {
        fn kill(process_id: i32, signal: i32) -> i32;
    }

    const SIGKILL: i32 = 9;
    if let Ok(process_group) = i32::try_from(child.id()) {
        // The worker is placed in a fresh process group whose ID is its PID.
        // A negative target kills ordinary descendants that inherited our
        // stdout/stderr pipes; killing only the Python parent can leave those
        // pipes open forever and defeat the request deadline.
        unsafe {
            let _ = kill(-process_group, SIGKILL);
        }
    }
    let _ = child.kill();
}

#[cfg(not(unix))]
fn kill_worker_process_group(child: &mut std::process::Child) {
    let _ = child.kill();
}

fn join_with_deadline<T>(
    handle: thread::JoinHandle<Result<T, String>>,
    label: &str,
    deadline: Instant,
) -> Result<Result<T, String>, MlxProviderError> {
    while !handle.is_finished() {
        if Instant::now() >= deadline {
            return Err(boundary(format!(
                "worker {label} did not close after process termination"
            )));
        }
        thread::sleep(Duration::from_millis(5));
    }
    handle
        .join()
        .map_err(|_| boundary(format!("worker {label} task panicked")))
}

fn run_bounded(
    mut command: Command,
    request: Vec<u8>,
    timeout: Duration,
) -> Result<(ExitStatus, Vec<u8>, Vec<u8>), MlxProviderError> {
    configure_worker_process_group(&mut command);
    let mut child = command
        .spawn()
        .map_err(|error| MlxProviderError::Launch(error.to_string()))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| MlxProviderError::Launch("stdin was not captured".to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| MlxProviderError::Launch("stdout was not captured".to_string()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| MlxProviderError::Launch("stderr was not captured".to_string()))?;

    let (failure_tx, failure_rx) = mpsc::channel();
    let stdout_reader = bounded_reader(stdout, "stdout", failure_tx.clone());
    let stderr_reader = bounded_reader(stderr, "stderr", failure_tx);
    let stdin_writer = thread::spawn(move || -> Result<(), String> {
        stdin
            .write_all(&request)
            .and_then(|()| stdin.flush())
            .map_err(|_| "cannot write the complete worker request".to_string())
    });

    let started = Instant::now();
    let mut capture_failure = None;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                kill_worker_process_group(&mut child);
                break status;
            }
            Ok(None) => {}
            Err(_) => {
                kill_worker_process_group(&mut child);
                let _ = child.wait();
                return Err(boundary("cannot wait for worker process"));
            }
        }
        if started.elapsed() >= timeout {
            capture_failure = Some("worker exceeded its fixed runtime limit".to_string());
            kill_worker_process_group(&mut child);
            break child
                .wait()
                .map_err(|_| boundary("cannot reap worker after timeout"))?;
        }
        match failure_rx.recv_timeout(Duration::from_millis(10)) {
            Ok(message) => {
                capture_failure = Some(message);
                kill_worker_process_group(&mut child);
                break child
                    .wait()
                    .map_err(|_| boundary("cannot reap worker after output violation"))?;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                thread::sleep(Duration::from_millis(10));
            }
        }
    };

    let cleanup_deadline = Instant::now() + Duration::from_secs(2);
    let write_result = join_with_deadline(stdin_writer, "stdin writer", cleanup_deadline)?;
    let stdout = join_with_deadline(stdout_reader, "stdout reader", cleanup_deadline)?;
    let stderr = join_with_deadline(stderr_reader, "stderr reader", cleanup_deadline)?;
    if let Some(message) = capture_failure {
        return Err(boundary(message));
    }
    let stdout = stdout.map_err(boundary)?;
    let stderr = stderr.map_err(boundary)?;
    if status.success() {
        write_result.map_err(boundary)?;
    }
    Ok((status, stdout, stderr))
}

fn parse_single_json_line(payload: &[u8], label: &str) -> Result<Value, MlxProviderError> {
    if payload.is_empty()
        || payload.len() > MAX_OUTPUT_BYTES
        || payload.last() != Some(&b'\n')
        || payload[..payload.len() - 1]
            .iter()
            .any(|byte| matches!(byte, b'\n' | b'\r'))
    {
        return Err(boundary(format!(
            "{label} must contain exactly one newline-terminated JSON line"
        )));
    }
    let body = &payload[..payload.len() - 1];
    if body.is_empty() {
        return Err(boundary(format!("{label} JSON line must not be empty")));
    }
    let value: Value =
        serde_json::from_slice(body).map_err(|_| boundary(format!("{label} is not valid JSON")))?;
    reject_duplicate_object_keys(body)
        .map_err(|_| boundary(format!("{label} contains duplicate JSON object keys")))?;
    Ok(value)
}

fn exact_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    label: &str,
) -> Result<(), MlxProviderError> {
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        return Err(boundary(format!("{label} keys do not match the contract")));
    }
    Ok(())
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, MlxProviderError> {
    value
        .as_object()
        .ok_or_else(|| boundary(format!("{label} must be an object")))
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a str, MlxProviderError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| boundary(format!("{label}.{key} must be a string")))
}

fn bounded_clean_string(value: &str, max_chars: usize) -> bool {
    !value.is_empty()
        && value.trim() == value
        && value.chars().count() <= max_chars
        && !value.chars().any(char::is_control)
}

fn exact_usize(value: &Value, label: &str) -> Result<usize, MlxProviderError> {
    let number = value
        .as_u64()
        .ok_or_else(|| boundary(format!("{label} must be an unsigned integer")))?;
    usize::try_from(number).map_err(|_| boundary(format!("{label} is too large")))
}

fn token_id_from_value(value: &Value, label: &str) -> Result<u32, MlxProviderError> {
    let number = value
        .as_u64()
        .ok_or_else(|| boundary(format!("{label} must be an unsigned integer")))?;
    let token = u32::try_from(number).map_err(|_| boundary(format!("{label} is too large")))?;
    if token > MAX_TOKEN_ID {
        return Err(boundary(format!("{label} exceeds {MAX_TOKEN_ID}")));
    }
    Ok(token)
}

fn finite_nonnegative(value: &Value, label: &str) -> Result<f64, MlxProviderError> {
    let number = value
        .as_f64()
        .ok_or_else(|| boundary(format!("{label} must be a number")))?;
    if !number.is_finite() || number < 0.0 {
        return Err(boundary(format!("{label} must be finite and non-negative")));
    }
    Ok(number)
}

fn validate_receipt(
    receipt: &Value,
    local_model: &LocalModelIdentity,
    local_runtime: &LocalRuntimeIdentity,
    prompt_token_ids: &[u32],
    max_tokens: usize,
    eos_token_id: Option<u32>,
    stop_on_eos: bool,
) -> Result<MlxGeneration, MlxProviderError> {
    let root = object(receipt, "receipt")?;
    exact_keys(
        root,
        &[
            "format",
            "request",
            "model",
            "packages",
            "runtime",
            "metrics",
            "generation",
        ],
        "receipt",
    )?;
    if root.get("format").and_then(Value::as_str) != Some(RECEIPT_FORMAT) {
        return Err(boundary("receipt.format does not match the contract"));
    }

    let request = object(&root["request"], "receipt.request")?;
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
        "receipt.request",
    )?;
    if exact_usize(
        &request["prompt_token_count"],
        "receipt.request.prompt_token_count",
    )? != prompt_token_ids.len()
        || exact_usize(&request["max_tokens"], "receipt.request.max_tokens")? != max_tokens
        || request["stop_on_eos"].as_bool() != Some(stop_on_eos)
    {
        return Err(boundary(
            "receipt.request differs from the submitted request",
        ));
    }
    if request["greedy_strategy"].as_str() != Some(GREEDY_STRATEGY) {
        return Err(boundary(
            "receipt greedy strategy differs from the contract",
        ));
    }
    let prompt_hash = required_string(request, "prompt_token_ids_sha256", "receipt.request")?;
    if prompt_hash != token_ids_sha256(prompt_token_ids)? {
        return Err(boundary(
            "receipt prompt token hash differs from the submitted request",
        ));
    }
    match eos_token_id {
        Some(expected) => {
            if token_id_from_value(
                &request["requested_eos_token_id"],
                "receipt.request.requested_eos_token_id",
            )? != expected
            {
                return Err(boundary("receipt requested EOS token differs"));
            }
        }
        None if !request["requested_eos_token_id"].is_null() => {
            return Err(boundary(
                "receipt unexpectedly reports a requested EOS token",
            ));
        }
        None => {}
    }
    let effective_values = request["effective_eos_token_ids"]
        .as_array()
        .ok_or_else(|| boundary("receipt.request.effective_eos_token_ids must be an array"))?;
    let mut effective_eos = Vec::with_capacity(effective_values.len());
    for (index, value) in effective_values.iter().enumerate() {
        effective_eos.push(token_id_from_value(
            value,
            &format!("receipt.request.effective_eos_token_ids[{index}]"),
        )?);
    }
    if !effective_eos.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(boundary(
            "receipt effective EOS tokens must be unique and sorted",
        ));
    }
    if let Some(expected) = eos_token_id {
        if effective_eos != [expected] {
            return Err(boundary("receipt effective EOS token differs"));
        }
    }
    if stop_on_eos && max_tokens > 0 && effective_eos.is_empty() {
        return Err(boundary("EOS stopping requires an effective EOS token"));
    }

    let model = object(&root["model"], "receipt.model")?;
    exact_keys(
        model,
        &["model_dir", "model_type", "quantization", "config_sha256"],
        "receipt.model",
    )?;
    if model["model_dir"].as_str() != local_model.model_dir.to_str() {
        return Err(boundary(
            "receipt model path differs from the requested model",
        ));
    }
    let model_type = required_string(model, "model_type", "receipt.model")?;
    if !bounded_clean_string(model_type, 128) || model_type != local_model.model_type.as_str() {
        return Err(boundary(
            "receipt.model.model_type differs from the local model config",
        ));
    }
    if !matches!(model["quantization"], Value::Null | Value::Object(_)) {
        return Err(boundary(
            "receipt.model.quantization must be null or an object",
        ));
    }
    let config_sha256 = required_string(model, "config_sha256", "receipt.model")?;
    if config_sha256.len() != 64
        || !config_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(boundary("receipt.model.config_sha256 is invalid"));
    }
    if config_sha256 != local_model.config_sha256 {
        return Err(boundary(
            "receipt.model.config_sha256 differs from the local model config",
        ));
    }

    let packages = object(&root["packages"], "receipt.packages")?;
    exact_keys(
        packages,
        &[
            "huggingface-hub",
            "mlx",
            "mlx-lm",
            "mlx-metal",
            "numpy",
            "safetensors",
            "tokenizers",
            "transformers",
        ],
        "receipt.packages",
    )?;
    for (name, expected_version) in PINNED_PACKAGE_VERSIONS {
        let version = required_string(packages, name, "receipt.packages")?;
        if !bounded_clean_string(version, 128) || version != expected_version {
            return Err(boundary(format!(
                "receipt package {name} does not match the pinned version"
            )));
        }
    }

    let runtime = object(&root["runtime"], "receipt.runtime")?;
    exact_keys(
        runtime,
        &[
            "offline",
            "trust_remote_code",
            "python_version",
            "python_executable",
            "python_executable_sha256",
            "runner",
            "runner_sha256",
        ],
        "receipt.runtime",
    )?;
    if runtime["offline"].as_bool() != Some(true)
        || runtime["trust_remote_code"].as_bool() != Some(false)
    {
        return Err(boundary("receipt runtime policy is unsafe"));
    }
    if required_string(runtime, "python_version", "receipt.runtime")? != PINNED_PYTHON_VERSION {
        return Err(boundary(
            "receipt Python version does not match the pinned version",
        ));
    }
    if runtime["python_executable"].as_str() != local_runtime.python.path.to_str()
        || runtime["python_executable_sha256"].as_str()
            != Some(local_runtime.python.sha256.as_str())
        || runtime["runner"].as_str() != local_runtime.runner.path.to_str()
        || runtime["runner_sha256"].as_str() != Some(local_runtime.runner.sha256.as_str())
    {
        return Err(boundary(
            "receipt runtime files differ from the independently inspected files",
        ));
    }

    let generation = object(&root["generation"], "receipt.generation")?;
    exact_keys(
        generation,
        &[
            "generated_token_ids",
            "generated_token_count",
            "stop_reason",
        ],
        "receipt.generation",
    )?;
    let generated_values = generation["generated_token_ids"]
        .as_array()
        .ok_or_else(|| boundary("receipt generated_token_ids must be an array"))?;
    if generated_values.len() > max_tokens || (max_tokens > 0 && generated_values.is_empty()) {
        return Err(boundary("receipt generated token count is out of range"));
    }
    let mut generated_token_ids = Vec::with_capacity(generated_values.len());
    for (index, value) in generated_values.iter().enumerate() {
        generated_token_ids.push(token_id_from_value(
            value,
            &format!("receipt.generation.generated_token_ids[{index}]"),
        )?);
    }
    if exact_usize(
        &generation["generated_token_count"],
        "receipt.generation.generated_token_count",
    )? != generated_token_ids.len()
    {
        return Err(boundary("receipt generated token count is inconsistent"));
    }
    let stop_reason = required_string(generation, "stop_reason", "receipt.generation")?;
    match stop_reason {
        "length" => {
            if generated_token_ids.len() != max_tokens {
                return Err(boundary("length stop did not produce max_tokens"));
            }
            if stop_on_eos
                && generated_token_ids
                    .iter()
                    .any(|token| effective_eos.contains(token))
            {
                return Err(boundary("length stop contains an effective EOS token"));
            }
        }
        "eos" => {
            if !stop_on_eos
                || !generated_token_ids
                    .last()
                    .is_some_and(|token| effective_eos.contains(token))
                || generated_token_ids[..generated_token_ids.len() - 1]
                    .iter()
                    .any(|token| effective_eos.contains(token))
            {
                return Err(boundary("EOS stop rule is inconsistent"));
            }
        }
        _ => return Err(boundary("receipt stop_reason is invalid")),
    }

    let metrics_object = object(&root["metrics"], "receipt.metrics")?;
    exact_keys(
        metrics_object,
        &[
            "load_ms",
            "ttft_ms",
            "tpot_ms",
            "tps",
            "timed_decode_tokens",
            "mlx_peak_memory_bytes",
        ],
        "receipt.metrics",
    )?;
    let timed_decode_tokens = exact_usize(
        &metrics_object["timed_decode_tokens"],
        "receipt.metrics.timed_decode_tokens",
    )?;
    if timed_decode_tokens != generated_token_ids.len().saturating_sub(1) {
        return Err(boundary("receipt timed decode token count is inconsistent"));
    }
    let metrics = MlxMetrics {
        load_ms: finite_nonnegative(&metrics_object["load_ms"], "receipt.metrics.load_ms")?,
        ttft_ms: finite_nonnegative(&metrics_object["ttft_ms"], "receipt.metrics.ttft_ms")?,
        tpot_ms: finite_nonnegative(&metrics_object["tpot_ms"], "receipt.metrics.tpot_ms")?,
        tps: finite_nonnegative(&metrics_object["tps"], "receipt.metrics.tps")?,
        timed_decode_tokens,
        mlx_peak_memory_bytes: metrics_object["mlx_peak_memory_bytes"]
            .as_u64()
            .ok_or_else(|| boundary("receipt.metrics.mlx_peak_memory_bytes must be an integer"))?,
    };
    if (timed_decode_tokens == 0 && (metrics.tpot_ms != 0.0 || metrics.tps != 0.0))
        || (timed_decode_tokens > 0 && (metrics.tpot_ms <= 0.0 || metrics.tps <= 0.0))
        || (generated_token_ids.is_empty() && metrics.ttft_ms != 0.0)
    {
        return Err(boundary("receipt decode timing metrics are inconsistent"));
    }

    Ok(MlxGeneration {
        generated_token_ids,
        metrics,
        receipt: receipt.clone(),
    })
}

fn validate_worker_error(
    value: &Value,
    exit_code: Option<i32>,
) -> Result<MlxProviderError, MlxProviderError> {
    let root = object(value, "worker error")?;
    exact_keys(root, &["format", "error"], "worker error")?;
    if root.get("format").and_then(Value::as_str) != Some(ERROR_FORMAT) {
        return Err(boundary("worker error format does not match the contract"));
    }
    let error = object(&root["error"], "worker error.error")?;
    exact_keys(error, &["code", "message"], "worker error.error")?;
    let code = required_string(error, "code", "worker error.error")?;
    if code.is_empty()
        || code.len() > 64
        || !code
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(boundary("worker error code is unsafe"));
    }
    let message = required_string(error, "message", "worker error.error")?;
    if !bounded_clean_string(message, 1024) {
        return Err(boundary("worker error message is unsafe"));
    }
    Ok(MlxProviderError::Worker {
        code: code.to_string(),
        message: message.to_string(),
        exit_code,
    })
}

struct JsonCursor<'a> {
    payload: &'a [u8],
    offset: usize,
}

impl<'a> JsonCursor<'a> {
    fn skip_whitespace(&mut self) {
        while self
            .payload
            .get(self.offset)
            .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
        {
            self.offset += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> Result<(), ()> {
        self.skip_whitespace();
        if self.payload.get(self.offset) != Some(&expected) {
            return Err(());
        }
        self.offset += 1;
        Ok(())
    }

    fn string(&mut self) -> Result<String, ()> {
        self.skip_whitespace();
        let start = self.offset;
        self.consume(b'"')?;
        let mut escaped = false;
        while let Some(&byte) = self.payload.get(self.offset) {
            self.offset += 1;
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                return serde_json::from_slice(&self.payload[start..self.offset]).map_err(|_| ());
            }
        }
        Err(())
    }

    fn value(&mut self) -> Result<(), ()> {
        self.skip_whitespace();
        match self.payload.get(self.offset) {
            Some(b'{') => self.map(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(|_| ()),
            Some(_) => {
                while self.payload.get(self.offset).is_some_and(|byte| {
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
        self.skip_whitespace();
        if self.payload.get(self.offset) == Some(&b'}') {
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
            self.skip_whitespace();
            match self.payload.get(self.offset) {
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
        self.skip_whitespace();
        if self.payload.get(self.offset) == Some(&b']') {
            self.offset += 1;
            return Ok(());
        }
        loop {
            self.value()?;
            self.skip_whitespace();
            match self.payload.get(self.offset) {
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

fn reject_duplicate_object_keys(payload: &[u8]) -> Result<(), ()> {
    let mut cursor = JsonCursor { payload, offset: 0 };
    cursor.value()?;
    cursor.skip_whitespace();
    if cursor.offset == payload.len() {
        Ok(())
    } else {
        Err(())
    }
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
    }

    impl Fixture {
        fn new(program_body: &str) -> Self {
            let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "apxinf-mlx-provider-{}-{serial}",
                std::process::id()
            ));
            fs::create_dir(&root).unwrap();
            let python = root.join("fake-python");
            let runner = root.join("fake-runner.py");
            let model = root.join("model");
            fs::create_dir(&model).unwrap();
            fs::write(model.join("config.json"), r#"{"model_type":"qwen3_5"}"#).unwrap();
            fs::write(&runner, "# fake runner\n").unwrap();
            fs::write(&python, format!("#!/bin/sh\n{program_body}\n")).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
            }
            Self {
                root,
                python,
                runner,
                model,
            }
        }

        fn set_program(&self, program_body: &str) {
            fs::write(&self.python, format!("#!/bin/sh\n{program_body}\n")).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&self.python, fs::Permissions::from_mode(0o700)).unwrap();
            }
        }

        fn run(
            &self,
            prompt: &[u32],
            max_tokens: usize,
            eos: Option<u32>,
            stop_on_eos: bool,
        ) -> Result<MlxGeneration, MlxProviderError> {
            generate_with_mlx(
                &self.python,
                &self.runner,
                &self.model,
                prompt,
                max_tokens,
                eos,
                stop_on_eos,
            )
        }

        fn run_with_timeout(
            &self,
            prompt: &[u32],
            max_tokens: usize,
            eos: Option<u32>,
            stop_on_eos: bool,
            timeout: Duration,
        ) -> Result<MlxGeneration, MlxProviderError> {
            generate_with_mlx_timeout(
                &self.python,
                &self.runner,
                &self.model,
                prompt,
                max_tokens,
                eos,
                stop_on_eos,
                timeout,
            )
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn shell_quote(value: &str) -> String {
        format!("'{}'", value.replace('\'', "'\\''"))
    }

    fn receipt(model: &Path, generated: &[u32], eos: Option<u32>, stop: bool) -> Value {
        let stop_reason = if stop && eos.is_some() && generated.last() == eos.as_ref() {
            "eos"
        } else {
            "length"
        };
        serde_json::json!({
            "format": RECEIPT_FORMAT,
            "request": {
                "prompt_token_count": 2,
                "prompt_token_ids_sha256": "def4fe4f74f38325f2f5e330fcb0e51d476035250b64f6158662026485f0e557",
                "max_tokens": if stop_reason == "eos" { 4 } else { generated.len() },
                "stop_on_eos": stop,
                "greedy_strategy": GREEDY_STRATEGY,
                "requested_eos_token_id": eos,
                "effective_eos_token_ids": eos.map_or_else(|| vec![99], |token| vec![token]),
            },
            "model": {
                "model_dir": model,
                "model_type": "qwen3_5",
                "quantization": null,
                "config_sha256": "298e8a955b83c4660c1740d6da3979855d73fd11dd2f1b2d71845df81600c7f0",
            },
            "packages": {
                "huggingface-hub": "1.28.0",
                "mlx": "0.32.1",
                "mlx-lm": "0.31.3",
                "mlx-metal": "0.32.1",
                "numpy": "2.5.2",
                "safetensors": "0.8.0",
                "tokenizers": "0.22.2",
                "transformers": "5.15.1",
            },
            "runtime": {
                "offline": true,
                "trust_remote_code": false,
                "python_version": "3.14.3",
                "python_executable": "__PYTHON_PATH__",
                "python_executable_sha256": "__PYTHON_SHA256__",
                "runner": "__RUNNER_PATH__",
                "runner_sha256": "__RUNNER_SHA256__",
            },
            "metrics": {
                "load_ms": 10.0,
                "ttft_ms": if generated.is_empty() { 0.0 } else { 2.0 },
                "tpot_ms": if generated.len() > 1 { 1.0 } else { 0.0 },
                "tps": if generated.len() > 1 { 1000.0 } else { 0.0 },
                "timed_decode_tokens": generated.len().saturating_sub(1),
                "mlx_peak_memory_bytes": 1234,
            },
            "generation": {
                "generated_token_ids": generated,
                "generated_token_count": generated.len(),
                "stop_reason": stop_reason,
            },
        })
    }

    fn emitted_payload_body(fixture: &Fixture, payload: &Value) -> String {
        let mut payload = payload.clone();
        payload["runtime"]["python_executable"] =
            Value::String(fixture.python.canonicalize().unwrap().display().to_string());
        payload["runtime"]["runner"] =
            Value::String(fixture.runner.canonicalize().unwrap().display().to_string());
        let line = serde_json::to_string(&payload)
            .unwrap()
            .replace('%', "%%")
            .replace("__PYTHON_SHA256__", "%s")
            .replace("__RUNNER_SHA256__", "%s");
        format!(
            r#"hash_file() {{
  if command -v shasum >/dev/null 2>&1; then
    output=$(shasum -a 256 "$1") || exit 94
  elif command -v sha256sum >/dev/null 2>&1; then
    output=$(sha256sum "$1") || exit 94
  else
    exit 94
  fi
  printf '%s' "${{output%% *}}"
}}
python_hash=$(hash_file "$0") || exit 94
runner_hash=$(hash_file "$1") || exit 94
printf {} "$python_hash" "$runner_hash""#,
            shell_quote(&(line + "\n"))
        )
    }

    fn set_emitted_payload(fixture: &Fixture, payload: &Value) {
        fixture.set_program(&format!(
            "IFS= read -r request || exit 91\n{}",
            emitted_payload_body(fixture, payload)
        ));
    }

    fn emitting_fixture(mut payload: Value) -> Fixture {
        let fixture = Fixture::new("exit 1");
        payload["model"]["model_dir"] =
            Value::String(fixture.model.canonicalize().unwrap().display().to_string());
        set_emitted_payload(&fixture, &payload);
        fixture
    }

    #[test]
    fn accepts_strict_success_and_clears_ambient_environment() {
        let fixture = Fixture::new("exit 1");
        let mut payload = receipt(Path::new("/placeholder"), &[11, 12], None, true);
        payload["model"]["model_dir"] =
            Value::String(fixture.model.canonicalize().unwrap().display().to_string());
        let prelude = r#"
IFS= read -r request || exit 91
if [ "${HOME+x}" = x ] || [ "${SSH_AUTH_SOCK+x}" = x ]; then
  printf '%s\n' '{"format":"apxinf-mlx-generation-error-v1","error":{"code":"ambient_environment","message":"ambient environment leaked"}}' >&2
  exit 2
fi
if [ "$HF_HUB_OFFLINE" != 1 ] || [ "$TRANSFORMERS_OFFLINE" != 1 ] || [ "$HF_DATASETS_OFFLINE" != 1 ] || [ "$HF_HUB_DISABLE_TELEMETRY" != 1 ] || [ "$PYTHONNOUSERSITE" != 1 ]; then
  printf '%s\n' '{"format":"apxinf-mlx-generation-error-v1","error":{"code":"policy_environment","message":"fixed environment missing"}}' >&2
  exit 2
fi
case "$request" in
  *'"format":"apxinf-mlx-generation-request-v1"'*'"max_tokens":2'*'"prompt_token_ids":[7,8]'*'"stop_on_eos":true'*) ;;
  *) exit 92 ;;
esac
case "$request" in *'"eos_token_id"'*) exit 93 ;; esac
"#;
        fixture.set_program(&format!(
            "{prelude}\n{}",
            emitted_payload_body(&fixture, &payload)
        ));
        let result = fixture.run(&[7, 8], 2, None, true).unwrap();
        assert_eq!(result.generated_token_ids, [11, 12]);
        assert_eq!(result.metrics.timed_decode_tokens, 1);
        assert_eq!(result.receipt["runtime"]["offline"], true);
    }

    #[test]
    fn accepts_eos_termination() {
        let value = receipt(Path::new("/placeholder"), &[11, 0], Some(0), true);
        let fixture = emitting_fixture(value);
        let result = fixture.run(&[7, 8], 4, Some(0), true).unwrap();
        assert_eq!(result.generated_token_ids, [11, 0]);
    }

    #[test]
    fn accepts_zero_token_budget() {
        let mut value = receipt(Path::new("/placeholder"), &[], None, true);
        value["request"]["effective_eos_token_ids"] = serde_json::json!([]);
        let fixture = emitting_fixture(value);
        let result = fixture.run(&[7, 8], 0, None, true).unwrap();
        assert!(result.generated_token_ids.is_empty());
        assert_eq!(result.metrics.timed_decode_tokens, 0);
        assert_eq!(result.metrics.ttft_ms, 0.0);
        assert_eq!(result.metrics.tpot_ms, 0.0);
        assert_eq!(result.metrics.tps, 0.0);
    }

    #[test]
    fn returns_only_validated_worker_failure() {
        let fixture = Fixture::new("
IFS= read -r request || exit 91
printf '%s\\n' '{\"format\":\"apxinf-mlx-generation-error-v1\",\"error\":{\"code\":\"model_load_failed\",\"message\":\"model load failed\"}}' >&2
exit 2
");
        let error = fixture.run(&[7, 8], 2, None, true).unwrap_err();
        assert_eq!(
            error,
            MlxProviderError::Worker {
                code: "model_load_failed".to_string(),
                message: "model load failed".to_string(),
                exit_code: Some(2),
            }
        );
    }

    #[test]
    fn rejects_root_schema_drift() {
        let mut value = receipt(Path::new("/placeholder"), &[11, 12], None, true);
        value["unexpected"] = Value::Bool(true);
        let fixture = emitting_fixture(value);
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
    }

    #[test]
    fn rejects_unpinned_package_version() {
        let mut value = receipt(Path::new("/placeholder"), &[11, 12], None, true);
        value["packages"]["mlx"] = Value::String("9.9.9".into());
        let fixture = emitting_fixture(value);
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
    }

    #[test]
    fn rejects_runtime_file_identity_drift() {
        let mut value = receipt(Path::new("/placeholder"), &[11, 12], None, true);
        value["runtime"]["runner_sha256"] = Value::String("1".repeat(64));
        let fixture = emitting_fixture(value);
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
    }

    #[test]
    fn rejects_model_path_drift() {
        let fixture = Fixture::new("exit 1");
        let value = receipt(Path::new("/different/model"), &[11, 12], None, true);
        set_emitted_payload(&fixture, &value);
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
    }

    #[test]
    fn rejects_model_config_content_drift() {
        let mut value = receipt(Path::new("/placeholder"), &[11, 12], None, true);
        value["model"]["config_sha256"] = Value::String("1".repeat(64));
        let fixture = emitting_fixture(value);
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
    }

    #[test]
    fn rejects_model_type_drift() {
        let mut value = receipt(Path::new("/placeholder"), &[11, 12], None, true);
        value["model"]["model_type"] = Value::String("another_model".into());
        let fixture = emitting_fixture(value);
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
    }

    #[test]
    fn rejects_generation_count_drift() {
        let mut value = receipt(Path::new("/placeholder"), &[11, 12], None, true);
        value["generation"]["generated_token_count"] = Value::from(1);
        let fixture = emitting_fixture(value);
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
    }

    #[test]
    fn rejects_request_field_drift() {
        let mut value = receipt(Path::new("/placeholder"), &[11, 12], None, true);
        value["request"]["max_tokens"] = Value::from(3);
        let fixture = emitting_fixture(value);
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
    }

    #[test]
    fn rejects_prompt_content_drift_with_the_same_token_count() {
        let mut value = receipt(Path::new("/placeholder"), &[11, 12], None, true);
        value["request"]["prompt_token_ids_sha256"] = Value::String(
            "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4".into(),
        );
        let fixture = emitting_fixture(value);
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
    }

    #[test]
    fn rejects_duplicate_json_keys() {
        let fixture = Fixture::new("
IFS= read -r request || exit 91
printf '%s\\n' '{\"format\":\"apxinf-mlx-generation-receipt-v1\",\"format\":\"apxinf-mlx-generation-receipt-v1\"}'
");
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
    }

    #[test]
    fn kills_worker_when_stdout_exceeds_bound() {
        let fixture = Fixture::new("
IFS= read -r request || exit 91
chunk=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
while :; do printf '%s' \"$chunk\"; done
");
        let started = std::time::Instant::now();
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn kills_worker_when_stderr_exceeds_bound() {
        let fixture = Fixture::new("
IFS= read -r request || exit 91
chunk=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
while :; do printf '%s' \"$chunk\" >&2; done
");
        let started = std::time::Instant::now();
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::Boundary(_))
        ));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn kills_worker_at_fixed_deadline() {
        let fixture = Fixture::new(
            "
IFS= read -r request || exit 91
while :; do :; done
",
        );
        let started = std::time::Instant::now();
        let error = fixture
            .run_with_timeout(&[7, 8], 2, None, true, Duration::from_millis(100))
            .unwrap_err();
        assert!(matches!(error, MlxProviderError::Boundary(_)));
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    #[cfg(unix)]
    fn kills_descendants_that_inherit_worker_output_pipes() {
        let fixture = Fixture::new(
            "
IFS= read -r request || exit 91
sleep 5 &
exit 7
",
        );
        let started = std::time::Instant::now();
        let error = fixture
            .run_with_timeout(&[7, 8], 2, None, true, Duration::from_secs(3))
            .unwrap_err();
        match error {
            MlxProviderError::Boundary(message) => {
                assert!(!message.contains("did not close after process termination"));
            }
            other => panic!("unexpected error: {other}"),
        }
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    #[cfg(unix)]
    fn rejects_symlink_runner_and_non_executable_python() {
        use std::os::unix::fs::PermissionsExt;

        let fixture = Fixture::new("exit 1");
        fs::set_permissions(&fixture.python, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(matches!(
            fixture.run(&[7, 8], 2, None, true),
            Err(MlxProviderError::InvalidInput(_))
        ));
        use std::os::unix::fs::symlink;
        fs::set_permissions(&fixture.python, fs::Permissions::from_mode(0o700)).unwrap();
        let symlink_runner = fixture.root.join("runner-link.py");
        symlink(&fixture.runner, &symlink_runner).unwrap();
        assert!(matches!(
            generate_with_mlx(
                &fixture.python,
                &symlink_runner,
                &fixture.model,
                &[7, 8],
                2,
                None,
                true,
            ),
            Err(MlxProviderError::InvalidInput(_))
        ));
    }
}

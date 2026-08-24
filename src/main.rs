//! ApxInf LLM inference engine CLI.

mod mlx_provider;
mod mlx_service;
mod mlx_service_cli;

use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus, Stdio};
use std::time::Duration;

use apxinf_core::{DType, Device, Tensor};
use apxinf_model::{AutoModel, GenerationProfile, ImageInput, LlmInput, LoadOptions};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum GenerateProvider {
    #[default]
    Native,
    Mlx,
}

#[derive(Parser)]
#[command(name = "apxinf")]
#[command(about = "LLM inference engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt
    Generate {
        /// Path to HuggingFace model directory (contains model.safetensors and tokenizer.json)
        #[arg(short, long)]
        model: PathBuf,

        /// Input prompt (treated as user message in chat mode, or raw text if no chat template)
        #[arg(short, long)]
        prompt: String,

        /// Path to an image file (for Qwen3-VL multimodal). When set, the
        /// image is preprocessed by a Python helper and fed alongside the
        /// prompt. Only for qwen3_vl models.
        #[arg(long)]
        image: Option<PathBuf>,

        /// Maximum new tokens to generate
        #[arg(long, default_value = "50")]
        max_tokens: usize,

        /// Maximum context allocated for model caches. The checkpoint may
        /// advertise a much larger limit than a local Mac can safely reserve.
        #[arg(long, default_value = "4096")]
        max_context: usize,

        /// Disable EOS-based early stopping (generate until max_tokens)
        #[arg(long)]
        no_eos_stop: bool,

        /// Print generated token IDs after the decoded text (useful for
        /// reproducible model-oracle and tokenizer checks).
        #[arg(long)]
        show_token_ids: bool,

        /// Emit one machine-readable JSON object on stdout. Progress and
        /// streamed text are suppressed; errors remain on stderr.
        #[arg(long)]
        json: bool,

        /// System prompt for chat mode
        #[arg(long)]
        system: Option<String>,

        /// Device to run inference on (cpu or cuda)
        #[arg(short, long, default_value = "cpu")]
        device: String,

        /// Weight dtype ("fp32" or "bf16"). On CUDA, "bf16" halves weight-
        /// bandwidth and enables the bf16 fast path. Ordinary CPU inference
        /// ignores this value; either Metal W8 lane requires explicit "fp32".
        #[arg(long, default_value = "fp32")]
        dtype: String,

        /// Inference implementation. `native` uses ApxInf's in-process model;
        /// `mlx` uses the trusted local-only MLX-LM worker boundary on Apple Silicon.
        #[arg(long, value_enum, default_value_t = GenerateProvider::Native)]
        provider: GenerateProvider,

        /// Direct executable from the isolated MLX Python environment. Required
        /// with `--provider mlx`; symlinks are rejected by the process boundary.
        #[arg(long, value_name = "PATH")]
        mlx_python: Option<PathBuf>,

        /// Direct path to ApxInf's trusted MLX worker script. Required with
        /// `--provider mlx`.
        #[arg(long, value_name = "PATH")]
        mlx_runner: Option<PathBuf>,

        /// Use the feature-gated Metal W8 tied lm_head for the first generated
        /// token and cached decode. The CPU/Accelerate model body is unchanged;
        /// unsupported builds or checkpoints fail instead of silently falling
        /// back.
        #[arg(long)]
        metal_w8_lm_head: bool,

        /// Use complete Metal W8 MLP blocks for every Qwen3.5 decode layer.
        /// Prefill, attention, residuals, and state remain CPU/F32. This is an
        /// explicit experimental path and can be combined with
        /// `--metal-w8-lm-head`; unsupported requests fail closed.
        #[arg(long)]
        metal_w8_mlp_block: bool,
    },

    /// Run the deterministic Hugging Face -> macOS onboarding controller.
    ///
    /// The Python controller is an explicit trust boundary because `cargo
    /// install` installs this binary but not ApxInf's Python assets. In a
    /// source checkout, pass `--controller scripts/onboard_hf_macos.py`; for
    /// an installed binary, point to the same file in a trusted controller
    /// bundle. Captured output is bounded, but the explicitly selected
    /// controller remains trusted for termination and filesystem effects.
    Onboard(OnboardArgs),

    /// Start one validated local MLX model and serve strict JSONL on stdin/stdout.
    MlxServe {
        /// Existing local MLX model bundle. Network downloads are never attempted.
        #[arg(long, value_name = "PATH")]
        model: PathBuf,

        /// Direct executable from the pinned MLX Python environment.
        #[arg(long, value_name = "PATH")]
        mlx_python: PathBuf,

        /// Direct path to ApxInf's trusted MLX service runner.
        #[arg(long, value_name = "PATH")]
        mlx_runner: PathBuf,

        /// Per-handshake and per-request service deadline.
        #[arg(long, default_value_t = 120)]
        timeout_seconds: u64,
    },

    /// Run a quick test of the engine
    Test,
}

#[derive(Args)]
struct OnboardArgs {
    /// Canonical Hugging Face model URL accepted by the controller
    #[arg(value_name = "HF_URL")]
    model_url: String,

    /// Explicit path to the trusted ApxInf Python controller
    #[arg(long, value_name = "PATH")]
    controller: PathBuf,

    /// Python interpreter used to launch the controller
    #[arg(long, value_name = "PATH")]
    python: PathBuf,

    /// Exact Hugging Face commit
    #[arg(long)]
    revision: String,

    /// Existing source lock, or exclusive output path in online mode
    #[arg(long)]
    source_lock: PathBuf,

    /// Existing pinned model bundle
    #[arg(long)]
    model_dir: PathBuf,

    /// Existing oracle bundle containing manifest.json and metrics
    #[arg(long)]
    oracle_dir: PathBuf,

    /// Existing ApxInf release binary to verify
    #[arg(long)]
    binary: PathBuf,

    /// Exclusive generation receipt output path
    #[arg(long)]
    receipt_output: PathBuf,

    /// Exclusive deployment lock output path
    #[arg(long)]
    lock_output: PathBuf,

    /// Reuse and verify the source lock without metadata network access
    #[arg(long)]
    offline: bool,

    /// Download missing files required by the pinned model bundle
    #[arg(long)]
    download_missing: bool,

    /// Emit the controller's exact stage plan without executing it
    #[arg(long)]
    dry_run: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match run_cli(cli) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run_cli(cli: Cli) -> Result<ExitCode, String> {
    match cli.command {
        Commands::Generate {
            model,
            prompt,
            image,
            max_tokens,
            max_context,
            no_eos_stop,
            show_token_ids,
            json,
            system,
            device,
            dtype,
            provider,
            mlx_python,
            mlx_runner,
            metal_w8_lm_head,
            metal_w8_mlp_block,
        } => {
            let device = parse_device(&device)?;
            let text_weight_dtype = parse_dtype(&dtype)?;
            run_generate(
                &model,
                &prompt,
                image.as_ref(),
                max_tokens,
                max_context,
                !no_eos_stop,
                show_token_ids,
                json,
                system.as_deref(),
                device,
                text_weight_dtype,
                &dtype,
                provider,
                mlx_python.as_deref(),
                mlx_runner.as_deref(),
                metal_w8_lm_head,
                metal_w8_mlp_block,
            )?;
            Ok(ExitCode::SUCCESS)
        }
        Commands::Onboard(args) => match run_onboard(&args) {
            Ok(code) => Ok(code),
            Err(error) => emit_onboard_launcher_error(&error),
        },
        Commands::MlxServe {
            model,
            mlx_python,
            mlx_runner,
            timeout_seconds,
        } => Ok(mlx_service_cli::run(
            &mlx_python,
            &mlx_runner,
            &model,
            timeout_seconds,
        )),
        Commands::Test => {
            run_test();
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn parse_device(s: &str) -> Result<Device, String> {
    match s.to_lowercase().as_str() {
        "cuda" | "gpu" => Ok(Device::Cuda(0)),
        "cpu" => Ok(Device::Cpu),
        _ => Err(format!("Unknown device '{s}'. Use 'cpu' or 'cuda'.")),
    }
}

fn parse_dtype(s: &str) -> Result<DType, String> {
    match s.to_ascii_lowercase().as_str() {
        "fp32" | "f32" => Ok(DType::F32),
        "bf16" => Ok(DType::BF16),
        other => Err(format!(
            "Unsupported text weight dtype `{other}`; use fp32 or bf16"
        )),
    }
}

const MAX_CONTROLLER_JSON_BYTES: usize = 16 * 1024 * 1024;
const ONBOARD_RECEIPT_FORMAT: &str = "apxinf-hf-macos-onboard-receipt-v2";
const ONBOARD_PLAN_FORMAT: &str = "apxinf-hf-macos-onboard-plan-v2";

fn absolute_cli_path(path: &Path, current_dir: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        current_dir.join(path)
    }
}

fn validate_controller_program(path: &Path, label: &str, executable: bool) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("Cannot inspect {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        ));
    }
    #[cfg(unix)]
    if executable {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(format!(
                "{label} must have an executable permission bit: {}",
                path.display()
            ));
        }
    }
    #[cfg(not(unix))]
    let _ = executable;
    Ok(())
}

fn onboard_controller_argv(
    args: &OnboardArgs,
    controller: &Path,
    current_dir: &Path,
) -> Vec<OsString> {
    let mut argv = vec![
        controller.as_os_str().to_owned(),
        OsString::from(&args.model_url),
        OsString::from("--revision"),
        OsString::from(&args.revision),
    ];
    for (flag, path) in [
        ("--source-lock", &args.source_lock),
        ("--model-dir", &args.model_dir),
        ("--oracle-dir", &args.oracle_dir),
        ("--binary", &args.binary),
        ("--receipt-output", &args.receipt_output),
        ("--lock-output", &args.lock_output),
    ] {
        argv.push(OsString::from(flag));
        argv.push(absolute_cli_path(path, current_dir).into_os_string());
    }
    if args.offline {
        argv.push(OsString::from("--offline"));
    }
    if args.download_missing {
        argv.push(OsString::from("--download-missing"));
    }
    if args.dry_run {
        argv.push(OsString::from("--dry-run"));
    }
    argv
}

fn onboard_controller_command(python: &Path, argv: &[OsString]) -> Command {
    let mut command = Command::new(python);
    command
        .args(argv)
        .current_dir("/")
        .env_clear()
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .env("PYTHONNOUSERSITE", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn bounded_controller_reader<R>(
    mut reader: R,
    label: &'static str,
    limit: usize,
    failure_tx: std::sync::mpsc::Sender<String>,
) -> std::thread::JoinHandle<Result<Vec<u8>, String>>
where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut payload = Vec::new();
        let mut buffer = [0_u8; 8192];
        loop {
            let count = match reader.read(&mut buffer) {
                Ok(count) => count,
                Err(error) => {
                    let message = format!("Cannot read controller {label}: {error}");
                    let _ = failure_tx.send(message.clone());
                    return Err(message);
                }
            };
            if count == 0 {
                return Ok(payload);
            }
            let next_len = payload.len().checked_add(count).unwrap_or(usize::MAX);
            if next_len > limit {
                let message = format!("Controller {label} exceeded the {limit} byte capture limit");
                let _ = failure_tx.send(message.clone());
                return Err(message);
            }
            payload.extend_from_slice(&buffer[..count]);
        }
    })
}

fn run_controller_bounded(mut command: Command) -> Result<(ExitStatus, Vec<u8>, Vec<u8>), String> {
    let mut child = command
        .spawn()
        .map_err(|error| format!("Cannot launch onboarding controller: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Controller stdout was not captured".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Controller stderr was not captured".to_string())?;
    let (failure_tx, failure_rx) = std::sync::mpsc::channel();
    let stdout_reader = bounded_controller_reader(
        stdout,
        "stdout",
        MAX_CONTROLLER_JSON_BYTES,
        failure_tx.clone(),
    );
    let stderr_reader = bounded_controller_reader(
        stderr,
        "stderr",
        MAX_CONTROLLER_JSON_BYTES,
        failure_tx.clone(),
    );
    drop(failure_tx);

    // The explicit controller is trusted for its overall runtime. Polling is
    // only needed so an output-limit violation can terminate it promptly.
    let mut capture_failure = None;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("Cannot wait for onboarding controller: {error}"))?
        {
            break status;
        }
        match failure_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(error) => {
                capture_failure = Some(error);
                let _ = child.kill();
                break child.wait().map_err(|wait_error| {
                    format!("Cannot reap onboarding controller: {wait_error}")
                })?;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    };

    let stdout = stdout_reader
        .join()
        .map_err(|_| "Controller stdout reader panicked".to_string());
    let stderr = stderr_reader
        .join()
        .map_err(|_| "Controller stderr reader panicked".to_string());
    if let Some(error) = capture_failure {
        return Err(error);
    }
    Ok((status, stdout??, stderr??))
}

fn validate_single_controller_json(
    payload: &[u8],
    label: &str,
    passed: bool,
    expected_format: &str,
) -> Result<(), String> {
    if payload.is_empty() || payload.len() > MAX_CONTROLLER_JSON_BYTES {
        return Err(format!(
            "{label} must contain one bounded JSON object (observed {} bytes)",
            payload.len()
        ));
    }
    let text =
        std::str::from_utf8(payload).map_err(|error| format!("{label} is not UTF-8: {error}"))?;
    if text.lines().count() != 1 {
        return Err(format!("{label} must contain exactly one JSON line"));
    }
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|error| format!("{label} is not valid JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} JSON root must be an object"))?;
    if object.get("format").and_then(serde_json::Value::as_str) != Some(expected_format) {
        return Err(format!("{label} JSON `format` must be `{expected_format}`"));
    }
    if object.get("passed").and_then(serde_json::Value::as_bool) != Some(passed) {
        return Err(format!(
            "{label} JSON `passed` must be {passed} for this exit status"
        ));
    }
    Ok(())
}

fn validate_controller_result(
    code: i32,
    stdout: &[u8],
    stderr: &[u8],
    dry_run: bool,
) -> Result<u8, String> {
    let code = u8::try_from(code)
        .map_err(|_| format!("Controller returned an invalid exit code: {code}"))?;
    if code == 0 {
        if !stderr.is_empty() {
            return Err("Controller wrote stderr despite a successful exit".to_string());
        }
        let expected_format = if dry_run {
            ONBOARD_PLAN_FORMAT
        } else {
            ONBOARD_RECEIPT_FORMAT
        };
        validate_single_controller_json(stdout, "Controller stdout", true, expected_format)?;
    } else {
        if !stdout.is_empty() {
            return Err("Controller wrote stdout despite a failed exit".to_string());
        }
        validate_single_controller_json(
            stderr,
            "Controller stderr",
            false,
            ONBOARD_RECEIPT_FORMAT,
        )?;
    }
    Ok(code)
}

fn run_onboard(args: &OnboardArgs) -> Result<ExitCode, String> {
    let current_dir = std::env::current_dir()
        .map_err(|error| format!("Cannot resolve the current directory: {error}"))?;
    let controller = absolute_cli_path(&args.controller, &current_dir);
    let python = absolute_cli_path(&args.python, &current_dir);
    validate_controller_program(&controller, "Onboarding controller", false)?;
    validate_controller_program(&python, "Python interpreter", true)?;
    let argv = onboard_controller_argv(args, &controller, &current_dir);
    let (status, captured_stdout, captured_stderr) =
        run_controller_bounded(onboard_controller_command(&python, &argv))?;
    let raw_code = status
        .code()
        .ok_or_else(|| "Onboarding controller terminated without an exit code".to_string())?;
    let code =
        validate_controller_result(raw_code, &captured_stdout, &captured_stderr, args.dry_run)?;
    if code == 0 {
        let mut stdout = std::io::stdout().lock();
        stdout
            .write_all(&captured_stdout)
            .and_then(|_| stdout.flush())
            .map_err(|error| format!("Cannot relay controller stdout: {error}"))?;
    } else {
        let mut stderr = std::io::stderr().lock();
        stderr
            .write_all(&captured_stderr)
            .and_then(|_| stderr.flush())
            .map_err(|error| format!("Cannot relay controller stderr: {error}"))?;
    }
    Ok(ExitCode::from(code))
}

fn emit_onboard_launcher_error(error: &str) -> Result<ExitCode, String> {
    let receipt = serde_json::json!({
        "format": "apxinf-hf-macos-onboard-launcher-v1",
        "passed": false,
        "error": {
            "code": "ONBOARD_LAUNCH_FAILED",
            "message": error,
        }
    });
    let payload = serde_json::to_string(&receipt).map_err(|serialize_error| {
        format!("Cannot serialize onboarding error: {serialize_error}")
    })?;
    let mut stderr = std::io::stderr().lock();
    writeln!(stderr, "{payload}")
        .and_then(|_| stderr.flush())
        .map_err(|write_error| format!("Cannot write onboarding error: {write_error}"))?;
    Ok(ExitCode::from(2))
}

fn run_generate(
    model_dir: &PathBuf,
    prompt: &str,
    image_path: Option<&PathBuf>,
    max_tokens: usize,
    max_context: usize,
    eos_stop: bool,
    show_token_ids: bool,
    json_output: bool,
    system_prompt: Option<&str>,
    device: Device,
    text_weight_dtype: DType,
    dtype: &str,
    provider: GenerateProvider,
    mlx_python: Option<&Path>,
    mlx_runner: Option<&Path>,
    metal_w8_lm_head: bool,
    metal_w8_mlp_block: bool,
) -> Result<(), String> {
    let mlx_boundary = validate_generate_provider(
        provider,
        mlx_python,
        mlx_runner,
        image_path.is_some(),
        device,
        text_weight_dtype,
        metal_w8_lm_head,
        metal_w8_mlp_block,
    )?;
    if !json_output {
        println!("apxinf — LLM/VLM inference engine");
        println!();
    }

    let model_name = AutoModel::detect_model_name(model_dir)
        .map_err(|error| format!("Failed to detect model type: {error}"))?;
    if image_path.is_some() && !matches!(model_name.as_str(), "qwen3_vl" | "qwen3vl") {
        return Err(format!("Model `{model_name}` does not support image input"));
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !json_output {
        println!("Loading tokenizer from {:?}...", tokenizer_path);
    }
    let tok = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("Failed to load tokenizer: {error}"))?;
    if !json_output {
        println!("Vocab size: {}", tok.vocab_size());
    }

    let eos_token_id = if eos_stop { tok.eos_token_id() } else { None };
    if !json_output {
        if let Some(eos) = eos_token_id {
            println!("EOS token ID: {eos}");
        }
    }

    // Model-specific processors turn raw media into tensors, while generation
    // itself always receives the model-neutral LlmInput request.
    let (tokens, prepared_image) = if let Some(image_path) = image_path {
        if !json_output {
            println!("Preprocessing image via the Hugging Face processor...");
        }
        let (data, shape, grid, tokens) =
            preprocess_image(model_dir, image_path, prompt, system_prompt)
                .map_err(|error| format!("Preprocessing failed: {error}"))?;
        if !json_output {
            println!(
                "pixel_values: {:?}, grid_thw: {:?}, prompt tokens: {}",
                shape,
                grid,
                tokens.len()
            );
        }
        let pixels = Tensor::from_bf16(shape, &data)
            .map_err(|error| format!("Invalid processor output: {error}"))?;
        (tokens, Some((pixels, vec![grid])))
    } else {
        let tokens = encode_prompt(&tok, prompt, system_prompt, json_output)
            .map_err(|error| format!("Failed to encode prompt: {error}"))?;
        (tokens, None)
    };

    let requested_context = tokens
        .len()
        .checked_add(max_tokens)
        .ok_or_else(|| "Prompt plus generation budget overflows usize".to_string())?;
    if max_context == 0 || requested_context > max_context {
        return Err(format!(
            "Prompt length {} + generation budget {max_tokens} exceeds --max-context {max_context}",
            tokens.len()
        ));
    }

    if let Some((python, runner)) = mlx_boundary {
        return run_generate_mlx(
            model_dir,
            &model_name,
            &tok,
            &tokens,
            max_tokens,
            eos_token_id,
            eos_stop,
            show_token_ids,
            json_output,
            python,
            runner,
        );
    }

    let options = LoadOptions {
        model_name: Some(model_name.clone()),
        text_weight_dtype: Some(text_weight_dtype),
        max_context: Some(max_context),
        metal_w8_lm_head,
        metal_w8_mlp_block,
        ..LoadOptions::default()
    };

    if !json_output {
        println!(
            "Loading {model_name} from {:?}... (dtype: {dtype})",
            model_dir
        );
    }
    let mut model = AutoModel::load_model(device, model_dir, &options)
        .map_err(|error| format!("Failed to load model: {error}"))?;
    if prepared_image.is_some() {
        match model.text_capabilities() {
            Ok(capabilities) if capabilities.image => {}
            Ok(_) => {
                return Err(format!("Model `{model_name}` does not support image input"));
            }
            Err(error) => {
                return Err(format!("Cannot generate with this model: {error}"));
            }
        }
    }
    if !json_output {
        println!("Model ready.");
    }

    let input = match prepared_image.as_ref() {
        Some((pixels, grids)) => LlmInput::with_image(&tokens, ImageInput::new(pixels, grids)),
        None => LlmInput::text(&tokens),
    };

    if !json_output {
        println!();
        println!("Generating {max_tokens} tokens...");
    }
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut all_tokens = tokens.clone();
    let mut stream_error = None;

    let (generated_tokens, profile) = model
        .generate_streaming(
            input,
            max_tokens,
            |token_id| {
                if json_output {
                    return;
                }
                if stream_error.is_some() {
                    return;
                }
                all_tokens.push(token_id);
                let text = match tok.decode(&all_tokens) {
                    Ok(text) => text,
                    Err(error) => {
                        stream_error = Some(format!("Failed to decode generated text: {error}"));
                        return;
                    }
                };
                let previous = match tok.decode(&all_tokens[..all_tokens.len() - 1]) {
                    Ok(previous) => previous,
                    Err(error) => {
                        stream_error = Some(format!("Failed to decode generated text: {error}"));
                        return;
                    }
                };
                let delta = text.strip_prefix(&previous).unwrap_or(&text);
                if let Err(error) = write!(out, "{delta}").and_then(|_| out.flush()) {
                    stream_error = Some(format!("Failed to write generated text: {error}"));
                }
            },
            eos_token_id,
        )
        .map_err(|error| format!("Generation failed: {error}"))?;
    if let Some(error) = stream_error {
        return Err(error);
    }

    drop(out);
    if json_output {
        let payload = generation_json(
            &model_name,
            device,
            text_weight_dtype,
            tokens.len(),
            &generated_tokens,
            &profile,
            metal_w8_lm_head,
            metal_w8_mlp_block,
            model
                .generation_path_receipt()
                .map_err(|error| format!("Cannot read generation path receipt: {error}"))?,
        )?;
        let mut stdout = std::io::stdout().lock();
        writeln!(stdout, "{payload}")
            .and_then(|_| stdout.flush())
            .map_err(|error| format!("Failed to write JSON result: {error}"))?;
    } else {
        println!();
        if show_token_ids {
            println!("Generated token IDs: {generated_tokens:?}");
            println!();
        }
        println!();
        println!("{}", profile.summary());
    }
    Ok(())
}

fn validate_generate_provider<'a>(
    provider: GenerateProvider,
    mlx_python: Option<&'a Path>,
    mlx_runner: Option<&'a Path>,
    has_image: bool,
    device: Device,
    text_weight_dtype: DType,
    metal_w8_lm_head: bool,
    metal_w8_mlp_block: bool,
) -> Result<Option<(&'a Path, &'a Path)>, String> {
    match provider {
        GenerateProvider::Native => {
            if mlx_python.is_some() || mlx_runner.is_some() {
                return Err("--mlx-python and --mlx-runner require `--provider mlx`".to_string());
            }
            Ok(None)
        }
        GenerateProvider::Mlx => {
            if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                return Err("The MLX provider requires Apple Silicon macOS".to_string());
            }
            if has_image {
                return Err(
                    "The current MLX worker is text-only; --image requires the native provider"
                        .to_string(),
                );
            }
            if device != Device::Cpu || text_weight_dtype != DType::F32 {
                return Err(
                    "--device and --dtype configure the native provider; use their defaults (`cpu`, `fp32`) with `--provider mlx`"
                        .to_string(),
                );
            }
            if metal_w8_lm_head || metal_w8_mlp_block {
                let flags = match (metal_w8_mlp_block, metal_w8_lm_head) {
                    (true, true) => "--metal-w8-mlp-block and --metal-w8-lm-head",
                    (true, false) => "--metal-w8-mlp-block",
                    (false, true) => "--metal-w8-lm-head",
                    (false, false) => unreachable!(),
                };
                return Err(format!(
                    "{flags} belong to the native provider and cannot be combined with `--provider mlx`"
                ));
            }
            let python = mlx_python
                .ok_or_else(|| "--provider mlx requires --mlx-python PATH".to_string())?;
            let runner = mlx_runner
                .ok_or_else(|| "--provider mlx requires --mlx-runner PATH".to_string())?;
            Ok(Some((python, runner)))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_generate_mlx(
    model_dir: &Path,
    model_name: &str,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    eos_token_id: Option<u32>,
    eos_stop: bool,
    show_token_ids: bool,
    json_output: bool,
    python: &Path,
    runner: &Path,
) -> Result<(), String> {
    // Keep the final path component unresolved so the provider can enforce its
    // direct non-symlink model-directory boundary before canonicalising the
    // verified directory for the worker and receipt.
    let current_dir = std::env::current_dir()
        .map_err(|error| format!("Cannot resolve the current directory: {error}"))?;
    let absolute_model_dir = absolute_cli_path(model_dir, &current_dir);
    if !json_output {
        println!(
            "Loading {model_name} through the trusted local-only MLX provider from {:?}...",
            absolute_model_dir
        );
    }
    let generation = mlx_provider::generate_with_mlx(
        python,
        runner,
        &absolute_model_dir,
        prompt_tokens,
        max_tokens,
        eos_token_id,
        eos_stop,
    )
    .map_err(|error| error.to_string())?;

    if json_output {
        let payload = serde_json::to_string(&generation.receipt)
            .map_err(|error| format!("Failed to serialize MLX receipt: {error}"))?;
        let mut stdout = std::io::stdout().lock();
        writeln!(stdout, "{payload}")
            .and_then(|_| stdout.flush())
            .map_err(|error| format!("Failed to write MLX JSON result: {error}"))?;
        return Ok(());
    }

    let prompt_text = tokenizer
        .decode(prompt_tokens)
        .map_err(|error| format!("Failed to decode MLX prompt tokens: {error}"))?;
    let mut all_tokens =
        Vec::with_capacity(prompt_tokens.len() + generation.generated_token_ids.len());
    all_tokens.extend_from_slice(prompt_tokens);
    all_tokens.extend_from_slice(&generation.generated_token_ids);
    let full_text = tokenizer
        .decode(&all_tokens)
        .map_err(|error| format!("Failed to decode MLX generated tokens: {error}"))?;
    let generated_text = full_text.strip_prefix(&prompt_text).unwrap_or(&full_text);
    print!("{generated_text}");
    std::io::stdout()
        .flush()
        .map_err(|error| format!("Failed to write MLX generated text: {error}"))?;
    println!();
    if show_token_ids {
        println!("Generated token IDs: {:?}", generation.generated_token_ids);
        println!();
    }
    println!();
    println!(
        "MLX: load {:.2} ms, TTFT {:.2} ms, TPOT {:.2} ms, {:.2} tok/s, peak {:.2} GiB",
        generation.metrics.load_ms,
        generation.metrics.ttft_ms,
        generation.metrics.tpot_ms,
        generation.metrics.tps,
        generation.metrics.mlx_peak_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok(())
}

fn generation_json(
    model_type: &str,
    device: Device,
    dtype: DType,
    prompt_token_count: usize,
    generated_token_ids: &[u32],
    profile: &GenerationProfile,
    metal_w8_lm_head: bool,
    metal_w8_mlp_block: bool,
    generation_path_receipt: Option<serde_json::Value>,
) -> Result<String, String> {
    let dtype = match dtype {
        DType::F32 => "fp32",
        DType::BF16 => "bf16",
        other => {
            return Err(format!(
                "Cannot serialize unsupported text weight dtype `{other}`"
            ));
        }
    };
    let value = serde_json::json!({
        "format": "apxinf-generation-v1",
        "model_type": model_type,
        "device": device.to_string(),
        "dtype": dtype,
        "build": {
            "target_os": std::env::consts::OS,
            "target_arch": std::env::consts::ARCH,
            "matmul_feature": compiled_matmul_feature(),
            "metal_w8_lm_head": metal_w8_lm_head,
            "metal_w8_mlp_block": metal_w8_mlp_block,
        },
        "generation_path": generation_path_receipt,
        "prompt_token_count": prompt_token_count,
        "generated_token_ids": generated_token_ids,
        "profile": {
            "input_tokens": profile.input_tokens(),
            "output_tokens": profile.output_tokens(),
            "ttft_ms": profile.ttft_ms(),
            "tpot_ms": profile.tpot_ms(),
            "generation_tps": profile.generation_tps(),
            "total_latency_ms": profile.total_latency_ms(),
        }
    });
    serde_json::to_string(&value)
        .map_err(|error| format!("Failed to serialize JSON result: {error}"))
}

fn compiled_matmul_feature() -> &'static str {
    if cfg!(feature = "accelerate") {
        "accelerate"
    } else if cfg!(feature = "openblas") {
        "openblas"
    } else {
        "naive"
    }
}

fn encode_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    system_prompt: Option<&str>,
    quiet: bool,
) -> Result<Vec<u32>, String> {
    if tokenizer.has_chat_template() {
        let mut messages = Vec::new();
        if let Some(system) = system_prompt {
            messages.push(ChatMessage::system(system));
        }
        messages.push(ChatMessage::user(prompt));
        if quiet {
            let formatted = tokenizer
                .apply_chat_template(&messages)
                .map_err(|error| error.to_string())?;
            tokenizer
                .encode(&formatted)
                .map_err(|error| error.to_string())
        } else {
            tokenizer
                .encode_chat(&messages)
                .map_err(|error| error.to_string())
        }
    } else {
        tokenizer.encode(prompt).map_err(|error| error.to_string())
    }
}
/// Preprocess an image with the model's Hugging Face processor. Raw image
/// decoding and chat templating stay outside the model runtime; the resulting
/// borrowed tensor is attached to LlmInput for unified generation.
fn preprocess_image(
    model_dir: &PathBuf,
    image_path: &PathBuf,
    prompt: &str,
    system_prompt: Option<&str>,
) -> Result<(Vec<half::bf16>, Vec<usize>, [u32; 3], Vec<u32>), String> {
    use std::process::Command;

    let suffix = std::process::id();
    let pixel_path = std::env::temp_dir().join(format!("apxinf-cli-{suffix}-pixels.npy"));
    let metadata_path = std::env::temp_dir().join(format!("apxinf-cli-{suffix}-metadata.json"));
    let script = r#"
import json
import sys
import numpy as np
from transformers import AutoProcessor
from PIL import Image

model_dir, image_path, prompt, system, pixel_path, metadata_path = sys.argv[1:]
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
image = Image.open(image_path).convert("RGB")
messages = []
if system:
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": system}],
    })
messages.append({
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ],
})
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
pixels = inputs["pixel_values"].cpu().numpy().astype(np.float32)
grid = inputs["image_grid_thw"][0].cpu().numpy().tolist()
tokens = inputs["input_ids"][0].cpu().numpy().astype(np.int64).tolist()
np.save(pixel_path, pixels)
with open(metadata_path, "w") as output:
    json.dump({"grid": grid, "tokens": tokens}, output)
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(model_dir)
        .arg(image_path)
        .arg(prompt)
        .arg(system_prompt.unwrap_or(""))
        .arg(&pixel_path)
        .arg(&metadata_path)
        .output()
        .map_err(|error| format!("python3: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "python preprocessing failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let metadata_raw = std::fs::read_to_string(&metadata_path)
        .map_err(|error| format!("read {}: {error}", metadata_path.display()))?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_raw)
        .map_err(|error| format!("parse {}: {error}", metadata_path.display()))?;
    let grid_values = metadata
        .get("grid")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "processor metadata has no grid array".to_string())?;
    if grid_values.len() != 3 {
        return Err(format!(
            "processor grid must have three values, got {}",
            grid_values.len()
        ));
    }
    let grid = [
        grid_values[0]
            .as_u64()
            .ok_or_else(|| "processor grid T is not an integer".to_string())? as u32,
        grid_values[1]
            .as_u64()
            .ok_or_else(|| "processor grid H is not an integer".to_string())? as u32,
        grid_values[2]
            .as_u64()
            .ok_or_else(|| "processor grid W is not an integer".to_string())? as u32,
    ];
    let tokens = metadata
        .get("tokens")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "processor metadata has no tokens array".to_string())?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .map(|token| token as u32)
                .ok_or_else(|| "processor returned a non-integer token".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (pixel_shape, pixel_data) = read_npy_f32_to_bf16(&pixel_path)?;

    let _ = std::fs::remove_file(&pixel_path);
    let _ = std::fs::remove_file(&metadata_path);
    Ok((pixel_data, pixel_shape, grid, tokens))
}

/// Read a NumPy v1 f32 array and convert it to bf16.
fn read_npy_f32_to_bf16(path: &std::path::Path) -> Result<(Vec<usize>, Vec<half::bf16>), String> {
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if buffer.len() < 10 || &buffer[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a NumPy array", path.display()));
    }
    if buffer[6] != 1 {
        return Err(format!(
            "{} uses unsupported NumPy format version {}",
            path.display(),
            buffer[6]
        ));
    }
    let header_len = u16::from_le_bytes([buffer[8], buffer[9]]) as usize;
    let data_start = 10usize
        .checked_add(header_len)
        .ok_or_else(|| "NumPy header length overflow".to_string())?;
    if data_start > buffer.len() {
        return Err("NumPy header exceeds file length".to_string());
    }
    let header = std::str::from_utf8(&buffer[10..data_start])
        .map_err(|error| format!("invalid NumPy header: {error}"))?;
    if !header.contains("<f4") {
        return Err("processor pixel array is not little-endian f32".to_string());
    }
    let shape = parse_npy_shape(header)?;
    let raw = &buffer[data_start..];
    let expected_bytes = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
    if raw.len() != expected_bytes {
        return Err(format!(
            "NumPy payload has {} bytes, expected {expected_bytes}",
            raw.len()
        ));
    }
    let data = raw
        .chunks_exact(4)
        .map(|bytes| half::bf16::from_f32(f32::from_le_bytes(bytes.try_into().unwrap())))
        .collect();
    Ok((shape, data))
}

fn parse_npy_shape(header: &str) -> Result<Vec<usize>, String> {
    let shape_offset = header
        .find("shape")
        .ok_or_else(|| "NumPy header has no shape".to_string())?;
    let open_offset = header[shape_offset..]
        .find('(')
        .ok_or_else(|| "NumPy shape has no opening parenthesis".to_string())?;
    let shape_start = shape_offset + open_offset + 1;
    let close_offset = header[shape_start..]
        .find(')')
        .ok_or_else(|| "NumPy shape has no closing parenthesis".to_string())?;
    let shape_text = &header[shape_start..shape_start + close_offset];
    let shape = shape_text
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid NumPy shape: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shape.is_empty() {
        return Err("NumPy array has an empty shape".to_string());
    }
    Ok(shape)
}
fn run_test() {
    println!("apxinf — LLM inference engine (test mode)");
    println!();

    // ── CPU matmul smoke test ───────────────────────────────────────
    use apxinf_core::Tensor;

    let a = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_f32(vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c_cpu = a.matmul_cpu(&b).unwrap();
    println!("[CPU] A: {a}");
    println!("[CPU] B: {b}");
    println!("[CPU] C = A @ B: {c_cpu}");
    println!("[CPU] C data: {:?}", c_cpu.as_f32().unwrap());
    println!();

    #[cfg(feature = "cuda")]
    cuda_test();
}

#[cfg(feature = "cuda")]
fn cuda_test() {
    use apxinf_core::Tensor;
    use apxinf_cuda::{
        kernels::{activation, attention, elementwise, gemm, norm, rope},
        transfers, CudaContext,
    };

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("[CUDA] Not available: {e}");
            return;
        }
    };
    println!("[CUDA] Device: {}", ctx.device_id());

    // Matmul test
    let a = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_f32(vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

    let a_gpu = transfers::to_cuda(&a, 0).unwrap();
    let b_gpu = transfers::to_cuda(&b, 0).unwrap();

    let c_gpu = gemm::matmul(&ctx, &a_gpu, &b_gpu).unwrap();
    let c_cpu = transfers::to_cpu(&c_gpu).unwrap();
    let data = c_cpu.as_f32().unwrap();
    println!("[CUDA] matmul: {:?}", data);

    // SiLU test
    let x = Tensor::from_f32(vec![4], &[1.0, -1.0, 0.0, 2.0]).unwrap();
    let x_gpu = transfers::to_cuda(&x, 0).unwrap();
    let silu_gpu = activation::silu(&ctx, &x_gpu).unwrap();
    let silu_cpu = transfers::to_cpu(&silu_gpu).unwrap();
    let silu_data = silu_cpu.as_f32().unwrap();
    let _silu_expected: Vec<f32> = [1.0f32, -1.0, 0.0, 2.0]
        .iter()
        .map(|x| x / (1.0 + (-x).exp()))
        .collect();
    println!("[CUDA] silu: {:?}", silu_data);

    // Add test
    let a2 = Tensor::from_f32(vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b2 = Tensor::from_f32(vec![4], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let a2_gpu = transfers::to_cuda(&a2, 0).unwrap();
    let b2_gpu = transfers::to_cuda(&b2, 0).unwrap();
    let add_gpu = elementwise::add(&ctx, &a2_gpu, &b2_gpu).unwrap();
    let add_cpu = transfers::to_cpu(&add_gpu).unwrap();
    println!("[CUDA] add: {:?}", add_cpu.as_f32().unwrap());

    // Mul test
    let mul_gpu = elementwise::mul(&ctx, &a2_gpu, &b2_gpu).unwrap();
    let mul_cpu = transfers::to_cpu(&mul_gpu).unwrap();
    println!("[CUDA] mul: {:?}", mul_cpu.as_f32().unwrap());

    // RMSNorm test
    let input = Tensor::from_f32(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let weight = Tensor::from_f32(vec![4], &[1.0, 1.0, 1.0, 1.0]).unwrap();
    let input_gpu = transfers::to_cuda(&input, 0).unwrap();
    let weight_gpu = transfers::to_cuda(&weight, 0).unwrap();
    let norm_gpu = norm::rms(&ctx, &input_gpu, &weight_gpu, 1e-5).unwrap();
    let norm_cpu = transfers::to_cpu(&norm_gpu).unwrap();
    println!("[CUDA] rms_norm: {:?}", norm_cpu.as_f32().unwrap());

    // Softmax test
    let sm_input = Tensor::from_f32(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let sm_gpu = transfers::to_cuda(&sm_input, 0).unwrap();
    let softmax_gpu = attention::softmax(&ctx, &sm_gpu).unwrap();
    let softmax_cpu = transfers::to_cpu(&softmax_gpu).unwrap();
    println!("[CUDA] softmax: {:?}", softmax_cpu.as_f32().unwrap());

    // RoPE test
    let rope_input =
        Tensor::from_f32(vec![2, 4], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]).unwrap();
    let rope_gpu = transfers::to_cuda(&rope_input, 0).unwrap();
    let rope_out = rope::apply(&ctx, &rope_gpu, 2, 4, 10000.0, 0).unwrap();
    let rope_cpu = transfers::to_cpu(&rope_out).unwrap();
    println!("[CUDA] rope: {:?}", rope_cpu.as_f32().unwrap());

    // Causal mask test
    let mask_input = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mask_gpu = transfers::to_cuda(&mask_input, 0).unwrap();
    let mask_out = attention::causal_mask(&ctx, &mask_gpu, 0).unwrap();
    let mask_cpu = transfers::to_cpu(&mask_out).unwrap();
    println!("[CUDA] causal_mask: {:?}", mask_cpu.as_f32().unwrap());

    println!("[CUDA] All kernel tests completed.");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn onboard_args() -> OnboardArgs {
        OnboardArgs {
            model_url: "https://huggingface.co/Qwen/Qwen3.5-0.8B".to_string(),
            controller: PathBuf::from("scripts/onboard_hf_macos.py"),
            python: PathBuf::from("/usr/bin/python3"),
            revision: "0123456789abcdef".to_string(),
            source_lock: PathBuf::from("locks/source.json"),
            model_dir: PathBuf::from("bundle/model"),
            oracle_dir: PathBuf::from("bundle/oracle"),
            binary: PathBuf::from("target/release/apxinf"),
            receipt_output: PathBuf::from("out/receipt.json"),
            lock_output: PathBuf::from("out/deployment.lock.json"),
            offline: true,
            download_missing: true,
            dry_run: true,
        }
    }

    #[test]
    fn parse_device_accepts_supported_names() {
        assert!(matches!(parse_device("cpu"), Ok(Device::Cpu)));
        assert!(matches!(parse_device("CUDA"), Ok(Device::Cuda(0))));
        assert!(matches!(parse_device("gpu"), Ok(Device::Cuda(0))));
    }

    #[test]
    fn parse_device_rejects_unknown_name() {
        let error = parse_device("metal").unwrap_err();
        assert!(error.contains("Unknown device 'metal'"));
    }

    #[test]
    fn parse_dtype_accepts_aliases_and_rejects_unknown_name() {
        assert!(matches!(parse_dtype("fp32"), Ok(DType::F32)));
        assert!(matches!(parse_dtype("F32"), Ok(DType::F32)));
        assert!(matches!(parse_dtype("BF16"), Ok(DType::BF16)));

        let error = parse_dtype("fp16").unwrap_err();
        assert!(error.contains("Unsupported text weight dtype `fp16`"));
    }

    #[test]
    fn generate_command_parses_json_mode() {
        let cli = Cli::try_parse_from([
            "apxinf",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "hello",
            "--json",
        ])
        .unwrap();

        assert!(matches!(
            cli.command,
            Commands::Generate {
                json: true,
                show_token_ids: false,
                ..
            }
        ));
    }

    #[test]
    fn generate_command_parses_explicit_metal_w8_lm_head() {
        let cli = Cli::try_parse_from([
            "apxinf",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "hello",
            "--metal-w8-lm-head",
        ])
        .unwrap();

        assert!(matches!(
            cli.command,
            Commands::Generate {
                metal_w8_lm_head: true,
                metal_w8_mlp_block: false,
                ..
            }
        ));
    }

    #[test]
    fn generate_command_parses_explicit_metal_w8_mlp_block() {
        let cli = Cli::try_parse_from([
            "apxinf",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "hello",
            "--metal-w8-mlp-block",
        ])
        .unwrap();

        assert!(matches!(
            cli.command,
            Commands::Generate {
                metal_w8_mlp_block: true,
                metal_w8_lm_head: false,
                ..
            }
        ));
    }

    #[test]
    fn generate_command_parses_explicit_mlx_boundary() {
        let cli = Cli::try_parse_from([
            "apxinf",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "hello",
            "--provider",
            "mlx",
            "--mlx-python",
            "/tmp/python",
            "--mlx-runner",
            "/tmp/runner.py",
        ])
        .unwrap();

        match cli.command {
            Commands::Generate {
                provider,
                mlx_python,
                mlx_runner,
                ..
            } => {
                assert_eq!(provider, GenerateProvider::Mlx);
                assert_eq!(mlx_python.as_deref(), Some(Path::new("/tmp/python")));
                assert_eq!(mlx_runner.as_deref(), Some(Path::new("/tmp/runner.py")));
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn mlx_serve_command_parses_explicit_local_boundary() {
        let cli = Cli::try_parse_from([
            "apxinf",
            "mlx-serve",
            "--model",
            "/models/qwen",
            "--mlx-python",
            "/runtime/python",
            "--mlx-runner",
            "/runtime/apxinf_mlx_serve.py",
            "--timeout-seconds",
            "45",
        ])
        .unwrap();
        match cli.command {
            Commands::MlxServe {
                model,
                mlx_python,
                mlx_runner,
                timeout_seconds,
            } => {
                assert_eq!(model, PathBuf::from("/models/qwen"));
                assert_eq!(mlx_python, PathBuf::from("/runtime/python"));
                assert_eq!(mlx_runner, PathBuf::from("/runtime/apxinf_mlx_serve.py"));
                assert_eq!(timeout_seconds, 45);
            }
            _ => panic!("expected mlx-serve command"),
        }
    }

    #[test]
    fn native_provider_rejects_ignored_mlx_inputs() {
        let error = validate_generate_provider(
            GenerateProvider::Native,
            Some(Path::new("/tmp/python")),
            None,
            false,
            Device::Cpu,
            DType::F32,
            false,
            false,
        )
        .unwrap_err();
        assert!(error.contains("require `--provider mlx`"));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_provider_is_explicit_and_rejects_native_only_options() {
        let error = validate_generate_provider(
            GenerateProvider::Mlx,
            None,
            None,
            false,
            Device::Cpu,
            DType::F32,
            false,
            false,
        )
        .unwrap_err();
        assert!(error.contains("requires --mlx-python"));

        let error = validate_generate_provider(
            GenerateProvider::Mlx,
            Some(Path::new("/tmp/python")),
            Some(Path::new("/tmp/runner.py")),
            false,
            Device::Cpu,
            DType::F32,
            true,
            false,
        )
        .unwrap_err();
        assert!(error.contains("cannot be combined"));

        let error = validate_generate_provider(
            GenerateProvider::Mlx,
            Some(Path::new("/tmp/python")),
            Some(Path::new("/tmp/runner.py")),
            false,
            Device::Cpu,
            DType::F32,
            false,
            true,
        )
        .unwrap_err();
        assert!(error.contains("--metal-w8-mlp-block"));
        assert!(error.contains("cannot be combined"));
    }

    #[test]
    fn onboard_command_requires_and_parses_explicit_controller() {
        let cli = Cli::try_parse_from([
            "apxinf",
            "onboard",
            "https://huggingface.co/Qwen/Qwen3.5-0.8B",
            "--controller",
            "scripts/onboard_hf_macos.py",
            "--python",
            "/usr/bin/python3",
            "--revision",
            "0123456789abcdef",
            "--source-lock",
            "source.lock.json",
            "--model-dir",
            "model",
            "--oracle-dir",
            "oracle",
            "--binary",
            "target/release/apxinf",
            "--receipt-output",
            "receipt.json",
            "--lock-output",
            "deployment.lock.json",
            "--offline",
            "--download-missing",
            "--dry-run",
        ])
        .unwrap();

        match cli.command {
            Commands::Onboard(args) => {
                assert_eq!(
                    args.controller,
                    PathBuf::from("scripts/onboard_hf_macos.py")
                );
                assert_eq!(args.python, PathBuf::from("/usr/bin/python3"));
                assert_eq!(args.revision, "0123456789abcdef");
                assert!(args.offline);
                assert!(args.download_missing);
                assert!(args.dry_run);
            }
            _ => panic!("expected onboard command"),
        }

        let error = Cli::try_parse_from([
            "apxinf",
            "onboard",
            "https://huggingface.co/Qwen/Qwen3.5-0.8B",
        ])
        .err()
        .expect("missing required options must fail parsing");
        let message = error.to_string();
        assert!(message.contains("--controller"));
        assert!(message.contains("--python"));
    }

    #[test]
    fn onboard_argv_is_exact_and_absolutizes_paths() {
        let args = onboard_args();
        let checkout = Path::new("/trusted/checkout");
        let controller = checkout.join(&args.controller);
        let argv = onboard_controller_argv(&args, &controller, checkout);
        let expected = [
            "/trusted/checkout/scripts/onboard_hf_macos.py",
            "https://huggingface.co/Qwen/Qwen3.5-0.8B",
            "--revision",
            "0123456789abcdef",
            "--source-lock",
            "/trusted/checkout/locks/source.json",
            "--model-dir",
            "/trusted/checkout/bundle/model",
            "--oracle-dir",
            "/trusted/checkout/bundle/oracle",
            "--binary",
            "/trusted/checkout/target/release/apxinf",
            "--receipt-output",
            "/trusted/checkout/out/receipt.json",
            "--lock-output",
            "/trusted/checkout/out/deployment.lock.json",
            "--offline",
            "--download-missing",
            "--dry-run",
        ];

        assert_eq!(argv, expected.map(OsString::from));
    }

    #[test]
    fn onboard_argv_omits_download_flag_without_authorization() {
        let mut args = onboard_args();
        args.download_missing = false;
        let checkout = Path::new("/trusted/checkout");
        let controller = checkout.join(&args.controller);
        let argv = onboard_controller_argv(&args, &controller, checkout);

        assert!(!argv.iter().any(|value| value == "--download-missing"));
        assert!(argv.iter().any(|value| value == "--offline"));
        assert!(argv.iter().any(|value| value == "--dry-run"));
    }

    #[test]
    fn onboard_command_uses_python_directly_with_allowlisted_environment() {
        let argv = vec![
            OsString::from("/trusted/controller.py"),
            OsString::from("--dry-run"),
        ];
        let command = onboard_controller_command(Path::new("/usr/bin/python3"), &argv);

        assert_eq!(command.get_program(), "/usr/bin/python3");
        assert_eq!(command.get_current_dir(), Some(Path::new("/")));
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            argv.iter().collect::<Vec<_>>()
        );
        let mut environment = command
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().into_owned(),
                    value.unwrap().to_string_lossy().into_owned(),
                )
            })
            .collect::<Vec<_>>();
        environment.sort();
        assert_eq!(
            environment,
            vec![
                ("LANG".to_string(), "C".to_string()),
                ("LC_ALL".to_string(), "C".to_string()),
                (
                    "PATH".to_string(),
                    "/usr/bin:/bin:/usr/sbin:/sbin".to_string(),
                ),
                ("PYTHONDONTWRITEBYTECODE".to_string(), "1".to_string()),
                ("PYTHONNOUSERSITE".to_string(), "1".to_string()),
                ("PYTHONUTF8".to_string(), "1".to_string()),
            ]
        );
    }

    #[test]
    fn onboard_result_accepts_only_one_json_object_on_the_status_stream() {
        assert_eq!(
            validate_controller_result(
                0,
                br#"{"format":"apxinf-hf-macos-onboard-plan-v2","passed":true,"stage":"dry-run"}
"#,
                b"",
                true,
            ),
            Ok(0)
        );
        assert_eq!(
            validate_controller_result(
                7,
                b"",
                br#"{"format":"apxinf-hf-macos-onboard-receipt-v2","passed":false,"error":{"code":"X"}}
"#,
                false,
            ),
            Ok(7)
        );

        let success = br#"{"format":"apxinf-hf-macos-onboard-receipt-v2","passed":true}
"#;
        let failure = br#"{"format":"apxinf-hf-macos-onboard-receipt-v2","passed":false}
"#;
        assert!(validate_controller_result(0, failure, b"", false).is_err());
        assert!(validate_controller_result(2, success, failure, false).is_err());
        assert!(validate_controller_result(
            0,
            br#"{"format":"apxinf-hf-macos-onboard-receipt-v2","passed":true}
{"format":"apxinf-hf-macos-onboard-receipt-v2","passed":true}
"#,
            b"",
            false,
        )
        .is_err());
        assert!(validate_controller_result(
            0,
            br#"[true]
"#,
            b"",
            false
        )
        .is_err());
        assert!(validate_controller_result(0, success, b"progress", false).is_err());
        assert!(validate_controller_result(
            0,
            br#"{"format":"unknown","passed":true}
"#,
            b"",
            false,
        )
        .is_err());
    }

    #[test]
    fn onboard_program_validation_fails_closed_for_missing_file() {
        let missing = std::env::temp_dir().join(format!(
            "apxinf-missing-controller-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        assert!(!missing.exists());

        let error = validate_controller_program(&missing, "Onboarding controller", false)
            .expect_err("missing controller must be rejected");
        assert!(error.starts_with("Cannot inspect Onboarding controller"));
    }

    #[test]
    fn onboard_reader_rejects_output_over_its_bound() {
        let (failure_tx, failure_rx) = std::sync::mpsc::channel();
        let reader = bounded_controller_reader(
            std::io::Cursor::new(b"12345".to_vec()),
            "stdout",
            4,
            failure_tx,
        );

        let error = reader.join().unwrap().unwrap_err();
        assert_eq!(error, "Controller stdout exceeded the 4 byte capture limit");
        assert_eq!(failure_rx.recv().unwrap(), error);
    }

    #[test]
    fn generation_json_is_one_machine_readable_object() {
        let mut profile = GenerationProfile::new();
        profile.finalize(3, 2);
        let output = generation_json(
            "qwen3_5",
            Device::Cpu,
            DType::F32,
            3,
            &[11, 22],
            &profile,
            true,
            true,
            Some(serde_json::json!({"format": "test-path-v1", "hit": true})),
        )
        .unwrap();

        assert_eq!(output.lines().count(), 1);
        let value: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert!(value.is_object());
        assert_eq!(value["format"], "apxinf-generation-v1");
        assert_eq!(value["model_type"], "qwen3_5");
        assert_eq!(value["device"], "cpu");
        assert_eq!(value["dtype"], "fp32");
        assert_eq!(value["build"]["target_os"], std::env::consts::OS);
        assert_eq!(value["build"]["target_arch"], std::env::consts::ARCH);
        assert_eq!(value["build"]["matmul_feature"], compiled_matmul_feature());
        assert_eq!(value["build"]["metal_w8_lm_head"], true);
        assert_eq!(value["build"]["metal_w8_mlp_block"], true);
        assert_eq!(value["generation_path"]["hit"], true);
        assert_eq!(value["prompt_token_count"], 3);
        assert_eq!(value["generated_token_ids"], serde_json::json!([11, 22]));
        assert_eq!(value["profile"]["input_tokens"], 3);
        assert_eq!(value["profile"]["output_tokens"], 2);
        assert!(value["profile"]["total_latency_ms"].is_number());
    }

    #[test]
    fn missing_model_directory_is_an_error() {
        let missing =
            std::env::temp_dir().join(format!("apxinf-cli-missing-model-{}", std::process::id()));
        assert!(
            !missing.exists(),
            "test path unexpectedly exists: {missing:?}"
        );

        let error = run_generate(
            &missing,
            "hello",
            None,
            1,
            16,
            true,
            false,
            false,
            None,
            Device::Cpu,
            DType::F32,
            "fp32",
            GenerateProvider::Native,
            None,
            None,
            false,
            false,
        )
        .unwrap_err();

        assert!(error.starts_with("Failed to detect model type:"));
    }
}

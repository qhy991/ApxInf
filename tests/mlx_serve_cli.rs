use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

static FIXTURE_COUNTER: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    python: PathBuf,
    runner: PathBuf,
    model: PathBuf,
    marker: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let sequence = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "apxinf-mlx-cli-test-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let python = root.join("python3.14");
        let runner = root.join("apxinf_mlx_serve.py");
        let helper = root.join("apxinf_mlx_generate.py");
        let model = root.join("model");
        let marker = root.join("loads.txt");
        fs::create_dir(&model).unwrap();
        fs::write(model.join("config.json"), r#"{"model_type":"qwen3_5"}"#).unwrap();
        fs::write(helper, "# fake helper\n").unwrap();
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

    fn start(&self) -> RunningCli {
        self.start_with_timeout(5)
    }

    fn start_with_timeout(&self, timeout_seconds: u64) -> RunningCli {
        let mut child = Command::new(env!("CARGO_BIN_EXE_apxinf"))
            .arg("mlx-serve")
            .arg("--model")
            .arg(&self.model)
            .arg("--mlx-python")
            .arg(&self.python)
            .arg("--mlx-runner")
            .arg(&self.runner)
            .arg("--timeout-seconds")
            .arg(timeout_seconds.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        RunningCli {
            child,
            stdin: Some(stdin),
            stdout,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

struct RunningCli {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl RunningCli {
    fn ready(&mut self) -> Value {
        self.read()
    }

    fn exchange(&mut self, value: &Value) -> Value {
        let payload = serde_json::to_vec(value).unwrap();
        self.write_raw(&payload);
        self.read()
    }

    fn write_raw(&mut self, payload: &[u8]) {
        let stdin = self.stdin.as_mut().unwrap();
        stdin.write_all(payload).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
    }

    fn read(&mut self) -> Value {
        let mut line = String::new();
        assert!(self.stdout.read_line(&mut line).unwrap() > 0);
        serde_json::from_str(&line).unwrap()
    }

    fn close_stdin(&mut self) {
        self.stdin.take();
    }

    fn finish(mut self) -> (i32, String, String) {
        self.stdin.take();
        let status = self.child.wait().unwrap().code().unwrap();
        let mut stdout = String::new();
        self.stdout.read_to_string(&mut stdout).unwrap();
        let mut stderr = String::new();
        self.child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut stderr)
            .unwrap();
        (status, stdout, stderr)
    }

    #[cfg(unix)]
    fn finish_bounded(
        mut self,
        maximum: Duration,
        worker_pid_path: &std::path::Path,
    ) -> (i32, String, String) {
        self.stdin.take();
        let deadline = Instant::now() + maximum;
        let status = loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                if let Ok(payload) = fs::read_to_string(worker_pid_path) {
                    if let Ok(worker_pid) = payload.trim().parse::<i32>() {
                        unsafe extern "C" {
                            fn kill(process_id: i32, signal: i32) -> i32;
                        }
                        unsafe {
                            let _ = kill(-worker_pid, 9);
                        }
                    }
                }
                let _ = self.child.kill();
                let _ = self.child.wait();
                panic!("mlx-serve did not exit within {maximum:?}");
            }
            thread::sleep(Duration::from_millis(5));
        };
        let mut stdout = String::new();
        self.stdout.read_to_string(&mut stdout).unwrap();
        let mut stderr = String::new();
        self.child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut stderr)
            .unwrap();
        (status.code().unwrap(), stdout, stderr)
    }
}

use std::io::Read;

fn request(request_id: &str, operation: &str) -> Value {
    json!({
        "format": "apxinf-mlx-cli-request-v1",
        "request_id": request_id,
        "operation": operation,
        "prompt_token_ids": [1, 2],
        "max_tokens": 2,
        "stop_on_eos": true,
        "eos_token_id": 9,
    })
}

fn shutdown(running: &mut RunningCli, request_id: &str) -> Value {
    running.exchange(&json!({
        "format": "apxinf-mlx-cli-request-v1",
        "request_id": request_id,
        "operation": "shutdown",
    }))
}

#[test]
fn two_requests_load_once_and_shutdown_cleanly() {
    let fixture = Fixture::new();
    let mut running = fixture.start();
    let ready = running.ready();
    assert_eq!(ready["format"], "apxinf-mlx-cli-ready-v1");
    assert_eq!(ready["protocol"], "apxinf-mlx-cli-v1");
    assert_eq!(ready["network_listener"], false);
    assert_eq!(
        ready["validated_service_ready"]["format"],
        "apxinf-mlx-service-ready-v1"
    );
    for id in ["generate-1", "generate-2"] {
        let response = running.exchange(&request(id, "generate"));
        assert_eq!(response["format"], "apxinf-mlx-cli-response-v1");
        assert_eq!(response["request_id"], id);
        assert_eq!(
            response["validated_service_receipt"]["generation"]["generated_token_ids"],
            json!([7, 9])
        );
    }
    assert_eq!(
        shutdown(&mut running, "shutdown")["format"],
        "apxinf-mlx-cli-shutdown-v1"
    );
    let (code, trailing_stdout, stderr) = running.finish();
    assert_eq!(code, 0);
    assert_eq!(trailing_stdout, "");
    assert_eq!(stderr, "");
    assert_eq!(fs::read_to_string(&fixture.marker).unwrap(), "load\n");
}

#[cfg(unix)]
#[test]
fn explicit_shutdown_reaps_pipe_inheriting_descendants() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("fork-on-shutdown"), "yes\n").unwrap();
    let escaped = fixture.root.join("descendant-escaped");
    let mut running = fixture.start();
    running.ready();

    let started = Instant::now();
    assert_eq!(
        shutdown(&mut running, "shutdown-with-descendant")["format"],
        "apxinf-mlx-cli-shutdown-v1"
    );
    let (code, trailing_stdout, stderr) = running.finish();
    assert_eq!(code, 0);
    assert_eq!(trailing_stdout, "");
    assert_eq!(stderr, "");
    assert!(started.elapsed() < Duration::from_millis(750));

    thread::sleep(Duration::from_millis(1200));
    assert!(!escaped.exists());
}

#[test]
fn session_append_reset_and_model_error_are_application_scoped() {
    let fixture = Fixture::new();
    let mut running = fixture.start();
    running.ready();
    let mut create = request("create", "session_generate");
    create["session_id"] = json!("chat-1");
    let created = running.exchange(&create);
    assert_eq!(
        created["validated_service_receipt"]["session"]["prefix_token_count"],
        4
    );
    let mut branch = request("branch", "session_generate");
    branch["session_id"] = json!("chat-1");
    branch["prompt_token_ids"] = json!([1, 2, 7, 8, 3]);
    branch["max_tokens"] = json!(1);
    branch["stop_on_eos"] = json!(false);
    branch.as_object_mut().unwrap().remove("eos_token_id");
    let rejected_branch = running.exchange(&branch);
    assert_eq!(
        rejected_branch["format"],
        "apxinf-mlx-cli-response-error-v1"
    );
    assert_eq!(rejected_branch["error"]["code"], "invalid_request");
    assert_eq!(rejected_branch["error"]["recoverable"], true);
    let mut append = request("append", "session_generate");
    append["session_id"] = json!("chat-1");
    append["prompt_token_ids"] = json!([1, 2, 7, 9, 3]);
    append["max_tokens"] = json!(1);
    append["stop_on_eos"] = json!(false);
    append.as_object_mut().unwrap().remove("eos_token_id");
    let appended = running.exchange(&append);
    assert_eq!(
        appended["validated_service_receipt"]["session"]["reused_prefix_token_count"],
        4
    );
    assert_eq!(
        appended["validated_service_receipt"]["session"]["evaluated_prompt_token_count"],
        1
    );
    let reset = running.exchange(&json!({
        "format": "apxinf-mlx-cli-request-v1",
        "request_id": "reset",
        "operation": "session_reset",
        "session_id": "chat-1",
    }));
    assert_eq!(
        reset["validated_service_receipt"]["format"],
        "apxinf-mlx-cli-session-reset-result-v1"
    );
    assert_eq!(
        reset["validated_service_receipt"]["validated_service_receipt"]["format"],
        "apxinf-mlx-session-reset-v1"
    );

    let mut failure_create = request("failure-create", "session_generate");
    failure_create["session_id"] = json!("chat-fail");
    failure_create["prompt_token_ids"] = json!([3]);
    let failure_created = running.exchange(&failure_create);
    assert_eq!(
        failure_created["validated_service_receipt"]["session"]["prefix_token_count"],
        3
    );
    let mut failure_append = request("failure-append", "session_generate");
    failure_append["session_id"] = json!("chat-fail");
    failure_append["prompt_token_ids"] = json!([3, 7, 9, 666]);
    let session_error = running.exchange(&failure_append);
    assert_eq!(session_error["error"]["code"], "generation_failed");
    assert_eq!(session_error["error"]["recoverable"], true);
    let mut recreated = request("failure-recreate", "session_generate");
    recreated["session_id"] = json!("chat-fail");
    recreated["prompt_token_ids"] = json!([4]);
    let recreated = running.exchange(&recreated);
    assert_eq!(
        recreated["validated_service_receipt"]["request"]["operation"],
        "create"
    );

    let mut failing = request("model-error", "generate");
    failing["prompt_token_ids"] = json!([666]);
    let error = running.exchange(&failing);
    assert_eq!(error["format"], "apxinf-mlx-cli-response-error-v1");
    assert_eq!(error["error"]["code"], "generation_failed");
    assert_eq!(error["error"]["recoverable"], true);
    let recovered = running.exchange(&request("after-error", "generate"));
    assert_eq!(recovered["format"], "apxinf-mlx-cli-response-v1");

    shutdown(&mut running, "shutdown");
    let (code, _, stderr) = running.finish();
    assert_eq!(code, 0);
    assert_eq!(stderr, "");
    assert_eq!(fs::read_to_string(&fixture.marker).unwrap(), "load\n");
}

#[test]
fn unknown_worker_error_code_is_fatal() {
    let fixture = Fixture::new();
    let mut running = fixture.start();
    running.ready();
    let mut failing = request("unknown-worker-error", "generate");
    failing["prompt_token_ids"] = json!([667]);
    running.write_raw(&serde_json::to_vec(&failing).unwrap());

    let (code, stdout, stderr) = running.finish();
    assert_eq!(code, 3);
    assert_eq!(stdout, "");
    let fatal: Value = serde_json::from_str(&stderr).unwrap();
    assert_eq!(fatal["format"], "apxinf-mlx-cli-fatal-error-v1");
    assert_eq!(fatal["error"]["code"], "impossible_worker_code");
}

#[test]
fn malformed_duplicate_unknown_and_oversized_lines_fail_closed() {
    for (payload, expected_code) in [
        (
            br#"{"format":"apxinf-mlx-cli-request-v1","request_id":"a","request_id":"b","operation":"shutdown"}"#.as_slice(),
            "invalid_json",
        ),
        (
            br#"{"format":"apxinf-mlx-cli-request-v1","request_id":"a","operation":"shutdown","unknown":true}"#.as_slice(),
            "invalid_request",
        ),
    ] {
        let fixture = Fixture::new();
        let mut running = fixture.start();
        running.ready();
        running.write_raw(payload);
        let (code, stdout, stderr) = running.finish();
        assert_eq!(code, 2);
        assert_eq!(stdout, "");
        let fatal: Value = serde_json::from_str(&stderr).unwrap();
        assert_eq!(fatal["format"], "apxinf-mlx-cli-fatal-error-v1");
        assert_eq!(fatal["error"]["code"], expected_code);
    }

    let fixture = Fixture::new();
    let mut running = fixture.start();
    running.ready();
    running.write_raw(&vec![b'x'; 1024 * 1024 + 1]);
    let (code, stdout, stderr) = running.finish();
    assert_eq!(code, 2);
    assert_eq!(stdout, "");
    let fatal: Value = serde_json::from_str(&stderr).unwrap();
    assert_eq!(fatal["error"]["code"], "line_too_large");
}

#[cfg(unix)]
#[test]
fn bad_input_aborts_immediately_even_when_worker_ignores_shutdown() {
    for (oversized, expected_code) in [(false, "unexpected_eof"), (true, "line_too_large")] {
        let fixture = Fixture::new();
        fs::write(fixture.root.join("ignore-shutdown"), "yes\n").unwrap();
        let worker_pid = fixture.root.join("worker-pid");
        let mut running = fixture.start_with_timeout(3600);
        running.ready();
        let started = Instant::now();
        if oversized {
            running.write_raw(&vec![b'x'; 1024 * 1024 + 1]);
        } else {
            running.close_stdin();
        }
        let (code, stdout, stderr) =
            running.finish_bounded(Duration::from_millis(750), &worker_pid);
        assert!(started.elapsed() < Duration::from_millis(750));
        assert_eq!(code, 2);
        assert_eq!(stdout, "");
        let fatal: Value = serde_json::from_str(&stderr).unwrap();
        assert_eq!(fatal["error"]["code"], expected_code);
    }
}

#[test]
fn duplicate_request_id_and_eof_are_nonzero_and_reap_the_worker() {
    let fixture = Fixture::new();
    let mut running = fixture.start();
    running.ready();
    let first = request("same-id", "generate");
    assert_eq!(
        running.exchange(&first)["format"],
        "apxinf-mlx-cli-response-v1"
    );
    running.write_raw(&serde_json::to_vec(&first).unwrap());
    let (code, stdout, stderr) = running.finish();
    assert_eq!(code, 2);
    assert_eq!(stdout, "");
    let fatal: Value = serde_json::from_str(&stderr).unwrap();
    assert_eq!(fatal["error"]["code"], "duplicate_request_id");
    assert_eq!(fs::read_to_string(&fixture.marker).unwrap(), "load\n");

    let fixture = Fixture::new();
    let mut running = fixture.start();
    running.ready();
    running.close_stdin();
    let (code, stdout, stderr) = running.finish();
    assert_eq!(code, 2);
    assert_eq!(stdout, "");
    let fatal: Value = serde_json::from_str(&stderr).unwrap();
    assert_eq!(fatal["error"]["code"], "unexpected_eof");
    assert_eq!(fs::read_to_string(&fixture.marker).unwrap(), "load\n");
}

#[cfg(unix)]
#[test]
fn symlinked_runtime_is_rejected_before_the_worker_starts() {
    use std::os::unix::fs::symlink;

    let fixture = Fixture::new();
    let linked_python = fixture.root.join("linked-python");
    symlink(&fixture.python, &linked_python).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_apxinf"))
        .arg("mlx-serve")
        .arg("--model")
        .arg(&fixture.model)
        .arg("--mlx-python")
        .arg(&linked_python)
        .arg("--mlx-runner")
        .arg(&fixture.runner)
        .arg("--timeout-seconds")
        .arg("5")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let fatal: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(fatal["error"]["code"], "service_start_failed");
    assert!(!fixture.marker.exists());
}

const FAKE_RUNNER: &str = r#"import hashlib,json,os,pathlib,subprocess,sys,time
args=sys.argv[1:]
fake_python=pathlib.Path(args[args.index('--fake-python')+1]).resolve()
model=pathlib.Path(args[args.index('--model-dir')+1]).resolve()
runner=pathlib.Path(__file__).resolve()
helper=runner.with_name('apxinf_mlx_generate.py')
marker=pathlib.Path(__MARKER__)
marker.write_text((marker.read_text() if marker.exists() else '')+'load\n')
(runner.parent/'worker-pid').write_text(str(os.getpid()))
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
 if request['format']=='apxinf-mlx-service-control-v1':
  if (runner.parent/'ignore-shutdown').exists(): time.sleep(60)
  if (runner.parent/'fork-on-shutdown').exists():
   escaped=runner.parent/'descendant-escaped'
   subprocess.Popen(['/bin/sh','-c','sleep 1; : > "$1"','sh',str(escaped)])
  emit({'format':'apxinf-mlx-service-shutdown-v1','protocol':'apxinf-mlx-service-v1','request_id':request_id}); raise SystemExit(0)
 if request['format']=='apxinf-mlx-session-control-v1':
  previous=sessions.pop(request['session_id'])
  emit({'format':'apxinf-mlx-session-reset-v1','protocol':'apxinf-mlx-session-v1','request_id':request_id,'session_id':request['session_id'],'released_cache_bytes':64,'previous_prefix':{'format':'apxinf-mlx-session-prefix-v1','token_count':len(previous),'token_ids_sha256':token_hash(previous)},'binding':binding,'session_cache':{'policy':'exact-append-only-in-process-lru-v1','session_count':len(sessions),'total_cache_bytes':64*len(sessions),'max_sessions':4,'max_bytes':536870912}}); continue
 if 667 in request['prompt_token_ids']:
  emit({'format':'apxinf-mlx-service-response-error-v1','protocol':'apxinf-mlx-service-v1','request_id':request_id,'error':{'code':'impossible_worker_code','message':'injected unknown worker error'}}); continue
 if 666 in request['prompt_token_ids']:
  if request['format']=='apxinf-mlx-session-request-v1':
   sessions.pop(request['session_id'],None); emit({'format':'apxinf-mlx-session-response-error-v1','protocol':'apxinf-mlx-session-v1','request_id':request_id,'error':{'code':'generation_failed','message':'injected model failure'}}); continue
  emit({'format':'apxinf-mlx-service-response-error-v1','protocol':'apxinf-mlx-service-v1','request_id':request_id,'error':{'code':'generation_failed','message':'injected model failure'}}); continue
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

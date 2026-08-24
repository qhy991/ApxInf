from __future__ import annotations

import hashlib
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/apxinf_mlx_serve.py"
ONE_SHOT_TEST = ROOT / "tests/python/test_apxinf_mlx_generate.py"
SPEC = importlib.util.spec_from_file_location(
    "apxinf_mlx_generate_tests", ONE_SHOT_TEST
)
assert SPEC is not None and SPEC.loader is not None
ONE_SHOT_TESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ONE_SHOT_TESTS)
write_fake_runtime = ONE_SHOT_TESTS.write_fake_runtime


def write_session_fake_runtime(
    root: Path, tokens: list[int], *, cache_bytes: int = 64
) -> Path:
    packages = write_fake_runtime(root, tokens, eos_token_ids=())
    (packages / "mlx_lm/__init__.py").write_text(
        textwrap.dedent(
            f"""
            import json
            import os
            import sys

            class Cache:
                def __init__(self):
                    self.tokens = []

                @property
                def nbytes(self):
                    return {cache_bytes}

            class Model:
                def make_cache(self):
                    return [Cache()]

            class Tokenizer:
                eos_token_ids = []

            def load(path, **kwargs):
                print("dependency stdout noise")
                print("dependency stderr noise", file=sys.stderr)
                with open(os.environ["FAKE_MLX_CALL_LOG"], "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({{"event": "load", "path": path}}) + "\\n")
                return Model(), Tokenizer(), {{"model_type": "qwen3_5"}}
            """
        ),
        encoding="utf-8",
    )
    (packages / "mlx_lm/generate.py").write_text(
        textwrap.dedent(
            f"""
            import json
            import os

            TOKENS = {tokens!r}

            def generate_step(
                prompt, model, *, max_tokens, sampler=None, prompt_cache=None
            ):
                before = None if prompt_cache is None else list(prompt_cache[0].tokens)
                with open(os.environ["FAKE_MLX_CALL_LOG"], "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({{
                        "event": "generate",
                        "prompt": list(prompt),
                        "cache_before": before,
                    }}) + "\\n")
                if sampler is None:
                    raise TypeError("ApxInf must bind its explicit greedy sampler")
                if prompt_cache is not None:
                    if 666 in prompt:
                        prompt_cache[0].tokens.append(prompt[0])
                        raise RuntimeError("injected failure after partial cache mutation")
                    prompt_cache[0].tokens.extend(prompt)
                for token in TOKENS[:max_tokens]:
                    if prompt_cache is not None:
                        prompt_cache[0].tokens.append(token)
                    yield token, None

            def make_prompt_cache(model, max_kv_size=None):
                if max_kv_size is not None:
                    raise ValueError("fake does not support rotating cache")
                return model.make_cache()
            """
        ),
        encoding="utf-8",
    )
    return packages


def start_service(
    root: Path, packages: Path
) -> tuple[subprocess.Popen[str], dict[str, object]]:
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"qwen3_5"}', encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(packages)
    environment["FAKE_MLX_CALL_LOG"] = str(root / "calls.jsonl")
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--model-dir", str(model_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    ready = json.loads(process.stdout.readline())
    return process, ready


def exchange(process: subprocess.Popen[str], value: object) -> dict[str, object]:
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


def shutdown_service(
    process: subprocess.Popen[str], request_id: str = "shutdown"
) -> None:
    response = exchange(
        process,
        {
            "format": "apxinf-mlx-service-control-v1",
            "request_id": request_id,
            "operation": "shutdown",
        },
    )
    assert response["format"] == "apxinf-mlx-service-shutdown-v1"
    assert process.stderr is not None
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.wait(timeout=5) == 0
    assert process.stderr.read() == ""
    process.stdin.close()
    process.stdout.close()
    process.stderr.close()


def token_hash(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def session_binding(ready: dict[str, object]) -> dict[str, object]:
    model = ready["model"]
    session_cache = ready["session_cache"]
    assert isinstance(model, dict) and isinstance(model["bundle"], dict)
    assert isinstance(session_cache, dict)
    return {
        "format": "apxinf-mlx-session-binding-v1",
        "model_config_sha256": model["config_sha256"],
        "model_bundle_sha256": model["bundle"]["sha256"],
        "greedy_strategy": ready["greedy_strategy"],
        "cache_policy": session_cache["policy"],
    }


def session_request(
    ready: dict[str, object],
    *,
    request_id: str,
    session_id: str,
    operation: str,
    prompt: list[int],
    prefix: list[int],
    max_tokens: int = 1,
) -> dict[str, object]:
    return {
        "format": "apxinf-mlx-session-request-v1",
        "request_id": request_id,
        "session_id": session_id,
        "operation": operation,
        "prompt_token_ids": prompt,
        "expected_prefix": {
            "format": "apxinf-mlx-session-prefix-v1",
            "token_count": len(prefix),
            "token_ids_sha256": token_hash(prefix),
        },
        "binding": session_binding(ready),
        "max_tokens": max_tokens,
        "stop_on_eos": False,
    }


class MlxPersistentServiceTests(unittest.TestCase):
    def test_explicit_session_reuses_only_the_exact_cached_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packages = write_session_fake_runtime(root, [7])
            process, ready = start_service(root, packages)
            self.assertEqual(
                ready["session_cache"],
                {
                    "format": "apxinf-mlx-session-cache-ready-v1",
                    "protocol": "apxinf-mlx-session-v1",
                    "policy": "exact-append-only-in-process-lru-v1",
                    "request_format": "apxinf-mlx-session-request-v1",
                    "control_format": "apxinf-mlx-session-control-v1",
                    "max_sessions": 4,
                    "max_bytes": 536870912,
                },
            )

            first_prompt = [1, 2]
            first = exchange(
                process,
                session_request(
                    ready,
                    request_id="session-create",
                    session_id="chat-1",
                    operation="create",
                    prompt=first_prompt,
                    prefix=[],
                ),
            )
            self.assertEqual(first["format"], "apxinf-mlx-session-response-v1")
            self.assertEqual(first["protocol"], "apxinf-mlx-session-v1")
            first_context = [1, 2, 7]
            self.assertEqual(first["session"]["prefix_token_count"], 3)
            self.assertEqual(
                first["session"]["prefix_token_ids_sha256"],
                token_hash(first_context),
            )
            self.assertEqual(first["session"]["reused_prefix_token_count"], 0)
            self.assertEqual(first["session"]["evaluated_prompt_token_count"], 2)

            second_prompt = [*first_context, 3, 4]
            second = exchange(
                process,
                session_request(
                    ready,
                    request_id="session-append",
                    session_id="chat-1",
                    operation="append",
                    prompt=second_prompt,
                    prefix=first_context,
                ),
            )
            self.assertEqual(second["generation"]["generated_token_ids"], [7])
            self.assertEqual(second["session"]["reused_prefix_token_count"], 3)
            self.assertEqual(second["session"]["evaluated_prompt_token_count"], 2)
            self.assertEqual(second["session"]["prefix_token_count"], 6)

            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            generations = [call for call in calls if call["event"] == "generate"]
            self.assertEqual(
                generations,
                [
                    {"event": "generate", "prompt": [1, 2], "cache_before": []},
                    {
                        "event": "generate",
                        "prompt": [3, 4],
                        "cache_before": first_context,
                    },
                ],
            )
            shutdown_service(process)

    def test_session_mismatch_fails_closed_without_mutating_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packages = write_session_fake_runtime(root, [7])
            process, ready = start_service(root, packages)
            create = session_request(
                ready,
                request_id="create",
                session_id="chat-1",
                operation="create",
                prompt=[1, 2],
                prefix=[],
            )
            self.assertEqual(
                exchange(process, create)["format"],
                "apxinf-mlx-session-response-v1",
            )
            context = [1, 2, 7]

            wrong_prefix = session_request(
                ready,
                request_id="wrong-prefix",
                session_id="chat-1",
                operation="append",
                prompt=[1, 99, 7, 3],
                prefix=context,
            )
            rejected = exchange(process, wrong_prefix)
            self.assertEqual(rejected["error"]["code"], "session_prefix_mismatch")

            wrong_binding = session_request(
                ready,
                request_id="wrong-binding",
                session_id="chat-1",
                operation="append",
                prompt=[*context, 3],
                prefix=context,
            )
            wrong_binding["binding"]["greedy_strategy"] = "different-strategy"
            rejected = exchange(process, wrong_binding)
            self.assertEqual(rejected["error"]["code"], "session_binding_mismatch")

            accepted = exchange(
                process,
                session_request(
                    ready,
                    request_id="valid-after-rejections",
                    session_id="chat-1",
                    operation="append",
                    prompt=[*context, 3],
                    prefix=context,
                ),
            )
            self.assertEqual(accepted["format"], "apxinf-mlx-session-response-v1")
            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            generations = [call for call in calls if call["event"] == "generate"]
            self.assertEqual(len(generations), 2)
            self.assertEqual(generations[-1]["cache_before"], context)
            shutdown_service(process)

    def test_session_generation_failure_invalidates_partially_mutated_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packages = write_session_fake_runtime(root, [7])
            process, ready = start_service(root, packages)
            created = exchange(
                process,
                session_request(
                    ready,
                    request_id="create",
                    session_id="chat-1",
                    operation="create",
                    prompt=[1, 2],
                    prefix=[],
                ),
            )
            self.assertEqual(created["generation"]["generated_token_ids"], [7])
            context = [1, 2, 7]
            failed = exchange(
                process,
                session_request(
                    ready,
                    request_id="partial-failure",
                    session_id="chat-1",
                    operation="append",
                    prompt=[*context, 666, 8],
                    prefix=context,
                ),
            )
            self.assertEqual(failed["error"]["code"], "generation_failed")
            missing = exchange(
                process,
                session_request(
                    ready,
                    request_id="must-not-reuse-partial",
                    session_id="chat-1",
                    operation="append",
                    prompt=[*context, 9],
                    prefix=context,
                ),
            )
            self.assertEqual(missing["error"]["code"], "session_not_found")
            shutdown_service(process)

    def test_session_lru_eviction_and_explicit_reset_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packages = write_session_fake_runtime(root, [7], cache_bytes=300 << 20)
            process, ready = start_service(root, packages)
            first = exchange(
                process,
                session_request(
                    ready,
                    request_id="create-1",
                    session_id="chat-1",
                    operation="create",
                    prompt=[1],
                    prefix=[],
                ),
            )
            self.assertEqual(first["session_cache"]["evicted_session_ids"], [])
            second = exchange(
                process,
                session_request(
                    ready,
                    request_id="create-2",
                    session_id="chat-2",
                    operation="create",
                    prompt=[2],
                    prefix=[],
                ),
            )
            self.assertEqual(second["session_cache"]["evicted_session_ids"], ["chat-1"])
            self.assertLessEqual(
                second["session_cache"]["total_cache_bytes"],
                ready["session_cache"]["max_bytes"],
            )
            missing = exchange(
                process,
                session_request(
                    ready,
                    request_id="append-evicted",
                    session_id="chat-1",
                    operation="append",
                    prompt=[1, 7, 3],
                    prefix=[1, 7],
                ),
            )
            self.assertEqual(missing["error"]["code"], "session_not_found")

            reset = exchange(
                process,
                {
                    "format": "apxinf-mlx-session-control-v1",
                    "request_id": "reset-2",
                    "operation": "reset",
                    "session_id": "chat-2",
                    "expected_prefix": {
                        "format": "apxinf-mlx-session-prefix-v1",
                        "token_count": 2,
                        "token_ids_sha256": token_hash([2, 7]),
                    },
                    "binding": session_binding(ready),
                },
            )
            self.assertEqual(reset["format"], "apxinf-mlx-session-reset-v1")
            self.assertEqual(reset["released_cache_bytes"], 300 << 20)
            self.assertEqual(reset["session_cache"]["session_count"], 0)
            missing = exchange(
                process,
                session_request(
                    ready,
                    request_id="append-reset",
                    session_id="chat-2",
                    operation="append",
                    prompt=[2, 7, 4],
                    prefix=[2, 7],
                ),
            )
            self.assertEqual(missing["error"]["code"], "session_not_found")
            shutdown_service(process)

    def test_ordinary_request_never_receives_a_session_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packages = write_session_fake_runtime(root, [5])
            process, _ready = start_service(root, packages)
            response = exchange(
                process,
                {
                    "format": "apxinf-mlx-service-request-v1",
                    "request_id": "ordinary",
                    "prompt_token_ids": [1, 2],
                    "max_tokens": 1,
                    "stop_on_eos": False,
                },
            )
            self.assertEqual(response["format"], "apxinf-mlx-service-response-v1")
            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(calls[-1]["cache_before"], None)
            shutdown_service(process)

    def test_loads_once_serves_two_requests_and_shuts_down_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"model_type":"qwen3_5"}', encoding="utf-8"
            )
            packages = write_fake_runtime(root, [7, 9, 11])
            fake_loader = packages / "mlx_lm/__init__.py"
            fake_loader.write_text(
                fake_loader.read_text(encoding="utf-8").replace(
                    'os.environ["FAKE_MLX_CALL_LOG"], "w",',
                    'os.environ["FAKE_MLX_CALL_LOG"], "a",',
                ),
                encoding="utf-8",
            )
            call_log = root / "load.json"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(packages)
            environment["FAKE_MLX_CALL_LOG"] = str(call_log)

            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--model-dir", str(model_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            ready = json.loads(process.stdout.readline())
            self.assertEqual(ready["format"], "apxinf-mlx-service-ready-v1")
            self.assertEqual(ready["protocol"], "apxinf-mlx-service-v1")

            first = {
                "format": "apxinf-mlx-service-request-v1",
                "request_id": "request-1",
                "prompt_token_ids": [1, 2],
                "max_tokens": 3,
                "stop_on_eos": True,
            }
            process.stdin.write(json.dumps(first, separators=(",", ":")) + "\n")
            process.stdin.flush()
            first_response = json.loads(process.stdout.readline())
            self.assertEqual(first_response["request_id"], "request-1")
            self.assertEqual(
                first_response["generation"]["generated_token_ids"], [7, 9]
            )
            self.assertEqual(first_response["generation"]["stop_reason"], "eos")

            second = {
                "format": "apxinf-mlx-service-request-v1",
                "request_id": "request-2",
                "prompt_token_ids": [3],
                "max_tokens": 0,
                "stop_on_eos": True,
            }
            process.stdin.write(json.dumps(second, separators=(",", ":")) + "\n")
            process.stdin.flush()
            second_response = json.loads(process.stdout.readline())
            self.assertEqual(second_response["request_id"], "request-2")
            self.assertEqual(second_response["generation"]["generated_token_ids"], [])
            self.assertEqual(second_response["metrics"]["ttft_ms"], 0.0)

            process.stdin.write(json.dumps(second, separators=(",", ":")) + "\n")
            process.stdin.flush()
            duplicate = json.loads(process.stdout.readline())
            self.assertEqual(
                duplicate["format"], "apxinf-mlx-service-response-error-v1"
            )
            self.assertEqual(duplicate["request_id"], "request-2")
            self.assertEqual(duplicate["error"]["code"], "duplicate_request_id")

            shutdown = {
                "format": "apxinf-mlx-service-control-v1",
                "request_id": "shutdown-1",
                "operation": "shutdown",
            }
            process.stdin.write(json.dumps(shutdown, separators=(",", ":")) + "\n")
            process.stdin.flush()
            acknowledgement = json.loads(process.stdout.readline())
            self.assertEqual(
                acknowledgement,
                {
                    "format": "apxinf-mlx-service-shutdown-v1",
                    "protocol": "apxinf-mlx-service-v1",
                    "request_id": "shutdown-1",
                },
            )
            self.assertEqual(process.wait(timeout=5), 0)
            self.assertEqual(process.stderr.read(), "")
            self.assertEqual(call_log.read_text(encoding="utf-8").count('"path"'), 1)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()

    def test_invalid_request_is_scoped_and_does_not_kill_the_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"model_type":"qwen3_5"}', encoding="utf-8"
            )
            packages = write_fake_runtime(root, [5])
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(packages)
            environment["FAKE_MLX_CALL_LOG"] = str(root / "load.json")
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--model-dir", str(model_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            json.loads(process.stdout.readline())

            process.stdin.write(
                '{"format":"apxinf-mlx-service-request-v1",'
                '"request_id":"bad-1","prompt_token_ids":[1],'
                '"max_tokens":65537,"stop_on_eos":true}\n'
            )
            process.stdin.flush()
            rejected = json.loads(process.stdout.readline())
            self.assertEqual(rejected["format"], "apxinf-mlx-service-response-error-v1")
            self.assertEqual(rejected["request_id"], "bad-1")
            self.assertEqual(rejected["error"]["code"], "invalid_request")

            process.stdin.write(
                '{"format":"apxinf-mlx-service-request-v1",'
                '"request_id":"good-1","prompt_token_ids":[1],'
                '"max_tokens":1,"stop_on_eos":false}\n'
            )
            process.stdin.flush()
            accepted = json.loads(process.stdout.readline())
            self.assertEqual(accepted["format"], "apxinf-mlx-service-response-v1")
            self.assertEqual(accepted["generation"]["generated_token_ids"], [5])

            process.stdin.write(
                '{"format":"apxinf-mlx-service-control-v1",'
                '"request_id":"shutdown-2","operation":"shutdown"}\n'
            )
            process.stdin.flush()
            json.loads(process.stdout.readline())
            self.assertEqual(process.wait(timeout=5), 0)
            self.assertEqual(process.stderr.read(), "")
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()

    def test_duplicate_json_key_fails_closed_without_desynchronizing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"model_type":"qwen3_5"}', encoding="utf-8"
            )
            packages = write_fake_runtime(root, [5])
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(packages)
            environment["FAKE_MLX_CALL_LOG"] = str(root / "load.json")
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--model-dir", str(model_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            json.loads(process.stdout.readline())
            process.stdin.write(
                '{"format":"apxinf-mlx-service-request-v1",'
                '"request_id":"duplicate-1","prompt_token_ids":[1],'
                '"max_tokens":1,"max_tokens":2,"stop_on_eos":false}\n'
            )
            process.stdin.flush()

            self.assertEqual(process.stdout.readline(), "")
            self.assertEqual(process.wait(timeout=5), 2)
            error = json.loads(process.stderr.read())
            self.assertEqual(error["format"], "apxinf-mlx-generation-error-v1")
            self.assertEqual(error["error"]["code"], "invalid_json")
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()

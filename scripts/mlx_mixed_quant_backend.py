#!/usr/bin/env python3
"""Evaluation-only MLX mixed-quant materialization backend.

Every candidate is loaded from the frozen BF16 source, saved into its own
backend-owned temporary directory, statically inspected, and only then loaded
back from disk.  Public handles never expose the model object or temporary
path.  The backend does not publish model bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import stat
import sys
import tempfile
from typing import NoReturn, Protocol


MATERIALIZATION = "independent-saved-static-verified-reload-v1"
STATE_ALIGNED_HOOK_CONTRACT = "apxinf-mlx-state-aligned-capture-hook-v1"
_SHA256_LENGTH = 64


class BackendError(RuntimeError):
    """A fail-closed backend custody, materialization, or evaluation error."""


class StateAlignedCaptureHook(Protocol):
    """Minimal audited extension point missing from public MLX-LM 0.31.3.

    A version-and-source-SHA-bound read-only manual-forward implementation is
    permitted.  It must run each candidate module on the BF16 reference
    module's frozen teacher-prefix input without monkeypatching, ``update_modules``,
    or otherwise replacing modules on either live model.  The runner validates
    the returned integer-only screening receipt.
    """

    contract: str

    def screen(
        self,
        reference_model: object,
        candidate_model: object,
        cache_dir: Path,
        *,
        trace: dict[str, object],
        candidate_modules: list[dict[str, object]],
        transition: dict[str, object] | None,
    ) -> dict[str, object]: ...


def _fail(message: str) -> NoReturn:
    raise BackendError(message)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BackendError(
            f"backend identity is not canonical JSON: {error}"
        ) from error


def _detached(value: object) -> object:
    return json.loads(_canonical_bytes(value))


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_regular_bytes(path: Path, label: str, maximum_bytes: int) -> bytes:
    """Read one bounded single-link regular file without following a symlink."""

    try:
        before = path.lstat()
    except OSError as error:
        raise BackendError(f"cannot inspect {label}: {error}") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > maximum_bytes
    ):
        _fail(f"{label} is not a bounded single-link regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BackendError(f"cannot open {label}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        stable_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_opened = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if stable_opened != stable_before:
            _fail(f"{label} changed while it was opened")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(4 * 1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        stable_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(payload) > maximum_bytes or stable_after != stable_opened:
            _fail(f"{label} changed while it was read")
        return payload
    finally:
        os.close(descriptor)


@dataclass
class _LiveBundle:
    handle: dict[str, object]
    build_dir: Path
    bundle_dir: Path
    loaded: object
    kind: str


@dataclass
class _MlxLoaded:
    api: object
    model: object
    tokenizer: object
    config: dict[str, object]
    bundle_dir: Path


class PinnedMlxAdapter:
    """Pinned MLX-LM adapter; state capture requires an audited explicit hook.

    MLX-LM 0.31.3 exposes generation and model execution but no stable public
    per-module counterfactual-output capture API.  Consequently this adapter
    never guesses at internal module structure: ``screen_state_aligned`` fails
    closed unless a separately audited hook with ``STATE_ALIGNED_HOOK_CONTRACT``
    is injected.
    """

    def __init__(
        self,
        builder_api: object,
        *,
        mlx_api: object | None = None,
        state_aligned_hook: StateAlignedCaptureHook | None = None,
    ) -> None:
        self._builder = builder_api
        self._api = mlx_api
        self._hook = state_aligned_hook

    def _selected_api(self) -> object:
        if self._api is None:
            try:
                self._api = self._builder._load_mlx_api()
            except Exception as error:
                raise BackendError(f"cannot load pinned MLX-LM API: {error}") from error
        return self._api

    def runtime_identity(self) -> dict[str, object]:
        """Hash CPython plus every pinned distribution RECORD manifest."""

        expected = getattr(self._builder, "PINNED_PACKAGES", None)
        if type(expected) is not dict or len(expected) != 8:
            _fail("pinned MLX adapter requires exactly eight locked packages")
        packages: list[dict[str, object]] = []
        for name in sorted(expected):
            try:
                distribution = metadata.distribution(name)
            except metadata.PackageNotFoundError as error:
                raise BackendError(f"runtime package is unavailable: {name}") from error
            version = distribution.version
            record = distribution.read_text("RECORD")
            if version != expected[name] or record is None:
                _fail(f"runtime package {name} is not the pinned RECORD artifact")
            packages.append(
                {
                    "name": name,
                    "version": version,
                    "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
                }
            )
        try:
            executable = Path(sys.executable).resolve(strict=True)
            executable_payload = executable.read_bytes()
        except OSError as error:
            raise BackendError(
                f"cannot hash the CPython executable: {error}"
            ) from error
        return {
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "executable_sha256": hashlib.sha256(executable_payload).hexdigest(),
            },
            "packages": packages,
            "offline": True,
            # Python socket interception is useful defense-in-depth, but is not
            # an OS-enforced network sandbox and must not be represented as one.
            "network_blocked": False,
            "trust_remote_code": False,
        }

    @staticmethod
    def _load_local(
        api: object, directory: Path
    ) -> tuple[object, object, dict[str, object]]:
        try:
            value = api.load(
                str(directory),
                tokenizer_config={
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
                lazy=False,
                return_config=True,
            )
        except Exception as error:
            raise BackendError(f"local MLX-LM load failed: {error}") from error
        if type(value) is not tuple or len(value) != 3 or type(value[2]) is not dict:
            _fail("local MLX-LM load returned an unexpected value")
        return value

    def save_candidate(
        self,
        source: object,
        bundle_dir: Path,
        cache_dir: Path,
        *,
        mode: str,
        selective: dict[str, object] | None,
    ) -> None:
        api = self._selected_api()
        with self._builder._offline_runtime(cache_dir):
            model, tokenizer, config = self._load_local(api, source.directory)
            expected_model_type = getattr(self._builder, "SUPPORTED_MODEL_TYPE", None)
            if expected_model_type is not None and config.get("model_type") != (
                expected_model_type
            ):
                _fail("loaded MLX model type drifted from the certified source")
            if mode == "affine-w4-g64":
                tiers = selective.get("tiers") if type(selective) is dict else None
                if type(tiers) is not dict or not tiers:
                    _fail("mixed candidate lacks a frozen W4/W8/BF16 tier map")

                def quant_predicate(path: str, _module: object) -> object:
                    tier = tiers.get(path)
                    if tier == "w4":
                        return True
                    if tier == "w8":
                        return {"bits": 8, "group_size": 64, "mode": "affine"}
                    if tier == "bf16":
                        return False
                    raise BackendError(
                        f"MLX exposed a module outside the frozen candidate set: {path}"
                    )

                try:
                    quantized = api.quantize_model(
                        model,
                        config,
                        group_size=64,
                        bits=4,
                        mode="affine",
                        quant_predicate=quant_predicate,
                    )
                except BackendError:
                    raise
                except Exception as error:
                    raise BackendError(
                        f"pinned mixed quantization failed: {error}"
                    ) from error
                if (
                    type(quantized) is not tuple
                    or len(quantized) != 2
                    or type(quantized[1]) is not dict
                ):
                    _fail("pinned mixed quantization returned an unexpected value")
                model, config = quantized
                config[getattr(self._builder, "SELECTIVE_CONFIG_KEY")] = _detached(
                    selective["config_manifest"]
                )
            elif mode != "mixed-bf16" or selective is not None:
                _fail("unsupported evaluation-only materialization mode")
            try:
                api.save(
                    bundle_dir,
                    str(source.directory),
                    model,
                    tokenizer,
                    config,
                    donate_model=True,
                )
            except Exception as error:
                raise BackendError(f"pinned MLX-LM save failed: {error}") from error
        if not bundle_dir.is_dir() or bundle_dir.is_symlink():
            _fail("pinned MLX-LM did not save a regular bundle directory")
        tokenizer_payloads = getattr(source, "tokenizer_payloads", None)
        if type(tokenizer_payloads) is dict:
            writer = getattr(self._builder, "_write_private_regular", None)
            if tokenizer_payloads and not callable(writer):
                _fail("builder cannot restore byte-exact tokenizer artifacts")
            for name, payload in tokenizer_payloads.items():
                writer(bundle_dir / name, payload)
        for entry in os.scandir(bundle_dir):
            if not entry.is_file(follow_symlinks=False):
                _fail("saved MLX bundle contains a non-regular entry")
            os.chmod(entry.path, 0o600, follow_symlinks=False)

    def reload_candidate(
        self,
        bundle_dir: Path,
        cache_dir: Path,
        *,
        mode: str,
        selective: dict[str, object] | None,
    ) -> _MlxLoaded:
        api = self._selected_api()
        with self._builder._offline_runtime(cache_dir):
            model, tokenizer, config = self._load_local(api, bundle_dir)
        try:
            self._builder._validate_output_config(config, mode, None, selective)
        except Exception as error:
            raise BackendError(
                f"reloaded MLX config failed validation: {error}"
            ) from error
        return _MlxLoaded(api, model, tokenizer, config, bundle_dir)

    def inspect_saved(
        self,
        source: object,
        bundle_dir: Path,
        *,
        mode: str,
        selective: dict[str, object] | None,
    ) -> dict[str, object]:
        """Run the builder's full static schema and content inspection."""

        try:
            self._builder._assert_source_unchanged(source)
            result = self._builder._inspect_output(
                bundle_dir, source, mode, None, selective
            )
            self._builder._assert_source_unchanged(source)
        except Exception as error:
            raise BackendError(
                f"saved candidate failed static inspection: {error}"
            ) from error
        if type(result) is not tuple or len(result) != 2 or type(result[1]) is not dict:
            _fail("builder static inspection returned an unexpected value")
        evidence = result[1]
        if not _is_sha256(evidence.get("manifest_sha256")):
            _fail("builder static inspection omitted the bundle manifest")
        return _detached(evidence)  # type: ignore[return-value]

    @staticmethod
    def _token_list(value: object, label: str, steps: int) -> list[int]:
        if type(value) is not list or len(value) != steps:
            _fail(f"{label} returned an unexpected token count")
        result: list[int] = []
        for token in value:
            if type(token) is bool:
                _fail(f"{label} returned a boolean token")
            try:
                result.append(int(token))
            except (TypeError, ValueError, OverflowError) as error:
                raise BackendError(f"{label} returned an invalid token") from error
        return result

    def evaluate_gate(
        self,
        loaded: _MlxLoaded,
        cache_dir: Path,
        *,
        trace: dict[str, object],
        role: str,
    ) -> dict[str, object]:
        """Run two 128-step teacher-forced and asynchronous greedy lanes."""

        if not isinstance(loaded, _MlxLoaded):
            _fail("pinned gate requires a reloaded MLX bundle")
        if loaded.model is None or loaded.tokenizer is None:
            _fail("cannot evaluate a closed MLX bundle")
        prompt_ids = trace.get("prompt_token_ids")
        teacher_ids = trace.get("teacher_token_ids")
        if (
            trace.get("api") != "mlx_lm.generate.generate_step"
            or trace.get("semantics") != "mlx-generate-step-argmax-v1"
            or trace.get("teacher_steps") != 128
            or trace.get("free_run_steps") != 128
            or trace.get("repeat_count") != 2
            or type(prompt_ids) is not list
            or not prompt_ids
            or type(teacher_ids) is not list
            or len(teacher_ids) != 128
            or any(type(token) is not int for token in [*prompt_ids, *teacher_ids])
        ):
            _fail("pinned MLX gate trace drifted from the frozen double-128 contract")
        api = loaded.api
        teacher_runs: list[list[int]] = []
        async_runs: list[list[int]] = []

        def greedy_argmax(logprobs: object) -> object:
            return api.argmax(logprobs, axis=-1)

        with self._builder._offline_runtime(cache_dir):
            for repeat in range(2):
                try:
                    raw_teacher = api.teacher_forced_step(
                        api.array(prompt_ids), loaded.model, list(teacher_ids)
                    )
                except Exception as error:
                    raise BackendError(
                        f"{role} teacher-forced repeat {repeat + 1} failed: {error}"
                    ) from error
                teacher_runs.append(
                    self._token_list(raw_teacher, f"{role} teacher-forced gate", 128)
                )
            for repeat in range(2):
                try:
                    generated = api.generate_step(
                        api.array(prompt_ids),
                        loaded.model,
                        max_tokens=128,
                        sampler=greedy_argmax,
                    )
                    raw_async: list[object] = []
                    for item in generated:
                        if type(item) is not tuple or len(item) != 2:
                            _fail("mlx_lm.generate_step yielded an unexpected value")
                        raw_async.append(item[0])
                except BackendError:
                    raise
                except Exception as error:
                    raise BackendError(
                        f"{role} asynchronous repeat {repeat + 1} failed: {error}"
                    ) from error
                async_runs.append(
                    self._token_list(raw_async, f"{role} asynchronous gate", 128)
                )
        return {
            "api": trace["api"],
            "semantics": trace["semantics"],
            "prompt_token_ids": list(prompt_ids),
            "teacher_forced_token_ids": teacher_runs,
            "async_free_run_token_ids": async_runs,
        }

    def screen_state_aligned(
        self,
        reference: _MlxLoaded,
        candidate: _MlxLoaded,
        cache_dir: Path,
        *,
        trace: dict[str, object],
        candidate_modules: list[dict[str, object]],
        transition: dict[str, object] | None,
    ) -> dict[str, object]:
        if (
            not isinstance(reference, _MlxLoaded)
            or not isinstance(candidate, _MlxLoaded)
            or reference.model is None
            or candidate.model is None
        ):
            _fail("state-aligned capture requires two live reloaded MLX bundles")
        hook = self._hook
        if hook is None or getattr(hook, "contract", None) != (
            STATE_ALIGNED_HOOK_CONTRACT
        ):
            _fail(
                "mlx-lm 0.31.3 has no audited state-aligned capture hook; "
                f"inject {STATE_ALIGNED_HOOK_CONTRACT!r} to enable screening"
            )
        try:
            value = hook.screen(
                reference.model,
                candidate.model,
                cache_dir,
                trace=_detached(trace),
                candidate_modules=_detached(candidate_modules),
                transition=_detached(transition),
            )
        except BackendError:
            raise
        except Exception as error:
            raise BackendError(
                f"audited state-aligned capture hook failed: {error}"
            ) from error
        return _detached(value)  # type: ignore[return-value]

    @staticmethod
    def close_loaded(loaded: _MlxLoaded) -> None:
        """Release the model/tokenizer references owned by one live handle."""

        if not isinstance(loaded, _MlxLoaded):
            _fail("cannot close an object not loaded by the pinned MLX adapter")
        if loaded.model is None and loaded.tokenizer is None:
            _fail("cannot close an already closed MLX bundle")
        loaded.model = None
        loaded.tokenizer = None


class MlxMixedQuantBackend:
    """Materialize one certified search generation without publishing it."""

    def __init__(
        self,
        certification: object,
        scratch_root: str | os.PathLike[str],
        adapter: object,
    ) -> None:
        required = {
            "generation",
            "builder_api",
            "source_bundle",
            "policy_path",
        }
        if any(not hasattr(certification, name) for name in required):
            _fail("backend requires a locally certified generation")
        self._certification = certification
        self._generation = certification.generation
        self._builder = certification.builder_api
        self._source = certification.source_bundle
        self._policy_path = Path(certification.policy_path)
        self._adapter = adapter
        self._live: dict[str, _LiveBundle] = {}
        self._failed = False

        scratch = Path(scratch_root)
        try:
            scratch_info = scratch.lstat()
            resolved_scratch = scratch.resolve(strict=True)
        except OSError as error:
            raise BackendError(
                f"cannot inspect backend scratch root: {error}"
            ) from error
        if (
            scratch != resolved_scratch
            or stat.S_ISLNK(scratch_info.st_mode)
            or not stat.S_ISDIR(scratch_info.st_mode)
            or scratch_info.st_uid != os.getuid()
        ):
            _fail("backend scratch root must be an absolute owned real directory")
        source_directory = Path(self._source.directory).resolve(strict=True)
        if (
            scratch == source_directory
            or source_directory in scratch.parents
            or scratch in source_directory.parents
        ):
            _fail("backend scratch root and frozen source must not overlap")
        self._scratch = scratch

        self._runtime_identity = self._validate_runtime_identity(
            self._adapter.runtime_identity()
        )
        self._check_certified_identity()
        self._selective = self._load_selective_descriptor()
        self._session_root = Path(
            tempfile.mkdtemp(prefix=".apxinf-mlx-mixed-backend-", dir=self._scratch)
        ).resolve(strict=True)
        self._session_root.chmod(0o700)

    @property
    def certified_generation(self) -> object:
        """Return the exact generation object accepted by backend methods."""

        return self._generation

    def _validate_runtime_identity(self, value: object) -> dict[str, object]:
        if type(value) is not dict or set(value) != {
            "python",
            "packages",
            "offline",
            "network_blocked",
            "trust_remote_code",
        }:
            _fail("MLX runtime identity fields drifted")
        python = value.get("python")
        if type(python) is not dict or set(python) != {
            "implementation",
            "version",
            "executable_sha256",
        }:
            _fail("Python runtime identity fields drifted")
        if (
            python.get("implementation") != "CPython"
            or type(python.get("version")) is not str
            or not _is_sha256(python.get("executable_sha256"))
            or value.get("offline") is not True
            or value.get("trust_remote_code") is not False
            or type(value.get("network_blocked")) is not bool
        ):
            _fail("MLX runtime is not the pinned local-only runtime")
        expected = getattr(self._builder, "PINNED_PACKAGES", None)
        packages = value.get("packages")
        if (
            type(expected) is not dict
            or len(expected) != 8
            or type(packages) is not list
        ):
            _fail("backend requires the exact eight-package runtime lock")
        normalized: list[dict[str, object]] = []
        for item in packages:
            if type(item) is not dict or set(item) != {
                "name",
                "version",
                "record_sha256",
            }:
                _fail("runtime package identity fields drifted")
            name = item.get("name")
            version = item.get("version")
            if (
                type(name) is not str
                or version != expected.get(name)
                or not _is_sha256(item.get("record_sha256"))
            ):
                _fail("runtime package identity drifted from the eight-package lock")
            normalized.append(dict(item))
        if [item["name"] for item in normalized] != sorted(expected):
            _fail("runtime package identity is incomplete, duplicated, or unsorted")
        return _detached(value)  # type: ignore[return-value]

    def _check_certified_identity(self) -> None:
        if self._failed:
            _fail("backend was invalidated by an earlier failure")
        if self._validate_runtime_identity(self._adapter.runtime_identity()) != (
            self._runtime_identity
        ):
            _fail("MLX runtime identity changed during evaluation")
        try:
            self._builder._assert_source_unchanged(self._source)
        except Exception as error:
            raise BackendError(f"frozen source identity changed: {error}") from error
        payload = _read_regular_bytes(
            self._policy_path,
            "mixed policy",
            int(getattr(self._builder, "MAX_JSON_BYTES", 64 * 1024 * 1024)),
        )
        generation = self._generation
        if hashlib.sha256(payload).hexdigest() != generation.policy_artifact_sha256:
            _fail("mixed policy artifact changed during evaluation")
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BackendError(f"mixed policy is not valid JSON: {error}") from error
        expected_document = generation.inputs.get("policy_document_sha256")
        if hashlib.sha256(_canonical_bytes(document)).hexdigest() != expected_document:
            _fail("mixed policy document identity changed during evaluation")

    def _load_selective_descriptor(self) -> dict[str, object]:
        policy = self._generation.policy
        source_policy = policy.get("source") if type(policy) is dict else None
        revision = (
            source_policy.get("revision") if type(source_policy) is dict else None
        )
        if type(revision) is not str:
            _fail("certified policy source revision is unavailable")
        try:
            selective = self._builder._load_selective_policy(
                self._source,
                str(self._policy_path),
                revision,
                "affine-w4-g64",
            )
        except Exception as error:
            raise BackendError(f"cannot bind selective policy: {error}") from error
        if (
            type(selective) is not dict
            or selective.get("policy_sha256") != self._generation.policy_sha256
            or selective.get("source_manifest_sha256")
            != self._generation.inputs.get("source_manifest_sha256")
        ):
            _fail("selective materialization facts drifted from certification")
        return _detached(selective)  # type: ignore[return-value]

    def _require_generation(self, certified: object) -> None:
        if certified is not self._generation:
            _fail("backend method received a different certified generation")
        try:
            self._check_certified_identity()
        except Exception as error:
            self._failed = True
            if not self._live and hasattr(self, "_session_root"):
                try:
                    self._remove_session_root()
                except Exception as cleanup_error:
                    try:
                        error.add_note(
                            f"backend session cleanup failed: {cleanup_error}"
                        )
                    except AttributeError:
                        pass
            raise

    def _new_build_directory(self) -> tuple[Path, Path, Path]:
        if not self._session_root.is_dir():
            _fail("backend temporary session is closed")
        build = Path(
            tempfile.mkdtemp(prefix="candidate-", dir=self._session_root)
        ).resolve(strict=True)
        build.chmod(0o700)
        cache = build / "runtime-cache"
        cache.mkdir(mode=0o700)
        return build, build / "bundle", cache

    def _remove_build_directory(self, build: Path) -> None:
        try:
            info = build.lstat()
        except FileNotFoundError:
            return
        if (
            build.parent != self._session_root
            or not build.name.startswith("candidate-")
            or stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
        ):
            _fail("refusing to clean an unexpected candidate directory")
        shutil.rmtree(build)

    def _materialize(
        self,
        certified: object,
        *,
        mode: str,
        selective: dict[str, object] | None,
        transition: dict[str, object] | None,
        kind: str,
    ) -> dict[str, object]:
        self._require_generation(certified)
        build, bundle, cache = self._new_build_directory()
        loaded: object | None = None
        try:
            self._adapter.save_candidate(
                self._source,
                bundle,
                cache,
                mode=mode,
                selective=_detached(selective),
            )
            self._check_certified_identity()
            evidence = self._adapter.inspect_saved(
                self._source,
                bundle,
                mode=mode,
                selective=_detached(selective),
            )
            manifest = (
                evidence.get("manifest_sha256") if type(evidence) is dict else None
            )
            if not _is_sha256(manifest):
                _fail("static candidate inspection did not return a manifest SHA-256")
            loaded = self._adapter.reload_candidate(
                bundle,
                cache,
                mode=mode,
                selective=_detached(selective),
            )
            if loaded is None:
                _fail("candidate reload returned no model")
            self._check_certified_identity()
            handle_id = secrets.token_hex(16)
            handle: dict[str, object] = {
                "handle_id": handle_id,
                "manifest_sha256": manifest,
                "policy_sha256": self._generation.policy_sha256,
                "evaluation_only": True,
                "publishable": False,
                "materialization": MATERIALIZATION,
            }
            if transition is not None:
                handle["transition"] = _detached(transition)
            self._live[handle_id] = _LiveBundle(
                handle=_detached(handle),  # type: ignore[arg-type]
                build_dir=build,
                bundle_dir=bundle,
                loaded=loaded,
                kind=kind,
            )
            return _detached(handle)  # type: ignore[return-value]
        except Exception as error:
            self._failed = True
            cleanup_failures: list[Exception] = []
            if loaded is not None:
                try:
                    self._adapter.close_loaded(loaded)
                except Exception as cleanup_error:
                    cleanup_failures.append(cleanup_error)
            try:
                self._remove_build_directory(build)
            except Exception as cleanup_error:
                cleanup_failures.append(cleanup_error)
            if not self._live:
                try:
                    self._remove_session_root()
                except Exception as cleanup_error:
                    cleanup_failures.append(cleanup_error)
            if cleanup_failures:
                message = "; ".join(
                    " ".join(str(failure).split())[:256] or type(failure).__name__
                    for failure in cleanup_failures
                )
                try:
                    error.add_note(f"candidate cleanup failed: {message}")
                except AttributeError:
                    pass
            raise

    def materialize_current(self, certified: object) -> dict[str, object]:
        """Save, inspect, and reload the policy's current mixed candidate."""

        return self._materialize(
            certified,
            mode="affine-w4-g64",
            selective=self._selective,
            transition=None,
            kind="current-candidate",
        )

    def open_bf16_reference(self, certified: object) -> dict[str, object]:
        """Save, inspect, and reload an independent BF16 reference bundle."""

        return self._materialize(
            certified,
            mode="mixed-bf16",
            selective=None,
            transition=None,
            kind="bf16-reference",
        )

    def materialize_counterfactual(
        self,
        certified: object,
        transition: dict[str, object],
    ) -> dict[str, object]:
        """Save and reload one W4→W8 or W8→BF16 evaluation candidate."""

        self._require_generation(certified)
        if type(transition) is not dict or set(transition) != {"path", "from", "to"}:
            _fail("counterfactual transition fields drifted")
        path = transition.get("path")
        previous = transition.get("from")
        following = transition.get("to")
        tiers = self._selective.get("tiers")
        if (
            type(path) is not str
            or type(tiers) is not dict
            or tiers.get(path) != previous
            or (previous, following) not in {("w4", "w8"), ("w8", "bf16")}
        ):
            _fail("counterfactual is not the next tier for one frozen module")
        changed = _detached(self._selective)
        assert type(changed) is dict
        changed_tiers = changed["tiers"]
        assert type(changed_tiers) is dict
        changed_tiers[path] = following
        changed["w4_paths"] = sorted(
            name for name, tier in changed_tiers.items() if tier == "w4"
        )
        changed["w8_paths"] = sorted(
            name for name, tier in changed_tiers.items() if tier == "w8"
        )
        changed["retained_bf16_paths"] = sorted(
            name for name, tier in changed_tiers.items() if tier == "bf16"
        )
        try:
            changed["weight_ledger"] = self._builder._selective_weight_ledger(
                self._source.tensor_schema, changed_tiers
            )
            changed["config_manifest"] = self._builder._selective_config_manifest(
                changed
            )
        except Exception as error:
            raise BackendError(
                f"cannot construct counterfactual static descriptor: {error}"
            ) from error
        return self._materialize(
            certified,
            mode="affine-w4-g64",
            selective=changed,
            transition=_detached(transition),  # type: ignore[arg-type]
            kind="counterfactual",
        )

    def _require_live(self, handle: object) -> _LiveBundle:
        if type(handle) is not dict or type(handle.get("handle_id")) is not str:
            _fail("backend received an invalid handle")
        live = self._live.get(handle["handle_id"])
        if live is None or live.handle != handle:
            _fail("backend handle is unknown or changed")
        return live

    def evaluate_gate(
        self,
        handle: dict[str, object],
        *,
        certified: object,
        role: str,
    ) -> dict[str, object]:
        """Run the frozen double gate on a registered, reloaded bundle."""

        self._require_generation(certified)
        live = self._require_live(handle)
        expected_kind = {
            "bf16-reference": "bf16-reference",
            "current-candidate": "current-candidate",
            "selected-counterfactual": "counterfactual",
        }.get(role)
        if live.kind != expected_kind:
            _fail("gate role does not match the registered bundle kind")
        trace = self._generation.policy.get("trace")
        if type(trace) is not dict:
            _fail("certified gate trace is unavailable")
        try:
            value = self._adapter.evaluate_gate(
                live.loaded,
                live.build_dir / "runtime-cache",
                trace=_detached(trace),
                role=role,
            )
        except BackendError:
            raise
        except Exception as error:
            raise BackendError(f"MLX quality gate failed: {error}") from error
        self._check_certified_identity()
        return _detached(value)  # type: ignore[return-value]

    def screen_state_aligned(
        self,
        reference: dict[str, object],
        candidate: dict[str, object],
        *,
        certified: object,
        transition: dict[str, object] | None,
    ) -> dict[str, object]:
        """Run an audited BF16-teacher-prefix-aligned capture hook."""

        self._require_generation(certified)
        reference_live = self._require_live(reference)
        candidate_live = self._require_live(candidate)
        if reference_live.kind != "bf16-reference" or candidate_live.kind not in {
            "current-candidate",
            "counterfactual",
        }:
            _fail("state-aligned screen requires BF16 reference and mixed candidate")
        expected_transition = candidate_live.handle.get("transition")
        if expected_transition != transition:
            _fail("state-aligned transition does not match the candidate bundle")
        trace = self._generation.policy.get("trace")
        candidates = self._generation.policy.get("candidate_modules")
        if type(trace) is not dict or type(candidates) is not list:
            _fail("certified state-aligned inputs are unavailable")
        try:
            value = self._adapter.screen_state_aligned(
                reference_live.loaded,
                candidate_live.loaded,
                candidate_live.build_dir / "runtime-cache",
                trace=_detached(trace),
                candidate_modules=_detached(candidates),
                transition=_detached(transition),
            )
        except BackendError:
            raise
        except Exception as error:
            raise BackendError(f"state-aligned capture failed: {error}") from error
        self._check_certified_identity()
        return _detached(value)  # type: ignore[return-value]

    def _remove_session_root(self) -> None:
        try:
            info = self._session_root.lstat()
        except FileNotFoundError:
            return
        if (
            self._session_root.parent != self._scratch
            or not self._session_root.name.startswith(".apxinf-mlx-mixed-backend-")
            or stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
        ):
            _fail("refusing to clean an unexpected backend session directory")
        shutil.rmtree(self._session_root)

    def close(self, handle: dict[str, object]) -> None:
        """Unload and remove exactly one backend-owned candidate handle."""

        if type(handle) is not dict or type(handle.get("handle_id")) is not str:
            _fail("cannot close an invalid backend handle")
        handle_id = handle["handle_id"]
        live = self._live.pop(handle_id, None)
        if live is None or live.handle != handle:
            _fail("cannot close an unknown, changed, or already closed handle")
        failure: Exception | None = None
        try:
            self._adapter.close_loaded(live.loaded)
        except Exception as error:
            failure = error
        try:
            self._remove_build_directory(live.build_dir)
        except Exception as error:
            if failure is None:
                failure = error
        if not self._live:
            try:
                self._remove_session_root()
            except Exception as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise BackendError(f"candidate cleanup failed: {failure}") from failure

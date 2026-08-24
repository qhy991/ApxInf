#!/usr/bin/env python3
"""Offline validator for the fixed MLX multi-prompt quality contract.

This module never imports MLX, opens model weights, or accesses the network.
It validates frozen raw-token prompts and, later in the file, synthetic or
Host-produced reference/candidate trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import NoReturn


CONTRACT_FORMAT = "apxinf-mlx-multi-prompt-quality-contract-v1"
EVIDENCE_FORMAT = "apxinf-mlx-multi-prompt-quality-evidence-v1"
RECEIPT_FORMAT = "apxinf-mlx-multi-prompt-quality-receipt-v1"
RUN_ENVELOPE_FORMAT = "apxinf-mlx-multi-prompt-quality-run-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = {
    "format",
    "schema_version",
    "model",
    "generation",
    "suite",
    "admission",
    "content_sha256",
}
_MODEL = {
    "repo_id": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "tokenizer_sha256": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "vocab_size": 248320,
}
_GENERATION = {
    "api": "mlx_lm.generate.generate_step",
    "semantics": "mlx-generate-step-argmax-v1",
    "sampler": "mx.argmax(logprobs,axis=-1)",
    "evaluation": "asynchronous",
    "prompt_prefill": "native-prompt-chunking",
    "stop_policy": "fixed-steps-no-eos-stop",
    "reference_precision": "bf16",
    "repeat_count": 2,
}
_PROMPT_FIELDS = {
    "id",
    "domain",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "teacher_steps",
}
_PROMPTS = (
    (
        "english-explanation",
        "english",
        "4d7bdaa5aface0377775ae53769765d715138de2c6dd7f5c3b118f2994495abc",
        64,
    ),
    (
        "chinese-explanation",
        "chinese",
        "2831f3f47ee9fa92a0f819505fee7f0d86301e7a25aacce2e4a40f94bcd7dcb5",
        64,
    ),
    (
        "python-code",
        "code",
        "f3bc042a561982e7e9f06355b2d9eb9857df617a2e0b8250b1ca830a009c3810",
        32,
    ),
    (
        "math-structured-json",
        "math_structured",
        "6bb5aebf1cd0faa5a0b05153f2e653e02c0b73294cc7bf46b18e72df45ecc1dd",
        32,
    ),
)
_ADMISSION = {
    "exact": {
        "claim": "fixed-suite-exact-parity",
        "required_prompt_pass_rate": 1.0,
    },
    "threshold": {
        "claim": "fixed-suite-threshold-match",
        "minimum_exact_prefix_tokens": 8,
        "minimum_position_match_ratio": 0.75,
        "required_prompt_pass_rate": 1.0,
    },
    "forbidden_claims": [
        "general-parity",
        "universal-parity",
        "model-parity",
        "all-prompts-parity",
    ],
    "claims_general_parity": False,
}
_EVIDENCE_FIELDS = {
    "format",
    "schema_version",
    "contract_sha256",
    "execution",
    "candidate",
    "records",
}
_CANDIDATE_FIELDS = {
    "candidate_id",
    "precision_profile",
    "requested_claim",
    "claims_general_parity",
}
_RECORD_FIELDS = {
    "prompt_id",
    "prompt_token_ids",
    "teacher_steps",
    "reference",
    "candidate",
}
_RUN_FIELDS = {"precision", "runs", "run_sha256s"}
_CANDIDATE_RUN_FIELDS = {"precision_profile", "runs", "run_sha256s"}
_PRECISION_PROFILES = {
    "bf16",
    "w8-g64",
    "w4-g64",
    "hybrid-w8-bf16-g64",
    "hybrid-w8-bf16-g64-chinese-top1-counterfactual-v1",
    "mixed-w4-w8-bf16",
}
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_RUN_ENVELOPE_FIELDS = {
    "format",
    "schema_version",
    "status",
    "policy",
    "custody",
    "evidence",
    "validation_receipt",
    "content_sha256",
}
_RUN_POLICY = {
    "network": "hf-offline-direct-local-bundles-v1",
    "remote_code": False,
    "generation": "mlx-lm-generate-step-explicit-axis-minus-one-argmax-v1",
    "publication": "same-filesystem-atomic-no-replace-v1",
    "claim_scope": "fixed-suite-only-never-general-parity-v1",
}
_CUSTODY_FIELDS = {
    "contract",
    "producer",
    "validator",
    "runtime",
    "offline_environment",
    "bundles",
}
_PAIR_FIELDS = {"before", "after"}
_FILE_IDENTITY_FIELDS = {"path", "size", "sha256"}
_RUNTIME_FIELDS = {"python", "packages"}
_PYTHON_FIELDS = {"implementation", "version", "executable"}
_PINNED_PYTHON_VERSION = "3.14.3"
_PINNED_PACKAGES = {
    "huggingface-hub": "1.28.0",
    "mlx": "0.32.1",
    "mlx-lm": "0.31.3",
    "mlx-metal": "0.32.1",
    "numpy": "2.5.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "transformers": "5.15.1",
}
_OFFLINE_ENVIRONMENT = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "NO_PROXY": "*",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "no_proxy": "*",
}
_BUNDLE_SNAPSHOT_FIELDS = {
    "path",
    "precision_profile",
    "files",
    "file_count",
    "total_bytes",
    "manifest_sha256",
}
_BUNDLE_FILE_FIELDS = {"size", "sha256"}
_FIXED_BUNDLE_FILES = {
    "README.md",
    "chat_template.jinja",
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
_MODEL_SHARD = re.compile(
    r"^model(?:\.safetensors|-[0-9]{5}-of-[0-9]{5}\.safetensors)$"
)
_MAX_BUNDLE_FILES = 64


class QualityGateError(ValueError):
    """Fail-closed contract or evidence violation."""


def _fail(message: str) -> NoReturn:
    raise QualityGateError(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _copy(value: object) -> object:
    return json.loads(canonical_bytes(value))


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail(f"JSON contains duplicate key: {key}")
        value[key] = item
    return value


def _parse_json(payload: bytes, label: str) -> object:
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except QualityGateError:
        raise
    except (UnicodeError, ValueError) as error:
        raise QualityGateError(f"{label} is not valid UTF-8 JSON") from error


def _read_json_file(path: Path, label: str, maximum_bytes: int) -> object:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        _fail(f"{label} must be an absolute direct regular file")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise QualityGateError(f"cannot read {label}: {error}") from error
    if len(payload) > maximum_bytes:
        _fail(f"{label} exceeds the {maximum_bytes}-byte limit")
    return _parse_json(payload, label)


def _token_ids(
    value: object, label: str, *, exact_length: int | None = None
) -> list[int]:
    if type(value) is not list or not value:
        _fail(f"{label} must be a non-empty token-id array")
    if exact_length is not None and len(value) != exact_length:
        _fail(f"{label} must contain exactly {exact_length} token IDs")
    if any(
        type(token) is not int or token < 0 or token >= _MODEL["vocab_size"]
        for token in value
    ):
        _fail(f"{label} contains an invalid Qwen3.5 token ID")
    return list(value)


def validate_contract(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _TOP_LEVEL_FIELDS:
        _fail("quality contract must contain the exact top-level fields")
    if value.get("format") != CONTRACT_FORMAT or value.get("schema_version") != 1:
        _fail("quality contract format/schema drifted")
    if value.get("model") != _MODEL:
        _fail("quality contract model/tokenizer identity drifted")
    if value.get("generation") != _GENERATION:
        _fail("quality contract production generate_step semantics drifted")
    if value.get("admission") != _ADMISSION:
        _fail("quality contract admission claims or thresholds drifted")

    suite = value.get("suite")
    if type(suite) is not dict or set(suite) != {
        "scope_id",
        "minimum_prompts",
        "minimum_steps_per_prompt",
        "minimum_long_prompts",
        "long_prompt_steps",
        "prompts",
    }:
        _fail("quality contract suite fields drifted")
    if (
        suite.get("scope_id") != "qwen35-0.8b-fixed-raw-multiprompt-v1"
        or suite.get("minimum_prompts") != 4
        or suite.get("minimum_steps_per_prompt") != 32
        or suite.get("minimum_long_prompts") != 2
        or suite.get("long_prompt_steps") != 64
    ):
        _fail("quality contract suite floors drifted")
    prompts = suite.get("prompts")
    if type(prompts) is not list or len(prompts) != len(_PROMPTS):
        _fail("quality contract must contain exactly four fixed prompts")
    for index, expected in enumerate(_PROMPTS):
        prompt = prompts[index]
        if type(prompt) is not dict or set(prompt) != _PROMPT_FIELDS:
            _fail("quality contract prompt fields drifted")
        prompt_ids = _token_ids(prompt.get("prompt_token_ids"), f"prompt {expected[0]}")
        if (
            prompt.get("id") != expected[0]
            or prompt.get("domain") != expected[1]
            or prompt.get("prompt_token_ids_sha256") != expected[2]
            or prompt.get("teacher_steps") != expected[3]
            or object_sha256(prompt_ids) != expected[2]
        ):
            _fail(f"quality contract prompt {expected[0]} drifted")

    content_hash = value.get("content_sha256")
    body = dict(value)
    body.pop("content_sha256")
    if type(content_hash) is not str or _SHA256.fullmatch(content_hash) is None:
        _fail("quality contract content_sha256 must be lowercase SHA-256")
    if content_hash != object_sha256(body):
        _fail("quality contract content_sha256 does not match its body")
    return _copy(value)  # type: ignore[return-value]


def load_contract(path: Path) -> dict[str, object]:
    value = _read_json_file(Path(path), "quality contract", 1024 * 1024)
    return validate_contract(value)


def _validate_two_runs(
    value: object,
    label: str,
    *,
    teacher_steps: int,
) -> list[int]:
    if type(value) is not dict or (
        set(value) != _RUN_FIELDS and set(value) != _CANDIDATE_RUN_FIELDS
    ):
        _fail(f"{label} run fields drifted")
    runs = value.get("runs")
    hashes = value.get("run_sha256s")
    if type(runs) is not list or len(runs) != 2:
        _fail(f"{label} must contain exactly two runs")
    if type(hashes) is not list or len(hashes) != 2:
        _fail(f"{label} must contain exactly two run hashes")
    normalized = [
        _token_ids(run, f"{label} run {index + 1}", exact_length=teacher_steps)
        for index, run in enumerate(runs)
    ]
    expected_hashes = [object_sha256(run) for run in normalized]
    if hashes != expected_hashes:
        _fail(f"{label} run hashes do not match their token IDs")
    if normalized[0] != normalized[1]:
        _fail(f"{label} two runs are not identical")
    return normalized[0]


def _prefix_match(left: list[int], right: list[int]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matched += 1
    return matched


def validate_evidence(
    contract_value: object,
    evidence: object,
) -> dict[str, object]:
    """Validate deterministic BF16/candidate traces and recompute admission."""

    contract = validate_contract(contract_value)
    if type(evidence) is not dict or set(evidence) != _EVIDENCE_FIELDS:
        _fail("quality evidence must contain the exact top-level fields")
    if evidence.get("format") != EVIDENCE_FORMAT or evidence.get("schema_version") != 1:
        _fail("quality evidence format/schema drifted")
    if evidence.get("contract_sha256") != contract["content_sha256"]:
        _fail("quality evidence is not bound to this contract")
    if evidence.get("execution") != contract["generation"]:
        _fail(
            "quality evidence did not use the frozen production generate_step semantics"
        )

    candidate = evidence.get("candidate")
    if type(candidate) is not dict or set(candidate) != _CANDIDATE_FIELDS:
        _fail("quality evidence candidate fields drifted")
    candidate_id = candidate.get("candidate_id")
    precision_profile = candidate.get("precision_profile")
    requested_claim = candidate.get("requested_claim")
    if type(candidate_id) is not str or _CANDIDATE_ID.fullmatch(candidate_id) is None:
        _fail("candidate_id must be a bounded canonical identifier")
    if precision_profile not in _PRECISION_PROFILES:
        _fail("candidate precision_profile is not a supported BF16/W8/W4/hybrid tier")
    allowed_claims = {
        contract["admission"]["exact"]["claim"],  # type: ignore[index]
        contract["admission"]["threshold"]["claim"],  # type: ignore[index]
    }
    if requested_claim not in allowed_claims:
        _fail("requested claim is outside the fixed-suite admission contract")
    if candidate.get("claims_general_parity") is not False:
        _fail("a fixed prompt suite may not claim general model parity")

    records = evidence.get("records")
    prompts = contract["suite"]["prompts"]  # type: ignore[index]
    if type(records) is not list or len(records) != len(prompts):
        _fail("quality evidence must contain every fixed prompt exactly once")
    results = []
    for index, prompt in enumerate(prompts):
        record = records[index]
        if type(record) is not dict or set(record) != _RECORD_FIELDS:
            _fail("quality evidence prompt record fields drifted")
        if (
            record.get("prompt_id") != prompt["id"]
            or record.get("prompt_token_ids") != prompt["prompt_token_ids"]
            or record.get("teacher_steps") != prompt["teacher_steps"]
        ):
            _fail(f"quality evidence record {prompt['id']} is not bound to its prompt")
        reference = record.get("reference")
        candidate_runs = record.get("candidate")
        if type(reference) is not dict or reference.get("precision") != "bf16":
            _fail(f"quality evidence record {prompt['id']} reference is not BF16")
        if (
            type(candidate_runs) is not dict
            or candidate_runs.get("precision_profile") != precision_profile
        ):
            _fail(f"quality evidence record {prompt['id']} candidate tier drifted")
        teacher = _validate_two_runs(
            reference,
            f"{prompt['id']} BF16 reference",
            teacher_steps=prompt["teacher_steps"],
        )
        observed = _validate_two_runs(
            candidate_runs,
            f"{prompt['id']} candidate",
            teacher_steps=prompt["teacher_steps"],
        )
        exact_matches = sum(left == right for left, right in zip(teacher, observed))
        prefix = _prefix_match(teacher, observed)
        results.append(
            {
                "prompt_id": prompt["id"],
                "domain": prompt["domain"],
                "teacher_steps": prompt["teacher_steps"],
                "exact": observed == teacher,
                "exact_prefix_tokens": prefix,
                "position_match_ratio": exact_matches / prompt["teacher_steps"],
                "reference_sha256": object_sha256(teacher),
                "candidate_sha256": object_sha256(observed),
            }
        )

    problems: list[str] = []
    exact_claim = contract["admission"]["exact"]["claim"]  # type: ignore[index]
    if requested_claim == exact_claim:
        failed = [result["prompt_id"] for result in results if not result["exact"]]
        if failed:
            problems.append("exact fixed-suite mismatch: " + ", ".join(failed))
    else:
        threshold = contract["admission"]["threshold"]  # type: ignore[index]
        failed = [
            result["prompt_id"]
            for result in results
            if result["exact_prefix_tokens"] < threshold["minimum_exact_prefix_tokens"]
            or result["position_match_ratio"]
            < threshold["minimum_position_match_ratio"]
        ]
        pass_rate = (len(results) - len(failed)) / len(results)
        if pass_rate < threshold["required_prompt_pass_rate"]:
            problems.append("fixed-suite threshold mismatch: " + ", ".join(failed))

    accepted = not problems
    return {
        "format": RECEIPT_FORMAT,
        "schema_version": 1,
        "accepted": accepted,
        "claim": requested_claim if accepted else None,
        "scope_id": contract["suite"]["scope_id"],  # type: ignore[index]
        "claims_general_parity": False,
        "candidate_id": candidate_id,
        "precision_profile": precision_profile,
        "prompt_count": len(results),
        "contract_sha256": contract["content_sha256"],
        "evidence_sha256": object_sha256(evidence),
        "prompts": results,
        "problems": problems,
    }


def _recorded_absolute_path(value: object, label: str) -> str:
    if type(value) is not str or not value or not Path(value).is_absolute():
        _fail(f"{label} custody path must be absolute")
    return value


def _lowercase_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{label} custody SHA-256 must be lowercase")
    return value


def _validate_custody_pair(value: object, label: str, validator):
    if type(value) is not dict or set(value) != _PAIR_FIELDS:
        _fail(f"{label} custody must contain exact before/after snapshots")
    before = validator(value.get("before"), f"{label} before")
    after = validator(value.get("after"), f"{label} after")
    if before != after:
        _fail(f"{label} custody changed between before and after")
    return before


def _validate_file_identity(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != _FILE_IDENTITY_FIELDS:
        _fail(f"{label} custody file identity fields drifted")
    _recorded_absolute_path(value.get("path"), label)
    if type(value.get("size")) is not int or value["size"] < 0:
        _fail(f"{label} custody size must be a non-negative integer")
    _lowercase_sha256(value.get("sha256"), label)
    return _copy(value)  # type: ignore[return-value]


def _validate_runtime_identity(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != _RUNTIME_FIELDS:
        _fail(f"{label} custody runtime fields drifted")
    python = value.get("python")
    if (
        type(python) is not dict
        or set(python) != _PYTHON_FIELDS
        or python.get("implementation") != "CPython"
        or python.get("version") != _PINNED_PYTHON_VERSION
    ):
        _fail(f"{label} custody runtime is not pinned CPython")
    _recorded_absolute_path(python.get("executable"), f"{label} Python executable")
    if value.get("packages") != _PINNED_PACKAGES:
        _fail(f"{label} custody runtime does not match the pinned package lock")
    return _copy(value)  # type: ignore[return-value]


def _validate_shard_portfolio(names: set[str], label: str) -> None:
    shards = sorted(name for name in names if _MODEL_SHARD.fullmatch(name))
    if not shards:
        _fail(f"{label} custody has no model safetensors shard")
    if shards == ["model.safetensors"]:
        return
    parsed = []
    for name in shards:
        match = re.fullmatch(r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", name)
        if match is None:
            _fail(f"{label} custody model shard portfolio is invalid")
        parsed.append((int(match.group(1)), int(match.group(2))))
    totals = {total for _, total in parsed}
    if len(totals) != 1:
        _fail(f"{label} custody model shard totals disagree")
    total = totals.pop()
    if total != len(parsed) or sorted(index for index, _ in parsed) != list(
        range(1, total + 1)
    ):
        _fail(f"{label} custody model shard sequence is incomplete")


def _validate_bundle_snapshot(
    value: object,
    label: str,
    *,
    expected_precision: str | None,
    tokenizer_sha256: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _BUNDLE_SNAPSHOT_FIELDS:
        _fail(f"{label} custody bundle snapshot fields drifted")
    _recorded_absolute_path(value.get("path"), label)
    precision = value.get("precision_profile")
    if precision not in _PRECISION_PROFILES or (
        expected_precision is not None and precision != expected_precision
    ):
        _fail(f"{label} custody precision profile drifted")
    files = value.get("files")
    if type(files) is not dict or not files or len(files) > _MAX_BUNDLE_FILES:
        _fail(f"{label} custody files must be a bounded non-empty manifest")
    names = set(files)
    _validate_shard_portfolio(names, label)
    shards = {name for name in names if _MODEL_SHARD.fullmatch(name)}
    if names != _FIXED_BUNDLE_FILES | shards:
        _fail(f"{label} custody is not the controlled flat bundle layout")
    total_bytes = 0
    for name, record in files.items():
        if type(record) is not dict or set(record) != _BUNDLE_FILE_FIELDS:
            _fail(f"{label}/{name} custody file identity fields drifted")
        if type(record.get("size")) is not int or record["size"] < 0:
            _fail(f"{label}/{name} custody size must be a non-negative integer")
        _lowercase_sha256(record.get("sha256"), f"{label}/{name}")
        total_bytes += record["size"]
    if files["tokenizer.json"]["sha256"] != tokenizer_sha256:
        _fail(f"{label} custody tokenizer is not bound to the contract")
    if value.get("file_count") != len(files) or value.get("total_bytes") != total_bytes:
        _fail(f"{label} custody bundle counts do not match its manifest")
    manifest_sha256 = _lowercase_sha256(
        value.get("manifest_sha256"), f"{label} manifest"
    )
    if manifest_sha256 != object_sha256(files):
        _fail(f"{label} custody manifest SHA-256 does not match its files")
    return _copy(value)  # type: ignore[return-value]


def _validate_custody(
    value: object,
    contract: dict[str, object],
    *,
    contract_path: Path | None,
) -> str:
    if type(value) is not dict or set(value) != _CUSTODY_FIELDS:
        _fail("quality run envelope custody fields drifted")
    contract_identity = _validate_custody_pair(
        value.get("contract"), "contract", _validate_file_identity
    )
    _validate_custody_pair(value.get("producer"), "producer", _validate_file_identity)
    _validate_custody_pair(value.get("validator"), "validator", _validate_file_identity)
    _validate_custody_pair(value.get("runtime"), "runtime", _validate_runtime_identity)
    if value.get("offline_environment") != _OFFLINE_ENVIRONMENT:
        _fail("quality run envelope custody offline environment drifted")
    bundles = value.get("bundles")
    if type(bundles) is not dict or set(bundles) != {"reference", "candidate"}:
        _fail("quality run envelope custody bundle fields drifted")
    tokenizer_sha256 = contract["model"]["tokenizer_sha256"]  # type: ignore[index]
    reference = _validate_custody_pair(
        bundles.get("reference"),
        "reference bundle",
        lambda snapshot, label: _validate_bundle_snapshot(
            snapshot,
            label,
            expected_precision="bf16",
            tokenizer_sha256=tokenizer_sha256,
        ),
    )
    candidate = _validate_custody_pair(
        bundles.get("candidate"),
        "candidate bundle",
        lambda snapshot, label: _validate_bundle_snapshot(
            snapshot,
            label,
            expected_precision=None,
            tokenizer_sha256=tokenizer_sha256,
        ),
    )
    if reference["path"] == candidate["path"]:
        _fail("quality run envelope custody bundles must be distinct")
    if reference["files"]["tokenizer.json"] != candidate["files"]["tokenizer.json"]:
        _fail("quality run envelope custody tokenizer identities differ")
    if contract_path is not None:
        try:
            contract_payload = Path(contract_path).read_bytes()
        except OSError as error:
            raise QualityGateError(
                f"cannot re-read quality contract: {error}"
            ) from error
        if (
            contract_identity["size"] != len(contract_payload)
            or contract_identity["sha256"]
            != hashlib.sha256(contract_payload).hexdigest()
        ):
            _fail("quality run envelope contract custody is not bound to --contract")
    return candidate["precision_profile"]  # type: ignore[return-value]


def validate_run_envelope(
    contract_value: object,
    envelope: object,
    *,
    contract_path: Path | None = None,
) -> dict[str, object]:
    """Validate a producer envelope and return its recomputed inner receipt."""

    contract = validate_contract(contract_value)
    if type(envelope) is not dict or set(envelope) != _RUN_ENVELOPE_FIELDS:
        _fail("quality run envelope must contain the exact top-level fields")
    if (
        envelope.get("format") != RUN_ENVELOPE_FORMAT
        or envelope.get("schema_version") != 1
    ):
        _fail("quality run envelope format/schema drifted")
    content_sha256 = envelope.get("content_sha256")
    body = dict(envelope)
    body.pop("content_sha256")
    if type(content_sha256) is not str or _SHA256.fullmatch(content_sha256) is None:
        _fail("quality run envelope content_sha256 must be lowercase SHA-256")
    if content_sha256 != object_sha256(body):
        _fail("quality run envelope content_sha256 does not match its body")

    if envelope.get("policy") != _RUN_POLICY:
        _fail("quality run envelope policy drifted")
    custody_precision = _validate_custody(
        envelope.get("custody"), contract, contract_path=contract_path
    )

    receipt = validate_evidence(contract, envelope.get("evidence"))
    if envelope.get("validation_receipt") != receipt:
        _fail("quality run envelope receipt is not bound to its inner evidence")
    if receipt["precision_profile"] != custody_precision:
        _fail("quality run envelope bundle custody is not bound to its inner evidence")
    expected_status = "accepted" if receipt["accepted"] is True else "failed_comparison"
    if envelope.get("status") != expected_status:
        _fail("quality run envelope status does not match its inner receipt")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        contract = load_contract(args.contract)
        evidence = _read_json_file(
            args.evidence,
            "quality evidence",
            4 * 1024 * 1024,
        )
        if type(evidence) is dict and evidence.get("format") == RUN_ENVELOPE_FORMAT:
            receipt = validate_run_envelope(
                contract,
                evidence,
                contract_path=args.contract,
            )
        else:
            receipt = validate_evidence(contract, evidence)
    except QualityGateError as error:
        receipt = {
            "format": RECEIPT_FORMAT,
            "schema_version": 1,
            "accepted": False,
            "claim": None,
            "claims_general_parity": False,
            "problems": [str(error)],
        }
        return_code = 2
    else:
        return_code = 0 if receipt["accepted"] is True else 1
    sys.stdout.write(canonical_bytes(receipt).decode("utf-8") + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

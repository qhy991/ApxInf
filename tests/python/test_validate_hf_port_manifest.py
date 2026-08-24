from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/validate_hf_port_manifest.py"
SPEC = importlib.util.spec_from_file_location("validate_hf_port_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SOURCE_LOCK_CONTENT_SHA256 = "b" * 64
QWEN_COMMIT = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN_SOURCE_LOCK_SHA256 = (
    "021209cc96e398db4aac6d126890f7bb5a5a3b5fce7204fed0328f544cbb7500"
)
QWEN_READY_GATES = [
    "source-lock",
    "bundle-integrity",
    "pinned-macos-arm64-binary",
    "exact-greedy-token-trajectory",
    "transformers-oracle-parity",
    "macos-memory-smoke",
]


def identity_args() -> list[str]:
    return [
        "--expected-repo-id",
        "Qwen/Qwen3.5-0.8B",
        "--expected-requested-revision",
        "main",
        "--expected-resolved-commit",
        "2fc06364715b967f1860aea9cf38778875588b17",
        "--expected-source-lock-content-sha256",
        SOURCE_LOCK_CONTENT_SHA256,
    ]


def qwen_host_args() -> list[str]:
    return [
        "--source-lock",
        str(ROOT / ".apxinf/onboarding/qwen35-0.8b/source-lock.json"),
        "--deployment-profile",
        str(ROOT / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"),
        "--expected-repo-id",
        "Qwen/Qwen3.5-0.8B",
        "--expected-requested-revision",
        QWEN_COMMIT,
        "--expected-resolved-commit",
        QWEN_COMMIT,
        "--expected-source-lock-content-sha256",
        QWEN_SOURCE_LOCK_SHA256,
    ]


def base_manifest() -> dict[str, object]:
    return {
        "schema_version": 2,
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "requested_revision": "main",
        "resolved_commit": "2fc06364715b967f1860aea9cf38778875588b17",
        "source_lock_content_sha256": SOURCE_LOCK_CONTENT_SHA256,
        "task": "text-generation",
        "input_modalities": ["text"],
        "profile_id": None,
        "target": "macos-arm64",
        "route": "PORT_MODEL",
        "provider": "native-apxinf-cpu",
        "blockers": ["Host verification is required before implementation."],
        "user_checkpoint_required": True,
        "transaction_paths": [
            "crates/apxinf-model/src/builtin.rs",
            "crates/apxinf-model/src/new_family",
        ],
        "new_paths": ["crates/apxinf-model/src/new_family"],
        "required_gates": ["macos-arm64-build", "numerical-parity"],
    }


def qwen_ready_manifest() -> dict[str, object]:
    return {
        "schema_version": 2,
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "requested_revision": QWEN_COMMIT,
        "resolved_commit": QWEN_COMMIT,
        "source_lock_content_sha256": QWEN_SOURCE_LOCK_SHA256,
        "task": "text-generation",
        "input_modalities": ["text"],
        "profile_id": "qwen35-0.8b-macos-cpu",
        "target": "macos-arm64",
        "route": "READY_EXISTING",
        "provider": "native-apxinf-cpu",
        "blockers": [],
        "user_checkpoint_required": False,
        "transaction_paths": [],
        "new_paths": [],
        "required_gates": list(QWEN_READY_GATES),
    }


class WorkspaceCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "ApxInf"
        source = self.workspace / "crates/apxinf-model/src"
        source.mkdir(parents=True)
        (source / "builtin.rs").write_text("// fixture\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def validate(self, manifest: dict[str, object]) -> dict[str, object]:
        return MODULE.validate_manifest(
            manifest,
            self.workspace,
            expected_repo_id="Qwen/Qwen3.5-0.8B",
            expected_requested_revision="main",
            expected_resolved_commit="2fc06364715b967f1860aea9cf38778875588b17",
            expected_source_lock_content_sha256=SOURCE_LOCK_CONTENT_SHA256,
        )


class ValidManifestTests(WorkspaceCase):
    def test_ready_required_mode_rejects_a_valid_non_ready_route(self) -> None:
        with self.assertRaisesRegex(MODULE.ManifestError, "READY_EXISTING"):
            MODULE.validate_manifest(
                base_manifest(),
                self.workspace,
                expected_repo_id="Qwen/Qwen3.5-0.8B",
                expected_requested_revision="main",
                expected_resolved_commit=("2fc06364715b967f1860aea9cf38778875588b17"),
                expected_source_lock_content_sha256=SOURCE_LOCK_CONTENT_SHA256,
                require_ready_existing=True,
            )

    def test_accepts_existing_file_and_declared_new_root(self) -> None:
        receipt = self.validate(base_manifest())
        self.assertEqual(receipt["passed"], True)
        self.assertIs(receipt["route_verified"], False)
        self.assertIs(receipt["decision_complete"], False)
        self.assertEqual(receipt["repo_id"], "Qwen/Qwen3.5-0.8B")
        self.assertEqual(
            receipt["source_lock_content_sha256"], SOURCE_LOCK_CONTENT_SHA256
        )
        self.assertEqual(
            receipt["path_facts"],
            [
                {
                    "path": "crates/apxinf-model/src/builtin.rs",
                    "state": "existing-file",
                },
                {
                    "path": "crates/apxinf-model/src/new_family",
                    "state": "new",
                },
            ],
        )
        self.assertFalse(
            (self.workspace / "crates/apxinf-model/src/new_family").exists()
        )

    def test_blocked_manifest_may_have_no_transactions_or_gates(self) -> None:
        manifest = base_manifest()
        manifest.update(
            {
                "route": "BLOCKED",
                "provider": "none",
                "blockers": ["Gated model requires a user decision."],
                "user_checkpoint_required": True,
                "transaction_paths": [],
                "new_paths": [],
                "required_gates": [],
            }
        )
        self.assertTrue(self.validate(manifest)["passed"])

    def test_cli_accepts_stdin_and_emits_exactly_one_json_receipt(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-B",
                str(MODULE_PATH),
                "--workspace",
                str(ROOT),
                *qwen_host_args(),
            ],
            input=json.dumps(qwen_ready_manifest()),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertTrue(json.loads(result.stdout)["passed"])

    def test_cli_accepts_inline_artifact_json(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-B",
                str(MODULE_PATH),
                "--workspace",
                str(ROOT),
                "--json",
                json.dumps(qwen_ready_manifest()),
                *qwen_host_args(),
                "--require-ready-existing",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertTrue(json.loads(result.stdout)["passed"])


class ReadyExistingProfileTests(unittest.TestCase):
    def test_accepts_qwen_only_after_host_profile_and_source_identity_match(
        self,
    ) -> None:
        manifest = qwen_ready_manifest()
        receipt = MODULE.validate_manifest(
            manifest,
            ROOT,
            expected_repo_id="Qwen/Qwen3.5-0.8B",
            expected_requested_revision=QWEN_COMMIT,
            expected_resolved_commit=QWEN_COMMIT,
            expected_source_lock_content_sha256=QWEN_SOURCE_LOCK_SHA256,
            source_lock_path=(ROOT / ".apxinf/onboarding/qwen35-0.8b/source-lock.json"),
            deployment_profile_path=(
                ROOT / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"
            ),
            require_ready_existing=True,
        )

        self.assertTrue(receipt["passed"])
        self.assertIs(receipt["route_verified"], True)
        self.assertIs(receipt["decision_complete"], True)
        self.assertEqual(receipt["route"], "READY_EXISTING")
        self.assertEqual(receipt["provider"], "native-apxinf-cpu")
        self.assertEqual(receipt["required_gates"], QWEN_READY_GATES)
        self.assertEqual(
            receipt["profile_file_sha256"], MODULE.QWEN_PROFILE_FILE_SHA256
        )

    def test_qwen_ready_contract_rejects_agent_selected_provider_gates_or_paths(
        self,
    ) -> None:
        mutations = (
            ("provider", "invented-provider"),
            ("required_gates", ["made-up-gate"]),
            ("transaction_paths", ["Cargo.toml"]),
            ("blockers", ["agent says maybe"]),
            ("user_checkpoint_required", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                manifest = qwen_ready_manifest()
                manifest[field] = value
                with self.assertRaises(MODULE.ManifestError):
                    MODULE.validate_manifest(
                        manifest,
                        ROOT,
                        expected_repo_id="Qwen/Qwen3.5-0.8B",
                        expected_requested_revision=QWEN_COMMIT,
                        expected_resolved_commit=QWEN_COMMIT,
                        expected_source_lock_content_sha256=QWEN_SOURCE_LOCK_SHA256,
                        source_lock_path=(
                            ROOT / ".apxinf/onboarding/qwen35-0.8b/source-lock.json"
                        ),
                        deployment_profile_path=(
                            ROOT / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"
                        ),
                    )

    def test_host_verified_qwen_cannot_be_downgraded_to_a_write_route(self) -> None:
        manifest = qwen_ready_manifest()
        manifest.update(
            {
                "route": "PORT_MODEL",
                "profile_id": None,
                "provider": "apxinf-native",
                "required_gates": ["macos-arm64-build"],
            }
        )
        with self.assertRaises(MODULE.ManifestError):
            MODULE.validate_manifest(
                manifest,
                ROOT,
                expected_repo_id="Qwen/Qwen3.5-0.8B",
                expected_requested_revision=QWEN_COMMIT,
                expected_resolved_commit=QWEN_COMMIT,
                expected_source_lock_content_sha256=QWEN_SOURCE_LOCK_SHA256,
                source_lock_path=(
                    ROOT / ".apxinf/onboarding/qwen35-0.8b/source-lock.json"
                ),
                deployment_profile_path=(
                    ROOT / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"
                ),
            )

    def _assert_mutated_source_identity_is_rejected(self, field: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "ApxInf"
            profile_path = (
                workspace / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json"
            )
            profile_path.parent.mkdir(parents=True)
            profile_path.write_bytes(
                (ROOT / "configs/hf-onboarding/qwen35-0.8b-macos-cpu.json").read_bytes()
            )
            source_path = workspace / ".apxinf/onboarding/qwen35-0.8b/source-lock.json"
            source_path.parent.mkdir(parents=True)
            source = json.loads(
                (ROOT / ".apxinf/onboarding/qwen35-0.8b/source-lock.json").read_text(
                    encoding="utf-8"
                )
            )
            if field == "config":
                source["architecture"]["config_sha256"] = "e" * 64
                for record in source["metadata"]["files"]:
                    if record["path"] == "config.json":
                        record["sha256"] = "e" * 64
            else:
                source["weights"]["files"][0]["sha256"] = "e" * 64
            del source["content_sha256"]
            encoded = json.dumps(
                source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            source_hash = hashlib.sha256(encoded).hexdigest()
            source["content_sha256"] = source_hash
            source_path.write_text(json.dumps(source), encoding="utf-8")
            manifest = qwen_ready_manifest()
            manifest["source_lock_content_sha256"] = source_hash

            with self.assertRaises(MODULE.ManifestError):
                MODULE.validate_manifest(
                    manifest,
                    workspace,
                    expected_repo_id="Qwen/Qwen3.5-0.8B",
                    expected_requested_revision=QWEN_COMMIT,
                    expected_resolved_commit=QWEN_COMMIT,
                    expected_source_lock_content_sha256=source_hash,
                    source_lock_path=source_path,
                    deployment_profile_path=profile_path,
                )

    def test_rejects_qwen_when_config_identity_differs_from_profile(self) -> None:
        self._assert_mutated_source_identity_is_rejected("config")

    def test_rejects_qwen_when_shard_identity_differs_from_profile(self) -> None:
        self._assert_mutated_source_identity_is_rejected("shard")


class SchemaTests(WorkspaceCase):
    def assert_rejected(self, manifest: dict[str, object]) -> None:
        with self.assertRaises(MODULE.ManifestError):
            self.validate(manifest)

    def test_requires_exact_schema_keys(self) -> None:
        missing = base_manifest()
        del missing["provider"]
        self.assert_rejected(missing)
        unknown = base_manifest()
        unknown["confidence"] = 1.0
        self.assert_rejected(unknown)

    def test_rejects_bool_or_string_schema_version(self) -> None:
        for value in (True, "2", 1):
            with self.subTest(value=value):
                manifest = base_manifest()
                manifest["schema_version"] = value
                self.assert_rejected(manifest)

    def test_rejects_bad_identity_and_decision_fields(self) -> None:
        cases = {
            "repo_id": "Qwen/Model/extra",
            "requested_revision": "../main",
            "resolved_commit": "A" * 40,
            "target": "linux-x86_64",
            "route": "SUPPORTED",
            "provider": "MLX provider; run me",
            "user_checkpoint_required": 1,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                manifest = base_manifest()
                manifest[field] = value
                self.assert_rejected(manifest)

    def test_generic_routes_cannot_claim_a_checked_in_profile(self) -> None:
        manifest = base_manifest()
        manifest["profile_id"] = "qwen35-0.8b-macos-cpu"
        self.assert_rejected(manifest)

    def test_non_ready_route_requires_a_host_visible_checkpoint(self) -> None:
        manifest = base_manifest()
        manifest["user_checkpoint_required"] = False
        manifest["blockers"] = []
        self.assert_rejected(manifest)

    def test_non_ready_route_rejects_an_unregistered_provider(self) -> None:
        manifest = base_manifest()
        manifest["provider"] = "invented-provider"
        self.assert_rejected(manifest)

    def test_non_ready_route_rejects_an_unregistered_gate(self) -> None:
        manifest = base_manifest()
        manifest["required_gates"] = ["made-up-gate"]
        self.assert_rejected(manifest)

    def test_rejects_identity_that_differs_from_host_expectations(self) -> None:
        cases = {
            "repo_id": "Other/Model",
            "requested_revision": "dev",
            "resolved_commit": "c" * 40,
            "source_lock_content_sha256": "d" * 64,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                manifest = base_manifest()
                manifest[field] = value
                self.assert_rejected(manifest)

    def test_rejects_wrong_host_source_lock_expectation(self) -> None:
        with self.assertRaises(MODULE.ManifestError):
            MODULE.validate_manifest(
                base_manifest(),
                self.workspace,
                expected_repo_id="Qwen/Qwen3.5-0.8B",
                expected_requested_revision="main",
                expected_resolved_commit="2fc06364715b967f1860aea9cf38778875588b17",
                expected_source_lock_content_sha256="e" * 64,
            )

    def test_rejects_non_arrays_and_unsafe_gate_ids(self) -> None:
        for field in ("blockers", "transaction_paths", "new_paths", "required_gates"):
            with self.subTest(field=field):
                manifest = base_manifest()
                manifest[field] = "not-an-array"
                self.assert_rejected(manifest)
        manifest = base_manifest()
        manifest["required_gates"] = ["cargo test --workspace"]
        self.assert_rejected(manifest)

    def test_non_blocked_route_requires_gate(self) -> None:
        manifest = base_manifest()
        manifest["required_gates"] = []
        self.assert_rejected(manifest)

    def test_duplicate_json_keys_and_non_finite_numbers_fail(self) -> None:
        duplicate = b'{"schema_version":1,"schema_version":1}'
        with self.assertRaises(MODULE.ManifestError):
            MODULE.parse_json(duplicate)
        with self.assertRaises(MODULE.ManifestError):
            MODULE.parse_json(b'{"schema_version":NaN}')

    def test_cli_failure_is_one_json_line_and_nonzero(self) -> None:
        result = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--workspace", str(self.workspace)],
            input="{}",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["passed"], False)
        self.assertEqual(receipt["error"]["code"], "INVALID_PORT_MANIFEST")

    def test_cli_expected_identity_arguments_are_mandatory(self) -> None:
        result = subprocess.run(
            [sys.executable, str(MODULE_PATH), "--workspace", str(self.workspace)],
            input=json.dumps(base_manifest()),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["error"]["code"], "INVALID_PORT_MANIFEST")
        self.assertIn("required", receipt["error"]["message"])

    def test_cli_requires_host_source_and_profile_bindings(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(MODULE_PATH),
                "--workspace",
                str(self.workspace),
                *identity_args(),
            ],
            input=json.dumps(base_manifest()),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        receipt = json.loads(result.stdout)
        self.assertIn("source-lock", receipt["error"]["message"])


class PathAdversaryTests(WorkspaceCase):
    def assert_path_rejected(self, path: str, *, new: bool = True) -> None:
        manifest = base_manifest()
        manifest["transaction_paths"] = [path]
        manifest["new_paths"] = [path] if new else []
        with self.assertRaises(MODULE.ManifestError):
            self.validate(manifest)

    def test_rejects_noncanonical_or_escaping_paths(self) -> None:
        for path in (
            "",
            "/tmp/payload",
            "../payload",
            "crates/../payload",
            "crates//payload",
            "crates/./payload",
            "C:\\temp\\payload",
            "~/payload",
        ):
            with self.subTest(path=path):
                self.assert_path_rejected(path)

    def test_rejects_forbidden_state_credentials_and_weight_paths(self) -> None:
        for path in (
            ".git/hooks/post-commit",
            ".kersor/autonomous-runs/job",
            "cache/download",
            "target/release/app",
            "models/weights/model.safetensors",
            "artifacts/pytorch_model.bin",
            "artifacts/pytorch_model.bin.index.json",
            "config/.env.local",
            "credentials/hf_token",
            "config/credentials.json",
            "host-runs/job/output.json",
        ):
            with self.subTest(path=path):
                self.assert_path_rejected(path)

    def test_rejects_existing_directory_as_transaction_path(self) -> None:
        self.assert_path_rejected("crates/apxinf-model/src", new=False)

    def test_rejects_missing_undeclared_path_and_existing_new_path(self) -> None:
        self.assert_path_rejected("crates/apxinf-model/src/missing.rs", new=False)
        self.assert_path_rejected("crates/apxinf-model/src/builtin.rs", new=True)

    def test_rejects_new_path_with_missing_parent(self) -> None:
        self.assert_path_rejected("crates/apxinf-model/missing/child", new=True)

    def test_rejects_new_path_not_in_transaction_boundary(self) -> None:
        manifest = base_manifest()
        manifest["transaction_paths"] = ["crates/apxinf-model/src/builtin.rs"]
        manifest["new_paths"] = ["crates/apxinf-model/src/new_family"]
        with self.assertRaises(MODULE.ManifestError):
            self.validate(manifest)

    def test_new_path_must_match_transaction_path_exactly(self) -> None:
        manifest = base_manifest()
        manifest["transaction_paths"] = ["crates/apxinf-model/src/new_family"]
        manifest["new_paths"] = ["crates/apxinf-model/src/NEW_FAMILY"]
        with self.assertRaises(MODULE.ManifestError):
            self.validate(manifest)

    def test_rejects_duplicates_and_parent_child_overlap(self) -> None:
        manifest = base_manifest()
        manifest["transaction_paths"] = [
            "crates/apxinf-model/src/builtin.rs",
            "crates/apxinf-model/src/builtin.rs",
        ]
        manifest["new_paths"] = []
        with self.assertRaises(MODULE.ManifestError):
            self.validate(manifest)

        parent = self.workspace / "crates/apxinf-model/new_tree"
        parent.mkdir()
        manifest = base_manifest()
        manifest["transaction_paths"] = [
            "crates/apxinf-model/new_tree",
            "crates/apxinf-model/new_tree/child.rs",
        ]
        manifest["new_paths"] = []
        with self.assertRaises(MODULE.ManifestError):
            self.validate(manifest)

    def test_rejects_symbolic_link_in_parent_chain(self) -> None:
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        link = self.workspace / "crates/linked"
        link.symlink_to(outside, target_is_directory=True)
        self.assert_path_rejected("crates/linked/new_file.rs", new=True)

    def test_rejects_hard_link_target(self) -> None:
        source = self.workspace / "crates/apxinf-model/src/builtin.rs"
        hardlink = self.workspace / "crates/apxinf-model/src/alias.rs"
        os.link(source, hardlink)
        self.assert_path_rejected("crates/apxinf-model/src/builtin.rs", new=False)

    def test_rejects_case_collision_suitable_for_default_macos_filesystems(
        self,
    ) -> None:
        (self.workspace / "crates/apxinf-model/src/NewFamily").mkdir()
        self.assert_path_rejected("crates/apxinf-model/src/newfamily", new=True)

    def test_blocked_route_cannot_smuggle_transaction_paths(self) -> None:
        manifest = base_manifest()
        manifest["route"] = "BLOCKED"
        with self.assertRaises(MODULE.ManifestError):
            self.validate(manifest)


if __name__ == "__main__":
    unittest.main()

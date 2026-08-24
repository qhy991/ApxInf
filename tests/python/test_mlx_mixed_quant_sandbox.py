from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/mlx_mixed_quant_sandbox.py"
PROBE_PATH = ROOT / "tests/fixtures/mlx_mixed_quant_sandbox_probe.py"
PYTHON_PATH = (ROOT / ".apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python3.14").resolve(
    strict=True
)
TOOLCHAIN_PATH = PYTHON_PATH.parents[1]


def load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module("_apxinf_mixed_quant_sandbox_under_test", MODULE_PATH)


def object_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class MixedQuantSandboxTests(unittest.TestCase):
    def fixture(
        self, root: Path, *, mode: str = "probe"
    ) -> tuple[object, dict[str, Path]]:
        source = root / "source"
        source.mkdir(mode=0o700)
        source_file = source / "weights.txt"
        source_file.write_text("source-ok", encoding="utf-8")
        policy = root / "policy.json"
        policy.write_text('{"policy":"ok"}\n', encoding="utf-8")
        scratch = root / "scratch"
        scratch.mkdir(mode=0o700)
        outside_read = root / "outside-read.txt"
        outside_read.write_text("outside-secret", encoding="utf-8")
        outside_write = root / "outside-write.txt"
        paths = {
            "source": source,
            "source_file": source_file,
            "policy": policy,
            "scratch": scratch,
            "scratch_file": scratch / "allowed.txt",
            "outside_read": outside_read,
            "outside_write": outside_write,
        }
        child_argv = (
            "--mode",
            mode,
            "--source-file",
            str(source_file),
            "--policy-file",
            str(policy),
            "--scratch-file",
            str(paths["scratch_file"]),
            "--outside-read",
            str(outside_read),
            "--outside-write",
            str(outside_write),
        )
        spec = MODULE.SandboxLaunchSpec(
            python_path=PYTHON_PATH,
            backend_script_path=PROBE_PATH.resolve(strict=True),
            toolchain_dir=TOOLCHAIN_PATH,
            source_dir=source,
            source_manifest_sha256=hashlib.sha256(b"tiny-source-manifest").hexdigest(),
            policy_path=policy,
            scratch_dir=scratch,
            child_argv=child_argv,
            timeout_seconds=5.0,
            stdout_limit_bytes=256 * 1024,
            stderr_limit_bytes=256 * 1024,
        )
        return spec, paths

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
    def test_real_seatbelt_denies_network_and_outside_io_but_allows_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix=".apxinf-seatbelt-", dir=ROOT) as temp:
            root = Path(temp).resolve(strict=True)
            spec, paths = self.fixture(root)
            previous = os.environ.get("APXINF_AMBIENT_SECRET")
            os.environ["APXINF_AMBIENT_SECRET"] = "must-not-cross"
            try:
                receipt = MODULE.launch_backend_child(spec)
            finally:
                if previous is None:
                    os.environ.pop("APXINF_AMBIENT_SECRET", None)
                else:
                    os.environ["APXINF_AMBIENT_SECRET"] = previous

            verified = MODULE.verify_launch_receipt(receipt, spec)
            child = json.loads(
                base64.b64decode(verified["body"]["output"]["stdout_base64"])
            )
            self.assertTrue(verified["body"]["passed"])
            self.assertTrue(verified["body"]["network_blocked"])
            self.assertEqual(child["source_read"]["value"], "source-ok")
            self.assertEqual(child["policy_read"]["value"], '{"policy":"ok"}\n')
            self.assertTrue(child["scratch_write"]["allowed"])
            self.assertFalse(child["outside_read"]["allowed"])
            self.assertFalse(child["outside_write"]["allowed"])
            self.assertFalse(child["network"]["allowed"])
            self.assertIn(child["outside_read"]["errno"], child["denied_errnos"])
            self.assertIn(child["outside_write"]["errno"], child["denied_errnos"])
            self.assertIn(child["network"]["errno"], child["denied_errnos"])
            self.assertNotIn("APXINF_AMBIENT_SECRET", child["environment"])
            self.assertEqual(
                child["environment"],
                verified["body"]["process"]["environment"],
            )
            self.assertEqual(paths["scratch_file"].read_text(), "scratch-ok")
            self.assertFalse(paths["outside_write"].exists())
            self.assertEqual(
                stat.S_IMODE(paths["scratch"].stat().st_mode),
                0o700,
            )

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
    def test_each_output_lane_is_bounded_while_child_is_live(self) -> None:
        for lane in ("stdout", "stderr"):
            with self.subTest(lane=lane):
                with tempfile.TemporaryDirectory(
                    prefix=".apxinf-seatbelt-", dir=ROOT
                ) as temp:
                    spec, _paths = self.fixture(
                        Path(temp).resolve(strict=True), mode=lane
                    )
                    spec = MODULE.SandboxLaunchSpec(
                        **{
                            **spec.__dict__,
                            "stdout_limit_bytes": 4096,
                            "stderr_limit_bytes": 4096,
                        }
                    )
                    with (
                        mock.patch.object(
                            MODULE,
                            "_kill_process_group",
                            wraps=MODULE._kill_process_group,
                        ) as killed,
                        self.assertRaisesRegex(
                            MODULE.SandboxError,
                            f"{lane}.*limit.*process group.*killed",
                        ),
                    ):
                        MODULE.launch_backend_child(spec)
                    killed.assert_called_once()

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
    def test_timeout_kills_the_owned_process_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".apxinf-seatbelt-", dir=ROOT) as temp:
            spec, _paths = self.fixture(Path(temp).resolve(strict=True), mode="sleep")
            spec = MODULE.SandboxLaunchSpec(**{**spec.__dict__, "timeout_seconds": 0.1})
            with (
                mock.patch.object(
                    MODULE,
                    "_kill_process_group",
                    wraps=MODULE._kill_process_group,
                ) as killed,
                self.assertRaisesRegex(
                    MODULE.SandboxError,
                    "timed out.*process group.*killed",
                ),
            ):
                MODULE.launch_backend_child(spec)
            killed.assert_called_once()

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
    def test_success_sweeps_process_group_before_reaping_leader(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".apxinf-seatbelt-", dir=ROOT) as temp:
            spec, _paths = self.fixture(Path(temp).resolve(strict=True))
            with mock.patch.object(
                MODULE,
                "_kill_process_group",
                wraps=MODULE._kill_process_group,
            ) as killed:
                receipt = MODULE.launch_backend_child(spec)

            verified = MODULE.verify_launch_receipt(receipt, spec)
            killed.assert_called_once()
            self.assertTrue(
                verified["body"]["process"]["process_group_swept_before_leader_reap"]
            )

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
    def test_verifier_reconstructs_profile_argv_runtime_script_and_environment(
        self,
    ) -> None:
        mutations = {
            "profile": lambda body: body["sandbox"].__setitem__(
                "profile_sha256", "0" * 64
            ),
            "argv": lambda body: body["process"]["argv"].append("--drift"),
            "python": lambda body: body["identities"]["python"].__setitem__(
                "sha256", "1" * 64
            ),
            "script": lambda body: body["identities"]["backend_script"].__setitem__(
                "sha256", "2" * 64
            ),
            "environment": lambda body: body["process"]["environment"].__setitem__(
                "AMBIENT", "forbidden"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    prefix=".apxinf-seatbelt-", dir=ROOT
                ) as temp:
                    spec, _paths = self.fixture(Path(temp).resolve(strict=True))
                    receipt = MODULE.launch_backend_child(spec)
                    changed = json.loads(json.dumps(receipt))
                    mutate(changed["body"])
                    changed["receipt_sha256"] = object_sha256(changed["body"])

                    with self.assertRaises(MODULE.SandboxError):
                        MODULE.verify_launch_receipt(changed, spec)


if __name__ == "__main__":
    unittest.main()

"""CPU tests for portable inputs and Metal measurement coordination."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import compare_native as native


class PortableInputsTests(unittest.TestCase):
    def test_asset_check_rejects_changed_bytes_and_extra_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            weight = model / "model.safetensors"
            weight.write_bytes(b"abc")
            manifest = {
                "model": "test-model", "revision": "test-revision",
                "assets": {weight.name: {
                    "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest(),
                }},
            }
            (root / "assets.json").write_text(json.dumps(manifest))
            contract = {"target": "test-model", "revision": "test-revision",
                        "asset_manifest": "assets.json"}
            with mock.patch.object(native, "HERE", root):
                assets, _ = native.pinned_assets(model, contract)
                self.assertEqual(assets, manifest["assets"])
                weight.write_bytes(b"abd")
                with self.assertRaisesRegex(ValueError, "fixed asset manifest"):
                    native.pinned_assets(model, contract)
                weight.write_bytes(b"abc")
                (model / "extra.jinja").write_text("unexpected")
                with self.assertRaisesRegex(ValueError, "fixed asset manifest"):
                    native.pinned_assets(model, contract)

    def test_manifest_cannot_select_another_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets.json").write_text(json.dumps({
                "model": "test-model", "revision": "other-revision", "assets": {},
            }))
            contract = {"target": "test-model", "revision": "test-revision",
                        "asset_manifest": "assets.json"}
            with mock.patch.object(native, "HERE", root):
                with self.assertRaisesRegex(ValueError, "contract differ"):
                    native.pinned_assets(root, contract)

    def test_prepare_requires_an_explicit_model_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["compare_native.py", "prepare", "--source", str(root),
                    "--snapshot", str(root / "snapshot"), "--output", str(root / "output")]
            with mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr", new_callable=io.StringIO) as errors:
                with self.assertRaises(SystemExit) as stopped:
                    native.main()
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("prepare requires --model", errors.getvalue())
            self.assertFalse((root / "snapshot").exists())
            self.assertFalse((root / "output").exists())


class MetalLockTests(unittest.TestCase):
    def test_shared_lock_rejects_a_second_owner_and_releases_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "metal.lock"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with native.metal_lock(path, 0):
                    inode = path.stat().st_ino
                    with self.assertRaises(TimeoutError):
                        with native.metal_lock(path, 0):
                            self.fail("A second owner acquired the lock")
                    raise RuntimeError("injected")
            with native.metal_lock(path, 0):
                self.assertEqual(path.stat().st_ino, inode)

    def test_runtime_holds_the_lock_before_starting_any_measured_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metal.lock"
            args = argparse.Namespace(metal_lock=path, lock_wait_seconds=0)
            with native.metal_lock(path, 0):
                with mock.patch.object(native, "run_runtime") as measured:
                    with self.assertRaises(TimeoutError):
                        native.runtime(args)
                    measured.assert_not_called()

            def check_ownership(_args):
                with self.assertRaises(TimeoutError):
                    with native.metal_lock(path, 0):
                        self.fail("Runtime did not hold its shared lock")

            with mock.patch.object(native, "run_runtime", side_effect=check_ownership) as measured:
                native.runtime(args)
                measured.assert_called_once_with(args)

    def test_invalid_wait_does_not_create_a_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metal.lock"
            for wait in [-1, float("nan"), float("inf")]:
                with self.assertRaises(ValueError):
                    with native.metal_lock(path, wait):
                        self.fail("Invalid wait was accepted")
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

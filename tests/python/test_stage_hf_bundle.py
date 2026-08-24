from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from email.message import Message
import hashlib
from http.client import HTTPException, IncompleteRead
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from urllib.parse import unquote, urlparse
from urllib.request import ProxyHandler, Request


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/stage_hf_bundle.py"
SPEC = importlib.util.spec_from_file_location("stage_hf_bundle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        final_url: str,
        content_length: int | None = None,
        content_range: str | None = None,
        content_encoding: str | None = None,
    ) -> None:
        self._payload = io.BytesIO(payload)
        self.status = status
        self._final_url = final_url
        self.closed = False
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        if content_range is not None:
            self.headers["Content-Range"] = content_range
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding

    def geturl(self) -> str:
        return self._final_url

    def getcode(self) -> int:
        return self.status

    def read(self, size: int = -1) -> bytes:
        return self._payload.read(size)

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.requests: list[Request] = []
        self.timeouts: list[float] = []
        self.override_status: int | None = None
        self.override_content_range: str | None = None
        self.override_content_length: int | None = None
        self.override_payload: bytes | None = None
        self.payload_sequence: list[bytes] = []
        self.final_url = "https://us.aws.cdn.hf.co/signed-object?signature=SECRET"

    def open(self, request: Request, *, timeout: float) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        name = unquote(urlparse(request.full_url).path.rsplit("/", 1)[-1])
        payload = self.payloads[name]
        range_header = request.get_header("Range")
        if range_header is None:
            start = 0
            status = 200
            content_range = None
        else:
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            status = 206
            content_range = f"bytes {start}-{len(payload) - 1}/{len(payload)}"
        body = payload[start:]
        if self.override_payload is not None:
            body = self.override_payload
        if self.payload_sequence:
            body = self.payload_sequence.pop(0)
        if self.override_status is not None:
            status = self.override_status
        if self.override_content_range is not None:
            content_range = self.override_content_range
        content_length = len(payload) - start
        if self.override_content_length is not None:
            content_length = self.override_content_length
        return FakeResponse(
            body,
            status=status,
            final_url=self.final_url,
            content_length=content_length,
            content_range=content_range,
        )


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def plan_for(payloads: dict[str, bytes]) -> object:
    artifacts = tuple(
        MODULE.Artifact(name, len(payload), digest(payload))
        for name, payload in sorted(payloads.items())
    )
    manifest = {
        artifact.path: {"sha256": artifact.sha256, "size": artifact.size}
        for artifact in artifacts
    }
    return MODULE.BundlePlan(
        profile_id="fixture-profile",
        repo_id="org/model",
        resolved_commit="a" * 40,
        source_lock_content_sha256="b" * 64,
        artifact_manifest_sha256=digest(MODULE._canonical_bytes(manifest)),
        artifacts=artifacts,
    )


class StagingCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.parent = Path(self.temporary.name).resolve()
        self.model_dir = self.parent / "bundle"
        self.payloads = {
            "config.json": b'{"model_type":"fixture"}\n',
            "model.safetensors-00001-of-00001.safetensors": b"safe-tensors-fixture",
        }
        self.plan = plan_for(self.payloads)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @property
    def staging(self) -> Path:
        return self.parent / ".bundle.apxinf-staging"

    @property
    def lock(self) -> Path:
        return self.parent / ".bundle.apxinf-stage.lock"

    def create_staging(self) -> None:
        self.staging.mkdir(mode=0o700)
        self.staging.chmod(0o700)

    def create_existing_bundle(self, *, cache: bool = False) -> None:
        self.model_dir.mkdir(mode=0o755)
        self.model_dir.chmod(0o755)
        for name, payload in self.payloads.items():
            path = self.model_dir / name
            path.write_bytes(payload)
            path.chmod(0o644)
        if cache:
            cache_dir = self.model_dir / ".cache/huggingface/download"
            cache_dir.mkdir(parents=True, mode=0o755)
            for directory in (self.model_dir / ".cache", cache_dir.parent, cache_dir):
                directory.chmod(0o755)
            metadata = cache_dir / "config.json.metadata"
            metadata.write_bytes(b"fixture cache metadata")
            metadata.chmod(0o644)
            lock = cache_dir / "config.json.lock"
            lock.write_bytes(b"")
            lock.chmod(0o664)


class DownloadTests(StagingCase):
    def test_stages_exact_tree_with_private_modes_and_receipt(self) -> None:
        opener = FakeOpener(self.payloads)
        receipt = MODULE.stage_bundle(self.plan, self.model_dir, opener=opener)

        self.assertTrue(receipt["passed"])
        self.assertTrue(receipt["published"])
        self.assertEqual(receipt["downloaded_bytes"], self.plan.total_bytes)
        self.assertEqual(receipt["resumed_from_bytes"], 0)
        self.assertEqual(receipt["reused_bytes"], 0)
        self.assertEqual(receipt["model_dir"], str(self.model_dir))
        self.assertEqual(
            receipt["policy"]["filesystem"]["trust_boundary"],
            "same-uid-local-filesystem-v1",
        )
        self.assertFalse(receipt["evidence"]["builtin_opener"])
        self.assertTrue(receipt["evidence"]["opener_injected"])
        self.assertTrue(receipt["evidence"]["lock_acquired"])
        self.assertTrue(receipt["evidence"]["network_used"])
        self.assertFalse(receipt["evidence"]["ambient_proxy_disabled"])
        self.assertTrue(receipt["evidence"]["published_by_this_invocation"])
        self.assertTrue(receipt["evidence"]["atomic_no_replace_publish_observed"])
        self.assertEqual(
            set(path.name for path in self.model_dir.iterdir()), set(self.payloads)
        )
        self.assertEqual(self.model_dir.stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.lock.stat().st_mode & 0o777, 0o600)
        for name, payload in self.payloads.items():
            path = self.model_dir / name
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.stat().st_nlink, 1)
        for request in opener.requests:
            headers = {key.lower(): value for key, value in request.header_items()}
            self.assertNotIn("authorization", headers)
            self.assertNotIn("cookie", headers)
            self.assertEqual(headers["accept-encoding"], "identity")
            self.assertTrue(request.full_url.startswith("https://huggingface.co/"))
        self.assertNotIn("SECRET", json.dumps(receipt))

    def test_builtin_opener_is_distinct_receipt_evidence(self) -> None:
        fake = FakeOpener(self.payloads)
        with mock.patch.object(MODULE, "_build_http_opener", return_value=fake):
            receipt = MODULE.stage_bundle(self.plan, self.model_dir)
        self.assertTrue(receipt["evidence"]["builtin_opener"])
        self.assertFalse(receipt["evidence"]["opener_injected"])
        self.assertTrue(receipt["evidence"]["ambient_proxy_disabled"])

    def test_resumes_with_exact_range_and_content_range(self) -> None:
        self.create_staging()
        artifact = self.plan.artifacts[0]
        payload = self.payloads[artifact.path]
        part = self.staging / f"{artifact.path}.part"
        prefix = payload[:5]
        part.write_bytes(prefix)
        part.chmod(0o600)
        # Pre-stage the other artifact to make the request under test unambiguous.
        other = self.plan.artifacts[1]
        complete = self.staging / other.path
        complete.write_bytes(self.payloads[other.path])
        complete.chmod(0o600)

        opener = FakeOpener(self.payloads)
        receipt = MODULE.stage_bundle(self.plan, self.model_dir, opener=opener)

        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(opener.requests[0].get_header("Range"), "bytes=5-")
        self.assertEqual(receipt["resumed_from_bytes"], 5)
        self.assertEqual(receipt["downloaded_bytes"], len(payload) - 5)
        self.assertEqual(receipt["reused_bytes"], len(self.payloads[other.path]))
        self.assertEqual((self.model_dir / artifact.path).read_bytes(), payload)

    def test_rejects_incorrect_content_range_without_publication(self) -> None:
        self.create_staging()
        artifact = self.plan.artifacts[0]
        part = self.staging / f"{artifact.path}.part"
        part.write_bytes(self.payloads[artifact.path][:3])
        part.chmod(0o600)
        opener = FakeOpener(self.payloads)
        opener.override_content_range = f"bytes 0-{artifact.size - 1}/{artifact.size}"

        with self.assertRaisesRegex(MODULE.StageError, "Content-Range"):
            MODULE.stage_bundle(self.plan, self.model_dir, opener=opener)

        self.assertFalse(self.model_dir.exists())
        self.assertEqual(part.read_bytes(), self.payloads[artifact.path][:3])

    def test_rejects_ignored_range_http_200(self) -> None:
        self.create_staging()
        artifact = self.plan.artifacts[0]
        part = self.staging / f"{artifact.path}.part"
        part.write_bytes(self.payloads[artifact.path][:3])
        part.chmod(0o600)
        opener = FakeOpener(self.payloads)
        opener.override_status = 200

        with self.assertRaisesRegex(MODULE.StageError, "Content-Range"):
            MODULE.stage_bundle(self.plan, self.model_dir, opener=opener)

    def test_hash_mismatch_restarts_once_then_succeeds(self) -> None:
        payloads = {"config.json": b"right"}
        plan = plan_for(payloads)
        opener = FakeOpener(payloads)
        opener.payload_sequence = [b"wrong", b"right"]

        receipt = MODULE.stage_bundle(plan, self.model_dir, opener=opener)

        self.assertTrue(receipt["published"])
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(receipt["downloaded_bytes"], 10)
        self.assertEqual(receipt["evidence"]["recovered_artifacts"], ["config.json"])
        self.assertEqual(receipt["evidence"]["recovery_bytes_discarded"], 5)

    def test_two_bad_hashes_fail_but_same_target_next_call_recovers(self) -> None:
        payloads = {"config.json": b"right"}
        plan = plan_for(payloads)
        bad = FakeOpener(payloads)
        bad.override_payload = b"wrong"
        with self.assertRaisesRegex(MODULE.StageError, "one safe restart"):
            MODULE.stage_bundle(plan, self.model_dir, opener=bad)
        self.assertFalse(self.model_dir.exists())
        part = self.staging / "config.json.part"
        self.assertEqual(part.read_bytes(), b"wrong")

        receipt = MODULE.stage_bundle(plan, self.model_dir, opener=FakeOpener(payloads))
        self.assertTrue(receipt["published"])
        self.assertEqual((self.model_dir / "config.json").read_bytes(), b"right")
        self.assertEqual(receipt["evidence"]["recovered_artifacts"], ["config.json"])

    def test_corrupt_completed_staging_file_is_safely_recovered(self) -> None:
        payloads = {"config.json": b"right"}
        plan = plan_for(payloads)
        self.create_staging()
        completed = self.staging / "config.json"
        completed.write_bytes(b"wrong")
        completed.chmod(0o600)

        receipt = MODULE.stage_bundle(plan, self.model_dir, opener=FakeOpener(payloads))

        self.assertTrue(receipt["published"])
        self.assertEqual((self.model_dir / "config.json").read_bytes(), b"right")
        self.assertEqual(receipt["evidence"]["recovery_bytes_discarded"], 5)

    def test_rejects_excess_payload(self) -> None:
        second = self.parent / "bundle-two"
        one_payload = {"config.json": self.payloads["config.json"]}
        opener = FakeOpener(one_payload)
        opener.override_payload = one_payload["config.json"] + b"x"
        with self.assertRaisesRegex(MODULE.StageError, "exceeded pinned size"):
            MODULE.stage_bundle(plan_for(one_payload), second, opener=opener)

    def test_missing_or_transformed_response_headers_fail_closed(self) -> None:
        artifact = self.plan.artifacts[0]
        response = FakeResponse(
            self.payloads[artifact.path],
            status=200,
            final_url="https://us.aws.cdn.hf.co/object",
            content_length=None,
        )
        with self.assertRaisesRegex(MODULE.StageError, "Content-Length"):
            MODULE._validate_response(response, artifact=artifact, start=0)

        response = FakeResponse(
            self.payloads[artifact.path],
            status=200,
            final_url="https://us.aws.cdn.hf.co/object",
            content_length=artifact.size,
            content_encoding="gzip",
        )
        with self.assertRaisesRegex(MODULE.StageError, "content encoding"):
            MODULE._validate_response(response, artifact=artifact, start=0)

    def test_duplicate_oversized_and_transfer_headers_fail_closed(self) -> None:
        artifact = self.plan.artifacts[0]
        cases: list[tuple[Message, str]] = []
        duplicate = Message()
        duplicate["Content-Length"] = str(artifact.size)
        duplicate["Content-Length"] = str(artifact.size)
        cases.append((duplicate, "duplicate Content-Length"))
        transfer = Message()
        transfer["Content-Length"] = str(artifact.size)
        transfer["Transfer-Encoding"] = "chunked"
        cases.append((transfer, "Transfer-Encoding"))
        oversized = Message()
        oversized["Content-Length"] = "1" * (MODULE.MAX_NUMERIC_HEADER_DIGITS + 1)
        cases.append((oversized, "Content-Length"))
        for headers, message in cases:
            with self.subTest(message=message):
                response = FakeResponse(
                    self.payloads[artifact.path],
                    status=200,
                    final_url="https://us.aws.cdn.hf.co/object",
                    content_length=artifact.size,
                )
                response.headers = headers
                with self.assertRaisesRegex(MODULE.StageError, message):
                    MODULE._validate_response(response, artifact=artifact, start=0)


class FilesystemSafetyTests(StagingCase):
    def test_dry_run_is_read_only_and_never_opens_network(self) -> None:
        opener = mock.Mock()
        receipt = MODULE.stage_bundle(
            self.plan, self.model_dir, dry_run=True, opener=opener
        )
        self.assertEqual(receipt["action"], "dry-run")
        self.assertEqual(receipt["would_action"], "publish-new")
        self.assertEqual(receipt["would_download_bytes"], self.plan.total_bytes)
        self.assertFalse(receipt["published"])
        self.assertFalse(receipt["evidence"]["lock_acquired"])
        self.assertFalse(receipt["evidence"]["network_used"])
        self.assertFalse(receipt["evidence"]["builtin_opener"])
        self.assertFalse(self.staging.exists())
        self.assertFalse(self.lock.exists())
        self.assertFalse(self.model_dir.exists())
        opener.open.assert_not_called()

    def test_reuses_strict_existing_bundle_without_network(self) -> None:
        self.create_existing_bundle(cache=True)
        opener = mock.Mock()
        receipt = MODULE.stage_bundle(self.plan, self.model_dir, opener=opener)

        self.assertEqual(receipt["action"], "reused-existing")
        self.assertTrue(receipt["published"])
        self.assertEqual(receipt["downloaded_bytes"], 0)
        self.assertEqual(receipt["reused_bytes"], self.plan.total_bytes)
        self.assertTrue(receipt["evidence"]["lock_acquired"])
        self.assertTrue(receipt["evidence"]["existing_bundle_verified"])
        self.assertTrue(receipt["evidence"]["cache_tree_present"])
        self.assertFalse(receipt["evidence"]["network_used"])
        self.assertFalse(receipt["evidence"]["builtin_opener"])
        self.assertFalse(receipt["evidence"]["published_by_this_invocation"])
        opener.open.assert_not_called()

    def test_existing_only_reuses_existing_and_records_enforcement(self) -> None:
        self.create_existing_bundle()
        dry_receipt = MODULE.stage_bundle(
            self.plan, self.model_dir, existing_only=True, dry_run=True
        )
        self.assertEqual(dry_receipt["would_action"], "reused-existing")
        self.assertFalse(dry_receipt["published"])
        self.assertTrue(dry_receipt["evidence"]["existing_only_enforced"])
        self.assertFalse(dry_receipt["evidence"]["lock_acquired"])
        self.assertFalse(self.lock.exists())

        opener = mock.Mock()
        receipt = MODULE.stage_bundle(
            self.plan, self.model_dir, existing_only=True, opener=opener
        )
        self.assertEqual(receipt["action"], "reused-existing")
        self.assertTrue(receipt["policy"]["operation"]["existing_only_requested"])
        self.assertTrue(receipt["evidence"]["existing_only_enforced"])
        self.assertFalse(receipt["evidence"]["network_used"])
        opener.open.assert_not_called()

    def test_existing_only_missing_never_inspects_staging_or_network(self) -> None:
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                opener = mock.Mock()
                with mock.patch.object(MODULE, "_inspect_staging") as inspect_staging:
                    with mock.patch.object(
                        MODULE, "_build_http_opener"
                    ) as build_opener:
                        with self.assertRaisesRegex(MODULE.StageError, "existing-only"):
                            MODULE.stage_bundle(
                                self.plan,
                                self.model_dir,
                                existing_only=True,
                                dry_run=dry_run,
                                opener=opener,
                            )
                inspect_staging.assert_not_called()
                build_opener.assert_not_called()
                opener.open.assert_not_called()
                self.assertFalse(self.staging.exists())
                self.assertFalse(self.lock.exists())

    def test_existing_only_target_disappearing_after_preflight_fails_closed(
        self,
    ) -> None:
        self.create_existing_bundle()
        moved = self.parent / "moved-after-preflight"

        @contextmanager
        def disappearing_lock(target: Path) -> object:
            target.rename(moved)
            yield

        opener = mock.Mock()
        with mock.patch.object(
            MODULE, "_exclusive_stage_lock", side_effect=disappearing_lock
        ):
            with mock.patch.object(MODULE, "_inspect_staging") as inspect_staging:
                with mock.patch.object(MODULE, "_build_http_opener") as build_opener:
                    with self.assertRaisesRegex(MODULE.StageError, "disappeared"):
                        MODULE.stage_bundle(
                            self.plan,
                            self.model_dir,
                            existing_only=True,
                            opener=opener,
                        )
        inspect_staging.assert_not_called()
        build_opener.assert_not_called()
        opener.open.assert_not_called()
        self.assertFalse(self.model_dir.exists())
        self.assertTrue(moved.exists())

    def test_dry_run_validates_existing_without_lock_or_publication_claim(self) -> None:
        self.create_existing_bundle()
        receipt = MODULE.stage_bundle(self.plan, self.model_dir, dry_run=True)
        self.assertEqual(receipt["action"], "dry-run")
        self.assertEqual(receipt["would_action"], "reused-existing")
        self.assertFalse(receipt["published"])
        self.assertTrue(receipt["evidence"]["existing_bundle_verified"])
        self.assertFalse(receipt["evidence"]["lock_acquired"])
        self.assertFalse(receipt["evidence"]["published_by_this_invocation"])
        self.assertFalse(self.lock.exists())

    def test_rejects_invalid_existing_tree_without_network(self) -> None:
        cases = ("extra", "bad-hash", "symlink", "hardlink", "unsafe-mode")
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                model_dir = self.parent / f"existing-{index}"
                model_dir.mkdir(mode=0o755)
                for name, payload in self.payloads.items():
                    path = model_dir / name
                    path.write_bytes(payload)
                    path.chmod(0o644)
                artifact = self.plan.artifacts[0]
                path = model_dir / artifact.path
                if case == "extra":
                    (model_dir / "unexpected.txt").write_bytes(b"x")
                elif case == "bad-hash":
                    path.write_bytes(b"x" * artifact.size)
                elif case == "symlink":
                    path.unlink()
                    path.symlink_to("missing")
                elif case == "hardlink":
                    outside = self.parent / f"existing-hardlink-{index}"
                    outside.write_bytes(self.payloads[artifact.path])
                    path.unlink()
                    os.link(outside, path)
                else:
                    path.chmod(0o664)
                opener = mock.Mock()
                with self.assertRaises(MODULE.StageError):
                    MODULE.stage_bundle(self.plan, model_dir, opener=opener)
                opener.open.assert_not_called()

    def test_optional_cache_rejects_link_script_and_extra_top_level(self) -> None:
        cases = ("cache-symlink", "cache-hardlink", "cache-script")
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                model_dir = self.parent / f"cached-{index}"
                model_dir.mkdir(mode=0o755)
                for name, payload in self.payloads.items():
                    path = model_dir / name
                    path.write_bytes(payload)
                    path.chmod(0o644)
                cache = model_dir / ".cache"
                cache.mkdir(mode=0o755)
                if case == "cache-symlink":
                    (cache / "entry").symlink_to("missing")
                elif case == "cache-hardlink":
                    outside = self.parent / f"cache-outside-{index}"
                    outside.write_bytes(b"data")
                    os.link(outside, cache / "entry")
                else:
                    script = cache / "entry.txt"
                    script.write_bytes(b"#!/bin/sh\n")
                    script.chmod(0o644)
                with self.assertRaises(MODULE.StageError):
                    MODULE.stage_bundle(self.plan, model_dir, opener=mock.Mock())

    def test_refuses_unexpected_symlink_and_hardlink_in_staging(self) -> None:
        cases = ("unexpected", "symlink", "hardlink", "oversized")
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                model_dir = self.parent / f"bundle-{index}"
                staging = self.parent / f".{model_dir.name}.apxinf-staging"
                staging.mkdir(mode=0o700)
                staging.chmod(0o700)
                artifact = self.plan.artifacts[0]
                if case == "unexpected":
                    path = staging / "surprise.txt"
                    path.write_bytes(b"surprise")
                    path.chmod(0o600)
                elif case == "symlink":
                    (staging / f"{artifact.path}.part").symlink_to("missing")
                else:
                    if case == "hardlink":
                        outside = self.parent / f"outside-{index}"
                        outside.write_bytes(b"x")
                        outside.chmod(0o600)
                        os.link(outside, staging / f"{artifact.path}.part")
                    else:
                        part = staging / f"{artifact.path}.part"
                        part.write_bytes(b"x" * (artifact.size + 1))
                        part.chmod(0o600)
                with self.assertRaises(MODULE.StageError):
                    MODULE.stage_bundle(
                        self.plan, model_dir, opener=FakeOpener(self.payloads)
                    )

    def test_refuses_concurrent_writer_lock(self) -> None:
        with MODULE._exclusive_stage_lock(self.model_dir):
            with self.assertRaisesRegex(MODULE.StageError, "another bundle staging"):
                MODULE.stage_bundle(
                    self.plan, self.model_dir, opener=FakeOpener(self.payloads)
                )
        self.assertFalse(self.model_dir.exists())

    def test_byte_and_timeout_caps_fail_before_network(self) -> None:
        opener = mock.Mock()
        with self.assertRaisesRegex(MODULE.StageError, "max-total-bytes"):
            MODULE.stage_bundle(
                self.plan,
                self.model_dir,
                max_total_bytes=self.plan.total_bytes - 1,
                opener=opener,
            )
        with self.assertRaisesRegex(MODULE.StageError, "timeout-seconds"):
            MODULE.stage_bundle(
                self.plan, self.model_dir, timeout_seconds=0, opener=opener
            )
        opener.open.assert_not_called()

    def test_refuses_relative_destination(self) -> None:
        with self.assertRaisesRegex(MODULE.StageError, "absolute"):
            MODULE.stage_bundle(self.plan, Path("relative/bundle"), dry_run=True)

    def test_refuses_destination_whose_parent_traverses_symlink(self) -> None:
        real_parent = self.parent / "real-parent"
        real_parent.mkdir()
        linked_parent = self.parent / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(MODULE.StageError, "parent must not traverse"):
            MODULE.stage_bundle(self.plan, linked_parent / "bundle", dry_run=True)


class NetworkPolicyTests(unittest.TestCase):
    def test_redirect_host_matching_is_label_aware_and_https_only(self) -> None:
        allowed = (
            "https://huggingface.co/org/model/resolve/commit/file",
            "https://us.aws.cdn.hf.co/object?signature=secret",
            "https://cas-bridge.xethub.hf.co/object",
            "https://sub.huggingface.co/object",
        )
        forbidden = (
            "http://huggingface.co/object",
            "https://evilhf.co/object",
            "https://hf.co.attacker.example/object",
            "https://huggingface.co.example/object",
            "https://user:password@huggingface.co/object",
            "https://huggingface.co:444/object",
            "https://huggingface.co/object#fragment",
        )
        for url in allowed:
            with self.subTest(url=url):
                MODULE._validate_download_url(url, label="fixture")
        for url in forbidden:
            with self.subTest(url=url), self.assertRaises(MODULE.StageError) as caught:
                MODULE._validate_download_url(url, label="fixture")
            self.assertNotIn(url, str(caught.exception))

    def test_redirect_handler_rejects_off_domain_before_request(self) -> None:
        handler = MODULE._HuggingFaceContentRedirectHandler()
        request = Request("https://huggingface.co/org/model/resolve/commit/file")
        for location in (
            "https://attacker.example/collect?secret=SIGNED",
            "//evilhf.co/collect",
            "http://us.aws.cdn.hf.co/object",
        ):
            with self.subTest(location=location), self.assertRaises(MODULE.StageError):
                handler.redirect_request(
                    request, io.BytesIO(), 302, "Found", Message(), location
                )

    def test_redirect_handler_accepts_relative_and_official_content_domain(
        self,
    ) -> None:
        handler = MODULE._HuggingFaceContentRedirectHandler()
        request = Request("https://huggingface.co/org/model/resolve/commit/file")
        for location in (
            "/org/model/resolve/commit/file",
            "https://us.aws.cdn.hf.co/object?signature=secret",
        ):
            with self.subTest(location=location):
                redirected = handler.redirect_request(
                    request, io.BytesIO(), 302, "Found", Message(), location
                )
                assert redirected is not None
                MODULE._validate_download_url(redirected.full_url, label="fixture")

    def test_http_opener_disables_ambient_proxies(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1"},
            clear=False,
        ):
            opener = MODULE._build_http_opener()
        proxy_handlers = [
            handler for handler in opener.handlers if isinstance(handler, ProxyHandler)
        ]
        self.assertEqual(proxy_handlers, [])


def fixture_contract(
    *, private: bool = False, remote_code: bool = False
) -> tuple[dict[str, object], dict[str, object], object]:
    payloads = {
        "config.json": b"config",
        "model.safetensors.index.json": b"index",
        "tokenizer_config.json": b"tokenizer",
        "model.safetensors": b"weights",
    }
    artifacts = {
        name: {"size": len(payload), "sha256": digest(payload)}
        for name, payload in payloads.items()
    }
    repo = "org/model"
    commit = "a" * 40
    lock: dict[str, object] = {
        "format": MODULE.SOURCE_LOCK_FORMAT,
        "repo_id": repo,
        "requested_revision": commit,
        "resolved_commit": commit,
        "source": {
            "url": f"https://huggingface.co/{repo}",
            "private": private,
            "gated": False,
            "disabled": False,
        },
        "security": {
            "remote_code_indicators": {
                "auto_map_keys": ["AutoModel"] if remote_code else [],
                "python_files": ["modeling.py"] if remote_code else [],
            },
            "unsafe_weight_files": [],
            "safetensors_only_plan": True,
        },
        "weights": {
            "format": "safetensors",
            "index_file": "model.safetensors.index.json",
            "files": [
                {
                    "path": "model.safetensors",
                    "size": artifacts["model.safetensors"]["size"],
                    "sha256": artifacts["model.safetensors"]["sha256"],
                    "git_blob_sha1": "b" * 40,
                }
            ],
            "total_bytes": artifacts["model.safetensors"]["size"],
        },
        "metadata": {
            "files": [
                {
                    "path": name,
                    "size": artifacts[name]["size"],
                    "sha256": artifacts[name]["sha256"],
                    "git_blob_sha1": "c" * 40,
                }
                for name in sorted(MODULE.SOURCE_LOCK_METADATA_ARTIFACTS)
            ]
        },
        "policy_receipt": {
            "metadata_only": True,
            "weight_payload_bytes_downloaded": 0,
            "remote_code_executed": False,
            "hf_token_read": False,
        },
    }
    lock["content_sha256"] = digest(MODULE._canonical_bytes(lock))
    source = {
        "config_sha256": artifacts["config.json"]["sha256"],
        "license": "Apache-2.0",
        "repo_id": repo,
        "resolved_commit": commit,
        "source_lock_content_sha256": lock["content_sha256"],
    }
    contract = MODULE.ProfileContract(
        profile_id="fixture",
        repo_id=repo,
        resolved_commit=commit,
        source_lock_content_sha256=lock["content_sha256"],
        source=source,
        artifacts=artifacts,
    )
    profile = {
        "format": MODULE.PROFILE_FORMAT,
        "profile_id": "fixture",
        "source": source,
        "artifacts": json.loads(json.dumps(artifacts)),
        "binary": {},
        "runtime": {},
        "gate": {},
        "memory_smoke": {},
        "oracle": {},
    }
    return profile, lock, contract


class ContractTests(unittest.TestCase):
    def test_accepts_jointly_pinned_profile_and_source_lock(self) -> None:
        profile, lock, contract = fixture_contract()
        plan = MODULE.validate_profile_and_source_lock(profile, lock, contract=contract)
        self.assertEqual(plan.repo_id, "org/model")
        self.assertEqual(
            plan.total_bytes, sum(x["size"] for x in profile["artifacts"].values())
        )
        self.assertEqual(
            plan.artifact_manifest_sha256,
            digest(MODULE._canonical_bytes(profile["artifacts"])),
        )

    def test_rejects_tampered_profile_allowlist_and_source_lock(self) -> None:
        profile, lock, contract = fixture_contract()
        profile["artifacts"]["config.json"]["size"] += 1
        with self.assertRaisesRegex(MODULE.StageError, "allowlist"):
            MODULE.validate_profile_and_source_lock(profile, lock, contract=contract)

        profile, lock, contract = fixture_contract()
        lock["resolved_commit"] = "d" * 40
        with self.assertRaisesRegex(MODULE.StageError, "content hash"):
            MODULE.validate_profile_and_source_lock(profile, lock, contract=contract)

    def test_rejects_private_or_remote_code_source_locks(self) -> None:
        for kwargs in ({"private": True}, {"remote_code": True}):
            with self.subTest(kwargs=kwargs):
                profile, lock, contract = fixture_contract(**kwargs)
                with self.assertRaises(MODULE.StageError):
                    MODULE.validate_profile_and_source_lock(
                        profile, lock, contract=contract
                    )

    def test_rejects_executable_artifact_and_unsafe_json(self) -> None:
        profile, lock, contract = fixture_contract()
        profile["artifacts"]["modeling.py"] = {
            "size": 1,
            "sha256": "0" * 64,
        }
        with self.assertRaisesRegex(MODULE.StageError, "executable code"):
            MODULE.validate_profile_and_source_lock(profile, lock, contract=contract)
        for payload in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.subTest(payload=payload), self.assertRaises(MODULE.StageError):
                MODULE._parse_json(payload, "fixture")

    def test_current_qwen_files_match_embedded_contract_when_present(self) -> None:
        source_lock = MODULE.DEFAULT_SOURCE_LOCK
        if not source_lock.exists():
            self.skipTest(
                "local metadata-only source lock is intentionally not checked in"
            )
        plan = MODULE.load_fixed_plan(MODULE.DEFAULT_PROFILE, source_lock)
        self.assertEqual(plan.profile_id, MODULE.QWEN_CONTRACT.profile_id)
        self.assertEqual(plan.total_bytes, 1_759_828_853)


class CliTests(StagingCase):
    def invoke_with_failure(
        self, error: BaseException
    ) -> tuple[int, dict[str, object], int]:
        output = io.StringIO()
        with mock.patch.object(MODULE, "load_fixed_plan", side_effect=error):
            with redirect_stdout(output):
                code = MODULE.main(["--model-dir", str(self.model_dir)])
        lines = output.getvalue().splitlines()
        return code, json.loads(lines[0]), len(lines)

    def test_main_emits_exactly_one_success_json_line(self) -> None:
        output = io.StringIO()
        with mock.patch.object(MODULE, "load_fixed_plan", return_value=self.plan):
            with redirect_stdout(output):
                code = MODULE.main(["--model-dir", str(self.model_dir), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertTrue(json.loads(output.getvalue())["passed"])
        self.assertFalse(self.lock.exists())

    def test_main_existing_only_flag_reaches_core_contract(self) -> None:
        self.create_existing_bundle()
        output = io.StringIO()
        with mock.patch.object(MODULE, "load_fixed_plan", return_value=self.plan):
            with redirect_stdout(output):
                code = MODULE.main(
                    ["--model-dir", str(self.model_dir), "--existing-only"]
                )
        self.assertEqual(code, 0)
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt["action"], "reused-existing")
        self.assertTrue(receipt["evidence"]["existing_only_enforced"])
        self.assertFalse(receipt["evidence"]["network_used"])

    def test_main_failure_is_one_json_line_and_nonzero(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = MODULE.main([])
        self.assertNotEqual(code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        receipt = json.loads(output.getvalue())
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["error"]["code"], "HF_BUNDLE_STAGE_FAILED")

    def test_main_transport_and_unexpected_failures_are_sanitized_one_line(
        self,
    ) -> None:
        failures = (
            HTTPException("https://us.aws.cdn.hf.co/object?signature=SECRET"),
            IncompleteRead(b"partial", 100),
            RuntimeError("https://attacker.example/?secret=SECRET"),
        )
        for error in failures:
            with self.subTest(error=type(error).__name__):
                code, receipt, line_count = self.invoke_with_failure(error)
                self.assertEqual(code, 2)
                self.assertEqual(line_count, 1)
                self.assertFalse(receipt["passed"])
                self.assertNotIn("SECRET", json.dumps(receipt))

    def test_main_keyboard_interrupt_is_one_json_line_and_exit_130(self) -> None:
        code, receipt, line_count = self.invoke_with_failure(KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertEqual(line_count, 1)
        self.assertEqual(receipt["error"]["code"], "HF_BUNDLE_STAGE_INTERRUPTED")


if __name__ == "__main__":
    unittest.main()

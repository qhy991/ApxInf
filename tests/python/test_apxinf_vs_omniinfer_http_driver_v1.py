from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest


DRIVER_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "cross_runtime"
    / "apxinf_vs_omniinfer_http_driver_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "apxinf_vs_omniinfer_http_driver_v1", DRIVER_PATH
)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def response_fixture(arm: str, token: int = 7) -> dict:
    settings = {
        "ignore_eos": True,
        "max_tokens": 128,
        "seed": 0,
        "temperature": 0,
    }
    verbose = {
        "prompt": driver.RENDERED_PROMPT,
        "tokens": [token] * 128,
        "tokens_evaluated": 13,
        "tokens_predicted": 128,
        "stop_type": "limit",
    }
    if arm == "B":
        settings.update(
            {
                "policy": "-inf-before-greedy",
                "suppressed_eog_token_ids": driver.SUPPRESSED_EOG_TOKEN_IDS,
            }
        )
        verbose.update(
            {
                "qualification": driver.APX_SERVER_QUALIFICATION,
                "formal_evidence_eligible": False,
                "prompt_token_ids": driver.PROMPT_TOKEN_IDS,
            }
        )
    else:
        settings["logit_bias"] = [
            {"bias": None, "token": token_id}
            for token_id in driver.SUPPRESSED_EOG_TOKEN_IDS
        ]
        verbose.update({"id_slot": 0, "tokens_cached": 140, "truncated": False})
    verbose["generation_settings"] = settings
    return {
        "object": "chat.completion",
        "model": driver.MODEL_ALIAS,
        "choices": [
            {
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "fixture"},
            }
        ],
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 128,
            "total_tokens": 141,
            **({"prompt_tokens_details": {"cached_tokens": 0}} if arm == "G" else {}),
        },
        **(
            {"timings": {"prompt_n": 13, "predicted_n": 128, "cache_n": 0}}
            if arm == "G"
            else {}
        ),
        "__verbose": verbose,
    }


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False

    def fileno(self) -> int:
        return -1 if self.closed else 23

    def getsockname(self) -> tuple[str, int]:
        return ("127.0.0.1", 43123)

    def getpeername(self) -> tuple[str, int]:
        return ("127.0.0.1", 9001)

    def setsockopt(self, *_: object) -> None:
        return None

    def sendall(self, value: bytes) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


class FakeResponse:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.status = 200
        self.version = 11
        self.will_close = False

    def begin(self) -> None:
        return None

    def getheaders(self) -> list[tuple[str, str]]:
        return [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(self.raw))),
        ]

    def read(self, _: int) -> bytes:
        return self.raw

    def close(self) -> None:
        return None


class ContractAndParserTests(unittest.TestCase):
    def test_canonical_request_is_exact_frozen_383_bytes(self) -> None:
        receipt = driver.validate_static_contract()
        self.assertEqual(receipt["request_size_bytes"], 383)
        self.assertEqual(receipt["request_sha256"], driver.REQUEST_SHA256)
        self.assertEqual(
            driver.REQUEST_BYTES, driver.canonical_json_bytes(driver.REQUEST)
        )

    def test_strict_json_rejects_duplicates_nonfinite_and_trailing_data(self) -> None:
        self.assertEqual(
            driver.parse_strict_json_document(b'{"ok":true}'), {"ok": True}
        )
        for raw in (
            b'{"ok":true,"ok":false}',
            b'{"value":NaN}',
            b'{"ok":true} trailing',
        ):
            with self.subTest(raw=raw), self.assertRaises(driver.AdmissionError):
                driver.parse_strict_json_document(raw)

    def test_exclusive_output_is_one_document_and_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "raw.json"
            output = driver.ExclusiveJsonOutput(str(output_path))
            receipt = output.write({"ok": True})
            self.assertTrue(receipt["exclusive_create"])
            self.assertEqual(output_path.read_bytes(), b'{"ok":true}\n')
            with self.assertRaises(FileExistsError):
                driver.ExclusiveJsonOutput(str(output_path))

    def test_quiet_host_gate_status_must_match_supplied_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "quiet.json"
            receipt_path.write_bytes(b'{"passed":true}')
            passed = driver.quiet_host_gate(
                argparse.Namespace(
                    quiet_host_status="passed",
                    quiet_host_receipt=str(receipt_path),
                )
            )
            self.assertTrue(passed["passed"])
            with self.assertRaises(driver.AdmissionError):
                driver.quiet_host_gate(
                    argparse.Namespace(
                        quiet_host_status="failed",
                        quiet_host_receipt=str(receipt_path),
                    )
                )

    def test_fixed_schedule_is_balanced_with_four_warmups_and_sixteen_blocks(
        self,
    ) -> None:
        schedule = driver.declared_schedule()
        warmups = [entry for entry in schedule if entry["phase"] == "warmup"]
        measured = [entry for entry in schedule if entry["phase"] == "measured"]
        self.assertEqual(
            [entry["order"] for entry in warmups], list(driver.WARMUP_ORDERS)
        )
        self.assertEqual(len(measured), 64)
        self.assertEqual(sum(entry["order"] == "BG" for entry in measured), 32)
        self.assertEqual(sum(entry["order"] == "GB" for entry in measured), 32)
        for block in range(1, 17):
            entries = [entry for entry in measured if entry["block"] == block]
            self.assertEqual(len(entries), 4)
            self.assertEqual(sum(entry["order"] == "BG" for entry in entries), 2)
            self.assertEqual(sum(entry["order"] == "GB" for entry in entries), 2)


class PersistentTransportTests(unittest.TestCase):
    def test_complete_wire_wall_ends_before_json_parse_and_validation(self) -> None:
        fake_socket = FakeSocket()
        raw = b'{"ok":true}'
        clock_values = iter((100, 200, 220, 230, 240, 250))
        connection = driver.PersistentHttpConnection(
            "http://127.0.0.1:9001",
            "fixture",
            socket_factory=lambda *_args, **_kwargs: fake_socket,
            response_factory=lambda _socket: FakeResponse(raw),
            clock_ns=lambda: next(clock_values),
        )
        connection.connect()
        payload, validated, transport = connection.request_json(
            "POST",
            "/v1/chat/completions",
            driver.REQUEST_BYTES,
            primary_timed=True,
            semantic_validator=lambda value: {"ok": value["ok"]},
        )
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(validated, {"ok": True})
        self.assertEqual(len(fake_socket.sent), 1)
        self.assertTrue(fake_socket.sent[0].endswith(driver.REQUEST_BYTES))
        self.assertEqual(transport["single_sendall_call_count"], 1)
        self.assertEqual(transport["client_full_response_wall_ns"], 100)
        self.assertEqual(transport["json_parse_ns"], 10)
        self.assertEqual(transport["semantic_validation_ns"], 10)
        self.assertTrue(transport["json_parse_excluded_from_wall"])
        self.assertTrue(transport["semantic_validation_excluded_from_wall"])
        self.assertLess(
            transport["ended_monotonic_ns"],
            220,
        )
        connection.close()

    def test_transport_does_not_accept_non_loopback_or_https_endpoints(self) -> None:
        for endpoint in (
            "https://127.0.0.1:9000",
            "http://localhost:9000",
            "http://0.0.0.0:9000",
        ):
            with (
                self.subTest(endpoint=endpoint),
                self.assertRaises(driver.AdmissionError),
            ):
                driver.PersistentHttpConnection(endpoint, "bad")

    def test_semantic_failure_receipts_identify_the_exact_request(self) -> None:
        fake_socket = FakeSocket()
        raw = b'{"ok":true}'
        connection = driver.PersistentHttpConnection(
            "http://127.0.0.1:9001",
            "fixture",
            socket_factory=lambda *_args, **_kwargs: fake_socket,
            response_factory=lambda _socket: FakeResponse(raw),
        )
        connection.connect()
        with self.assertRaises(driver.TransportFailure) as caught:
            connection.request_json(
                "POST",
                "/v1/chat/completions",
                driver.REQUEST_BYTES,
                primary_timed=True,
                semantic_validator=lambda _value: driver.require(False, "fixture"),
            )
        observation = caught.exception.observation
        self.assertEqual(observation["method"], "POST")
        self.assertEqual(observation["path"], "/v1/chat/completions")
        self.assertTrue(observation["primary_timed_interval"])
        self.assertEqual(observation["attempted_request_index_on_connection"], 1)
        self.assertEqual(observation["status"], 200)
        self.assertEqual(observation["http_version"], 11)
        self.assertEqual(
            observation["response_headers"]["content-length"], [str(len(raw))]
        )
        connection.close()

    def test_generation_failure_preserves_the_successful_clear_receipt(self) -> None:
        class ClearThenFail:
            def __init__(self) -> None:
                self.calls = 0

            def request_json(self, *_args: object, **_kwargs: object) -> tuple:
                self.calls += 1
                if self.calls == 1:
                    return (
                        {"ok": True},
                        {"acknowledged": True},
                        {"ended_monotonic_ns": 10},
                    )
                raise driver.TransportFailure(
                    "generation failed",
                    {"path": "/v1/chat/completions"},
                )

        apx = ClearThenFail()
        with self.assertRaises(driver.TransportFailure) as caught:
            driver.run_one_sample(
                arm="B",
                sequence_index=1,
                schedule_entry={
                    "phase": "measured",
                    "block": 1,
                    "pair_index": 1,
                    "order": "BG",
                },
                apx=apx,
                omni=object(),
                omni_clear=object(),
                args=argparse.Namespace(),
            )
        observation = caught.exception.observation
        self.assertEqual(observation["path"], "/v1/chat/completions")
        self.assertEqual(observation["sample_context"]["arm"], "B")
        self.assertEqual(
            observation["successful_cache_clear"]["validated"],
            {"acknowledged": True},
        )


class SemanticAndStatisticsTests(unittest.TestCase):
    def test_omni_state_cross_checks_backend_endpoint_and_runtime_identity(
        self,
    ) -> None:
        state = {
            "backend": "llama.cpp-mac",
            "backend_ready": True,
            "backend_pid": 1234,
            "backend_port": 51090,
            "generation": 1,
            "client_endpoint": "http://127.0.0.1:51090",
            "model_path": driver.MODEL_ALIAS,
            "runtime": {
                "pid": 1234,
                "port": 51090,
                "client_endpoint": "http://127.0.0.1:51090",
            },
        }
        receipt = driver.validate_omni_state(state)
        self.assertTrue(receipt["runtime_identity_cross_checked"])
        for path, replacement in (
            (("backend_port",), 51091),
            (("runtime", "pid"), 4321),
            (("runtime", "port"), 51091),
            (("runtime", "client_endpoint"), "http://127.0.0.1:51091"),
        ):
            mutated = {
                **state,
                "runtime": dict(state["runtime"]),
            }
            target = mutated
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = replacement
            with self.subTest(path=path), self.assertRaises(driver.AdmissionError):
                driver.validate_omni_state(mutated)

    def test_clear_endpoint_is_bound_to_the_gateway_or_reported_backend(self) -> None:
        gateway = "http://127.0.0.1:9000"
        backend = "http://127.0.0.1:51090"
        gateway_receipt = driver.validate_omni_clear_binding(
            generation_endpoint=gateway,
            clear_endpoint=gateway,
            resident_backend_endpoint=backend,
            clear_contract="omni-gateway",
        )
        self.assertEqual(gateway_receipt["bound_to"], "generation-gateway")
        backend_receipt = driver.validate_omni_clear_binding(
            generation_endpoint=gateway,
            clear_endpoint=backend,
            resident_backend_endpoint=backend,
            clear_contract="llama-slot-erase",
        )
        self.assertEqual(
            backend_receipt["bound_to"], "resident-backend-from-omni-state"
        )
        for contract, wrong_endpoint in (
            ("omni-gateway", backend),
            ("llama-slot-erase", "http://127.0.0.1:51091"),
        ):
            with (
                self.subTest(contract=contract),
                self.assertRaises(driver.AdmissionError),
            ):
                driver.validate_omni_clear_binding(
                    generation_endpoint=gateway,
                    clear_endpoint=wrong_endpoint,
                    resident_backend_endpoint=backend,
                    clear_contract=contract,
                )

    def test_both_policy_encodings_normalize_to_the_same_five_eog_tokens(self) -> None:
        apx = driver.validate_chat_response(response_fixture("B"), "B")
        omni = driver.validate_chat_response(response_fixture("G", token=8), "G")
        self.assertEqual(
            apx["generation_policy"]["suppressed_eog_token_ids"],
            driver.SUPPRESSED_EOG_TOKEN_IDS,
        )
        self.assertEqual(
            omni["generation_policy"]["suppressed_eog_token_ids"],
            driver.SUPPRESSED_EOG_TOKEN_IDS,
        )
        self.assertTrue(apx["prompt_token_ids_observed_in_response"])
        self.assertEqual(apx["prompt_token_ids"], driver.PROMPT_TOKEN_IDS)
        self.assertFalse(omni["prompt_token_ids_observed_in_response"])
        self.assertIsNone(omni["prompt_token_ids"])
        self.assertNotEqual(
            apx["generated_token_ids_sha256"], omni["generated_token_ids_sha256"]
        )

    def test_eog_token_in_raw_output_is_rejected(self) -> None:
        with self.assertRaises(driver.AdmissionError):
            driver.validate_chat_response(
                response_fixture("B", token=driver.SUPPRESSED_EOG_TOKEN_IDS[0]), "B"
            )

    def test_omni_response_rejects_warm_wrong_slot_or_truncated_generation(
        self,
    ) -> None:
        mutations = (
            ("usage", "prompt_tokens_details", "cached_tokens", 13),
            ("timings", "cache_n", None, 13),
            ("__verbose", "id_slot", None, 9),
            ("__verbose", "id_slot", None, False),
            ("__verbose", "tokens_cached", None, 139),
            ("__verbose", "truncated", None, True),
        )
        for outer, inner, leaf, replacement in mutations:
            response = response_fixture("G")
            if leaf is None:
                response[outer][inner] = replacement
            else:
                response[outer][inner][leaf] = replacement
            with (
                self.subTest(field=(outer, inner, leaf)),
                self.assertRaises(driver.AdmissionError),
            ):
                driver.validate_chat_response(response, "G")

    def test_block_statistics_use_omni_minus_apx_without_ranking_claim(self) -> None:
        pairs: list[dict] = []
        for block in range(1, 17):
            orders = driver.ODD_BLOCK_ORDERS if block % 2 else driver.EVEN_BLOCK_ORDERS
            for pair_index, order in enumerate(orders, start=1):
                apx_wall = 100.0 + block * 0.1 + pair_index * 0.01
                samples = {
                    "B": {
                        "arm": "B",
                        "transport": {"client_full_response_wall_ms": apx_wall},
                    },
                    "G": {
                        "arm": "G",
                        "transport": {"client_full_response_wall_ms": apx_wall + 10.0},
                    },
                }
                pairs.append(
                    {
                        "block": block,
                        "pair_index": pair_index,
                        "order": order,
                        "samples": [samples[arm] for arm in order],
                    }
                )
        result = driver.analyze_measured_pairs(pairs, quiet_host_passed=True)
        primary = result["primary_omniinfer_minus_apxinf_client_wall_ms"]
        self.assertAlmostEqual(primary["mean"], 10.0)
        self.assertFalse(result["formal_summary_allowed"])
        self.assertFalse(result["engine_winner_or_ranking_claim_allowed"])

    def test_per_runtime_determinism_does_not_require_cross_runtime_equality(
        self,
    ) -> None:
        pairs: list[dict] = []
        for index in range(driver.EXPECTED_REQUESTS_PER_ARM):
            pairs.append(
                {
                    "samples": [
                        {
                            "arm": "B",
                            "validated": {
                                "generated_token_ids_sha256": "b" * 64,
                                "content_sha256": "c" * 64,
                            },
                        },
                        {
                            "arm": "G",
                            "validated": {
                                "generated_token_ids_sha256": "d" * 64,
                                "content_sha256": "e" * 64,
                            },
                        },
                    ]
                }
            )
        result = driver.validate_per_runtime_determinism(pairs[:4], pairs[4:])
        self.assertTrue(result["per_runtime"]["apxinf"]["deterministic_within_runtime"])
        self.assertTrue(
            result["per_runtime"]["omniinfer"]["deterministic_within_runtime"]
        )
        self.assertFalse(result["cross_runtime_trajectory_equality_required"])
        self.assertTrue(result["cross_runtime_trajectory_hash_comparison_omitted"])


if __name__ == "__main__":
    unittest.main()

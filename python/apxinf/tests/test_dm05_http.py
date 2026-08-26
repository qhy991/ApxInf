from __future__ import annotations

import json

from test_dm05_policy import FakeBackend, FakeCombinedBackend, observation


def service(backend=None):
    from apxinf import Dm05Policy
    from apxinf.serving import Dm05HttpService

    return Dm05HttpService(Dm05Policy(backend or FakeBackend()))


def request_body():
    value = observation()
    sampling = value.pop("sampling")
    return json.dumps({"observation": value, "sampling": sampling}).encode()


def test_dm05_http_health_and_infer():
    api = service()
    status, health = api.handle("GET", "/health")
    assert status == 200
    assert health["schema"] == "apxinf.dm05.libero.http.v1"
    assert health["policy"]["execution_backend"] == "default"

    status, response = api.handle("POST", "/v1/infer", request_body())
    assert status == 200
    assert len(response["actions"]) == 10
    assert all(len(row) == 7 for row in response["actions"])
    assert response["metadata"]["schema"] == "apxinf.dm05.libero.response.v1"
    assert "path_proof" not in response["metadata"]


def test_dm05_combined_http_exposes_outer_contract_and_independent_proofs():
    api = service(FakeCombinedBackend())
    _, health_before = api.handle("GET", "/health")
    policy = health_before["policy"]
    assert policy["execution_backend"] == "default_exact_combined"
    assert policy["runtime_selector"] == "default_exact_combined"
    assert policy["host_thread_policy"] == "fixed_intraop_2"
    assert policy["torch_intraop_threads"] == 2
    assert policy["process_inference_policy"] == "serialized_all_dm05"
    assert policy["path_proof"]["initialized"] is False

    _, first_response = api.handle("POST", "/v1/infer", request_body())
    first = first_response["metadata"]["path_proof"]
    assert first["prefix_graph_replay_count"] == 1
    assert first_response["metadata"]["execution_backend"] == "default_exact_combined"
    assert first_response["metadata"]["runtime_selector"] == "default_exact_combined"
    assert first_response["metadata"]["host_thread_policy"] == "fixed_intraop_2"
    assert first_response["metadata"]["torch_intraop_threads"] == 2
    assert (
        first_response["metadata"]["process_inference_policy"]
        == "serialized_all_dm05"
    )
    assert first_response["metadata"]["precision"] == "bf16"
    assert first_response["metadata"]["llm_attention"] == "eager"
    assert first_response["metadata"]["vision_attention"] == "sdpa"
    assert first_response["metadata"]["action_attention"] == "sdpa"

    _, second_response = api.handle("POST", "/v1/infer", request_body())
    second = second_response["metadata"]["path_proof"]
    assert second["prefix_graph_replay_count"] == 2
    assert first["prefix_graph_replay_count"] == 1

    _, health_after = api.handle("GET", "/health")
    assert (
        health_after["policy"]["path_proof"]["prefix_graph_replay_count"]
        == 2
    )
    for proof in (first, second, health_after["policy"]["path_proof"]):
        assert proof["execution_backend"] == "default_exact_combined"
        assert proof["fallback_count"] == 0
        assert proof["startup_native_reference_bitwise"] is True
        assert proof["mask_static_address_verified"] is True
        assert proof["pack_workspace_addresses_stable"] is True
        assert not any("ptr" in key for key in proof)


def test_dm05_http_rejects_wrong_wire_contract():
    api = service()
    assert api.handle("GET", "/v1/infer")[0] == 405
    assert api.handle("POST", "/unknown", b"{}")[0] == 404
    assert api.handle("POST", "/v1/infer", b"not json")[0] == 400
    assert (
        api.handle(
            "POST", "/v1/infer", b"{}", content_type="application/octet-stream"
        )[0]
        == 415
    )

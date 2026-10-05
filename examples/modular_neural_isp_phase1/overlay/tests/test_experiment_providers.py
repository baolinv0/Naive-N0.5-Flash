"""Local HTTP, process, and file fixtures for model boundary contracts."""
import base64
import io
import json
import multiprocessing
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image


def limits(**changes):
    return {"max_calls": 20, "max_input_tokens_per_call": 20000,
            "max_output_tokens_per_call": 1000, "max_total_tokens": 100000,
            "input_usd_per_million": 1.0, "output_usd_per_million": 2.0,
            "max_cost_usd": 1.0, **changes}


@contextmanager
def server(events=None, *, status=200, body=None, content_type="text/event-stream", drip_delay=0):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append({"path": self.path, "headers": dict(self.headers),
                             "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])) )})
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            if body is not None:
                if drip_delay:
                    for byte in body:
                        try:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                        time.sleep(drip_delay)
                else:
                    self.wfile.write(body)
            else:
                for event in events:
                    content = event if isinstance(event, str) else json.dumps(event)
                    self.wfile.write(("data: " + content + "\n\n").encode())
            self.wfile.flush()

        def log_message(self, *args):
            pass

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/v1" % instance.server_port, received
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join()


def success_events(raw='{"proposal":{"learning_rate":0.0001},"decision":{}}', usage=True):
    events = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": raw[:10]},
                             "finish_reason": None}]},
              {"choices": [{"index": 0, "delta": {"content": raw[10:]}, "finish_reason": "stop"}]}]
    if usage:
        events.append({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                       "total_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 12}}})
    return events + ["[DONE]"]


def client(tmp_path, endpoint, ledger_limits=None, **config):
    from tm_research.experiment_costs import CostLedger
    from tm_research.experiment_provider import ChatClient
    ledger = CostLedger(tmp_path / "ledger.json", ledger_limits or limits())
    config = {"base_url": endpoint, "model": "local-naive", "timeout_seconds": 2,
              "log_dir": str(tmp_path / "logs"), "input_token_reservation": 1000, **config}
    return ChatClient(config, ledger), ledger


def test_streaming_keeps_visible_json_and_counts_reasoning_once(tmp_path, monkeypatch):
    raw = ' { "proposal": {"x": 1}, "decision": {} } '
    monkeypatch.setenv("EXPERIMENT_TEST_API_KEY", "secret-provider-test-token")
    with server(success_events(raw)) as (endpoint, received):
        provider, ledger = client(tmp_path, endpoint, api_key_env="EXPERIMENT_TEST_API_KEY")
        result = provider.complete([{"role": "user", "content": "feedback"}], request_id="r1",
                                   role="naive_proposer", max_output_tokens=100)
    assert result["raw_text"] == raw
    assert result["payload"] == {"proposal": {"x": 1}, "decision": {}}
    assert result["model"] == "local-naive"
    assert result["role"] == "naive_proposer"
    assert received[0]["path"] == "/v1/chat/completions"
    assert received[0]["body"]["stream"] is True
    assert received[0]["body"]["max_tokens"] == 100
    assert received[0]["headers"]["Authorization"] == "Bearer secret-provider-test-token"
    summary = ledger.summary()
    assert summary["actual_tokens"] == 30
    assert summary["charged_tokens"] == 30
    assert summary["actual_cost_usd"] == pytest.approx(0.00005)
    assert summary["requests"][0]["reasoning_tokens"] == 12
    assert "secret-provider-test-token" not in "".join(p.read_text() for p in (tmp_path / "logs").glob("*.json"))


@pytest.mark.parametrize("events", [
    success_events()[:-1],
    [{"choices": [{"index": 0, "delta": {"content": "{}"}, "finish_reason": "length"}]}, "[DONE]"],
    [{"choices": [{"index": 0, "delta": {"content": "{}"}, "finish_reason": None}]}, "[DONE]"],
    success_events("```json\n{}\n```"),
    success_events('{"score": NaN}'),
    success_events('{"score": 1e309}'),
    success_events('{"x":1,"x":2}'),
])
def test_partial_or_invalid_response_records_failure_and_preserves_budget(tmp_path, events):
    from tm_research.experiment_provider import ProviderError
    with server(events) as (endpoint, received):
        provider, ledger = client(tmp_path, endpoint)
        with pytest.raises(ProviderError):
            provider.complete([{"role": "user", "content": "f"}], request_id="bad", role="naive",
                              max_output_tokens=100)
    assert len(received) == 1
    record = ledger.summary()["requests"][0]
    assert record["error"]
    assert (tmp_path / "logs" / "bad.response.json").exists()
    # Valid provider usage is retained even if visible JSON violates the contract.
    assert record["charged_tokens"] in (30, 1100)


def test_http_failure_unknown_usage_charges_reservation_and_no_retry(tmp_path):
    from tm_research.experiment_provider import ProviderError
    with server(status=503, body=b"unavailable") as (endpoint, received):
        provider, ledger = client(tmp_path, endpoint)
        with pytest.raises(ProviderError):
            provider.complete([{"role": "user", "content": "f"}], request_id="failed", role="naive",
                              max_output_tokens=100)
        with pytest.raises((ProviderError, ValueError)):
            provider.complete([{"role": "user", "content": "f"}], request_id="failed", role="naive",
                              max_output_tokens=100)
    assert len(received) == 1
    summary = ledger.summary()
    assert summary["actual_tokens"] is None
    assert summary["actual_cost_usd"] is None
    assert summary["charged_tokens"] == 1100
    assert summary["charged_cost_usd"] == pytest.approx(0.0012)
    assert "unavailable" in (tmp_path / "logs" / "failed.response.json").read_text()


def test_missing_usage_is_null_with_conservative_booking(tmp_path):
    with server(success_events(usage=False)) as (endpoint, _):
        provider, ledger = client(tmp_path, endpoint)
        result = provider.complete([{"role": "user", "content": "f"}], request_id="unknown",
                                   role="naive", max_output_tokens=100)
    assert result["usage"] is None
    assert ledger.summary()["actual_tokens"] is None
    assert ledger.summary()["charged_tokens"] == 1100


def test_quota_is_checked_before_network(tmp_path):
    from tm_research.experiment_costs import LedgerQuotaError
    with server(success_events()) as (endpoint, received):
        provider, ledger = client(tmp_path, endpoint, limits(max_total_tokens=100))
        with pytest.raises(LedgerQuotaError):
            provider.complete([{"role": "user", "content": "f"}], request_id="denied", role="naive",
                              max_output_tokens=100)
    assert received == []
    assert ledger.summary()["calls"] == 0


def test_ledger_separate_input_output_limits_and_unknown_cost(tmp_path):
    from tm_research.experiment_costs import CostLedger, LedgerQuotaError
    quotas = limits(max_total_input_tokens=15, max_total_output_tokens=8,
                    input_usd_per_million=None, output_usd_per_million=None, max_cost_usd=None)
    ledger = CostLedger(tmp_path / "ledger.json", quotas)
    ledger.reserve("one", 10, 6, "naive")
    ledger.settle("one", None, error="timeout")
    with pytest.raises(LedgerQuotaError):
        ledger.reserve("input", 6, 1, "naive")
    with pytest.raises(LedgerQuotaError):
        ledger.reserve("output", 1, 3, "naive")
    summary = ledger.summary()
    assert summary["charged_input_tokens"] == 10
    assert summary["charged_output_tokens"] == 6
    assert summary["charged_cost_usd"] is None
    assert summary["actual_cost_usd"] is None


def reserve_in_process(path, quotas, request_id, queue):
    from tm_research.experiment_costs import CostLedger, LedgerQuotaError
    try:
        CostLedger(path, quotas).reserve(request_id, 10, 10, "naive")
        queue.put(True)
    except LedgerQuotaError:
        queue.put(False)


def test_concurrent_processes_cannot_overreserve(tmp_path):
    from tm_research.experiment_costs import CostLedger
    quotas = limits(max_calls=3)
    path = str(tmp_path / "ledger.json")
    CostLedger(path, quotas)
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [context.Process(target=reserve_in_process, args=(path, quotas, "r" + str(i), queue))
                 for i in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert sum(queue.get(timeout=2) for _ in processes) == 3
    assert CostLedger(path, quotas).summary()["charged_tokens"] == 60


@pytest.mark.parametrize("changes", [{"max_calls": 0}, {"max_calls": True},
    {"max_total_tokens": float("inf")}, {"max_cost_usd": -1},
    {"input_usd_per_million": float("nan")}, {"max_cost_usd": 1, "output_usd_per_million": None}])
def test_ledger_rejects_nonfinite_or_unbookable_limits(tmp_path, changes):
    from tm_research.experiment_costs import CostLedger
    with pytest.raises(ValueError):
        CostLedger(tmp_path / "ledger.json", limits(**changes))


def test_usage_overrun_is_recorded_not_erased(tmp_path):
    from tm_research.experiment_costs import CostLedger, LedgerQuotaError
    ledger = CostLedger(tmp_path / "ledger.json", limits(max_total_tokens=30))
    ledger.reserve("r", 10, 10, "naive")
    record = ledger.settle("r", {"prompt_tokens": 15, "completion_tokens": 25})
    assert record["charged_tokens"] == 40
    assert record["reservation_overrun"] is True
    with pytest.raises(LedgerQuotaError):
        ledger.reserve("next", 1, 1, "naive")


def test_token_booking_is_explicitly_not_measured_usage(tmp_path):
    with server(success_events(usage=False)) as (endpoint, _):
        provider, ledger = client(tmp_path, endpoint, input_token_reservation=None)
        provider.complete([{"role": "user", "content": "科研反馈"}], request_id="bytes", role="naive",
                          max_output_tokens=100)
    record = ledger.summary()["requests"][0]
    assert record["input_booking_basis"] == "utf8_bytes_plus_template_allowance"
    assert record["actual_input_tokens"] is None


def make_png(path):
    Image.new("RGB", (8, 8), (20, 30, 40)).save(path)
    return path


@pytest.fixture
def image_root(tmp_path_factory):
    return tmp_path_factory.mktemp("dev_assets")


def test_vision_sends_authorized_actual_image_bytes_and_auxiliary_contract(image_root):
    from tm_research.experiment_vision import build_vision_messages, validate_auxiliary_report
    root = image_root / "dev"
    root.mkdir()
    image = make_png(root / "scene.png")
    packet = {"feedback_packet_revision": "rev1", "latest_observation": {"run_id": "run1"}}
    messages = build_vision_messages(packet, [str(image)], str(root))
    content = messages[-1]["content"]
    image_block = next(item for item in content if item["type"] == "image_url")
    assert base64.b64decode(image_block["image_url"]["url"].split(",", 1)[1]) == image.read_bytes()
    assert "run1" in json.dumps(messages)
    report = validate_auxiliary_report({"observations": ["slight clipping"], "limitations": ["one image"]})
    assert report["authority"] == "auxiliary_only"


@pytest.mark.parametrize("kind", ["outside", "symlink", "confirm", "test", "not_image", "too_many"])
def test_vision_rejects_unauthorized_or_unbounded_assets(image_root, kind):
    from tm_research.experiment_vision import build_vision_messages
    root = image_root / "dev"
    root.mkdir()
    image = make_png(root / "scene.png")
    paths = [str(image)]
    if kind == "outside":
        paths = [str(make_png(image_root / "outside.png"))]
    elif kind == "symlink":
        image.unlink()
        image.symlink_to(make_png(image_root / "outside.png"))
    elif kind in ("confirm", "test"):
        forbidden = root / kind.upper()
        forbidden.mkdir()
        paths = [str(make_png(forbidden / "scene.png"))]
    elif kind == "not_image":
        image.write_text("not PNG")
    else:
        paths *= 4
    with pytest.raises(ValueError):
        build_vision_messages({}, paths, str(root))


@pytest.mark.parametrize("report", [{"observations": [], "limitations": [], "psnr": 99},
    {"observations": [], "limitations": [], "decision": "accept"},
    {"observations": "looks good", "limitations": []}])
def test_auxiliary_report_cannot_claim_score_or_selection_authority(report):
    from tm_research.experiment_vision import validate_auxiliary_report
    with pytest.raises(ValueError):
        validate_auxiliary_report(report)


def test_existing_evidence_cannot_be_overwritten_by_new_ledger(tmp_path):
    from tm_research.experiment_provider import ProviderError
    logs = tmp_path / "logs"
    logs.mkdir()
    existing = logs / "existing.response.json"
    existing.write_text('{"original":"evidence"}')
    with server(success_events()) as (endpoint, received):
        provider, ledger = client(tmp_path, endpoint)
        with pytest.raises(ProviderError):
            provider.complete([{"role": "user", "content": "f"}], request_id="existing", role="naive",
                              max_output_tokens=100)
    assert existing.read_text() == '{"original":"evidence"}'
    assert received == []


def test_usage_overrun_records_provider_failure_with_actual_billing(tmp_path):
    from tm_research.experiment_provider import ProviderError
    with server(success_events()) as (endpoint, _):
        provider, ledger = client(tmp_path, endpoint)
        with pytest.raises(ProviderError, match="reservation"):
            provider.complete([{"role": "user", "content": "f"}], request_id="overrun", role="naive",
                              max_output_tokens=10)
    record = ledger.summary()["requests"][0]
    assert record["actual_tokens"] == 30
    assert record["reservation_overrun"] is True
    assert record["error"]


def test_credentials_echoed_by_endpoint_are_redacted_only_in_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_TEST_API_KEY", "secret-for-echo")
    raw = '{"unexpected":"secret-for-echo"}'
    with server(success_events(raw)) as (endpoint, _):
        provider, ledger = client(tmp_path, endpoint, api_key_env="EXPERIMENT_TEST_API_KEY")
        result = provider.complete([{"role": "user", "content": "f"}], request_id="echo", role="qwen",
                                   max_output_tokens=100)
    assert result["raw_text"] == raw
    assert "secret-for-echo" not in "".join(p.read_text() for p in (tmp_path / "logs").glob("*.json"))


@pytest.mark.parametrize("name", ["CONFIRM_holdout", "TEST_final"])
def test_named_confirmation_or_test_roots_are_forbidden(image_root, name):
    from tm_research.experiment_vision import build_vision_messages
    root = image_root / name
    root.mkdir()
    image = make_png(root / "scene.png")
    with pytest.raises(ValueError):
        build_vision_messages({}, [str(image)], str(root))


def test_oversized_image_rejected_before_decode(image_root):
    from tm_research.experiment_vision import build_vision_messages, MAX_IMAGE_BYTES
    root = image_root / "dev"
    root.mkdir()
    image = root / "big.png"
    image.write_bytes(b"x" * (MAX_IMAGE_BYTES + 1))
    with pytest.raises(ValueError, match="byte"):
        build_vision_messages({}, [str(image)], str(root))


def test_stream_response_byte_limit_preserves_partial_raw_and_failure(tmp_path):
    from tm_research.experiment_provider import ProviderError
    with server(success_events('{"text":"' + "x" * 1000 + '"}')) as (endpoint, _):
        provider, ledger = client(tmp_path, endpoint, max_response_bytes=128)
        with pytest.raises(ProviderError, match="byte limit"):
            provider.complete([{"role": "user", "content": "f"}], request_id="big", role="naive",
                              max_output_tokens=100)
    assert ledger.summary()["actual_tokens"] is None
    assert json.loads((tmp_path / "logs" / "big.response.json").read_text())["raw_stream"]


def test_text_cannot_use_a_reservation_smaller_than_conservative_byte_booking(tmp_path):
    with server(success_events(usage=False)) as (endpoint, received):
        provider, ledger = client(tmp_path, endpoint, input_token_reservation=1)
        provider.complete([{"role": "user", "content": "x" * 3000}], request_id="undersized", role="naive",
                          max_output_tokens=100)
    record = ledger.summary()["requests"][0]
    assert record["charged_input_tokens"] >= len(json.dumps(received[0]["body"]).encode()) + 512
    assert record["input_booking_basis"] == "configured_and_utf8_conservative_max"
    assert record["actual_tokens"] is None


def test_non_sse_response_preserves_raw_failure_body(tmp_path):
    from tm_research.experiment_provider import ProviderError
    with server(body=b'{"error":"unsupported streaming"}', content_type="application/json") as (endpoint, _):
        provider, ledger = client(tmp_path, endpoint)
        with pytest.raises(ProviderError, match="streaming"):
            provider.complete([{"role": "user", "content": "f"}], request_id="nonsse", role="naive",
                              max_output_tokens=100)
    response = json.loads((tmp_path / "logs" / "nonsse.response.json").read_text())
    assert response["raw_stream"] == '{"error":"unsupported streaming"}'


def test_credentials_in_extra_usage_fields_are_not_written_to_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_TEST_API_KEY", "secret-in-usage")
    events = success_events()
    events[-2]["usage"]["unexpected_echo"] = "secret-in-usage"
    with server(events) as (endpoint, _):
        provider, ledger = client(tmp_path, endpoint, api_key_env="EXPERIMENT_TEST_API_KEY")
        provider.complete([{"role": "user", "content": "f"}], request_id="usage", role="naive",
                          max_output_tokens=100)
    assert "secret-in-usage" not in (tmp_path / "ledger.json").read_text()
    assert ledger.summary()["actual_tokens"] == 30


def test_numeric_credential_does_not_corrupt_numeric_usage_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_TEST_API_KEY", "10")
    with server(success_events()) as (endpoint, _):
        provider, ledger = client(tmp_path, endpoint, api_key_env="EXPERIMENT_TEST_API_KEY")
        provider.complete([{"role": "user", "content": "f"}], request_id="numeric", role="naive",
                          max_output_tokens=100)
    assert ledger.summary()["actual_input_tokens"] == 10


@pytest.mark.parametrize("phase", ["CONFIRM_holdout", "test_final"])
def test_holdout_ancestor_of_allowed_run_is_rejected(image_root, phase):
    from tm_research.experiment_vision import build_vision_messages
    root = image_root / phase / "runs" / "run001"
    root.mkdir(parents=True)
    image = make_png(root / "scene.png")
    with pytest.raises(ValueError):
        build_vision_messages({}, [str(image)], str(root))


def test_truncated_jpeg_is_rejected_before_paid_model_call(image_root):
    from tm_research.experiment_vision import build_vision_messages
    root = image_root / "dev"
    root.mkdir()
    image = root / "scene.jpg"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(image)
    image.write_bytes(image.read_bytes()[:-2])
    with pytest.raises(ValueError):
        build_vision_messages({}, [str(image)], str(root))


def test_drip_fed_line_cannot_extend_total_body_walltime(tmp_path):
    from tm_research.experiment_provider import ProviderError
    with server(body=b"data: " + b"x" * 100, drip_delay=0.01) as (endpoint, _):
        provider, ledger = client(tmp_path, endpoint, timeout_seconds=0.15)
        started = time.monotonic()
        with pytest.raises(ProviderError):
            provider.complete([{"role": "user", "content": "f"}], request_id="drip", role="naive",
                              max_output_tokens=100)
        duration = time.monotonic() - started
    assert duration < 0.6
    assert ledger.summary()["actual_tokens"] is None


def test_json_escaped_credentials_are_redacted_in_wire_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_TEST_API_KEY", "secret-for-echo")
    raw = '{"unexpected":"\\u0073ecret-for-echo"}'
    with server(success_events(raw)) as (endpoint, _):
        provider, ledger = client(tmp_path, endpoint, api_key_env="EXPERIMENT_TEST_API_KEY")
        result = provider.complete([{"role": "user", "content": "f"}], request_id="escaped", role="naive",
                                   max_output_tokens=100)
    assert result["payload"] == {"unexpected": "secret-for-echo"}
    response = json.loads((tmp_path / "logs" / "escaped.response.json").read_text())
    assert "secret-for-echo" not in response["raw_text"]
    assert "\\u0073ecret-for-echo" not in response["raw_text"]


def test_scope_ceiling_rejects_before_charging_global_budget(tmp_path):
    from tm_research.experiment_costs import CostLedger, LedgerQuotaError
    path = tmp_path / "ledger.json"
    global_limits = limits(max_calls=10)
    scope_limits = limits(max_calls=1)
    first = CostLedger(path, global_limits, scope_id="campaign-a", scope_limits=scope_limits)
    second = CostLedger(path, global_limits, scope_id="campaign-b", scope_limits=scope_limits)
    first.reserve("a1", 10, 10, "naive")
    before = path.read_text()
    with pytest.raises(LedgerQuotaError, match="scope"):
        first.reserve("a2", 10, 10, "qwen")
    assert path.read_text() == before
    second.reserve("b1", 10, 10, "naive")
    assert first.summary()["calls"] == 2
    assert first.summary(scope_id="campaign-a")["calls"] == 1
    assert first.summary()["requests"][0]["scope_id"] == "campaign-a"


@pytest.mark.parametrize("dimension,new_input,new_output", [
    ("input", 6, 1), ("output", 1, 3), ("total", 4, 4)])
def test_scope_separate_and_combined_token_limits(tmp_path, dimension, new_input, new_output):
    from tm_research.experiment_costs import CostLedger, LedgerQuotaError
    scope_limits = limits(max_total_input_tokens=15, max_total_output_tokens=8, max_total_tokens=23)
    if dimension == "total":
        scope_limits.update(max_total_input_tokens=100, max_total_output_tokens=100)
    ledger = CostLedger(tmp_path / "ledger.json", limits(), scope_id="campaign-a", scope_limits=scope_limits)
    ledger.reserve("first", 10, 6, "naive")
    ledger.settle("first", None, error="usage unavailable")
    with pytest.raises(LedgerQuotaError):
        ledger.reserve("next", new_input, new_output, "qwen")
    summary = ledger.summary(scope_id="campaign-a")
    assert summary["actual_tokens"] is None
    assert summary["charged_input_tokens"] == 10
    assert summary["charged_output_tokens"] == 6
    assert summary["charged_tokens"] == 16


def test_scope_known_usage_releases_only_its_own_reservation(tmp_path):
    from tm_research.experiment_costs import CostLedger
    path = tmp_path / "ledger.json"
    quota = limits(max_total_tokens=30)
    a = CostLedger(path, limits(), scope_id="a", scope_limits=quota)
    b = CostLedger(path, limits(), scope_id="b", scope_limits=quota)
    a.reserve("a1", 10, 10, "naive")
    b.reserve("b1", 10, 10, "qwen")
    a.settle("a1", {"prompt_tokens": 1, "completion_tokens": 1})
    a.reserve("a2", 10, 10, "naive")
    assert a.summary(scope_id="a")["charged_tokens"] == 22
    assert a.summary(scope_id="b")["charged_tokens"] == 20
    assert a.summary()["charged_tokens"] == 42


def test_scope_identity_and_limits_frozen_across_clients(tmp_path):
    from tm_research.experiment_costs import CostLedger
    path = tmp_path / "ledger.json"
    a = CostLedger(path, limits(), scope_id="a", scope_limits=limits(max_calls=1))
    a.reserve("r", 1, 1, "naive")
    with pytest.raises(ValueError, match="scope"):
        CostLedger(path, limits(), scope_id="a", scope_limits=limits(max_calls=2))
    b = CostLedger(path, limits(), scope_id="b", scope_limits=limits(max_calls=2))
    with pytest.raises(ValueError, match="scope"):
        b.settle("r", None)
    with pytest.raises(ValueError):
        a.settle("r", None, scope_id="b")
    assert a.summary()["requests"][0]["scope_id"] == "a"


def test_scope_monetary_cap_uses_same_global_accounting_rates(tmp_path):
    from tm_research.experiment_costs import CostLedger, LedgerQuotaError
    scoped = limits(max_cost_usd=0.00002)
    scoped.pop("input_usd_per_million")
    scoped.pop("output_usd_per_million")
    ledger = CostLedger(tmp_path / "ledger.json", limits(), scope_id="a", scope_limits=scoped)
    ledger.reserve("first", 5, 5, "naive")
    with pytest.raises(LedgerQuotaError):
        ledger.reserve("second", 5, 5, "qwen")
    with pytest.raises(ValueError):
        CostLedger(tmp_path / "ledger.json", limits(), scope_id="b",
                   scope_limits=limits(input_usd_per_million=10))


@pytest.mark.parametrize("scope_id,scope_limits", [
    (None, limits()), ("a", None), ("", limits()), ("../escape", limits()), ("a" * 129, limits())])
def test_scope_requires_bounded_identity_and_limits(tmp_path, scope_id, scope_limits):
    from tm_research.experiment_costs import CostLedger
    with pytest.raises(ValueError):
        CostLedger(tmp_path / "ledger.json", limits(), scope_id=scope_id, scope_limits=scope_limits)


def scoped_reserve_process(path, global_limits, scope_limits, request_id, scope_id, queue):
    from tm_research.experiment_costs import CostLedger, LedgerQuotaError
    try:
        CostLedger(path, global_limits, scope_id=scope_id, scope_limits=scope_limits).reserve(
            request_id, 10, 10, "naive")
        queue.put(True)
    except LedgerQuotaError:
        queue.put(False)


def test_concurrent_scope_and_global_reservations_use_one_atomic_gate(tmp_path):
    from tm_research.experiment_costs import CostLedger
    global_limits, scope_limits = limits(max_calls=5), limits(max_calls=3)
    path = str(tmp_path / "ledger.json")
    CostLedger(path, global_limits)
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [context.Process(target=scoped_reserve_process,
        args=(path, global_limits, scope_limits, "r" + str(i), "a" if i % 2 else "b", queue))
        for i in range(12)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    ledger = CostLedger(path, global_limits)
    assert sum(queue.get(timeout=2) for _ in processes) == 5
    assert ledger.summary()["calls"] == 5
    assert ledger.summary(scope_id="a")["calls"] <= 3
    assert ledger.summary(scope_id="b")["calls"] <= 3


def test_legacy_unscoped_state_can_add_a_scope_without_reassigning_requests(tmp_path):
    from tm_research.experiment_costs import CostLedger
    path = tmp_path / "ledger.json"
    old = CostLedger(path, limits())
    old.reserve("legacy", 1, 1, "naive")
    state = json.loads(path.read_text())
    state.pop("scoped_limits", None)
    state["requests"]["legacy"].pop("scope_id", None)
    path.write_text(json.dumps(state))
    scoped = CostLedger(path, limits(), scope_id="new", scope_limits=limits())
    scoped.reserve("new", 10, 10, "naive")
    assert scoped.summary()["calls"] == 2
    assert scoped.summary(scope_id="new")["calls"] == 1
    assert "scope_id" not in scoped.summary()["requests"][0]

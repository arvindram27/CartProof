"""Offline tests for harness/shopify_probe.py (P42). Fake only; no network."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from harness import shopify_probe
from harness.shopify_probe import (LiveShopifyBackend, _guard_text,
    _write_json, main, run_probe)

try:  # tests/ has no __init__.py in this repo; both roots work
    from tests.shopify_fake import FAKE_CART_ID
    from tests.shopify_live_fake import FakeLiveShopify
except ImportError:  # pragma: no cover
    from shopify_fake import FAKE_CART_ID
    from shopify_live_fake import FakeLiveShopify

SHOP = shopify_probe.EXPECTED_SHOP
GID1 = "gid://shopify/ProductVariant/1"
GID2 = "gid://shopify/ProductVariant/2"
GID3 = "gid://shopify/ProductVariant/3"
CART_REF = hashlib.sha256(FAKE_CART_ID.encode("utf-8")).hexdigest()[:12]
FAKE_KEY = "fakekeycapability000000"


def _args(stage, tmp_path, allow=True, max_requests=30):
    argv = ["--stage", stage, "--shop", SHOP,
            "--mapping", str(tmp_path / "mapping.json"),
            "--log-dir", str(tmp_path / "logs"),
            "--max-requests", str(max_requests)]
    if allow:
        argv.append("--allow-live")
    return argv


def _discover(tmp_path, fake):
    return main(_args("discover", tmp_path), transport=fake.transport)


def _probe(tmp_path, fake):
    return main(_args("probe", tmp_path), transport=fake.transport)


def _read(log_dir, name):
    return json.loads((log_dir / name).read_text(encoding="utf-8"))


def _steps(log_dir):
    return [json.loads(line) for line in
            (log_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]


# ---- gating ---------------------------------------------------- #

def test_cli_refuses_without_allow_live(tmp_path, capsys):
    rc = main(_args("discover", tmp_path, allow=False))
    assert rc == 3
    assert "live_not_approved" in capsys.readouterr().err
    assert not (tmp_path / "logs").exists()


def test_backend_requires_allow_live():
    with pytest.raises(TypeError):
        LiveShopifyBackend(SHOP, None)


def test_backend_bound_to_dev_shop():
    with pytest.raises(ValueError):
        LiveShopifyBackend("evil.example.com", None, allow_live=True)


def test_backend_rejects_non_mock_injected_transport():
    with pytest.raises(TypeError):
        LiveShopifyBackend(SHOP, None, allow_live=True,
                           transport=httpx.HTTPTransport())


def test_probe_requires_budget_headroom(tmp_path, capsys):
    (tmp_path / "mapping.json").write_text(
        json.dumps({"variants": {"A": GID1, "B": GID2}}))
    rc = main(_args("probe", tmp_path, max_requests=10))
    assert rc == 4
    assert "request_budget_too_small" in capsys.readouterr().err


# ---- discover -------------------------------------------------- #

def test_discover_writes_mapping_read_only(tmp_path):
    fake = FakeLiveShopify()
    assert _discover(tmp_path, fake) == 0
    mapping = json.loads((tmp_path / "mapping.json").read_text())["variants"]
    assert mapping["TEST-SKU-1"] == GID1
    assert mapping["TEST-SKU-2"] == GID2
    assert mapping["VARIANT:3"] == GID3  # blank sku: full-tail fallback
    assert len(fake.requests) == 1
    assert "products(" in fake.requests[0]["query"]
    summary = _read(tmp_path / "logs", "discover-summary.json")
    assert summary["status"] == "ok"
    assert summary["catalog_pagination_risk"] is False


def test_discover_duplicate_sku_keys_unique(tmp_path):
    products = [{"id": "gid://shopify/Product/9", "title": "Dup",
                 "variants": [{"id": GID1, "sku": "DUP"},
                              {"id": GID2, "sku": "DUP"},
                              {"id": GID3, "sku": "DUP"}]}]
    fake = FakeLiveShopify(products=products)
    assert _discover(tmp_path, fake) == 0
    mapping = json.loads((tmp_path / "mapping.json").read_text())["variants"]
    assert len(mapping) == 3
    assert len(set(mapping.keys())) == 3
    assert len(set(mapping.values())) == 3
    assert mapping["DUP"] == GID1  # first occurrence keeps the sku key


def test_discover_pagination_risk_recorded(tmp_path):
    products = [{"id": f"gid://shopify/Product/{i}", "title": f"P{i}",
                 "variants": [{"id": f"gid://shopify/ProductVariant/{i}",
                               "sku": f"S{i}"}]}
                for i in range(1, 21)]
    fake = FakeLiveShopify(products=products)
    assert _discover(tmp_path, fake) == 0
    summary = _read(tmp_path / "logs", "discover-summary.json")
    assert summary["catalog_pagination_risk"] is True


def test_discover_http_500_stops_no_retry(tmp_path):
    fake = FakeLiveShopify(mode="http_500")
    assert _discover(tmp_path, fake) == 4
    summary = _read(tmp_path / "logs", "discover-summary.json")
    assert summary["status"] == "failed"
    assert summary["category"] == "http:500"
    assert len(fake.requests) == 1
    assert not (tmp_path / "mapping.json").exists()


# ---- probe ------------------------------------------------------ #

def test_probe_happy_path_merged(tmp_path):
    fake = FakeLiveShopify()
    assert _discover(tmp_path, fake) == 0
    assert _probe(tmp_path, fake) == 0
    final = _read(tmp_path / "logs", "final.json")
    assert final["overall_verdict"] == "COMPLETE"
    assert final["cart_verified_empty"] is True
    assert final["merge_behavior"] == "merged_single_line"
    assert final["requests_used"] == 14
    assert fake.lines == {}


def test_probe_no_merge_dynamic_expectation(tmp_path):
    fake = FakeLiveShopify(merge=False)
    assert _discover(tmp_path, fake) == 0
    assert _probe(tmp_path, fake) == 0
    final = _read(tmp_path / "logs", "final.json")
    assert final["overall_verdict"] == "COMPLETE"
    assert final["merge_behavior"] == "split_lines"
    assert final["requests_used"] == 15
    steps = {s["name"]: s for s in _steps(tmp_path / "logs")
             if s.get("event") == "step"}
    # first A line held qty 2; setting it to 1 leaves the second A line:
    # aggregate A = 2, never a false PARTIAL
    assert steps["set_first_A_line_to_1"]["expected"] == {
        "TEST-SKU-1": 2, "TEST-SKU-2": 1}
    pin = [s for s in _steps(tmp_path / "logs")
           if s.get("event") == "merge_pin"][0]
    assert pin["raw_line_count"] == 2
    assert pin["aggregate_quantity"] == 3


def test_cleanup_is_last_mutation_then_read(tmp_path):
    fake = FakeLiveShopify()
    assert _discover(tmp_path, fake) == 0
    assert _probe(tmp_path, fake) == 0
    assert "cartLinesRemove" in fake.requests[-2]["query"]
    assert "cart(" in fake.requests[-1]["query"]


def test_probe_deterministic_plan(tmp_path):
    plans = []
    for i in range(2):
        d = tmp_path / f"run{i}"
        d.mkdir()
        fake = FakeLiveShopify()
        assert main(_args("discover", d), transport=fake.transport) == 0
        assert main(_args("probe", d), transport=fake.transport) == 0
        plan = _read(d / "logs", "plan.json")
        plan.pop("frozen_at")
        plans.append(plan)
    assert plans[0] == plans[1]


def test_request_cap_stops_and_records(tmp_path):
    fake = FakeLiveShopify()
    assert _discover(tmp_path, fake) == 0
    backend = LiveShopifyBackend(
        SHOP, str(tmp_path / "mapping.json"), allow_live=True,
        transport=fake.transport, max_requests=5)
    rc = run_probe(backend, tmp_path / "logs")
    assert rc == 5
    final = _read(tmp_path / "logs", "final.json")
    assert final["overall_verdict"] == "UNKNOWN"
    assert "request_cap_exceeded" in final["stop_reason"]
    assert final["cart_verified_empty"] is False
    assert final["cart_not_cleaned"]["cart_ref"] == CART_REF
    assert len(fake.requests) == 1 + 5  # 1 discover + cap at 5, no retry


def test_probe_http_500_stops_no_retry(tmp_path):
    fake = FakeLiveShopify(mode="http_500")
    (tmp_path / "mapping.json").write_text(
        json.dumps({"variants": {"A": GID1, "B": GID2}}))
    backend = LiveShopifyBackend(
        SHOP, str(tmp_path / "mapping.json"), allow_live=True,
        transport=fake.transport)
    rc = run_probe(backend, tmp_path / "logs")
    assert rc == 5
    final = _read(tmp_path / "logs", "final.json")
    assert final["overall_verdict"] == "UNKNOWN"
    assert "acquisition_failed:baseline" in final["stop_reason"]
    assert len(fake.requests) == 1


def test_user_error_echo_stops_redacted(tmp_path):
    fake = FakeLiveShopify(mode="user_error_echo")
    assert _discover(tmp_path, fake) == 0
    rc = _probe(tmp_path, fake)
    assert rc == 5
    final = _read(tmp_path / "logs", "final.json")
    assert final["overall_verdict"] == "UNKNOWN"
    assert "mutation_failed:add_A_x2" in final["stop_reason"]
    for path in (tmp_path / "logs").iterdir():
        text = path.read_text(encoding="utf-8")
        assert FAKE_CART_ID not in text
        assert FAKE_KEY not in text


# ---- redaction --------------------------------------------------- #

def test_no_capability_in_any_evidence_file(tmp_path):
    fake = FakeLiveShopify()
    assert _discover(tmp_path, fake) == 0
    assert _probe(tmp_path, fake) == 0
    texts = {p.name: p.read_text(encoding="utf-8")
             for p in (tmp_path / "logs").iterdir()}
    assert texts  # evidence exists
    for text in texts.values():
        assert FAKE_CART_ID not in text
        assert FAKE_KEY not in text
    assert "cartref:" in texts["steps.jsonl"]
    assert CART_REF in texts["final.json"]


def test_mutation_500_no_further_requests(tmp_path):
    # baseline succeeds; the FIRST mutation fails with HTTP 500. No read,
    # no cleanup, no request of any kind afterwards, and the backend
    # latches fail-closed for later callers too.
    fake = FakeLiveShopify(mode="add_http_500")
    assert _discover(tmp_path, fake) == 0
    backend = LiveShopifyBackend(
        SHOP, str(tmp_path / "mapping.json"), allow_live=True,
        transport=fake.transport)
    before = len(fake.requests)
    rc = run_probe(backend, tmp_path / "logs")
    assert rc == 5
    after_probe = len(fake.requests)
    # 1 create + 1 baseline read + 1 failing add; then nothing
    assert after_probe - before == 3
    final = _read(tmp_path / "logs", "final.json")
    assert final["overall_verdict"] == "UNKNOWN"
    assert "mutation_failed:add_A_x2" in final["stop_reason"]
    assert final["cart_verified_empty"] is False
    assert final["cart_not_cleaned"]["cart_ref"] == CART_REF
    # sticky: even a direct later read is refused WITHOUT sending
    state = backend.acquire_state()
    assert state["lines"] is None
    assert len(fake.requests) == after_probe


def test_redirect_rejected_not_parsed(tmp_path):
    fake = FakeLiveShopify(mode="redirect")
    assert _discover(tmp_path, fake) == 4
    summary = _read(tmp_path / "logs", "discover-summary.json")
    assert summary["status"] == "failed"
    assert "redirect_or_status:302" in summary["category"]
    assert len(fake.requests) == 1
    assert not (tmp_path / "mapping.json").exists()


def test_invalid_gid_skipped_and_counted(tmp_path):
    products = [{"id": "gid://shopify/Product/1", "title": "T",
                 "variants": [
                     {"id": GID1, "sku": "A"},
                     {"id": GID2, "sku": "B"},
                     # capability-shaped junk must never enter the mapping
                     {"id": "gid://shopify/Cart/abc?key=evil", "sku": "C"}]}]
    fake = FakeLiveShopify(products=products)
    assert _discover(tmp_path, fake) == 0
    mapping = json.loads((tmp_path / "mapping.json").read_text())["variants"]
    assert mapping == {"A": GID1, "B": GID2}
    summary = _read(tmp_path / "logs", "discover-summary.json")
    assert summary["skipped_invalid_gid"] == 1


def test_probe_rejects_invalid_mapping(tmp_path, capsys):
    (tmp_path / "mapping.json").write_text(json.dumps({"variants": {
        "A": GID1, "B": GID1}}))  # duplicate gid values
    fake = FakeLiveShopify()
    rc = main(_args("probe", tmp_path), transport=fake.transport)
    assert rc == 4
    assert "mapping_invalid" in capsys.readouterr().err
    assert len(fake.requests) == 0


def test_api_version_pinned():
    with pytest.raises(ValueError):
        LiveShopifyBackend(SHOP, None, allow_live=True,
                           api_version="../../admin")
    with pytest.raises(ValueError):
        LiveShopifyBackend(SHOP, None, allow_live=True,
                           api_version="2025-01")


def test_no_complete_when_resolution_not_remote_cart(tmp_path, monkeypatch):
    fake = FakeLiveShopify()
    assert _discover(tmp_path, fake) == 0
    backend = LiveShopifyBackend(
        SHOP, str(tmp_path / "mapping.json"), allow_live=True,
        transport=fake.transport)
    monkeypatch.setattr(backend, "acquire_state", lambda: {
        "lines": {}, "line_ids": [], "version": None,
        "resolution": "mystery", "provenance": {}})
    rc = run_probe(backend, tmp_path / "logs")
    assert rc == 5
    final = _read(tmp_path / "logs", "final.json")
    assert final["overall_verdict"] == "UNKNOWN"
    assert "acquisition_failed:baseline" in final["stop_reason"]


def test_mutation_429_no_further_requests(tmp_path):
    fake = FakeLiveShopify(mode="add_http_429")
    assert _discover(tmp_path, fake) == 0
    backend = LiveShopifyBackend(
        SHOP, str(tmp_path / "mapping.json"), allow_live=True,
        transport=fake.transport)
    before = len(fake.requests)
    rc = run_probe(backend, tmp_path / "logs")
    assert rc == 5
    assert len(fake.requests) - before == 3  # create, read, failing add
    final = _read(tmp_path / "logs", "final.json")
    assert final["overall_verdict"] == "UNKNOWN"
    assert "mutation_failed:add_A_x2" in final["stop_reason"]
    assert final["cart_verified_empty"] is False


def test_capability_shaped_sku_refused_no_trace(tmp_path):
    bad = "gid://shopify/Cart/x?key=evil"
    products = [{"id": "gid://shopify/Product/1", "title": "T",
                 "variants": [{"id": GID1, "sku": "A"},
                              {"id": GID2, "sku": "B"},
                              {"id": GID3, "sku": bad}]}]
    fake = FakeLiveShopify(products=products)
    rc = _discover(tmp_path, fake)
    assert rc == 4
    assert not (tmp_path / "mapping.json").exists()
    summary = _read(tmp_path / "logs", "discover-summary.json")
    assert summary["status"] == "refused:capability_shaped_key"
    assert summary.get("catalog_raw_withheld") is True
    # no raw trace of the capability-shaped value anywhere on disk
    for path in (tmp_path / "logs").iterdir():
        text = path.read_text(encoding="utf-8")
        assert bad not in text
        assert "?key=" not in text


def test_evidence_guard_blocks_leak(tmp_path):
    fake = FakeLiveShopify()
    assert _discover(tmp_path, fake) == 0
    assert _probe(tmp_path, fake) == 0
    backend = LiveShopifyBackend(
        SHOP, str(tmp_path / "mapping.json"), allow_live=True,
        transport=fake.transport)
    backend._set_cart_id(FAKE_CART_ID)
    # the guard is the fail-closed backstop BEHIND redact(): test it raw
    with pytest.raises(RuntimeError):
        _guard_text('{"leak": "' + FAKE_CART_ID + '"}', backend)
    with pytest.raises(RuntimeError):
        _guard_text('{"leak": "' + FAKE_KEY + '"}', backend)
    # generic capability shapes are blocked even with nothing registered
    fresh = LiveShopifyBackend(SHOP, str(tmp_path / "mapping.json"),
                               allow_live=True, transport=fake.transport)
    with pytest.raises(RuntimeError):
        _guard_text('{"echo": "gid://shopify/Cart/AAAA?key=bbbb"}', fresh)
    with pytest.raises(RuntimeError):
        _guard_text('{"echo": "see ?key= here"}', fresh)
    # line IDs are NOT capabilities and must pass
    _guard_text('{"line_id": "gid://shopify/CartLine/1"}', fresh)
    # and redact() itself must neutralize the capability before writing
    out = _write_json(tmp_path / "logs", "redacted.json",
                      {"echo": FAKE_CART_ID}, backend)
    text = out.read_text(encoding="utf-8")
    assert FAKE_CART_ID not in text
    assert "cartref:" in text


# ---- error capture (P47) ---------------------------------------- #
# All offline: the fake's error_response seam returns a fixed >=400
# response to the products query. Every test asserts no extra request.

def _diag_summary(tmp_path):
    return _read(tmp_path / "logs", "discover-summary.json")


def test_discover_400_captures_allowlisted_fields(tmp_path):
    body = json.dumps({"errors": [
        {"message": "Field 'cart' doesn't exist on type 'Mutation'",
         "extensions": {"code": "undefinedField",
                        "documentation": "https://dropped.example"},
         "locations": [{"line": 1, "column": 2}]},
        {"message": "second problem", "code": "SOME_CODE",
         "path": ["cartCreate"]}]}).encode()
    resp = httpx.Response(400, headers={
        "x-shopify-request-id": "9f1b2c3d-1234-4abc-9def-0123456789ab",
        "set-cookie": "session=secretvalue; HttpOnly",
        "x-sorting-hat-podid": "123"}, content=body)
    fake = FakeLiveShopify(error_response=resp)
    assert _discover(tmp_path, fake) == 4
    summary = _diag_summary(tmp_path)
    assert summary["status"] == "failed"
    assert summary["category"] == "http:400"
    diag = summary["diagnostic"]
    assert diag["http_status"] == 400
    assert diag["request_id"] == "9f1b2c3d-1234-4abc-9def-0123456789ab"
    assert diag["errors"] == [
        {"message": "Field 'cart' doesn't exist on type 'Mutation'",
         "extensions_code": "undefinedField"},
        {"message": "second problem", "code": "SOME_CODE"}]
    assert diag["errors_truncated"] is False
    assert len(fake.requests) == 1  # capture made no extra request
    raw = (tmp_path / "logs" / "discover-summary.json").read_text(
        encoding="utf-8")
    assert "secretvalue" not in raw      # cookie header never persisted
    assert "x-sorting-hat" not in raw    # non-allowlisted header dropped
    assert "dropped.example" not in raw  # non-allowlisted body field
    assert "locations" not in raw and "path" not in raw


def test_discover_400_request_id_shape_withheld(tmp_path):
    resp = httpx.Response(400, headers={
        "x-request-id": 'bad id "quoted"'},
        content=json.dumps({"errors": []}).encode())
    fake = FakeLiveShopify(error_response=resp)
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag["request_id"] == "<withheld>"
    raw = (tmp_path / "logs" / "discover-summary.json").read_text(
        encoding="utf-8")
    assert "quoted" not in raw


def test_discover_400_capability_body_refused(tmp_path):
    body = json.dumps({"errors": [{
        "message": "see gid://shopify/Cart/zzz?key=evilkeycap000"}]
        }).encode()
    fake = FakeLiveShopify(error_response=httpx.Response(400, content=body))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag == {"capture_refused": "guard_hit"}
    for path in (tmp_path / "logs").iterdir():
        text = path.read_text(encoding="utf-8")
        assert "evilkeycap000" not in text
        assert "gid://shopify/Cart" not in text


def test_discover_400_secret_body_refused(tmp_path):
    body = json.dumps({"errors": [{
        "message": "debug echo: shpat_abc123def456"}]}).encode()
    fake = FakeLiveShopify(error_response=httpx.Response(400, content=body))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag == {"capture_refused": "guard_hit"}
    raw = (tmp_path / "logs" / "discover-summary.json").read_text(
        encoding="utf-8")
    assert "shpat_" not in raw


def test_discover_400_malformed_json_body(tmp_path):
    fake = FakeLiveShopify(error_response=httpx.Response(
        400, content=b'{"errors": [{"message":'))
    assert _discover(tmp_path, fake) == 4
    summary = _diag_summary(tmp_path)
    assert summary["category"] == "http:400"
    assert summary["diagnostic"]["body_json"] is False
    assert len(fake.requests) == 1


def test_discover_400_non_json_html_body(tmp_path):
    fake = FakeLiveShopify(error_response=httpx.Response(
        400, content=b"<html><body>Bad Request</body></html>"))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag["body_json"] is False
    raw = (tmp_path / "logs" / "discover-summary.json").read_text(
        encoding="utf-8")
    assert "<html>" not in raw


def test_discover_400_oversize_declared_not_read(tmp_path):
    fake = FakeLiveShopify(error_response=httpx.Response(
        400, headers={"content-length": str(
            shopify_probe.ERROR_BODY_CAP + 1)},
        content=b"{}"))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag["body"] == {"skipped": "content_length_exceeds_cap"}
    assert len(fake.requests) == 1


def test_discover_400_oversize_stream_capped(tmp_path):
    def gen():
        yield b"x" * (shopify_probe.ERROR_BODY_CAP + 100)
    fake = FakeLiveShopify(error_response=httpx.Response(
        400, content=gen()))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag["body"] == {"skipped": "exceeds_cap"}
    assert len(fake.requests) == 1


def test_discover_400_errors_bounded_and_capped(tmp_path):
    long_msg = "m" * 500
    body = json.dumps({"errors": [{"message": long_msg}] + [
        {"message": f"err{i}"} for i in range(7)]}).encode()
    fake = FakeLiveShopify(error_response=httpx.Response(400, content=body))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert len(diag["errors"]) == shopify_probe.ERROR_MAX_ENTRIES
    assert diag["errors_truncated"] is True
    assert diag["errors"][0]["message"] == "m" * 200
    assert diag["errors"][0]["message_truncated"] is True
    for item in diag["errors"]:
        assert len(item["message"]) <= shopify_probe.ERROR_MESSAGE_CAP


def test_discover_400_errors_not_a_list(tmp_path):
    fake = FakeLiveShopify(error_response=httpx.Response(
        400, content=json.dumps({"errors": "boom"}).encode()))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag["errors_present"] is False


def test_probe_400_diagnostic_reaches_final(tmp_path):
    # add_http_500: create + baseline read succeed, first mutation gets a
    # 500 with a non-list errors body; the diagnostic must land in
    # final.json and no further request may be sent.
    fake = FakeLiveShopify(mode="add_http_500")
    assert _discover(tmp_path, fake) == 0
    backend = LiveShopifyBackend(
        SHOP, str(tmp_path / "mapping.json"), allow_live=True,
        transport=fake.transport)
    before = len(fake.requests)
    rc = run_probe(backend, tmp_path / "logs")
    assert rc == 5
    final = _read(tmp_path / "logs", "final.json")
    assert final["overall_verdict"] == "UNKNOWN"
    assert final["diagnostic"]["http_status"] == 500
    assert final["diagnostic"]["errors_present"] is False
    assert len(fake.requests) - before == 3  # create, read, failing add


# ---- streamed real-path capture (P47) ---------------------------- #
# Constructed MockTransport responses arrive pre-consumed; a REAL
# response arrives as a live stream. These tests return custom
# SyncByteStream-backed responses through the full client path and
# assert the adapter still raises http:<status>, the latch still
# engages, and the request count is unchanged.

class _ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False
        self.iter_started = False

    def __iter__(self):
        self.iter_started = True
        yield from self._chunks

    def close(self):
        self.closed = True


class _CountingStreamTransport(httpx.MockTransport):
    def __init__(self, status, chunks, headers=None):
        self.calls = 0
        self.stream = _ChunkStream(chunks)
        self._status = status
        self._headers = headers or {}
        super().__init__(self._handler)

    def _handler(self, request):
        self.calls += 1
        return httpx.Response(self._status, headers=self._headers,
                              stream=self.stream, request=request)


_ERR_BODY = json.dumps({"errors": [{
    "message": "field x does not exist",
    "extensions": {"code": "undefinedField"}}]}).encode()


def _stream_discover(tmp_path, transport):
    return main(_args("discover", tmp_path), transport=transport)


def test_streamed_400_preserves_adapter_category_and_latch(tmp_path):
    t = _CountingStreamTransport(400, [_ERR_BODY])
    assert _stream_discover(tmp_path, t) == 4
    summary = _diag_summary(tmp_path)
    # the adapter's category is http:400 - NOT transport:StreamConsumed
    assert summary["category"] == "http:400"
    assert summary["requests_used"] == 1
    assert t.calls == 1  # capture added no request
    diag = summary["diagnostic"]
    assert diag["http_status"] == 400
    assert diag["errors"] == [{"message": "field x does not exist",
                               "extensions_code": "undefinedField"}]
    # fail-closed latch still engaged after the streamed failure: build
    # a backend on the same (already failed) state is not possible, so
    # prove the latch on a fresh backend whose first call fails.
    t2 = _CountingStreamTransport(400, [_ERR_BODY])
    backend = LiveShopifyBackend(SHOP, None, allow_live=True,
                                 transport=t2)
    with pytest.raises(Exception):
        backend.read_catalog()
    with pytest.raises(Exception) as second:
        backend.read_catalog()
    assert "backend_failed_closed" in str(second.value)
    assert t2.calls == 1  # the refusal sent nothing


def test_streamed_400_oversize_capped_and_category_kept(tmp_path):
    t = _CountingStreamTransport(
        400, [b"x" * (shopify_probe.ERROR_BODY_CAP + 100)])
    assert _stream_discover(tmp_path, t) == 4
    summary = _diag_summary(tmp_path)
    assert summary["category"] == "http:400"
    assert summary["diagnostic"]["body"] == {"skipped": "exceeds_cap"}
    assert t.calls == 1
    assert t.stream.closed  # the abandoned oversize stream was closed


def test_streamed_400_declared_oversize_not_read(tmp_path):
    t = _CountingStreamTransport(
        400, [b"{}"], {"content-length": str(
            shopify_probe.ERROR_BODY_CAP + 1)})
    assert _stream_discover(tmp_path, t) == 4
    summary = _diag_summary(tmp_path)
    assert summary["category"] == "http:400"
    assert summary["diagnostic"]["body"] == {
        "skipped": "content_length_exceeds_cap"}
    assert t.calls == 1


def test_streamed_400_encoded_body_never_read(tmp_path):
    # No bounded-decompression claim: an encoded body is skipped without
    # decompression or reading; status/category/latch are unaffected.
    import gzip
    t = _CountingStreamTransport(
        400, [gzip.compress(_ERR_BODY)], {"content-encoding": "gzip"})
    assert _stream_discover(tmp_path, t) == 4
    summary = _diag_summary(tmp_path)
    assert summary["category"] == "http:400"
    assert summary["diagnostic"]["body"] == {
        "skipped": "content_encoding"}
    assert summary["diagnostic"]["http_status"] == 400
    assert not t.stream.iter_started  # never decompressed, never read
    assert t.stream.closed            # original response still closed
    assert t.calls == 1


def test_constructed_400_encoded_body_skipped(tmp_path):
    import gzip
    resp = httpx.Response(400, headers={"content-encoding": "gzip"},
                          content=gzip.compress(_ERR_BODY))
    fake = FakeLiveShopify(error_response=resp)
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag["body"] == {"skipped": "content_encoding"}


def test_discover_400_identity_encoding_read_normally(tmp_path):
    resp = httpx.Response(400, headers={"content-encoding": "identity"},
                          content=_ERR_BODY)
    fake = FakeLiveShopify(error_response=resp)
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag["errors"] == [{"message": "field x does not exist",
                               "extensions_code": "undefinedField"}]


# ---- JSON-escaped secret evasion (P47) ---------------------------- #

def test_discover_400_unicode_escaped_token_straddling_cap(tmp_path):
    # "\u0073k-" decodes to "sk-": the raw-text scan cannot see it,
    # and the token straddles the 200-char message cap after decoding.
    body = ('{"errors": [{"message": "' + "x" * 195
            + "\\u0073k-abcdefghijklmnopqrstuvwxyz" + '"}]}').encode()
    fake = FakeLiveShopify(error_response=httpx.Response(400,
                                                         content=body))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag == {"capture_refused": "guard_hit"}
    raw = (tmp_path / "logs" / "discover-summary.json").read_text(
        encoding="utf-8")
    assert "sk-abcdef" not in raw
    assert "u0073" not in raw


def test_discover_400_unicode_escaped_secret_in_unextracted_field(tmp_path):
    # The secret sits in a field the extractor never touches, in escaped
    # form: only the decoded whole-body walk can catch it.
    body = ('{"errors": [{"message": "plain"}], "debug": {"note": "'
            + "\\u0073hpat_abc123def456" + '"}}').encode()
    fake = FakeLiveShopify(error_response=httpx.Response(400,
                                                         content=body))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag == {"capture_refused": "guard_hit"}
    raw = (tmp_path / "logs" / "discover-summary.json").read_text(
        encoding="utf-8")
    assert "shpat" not in raw
    assert "u0073" not in raw


def test_streamed_400_downstream_never_reads_error_body(tmp_path):
    # The response returned to the client after capture carries an
    # empty body: nothing downstream can read error payloads.
    t = _CountingStreamTransport(400, [_ERR_BODY])
    backend = LiveShopifyBackend(SHOP, None, allow_live=True,
                                 transport=t)
    with pytest.raises(Exception) as caught:
        backend.read_catalog()
    assert str(caught.value) == "http:400"  # adapter category intact


# ---- per-field secret guard (P47) --------------------------------- #

@pytest.mark.parametrize("payload", [
    "ghp_abcdefghijklmnop",                       # GitHub token
    "gho_abcdefghijklmnop",
    "github_pat_11ABCDEFG_abcdefghijklm",
    "sk-abcdefghijklmnop",                        # OpenAI-style key
    "AKIAIOSFODNN7EXAMPLE",                       # AWS access key id
    "eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0",      # JWT
    "-----BEGIN PRIVATE KEY-----",
    "Bearer abcdefghijklmnop",
    "password=hunter2",
    "token: abcdefghij",
    "cookie: session=abc",
    "api_key = 1234567890",
    "user@example.com",                           # email address
])
def test_discover_400_each_secret_class_refused(tmp_path, payload):
    body = json.dumps({"errors": [{"message": payload}]}).encode()
    fake = FakeLiveShopify(error_response=httpx.Response(400,
                                                         content=body))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag == {"capture_refused": "guard_hit"}
    raw = (tmp_path / "logs" / "discover-summary.json").read_text(
        encoding="utf-8")
    assert payload not in raw


def test_discover_400_secret_beyond_truncation_refused(tmp_path):
    # The token starts AFTER the 200-char message cap: scanning the
    # original before truncation must still refuse everything.
    body = json.dumps({"errors": [{
        "message": "x" * 300 + " ghp_abcdefghijklmnop"}]}).encode()
    fake = FakeLiveShopify(error_response=httpx.Response(400,
                                                         content=body))
    assert _discover(tmp_path, fake) == 4
    assert _diag_summary(tmp_path)["diagnostic"] == {
        "capture_refused": "guard_hit"}


def test_discover_400_secret_straddling_boundary_refused(tmp_path):
    # The token starts before the cap but extends past it.
    body = json.dumps({"errors": [{
        "message": "x" * 195 + "sk-abcdefghijklmnopqrstuvwxyz"}]
    }).encode()
    fake = FakeLiveShopify(error_response=httpx.Response(400,
                                                         content=body))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag == {"capture_refused": "guard_hit"}
    raw = (tmp_path / "logs" / "discover-summary.json").read_text(
        encoding="utf-8")
    assert "sk-abcdef" not in raw


def test_discover_400_code_shape_policy(tmp_path):
    body = json.dumps({"errors": [{
        "message": "plain",
        "code": "THROTTLED",
        "extensions": {"code": 'bad code "quoted"'}}]}).encode()
    fake = FakeLiveShopify(error_response=httpx.Response(400,
                                                         content=body))
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag["errors"] == [{"message": "plain", "code": "THROTTLED",
                               "extensions_code": "<withheld>"}]
    raw = (tmp_path / "logs" / "discover-summary.json").read_text(
        encoding="utf-8")
    assert "quoted" not in raw


def test_discover_400_request_id_token_like_withheld(tmp_path):
    # Passes the character shape but matches a secret pattern.
    resp = httpx.Response(400, headers={
        "x-request-id": "sk-abcdefghijklmnop"},
        content=json.dumps({"errors": []}).encode())
    fake = FakeLiveShopify(error_response=resp)
    assert _discover(tmp_path, fake) == 4
    diag = _diag_summary(tmp_path)["diagnostic"]
    assert diag["request_id"] == "<withheld>"
    raw = (tmp_path / "logs" / "discover-summary.json").read_text(
        encoding="utf-8")
    assert "sk-abcdef" not in raw

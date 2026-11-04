"""Offline tests for the WB2 live wiring (harness/shopify_live_backend.py +
harness/shopify_live_runner.py). LABELED SIMULATION - NOT LIVE PROOF: these
run against the stateful fake; they prove the new wiring is correct before
any approved live request, and they are the gate that runs first in the
Harness procedure."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from harness.shopify_live_backend import (LIVE_OPT_IN_ENV, MAX_REQUESTS,
                                          TOKEN_ENV, RequestBudgetExceeded,
                                          ShopifyLiveBackend)
from harness import shopify_live_runner as runner

try:  # tests/ has no __init__.py in this repo; both roots work
    from tests.shopify_fake import FakeShopify
except ImportError:  # pragma: no cover
    from shopify_fake import FakeShopify

GIDS = {f"CARTPROOF-TEST-{c}": f"gid://shopify/ProductVariant/{i}"
        for i, c in enumerate("ABC", start=1)}
MAPPING = {"variants": dict(GIDS)}


def _mapping_file(tmp_path: Path) -> str:
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(MAPPING), encoding="utf-8")
    return str(path)


def _backend(fake, tmp_path):
    return ShopifyLiveBackend("cartproof.myshopify.com",
                              _mapping_file(tmp_path), token="fake-token",
                              transport=fake.transport)


def _fake(valid=None):
    return FakeShopify(valid_variant_gids=valid or list(GIDS.values()))


# ---------------------------------------------------------- construction
def test_mock_transport_constructs_without_live_opt_in(tmp_path,
                                                       monkeypatch):
    monkeypatch.delenv(LIVE_OPT_IN_ENV, raising=False)
    be = _backend(_fake(), tmp_path)
    assert be.known_item_keys() == set(GIDS)


def test_real_transport_requires_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv(LIVE_OPT_IN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="live_opt_in_missing"):
        ShopifyLiveBackend("cartproof.myshopify.com",
                           _mapping_file(tmp_path), token="fake-token")


def test_token_resolves_from_file_path_env(tmp_path, monkeypatch):
    from harness.shopify_live_backend import TOKEN_FILE_ENV, resolve_token
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.delenv(TOKEN_FILE_ENV, raising=False)
    assert resolve_token() == ""
    tok = tmp_path / "tok.txt"
    tok.write_text("tok-from-file\n", encoding="utf-8")
    monkeypatch.setenv(TOKEN_FILE_ENV, str(tok))
    assert resolve_token() == "tok-from-file"
    monkeypatch.setenv(TOKEN_ENV, "tok-env-wins")
    assert resolve_token() == "tok-env-wins"


def test_token_must_come_from_env_or_arg(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="token_missing"):
        ShopifyLiveBackend("cartproof.myshopify.com",
                           _mapping_file(tmp_path),
                           transport=_fake().transport)


def test_token_header_sent_and_logged_safely(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get(
            "X-Shopify-Storefront-Access-Token")
        return httpx.Response(
            200, json={"data": {"cartCreate": {
                "cart": {"id": "gid://shopify/" + "Cart/QQ" + "?key=" + "secretkey"},
                "userErrors": []}}},
            headers={"X-Request-ID": "req-123"})

    be = ShopifyLiveBackend("cartproof.myshopify.com",
                            _mapping_file(tmp_path), token="fake-token",
                            transport=httpx.MockTransport(handler))
    be.ensure_cart()
    assert seen["auth"] == "fake-token"
    entry = be.request_log[-1]
    assert entry["status"] == 200 and entry["x_request_id"] == "req-123"
    assert entry["elapsed_ms"] is not None
    blob = json.dumps(be.request_log)
    assert "fake-token" not in blob and "secretkey" not in blob
    assert "gid://shopify/" + "Cart/QQ" not in blob
    # cart tracking records only the redacted ref
    assert be.carts_created == [be.cart_ref]
    assert be.cart_ref not in ("secretkey",) and len(be.cart_ref) == 12


def test_install_mapping_keeps_accounting_and_cart(tmp_path):
    be = _backend(_fake(), tmp_path)
    be.ensure_cart()
    count = be.request_count
    ref = be.cart_ref
    be.install_mapping({"variants": {"NEW-SKU": "gid://x/9"}})
    assert be.known_item_keys() == {"NEW-SKU"}
    assert be.request_count == count and be.cart_ref == ref


# --------------------------------------------------------------- budget
def test_budget_aborts_before_extra_send(tmp_path):
    be = _backend(_fake(), tmp_path)
    be._max_requests = 2
    be.ensure_cart()          # request 1
    be.acquire_state()        # request 2
    with pytest.raises(RequestBudgetExceeded):
        be.acquire_state()    # would be request 3: refused before send
    assert be.request_count == 2


# ------------------------------------------------------------- full run
def test_three_scenario_run_against_fake(tmp_path):
    be = _backend(_fake(), tmp_path)
    workdir = tmp_path / "run"
    summary = runner.run(be, workdir, mapping=dict(MAPPING))
    assert summary["outcome"] == "completed"
    assert summary["all_expectations_met"] is True
    by_name = {s["scenario"]: s for s in summary["scenarios"]}
    assert by_name["s1_match"]["verdict"] == "COMPLETE"
    assert by_name["s1_match"]["seed_lines"][0]["survived_unchanged"]
    s2 = by_name["s2_seeded_mismatch"]
    assert s2["verdict"] == "PARTIAL"
    assert {m["sku"] for m in s2["mismatches"]} == {
        "CARTPROOF-TEST-A", "CARTPROOF-TEST-C"}
    s3 = by_name["s3_bounded_recovery"]
    assert s3["verdict"] == "COMPLETE"
    assert all(s["survived_unchanged"] for s in s3["seed_lines"])
    assert all(s.get("validated") for s in summary["scenarios"])
    # reconciliation + residual-cart report
    assert summary["request_count"] == 11
    assert summary["request_planned"] == 11
    assert summary["request_count"] <= MAX_REQUESTS
    assert len(summary["carts"]) == 1
    assert set(summary["carts"][0]) == {"cart_ref", "disposition"}
    assert len(summary["carts"][0]["cart_ref"]) == 12
    assert "abandoned" in summary["carts"][0]["disposition"]
    # redaction held end-to-end
    assert runner.guard_scan(workdir, "fake-token") == []


def test_mutation_failure_stops_before_later_scenarios(tmp_path):
    # SKU A's variant GID unknown to the store: S1's request add fails.
    fake = _fake(valid=[GIDS["CARTPROOF-TEST-B"], GIDS["CARTPROOF-TEST-C"]])
    be = _backend(fake, tmp_path)
    workdir = tmp_path / "run"
    with pytest.raises(runner.UnexpectedFailure, match="mutation_failed"):
        runner.run(be, workdir, mapping=dict(MAPPING))
    summary = json.loads((workdir / "run_summary.json")
                         .read_text(encoding="utf-8"))
    assert summary["outcome"] == "unexpected_failure"
    assert summary["scenarios_completed"] == []
    assert not (workdir / "s2_seeded_mismatch").exists()


def test_s2_validation_requires_exact_discrepancy(tmp_path):
    be = _backend(_fake(), tmp_path)
    outcome = {"scenario": "s2_seeded_mismatch", "verdict": "PARTIAL",
               "mismatches": [{"sku": "CARTPROOF-TEST-A",
                               "expected_quantity": 3,
                               "actual_quantity": 2}],
               "seed_lines": [], "resolution": "remote-cart",
               "check_duration_ms": 1}
    with pytest.raises(runner.UnexpectedFailure,
                       match="s2_seeded_mismatch_unexpected_verdict"):
        runner._validate(be, outcome, "PARTIAL",
                         want_mismatches={("CARTPROOF-TEST-A", 3, 2),
                                          ("CARTPROOF-TEST-C", 0, 1)})


def test_run_reports_unexpected_failure(tmp_path):
    fake = FakeShopify(mode="http_500",
                       valid_variant_gids=list(GIDS.values()))
    be = _backend(fake, tmp_path)
    with pytest.raises(runner.UnexpectedFailure):
        runner.run(be, tmp_path / "run", mapping=dict(MAPPING))
    saved = json.loads((tmp_path / "run" / "run_summary.json")
                       .read_text(encoding="utf-8"))
    assert saved["outcome"] == "unexpected_failure"


# ----------------------------------------------------- finish + finalize
def test_finish_guard_refusal_writes_no_manifest(tmp_path):
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "leak.json").write_text('{"note": "fake-token"}',
                                       encoding="utf-8")
    assert runner.finish(workdir, "fake-token", 0) == 4
    assert not (workdir / "manifest.json").exists()
    guard = json.loads((workdir / "guard.json").read_text(encoding="utf-8"))
    assert guard["clean"] is False and guard["offenders"]


def test_manifest_keys_use_posix_separators_on_every_os(tmp_path):
    """Regression: manifest keys must be POSIX-relative on Windows too
    (the WB2 G2 failure: backslash keys broke key equality)."""
    workdir = tmp_path / "run"
    (workdir / "a" / "b").mkdir(parents=True)
    (workdir / "a" / "b" / "c.json").write_text("{}", encoding="utf-8")
    assert runner.finish(workdir, "fake-token", 0) == 0
    manifest = json.loads((workdir / "manifest.json")
                          .read_text(encoding="utf-8"))
    for key in manifest["files"]:
        assert "\\" not in key, key
    assert "a/b/c.json" in manifest["files"]


def test_manifest_hashes_nested_manifest_files(tmp_path):
    workdir = tmp_path / "run"
    (workdir / "nested").mkdir(parents=True)
    (workdir / "nested" / "manifest.json").write_text("{}",
                                                     encoding="utf-8")
    assert runner.finish(workdir, "fake-token", 0) == 0
    manifest = json.loads((workdir / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert "nested/manifest.json" in manifest["files"]
    assert "manifest.json" not in manifest["files"]


def test_finish_clean_writes_manifest(tmp_path):
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "a.json").write_text("{}", encoding="utf-8")
    assert runner.finish(workdir, "fake-token", 2) == 2
    manifest = json.loads((workdir / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["count"] == 2  # a.json + guard.json


def test_finalize_copies_logs_guards_closure(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, ("tok-zz-" + "not-in-source"))
    _pin_gates(monkeypatch)
    run_dir = tmp_path / "run-x"
    run_dir.mkdir()
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    console = tmp_path / "console-x.txt"
    console.write_text("status: completed", encoding="utf-8")
    offline = tmp_path / "offline-tests-x.txt"
    offline.write_text("11 passed", encoding="utf-8")
    g0 = tmp_path / "g0-precheck-x.txt"
    g0.write_text("HEAD=168d9d65\nDIGEST=8ca0708a", encoding="utf-8")
    code = runner.main(["--finalize", str(run_dir), "--console",
                        str(console), "--offline", str(offline),
                        "--gate-log", str(g0)])
    assert code == 0
    manifest = json.loads((run_dir / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {
        "run_summary.json", console.name, offline.name, g0.name,
        "guard.json", "postcheck.json",
        "payload-snapshot/shopify_live_backend.py",
        "payload-snapshot/shopify_live_runner.py",
        "payload-snapshot/test_shopify_live_wb2.py"}
    post = json.loads((run_dir / "postcheck.json").read_text(encoding="utf-8"))
    assert post["tree_unchanged"] is True


def test_finalize_postcheck_mismatch_stops_no_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "fake-token")
    _pin_gates(monkeypatch, digest="f" * 64)
    run_dir = tmp_path / "run-pm"
    run_dir.mkdir()
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    code = runner.main(["--finalize", str(run_dir)])
    assert code == 1
    assert not (run_dir / "manifest.json").exists()
    post = json.loads((run_dir / "postcheck.json").read_text(encoding="utf-8"))
    assert post["tree_unchanged"] is False


def test_finalize_refuses_leak_no_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, ("tok-zz-" + "not-in-source"))
    _pin_gates(monkeypatch)
    run_dir = tmp_path / "run-y"
    run_dir.mkdir()
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    # a stale manifest from an earlier successful finish must NOT survive
    # a refusal
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    console = tmp_path / "console-y.txt"
    console.write_text("oops " + ("tok-zz-" + "not-in-source"), encoding="utf-8")
    code = runner.main(["--finalize", str(run_dir), "--console",
                        str(console), "--offline", None])
    assert code == 4
    assert not (run_dir / "manifest.json").exists()


def test_finalize_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "fake-token")
    run_dir = tmp_path / "run-z"
    run_dir.mkdir()
    (run_dir / "console-z.txt").write_text("old", encoding="utf-8")
    console = tmp_path / "console-z.txt"
    console.write_text("new", encoding="utf-8")
    code = runner.main(["--finalize", str(run_dir), "--console",
                        str(console), "--offline", None])
    assert code == 1
    assert (run_dir / "console-z.txt").read_text() == "old"


# ---------------------------------------------------------------- guard
def test_guard_refuses_cart_capability(tmp_path):
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "leak.json").write_text(
        '{"cart": "' + "gid://shopify/" + "Cart/QQ" + "?key=" + "abc" + '"}', encoding="utf-8")
    assert runner.guard_scan(workdir, "other-token")


def test_executed_source_files_are_guard_clean(tmp_path):
    import shutil
    repo = Path(runner.__file__).resolve().parents[1]
    snap = tmp_path / "snap"
    snap.mkdir()
    for rel in runner.PAYLOAD_FILES:
        shutil.copy(repo / rel, snap / Path(rel).name)
    assert runner.guard_scan(snap, ("tok-zz-" + "not-in-source")) == []


def test_guard_scans_nested_manifest_too(tmp_path):
    workdir = tmp_path / "run"
    (workdir / "nested").mkdir(parents=True)
    (workdir / "nested" / "manifest.json").write_text(
        '{"cart": "' + "gid://shopify/" + "Cart/QQ" + "?key=" + "abc" + '"}', encoding="utf-8")
    offenders = runner.guard_scan(workdir, "other-token")
    assert offenders and offenders[0]["file"] == "manifest.json"


def test_s2_seed_loss_stops_run(tmp_path):
    be = _backend(_fake(), tmp_path)
    outcome = {"scenario": "s2_seeded_mismatch", "verdict": "PARTIAL",
               "mismatches": [{"sku": "CARTPROOF-TEST-A",
                               "expected_quantity": 3,
                               "actual_quantity": 2},
                              {"sku": "CARTPROOF-TEST-C",
                               "expected_quantity": 0,
                               "actual_quantity": 1}],
               "seed_lines": [{"survived_unchanged": False}],
               "resolution": "remote-cart", "check_duration_ms": 1}
    with pytest.raises(runner.UnexpectedFailure,
                       match="did not survive unchanged"):
        runner._validate(be, outcome, "PARTIAL",
                         want_mismatches={("CARTPROOF-TEST-A", 3, 2),
                                          ("CARTPROOF-TEST-C", 0, 1)})


# ------------------------------------------------------------ preflight
def _pin_gates(monkeypatch, head=None, digest=None):
    monkeypatch.setattr(runner, "_head_rev",
                        lambda root: head or runner.EXPECTED_HEAD)
    monkeypatch.setattr(runner, "_source_digest",
                        lambda root: digest or runner.EXPECTED_SOURCE_DIGEST)


def test_preflight_gates_and_records_hashes(tmp_path, monkeypatch):
    (tmp_path / "var" / "acceptance-p52").mkdir(parents=True)
    monkeypatch.setenv(TOKEN_ENV, "fake-token")
    monkeypatch.setenv(LIVE_OPT_IN_ENV, "1")
    _pin_gates(monkeypatch)
    gates = runner.preflight(tmp_path, "t1")
    assert gates["ok"] is False  # offline gate file missing
    (tmp_path / "var" / "acceptance-p52"
     / "offline-tests-t1.txt").write_text("23 passed\noffline exit code: 0", encoding="utf-8")
    gates = runner.preflight(tmp_path, "t1")
    assert gates["ok"] is True
    assert gates["head_rev_matches"] is True
    assert gates["source_digest_matches"] is True
    assert "store/backends.py" in gates["dependency_sha256"]
    rec = gates["dependency_sha256"]["store/backends.py"]
    assert set(rec) == {"present", "sha256"}


def test_preflight_head_mismatch_stops(tmp_path, monkeypatch):
    (tmp_path / "var" / "acceptance-p52").mkdir(parents=True)
    (tmp_path / "var" / "acceptance-p52"
     / "offline-tests-t1.txt").write_text("23 passed\noffline exit code: 0", encoding="utf-8")
    monkeypatch.setenv(TOKEN_ENV, "fake-token")
    monkeypatch.setenv(LIVE_OPT_IN_ENV, "1")
    _pin_gates(monkeypatch, head="0" * 40)
    gates = runner.preflight(tmp_path, "t1")
    assert gates["ok"] is False
    assert gates["head_rev_matches"] is False
    assert gates["source_digest_matches"] is True


def test_preflight_digest_mismatch_stops(tmp_path, monkeypatch):
    (tmp_path / "var" / "acceptance-p52").mkdir(parents=True)
    (tmp_path / "var" / "acceptance-p52"
     / "offline-tests-t1.txt").write_text("23 passed\noffline exit code: 0", encoding="utf-8")
    monkeypatch.setenv(TOKEN_ENV, "fake-token")
    monkeypatch.setenv(LIVE_OPT_IN_ENV, "1")
    _pin_gates(monkeypatch, digest="f" * 64)
    gates = runner.preflight(tmp_path, "t1")
    assert gates["ok"] is False
    assert gates["source_digest_matches"] is False


# ---------------------------------------------------------- R2: capture
from store.backends import ShopifyBackendError  # noqa: E402

CATALOG_BODY = {"data": {"products": {"edges": [
    {"node": {"id": f"gid://shopify/Product/{i}", "title": f"T{i}",
              "variants": {"edges": [
                  {"node": {"id": f"gid://shopify/ProductVariant/{i}",
                            "sku": sku,
                            "price": {"amount": "1.00",
                                      "currencyCode": "USD"}}}]}}}
    for i, sku in enumerate(
        ["CARTPROOF-TEST-A", "CARTPROOF-TEST-B", "CARTPROOF-TEST-C"],
        start=1)]}}}


def _resp(status=200, body=None, raw=None):
    kwargs = {"headers": {"X-Request-ID": "req-1"}}
    if raw is not None:
        kwargs["content"] = raw
    else:
        kwargs["json"] = body
    return httpx.Response(status, **kwargs)


def _stream_resp(raw: bytes, status: int = 200,
                 encoding: str = None):
    """A genuinely STREAMING mock response (not pre-buffered), so tests
    exercise the real raw-read + bounded-decode path, not the in-memory
    fallback."""
    from httpx._content import IteratorByteStream  # test-only
    headers = {"X-Request-ID": "req-1"}
    if encoding:
        headers["Content-Encoding"] = encoding
    chunks = [raw[i:i + 4096] for i in range(0, len(raw), 4096)] or [b""]
    return httpx.Response(status, headers=headers,
                          stream=IteratorByteStream(iter(chunks)))


def _transport_backend(tmp_path, handler):
    return ShopifyLiveBackend(
        "cartproof.myshopify.com", _mapping_file(tmp_path),
        token="fake-token", transport=httpx.MockTransport(handler))


def test_gql_error_categories_match_parent(tmp_path):
    """The R2 override keeps the reviewed parent's categories: transport /
    http:<status> / invalid_json / no_data / graphql_errors."""
    be = _transport_backend(
        tmp_path, lambda req: (_ for _ in ()).throw(
            httpx.ConnectError("boom")))
    with pytest.raises(ShopifyBackendError, match="transport:"):
        be._gql("query", {})

    be = _transport_backend(tmp_path, lambda req: _resp(401, body={}))
    with pytest.raises(ShopifyBackendError, match="http:401"):
        be._gql("query", {})

    be = _transport_backend(tmp_path,
                            lambda req: _resp(200, raw=b"not json"))
    with pytest.raises(ShopifyBackendError, match="invalid_json"):
        be._gql("query", {})

    be = _transport_backend(tmp_path, lambda req: _resp(200, {"data": None}))
    with pytest.raises(ShopifyBackendError, match="no_data"):
        be._gql("query", {})

    be = _transport_backend(
        tmp_path, lambda req: _resp(200, {"errors": [{"message": "x"}]}))
    with pytest.raises(ShopifyBackendError, match="graphql_errors"):
        be._gql("query", {})


def test_body_too_large_is_bounded_before_decode(tmp_path):
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(b"x" * 70000))
    with pytest.raises(ShopifyBackendError, match="body_too_large"):
        be._gql("query", {})


def test_capture_retains_bounded_safe_details(tmp_path):
    body = {"errors": [
        {"message": "Field 'price' is not accessible",
         "extensions": {"code": "ACCESS_DENIED"}},
        {"message": "second problem"},
        {"message": "third problem"},
        {"message": "fourth problem"}]}
    be = _transport_backend(tmp_path, lambda req: _resp(200, body))
    with pytest.raises(ShopifyBackendError):
        be.read_catalog()
    entry = be.request_log[-1]
    assert entry["status"] == 200 and entry["x_request_id"] == "req-1"
    assert entry["graphql_error_codes"] == ["ACCESS_DENIED"]
    assert entry["graphql_error_messages"] == [
        "Field 'price' is not accessible", "second problem",
        "third problem"]
    assert entry["graphql_error_dropped"] == 1


def test_capture_never_retains_token_or_capability(tmp_path):
    token_echo = "denied: token fake-token echoed back"
    # Built by concatenation so this SOURCE file never self-matches the
    # guard patterns (the archive snapshot scans the payload files).
    plants = [token_echo,
              "cart " + "gid://" + "shopify/Cart/abcdef123456 leaked",
              "see ?" + "key=abcdef0123456789 for details",
              "sh" + "pat_abcdef0123 leaked",
              "Authorization: Bear" + "er abc.def-123"]
    body = {"errors": [{"message": p} for p in plants]
            + [{"message": "safe complaint"}]}
    be = _transport_backend(tmp_path, lambda req: _resp(200, body))
    with pytest.raises(ShopifyBackendError):
        be.read_catalog()
    entry = be.request_log[-1]
    # ANY unsafe candidate refuses the WHOLE capture - no partial detail
    assert entry["graphql_capture"] == "capture_refused_unsafe_or_unbounded"
    assert "graphql_error_messages" not in entry
    blob = json.dumps(be.request_log)
    for plant in plants:
        assert plant not in blob


def test_capture_scans_full_length_before_truncation(tmp_path):
    long_safe = "y" * 200
    token_at_tail = "z" * 200 + "fake-token"
    body = {"errors": [{"message": token_at_tail}, {"message": long_safe}]}
    be = _transport_backend(tmp_path, lambda req: _resp(200, body))
    with pytest.raises(ShopifyBackendError):
        be.read_catalog()
    entry = be.request_log[-1]
    # the tail-planted token is caught by the FULL-LENGTH scan of the
    # whole decoded structure - the entire capture is refused, never
    # truncated into retention
    assert entry["graphql_capture"] == "capture_refused_unsafe_or_unbounded"
    blob = json.dumps(be.request_log)
    assert "fake-token" not in blob and "y" * 121 not in blob


def test_failure_path_persists_request_log_and_summary(tmp_path):
    body = {"errors": [{"message": "denied: fake-token echo"},
                       {"message": "plain message"}]}
    be = _transport_backend(tmp_path, lambda req: _resp(200, body))
    with pytest.raises(ShopifyBackendError):
        be.read_catalog()
    workdir = tmp_path / "run"
    workdir.mkdir()
    runner.persist_failure_evidence(be, workdir, "unexpected_failure",
                                    "ShopifyBackendError: graphql_errors")
    log = json.loads((workdir / "request_log.json").read_text())
    # the planted token refused the WHOLE capture pre-retention
    assert log[-1]["graphql_capture"] == "capture_refused_unsafe_or_unbounded"
    summary = json.loads((workdir / "run_summary.json").read_text())
    assert summary["outcome"] == "unexpected_failure"
    assert summary["request_count"] == 1
    # nothing unsafe was retained, so the archive scan over the persisted
    # evidence is clean
    assert runner.guard_scan(workdir, "fake-token") == []


def test_build_mapping_parses_parent_shape(tmp_path):
    """Regression: read_catalog returns body["data"] directly; the parser
    must not unwrap a second 'data' level (found by inspection in R2)."""
    be = _transport_backend(tmp_path, lambda req: _resp(200, CATALOG_BODY))
    workdir = tmp_path / "run"
    workdir.mkdir()
    mapping = runner.build_mapping_from_catalog(be, workdir)
    assert mapping["variants"] == dict(GIDS)
    summary = json.loads((workdir / "catalog_summary.json").read_text())
    assert summary["product_edges_seen"] == 3


def test_probe_mode_success_and_request_cap(tmp_path):
    be = _transport_backend(tmp_path, lambda req: _resp(200, CATALOG_BODY))
    be.set_request_cap(1)
    workdir = tmp_path / "run"
    workdir.mkdir()
    summary = runner.probe_catalog(be, workdir)
    assert summary["product_edges_seen"] == 3
    assert summary["synthetic_skus_found"] == sorted(GIDS)
    assert be.request_count == 1
    assert (workdir / "catalog_summary.json").exists()
    with pytest.raises(RequestBudgetExceeded):
        be.read_catalog()
    assert be.request_count == 1


def test_capture_refusal_covers_unretained_fields_and_keys(tmp_path):
    """Guard hits in UNRETAINED places - a custom error field and even a
    KEY - still refuse the whole capture."""
    body = {"errors": [{"message": "plain",
                        "debug_note": "echo of fake-token here"},
                       {"fake-token-key": {"message": "plain"}}]}
    be = _transport_backend(tmp_path, lambda req: _resp(200, body))
    with pytest.raises(ShopifyBackendError):
        be.read_catalog()
    entry = be.request_log[-1]
    assert entry["graphql_capture"] == "capture_refused_unsafe_or_unbounded"
    assert "fake-token" not in json.dumps(be.request_log)


def test_capture_scans_decoded_form_not_just_raw_text(tmp_path):
    """A token that exists only AFTER JSON unescaping (raw text carries
    the \\u-escaped form) is still caught via the structure scan."""
    escaped = "".join("\\u%04x" % ord(c) for c in "fake-token")
    raw = ('{"errors": [{"message": "denied: ' + escaped + '"}]}').encode()
    assert b"fake-token" not in raw  # raw text alone would miss it
    be = _transport_backend(tmp_path, lambda req: _resp(200, raw=raw))
    with pytest.raises(ShopifyBackendError):
        be.read_catalog()
    assert be.request_log[-1]["graphql_capture"] == \
        "capture_refused_unsafe_or_unbounded"


def test_capture_refuses_excessive_depth(tmp_path):
    deep = current = {}
    for _ in range(20):
        current["nested"] = {}
        current = current["nested"]
    body = {"errors": [{"message": "plain", "extensions": deep}]}
    be = _transport_backend(tmp_path, lambda req: _resp(200, body))
    with pytest.raises(ShopifyBackendError):
        be.read_catalog()
    assert be.request_log[-1]["graphql_capture"] == \
        "capture_refused_unsafe_or_unbounded"


def test_compressed_bomb_and_decode_cap(tmp_path):
    import zlib
    bomb = zlib.compress(b"A" * 1_000_000)  # small wire, huge decoded
    assert len(bomb) < 65536
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(bomb, encoding="deflate"))
    with pytest.raises(ShopifyBackendError, match="body_too_large"):
        be.read_catalog()


def test_compressed_wire_cap_applies_before_decode(tmp_path):
    import os
    import zlib
    payload = zlib.compress(os.urandom(70000))  # incompressible: >64KiB
    assert len(payload) > 65536
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(payload, encoding="deflate"))
    with pytest.raises(ShopifyBackendError, match="body_too_large"):
        be.read_catalog()


def test_valid_gzip_body_decodes_and_captures(tmp_path):
    import gzip
    raw = gzip.compress(json.dumps(
        {"errors": [{"message": "plain",
                     "extensions": {"code": "THROTTLED"}}]}).encode())
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(raw, encoding="gzip"))
    with pytest.raises(ShopifyBackendError, match="graphql_errors"):
        be.read_catalog()
    entry = be.request_log[-1]
    assert entry["graphql_error_codes"] == ["THROTTLED"]
    assert entry["graphql_error_messages"] == ["plain"]


def test_invalid_utf8_is_invalid_json(tmp_path):
    raw = b'{"errors": [{"message": "\xff\xfe bad bytes"}]}'
    be = _transport_backend(tmp_path, lambda req: _resp(200, raw=raw))
    with pytest.raises(ShopifyBackendError, match="invalid_json"):
        be.read_catalog()


def test_unsupported_encoding_is_a_fixed_category(tmp_path):
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(b"whatever", encoding="br"))
    with pytest.raises(ShopifyBackendError, match="unsupported_encoding"):
        be.read_catalog()


def test_x_request_id_is_guarded_before_retention(tmp_path):
    body = {"errors": [{"message": "plain"}]}
    evil = "fake-token" + "x" * 100  # fails charset, length and scan
    be = _transport_backend(
        tmp_path, lambda req: httpx.Response(
            200, json=body, headers={"X-Request-ID": evil}))
    with pytest.raises(ShopifyBackendError):
        be.read_catalog()
    entry = be.request_log[-1]
    assert entry["x_request_id"] == "<withheld>"
    assert evil not in json.dumps(be.request_log)


def test_truncated_gzip_is_decode_failed(tmp_path):
    import gzip
    raw = gzip.compress(b'{"errors": []}')[:-4]  # cut the trailer
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(raw, encoding="gzip"))
    with pytest.raises(ShopifyBackendError, match="decode_failed"):
        be.read_catalog()


def test_trailing_garbage_after_gzip_is_decode_failed(tmp_path):
    import gzip
    raw = gzip.compress(b'{"errors": []}') + b"garbage"
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(raw, encoding="gzip"))
    with pytest.raises(ShopifyBackendError, match="decode_failed"):
        be.read_catalog()


def test_concatenated_gzip_members_are_decode_failed(tmp_path):
    import gzip
    raw = gzip.compress(b'{"errors": []}') + gzip.compress(b'{"more": 1}')
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(raw, encoding="gzip"))
    with pytest.raises(ShopifyBackendError, match="decode_failed"):
        be.read_catalog()


def test_corrupt_deflate_is_decode_failed(tmp_path):
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(b"\x00\x01\x02\x03not-zlib",
                                           encoding="deflate"))
    with pytest.raises(ShopifyBackendError, match="decode_failed"):
        be.read_catalog()


def test_decode_exact_cap_accepted_and_one_over_refused(tmp_path):
    import zlib
    from harness.shopify_live_backend import MAX_DECODED_BYTES
    # a valid errors body padded to EXACTLY the cap is accepted
    base = '{"errors": [{"message": "'
    tail = '"}]}'
    exact = (base + "y" * (MAX_DECODED_BYTES - len(base) - len(tail))
             + tail).encode()
    assert len(exact) == MAX_DECODED_BYTES
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(zlib.compress(exact),
                                           encoding="deflate"))
    with pytest.raises(ShopifyBackendError, match="graphql_errors"):
        be.read_catalog()
    # one byte over the cap fails closed
    over = (base + "y" * (MAX_DECODED_BYTES - len(base) - len(tail) + 1)
            + tail).encode()
    be = _transport_backend(
        tmp_path, lambda req: _stream_resp(zlib.compress(over),
                                           encoding="deflate"))
    with pytest.raises(ShopifyBackendError, match="body_too_large"):
        be.read_catalog()


def test_transport_failure_records_fixed_non_http_outcome(tmp_path):
    be = _transport_backend(
        tmp_path, lambda req: (_ for _ in ()).throw(
            httpx.ConnectError("boom")))
    with pytest.raises(ShopifyBackendError, match="transport:"):
        be.read_catalog()
    entry = be.request_log[-1]
    # request count + fixed marker + error type only - NO HTTP claims
    assert entry["seq"] == 1
    assert entry["outcome"] == "transport_failed_no_response"
    assert entry["error_type"] == "ConnectError"
    assert "status" not in entry and "x_request_id" not in entry


def test_error_codes_are_charset_allowlisted(tmp_path):
    body = {"errors": [
        {"message": "m1", "extensions": {"code": "ACCESS_DENIED"}},
        {"message": "m2", "extensions": {"code": "NOT A CODE!!"}},
        {"message": "m3", "extensions": {"code": "x" * 41}}]}
    be = _transport_backend(tmp_path, lambda req: _resp(200, body))
    with pytest.raises(ShopifyBackendError):
        be.read_catalog()
    entry = be.request_log[-1]
    assert entry["graphql_error_codes"] == ["ACCESS_DENIED"]


def test_probe_success_archive_contains_request_evidence(tmp_path):
    """A SUCCESSFUL probe archives the request metadata too: count 1, the
    enforced cap, and the log entry with status + X-Request-ID - not only
    the catalog summary."""
    be = _transport_backend(tmp_path, lambda req: _resp(200, CATALOG_BODY))
    be.set_request_cap(1)
    workdir = tmp_path / "run"
    workdir.mkdir()
    runner.probe_catalog(be, workdir)
    assert runner.finish(workdir, "fake-token", 0) == 0
    manifest = json.loads((workdir / "manifest.json").read_text())
    for name in ("request_log.json", "run_summary.json",
                 "catalog_summary.json"):
        assert name in manifest["files"]
    summary = json.loads((workdir / "run_summary.json").read_text())
    assert summary["outcome"] == "probe_completed"
    assert summary["request_count"] == 1
    assert summary["request_cap"] == 1
    log = json.loads((workdir / "request_log.json").read_text())
    assert log[0]["status"] == 200
    assert log[0]["x_request_id"] == "req-1"
    assert runner.guard_scan(workdir, "fake-token") == []

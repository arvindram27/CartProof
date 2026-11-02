"""Offline tests for ServerState + run_check over BOTH backends (P40).

Covers what the v1 suite could not: the check path over the SHOPIFY
backend (stateful fake) - success, seed deletion, duplicate merchandise
lines, truncated/malformed/unavailable degradation, and provenance. Local
backend: loopback mock store via the same `store` fixture family as
tests/test_mcp_server.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from store import models
from store.backends import LocalSQLiteBackend, ShopifyStorefrontBackend
from store.mcp_server_v2 import ServerState, run_check

try:  # tests/ has no __init__.py in this repo; both roots work
    from tests.shopify_fake import (FAKE_CART_ID, FAKE_VARIANT_GID,
                                    FakeShopify)
except ImportError:  # pragma: no cover
    from shopify_fake import (FAKE_CART_ID, FAKE_VARIANT_GID, FakeShopify)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "shopify"
NAVY = "BTL-HIK-500-NAVY"


def _task(tmp_path, items, request_text, seed_lines):
    path = tmp_path / "mcp-task-v2.json"
    path.write_text(json.dumps({
        "runtime": {"request_text": request_text, "seed_lines": seed_lines},
        "expected_interpretation": {"items": items,
                                    "budget_max_cents": None,
                                    "substitution_allowed": False},
    }), encoding="utf-8")
    return path


def _seed(store):
    conn = models.connect(store["db"])
    try:
        cart_id, sid, _ = models.get_or_create_session(conn, None)
        models.add_line_guarded(conn, cart_id, 0, NAVY, 1, 2400)
    finally:
        conn.close()
    return sid


def _local_backend(store):
    return LocalSQLiteBackend(
        store["base"], store["db"],
        str(PROJECT_ROOT / "store" / "seed_catalog.json"), _seed(store))


def _shopify_backend(fake):
    return ShopifyStorefrontBackend(
        "cartproof.myshopify.com", str(FIXTURES / "mapping.json"),
        transport=fake.transport)


# ---------------------------------------------------------------- local
def test_check_local_complete(store, tmp_path):
    backend = _local_backend(store)
    task = _task(tmp_path, [{"sku": NAVY, "quantity": 1}],
                 "Add 1 x BTL-HIK-500-NAVY.",
                 [{"sku": NAVY, "quantity": 1}])
    state = ServerState(backend, str(task), str(tmp_path / "log"))
    version = backend.acquire_state()["version"]
    added = backend.add_line(NAVY, 1, expected_version=version)
    assert added["ok"] is True
    result = run_check(state)
    assert result["verdict"] == "COMPLETE", result
    assert result["capabilities"]["atomic_version_guard"] is True
    assert result["baseline_provenance"]["source"] == "sqlite-snapshot"
    assert result["check_provenance"]["source"] == "sqlite-snapshot"
    assert result["seed_lines"][0]["survived_unchanged"] is True


def test_check_local_partial_when_request_not_fulfilled(store, tmp_path):
    backend = _local_backend(store)
    task = _task(tmp_path, [{"sku": NAVY, "quantity": 1}],
                 "Add 1 x BTL-HIK-500-NAVY.",
                 [{"sku": NAVY, "quantity": 1}])
    state = ServerState(backend, str(task), str(tmp_path / "log"))
    result = run_check(state)
    assert result["verdict"] == "PARTIAL"
    assert result["mismatches"] == [{"sku": NAVY, "expected_quantity": 2,
                                     "actual_quantity": 1}]


def test_check_local_seed_delete_detected(store, tmp_path):
    backend = _local_backend(store)
    task = _task(tmp_path, [{"sku": NAVY, "quantity": 1}],
                 "Add 1 x BTL-HIK-500-NAVY.",
                 [{"sku": NAVY, "quantity": 1}])
    state = ServerState(backend, str(task), str(tmp_path / "log"))
    now = backend.acquire_state()
    line_id = now["line_ids"][0]["line_id"]
    deleted = backend.delete_line(line_id, expected_version=now["version"])
    assert deleted["ok"] is True
    result = run_check(state)
    assert result["verdict"] == "PARTIAL"
    assert result["seed_lines"][0]["present"] is False
    assert result["seed_lines"][0]["survived_unchanged"] is False


# -------------------------------------------------------------- shopify
def test_check_shopify_complete_with_provenance(tmp_path):
    fake = FakeShopify()
    backend = _shopify_backend(fake)
    backend.add_line(NAVY, 1, expected_version=None)  # the "seeded" line
    task = _task(tmp_path, [{"sku": NAVY, "quantity": 1}],
                 "Add 1 x BTL-HIK-500-NAVY.",
                 [{"sku": NAVY, "quantity": 1}])
    state = ServerState(backend, str(task), str(tmp_path / "log"))
    backend.add_line(NAVY, 1, expected_version=None)  # the agent's action
    result = run_check(state)
    assert result["verdict"] == "COMPLETE", result
    assert result["capabilities"] == {"atomic_version_guard": False,
                                      "cart_identity": "remote-cart-id"}
    assert result["baseline_provenance"]["source"] \
        == "shopify-storefront-api"
    assert result["check_provenance"]["source"] == "shopify-storefront-api"
    assert result["baseline_provenance"]["cart_ref"] == backend.cart_ref
    assert "no concurrency guard" in result["evidence_limit"]
    assert "aggregate" in result["seed_identity"]
    assert "fakekeycapability" not in json.dumps(result)


def test_check_shopify_seed_delete_detected(tmp_path):
    fake = FakeShopify()
    backend = _shopify_backend(fake)
    backend.add_line(NAVY, 1, expected_version=None)  # the "seeded" line
    task = _task(tmp_path, [], "No additions expected.",
                 [{"sku": NAVY, "quantity": 1}])
    state = ServerState(backend, str(task), str(tmp_path / "log"))
    fake.lines.clear()  # an agent deleted the seeded line behind our back
    result = run_check(state)
    assert result["verdict"] == "PARTIAL"
    assert result["seed_lines"][0]["present"] is False
    assert result["seed_lines"][0]["survived_unchanged"] is False


def test_check_shopify_duplicate_lines_no_false_partial(tmp_path):
    """A variant split across two cart lines must not false-PARTIAL the
    aggregate seed evidence (Shopify line-ID stability is not guaranteed).
    """
    fake = FakeShopify()
    fake.lines["gid://shopify/CartLine/1"] = {
        "merchandise": FAKE_VARIANT_GID, "quantity": 1}
    fake.lines["gid://shopify/CartLine/2"] = {
        "merchandise": FAKE_VARIANT_GID, "quantity": 1}
    backend = _shopify_backend(fake)
    task = _task(tmp_path, [], "No additions expected.",
                 [{"sku": NAVY, "quantity": 2}])
    state = ServerState(backend, str(task), str(tmp_path / "log"))
    result = run_check(state)
    assert result["baseline_lines"] == {NAVY: 2}
    assert result["verdict"] == "COMPLETE", result
    assert result["seed_lines"][0]["survived_unchanged"] is True


def test_check_shopify_degraded_shapes_unknown_not_guessed(tmp_path):
    # Baseline is frozen from a HEALTHY fake; acquisitions then degrade.
    for mode in ("truncated", "string_quantity", "http_500"):
        task = _task(tmp_path, [], "No additions expected.", [])
        state = ServerState(_shopify_backend(FakeShopify()), str(task),
                            str(tmp_path / "log"))
        state.backend = _shopify_backend(FakeShopify(mode=mode))
        result = run_check(state)
        assert result["verdict"] == "UNKNOWN", mode
        assert result["final_lines"] is None, mode


def test_check_baseline_unavailable_fails_closed(tmp_path):
    fake = FakeShopify(mode="http_500")
    backend = _shopify_backend(fake)
    task = _task(tmp_path, [], "No additions expected.", [])
    with pytest.raises(RuntimeError, match="mcp_baseline_unavailable"):
        ServerState(backend, str(task), str(tmp_path / "log"))


def test_check_cart_identity_change_unknown(tmp_path):
    """Baseline and check against DIFFERENT remote carts: no comparison is
    meaningful, so the verdict is UNKNOWN, never COMPLETE/PARTIAL."""
    task = _task(tmp_path, [], "No additions expected.", [])
    state = ServerState(_shopify_backend(FakeShopify()), str(task),
                        str(tmp_path / "log"))
    other = FakeShopify(cart_id="gid://shopify/Cart/b3RoZXI"
                                "?key=othercapability999999")
    state.backend = _shopify_backend(other)
    result = run_check(state)
    assert result["verdict"] == "UNKNOWN"
    assert result["resolution"] == "cart_identity_changed"


def test_check_wrong_cart_response_unknown(tmp_path):
    """A check whose fetch returns a DIFFERENT cart than the backend
    created is UNKNOWN (resolution wrong_cart), never a comparison."""
    task = _task(tmp_path, [], "No additions expected.", [])
    state = ServerState(_shopify_backend(FakeShopify()), str(task),
                        str(tmp_path / "log"))
    state.backend = _shopify_backend(FakeShopify(mode="wrong_cart"))
    result = run_check(state)
    assert result["verdict"] == "UNKNOWN"
    assert result["resolution"] == "wrong_cart"


def test_log_redaction_strips_capability_from_logged_args(tmp_path):
    """Tool arguments are caller-controlled strings; the evidence log must
    strip capability material even when a caller pastes it into an arg."""
    fake = FakeShopify()
    backend = _shopify_backend(fake)
    task = _task(tmp_path, [], "No additions expected.", [])
    state = ServerState(backend, str(task), str(tmp_path / "log"))
    state._log({"event": "tool_call", "tool": "cart_delete_line",
                "args": {"line_id": FAKE_CART_ID},
                "result": {"ok": False}})
    for raw in state.log_path.read_text(encoding="utf-8").splitlines():
        assert FAKE_CART_ID not in raw
        assert "fakekeycapability000000" not in raw
    blob = state.log_path.read_text(encoding="utf-8")
    assert "cartref:" in blob  # correlation survives, capability does not

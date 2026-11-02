"""Offline tests for the P40 cart-state backends.

Local backend: loopback mock store only (same `store` fixture family as
tests/test_mcp_server.py). Shopify backend: the STATEFUL in-memory fake
in tests/shopify_fake.py (so lifecycle tests verify STATE correctness,
not just call success) plus the official doc-example fixtures - all
labeled simulation/fixture, NOT live proof.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from store import models
from store.backends import LocalSQLiteBackend, ShopifyStorefrontBackend

try:  # tests/ has no __init__.py in this repo; both roots work
    from tests.shopify_fake import (FAKE_CART_ID, FAKE_VARIANT_GID,
                                    FakeShopify)
except ImportError:  # pragma: no cover
    from shopify_fake import (FAKE_CART_ID, FAKE_VARIANT_GID, FakeShopify)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "shopify"
NAVY = "BTL-HIK-500-NAVY"
FAKE_KEY = "fakekeycapability000000"


def _seed(store):
    conn = models.connect(store["db"])
    try:
        cart_id, sid, _ = models.get_or_create_session(conn, None)
        models.add_line_guarded(conn, cart_id, 0, NAVY, 1, 2400)
    finally:
        conn.close()
    return sid


def _local(store):
    return LocalSQLiteBackend(
        store["base"], store["db"],
        str(PROJECT_ROOT / "store" / "seed_catalog.json"), _seed(store))


def _shopify(fake=None):
    fake = fake or FakeShopify()
    backend = ShopifyStorefrontBackend(
        "cartproof.myshopify.com", str(FIXTURES / "mapping.json"),
        transport=fake.transport)
    return backend, fake


# ---------------------------------------------------------------- local
def test_local_backend_acquire_state_matches_snapshot(store):
    backend = _local(store)
    state = backend.acquire_state()
    assert state["lines"] == {NAVY: 1}
    assert state["version"] == 1
    assert state["resolution"] == "cookie"
    assert state["provenance"]["source"] == "sqlite-snapshot"
    assert state["line_ids"][0]["item_key"] == NAVY
    assert backend.capabilities() == {"atomic_version_guard": True,
                                      "cart_identity": "session"}


def test_local_backend_native_guard_enforced(store):
    backend = _local(store)
    stale = backend.add_line(NAVY, 1, expected_version=999)
    assert stale["ok"] is False
    assert stale["payload"]["error"]["code"] == "version_conflict"
    assert stale["version_enforcement"] == "enforced"
    assert backend.acquire_state()["lines"] == {NAVY: 1}
    current = backend.acquire_state()["version"]
    good = backend.add_line(NAVY, 1, expected_version=current)
    assert good["ok"] is True
    assert backend.acquire_state()["lines"] == {NAVY: 2}


def test_local_backend_requires_expected_version(store):
    backend = _local(store)
    refused = backend.add_line(NAVY, 1, expected_version=None)
    assert refused["ok"] is False
    assert refused["payload"]["error"]["code"] == "expected_version_required"
    assert refused["version_enforcement"] == "required"
    assert backend.acquire_state()["lines"] == {NAVY: 1}


# -------------------------------------------------------------- shopify
def test_shopify_construction_guard_requires_transport():
    with pytest.raises(TypeError):
        ShopifyStorefrontBackend("cartproof.myshopify.com",
                                 str(FIXTURES / "mapping.json"),
                                 transport=None)


def test_shopify_stateful_lifecycle_state_correctness():
    backend, fake = _shopify()
    added = backend.add_line(NAVY, 2, expected_version=7)
    assert added["ok"] is True
    assert added["version_enforcement"] == "unavailable"
    assert added["expected_version_received"] == 7
    # STATE correctness, not call success: the fake applied the mutation.
    assert backend.acquire_state()["lines"] == {NAVY: 2}
    line_id = next(iter(fake.lines))
    updated = backend.set_line_quantity(line_id, 5, expected_version=None)
    assert updated["ok"] is True
    assert backend.acquire_state()["lines"] == {NAVY: 5}
    removed = backend.delete_line(line_id, expected_version=None)
    assert removed["ok"] is True
    assert backend.acquire_state()["lines"] == {}
    # the first mutation carried a REAL cart ID, never null
    first_mutation_vars = fake.requests[1]["variables"]
    assert first_mutation_vars["cartId"] == FAKE_CART_ID


def test_shopify_redaction_of_cart_capability():
    backend, fake = _shopify()
    backend.add_line(NAVY, 1, expected_version=None)
    state = backend.acquire_state()
    blobs = [json.dumps(state),
             json.dumps(backend.add_line(NAVY, 1, None)["payload"])]
    for blob in blobs:
        assert FAKE_KEY not in blob
        assert FAKE_CART_ID not in blob
    assert backend.cart_ref == hashlib.sha256(
        FAKE_CART_ID.encode("utf-8")).hexdigest()[:12]
    assert state["provenance"]["cart_ref"] == backend.cart_ref
    assert state["provenance"]["source"] == "shopify-storefront-api"
    assert state["provenance"]["api_version"] == "2026-07"


def test_shopify_user_errors_fail_closed():
    backend, _ = _shopify(FakeShopify(valid_variant_gids=[]))
    result = backend.add_line(NAVY, 1, expected_version=None)
    assert result["ok"] is False
    assert result["version_enforcement"] == "unavailable"
    blob = json.dumps(result["payload"])
    assert FAKE_KEY not in blob  # userErrors echoes are redacted too


def test_shopify_degraded_shapes_never_guess_state():
    for mode, resolution in [("truncated", "truncated"),
                             ("missing_lines", "malformed"),
                             ("missing_page_info", "malformed"),
                             ("string_quantity", "malformed"),
                             ("null_cart", "cart_not_found"),
                             ("http_500", "unavailable"),
                             ("json_list", "unavailable")]:
        backend, _ = _shopify(FakeShopify(mode=mode))
        state = backend.acquire_state()
        assert state["lines"] is None, mode
        assert state["resolution"] == resolution, mode
        assert state["provenance"]["source"] == "shopify-storefront-api"


def test_canonicalizer_sums_duplicate_merchandise_lines():
    dup_cart = {
        "id": FAKE_CART_ID,
        "lines": {"edges": [
            {"node": {"id": "gid://shopify/CartLine/1", "quantity": 1,
                      "merchandise": {"id": FAKE_VARIANT_GID}}},
            {"node": {"id": "gid://shopify/CartLine/2", "quantity": 2,
                      "merchandise": {"id": FAKE_VARIANT_GID}}},
        ], "pageInfo": {"hasNextPage": False}},
    }
    state = ShopifyStorefrontBackend._canonicalize(dup_cart)
    assert state["lines"] == {FAKE_VARIANT_GID: 3}
    assert len(state["line_ids"]) == 2
    assert state["resolution"] == "remote-cart"


def test_shopify_transport_type_enforced():
    import httpx
    with pytest.raises(TypeError):
        ShopifyStorefrontBackend("cartproof.myshopify.com",
                                 str(FIXTURES / "mapping.json"),
                                 transport=httpx.HTTPTransport())


def test_shopify_user_error_echo_redacted():
    """userErrors can echo input values; capability material in them is
    redacted from the returned payload."""
    backend, _ = _shopify(FakeShopify(mode="user_error_echo"))
    result = backend.add_line(NAVY, 1, expected_version=None)
    assert result["ok"] is False
    blob = json.dumps(result["payload"])
    assert FAKE_KEY not in blob
    assert FAKE_CART_ID not in blob
    assert "cartref:" in blob


def test_shopify_wrong_cart_response_not_trusted():
    """A response for a DIFFERENT cart than the one created is never
    canonicalized: provenance must not hash the requested ID over a
    foreign response."""
    backend, _ = _shopify(FakeShopify(mode="wrong_cart"))
    state = backend.acquire_state()
    assert state["lines"] is None
    assert state["resolution"] == "wrong_cart"


def test_unknown_sku_echo_is_redacted():
    """Even the early unknown_sku error path strips capability material
    from the echoed value."""
    backend, _ = _shopify()
    backend.add_line(NAVY, 1, expected_version=None)  # establishes the cart
    result = backend.add_line(FAKE_CART_ID, 1, expected_version=None)
    assert result["ok"] is False
    assert result["payload"]["error"]["code"] == "unknown_sku"
    blob = json.dumps(result["payload"])
    assert FAKE_KEY not in blob
    assert FAKE_CART_ID not in blob


def test_canonicalize_requires_explicit_complete_page_info():
    base = {"id": FAKE_CART_ID,
            "lines": {"edges": [], "pageInfo": {"hasNextPage": False}}}
    ok = ShopifyStorefrontBackend._canonicalize(base)
    assert ok["resolution"] == "remote-cart"
    missing = {"id": FAKE_CART_ID, "lines": {"edges": []}}
    assert ShopifyStorefrontBackend._canonicalize(missing)["resolution"] \
        == "malformed"
    non_bool = {"id": FAKE_CART_ID,
                "lines": {"edges": [], "pageInfo": {"hasNextPage": "no"}}}
    assert ShopifyStorefrontBackend._canonicalize(non_bool)["resolution"] \
        == "malformed"


def test_canonicalize_rejects_bad_quantity_and_line_id():
    def cart(quantity=None, line_id="gid://shopify/CartLine/1"):
        node = {"merchandise": {"id": FAKE_VARIANT_GID}}
        if quantity is not None:
            node["quantity"] = quantity
        if line_id is not None:
            node["id"] = line_id
        return {"id": FAKE_CART_ID,
                "lines": {"edges": [{"node": node}],
                          "pageInfo": {"hasNextPage": False}}}
    for bad in (cart(quantity=0), cart(quantity=-1),
                cart(quantity="1"), cart(quantity=True),
                cart(), cart(line_id=None), cart(line_id="")):
        assert ShopifyStorefrontBackend._canonicalize(bad)["resolution"] \
            == "malformed", bad


def test_official_doc_example_shape_parses():
    """The official shopify.dev cart-query example (fixture, not live)
    canonicalizes to the documented line."""
    doc = json.loads((FIXTURES / "cart_query.json").read_text(
        encoding="utf-8"))
    state = ShopifyStorefrontBackend._canonicalize(doc["data"]["cart"])
    assert state["lines"] == {FAKE_VARIANT_GID: 1}
    assert state["resolution"] == "remote-cart"

"""Concrete cart-state backends (P40). Seam: store/cart_backend.py.

LocalSQLiteBackend: the reviewed mock store. Mutations go through its own
HTTP API with the native X-Cart-Version guard (version_enforcement
"enforced"; the guard is REQUIRED - a mutation without expected_version
fails closed with "expected_version_required"). State acquisition is a
consistent SQLite snapshot read, exactly as harness/snapshots.py read_cart
does for capture_final_db.

ShopifyStorefrontBackend: the CartProof dev store over the tokenless
Storefront API. Honest limits, by design:
- Shopify carts expose no version, ETag or concurrency token, so none is
  invented: version is None and every mutation result carries
  version_enforcement "unavailable".
- Cart reads request lines(first: 100) with pageInfo. A cart with more
  than 100 lines (hasNextPage true) is reported resolution "truncated"
  with lines None - never silently partial. Malformed responses (missing
  lines object, non-integer quantity, missing merchandise) are reported
  resolution "malformed" with lines None. Both make cartproof_check return
  UNKNOWN instead of a false COMPLETE/PARTIAL.
- The cart ID's ?key= parameter is an ephemeral capability. The full cart
  ID AND the key value on its own are redacted from every returned payload
  (including userErrors and error echoes); only a truncated sha256
  cart_ref is exposed for correlation.
- OFFLINE CONSTRUCTION GUARD: the httpx transport is a REQUIRED argument.
  There is no default, so this backend cannot make a live network call
  unless the caller deliberately supplies a transport. In this brief only
  tests supply one (httpx.MockTransport).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from harness.snapshots import backup_copy, read_cart
from store.cart_backend import (
    CanonicalCartState,
    Capabilities,
    MutationResult,
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _transport_error() -> dict[str, Any]:
    return {"error": {"code": "backend_transport",
                      "message": "internal backend error"}}


class LocalSQLiteBackend:
    """The reviewed mock store behind the seam."""

    def __init__(self, base_url: str, db_path: str, catalog_path: str,
                 session_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.db_path = db_path
        self.client = httpx.Client(base_url=self.base_url, timeout=15.0,
                                   follow_redirects=False, trust_env=False)
        self.cookie = session_id
        headers = {"Cookie": f"cp_session={session_id}"} if session_id else {}
        response = self.client.get("/api/cart", headers=headers)
        response.raise_for_status()
        if not self.cookie:
            self.cookie = response.cookies.get("cp_session")
        if not self.cookie:
            raise RuntimeError("local_session_unavailable")
        self.client.cookies.clear()
        self.catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))

    # ----------------------------------------------------------- #
    def capabilities(self) -> Capabilities:
        return {"atomic_version_guard": True, "cart_identity": "session"}

    def known_item_keys(self) -> set[str]:
        skus: set[str] = set()
        for product in self.catalog.get("products", []):
            for variant in product.get("variants", []):
                skus.add(variant["sku"])
        return skus

    def _line_ids(self, db_file: str) -> list[dict[str, Any]]:
        conn = sqlite3.connect(db_file)
        try:
            row = conn.execute(
                "SELECT cart_id FROM sessions WHERE session_id = ?",
                (self.cookie,)).fetchone()
            if row is None:
                return []
            rows = conn.execute(
                "SELECT line_id, sku, quantity FROM cart_lines "
                "WHERE cart_id = ? ORDER BY line_id", (row[0],)).fetchall()
            return [{"line_id": r[0], "item_key": r[1], "quantity": r[2]}
                    for r in rows]
        finally:
            conn.close()

    def acquire_state(self) -> CanonicalCartState:
        import os
        fd, copy_path = tempfile.mkstemp(prefix="cp40-local-", suffix=".db")
        try:
            os.close(fd)
            backup_copy(self.db_path, copy_path)
            cart = read_cart(copy_path, self.cookie)
            ids = self._line_ids(copy_path)
        finally:
            try:
                Path(copy_path).unlink()
            except OSError:
                pass
        return {
            "lines": cart["lines"],
            "line_ids": ids,
            "version": cart["version"],
            "resolution": cart["resolution"],
            "provenance": {"source": "sqlite-snapshot",
                           "db_copy": Path(copy_path).name,
                           "fetched_at": _now()},
        }

    def redact(self, obj: Any) -> Any:
        """No bearer capabilities exist on the mock store path; the
        cp_session cookie value is never placed in payloads or logs."""
        return obj

    def read_catalog(self) -> dict[str, Any]:
        return self.catalog

    def _mutate(self, method: str, path: str, expected_version: Optional[int],
                **kwargs) -> MutationResult:
        if expected_version is None:
            # The mock store's native guard is required; fail closed rather
            # than send an unguarded write.
            return {"ok": False,
                    "payload": {"error": {"code": "expected_version_required",
                                          "message": "this backend enforces "
                                                     "the native version "
                                                     "guard"}},
                    "version_enforcement": "required",
                    "expected_version_received": None}
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Cookie"] = f"cp_session={self.cookie}"
        headers["X-Cart-Version"] = str(expected_version)
        try:
            response = self.client.request(method, path, headers=headers,
                                           **kwargs)
            payload = response.json()
        except Exception:
            return {"ok": False, "payload": _transport_error(),
                    "version_enforcement": "enforced",
                    "expected_version_received": expected_version}
        if response.status_code >= 400:
            if not (isinstance(payload, dict)
                    and isinstance(payload.get("error"), dict)):
                payload = _transport_error()
            return {"ok": False, "payload": payload,
                    "version_enforcement": "enforced",
                    "expected_version_received": expected_version}
        return {"ok": True, "payload": payload,
                "version_enforcement": "enforced",
                "expected_version_received": expected_version}

    def add_line(self, sku: str, quantity: int,
                 expected_version: Optional[int]) -> MutationResult:
        return self._mutate("POST", "/api/cart/lines", expected_version,
                            json={"sku": sku, "quantity": quantity})

    def set_line_quantity(self, line_id: str, quantity: int,
                          expected_version: Optional[int]) -> MutationResult:
        return self._mutate("PATCH", f"/api/cart/lines/{line_id}",
                            expected_version, json={"quantity": quantity})

    def delete_line(self, line_id: str,
                    expected_version: Optional[int]) -> MutationResult:
        return self._mutate("DELETE", f"/api/cart/lines/{line_id}",
                            expected_version)


class ShopifyBackendError(Exception):
    """Raised for transport/HTTP/GraphQL-level failures. Carries only a
    short code - never the cart ID or any response body."""


_CART_FIELDS = """
    id
    lines(first: 100) {
      edges {
        node {
          id
          quantity
          merchandise {
            ... on ProductVariant {
              id
            }
          }
        }
      }
      pageInfo {
        hasNextPage
      }
    }
"""


class ShopifyStorefrontBackend:
    """The CartProof dev store over the tokenless Storefront API.

    ``transport`` is REQUIRED (no default): the offline construction guard.
    Tests pass httpx.MockTransport; a live caller must deliberately build
    and pass a real transport, which this brief does not do.
    """

    def __init__(self, shop: str, mapping_path: str,
                 transport: httpx.BaseTransport,
                 api_version: str = "2026-07"):
        if not isinstance(transport, httpx.MockTransport):
            # OFFLINE CONSTRUCTION GUARD: this build cannot speak live HTTP
            # at all. Relaxing this check belongs to a future Stage 1
            # live smoke that would require separate approval; none is
            # granted.
            raise TypeError("offline build: transport must be an "
                            "httpx.MockTransport; no live HTTP client can "
                            "be constructed")
        self.shop = shop
        self.api_version = api_version
        self.client = httpx.Client(
            base_url=f"https://{shop}/api/{api_version}/",
            transport=transport, timeout=15.0,
            follow_redirects=False, trust_env=False)
        mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        self._sku_to_gid = dict(mapping.get("variants", {}))
        self._gid_to_sku = {gid: sku for sku, gid in self._sku_to_gid.items()}
        self._cart_id: Optional[str] = None
        self._cart_key: Optional[str] = None
        self._cart_ref: Optional[str] = None

    # ----------------------------------------------------------- #
    def capabilities(self) -> Capabilities:
        return {"atomic_version_guard": False,
                "cart_identity": "remote-cart-id"}

    def known_item_keys(self) -> set[str]:
        return set(self._sku_to_gid)

    @property
    def cart_ref(self) -> Optional[str]:
        return self._cart_ref

    def _set_cart_id(self, cart_id: str) -> None:
        self._cart_id = cart_id
        if "?key=" in cart_id:
            self._cart_key = cart_id.split("?key=", 1)[1]
        self._cart_ref = hashlib.sha256(cart_id.encode("utf-8")).hexdigest()[:12]

    def redact(self, obj: Any) -> Any:
        """Remove the cart capability (full ID and bare ?key= value)
        from anything before it is returned or logged."""
        if isinstance(obj, str):
            if self._cart_id is not None:
                obj = obj.replace(self._cart_id, f"cartref:{self._cart_ref}")
            if self._cart_key:
                obj = obj.replace(self._cart_key, "<cart-key-redacted>")
            return obj
        if isinstance(obj, list):
            return [self.redact(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self.redact(v) for k, v in obj.items()}
        return obj

    def _gql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.post(
                "graphql.json",
                json={"query": query, "variables": variables})
        except Exception as exc:
            raise ShopifyBackendError(
                f"transport:{type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise ShopifyBackendError(f"http:{response.status_code}")
        try:
            body = response.json()
        except Exception:
            raise ShopifyBackendError("invalid_json")
        if not isinstance(body, dict):
            raise ShopifyBackendError("invalid_json")
        if body.get("errors"):
            raise ShopifyBackendError("graphql_errors")
        data = body.get("data")
        if not isinstance(data, dict):
            raise ShopifyBackendError("no_data")
        return data

    def ensure_cart(self) -> None:
        if self._cart_id is not None:
            return
        data = self._gql(
            "mutation { cartCreate(input: {lines: []}) { cart { id } "
            "userErrors { field message } } }", {})
        payload = data.get("cartCreate") or {}
        if payload.get("userErrors"):
            raise ShopifyBackendError("cart_create_user_errors")
        cart = payload.get("cart") or {}
        cart_id = cart.get("id")
        if not isinstance(cart_id, str) or not cart_id:
            raise ShopifyBackendError("cart_create_no_id")
        self._set_cart_id(cart_id)

    @staticmethod
    def _canonicalize(cart: Any) -> CanonicalCartState:
        """Cart object -> canonical state. Truncation or a malformed shape
        yields lines None (the check must go UNKNOWN, never guess)."""
        if not isinstance(cart, dict):
            return {"lines": None, "line_ids": [], "version": None,
                    "resolution": "malformed", "provenance": {}}
        lines_obj = cart.get("lines")
        if not isinstance(lines_obj, dict):
            return {"lines": None, "line_ids": [], "version": None,
                    "resolution": "malformed", "provenance": {}}
        page_info = lines_obj.get("pageInfo")
        if not isinstance(page_info, dict) \
                or not isinstance(page_info.get("hasNextPage"), bool):
            # pageInfo was requested; a missing/invalid one means the
            # shape is not what we asked for - never assume completeness.
            return {"lines": None, "line_ids": [], "version": None,
                    "resolution": "malformed", "provenance": {}}
        if page_info["hasNextPage"]:
            return {"lines": None, "line_ids": [], "version": None,
                    "resolution": "truncated", "provenance": {}}
        edges = lines_obj.get("edges")
        if not isinstance(edges, list):
            return {"lines": None, "line_ids": [], "version": None,
                    "resolution": "malformed", "provenance": {}}
        lines: dict[str, int] = {}
        ids: list[dict[str, Any]] = []
        for edge in edges:
            node = (edge or {}).get("node")
            if not isinstance(node, dict):
                return {"lines": None, "line_ids": [], "version": None,
                        "resolution": "malformed", "provenance": {}}
            quantity = node.get("quantity")
            if isinstance(quantity, bool) or not isinstance(quantity, int) \
                    or quantity < 1:
                return {"lines": None, "line_ids": [], "version": None,
                        "resolution": "malformed", "provenance": {}}
            line_id = node.get("id")
            if not isinstance(line_id, str) or not line_id:
                return {"lines": None, "line_ids": [], "version": None,
                        "resolution": "malformed", "provenance": {}}
            merchandise = node.get("merchandise")
            gid = merchandise.get("id") if isinstance(merchandise, dict) \
                else None
            if not isinstance(gid, str) or not gid:
                return {"lines": None, "line_ids": [], "version": None,
                        "resolution": "malformed", "provenance": {}}
            item_key = gid  # mapped to SKU by the caller
            lines[item_key] = lines.get(item_key, 0) + quantity
            ids.append({"line_id": line_id,
                        "item_key": item_key, "quantity": quantity})
        return {"lines": lines, "line_ids": ids, "version": None,
                "resolution": "remote-cart", "provenance": {}}

    def acquire_state(self) -> CanonicalCartState:
        try:
            self.ensure_cart()
            data = self._gql(
                "query ($id: ID!) { cart(id: $id) { " + _CART_FIELDS
                + " } }",
                {"id": self._cart_id})
        except ShopifyBackendError:
            return {"lines": None, "line_ids": [], "version": None,
                    "resolution": "unavailable", "provenance": self._prov()}
        cart = data.get("cart")
        if cart is None:
            return {"lines": None, "line_ids": [], "version": None,
                    "resolution": "cart_not_found",
                    "provenance": self._prov()}
        returned_id = cart.get("id") if isinstance(cart, dict) else None
        if returned_id != self._cart_id:
            # The response is for a missing or DIFFERENT cart than the one
            # this backend created: provenance must not hash the requested
            # ID over a foreign response. UNKNOWN, never a comparison.
            return {"lines": None, "line_ids": [], "version": None,
                    "resolution": "wrong_cart",
                    "provenance": self._prov()}
        state = self._canonicalize(cart)
        if state["lines"] is not None:
            mapped: dict[str, int] = {}
            for gid, qty in state["lines"].items():
                key = self._gid_to_sku.get(gid, gid)
                mapped[key] = mapped.get(key, 0) + qty
            state["lines"] = mapped
            for entry in state["line_ids"]:
                entry["item_key"] = self._gid_to_sku.get(
                    entry["item_key"], entry["item_key"])
        state["provenance"] = self._prov()
        return state

    def _prov(self) -> dict[str, Any]:
        return {"source": "shopify-storefront-api",
                "api_version": self.api_version,
                "fetched_at": _now(),
                "cart_ref": self._cart_ref}

    def read_catalog(self) -> dict[str, Any]:
        data = self._gql(
            "query { products(first: 20) { edges { node { id title "
            "variants(first: 20) { edges { node { id sku price { amount "
            "currencyCode } } } } } } } }", {})
        return self.redact(data)

    def _cart_mutation(self, name: str, query: str,
                       variables_factory: Callable[[str], dict[str, Any]],
                       expected_version: Optional[int]) -> MutationResult:
        try:
            self.ensure_cart()
            # variables are built AFTER ensure_cart: the cart ID never
            # leaves this method as None on the first mutation.
            data = self._gql(query, variables_factory(self._cart_id))
        except ShopifyBackendError as exc:
            return {"ok": False,
                    "payload": {"error": {"code": "shopify_backend",
                                          "message": str(exc)}},
                    "version_enforcement": "unavailable",
                    "expected_version_received": expected_version}
        payload = data.get(name) or {}
        user_errors = payload.get("userErrors") or []
        ok = not user_errors and payload.get("cart") is not None
        return {"ok": ok,
                "payload": self.redact({name: payload}),
                "version_enforcement": "unavailable",
                "expected_version_received": expected_version}

    def add_line(self, sku: str, quantity: int,
                 expected_version: Optional[int]) -> MutationResult:
        gid = self._sku_to_gid.get(sku)
        if gid is None:
            return {"ok": False,
                    "payload": self.redact({"error": {
                        "code": "unknown_sku", "message": sku}}),
                    "version_enforcement": "unavailable",
                    "expected_version_received": expected_version}
        return self._cart_mutation(
            "cartLinesAdd",
            "mutation ($cartId: ID!, $lines: [CartLineInput!]!) { "
            "cartLinesAdd(cartId: $cartId, lines: $lines) { "
            "cart { " + _CART_FIELDS + " } userErrors { field message } } }",
            lambda cart_id: {"cartId": cart_id,
                             "lines": [{"merchandiseId": gid,
                                        "quantity": quantity}]},
            expected_version)

    def set_line_quantity(self, line_id: str, quantity: int,
                          expected_version: Optional[int]) -> MutationResult:
        return self._cart_mutation(
            "cartLinesUpdate",
            "mutation ($cartId: ID!, $lines: [CartLineUpdateInput!]!) { "
            "cartLinesUpdate(cartId: $cartId, lines: $lines) { "
            "cart { " + _CART_FIELDS + " } userErrors { field message } } }",
            lambda cart_id: {"cartId": cart_id,
                             "lines": [{"id": line_id,
                                        "quantity": quantity}]},
            expected_version)

    def delete_line(self, line_id: str,
                    expected_version: Optional[int]) -> MutationResult:
        return self._cart_mutation(
            "cartLinesRemove",
            "mutation ($cartId: ID!, $lineIds: [ID!]!) { "
            "cartLinesRemove(cartId: $cartId, lineIds: $lineIds) { "
            "cart { " + _CART_FIELDS + " } userErrors { field message } } }",
            lambda cart_id: {"cartId": cart_id, "lineIds": [line_id]},
            expected_version)

"""Stateful in-memory fake of Shopify's tokenless Storefront API cart surface.

LABELED SIMULATION - NOT LIVE PROOF. The fake exists so offline tests can
verify STATE correctness (not just call success) of the P40 Shopify
backend. Its behaviors are explicit assumptions:
- cartLinesAdd on an EXISTING merchandise line INCREMENTS that line's
  quantity (merge behavior is undocumented - P39 remote unknown #2; the
  Stage 1 live smoke pins the real behavior). The canonicalizer is
  separately tested against duplicate-line carts, so the code is correct
  under EITHER real behavior.
- cartCreate issues a cart ID with a ?key= capability, like the real API.
- Line IDs are arbitrary fake GIDs; nothing about their stability is
  claimed.
Failure modes (mode=...) exist to prove the backend degrades to UNKNOWN,
never to a guessed verdict.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import httpx

FAKE_CART_ID = "gid://shopify/Cart/ZmFrZS1jYXJ0?key=fakekeycapability000000"
FAKE_VARIANT_GID = "gid://shopify/ProductVariant/1"


class FakeShopify:
    def __init__(self, mode: Optional[str] = None,
                 valid_variant_gids: Optional[list[str]] = None,
                 cart_id: str = FAKE_CART_ID):
        self.mode = mode  # None | "truncated" | "missing_lines"
                          # | "missing_page_info" | "string_quantity"
                          # | "null_cart" | "http_500" | "json_list"
                          # | "user_error_echo" | "wrong_cart"
        self.cart_id = cart_id
        self.valid_variant_gids = set(
            [FAKE_VARIANT_GID] if valid_variant_gids is None
            else valid_variant_gids)
        self.lines: dict[str, dict[str, Any]] = {}
        self._next_line = 0
        self.requests: list[dict[str, Any]] = []
        self.transport = httpx.MockTransport(self._handler)

    # ----------------------------------------------------------- #
    def _cart_payload(self) -> dict[str, Any]:
        return {
            "id": self.cart_id,
            "lines": {
                "edges": [
                    {"node": {"id": line_id,
                              "quantity": line["quantity"],
                              "merchandise": {"id": line["merchandise"]}}}
                    for line_id, line in self.lines.items()
                ],
                "pageInfo": {"hasNextPage": False},
            },
        }

    def _wrap(self, name: str, payload: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json={"data": {name: payload}})

    def _handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        query = body.get("query", "")
        variables = body.get("variables") or {}
        self.requests.append({"query": query, "variables": variables})

        if self.mode == "http_500":
            return httpx.Response(500, json={"errors": "boom"})

        if "cartCreate" in query:
            return self._wrap("cartCreate", {
                "cart": {"id": self.cart_id}, "userErrors": [],
                "warnings": []})

        if "cartLinesAdd" in query:
            if self.mode == "user_error_echo":
                # userErrors can ECHO input values; the backend must
                # redact capability material from them too.
                return self._wrap("cartLinesAdd", {
                    "cart": None,
                    "userErrors": [{
                        "field": ["cartId"],
                        "message": f"cart {self.cart_id} rejected"}],
                    "warnings": []})
            for line in variables.get("lines", []):
                gid = line.get("merchandiseId")
                if gid not in self.valid_variant_gids:
                    return self._wrap("cartLinesAdd", {
                        "cart": None,
                        "userErrors": [{
                            "field": ["lines", "0", "merchandiseId"],
                            "message": "Merchandise does not exist"}],
                        "warnings": []})
                existing = next(
                    (lid for lid, l in self.lines.items()
                     if l["merchandise"] == gid), None)
                if existing is not None:
                    # ASSUMPTION (undocumented live behavior): increment
                    self.lines[existing]["quantity"] += line.get(
                        "quantity", 1)
                else:
                    self._next_line += 1
                    lid = f"gid://shopify/CartLine/{self._next_line}"
                    self.lines[lid] = {"merchandise": gid,
                                       "quantity": line.get("quantity", 1)}
            return self._wrap("cartLinesAdd", {
                "cart": self._cart_payload(), "userErrors": [],
                "warnings": []})

        if "cartLinesUpdate" in query:
            for line in variables.get("lines", []):
                lid = line.get("id")
                if lid in self.lines:
                    self.lines[lid]["quantity"] = line.get("quantity", 1)
            return self._wrap("cartLinesUpdate", {
                "cart": self._cart_payload(), "userErrors": [],
                "warnings": []})

        if "cartLinesRemove" in query:
            for lid in variables.get("lineIds", []):
                self.lines.pop(lid, None)
            return self._wrap("cartLinesRemove", {
                "cart": self._cart_payload(), "userErrors": [],
                "warnings": []})

        if "cart(" in query:
            if self.mode == "wrong_cart":
                payload = self._cart_payload()
                payload["id"] = ("gid://shopify/Cart/d3Jvbmc"
                                 "?key=wrongcapability777777")
                return httpx.Response(200, json={"data": {"cart": payload}})
            if self.mode == "json_list":
                return httpx.Response(200, json=["not", "a", "dict"])
            if self.mode == "missing_page_info":
                return httpx.Response(200, json={"data": {"cart": {
                    "id": self.cart_id,
                    "lines": {"edges": []}}}})
            if self.mode == "null_cart":
                return httpx.Response(200, json={"data": {"cart": None}})
            if self.mode == "missing_lines":
                return httpx.Response(200, json={"data": {"cart": {
                    "id": FAKE_CART_ID}}})
            if self.mode == "string_quantity":
                return httpx.Response(200, json={"data": {"cart": {
                    "id": self.cart_id,
                    "lines": {"edges": [{"node": {
                        "id": "gid://shopify/CartLine/1",
                        "quantity": "1",
                        "merchandise": {"id": FAKE_VARIANT_GID}}}],
                        "pageInfo": {"hasNextPage": False}}}}})
            if self.mode == "truncated":
                return httpx.Response(200, json={"data": {"cart": {
                    "id": self.cart_id,
                    "lines": {"edges": [{"node": {
                        "id": "gid://shopify/CartLine/1",
                        "quantity": 1,
                        "merchandise": {"id": FAKE_VARIANT_GID}}}],
                        "pageInfo": {"hasNextPage": True}}}}})
            return httpx.Response(200,
                                  json={"data": {"cart": self._cart_payload()}})

        return httpx.Response(200, json={"data": {}})

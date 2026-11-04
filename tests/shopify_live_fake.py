"""P42 fake extension (NEW file; the reviewed tests/shopify_fake.py is NOT
modified). Adds the tokenless products query used by discover and an
optional no-merge cart behavior used to prove the probe's dynamic
expectations. LABELED SIMULATION - NOT LIVE PROOF.
"""
from __future__ import annotations

import json
from typing import Optional

import httpx

try:  # tests/ has no __init__.py in this repo; both roots work
    from tests.shopify_fake import FakeShopify
except ImportError:  # pragma: no cover
    from shopify_fake import FakeShopify


def default_products() -> list:
    return [
        {"id": "gid://shopify/Product/1", "title": "Test Product One",
         "variants": [
             {"id": "gid://shopify/ProductVariant/1", "sku": "TEST-SKU-1"},
             {"id": "gid://shopify/ProductVariant/2", "sku": "TEST-SKU-2"}]},
        {"id": "gid://shopify/Product/2", "title": "Test Product Two",
         "variants": [
             {"id": "gid://shopify/ProductVariant/3", "sku": ""}]},
    ]


class FakeLiveShopify(FakeShopify):
    """Reviewed fake + products query + optional no-merge add behavior."""

    def __init__(self, *args, merge: bool = True,
                 products: Optional[list] = None,
                 error_response: Optional[httpx.Response] = None,
                 **kwargs):
        prods = products if products is not None else default_products()
        gids = [v["id"] for p in prods for v in p["variants"]]
        kwargs.setdefault("valid_variant_gids", gids)
        super().__init__(*args, **kwargs)
        self.merge = merge
        self.products = prods
        # P47 test seam: when set, the products query gets this exact
        # response (status/body/headers), e.g. a 400 with a JSON errors
        # body, a secret-shaped payload, an oversized body, ...
        self.error_response = error_response

    def _handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        query = body.get("query", "")
        if "products(" in query:
            self.requests.append({"query": query,
                                  "variables": body.get("variables") or {}})
            if self.error_response is not None:
                return self.error_response
            if self.mode == "http_500":
                return httpx.Response(500, json={"errors": "boom"})
            if self.mode == "redirect":
                # A password-page style redirect: must be REJECTED, never
                # followed or parsed.
                return httpx.Response(
                    302, headers={
                        "location": "https://" + "cartproof.myshopify.com"
                                    + "/password"})
            return httpx.Response(200, json={"data": {"products": {
                "edges": [
                    {"node": {
                        "id": p["id"], "title": p["title"],
                        "variants": {"edges": [
                            {"node": {
                                "id": v["id"], "sku": v["sku"],
                                "price": {"amount": "1.00",
                                          "currencyCode": "INR"}}}
                            for v in p["variants"]]}}}
                    for p in self.products]}}})
        if "cartLinesAdd" in query and self.mode in (
                "add_http_500", "add_http_429"):
            # cartCreate and reads work; ONLY the first mutation fails.
            self.requests.append({"query": query,
                                  "variables": body.get("variables") or {}})
            status = 500 if self.mode == "add_http_500" else 429
            return httpx.Response(status, json={"errors": "boom"})
        if "cartLinesAdd" in query and not self.merge \
                and self.mode != "user_error_echo":
            self.requests.append({"query": query,
                                  "variables": body.get("variables") or {}})
            variables = body.get("variables") or {}
            for line in variables.get("lines", []):
                gid = line.get("merchandiseId")
                if gid not in self.valid_variant_gids:
                    return self._wrap("cartLinesAdd", {
                        "cart": None,
                        "userErrors": [{
                            "field": ["lines", "0", "merchandiseId"],
                            "message": "Merchandise does not exist"}],
                        "warnings": []})
                self._next_line += 1
                lid = f"gid://shopify/CartLine/{self._next_line}"
                self.lines[lid] = {"merchandise": gid,
                                   "quantity": line.get("quantity", 1)}
            return self._wrap("cartLinesAdd", {
                "cart": self._cart_payload(), "userErrors": [],
                "warnings": []})
        return super()._handler(request)

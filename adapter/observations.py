"""Observations: read the DOM cart, read the API cart, capture a screenshot.

Every observation returns a labelled :class:`Evidence`. The API read reuses the
same browser session cookie so DOM and API describe the same cart.
"""
from __future__ import annotations

import os

from .evidence import Evidence


def read_dom_cart(session) -> Evidence:
    session._goto("/cart")
    lines = []
    for li in session.page.query_selector_all("li.cart-line"):
        lines.append({
            "line_id": li.get_attribute("data-line-id"),
            "sku": li.get_attribute("data-sku"),
            "quantity": int(li.get_attribute("data-quantity")),
            "unit_price_cents": int(li.get_attribute("data-unit-price-cents")),
        })
    total_el = session.page.query_selector("#cart-total")
    total_cents = int(total_el.get_attribute("data-total-cents")) if total_el else None
    version_el = session.page.query_selector("#cart-version")
    version = int(version_el.text_content().strip()) if version_el else None
    return Evidence(
        source="dom_reread",
        payload={"lines": lines, "total_cents": total_cents, "version": version},
    )


def read_api_cart(session) -> Evidence:
    resp = session.context.request.get(f"{session.base_url}/api/cart")
    data = resp.json()
    return Evidence(source="api_reread", payload=data)


def screenshot(session, path: str) -> Evidence:
    session.page.screenshot(path=path, full_page=True)
    size = os.path.getsize(path)
    return Evidence(source="screenshot", payload={"path": path, "bytes": size})

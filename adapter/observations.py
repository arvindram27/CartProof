"""Observations: read the DOM cart, read the API cart, capture a screenshot.

Every observation returns a labelled :class:`Evidence`. The API read reuses the
same browser session cookie so DOM and API describe the same cart.

``read_dom_cart`` and ``read_api_cart`` accept an optional ``timeout_ms`` (the
remaining settle budget) so the verifier can bound each read. On a Playwright
timeout they return ``{"read_timeout": True}`` (insufficient evidence), never a
crash; on a parse/JSON error they return a malformed payload (version None).
With ``timeout_ms=None`` behaviour is unchanged from phase 1.
"""
from __future__ import annotations

import os

from .evidence import Evidence


def read_dom_cart(session, timeout_ms=None) -> Evidence:
    url = f"{session.base_url}/cart"
    try:
        if timeout_ms is None:
            session._goto("/cart")
        else:
            session.page.goto(url, timeout=timeout_ms)
    except Exception:
        return Evidence(source="dom_reread", payload={"read_timeout": True})
    try:
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
    except Exception:
        return Evidence(source="dom_reread", payload={"version": None, "lines": [], "total_cents": None})
    return Evidence(
        source="dom_reread",
        payload={"lines": lines, "total_cents": total_cents, "version": version},
    )


def read_api_cart(session, timeout_ms=None) -> Evidence:
    kwargs = {"timeout": timeout_ms} if timeout_ms is not None else {}
    try:
        resp = session.context.request.get(f"{session.base_url}/api/cart", **kwargs)
    except Exception:
        return Evidence(source="api_reread", payload={"read_timeout": True})
    try:
        data = resp.json()
    except Exception:
        return Evidence(source="api_reread", payload={"version": None, "lines": [], "total_cents": None})
    return Evidence(source="api_reread", payload=data)


def screenshot(session, path: str) -> Evidence:
    session.page.screenshot(path=path, full_page=True)
    size = os.path.getsize(path)
    return Evidence(source="screenshot", payload={"path": path, "bytes": size})

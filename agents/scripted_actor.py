"""Deterministic, non-LLM actor that assembles a cart through the adapter.

Takes a task spec (product query, target sku, quantity), executes it in a fixed
order, and returns an evidence log. No randomness, no retries, no checkout.
"""
from __future__ import annotations

from typing import Any

from adapter.actions import BrowserSession
from adapter.observations import read_api_cart, read_dom_cart, screenshot


def run_actor(session: BrowserSession, spec: dict[str, Any], screenshot_path: str | None = None) -> list[dict[str, Any]]:
    query = spec["product_query"]
    target_sku = spec["target_sku"]
    quantity = int(spec.get("quantity", 1))

    log: list[dict[str, Any]] = []

    search_ev = session.search(query)
    log.append(search_ev.to_dict())

    results = search_ev.payload["results"]
    if not results:
        raise RuntimeError(f"search returned no results for query {query!r}")
    product_id = results[0]["product_id"]

    open_ev = session.open_product(product_id)
    log.append(open_ev.to_dict())

    select_ev = session.select_variant(target_sku)
    log.append(select_ev.to_dict())

    add_ev = session.add_to_cart(quantity)
    log.append(add_ev.to_dict())

    dom_ev = read_dom_cart(session)
    log.append(dom_ev.to_dict())

    api_ev = read_api_cart(session)
    log.append(api_ev.to_dict())

    if screenshot_path is not None:
        shot_ev = screenshot(session, screenshot_path)
        log.append(shot_ev.to_dict())

    return log

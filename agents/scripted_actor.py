"""Deterministic, non-LLM actor that assembles a cart through the adapter.

Takes a task spec (either ``product_query``/``target_sku``/``quantity``, or an
``items`` list of the same), executes it in a fixed order, and returns an
evidence log. No randomness, no retries, no checkout.

With ``hook=None`` this behaves exactly as in phase 1. When a ``hook`` is
supplied it is invoked at three points per item and its returned
``{"stop": bool, "reason": str}`` result is honoured:

- ``before_add`` (before the page is re-opened) - the verifier settles the
  before-snapshot here;
- ``pre_add`` (after open+select, before the POST) - page version validation;
- ``after_add`` (after the mutating add) - diagnosis/recovery.
"""
from __future__ import annotations

from typing import Any

from adapter.actions import BrowserSession
from adapter.observations import read_api_cart, read_dom_cart, screenshot


def _item_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if "items" in spec:
        return [
            {
                "product_query": it["product_query"],
                "target_sku": it["target_sku"],
                "quantity": int(it.get("quantity", 1)),
            }
            for it in spec["items"]
        ]
    return [
        {
            "product_query": spec["product_query"],
            "target_sku": spec["target_sku"],
            "quantity": int(spec.get("quantity", 1)),
        }
    ]


def run_actor(session: BrowserSession, spec: dict[str, Any],
              screenshot_path: str | None = None, hook=None) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []

    for index, item in enumerate(_item_specs(spec)):
        query = item["product_query"]
        target_sku = item["target_sku"]
        quantity = item["quantity"]
        action_id = f"add-{index}"

        # before_add: let the verifier settle the before-snapshot first
        # (settlement navigates to /cart, so the page is restored after).
        if hook is not None:
            r = hook({"kind": "before_add", "action_id": action_id,
                      "item": {"index": index, "sku": target_sku, "quantity": quantity}})
            if r and r.get("stop"):
                break

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

        item_ctx = {"index": index, "product_id": product_id,
                    "sku": target_sku, "quantity": quantity}

        if hook is not None:
            r = hook({"kind": "pre_add", "action_id": action_id, "item": item_ctx,
                      "open": open_ev.to_dict()})
            if r and r.get("stop"):
                break

        add_ev = session.add_to_cart(quantity, sku=target_sku)
        log.append(add_ev.to_dict())

        if hook is not None:
            r = hook({"kind": "after_add", "action_id": action_id, "item": item_ctx,
                      "add": add_ev.to_dict()})
            if r and r.get("stop"):
                break

    dom_ev = read_dom_cart(session)
    log.append(dom_ev.to_dict())

    api_ev = read_api_cart(session)
    log.append(api_ev.to_dict())

    if screenshot_path is not None:
        shot_ev = screenshot(session, screenshot_path)
        log.append(shot_ev.to_dict())

    return log

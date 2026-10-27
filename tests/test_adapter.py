"""Adapter tests: actions and observations return labelled Evidence."""
from __future__ import annotations

import os

from adapter.evidence import Evidence
from adapter.observations import read_api_cart, read_dom_cart, screenshot

TARGET_SKU = "BTL-HIK-500-NAVYBLUE"


def test_actions_return_action_confirmation(browser_session):
    s = browser_session

    ev = s.search("Hiker Water Bottle")
    assert isinstance(ev, Evidence)
    assert ev.source == "action_confirmation"
    assert ev.payload["action"] == "search"
    assert len(ev.payload["results"]) == 1
    assert ev.payload["results"][0]["name"] == "Hiker Water Bottle"

    product_id = ev.payload["results"][0]["product_id"]
    ev = s.open_product(product_id)
    assert ev.source == "action_confirmation"
    assert ev.payload["action"] == "open_product"
    assert TARGET_SKU in ev.payload["variant_skus"]

    ev = s.select_variant(TARGET_SKU)
    assert ev.source == "action_confirmation"
    assert ev.payload["sku"] == TARGET_SKU

    ev = s.add_to_cart(1)
    assert ev.source == "action_confirmation"
    assert "Added" in ev.payload["status"]


def test_observations_return_labeled_evidence(browser_session, tmp_path):
    s = browser_session
    s.search("Hiker Water Bottle")
    s.open_product("bottle-hiker")
    s.select_variant(TARGET_SKU)
    s.add_to_cart(1)

    dom = read_dom_cart(s)
    assert isinstance(dom, Evidence)
    assert dom.source == "dom_reread"
    assert any(l["sku"] == TARGET_SKU for l in dom.payload["lines"])

    api = read_api_cart(s)
    assert isinstance(api, Evidence)
    assert api.source == "api_reread"
    assert any(l["sku"] == TARGET_SKU for l in api.payload["lines"])

    shot_path = str(tmp_path / "shot.png")
    shot = screenshot(s, shot_path)
    assert shot.source == "screenshot"
    assert os.path.exists(shot_path)

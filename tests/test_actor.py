"""Scripted actor: completes task1 end to end through the adapter."""
from __future__ import annotations

import json
from pathlib import Path

from agents.scripted_actor import run_actor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK1 = PROJECT_ROOT / "examples" / "task1.json"


def test_actor_completes_task1(browser_session):
    spec = json.loads(TASK1.read_text(encoding="utf-8"))
    log = run_actor(browser_session, spec)

    sources = [e["source"] for e in log]
    assert sources == [
        "action_confirmation",  # search
        "action_confirmation",  # open_product
        "action_confirmation",  # select_variant
        "action_confirmation",  # add_to_cart
        "dom_reread",
        "api_reread",
    ]

    dom = next(e for e in log if e["source"] == "dom_reread")
    api = next(e for e in log if e["source"] == "api_reread")

    # DOM and API carts agree on the assembled cart.
    dom_sig = sorted((l["sku"], l["quantity"]) for l in dom["payload"]["lines"])
    api_sig = sorted((l["sku"], l["quantity"]) for l in api["payload"]["lines"])
    assert dom_sig == api_sig
    assert dom_sig == [(spec["target_sku"], spec["quantity"])]

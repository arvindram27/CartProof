"""Supported-grammar intent tests (Brief 5 section 2 + R1 edges + M2)."""
from __future__ import annotations

import json
from pathlib import Path

from agents.intent import extract_intent, validate_proposal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = []
_data = json.loads((PROJECT_ROOT / "store" / "seed_catalog.json")
                   .read_text(encoding="utf-8"))
for _p in _data["products"]:
    for _v in _p["variants"]:
        CATALOG.append({"sku": _v["sku"], "product_id": _p["id"],
                        "product_name": _p["name"], "label": _v["label"],
                        "size": _v.get("size"), "color": _v.get("color"),
                        "price_cents": _v["price_cents"]})

NAVYBLUE = "BTL-HIK-500-NAVYBLUE"


def _proposal(items, budget=None, substitution=False):
    return {"items": [{"sku": s, "quantity": q} for s, q in items],
            "budget_max_cents": budget, "substitution_allowed": substitution}


def test_explicit_sku_proceeds_with_provenance():
    out = validate_proposal("Add 1 x BTL-HIK-500-NAVYBLUE.", CATALOG,
                            _proposal([(NAVYBLUE, 1)]))
    assert out["outcome_kind"] == "proceed"
    ei = out["executable_intent"]
    assert ei["items"][0]["sku"] == NAVYBLUE
    assert ei["provenance"][0]["phrase"]


def test_attribute_form_grounds_unique_variant():
    out = validate_proposal("Add a 750ml steel Hiker Water Bottle.", CATALOG,
                            _proposal([("BTL-HIK-750-STEEL", 1)]))
    assert out["outcome_kind"] == "proceed"


def test_budget_language_mappings_m2():
    under = extract_intent("Add 1 x BTL-THR-500-BLACK, keep it under $50.",
                           CATALOG)
    assert under["budget_max_cents"] == 4999
    at_most = extract_intent(
        "Add 1 x BTL-THR-500-BLACK, keep the total at most $50.", CATALOG)
    assert at_most["budget_max_cents"] == 5000


def test_invented_budget_is_rejected():
    out = validate_proposal("Add 1 x BTL-HIK-500-NAVYBLUE.", CATALOG,
                            _proposal([(NAVYBLUE, 1)], budget=5000))
    # Fully grounded extraction + cross-check disagreement -> semantic clarify,
    # with the rejection recorded as an unresolved point (not operational).
    assert out["outcome_kind"] == "semantic_clarify"
    assert any("budget" in p for p in out["unresolved_points"])


def test_invented_substitution_is_rejected():
    out = validate_proposal("Add 1 x BTL-HIK-500-NAVYBLUE.", CATALOG,
                            _proposal([(NAVYBLUE, 1)], substitution=True))
    assert out["outcome_kind"] == "semantic_clarify"
    assert any("substitution" in p for p in out["unresolved_points"])


def test_no_substitutions_removes_permission():
    out = extract_intent(
        "Add 1 x BTL-HIK-500-NAVYBLUE. No substitutions.", CATALOG)
    assert out["substitution_allowed"] is False


def test_malformed_proposal_is_operational_failure():
    out = validate_proposal("Add 1 x BTL-HIK-500-NAVYBLUE.", CATALOG,
                            "<malformed-output>")
    assert out["outcome_kind"] == "operational_failure"


def test_omitted_grounded_item_clarifies_or_rejects():
    # Proposal drops the second grounded item: disagreement, never proceeds.
    out = validate_proposal(
        "Add 1 x BTL-HIK-500-NAVYBLUE and add 1 x BTL-TMB-12-SLATE.",
        CATALOG, _proposal([(NAVYBLUE, 1)]))
    assert out["outcome_kind"] in ("operational_failure", "semantic_clarify")
    assert out["executable_intent"] is None


import pytest


@pytest.mark.parametrize("text", [
    "Don't add BTL-HIK-500-NAVYBLUE.",
    "Add a Hiker Water Bottle except the steel one.",
    "Add a Hiker Water Bottle or a Thermo Flask.",
    "Add a couple of Kids Bottle.",
    "Add 0 x BTL-HIK-500-NAVYBLUE.",
    "Add 100 x BTL-HIK-500-NAVYBLUE.",
    "Add 1 or 2 x BTL-HIK-500-NAVYBLUE.",
    "Add some Kids Bottle.",
    "Add a 500ml Hiker Water Bottle.",
    "Add a water bottle.",
])
def test_edges_never_produce_executable_intent(text):
    out = validate_proposal(text, CATALOG,
                            _proposal([(NAVYBLUE, 1)]))
    assert out["outcome_kind"] != "proceed"
    assert out["executable_intent"] is None

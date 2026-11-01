"""Offline regressions for the careful-agent inspection validator (P26).

The inspection menu advertises finish, add_to_cart (intent items), and
set_line_quantity/delete_line (observed cart lines). These tests use a fake
session and the stub model client only; nothing here touches the network, a
browser, a provider, or project evidence.
"""
from __future__ import annotations

import httpx
import pytest

from adapter.observations import read_api_cart
from agents.careful_agent import CarefulAgent
from agents.model_actor import ModelActor
from agents.model_client import BudgetedModelClient, ClientConfig, FakeModelClient


CATALOG = [
    {"sku": "SKU-A", "product_id": "p1", "product_name": "Prod",
     "label": "A", "size": None, "color": None, "price_cents": 100},
    {"sku": "SKU-B", "product_id": "p1", "product_name": "Prod",
     "label": "B", "size": None, "color": None, "price_cents": 100},
]
INTENT = {"payload": {"items": [{"sku": "SKU-A", "quantity": 1}],
                      "budget_max_cents": None, "substitution_allowed": False}}


def _line(line_id, sku, quantity):
    return {"line_id": line_id, "sku": sku, "quantity": quantity,
            "unit_price_cents": 100}


class Ev:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return {"payload": self.payload}


class FakeSession:
    """Fake adapter boundary that keeps a cart and records every call."""

    def __init__(self, lines=None, version=1):
        self.cart = {"version": version,
                     "lines": [dict(l) for l in (lines or [])],
                     "total_cents": 0}
        self.calls = []

    def _bump(self):
        self.cart["version"] += 1
        return self.cart["version"]

    def snapshot(self):
        return {"version": self.cart["version"],
                "lines": [dict(l) for l in self.cart["lines"]],
                "total_cents": 0}

    def search(self, query):
        self.calls.append(("search", query))
        return Ev({"results": [{"product_id": "p1", "name": "Prod"}]})

    def open_product(self, product_id):
        self.calls.append(("open_product", product_id))
        return Ev({"variant_skus": ["SKU-A", "SKU-B"],
                   "cart_version": self.cart["version"]})

    def select_variant(self, sku):
        self.calls.append(("select_variant", sku))
        return Ev({"sku": sku})

    def add_to_cart(self, quantity=1, sku=None):
        self.calls.append(("add_to_cart", sku, quantity))
        line = next((l for l in self.cart["lines"] if l["sku"] == sku), None)
        if line is None:
            self.cart["lines"].append(_line(f"L-{sku}", sku, quantity))
        else:
            line["quantity"] += quantity
        self._bump()
        return Ev({"response_status": 200,
                   "response_version": self.cart["version"]})

    def set_line_quantity(self, line_id, sku, quantity, expected_version):
        self.calls.append(("set_line_quantity", line_id, sku, quantity,
                           expected_version))
        for line in self.cart["lines"]:
            if line["line_id"] == line_id:
                line["quantity"] = quantity
        self._bump()
        return Ev({"response_status": 200})

    def delete_line(self, line_id, sku, expected_version):
        self.calls.append(("delete_line", line_id, sku, expected_version))
        self.cart["lines"] = [l for l in self.cart["lines"]
                              if l["line_id"] != line_id]
        self._bump()
        return Ev({"response_status": 200})


def _wire_read(monkeypatch):
    import adapter.observations as obs

    monkeypatch.setattr(obs, "read_api_cart",
                        lambda session: Ev(session.snapshot()))


def _client(responses):
    return BudgetedModelClient(FakeModelClient(responses),
                               ClientConfig(enabled=True,
                                            token_budget=100000))


def _agent(session, responses, items=None):
    agent = CarefulAgent(session, _client(responses), CATALOG, masked=False)
    agent.executable_intent = {
        "items": items or [{"sku": "SKU-A", "quantity": 1}],
        "budget_max_cents": None,
        "substitution_allowed": False,
    }
    agent.observed_cart = session.snapshot()
    return agent


def _mutations(session):
    return [c for c in session.calls
            if c[0] in ("add_to_cart", "set_line_quantity", "delete_line")]


# --------------------------------------------------------------------------- #
# 1. delete_line admitted and executed end to end
# --------------------------------------------------------------------------- #
def test_inspection_delete_line_admitted_and_executed(monkeypatch):
    _wire_read(monkeypatch)
    session = FakeSession([_line("L-A", "SKU-A", 1), _line("L-B", "SKU-B", 1)])
    agent = _agent(session, [
        {"payload": {"action": "delete_line", "sku": "SKU-B"}},
        {"payload": {"action": "finish"}},
        {"payload": {"declaration": "COMPLETE", "rationale": "wrong line gone"}},
    ])
    out = agent.inspect_and_review()
    assert ("delete_line", "L-B", "SKU-B", 1) in session.calls
    assert out["declared_outcome"] == "COMPLETE"
    assert session.snapshot()["lines"] == [_line("L-A", "SKU-A", 1)]
    # The cart was re-read after the mutation.
    assert agent.observed_cart["lines"] == [_line("L-A", "SKU-A", 1)]


# --------------------------------------------------------------------------- #
# 2. set_line_quantity admitted with an in-bound quantity
# --------------------------------------------------------------------------- #
def test_inspection_set_line_quantity_admitted(monkeypatch):
    _wire_read(monkeypatch)
    session = FakeSession([_line("L-A", "SKU-A", 1)])
    agent = _agent(session, [
        {"payload": {"action": "set_line_quantity", "sku": "SKU-A",
                     "quantity": 2}},
        {"payload": {"action": "finish"}},
        {"payload": {"declaration": "COMPLETE", "rationale": "corrected"}},
    ], items=[{"sku": "SKU-A", "quantity": 2}])
    out = agent.inspect_and_review()
    assert ("set_line_quantity", "L-A", "SKU-A", 2, 1) in session.calls
    assert out["declared_outcome"] == "COMPLETE"
    assert session.snapshot()["lines"] == [_line("L-A", "SKU-A", 2)]


# --------------------------------------------------------------------------- #
# 3. finish admitted: loop terminates and the review call happens
# --------------------------------------------------------------------------- #
def test_inspection_finish_admitted_and_review_runs(monkeypatch):
    _wire_read(monkeypatch)
    session = FakeSession([_line("L-A", "SKU-A", 1)])
    agent = _agent(session, [
        {"payload": {"action": "finish"}},
        {"payload": {"declaration": "COMPLETE", "rationale": "already right"}},
    ])
    out = agent.inspect_and_review()
    assert out["declared_outcome"] == "COMPLETE"
    assert session.calls == []


# --------------------------------------------------------------------------- #
# 4. Actor build loop still rejects the inspection-only actions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload", [
    {"action": "finish"},
    {"action": "delete_line", "sku": "SKU-A"},
    {"action": "set_line_quantity", "sku": "SKU-A", "quantity": 1},
])
def test_actor_build_loop_rejects_inspection_only_actions(payload):
    session = FakeSession([])
    agent = ModelActor(session, _client([INTENT, {"payload": payload}]),
                       CATALOG, masked=False)
    agent.resolve_intent("Add 1 x SKU-A.")
    agent.execute_items([{"product_query": "Prod", "target_sku": "SKU-A",
                          "quantity": 1}])
    assert session.calls == []
    assert agent.failures
    assert "action not in available menu" in agent.failures[-1]["reason"]


# --------------------------------------------------------------------------- #
# 6. Refusal cases: no mutation, operational failure, PARTIAL
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload, reason", [
    ({"action": "delete_line", "sku": "SKU-X"}, "not in menu"),
    ({"action": "set_line_quantity", "sku": "SKU-A", "quantity": 0},
     "integer >= 1"),
    ({"action": "set_line_quantity", "sku": "SKU-A", "quantity": -1},
     "integer >= 1"),
    ({"action": "set_line_quantity", "sku": "SKU-A", "quantity": True},
     "integer >= 1"),
    ({"action": "set_line_quantity", "sku": "SKU-A", "quantity": "2"},
     "integer >= 1"),
    ({"action": "set_line_quantity", "sku": "SKU-A", "quantity": 2.0},
     "integer >= 1"),
    ({"action": "set_line_quantity", "sku": "SKU-A", "quantity": None},
     "integer >= 1"),
    ({"action": "set_line_quantity", "sku": "SKU-A", "quantity": 2},
     "exceeds intent quantity"),
    ({"action": "set_line_quantity", "sku": "SKU-B", "quantity": 1},
     "sku not in intent"),
])
def test_inspection_refusals_do_not_mutate(monkeypatch, payload, reason):
    _wire_read(monkeypatch)
    session = FakeSession([_line("L-A", "SKU-A", 1), _line("L-B", "SKU-B", 1)])
    agent = _agent(session, [
        {"payload": payload},
        {"payload": {"declaration": "PARTIAL", "rationale": "refused"}},
    ])
    out = agent.inspect_and_review()
    assert out["declared_outcome"] == "PARTIAL"
    assert _mutations(session) == []
    assert agent.failures
    assert reason in agent.failures[-1]["reason"]


# --------------------------------------------------------------------------- #
# 7. Replay of the observed stage-2 failure shape: wrong variant seeded and
#    the intended item missing; delete the wrong line, add the intended one.
# --------------------------------------------------------------------------- #
def test_inspection_replays_wrong_variant_plus_missing_intent(monkeypatch):
    _wire_read(monkeypatch)
    session = FakeSession([_line("L-B", "SKU-B", 1)])
    agent = _agent(session, [
        {"payload": {"action": "delete_line", "sku": "SKU-B"}},
        {"payload": {"action": "add_to_cart", "sku": "SKU-A", "quantity": 1}},
        {"payload": {"action": "finish"}},
        {"payload": {"declaration": "COMPLETE", "rationale": "replaced"}},
    ])
    out = agent.inspect_and_review()
    assert out["declared_outcome"] == "COMPLETE"
    assert ("delete_line", "L-B", "SKU-B", 1) in session.calls
    assert ("add_to_cart", "SKU-A", 1) in session.calls
    assert [(l["sku"], l["quantity"])
            for l in session.snapshot()["lines"]] == [("SKU-A", 1)]
    assert any(c[0] == "search" for c in session.calls)


# --------------------------------------------------------------------------- #
# Localhost CA transient-fault coverage (P27)
#
# These use the real store/control apps on ephemeral 127.0.0.1 ports and a real
# local Playwright browser. The model client is still the scripted fake; no
# provider is constructed or called.
# --------------------------------------------------------------------------- #
SKU_NAVYBLUE = "BTL-HIK-500-NAVYBLUE"
SKU_NAVY = "BTL-HIK-500-NAVY"


def _reset_store(store):
    httpx.post(f"{store['control']}/_control/reset").raise_for_status()


def _set_fault(store, name):
    httpx.post(f"{store['control']}/_control/fault",
               json={"name": name, "params": {}}).raise_for_status()


def _clear_fault(store, name):
    httpx.post(f"{store['control']}/_control/fault/clear",
               json={"name": name}).raise_for_status()


def _local_catalog():
    from harness.model_compare import load_catalog
    return load_catalog()


def _intent_payload(sku, quantity=1):
    return {"payload": {"items": [{"sku": sku, "quantity": quantity}],
                        "budget_max_cents": None,
                        "substitution_allowed": False}}


def _build_responses(sku):
    return [
        {"payload": {"action": "search"}},
        {"payload": {"action": "open_product", "product_id": "bottle-hiker"}},
        {"payload": {"action": "select_variant", "sku": sku}},
        {"payload": {"action": "add_to_cart", "sku": sku, "quantity": 1}},
    ]


def _cart(session):
    return read_api_cart(session).payload


def _cart_quantities(session):
    return {line["sku"]: line["quantity"]
            for line in _cart(session)["lines"]}


def test_ca_silent_drop_transient_recovers(store, browser_session):
    """Stage-2 drop shape: first add is dropped, the retry commits."""
    _reset_store(store)
    _set_fault(store, "silent-drop")
    responses = [_intent_payload(SKU_NAVYBLUE)] + _build_responses(SKU_NAVYBLUE) + [
        {"payload": {"action": "read_cart"}},
        {"payload": {"action": "add_to_cart", "sku": SKU_NAVYBLUE,
                     "quantity": 1}},
        {"payload": {"action": "finish"}},
        {"payload": {"declaration": "COMPLETE", "rationale": "retry committed"}},
    ]
    agent = CarefulAgent(browser_session, _client(responses),
                         _local_catalog(), masked=False)
    agent.resolve_intent("Add 1 x BTL-HIK-500-NAVYBLUE.")
    agent.execute_items([{"product_query": "Hiker Water Bottle",
                          "target_sku": SKU_NAVYBLUE, "quantity": 1}])

    # Silent drop is observable as the intended line missing from persisted
    # state after the first (HTTP 200) add. CA has no RecoveryController, so no
    # failure entry is required here.
    assert _cart_quantities(browser_session) == {}

    _clear_fault(store, "silent-drop")
    out = agent.inspect_and_review()

    assert _cart_quantities(browser_session) == {SKU_NAVYBLUE: 1}
    assert out["declared_outcome"] == "COMPLETE"
    assert agent.add_statuses == [200, 200]
    assert agent.failures == []
    assert [a["action"] for a in agent.inspection_actions] == [
        "read_cart", "add_to_cart", "finish"]


def test_ca_wrong_variant_transient_recovers(store, browser_session,
                                             monkeypatch):
    """Stage-2 variant shape: delete the server-observed wrong line, re-add."""
    _reset_store(store)
    _set_fault(store, "wrong-variant")
    responses = [_intent_payload(SKU_NAVYBLUE)] + _build_responses(SKU_NAVYBLUE) + [
        {"payload": {"action": "read_cart"}},
        {"payload": {"action": "delete_line", "sku": SKU_NAVY}},
        {"payload": {"action": "add_to_cart", "sku": SKU_NAVYBLUE,
                     "quantity": 1}},
        {"payload": {"action": "finish"}},
        {"payload": {"declaration": "COMPLETE", "rationale": "corrected"}},
    ]
    agent = CarefulAgent(browser_session, _client(responses),
                         _local_catalog(), masked=False)
    agent.resolve_intent("Add 1 x BTL-HIK-500-NAVYBLUE.")
    agent.execute_items([{"product_query": "Hiker Water Bottle",
                          "target_sku": SKU_NAVYBLUE, "quantity": 1}])

    before = _cart(browser_session)
    assert [line["sku"] for line in before["lines"]] == [SKU_NAVY]
    wrong = before["lines"][0]
    expected_version = before["version"]

    _clear_fault(store, "wrong-variant")
    delete_calls = []
    original_delete = browser_session.delete_line

    def record_delete(line_id, sku, version):
        delete_calls.append((line_id, sku, version))
        return original_delete(line_id, sku, version)

    monkeypatch.setattr(browser_session, "delete_line", record_delete)
    out = agent.inspect_and_review()

    assert delete_calls == [(wrong["line_id"], SKU_NAVY, expected_version)]
    assert _cart_quantities(browser_session) == {SKU_NAVYBLUE: 1}
    assert out["declared_outcome"] == "COMPLETE"
    assert agent.add_statuses == [200, 200]
    assert agent.failures == []
    assert [a["action"] for a in agent.inspection_actions] == [
        "read_cart", "delete_line", "add_to_cart", "finish"]

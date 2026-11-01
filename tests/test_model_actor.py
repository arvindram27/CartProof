"""Model actor, careful-agent judgment, and full-cart guard tests."""
from __future__ import annotations

import pytest

from agents.careful_agent import CarefulAgent
from agents.full_cart_guard import apply_full_cart_guard, completion_settled_view
from agents.model_actor import ModelActor
from agents.model_client import BudgetedModelClient, ClientConfig, FakeModelClient


class _Ev:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return {"payload": self.payload}


class StubSession:
    def __init__(self):
        self.calls = []
        self.writes = []

    def search(self, query):
        self.calls.append("search")
        return _Ev({"action": "search", "results": [{"product_id": "p1"}]})

    def open_product(self, pid):
        self.calls.append("open_product")
        return _Ev({"action": "open_product", "product_id": pid,
                    "variant_skus": ["SKU-A", "SKU-B"], "cart_version": 0})

    def select_variant(self, sku):
        self.calls.append("select_variant")
        return _Ev({"action": "select_variant", "sku": sku})

    def add_to_cart(self, quantity=1, sku=None):
        self.calls.append("add_to_cart")
        self.writes.append((sku, quantity))
        return _Ev({"action": "add_to_cart", "response_status": 200,
                    "response_version": 1})


CATALOG = [
    {"sku": "SKU-A", "product_id": "p1", "product_name": "Prod",
     "label": "A", "size": None, "color": None, "price_cents": 100},
    {"sku": "SKU-B", "product_id": "p1", "product_name": "Prod",
     "label": "B", "size": None, "color": None, "price_cents": 100},
]

INTENT = {"payload": {"items": [{"sku": "SKU-A", "quantity": 1}],
                      "budget_max_cents": None, "substitution_allowed": False}}


def _client(responses):
    fake = FakeModelClient(responses)
    return BudgetedModelClient(fake, ClientConfig(enabled=True,
                                                  token_budget=100000))


def _actions(sku="SKU-A"):
    return [{"payload": {"action": "search"}},
            {"payload": {"action": "open_product", "product_id": "p1"}},
            {"payload": {"action": "select_variant", "sku": sku}},
            {"payload": {"action": "add_to_cart", "sku": sku, "quantity": 1}}]


def test_masked_actor_rejects_wrong_sku_before_any_write():
    session = StubSession()
    actor = ModelActor(session, _client([INTENT] + _actions("SKU-B")),
                       CATALOG, masked=True)
    actor.resolve_intent("Add 1 x SKU-A.")
    # Masked menu only offers SKU-A; the SKU-B proposal is not in the menu.
    out = actor.execute_items([{"product_query": "Prod", "target_sku": "SKU-A",
                                "quantity": 1}])
    assert session.writes == []  # no write reached the adapter
    assert actor.failures


def test_unmasked_actor_may_add_wrong_sku():
    session = StubSession()
    actor = ModelActor(session, _client([INTENT] + _actions("SKU-B")),
                       CATALOG, masked=False)
    actor.resolve_intent("Add 1 x SKU-A.")
    actor.execute_items([{"product_query": "Prod", "target_sku": "SKU-A",
                          "quantity": 1}])
    assert session.writes == [("SKU-B", 1)]


def test_malformed_action_proposal_never_writes():
    session = StubSession()
    actor = ModelActor(session,
                       _client([INTENT, {"malformed": True}]),
                       CATALOG, masked=True)
    actor.resolve_intent("Add 1 x SKU-A.")
    actor.execute_items([{"product_query": "Prod", "target_sku": "SKU-A",
                          "quantity": 1}])
    assert session.writes == []


def test_hook_events_match_scripted_actor_shapes():
    session = StubSession()
    seen = []

    def hook(event):
        seen.append((event["kind"], event["action_id"],
                     event["item"]["index"]))
        return None

    actor = ModelActor(session, _client([INTENT] + _actions()),
                       CATALOG, masked=True, hook=hook)
    actor.resolve_intent("Add 1 x SKU-A.")
    actor.execute_items([{"product_query": "Prod", "target_sku": "SKU-A",
                          "quantity": 1}])
    assert seen == [("before_add", "add-0", 0), ("pre_add", "add-0", 0),
                    ("after_add", "add-0", 0)]
    assert actor.completed_items == [0]


def test_naive_declared_requires_every_intended_item_answered():
    # (a) two intended items, early stop after one 200 add -> PARTIAL
    actor = ModelActor(StubSession(), _client([]), CATALOG, masked=False)
    actor.executable_intent = {"items": [{"sku": "SKU-A", "quantity": 1},
                                         {"sku": "SKU-B", "quantity": 1}]}
    actor.add_statuses = [200]
    actor.add_status_by_index = {0: 200}
    assert actor.naive_declared() == "PARTIAL"

    # (c) an intended item never attempted -> PARTIAL (zero executed items too)
    actor2 = ModelActor(StubSession(), _client([]), CATALOG, masked=False)
    actor2.executable_intent = {"items": [{"sku": "SKU-A", "quantity": 1}]}
    assert actor2.naive_declared() == "PARTIAL"

    # (d) duplicate same-SKU items each require their own answered add
    actor3 = ModelActor(StubSession(), _client([]), CATALOG, masked=False)
    actor3.executable_intent = {"items": [{"sku": "SKU-A", "quantity": 1},
                                          {"sku": "SKU-A", "quantity": 1}]}
    actor3.add_status_by_index = {0: 200}
    assert actor3.naive_declared() == "PARTIAL"
    actor3.add_status_by_index = {0: 200, 1: 200}
    assert actor3.naive_declared() == "COMPLETE"


def test_naive_declared_malformed_after_first_add_is_partial():
    # (b) first item answered 200, then a malformed action proposal stops the
    # run before the second item is attempted -> PARTIAL.
    session = StubSession()
    intent2 = {"payload": {"items": [{"sku": "SKU-A", "quantity": 1},
                                     {"sku": "SKU-B", "quantity": 1}],
                           "budget_max_cents": None,
                           "substitution_allowed": False}}
    actor = ModelActor(session, _client([intent2] + _actions("SKU-A")
                                        + [{"malformed": True}]),
                       CATALOG, masked=True)
    actor.resolve_intent("Add 1 x SKU-A and add 1 x SKU-B.")
    actor.execute_items([{"product_query": "Prod", "target_sku": "SKU-A",
                          "quantity": 1},
                         {"product_query": "Prod", "target_sku": "SKU-B",
                          "quantity": 1}])
    assert session.writes == [("SKU-A", 1)]
    assert actor.add_status_by_index == {0: 200}
    assert actor.naive_declared() == "PARTIAL"


def test_naive_declared_all_items_answered_complete():
    # (e) all intended items answered 200 -> COMPLETE (behavior preserved)
    session = StubSession()
    actor = ModelActor(session, _client([INTENT] + _actions()),
                       CATALOG, masked=True)
    actor.resolve_intent("Add 1 x SKU-A.")
    actor.execute_items([{"product_query": "Prod", "target_sku": "SKU-A",
                          "quantity": 1}])
    assert actor.add_status_by_index == {0: 200}
    assert actor.naive_declared() == "COMPLETE"


def test_ca_declaration_is_the_models_judgment():
    # Model declares COMPLETE in review; code must not override it, even
    # though no cart observation supports it.
    session = StubSession()
    responses = [INTENT] + _actions() + [
        {"payload": {"action": "finish"}},
        {"payload": {"declaration": "COMPLETE", "rationale": "looks done"}}]
    agent = CarefulAgent(session, _client(responses), CATALOG, masked=False)
    agent.resolve_intent("Add 1 x SKU-A.")
    agent.execute_items([{"product_query": "Prod", "target_sku": "SKU-A",
                          "quantity": 1}])
    out = agent.inspect_and_review()
    assert out["declared_outcome"] == "COMPLETE"


def test_ca_partial_judgment_is_respected():
    session = StubSession()
    responses = [INTENT] + _actions() + [
        {"payload": {"action": "finish"}},
        {"payload": {"declaration": "PARTIAL", "rationale": "not sure"}}]
    agent = CarefulAgent(session, _client(responses), CATALOG, masked=False)
    agent.resolve_intent("Add 1 x SKU-A.")
    agent.execute_items([{"product_query": "Prod", "target_sku": "SKU-A",
                          "quantity": 1}])
    assert agent.inspect_and_review()["declared_outcome"] == "PARTIAL"


def test_ca_inspection_add_cap_bounds_retries(monkeypatch):
    import adapter.observations as obs

    class _Cart:
        payload = {"version": 0, "lines": [], "total_cents": 0}

    monkeypatch.setattr(obs, "read_api_cart", lambda session: _Cart())
    session = StubSession()
    responses = [INTENT] + _actions() + [
        {"payload": {"action": "add_to_cart", "sku": "SKU-A", "quantity": 1}}
        for _ in range(4)
    ] + [{"payload": {"declaration": "PARTIAL", "rationale": "cannot add"}}]
    agent = CarefulAgent(session, _client(responses), CATALOG, masked=False)
    agent.resolve_intent("Add 1 x SKU-A.")
    agent.execute_items([{"product_query": "Prod", "target_sku": "SKU-A",
                          "quantity": 1}])
    out = agent.inspect_and_review()
    assert out["declared_outcome"] == "PARTIAL"
    # 1 add in execute_items + 3 bounded inspection adds; the 4th is refused
    # before it reaches the adapter.
    assert len(session.writes) == 4
    assert agent.failures
    assert "add cap" in agent.failures[-1]["reason"]


# ------------------------------------------------------------------ guard

def _line(sku, qty, price=100):
    return {"line_id": sku, "sku": sku, "quantity": qty,
            "unit_price_cents": price}


def _pair(lines, version=1, total=None):
    view = {"version": version, "lines": lines,
            "total_cents": total if total is not None
            else sum(l["quantity"] * l["unit_price_cents"] for l in lines)}
    return (dict(view), dict(view))


def test_completion_settled_view_uses_last_recorded_pair():
    report = {"final": {"version": 1}}
    raw = {1: _pair([_line("A", 1)]), 3: _pair([_line("A", 1), _line("X", 1)])}
    # Last pair is the completion settle: it carries the extra SKU.
    settled = completion_settled_view(report, raw)
    assert settled["version"] == 1
    assert settled["lines"][-1]["sku"] == "X"


def test_guard_catches_unrelated_extra_sku_and_preserves_raw():
    report = {"final": {"version": 1}}
    raw = {1: _pair([_line("A", 1), _line("X", 1)])}
    guard = apply_full_cart_guard(report, raw, {}, {"A": 1}, None, "COMPLETE")
    assert guard["status"] == "downgrade"
    assert guard["guarded_verdict"] == "PARTIAL"
    assert guard["mismatches"] == [{"sku": "X", "expected_quantity": 0,
                                    "actual_quantity": 1}]


def test_guard_catches_missing_seeded_line():
    report = {"final": {"version": 1}}
    raw = {1: _pair([_line("A", 1)])}
    guard = apply_full_cart_guard(report, raw, {"S": 1}, {"A": 1}, None,
                                  "COMPLETE")
    assert guard["guarded_verdict"] == "PARTIAL"
    assert any(m["sku"] == "S" for m in guard["mismatches"])


def test_guard_aggregates_duplicate_sku_lines():
    report = {"final": {"version": 1}}
    raw = {1: _pair([_line("A", 1), _line("A", 1)])}
    guard = apply_full_cart_guard(report, raw, {}, {"A": 2}, None, "COMPLETE")
    assert guard["status"] == "pass"
    assert guard["guarded_verdict"] == "COMPLETE"


def test_guard_unsettled_or_invalid_never_permits_complete():
    guard = apply_full_cart_guard({"final": {"version": None}}, {}, {}, {"A": 1},
                                  None, "COMPLETE")
    assert guard["status"] == "not_eligible"
    assert guard["guarded_verdict"] == "PARTIAL"

    bad = {1: ({"version": 1, "lines": [{"sku": "A", "quantity": True,
                                         "unit_price_cents": 100}],
                "total_cents": 100},
               {"version": 1, "lines": [_line("A", 1)], "total_cents": 100})}
    guard2 = apply_full_cart_guard({"final": {"version": 1}}, bad, {},
                                   {"A": 1}, None, "COMPLETE")
    assert guard2["status"] == "not_eligible"
    assert guard2["guarded_verdict"] == "PARTIAL"


def test_guard_budget_recompute_from_line_prices():
    report = {"final": {"version": 1}}
    raw = {1: _pair([_line("A", 1, price=6000)])}
    guard = apply_full_cart_guard(report, raw, {}, {"A": 1}, 5000, "COMPLETE")
    assert guard["guarded_verdict"] == "PARTIAL"
    assert guard["recomputed_total_cents"] == 6000

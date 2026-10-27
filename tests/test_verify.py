"""Verification-layer tests: checks, settlement, anomalies, provenance, and the
adapter's response_version extraction (integration via the real store + browser).
"""
from __future__ import annotations

import httpx
import pytest

from core.checks import evaluate, quantities, recompute_total
from core.ledger import CONFIRMED, CONTRADICTED, UNKNOWN, Constraint, Ledger
from core.verify import Verifier

SKU_NAVYBLUE = "BTL-HIK-500-NAVYBLUE"


# --------------------------------------------------------------------------- #
# Deterministic checks (pure)
# --------------------------------------------------------------------------- #

def test_item_quantity_pass_and_fail():
    c = Constraint.from_spec({"type": "item_quantity", "sku": "A", "quantity": 2})
    ok = {"lines": [{"sku": "A", "quantity": 2, "unit_price_cents": 100}], "total_cents": 200}
    assert evaluate(c, ok)[0] == CONFIRMED
    bad = {"lines": [{"sku": "A", "quantity": 1, "unit_price_cents": 100}], "total_cents": 100}
    assert evaluate(c, bad)[0] == CONTRADICTED


def test_count_distinct_exact_semantics():
    c = Constraint.from_spec({"type": "count_distinct", "skus": ["A", "B", "C"], "count": 2})
    two = {"lines": [
        {"sku": "A", "quantity": 1, "unit_price_cents": 1},
        {"sku": "B", "quantity": 1, "unit_price_cents": 1},
    ], "total_cents": 2}
    assert evaluate(c, two)[0] == CONFIRMED
    one = {"lines": [{"sku": "A", "quantity": 1, "unit_price_cents": 1}], "total_cents": 1}
    assert evaluate(c, one)[0] == CONTRADICTED
    three = {"lines": [
        {"sku": "A", "quantity": 1, "unit_price_cents": 1},
        {"sku": "B", "quantity": 1, "unit_price_cents": 1},
        {"sku": "C", "quantity": 1, "unit_price_cents": 1},
    ], "total_cents": 3}
    assert evaluate(c, three)[0] == CONTRADICTED  # exactly N, not at least N


def test_count_distinct_counts_pre_existing_baseline_items():
    c = Constraint.from_spec({"type": "count_distinct", "skus": ["SEEDED", "ADDED"], "count": 2})
    view = {"lines": [
        {"sku": "SEEDED", "quantity": 1, "unit_price_cents": 1},
        {"sku": "ADDED", "quantity": 1, "unit_price_cents": 1},
    ], "total_cents": 2}
    assert evaluate(c, view)[0] == CONFIRMED


def test_budget_pass_fail_and_internal_conflict():
    c = Constraint.from_spec({"type": "budget_max", "max_cents": 100})
    under = {"lines": [{"sku": "A", "quantity": 1, "unit_price_cents": 50}], "total_cents": 50}
    assert evaluate(c, under)[0] == CONFIRMED
    over = {"lines": [{"sku": "A", "quantity": 1, "unit_price_cents": 200}], "total_cents": 200}
    assert evaluate(c, over)[0] == CONTRADICTED
    conflict = {"lines": [{"sku": "A", "quantity": 1, "unit_price_cents": 200}], "total_cents": 0}
    assert evaluate(c, conflict)[0] == UNKNOWN


def test_final_absolute_no_baseline_double_count():
    exact = Constraint.from_spec({"type": "item_quantity", "sku": "A", "quantity": 2})
    view = {"lines": [{"sku": "A", "quantity": 2, "unit_price_cents": 100}], "total_cents": 200}
    assert evaluate(exact, view)[0] == CONFIRMED
    wrong = Constraint.from_spec({"type": "item_quantity", "sku": "A", "quantity": 1})
    assert evaluate(wrong, view)[0] == CONTRADICTED  # 2 != 1 (no +baseline)


def test_recompute_and_quantities_helpers():
    lines = [{"sku": "A", "quantity": 2, "unit_price_cents": 150},
             {"sku": "A", "quantity": 1, "unit_price_cents": 150},
             {"sku": "B", "quantity": 1, "unit_price_cents": 200}]
    assert quantities(lines) == {"A": 3, "B": 1}
    assert recompute_total(lines) == 3 * 150 + 200


# --------------------------------------------------------------------------- #
# Settlement / freshness / anomaly (pure, fake clock)
# --------------------------------------------------------------------------- #

class FakeClock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        return self.t

    def sleep_ms(self, ms):
        self.t += ms


def _verifier(reader, settle_ms=2000, interval_ms=100):
    clock = FakeClock()
    v = Verifier(reader, settle_ms=settle_ms, interval_ms=interval_ms,
                 now_ms_fn=clock.now_ms, sleep_ms_fn=clock.sleep_ms)
    v.known_version = 1
    v.baseline = {"version": 0, "lines": [], "total_cents": 0}
    return v


def _ledger_one_item(quantity=1):
    return Ledger.build([{"type": "item_quantity", "sku": "X", "quantity": quantity}])


def _round(version, qty):
    lines = [{"sku": "X", "quantity": qty, "unit_price_cents": 100}] if qty else []
    total = qty * 100
    return ({"version": version, "lines": lines, "total_cents": total},
            {"version": version, "lines": lines, "total_cents": total})


def test_stale_view_never_upgrades():
    def reader(timeout_ms):
        return _round(0, 0)  # always stale (version 0 < known 1)

    v = _verifier(reader, settle_ms=500)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "PARTIAL"
    assert report["constraints"][0]["state"] == UNKNOWN  # never contradicted


def test_fresh_view_arriving_mid_window_settles():
    calls = {"n": 0}

    def reader(timeout_ms):
        calls["n"] += 1
        if calls["n"] <= 3:
            return _round(0, 0)
        return _round(1, 1)

    v = _verifier(reader, settle_ms=2000)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "COMPLETE"
    assert report["constraints"][0]["state"] == CONFIRMED


def test_deadline_expiry_leaves_unknown():
    def reader(timeout_ms):
        return _round(0, 0)

    v = _verifier(reader, settle_ms=100, interval_ms=100)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "PARTIAL"
    assert report["constraints"][0]["state"] == UNKNOWN


def test_anomaly_on_unexplained_version_advance():
    def reader(timeout_ms):
        return _round(2, 2)  # version 2 exceeds known_version 1

    v = _verifier(reader)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "PARTIAL"
    assert report["constraints"][0]["state"] == UNKNOWN
    assert any(a["kind"] == "external_version_advance" for a in report["anomalies"])
    assert "quantity 2 vs required 1" in report["constraints"][0]["reason"]


def test_conflicting_dom_api_sources_leave_unknown():
    def reader(timeout_ms):
        return ({"version": 1, "lines": [], "total_cents": 0},
                {"version": 1, "lines": [{"sku": "X", "quantity": 1, "unit_price_cents": 100}],
                 "total_cents": 100})

    v = _verifier(reader, settle_ms=300)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "PARTIAL"
    assert report["constraints"][0]["state"] == UNKNOWN
    assert "conflicting DOM/API" in report["constraints"][0]["reason"]


# --------------------------------------------------------------------------- #
# Per-action version tracking
# --------------------------------------------------------------------------- #

def _on_action_verifier():
    v = Verifier(lambda timeout_ms: _round(0, 0))
    v.known_version = 0
    v.baseline = {"version": 0, "lines": [], "total_cents": 0}
    return v


def test_on_action_committed_advances_version():
    v = _on_action_verifier()
    v.on_action({"action": "add_to_cart", "response_status": 200, "response_version": 1})
    assert v.known_version == 1
    assert v.anomalies == []


def test_on_action_silent_drop_mismatch():
    v = _on_action_verifier()
    v.on_action({"action": "add_to_cart", "response_status": 200, "response_version": 0})
    assert v.known_version == 0
    assert any(a["kind"] == "action_outcome_mismatch" for a in v.anomalies)


def test_on_action_external_version_jump_enters_epoch():
    v = _on_action_verifier()
    v.on_action({"action": "add_to_cart", "response_status": 200, "response_version": 5})
    assert v.known_version == 5
    assert v.anomaly_epoch
    assert any(a["kind"] == "external_version_advance" for a in v.anomalies)


def test_on_action_409_with_current_version_above_known_enters_epoch():
    v = _on_action_verifier()
    v.on_action({"action": "add_to_cart", "response_status": 409,
                 "response_version": None, "response_current_version": 3})
    assert v.known_version == 3
    assert v.anomaly_epoch


def test_on_action_unknown_commit_keeps_version():
    v = _on_action_verifier()
    v.on_action({"action": "add_to_cart", "response_status": None, "response_version": None})
    assert v.known_version == 0
    assert any(a["kind"] == "unknown_commit" for a in v.anomalies)


# --------------------------------------------------------------------------- #
# Regression tests (brief 2 follow-up)
# --------------------------------------------------------------------------- #

def test_version_skew_pair_is_stale_not_conflict():
    calls = {"n": 0}

    def reader(timeout_ms):
        calls["n"] += 1
        if calls["n"] == 1:
            # skewed: dom@v0 (below known 1), api@v1 (== known)
            return ({"version": 0, "lines": [], "total_cents": 0},
                    {"version": 1, "lines": [{"sku": "X", "quantity": 1, "unit_price_cents": 100}],
                     "total_cents": 100})
        return _round(1, 1)

    v = _verifier(reader, settle_ms=2000)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "COMPLETE"
    assert all("conflict" not in c["reason"].lower() for c in report["constraints"])


def test_version_skew_pair_perpetual_is_expiry_not_conflict():
    def reader(timeout_ms):
        return ({"version": 0, "lines": [], "total_cents": 0},
                {"version": 1, "lines": [{"sku": "X", "quantity": 1, "unit_price_cents": 100}],
                 "total_cents": 100})

    v = _verifier(reader, settle_ms=500)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "PARTIAL"
    assert report["constraints"][0]["state"] == UNKNOWN
    assert "conflicting" not in report["constraints"][0]["reason"].lower()


def test_skew_above_known_still_records_anomaly():
    def reader(timeout_ms):
        return ({"version": 2, "lines": [{"sku": "X", "quantity": 2, "unit_price_cents": 100}],
                 "total_cents": 200},
                {"version": 1, "lines": [{"sku": "X", "quantity": 1, "unit_price_cents": 100}],
                 "total_cents": 100})

    v = _verifier(reader)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert v.anomaly_epoch
    assert any(a["kind"] == "external_version_advance" for a in report["anomalies"])
    assert report["verdict"] == "PARTIAL"


def test_malformed_sample_is_unknown_not_conflict():
    def reader(timeout_ms):
        return ({"lines": "not-a-list"},
                {"version": 1, "lines": [{"sku": "X", "quantity": 1, "unit_price_cents": 100}],
                 "total_cents": 100})

    v = _verifier(reader, settle_ms=300)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["constraints"][0]["state"] == UNKNOWN
    assert "conflicting" not in report["constraints"][0]["reason"].lower()
    assert ("malformed" in report["constraints"][0]["reason"].lower()
            or "insufficient" in report["constraints"][0]["reason"].lower())


def test_slow_reader_cannot_confirm_after_expiry():
    clock = FakeClock()

    def reader(timeout_ms):
        clock.t += 500  # the read "blocks" past the 100 ms budget
        return _round(1, 1)

    v = Verifier(reader, settle_ms=100, interval_ms=100,
                 now_ms_fn=clock.now_ms, sleep_ms_fn=clock.sleep_ms)
    v.known_version = 1
    v.baseline = {"version": 0, "lines": [], "total_cents": 0}
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "PARTIAL"
    assert report["constraints"][0]["state"] == UNKNOWN
    assert "deadline" in report["constraints"][0]["reason"].lower()


def test_anomaly_epoch_blocks_confirmation_after_reconcile():
    clock = FakeClock()
    v = Verifier(lambda timeout_ms: _round(3, 1), settle_ms=2000,
                 now_ms_fn=clock.now_ms, sleep_ms_fn=clock.sleep_ms)
    v.known_version = 1
    v.baseline = {"version": 0, "lines": [], "total_cents": 0}
    v.on_action({"action": "add_to_cart", "response_status": 200, "response_version": 3})
    assert v.anomaly_epoch
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "PARTIAL"
    assert report["constraints"][0]["state"] == UNKNOWN
    assert "anomaly epoch" in report["constraints"][0]["reason"].lower()


def test_provenance_references_recorded_samples():
    def reader(timeout_ms):
        return _round(1, 1)

    v = _verifier(reader)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "COMPLETE"
    recorded = {(s["source"], s["version"], s["timestamp"]) for s in report["samples"]}
    for c in report["constraints"]:
        for p in c["provenance"]:
            assert (p["source"], p["version"], p["timestamp"]) in recorded


def test_timestamps_do_not_affect_settlement():
    calls = {"n": 0}

    def reader(timeout_ms):
        calls["n"] += 1
        ts = f"2026-10-28T00:00:{calls['n']:02d}+00:00"
        lines = [{"sku": "X", "quantity": 1, "unit_price_cents": 100}]
        return ({"version": 1, "lines": lines, "total_cents": 100, "timestamp": ts},
                {"version": 1, "lines": lines, "total_cents": 100, "timestamp": ts})

    v = _verifier(reader)
    report = v.finish(_ledger_one_item(), {"X": 1})
    assert report["verdict"] == "COMPLETE"


# --------------------------------------------------------------------------- #
# Adapter response_version extraction (integration: real store + browser)
# --------------------------------------------------------------------------- #

def _goto_target(session):
    session.search("Hiker Water Bottle")
    session.open_product("bottle-hiker")
    session.select_variant(SKU_NAVYBLUE)


def test_add_response_version_from_http_response(store, browser_session):
    s = browser_session
    _goto_target(s)
    ev = s.add_to_cart(1)
    assert ev.payload["response_status"] == 200
    assert ev.payload["response_version"] == 1  # read from the HTTP body, not the DOM


def test_add_response_version_silent_drop_unchanged(store, browser_session):
    s = browser_session
    httpx.post(f"{store['control']}/_control/fault",
               json={"name": "silent-drop", "params": {}}).raise_for_status()
    _goto_target(s)
    ev = s.add_to_cart(1)
    assert ev.payload["response_status"] == 200
    assert ev.payload["response_version"] == 0  # unchanged version under silent-drop
    httpx.post(f"{store['control']}/_control/reset").raise_for_status()

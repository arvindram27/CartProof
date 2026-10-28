"""Focused regressions for the five brief-3 recovery corrections (28 Oct 2026).

Every test here exercises the real controller methods. Where a write behaviour
matters, a controlled adapter boundary (fake session) records the actual calls so
the assertion is about what was dispatched, not about an internal counter alone.
Persisted store state is never imported by the controller; the integration-style
assertions live in ``tests/test_recovery.py``.
"""
from __future__ import annotations

import copy

import pytest

from core.ledger import Ledger
from core.recovery import (
    MAX_WRITES_PER_RUN,
    RECOVERABLE_KINDS,
    RecoveryController,
    UnsafeState,
    _coherent_full,
    _signature_full,
    _snapshot,
    establish_recovery_baseline,
    render_recovery_report,
)
from core.verify import Verifier


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _line(lid, sku, qty, price=100):
    return {"line_id": lid, "sku": sku, "quantity": qty, "unit_price_cents": price}


def _view(version, lines):
    return {
        "version": version,
        "lines": lines,
        "total_cents": sum(l["quantity"] * l["unit_price_cents"] for l in lines),
    }


class _Ev:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return {"payload": self.payload}


class FakeVerifier:
    """Scripted ``settle`` with no reader, so no observation wrapper is installed."""

    def __init__(self, views=None):
        self.views = list(views or [])
        self.known_version = 0
        self.samples = []
        self.anomaly_epoch = False
        self.settle_ms = 2000
        self.phases = []

    def settle(self, known_version=None, deadline_ms=None, phase="recovery", **_kwargs):
        self.phases.append(phase)
        if not self.views:
            return {"status": "timeout", "view": None, "provenance": (), "sample_ids": []}
        view = self.views.pop(0)
        return {"status": "settled", "view": view, "provenance": (), "sample_ids": []}


class FakeSession:
    """Controlled adapter boundary: records every call that would reach HTTP."""

    def __init__(self, *, page_version=0, add_payload=None, delete_payload=None,
                 patch_payload=None, raise_on=None):
        self.page_version = page_version
        self.payloads = {"add": add_payload, "delete": delete_payload,
                         "patch": patch_payload}
        self.raise_on = set(raise_on or ())
        self.calls = []

    def open_product(self, product_id):
        self.calls.append("open_product")
        return _Ev({"action": "open_product", "product_id": product_id,
                    "cart_version": self.page_version})

    def select_variant(self, sku):
        self.calls.append("select_variant")
        return _Ev({"action": "select_variant", "sku": sku})

    def _mutate(self, kind):
        self.calls.append(kind)
        if kind in self.raise_on:
            raise RuntimeError(f"transport failed during {kind}")
        spec = self.payloads.get(kind)
        payload = spec() if callable(spec) else dict(spec or {})
        return _Ev(payload)

    def add_to_cart(self, quantity, sku=None):
        return self._mutate("add")

    def set_line_quantity(self, line_id, sku, quantity, expected_version):
        return self._mutate("patch")

    def delete_line(self, line_id, sku, expected_version):
        return self._mutate("delete")


def _controller(verifier, session, baseline=None):
    ledger = Ledger.build([{"type": "item_quantity", "sku": "X", "quantity": 1}])
    return RecoveryController(verifier, session, ledger, {},
                              baseline or _view(0, []))


# --------------------------------------------------------------------------- #
# 1. Full-state attribution: nothing short of exact identity is ownership
# --------------------------------------------------------------------------- #

def test_unrelated_line_id_replacement_is_never_a_correction():
    c = _controller(FakeVerifier(), FakeSession())
    before = _view(0, [_line("l1", "X", 1)])
    after = _view(1, [_line("l2", "X", 2)])  # ID replaced, aggregate looks like +1
    incident = c._diagnose({"requested_sku": "X", "quantity": 1, "response_version": 1},
                           before, after)
    assert incident["kind"] == "unsupported"
    assert incident["kind"] not in RECOVERABLE_KINDS


def test_price_change_on_an_unrelated_line_stops_attribution():
    c = _controller(FakeVerifier(), FakeSession())
    before = _view(0, [_line("l1", "X", 1), _line("l2", "Y", 1, price=100)])
    after = _view(1, [_line("l1", "X", 1), _line("l2", "Y", 1, price=150)])
    incident = c._diagnose({"requested_sku": "X", "quantity": 1, "response_version": 1},
                           before, after)
    assert incident["kind"] == "unsupported"
    assert "changed" in incident["detail"]


def test_same_aggregate_quantities_with_churned_ids_is_not_a_silent_drop():
    c = _controller(FakeVerifier(), FakeSession())
    before = _view(3, [_line("l1", "X", 1), _line("l2", "Y", 1)])
    after = _view(3, [_line("l9", "X", 1), _line("l8", "Y", 1)])
    incident = c._diagnose({"requested_sku": "X", "quantity": 1, "response_version": 3},
                           before, after)
    assert incident["kind"] == "unknown_commit"
    assert incident["kind"] not in RECOVERABLE_KINDS


@pytest.mark.parametrize("view", [
    _view(0, [_line("l1", "X", 1), _line("l1", "Y", 1)]),          # duplicate ID
    {"version": 0, "lines": [{"sku": "X", "quantity": 1,
                              "unit_price_cents": 100}], "total_cents": 100},  # no ID
    _view(0, [_line("", "X", 1)]),                                  # empty ID
    {"version": 0, "lines": [_line("l1", "X", 1)], "total_cents": 99},  # total mismatch
    {"version": True, "lines": [], "total_cents": 0},               # boolean version
    {"version": 0, "lines": [{"line_id": "l1", "sku": "X", "quantity": True,
                              "unit_price_cents": 100}], "total_cents": 100},
    {"version": 0, "lines": [{"line_id": "l1", "sku": "X", "quantity": 0,
                              "unit_price_cents": 100}], "total_cents": 0},
    {"version": 0, "lines": [_line("l1", "X", 1)], "total_cents": 100.0},
])
def test_invalid_or_ambiguous_identity_is_unsafe(view):
    with pytest.raises(UnsafeState):
        _snapshot(view)


def test_wrong_variant_reusing_a_baseline_sku_is_not_owned():
    baseline = _view(0, [_line("l1", "X", 1), _line("l9", "W", 1)])
    c = _controller(FakeVerifier(), FakeSession(), baseline=baseline)
    before = _view(0, [_line("l1", "X", 1), _line("l9", "W", 1)])
    after = _view(1, [_line("l1", "X", 1), _line("l9", "W", 1),
                      _line("w1", "W", 1)])  # W is seeded, so not newly owned
    incident = c._diagnose({"requested_sku": "X", "quantity": 1, "response_version": 1},
                           before, after)
    assert incident["kind"] == "ownership_uncertain"


def test_seeded_requested_sku_merge_of_exactly_the_request_is_correct():
    """Clarified legitimate addition: clean task3 allows NAVYBLUE 1 -> 2."""
    baseline = _view(0, [_line("l1", "X", 1)])
    c = _controller(FakeVerifier(), FakeSession(), baseline=baseline)
    before = _view(1, [_line("l1", "X", 1)])
    after = _view(2, [_line("l1", "X", 2)])
    incident = c._diagnose({"requested_sku": "X", "quantity": 1, "response_version": 2},
                           before, after)
    assert incident["kind"] == "correct"


def test_merged_requested_sku_with_extra_units_is_not_ours_to_reduce():
    baseline = _view(0, [_line("l1", "X", 1)])
    c = _controller(FakeVerifier(), FakeSession(), baseline=baseline)
    before = _view(1, [_line("l1", "X", 1)])
    after = _view(2, [_line("l1", "X", 3)])  # +2 for a +1 request
    incident = c._diagnose({"requested_sku": "X", "quantity": 1, "response_version": 2},
                           before, after)
    assert incident["kind"] == "ownership_uncertain"
    assert "already present before add" in incident["detail"]


def test_delete_commit_with_wrong_observed_effect_sends_no_replacement_post():
    before = _view(0, [_line("l1", "X", 1)])
    after_add = _view(1, [_line("l1", "X", 1), _line("w1", "W", 1)])
    # The DELETE committed (200/version 2) but also changed a neighbour.
    wrong_effect = _view(2, [_line("l1", "X", 2)])
    session = FakeSession(delete_payload={"response_status": 200, "response_version": 2})
    verifier = FakeVerifier([wrong_effect])
    c = _controller(verifier, session, baseline=before)
    c.verifier.known_version = 1
    c._before_snap = _snapshot(before)
    c._last_snap = _snapshot(after_add)
    c.last_state = after_add
    incident = {"kind": "wrong_variant", "req_sku": "X", "req_qty": 1,
                "wrong_sku": "W", "wrong_qty": 1, "line_id": "w1",
                "action_id": "add-0", "incident_id": "i1"}

    c.recover("add-0", {"product_id": "p", "sku": "X", "quantity": 1}, incident)

    assert session.calls == ["delete"]
    assert c.writes_attempted == 1
    assert any("DELETE did not remove exactly" in b["detail"] for b in c.blocking)


def test_patch_commit_with_collateral_change_is_not_success():
    pre_patch = _view(1, [_line("l1", "X", 2), _line("l2", "P", 1, price=100)])
    after_patch = _view(2, [_line("l1", "X", 1), _line("l2", "P", 1, price=150)])
    session = FakeSession(patch_payload={"response_status": 200, "response_version": 2})
    c = _controller(FakeVerifier([after_patch]), session, baseline=pre_patch)
    c.verifier.known_version = 1
    c._last_snap = _snapshot(pre_patch)
    c.last_state = pre_patch
    incident = {"kind": "excess_quantity", "req_sku": "X", "req_qty": 1,
                "sku": "X", "intended": 1, "line_id": "l1",
                "action_id": "add-0", "incident_id": "i1"}

    c.recover("add-0", {"product_id": "p", "sku": "X", "quantity": 1}, incident)

    assert session.calls == ["patch"]
    assert not c.resolved
    assert any("planned PATCH effect" in b["detail"] for b in c.blocking)


def test_patch_proof_preserves_protected_seeded_lines():
    pre_patch = _view(1, [_line("l1", "X", 2), _line("l2", "SEED", 1)])
    after_patch = _view(2, [_line("l1", "X", 1), _line("l2", "SEED", 1)])
    session = FakeSession(patch_payload={"response_status": 200, "response_version": 2})
    c = _controller(FakeVerifier([after_patch]), session, baseline=pre_patch)
    c.verifier.known_version = 1
    c._last_snap = _snapshot(pre_patch)
    c.last_state = pre_patch
    incident = {"kind": "excess_quantity", "req_sku": "X", "req_qty": 1,
                "sku": "X", "intended": 1, "line_id": "l1",
                "action_id": "add-0", "incident_id": "i1"}

    c.recover("add-0", {"product_id": "p", "sku": "X", "quantity": 1}, incident)

    assert c.resolved and c.resolved[0]["kind"] == "excess_quantity"
    assert c.resolved[0]["observed_lines"]["l2"] == {
        "sku": "SEED", "quantity": 1, "unit_price_cents": 100}


# --------------------------------------------------------------------------- #
# 2. Action-relative targets (no absolute-quantity success check, no KeyError)
# --------------------------------------------------------------------------- #

def test_plan_achieved_is_relative_to_a_seeded_before_state():
    c = _controller(FakeVerifier(), FakeSession())
    incident = {"kind": "silent_drop", "req_sku": "X", "req_qty": 1}
    before = _snapshot(_view(0, [_line("l1", "X", 1)]))
    after = _snapshot(_view(1, [_line("l1", "X", 2)]))
    assert c._plan_achieved(incident, before, after) is True


def test_plan_achieved_for_a_second_action_on_an_already_added_sku():
    c = _controller(FakeVerifier(), FakeSession())
    incident = {"kind": "silent_drop", "req_sku": "X", "req_qty": 1}
    before = _snapshot(_view(2, [_line("l1", "X", 2)]))
    after = _snapshot(_view(3, [_line("l1", "X", 3)]))
    assert c._plan_achieved(incident, before, after) is True
    # the old absolute rule (qty == req_qty) must not be used any more
    assert c._plan_achieved(incident, before, _snapshot(_view(2, [_line("l1", "X", 2)]))) is False


def test_plan_achieved_rejects_a_neighbour_price_change():
    c = _controller(FakeVerifier(), FakeSession())
    incident = {"kind": "silent_drop", "req_sku": "X", "req_qty": 1}
    before = _snapshot(_view(0, [_line("l1", "X", 1), _line("l2", "P", 1, price=100)]))
    after = _snapshot(_view(1, [_line("l1", "X", 2), _line("l2", "P", 1, price=150)]))
    assert c._plan_achieved(incident, before, after) is False


def test_plan_achieved_wrong_variant_requires_the_wrong_line_gone():
    c = _controller(FakeVerifier(), FakeSession())
    incident = {"kind": "wrong_variant", "req_sku": "X", "req_qty": 1,
                "wrong_sku": "W", "wrong_qty": 1, "line_id": "w1"}
    before = _snapshot(_view(0, [_line("l1", "X", 1)]))
    still_there = _snapshot(_view(2, [_line("l1", "X", 2), _line("w1", "W", 1)]))
    assert c._plan_achieved(incident, before, still_there) is False


def test_retry_diagnosed_correct_resolves_without_keyerror(monkeypatch):
    """The reported crash: ``correct`` has no ``detail``. It must resolve."""
    payload = {"action": "add_to_cart", "requested_sku": "X", "quantity": 1,
               "sent_version": 0, "sent_sku": "X", "sent_quantity": 1,
               "response_status": 200, "response_version": 1}
    session = FakeSession(page_version=0, add_payload=payload)
    c = _controller(FakeVerifier([_view(1, [_line("l1", "X", 1)])]), session)
    c.verifier.known_version = 0
    c._before_snap = _snapshot(_view(0, []))
    c._before = _view(0, [])
    c.last_state = _view(0, [])
    c._last_snap = _snapshot(_view(0, []))
    monkeypatch.setattr(c, "_plan_achieved", lambda *a, **k: False)
    incident = {"kind": "silent_drop", "req_sku": "X", "req_qty": 1,
                "action_id": "add-0", "incident_id": "i1"}

    c.recover("add-0", {"product_id": "p", "sku": "X", "quantity": 1}, incident)

    assert c.resolved and c.resolved[0]["kind"] == "silent_drop"
    assert not c.blocking
    assert c.writes_attempted == 1


# --------------------------------------------------------------------------- #
# 3. Strict response typing, dispatch accounting, caps
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("status,version", [
    (True, 1), (200.0, 1), (200, True), (200, 1.0), (200, -1), (201, 1),
    (302, 1), (422, 1), (None, 1), (500, 1),
])
def test_only_strict_200_and_genuine_int_version_can_commit(status, version):
    c = _controller(FakeVerifier(), FakeSession())
    c.verifier.known_version = 0
    step = {"action_id": "a", "plan_id": "p", "operation": "add"}
    outcome = c._classify_correction(
        {"response_status": status, "response_version": version}, 0, step)
    assert outcome != "committed"
    assert c.verifier.known_version == 0
    assert c.blocking


def test_strict_200_with_the_expected_increment_commits():
    c = _controller(FakeVerifier(), FakeSession())
    c.verifier.known_version = 0
    step = {"action_id": "a", "plan_id": "p", "operation": "add"}
    outcome = c._classify_correction(
        {"response_status": 200, "response_version": 1}, 0, step)
    assert outcome == "committed"
    assert c.verifier.known_version == 1


def test_boolean_version_is_rejected_before_any_correction():
    c = _controller(FakeVerifier(), FakeSession())
    c.verifier.known_version = 0
    c._before_snap = _snapshot(_view(0, []))
    c._before = _view(0, [])
    result = c.diagnose("add-0", {"sku": "X", "quantity": 1},
                        {"sent_version": 0, "sent_sku": "X", "sent_quantity": 1,
                         "response_status": 200, "response_version": True})
    assert result["_stop"]["stop"] is True
    assert "invalid response version" in result["_stop"]["reason"]
    assert c.writes_attempted == 0


def test_missing_outgoing_capture_counts_one_and_stays_uncertain():
    payload = {"action": "add_to_cart", "requested_sku": "X",
               "sent_version": None, "sent_sku": None, "sent_quantity": None,
               "response_status": 200, "response_version": 1}
    session = FakeSession(page_version=0, add_payload=payload)
    c = _controller(FakeVerifier(), session)
    c.verifier.known_version = 0

    result = c._corrective_add("add-0", {"product_id": "p", "sku": "X", "quantity": 1})

    assert result["outcome"] == "uncertain"
    assert session.calls == ["open_product", "select_variant", "add"]
    assert c.writes_attempted == 1
    step = c.audit[-1]
    assert step["dispatched"] is True
    assert step["response"]["sent_version"] is None
    assert any("sent version" in b["detail"] for b in c.blocking)


def test_adapter_exception_after_dispatch_is_counted_and_retained():
    session = FakeSession(raise_on={"patch"})
    c = _controller(FakeVerifier(), session, baseline=_view(1, [_line("l1", "X", 2)]))
    c.verifier.known_version = 1
    c._last_snap = _snapshot(_view(1, [_line("l1", "X", 2)]))
    c.last_state = _view(1, [_line("l1", "X", 2)])
    incident = {"kind": "excess_quantity", "req_sku": "X", "req_qty": 1,
                "sku": "X", "intended": 1, "line_id": "l1",
                "action_id": "add-0", "incident_id": "i1"}

    c.recover("add-0", {"product_id": "p", "sku": "X", "quantity": 1}, incident)

    assert c.writes_attempted == 1
    step = c.audit[-1]
    assert step["operation"] == "patch"
    assert step["dispatched"] is True
    assert step["outcome"] == "unknown"
    assert "transport failed" in step["error"]
    assert any("raised" in b["detail"] for b in c.blocking)


def test_preflight_stop_before_dispatch_counts_zero():
    session = FakeSession(page_version=7)  # page version != proven version
    c = _controller(FakeVerifier(), session)
    c.verifier.known_version = 0

    result = c._corrective_add("add-0", {"product_id": "p", "sku": "X", "quantity": 1})

    assert result is None
    assert c.writes_attempted == 0
    assert c.audit == []
    assert session.calls == ["open_product"]
    assert any("page version" in b["detail"] for b in c.blocking)


def _excess_incident(i):
    return {"kind": "excess_quantity", "req_sku": "X", "req_qty": 1,
            "sku": "X", "intended": 1, "line_id": f"l{i}",
            "action_id": f"add-{i}", "incident_id": f"i{i}"}


def _lines_with(qty_overrides):
    return [_line(f"l{i}", "X", qty_overrides.get(i, 1)) for i in range(1, 6)]


def test_global_write_cap_is_exhausted_by_real_dispatches():
    expected = {"version": 0}
    session = FakeSession(patch_payload=lambda: {
        "response_status": 200, "response_version": expected["version"] + 1})
    views = []
    qtys = {i: 1 for i in range(1, 6)}
    for i in (1, 2, 3, 4):
        views.append(_view(i, _lines_with(qtys)))
    c = _controller(FakeVerifier(views), session)
    c.verifier.known_version = 0

    for i in (1, 2, 3, 4):
        qtys[i] = 2
        c._last_snap = _snapshot(_view(i, _lines_with(qtys)))
        c.last_state = _view(i, _lines_with(qtys))
        c.verifier.known_version = i
        expected["version"] = i
        c.recover(f"add-{i}", {"product_id": "p", "sku": "X", "quantity": 1},
                  _excess_incident(i))
        qtys[i] = 1

    assert c.writes_attempted == MAX_WRITES_PER_RUN
    assert len(c.resolved) == 4
    calls_before = len(session.calls)

    # Fifth target: the real global cap must refuse the write.
    qtys[5] = 2
    c._last_snap = _snapshot(_view(4, _lines_with(qtys)))
    c.verifier.known_version = 4
    c.recover("add-5", {"product_id": "p", "sku": "X", "quantity": 1},
              _excess_incident(5))

    assert c.writes_attempted == MAX_WRITES_PER_RUN
    assert len(session.calls) == calls_before  # no PATCH reached the adapter
    assert any("write budget exhausted" in b["detail"] for b in c.blocking)


def test_two_step_plan_cannot_start_with_only_one_slot_left():
    expected = {"version": 0}
    session = FakeSession(patch_payload=lambda: {
        "response_status": 200, "response_version": expected["version"] + 1})
    views = []
    qtys = {i: 1 for i in range(1, 6)}
    for i in (1, 2, 3):
        views.append(_view(i, _lines_with(qtys)))
    c = _controller(FakeVerifier(views), session)
    c.verifier.known_version = 0

    for i in (1, 2, 3):
        qtys[i] = 2
        c._last_snap = _snapshot(_view(i, _lines_with(qtys)))
        c.last_state = _view(i, _lines_with(qtys))
        c.verifier.known_version = i
        expected["version"] = i
        c.recover(f"add-{i}", {"product_id": "p", "sku": "X", "quantity": 1},
                  _excess_incident(i))
        qtys[i] = 1

    assert c.writes_attempted == MAX_WRITES_PER_RUN - 1
    calls_before = len(session.calls)
    incident = {"kind": "wrong_variant", "req_sku": "X", "req_qty": 1,
                "wrong_sku": "W", "wrong_qty": 1, "line_id": "w1",
                "action_id": "add-9", "incident_id": "i9"}

    c.recover("add-9", {"product_id": "p", "sku": "X", "quantity": 1}, incident)

    assert session.calls == session.calls[:calls_before]  # no DELETE/ADD dispatched
    assert c.writes_attempted == MAX_WRITES_PER_RUN - 1
    assert any("write budget exhausted (need 2" in b["detail"] for b in c.blocking)


# --------------------------------------------------------------------------- #
# 2 (clarification). Full-state settlement: three identical eligible pairs
# --------------------------------------------------------------------------- #

def _run_settle(scripted, *, full_state: bool, settle_ms=2000):
    clock = {"t": 0}
    reads = list(scripted)

    def now_ms():
        return clock["t"]

    def sleep_ms(ms):
        clock["t"] += ms

    def reader(timeout_ms):
        clock["t"] += 10
        if not reads:
            raise AssertionError("reader script exhausted")
        return reads.pop(0)

    verifier = Verifier(reader, settle_ms=settle_ms, interval_ms=100,
                        now_ms_fn=now_ms, sleep_ms_fn=sleep_ms)
    kwargs = {"coherence": _coherent_full, "signature": _signature_full} if full_state else {}
    res = verifier.settle(1, phase="test", **kwargs)
    return verifier, res


def _pair(version, lid, price=100):
    view = _view(version, [_line(lid, "X", 1, price=price)])
    return (view, view)


def test_full_state_settle_ignores_churn_and_needs_three_identical_pairs():
    # Three identity/price churns that look identical in SKU/quantity/total,
    # then three genuinely identical full-state pairs.
    scripted = [_pair(1, "a"), _pair(1, "b"), _pair(1, "c"),
                _pair(1, "d"), _pair(1, "d"), _pair(1, "d")]
    verifier, res = _run_settle(scripted, full_state=True)

    assert res["status"] == "settled"
    assert len(verifier.samples) == 12  # exactly 6 reads: no early confirmation
    cited = {s["seq"] for s in res["provenance"]}
    assert cited == {7, 8, 9, 10, 11, 12}
    assert all(s["version"] == 1 for s in res["provenance"])


def test_full_state_settle_never_confirms_on_a_single_trailing_valid_pair():
    # The first two pairs disagree between DOM and API on line identity; only
    # the last three agree. Confirmation must cite all three, not the last pair.
    mismatch = (_view(1, [_line("d", "X", 1)]), _view(1, [_line("e", "X", 1)]))
    scripted = [mismatch, mismatch, _pair(1, "d"), _pair(1, "d"), _pair(1, "d")]
    verifier, res = _run_settle(scripted, full_state=True)

    assert res["status"] == "settled"
    assert len(verifier.samples) == 10  # 5 reads; pairs 1-2 could not confirm
    assert [s["seq"] for s in res["provenance"]] == [5, 6, 7, 8, 9, 10]


def test_closed_verification_only_settlement_keeps_its_own_rules():
    # Same churn script without recovery comparators: the closed SKU/quantity/
    # total rules still settle after three aggregate-identical pairs.
    scripted = [_pair(1, "a"), _pair(1, "b"), _pair(1, "c"),
                _pair(1, "d"), _pair(1, "d"), _pair(1, "d")]
    verifier, res = _run_settle(scripted, full_state=False)

    assert res["status"] == "settled"
    assert len(verifier.samples) == 6  # three aggregate-identical pairs


# --------------------------------------------------------------------------- #
# 4. Ownership key namespaces: earlier actions and baseline ID reuse
# --------------------------------------------------------------------------- #

def test_wrong_variant_with_earlier_action_same_sku_is_not_owned():
    """SKU Y already exists from an earlier action; a second Y line is not ours."""
    c = _controller(FakeVerifier(), FakeSession())
    before = _view(0, [_line("ly", "Y", 1)])
    after = _view(1, [_line("ly", "Y", 1), _line("ly2", "Y", 1)])
    incident = c._diagnose({"requested_sku": "X", "quantity": 1, "response_version": 1},
                           before, after)
    assert incident["kind"] == "ownership_uncertain"
    assert incident["kind"] not in RECOVERABLE_KINDS

    c.recover("add-0", {"product_id": "p", "sku": "X", "quantity": 1}, incident)
    assert c.writes_attempted == 0
    assert c.session.calls == []          # zero corrections dispatched


def test_excess_with_earlier_action_same_sku_is_not_owned():
    """The requested SKU already exists from an earlier action; an extra new line
    for the same SKU is not a newly-owned excess."""
    c = _controller(FakeVerifier(), FakeSession())
    before = _view(0, [_line("lx1", "X", 1)])
    after = _view(1, [_line("lx1", "X", 1), _line("lx2", "X", 2)])
    incident = c._diagnose({"requested_sku": "X", "quantity": 1, "response_version": 1},
                           before, after)
    assert incident["kind"] == "ownership_uncertain"
    assert incident["kind"] not in RECOVERABLE_KINDS

    c.recover("add-0", {"product_id": "p", "sku": "X", "quantity": 1}, incident)
    assert c.writes_attempted == 0
    assert c.session.calls == []          # zero corrections dispatched


def test_baseline_line_id_reused_under_different_sku_is_not_owned():
    """An original baseline line ID must never be re-owned under another SKU,
    even when that ID is absent from the immediate before-state."""
    baseline = _view(0, [_line("l1", "SEED", 1)])
    c = _controller(FakeVerifier(), FakeSession(), baseline=baseline)
    before = _view(0, [])                 # seeded line absent from the immediate view
    after = _view(1, [_line("l1", "Y", 1)])
    incident = c._diagnose({"requested_sku": "X", "quantity": 1, "response_version": 1},
                           before, after)
    assert incident["kind"] == "ownership_uncertain"
    assert incident["kind"] not in RECOVERABLE_KINDS

    c.recover("add-0", {"product_id": "p", "sku": "X", "quantity": 1}, incident)
    assert c.writes_attempted == 0
    assert c.session.calls == []          # zero corrections dispatched


# --------------------------------------------------------------------------- #
# 5. Strict full-state evidence: typed totals, complete lines, valid streaks
# --------------------------------------------------------------------------- #

def _scripted_verifier(scripted, settle_ms=2000):
    clock = {"t": 0}
    reads = list(scripted)

    def now_ms():
        return clock["t"]

    def sleep_ms(ms):
        clock["t"] += ms

    def reader(timeout_ms):
        clock["t"] += 10
        if not reads:
            raise AssertionError("reader script exhausted")
        return reads.pop(0)

    return Verifier(reader, settle_ms=settle_ms, interval_ms=100,
                    now_ms_fn=now_ms, sleep_ms_fn=sleep_ms)


def _pair_with_api_total(version, lid, api_total):
    view = _view(version, [_line(lid, "X", 1)])
    api = {"version": version, "lines": [dict(_line(lid, "X", 1))],
           "total_cents": api_total}
    return (view, api)


@pytest.mark.parametrize("bad_total", [100.0, True])
def test_invalid_total_in_an_earlier_pair_cannot_join_the_streak(bad_total):
    """A float/bool total equal by Python comparison is not eligible evidence, but
    a later genuine three-pair streak may still confirm within the deadline."""
    bad = _pair_with_api_total(1, "a", bad_total)
    assert _coherent_full(*bad)[0] is False

    scripted = [bad, _pair(1, "b"), _pair(1, "b"), _pair(1, "b")]
    verifier = _scripted_verifier(scripted)
    res = verifier.settle(1, phase="test", coherence=_coherent_full,
                          signature=_signature_full)

    assert res["status"] == "settled"
    assert [s["seq"] for s in res["provenance"]] == [3, 4, 5, 6, 7, 8]


@pytest.mark.parametrize("bad_lines", [None, "not-a-list", {"a": 1}])
def test_missing_or_malformed_lines_is_not_an_empty_cart(bad_lines):
    payload = {"version": 0, "lines": bad_lines, "total_cents": 0}
    assert _coherent_full(payload, copy.deepcopy(payload))[0] is False


# --------------------------------------------------------------------------- #
# 5 (diagnostic fix). A shared stale total is arithmetic, not a source conflict
# --------------------------------------------------------------------------- #

def _arithmetic_pair(version, lid, displayed=0):
    """DOM and API agree on version/lines/total, but the total is not the sum."""
    line = _line(lid, "X", 1)
    payload = {"version": version, "lines": [dict(line)], "total_cents": displayed}
    return (copy.deepcopy(payload), payload)


def test_shared_stale_total_is_arithmetic_not_a_channel_conflict():
    # Both sides display total 0 for a line that costs 100: the channels agree
    # with each other, so this is arithmetic inconsistency, not a DOM/API split.
    assert _coherent_full(*_arithmetic_pair(1, "a")) == (False, "arithmetic")


@pytest.mark.parametrize("dom_total,api_total,kind", [
    (100, 0, "total"),
    (0, 100, "total"),
    (100.0, 100.0, "malformed"),
    (True, True, "malformed"),
    (-100, -100, "malformed"),
])
def test_total_kinds_distinguish_source_conflict_from_malformed(dom_total, api_total, kind):
    line = _line("a", "X", 1)
    dom = {"version": 1, "lines": [dict(line)], "total_cents": dom_total}
    api = {"version": 1, "lines": [dict(line)], "total_cents": api_total}
    assert _coherent_full(dom, api) == (False, kind)


def test_arithmetic_stop_reason_cites_displayed_recomputed_and_observations():
    verifier = _scripted_verifier([_arithmetic_pair(1, "a")], settle_ms=100)
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "malformed"
    assert res["view"] is None
    assert [d["kind"] for d in res["incoherent"]] == ["arithmetic"]
    reason = c._settle_stop_reason(res["status"], res)
    assert reason == ("displayed total 0 != recomputed total 100 "
                      "(DOM/API agree on full state; observations #1/#2)")
    assert "conflicting DOM/API sources" not in reason
    # The completion report uses the same eligible evidence wording.
    assert c._completion_failure_reason("malformed", c.ledger.constraints[0], res) == reason


def test_true_dom_api_total_disagreement_keeps_the_conflict_reason():
    conflict = _pair_with_api_total(1, "a", 0)  # DOM 100, API 0, same lines
    verifier = _scripted_verifier([conflict], settle_ms=100)
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "conflict"
    assert c._settle_stop_reason(res["status"], res) == "conflicting DOM/API sources"


def test_malformed_totals_keep_the_malformed_reason():
    bad = _pair_with_api_total(1, "a", 100.0)
    verifier = _scripted_verifier([bad], settle_ms=100)
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "malformed"
    assert "incoherent" not in res
    assert c._settle_stop_reason(res["status"], res) == "malformed or insufficient evidence"


def test_a_late_arithmetic_pair_cannot_overwrite_the_deadline_reason():
    # The first read already lands past the deadline: it is not fresh evidence,
    # so it must neither be reported nor turn the stop into malformed.
    verifier = _scripted_verifier([_arithmetic_pair(1, "a")], settle_ms=5)
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "timeout"
    assert "incoherent" not in res
    assert c._settle_stop_reason(res["status"], res) == (
        "no fresh settled view within deadline (settlement timed out)")


def test_an_earlier_real_conflict_is_not_erased_by_later_arithmetic_trouble():
    conflict = _pair_with_api_total(1, "a", 0)
    verifier = _scripted_verifier([conflict, _arithmetic_pair(1, "a"),
                                   _arithmetic_pair(1, "a")],
                                  settle_ms=300)
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "conflict"
    assert res.get("incoherent")  # later arithmetic was seen ...
    assert c._settle_stop_reason(res["status"], res) == "conflicting DOM/API sources"


def test_a_later_valid_streak_can_still_settle_after_arithmetic_trouble():
    verifier = _scripted_verifier([_arithmetic_pair(1, "a")] + [_pair(1, "b")] * 3,
                                  settle_ms=2000)
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "settled"
    assert res["view"]["total_cents"] == 100
    assert [s["seq"] for s in res["provenance"]] == [3, 4, 5, 6, 7, 8]


def test_a_below_known_arithmetic_pair_is_not_fresh_diagnostic_evidence():
    # known_version=2 but the read is version 1: provably stale, so arithmetic
    # trouble in it must not become malformed evidence or a cited reason.
    verifier = _scripted_verifier([_arithmetic_pair(1, "a")], settle_ms=100)
    verifier.known_version = 2
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "timeout"
    assert res["view"] is None
    assert "incoherent" not in res
    assert c.session.calls == []
    assert c._settle_stop_reason(res["status"], res) == (
        "no fresh settled view within deadline (settlement timed out)")


def test_only_the_fresh_arithmetic_pair_supplies_the_reason_and_references():
    # A stale v1 arithmetic pair is ignored; two current v2 arithmetic pairs are
    # eligible and the first fresh one supplies the observation references.
    verifier = _scripted_verifier([_arithmetic_pair(1, "a"),
                                   _arithmetic_pair(2, "a"),
                                   _arithmetic_pair(2, "a")],
                                  settle_ms=250)
    verifier.known_version = 2
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "malformed"
    assert [(d["seq_dom"], d["seq_api"]) for d in res["incoherent"]] == [(3, 4), (5, 6)]
    assert c._settle_stop_reason(res["status"], res) == (
        "displayed total 0 != recomputed total 100 "
        "(DOM/API agree on full state; observations #3/#4)")


def test_stale_arithmetic_then_a_current_valid_streak_settles_only_on_current_pairs():
    verifier = _scripted_verifier([_arithmetic_pair(1, "a")] + [_pair(2, "b")] * 3,
                                  settle_ms=2000)
    verifier.known_version = 2
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "settled"
    assert res["view"]["version"] == 2
    assert "incoherent" not in res
    assert [s["seq"] for s in res["provenance"]] == [3, 4, 5, 6, 7, 8]


def test_read_timeout_keeps_deadline_treatment_without_arithmetic_diagnosis():
    timeout_pair = ({"read_timeout": True}, {"read_timeout": True})
    verifier = _scripted_verifier([timeout_pair], settle_ms=100)
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "timeout"
    assert "incoherent" not in res
    assert c._settle_stop_reason(res["status"], res) == (
        "no fresh settled view within deadline (settlement timed out)")


def test_above_known_arithmetic_still_hits_the_external_version_guard_first():
    verifier = _scripted_verifier([_arithmetic_pair(3, "a")], settle_ms=100)
    verifier.known_version = 2
    c = _controller(verifier, FakeSession())
    res = c._settle("test")

    assert res["status"] == "anomaly"
    assert verifier.anomaly_epoch is True
    assert any(a["kind"] == "external_version_advance" for a in verifier.anomalies)
    assert "incoherent" not in res
    assert c._settle_stop_reason(res["status"], res) == (
        "unexplained version advance during settle")


def test_absent_lines_key_is_not_an_empty_cart_or_a_baseline():
    payload = {"version": 0, "total_cents": 0}
    assert _coherent_full(payload, copy.deepcopy(payload))[0] is False

    # The same malformed pair repeated can never form a baseline streak.
    verifier = _scripted_verifier([(payload, dict(payload))] * 25)
    with pytest.raises(RuntimeError):
        establish_recovery_baseline(verifier)
    assert verifier.baseline is None


def test_whitespace_only_identity_is_rejected_without_normalizing_padding():
    blank_id = {"version": 0, "lines": [{"line_id": "   ", "sku": "X", "quantity": 1,
                                         "unit_price_cents": 100}], "total_cents": 100}
    assert _coherent_full(blank_id, copy.deepcopy(blank_id))[0] is False

    blank_sku = {"version": 0, "lines": [{"line_id": "l1", "sku": " \t ", "quantity": 1,
                                          "unit_price_cents": 100}], "total_cents": 100}
    assert _coherent_full(blank_sku, copy.deepcopy(blank_sku))[0] is False

    # Nonblank padded values are valid but preserved exactly, never stripped.
    padded = {"version": 0, "lines": [{"line_id": " l1 ", "sku": " X ", "quantity": 1,
                                       "unit_price_cents": 100}], "total_cents": 100}
    assert _coherent_full(padded, copy.deepcopy(padded)) == (True, "ok")
    assert _signature_full(0, padded)[1] == ((" l1 ", " X ", 1, 100),)


def test_a_single_final_valid_pair_cannot_confirm_settlement():
    # 18 coherent but identity-changing pairs, then one last valid pair: the
    # deadline expires before any three-pair streak exists.
    scripted = [_pair(1, f"l{i}") for i in range(18)] + [_pair(1, "last")]
    verifier = _scripted_verifier(scripted)
    res = verifier.settle(1, phase="test", coherence=_coherent_full,
                          signature=_signature_full)

    assert res["status"] != "settled"
    assert res["provenance"] == ()


def test_aggregate_identical_but_changing_baseline_pairs_do_not_establish_a_baseline():
    verifier = _scripted_verifier([_pair(1, f"l{i}") for i in range(25)])
    with pytest.raises(RuntimeError):
        establish_recovery_baseline(verifier)
    assert verifier.baseline is None


def test_later_full_state_streak_establishes_the_baseline():
    scripted = [_pair(1, "a"), _pair(1, "b"), _pair(1, "c"),
                _pair(1, "d"), _pair(1, "d"), _pair(1, "d")]
    verifier = _scripted_verifier(scripted)
    baseline = establish_recovery_baseline(verifier)

    assert baseline == {"version": 1, "lines": [_line("d", "X", 1)],
                        "total_cents": 100}
    assert verifier.known_version == 1
    assert verifier.baseline_provenance
    assert [s["seq"] for s in verifier.baseline_provenance] == [7, 8, 9, 10, 11, 12]


# --------------------------------------------------------------------------- #
# 6. Completion event precedes the returned/rendered report snapshot
# --------------------------------------------------------------------------- #

def test_completion_event_present_in_returned_and_rendered_report():
    view = _view(1, [_line("l1", "X", 1)])
    c = _controller(FakeVerifier([view]), FakeSession())

    report, verdict = c.complete()

    last = report["events"][-1]
    assert last["kind"] == "final"
    assert last["verdict"] == verdict
    assert last["final_status"] == "settled"
    assert last["confirmed"] == 1
    assert last["total"] == 1
    assert last["citations"] == []

    n_events = len(report["events"])
    first = render_recovery_report(report)
    second = render_recovery_report(report)
    assert first == second
    assert len(report["events"]) == n_events      # rendering mutates nothing
    assert '"kind": "final"' in first
    assert f'"verdict": "{verdict}"' in first

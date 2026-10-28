"""Bounded recovery controller tests (brief 3).

Integration tests drive the real adapter + store + verifier + controller for the
HTTP path; deterministic unit tests cover the caps, diagnosis attribution and
the completion contract. Persisted state is asserted only from tests/harness.
"""
from __future__ import annotations

import sqlite3

import httpx
import pytest

from adapter.evidence import Evidence
from core.checks import quantities
from core.ledger import CONFIRMED, Ledger
from core.recovery import MAX_PLANS_PER_TARGET, MAX_WRITES_PER_RUN, RecoveryController
from core.verify import Verifier

SKU_NAVYBLUE = "BTL-HIK-500-NAVYBLUE"
SKU_NAVY = "BTL-HIK-500-NAVY"
SKU_STEEL = "BTL-HIK-750-STEEL"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _reader(session):
    from adapter.observations import read_api_cart, read_dom_cart

    def reader(timeout_ms):
        dom = read_dom_cart(session, timeout_ms=timeout_ms)
        api = read_api_cart(session, timeout_ms=timeout_ms)
        return ({**dom.payload, "timestamp": dom.timestamp},
                {**api.payload, "timestamp": api.timestamp})

    return reader


def _build_controller(store, session, spec):
    from harness.run_task import _declared_additions

    verifier = Verifier(_reader(session))
    verifier.take_baseline()
    ledger = Ledger.build(spec["constraints"])
    declared = _declared_additions(spec)
    controller = RecoveryController(verifier, session, ledger, declared, verifier.baseline)
    return controller, verifier


def _reset_and_fault(store, fault=None, params=None):
    httpx.post(f"{store['control']}/_control/reset").raise_for_status()
    if fault:
        httpx.post(f"{store['control']}/_control/fault",
                   json={"name": fault, "params": params or {}}).raise_for_status()


def _seed(session, spec):
    for sl in spec.get("seed_lines") or []:
        session.seed_line(sl["sku"], sl.get("quantity", 1))


def _clear_fault(db_path, name):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM faults WHERE name = ?", (name,))
        conn.commit()
    finally:
        conn.close()


def _run_recovery(store, session, spec, fault=None, params=None, settle_ms=2000):
    """Seed, build controller, apply fault, run the actor with the recovery hook."""
    from agents.scripted_actor import run_actor

    _reset_and_fault(store)  # fresh boot
    _seed(session, spec)
    controller, verifier = _build_controller(store, session, spec)
    verifier.settle_ms = settle_ms
    controller.verifier = verifier
    if fault:
        httpx.post(f"{store['control']}/_control/fault",
                   json={"name": fault, "params": params or {}}).raise_for_status()

    def hook(event):
        k = event.get("kind")
        if k == "before_add":
            return controller.before_add(event)
        if k == "pre_add":
            return controller.pre_add(event)
        if k == "after_add":
            return controller.after_add(event)
        return None

    log = run_actor(session, spec, hook=hook)
    report, verdict = controller.complete()
    return log, report, verdict, controller


def _task1_spec():
    return {"product_query": "Hiker Water Bottle", "target_sku": SKU_NAVYBLUE, "quantity": 1,
            "constraints": [{"type": "item_quantity", "sku": SKU_NAVYBLUE, "quantity": 1}]}


def _task3_spec():
    return {"product_query": "Hiker Water Bottle", "target_sku": SKU_NAVYBLUE, "quantity": 1,
            "seed_lines": [{"sku": SKU_NAVY, "quantity": 1}, {"sku": SKU_NAVYBLUE, "quantity": 1}],
            "constraints": [{"type": "item_quantity", "sku": SKU_NAVY, "quantity": 1},
                            {"type": "item_quantity", "sku": SKU_NAVYBLUE, "quantity": 2}]}


# --------------------------------------------------------------------------- #
# Clean recovery + conservative limits (acceptance cases 1, 6)
# --------------------------------------------------------------------------- #

def test_recovery_clean_task1_complete(store, browser_session):
    _, report, verdict, controller = _run_recovery(store, browser_session, _task1_spec())
    assert verdict == "COMPLETE"
    assert controller.writes_attempted == 0
    assert report["final"]["lines"] == {SKU_NAVYBLUE: 1}


def test_recovery_clean_task3_complete_preserves_seeds(store, browser_session):
    _, report, verdict, controller = _run_recovery(store, browser_session, _task3_spec())
    assert verdict == "COMPLETE"
    assert controller.writes_attempted == 0
    assert report["final"]["lines"] == {SKU_NAVY: 1, SKU_NAVYBLUE: 2}
    assert report["protected_ok"] is True


def test_recovery_ownership_uncertain_merged_line(store, browser_session):
    _, report, verdict, controller = _run_recovery(store, browser_session, _task3_spec(),
                                                   fault="quantity-drift")
    assert verdict == "PARTIAL"
    assert controller.writes_attempted == 0
    assert any("already present before add" in b["detail"] for b in report["blocking"])
    # seeded NAVY preserved (unrelated), NAVYBLUE left at the drifted 3 (no reduction)
    assert report["final"]["lines"][SKU_NAVY] == 1


# --------------------------------------------------------------------------- #
# Recovery success (acceptance cases 2, 3)
# --------------------------------------------------------------------------- #

def test_recovery_quantity_drift_restores(store, browser_session):
    _, report, verdict, controller = _run_recovery(store, browser_session, _task1_spec(),
                                                   fault="quantity-drift")
    assert verdict == "COMPLETE"
    assert controller.writes_attempted == 1
    assert report["final"]["lines"] == {SKU_NAVYBLUE: 1}
    # the initial failure remains in the audit/resolved incidents
    assert any(i["kind"] == "excess_quantity" for i in report["incidents"])
    assert any(i["kind"] == "excess_quantity" for i in report["resolved"])


def test_recovery_transient_wrong_variant(store, browser_session):
    """Transient wrong-variant: the test clears the fault row between diagnosis
    and the corrective add; the controller receives no fault info."""
    session = browser_session
    spec = _task1_spec()
    _reset_and_fault(store)
    _seed(session, spec)
    controller, verifier = _build_controller(store, session, spec)
    httpx.post(f"{store['control']}/_control/fault",
               json={"name": "wrong-variant", "params": {}}).raise_for_status()

    item = {"index": 0, "product_id": "bottle-hiker", "sku": SKU_NAVYBLUE, "quantity": 1}
    controller.before_add({"action_id": "add-0", "item": item})
    session.search("Hiker Water Bottle")
    open_ev = session.open_product("bottle-hiker")
    session.select_variant(SKU_NAVYBLUE)
    r = controller.pre_add({"action_id": "add-0", "item": item, "open": open_ev.to_dict()})
    assert not (r and r.get("stop"))

    add_ev = session.add_to_cart(1, sku=SKU_NAVYBLUE)
    incident = controller.diagnose("add-0", item, add_ev.payload)
    assert incident["kind"] == "wrong_variant"

    # Clear the fault in our own temp DB (tests-only); controller sees nothing.
    _clear_fault(store["db"], "wrong-variant")
    controller.recover("add-0", item, incident)

    report, verdict = controller.complete()
    assert verdict == "COMPLETE"
    assert report["final"]["lines"] == {SKU_NAVYBLUE: 1}
    # the corrective add explicitly selected NAVYBLUE (not the default NAVY)
    add_step = next(s for s in report["audit"] if s.get("operation") == "add")
    assert add_step["evidence"]["sent_sku"] == SKU_NAVYBLUE
    assert add_step["evidence"]["requested_sku"] == SKU_NAVYBLUE


# --------------------------------------------------------------------------- #
# Persistent faults end PARTIAL (acceptance cases 4, 5)
# --------------------------------------------------------------------------- #

def test_recovery_persistent_wrong_variant_partial(store, browser_session):
    _, report, verdict, controller = _run_recovery(store, browser_session, _task1_spec(),
                                                   fault="wrong-variant")
    assert verdict == "PARTIAL"
    assert controller.writes_attempted == 2  # delete + corrective add
    assert any("did not achieve" in b["detail"] for b in report["blocking"])


def test_recovery_persistent_silent_drop_partial(store, browser_session):
    _, report, verdict, controller = _run_recovery(store, browser_session, _task1_spec(),
                                                   fault="silent-drop")
    assert verdict == "PARTIAL"
    assert controller.writes_attempted == 2  # two retries
    assert controller.plans_by_target.get("add-0") == MAX_PLANS_PER_TARGET
    assert any("budget exhausted" in b["detail"] for b in report["blocking"])


# --------------------------------------------------------------------------- #
# Stale-version 409 and unknown corrective outcomes (acceptance cases 7, 8)
# --------------------------------------------------------------------------- #

def _session_cookie(session):
    for c in session.context.cookies():
        if c["name"] == "cp_session":
            return c["value"]
    return None


def test_recovery_stale_version_correction_gets_409(store, browser_session):
    """A second client advances the version between diagnosis and the correction;
    the PATCH gets 409 and nothing changes."""
    session = browser_session
    spec = _task1_spec()
    _reset_and_fault(store)
    _seed(session, spec)
    controller, verifier = _build_controller(store, session, spec)
    httpx.post(f"{store['control']}/_control/fault",
               json={"name": "quantity-drift", "params": {}}).raise_for_status()

    item = {"index": 0, "product_id": "bottle-hiker", "sku": SKU_NAVYBLUE, "quantity": 1}
    controller.before_add({"action_id": "add-0", "item": item})
    session.search("Hiker Water Bottle")
    open_ev = session.open_product("bottle-hiker")
    session.select_variant(SKU_NAVYBLUE)
    controller.pre_add({"action_id": "add-0", "item": item, "open": open_ev.to_dict()})
    add_ev = session.add_to_cart(1, sku=SKU_NAVYBLUE)
    incident = controller.diagnose("add-0", item, add_ev.payload)
    assert incident["kind"] == "excess_quantity"

    # External client advances the version (same cookie).
    c2 = httpx.Client()
    c2.cookies.set("cp_session", _session_cookie(session), domain="127.0.0.1")
    v = c2.get(f"{store['base']}/api/cart").json()["version"]
    c2.post(f"{store['base']}/api/cart/lines",
            headers={"X-Cart-Version": str(v)}, json={"sku": SKU_STEEL, "quantity": 1})

    controller.recover("add-0", item, incident)

    report, verdict = controller.complete()
    assert verdict == "PARTIAL"
    assert controller.writes_attempted == 1  # the PATCH was attempted once
    assert any("conflict" in b["detail"] for b in report["blocking"])
    # the rejected PATCH did not reduce the drifted line
    api = session.context.request.get(f"{store['base']}/api/cart").json()
    amap = quantities(api["lines"])
    assert amap[SKU_NAVYBLUE] == 2  # unchanged by the rejected correction


class _ProxySession:
    """Wraps the real session; drops the response of corrective writes."""

    def __init__(self, real, commit: bool):
        self._real = real
        self.commit = commit
        self.writes = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def set_line_quantity(self, line_id, sku, quantity, expected_version):
        if self.commit:
            self._real.set_line_quantity(line_id, sku, quantity, expected_version)
        self.writes.append(("patch", line_id, sku, quantity, expected_version))
        return Evidence(source="action_confirmation", payload={
            "action": "set_line_quantity", "line_id": line_id, "sku": sku,
            "quantity": quantity, "expected_version": expected_version,
            "response_status": None, "response_version": None, "response_current_version": None})

    def delete_line(self, line_id, sku, expected_version):
        if self.commit:
            self._real.delete_line(line_id, sku, expected_version)
        self.writes.append(("delete", line_id, sku, expected_version))
        return Evidence(source="action_confirmation", payload={
            "action": "delete_line", "line_id": line_id, "sku": sku,
            "expected_version": expected_version,
            "response_status": None, "response_version": None, "response_current_version": None})


def _run_recovery_proxy(store, session, commit):
    """Same as _run_recovery but with a proxy session for corrective writes."""
    from agents.scripted_actor import run_actor

    spec = _task1_spec()
    _reset_and_fault(store)
    _seed(session, spec)
    verifier = Verifier(_reader(session))
    verifier.take_baseline()
    ledger = Ledger.build(spec["constraints"])
    from harness.run_task import _declared_additions
    declared = _declared_additions(spec)
    proxy = _ProxySession(session, commit=commit)
    controller = RecoveryController(verifier, proxy, ledger, declared, verifier.baseline)
    httpx.post(f"{store['control']}/_control/fault",
               json={"name": "quantity-drift", "params": {}}).raise_for_status()

    def hook(event):
        k = event.get("kind")
        if k == "before_add":
            return controller.before_add(event)
        if k == "pre_add":
            return controller.pre_add(event)
        if k == "after_add":
            return controller.after_add(event)
        return None

    run_actor(session, spec, hook=hook)
    report, verdict = controller.complete()
    return report, verdict, controller


def test_recovery_commit_with_lost_response_partial(store, browser_session):
    report, verdict, controller = _run_recovery_proxy(store, browser_session, commit=True)
    assert verdict == "PARTIAL"
    assert any("unknown" in b["detail"] for b in report["blocking"])
    # the write actually committed (persisted state), but the controller did not know
    api = browser_session.context.request.get(f"{store['base']}/api/cart").json()
    assert quantities(api["lines"])[SKU_NAVYBLUE] == 1  # the PATCH committed


def test_recovery_nocommit_lost_response_partial(store, browser_session):
    report, verdict, controller = _run_recovery_proxy(store, browser_session, commit=False)
    assert verdict == "PARTIAL"
    assert any("unknown" in b["detail"] for b in report["blocking"])
    api = browser_session.context.request.get(f"{store['base']}/api/cart").json()
    assert quantities(api["lines"])[SKU_NAVYBLUE] == 2  # no commit happened


# --------------------------------------------------------------------------- #
# Unsafe cases stop without correction (acceptance case 9)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fault,reason", [
    ("dom-api-mismatch", "conflicting DOM/API sources"),
    ("stale-total", "displayed total"),
    ("external-mutation", "unexplained version advance"),
])
def test_recovery_unsafe_stop(store, browser_session, fault, reason):
    _, report, verdict, controller = _run_recovery(store, browser_session, _task1_spec(),
                                                   fault=fault)
    assert verdict == "PARTIAL"
    assert controller.writes_attempted == 0
    assert any(reason in (b["detail"] or "") or reason in (report["stop_reason"] or "")
               for b in report["blocking"])


def test_recovery_deadline_expiry_stop(store, browser_session):
    _, report, verdict, controller = _run_recovery(store, browser_session, _task1_spec(),
                                                   fault="slow-consistency",
                                                   params={"delay_ms": 800}, settle_ms=100)
    assert verdict == "PARTIAL"
    assert controller.writes_attempted == 0
    assert "deadline" in (report["stop_reason"] or "")


# --------------------------------------------------------------------------- #
# Product refresh must not adopt an unproven version (clarification 1)
# --------------------------------------------------------------------------- #

def test_external_edit_before_refresh_sends_zero_posts(store, browser_session):
    session = browser_session
    spec = _task1_spec()
    _reset_and_fault(store)
    _seed(session, spec)
    controller, verifier = _build_controller(store, session, spec)

    # The proven version is 0; an external client advances the store to v1.
    c2 = httpx.Client()
    c2.cookies.set("cp_session", _session_cookie(session), domain="127.0.0.1")
    v = c2.get(f"{store['base']}/api/cart").json()["version"]
    c2.post(f"{store['base']}/api/cart/lines",
            headers={"X-Cart-Version": str(v)}, json={"sku": SKU_STEEL, "quantity": 1})

    item = {"index": 0, "product_id": "bottle-hiker", "sku": SKU_NAVYBLUE, "quantity": 1}
    result = controller._corrective_add("add-0", item)
    assert result is None  # blocked before the add
    assert any("page version" in b["detail"] for b in controller.blocking)
    # zero corrective POSTs: no NAVYBLUE line was added
    api = session.context.request.get(f"{store['base']}/api/cart").json()
    assert SKU_NAVYBLUE not in quantities(api["lines"])


# --------------------------------------------------------------------------- #
# Deterministic unit tests: caps, sent validation, contract
# --------------------------------------------------------------------------- #

def _unit_controller(writes_attempted=0):
    ledger = Ledger.build([{"type": "item_quantity", "sku": "X", "quantity": 1}])
    c = RecoveryController(None, None, ledger, {"X": 1}, {"version": 0, "lines": [], "total_cents": 0})
    c.writes_attempted = writes_attempted
    return c


def test_sent_version_mismatch_stops():
    c = _unit_controller()
    c._before = {"version": 0, "lines": [], "total_cents": 0}
    res = c.diagnose("a", {"sku": "X", "quantity": 1},
                     {"sent_version": 1, "sent_sku": "X", "sent_quantity": 1,
                      "response_status": 200, "response_version": 1})
    assert res.get("_stop") is not None
    assert "sent version" in res["_stop"]["reason"]


def test_budget_write_cap_blocks_plan():
    c = _unit_controller(writes_attempted=MAX_WRITES_PER_RUN - 1)
    assert c._budget_ok("a", 2) is False
    assert any("write budget" in b["detail"] for b in c.blocking)


def test_budget_plan_cap_blocks_plan():
    c = _unit_controller()
    c.plans_by_target["a"] = MAX_PLANS_PER_TARGET
    assert c._budget_ok("a", 1) is False
    assert any("plan budget" in b["detail"] for b in c.blocking)


def test_final_contract_net_delta_violation_not_complete():
    ledger = Ledger.build([{"type": "item_quantity", "sku": "X", "quantity": 1}])

    class FakeVerifier:
        known_version = 1
        samples, anomalies, anomaly_epoch, settle_ms = [], [], False, 2000

        def settle(self, known_version=None, deadline_ms=None, phase="recovery"):
            view = {"version": 1, "lines": [{"line_id": "l1", "sku": "X", "quantity": 2,
                                              "unit_price_cents": 100}], "total_cents": 200}
            return {"status": "settled", "view": view, "provenance": ()}

    c = RecoveryController(FakeVerifier(), None, ledger, {"X": 1},
                           {"version": 0, "lines": [], "total_cents": 0})
    report, verdict = c.complete()
    assert verdict == "PARTIAL"
    assert report["delta_consistent"] is False


def test_final_contract_protected_item_changed_not_complete():
    ledger = Ledger.build([{"type": "item_quantity", "sku": "X", "quantity": 2}])

    class FakeVerifier:
        known_version = 1
        samples, anomalies, anomaly_epoch, settle_ms = [], [], False, 2000

        def settle(self, known_version=None, deadline_ms=None, phase="recovery"):
            # baseline line l1 (X qty 1) is now a different line_id l9 (identity lost)
            view = {"version": 1, "lines": [{"line_id": "l9", "sku": "X", "quantity": 2,
                                              "unit_price_cents": 100}], "total_cents": 200}
            return {"status": "settled", "view": view, "provenance": ()}

    baseline = {"version": 0, "lines": [{"line_id": "l1", "sku": "X", "quantity": 1,
                                         "unit_price_cents": 100}], "total_cents": 100}
    c = RecoveryController(FakeVerifier(), None, ledger, {"X": 1}, baseline)
    report, verdict = c.complete()
    assert verdict == "PARTIAL"
    assert report["protected_ok"] is False


def test_diagnose_ownership_uncertain_for_merged_sku():
    c = _unit_controller()
    # baseline already contains SKU X
    c.baseline_skus = {"X"}
    payload = {"requested_sku": "X", "quantity": 1, "response_version": 1}
    before = {"version": 0, "lines": [{"line_id": "l1", "sku": "X", "quantity": 1,
                                       "unit_price_cents": 100}], "total_cents": 100}
    after = {"version": 1, "lines": [{"line_id": "l1", "sku": "X", "quantity": 3,
                                      "unit_price_cents": 100}], "total_cents": 300}
    incident = c._diagnose(payload, before, after)
    assert incident["kind"] == "ownership_uncertain"


def test_recover_requires_constraints(tmp_path):
    """--recover without a constraints ledger is a hard configuration error (exit 2)."""
    import json

    from harness.run_task import main

    spec = {"product_query": "Hiker Water Bottle", "target_sku": SKU_NAVYBLUE, "quantity": 1}
    p = tmp_path / "nocons.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    assert main(["--task", str(p), "--recover"]) == 2

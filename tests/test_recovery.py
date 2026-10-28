"""Bounded recovery controller tests (brief 3).

Integration tests drive the real adapter + store + verifier + controller for the
HTTP path; deterministic unit tests cover the caps, diagnosis attribution and
the completion contract. Persisted state is asserted only from tests/harness.
"""
from __future__ import annotations

import copy
import sqlite3

import httpx
import pytest

from adapter.evidence import Evidence
from core.checks import quantities
from core.ledger import CONFIRMED, Ledger
from core.recovery import (MAX_PLANS_PER_TARGET, MAX_WRITES_PER_RUN, RecoveryController,
                           attach_raw_pair_recorder, establish_recovery_baseline,
                           render_recovery_report)
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


def _build_controller(store, session, spec, controller_session=None):
    from core.recovery import attach_raw_pair_recorder, establish_recovery_baseline
    from harness.run_task import _declared_additions

    verifier = Verifier(_reader(session))
    raw_pairs = {}
    attach_raw_pair_recorder(verifier, raw_pairs)
    establish_recovery_baseline(verifier)
    ledger = Ledger.build(spec["constraints"])
    declared = _declared_additions(spec)
    controller = RecoveryController(verifier, controller_session or session, ledger, declared,
                                    verifier.baseline, raw_pairs=raw_pairs)
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


def _task2_spec():
    """Two-item actor task: the second item is real remaining work to be skipped."""
    return {"items": [
        {"product_query": "Hiker Water Bottle", "target_sku": SKU_NAVYBLUE, "quantity": 1},
        {"product_query": "Hiker Water Bottle", "target_sku": SKU_STEEL, "quantity": 1}],
        "constraints": [{"type": "item_quantity", "sku": SKU_NAVYBLUE, "quantity": 1},
                        {"type": "item_quantity", "sku": SKU_STEEL, "quantity": 1}]}


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


def test_baseline_observations_have_complete_lines(store, browser_session):
    """The recovery baseline is accepted from complete full-state pairs recorded
    BEFORE any mutation, with eligible citations and complete line fields."""
    session = browser_session
    spec = _task3_spec()
    _reset_and_fault(store)
    _seed(session, spec)
    controller, verifier = _build_controller(store, session, spec)

    assert verifier.baseline_provenance            # citations exist pre-mutation
    assert controller.writes_attempted == 0
    controller._record_observations()              # materialize, no new reads
    baseline_seqs = {s["seq"] for s in verifier.baseline_provenance}
    obs = [o for o in controller.observations.values() if o["seq"] in baseline_seqs]
    assert {o["seq"] for o in obs} == baseline_seqs
    assert {o["phase"] for o in obs} == {"baseline"}
    for o in obs:
        assert o["lines_full"] is not None
        assert all({"line_id", "sku", "quantity", "unit_price_cents"} <= set(ln)
                   for ln in o["lines_full"])
    # Exact seeded identity/price is retained, never aggregated away.
    observed = {(ln["sku"], ln["quantity"], ln["unit_price_cents"])
                for o in obs for ln in o["lines_full"]}
    assert {(sku, qty) for sku, qty, _price in observed} == {
        (SKU_NAVY, 1), (SKU_NAVYBLUE, 1)}
    assert all(isinstance(price, int) and price > 0 for _sku, _qty, price in observed)


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


def _persisted_cart(store, session):
    """DB truth for this test's cart.

    ``cp_session`` is a SESSION id, not a cart id. Resolve ``sessions.cart_id``
    with a direct SELECT (never a session-creating helper), assert exactly one
    mapping, read the persisted cart, then close. Direct reads bypass fault
    projections and do not consume one-shot faults.
    """
    from store import models

    sid = _session_cookie(session)
    assert sid, "no cp_session cookie on the browser session"
    conn = models.connect(store["db"])
    try:
        rows = conn.execute(
            "SELECT cart_id FROM sessions WHERE session_id = ?", (sid,)).fetchall()
        assert len(rows) == 1, f"expected one cart mapping for the session, got {len(rows)}"
        return models.read_persisted_cart(conn, rows[0]["cart_id"])
    finally:
        conn.close()


class _RecordingSession:
    """Delegates to the real session and records outbound mutating calls."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "mutations", [])

    def __getattr__(self, name):
        return getattr(self._real, name)

    def add_to_cart(self, quantity=1, sku=None):
        self.mutations.append(("add_to_cart", sku, quantity))
        return self._real.add_to_cart(quantity, sku=sku)

    def set_line_quantity(self, line_id, sku, quantity, expected_version):
        self.mutations.append(("set_line_quantity", line_id, sku, quantity, expected_version))
        return self._real.set_line_quantity(line_id, sku, quantity, expected_version)

    def delete_line(self, line_id, sku, expected_version):
        self.mutations.append(("delete_line", line_id, sku, expected_version))
        return self._real.delete_line(line_id, sku, expected_version)


def test_recovery_stale_version_correction_gets_409(store, browser_session):
    """A real second client advances the version between diagnosis and the
    corrective PATCH: the guarded write returns HTTP 409, no line/version changes,
    and the remaining normal (second-item) add never runs."""
    from agents.scripted_actor import run_actor

    session = browser_session
    spec = _task2_spec()
    _reset_and_fault(store)
    _seed(session, spec)
    proxy = _RecordingSession(session)
    controller, verifier = _build_controller(store, session, spec, controller_session=proxy)
    httpx.post(f"{store['control']}/_control/fault",
               json={"name": "quantity-drift", "params": {}}).raise_for_status()

    seen = {}

    def hook(event):
        k = event.get("kind")
        if k == "before_add":
            return controller.before_add(event)
        if k == "pre_add":
            return controller.pre_add(event)
        # after_add is split here ONLY to inject the external edit between the
        # diagnosis and the corrective write (real code path otherwise).
        item = event["item"]
        payload = (event.get("add") or {}).get("payload", {})
        incident = controller.diagnose(event["action_id"], item, payload)
        assert incident["kind"] == "excess_quantity"
        seen["stale_version"] = verifier.known_version

        # Clear the drift fault (test-only DB control) so the second client's real
        # mutation is exactly its intended line, then advance the version.
        _clear_fault(store["db"], "quantity-drift")
        c2 = httpx.Client()
        c2.cookies.set("cp_session", _session_cookie(session), domain="127.0.0.1")
        v = c2.get(f"{store['base']}/api/cart").json()["version"]
        ext = c2.post(f"{store['base']}/api/cart/lines",
                      headers={"X-Cart-Version": str(v)},
                      json={"sku": SKU_STEEL, "quantity": 1})
        seen["status"] = ext.status_code
        seen["after_external"] = _persisted_cart(store, session)

        controller.recover(event["action_id"], item, incident)
        return controller._stop_result()

    run_actor(proxy, spec, hook=hook)
    report, verdict = controller.complete()

    # The second client's mutation really succeeded and advanced the version.
    assert seen["status"] == 200
    assert seen["stale_version"] == 1
    after_external = seen["after_external"]
    assert after_external["version"] == seen["stale_version"] + 1
    assert quantities(after_external["lines"])[SKU_STEEL] == 1
    assert quantities(after_external["lines"])[SKU_NAVYBLUE] == 2

    # Exactly one corrective PATCH, sent against the now-stale version, got 409.
    steps = [s for s in report["audit"] if s.get("phase") == "correction"]
    assert len(steps) == 1
    step = steps[0]
    assert step["operation"] == "patch"
    assert step["outcome"] == "conflict"
    assert step["request"]["expected_version"] == seen["stale_version"]
    assert step["request"]["sku"] == SKU_NAVYBLUE
    assert step["request"]["quantity"] == 1
    assert step["response"]["response_status"] == 409
    assert step["response"]["response_current_version"] == after_external["version"]
    assert controller.writes_attempted == 1
    assert controller.stop_reason and "conflict" in controller.stop_reason

    # The rejected correction changed nothing: same full persisted cart/version.
    assert _persisted_cart(store, session) == after_external

    # Terminal stop skipped the remaining NORMAL second-item add: only the first
    # item's real add reached the adapter.
    adds = [m for m in proxy.mutations if m[0] == "add_to_cart"]
    assert adds == [("add_to_cart", SKU_NAVYBLUE, 1)]
    assert [m for m in proxy.mutations if m[0] == "set_line_quantity"] == [
        ("set_line_quantity", step["request"]["line_id"], SKU_NAVYBLUE, 1,
         seen["stale_version"])]
    assert verdict == "PARTIAL"
    assert report["plans_by_target"] == {"add-0": 1}


class _ProxySession(_RecordingSession):
    """Records real adapter calls and drops the response of corrective writes.

    With ``commit=True`` the real guarded PATCH/DELETE is executed and then its
    response is discarded (a committed-but-lost response). With ``commit=False``
    no mutation is sent and a lost response is supplied (an uncommitted case).
    """

    def __init__(self, real, commit: bool):
        super().__init__(real)
        self.commit = commit

    @staticmethod
    def _lost(action, **fields):
        return Evidence(source="action_confirmation", payload={
            "action": action, **fields, "response_status": None,
            "response_version": None, "response_current_version": None})

    def set_line_quantity(self, line_id, sku, quantity, expected_version):
        if self.commit:
            self._real.set_line_quantity(line_id, sku, quantity, expected_version)
        self.mutations.append(("set_line_quantity", line_id, sku, quantity, expected_version))
        return self._lost("set_line_quantity", line_id=line_id, sku=sku,
                          quantity=quantity, expected_version=expected_version)

    def delete_line(self, line_id, sku, expected_version):
        if self.commit:
            self._real.delete_line(line_id, sku, expected_version)
        self.mutations.append(("delete_line", line_id, sku, expected_version))
        return self._lost("delete_line", line_id=line_id, sku=sku,
                          expected_version=expected_version)


def _run_recovery_proxy(store, session, commit):
    """Multi-item actor run with a proxy session for corrective writes."""
    from agents.scripted_actor import run_actor

    spec = _task2_spec()
    _reset_and_fault(store)
    _seed(session, spec)
    proxy = _ProxySession(session, commit=commit)
    controller, verifier = _build_controller(store, session, spec, controller_session=proxy)
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

    run_actor(proxy, spec, hook=hook)
    report, verdict = controller.complete()
    return report, verdict, controller, proxy


def _assert_single_unknown_correction(report, controller, proxy, expected_version):
    assert controller.writes_attempted == 1
    steps = [s for s in report["audit"] if s.get("phase") == "correction"]
    assert len(steps) == 1
    step = steps[0]
    assert step["operation"] == "patch"
    assert step["outcome"] == "unknown"
    assert step["request"]["expected_version"] == expected_version
    assert step["request"]["sku"] == SKU_NAVYBLUE
    assert step["request"]["quantity"] == 1
    assert step["response"]["response_status"] is None
    assert any("correction unknown outcome" in (b["detail"] or "")
               for b in report["blocking"])
    # no replay: one plan for the target and one corrective adapter call
    assert report["plans_by_target"] == {"add-0": 1}
    assert len([m for m in proxy.mutations if m[0] == "set_line_quantity"]) == 1
    # the remaining NORMAL second-item add never ran
    assert [m for m in proxy.mutations if m[0] == "add_to_cart"] == [
        ("add_to_cart", SKU_NAVYBLUE, 1)]


def test_recovery_commit_with_lost_response_partial(store, browser_session):
    report, verdict, controller, proxy = _run_recovery_proxy(store, browser_session, commit=True)
    assert verdict == "PARTIAL"
    # DB truth: the real guarded PATCH committed even though its response was lost.
    persisted = _persisted_cart(store, browser_session)
    assert quantities(persisted["lines"])[SKU_NAVYBLUE] == 1
    assert SKU_STEEL not in quantities(persisted["lines"])
    _assert_single_unknown_correction(report, controller, proxy, expected_version=1)


def test_recovery_nocommit_lost_response_partial(store, browser_session):
    report, verdict, controller, proxy = _run_recovery_proxy(store, browser_session, commit=False)
    assert verdict == "PARTIAL"
    # DB truth: no PATCH was sent, so the drifted quantity remains.
    persisted = _persisted_cart(store, browser_session)
    assert quantities(persisted["lines"])[SKU_NAVYBLUE] == 2
    assert SKU_STEEL not in quantities(persisted["lines"])
    _assert_single_unknown_correction(report, controller, proxy, expected_version=1)


# --------------------------------------------------------------------------- #
# Unsafe cases stop without correction (acceptance case 9)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fault", ["dom-api-mismatch", "stale-total"])
def test_recovery_unsafe_stop(store, browser_session, fault):
    _, report, verdict, controller = _run_recovery(store, browser_session, _task1_spec(),
                                                   fault=fault)
    assert verdict == "PARTIAL"
    assert controller.writes_attempted == 0
    # Every eligible pair must be full-state coherent, but the diagnosis differs:
    # a real DOM/API split is a source conflict, while the stale-total fault gives
    # both sides the same stale displayed total, which is an arithmetic
    # inconsistency and must not be reported as conflicting sources.
    stop = report["stop_reason"] or ""
    if fault == "dom-api-mismatch":
        expected = "conflicting DOM/API sources"
        assert expected in stop
    else:
        assert "conflicting DOM/API sources" not in stop
        assert "recomputed total" in stop
        expected = "recomputed total"
    assert any(expected in (b["detail"] or "") for b in report["blocking"])


def test_recovery_seeded_external_mutation_stops_before_any_post(store, browser_session):
    """SEEDED fixture (changed 28 Oct 2026, clarification 1).

    Claim: a real external mutation stops before another write. A seeded cart is
    required because ``store.models._apply_external_mutation`` is a no-op on an
    empty cart. Here the read-triggered mutation really changes state, so the
    fresh pre-action read must stop with zero subsequent shopping POSTs.
    """
    log, report, verdict, controller = _run_recovery(store, browser_session,
                                                     _task3_spec(), fault="external-mutation")
    assert verdict == "PARTIAL"
    assert controller.writes_attempted == 0
    assert any("unexplained version advance" in (b["detail"] or "")
               or "unexplained version advance" in (report["stop_reason"] or "")
               for b in report["blocking"])
    assert log == [] or [e for e in log
                         if (e.get("payload") or {}).get("action") in
                         ("search", "open_product", "select_variant", "add_to_cart")] == []


def test_recovery_empty_cart_external_mutation_noop_completes(store, browser_session):
    """Separately named empty-cart no-op regression (clarification 1).

    The one-shot external-mutation fixture is a no-op when the cart has no lines:
    the required fresh pre-action read consumes it without changing state. This
    is NOT detection of an external edit (none occurred); the intended add may
    COMPLETE with zero corrective writes. The original no-recovery matrix keeps
    its own read timing and is unchanged.
    """
    log, report, verdict, controller = _run_recovery(store, browser_session,
                                                     _task1_spec(), fault="external-mutation")
    assert verdict == "COMPLETE"
    assert controller.writes_attempted == 0
    assert report["final"]["lines"] == {SKU_NAVYBLUE: 1}
    assert log  # the intended add really ran


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

        def settle(self, known_version=None, deadline_ms=None, phase="recovery", **_recovery):
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

        def settle(self, known_version=None, deadline_ms=None, phase="recovery", **_recovery):
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


def test_recovery_silent_drop_into_seeded_requested_sku(store, browser_session):
    """Silent drop with a seeded requested SKU (NAVYBLUE 1 -> 2).

    The original add is dropped (no commit); the retry merges into the seeded
    line. The action-relative rule (before + requested quantity) must resolve it
    without the old absolute-quantity mismatch or the ``correct``/``detail``
    KeyError reported in review.
    """
    session = browser_session
    spec = {"product_query": "Hiker Water Bottle", "target_sku": SKU_NAVYBLUE,
            "quantity": 1, "seed_lines": [{"sku": SKU_NAVYBLUE, "quantity": 1}],
            "constraints": [{"type": "item_quantity", "sku": SKU_NAVYBLUE, "quantity": 2}]}
    _reset_and_fault(store)
    _seed(session, spec)
    controller, verifier = _build_controller(store, session, spec)
    httpx.post(f"{store['control']}/_control/fault",
               json={"name": "silent-drop", "params": {}}).raise_for_status()

    item = {"index": 0, "product_id": "bottle-hiker", "sku": SKU_NAVYBLUE, "quantity": 1}
    assert controller.before_add({"action_id": "add-0", "item": item}) is None
    session.search("Hiker Water Bottle")
    open_ev = session.open_product("bottle-hiker")
    session.select_variant(SKU_NAVYBLUE)
    assert not (controller.pre_add({"action_id": "add-0", "item": item,
                                    "open": open_ev.to_dict()}) or {}).get("stop")
    add_ev = session.add_to_cart(1, sku=SKU_NAVYBLUE)
    incident = controller.diagnose("add-0", item, add_ev.payload)
    assert incident["kind"] == "silent_drop"

    _clear_fault(store["db"], "silent-drop")  # transient: the retry can commit
    controller.recover("add-0", item, incident)

    report, verdict = controller.complete()
    assert verdict == "COMPLETE"
    assert controller.writes_attempted == 1
    assert report["final"]["lines"] == {SKU_NAVYBLUE: 2}
    assert report["delta"] == {SKU_NAVYBLUE: 1}
    assert report["resolved"] and report["resolved"][0]["kind"] == "silent_drop"
    # Seeded before-state is recorded in full, not as an aggregate.
    diag = next(e for e in report["events"] if e["kind"] == "diagnosis")
    assert len(diag["before_lines"]) == 1
    only = next(iter(diag["before_lines"].values()))
    assert only["sku"] == SKU_NAVYBLUE and only["quantity"] == 1
    assert isinstance(only["unit_price_cents"], int)
    assert diag["after_lines"] == diag["before_lines"]  # the drop changed nothing


def test_recovery_report_citations_and_stable_render(store, browser_session):
    """§5: citations reference emitted observations, incidents survive resolution,
    and rendering twice is stable and does not mutate the audit."""
    session = browser_session
    spec = _task1_spec()
    _reset_and_fault(store)
    _seed(session, spec)
    controller, verifier = _build_controller(store, session, spec)
    httpx.post(f"{store['control']}/_control/fault",
               json={"name": "wrong-variant", "params": {}}).raise_for_status()

    item = {"index": 0, "product_id": "bottle-hiker", "sku": SKU_NAVYBLUE, "quantity": 1}
    assert controller.before_add({"action_id": "add-0", "item": item}) is None
    session.search("Hiker Water Bottle")
    open_ev = session.open_product("bottle-hiker")
    session.select_variant(SKU_NAVYBLUE)
    controller.pre_add({"action_id": "add-0", "item": item, "open": open_ev.to_dict()})
    add_ev = session.add_to_cart(1, sku=SKU_NAVYBLUE)
    incident = controller.diagnose("add-0", item, add_ev.payload)
    assert incident["kind"] == "wrong_variant"

    _clear_fault(store["db"], "wrong-variant")
    controller.recover("add-0", item, incident)
    report, verdict = controller.complete()
    assert verdict == "COMPLETE"

    # Emitted observations carry complete line fields (IDs and unit prices).
    seqs = {s["seq"] for s in report["observations"]}
    non_baseline = [s for s in report["observations"] if s["phase"] != "baseline"]
    assert non_baseline
    assert all(s["lines_full"] is not None for s in non_baseline)
    assert all("line_id" in ln and "unit_price_cents" in ln
               for s in non_baseline for ln in s["lines_full"])

    # Every confirmation and resolution cites samples actually in the report.
    for c in report["constraints"]:
        if c["state"] == CONFIRMED:
            assert c["citations"]
            for pair in c["citations"]:
                assert all(seq in seqs for seq in pair)
    for r in report["resolved"]:
        assert r["citations"]
        for pair in r["citations"]:
            assert all(seq in seqs for seq in pair)

    # Original incidents survive; resolution is a new record, not a mutation.
    assert report["incidents"]
    assert all("resolution" not in i for i in report["incidents"])
    assert all(i["kind"] == "wrong_variant" for i in report["incidents"])

    # Before/after diagnosis evidence is present and complete.
    before_ev = [e for e in report["events"] if e["kind"] == "before"]
    diag_ev = [e for e in report["events"] if e["kind"] == "diagnosis"]
    assert before_ev and "lines" in before_ev[0] and "citations" in before_ev[0]
    assert diag_ev
    assert diag_ev[0]["before_lines"] is not None
    assert diag_ev[0]["after_lines"]  # the wrong variant is recorded
    assert all(isinstance(lid, str) and lid and "unit_price_cents" in info
               for lid, info in diag_ev[0]["after_lines"].items())

    snapshot = copy.deepcopy(report)
    first = render_recovery_report(report)
    second = render_recovery_report(report)
    assert first == second
    assert report == snapshot
    assert "Recorded observations (complete line fields, no aggregation)" in first


def test_recover_requires_constraints(tmp_path):
    """--recover without a constraints ledger is a hard configuration error (exit 2)."""
    import json

    from harness.run_task import main

    spec = {"product_query": "Hiker Water Bottle", "target_sku": SKU_NAVYBLUE, "quantity": 1}
    p = tmp_path / "nocons.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    assert main(["--task", str(p), "--recover"]) == 2

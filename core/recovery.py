"""Bounded, deterministic recovery controller for the owned mock store.

Composed with the existing verifier through its narrow public ``settle()``
method. It never imports from ``store/``, never reads the SQLite file, fault
configuration, control endpoint, or expected fixture results, and never trusts
``added.sku``/``added.line_id`` as ownership proof.

Recovery is opt-in, conservative, and capped: it corrects only a proven,
safely-attributable mismatch (wrong variant, excess quantity on a newly created
line, or a proven no-commit silent drop), using the store's existing guarded
PATCH/DELETE and the existing browser add action. Every correction sends the
last proven version and is re-verified; 409, external advance, timeout, malformed
response and unknown commit stop the run.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .checks import evaluate, quantities, recompute_total
from .ledger import CONFIRMED, UNKNOWN, Constraint, Ledger

MAX_PLANS_PER_TARGET = 2
MAX_WRITES_PER_RUN = 4

RECOVERABLE_KINDS = {"wrong_variant", "excess_quantity", "silent_drop"}


def _delta(before_map: dict, after_map: dict) -> dict:
    out = {}
    for sku in sorted(set(before_map) | set(after_map)):
        d = after_map.get(sku, 0) - before_map.get(sku, 0)
        if d != 0:
            out[sku] = d
    return out


class RecoveryController:
    def __init__(self, verifier, session, ledger: Ledger, declared_additions: dict,
                 original_baseline: dict):
        self.verifier = verifier
        self.session = session
        self.ledger = ledger
        self.declared = declared_additions
        self.original_baseline = original_baseline
        self.baseline_lines = {l["line_id"]: l for l in (original_baseline.get("lines") or [])}
        self.baseline_skus = {l["sku"] for l in (original_baseline.get("lines") or [])}

        self.audit: list = []          # append-only step records
        self.incidents: list = []      # all incidents (resolved + blocking)
        self.blocking: list = []       # terminal incidents
        self.resolved: list = []       # resolved recoverable incidents
        self.plans_by_target: dict = {}
        self.writes_attempted = 0
        self.stop_reason: Optional[str] = None
        self.last_state = original_baseline  # last known fresh settled state
        self._before: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Hook API (called by the actor through harness/run_task)
    # ------------------------------------------------------------------ #

    def before_add(self, event: dict):
        action_id = event["action_id"]
        # The before-state is the last fresh settled snapshot: the original
        # baseline for the first add, the prior action's settled after-state
        # otherwise. No read is taken here, so a read-triggered fault (e.g.
        # external-mutation) is not consumed before the add.
        self._before = self.last_state
        self.audit.append({
            "action_id": action_id, "phase": "before", "version": self._before["version"],
            "lines": quantities(self._before["lines"]),
        })
        return None

    def pre_add(self, event: dict):
        action_id = event["action_id"]
        item = event["item"]
        open_ev = event.get("open") or {}
        embedded = open_ev.get("payload", {}).get("cart_version") if isinstance(open_ev, dict) else None
        if self._before is None:
            return self._stop("no before snapshot")
        if embedded != self._before["version"]:
            return self._stop(
                f"page version {embedded!r} != proven {self._before['version']}")
        return None

    def after_add(self, event: dict):
        action_id = event["action_id"]
        item = event["item"]
        payload = (event.get("add") or {}).get("payload", {})
        if self._before is None:
            return self._stop("no before snapshot")
        incident = self.diagnose(action_id, item, payload)
        if incident is None:
            return self._stop_result()
        if isinstance(incident, dict) and incident.get("_stop"):
            return incident["_stop"]
        if incident["kind"] in RECOVERABLE_KINDS:
            self.recover(action_id, item, incident)
        else:
            self._block(incident["detail"])
        return self._stop_result()

    def diagnose(self, action_id: str, item: dict, payload: dict):
        """Validate, reconcile the version, settle after, and diagnose.

        Returns ``None`` for a correct (clean) add, a ``{"_stop": ...}`` dict for
        a safety stop, or an incident dict for a recoverable/blocking mismatch.
        Does NOT perform any correction.
        """
        if self._before is None:
            return {"_stop": self._stop("no before snapshot")}

        sent_v, sent_sku, sent_qty = (payload.get("sent_version"), payload.get("sent_sku"),
                                      payload.get("sent_quantity"))
        if sent_v != self._before["version"]:
            return {"_stop": self._stop(f"sent version {sent_v!r} != proven {self._before['version']}")}
        if sent_sku != item["sku"]:
            return {"_stop": self._stop(f"sent sku {sent_sku!r} != requested {item['sku']!r}")}
        if sent_qty != item["quantity"]:
            return {"_stop": self._stop(f"sent quantity {sent_qty!r} != requested {item['quantity']!r}")}

        status = payload.get("response_status")
        rv = payload.get("response_version")
        before_v = self._before["version"]
        if status is None:
            self._record_incident({"kind": "unknown_commit", "detail": "no response (transport failure)",
                                   "action_id": action_id})
            return {"_stop": self._stop("unknown commit (no response)")}
        if status == 409:
            self._record_incident({"kind": "conflict_409", "detail": "version conflict",
                                   "action_id": action_id,
                                   "current_version": payload.get("response_current_version")})
            return {"_stop": self._stop("version conflict on add")}
        if status in (400, 404):
            self._record_incident({"kind": "rejected", "detail": f"documented rejection HTTP {status}",
                                   "action_id": action_id})
            return {"_stop": self._stop(f"add rejected (HTTP {status})")}
        if status >= 500 or status < 200 or not isinstance(rv, int) or isinstance(rv, bool):
            self._record_incident({"kind": "unknown_commit",
                                   "detail": f"status {status} version {rv!r}", "action_id": action_id})
            return {"_stop": self._stop("unknown commit")}

        if rv == before_v:
            self.verifier.known_version = before_v
        elif rv == before_v + 1:
            self.verifier.known_version = before_v + 1
        else:
            self.verifier.known_version = rv
            self._record_incident({"kind": "external_mutation",
                                   "detail": f"version advanced to {rv}", "action_id": action_id})
            return {"_stop": self._stop("unexplained version advance")}

        res = self.verifier.settle(self.verifier.known_version, phase=f"after:{action_id}")
        if res["status"] != "settled":
            return {"_stop": self._stop(self._settle_stop_reason(res["status"]))}

        after = res["view"]
        self.last_state = after
        incident = self._diagnose(payload, self._before, after)
        incident["action_id"] = action_id
        self._record_incident(incident)
        if incident["kind"] == "correct":
            return None
        return incident

    # ------------------------------------------------------------------ #
    # Diagnosis
    # ------------------------------------------------------------------ #

    def _diagnose(self, payload: dict, before: dict, after: dict) -> dict:
        req_sku = payload.get("requested_sku")
        req_qty = payload.get("quantity")
        rv = payload.get("response_version")
        before_v = before["version"]
        after_v = after["version"]
        before_map = quantities(before["lines"])
        after_map = quantities(after["lines"])
        base = {"req_sku": req_sku, "req_qty": req_qty}

        # Silent drop: a documented 200 with an unchanged version is only a
        # silent drop when the settled state is provably unchanged (lines AND
        # total). Anything else with an unchanged version is unknown_commit.
        if isinstance(rv, int) and not isinstance(rv, bool) and rv == before_v:
            if (after_v == before_v and after_map == before_map
                    and after.get("total_cents") == before.get("total_cents")):
                return {**base, "kind": "silent_drop",
                        "detail": "no commit: version and state unchanged"}
            return {**base, "kind": "unknown_commit",
                    "detail": "version unchanged but state changed"}

        # stale total first (computed total vs displayed total).
        if recompute_total(after["lines"]) != after.get("total_cents"):
            return {**base, "kind": "stale_total",
                    "detail": "displayed total != recomputed total"}

        delta = _delta(before_map, after_map)

        if delta == {req_sku: req_qty}:
            return {**base, "kind": "correct", "delta": delta}

        if len(delta) == 1 and req_sku not in delta:
            wrong_sku, dq = next(iter(delta.items()))
            if dq > 0 and wrong_sku not in self.baseline_skus and wrong_sku not in before_map:
                line_id, _ = self._line_for_sku(after, wrong_sku)
                return {**base, "kind": "wrong_variant", "wrong_sku": wrong_sku,
                        "wrong_qty": dq, "line_id": line_id,
                        "detail": f"wrote {wrong_sku} instead of {req_sku}"}
            return {**base, "kind": "ownership_uncertain",
                    "detail": f"affected sku {wrong_sku} not newly created"}

        if len(delta) == 1 and delta.get(req_sku, 0) > req_qty:
            dq = delta[req_sku]
            if req_sku not in self.baseline_skus and req_sku not in before_map:
                line_id, _ = self._line_for_sku(after, req_sku)
                return {**base, "kind": "excess_quantity", "sku": req_sku,
                        "observed_delta": dq, "intended": req_qty, "line_id": line_id,
                        "detail": f"{req_sku} written x{req_qty + (dq - req_qty)} extra"}
            return {**base, "kind": "ownership_uncertain",
                    "detail": f"requested sku {req_sku} already present before add"}

        return {**base, "kind": "unsupported", "delta": delta,
                "detail": f"unexplained delta {delta}"}

    def _line_for_sku(self, view: dict, sku: str):
        for l in view.get("lines") or []:
            if l.get("sku") == sku:
                return l.get("line_id"), l.get("quantity")
        return None, None

    # ------------------------------------------------------------------ #
    # Recovery execution
    # ------------------------------------------------------------------ #

    def recover(self, action_id: str, item: dict, incident: dict):
        kind = incident["kind"]
        if kind == "silent_drop":
            self._recover_silent_drop(action_id, item, incident)
        elif kind == "wrong_variant":
            self._recover_wrong_variant(action_id, item, incident)
        elif kind == "excess_quantity":
            self._recover_excess(action_id, item, incident)

    def _budget_ok(self, action_id: str, steps: int) -> bool:
        if self.writes_attempted + steps > MAX_WRITES_PER_RUN:
            self._block(f"write budget exhausted (need {steps}, "
                        f"{MAX_WRITES_PER_RUN - self.writes_attempted} left)")
            return False
        if self.plans_by_target.get(action_id, 0) >= MAX_PLANS_PER_TARGET:
            self._block(f"plan budget exhausted for {action_id}")
            return False
        return True

    def _recover_excess(self, action_id, item, incident):
        steps = 1
        if not self._budget_ok(action_id, steps):
            return
        self.plans_by_target[action_id] = self.plans_by_target.get(action_id, 0) + 1
        expected = self.verifier.known_version
        ev = self.session.set_line_quantity(incident["line_id"], incident["sku"],
                                            incident["intended"], expected)
        self.writes_attempted += 1
        self.audit.append({"action_id": action_id, "phase": "correction",
                           "operation": "patch", "evidence": ev.to_dict()})
        outcome = self._classify_correction(ev.payload, expected)
        if outcome != "committed":
            self._block(f"correction PATCH {outcome}")
            return
        res = self.verifier.settle(self.verifier.known_version, phase=f"plan:{action_id}")
        if res["status"] != "settled":
            self._block(f"post-correction settle {res['status']}")
            return
        self.last_state = res["view"]
        if self._plan_achieved(incident, res["view"]):
            self._resolve(incident)
        else:
            self._block("correction did not achieve intended quantity")

    def _recover_wrong_variant(self, action_id, item, incident):
        steps = 2
        if not self._budget_ok(action_id, steps):
            return
        self.plans_by_target[action_id] = self.plans_by_target.get(action_id, 0) + 1

        # Step 1: delete the wrong line.
        expected = self.verifier.known_version
        ev = self.session.delete_line(incident["line_id"], incident["wrong_sku"], expected)
        self.writes_attempted += 1
        self.audit.append({"action_id": action_id, "phase": "correction",
                           "operation": "delete", "evidence": ev.to_dict()})
        outcome = self._classify_correction(ev.payload, expected)
        if outcome != "committed":
            self._block(f"correction DELETE {outcome}")
            return
        res = self.verifier.settle(self.verifier.known_version, phase=f"plan-delete:{action_id}")
        if res["status"] != "settled":
            self._block(f"post-delete settle {res['status']}")
            return
        self.last_state = res["view"]

        # Step 2: add the intended sku through the browser action.
        add_payload = self._corrective_add(action_id, item)
        if add_payload is None:
            return  # already blocked
        self.writes_attempted += 1
        self.audit.append({"action_id": action_id, "phase": "correction",
                           "operation": "add", "evidence": add_payload})
        # classify the corrective add's response
        status = add_payload.get("response_status")
        rv = add_payload.get("response_version")
        exp = self.verifier.known_version
        if status == 200 and isinstance(rv, int) and rv == exp + 1:
            self.verifier.known_version = exp + 1
        else:
            self._block(f"corrective add failed (status {status}, version {rv!r})")
            return
        res = self.verifier.settle(self.verifier.known_version, phase=f"plan-add:{action_id}")
        if res["status"] != "settled":
            self._block(f"post-add settle {res['status']}")
            return
        self.last_state = res["view"]
        if self._plan_achieved(incident, res["view"]):
            self._resolve(incident)
        else:
            self._block("correction did not achieve intended state (fault may persist)")

    def _recover_silent_drop(self, action_id, item, incident):
        while self.plans_by_target.get(action_id, 0) < MAX_PLANS_PER_TARGET:
            if self.writes_attempted + 1 > MAX_WRITES_PER_RUN:
                self._block("write budget exhausted")
                return
            self.plans_by_target[action_id] = self.plans_by_target.get(action_id, 0) + 1
            add_payload = self._corrective_add(action_id, item)
            if add_payload is None:
                return
            self.writes_attempted += 1
            self.audit.append({"action_id": action_id, "phase": "correction",
                               "operation": "retry", "evidence": add_payload})
            status = add_payload.get("response_status")
            rv = add_payload.get("response_version")
            before_v = self._before["version"]
            if status == 200 and isinstance(rv, int) and rv == before_v + 1:
                self.verifier.known_version = before_v + 1
            elif status == 200 and isinstance(rv, int) and rv == before_v:
                self.verifier.known_version = before_v
            else:
                self._block(f"retry add failed (status {status}, version {rv!r})")
                return
            res = self.verifier.settle(self.verifier.known_version, phase=f"retry:{action_id}")
            if res["status"] != "settled":
                self._block(f"post-retry settle {res['status']}")
                return
            self.last_state = res["view"]
            if self._plan_achieved(incident, res["view"]):
                self._resolve(incident)
                return
            # still not committed: re-diagnose and loop (retry again if allowed)
            d = self._diagnose(add_payload, self._before, res["view"])
            if d["kind"] != "silent_drop":
                self._block(f"retry produced {d['kind']}: {d['detail']}")
                return
        self._block(f"silent-drop plan budget exhausted for {action_id}")

    def _corrective_add(self, action_id, item):
        product_id = item.get("product_id")
        sku = item["sku"]
        qty = item["quantity"]
        expected = self.verifier.known_version
        open_ev = self.session.open_product(product_id)
        embedded = open_ev.payload.get("cart_version")
        if embedded != expected:
            self._block(f"page version {embedded!r} != proven {expected}")
            return None
        self.session.select_variant(sku)
        add_ev = self.session.add_to_cart(qty, sku=sku)
        p = add_ev.payload
        if p.get("sent_version") != expected:
            self._block(f"sent version {p.get('sent_version')!r} != proven {expected}")
            return None
        if p.get("sent_sku") != sku or p.get("sent_quantity") != qty:
            self._block(f"sent sku/qty mismatch: {p.get('sent_sku')}/{p.get('sent_quantity')} vs {sku}/{qty}")
            return None
        return add_ev.to_dict()["payload"]

    def _classify_correction(self, payload: dict, expected: int) -> str:
        status = payload.get("response_status")
        rv = payload.get("response_version")
        if status == 200 and isinstance(rv, int) and rv == expected + 1:
            self.verifier.known_version = expected + 1
            return "committed"
        if status == 409:
            self._block(f"correction version conflict (current {payload.get('response_current_version')})")
            return "conflict"
        if status in (400, 404):
            self._block(f"correction rejected (HTTP {status})")
            return "rejected"
        self._block(f"correction unknown outcome (status {status}, version {rv!r})")
        return "unknown"

    def _plan_achieved(self, incident: dict, after: dict) -> bool:
        amap = quantities(after["lines"])
        if incident["kind"] == "silent_drop":
            return amap.get(incident["req_sku"], 0) == incident["req_qty"]
        if incident["kind"] == "wrong_variant":
            return (amap.get(incident["req_sku"], 0) == incident["req_qty"]
                    and amap.get(incident["wrong_sku"], 0) == 0)
        if incident["kind"] == "excess_quantity":
            return amap.get(incident["sku"], 0) == incident["intended"]
        return False

    # ------------------------------------------------------------------ #
    # Completion
    # ------------------------------------------------------------------ #

    def complete(self):
        res = self.verifier.settle(self.verifier.known_version, phase="final")
        final_status = res["status"]
        final_view = res["view"]
        constraints = []
        confirmed = 0
        delta = None
        delta_ok = None
        protected_ok = True
        protected_detail = None

        if final_status == "settled" and not self.verifier.anomaly_epoch:
            for c in self.ledger.constraints:
                state, reason = evaluate(c, final_view)
                constraints.append({"label": c.label(), "state": state, "reason": reason,
                                    "provenance": res["provenance"]})
                if state == CONFIRMED:
                    confirmed += 1
            delta = _delta(quantities(self.original_baseline["lines"]), quantities(final_view["lines"]))
            delta_ok = (delta == self.declared)
            protected_ok, protected_detail = self._check_protected(final_view)
            self.last_state = final_view
        else:
            for c in self.ledger.constraints:
                constraints.append({"label": c.label(), "state": UNKNOWN,
                                    "reason": self._completion_failure_reason(final_status, c),
                                    "provenance": ()})
            if final_view is not None:
                delta = _delta(quantities(self.original_baseline["lines"]), quantities(final_view["lines"]))
                delta_ok = (delta == self.declared)
            protected_ok = False
            protected_detail = f"final view not settled ({final_status})"

        blocking = bool(self.blocking) or self.stop_reason is not None
        complete = (final_status == "settled" and not self.verifier.anomaly_epoch
                    and confirmed == len(self.ledger.constraints)
                    and delta_ok is True and protected_ok and not blocking)
        verdict = "COMPLETE" if complete else "PARTIAL"

        report = {
            "ledger": self.ledger.to_dict(),
            "constraints": constraints,
            "incidents": list(self.incidents),
            "blocking": list(self.blocking),
            "resolved": list(self.resolved),
            "stop_reason": self.stop_reason,
            "writes_attempted": self.writes_attempted,
            "plans_by_target": dict(self.plans_by_target),
            "original_baseline": {"version": self.original_baseline["version"],
                                  "lines": quantities(self.original_baseline["lines"])},
            "final": _view_summary(final_view),
            "last_state": _view_summary(self.last_state),
            "declared_additions": self.declared,
            "delta": delta,
            "delta_consistent": delta_ok,
            "protected_ok": protected_ok,
            "protected_detail": protected_detail,
            "audit": list(self.audit),
            "samples": list(self.verifier.samples),
            "settle_ms": self.verifier.settle_ms,
            "verdict": verdict,
            "confirmed_count": confirmed,
            "total": len(self.ledger.constraints),
        }
        return report, verdict

    def _check_protected(self, final_view: dict):
        final_by_id = {l["line_id"]: l for l in (final_view.get("lines") or [])}
        for line_id, bl in self.baseline_lines.items():
            fl = final_by_id.get(line_id)
            if fl is None:
                return False, f"baseline line {line_id} ({bl['sku']}) missing"
            if fl.get("sku") != bl.get("sku"):
                return False, f"baseline line {line_id} sku changed"
            if fl.get("quantity", 0) < bl.get("quantity", 0):
                return False, f"baseline line {line_id} quantity reduced"
            if fl.get("unit_price_cents") != bl.get("unit_price_cents"):
                return False, f"baseline line {line_id} unit price changed"
        return True, None

    def _completion_failure_reason(self, status: str, c: Constraint) -> str:
        if self.verifier.anomaly_epoch:
            return "anomaly epoch: state changed by an unexplained external write"
        if status == "conflict":
            return "conflicting DOM/API sources"
        if status == "malformed":
            return "malformed or insufficient evidence"
        if status == "stale":
            return "no fresh settled view within deadline (version below known_version)"
        return "no fresh settled view within deadline"

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #

    def _settle_stop_reason(self, status: str) -> str:
        if status == "conflict":
            return "conflicting DOM/API sources"
        if status == "malformed":
            return "malformed or insufficient evidence"
        if status == "stale":
            return "no fresh settled view within deadline"
        if status == "anomaly":
            return "unexplained version advance during settle"
        return f"settlement failed ({status})"

    def _record_incident(self, incident: dict):
        self.incidents.append(incident)

    def _resolve(self, incident: dict):
        incident["resolution"] = "resolved"
        self.resolved.append(incident)

    def _block(self, detail: str):
        self.blocking.append({"kind": "blocked", "detail": detail})
        self.stop_reason = self.stop_reason or detail

    def _stop(self, reason: str):
        self.stop_reason = self.stop_reason or reason
        self.blocking.append({"kind": "stop", "detail": reason})
        return {"stop": True, "reason": reason}

    def _stop_result(self):
        if self.stop_reason:
            return {"stop": True, "reason": self.stop_reason}
        return None


def _view_summary(view: Optional[dict]) -> dict:
    if not view:
        return {"version": None, "lines": {}, "total_cents": None}
    return {"version": view.get("version"),
            "lines": quantities(view.get("lines") or []),
            "total_cents": view.get("total_cents")}


def _fmt_delta(d: dict) -> str:
    return "{" + ", ".join(f"{k}: {v:+d}" for k, v in sorted(d.items())) + "}"


def render_recovery_report(report: dict) -> str:
    out = []
    out.append("")
    out.append(f"=== CartProof recovery report (settle_ms={report.get('settle_ms')}) ===")
    out.append("")
    out.append(f"Ledger (version {report['ledger']['version']}):")
    out.append(json.dumps(report["ledger"], indent=2))
    out.append("")
    out.append("Constraints:")
    for c in report["constraints"]:
        prov = ", ".join(f"{p['source']}@v{p['version']}" for p in c.get("provenance") or ())
        suffix = f"{c.get('reason')}; {prov}" if prov else c.get("reason") or ""
        out.append(f"  - {c['label']} -> {c['state']}  ({suffix})")
    out.append("")
    out.append(f"Original baseline: version {report['original_baseline']['version']}, "
               f"lines {report['original_baseline']['lines']}")
    out.append(f"Final: {_fmt_view(report.get('final'))}")
    out.append(f"Last known state: {_fmt_view(report.get('last_state'))}")
    if report.get("delta") is not None:
        out.append("Delta (final - baseline): " + _fmt_delta(report["delta"]))
        out.append("Declared additions: " + _fmt_delta(report["declared_additions"]))
    out.append(f"Protected items preserved: {report['protected_ok']}"
               + (f" ({report['protected_detail']})" if report.get("protected_detail") else ""))
    out.append("")
    out.append(f"Budget: {report['writes_attempted']}/{MAX_WRITES_PER_RUN} corrective writes, "
               f"plans per target {report['plans_by_target']} (max {MAX_PLANS_PER_TARGET})")
    out.append("")
    if report["resolved"]:
        out.append("Resolved incidents:")
        for i in report["resolved"]:
            out.append(f"  - [{i['kind']}] action {i.get('action_id')}: {i.get('detail')} -> resolved")
    else:
        out.append("Resolved incidents: (none)")
    if report["blocking"]:
        out.append("Blocking incidents:")
        for b in report["blocking"]:
            out.append(f"  - {b['kind']}: {b['detail']}")
    else:
        out.append("Blocking incidents: (none)")
    if report.get("stop_reason"):
        out.append(f"Stop reason: {report['stop_reason']}")
    out.append("")
    out.append("Recovery audit:")
    for step in report.get("audit") or []:
        out.append(f"  - {step}")
    out.append("")
    if report["verdict"] == "COMPLETE":
        out.append("VERDICT: COMPLETE")
    else:
        out.append(f"VERDICT: PARTIAL ({report['confirmed_count']}/{report['total']} constraints confirmed)")
        for c in report["constraints"]:
            if c["state"] != CONFIRMED:
                out.append(f"  - {c['label']}: {c['state']} ({c['reason']})")
        for b in report["blocking"]:
            out.append(f"  - blocking: {b['kind']} - {b['detail']}")
    return "\n".join(out)


def _fmt_view(view: Optional[dict]) -> str:
    if not view or view.get("version") is None:
        return "n/a"
    return f"version {view['version']}, lines {view['lines']}, total {view['total_cents']}"

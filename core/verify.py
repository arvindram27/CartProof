"""Verification loop: baseline, per-action version tracking, settlement,
anomaly detection, and the completion tier.

This module is deterministic and evidence-only. It never imports from ``store/``
and never touches ``/_control/`` or fault configuration - it sees only what a
real agent could see through the adapter: observation rereads (DOM and API,
each carrying the cart ``version`` and contents), action evidence, and the task
spec.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .checks import evaluate, quantities, recompute_total
from .ledger import CONFIRMED, UNKNOWN, Constraint, Ledger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version(payload: Any) -> Optional[int]:
    if isinstance(payload, dict):
        v = payload.get("version")
        return v if isinstance(v, int) and not isinstance(v, bool) else None
    return None


def _norm_lines(lines: Any):
    if not isinstance(lines, list):
        return None
    try:
        return tuple(sorted((str(l["sku"]), int(l["quantity"])) for l in lines))
    except (KeyError, TypeError, ValueError):
        return None


def _coherent(dom: Any, api: Any):
    """Compare DOM and API views: same version first, then content.

    Returns (ok, kind) where kind is 'ok', 'version', 'lines', 'total', or
    'malformed'.
    """
    if not isinstance(dom, dict) or not isinstance(api, dict):
        return False, "malformed"
    dv, av = _version(dom), _version(api)
    if dv is None or av is None:
        return False, "malformed"
    if dv != av:
        return False, "version"
    dl = _norm_lines(dom.get("lines") or [])
    al = _norm_lines(api.get("lines") or [])
    if dl is None or al is None:
        return False, "malformed"
    if dl != al:
        return False, "lines"
    if dom.get("total_cents") != api.get("total_cents"):
        return False, "total"
    return True, "ok"


def _delta(baseline_lines: list, final_lines: list) -> dict:
    b = quantities(baseline_lines or [])
    f = quantities(final_lines or [])
    out = {}
    for sku in sorted(set(b) | set(f)):
        d = f.get(sku, 0) - b.get(sku, 0)
        if d != 0:
            out[sku] = d
    return out


def _best_effort_state(dom: Any, api: Any) -> dict:
    for p in (api, dom):
        if isinstance(p, dict):
            return {
                "version": p.get("version"),
                "lines": p.get("lines") or [],
                "total_cents": p.get("total_cents"),
            }
    return {"version": None, "lines": [], "total_cents": None}


def _state_summary(view: Optional[dict]) -> dict:
    if not view:
        return {"version": None, "lines": [], "total_cents": None}
    return {
        "version": view["version"],
        "lines": view.get("lines") or [],
        "total_cents": view.get("total_cents"),
    }


class Verifier:
    def __init__(self, reader, settle_ms: int = 2000, interval_ms: int = 100,
                 now_ms_fn=None, sleep_ms_fn=None):
        self.reader = reader  # () -> (dom_payload, api_payload)
        self.settle_ms = settle_ms
        self.interval_ms = interval_ms
        self.now_ms_fn = now_ms_fn or (lambda: int(time.monotonic() * 1000))
        self.sleep_ms_fn = sleep_ms_fn or (lambda ms: time.sleep(ms / 1000.0))
        self.known_version: Optional[int] = None
        self.baseline: Optional[dict] = None
        self.anomalies: list = []
        self.anomaly_epoch = False
        self.action_outcomes: list = []
        # Baseline has no freshness requirement and no fault active, so it uses a
        # fixed generous deadline (the standard 3-sample rule) independent of the
        # tuneable settle deadline.
        self.baseline_settle_ms = 2000

    # ------------------------------------------------------------------ #
    # Baseline
    # ------------------------------------------------------------------ #
    def take_baseline(self) -> dict:
        status, dom, api = self._sample(None, deadline_ms=self.baseline_settle_ms)
        if status != "settled":
            raise RuntimeError(f"baseline did not settle (status={status})")
        self.baseline = {
            "version": api["version"],
            "lines": api["lines"],
            "total_cents": api["total_cents"],
        }
        self.known_version = self.baseline["version"]
        return self.baseline

    # ------------------------------------------------------------------ #
    # Per-action version tracking + anomaly detection
    # ------------------------------------------------------------------ #
    def on_action(self, payload: dict) -> dict:
        status = payload.get("response_status")
        rv = payload.get("response_version")
        known = self.known_version
        outcome = {"action": payload.get("action"), "status": status, "response_version": rv}

        if status == 200:
            if isinstance(rv, int) and isinstance(known, int) and rv == known + 1:
                self.known_version = known + 1
                outcome["outcome"] = "committed"
            elif isinstance(rv, int) and known is not None and rv == known:
                self._anomaly(
                    "action_outcome_mismatch",
                    "action reported success but the store version did not advance",
                )
                outcome["outcome"] = "mismatch"
            elif isinstance(rv, int) and known is not None and rv > known + 1:
                self._anomaly(
                    "external_version_advance",
                    f"response version {rv} jumped past known_version {known}",
                )
                self.known_version = rv
                self.anomaly_epoch = True
                outcome["outcome"] = "external"
            else:
                self._anomaly(
                    "unknown_commit",
                    f"cannot prove the add committed (status {status}, response_version {rv!r})",
                )
                outcome["outcome"] = "unknown_commit"
        elif status in (400, 404, 409):
            # Documented store error -> contractual rollback (no version increment).
            if status == 409:
                cur = payload.get("response_current_version")
                if isinstance(cur, int) and known is not None and cur > known:
                    self._anomaly(
                        "external_version_advance",
                        f"409 rejection carries current_version {cur} > known_version {known}",
                    )
                    self.known_version = cur
                    self.anomaly_epoch = True
                    outcome["outcome"] = "rejected_external"
                else:
                    outcome["outcome"] = "rejected"
            else:
                outcome["outcome"] = "rejected"
        else:
            self._anomaly("unknown_commit", f"add failed to prove a commit (status {status!r})")
            outcome["outcome"] = "unknown_commit"

        self.action_outcomes.append(outcome)
        return outcome

    # ------------------------------------------------------------------ #
    # Completion tier
    # ------------------------------------------------------------------ #
    def finish(self, ledger: Ledger, declared_additions: dict) -> dict:
        status, dom, api = self._sample(self.known_version)

        constraints_report = []
        confirmed = 0
        total = len(ledger.constraints)

        if status == "settled":
            view = {
                "version": api["version"],
                "lines": api["lines"],
                "total_cents": api["total_cents"],
            }
            ts = _now_iso()
            for c in ledger.constraints:
                state, reason = evaluate(c, view)
                prov = (
                    {"source": "dom_reread", "version": view["version"], "timestamp": ts},
                    {"source": "api_reread", "version": view["version"], "timestamp": ts},
                )
                constraints_report.append({
                    "label": c.label(), "state": state, "provenance": prov,
                    "reason": reason, "historical": False,
                })
                if state == CONFIRMED:
                    confirmed += 1
            delta = _delta(self.baseline["lines"], view["lines"])
            delta_ok = (delta == declared_additions)
            if not delta_ok:
                self._anomaly(
                    "delta_mismatch",
                    f"declared additions {declared_additions} but observed delta {delta}",
                )
            preserved = [sku for sku in quantities(self.baseline["lines"]) if sku not in delta]
            final_info = view
        else:
            delta = None
            delta_ok = None
            preserved = None
            final_info = _best_effort_state(dom, api)
            obs = final_info.get("lines")
            for c in ledger.constraints:
                constraints_report.append({
                    "label": c.label(), "state": UNKNOWN, "provenance": (),
                    "reason": self._completion_failure_reason(status, c, obs),
                    "historical": False,
                })

        complete = (confirmed == total) and (not self.anomalies) and (delta_ok is True)
        verdict = "COMPLETE" if complete else "PARTIAL"

        return {
            "ledger": ledger.to_dict(),
            "constraints": constraints_report,
            "anomalies": list(self.anomalies),
            "baseline": _state_summary(self.baseline),
            "final": final_info,
            "declared_additions": declared_additions,
            "delta": delta,
            "delta_consistent": delta_ok,
            "preserved": preserved,
            "verdict": verdict,
            "confirmed_count": confirmed,
            "total": total,
        }

    def _completion_failure_reason(self, status: str, c: Constraint, obs) -> str:
        if self.anomaly_epoch:
            extra = ""
            if obs is not None:
                if c.type == "item_quantity":
                    q = quantities(obs).get(c.sku, 0)
                    extra = (f" (current view shows quantity {q} vs required {c.quantity}; "
                             f"extra unit not attributable to the agent)")
                elif c.type == "count_distinct":
                    present = {s for s, n in quantities(obs).items() if n > 0}
                    extra = f" (current view shows {len(present & set(c.skus))} distinct)"
                elif c.type == "budget_max":
                    extra = f" (current view total {recompute_total(obs)})"
            return "anomaly epoch: state changed by an unexplained external write" + extra
        if status == "conflict":
            return "conflicting DOM/API sources"
        if status == "stale":
            return "no fresh settled view within deadline (version below known_version)"
        return "no fresh settled view within deadline"

    # ------------------------------------------------------------------ #
    # Settlement sampling
    # ------------------------------------------------------------------ #
    def _sample(self, known_version: Optional[int], deadline_ms: Optional[int] = None):
        deadline = self.now_ms_fn() + (deadline_ms if deadline_ms is not None else self.settle_ms)
        last_sig = None
        streak = 0
        conflict_kind = None
        last_coherent_version = None
        dom = api = None
        while True:
            dom, api = self.reader()
            dv, av = _version(dom), _version(api)
            if known_version is not None:
                for v in (dv, av):
                    if v is not None and v > known_version:
                        self._anomaly(
                            "external_version_advance",
                            f"observed version {v} exceeds known_version {known_version}",
                        )
                        self.anomaly_epoch = True
                        return "anomaly", dom, api
            ok, kind = _coherent(dom, api)
            if not ok:
                conflict_kind = kind
                streak = 0
                last_sig = None
            else:
                last_coherent_version = dv
                sig = (dv, _norm_lines(api.get("lines") or []), api.get("total_cents"))
                if known_version is not None and dv is not None and dv < known_version:
                    # provably stale: never accumulates settlement
                    streak = 0
                    last_sig = None
                else:
                    if sig == last_sig:
                        streak += 1
                    else:
                        streak = 1
                        last_sig = sig
                    if streak >= 3:
                        return "settled", dom, api
            if self.now_ms_fn() >= deadline:
                break
            self.sleep_ms_fn(self.interval_ms)
        if conflict_kind is not None:
            return "conflict", dom, api
        if (known_version is not None and last_coherent_version is not None
                and last_coherent_version < known_version):
            return "stale", dom, api
        return "timeout", dom, api

    def _anomaly(self, kind: str, detail: str) -> None:
        self.anomalies.append({"kind": kind, "detail": detail})


def _fmt_delta(d: dict) -> str:
    return "{" + ", ".join(f"{k}: {v:+d}" for k, v in sorted(d.items())) + "}"


def _fmt_state(view: Optional[dict]) -> str:
    if not view or view.get("version") is None:
        return "n/a"
    lines = quantities(view.get("lines") or [])
    return f"version {view['version']}, lines {lines}, total {view.get('total_cents')}"


def render_report(report: dict) -> str:
    out = []
    out.append("")
    out.append("=== CartProof verification report ===")
    out.append("")
    out.append(f"Ledger (version {report['ledger']['version']}):")
    out.append(json.dumps(report["ledger"], indent=2))
    out.append("")
    out.append("Constraints:")
    for c in report["constraints"]:
        prov = ", ".join(f"{p['source']}@v{p['version']}" for p in c.get("provenance") or ())
        detail = c.get("reason") or ""
        suffix = f"{detail}; {prov}" if prov else detail
        hist = " [historical]" if c.get("historical") else ""
        out.append(f"  - {c['label']} -> {c['state']}{hist}  ({suffix})")
    out.append("")
    out.append("Baseline: " + _fmt_state(report.get("baseline")))
    if report.get("final") and report["final"].get("version") is not None:
        out.append("Final: " + _fmt_state(report["final"]))
    if report.get("delta") is not None:
        out.append("Delta (final - baseline): " + _fmt_delta(report["delta"]))
        out.append("Declared additions: " + _fmt_delta(report["declared_additions"]))
    if report.get("preserved"):
        out.append("Preserved (seeded, unchanged): " + ", ".join(sorted(report["preserved"])))
    out.append("")
    if report["anomalies"]:
        out.append("Anomalies:")
        for a in report["anomalies"]:
            out.append(f"  - {a['kind']}: {a['detail']}")
    else:
        out.append("Anomalies: (none)")
    out.append("")
    if report["verdict"] == "COMPLETE":
        out.append("VERDICT: COMPLETE")
    else:
        out.append(f"VERDICT: PARTIAL ({report['confirmed_count']}/{report['total']} constraints confirmed)")
        for c in report["constraints"]:
            if c["state"] != CONFIRMED:
                out.append(f"  - {c['label']}: {c['state']} ({c['reason']})")
        for a in report["anomalies"]:
            out.append(f"  - anomaly: {a['kind']} - {a['detail']}")
    return "\n".join(out)

"""Verification loop: baseline, per-action version tracking, settlement,
anomaly detection, and the completion tier.

This module is deterministic and evidence-only. It never imports from ``store/``
and never touches ``/_control/`` or fault configuration - it sees only what a
real agent could see through the adapter: observation rereads (DOM and API,
each carrying the cart ``version`` and contents), action evidence, and the task
spec.

Settlement uses the owned store's monotonic per-cart version to order reads
against writes. A sample completed after the settle deadline cannot confirm; a
sample whose channels disagree on version (with the lower below ``known_version``)
is provably stale, not a source conflict; a genuinely conflicting same-version
line/total mismatch is a source conflict; malformed evidence is ``unknown``.
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


def _is_read_timeout(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("read_timeout") is True


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
    'malformed'. Timestamps are never part of this comparison.
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


def _min_version(dom: Any, api: Any) -> Optional[int]:
    vs = [v for v in (_version(dom), _version(api)) if v is not None]
    return min(vs) if vs else None


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
        if isinstance(p, dict) and not _is_read_timeout(p):
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
        self.reader = reader  # reader(timeout_ms) -> (dom_payload, api_payload)
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
        # Append-only audit log of every settlement reread; provenance references
        # entries in here, never report-time placeholders.
        self.samples: list = []
        self._sample_counter = 0
        self._settled_provenance: tuple = ()

    # ------------------------------------------------------------------ #
    # Baseline
    # ------------------------------------------------------------------ #
    def take_baseline(self) -> dict:
        status, dom, api = self._sample(None, deadline_ms=self.baseline_settle_ms, phase="baseline")
        if status != "settled":
            raise RuntimeError(f"baseline did not settle (status={status})")
        self.baseline = {
            "version": api["version"],
            "lines": api["lines"],
            "total_cents": api["total_cents"],
        }
        self.known_version = self.baseline["version"]
        return self.baseline

    def settle(self, known_version: Optional[int] = None, deadline_ms: Optional[int] = None,
               phase: str = "recovery", coherence=None, signature=None,
               on_incoherent=None):
        """Public settlement entry used by the recovery controller.

        Wraps the settlement machinery without exposing private internals. With
        ``coherence``/``signature`` unset it uses the closed SKU/quantity/total
        rules; recovery passes full-state comparators. Returns ``{"status",
        "view", "dom", "api", "provenance", "sample_ids"}`` where ``view`` is a
        settled snapshot ``{version, lines, total_cents}`` only when ``status``
        is ``"settled"``, and ``dom``/``api`` are the raw last pair.

        ``on_incoherent`` is an optional recovery-only diagnostic hook invoked
        with ``(kind, dom, api, e_dom, e_api)`` for a fresh incoherent pair. It
        never changes the settlement outcome or the returned fields.
        """
        kv = self.known_version if known_version is None else known_version
        start = len(self.samples)
        status, dom, api = self._sample(kv, deadline_ms=deadline_ms, phase=phase,
                                        coherence=coherence, signature=signature,
                                        on_incoherent=on_incoherent)
        sample_ids = [s["seq"] for s in self.samples[start:]]
        if status == "settled":
            view = {
                "version": api["version"],
                "lines": api["lines"],
                "total_cents": api["total_cents"],
            }
            return {"status": "settled", "view": view, "dom": dom, "api": api,
                    "provenance": tuple(self._settled_provenance),
                    "sample_ids": sample_ids}
        return {"status": status, "view": None, "dom": dom, "api": api,
                "provenance": (), "sample_ids": sample_ids}

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
        status, dom, api = self._sample(self.known_version, phase="completion")

        constraints_report = []
        confirmed = 0
        total = len(ledger.constraints)

        if self.anomaly_epoch:
            # Defect 3: an active anomaly epoch blocks per-constraint confirmation
            # even if _sample returned "settled" at the reconciled version.
            final_info = _best_effort_state(dom, api)
            obs_lines = final_info.get("lines") or []
            delta = _delta(self.baseline["lines"], obs_lines)
            delta_ok = (delta == declared_additions)
            for c in ledger.constraints:
                constraints_report.append({
                    "label": c.label(), "state": UNKNOWN, "provenance": (),
                    "reason": self._completion_failure_reason(status, c, obs_lines),
                    "historical": False,
                })
            preserved = [sku for sku in quantities(self.baseline["lines"]) if sku not in delta]
        elif status == "settled":
            view = {
                "version": api["version"],
                "lines": api["lines"],
                "total_cents": api["total_cents"],
            }
            for c in ledger.constraints:
                state, reason = evaluate(c, view)
                constraints_report.append({
                    "label": c.label(), "state": state,
                    "provenance": self._settled_provenance,
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
            "samples": list(self.samples),
            "settle_ms": self.settle_ms,
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
        if status == "malformed":
            return "malformed or insufficient evidence"
        if status == "stale":
            return "no fresh settled view within deadline (version below known_version)"
        return "no fresh settled view within deadline"

    # ------------------------------------------------------------------ #
    # Settlement sampling
    # ------------------------------------------------------------------ #
    def _sample(self, known_version: Optional[int], deadline_ms: Optional[int] = None,
                phase: str = "completion", coherence=None, signature=None,
                on_incoherent=None):
        coh = coherence or _coherent
        sig_fn = signature or (lambda dv, api: (dv, _norm_lines(api.get("lines") or []),
                                                api.get("total_cents")))
        deadline = self.now_ms_fn() + (deadline_ms if deadline_ms is not None else self.settle_ms)
        self._settled_provenance = ()  # clear any prior attempt's provenance
        last_sig = None
        streak = 0
        streak_entries = []
        conflict_kind = None
        saw_malformed = False
        saw_read_timeout = False
        last_coherent_version = None
        dom = api = None
        while True:
            if self.now_ms_fn() >= deadline:
                break
            remaining = max(1, deadline - self.now_ms_fn())
            dom, api = self.reader(remaining)
            read_at = _now_iso()
            late = self.now_ms_fn() >= deadline
            e_dom = self._record_sample("dom_reread", dom, read_at, phase, late)
            e_api = self._record_sample("api_reread", api, read_at, phase, late)

            # Upstream guard (unchanged): any validated version ABOVE known is an
            # unexplained external advance and is always recorded.
            if known_version is not None:
                for v in (_version(dom), _version(api)):
                    if v is not None and v > known_version:
                        self._anomaly(
                            "external_version_advance",
                            f"observed version {v} exceeds known_version {known_version}",
                        )
                        self.anomaly_epoch = True
                        return "anomaly", dom, api

            # Read timeout is distinct from malformed evidence and from conflicts:
            # it feeds the deadline-expiry outcome.
            if _is_read_timeout(dom) or _is_read_timeout(api):
                saw_read_timeout = True
                streak = 0
                last_sig = None
                streak_entries = []
                if late:
                    break
                self.sleep_ms_fn(self.interval_ms)
                continue

            ok, kind = coh(dom, api)
            if not ok:
                if kind == "arithmetic":
                    # Recovery-only category: DOM and API agree with each other
                    # but not with line arithmetic. Only a version-current,
                    # non-late pair is fresh enough to diagnose: a below-known
                    # read is provably stale and must not become malformed
                    # evidence or cite itself as an arithmetic inconsistency.
                    lower = _min_version(dom, api)
                    is_stale = (known_version is not None and lower is not None
                                and lower < known_version)
                    if not late and not is_stale:
                        saw_malformed = True
                        if on_incoherent is not None:
                            on_incoherent(kind, dom, api, e_dom, e_api)
                elif kind == "version":
                    lower = _min_version(dom, api)
                    is_stale = (known_version is not None and lower is not None
                                and lower < known_version)
                    if not is_stale:
                        conflict_kind = "version"
                elif kind in ("lines", "total"):
                    conflict_kind = kind
                else:
                    saw_malformed = True
                streak = 0
                last_sig = None
                streak_entries = []
            else:
                dv = _version(api)
                last_coherent_version = dv
                sig = sig_fn(dv, api)
                if known_version is not None and dv is not None and dv < known_version:
                    # provably stale: never accumulates settlement
                    streak = 0
                    last_sig = None
                    streak_entries = []
                else:
                    if sig == last_sig:
                        streak += 1
                        streak_entries.extend([e_dom, e_api])
                    else:
                        streak = 1
                        last_sig = sig
                        streak_entries = [e_dom, e_api]
                    if streak >= 3 and not late:
                        self._settled_provenance = tuple(streak_entries)
                        return "settled", dom, api

            if late:
                break
            if self.now_ms_fn() >= deadline:
                break
            self.sleep_ms_fn(self.interval_ms)

        if conflict_kind is not None:
            return "conflict", dom, api
        if saw_malformed:
            return "malformed", dom, api
        if saw_read_timeout:
            return "timeout", dom, api
        if (known_version is not None and last_coherent_version is not None
                and last_coherent_version < known_version):
            return "stale", dom, api
        return "timeout", dom, api

    def _record_sample(self, source: str, payload: Any, read_at: str,
                       phase: str, late: bool) -> dict:
        self._sample_counter += 1
        ts = payload.get("timestamp") if isinstance(payload, dict) else None
        raw_lines = payload.get("lines") if isinstance(payload, dict) else None
        entry = {
            "seq": self._sample_counter,
            "phase": phase,
            "source": source,
            "version": _version(payload),
            "lines": quantities(raw_lines) if isinstance(raw_lines, list) else {},
            "total_cents": payload.get("total_cents") if isinstance(payload, dict) else None,
            "timestamp": ts or read_at,
            "timed_out": _is_read_timeout(payload),
            "late": late,
        }
        self.samples.append(entry)
        return entry

    def _anomaly(self, kind: str, detail: str) -> None:
        self.anomalies.append({"kind": kind, "detail": detail})


def _fmt_delta(d: dict) -> str:
    return "{" + ", ".join(f"{k}: {v:+d}" for k, v in sorted(d.items())) + "}"


def _fmt_state(view: Optional[dict]) -> str:
    if not view or view.get("version") is None:
        return "n/a"
    lines = quantities(view.get("lines") or [])
    return f"version {view['version']}, lines {lines}, total {view.get('total_cents')}"


def _fmt_sample(s: dict) -> str:
    marks = []
    if s.get("late"):
        marks.append("late")
    if s.get("timed_out"):
        marks.append("read-timeout")
    suffix = f" [{', '.join(marks)}]" if marks else ""
    return (f"  #{s['seq']:<3} {s['phase']:<10} {s['source']:<11} v{s['version']} "
            f"lines={s['lines']} total={s['total_cents']} {s['timestamp']}{suffix}")


def render_report(report: dict) -> str:
    out = []
    out.append("")
    out.append(f"=== CartProof verification report (settle_ms={report.get('settle_ms')}) ===")
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
    out.append("Recorded settlement samples:")
    for s in report.get("samples") or []:
        out.append(_fmt_sample(s))
    late_count = sum(1 for s in report.get("samples") or [] if s.get("late"))
    timeout_count = sum(1 for s in report.get("samples") or [] if s.get("timed_out"))
    if late_count or timeout_count:
        out.append(f"Note: {late_count} sample(s) completed after expiry and "
                   f"{timeout_count} read(s) timed out; the settle deadline bounds sampling "
                   f"and such samples cannot confirm.")
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

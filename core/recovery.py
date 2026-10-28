"""Bounded, deterministic recovery controller for the owned mock store.

Composed with the existing verifier through its narrow public ``settle()``
method. It never imports from ``store/``, never reads the SQLite file, fault
configuration, control endpoint, or expected fixture results, and never trusts
``added.sku``/``added.line_id`` as ownership proof.

Attribution is full-state: every ownership decision and every post-correction
proof compares complete normalized line snapshots (unique nonempty line ID,
exact SKU, genuine integer quantity 1..99, integer unit price, and a total that
matches line arithmetic), never per-SKU aggregates alone. Recovery settles with
full-state comparators passed to ``Verifier.settle`` while the closed
verification-only rules stay untouched, and it records deep copies of every raw
DOM/API pair so citations point at complete observations, not aggregates.

Recovery is opt-in, conservative, and capped: it corrects only a proven,
safely-attributable mismatch (wrong variant, excess quantity on a newly created
line, or a proven no-commit silent drop), using the store's existing guarded
PATCH/DELETE and the existing browser add action. Every corrective mutation is
counted and appended at dispatch, before the adapter call, and keeps its raw
request/response/error evidence. 409, external advance, timeout, malformed or
unknown outcomes stop the run. History is append-only: a resolution is a new
event that cites the exact observations that justify it.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Optional

from .checks import evaluate, quantities
from .ledger import CONFIRMED, UNKNOWN, Constraint, Ledger

MAX_PLANS_PER_TARGET = 2
MAX_WRITES_PER_RUN = 4
MAX_QUANTITY = 99

RECOVERABLE_KINDS = {"wrong_variant", "excess_quantity", "silent_drop"}


class UnsafeState(Exception):
    """Observed state is invalid or ambiguous, so it cannot prove ownership."""


def _delta(before_map: dict, after_map: dict) -> dict:
    out = {}
    for sku in sorted(set(before_map) | set(after_map)):
        d = after_map.get(sku, 0) - before_map.get(sku, 0)
        if d != 0:
            out[sku] = d
    return out


# --------------------------------------------------------------------------- #
# Full-state validation helpers (recovery-only; never used by verify.py)
# --------------------------------------------------------------------------- #

def _is_int(v: Any) -> bool:
    """Genuine integer: booleans and floats are not acceptable evidence."""
    return type(v) is int


def _valid_version(v: Any) -> bool:
    return type(v) is int and v >= 0


def _valid_quantity(q: Any) -> bool:
    return type(q) is int and 1 <= q <= MAX_QUANTITY


def _norm_lines_full(lines: Any):
    """Canonical sorted tuple of ``(line_id, sku, quantity, unit_price_cents)``.

    Returns ``None`` when any line is invalid, identity is missing, or line IDs
    are not unique. Ambiguous identity is never treated as ownership.
    """
    if not isinstance(lines, (list, tuple)):
        return None
    out = []
    for ln in lines:
        if isinstance(ln, tuple) and len(ln) == 4:
            lid, sku, qty, price = ln
        elif isinstance(ln, dict):
            lid, sku = ln.get("line_id"), ln.get("sku")
            qty, price = ln.get("quantity"), ln.get("unit_price_cents")
        else:
            return None
        # Nonblank identity, but the actual values are preserved: whitespace is
        # only used to reject an empty-looking identity, never normalized away.
        if not (isinstance(lid, str) and lid.strip()
                and isinstance(sku, str) and sku.strip()):
            return None
        if not _valid_quantity(qty):
            return None
        if not _is_int(price) or price < 0:
            return None
        out.append((lid, sku, qty, price))
    ids = [o[0] for o in out]
    if len(set(ids)) != len(ids):
        return None
    return tuple(sorted(out))


def _snapshot(view: Any) -> dict:
    """Validate a settled view into a canonical full snapshot."""
    if not isinstance(view, dict):
        raise UnsafeState("view is not a dict")
    version = view.get("version")
    if not _valid_version(version):
        raise UnsafeState(f"invalid version {version!r}")
    lines = _norm_lines_full(view.get("lines"))
    if lines is None:
        raise UnsafeState("invalid or ambiguous line identity")
    total = view.get("total_cents")
    if not _is_int(total) or total < 0:
        raise UnsafeState(f"invalid total {total!r}")
    if total != sum(q * p for (_lid, _sku, q, p) in lines):
        raise UnsafeState(f"displayed total {total} != recomputed total")
    return {"version": version, "lines": lines, "total_cents": total}


def _line_map(snap: dict) -> dict:
    return {lid: {"sku": sku, "quantity": qty, "unit_price_cents": price}
            for (lid, sku, qty, price) in snap["lines"]}


def _snapshot_equal(a: dict, b: dict) -> bool:
    return (a["version"] == b["version"] and a["lines"] == b["lines"]
            and a["total_cents"] == b["total_cents"])


def _full_delta(before: dict, after: dict):
    """Split two full snapshots into added/removed/changed/unchanged by line ID."""
    b, a = _line_map(before), _line_map(after)
    added = {lid: a[lid] for lid in a if lid not in b}
    removed = {lid: b[lid] for lid in b if lid not in a}
    changed = {lid: (b[lid], a[lid]) for lid in b if lid in a and b[lid] != a[lid]}
    unchanged = {lid: a[lid] for lid in a if lid in b and a[lid] == b[lid]}
    return added, removed, changed, unchanged


def _coherent_full(dom: Any, api: Any):
    """Recovery coherence: typed, arithmetic-consistent full state on both sides.

    Every eligible pair must be genuine integer evidence. A float/bool total that
    compares equal by Python equality, a missing ``lines`` key, a whitespace-only
    identity or a total that disagrees with line arithmetic is malformed and can
    never contribute to the settlement streak.
    """
    if not isinstance(dom, dict) or not isinstance(api, dict):
        return False, "malformed"
    dv, av = dom.get("version"), api.get("version")
    if not _valid_version(dv) or not _valid_version(av):
        return False, "malformed"
    if dv != av:
        return False, "version"
    # No ``or []``: a missing/malformed line list must not become an empty cart.
    dl = _norm_lines_full(dom.get("lines"))
    al = _norm_lines_full(api.get("lines"))
    if dl is None or al is None:
        return False, "malformed"
    if dl != al:
        return False, "lines"
    dt, at = dom.get("total_cents"), api.get("total_cents")
    # Type/sign problems are malformed evidence, never a channel conflict.
    for total in (dt, at):
        if not _is_int(total) or total < 0:
            return False, "malformed"
    if dt != at:
        # Both sides are typed but disagree: a genuine source conflict.
        return False, "total"
    recomputed = sum(q * p for (_lid, _sku, q, p) in dl)
    if dt != recomputed:
        # DOM and API agree on version, full lines and displayed total, but that
        # shared total does not match line arithmetic. Recovery-only category.
        return False, "arithmetic"
    return True, "ok"


def _signature_full(dv: int, api: Any):
    """Full-state streak signature: version + typed lines + typed total.

    ``_coherent_full`` already rejects invalid pairs, so a valid signature is the
    normal path. The fallback carries a per-call unique object so that any
    unexpected invalid state can never equal another read's signature and
    therefore can never accumulate a settlement streak.
    """
    lines = _norm_lines_full(api.get("lines")) if isinstance(api, dict) else None
    total = api.get("total_cents") if isinstance(api, dict) else None
    if lines is None or not _is_int(total) or total < 0 or not _valid_version(dv):
        return ("invalid", id(api))
    return (dv, lines, total)


def _raw_lines(snap: dict) -> list:
    return [{"line_id": lid, "sku": sku, "quantity": qty, "unit_price_cents": price}
            for (lid, sku, qty, price) in snap["lines"]]


def _payload_lines(payload: Any) -> Optional[list]:
    """Complete relevant line fields from a raw observation payload."""
    if not isinstance(payload, dict):
        return None
    lines = payload.get("lines")
    if not isinstance(lines, list):
        return None
    return [
        {"line_id": ln.get("line_id"), "sku": ln.get("sku"),
         "quantity": ln.get("quantity"), "unit_price_cents": ln.get("unit_price_cents")}
        for ln in lines if isinstance(ln, dict)
    ]


def attach_raw_pair_recorder(verifier, raw_pairs: dict) -> None:
    """Install the recovery raw-pair recorder on a verifier's reader.

    Must be called BEFORE the first baseline read so the original baseline pairs
    are retained with their real sequence IDs. The wrapper is passive: it only
    deep-copies the pair the verifier already requested and keys it by the sample
    number that read is about to receive.
    """
    base_reader = getattr(verifier, "reader", None)
    if base_reader is None:
        return
    samples = verifier.samples

    def wrapped(timeout_ms):
        dom, api = base_reader(timeout_ms)
        raw_pairs[len(samples) + 1] = (deepcopy(dom), deepcopy(api))
        return dom, api

    verifier.reader = wrapped


def establish_recovery_baseline(verifier) -> dict:
    """Accept the original baseline only through full-state settlement.

    Recovery-only: the baseline must be proven by the same typed, full-identity
    comparators and the same three-consecutive-pair rule used for every later
    correction. The existing finite baseline deadline is kept (never the shorter
    user settle window). No-recovery initialization is untouched and keeps using
    ``Verifier.take_baseline``.
    """
    res = verifier.settle(None, deadline_ms=verifier.baseline_settle_ms,
                          phase="baseline", coherence=_coherent_full,
                          signature=_signature_full)
    if res["status"] != "settled":
        raise RuntimeError(f"recovery baseline did not settle (status={res['status']})")
    snap = _snapshot(res["view"])
    verifier.baseline = {
        "version": snap["version"],
        "lines": _raw_lines(snap),
        "total_cents": snap["total_cents"],
    }
    verifier.known_version = snap["version"]
    # Retain the exact eligible pair sequence IDs so baseline proof can be cited
    # before any mutation occurs.
    verifier.baseline_provenance = tuple(res["provenance"])
    return verifier.baseline


class RecoveryController:
    def __init__(self, verifier, session, ledger: Ledger, declared_additions: dict,
                 original_baseline: dict, raw_pairs: Optional[dict] = None):
        self.verifier = verifier
        self.session = session
        self.ledger = ledger
        self.declared = declared_additions
        self.original_baseline = original_baseline
        self.baseline_lines = {l["line_id"]: l for l in (original_baseline.get("lines") or [])}
        self.baseline_skus = {l["sku"] for l in (original_baseline.get("lines") or [])}
        self.baseline_ids = set(self.baseline_lines)
        try:
            self._baseline_snap = _snapshot(original_baseline)
        except UnsafeState:
            self._baseline_snap = None
        self._last_snap = self._baseline_snap

        self.audit: list = []          # append-only step records (write evidence)
        self.events: list = []         # append-only event log with stable IDs
        self.incidents: list = []      # all incidents (resolved + blocking)
        self.blocking: list = []       # terminal incidents
        self.resolved: list = []       # resolution records (never mutate originals)
        self.observations: dict = {}   # sample seq -> complete observation record
        self.plans_by_target: dict = {}
        self.writes_attempted = 0
        self.stop_reason: Optional[str] = None
        self.last_state = original_baseline  # last known fresh settled state
        self._before: Optional[dict] = None
        self._before_snap: Optional[dict] = None
        self._plan_counter = 0

        # Recovery-only recorder: deep copies of the complete raw DOM/API payloads,
        # keyed by the sample seq they will be recorded under. When the caller
        # already installed the recorder before the baseline read, the same dict
        # is injected here so baseline samples keep their real proof trail.
        if raw_pairs is not None:
            self.raw_pairs = raw_pairs
        else:
            self.raw_pairs = {}
            if verifier is not None and getattr(verifier, "reader", None) is not None:
                attach_raw_pair_recorder(verifier, self.raw_pairs)

        self._emit("init", version=(self._baseline_snap or {}).get("version"),
                   lines=_line_map(self._baseline_snap) if self._baseline_snap else {})

    # ------------------------------------------------------------------ #
    # Settlement wrapper (full-state only)
    # ------------------------------------------------------------------ #

    def _settle(self, phase: str) -> dict:
        incoherent = []

        def collect(kind: str, dom: Any, api: Any, e_dom: dict, e_api: dict) -> None:
            lines = _norm_lines_full(api.get("lines")) or []
            incoherent.append({
                "kind": kind,
                "version": api.get("version"),
                "displayed_total": api.get("total_cents"),
                "dom_total": dom.get("total_cents"),
                "api_total": api.get("total_cents"),
                "recomputed_total": sum(q * p for (_lid, _sku, q, p) in lines),
                "seq_dom": e_dom.get("seq"),
                "seq_api": e_api.get("seq"),
            })

        res = self.verifier.settle(self.verifier.known_version, phase=phase,
                                   coherence=_coherent_full, signature=_signature_full,
                                   on_incoherent=collect)
        self._record_observations()
        if incoherent:
            res["incoherent"] = incoherent
        return res

    def _record_observations(self) -> None:
        """Materialize complete observations for every sample recorded so far."""
        for s in self.verifier.samples:
            seq = s["seq"]
            if seq in self.observations:
                continue
            raw = self.raw_pairs.get(seq)
            if raw is None:
                raw = self.raw_pairs.get(seq - 1)
            payload = None
            if raw is not None:
                payload = raw[0] if str(s["source"]).startswith("dom") else raw[1]
            self.observations[seq] = {
                "seq": seq,
                "phase": s["phase"],
                "source": s["source"],
                "version": s["version"],
                "total_cents": s["total_cents"],
                "timestamp": s["timestamp"],
                "late": s["late"],
                "timed_out": s["timed_out"],
                "lines_full": _payload_lines(payload),
            }

    @staticmethod
    def _citations(res: dict) -> list:
        """Exact eligible full-state pairs that justified a settled outcome."""
        prov = list(res.get("provenance") or ())
        out = []
        for i in range(0, len(prov) - 1, 2):
            out.append([prov[i].get("seq"), prov[i + 1].get("seq")])
        return out

    # ------------------------------------------------------------------ #
    # Hook API (called by the actor through harness/run_task)
    # ------------------------------------------------------------------ #

    def before_add(self, event: dict):
        """Bounded fresh settled before-snapshot for every intended add.

        The ORIGINAL baseline stays immutable. A before-snapshot that disagrees
        with the last proven state is an external edit: stop, never adopt it.
        """
        action_id = event["action_id"]
        res = self._settle(f"before:{action_id}")
        if res["status"] != "settled":
            return self._stop(self._settle_stop_reason(res["status"], res))
        try:
            snap = _snapshot(res["view"])
        except UnsafeState as exc:
            return self._stop(f"before snapshot unsafe: {exc}")
        if self._last_snap is not None and not _snapshot_equal(snap, self._last_snap):
            return self._stop(
                "before snapshot drifted from the last proven state "
                f"(observed v{snap['version']} vs proven v{self._last_snap['version']})")
        self._before = res["view"]
        self._before_snap = snap
        self.audit.append({
            "action_id": action_id, "phase": "before", "version": snap["version"],
            "lines": quantities(res["view"]["lines"]), "lines_full": _line_map(snap),
            "citations": self._citations(res),
        })
        self._emit("before", action_id=action_id, version=snap["version"],
                   lines=_line_map(snap), citations=self._citations(res))
        return None

    def pre_add(self, event: dict):
        action_id = event["action_id"]
        open_ev = event.get("open") or {}
        embedded = open_ev.get("payload", {}).get("cart_version") if isinstance(open_ev, dict) else None
        if self._before is None or self._before_snap is None:
            return self._stop("no before snapshot")
        proven = self._before_snap["version"]
        if not _is_int(embedded) or embedded != proven:
            return self._stop(f"page version {embedded!r} != proven {proven}")
        self._emit("pre_add", action_id=action_id, page_version=embedded, proven_version=proven)
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
            self._block(incident.get("detail") or incident["kind"])
        return self._stop_result()

    def diagnose(self, action_id: str, item: dict, payload: dict):
        """Validate evidence, reconcile the version, settle after, and diagnose.

        Returns ``None`` for a correct (clean) add, a ``{"_stop": ...}`` dict for
        a safety stop, or an incident dict for a recoverable/blocking mismatch.
        Does NOT perform any correction.
        """
        if self._before is None:
            return {"_stop": self._stop("no before snapshot")}
        if self._before_snap is None:
            try:
                self._before_snap = _snapshot(self._before)
            except UnsafeState as exc:
                return {"_stop": self._stop(f"before snapshot unsafe: {exc}")}

        sent_v, sent_sku, sent_qty = (payload.get("sent_version"), payload.get("sent_sku"),
                                      payload.get("sent_quantity"))
        proven = self._before_snap["version"]
        if not _is_int(sent_v) or sent_v != proven:
            return {"_stop": self._stop(f"sent version {sent_v!r} != proven {proven}")}
        if not isinstance(sent_sku, str) or sent_sku != item["sku"]:
            return {"_stop": self._stop(f"sent sku {sent_sku!r} != requested {item['sku']!r}")}
        if not _is_int(sent_qty) or sent_qty != item["quantity"]:
            return {"_stop": self._stop(f"sent quantity {sent_qty!r} != requested {item['quantity']!r}")}

        status = payload.get("response_status")
        rv = payload.get("response_version")
        if status is None:
            self._record_incident({"kind": "unknown_commit",
                                   "detail": "no response (transport failure)",
                                   "action_id": action_id})
            return {"_stop": self._stop("unknown commit (no response)")}
        if not _is_int(status):
            self._record_incident({"kind": "unknown_commit",
                                   "detail": f"invalid HTTP status type {status!r}",
                                   "action_id": action_id})
            return {"_stop": self._stop("invalid HTTP status type")}
        if status == 409:
            self._record_incident({"kind": "conflict_409", "detail": "version conflict",
                                   "action_id": action_id,
                                   "current_version": payload.get("response_current_version")})
            return {"_stop": self._stop("version conflict on add")}
        if status in (400, 404):
            self._record_incident({"kind": "rejected", "detail": f"documented rejection HTTP {status}",
                                   "action_id": action_id})
            return {"_stop": self._stop(f"add rejected (HTTP {status})")}
        if status != 200:
            self._record_incident({"kind": "unknown_commit",
                                   "detail": f"undocumented status {status!r}", "action_id": action_id})
            return {"_stop": self._stop(f"undocumented HTTP status {status}")}
        if not _valid_version(rv):
            self._record_incident({"kind": "unknown_commit",
                                   "detail": f"invalid version {rv!r}", "action_id": action_id})
            return {"_stop": self._stop("invalid response version")}

        # Never adopt a response version to make a retry possible.
        if rv == proven:
            self.verifier.known_version = proven
        elif rv == proven + 1:
            self.verifier.known_version = proven + 1
        else:
            self._record_incident({"kind": "external_mutation",
                                   "detail": f"version advanced to {rv}", "action_id": action_id})
            return {"_stop": self._stop("unexplained version advance")}

        res = self._settle(f"after:{action_id}")
        if res["status"] != "settled":
            return {"_stop": self._stop(self._settle_stop_reason(res["status"], res))}
        try:
            after_snap = _snapshot(res["view"])
        except UnsafeState as exc:
            detail = str(exc)
            kind = "stale_total" if "total" in detail else "malformed_state"
            incident = {"kind": kind, "detail": detail, "action_id": action_id}
            self._record_incident(incident)
            self._emit("diagnosis", action_id=action_id,
                       incident_id=incident["incident_id"], incident_kind=kind, detail=detail)
            if kind == "stale_total":
                return {"_stop": self._stop("displayed total != recomputed total")}
            return {"_stop": self._stop(f"after state unsafe: {detail}")}

        self.last_state = res["view"]
        self._last_snap = after_snap
        incident = self._diagnose(payload, self._before_snap, after_snap)
        incident["action_id"] = action_id
        self._record_incident(incident)
        self._emit("diagnosis", action_id=action_id,
                   incident_id=incident["incident_id"], incident_kind=incident["kind"],
                   detail=incident.get("detail", ""),
                   before_lines=_line_map(self._before_snap),
                   after_lines=_line_map(after_snap),
                   citations=self._citations(res))
        if incident["kind"] == "correct":
            return None
        return incident

    # ------------------------------------------------------------------ #
    # Diagnosis (full-state only)
    # ------------------------------------------------------------------ #

    def _diagnose(self, payload: dict, before: dict, after: dict) -> dict:
        b = _snapshot(before)
        a = _snapshot(after)
        req_sku = payload.get("requested_sku")
        req_qty = payload.get("quantity")
        rv = payload.get("response_version")
        base = {"req_sku": req_sku, "req_qty": req_qty}

        if _valid_version(rv) and rv == b["version"]:
            if _snapshot_equal(b, a):
                return {**base, "kind": "silent_drop",
                        "detail": "no commit: version, full lines and total unchanged"}
            return {**base, "kind": "unknown_commit",
                    "detail": "version unchanged but full state changed"}

        added, removed, changed, _unchanged = _full_delta(b, a)
        before_map = _line_map(b)
        # Explicit namespaces: SKU membership must be tested against SKU sets, and
        # line-ID membership against ID sets. ``_line_map`` is keyed by line ID.
        before_skus = {info["sku"] for info in before_map.values()}
        before_ids = set(before_map)

        def _newly_owned(lid: str, sku: str) -> bool:
            """A new affected line needs a new ID and a SKU absent everywhere before."""
            return (sku not in self.baseline_skus and sku not in before_skus
                    and lid not in before_ids and lid not in self.baseline_ids)

        # Correct: exactly one new line for the requested sku/quantity.
        if not removed and not changed and len(added) == 1:
            lid, info = next(iter(added.items()))
            if info["sku"] == req_sku and info["quantity"] == req_qty:
                return {**base, "kind": "correct", "line_id": lid}

        # Correct: legitimate merge adding exactly the requested quantity in place.
        if not added and not removed and len(changed) == 1:
            lid, (bb, aa) = next(iter(changed.items()))
            if (bb["sku"] == req_sku and aa["sku"] == req_sku
                    and bb["unit_price_cents"] == aa["unit_price_cents"]
                    and aa["quantity"] == bb["quantity"] + req_qty):
                return {**base, "kind": "correct", "line_id": lid}

        if not removed and not changed and len(added) == 1:
            lid, info = next(iter(added.items()))
            if info["sku"] != req_sku:
                wrong_sku = info["sku"]
                if _newly_owned(lid, wrong_sku):
                    return {**base, "kind": "wrong_variant", "wrong_sku": wrong_sku,
                            "wrong_qty": info["quantity"], "line_id": lid,
                            "unit_price_cents": info["unit_price_cents"],
                            "detail": f"wrote {wrong_sku} instead of {req_sku}"}
                return {**base, "kind": "ownership_uncertain",
                        "detail": f"affected sku {wrong_sku} not newly created"}
            # Same requested sku on a brand-new line, but more than requested.
            if info["quantity"] > req_qty:
                if _newly_owned(lid, req_sku):
                    return {**base, "kind": "excess_quantity", "sku": req_sku,
                            "observed_delta": info["quantity"], "intended": req_qty,
                            "line_id": lid,
                            "unit_price_cents": info["unit_price_cents"],
                            "detail": f"{req_sku} written x{info['quantity']} "
                                      f"vs requested {req_qty}"}
                return {**base, "kind": "ownership_uncertain",
                        "detail": f"requested sku {req_sku} already present before add"}
            return {**base, "kind": "unsupported",
                    "detail": f"new {req_sku} line quantity {info['quantity']} != requested {req_qty}"}

        # A merged line for the requested sku is not automatically ours to reduce.
        if not added and not removed and len(changed) == 1:
            lid, (bb, aa) = next(iter(changed.items()))
            if bb["sku"] == req_sku and aa["sku"] == req_sku:
                return {**base, "kind": "ownership_uncertain",
                        "detail": f"requested sku {req_sku} already present before add"}

        return {**base, "kind": "unsupported",
                "detail": f"unexplained full-state change: added={sorted(added)}, "
                          f"removed={sorted(removed)}, changed={sorted(changed)}"}

    # ------------------------------------------------------------------ #
    # Recovery execution
    # ------------------------------------------------------------------ #

    def recover(self, action_id: str, item: dict, incident: dict):
        if self.stop_reason is not None:
            return
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

    def _next_plan_id(self) -> str:
        self._plan_counter += 1
        return f"plan-{self._plan_counter}"

    def _dispatch_mutate(self, action_id: str, plan_id: str, operation: str,
                         request: dict) -> dict:
        """Count and append an attempted write BEFORE calling the adapter."""
        self.writes_attempted += 1
        step = {
            "event_id": f"w{len(self.audit) + 1}",
            "action_id": action_id, "phase": "correction", "plan_id": plan_id,
            "operation": operation, "dispatched": True, "request": request,
            "evidence": {"dispatched": True, "request": request},
        }
        self.audit.append(step)
        self._emit("write_attempt", action_id=action_id, plan_id=plan_id,
                   operation=operation, request=request,
                   write_number=self.writes_attempted)
        return step

    def _finish_write(self, step: dict, outcome: str, response: Any = None,
                      error: Optional[str] = None) -> dict:
        step["outcome"] = outcome
        if response is not None:
            step["evidence"] = response
            step["response"] = response
        if error is not None:
            step["error"] = error
            step["evidence"] = {**step.get("evidence", {}), "error": error}
        self._emit("write_outcome", action_id=step.get("action_id"),
                   plan_id=step.get("plan_id"), operation=step.get("operation"),
                   outcome=outcome,
                   response=response if isinstance(response, dict) else None,
                   error=error)
        return step

    def _recover_excess(self, action_id, item, incident):
        if not self._budget_ok(action_id, 1):
            return
        self.plans_by_target[action_id] = self.plans_by_target.get(action_id, 0) + 1
        expected = self.verifier.known_version
        target = incident.get("intended")
        if not _valid_quantity(target):
            self._block(f"unsupported corrective quantity {target!r}")
            return
        plan_id = self._next_plan_id()
        step = self._dispatch_mutate(
            action_id, plan_id, "patch",
            {"line_id": incident["line_id"], "sku": incident["sku"],
             "quantity": target, "expected_version": expected})
        try:
            ev = self.session.set_line_quantity(incident["line_id"], incident["sku"],
                                                target, expected)
        except Exception as exc:
            self._finish_write(step, "unknown", error=repr(exc))
            self._block(f"correction PATCH raised: {exc!r}")
            return
        outcome = self._classify_correction(getattr(ev, "payload", None), expected, step)
        if outcome != "committed":
            return
        res = self._settle(f"plan:{action_id}")
        if res["status"] != "settled":
            self._block(f"post-correction settle {res['status']}")
            return
        try:
            after = _snapshot(res["view"])
        except UnsafeState as exc:
            self._block(f"post-correction snapshot unsafe: {exc}")
            return
        before = self._last_snap
        self.last_state = res["view"]
        self._last_snap = after
        if self._plan_achieved(incident, before, after):
            self._resolve(incident, res)
        else:
            self._block("correction did not achieve the planned PATCH effect")

    def _recover_wrong_variant(self, action_id, item, incident):
        if not self._budget_ok(action_id, 2):
            return
        self.plans_by_target[action_id] = self.plans_by_target.get(action_id, 0) + 1
        plan_id = self._next_plan_id()

        # Step 1: delete the owned wrong line.
        expected = self.verifier.known_version
        step = self._dispatch_mutate(
            action_id, plan_id, "delete",
            {"line_id": incident["line_id"], "sku": incident["wrong_sku"],
             "expected_version": expected})
        try:
            ev = self.session.delete_line(incident["line_id"], incident["wrong_sku"], expected)
        except Exception as exc:
            self._finish_write(step, "unknown", error=repr(exc))
            self._block(f"correction DELETE raised: {exc!r}")
            return
        outcome = self._classify_correction(getattr(ev, "payload", None), expected, step)
        if outcome != "committed":
            return
        res = self._settle(f"plan-delete:{action_id}")
        if res["status"] != "settled":
            self._block(f"post-delete settle {res['status']}")
            return
        try:
            after_delete = _snapshot(res["view"])
        except UnsafeState as exc:
            self._block(f"post-delete snapshot unsafe: {exc}")
            return

        # Prove the DELETE effect before any replacement POST: exactly the owned
        # wrong line is gone and every other line keeps ID, SKU, quantity, price.
        before_map = _line_map(self._last_snap)
        expected_map = {lid: info for lid, info in before_map.items()
                        if lid != incident["line_id"]}
        if _line_map(after_delete) != expected_map:
            self._block("DELETE did not remove exactly the owned wrong line "
                        "(no replacement POST sent)")
            return
        self.last_state = res["view"]
        self._last_snap = after_delete

        # Step 2: add the intended sku through the browser action.
        if self.stop_reason is not None:
            return
        add_res = self._corrective_add(action_id, item, plan_id=plan_id)
        if add_res is None or add_res.get("outcome") != "committed":
            return  # already counted/recorded/blocked
        res2 = self._settle(f"plan-add:{action_id}")
        if res2["status"] != "settled":
            self._block(f"post-add settle {res2['status']}")
            return
        try:
            after_add = _snapshot(res2["view"])
        except UnsafeState as exc:
            self._block(f"post-add snapshot unsafe: {exc}")
            return
        self.last_state = res2["view"]
        self._last_snap = after_add
        if self._plan_achieved(incident, self._before_snap, after_add):
            self._resolve(incident, res2)
        else:
            self._block("correction did not achieve intended state (fault may persist)")

    def _recover_silent_drop(self, action_id, item, incident):
        while self.plans_by_target.get(action_id, 0) < MAX_PLANS_PER_TARGET:
            if not self._budget_ok(action_id, 1):
                return
            self.plans_by_target[action_id] = self.plans_by_target.get(action_id, 0) + 1
            add_res = self._corrective_add(action_id, item, allow_no_commit=True)
            if add_res is None:
                return
            if add_res.get("outcome") not in ("committed", "no_commit"):
                return
            res = self._settle(f"retry:{action_id}")
            if res["status"] != "settled":
                self._block(f"post-retry settle {res['status']}")
                return
            try:
                after = _snapshot(res["view"])
            except UnsafeState as exc:
                self._block(f"post-retry snapshot unsafe: {exc}")
                return
            self.last_state = res["view"]
            self._last_snap = after
            if self._plan_achieved(incident, self._before_snap, after):
                self._resolve(incident, res)
                return
            d = self._diagnose(add_res.get("payload") or {}, self._before_snap, after)
            if d["kind"] == "silent_drop":
                continue
            if d["kind"] == "correct":
                # A retry that matches the original action resolves the incident.
                self._resolve(incident, res)
                return
            self._block(f"retry produced {d['kind']}: {d.get('detail', '')}")
            return
        self._block(f"silent-drop plan budget exhausted for {action_id}")

    def _corrective_add(self, action_id, item, plan_id: Optional[str] = None,
                        allow_no_commit: bool = False):
        """Preflight, then one counted corrective ADD.

        Returns ``None`` when no request was sent (preflight stop) and a dict
        ``{"outcome", "payload"}`` once a request was dispatched, so the caller
        can never lose an attempted write.
        """
        product_id = item.get("product_id")
        sku, qty = item.get("sku"), item.get("quantity")
        if not isinstance(sku, str) or not sku or not _valid_quantity(qty):
            self._block(f"unsupported corrective add target {sku!r} x {qty!r} "
                        "(no request sent)")
            return None
        expected = self.verifier.known_version
        try:
            open_ev = self.session.open_product(product_id)
        except Exception as exc:
            self._block(f"corrective add preflight raised: {exc!r} (no request sent)")
            return None
        open_payload = getattr(open_ev, "payload", None) or {}
        embedded = open_payload.get("cart_version")
        if not _is_int(embedded) or embedded != expected:
            self._block(f"page version {embedded!r} != proven {expected} (no request sent)")
            return None
        try:
            self.session.select_variant(sku)
        except Exception as exc:
            self._block(f"corrective add select raised: {exc!r} (no request sent)")
            return None

        step = self._dispatch_mutate(
            action_id, plan_id or self._next_plan_id(), "add",
            {"sku": sku, "quantity": qty, "expected_version": expected})
        try:
            add_ev = self.session.add_to_cart(qty, sku=sku)
        except Exception as exc:
            self._finish_write(step, "unknown", error=repr(exc))
            self._block(f"corrective add raised after dispatch: {exc!r}")
            return {"outcome": "unknown", "payload": None}
        payload = getattr(add_ev, "payload", None)
        if not isinstance(payload, dict):
            self._finish_write(step, "unknown", error="missing action payload")
            self._block("corrective add dispatched but no request/response captured")
            return {"outcome": "unknown", "payload": None}
        # Retain the raw captured request/response (never invent absent fields).
        self._finish_write(step, "dispatched", response=payload)
        if not _is_int(payload.get("sent_version")) or payload["sent_version"] != expected:
            self._block(f"sent version {payload.get('sent_version')!r} != proven {expected}")
            return {"outcome": "uncertain", "payload": payload}
        if (not isinstance(payload.get("sent_sku"), str)
                or payload.get("sent_sku") != sku
                or not _is_int(payload.get("sent_quantity"))
                or payload.get("sent_quantity") != qty):
            self._block(f"sent sku/qty mismatch: {payload.get('sent_sku')!r}/"
                        f"{payload.get('sent_quantity')!r} vs {sku}/{qty}")
            return {"outcome": "uncertain", "payload": payload}
        outcome = self._classify_correction(payload, expected, step,
                                            allow_no_commit=allow_no_commit)
        return {"outcome": outcome, "payload": payload}

    def _classify_correction(self, payload: Any, expected: int, step: dict,
                             allow_no_commit: bool = False) -> str:
        if not isinstance(payload, dict):
            self._finish_write(step, "unknown", error="missing correction payload")
            self._block("correction produced no captured payload")
            return "unknown"
        status = payload.get("response_status")
        rv = payload.get("response_version")
        if _is_int(status) and status == 200 and _valid_version(rv) and rv == expected + 1:
            self.verifier.known_version = expected + 1
            self._finish_write(step, "committed", response=payload)
            return "committed"
        if _is_int(status) and status == 200 and _valid_version(rv) and rv == expected:
            # Documented 200 with an unchanged version: a no-commit observation,
            # never success. Only the silent-drop retry may continue from here.
            self._finish_write(step, "no_commit", response=payload)
            if allow_no_commit:
                self.verifier.known_version = expected
                return "no_commit"
            self._block("correction reported HTTP 200 but the version did not advance")
            return "unknown"
        if _is_int(status) and status == 409:
            self._finish_write(step, "conflict", response=payload)
            self._block(f"correction version conflict "
                        f"(current {payload.get('response_current_version')})")
            return "conflict"
        if _is_int(status) and status in (400, 404):
            self._finish_write(step, "rejected", response=payload)
            self._block(f"correction rejected (HTTP {status})")
            return "rejected"
        self._finish_write(step, "unknown", response=payload)
        self._block(f"correction unknown outcome (status {status!r}, version {rv!r})")
        return "unknown"

    def _plan_achieved(self, incident: dict, before: dict, after: dict) -> bool:
        """Action-relative full-state proof that a plan reached its target.

        The target comes from the frozen original-action before-state, never from
        a fresh absolute quantity. Every unaffected line keeps its exact ID, SKU,
        quantity and unit price.
        """
        if before is None:
            return False
        added, removed, changed, _unchanged = _full_delta(before, after)
        req_sku, req_qty = incident.get("req_sku"), incident.get("req_qty")

        if incident["kind"] == "excess_quantity":
            if added or removed or len(changed) != 1:
                return False
            lid, (bb, aa) = next(iter(changed.items()))
            return (lid == incident["line_id"] and bb["sku"] == req_sku
                    and aa["sku"] == req_sku
                    and bb["unit_price_cents"] == aa["unit_price_cents"]
                    and aa["quantity"] == incident.get("intended"))

        if incident["kind"] == "wrong_variant":
            if incident["line_id"] in _line_map(after):
                return False

        # ADD: either exactly one new line with the requested SKU/quantity, or a
        # legitimate merge that adds exactly the requested quantity in place.
        if not removed and not changed and len(added) == 1:
            _lid, info = next(iter(added.items()))
            return info["sku"] == req_sku and info["quantity"] == req_qty
        if not added and not removed and len(changed) == 1:
            _lid, (bb, aa) = next(iter(changed.items()))
            return (bb["sku"] == req_sku and aa["sku"] == req_sku
                    and bb["unit_price_cents"] == aa["unit_price_cents"]
                    and aa["quantity"] == bb["quantity"] + req_qty)
        return False

    # ------------------------------------------------------------------ #
    # Completion
    # ------------------------------------------------------------------ #

    def complete(self):
        res = self._settle("final")
        final_status = res["status"]
        final_view = res["view"]
        final_snap = None
        if final_status == "settled":
            try:
                final_snap = _snapshot(final_view)
            except UnsafeState:
                final_status = "malformed"
        constraints = []
        confirmed = 0
        delta = None
        delta_ok = None
        protected_ok = True
        protected_detail = None

        if final_status == "settled" and not self.verifier.anomaly_epoch and final_snap is not None:
            for c in self.ledger.constraints:
                state, reason = evaluate(c, final_view)
                constraints.append({"label": c.label(), "state": state, "reason": reason,
                                    "provenance": res["provenance"],
                                    "citations": self._citations(res)})
                if state == CONFIRMED:
                    confirmed += 1
            delta = _delta(quantities(self.original_baseline["lines"]),
                           quantities(_raw_lines(final_snap)))
            delta_ok = (delta == self.declared)
            protected_ok, protected_detail = self._check_protected(final_snap)
            self.last_state = final_view
            self._last_snap = final_snap
        else:
            for c in self.ledger.constraints:
                constraints.append({"label": c.label(), "state": UNKNOWN,
                                    "reason": self._completion_failure_reason(final_status, c, res),
                                    "provenance": (), "citations": []})
            if final_snap is not None:
                delta = _delta(quantities(self.original_baseline["lines"]),
                               quantities(_raw_lines(final_snap)))
                delta_ok = (delta == self.declared)
            protected_ok = False
            protected_detail = f"final view not settled ({final_status})"

        blocking = bool(self.blocking) or self.stop_reason is not None
        complete = (final_status == "settled" and not self.verifier.anomaly_epoch
                    and confirmed == len(self.ledger.constraints)
                    and delta_ok is True and protected_ok and not blocking)
        verdict = "COMPLETE" if complete else "PARTIAL"

        # Emit completion BEFORE the report copies the append-only event list, so
        # the returned and rendered logs both contain the final event.
        self._emit("final", verdict=verdict, final_status=final_status,
                   citations=self._citations(res), confirmed=confirmed,
                   total=len(self.ledger.constraints))

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
            "events": list(self.events),
            "observations": [self.observations[s] for s in sorted(self.observations)],
            "samples": list(self.verifier.samples),
            "settle_ms": self.verifier.settle_ms,
            "verdict": verdict,
            "confirmed_count": confirmed,
            "total": len(self.ledger.constraints),
        }
        return report, verdict

    def _check_protected(self, final_snap: dict):
        final_by_id = _line_map(final_snap)
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

    def _completion_failure_reason(self, status: str, c: Constraint,
                                   res: Optional[dict] = None) -> str:
        if self.verifier.anomaly_epoch:
            return "anomaly epoch: state changed by an unexplained external write"
        return self._failure_reason(status, res, completion=True)

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #

    def _settle_stop_reason(self, status: str, res: Optional[dict] = None) -> str:
        return self._failure_reason(status, res)

    def _failure_reason(self, status: str, res: Optional[dict] = None,
                        completion: bool = False) -> str:
        """Explain a non-settled outcome from the evidence actually recorded.

        A genuine DOM/API disagreement always wins. Otherwise a fresh pair whose
        DOM and API agree but whose shared total fails line arithmetic is
        reported as the arithmetic inconsistency it is, with the observation
        references that prove it; no other path changes wording.
        """
        if status == "conflict":
            return "conflicting DOM/API sources"
        if status == "malformed":
            for diag in (res or {}).get("incoherent") or []:
                if diag.get("kind") != "arithmetic":
                    continue
                return (f"displayed total {diag['displayed_total']} != recomputed "
                        f"total {diag['recomputed_total']} (DOM/API agree on full "
                        f"state; observations #{diag['seq_dom']}/#{diag['seq_api']})")
            return "malformed or insufficient evidence"
        if completion:
            if status == "stale":
                return "no fresh settled view within deadline (version below known_version)"
            return "no fresh settled view within deadline"
        if status == "stale":
            return "no fresh settled view within deadline"
        if status == "anomaly":
            return "unexplained version advance during settle"
        if status == "timeout":
            return "no fresh settled view within deadline (settlement timed out)"
        return f"settlement failed ({status})"

    def _emit(self, event_kind: str, **fields) -> dict:
        ev = {"event_id": f"e{len(self.events) + 1}", "kind": event_kind, **fields}
        self.events.append(ev)
        return ev

    def _record_incident(self, incident: dict):
        """Append an incident record; the stored copy is never mutated later."""
        incident["incident_id"] = f"i{len(self.incidents) + 1}"
        self.incidents.append(incident)
        return incident["incident_id"]

    def _resolve(self, incident: dict, settle_res: dict):
        """Append a resolution event; the original incident record is untouched."""
        citations = self._citations(settle_res)
        record = dict(incident)
        record["resolution"] = "resolved"
        record["citations"] = citations
        record["observed_lines"] = _line_map(self._last_snap) if self._last_snap else {}
        self.resolved.append(record)
        self._emit("resolution", incident_id=incident.get("incident_id"),
                   action_id=incident.get("action_id"), incident_kind=incident["kind"],
                   citations=citations, observed_lines=record["observed_lines"])

    def _block(self, detail: str):
        self.blocking.append({"kind": "blocked", "detail": detail})
        self.stop_reason = self.stop_reason or detail
        self._emit("blocked", detail=detail)

    def _stop(self, reason: str):
        self.stop_reason = self.stop_reason or reason
        self.blocking.append({"kind": "stop", "detail": reason})
        self._emit("stop", reason=reason)
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


def _fmt_citation(c: list) -> str:
    return "/".join(f"#{s}" for s in c if s is not None)


def render_recovery_report(report: dict) -> str:
    """Render the recorded evidence. Takes no observations and mutates nothing."""
    out = []
    out.append("")
    out.append(f"=== CartProof recovery report (settle_ms={report.get('settle_ms')}) ===")
    out.append("")
    out.append(f"Ledger (version {report['ledger']['version']}):")
    out.append(json.dumps(report["ledger"], indent=2))
    out.append("")
    out.append("Constraints:")
    for c in report["constraints"]:
        prov = ", ".join(f"{p['source']}@v{p['version']}#s{p['seq']}"
                         for p in c.get("provenance") or ())
        cites = ", ".join(_fmt_citation(x) for x in c.get("citations") or ())
        suffix = c.get("reason") or ""
        if prov:
            suffix = f"{suffix}; {prov}" if suffix else prov
        if cites:
            suffix = f"{suffix}; cites {cites}" if suffix else f"cites {cites}"
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
            cites = ", ".join(_fmt_citation(x) for x in i.get("citations") or ())
            suffix = f" cites {cites}" if cites else ""
            out.append(f"  - [{i['kind']}] {i.get('incident_id')} action {i.get('action_id')}: "
                       f"{i.get('detail')} -> resolved{suffix}")
    else:
        out.append("Resolved incidents: (none)")
    if report["incidents"]:
        out.append("Recorded incidents (original records):")
        for i in report["incidents"]:
            out.append(f"  - {i.get('incident_id')} [{i['kind']}] action {i.get('action_id')}: "
                       f"{i.get('detail')}")
    else:
        out.append("Recorded incidents: (none)")
    if report["blocking"]:
        out.append("Blocking incidents:")
        for b in report["blocking"]:
            out.append(f"  - {b['kind']}: {b['detail']}")
    else:
        out.append("Blocking incidents: (none)")
    if report.get("stop_reason"):
        out.append(f"Stop reason: {report['stop_reason']}")
    out.append("")
    out.append("Recovery audit (attempted writes, dispatched-before-call):")
    for step in report.get("audit") or []:
        out.append("  - " + json.dumps(step, sort_keys=True, default=str))
    out.append("")
    out.append("Recorded observations (complete line fields, no aggregation):")
    for s in report.get("observations") or []:
        marks = []
        if s.get("late"):
            marks.append("late")
        if s.get("timed_out"):
            marks.append("read-timeout")
        suffix = f" [{', '.join(marks)}]" if marks else ""
        lines = s.get("lines_full")
        shown = "unavailable" if lines is None else json.dumps(lines, sort_keys=True)
        out.append(f"  #{s['seq']:<3} {s['phase']:<14} {s['source']:<11} v{s['version']} "
                   f"total={s['total_cents']} lines={shown} {s['timestamp']}{suffix}")
    out.append("")
    out.append("Event log (append-only):")
    for ev in report.get("events") or []:
        out.append("  - " + json.dumps(ev, sort_keys=True, default=str))
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

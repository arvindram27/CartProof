"""Full-cart guard for MB/MC (outside core/). Declaration honesty only.

The existing ledger constraint types cannot express full-cart exactness
(count_distinct counts an intersection; unrelated extras are invisible), and
core/ semantics stay frozen. This guard compares the run's final settled
runtime observation against the cart the run itself set out to produce
(observed baseline + validated executable additions). It only ever downgrades
a COMPLETE declaration to PARTIAL. It never upgrades, never touches recovery,
and never reads the evaluator or the SQLite file.

The final observation comes from the completion settle's own recorded raw
pair (M1): the same settled observations that supported the verdict, with no
extra reads.
"""
from __future__ import annotations

from typing import Any, Optional


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def aggregate_quantities(lines: Any) -> Optional[dict]:
    """Aggregate SKU quantities, or None on any invalid line shape."""
    if not isinstance(lines, list):
        return None
    out: dict[str, int] = {}
    for ln in lines:
        if not isinstance(ln, dict):
            return None
        sku, qty = ln.get("sku"), ln.get("quantity")
        if not isinstance(sku, str) or not sku:
            return None
        if not _is_int(qty) or qty <= 0:
            return None
        out[sku] = out.get(sku, 0) + qty
    return out


def _validate_view(view: Any, settled_version: int) -> Optional[dict]:
    """Validate one raw DOM/API view; return its lines or None (M1c)."""
    if not isinstance(view, dict):
        return None
    v = view.get("version")
    if not _is_int(v) or v < 0 or v != settled_version:
        return None
    total = view.get("total_cents")
    if not _is_int(total) or total < 0:
        return None
    lines = view.get("lines")
    if not isinstance(lines, list):
        return None
    for ln in lines:
        if not isinstance(ln, dict):
            return None
        if not isinstance(ln.get("sku"), str) or not ln.get("sku"):
            return None
        if not _is_int(ln.get("quantity")) or ln["quantity"] <= 0:
            return None
        if not _is_int(ln.get("unit_price_cents")) or ln["unit_price_cents"] < 0:
            return None
    return lines


def completion_settled_view(report: dict, raw_pairs: dict) -> Optional[dict]:
    """The completion settle's settled view, from its recorded raw pair (M1).

    Reads only pairs the recorder already captured; performs no new sampling.
    Returns None (guard NOT ELIGIBLE, fail closed) on any missing, ambiguous,
    or invalid association.
    """
    if not isinstance(report, dict) or not raw_pairs:
        return None
    final = report.get("final") or {}
    settled_version = final.get("version")
    if not _is_int(settled_version) or settled_version < 0:
        return None
    try:
        pair = raw_pairs[max(raw_pairs)]
    except (ValueError, TypeError):
        return None
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    dom, api = pair
    dom_lines = _validate_view(dom, settled_version)
    api_lines = _validate_view(api, settled_version)
    if dom_lines is None or api_lines is None:
        return None
    if aggregate_quantities(dom_lines) != aggregate_quantities(api_lines):
        return None
    return {"version": settled_version, "lines": api_lines,
            "total_cents": api["total_cents"]}


def expected_full_cart(baseline_quantities: dict, additions: dict) -> dict:
    out = dict(baseline_quantities)
    for sku, qty in (additions or {}).items():
        out[sku] = out.get(sku, 0) + int(qty)
    return out


def compare_full_cart(expected: dict, actual: dict) -> list[dict]:
    mismatches = []
    for sku in sorted(set(expected) | set(actual)):
        e, a = expected.get(sku, 0), actual.get(sku, 0)
        if e != a:
            mismatches.append({"sku": sku, "expected_quantity": e,
                               "actual_quantity": a})
    return mismatches


def apply_full_cart_guard(report: dict, raw_pairs: dict,
                          baseline_quantities: Optional[dict],
                          additions: dict, budget_max_cents: Optional[int],
                          raw_verdict: str) -> dict:
    """Guard a raw COMPLETE/PARTIAL verdict. Returns the guard record."""
    expected = None
    if baseline_quantities is not None:
        expected = expected_full_cart(baseline_quantities, additions)

    def record(status, guarded, mismatches=None, settled=None, reason=None):
        return {
            "applied": True, "status": status,
            "expected_full_cart": expected,
            "observed_final_cart": (
                aggregate_quantities(settled["lines"]) if settled else None),
            "baseline_cart": baseline_quantities,
            "mismatches": mismatches or [],
            "budget_max_cents": budget_max_cents,
            "final_total_cents": settled["total_cents"] if settled else None,
            "recomputed_total_cents": (
                sum(ln["quantity"] * ln["unit_price_cents"]
                    for ln in settled["lines"]) if settled else None),
            "reason": reason, "guarded_verdict": guarded,
        }

    if baseline_quantities is None:
        return record("baseline_unavailable",
                      "PARTIAL" if raw_verdict == "COMPLETE" else raw_verdict,
                      reason="observed baseline unavailable or inconsistent")

    settled = completion_settled_view(report, raw_pairs)
    if settled is None:
        # Unknown / inconsistent / unsettled final observation: never COMPLETE.
        return record("not_eligible",
                      "PARTIAL" if raw_verdict == "COMPLETE" else raw_verdict,
                      reason="completion settled observation unavailable, "
                             "inconsistent, or unsettled")

    actual = aggregate_quantities(settled["lines"])
    mismatches = compare_full_cart(expected, actual)
    recomputed = sum(ln["quantity"] * ln["unit_price_cents"]
                     for ln in settled["lines"])
    budget_bad = False
    if budget_max_cents is not None:
        if settled["total_cents"] != recomputed or recomputed > budget_max_cents:
            budget_bad = True

    if mismatches or budget_bad:
        reason = ("full-cart mismatch" if mismatches else "budget exceeded")
        if budget_bad:
            reason += "; budget check failed"
        return record("downgrade" if raw_verdict == "COMPLETE" else "mismatch",
                      "PARTIAL" if raw_verdict == "COMPLETE" else raw_verdict,
                      mismatches=mismatches, settled=settled, reason=reason)

    return record("pass", raw_verdict, settled=settled)

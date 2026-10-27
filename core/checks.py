"""Deterministic constraint checks against an observed cart state.

Pure functions only: ``(constraint, observed_state) -> (verdict, reason)`` for
final-absolute checks. Exact-match rules only: sku equality (never fuzzy), final
quantity equality, distinct-sku counting (EXACTLY N), and recomputed totals.
A ``confirmed``/``contradicted``/``unknown`` verdict here carries no freshness
or settlement context - the verifier supplies that separately.
"""
from __future__ import annotations

from typing import Any

from .ledger import (
    BUDGET_MAX,
    CONFIRMED,
    CONTRADICTED,
    COUNT_DISTINCT,
    ITEM_QUANTITY,
    UNKNOWN,
    Constraint,
)


def quantities(lines: list) -> dict:
    """Aggregate per-sku quantities from a line list."""
    out: dict = {}
    for l in lines:
        sku = l.get("sku")
        qty = l.get("quantity")
        if sku is None or not isinstance(qty, int) or isinstance(qty, bool):
            continue
        out[sku] = out.get(sku, 0) + qty
    return out


def recompute_total(lines: list) -> int:
    """Arithmetic total from observed lines (quantity * unit_price_cents)."""
    total = 0
    for l in lines:
        q = l.get("quantity")
        p = l.get("unit_price_cents")
        if isinstance(q, int) and not isinstance(q, bool) and isinstance(p, int) and not isinstance(p, bool):
            total += q * p
    return total


def check_item_quantity(c: Constraint, view: dict) -> tuple:
    qty = quantities(view.get("lines") or []).get(c.sku, 0)
    if qty == c.quantity:
        return CONFIRMED, f"observed {qty} == required {c.quantity}"
    return CONTRADICTED, f"observed {qty} != required {c.quantity}"


def check_count_distinct(c: Constraint, view: dict) -> tuple:
    q = quantities(view.get("lines") or [])
    present = {s for s, n in q.items() if n > 0}
    distinct = len(present & set(c.skus))
    if distinct == c.count:
        return CONFIRMED, f"observed exactly {distinct} distinct skus"
    return CONTRADICTED, f"observed {distinct} distinct skus != required exactly {c.count}"


def check_budget(c: Constraint, view: dict) -> tuple:
    displayed = view.get("total_cents")
    recomputed = recompute_total(view.get("lines") or [])
    if displayed is None or displayed != recomputed:
        return UNKNOWN, f"displayed total {displayed} != recomputed {recomputed}"
    if recomputed <= c.max_cents:
        return CONFIRMED, f"total {recomputed} <= {c.max_cents}"
    return CONTRADICTED, f"total {recomputed} exceeds {c.max_cents}"


def evaluate(c: Constraint, view: dict) -> tuple:
    """Dispatch a constraint against an observed state, returning (state, reason)."""
    if c.type == ITEM_QUANTITY:
        return check_item_quantity(c, view)
    if c.type == COUNT_DISTINCT:
        return check_count_distinct(c, view)
    if c.type == BUDGET_MAX:
        return check_budget(c, view)
    return UNKNOWN, f"unknown constraint type {c.type!r}"

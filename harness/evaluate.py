"""Independent evaluator: persisted-state ground truth vs authored expectation.

The evaluator compares the store's persisted cart against the task's
independently authored expected end-state. It never reads the fault-injection
log, the agent evidence log, the verifier report, or the recovery report; its
truth is persisted state versus authored expectation, nothing else.

Stage semantics (Brief 4 correction C2):

- Items are processed in a fixed order. Prefix ``k`` = seed lines plus the
  intended lines of items 1..k, aggregated by SKU.
- A ``before_add`` snapshot for original item ``i`` (0-based) compares against
  prefix ``i``; an ``after_add`` snapshot for item ``i`` compares against
  prefix ``i+1``.
- Only the final post-run verdict uses the full final expectation.

This module imports only the raw persisted-state reader (``snapshots.read_cart``)
plus standard-library collections; it never touches runtime report structures.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from .snapshots import backup_copy, read_cart


def compare_lines(expected: dict[str, int], actual: dict[str, int]) -> list[dict]:
    """Return every SKU where expected and actual quantities differ."""
    mismatches = []
    for sku in sorted(set(expected) | set(actual)):
        e = expected.get(sku, 0)
        a = actual.get(sku, 0)
        if e != a:
            mismatches.append({"sku": sku, "expected_quantity": e,
                               "actual_quantity": a})
    return mismatches


def derive_prefixes(seed_lines: list[dict], items: list[dict]) -> list[dict]:
    """Derive cumulative intended cart prefixes from seeds + original-order items.

    Never derived from run output. ``prefixes[0]`` is the seeded cart;
    ``prefixes[k]`` is the cart after items 0..k-1.
    """
    agg: Counter[str] = Counter()
    for s in seed_lines or []:
        agg[s["sku"]] += int(s["quantity"])
    prefixes: list[dict] = [dict(agg)]
    for it in items or []:
        agg[it["target_sku"]] += int(it["quantity"])
        prefixes.append(dict(agg))
    return prefixes


def final_expected_prefix(prefixes: list[dict]) -> dict:
    return prefixes[-1] if prefixes else {}


def measure_final_db(final_db: str, expected_lines: dict[str, int],
                     cookie_session_id: Optional[str] = None) -> dict:
    """Measure the final persisted cart against the authored expectation.

    Final evaluation requires unique active-cart resolution: ``empty``,
    ``ambiguous`` and ``error`` resolutions all yield ``unknown``.
    """
    cart = read_cart(final_db, cookie_session_id)
    if cart["resolution"] in ("ambiguous", "error", "empty"):
        return {"measured_outcome": "unknown", "resolution": cart["resolution"],
                "mismatches": [], "lines": cart["lines"]}
    mismatches = compare_lines(expected_lines, cart["lines"])
    outcome = "correct" if not mismatches else "incorrect"
    return {"measured_outcome": outcome, "resolution": cart["resolution"],
            "mismatches": mismatches, "lines": cart["lines"]}


def capture_final_db(db_path: str, final_db: str, cookie_session_id: Optional[str]) -> dict:
    """Backup the final DB and read its persisted cart (raw, no expectation)."""
    backup_copy(db_path, final_db)
    return read_cart(final_db, cookie_session_id)


def stage_mismatches(snapshot_lines: Optional[dict], expected_prefix: dict):
    """Compare a hook-entry snapshot's lines against a stage prefix.

    Returns ``None`` when the snapshot is unknown and cannot be compared.
    """
    if snapshot_lines is None:
        return None
    return compare_lines(expected_prefix, snapshot_lines)

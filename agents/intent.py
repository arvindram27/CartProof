"""Supported-grammar intent handling: parser-anchored, model as cross-check.

Within the supported grammar, interpretation is anchored by this deterministic
parser; the model proposal is a cross-check channel (agreement proceeds,
disagreement clarifies). No artifact may attribute natural-language
understanding to the model for in-grammar requests. Anything outside the
grammar, genuinely ambiguous, or underdetermined produces a semantic
clarification for the whole run: no writes at all.

Constraints (budget, substitution permission) come only from the request text.
The model never invents, widens, or relaxes one. Absent means absent.
"""
from __future__ import annotations

import re
from typing import Any, Optional

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_SKU_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", re.IGNORECASE)


def _find_span(text: str, phrase: str) -> tuple[int, int]:
    i = text.lower().find(phrase.lower())
    return (i, i + len(phrase)) if i >= 0 else (-1, -1)


def _quantity(clause: str, points: list[str]) -> Optional[int]:
    low = clause.lower()
    if "a couple" in low or "some " in low or low.strip().startswith("some"):
        points.append("quantity is vague ('a couple'/'some')")
        return None
    # Strip SKU tokens first so their digits (e.g. 500, 750) are not quantities.
    bare = _SKU_RE.sub("", clause)
    found: list[int] = []
    for m in re.finditer(r"(\d+)\s*(?:x|×)?\b", bare):
        found.append(int(m.group(1)))
    for w, v in _NUMBER_WORDS.items():
        if re.search(rf"\b{w}\b", low):
            found.append(v)
    # Singular article quantity only when an add/buy verb introduces the clause.
    if re.search(r"\b(add|buy|get|order)\s+(a|an)\b", low):
        found.append(1)
    uniq = sorted(set(found))
    if len(uniq) > 1:
        points.append(f"conflicting quantities {uniq}")
        return None
    if not uniq:
        points.append("quantity not specified")
        return None
    q = uniq[0]
    if q <= 0 or q > 99:
        points.append(f"quantity {q} outside 1..99")
        return None
    return q


def _budget(text: str, points: list[str]):
    """Return (budget_max_cents|None). M2 semantics for $ phrases."""
    low = text.lower()
    values: list[int] = []
    for m in re.finditer(r"(under|less than)\s*\$\s*(\d+)", low):
        values.append(int(m.group(2)) * 100 - 1)
    for m in re.finditer(r"(at most|no more than|up to)\s*\$\s*(\d+)", low):
        values.append(int(m.group(2)) * 100)
    for m in re.finditer(r"(under|less than)\s*(\d+)\s*cents", low):
        values.append(int(m.group(2)) - 1)
    for m in re.finditer(r"(at most|no more than|up to)\s*(\d+)\s*cents", low):
        values.append(int(m.group(2)))
    uniq = sorted(set(values))
    if len(uniq) > 1:
        points.append(f"conflicting budget limits {uniq}")
        return None
    return uniq[0] if uniq else None


def _substitution(text: str) -> bool:
    low = text.lower()
    if "no substitution" in low or "do not substitute" in low \
            or "don't substitute" in low or "no substitutions" in low:
        return False
    if "may substitute" in low or "substitutions allowed" in low \
            or "can substitute" in low or "you may substitute" in low:
        return True
    return False


def _split_clauses(text: str) -> list[str]:
    parts = re.split(r"[.;]", text)
    clauses: list[str] = []
    for p in parts:
        for c in re.split(r",|\band\b", p):
            c = c.strip()
            if c:
                clauses.append(c)
    return clauses


def _is_non_item_clause(clause: str) -> bool:
    low = clause.lower()
    if "$" in clause or "cents" in low:
        return True
    if "substitut" in low:
        return True
    if low.startswith(("keep", "stay", "total", "budget")):
        return True
    return False


def extract_intent(request_text: str, catalog: list[dict]) -> dict:
    """Deterministic grammar extraction. Returns items/budget/points."""
    points: list[str] = []
    by_sku = {c["sku"].upper(): c for c in catalog}
    products: dict[str, list[dict]] = {}
    for c in catalog:
        products.setdefault(c["product_name"], []).append(c)

    budget = _budget(request_text, points)
    substitution = _substitution(request_text)

    items: list[dict] = []
    for clause in _split_clauses(request_text):
        low = clause.lower()
        if _is_non_item_clause(clause):
            continue
        if re.search(r"\b(don't|do not|never)\b", low) and "substitut" not in low:
            points.append(f"negated clause: {clause!r}")
            continue
        if " or " in f" {low} " or " except " in f" {low} ":
            points.append(f"alternation/exclusion in clause: {clause!r}")
            continue

        sku = None
        entry = None
        m = _SKU_RE.search(clause)
        if m and m.group(0).upper() in by_sku:
            entry = by_sku[m.group(0).upper()]
            sku = entry["sku"]
        else:
            # Product-name grounding, then unique-variant narrowing.
            matched_products = [name for name in products
                                if name.lower() in low]
            if len(matched_products) != 1:
                points.append(
                    f"no unique catalog product grounded in clause: {clause!r}")
                continue
            variants = products[matched_products[0]]
            candidates = []
            for v in variants:
                attrs = [str(v.get("size") or "").lower(),
                         str(v.get("color") or "").lower(),
                         str(v.get("label") or "").lower()]
                mentioned = [a for a in attrs if a and a in low]
                # A variant is a candidate only if every attribute word the
                # clause mentions about variants is consistent with it.
                size_hit = any(str(x.get("size") or "").lower() in low
                               for x in variants if x.get("size"))
                color_hit = any(str(x.get("color") or "").lower() in low
                                for x in variants if x.get("color"))
                ok = True
                if size_hit and str(v.get("size") or "").lower() not in low:
                    ok = False
                if color_hit and str(v.get("color") or "").lower() not in low:
                    ok = False
                if ok:
                    candidates.append(v)
            if len(candidates) != 1:
                points.append(
                    f"variant not uniquely determined in clause: {clause!r} "
                    f"({len(candidates)} candidates)")
                continue
            entry = candidates[0]
            sku = entry["sku"]

        qty = _quantity(clause, points)
        if qty is None:
            continue
        s, e = _find_span(request_text, clause)
        items.append({
            "sku": sku, "quantity": qty,
            "product_query": entry["product_name"],
            "phrase": clause, "char_start": s, "char_end": e,
        })

    return {"items": items, "budget_max_cents": budget,
            "substitution_allowed": substitution, "unresolved_points": points}


def _agg(items: list[dict]) -> dict:
    out: dict[str, int] = {}
    for it in items:
        out[it["sku"]] = out.get(it["sku"], 0) + int(it["quantity"])
    return out


def validate_proposal(request_text: str, catalog: list[dict],
                      proposal_payload: Any) -> dict:
    """Decide executable intent from a model proposal + grammar extraction.

    Returns {outcome_kind, executable_intent, unresolved_points, rejections,
    proposed_intent}. Confidence is never consulted.
    """
    extraction = extract_intent(request_text, catalog)

    if not isinstance(proposal_payload, dict) \
            or not isinstance(proposal_payload.get("items"), list):
        return {"outcome_kind": "operational_failure",
                "executable_intent": None,
                "unresolved_points": [],
                "rejections": ["model output malformed: proposal is not an "
                                "object with an items list"],
                "proposed_intent": proposal_payload}

    proposed_items = []
    for it in proposal_payload["items"]:
        if not isinstance(it, dict) or not isinstance(it.get("sku"), str) \
                or not isinstance(it.get("quantity"), int) \
                or isinstance(it.get("quantity"), bool):
            return {"outcome_kind": "operational_failure",
                    "executable_intent": None,
                    "unresolved_points": [],
                    "rejections": ["model output malformed: bad item shape"],
                    "proposed_intent": proposal_payload}
        proposed_items.append({"sku": it["sku"], "quantity": it["quantity"]})

    rejections: list[str] = []
    # Invented/widened constraints are rejected, never adopted.
    p_budget = proposal_payload.get("budget_max_cents")
    if p_budget != extraction["budget_max_cents"]:
        rejections.append(
            f"proposal budget {p_budget!r} != request-grounded budget "
            f"{extraction['budget_max_cents']!r}")
    p_sub = proposal_payload.get("substitution_allowed", False)
    if bool(p_sub) != extraction["substitution_allowed"]:
        rejections.append(
            f"proposal substitution {p_sub!r} != request-grounded "
            f"{extraction['substitution_allowed']!r}")
    if _agg(proposed_items) != _agg(extraction["items"]):
        rejections.append(
            f"proposal items {_agg(proposed_items)} != grounded items "
            f"{_agg(extraction['items'])}")

    if extraction["unresolved_points"]:
        return {"outcome_kind": "semantic_clarify",
                "executable_intent": None,
                "unresolved_points": extraction["unresolved_points"],
                "rejections": rejections,
                "proposed_intent": proposal_payload}

    if rejections:
        # Fully grounded extraction, but the model cross-check disagrees
        # (invented/changed budget, substitution, or item set): this is a
        # semantic clarification, with the rejections as unresolved points.
        # Malformed proposals (handled above) remain operational failures.
        return {"outcome_kind": "semantic_clarify",
                "executable_intent": None,
                "unresolved_points": list(rejections),
                "rejections": rejections,
                "proposed_intent": proposal_payload}

    executable = {
        "items": extraction["items"],
        "budget_max_cents": extraction["budget_max_cents"],
        "substitution_allowed": extraction["substitution_allowed"],
        "provenance": [
            {"field": f"items[{i}]", "sku": it["sku"], "quantity": it["quantity"],
             "phrase": it["phrase"], "char_start": it["char_start"],
             "char_end": it["char_end"]}
            for i, it in enumerate(extraction["items"])
        ],
    }
    return {"outcome_kind": "proceed", "executable_intent": executable,
            "unresolved_points": [], "rejections": [],
            "proposed_intent": proposal_payload}

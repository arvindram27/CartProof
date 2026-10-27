"""Intent ledger: typed constraints, states, and an immutable versioned ledger.

The parser here is deterministic: it builds the ledger from the task spec's
explicit ``constraints`` list. This stands in for the model-based parse of a
free-text request, which is a later phase gated on provider access the user has
not approved; the interface (task spec in -> versioned ledger out) is what that
model parser will implement later.

Semantics note (per the approved brief): ``count_distinct`` means EXACTLY N
distinct skus from the named set at completion, not "at least N".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

#: Constraint verdict states.
UNVERIFIED = "unverified"
CONFIRMED = "confirmed"
CONTRADICTED = "contradicted"
UNKNOWN = "unknown"
STATES = {UNVERIFIED, CONFIRMED, CONTRADICTED, UNKNOWN}

#: Constraint types.
ITEM_QUANTITY = "item_quantity"
COUNT_DISTINCT = "count_distinct"
BUDGET_MAX = "budget_max"


class ConstraintError(ValueError):
    """Raised when a constraint spec is malformed or invalid."""


@dataclass(frozen=True)
class Constraint:
    """One typed constraint. Terms are immutable (frozen dataclass)."""

    type: str
    sku: Optional[str] = None
    quantity: Optional[int] = None
    skus: Tuple[str, ...] = ()
    count: Optional[int] = None
    max_cents: Optional[int] = None
    count_mode: str = "exactly"

    @classmethod
    def from_spec(cls, spec: Any) -> "Constraint":
        if not isinstance(spec, dict):
            raise ConstraintError("constraint must be a JSON object")
        ctype = spec.get("type")
        if ctype == ITEM_QUANTITY:
            sku = spec.get("sku")
            quantity = spec.get("quantity")
            if not isinstance(sku, str) or not sku:
                raise ConstraintError("item_quantity requires a non-empty 'sku'")
            if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
                raise ConstraintError("item_quantity requires a positive integer 'quantity'")
            return cls(type=ctype, sku=sku, quantity=quantity)
        if ctype == COUNT_DISTINCT:
            skus = spec.get("skus")
            count = spec.get("count")
            if not isinstance(skus, list) or len(skus) == 0:
                raise ConstraintError("count_distinct requires a non-empty 'skus' list")
            if not all(isinstance(s, str) and s for s in skus):
                raise ConstraintError("count_distinct 'skus' must be non-empty strings")
            unique: Tuple[str, ...] = tuple(dict.fromkeys(skus))  # dedupe, keep order
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ConstraintError("count_distinct requires a positive integer 'count'")
            if count > len(unique):
                raise ConstraintError("count_distinct 'count' exceeds the size of the sku set")
            return cls(type=ctype, skus=unique, count=count, count_mode="exactly")
        if ctype == BUDGET_MAX:
            max_cents = spec.get("max_cents")
            if not isinstance(max_cents, int) or isinstance(max_cents, bool) or max_cents < 0:
                raise ConstraintError("budget_max requires a non-negative integer 'max_cents'")
            return cls(type=ctype, max_cents=max_cents)
        raise ConstraintError(f"unknown constraint type: {ctype!r}")

    def label(self) -> str:
        if self.type == ITEM_QUANTITY:
            return f"{self.type} {self.sku} = {self.quantity}"
        if self.type == COUNT_DISTINCT:
            return f"{self.type} exactly {self.count} of {list(self.skus)}"
        if self.type == BUDGET_MAX:
            return f"{self.type} <= {self.max_cents} cents"
        return self.type

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": self.type}
        if self.type == ITEM_QUANTITY:
            d["sku"] = self.sku
            d["quantity"] = self.quantity
        elif self.type == COUNT_DISTINCT:
            d["skus"] = list(self.skus)
            d["count"] = self.count
            d["count_mode"] = self.count_mode
        elif self.type == BUDGET_MAX:
            d["max_cents"] = self.max_cents
        return d


@dataclass(frozen=True)
class Ledger:
    """Immutable, versioned set of constraints.

    ``build`` produces version 1; ``rebuild`` is the only way to change terms
    and always produces a new (incremented) version. Terms are never edited in
    place.
    """

    version: int
    constraints: Tuple[Constraint, ...]

    @classmethod
    def build(cls, constraint_specs: Any) -> "Ledger":
        if not isinstance(constraint_specs, list):
            raise ConstraintError("'constraints' must be a list")
        if len(constraint_specs) == 0:
            raise ConstraintError("'constraints' must not be empty")
        constraints = tuple(Constraint.from_spec(s) for s in constraint_specs)
        return cls(version=1, constraints=constraints)

    def rebuild(self, constraint_specs: Any) -> "Ledger":
        constraints = tuple(Constraint.from_spec(s) for s in constraint_specs)
        return Ledger(version=self.version + 1, constraints=constraints)

    def to_dict(self) -> dict:
        return {"version": self.version, "constraints": [c.to_dict() for c in self.constraints]}


@dataclass(frozen=True)
class Verdict:
    """A constraint's verification state with evidence provenance."""

    state: str
    provenance: Tuple[dict, ...] = ()
    reason: str = ""
    historical: bool = False

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "provenance": list(self.provenance),
            "reason": self.reason,
            "historical": self.historical,
        }

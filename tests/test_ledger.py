"""Intent ledger tests: validation, immutability, re-versioning, states."""
from __future__ import annotations

import pytest

from core.ledger import (
    CONFIRMED,
    CONTRADICTED,
    UNKNOWN,
    UNVERIFIED,
    Constraint,
    ConstraintError,
    Ledger,
    STATES,
)


def _specs(*constraints):
    return list(constraints)


# --------------------------------------------------------------------------- #
# Validation (fail closed on malformed input)
# --------------------------------------------------------------------------- #

def test_item_quantity_validation():
    assert Constraint.from_spec({"type": "item_quantity", "sku": "A", "quantity": 2}).quantity == 2
    with pytest.raises(ConstraintError):
        Constraint.from_spec({"type": "item_quantity", "quantity": 2})          # missing sku
    with pytest.raises(ConstraintError):
        Constraint.from_spec({"type": "item_quantity", "sku": "A", "quantity": 0})   # zero
    with pytest.raises(ConstraintError):
        Constraint.from_spec({"type": "item_quantity", "sku": "A", "quantity": -3})  # negative
    with pytest.raises(ConstraintError):
        Constraint.from_spec({"type": "item_quantity", "sku": "", "quantity": 2})


def test_count_distinct_validation():
    ok = Constraint.from_spec({"type": "count_distinct", "skus": ["A", "B"], "count": 2})
    assert ok.count_mode == "exactly"
    with pytest.raises(ConstraintError):
        Constraint.from_spec({"type": "count_distinct", "skus": [], "count": 1})   # empty set
    with pytest.raises(ConstraintError):
        Constraint.from_spec({"type": "count_distinct", "skus": ["A"], "count": 0})  # zero count
    with pytest.raises(ConstraintError):
        Constraint.from_spec({"type": "count_distinct", "skus": ["A"], "count": 2})  # count > set size


def test_budget_max_validation():
    assert Constraint.from_spec({"type": "budget_max", "max_cents": 0}).max_cents == 0
    with pytest.raises(ConstraintError):
        Constraint.from_spec({"type": "budget_max", "max_cents": -1})  # negative budget


def test_unknown_type_rejected():
    with pytest.raises(ConstraintError):
        Constraint.from_spec({"type": "spend_at_most", "max_cents": 5})


# --------------------------------------------------------------------------- #
# Ledger immutability and re-versioning
# --------------------------------------------------------------------------- #

def test_build_produces_version_1():
    ledger = Ledger.build(_specs({"type": "item_quantity", "sku": "A", "quantity": 1}))
    assert ledger.version == 1
    assert len(ledger.constraints) == 1


def test_empty_constraints_is_an_error():
    with pytest.raises(ConstraintError):
        Ledger.build([])


def test_rebuild_increments_version_and_preserves_original():
    original = Ledger.build(_specs({"type": "item_quantity", "sku": "A", "quantity": 1}))
    rebuilt = original.rebuild([{"type": "item_quantity", "sku": "B", "quantity": 5}])
    assert rebuilt.version == 2
    # Original is untouched.
    assert original.version == 1
    assert original.constraints[0].sku == "A"
    assert rebuilt.constraints[0].sku == "B"


def test_constraint_is_immutable():
    c = Constraint.from_spec({"type": "item_quantity", "sku": "A", "quantity": 1})
    with pytest.raises(Exception):
        c.sku = "B"  # frozen dataclass


def test_ledger_echo_round_trips():
    ledger = Ledger.build(_specs(
        {"type": "item_quantity", "sku": "A", "quantity": 1},
        {"type": "count_distinct", "skus": ["A", "B"], "count": 1},
        {"type": "budget_max", "max_cents": 100},
    ))
    d = ledger.to_dict()
    assert d["version"] == 1
    assert d["constraints"][1]["count_mode"] == "exactly"
    # Rebuild from the echoed dict is possible for the non-list specs via from_spec.
    assert Constraint.from_spec(d["constraints"][0]).sku == "A"


def test_states_constants():
    assert STATES == {UNVERIFIED, CONFIRMED, CONTRADICTED, UNKNOWN}

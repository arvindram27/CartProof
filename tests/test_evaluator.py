"""Evaluator honesty tests (Brief 4 section 6).

Uses only synthetic SQLite fixtures and ``eval/dev`` tasks — never ``eval/frozen``
tasks. Proves the evaluator reads persisted state directly, ignores the
fault-injection log and runtime reports, and can fail a run.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from harness import evaluate
from harness.snapshots import backup_copy, read_cart

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_db(path: Path, lines: dict, n_sessions: int = 1, with_faults: bool = False):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, cart_id TEXT NOT NULL)")
    conn.execute("CREATE TABLE carts (cart_id TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)")
    conn.execute(
        "CREATE TABLE cart_lines (line_id TEXT PRIMARY KEY, cart_id TEXT NOT NULL, "
        "sku TEXT NOT NULL, quantity INTEGER NOT NULL, unit_price_cents INTEGER NOT NULL)")
    if with_faults:
        conn.execute("CREATE TABLE faults (name TEXT PRIMARY KEY, params TEXT NOT NULL)")
        conn.execute("INSERT INTO faults VALUES ('quantity-drift', '{}')")
    for i in range(n_sessions):
        cid = f"cart-{i}"
        conn.execute("INSERT INTO carts VALUES (?, 1)", (cid,))
        conn.execute("INSERT INTO sessions VALUES (?, ?)", (f"sid-{i}", cid))
        for sku, qty in lines.items():
            conn.execute(
                "INSERT INTO cart_lines VALUES (?, ?, ?, ?, 100)",
                (f"{cid}-{sku}", cid, sku, qty))
    conn.commit()
    conn.close()


def test_compare_lines_detects_all_kinds():
    expected = {"A": 1, "B": 2, "C": 1}
    actual = {"A": 1, "B": 3, "D": 1}
    mism = evaluate.compare_lines(expected, actual)
    got = {(m["sku"], m["expected_quantity"], m["actual_quantity"]) for m in mism}
    assert got == {("B", 2, 3), ("C", 1, 0), ("D", 0, 1)}


def test_derive_prefixes_multi_item_and_repeated_sku():
    seeds = [{"sku": "A", "quantity": 1}]
    items = [{"target_sku": "B", "quantity": 1},
             {"target_sku": "A", "quantity": 2}]
    prefixes = evaluate.derive_prefixes(seeds, items)
    assert prefixes == [{"A": 1}, {"A": 1, "B": 1}, {"A": 3, "B": 1}]
    assert evaluate.final_expected_prefix(prefixes) == {"A": 3, "B": 1}


def test_measure_final_correct(tmp_path):
    db = tmp_path / "cart.db"
    _make_db(db, {"NAVYBLUE": 1})
    out = evaluate.measure_final_db(str(db), {"NAVYBLUE": 1})
    assert out["measured_outcome"] == "correct"
    assert out["mismatches"] == []


def test_wrong_expected_end_state_is_caught(tmp_path):
    # The planted wrong expectation (quantity 2) must be reported as a mismatch.
    db = tmp_path / "cart.db"
    _make_db(db, {"NAVYBLUE": 1})
    out = evaluate.measure_final_db(str(db), {"NAVYBLUE": 2})
    assert out["measured_outcome"] == "incorrect"
    assert out["mismatches"] == [{"sku": "NAVYBLUE", "expected_quantity": 2,
                                  "actual_quantity": 1}]


def test_measure_final_ambiguous_is_unknown(tmp_path):
    db = tmp_path / "cart.db"
    _make_db(db, {"A": 1}, n_sessions=2)
    out = evaluate.measure_final_db(str(db), {"A": 1}, cookie_session_id=None)
    assert out["measured_outcome"] == "unknown"
    assert out["resolution"] == "ambiguous"


def test_measure_final_empty_is_unknown(tmp_path):
    db = tmp_path / "cart.db"
    _make_db(db, {}, n_sessions=0)
    out = evaluate.measure_final_db(str(db), {})
    assert out["measured_outcome"] == "unknown"
    assert out["resolution"] == "empty"


def test_measure_final_cookie_resolves(tmp_path):
    db = tmp_path / "cart.db"
    _make_db(db, {"A": 1}, n_sessions=2)
    out = evaluate.measure_final_db(str(db), {"A": 1}, cookie_session_id="sid-1")
    assert out["measured_outcome"] == "correct"
    assert out["resolution"] == "cookie"


def test_evaluator_ignores_injection_log(tmp_path):
    # Same persisted lines, one DB with a fault row, one without: identical verdict.
    with_faults = tmp_path / "with.db"
    without = tmp_path / "without.db"
    _make_db(with_faults, {"A": 1}, with_faults=True)
    _make_db(without, {"A": 1}, with_faults=False)
    a = evaluate.measure_final_db(str(with_faults), {"A": 1})
    b = evaluate.measure_final_db(str(without), {"A": 1})
    assert a["measured_outcome"] == b["measured_outcome"] == "correct"


def test_backup_copy_survives_uncheckpointed_wal(tmp_path):
    db = tmp_path / "wal.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (a int)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.execute("INSERT INTO t VALUES (2)")
    conn.commit()  # committed but possibly still in WAL
    conn.close()
    snap = tmp_path / "snap.db"
    backup_copy(str(db), str(snap))
    c2 = sqlite3.connect(str(snap))
    assert [r[0] for r in c2.execute("SELECT a FROM t").fetchall()] == [1, 2]
    c2.close()


def _forbidden_imports(source: str):
    return [tok for tok in ("core.verify", "core.recovery", "core.ledger",
                            "agents.", "adapter.", "store.") if tok in source]


def test_evaluator_and_observer_have_no_runtime_imports():
    for rel in ("harness/evaluate.py", "harness/snapshots.py"):
        src = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        assert _forbidden_imports(src) == [], f"{rel} imports runtime/store code"


def test_observer_has_no_expectation_access():
    src = (PROJECT_ROOT / "harness/snapshots.py").read_text(encoding="utf-8")
    assert "expected_lines" not in src
    assert "end_state" not in src
    assert "evaluate" not in src
    assert "metrics" not in src

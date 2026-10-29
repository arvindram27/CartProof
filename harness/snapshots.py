"""Harness-only persisted-state snapshot observer for the Brief 4 evaluator.

Captures raw persisted cart state and adapter call context. It never imports or
receives eval expectations, arm-outcome expectations, or anything beyond the
runtime spec. It reads the store's SQLite file directly through the backup API so
committed data still in a WAL is never missed, and it never goes through the
store's HTTP read path (so it cannot consume the one-shot ``external-mutation``
fault or trigger read-path fault behaviour).

Only ``tests/`` and ``harness/`` may import the store or touch its SQLite file;
this module lives at harness level and is the component allowed to read state
directly, which is exactly its job.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional


def _open(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(Path(path)), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
    except Exception:
        pass
    return conn


def backup_copy(src_db: str, dst_db: str) -> None:
    """Copy a live (possibly WAL) SQLite database consistently.

    Uses the SQLite backup API so committed pages still held in the ``-wal`` file
    are included. The source is opened read-only so it can never modify the live
    store file.
    """
    src = _open(src_db)
    dst = sqlite3.connect(str(Path(dst_db)))
    try:
        with dst:
            src.backup(dst)
    finally:
        dst.close()
        src.close()


def read_cart(db_file: str, cookie_session_id: Optional[str] = None) -> dict:
    """Read persisted cart state from a database file.

    Returns ``{"lines", "version", "total_cents", "resolution"}`` where
    ``lines`` is ``{sku: quantity}`` (or ``None`` when unresolvable) and
    ``resolution`` is one of ``cookie``, ``sole_session``, ``empty``,
    ``ambiguous``, or ``error``.
    """
    conn = None
    try:
        conn = _open(db_file)
    except Exception as exc:
        return {"lines": None, "version": None, "total_cents": None,
                "resolution": "error", "error": repr(exc)}
    try:
        if cookie_session_id:
            row = conn.execute(
                "SELECT cart_id FROM sessions WHERE session_id = ?",
                (cookie_session_id,),
            ).fetchone()
            if row is not None:
                return _lines_for(conn, row["cart_id"], "cookie")
        rows = conn.execute("SELECT cart_id FROM sessions").fetchall()
        if len(rows) == 1:
            return _lines_for(conn, rows[0]["cart_id"], "sole_session")
        if len(rows) == 0:
            carts = conn.execute("SELECT cart_id FROM carts").fetchall()
            if len(carts) == 0:
                return {"lines": {}, "version": 0, "total_cents": 0,
                        "resolution": "empty"}
            return {"lines": None, "version": None, "total_cents": None,
                    "resolution": "ambiguous"}
        return {"lines": None, "version": None, "total_cents": None,
                "resolution": "ambiguous"}
    except Exception as exc:
        return {"lines": None, "version": None, "total_cents": None,
                "resolution": "error", "error": repr(exc)}
    finally:
        if conn is not None:
            conn.close()


def _lines_for(conn: sqlite3.Connection, cart_id: str, resolution: str) -> dict:
    crow = conn.execute(
        "SELECT version FROM carts WHERE cart_id = ?", (cart_id,)
    ).fetchone()
    version = crow["version"] if crow else 0
    lrows = conn.execute(
        "SELECT sku, quantity FROM cart_lines WHERE cart_id = ?", (cart_id,)
    ).fetchall()
    lines: dict[str, int] = {}
    for r in lrows:
        lines[r["sku"]] = lines.get(r["sku"], 0) + int(r["quantity"])
    total = conn.execute(
        "SELECT COALESCE(SUM(quantity * unit_price_cents), 0) AS t "
        "FROM cart_lines WHERE cart_id = ?",
        (cart_id,),
    ).fetchone()["t"]
    return {"lines": lines, "version": version, "total_cents": int(total),
            "resolution": resolution}


class Observer:
    """Captures raw persisted cart state and adapter call context.

    Knows nothing about expectations. It resolves the active cart from the
    ``cp_session`` cookie (mapped through the ``sessions`` table) with a
    sole-session fallback, and reports ``unknown`` on ambiguity.
    """

    def __init__(self, db_path: str, snapshot_dir: str, session: Any) -> None:
        self.db_path = db_path
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.session = session  # the real BrowserSession (for cookie reads)
        self.snapshots: list[dict] = []
        self.mutations: list[dict] = []
        self.hook_calls: dict[str, int] = {"before_add": 0, "after_add": 0,
                                           "pre_add": 0}
        self._contexts: list[str] = []
        self._counter = 0
        self._current_item_index: Optional[int] = None
        self._current_action_id: Optional[str] = None
        self._current_attempt_id: Optional[int] = None

    @contextmanager
    def context(self, name: str):
        self._contexts.append(name)
        try:
            yield
        finally:
            self._contexts.pop()

    @property
    def current_context(self) -> str:
        return self._contexts[-1] if self._contexts else "actor"

    def _cookie_session_id(self) -> Optional[str]:
        try:
            for c in self.session.context.cookies():
                if c.get("name") == "cp_session":
                    return c.get("value")
        except Exception:
            return None
        return None

    def record_hook_call(self, kind: str) -> None:
        self.hook_calls[kind] = self.hook_calls.get(kind, 0) + 1

    def hook_entry(self, hook: str, event: dict) -> None:
        item = event.get("item") or {}
        index = item.get("index")
        action_id = event.get("action_id")
        self._current_item_index = index
        self._current_action_id = action_id
        self._current_attempt_id = 1  # the scripted actor never retries
        self._capture("hook_entry", hook, None, index, action_id, 1,
                      self.current_context)

    def record_mutation(self, operation: str, target: dict) -> None:
        context = self.current_context
        corrective = (context == "after_add_hook")
        rec: dict[str, Any] = {
            "operation": operation,
            "context": context,
            "corrective": corrective,
            "target": target,
        }
        if corrective:
            snap = self._capture(
                "before_write", None, operation,
                self._current_item_index, self._current_action_id,
                self._current_attempt_id, context)
            rec["before_write_snapshot"] = snap
        self.mutations.append(rec)

    def _capture(self, kind: str, hook: Optional[str], operation: Optional[str],
                 item_index, action_id, attempt_id, context) -> dict:
        self._counter += 1
        t0 = time.perf_counter()
        snap_file = self.snapshot_dir / f"snap-{self._counter:04d}.db"
        try:
            backup_copy(self.db_path, str(snap_file))
            cart = read_cart(str(snap_file), self._cookie_session_id())
        except Exception as exc:
            cart = {"lines": None, "version": None, "total_cents": None,
                    "resolution": "error", "error": repr(exc)}
        dt_ms = (time.perf_counter() - t0) * 1000.0
        rec = {
            "seq": self._counter,
            "kind": kind,
            "hook": hook,
            "operation": operation,
            "item_index": item_index,
            "action_id": action_id,
            "attempt_id": attempt_id,
            "context": context,
            "capture_ms": round(dt_ms, 3),
            "resolution": cart["resolution"],
            "lines": cart["lines"],
            "version": cart["version"],
            "total_cents": cart["total_cents"],
            "error": cart.get("error"),
        }
        self.snapshots.append(rec)
        return rec

    def summary(self) -> dict:
        adapter = {"add_to_cart": 0, "set_line_quantity": 0, "delete_line": 0}
        for m in self.mutations:
            op = m["operation"]
            if op in adapter:
                adapter[op] += 1
        return {
            "hook_counts": dict(self.hook_calls),
            "adapter_mutations": adapter,
            "snapshots": list(self.snapshots),
            "mutations": list(self.mutations),
            "capture_ms_total": round(sum(s["capture_ms"] for s in self.snapshots), 3),
        }


class ObservedSession:
    """Pass-through wrapper that observes adapter mutations without changing them.

    All non-mutating attributes and methods delegate unchanged. The three
    mutating methods record the call (and, for corrective calls, capture a
    before-write snapshot) then delegate to the real session verbatim.
    """

    def __init__(self, real_session: Any, observer: Observer) -> None:
        self._real = real_session
        self._observer = observer

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def add_to_cart(self, quantity: int = 1, sku: str | None = None):
        self._observer.record_mutation(
            "add_to_cart", {"sku": sku, "quantity": quantity})
        return self._real.add_to_cart(quantity, sku=sku)

    def set_line_quantity(self, line_id: str, sku: str, quantity: int,
                          expected_version: int):
        self._observer.record_mutation(
            "set_line_quantity",
            {"line_id": line_id, "sku": sku, "quantity": quantity})
        return self._real.set_line_quantity(line_id, sku, quantity,
                                            expected_version)

    def delete_line(self, line_id: str, sku: str, expected_version: int):
        self._observer.record_mutation(
            "delete_line", {"line_id": line_id, "sku": sku})
        return self._real.delete_line(line_id, sku, expected_version)

"""SQLite schema and low-level data access for the CartProof mock store.

Everything here lives inside a project-owned SQLite file. No external
services, no secrets. Prices are integer cents. Cart mutations are version
guarded inside a single SQLite transaction so a stale writer can never leave
a partial write behind.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    cart_id     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS carts (
    cart_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cart_lines (
    line_id           TEXT PRIMARY KEY,
    cart_id           TEXT NOT NULL,
    sku               TEXT NOT NULL,
    quantity          INTEGER NOT NULL,
    unit_price_cents  INTEGER NOT NULL,
    added_at          TEXT NOT NULL,
    UNIQUE (cart_id, sku)
);
CREATE TABLE IF NOT EXISTS products (
    product_id       TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    category         TEXT NOT NULL,
    base_price_cents INTEGER NOT NULL,
    sort_order       INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS variants (
    sku         TEXT PRIMARY KEY,
    product_id   TEXT NOT NULL,
    label        TEXT NOT NULL,
    size         TEXT,
    color        TEXT,
    price_cents  INTEGER NOT NULL,
    sort_order   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS faults (
    name   TEXT PRIMARY KEY,
    params TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cart_snapshots (
    cart_id       TEXT NOT NULL,
    version       INTEGER NOT NULL,
    lines_json    TEXT NOT NULL,
    total_cents   INTEGER NOT NULL,
    created_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS stale_totals (
    cart_id     TEXT PRIMARY KEY,
    version     INTEGER NOT NULL,
    total_cents INTEGER NOT NULL
);
"""

class VersionConflict(Exception):
    """Raised when a mutation carries a stale version."""

    def __init__(self, current_version: int):
        super().__init__(f"version conflict: current version is {current_version}")
        self.current_version = current_version


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> int:
    return int(time.time() * 1000)


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def seed_catalog(conn: sqlite3.Connection, catalog_path: str) -> None:
    """Load the seed catalog into the products/variants tables.

    Idempotent: if products already exist, do nothing.
    """
    existing = conn.execute("SELECT COUNT(*) AS n FROM products").fetchone()["n"]
    if existing:
        return
    data = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    conn.execute("BEGIN IMMEDIATE")
    try:
        for order, p in enumerate(data["products"]):
            conn.execute(
                "INSERT INTO products (product_id, name, category, base_price_cents, sort_order) "
                "VALUES (?, ?, ?, ?, ?)",
                (p["id"], p["name"], p["category"], p["base_price_cents"], order),
            )
            for vorder, v in enumerate(p["variants"]):
                conn.execute(
                    "INSERT INTO variants (sku, product_id, label, size, color, price_cents, sort_order) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (v["sku"], p["id"], v["label"], v.get("size"), v.get("color"),
                     v["price_cents"], vorder),
                )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def init_and_seed(db_path: str, catalog_path: str) -> None:
    conn = connect(db_path)
    try:
        init_schema(conn)
        seed_catalog(conn, catalog_path)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Faults
# --------------------------------------------------------------------------- #

def get_faults(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute("SELECT name, params FROM faults").fetchall()
    return {r["name"]: json.loads(r["params"]) for r in rows}


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

def get_or_create_session(conn: sqlite3.Connection, session_id: Optional[str]):
    """Return ``(cart_id, session_id, created)``.

    Creates a new cart (version 0) and its initial snapshot when needed.
    """
    if session_id:
        row = conn.execute(
            "SELECT cart_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row:
            return row["cart_id"], session_id, False
    sid = uuid.uuid4().hex
    cart_id = uuid.uuid4().hex
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("INSERT INTO carts (cart_id, version) VALUES (?, 0)", (cart_id,))
        conn.execute(
            "INSERT INTO sessions (session_id, cart_id, created_at) VALUES (?, ?, ?)",
            (sid, cart_id, _now_iso()),
        )
        _record_snapshot(conn, cart_id, 0, [], 0)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return cart_id, sid, True


# --------------------------------------------------------------------------- #
# Line and cart reads
# --------------------------------------------------------------------------- #

def _load_lines(conn: sqlite3.Connection, cart_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT cl.line_id, cl.sku, cl.quantity, cl.unit_price_cents,
               v.label, v.product_id, p.name
        FROM cart_lines cl
        JOIN variants v ON v.sku = cl.sku
        JOIN products p ON p.product_id = v.product_id
        WHERE cl.cart_id = ?
        ORDER BY cl.added_at, cl.line_id
        """,
        (cart_id,),
    ).fetchall()
    lines = []
    for r in rows:
        lines.append({
            "line_id": r["line_id"],
            "sku": r["sku"],
            "name": r["name"],
            "product_id": r["product_id"],
            "label": r["label"],
            "quantity": r["quantity"],
            "unit_price_cents": r["unit_price_cents"],
            "line_total_cents": r["quantity"] * r["unit_price_cents"],
        })
    return lines


def _total(conn: sqlite3.Connection, cart_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(quantity * unit_price_cents), 0) AS t "
        "FROM cart_lines WHERE cart_id = ?",
        (cart_id,),
    ).fetchone()
    return int(row["t"])


def read_persisted_cart(conn: sqlite3.Connection, cart_id: str) -> dict[str, Any]:
    """Direct read of persisted state (bypasses the slow-consistency projection).

    Tests and the harness use this to prove a committed write was never lost.
    """
    row = conn.execute("SELECT version FROM carts WHERE cart_id = ?", (cart_id,)).fetchone()
    version = row["version"] if row else 0
    lines = _load_lines(conn, cart_id)
    total = sum(l["quantity"] * l["unit_price_cents"] for l in lines)
    return {"cart_id": cart_id, "version": version, "lines": lines, "total_cents": total}


def _record_snapshot(conn: sqlite3.Connection, cart_id: str, version: int,
                     lines: list[dict[str, Any]], total_cents: int) -> None:
    conn.execute(
        "INSERT INTO cart_snapshots (cart_id, version, lines_json, total_cents, created_at_ms) "
        "VALUES (?, ?, ?, ?, ?)",
        (cart_id, version, json.dumps(lines), total_cents, _now_ms()),
    )


def _record_snapshot_for_cart(conn: sqlite3.Connection, cart_id: str, version: int) -> None:
    lines = _load_lines(conn, cart_id)
    total = sum(l["quantity"] * l["unit_price_cents"] for l in lines)
    _record_snapshot(conn, cart_id, version, lines, total)


def _read_from_snapshot(conn: sqlite3.Connection, cart_id: str, delay_ms: int) -> dict[str, Any]:
    cutoff = _now_ms() - delay_ms
    row = conn.execute(
        "SELECT version, lines_json, total_cents FROM cart_snapshots "
        "WHERE cart_id = ? AND created_at_ms <= ? "
        "ORDER BY created_at_ms DESC, version DESC LIMIT 1",
        (cart_id, cutoff),
    ).fetchone()
    if row is None:
        return {"cart_id": cart_id, "version": 0, "lines": [], "total_cents": 0}
    return {
        "cart_id": cart_id,
        "version": row["version"],
        "lines": json.loads(row["lines_json"]),
        "total_cents": row["total_cents"],
    }


def _apply_external_mutation(conn: sqlite3.Connection, cart_id: str, delta: int) -> None:
    """One-shot: bump the first line's quantity, version it, snapshot it."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT line_id FROM cart_lines WHERE cart_id = ? ORDER BY line_id LIMIT 1",
            (cart_id,),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return
        conn.execute(
            "UPDATE cart_lines SET quantity = quantity + ? WHERE line_id = ?",
            (delta, row["line_id"]),
        )
        conn.execute("UPDATE carts SET version = version + 1 WHERE cart_id = ?", (cart_id,))
        new_version = conn.execute(
            "SELECT version FROM carts WHERE cart_id = ?", (cart_id,)
        ).fetchone()["version"]
        _record_snapshot_for_cart(conn, cart_id, new_version)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def read_cart_view(conn: sqlite3.Connection, cart_id: str) -> dict[str, Any]:
    """Read projection served to ``GET /api/cart`` and ``/cart``.

    Applies the fault behaviours that affect reads: external-mutation (once),
    slow-consistency (lagged snapshot), and stale-total (pre-write total).
    dom-api-mismatch is applied only at render time by the page route.
    """
    faults = get_faults(conn)
    if "external-mutation" in faults:
        delta = int(faults["external-mutation"].get("delta", 1))
        _apply_external_mutation(conn, cart_id, delta)
        conn.execute("DELETE FROM faults WHERE name = 'external-mutation'")
        faults = get_faults(conn)

    if "slow-consistency" in faults:
        delay_ms = int(faults["slow-consistency"].get("delay_ms", 500))
        return _read_from_snapshot(conn, cart_id, delay_ms)

    view = read_persisted_cart(conn, cart_id)
    if "stale-total" in faults:
        row = conn.execute(
            "SELECT total_cents FROM stale_totals WHERE cart_id = ?", (cart_id,)
        ).fetchone()
        if row is not None:
            view = dict(view)
            view["total_cents"] = row["total_cents"]
    return view


# --------------------------------------------------------------------------- #
# Mutations
# --------------------------------------------------------------------------- #

def mutate_cart(conn: sqlite3.Connection, cart_id: str, expected_version: int,
                apply_fn) -> dict[str, Any]:
    """Run ``apply_fn(conn, cart_id)`` inside a version-guarded transaction.

    Raises :class:`VersionConflict` if the caller's version is stale. The
    version increment is a guarded ``UPDATE`` whose affected-row count is
    checked, so no error path leaves a partial write.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT version FROM carts WHERE cart_id = ?", (cart_id,)).fetchone()
        current = row["version"] if row else 0
        if current != expected_version:
            conn.execute("ROLLBACK")
            raise VersionConflict(current)

        pre_write_total = _total(conn, cart_id)
        result = apply_fn(conn, cart_id)

        cur = conn.execute(
            "UPDATE carts SET version = version + 1 WHERE cart_id = ? AND version = ?",
            (cart_id, current),
        )
        if cur.rowcount != 1:
            conn.execute("ROLLBACK")
            raise RuntimeError("version guard failed: no row updated")

        new_version = current + 1
        _record_snapshot_for_cart(conn, cart_id, new_version)

        if "stale-total" in get_faults(conn):
            conn.execute(
                "INSERT INTO stale_totals (cart_id, version, total_cents) VALUES (?, ?, ?) "
                "ON CONFLICT(cart_id) DO UPDATE SET version = excluded.version, "
                "total_cents = excluded.total_cents",
                (cart_id, current, pre_write_total),
            )

        conn.execute("COMMIT")
        return result
    except VersionConflict:
        raise
    except Exception:
        conn.execute("ROLLBACK")
        raise


def add_line_guarded(conn: sqlite3.Connection, cart_id: str, expected_version: int,
                     sku: str, quantity: int, unit_price_cents: int) -> dict[str, Any]:
    """Version-guarded add. Merges quantities when the same sku already exists."""

    def _apply(conn, cart_id):
        existing = conn.execute(
            "SELECT line_id, quantity FROM cart_lines WHERE cart_id = ? AND sku = ?",
            (cart_id, sku),
        ).fetchone()
        if existing:
            new_qty = existing["quantity"] + quantity
            conn.execute(
                "UPDATE cart_lines SET quantity = ? WHERE line_id = ?",
                (new_qty, existing["line_id"]),
            )
            line_id = existing["line_id"]
            final_qty = new_qty
        else:
            line_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO cart_lines (line_id, cart_id, sku, quantity, unit_price_cents, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (line_id, cart_id, sku, quantity, unit_price_cents, _now_iso()),
            )
            final_qty = quantity
        return {"line_id": line_id, "sku": sku, "quantity": final_qty}

    return mutate_cart(conn, cart_id, expected_version, _apply)


def update_line_guarded(conn: sqlite3.Connection, cart_id: str, expected_version: int,
                        line_id: str, quantity: int) -> dict[str, Any]:
    def _apply(conn, cart_id):
        cur = conn.execute(
            "UPDATE cart_lines SET quantity = ? WHERE line_id = ? AND cart_id = ?",
            (quantity, line_id, cart_id),
        )
        if cur.rowcount != 1:
            raise LineNotFound(line_id)
        return {"line_id": line_id, "quantity": quantity}

    return mutate_cart(conn, cart_id, expected_version, _apply)


def delete_line_guarded(conn: sqlite3.Connection, cart_id: str, expected_version: int,
                        line_id: str) -> dict[str, Any]:
    def _apply(conn, cart_id):
        cur = conn.execute(
            "DELETE FROM cart_lines WHERE line_id = ? AND cart_id = ?", (line_id, cart_id)
        )
        if cur.rowcount != 1:
            raise LineNotFound(line_id)
        return {"line_id": line_id}

    return mutate_cart(conn, cart_id, expected_version, _apply)


class LineNotFound(Exception):
    def __init__(self, line_id: str):
        super().__init__(f"no such cart line: {line_id}")
        self.line_id = line_id


# --------------------------------------------------------------------------- #
# Catalog lookups (used by pages and the API)
# --------------------------------------------------------------------------- #

def list_products(conn: sqlite3.Connection, q: str = "") -> list[dict[str, Any]]:
    if q:
        like = f"%{q}%"
        rows = conn.execute(
            "SELECT product_id, name, category, base_price_cents FROM products "
            "WHERE name LIKE ? OR category LIKE ? ORDER BY sort_order",
            (like, like),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT product_id, name, category, base_price_cents FROM products ORDER BY sort_order"
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["product_id"],
            "name": r["name"],
            "category": r["category"],
            "base_price_cents": r["base_price_cents"],
            "price_label": f"${r['base_price_cents'] / 100:.2f}",
        })
    return out


def get_product(conn: sqlite3.Connection, product_id: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT product_id, name, category, base_price_cents FROM products WHERE product_id = ?",
        (product_id,),
    ).fetchone()
    if row is None:
        return None
    vs = conn.execute(
        "SELECT sku, label, size, color, price_cents FROM variants "
        "WHERE product_id = ? ORDER BY sort_order",
        (product_id,),
    ).fetchall()
    variants = [
        {
            "sku": v["sku"],
            "label": v["label"],
            "size": v["size"],
            "color": v["color"],
            "price_cents": v["price_cents"],
            "price_label": f"${v['price_cents'] / 100:.2f}",
        }
        for v in vs
    ]
    return {
        "id": row["product_id"],
        "name": row["name"],
        "category": row["category"],
        "base_price_cents": row["base_price_cents"],
        "variants": variants,
    }


def get_variant(conn: sqlite3.Connection, sku: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT sku, product_id, label, size, color, price_cents FROM variants WHERE sku = ?",
        (sku,),
    ).fetchone()
    if row is None:
        return None
    return {
        "sku": row["sku"],
        "product_id": row["product_id"],
        "label": row["label"],
        "size": row["size"],
        "color": row["color"],
        "price_cents": row["price_cents"],
    }


def confusable_sibling(conn: sqlite3.Connection, sku: str) -> Optional[str]:
    """Return a same-product variant whose label differs only by ' Blue', else the
    first other variant of the product, else None."""
    var = get_variant(conn, sku)
    if var is None:
        return None
    label = var["label"]
    rows = conn.execute(
        "SELECT sku, label FROM variants WHERE product_id = ? AND sku != ? ORDER BY sort_order",
        (var["product_id"], sku),
    ).fetchall()
    if not rows:
        return None
    for r in rows:
        if r["label"] == label + " Blue" or label == r["label"] + " Blue":
            return r["sku"]
    return rows[0]["sku"]

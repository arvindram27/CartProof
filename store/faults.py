"""Fault flags and injection logic.

Faults are stored in the project-owned SQLite file and only ever acted on by
``harness/`` and ``tests/`` through the ``/_control/`` endpoints. The
injection behaviours themselves live in ``models.py`` (write path) and
``api.py`` / ``pages.py`` (read/render path).
"""
from __future__ import annotations

import json
from typing import Any

import sqlite3

#: Fault names recognised by ``POST /_control/fault`` and their default params.
KNOWN_FAULTS: dict[str, dict[str, Any]] = {
    "silent-drop": {},
    "wrong-variant": {},
    "quantity-drift": {"delta": 1},
    "stale-total": {},
    "dom-api-mismatch": {},
    "slow-consistency": {"delay_ms": 500},
    "external-mutation": {"delta": 1},
}


def normalize_fault(name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Validate a fault name and merge caller params over the defaults."""
    if name not in KNOWN_FAULTS:
        raise ValueError(f"unknown fault: {name}")
    merged = dict(KNOWN_FAULTS[name])
    if params:
        merged.update(params)
    return merged


def set_fault(conn: sqlite3.Connection, name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    merged = normalize_fault(name, params)
    conn.execute(
        "INSERT INTO faults (name, params) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET params = excluded.params",
        (name, json.dumps(merged)),
    )
    return merged


def clear_all_faults(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM faults")


def reset_store(conn: sqlite3.Connection, catalog_path: str) -> None:
    """Clear faults, carts, sessions, snapshots and re-seed the catalog.

    Acts only on tables/files this project itself created.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM faults")
        conn.execute("DELETE FROM cart_snapshots")
        conn.execute("DELETE FROM stale_totals")
        conn.execute("DELETE FROM cart_lines")
        conn.execute("DELETE FROM carts")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM variants")
        conn.execute("DELETE FROM products")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    from .models import seed_catalog

    seed_catalog(conn, catalog_path)

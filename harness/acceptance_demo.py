"""Demonstrations for acceptance checks 5 and 6.

Check 5: transactional guard with two overlapping clients (stale writer gets
409 and the cart is left exactly as the winning writer left it).

Check 6: slow-consistency — a committed write is visible to a direct
persisted-state read immediately, hidden by the lagging projection until the
delay elapses, and never lost.

Run:  python -m harness.acceptance_demo
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from harness.run_task import boot_server, stop_server  # noqa: E402
from store import models  # noqa: E402  (harness may import store)
from store.app import create_app, create_control_app  # noqa: E402

SKU_A = "BTL-HIK-500-NAVY"
SKU_B = "BTL-HIK-750-STEEL"


def _client(base: str) -> httpx.Client:
    c = httpx.Client()
    c.get(f"{base}/api/cart").raise_for_status()
    return c


def _cart(client: httpx.Client, base: str) -> dict:
    return client.get(f"{base}/api/cart").json()


def main() -> None:
    tmp = PROJECT_ROOT / "var" / "demo"
    tmp.mkdir(parents=True, exist_ok=True)
    db_path = str(tmp / "demo.db")
    catalog_path = str(PROJECT_ROOT / "store" / "seed_catalog.json")

    store_app = create_app(db_path, catalog_path)
    control_app = create_control_app(db_path, catalog_path)
    s_srv, s_thr, s_port = boot_server(store_app)
    c_srv, c_thr, c_port = boot_server(control_app)
    base = f"http://127.0.0.1:{s_port}"
    control = f"http://127.0.0.1:{c_port}"

    try:
        # -------------------------------------------------------------- Check 5
        print("=" * 70)
        print("CHECK 5: transactional guard with two overlapping clients")
        print("=" * 70)
        httpx.post(f"{control}/_control/reset").raise_for_status()

        a = _client(base)
        b = httpx.Client()
        b.cookies.set("cp_session", a.cookies["cp_session"])

        n = _cart(a, base)["version"]
        print(f"client A reads version  N   = {n}")
        print(f"client B reads version  N   = {_cart(b, base)['version']}")

        rb = b.post(
            f"{base}/api/cart/lines",
            json={"sku": SKU_A, "quantity": 1},
            headers={"X-Cart-Version": str(n)},
        )
        print(f"client B add {SKU_A} -> HTTP {rb.status_code}, version {rb.json()['version']}")

        ra = a.post(
            f"{base}/api/cart/lines",
            json={"sku": SKU_B, "quantity": 1},
            headers={"X-Cart-Version": str(n)},  # still carrying N
        )
        print(f"client A add {SKU_B} (version {n}) -> HTTP {ra.status_code}")
        print(f"  body: {ra.json()}")

        after = _cart(a, base)
        print(f"follow-up read -> version {after['version']}, "
              f"lines = {[l['sku'] for l in after['lines']]}")

        # -------------------------------------------------------------- Check 6
        print()
        print("=" * 70)
        print("CHECK 6: slow-consistency (persisted write vs lagging projection)")
        print("=" * 70)
        httpx.post(f"{control}/_control/reset").raise_for_status()
        httpx.post(
            f"{control}/_control/fault",
            json={"name": "slow-consistency", "params": {"delay_ms": 800}},
        ).raise_for_status()

        c = _client(base)
        cart_id = _cart_id(db_path, c)

        r = c.post(
            f"{base}/api/cart/lines",
            json={"sku": SKU_A, "quantity": 1},
            headers={"X-Cart-Version": "0"},
        )
        print(f"write (add {SKU_A}) -> HTTP {r.status_code}")

        persisted = _persisted(db_path, cart_id)
        print(f"persisted state NOW    -> version {persisted['version']}, "
              f"lines = {[l['sku'] for l in persisted['lines']]}")

        projection = _cart(c, base)
        print(f"delayed projection NOW -> version {projection['version']}, "
              f"lines = {[l['sku'] for l in projection['lines']]}")

        time.sleep(0.9)
        projection_after = _cart(c, base)
        print(f"projection after delay -> version {projection_after['version']}, "
              f"lines = {[l['sku'] for l in projection_after['lines']]}")
    finally:
        stop_server(s_srv, s_thr)
        stop_server(c_srv, c_thr)


def _cart_id(db_path: str, client: httpx.Client) -> str:
    conn = models.connect(db_path)
    try:
        row = conn.execute(
            "SELECT cart_id FROM sessions WHERE session_id = ?",
            (client.cookies["cp_session"],),
        ).fetchone()
        return row["cart_id"]
    finally:
        conn.close()


def _persisted(db_path: str, cart_id: str) -> dict:
    conn = models.connect(db_path)
    try:
        return models.read_persisted_cart(conn, cart_id)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

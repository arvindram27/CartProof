"""Store API tests: versioning, CAS conflicts, and fault injection."""
from __future__ import annotations

import time

import httpx

from store import models  # tests may import the store to boot/verify

# Catalog fixture SKUs (from store/seed_catalog.json)
SKU_A = "BTL-HIK-500-NAVY"       # "Navy"
SKU_A_CONFUSABLE = "BTL-HIK-500-NAVYBLUE"  # "Navy Blue"
SKU_B = "BTL-HIK-750-STEEL"
SKU_PRICE = 2400


def new_client(base: str) -> httpx.Client:
    c = httpx.Client()
    c.get(f"{base}/api/cart").raise_for_status()
    return c


def get_cart(client: httpx.Client, base: str) -> dict:
    r = client.get(f"{base}/api/cart")
    r.raise_for_status()
    return r.json()


def add(client: httpx.Client, base: str, sku: str, quantity: int = 1, version: int | None = None):
    if version is None:
        version = get_cart(client, base)["version"]
    return client.post(
        f"{base}/api/cart/lines",
        json={"sku": sku, "quantity": quantity},
        headers={"X-Cart-Version": str(version)},
    )


def set_fault(control: str, name: str, params: dict | None = None):
    r = httpx.post(f"{control}/_control/fault", json={"name": name, "params": params})
    r.raise_for_status()
    return r.json()


def reset(control: str):
    r = httpx.post(f"{control}/_control/reset")
    r.raise_for_status()


# --------------------------------------------------------------------------- #
# Cart semantics
# --------------------------------------------------------------------------- #

def test_add_merge_and_version_increments(store):
    base = store["base"]
    c = new_client(base)

    assert get_cart(c, base)["version"] == 0
    assert get_cart(c, base)["lines"] == []

    r = add(c, base, SKU_A)
    assert r.status_code == 200
    assert r.json()["version"] == 1
    assert len(r.json()["lines"]) == 1
    assert r.json()["lines"][0]["quantity"] == 1
    assert r.json()["total_cents"] == SKU_PRICE

    # Adding the same sku again merges into a single line.
    r = add(c, base, SKU_A)
    assert r.status_code == 200
    assert r.json()["version"] == 2
    assert len(r.json()["lines"]) == 1
    assert r.json()["lines"][0]["quantity"] == 2

    # A different sku becomes a second line.
    r = add(c, base, SKU_B)
    assert r.json()["version"] == 3
    assert len(r.json()["lines"]) == 2


def test_unknown_sku_is_400_not_409(store):
    base = store["base"]
    c = new_client(base)
    r = add(c, base, "NOT-A-SKU")
    assert r.status_code == 400
    assert r.status_code != 409
    assert r.json()["error"]["code"] == "unknown_sku"
    # Nothing changed.
    assert get_cart(c, base)["version"] == 0
    assert get_cart(c, base)["lines"] == []


def test_missing_version_header_is_400(store):
    base = store["base"]
    c = new_client(base)
    r = c.post(f"{base}/api/cart/lines", json={"sku": SKU_A, "quantity": 1})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_version"


# --------------------------------------------------------------------------- #
# Transactional CAS: two overlapping clients
# --------------------------------------------------------------------------- #

def test_transactional_guard_two_clients(store):
    base = store["base"]

    client_a = new_client(base)
    client_b = httpx.Client()
    client_b.cookies.set("cp_session", client_a.cookies["cp_session"])

    # Both read version N.
    assert get_cart(client_a, base)["version"] == 0
    assert get_cart(client_b, base)["version"] == 0

    # B completes a mutation -> version N+1.
    rb = add(client_b, base, SKU_A)
    assert rb.status_code == 200
    assert rb.json()["version"] == 1

    # A submits its mutation still carrying version N -> 409.
    ra = add(client_a, base, SKU_B, version=0)
    assert ra.status_code == 409
    assert ra.json()["error"]["code"] == "version_conflict"
    assert ra.json()["error"]["current_version"] == 1

    # Follow-up read shows the cart exactly as B left it (version N+1).
    after = get_cart(client_a, base)
    assert after["version"] == 1
    skus = {l["sku"] for l in after["lines"]}
    assert skus == {SKU_A}
    assert SKU_B not in skus


# --------------------------------------------------------------------------- #
# Faults
# --------------------------------------------------------------------------- #

def test_fault_silent_drop(store):
    base, control = store["base"], store["control"]
    c = new_client(base)

    set_fault(control, "silent-drop")
    r = add(c, base, SKU_A)
    assert r.status_code == 200
    assert "added" in r.json()  # looks like success
    # But no line was written and the version did not change.
    assert get_cart(c, base)["version"] == 0
    assert get_cart(c, base)["lines"] == []

    reset(control)
    r = add(c, base, SKU_A)
    assert r.status_code == 200
    assert get_cart(c, base)["version"] == 1
    assert [l["sku"] for l in get_cart(c, base)["lines"]] == [SKU_A]


def test_fault_wrong_variant(store):
    base, control = store["base"], store["control"]
    c = new_client(base)

    set_fault(control, "wrong-variant")
    r = add(c, base, SKU_A)  # requests "Navy"
    assert r.status_code == 200
    # The confusable sibling "Navy Blue" was written instead.
    assert [l["sku"] for l in get_cart(c, base)["lines"]] == [SKU_A_CONFUSABLE]

    reset(control)
    c2 = new_client(base)
    r = add(c2, base, SKU_A)
    assert [l["sku"] for l in get_cart(c2, base)["lines"]] == [SKU_A]


def test_fault_quantity_drift(store):
    base, control = store["base"], store["control"]
    c = new_client(base)

    set_fault(control, "quantity-drift", {"delta": 1})
    r = add(c, base, SKU_A, quantity=1)
    assert r.status_code == 200
    assert get_cart(c, base)["lines"][0]["quantity"] == 2

    reset(control)
    c2 = new_client(base)
    add(c2, base, SKU_A, quantity=1)
    assert get_cart(c2, base)["lines"][0]["quantity"] == 1


def test_fault_stale_total(store):
    base, control = store["base"], store["control"]
    c = new_client(base)

    set_fault(control, "stale-total")
    add(c, base, SKU_A, quantity=1)
    view = get_cart(c, base)
    # Line was written, but the total is the pre-write total (0).
    assert len(view["lines"]) == 1
    assert view["lines"][0]["sku"] == SKU_A
    assert view["total_cents"] == 0

    reset(control)
    c2 = new_client(base)
    add(c2, base, SKU_A, quantity=1)
    assert get_cart(c2, base)["total_cents"] == SKU_PRICE


def test_fault_dom_api_mismatch(store):
    base, control = store["base"], store["control"]
    c = new_client(base)

    set_fault(control, "dom-api-mismatch")
    add(c, base, SKU_A)
    add(c, base, SKU_B)

    api_view = get_cart(c, base)
    assert {l["sku"] for l in api_view["lines"]} == {SKU_A, SKU_B}

    html = c.get(f"{base}/cart").text
    # The DOM drops the most recently added line (SKU_B).
    assert SKU_A in html
    assert SKU_B not in html

    reset(control)
    # After reset the fault is cleared and carts are emptied; rebuild and verify
    # the DOM now shows every line.
    c2 = new_client(base)
    add(c2, base, SKU_A)
    add(c2, base, SKU_B)
    html = c2.get(f"{base}/cart").text
    assert SKU_A in html and SKU_B in html


def test_fault_external_mutation(store):
    base, control = store["base"], store["control"]
    c = new_client(base)

    add(c, base, SKU_A, quantity=1)
    assert get_cart(c, base)["lines"][0]["quantity"] == 1

    set_fault(control, "external-mutation", {"delta": 1})
    # The next read applies the one-shot change.
    view = get_cart(c, base)
    assert view["lines"][0]["quantity"] == 2
    assert view["version"] == 2
    # It fires once: a second read does nothing further.
    view2 = get_cart(c, base)
    assert view2["lines"][0]["quantity"] == 2
    assert view2["version"] == 2


# --------------------------------------------------------------------------- #
# slow-consistency: persisted write vs lagging read projection
# --------------------------------------------------------------------------- #

def test_slow_consistency_persisted_vs_projection(store):
    base, control = store["base"], store["control"]
    c = new_client(base)
    cart_id = _cart_id_for_client(store["db"], c)

    set_fault(control, "slow-consistency", {"delay_ms": 800})

    r = add(c, base, SKU_A, quantity=1)
    assert r.status_code == 200

    # The write committed to persisted state immediately.
    persisted = _read_persisted(store["db"], cart_id)
    assert persisted["version"] == 1
    assert [l["sku"] for l in persisted["lines"]] == [SKU_A]

    # The read projection still lags: before the delay the write is invisible.
    projection = get_cart(c, base)
    assert projection["version"] == 0
    assert projection["lines"] == []

    # After the delay the projection catches up; the write was never lost.
    time.sleep(0.9)
    projection = get_cart(c, base)
    assert projection["version"] == 1
    assert [l["sku"] for l in projection["lines"]] == [SKU_A]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _cart_id_for_client(db_path: str, client: httpx.Client) -> str:
    conn = models.connect(db_path)
    try:
        row = conn.execute(
            "SELECT cart_id FROM sessions WHERE session_id = ?",
            (client.cookies["cp_session"],),
        ).fetchone()
        return row["cart_id"]
    finally:
        conn.close()


def _read_persisted(db_path: str, cart_id: str) -> dict:
    conn = models.connect(db_path)
    try:
        return models.read_persisted_cart(conn, cart_id)
    finally:
        conn.close()

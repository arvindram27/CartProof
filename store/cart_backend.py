"""Cart state backend seam (P40): state acquisition + provenance.

The Shopify platform change replaces "cart truth = a local SQLite file"
with "cart truth = whatever the active backend's acquire_state() fetched".
This module names that seam. It does not change how the mock store works,
and it does not pretend Shopify has a version counter it does not have:
capabilities().atomic_version_guard says which case the active backend is
in, and every mutation result says whether a version guard was enforced.

Interface follows the reviewed P39 design; the local side wraps exactly
what store/mcp_server.py (reviewed, pinned) and harness/snapshots.py do
today.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, TypedDict


class Capabilities(TypedDict):
    atomic_version_guard: bool
    cart_identity: str  # "session" | "remote-cart-id"


class Provenance(TypedDict, total=False):
    source: str       # "sqlite-snapshot" | "shopify-storefront-api"
    fetched_at: str   # UTC ISO-8601 of THIS acquisition
    db_copy: str      # local backend: snapshot file name (deleted after read)
    api_version: str  # shopify backend: Storefront API version
    cart_ref: str     # shopify backend: truncated sha256 of the cart ID


class CanonicalCartState(TypedDict):
    lines: Optional[dict[str, int]]  # {item_key: quantity}; None = unresolvable
    line_ids: list[dict[str, Any]]   # observational [{line_id, item_key, quantity}]
    version: Optional[int]           # None where no version concept exists
    resolution: str                  # backend-specific resolution detail
    provenance: Provenance


class MutationResult(TypedDict):
    ok: bool
    payload: dict[str, Any]          # backend response, capability-redacted
    version_enforcement: str         # "enforced" | "unavailable" | "required"
    expected_version_received: Optional[int]


class CartStateBackend(Protocol):
    """One seam for state acquisition, provenance and cart mutations."""

    def capabilities(self) -> Capabilities: ...

    def known_item_keys(self) -> set[str]: ...

    def acquire_state(self) -> CanonicalCartState: ...

    def read_catalog(self) -> dict[str, Any]: ...

    def add_line(self, sku: str, quantity: int,
                 expected_version: Optional[int]) -> MutationResult: ...

    def set_line_quantity(self, line_id: str, quantity: int,
                          expected_version: Optional[int]) -> MutationResult: ...

    def delete_line(self, line_id: str,
                    expected_version: Optional[int]) -> MutationResult: ...

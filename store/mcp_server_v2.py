"""Stdio MCP server v2 (P40): the six CartProof tools over a CartStateBackend.

ADAPTER CONTRACT (explicit; v2 is NOT a verbatim surface mirror of v1):
- cart_read returns CANONICAL state with provenance
  {lines, line_ids, version, resolution, provenance} - not the mock
  store's native HTTP response. v1 (store/mcp_server.py, pinned) remains
  the native-response surface and is untouched.
- Mutation tools return MutationResult
  {ok, payload, version_enforcement, expected_version_received}.
- expected_version is OPTIONAL in the tool schema: backends with a native
  guard (mock store) fail closed with "expected_version_required" when it
  is omitted; the Shopify backend records it but cannot enforce it
  (version_enforcement "unavailable"). No fake 0 is required or invented.
- cartproof_check adds capabilities, baseline_provenance and
  check_provenance to the v1 result shape, and its seed evidence is
  AGGREGATE per item_key (see run_check docstring).

Evidence limit, same discipline as v1: this server exposes RAW backend
semantics. It does NOT apply CartProof ownership checks or the full-cart
guard; those protect only the harness's own model arms.

OFFLINE BRIEF: --backend shopify is refused (exit 3). Live Shopify access
is a future Stage 1 smoke that would require separate approval; none is
granted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from mcp.server import MCPServer

from harness.evaluate import compare_lines
from store.backends import LocalSQLiteBackend
from store.cart_backend import CartStateBackend

MCP_TRANSPORT_ERROR = {"error": {"code": "mcp_transport",
                                 "message": "internal tool error"}}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_arg(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return "<non-scalar>"


class ServerState:
    """Frozen request + immutable baseline + evidence log, backend-driven.

    Baseline discipline unchanged from v1: one acquisition, captured after
    the backend session exists and before any tool call. No recapture
    fallback.
    """

    def __init__(self, backend: CartStateBackend, task_path: str,
                 log_dir: str):
        self.backend = backend
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "mcp-evidence-v2.jsonl"

        task = json.loads(Path(task_path).read_text(encoding="utf-8"))
        self.request_text = task["runtime"]["request_text"]
        known = backend.known_item_keys()
        self.request_items = []
        for item in task["expected_interpretation"].get("items") or []:
            sku = item.get("sku")
            qty = item.get("quantity")
            if sku not in known:
                raise RuntimeError("mcp_request_unknown_sku")
            if not isinstance(qty, int) or isinstance(qty, bool) \
                    or not 1 <= qty <= 99:
                raise RuntimeError("mcp_request_invalid_quantity")
            self.request_items.append({"sku": sku, "quantity": qty})
        self.seed_skus = []
        for seed in task["runtime"].get("seed_lines") or []:
            if seed.get("sku") not in known:
                raise RuntimeError("mcp_seed_unknown_sku")
            self.seed_skus.append(seed["sku"])

        # Central redaction: the backend strips its bearer capabilities
        # (Shopify cart ID / ?key=) from EVERYTHING logged, including
        # caller-controlled tool arguments.
        self._redact = getattr(backend, "redact", None) or (lambda o: o)
        self.baseline = backend.acquire_state()
        if self.baseline["lines"] is None:
            raise RuntimeError("mcp_baseline_unavailable")
        self.expected = dict(self.baseline["lines"])
        for item in self.request_items:
            self.expected[item["sku"]] = self.expected.get(item["sku"], 0) \
                + item["quantity"]
        self._log({"event": "startup", "request_text": self.request_text,
                   "request_items": self.request_items,
                   "baseline_lines": self.baseline["lines"],
                   "baseline_provenance": self.baseline["provenance"],
                   "capabilities": self.backend.capabilities()})

    def _log(self, entry: dict) -> None:
        entry = self._redact(dict(entry))
        entry.setdefault("ts", _now())
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def run_check(state: ServerState) -> dict:
    """Read-only check against the frozen request/baseline.

    Seed evidence is AGGREGATE per item_key, not per line ID: Shopify does
    not guarantee cart line IDs are stable across mutations, and a variant
    may legitimately occupy several lines, so line-ID matching would
    produce false PARTIALs. A seed "survived_unchanged" when the cart's
    aggregate quantity for its item_key equals the expected quantity
    (frozen baseline + request). Identity limit, stated honestly: a
    delete-then-readd of the same item_key at the same aggregate quantity
    is indistinguishable at this level - the same limit compare_lines has
    always had. Raw line IDs stay visible in baseline_line_ids /
    final_line_ids for manual inspection only.

    Any acquisition failure (unresolvable, truncated, malformed,
    unavailable) yields verdict UNKNOWN - never a guessed COMPLETE/PARTIAL.
    """
    try:
        check = state.backend.acquire_state()
    except Exception as exc:
        state._log({"event": "tool_error", "tool": "cartproof_check",
                    "error_code": "mcp_transport",
                    "error_type": type(exc).__name__})
        check = {"lines": None, "line_ids": [], "version": None,
                 "resolution": "error",
                 "provenance": {"source": "unknown", "fetched_at": _now()}}
    base_ref = state.baseline["provenance"].get("cart_ref")
    check_ref = check["provenance"].get("cart_ref")
    identity_mismatch = (base_ref is not None and check_ref is not None
                         and base_ref != check_ref)
    if check["lines"] is None or identity_mismatch:
        # No guessing: a failed acquisition - or a check taken against a
        # DIFFERENT remote cart than the baseline - is UNKNOWN.
        verdict = "UNKNOWN"
        mismatches = []
    else:
        mismatches = compare_lines(state.expected, check["lines"])
        verdict = "COMPLETE" if not mismatches else "PARTIAL"
    final_lines = check["lines"] or {}
    seed_evidence = []
    for sku in state.seed_skus:
        baseline_qty = (state.baseline["lines"] or {}).get(sku, 0)
        expected_qty = state.expected.get(sku, 0)
        actual_qty = final_lines.get(sku, 0)
        seed_evidence.append({
            "item_key": sku,
            "baseline_quantity": baseline_qty,
            "expected_quantity": expected_qty,
            "present": actual_qty > 0,
            "actual_quantity": actual_qty if check["lines"] is not None
                               else None,
            "survived_unchanged": (check["lines"] is not None
                                   and actual_qty == expected_qty),
        })
    if verdict == "COMPLETE" and not all(s["survived_unchanged"]
                                        for s in seed_evidence):
        verdict = "PARTIAL"
    caps = state.backend.capabilities()
    return {
        "verdict": verdict,
        "request_text": state.request_text,
        "request_items": state.request_items,
        "baseline_lines": state.baseline["lines"],
        "expected_lines": state.expected,
        "final_lines": check["lines"],
        "resolution": ("cart_identity_changed" if identity_mismatch
                       else check["resolution"]),
        "mismatches": mismatches,
        "seed_lines": seed_evidence,
        "seed_identity": "aggregate per item_key; line IDs observational "
                         "only (Shopify does not guarantee line-ID "
                         "stability across mutations)",
        "baseline_line_ids": state.baseline["line_ids"],
        "final_line_ids": check["line_ids"],
        "capabilities": caps,
        "baseline_provenance": state.baseline["provenance"],
        "check_provenance": check["provenance"],
        "evidence_limit": (
            "persisted SQLite snapshot only; no DOM/API coherence"
            if caps["atomic_version_guard"] else
            "remote state at fetch time; no concurrency guard active; "
            "time-of-check gap between agent action and fetch"),
    }


def build_server_v2(state: ServerState) -> MCPServer:
    backend = state.backend
    mcp = MCPServer("CartProof backend-seam store (v2)")

    def _result(tool: str, kwargs: dict, result: dict) -> dict:
        state._log({"event": "tool_call", "tool": tool,
                    "args": {k: _safe_arg(v) for k, v in kwargs.items()},
                    "result": result})
        return result

    def _read_state() -> dict:
        try:
            state_now = backend.acquire_state()
        except Exception as exc:
            state._log({"event": "tool_error", "tool": "cart_read",
                        "error_code": "mcp_transport",
                        "error_type": type(exc).__name__})
            return dict(MCP_TRANSPORT_ERROR)
        return {"lines": state_now["lines"],
                "line_ids": state_now["line_ids"],
                "version": state_now["version"],
                "resolution": state_now["resolution"],
                "provenance": state_now["provenance"]}

    @mcp.tool()
    def cart_read() -> dict:
        """Acquire current canonical cart state (with provenance)."""
        return _result("cart_read", {}, _read_state())

    @mcp.tool()
    def cart_add_line(sku: str, quantity: int,
                      expected_version: Optional[int] = None) -> dict:
        """Add a line. Backends with a native version guard (mock store)
        REQUIRE expected_version and fail closed without it; on Shopify no
        version concept exists and version_enforcement is "unavailable"."""
        result = backend.add_line(sku, quantity, expected_version)
        return _result("cart_add_line",
                       {"sku": sku, "quantity": quantity,
                        "expected_version": expected_version}, result)

    @mcp.tool()
    def cart_set_line_quantity(line_id: str, quantity: int,
                               expected_version: Optional[int] = None
                               ) -> dict:
        """Set a line quantity (guard semantics as cart_add_line)."""
        result = backend.set_line_quantity(line_id, quantity,
                                           expected_version)
        return _result("cart_set_line_quantity",
                       {"line_id": line_id, "quantity": quantity,
                        "expected_version": expected_version}, result)

    @mcp.tool()
    def cart_delete_line(line_id: str,
                         expected_version: Optional[int] = None) -> dict:
        """Delete a line (guard semantics as cart_add_line)."""
        result = backend.delete_line(line_id, expected_version)
        return _result("cart_delete_line",
                       {"line_id": line_id,
                        "expected_version": expected_version}, result)

    @mcp.tool()
    def catalog_read() -> dict:
        """Return the active backend's catalog data."""
        try:
            catalog = backend.read_catalog()
        except Exception as exc:
            state._log({"event": "tool_error", "tool": "catalog_read",
                        "error_code": "mcp_transport",
                        "error_type": type(exc).__name__})
            return dict(MCP_TRANSPORT_ERROR)
        return _result("catalog_read", {}, catalog)

    @mcp.tool()
    def cartproof_check() -> dict:
        """Read-only check against the frozen request/baseline. See
        run_check for the aggregate seed-identity semantics and the
        per-backend evidence limit. No writes, no corrections, no
        recapture."""
        result = run_check(state)
        return _result("cartproof_check", {}, result)

    return mcp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "shopify"],
                        required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--db-path")
    parser.add_argument("--task", required=True)
    parser.add_argument("--catalog")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--session-id")
    args = parser.parse_args()
    if args.backend == "shopify":
        # OFFLINE BRIEF GUARD: live Shopify access is a future Stage 1
        # smoke requiring separate approval (none granted). This CLI never
        # constructs the Shopify backend here (and ShopifyStorefrontBackend
        # itself requires an httpx.MockTransport in this build).
        print(json.dumps({"startup_error": "shopify_live_not_approved"}),
              file=sys.stderr)
        return 3
    if not (args.base_url and args.db_path and args.catalog):
        print(json.dumps({"startup_error": "missing_local_args"}),
              file=sys.stderr)
        return 2
    try:
        backend = LocalSQLiteBackend(args.base_url, args.db_path,
                                     args.catalog, args.session_id)
        state = ServerState(backend, args.task, args.log_dir)
    except Exception as exc:
        print(json.dumps({"startup_error": type(exc).__name__}),
              file=sys.stderr)
        return 2
    build_server_v2(state).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

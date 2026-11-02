"""Local stdio MCP server exposing the running CartProof mock store.

The MCP protocol in this SDK revision is stateless; all application state
(store session cookie, frozen request, immutable baseline) lives in this
server process for its lifetime.

This server exposes RAW store semantics (including native X-Cart-Version
guards). It does NOT apply CartProof ownership checks or the full-cart
guard; those protect only the harness's own model arms.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from mcp.server import MCPServer

from harness.evaluate import compare_lines
from harness.snapshots import backup_copy, read_cart

MCP_TRANSPORT_ERROR = {"error": {"code": "mcp_transport",
                                 "message": "internal tool error"}}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_arg(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return "<non-scalar>"


class StoreBridge:
    def __init__(self, base_url: str, db_path: str, task_path: str,
                 catalog_path: str, log_dir: str,
                 session_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.db_path = db_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "mcp-evidence.jsonl"
        self.client = httpx.Client(base_url=self.base_url, timeout=15.0,
                                   follow_redirects=False, trust_env=False)
        self.cookie = session_id
        headers = {"Cookie": f"cp_session={session_id}"} if session_id else {}
        # Order matters: establish/attach the store session first.
        response = self.client.get("/api/cart", headers=headers)
        response.raise_for_status()
        if not self.cookie:
            self.cookie = response.cookies.get("cp_session")
        if not self.cookie:
            raise RuntimeError("mcp_session_unavailable")
        # Use one explicit Cookie header on every request; never let the
        # httpx jar introduce a second session cookie.
        self.client.cookies.clear()

        self.catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self.skus = {}
        for product in self.catalog.get("products", []):
            for variant in product.get("variants", []):
                self.skus[variant["sku"]] = product["id"]

        task = json.loads(Path(task_path).read_text(encoding="utf-8"))
        self.request_text = task["runtime"]["request_text"]
        self.request_items = []
        for item in task["expected_interpretation"].get("items") or []:
            sku = item.get("sku")
            qty = item.get("quantity")
            if sku not in self.skus:
                raise RuntimeError("mcp_request_unknown_sku")
            if not isinstance(qty, int) or isinstance(qty, bool) or not 1 <= qty <= 99:
                raise RuntimeError("mcp_request_invalid_quantity")
            self.request_items.append({"sku": sku, "quantity": qty})
        for seed in task["runtime"].get("seed_lines") or []:
            if seed.get("sku") not in self.skus:
                raise RuntimeError("mcp_seed_unknown_sku")

        # Immutable baseline: one snapshot of the session cart, captured after
        # the session exists and before any tool call. No recapture fallback.
        self.baseline_db = self.log_dir / "baseline.db"
        backup_copy(self.db_path, str(self.baseline_db))
        self.baseline = read_cart(str(self.baseline_db), self.cookie)
        if self.baseline["resolution"] not in ("cookie", "sole_session") \
                or self.baseline["lines"] is None:
            raise RuntimeError("mcp_baseline_unavailable")
        self.baseline_ids = self._line_ids(str(self.baseline_db))
        self.expected = dict(self.baseline["lines"])
        for item in self.request_items:
            self.expected[item["sku"]] = self.expected.get(item["sku"], 0) \
                + item["quantity"]
        self._log({"event": "startup", "request_text": self.request_text,
                   "request_items": self.request_items,
                   "baseline_lines": self.baseline["lines"],
                   "baseline_ids": self.baseline_ids,
                   "baseline_ref": "baseline.db"})

    # ------------------------------------------------------------------ #
    def _line_ids(self, db_file: str) -> list[dict]:
        conn = sqlite3.connect(db_file)
        try:
            row = conn.execute(
                "SELECT cart_id FROM sessions WHERE session_id = ?",
                (self.cookie,)).fetchone()
            if row is None:
                return []
            rows = conn.execute(
                "SELECT line_id, sku, quantity FROM cart_lines "
                "WHERE cart_id = ? ORDER BY line_id", (row[0],)).fetchall()
            return [{"line_id": r[0], "sku": r[1], "quantity": r[2]}
                    for r in rows]
        finally:
            conn.close()

    def _log(self, entry: dict) -> None:
        entry = dict(entry)
        entry.setdefault("ts", _now())
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _store(self, method: str, path: str, **kwargs) -> dict:
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["Cookie"] = f"cp_session={self.cookie}"
        try:
            response = self.client.request(method, path, headers=headers, **kwargs)
        except Exception as exc:
            self._log({"event": "tool_error", "tool": path,
                       "error_code": "mcp_transport",
                       "error_type": type(exc).__name__})
            return dict(MCP_TRANSPORT_ERROR)
        try:
            payload = response.json()
        except Exception:
            self._log({"event": "tool_error", "tool": path,
                       "error_code": "mcp_transport",
                       "error_type": "invalid_json"})
            return dict(MCP_TRANSPORT_ERROR)
        if response.status_code >= 400:
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                return payload
            return dict(MCP_TRANSPORT_ERROR)
        return payload


def build_server(bridge: StoreBridge) -> MCPServer:
    mcp = MCPServer("CartProof local mock store")

    def _result(tool: str, kwargs: dict, result: dict) -> dict:
        bridge._log({"event": "tool_call", "tool": tool,
                     "args": {k: _safe_arg(v) for k, v in kwargs.items()},
                     "result": result})
        return result

    @mcp.tool()
    def cart_read() -> dict:
        """Read the current persisted cart (native store response)."""
        return _result("cart_read", {}, bridge._store("GET", "/api/cart"))

    @mcp.tool()
    def cart_add_line(sku: str, quantity: int, expected_version: int) -> dict:
        """Add a line with the required native X-Cart-Version guard."""
        result = bridge._store(
            "POST", "/api/cart/lines",
            json={"sku": sku, "quantity": quantity},
            headers={"X-Cart-Version": str(expected_version)})
        return _result("cart_add_line",
                       {"sku": sku, "quantity": quantity,
                        "expected_version": expected_version}, result)

    @mcp.tool()
    def cart_set_line_quantity(line_id: str, quantity: int,
                               expected_version: int) -> dict:
        """Set a line quantity with the native X-Cart-Version guard."""
        result = bridge._store(
            "PATCH", f"/api/cart/lines/{line_id}",
            json={"quantity": quantity},
            headers={"X-Cart-Version": str(expected_version)})
        return _result("cart_set_line_quantity",
                       {"line_id": line_id, "quantity": quantity,
                        "expected_version": expected_version}, result)

    @mcp.tool()
    def cart_delete_line(line_id: str, expected_version: int) -> dict:
        """Delete a line with the native X-Cart-Version guard."""
        result = bridge._store(
            "DELETE", f"/api/cart/lines/{line_id}",
            headers={"X-Cart-Version": str(expected_version)})
        return _result("cart_delete_line",
                       {"line_id": line_id,
                        "expected_version": expected_version}, result)

    @mcp.tool()
    def catalog_read() -> dict:
        """Return the store's catalog data (same source the store serves)."""
        return _result("catalog_read", {}, bridge.catalog)

    @mcp.tool()
    def cartproof_check() -> dict:
        """Read-only persisted-cart check against the frozen request/baseline.

        Evidence limit: persisted SQLite state only; the DOM/API coherence
        monitor stays harness-side. No writes, no corrections, no recapture.
        """
        check_db = bridge.log_dir / f"check-{uuid.uuid4().hex}.db"
        backup_copy(bridge.db_path, str(check_db))
        try:
            cart = read_cart(str(check_db), bridge.cookie)
            ids = bridge._line_ids(str(check_db))
        finally:
            try:
                check_db.unlink()
            except OSError:
                pass
        if cart["resolution"] not in ("cookie", "sole_session") \
                or cart["lines"] is None:
            verdict = "UNKNOWN"
            mismatches = []
        else:
            mismatches = compare_lines(bridge.expected, cart["lines"])
            verdict = "COMPLETE" if not mismatches else "PARTIAL"
        actual_ids = {line["line_id"]: line for line in ids}
        seed_evidence = []
        for seed in bridge.baseline_ids:
            found = actual_ids.get(seed["line_id"])
            seed_evidence.append({
                "line_id": seed["line_id"], "sku": seed["sku"],
                "baseline_quantity": seed["quantity"],
                "expected_quantity": bridge.expected.get(seed["sku"], 0),
                "present": found is not None,
                "actual_quantity": found["quantity"] if found else None,
                "survived_unchanged": (
                    found is not None
                    and found["sku"] == seed["sku"]
                    and found["quantity"] == bridge.expected.get(seed["sku"], 0)),
            })
        if verdict == "COMPLETE" and not all(s["survived_unchanged"]
                                            for s in seed_evidence):
            verdict = "PARTIAL"
        result = {
            "verdict": verdict,
            "request_text": bridge.request_text,
            "request_items": bridge.request_items,
            "baseline_lines": bridge.baseline["lines"],
            "expected_lines": bridge.expected,
            "final_lines": cart["lines"],
            "resolution": cart["resolution"],
            "mismatches": mismatches,
            "seed_lines": seed_evidence,
            "evidence_limit": "persisted SQLite state only; no DOM/API coherence",
        }
        return _result("cartproof_check", {}, result)

    return mcp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--session-id")
    args = parser.parse_args()
    try:
        bridge = StoreBridge(args.base_url, args.db_path, args.task,
                             args.catalog, args.log_dir, args.session_id)
    except Exception as exc:
        print(json.dumps({"startup_error": type(exc).__name__}),
              file=sys.stderr)
        return 2
    build_server(bridge).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Localhost-only mutation-path tests for the stdio MCP server (P37b).

Follow-up to tests/test_mcp_server.py: drives cart_set_line_quantity and
cart_delete_line through the real MCP SDK client and verifies that
cartproof_check reports the seeded line's loss. Helper copies are
intentionally duplicated so the reviewed P37 files stay byte-identical.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from store import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAVY = "BTL-HIK-500-NAVY"


def _seed(store):
    conn = models.connect(store["db"])
    try:
        cart_id, sid, _ = models.get_or_create_session(conn, None)
        models.add_line_guarded(conn, cart_id, 0, NAVY, 1, 2400)
    finally:
        conn.close()
    return sid


def _task(tmp_path, items, request_text, seed_lines):
    path = tmp_path / "mcp-task.json"
    path.write_text(json.dumps({
        "runtime": {"request_text": request_text, "seed_lines": seed_lines},
        "expected_interpretation": {"items": items,
                                    "budget_max_cents": None,
                                    "substitution_allowed": False},
    }), encoding="utf-8")
    return path


@pytest.fixture
def mcp_env(store, tmp_path):
    sid = _seed(store)
    task = _task(tmp_path,
                 [{"sku": NAVY, "quantity": 1}],
                 "Add 1 x BTL-HIK-500-NAVY.",
                 [{"sku": NAVY, "quantity": 1}])
    logdir = tmp_path / "mcp-log"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "store.mcp_server",
              "--base-url", store["base"],
              "--db-path", store["db"],
              "--task", str(task),
              "--catalog", str(PROJECT_ROOT / "store" / "seed_catalog.json"),
              "--log-dir", str(logdir),
              "--session-id", sid],
        env=env, cwd=str(PROJECT_ROOT))
    return {"params": params, "logdir": logdir, "sid": sid,
            "store": store, "task": task}


async def _call(client, name, args=None):
    result = await client.call_tool(name, args or {})
    if result.is_error:
        raise RuntimeError(str(result.content))
    content = result.structured_content
    if content is None:
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                return json.loads(text)
        return None
    if isinstance(content, dict) and set(content.keys()) == {"result"}:
        return content["result"]
    return content


async def _run_set_quantity_to_expected(params):
    async with Client(params) as client:
        cart = await _call(client, "cart_read")
        assert [line["sku"] for line in cart["lines"]] == [NAVY]
        line_id = cart["lines"][0]["line_id"]
        updated = await _call(client, "cart_set_line_quantity",
                              {"line_id": line_id, "quantity": 2,
                               "expected_version": cart["version"]})
        assert "error" not in updated
        after = await _call(client, "cart_read")
        assert after["lines"][0]["line_id"] == line_id
        assert after["lines"][0]["quantity"] == 2
        check = await _call(client, "cartproof_check")
        assert check["verdict"] == "COMPLETE"
        assert check["final_lines"] == {NAVY: 2}
        seed = check["seed_lines"][0]
        assert seed["present"] is True
        assert seed["actual_quantity"] == 2
        assert seed["survived_unchanged"] is True


def test_mcp_server_set_line_quantity_to_expected(mcp_env):
    asyncio.run(_run_set_quantity_to_expected(mcp_env["params"]))


async def _run_set_quantity_over_delivery(params):
    async with Client(params) as client:
        cart = await _call(client, "cart_read")
        line_id = cart["lines"][0]["line_id"]
        updated = await _call(client, "cart_set_line_quantity",
                              {"line_id": line_id, "quantity": 3,
                               "expected_version": cart["version"]})
        assert "error" not in updated
        check = await _call(client, "cartproof_check")
        assert check["verdict"] == "PARTIAL"
        assert check["final_lines"] == {NAVY: 3}
        seed = check["seed_lines"][0]
        assert seed["present"] is True
        assert seed["actual_quantity"] == 3
        assert seed["survived_unchanged"] is False
        assert any(m["sku"] == NAVY and m["expected_quantity"] == 2
                   and m["actual_quantity"] == 3
                   for m in check["mismatches"])


def test_mcp_server_set_line_quantity_over_delivery_measured(mcp_env):
    asyncio.run(_run_set_quantity_over_delivery(mcp_env["params"]))


async def _run_delete_seed_line(params):
    async with Client(params) as client:
        cart = await _call(client, "cart_read")
        line_id = cart["lines"][0]["line_id"]
        deleted = await _call(client, "cart_delete_line",
                              {"line_id": line_id,
                               "expected_version": cart["version"]})
        assert "error" not in deleted
        after = await _call(client, "cart_read")
        assert after["lines"] == []
        check = await _call(client, "cartproof_check")
        assert check["verdict"] == "PARTIAL"
        assert check["final_lines"] == {}
        seed = check["seed_lines"][0]
        assert seed["sku"] == NAVY
        assert seed["present"] is False
        assert seed["survived_unchanged"] is False
        assert any(m["sku"] == NAVY and m["expected_quantity"] == 2
                   and m["actual_quantity"] == 0
                   for m in check["mismatches"])


def test_mcp_server_delete_seed_line_detected(mcp_env):
    asyncio.run(_run_delete_seed_line(mcp_env["params"]))


async def _run_stale_patch_delete(params):
    async with Client(params) as client:
        cart = await _call(client, "cart_read")
        line_id = cart["lines"][0]["line_id"]
        stale_patch = await _call(client, "cart_set_line_quantity",
                                  {"line_id": line_id, "quantity": 5,
                                   "expected_version": 0})
        assert stale_patch["error"]["code"] == "version_conflict"
        stale_delete = await _call(client, "cart_delete_line",
                                   {"line_id": line_id,
                                    "expected_version": 0})
        assert stale_delete["error"]["code"] == "version_conflict"
        after = await _call(client, "cart_read")
        assert after == cart
        assert [line["line_id"] for line in after["lines"]] == [line_id]
        check = await _call(client, "cartproof_check")
        seed = check["seed_lines"][0]
        assert seed["present"] is True
        assert seed["actual_quantity"] == 1


def test_mcp_server_stale_patch_delete_no_change(mcp_env):
    asyncio.run(_run_stale_patch_delete(mcp_env["params"]))

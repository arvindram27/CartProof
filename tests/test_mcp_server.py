"""Localhost-only tests for the stdio MCP server wrapper (slice 1)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest
from mcp import Client, StdioServerParameters

from store import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAVY = "BTL-HIK-500-NAVY"
STEEL = "BTL-HIK-750-STEEL"


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


async def _list(client):
    return await client.list_tools()


async def _run_interop(params, store):
    async with Client(params) as client:
        tools = await _list(client)
        names = {tool.name for tool in tools.tools}
        add = next(t for t in tools.tools if t.name == "cart_add_line")
        required = set(add.input_schema.get("required", []))
        assert {"sku", "quantity", "expected_version"} <= required
        assert "cartproof_check" in names and "catalog_read" in names

        cart = await _call(client, "cart_read")
        assert [line["sku"] for line in cart["lines"]] == [NAVY]

        added = await _call(client, "cart_add_line",
                            {"sku": NAVY, "quantity": 1,
                             "expected_version": cart["version"]})
        assert added.get("version") == 2

        check = await _call(client, "cartproof_check")
        assert check["verdict"] == "COMPLETE"
        assert check["final_lines"] == {NAVY: 2}
        assert check["seed_lines"][0]["survived_unchanged"] is True

        catalog = await _call(client, "catalog_read")
        assert any(p.get("id") == "bottle-hiker"
                   for p in catalog.get("products", []))


def test_mcp_server_interop_add_and_check(mcp_env):
    asyncio.run(_run_interop(mcp_env["params"], mcp_env["store"]))


async def _run_errors(params, store):
    async with Client(params) as client:
        cart = await _call(client, "cart_read")
        unknown = await _call(client, "cart_add_line",
                              {"sku": "NOPE-404", "quantity": 1,
                               "expected_version": cart["version"]})
        assert unknown["error"]["code"] == "unknown_sku"
        stale = await _call(client, "cart_add_line",
                            {"sku": NAVY, "quantity": 1,
                             "expected_version": 0})
        assert stale["error"]["code"] == "version_conflict"
        result = await client.call_tool("cart_add_line",
                                        {"sku": NAVY, "quantity": 1})
        assert result.is_error is True


def test_mcp_server_native_guards_and_schema_required(mcp_env):
    asyncio.run(_run_errors(mcp_env["params"], mcp_env["store"]))


async def _run_silent_drop(params, store):
    async with Client(params) as client:
        cart = await _call(client, "cart_read")
        httpx.post(f"{store['control']}/_control/fault",
                   json={"name": "silent-drop", "params": {}}).raise_for_status()
        try:
            added = await _call(client, "cart_add_line",
                                {"sku": NAVY, "quantity": 1,
                                 "expected_version": cart["version"]})
            assert "error" not in added
            check = await _call(client, "cartproof_check")
            assert check["verdict"] == "PARTIAL"
            assert any(m["sku"] == NAVY for m in check["mismatches"])
        finally:
            httpx.post(f"{store['control']}/_control/fault/clear",
                       json={"name": "silent-drop"}).raise_for_status()


def test_mcp_server_silent_drop_measured_not_fixed(mcp_env):
    asyncio.run(_run_silent_drop(mcp_env["params"], mcp_env["store"]))


async def _run_baseline_log(params, logdir):
    async with Client(params) as client:
        first = await _call(client, "cartproof_check")
        cart = await _call(client, "cart_read")
        await _call(client, "cart_add_line",
                    {"sku": NAVY, "quantity": 1,
                     "expected_version": cart["version"]})
        second = await _call(client, "cartproof_check")
        assert first["baseline_lines"] == second["baseline_lines"] == {NAVY: 1}
        assert first["expected_lines"] == second["expected_lines"] == {NAVY: 2}
        assert second["verdict"] == "COMPLETE"
    text = (Path(logdir) / "mcp-evidence.jsonl").read_text(encoding="utf-8")
    assert '"event": "startup"' in text
    assert '"event": "tool_call"' in text
    lowered = text.lower()
    for forbidden in ("set-cookie", "cp_session", "authorization",
                      "traceback", "c:\\\\"):
        assert forbidden not in lowered


def test_mcp_server_baseline_immutable_and_log_sanitized(mcp_env):
    asyncio.run(_run_baseline_log(mcp_env["params"], mcp_env["logdir"]))


def test_mcp_server_rejects_unknown_request_sku(store, tmp_path):
    sid = _seed(store)
    bad_task = _task(tmp_path, [{"sku": "NOPE-404", "quantity": 1}],
                     "Add 1 x NOPE-404.", [])
    logdir = tmp_path / "bad-log"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "store.mcp_server", "--base-url", store["base"],
              "--db-path", store["db"], "--task", str(bad_task),
              "--catalog", str(PROJECT_ROOT / "store" / "seed_catalog.json"),
              "--log-dir", str(logdir), "--session-id", sid],
        env={"PATH": os.environ.get("PATH", ""),
             "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
             "PYTHONPATH": str(PROJECT_ROOT)},
        cwd=str(PROJECT_ROOT))
    with pytest.raises(Exception):
        async def _start():
            async with Client(params):
                pass
        asyncio.run(_start())

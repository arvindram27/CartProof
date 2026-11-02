"""Localhost-only interop tests for the v2 (backend-seam) MCP server.

Same stdio/SDK pattern as the reviewed tests/test_mcp_server.py (v1):
real MCP client over a server subprocess, loopback mock store, minimal
environment. Helpers intentionally duplicated so reviewed files stay
byte-identical.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
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
    path = tmp_path / "mcp-task-v2.json"
    path.write_text(json.dumps({
        "runtime": {"request_text": request_text, "seed_lines": seed_lines},
        "expected_interpretation": {"items": items,
                                    "budget_max_cents": None,
                                    "substitution_allowed": False},
    }), encoding="utf-8")
    return path


@pytest.fixture
def mcp_env_v2(store, tmp_path):
    sid = _seed(store)
    task = _task(tmp_path,
                 [{"sku": NAVY, "quantity": 1}],
                 "Add 1 x BTL-HIK-500-NAVY.",
                 [{"sku": NAVY, "quantity": 1}])
    logdir = tmp_path / "mcp-log-v2"
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
        args=["-m", "store.mcp_server_v2",
              "--backend", "local",
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


async def _run_surface(params):
    async with Client(params) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools.tools}
        assert {"cart_read", "cart_add_line", "cart_set_line_quantity",
                "cart_delete_line", "catalog_read",
                "cartproof_check"} == names
        add = next(t for t in tools.tools if t.name == "cart_add_line")
        required = set(add.input_schema.get("required", []))
        # item identity is required; expected_version is OPTIONAL in the
        # schema - backends with a native guard fail closed without it
        assert {"sku", "quantity"} <= required
        assert "expected_version" not in required
        read = await _call(client, "cart_read")
        assert read["provenance"]["source"] == "sqlite-snapshot"
        assert read["lines"] == {NAVY: 1}


def test_v2_tool_surface_and_provenance(mcp_env_v2):
    asyncio.run(_run_surface(mcp_env_v2["params"]))


async def _run_add_check(params):
    async with Client(params) as client:
        before = await _call(client, "cart_read")
        added = await _call(client, "cart_add_line",
                            {"sku": NAVY, "quantity": 1,
                             "expected_version": before["version"]})
        assert added["ok"] is True
        assert added["version_enforcement"] == "enforced"
        check = await _call(client, "cartproof_check")
        assert check["verdict"] == "COMPLETE", check
        assert check["capabilities"]["atomic_version_guard"] is True
        assert check["baseline_provenance"]["source"] == "sqlite-snapshot"
        assert check["check_provenance"]["source"] == "sqlite-snapshot"
        assert check["seed_lines"][0]["survived_unchanged"] is True


def test_v2_add_line_then_check_complete(mcp_env_v2):
    asyncio.run(_run_add_check(mcp_env_v2["params"]))


async def _run_stale(params):
    async with Client(params) as client:
        stale = await _call(client, "cart_add_line",
                            {"sku": NAVY, "quantity": 1,
                             "expected_version": 999})
        assert stale["ok"] is False
        assert stale["payload"]["error"]["code"] == "version_conflict"
        read = await _call(client, "cart_read")
        assert read["lines"] == {NAVY: 1}


def test_v2_stale_version_rejected_no_change(mcp_env_v2):
    asyncio.run(_run_stale(mcp_env_v2["params"]))


async def _run_missing_version(params):
    async with Client(params) as client:
        refused = await _call(client, "cart_add_line",
                              {"sku": NAVY, "quantity": 1})
        assert refused["ok"] is False
        assert refused["payload"]["error"]["code"] \
            == "expected_version_required"
        assert refused["version_enforcement"] == "required"
        read = await _call(client, "cart_read")
        assert read["lines"] == {NAVY: 1}


def test_v2_local_guard_requires_expected_version(mcp_env_v2):
    asyncio.run(_run_missing_version(mcp_env_v2["params"]))


async def _run_delete_seed(params):
    async with Client(params) as client:
        read = await _call(client, "cart_read")
        line_id = read["line_ids"][0]["line_id"]
        deleted = await _call(client, "cart_delete_line",
                              {"line_id": line_id,
                               "expected_version": read["version"]})
        assert deleted["ok"] is True
        check = await _call(client, "cartproof_check")
        assert check["verdict"] == "PARTIAL", check
        assert check["seed_lines"][0]["present"] is False
        assert check["seed_lines"][0]["survived_unchanged"] is False


def test_v2_delete_seed_line_detected(mcp_env_v2):
    asyncio.run(_run_delete_seed(mcp_env_v2["params"]))


def test_v2_cli_refuses_shopify_backend(tmp_path):
    """The offline guard: --backend shopify never starts a live path."""
    task = _task(tmp_path, [], "No additions expected.", [])
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "store.mcp_server_v2",
         "--backend", "shopify",
         "--task", str(task),
         "--log-dir", str(tmp_path / "log")],
        capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT))
    assert proc.returncode == 3
    assert "shopify_live_not_approved" in proc.stderr

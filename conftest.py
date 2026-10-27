"""Shared pytest fixtures: a fresh store + control pair on ephemeral ports."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from store.app import create_app, create_control_app  # noqa: E402  (tests may import store)
from harness.run_task import boot_server, stop_server  # noqa: E402


@pytest.fixture
def store(tmp_path):
    """Boot store + control on ephemeral 127.0.0.1 ports with a temp DB."""
    db_path = str(tmp_path / "store.db")
    catalog_path = str(PROJECT_ROOT / "store" / "seed_catalog.json")

    store_app = create_app(db_path, catalog_path)
    control_app = create_control_app(db_path, catalog_path)

    store_srv, store_thread, store_port = boot_server(store_app)
    control_srv, control_thread, control_port = boot_server(control_app)

    info = {
        "base": f"http://127.0.0.1:{store_port}",
        "control": f"http://127.0.0.1:{control_port}",
        "db": db_path,
        "catalog": catalog_path,
    }
    yield info

    stop_server(store_srv, store_thread)
    stop_server(control_srv, control_thread)


@pytest.fixture
def browser_session(store):
    """A headless Playwright browser session pointed at the store."""
    from adapter.actions import BrowserSession

    session = BrowserSession(store["base"], headed=False)
    yield session
    session.close()

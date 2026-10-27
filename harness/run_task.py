"""Run a scripted cart task end to end against a freshly booted store.

Owns the whole run: boots the store and its control endpoint on ephemeral
127.0.0.1 ports, applies any requested fault through the control endpoint after
boot and before the actor starts, runs the actor, prints the evidence log to
stdout, and shuts everything down.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from store.app import create_app, create_control_app  # noqa: E402  (harness may import store)


def boot_server(app) -> tuple[uvicorn.Server, threading.Thread, int]:
    """Run ``app`` on an ephemeral 127.0.0.1 port in a background thread."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    def _target():
        server.run(sockets=[sock])

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    return server, thread, port


def stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=5)


def _parse_fault_params(pairs: list[str] | None, delay_ms: int | None) -> dict:
    params: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param must be key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        params[key] = _coerce(value)
    if delay_ms is not None:
        params["delay_ms"] = delay_ms
    return params


def _coerce(value: str):
    if value.isdigit():
        return int(value)
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a scripted CartProof cart task")
    parser.add_argument("--task", required=True, help="path to a task spec JSON")
    parser.add_argument("--fault", help="fault name to apply after boot, before the actor")
    parser.add_argument("--param", action="append", help="fault param as key=value (repeatable)")
    parser.add_argument("--delay-ms", type=int, help="shorthand for slow-consistency delay_ms")
    parser.add_argument("--headed", action="store_true", help="run the browser headed")
    parser.add_argument("--db", help="path to the SQLite file (default: var/cartproof.db)")
    args = parser.parse_args(argv)

    task_path = Path(args.task).resolve()
    if not task_path.exists():
        raise SystemExit(f"task file not found: {task_path}")
    spec = json.loads(task_path.read_text(encoding="utf-8"))

    var_dir = PROJECT_ROOT / "var"
    var_dir.mkdir(exist_ok=True)
    db_path = str(Path(args.db).resolve() if args.db else var_dir / "cartproof.db")
    catalog_path = str(PROJECT_ROOT / "store" / "seed_catalog.json")

    # Fault selection: CLI --fault wins over a fault field in the task JSON.
    fault_name = args.fault or None
    fault_params: dict = {}
    if args.fault:
        fault_params = _parse_fault_params(args.param, args.delay_ms)
    elif "fault" in spec:
        f = spec["fault"]
        if isinstance(f, str):
            fault_name = f
        elif isinstance(f, dict):
            fault_name = f.get("name")
            fault_params = dict(f.get("params") or {})
        if args.param or args.delay_ms is not None:
            fault_params.update(_parse_fault_params(args.param, args.delay_ms))

    store_app = create_app(db_path, catalog_path)
    control_app = create_control_app(db_path, catalog_path)

    store_srv, store_thread, store_port = boot_server(store_app)
    control_srv, control_thread, control_port = boot_server(control_app)
    store_url = f"http://127.0.0.1:{store_port}"
    control_url = f"http://127.0.0.1:{control_port}"

    try:
        # Fresh boot: reset then apply the requested fault, all via the control endpoint.
        httpx.post(f"{control_url}/_control/reset", timeout=10).raise_for_status()
        if fault_name:
            resp = httpx.post(
                f"{control_url}/_control/fault",
                json={"name": fault_name, "params": fault_params},
                timeout=10,
            )
            resp.raise_for_status()

        from adapter.actions import BrowserSession
        from agents.scripted_actor import run_actor

        artifacts = var_dir / "screenshots"
        artifacts.mkdir(exist_ok=True)
        shot_path = str(artifacts / f"task-{uuid.uuid4().hex}.png")

        session = BrowserSession(store_url, headed=args.headed)
        try:
            log = run_actor(session, spec, screenshot_path=shot_path)
        finally:
            session.close()

        sys.stderr.write(f"store: {store_url}  control: {control_url}\n")
        if fault_name:
            sys.stderr.write(f"fault: {fault_name} {fault_params or ''}\n".rstrip() + "\n")
        _print_agreement(log)
        print(json.dumps(log, indent=2))
        return 0
    finally:
        stop_server(store_srv, store_thread)
        stop_server(control_srv, control_thread)


def _print_agreement(log: list[dict]) -> None:
    dom = next((e["payload"] for e in log if e["source"] == "dom_reread"), None)
    api = next((e["payload"] for e in log if e["source"] == "api_reread"), None)
    if dom is None or api is None:
        return

    def sig(lines):
        return sorted((l["sku"], l["quantity"]) for l in lines)

    dom_lines = dom.get("lines") or []
    api_lines = api.get("lines") or []
    agree = sig(dom_lines) == sig(api_lines)
    sys.stderr.write(
        f"DOM lines ({len(dom_lines)}) vs API lines ({len(api_lines)}): "
        f"{'AGREE' if agree else 'DISAGREE'}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a scripted cart task end to end against a freshly booted store.

Owns the whole run: boots the store and its control endpoint on ephemeral
127.0.0.1 ports, seeds the customer-owned cart, takes a baseline snapshot,
builds the intent ledger, applies any requested fault, runs the actor with the
per-action verification hook, and prints the evidence log and the verification
report.

Exit codes (verification mode): 0 = COMPLETE, 1 = PARTIAL, 2 = hard error.
A task with no ``constraints`` key keeps phase-1 behaviour (evidence log only).
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


def _declared_additions(spec: dict) -> dict:
    """The agent's own add amounts, from the actor fields only (never constraints)."""
    d: dict = {}
    if "items" in spec:
        for it in spec["items"]:
            sku = it.get("target_sku")
            qty = int(it.get("quantity", 1))
            if not sku or qty <= 0:
                raise ValueError(f"invalid item entry: {it!r}")
            d[sku] = d.get(sku, 0) + qty
        return d
    sku = spec.get("target_sku")
    qty = int(spec.get("quantity", 1))
    if not sku or qty <= 0:
        raise ValueError("task spec requires 'target_sku' and a positive 'quantity'")
    return {sku: qty}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a scripted CartProof cart task")
    parser.add_argument("--task", required=True, help="path to a task spec JSON")
    parser.add_argument("--fault", help="fault name to apply after baseline, before the actor")
    parser.add_argument("--param", action="append", help="fault param as key=value (repeatable)")
    parser.add_argument("--delay-ms", type=int, help="shorthand for slow-consistency delay_ms")
    parser.add_argument("--settle-ms", type=int, default=2000, help="settlement deadline (ms)")
    parser.add_argument("--recover", action="store_true", help="opt-in bounded recovery")
    parser.add_argument("--headed", action="store_true", help="run the browser headed")
    parser.add_argument("--db", help="path to the SQLite file (default: var/cartproof.db)")
    args = parser.parse_args(argv)

    task_path = Path(args.task).resolve()
    if not task_path.exists():
        print(f"error: task file not found: {task_path}", file=sys.stderr)
        return 2
    try:
        spec = json.loads(task_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"error: cannot parse task spec: {exc}", file=sys.stderr)
        return 2

    verification = "constraints" in spec
    if args.recover and not verification:
        print("error: --recover requires a task with a 'constraints' key", file=sys.stderr)
        return 2

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
        # Fresh boot, no fault yet.
        httpx.post(f"{control_url}/_control/reset", timeout=10).raise_for_status()

        from adapter.actions import BrowserSession
        from adapter.observations import read_api_cart, read_dom_cart
        from agents.scripted_actor import run_actor
        from core.ledger import ConstraintError, Ledger
        from core.recovery import (RecoveryController, attach_raw_pair_recorder,
                                   establish_recovery_baseline, render_recovery_report)
        from core.verify import Verifier, render_report

        artifacts = var_dir / "screenshots"
        artifacts.mkdir(exist_ok=True)
        shot_path = str(artifacts / f"task-{uuid.uuid4().hex}.png")

        session = BrowserSession(store_url, headed=args.headed)
        try:
            verifier = None
            ledger = None
            declared = None
            controller = None

            if verification:
                try:
                    # 3. Seed the customer-owned cart through the shared session.
                    for sl in spec.get("seed_lines") or []:
                        sku = sl.get("sku")
                        qty = int(sl.get("quantity", 1))
                        if not sku or qty <= 0:
                            raise ValueError(f"invalid seed line: {sl!r}")
                        session.seed_line(sku, qty)

                    def reader(timeout_ms):
                        # DOM and API share one settle budget: recompute remaining
                        # after the DOM read and give the API read only what is left.
                        start = time.monotonic()
                        dom = read_dom_cart(session, timeout_ms=timeout_ms)
                        dom_payload = dict(dom.payload)
                        dom_payload["timestamp"] = dom.timestamp
                        remaining = timeout_ms - int((time.monotonic() - start) * 1000)
                        if remaining <= 0:
                            api_payload = {"read_timeout": True, "timestamp": dom.timestamp}
                        else:
                            api = read_api_cart(session, timeout_ms=remaining)
                            api_payload = dict(api.payload)
                            api_payload["timestamp"] = api.timestamp
                        return dom_payload, api_payload

                    # 4. Baseline snapshot (same session; no fault active yet).
                    # Recovery installs its raw-pair recorder BEFORE the first
                    # baseline read and accepts the baseline only through the
                    # full-state three-pair proof; verification-only keeps the
                    # existing aggregate baseline unchanged.
                    verifier = Verifier(reader, settle_ms=args.settle_ms)
                    raw_pairs = None
                    if args.recover:
                        raw_pairs = {}
                        attach_raw_pair_recorder(verifier, raw_pairs)
                        establish_recovery_baseline(verifier)
                    else:
                        verifier.take_baseline()

                    # 5. Build the ledger (deterministic parse).
                    ledger = Ledger.build(spec["constraints"])
                    declared = _declared_additions(spec)
                    if args.recover:
                        controller = RecoveryController(
                            verifier, session, ledger, declared, verifier.baseline,
                            raw_pairs=raw_pairs)
                except (ConstraintError, ValueError, RuntimeError) as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2

            # 6. Apply the requested fault (after seeding + baseline).
            if fault_name:
                httpx.post(
                    f"{control_url}/_control/fault",
                    json={"name": fault_name, "params": fault_params},
                    timeout=10,
                ).raise_for_status()

            # 7. Run the actor with the per-action verification/recovery hook.
            if controller is not None:
                def hook(event):
                    kind = event.get("kind")
                    if kind == "before_add":
                        return controller.before_add(event)
                    if kind == "pre_add":
                        return controller.pre_add(event)
                    if kind == "after_add":
                        return controller.after_add(event)
                    return None
            elif verifier is not None:
                def hook(event):
                    if event.get("kind") == "after_add":
                        verifier.on_action(event["add"]["payload"])
                    return None
            else:
                hook = None
            log = run_actor(session, spec, screenshot_path=shot_path, hook=hook)

            sys.stderr.write(f"store: {store_url}  control: {control_url}\n")
            if fault_name:
                sys.stderr.write(f"fault: {fault_name} {fault_params or ''}\n".rstrip() + "\n")
            _print_agreement(log)
            print(json.dumps(log, indent=2))

            # 8. Completion tier + verdict report.
            if controller is not None:
                report, verdict = controller.complete()
                print(render_recovery_report(report))
                return 0 if verdict == "COMPLETE" else 1
            if verifier is not None:
                report = verifier.finish(ledger, declared)
                print(render_report(report))
                return 0 if report["verdict"] == "COMPLETE" else 1

            print("note: verification was not requested (task has no 'constraints' key)")
            return 0
        finally:
            session.close()
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

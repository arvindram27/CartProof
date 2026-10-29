"""Brief 4 comparison runner and per-run worker.

Three arms over the same tasks, same adapter, same privileges:

- Arm A: declaration only (trust the agent's action responses).
- Arm B: verification, no recovery.
- Arm C: full CartProof (verification + bounded recovery).

Orchestrator: ``python -m harness.compare --suite eval/dev/suite.json``

Worker (spawned per run): ``python -m harness.compare --run-one ...``

The orchestrator reads each eval task's authored expectation and enriches each
worker-produced run record with the measured outcome and stage/recovery metrics.
The worker sees only the runtime spec + fault, never the expectation.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from harness.run_task import boot_server, stop_server  # noqa: E402
from harness.snapshots import (  # noqa: E402
    ObservedSession, Observer, backup_copy, read_cart)
from harness import evaluate, metrics  # noqa: E402


def _items(runtime: dict) -> list[dict]:
    if "items" in runtime:
        return [{"product_query": it["product_query"], "target_sku": it["target_sku"],
                 "quantity": int(it.get("quantity", 1))} for it in runtime["items"]]
    return [{"product_query": runtime["product_query"],
             "target_sku": runtime["target_sku"],
             "quantity": int(runtime.get("quantity", 1))}]


def _declared_additions(runtime: dict) -> dict:
    d: dict[str, int] = {}
    for it in _items(runtime):
        d[it["target_sku"]] = d.get(it["target_sku"], 0) + it["quantity"]
    return d


def _arm_a_declared(log: list[dict]) -> str:
    adds = [e for e in log if e.get("source") == "action_confirmation"
            and (e.get("payload") or {}).get("action") == "add_to_cart"]
    if adds and all((a.get("payload") or {}).get("response_status") == 200
                    for a in adds):
        return "COMPLETE"
    return "PARTIAL"


def build_template_db(catalog_path: str, template_path: str) -> None:
    from store.app import create_control_app

    control_app = create_control_app(template_path, catalog_path)
    cs, ct, cp = boot_server(control_app)
    try:
        httpx.post(f"http://127.0.0.1:{cp}/_control/reset",
                   timeout=10).raise_for_status()
    finally:
        stop_server(cs, ct)
    conn = sqlite3.connect(template_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def execute_run(runtime: dict, fault: dict, arm: str, db_path: str, out_dir,
                settle_ms: int = 2000, headed: bool = False) -> dict:
    """Run one arm end to end and return its raw run record (no expectation)."""
    from adapter.actions import BrowserSession
    from adapter.observations import read_api_cart, read_dom_cart
    from agents.scripted_actor import run_actor
    from core.ledger import Ledger
    from core.verify import Verifier

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = str(PROJECT_ROOT / "store" / "seed_catalog.json")

    from store.app import create_app, create_control_app

    run: dict = {
        "arm": arm, "declared_outcome": None, "execution_status": "normal",
        "seed_lines": runtime.get("seed_lines") or [], "items": _items(runtime),
        "snapshots": [], "mutations": [],
        "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
        "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0,
                              "delete_line": 0},
        "stop_reasons": [], "error": None, "final_cookie": None,
        "final_persisted": None,
    }

    store_app = create_app(db_path, catalog)
    control_app = create_control_app(db_path, catalog)
    ss, st, sp = boot_server(store_app)
    cs, ct, cp = boot_server(control_app)
    store_url = f"http://127.0.0.1:{sp}"
    control_url = f"http://127.0.0.1:{cp}"

    session = None
    observer = None
    try:
        session = BrowserSession(store_url, headed=headed)
        observer = Observer(db_path, out_dir, session)
        wrapped = ObservedSession(session, observer)

        # Seeding (identical across arms, before any fault).
        with observer.context("setup"):
            for sl in runtime.get("seed_lines") or []:
                wrapped.seed_line(sl["sku"], sl["quantity"])

        verifier = None
        ledger = None
        declared = None
        controller = None

        if arm in ("B", "C"):
            import time as _time

            def reader(timeout_ms):
                start = _time.monotonic()
                dom = read_dom_cart(session, timeout_ms=timeout_ms)
                dom_payload = dict(dom.payload)
                dom_payload["timestamp"] = dom.timestamp
                remaining = timeout_ms - int((_time.monotonic() - start) * 1000)
                if remaining <= 0:
                    api_payload = {"read_timeout": True, "timestamp": dom.timestamp}
                else:
                    api = read_api_cart(session, timeout_ms=remaining)
                    api_payload = dict(api.payload)
                    api_payload["timestamp"] = api.timestamp
                return dom_payload, api_payload

            verifier = Verifier(reader, settle_ms=settle_ms)
            ledger = Ledger.build(runtime["constraints"])
            declared = _declared_additions(runtime)
            if arm == "C":
                from core.recovery import (
                    RecoveryController, attach_raw_pair_recorder,
                    establish_recovery_baseline)
                raw_pairs = {}
                attach_raw_pair_recorder(verifier, raw_pairs)
                establish_recovery_baseline(verifier)
                controller = RecoveryController(verifier, wrapped, ledger,
                                                declared, verifier.baseline,
                                                raw_pairs=raw_pairs)
            else:
                verifier.take_baseline()

        # Apply the fault (after seeding/baseline, before the actor).
        if fault and fault.get("name"):
            httpx.post(
                f"{control_url}/_control/fault",
                json={"name": fault["name"], "params": fault.get("params") or {}},
                timeout=10,
            ).raise_for_status()

        def hook(event):
            kind = event.get("kind")
            observer.record_hook_call(kind)
            if kind in ("before_add", "after_add"):
                observer.hook_entry(kind, event)
            if arm == "C" and controller is not None:
                if kind == "before_add":
                    return controller.before_add(event)
                if kind == "pre_add":
                    return controller.pre_add(event)
                if kind == "after_add":
                    with observer.context("after_add_hook"):
                        return controller.after_add(event)
                return None
            if arm == "B" and verifier is not None:
                if kind == "after_add":
                    verifier.on_action(event["add"]["payload"])
                return None
            return None

        log = run_actor(wrapped, runtime, hook=hook)

        if arm == "C" and controller is not None:
            _report, verdict = controller.complete()
            run["declared_outcome"] = verdict
            run["stop_reasons"] = ([controller.stop_reason]
                                   if controller.stop_reason else [])
        elif arm == "B" and verifier is not None:
            report = verifier.finish(ledger, declared)
            run["declared_outcome"] = report["verdict"]
            reasons = [a.get("detail") or a.get("kind")
                       for a in report.get("anomalies", [])]
            for c in report.get("constraints", []):
                if c.get("state") != "confirmed":
                    reasons.append(c.get("reason"))
            run["stop_reasons"] = reasons
        elif arm == "A":
            run["declared_outcome"] = _arm_a_declared(log)

        run["log_add_statuses"] = [
            (e.get("payload") or {}).get("response_status")
            for e in log
            if e.get("source") == "action_confirmation"
            and (e.get("payload") or {}).get("action") == "add_to_cart"]
    except Exception:
        run["execution_status"] = "crash"
        run["error"] = traceback.format_exc()
    finally:
        # Clean shutdown first, then capture the final persisted DB.
        stop_server(ss, st)
        stop_server(cs, ct)
        if observer is not None and session is not None:
            try:
                cookie = observer._cookie_session_id()
                final_db = out_dir / "final.db"
                backup_copy(db_path, str(final_db))
                cart = read_cart(str(final_db), cookie)
                run["final_persisted"] = {
                    "lines": cart["lines"], "resolution": cart["resolution"],
                    "version": cart["version"], "total_cents": cart["total_cents"],
                }
                run["final_cookie"] = cookie
            except Exception as exc:
                run["final_persisted"] = {"lines": None, "resolution": "error",
                                          "error": repr(exc)}
                run["final_cookie"] = None
        if session is not None:
            session.close()
        if observer is not None:
            run["snapshots"] = observer.snapshots
            run["mutations"] = observer.mutations
            run["hook_counts"] = observer.hook_calls
            run["adapter_mutations"] = observer.summary()["adapter_mutations"]
            run["capture_ms_total"] = observer.summary()["capture_ms_total"]

    return run


def run_one_cli(argv) -> int:
    p = argparse.ArgumentParser(description="Run one comparison arm")
    p.add_argument("--spec", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--settle-ms", type=int, default=2000)
    args = p.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    runtime = spec["runtime"]
    fault = spec.get("fault") or {"name": None, "params": {}}
    try:
        run = execute_run(runtime, fault, args.arm, args.db, args.out,
                          settle_ms=args.settle_ms)
    except Exception:
        run = {
            "arm": args.arm, "declared_outcome": None,
            "execution_status": "crash", "error": traceback.format_exc(),
            "seed_lines": runtime.get("seed_lines") or [], "items": _items(runtime),
            "snapshots": [], "mutations": [],
            "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
            "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0,
                                  "delete_line": 0},
            "stop_reasons": [], "final_cookie": None, "final_persisted": None,
        }
    (Path(args.out) / "run.json").write_text(
        json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"arm": args.arm,
                      "execution_status": run.get("execution_status"),
                      "declared_outcome": run.get("declared_outcome")}))
    return 0


def _base_run(task: dict, arm: str, status: str, error: str | None = None) -> dict:
    runtime = task["runtime"]
    return {
        "task_id": task["task_id"], "arm": arm,
        "fault_class": task.get("fault_class") or "clean",
        "fault_schedule": task.get("fault"),
        "execution_status": status,
        "declared_outcome": None, "measured_outcome": None,
        "seed_lines": runtime.get("seed_lines") or [], "items": _items(runtime),
        "snapshots": [], "mutations": [],
        "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
        "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0,
                              "delete_line": 0},
        "stop_reasons": [], "error": error, "final_cookie": None,
        "final_persisted": None,
    }


def _spawn_run(task: dict, arm: str, rundir: Path, args) -> dict:
    spec_file = rundir / "runtime_spec.json"
    spec_file.write_text(json.dumps({"fault": task.get("fault"),
                                     "runtime": task["runtime"]}),
                         encoding="utf-8")
    rundb = rundir / "run.db"
    cmd = [sys.executable, "-m", "harness.compare", "--run-one",
           "--spec", str(spec_file), "--arm", arm,
           "--db", str(rundb), "--out", str(rundir),
           "--settle-ms", str(args.settle_ms)]
    with open(rundir / "stdout.txt", "w", encoding="utf-8") as out, \
         open(rundir / "stderr.txt", "w", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=out, stderr=err)
    try:
        proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return _base_run(task, arm, "timeout")

    run_json = rundir / "run.json"
    if run_json.exists():
        try:
            run = json.loads(run_json.read_text(encoding="utf-8"))
        except Exception as exc:
            run = _base_run(task, arm, "harness-error", error=repr(exc))
    else:
        run = _base_run(task, arm, "harness-error", error="no run.json produced")

    run["task_id"] = task["task_id"]
    run["fault_class"] = task.get("fault_class") or "clean"
    run["fault_schedule"] = task.get("fault")

    final_db = rundir / "final.db"
    if not final_db.exists():
        try:
            backup_copy(str(rundir / "run.db"), str(final_db))
        except Exception:
            pass
    try:
        measured = evaluate.measure_final_db(
            str(final_db), task["expected"]["lines"], run.get("final_cookie"))
    except Exception as exc:
        measured = {"measured_outcome": "unknown", "resolution": "error",
                    "mismatches": []}
    run["measured_outcome"] = measured["measured_outcome"]
    run["measured_resolution"] = measured["resolution"]
    run["measured_mismatches"] = measured["mismatches"]
    run["physical_correct"] = (run["measured_outcome"] == "correct")
    return run


def run_suite_cli(argv) -> int:
    p = argparse.ArgumentParser(description="Run the three-arm comparison suite")
    p.add_argument("--suite", required=True)
    p.add_argument("--out-root", default=None)
    p.add_argument("--settle-ms", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--label", default=None)
    args = p.parse_args(argv)

    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_dir = suite_path.parent
    tasks = [json.loads((suite_dir / rel).read_text(encoding="utf-8"))
             for rel in suite["task_files"]]

    out_root = Path(args.out_root) if args.out_root else (
        PROJECT_ROOT / "var" / f"acceptance-{time.strftime('%Y%m%d-%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)

    catalog = str(PROJECT_ROOT / "store" / "seed_catalog.json")
    template = out_root / "template.db"
    build_template_db(catalog, str(template))

    expected_by_id = {t["task_id"]: t["expected"]["lines"] for t in tasks}
    runs: list[dict] = []
    for task in tasks:
        for arm in ("A", "B", "C"):
            rundir = out_root / "runs" / task["task_id"] / arm
            rundir.mkdir(parents=True, exist_ok=True)
            backup_copy(str(template), str(rundir / "run.db"))
            run = _spawn_run(task, arm, rundir, args)
            runs.append(run)
            print(f"{task['task_id']} {arm}: "
                  f"{run['execution_status']} declared={run['declared_outcome']} "
                  f"measured={run['measured_outcome']}")

    enriched = [metrics.enrich_run(r, expected_by_id[r["task_id"]]) for r in runs]
    report = metrics.aggregate(enriched)

    # Per-run records.
    per_run = []
    for r in enriched:
        rec = {k: r[k] for k in (
            "task_id", "arm", "fault_class", "fault_schedule",
            "execution_status", "declared_outcome", "measured_outcome",
            "measured_resolution", "measured_mismatches", "physical_correct",
            "deviation_observation", "recovery", "stop_reasons",
            "hook_counts", "adapter_mutations", "error")}
        rec["corrective_writes"] = [
            {k: m.get(k) for k in (
                "operation", "context", "corrective", "target", "unnecessary",
                "redundant_line")}
            for m in r.get("corrective_writes") or []]
        rec["final_persisted"] = r.get("final_persisted")
        rec["snapshot_summary"] = {
            "n_snapshots": len(r.get("snapshots") or []),
            "capture_ms_total": r.get("capture_ms_total"),
        }
        per_run.append(rec)

    summary = {
        "label": "Harness-produced demonstrations on controlled fixtures, "
                 "not measured model behavior",
        "suite": suite.get("suite"),
        "suite_label": args.label,
        "n_runs": len(per_run),
        "report": report,
        "runs": per_run,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_root / "summary.md").write_text(
        metrics.render_markdown(report, suite.get("suite") or "suite"),
        encoding="utf-8")
    print(f"summary -> {out_root}")
    return 0


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--run-one" in argv:
        return run_one_cli([a for a in argv if a != "--run-one"])
    return run_suite_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())

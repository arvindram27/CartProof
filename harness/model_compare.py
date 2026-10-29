"""Brief 5 model-arm comparison runner and per-run worker.

Arms: MA (naive model actor), CA (careful model agent), MB (model actor +
CartProof verification), MC (model actor + verification + bounded recovery).
Fake client only; no live transport exists in this brief.

Orchestrator: ``python -m harness.model_compare --suite eval/model/dev/suite.json``
Worker: ``python -m harness.model_compare --run-one ...``

The worker sees only the request text, seed lines, fault, arm, trial and the
fake-client script. Expectations are loaded only by the parent, after the run,
for measurement. Frozen suites are refused.
"""
from __future__ import annotations

import argparse
import copy
import json
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
from harness import evaluate, model_metrics  # noqa: E402
from harness.compare import build_template_db  # noqa: E402


def load_catalog() -> list[dict]:
    data = json.loads((PROJECT_ROOT / "store" / "seed_catalog.json")
                      .read_text(encoding="utf-8"))
    out = []
    for p in data["products"]:
        for v in p["variants"]:
            out.append({"sku": v["sku"], "product_id": p["id"],
                        "product_name": p["name"], "label": v["label"],
                        "size": v.get("size"), "color": v.get("color"),
                        "price_cents": v["price_cents"]})
    return out


def _seed_agg(seed_lines) -> dict:
    out: dict[str, int] = {}
    for s in seed_lines or []:
        out[s["sku"]] = out.get(s["sku"], 0) + int(s["quantity"])
    return out


def execute_model_run(setup: dict, fault: dict, arm: str, trial: int,
                      fake_script: dict, db_path: str, out_dir,
                      settle_ms: int = 2000) -> dict:
    from adapter.actions import BrowserSession
    from adapter.observations import read_api_cart, read_dom_cart
    from agents.careful_agent import CarefulAgent
    from agents.full_cart_guard import aggregate_quantities, apply_full_cart_guard
    from agents.model_actor import ModelActor
    from agents.model_client import (BudgetedModelClient, ClientConfig,
                                     FakeModelClient)
    from core.ledger import Ledger

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = str(PROJECT_ROOT / "store" / "seed_catalog.json")
    request_text = setup["request_text"]
    seed_lines = setup.get("seed_lines") or []

    run: dict = {
        "arm": arm, "trial": trial, "declared_outcome": None, "raw_verdict": None,
        "execution_status": "normal", "error": None,
        "seed_lines": seed_lines, "items": [],
        "snapshots": [], "mutations": [],
        "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
        "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0,
                              "delete_line": 0},
        "stop_reasons": [], "final_cookie": None, "final_persisted": None,
        "guard": {"applied": arm in ("MB", "MC")},
    }

    budget_cfg = dict(fake_script.get("budget") or {})
    config = ClientConfig(
        enabled=True, model_name=budget_cfg.get("model_name", "fake-model"),
        max_calls=int(budget_cfg.get("max_calls", 40)),
        token_budget=int(budget_cfg.get("token_budget", 20000)),
        max_output_tokens_per_call=int(budget_cfg.get("max_output_tokens", 512)))
    fake = FakeModelClient(copy.deepcopy(fake_script.get("responses") or []),
                           model_name=config.model_name)
    client = BudgetedModelClient(fake, config)
    catalog = load_catalog()

    from store.app import create_app, create_control_app

    store_app = create_app(db_path, catalog_path)
    control_app = create_control_app(db_path, catalog_path)
    ss, st, sp = boot_server(store_app)
    cs, ct, cp = boot_server(control_app)
    store_url = f"http://127.0.0.1:{sp}"
    control_url = f"http://127.0.0.1:{cp}"

    session = None
    observer = None
    actor = None
    try:
        session = BrowserSession(store_url, headed=False)
        observer = Observer(db_path, out_dir, session)
        wrapped = ObservedSession(session, observer)

        with observer.context("setup"):
            for sl in seed_lines:
                wrapped.seed_line(sl["sku"], sl["quantity"])

        actor_cls = CarefulAgent if arm == "CA" else ModelActor
        actor = actor_cls(wrapped, client, catalog,
                          masked=arm in ("MB", "MC"), hook=None,
                          model_name=config.model_name)
        actor.resolve_intent(request_text)

        if actor.outcome_kind != "proceed" or not actor.executable_intent \
                or not actor.executable_intent.get("items"):
            if actor.outcome_kind == "proceed":
                actor.outcome_kind = "operational_failure"
                actor.failures.append({
                    "stage": "intent", "category": "operational",
                    "reason": "executable intent has no items"})
            run["declared_outcome"] = (
                "CLARIFY" if actor.outcome_kind == "semantic_clarify" else None)
        else:
            ei = actor.executable_intent
            items = [{"product_query": it["product_query"],
                      "target_sku": it["sku"],
                      "quantity": int(it["quantity"])} for it in ei["items"]]
            run["items"] = items
            additions: dict[str, int] = {}
            for it in items:
                additions[it["target_sku"]] = \
                    additions.get(it["target_sku"], 0) + it["quantity"]
            run["declared_additions"] = additions

            verifier = None
            ledger = None
            controller = None
            raw_pairs: dict = {}
            baseline_quantities = None

            if arm in ("MB", "MC"):
                import time as _time

                def reader(timeout_ms):
                    start = _time.monotonic()
                    dom = read_dom_cart(session, timeout_ms=timeout_ms)
                    dom_payload = dict(dom.payload)
                    dom_payload["timestamp"] = dom.timestamp
                    remaining = timeout_ms - int((_time.monotonic() - start) * 1000)
                    if remaining <= 0:
                        api_payload = {"read_timeout": True,
                                       "timestamp": dom.timestamp}
                    else:
                        api = read_api_cart(session, timeout_ms=remaining)
                        api_payload = dict(api.payload)
                        api_payload["timestamp"] = api.timestamp
                    return dom_payload, api_payload

                from core.verify import Verifier
                verifier = Verifier(reader, settle_ms=settle_ms)

                final_agg = _seed_agg(seed_lines)
                for sku, q in additions.items():
                    final_agg[sku] = final_agg.get(sku, 0) + q
                specs = [{"type": "item_quantity", "sku": sku, "quantity": q}
                         for sku, q in sorted(final_agg.items()) if q > 0]
                if ei.get("budget_max_cents") is not None:
                    specs.append({"type": "budget_max",
                                  "max_cents": ei["budget_max_cents"]})
                ledger = Ledger.build(specs)

                from core.recovery import attach_raw_pair_recorder
                attach_raw_pair_recorder(verifier, raw_pairs)
                if arm == "MC":
                    from core.recovery import (RecoveryController,
                                               establish_recovery_baseline)
                    establish_recovery_baseline(verifier)
                    controller = RecoveryController(
                        verifier, wrapped, ledger, additions, verifier.baseline,
                        raw_pairs=raw_pairs)
                else:
                    verifier.take_baseline()
                baseline_quantities = aggregate_quantities(
                    verifier.baseline.get("lines"))

            if baseline_quantities is None and arm in ("MB", "MC"):
                run["guard"] = {"applied": True, "status": "baseline_unavailable",
                                "reason": "observed baseline unavailable or "
                                          "inconsistent"}
                run["declared_outcome"] = "PARTIAL"
                run["stop_reasons"] = ["full-cart guard: baseline unavailable"]
            else:
                if fault and fault.get("name"):
                    httpx.post(
                        f"{control_url}/_control/fault",
                        json={"name": fault["name"],
                              "params": fault.get("params") or {}},
                        timeout=10).raise_for_status()

                def hook(event):
                    kind = event.get("kind")
                    observer.record_hook_call(kind)
                    if kind in ("before_add", "after_add"):
                        observer.hook_entry(kind, event)
                    if arm == "MC" and controller is not None:
                        if kind == "before_add":
                            return controller.before_add(event)
                        if kind == "pre_add":
                            return controller.pre_add(event)
                        if kind == "after_add":
                            with observer.context("after_add_hook"):
                                return controller.after_add(event)
                        return None
                    if arm == "MB" and verifier is not None:
                        if kind == "after_add":
                            verifier.on_action(event["add"]["payload"])
                        return None
                    return None

                actor.hook = hook
                actor.execute_items(items)

                if arm == "CA":
                    review = actor.inspect_and_review()
                    run["declared_outcome"] = review["declared_outcome"]
                    run["raw_verdict"] = review["declared_outcome"]
                elif arm == "MA":
                    run["declared_outcome"] = actor.naive_declared()
                    run["raw_verdict"] = run["declared_outcome"]
                elif arm == "MB":
                    report = verifier.finish(ledger, additions)
                    run["raw_verdict"] = report["verdict"]
                    guard = apply_full_cart_guard(
                        report, raw_pairs, baseline_quantities, additions,
                        ei.get("budget_max_cents"), report["verdict"])
                    run["guard"] = guard
                    run["declared_outcome"] = guard["guarded_verdict"]
                    reasons = [a.get("detail") or a.get("kind")
                               for a in report.get("anomalies", [])]
                    for c in report.get("constraints", []):
                        if c.get("state") != "confirmed":
                            reasons.append(c.get("reason"))
                    if guard["status"] in ("downgrade", "not_eligible",
                                            "mismatch"):
                        reasons.append(f"full-cart guard: {guard['reason']}")
                    run["stop_reasons"] = reasons
                elif arm == "MC":
                    report, verdict = controller.complete()
                    run["raw_verdict"] = verdict
                    guard = apply_full_cart_guard(
                        report, raw_pairs, baseline_quantities, additions,
                        ei.get("budget_max_cents"), verdict)
                    run["guard"] = guard
                    run["declared_outcome"] = guard["guarded_verdict"]
                    reasons = ([controller.stop_reason]
                               if controller.stop_reason else [])
                    if guard["status"] in ("downgrade", "not_eligible",
                                            "mismatch"):
                        reasons.append(f"full-cart guard: {guard['reason']}")
                    run["stop_reasons"] = reasons
    except Exception:
        run["execution_status"] = "crash"
        run["error"] = traceback.format_exc()
    finally:
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
                    "version": cart["version"],
                    "total_cents": cart["total_cents"]}
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
            summ = observer.summary()
            run["adapter_mutations"] = summ["adapter_mutations"]
            run["capture_ms_total"] = summ["capture_ms_total"]
        if actor is not None:
            run.update(actor.record())
        run["budget"] = client.summary()
        run["model_client_requests"] = fake.requests

    return run


def run_one_cli(argv) -> int:
    p = argparse.ArgumentParser(description="Run one model arm")
    p.add_argument("--spec", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--settle-ms", type=int, default=2000)
    args = p.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    try:
        run = execute_model_run(spec["setup"], spec.get("fault") or
                                {"name": None, "params": {}}, spec["arm"],
                                int(spec.get("trial", 1)), spec["fake_script"],
                                args.db, args.out, settle_ms=args.settle_ms)
    except Exception:
        run = {"arm": spec.get("arm"), "trial": spec.get("trial", 1),
               "declared_outcome": None, "raw_verdict": None,
               "execution_status": "crash", "error": traceback.format_exc(),
               "seed_lines": (spec.get("setup") or {}).get("seed_lines") or [],
               "items": [], "snapshots": [], "mutations": [],
               "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
               "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0,
                                     "delete_line": 0},
               "stop_reasons": [], "final_cookie": None, "final_persisted": None,
               "guard": {"applied": spec.get("arm") in ("MB", "MC")}}
    (Path(args.out) / "run.json").write_text(
        json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"arm": run.get("arm"),
                      "execution_status": run.get("execution_status"),
                      "declared_outcome": run.get("declared_outcome")}))
    return 0


def _base_run(task: dict, arm: str, trial: int, status: str,
              error: str | None = None) -> dict:
    return {"task_id": task["task_id"], "arm": arm, "trial": trial,
            "fault_class": task.get("fault_class") or "clean",
            "fault_schedule": task.get("fault"),
            "execution_status": status, "declared_outcome": None,
            "raw_verdict": None, "seed_lines": task["runtime"].get("seed_lines") or [],
            "items": [], "snapshots": [], "mutations": [],
            "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
            "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0,
                                  "delete_line": 0},
            "stop_reasons": [], "error": error, "final_cookie": None,
            "final_persisted": None, "guard": {"applied": arm in ("MB", "MC")}}


def _spawn(task: dict, arm: str, trial: int, rundir: Path, args,
           fake_script: dict) -> dict:
    spec_file = rundir / "worker_spec.json"
    spec_file.write_text(json.dumps({
        "setup": {"request_text": task["runtime"]["request_text"],
                  "seed_lines": task["runtime"].get("seed_lines") or []},
        "fault": task.get("fault"), "arm": arm, "trial": trial,
        "fake_script": fake_script}), encoding="utf-8")
    cmd = [sys.executable, "-m", "harness.model_compare", "--run-one",
           "--spec", str(spec_file), "--db", str(rundir / "run.db"),
           "--out", str(rundir), "--settle-ms", str(args.settle_ms)]
    with open(rundir / "stdout.txt", "w", encoding="utf-8") as out, \
         open(rundir / "stderr.txt", "w", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=out, stderr=err)
    try:
        proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return _base_run(task, arm, trial, "timeout")

    run_json = rundir / "run.json"
    if run_json.exists():
        try:
            run = json.loads(run_json.read_text(encoding="utf-8"))
        except Exception as exc:
            run = _base_run(task, arm, trial, "harness-error", error=repr(exc))
    else:
        run = _base_run(task, arm, trial, "harness-error",
                        error="no run.json produced")
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
    except Exception:
        measured = {"measured_outcome": "unknown", "resolution": "error",
                    "mismatches": []}
    run["measured_outcome"] = measured["measured_outcome"]
    run["measured_resolution"] = measured["resolution"]
    run["measured_mismatches"] = measured["mismatches"]
    return run


def run_suite_cli(argv) -> int:
    p = argparse.ArgumentParser(description="Run the model-arm comparison")
    p.add_argument("--suite", required=True)
    p.add_argument("--out-root", default=None)
    p.add_argument("--settle-ms", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--trials", type=int, default=1)
    args = p.parse_args(argv)

    suite_path = Path(args.suite).resolve()
    if "frozen" in suite_path.parts:
        print("error: frozen model suites are never run in this brief",
              file=sys.stderr)
        return 2
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_dir = suite_path.parent
    tasks = [json.loads((suite_dir / rel).read_text(encoding="utf-8"))
             for rel in suite["task_files"]]
    fake_scripts = {}
    fs_path = suite_dir / "fake_scripts.json"
    if fs_path.exists():
        fake_scripts = json.loads(fs_path.read_text(encoding="utf-8"))

    out_root = Path(args.out_root) if args.out_root else (
        PROJECT_ROOT / "var" / f"acceptance-{time.strftime('%Y%m%d-%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)

    catalog = str(PROJECT_ROOT / "store" / "seed_catalog.json")
    template = out_root / "template.db"
    build_template_db(catalog, str(template))

    runs: list[dict] = []
    for task in tasks:
        for arm in ("MA", "CA", "MB", "MC"):
            for trial in range(1, args.trials + 1):
                rundir = out_root / "runs" / task["task_id"] / arm / f"t{trial}"
                rundir.mkdir(parents=True, exist_ok=True)
                backup_copy(str(template), str(rundir / "run.db"))
                script = (fake_scripts.get(task["task_id"]) or {}).get(arm) or {}
                run = _spawn(task, arm, trial, rundir, args, script)
                runs.append(run)
                print(f"{task['task_id']} {arm} t{trial}: "
                      f"{run['execution_status']} declared={run['declared_outcome']} "
                      f"measured={run.get('measured_outcome')}")

    task_by_id = {t["task_id"]: t for t in tasks}
    enriched = [model_metrics.enrich_model_run(r, task_by_id[r["task_id"]])
                for r in runs]
    report = model_metrics.aggregate(enriched)
    summary = {
        "label": "Fake-client plumbing demonstration only; not measured model "
                 "behavior",
        "suite": suite.get("suite"), "n_runs": len(enriched),
        "report": report, "runs": enriched,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_root / "summary.md").write_text(
        model_metrics.render_markdown(report, suite.get("suite") or "model"),
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

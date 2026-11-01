import copy, hashlib, json, math, os, re, shutil, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from harness import pilot_manifest as mft
from harness.model_pilot import (
    PilotError, _read_pinned_task, build_cell_spec, validate_worker_spec,
    preflight_stage, _run_stage_injected, _validate_receipt, export_safe,
    pilot_accounting_check, compute_source_digest, DIGEST_PATHS,
    _reserve_attempt, run_dry,
)

def make_root(tmp_path):
    root = tmp_path / "root"
    (root / "var").mkdir(parents=True)
    for tid, info in mft.TASKS.items():
        dst = root / info["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / info["path"], dst)
    (root / "store").mkdir(exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / "store" / "seed_catalog.json", root / "store" / "seed_catalog.json")
    fake_scripts = root / "eval" / "model" / "dev" / "fake_scripts.json"
    fake_scripts.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / "eval" / "model" / "dev" / "fake_scripts.json",
                    fake_scripts)
    return root

def fake_run(spec):
    prof = mft.PROFILES[next(c["profile"] for c in mft.STAGES[1] + mft.STAGES[2]
                             if c["cell_id"] == spec["pilot_cell_id"])]
    rates = prof["price_table"]["per_1m_tokens"][prof["model_name"]]
    prompt, completion, cached = 1000, 100, 0
    cost = round((prompt * rates["input"] + completion * rates["output"]) / 1_000_000, 8)
    return {
        "arm": spec["arm"], "trial": 1, "execution_status": "normal",
        "declared_outcome": "COMPLETE", "raw_marker": True,
        "seed_lines": spec["setup"]["seed_lines"], "items": [], "snapshots": [],
        "mutations": [], "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
        "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0, "delete_line": 0},
        "stop_reasons": [], "final_cookie": "c",
        "final_persisted": {"lines": [], "resolution": "ok", "version": 1, "total_cents": 0},
        "guard": {"applied": False}, "transport": "meta",
        "budget": {
            "calls_attempted": 1, "calls_made": 1, "calls_refused": 0,
            "consumed_tokens": 1100, "reserved_tokens": 5096, "overrun": False,
            "records": [{"call_index": 1, "kind": "intent", "status": "made",
                         "reserved": 5096, "consumed": 1100, "declared_usage": 1100}],
            "sidecar": [{
                "call_index": 1, "usage_present": True, "usage_malformed": False,
                "usage_detail_trustworthy": True, "finish_reason": "stop",
                "provider_usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                   "total_tokens": prompt + completion, "cached_tokens": cached,
                                   "reasoning_tokens": 0},
                "estimated_cost_usd": cost,
            }],
            "smoke_guard": {"enabled": True, "transport_kind": "meta", "stopped": False,
                            "accounting_mode": "provider", "provider_billing_verified": False},
        },
    }

def fake_spawn(ctx):
    spec = ctx["spec"]
    run = fake_run(spec)
    (ctx["rundir"] / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run, {"cleanup_status": "verified", "timed_out": False}

def test_task_hashes():
    for tid, info in mft.TASKS.items():
        assert hashlib.sha256((PROJECT_ROOT / info["path"]).read_bytes()).hexdigest() == info["sha256"]

def test_build_cell_spec_types_and_complete_config():
    task, _, _ = _read_pinned_task(PROJECT_ROOT, "m-clean-seeded")
    spec = build_cell_spec(mft.STAGES[1][0], task)
    assert spec["fake_script"]["budget"] == {"max_calls": 12, "token_budget": 20000, "max_output_tokens": 4096}
    assert spec["fake_script"]["client"]["model_name"] == "muse-spark-1.3"
    assert spec["fault"] == {"name": None, "params": {}}

def test_stage2_carries_canonical_fault():
    task, _, _ = _read_pinned_task(PROJECT_ROOT, "m-fault-silent-drop")
    spec = build_cell_spec(mft.STAGES[2][0], task)
    assert spec["fault"] == {"name": "silent-drop", "params": {}}

def test_stage_orders_fixed():
    assert [c["cell_id"] for c in mft.STAGES[1]] == [
        "s1-seeded-ca-spark", "s1-seeded-ca-wonder",
        "s1-ambig-ca-wonder", "s1-ambig-ca-spark"]
    assert [c["cell_id"] for c in mft.STAGES[2]][:4] == [
        "s2-drop-ca-spark", "s2-drop-ca-wonder",
        "s2-drop-mc-wonder", "s2-drop-mc-spark"]

def test_validation_rejects_extras_and_smoke(tmp_path):
    root = make_root(tmp_path)
    task, _, _ = _read_pinned_task(root, "m-clean-seeded")
    spec = build_cell_spec(mft.STAGES[1][0], task)
    out = root / "var" / "model-pilot-stage1-test" / "cells" / spec["pilot_cell_id"]
    out.mkdir(parents=True)
    spec["_startup_barrier"] = {"ready": str(out / "barrier-ready.json"),
                                "release": str(out / "barrier-release")}
    validate_worker_spec(spec, root)
    for mutation in ({"extra": 1}, {"smoke": True}):
        bad = copy.deepcopy(spec)
        bad.update(mutation)
        with pytest.raises(PilotError):
            validate_worker_spec(bad, root)

def test_validation_rejects_bool_budget_and_wrong_fault(tmp_path):
    root = make_root(tmp_path)
    task, _, _ = _read_pinned_task(root, "m-clean-seeded")
    spec = build_cell_spec(mft.STAGES[1][0], task)
    out = root / "var" / "model-pilot-stage1-test" / "cells" / spec["pilot_cell_id"]
    out.mkdir(parents=True)
    spec["_startup_barrier"] = {"ready": str(out / "barrier-ready.json"),
                                "release": str(out / "barrier-release")}
    bad = copy.deepcopy(spec)
    bad["fake_script"]["budget"]["max_calls"] = True
    with pytest.raises(PilotError):
        validate_worker_spec(bad, root)
    bad = copy.deepcopy(spec)
    bad["fault"] = {"name": "silent-drop", "params": {}}
    with pytest.raises(PilotError):
        validate_worker_spec(bad, root)

def test_task_hash_mismatch(tmp_path):
    root = make_root(tmp_path)
    p = root / mft.TASKS["m-clean-seeded"]["path"]
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(PilotError, match="task_hash"):
        _read_pinned_task(root, "m-clean-seeded")

def test_preflight_rejects_unknown_dirty_path(monkeypatch, tmp_path):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    fake_py = root / ".venv" / "Scripts" / "python.exe"
    fake_py.parent.mkdir(parents=True)
    fake_py.write_bytes(b"")
    monkeypatch.setattr(mp, "PROJECT_ROOT", root)
    monkeypatch.setattr(mp.sys, "executable", str(fake_py))
    monkeypatch.setattr(mp, "_git_scope", lambda r: (_ for _ in ()).throw(PilotError("worktree_dirty:x")))
    with pytest.raises(PilotError, match="worktree_dirty"):
        preflight_stage(root, 1, production=True)

def test_pilot_accounting_recomputes_cost(tmp_path):
    task, _, _ = _read_pinned_task(PROJECT_ROOT, "m-clean-seeded")
    spec = build_cell_spec(mft.STAGES[1][0], task)
    run = fake_run(spec)
    prof = mft.PROFILES["spark-medium"]
    ok, reasons = pilot_accounting_check(run, prof)
    assert ok, reasons
    bad = copy.deepcopy(run)
    bad["budget"]["sidecar"][0]["provider_usage"]["total_tokens"] = 1
    ok, reasons = pilot_accounting_check(bad, prof)
    assert not ok and "provider_total_mismatch" in reasons

def test_zero_usage_valid():
    from harness.model_smoke import validate_paid_accounting
    run = {"transport": "meta", "budget": {
        "calls_attempted": 1, "calls_made": 1, "calls_refused": 0,
        "consumed_tokens": 0, "reserved_tokens": 100, "overrun": False,
        "records": [{"call_index": 1, "kind": "intent", "status": "made", "reserved": 100, "consumed": 0}],
        "sidecar": [{"call_index": 1, "usage_present": True, "usage_malformed": False,
                     "usage_detail_trustworthy": True, "finish_reason": "stop",
                     "provider_usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                        "total_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0},
                     "estimated_cost_usd": 0.0}],
        "smoke_guard": {"enabled": True, "transport_kind": "meta", "stopped": False}}}
    ok, reasons, made = validate_paid_accounting(run)
    assert ok and made == {1}

def test_feasible_and_exhausted_traces():
    from agents.model_client import BudgetedModelClient, ClientConfig, ModelRequest, ModelProposal, Usage
    class Inner:
        def __init__(self, out): self.out = out
        def propose(self, req):
            return ModelProposal(kind=req.kind, payload={}, status="ok",
                                 usage=Usage(input_tokens=1000, output_tokens=self.out))
    cfg = ClientConfig(enabled=True, max_calls=12, token_budget=20000, max_output_tokens_per_call=4096)
    c = BudgetedModelClient(Inner(500), cfg)
    for _ in range(7):
        assert c.propose(ModelRequest(kind="action", max_output_tokens=4096)).status != "refusal"
    assert c.calls_made == 7
    c2 = BudgetedModelClient(Inner(4096), cfg)
    made = 0
    for _ in range(12):
        if c2.propose(ModelRequest(kind="action", max_output_tokens=4096)).status == "refusal":
            break
        made += 1
    assert made < 7 and c2.calls_refused >= 1

def test_max_calls_ceiling():
    from agents.model_client import BudgetedModelClient, ClientConfig, ModelRequest, ModelProposal, Usage
    class Inner:
        def propose(self, req):
            return ModelProposal(kind=req.kind, payload={}, status="ok",
                                 usage=Usage(input_tokens=10, output_tokens=10))
    cfg = ClientConfig(enabled=True, max_calls=12, token_budget=100000, max_output_tokens_per_call=4096)
    c = BudgetedModelClient(Inner(), cfg)
    for _ in range(12):
        assert c.propose(ModelRequest(kind="action", max_output_tokens=4096)).status != "refusal"
    assert "max_calls" in c.propose(ModelRequest(kind="action")).error

def test_injected_stage1_complete_and_raw_preserved(tmp_path):
    root = make_root(tmp_path)
    batch = _run_stage_injected(root, 1, fake_spawn, production=False, control_root=tmp_path/"pilot-control")
    data = json.loads((batch / "batch.json").read_text())
    assert data["status"] == "complete" and data["transport_label"] == "FAKE (injected)"
    assert len(data["cells"]) == 4
    raw = json.loads((batch / "cells" / "s1-seeded-ca-spark" / "run.json").read_text())
    assert raw.get("raw_marker") is True
    assert "task_id" not in raw
    ev = json.loads((batch / "cells" / "s1-seeded-ca-spark" / "evaluated.json").read_text())
    assert ev["task_id"] == "m-clean-seeded"
    assert (batch / "export" / "manifest.json").is_file()

def test_partial_halt_stops_before_next_cell(tmp_path):
    root = make_root(tmp_path)
    calls = {"n": 0}
    def one_bad(ctx):
        calls["n"] += 1
        if calls["n"] == 2:
            (ctx["rundir"] / "run.json").write_text(json.dumps({"execution_status": "timeout"}), encoding="utf-8")
            return {"execution_status": "timeout"}, {"cleanup_status": "terminated"}
        return fake_spawn(ctx)
    batch = _run_stage_injected(root, 1, one_bad, production=False, control_root=tmp_path/"pilot-control2")
    data = json.loads((batch / "batch.json").read_text())
    assert data["status"] == "halted"
    assert len(data["cells"]) == 2
    assert calls["n"] == 2

def test_stage2_closed_and_receipt_shape(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.setattr(mft, "ACCEPTED_STAGE1_RECEIPT_ID", None)
    monkeypatch.setattr(mft, "ACCEPTED_STAGE1_MANIFEST_ID", None)
    with pytest.raises(PilotError, match="receipt_not_accepted"):
        _run_stage_injected(root, 2, fake_spawn, production=False, control_root=tmp_path/"pilot-control3")
    with pytest.raises(PilotError):
        _validate_receipt({})


def test_reviewed_receipt_shape_matches_constants():
    receipt = mft.ACCEPTED_STAGE1_RECEIPT
    assert isinstance(receipt, dict)
    assert set(receipt) == {"receipt_id", "manifest_id", "stage", "status",
                            "known_cost", "unknown_cost_calls", "cells",
                            "file_hashes"}
    assert receipt["receipt_id"] == mft.ACCEPTED_STAGE1_RECEIPT_ID
    assert receipt["manifest_id"] == mft.ACCEPTED_STAGE1_MANIFEST_ID
    assert receipt["stage"] == 1
    assert receipt["status"] == "complete"
    assert receipt["unknown_cost_calls"] == 0
    assert receipt["cells"] == [c["cell_id"] for c in mft.STAGES[1]]
    expected = {"batch.json", "ledger.json", "summary.md", "manifest.json"}
    for cid in receipt["cells"]:
        expected |= {f"cells/{cid}/run.json", f"cells/{cid}/evaluated.json",
                     f"cells/{cid}/secret-scan.txt"}
    assert set(receipt["file_hashes"]) == expected
    assert len(receipt["file_hashes"]) == 16


def test_receipt_id_mismatch_refused():
    bad = dict(mft.ACCEPTED_STAGE1_RECEIPT)
    bad["receipt_id"] = "not-the-reviewed-id"
    with pytest.raises(PilotError, match="receipt_id_mismatch"):
        _validate_receipt(bad)


def test_stage2_accepts_reviewed_receipt(tmp_path):
    root = make_root(tmp_path)
    batch = _run_stage_injected(root, 2, fake_spawn, production=False,
                                control_root=tmp_path / "pilot-control-stage2",
                                receipt=mft.ACCEPTED_STAGE1_RECEIPT)
    data = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    assert data["status"] == "complete"
    assert data["stage"] == 2
    assert len(data["cells"]) == 8
    assert data["known_cost_usd_estimated"] >= \
        mft.ACCEPTED_STAGE1_RECEIPT["known_cost"]


def test_run_stage_threads_receipt_only_for_stage2(monkeypatch):
    import harness.model_pilot as mp
    seen = {}

    def fake_injected(root, stage, spawn_owned, **kwargs):
        seen["stage"] = stage
        seen["receipt"] = kwargs.get("receipt")
        return root

    monkeypatch.setattr(mp, "_run_stage_injected", fake_injected)
    mp.run_stage(mp.PROJECT_ROOT, 1, live=True)
    assert seen["stage"] == 1 and seen["receipt"] is None
    mp.run_stage(mp.PROJECT_ROOT, 2, live=True)
    assert seen["stage"] == 2
    assert seen["receipt"] is mft.ACCEPTED_STAGE1_RECEIPT

def test_export_safe_excludes_dbs_and_hashes_only_delivered(tmp_path):
    root = make_root(tmp_path)
    batch = _run_stage_injected(root, 1, fake_spawn, production=False, control_root=tmp_path/"pilot-control")
    exp = batch / "export"
    names = {p.relative_to(exp).as_posix() for p in exp.rglob("*") if p.is_file()}
    assert not any(n.endswith(".db") for n in names)
    assert "cells/s1-seeded-ca-spark/run.json" in names
    man = json.loads((exp / "manifest.json").read_text())
    assert "manifest.json" not in man
    assert man["files"]["batch.json"] == hashlib.sha256((exp / "batch.json").read_bytes()).hexdigest()

def test_scoring_wrong_and_unknown_cart(tmp_path):
    from harness import evaluate
    import sqlite3
    # wrong persisted cart is not a success even with COMPLETE declaration
    db = tmp_path / "wrong.db"
    from store import models as _models
    _models.init_and_seed(str(db), str(PROJECT_ROOT / "store" / "seed_catalog.json"))
    # no cookie/cart -> empty/unknown; assert it is not "correct"
    out = evaluate.measure_final_db(str(db), {"BTL-HIK-500-NAVYBLUE": 1}, "missing")
    assert out["measured_outcome"] != "correct"
    missing = tmp_path / "missing.db"
    out = evaluate.measure_final_db(str(missing), {"BTL-HIK-500-NAVYBLUE": 1}, "missing")
    assert out["measured_outcome"] == "unknown"

def test_made_reserved_consumed_declared_counterexample():
    from harness.pilot_manifest import PROFILES
    prof = PROFILES["spark-medium"]
    run = {"transport":"meta","budget":{
        "calls_attempted":1,"calls_made":1,"calls_refused":0,
        "consumed_tokens":1,"reserved_tokens":1,"overrun":False,
        "records":[{"call_index":1,"kind":"intent","status":"made","reserved":1,"consumed":1,"declared_usage":1100}],
        "sidecar":[{"call_index":1,"usage_present":True,"usage_malformed":False,
                     "usage_detail_trustworthy":True,"finish_reason":"stop",
                     "provider_usage":{"prompt_tokens":1000,"completion_tokens":100,"total_tokens":1100,"cached_tokens":0,"reasoning_tokens":0},
                     "estimated_cost_usd":0.001675}],
        "smoke_guard":{"enabled":True,"transport_kind":"meta","stopped":False}}}
    ok,reasons = pilot_accounting_check(run, prof)
    assert not ok

def test_persistent_totals_and_latch(tmp_path):
    root = make_root(tmp_path)
    b = _run_stage_injected(root,1,fake_spawn,production=False,control_root=tmp_path/"ctl")
    data = json.loads((b/"batch.json").read_text())
    reservations = list((tmp_path/"ctl"/"attempts"/"stage1").rglob("reservation.json"))
    assert len(reservations)==1
    res = json.loads(reservations[0].read_text())
    assert res["known_cost_usd"] == data["known_cost_usd_estimated"]
    with pytest.raises(PilotError):
        _run_stage_injected(root,1,fake_spawn,production=False,control_root=tmp_path/"ctl", out_root=tmp_path/"b2")

def test_production_digest_not_synthetic():
    from harness.model_pilot import compute_source_digest
    d = compute_source_digest(PROJECT_ROOT)
    assert d != "synthetic-test"
    assert len(d)==64


def _reservation(tmp_path, name):
    files = list((tmp_path / name).rglob("reservation.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())

def test_timeout_unresolved_never_zero(tmp_path):
    root = make_root(tmp_path)
    def timeout_spawn(ctx):
        (ctx["rundir"] / "run.json").write_text(json.dumps({"execution_status": "timeout"}), encoding="utf-8")
        return {"execution_status": "timeout"}, {"cleanup_status": "terminated"}
    batch = _run_stage_injected(root, 1, timeout_spawn, production=False,
                                control_root=tmp_path / "ctl-timeout")
    data = json.loads((batch / "batch.json").read_text())
    assert data["unresolved_cost_state"] == "unresolved"
    res = _reservation(tmp_path, "ctl-timeout")
    assert res["unresolved_cost_state"] == "unresolved"
    assert res["state"] == "halted"

def test_late_prefixed_error_remains_unresolved(tmp_path):
    # Text raised INSIDE spawn can never prove no-start (phase provenance).
    root = make_root(tmp_path)
    def refuse_spawn(ctx):
        raise PilotError("prerequisite_failed:fixture")
    batch = _run_stage_injected(root, 1, refuse_spawn, production=False,
                                control_root=tmp_path / "ctl-notstarted")
    data = json.loads((batch / "batch.json").read_text())
    assert data["cells"][0]["child_may_have_started"] is True
    assert data["unresolved_cost_state"] == "unresolved"

@pytest.mark.parametrize("prefix", ["prerequisite_failed:late", "review_threshold", "planning_allowance"])
def test_late_prefix_variants_unresolved(tmp_path, prefix):
    root = make_root(tmp_path)
    calls = {"n": 0}
    def late_spawn(ctx):
        calls["n"] += 1
        raise PilotError(prefix)
    batch = _run_stage_injected(root, 1, late_spawn, production=False,
                                control_root=tmp_path / "ctl-late")
    data = json.loads((batch / "batch.json").read_text())
    assert calls["n"] == 1
    assert len(data["cells"]) == 1
    assert data["cells"][0]["child_may_have_started"] is True
    assert data["unresolved_cost_state"] == "unresolved"
    res = _reservation(tmp_path, "ctl-late")
    assert res["unresolved_cost_state"] == "unresolved"
    assert res["child_may_have_started"] is True

def test_true_prespawn_gate_refusal_no_spawn(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    calls = {"n": 0}
    def counting_spawn(ctx):
        calls["n"] += 1
        return fake_spawn(ctx)
    monkeypatch.setattr(mp, "_cost_gate", lambda known, profile: "review_threshold")
    batch = _run_stage_injected(root, 1, counting_spawn, production=False,
                                control_root=tmp_path / "ctl-gate")
    data = json.loads((batch / "batch.json").read_text())
    assert calls["n"] == 0
    assert data["cells"][0]["cost"]["status"] == "not_started"
    assert data["cells"][0]["child_may_have_started"] is False
    assert data["unresolved_cost_state"] == "none"

def test_bad_raw_shapes_stop_before_cell2(tmp_path):
    for i, payload in enumerate(("[]", "null", "{broken")):
        root = make_root(tmp_path / f"case{i}")
        calls = {"n": 0}
        def bad_spawn(ctx, payload=payload):
            calls["n"] += 1
            (ctx["rundir"] / "run.json").write_text(payload, encoding="utf-8")
            return {"execution_status": "normal"}, {"cleanup_status": "verified"}
        batch = _run_stage_injected(root, 1, bad_spawn, production=False,
                                    control_root=tmp_path / f"ctl-bad{i}")
        data = json.loads((batch / "batch.json").read_text())
        assert data["status"] == "halted"
        assert calls["n"] == 1

def test_unexpected_exception_unresolved_and_stops(tmp_path):
    root = make_root(tmp_path)
    calls = {"n": 0}
    def boom(ctx):
        calls["n"] += 1
        raise RuntimeError("synthetic boom")
    batch = _run_stage_injected(root, 1, boom, production=False,
                                control_root=tmp_path / "ctl-boom")
    data = json.loads((batch / "batch.json").read_text())
    assert data["unresolved_cost_state"] == "unresolved"
    assert calls["n"] == 1

def test_concurrent_latch_admits_one(tmp_path):
    from harness.model_pilot import _reserve_attempt
    ctl = tmp_path / "ctl-concurrent"
    _reserve_attempt(ctl, 1, list(mft.STAGES[1]), "synthetic-test")
    with pytest.raises(PilotError):
        _reserve_attempt(ctl, 1, list(mft.STAGES[1]), "synthetic-test")

def test_export_preexisting_and_cross_stage_rejected(tmp_path):
    from harness.model_pilot import export_safe as _export
    batch = tmp_path / "batch"
    (batch / "cells").mkdir(parents=True)
    (batch / "batch.json").write_text(json.dumps({"stage": 1}), encoding="utf-8")
    (batch / "ledger.json").write_text("{}", encoding="utf-8")
    (batch / "summary.md").write_text("x", encoding="utf-8")
    (batch / "export").mkdir()
    sentinel = batch / "export" / "keep.txt"
    sentinel.write_text("old", encoding="utf-8")
    with pytest.raises(PilotError):
        _export(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    assert sentinel.read_text() == "old"
    with pytest.raises(PilotError):
        _export(batch, [{"cell_id": "s2-drop-ca-spark"}])

def test_export_scan_hit_withholds_flagged_bytes(tmp_path):
    from harness.model_pilot import export_safe as _export
    batch = tmp_path / "batch2"
    cell = batch / "cells" / "s1-seeded-ca-spark"
    cell.mkdir(parents=True)
    (batch / "batch.json").write_text(json.dumps({
        "stage": 1, "manifest_id": mft.MANIFEST_ID,
        "protocol_version": mft.PROTOCOL_VERSION, "status": "halted",
        "cells": [{"cell_id": "s1-seeded-ca-spark", "task_id": "m-clean-seeded",
                   "arm": "CA", "profile": "spark-medium"}]}), encoding="utf-8")
    (batch / "ledger.json").write_text("{}", encoding="utf-8")
    (batch / "summary.md").write_text("x", encoding="utf-8")
    (cell / "run.json").write_text("MODEL_API_KEY=SYNTHETIC-FLAGGED", encoding="utf-8")
    (cell / "evaluated.json").write_text("{}", encoding="utf-8")
    (cell / "secret-scan.txt").write_text("{}", encoding="utf-8")
    out = _export(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    names = {f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file()}
    assert names == {"halted-status.json", "manifest.json"}
    assert "SYNTHETIC-FLAGGED" not in (out / "halted-status.json").read_text()

def _cart_db(tmp_path, lines):
    from store import models
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "cart.db"
    models.init_and_seed(str(db), str(PROJECT_ROOT / "store" / "seed_catalog.json"))
    conn = models.connect(str(db))
    try:
        cart_id, sid, _ = models.get_or_create_session(conn, None)
        version = 0
        for sku, qty in lines.items():
            variant = models.get_variant(conn, sku)
            models.add_line_guarded(conn, cart_id, version, sku, qty, variant["price_cents"])
            version += 1
    finally:
        conn.close()
    return db, sid

def test_scoring_correct_wrong_missing_and_clarify(tmp_path):
    from harness import evaluate
    db, sid = _cart_db(tmp_path / "a", {"BTL-HIK-500-NAVYBLUE": 1})
    (tmp_path / "a").mkdir(exist_ok=True)
    out = evaluate.measure_final_db(str(db), {"BTL-HIK-500-NAVYBLUE": 1}, sid)
    assert out["measured_outcome"] == "correct"
    db2, sid2 = _cart_db(tmp_path / "b", {"BTL-HIK-750-STEEL": 1})
    out2 = evaluate.measure_final_db(str(db2), {"BTL-HIK-500-NAVYBLUE": 1}, sid2)
    assert out2["measured_outcome"] == "incorrect"
    out3 = evaluate.measure_final_db(str(tmp_path / "missing.db"),
                                     {"BTL-HIK-500-NAVYBLUE": 1}, "x")
    assert out3["measured_outcome"] == "unknown"
    # clarification: seed-exact persisted cart is correct, a write changes it
    db4, sid4 = _cart_db(tmp_path / "c", {"BTL-HIK-500-NAVY": 1})
    out4 = evaluate.measure_final_db(str(db4), {"BTL-HIK-500-NAVY": 1}, sid4)
    assert out4["measured_outcome"] == "correct"


def test_may_start_snapshot_visible_at_spawn(tmp_path):
    root = make_root(tmp_path)
    seen = {}
    def inspecting_spawn(ctx):
        if not seen:
            files = list((tmp_path / "ctl-may").rglob("reservation.json"))
            seen.update(json.loads(files[0].read_text()))
        return fake_spawn(ctx)
    _run_stage_injected(root, 1, inspecting_spawn, production=False,
                        control_root=tmp_path / "ctl-may")
    assert seen["state"] == "may_start"
    assert seen["child_may_have_started"] is True
    assert seen["unresolved_cost_state"] == "unresolved"
    assert seen["current_cell"] == "s1-seeded-ca-spark"

def test_interruption_at_spawn_leaves_may_start_blocked(tmp_path):
    root = make_root(tmp_path)
    def interrupting_spawn(ctx):
        raise SystemExit(3)
    with pytest.raises(SystemExit):
        _run_stage_injected(root, 1, interrupting_spawn, production=False,
                            control_root=tmp_path / "ctl-int")
    res = _reservation(tmp_path, "ctl-int")
    assert res["state"] == "may_start"
    assert res["unresolved_cost_state"] == "unresolved"
    with pytest.raises(PilotError):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-int",
                            out_root=tmp_path / "second")

def test_persistence_failure_before_may_start_no_spawn(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    calls = {"n": 0}
    def counting_spawn(ctx):
        calls["n"] += 1
        return fake_spawn(ctx)
    orig = mp._persist_attempt
    def failing(adir, state, name, **kw):
        if name == "may_start":
            raise OSError("synthetic persistence failure")
        return orig(adir, state, name, **kw)
    monkeypatch.setattr(mp, "_persist_attempt", failing)
    _run_stage_injected(root, 1, counting_spawn, production=False,
                        control_root=tmp_path / "ctl-persist")
    assert calls["n"] == 0

def test_rejected_accounting_retains_unresolved_diagnostic(tmp_path):
    root = make_root(tmp_path)
    calls = {"n": 0}
    def bad_accounting_spawn(ctx):
        calls["n"] += 1
        run = fake_run(ctx["spec"])
        # valid provider sidecar/cost, invalid consumed vs declared record
        run["budget"]["records"][0]["consumed"] = 1
        run["budget"]["consumed_tokens"] = 1
        (ctx["rundir"] / "run.json").write_text(json.dumps(run), encoding="utf-8")
        return run, {"cleanup_status": "verified", "timed_out": False}
    batch = _run_stage_injected(root, 1, bad_accounting_spawn, production=False,
                                control_root=tmp_path / "ctl-acct")
    data = json.loads((batch / "batch.json").read_text())
    assert calls["n"] == 1
    assert data["cells"][0]["accounting_ok"] is False
    assert data["unresolved_cost_state"] == "unresolved"
    assert data["cells"][0]["diagnostic_cost_usd"] is not None
    res = _reservation(tmp_path, "ctl-acct")
    assert res["unresolved_cost_state"] == "unresolved"
    ledger = json.loads((batch / "ledger.json").read_text())
    assert ledger["unresolved_cost_state"] == "unresolved"


def _make_batch_dir(tmp_path, recorded_ids, stage=1, status="halted"):
    from harness.model_pilot import export_safe_outcome, export_safe as _exp
    batch = tmp_path / "batchx"
    batch.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in recorded_ids:
        cell = next(c for c in mft.STAGES[stage] if c["cell_id"] == cid)
        rows.append({"cell_id": cid, "task_id": cell["task_id"],
                     "arm": cell["arm"], "profile": cell["profile"]})
        d = batch / "cells" / cid
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(json.dumps({"ok": cid}), encoding="utf-8")
        (d / "evaluated.json").write_text("{}", encoding="utf-8")
        (d / "secret-scan.txt").write_text("{}", encoding="utf-8")
    (batch / "batch.json").write_text(json.dumps({
        "stage": stage, "manifest_id": mft.MANIFEST_ID,
        "protocol_version": mft.PROTOCOL_VERSION, "status": status,
        "cells": rows}), encoding="utf-8")
    (batch / "ledger.json").write_text("{}", encoding="utf-8")
    (batch / "summary.md").write_text("x", encoding="utf-8")
    return batch

def test_export_matches_recorded_selection(tmp_path):
    from harness.model_pilot import export_safe as _exp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    with pytest.raises(PilotError):
        _exp(batch, [{"cell_id": "s1-seeded-ca-spark"}, {"cell_id": "s1-seeded-ca-wonder"}])
    with pytest.raises(PilotError):
        _exp(batch, [{"cell_id": "s1-seeded-ca-wonder"}])
    with pytest.raises(PilotError):
        _exp(batch, [{"cell_id": "s1-seeded-ca-spark"}, {"cell_id": "s1-seeded-ca-spark"}])
    with pytest.raises(PilotError):
        _exp(batch, [{"cell_id": "s2-drop-ca-spark"}])
    with pytest.raises(PilotError):
        _exp(batch, ["not-a-dict"])
    out = _exp(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    man = json.loads((out / "manifest.json").read_text())
    assert man["mode"] == "partial"
    assert man["selection"] == ["s1-seeded-ca-spark"]

def test_export_withheld_report_complete(tmp_path):
    from harness.model_pilot import export_safe as _exp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark", "s1-seeded-ca-wonder"])
    (batch / "cells" / "s1-seeded-ca-spark" / "run.json").write_text(
        "MODEL_API_KEY=FLAGGED", encoding="utf-8")
    (batch / "cells" / "s1-seeded-ca-wonder" / "evaluated.json").unlink()
    out = _exp(batch, [{"cell_id": "s1-seeded-ca-spark"},
                       {"cell_id": "s1-seeded-ca-wonder"}])
    names = {f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file()}
    assert names == {"halted-status.json", "manifest.json"}
    status = json.loads((out / "halted-status.json").read_text())
    assert "cells/s1-seeded-ca-spark/run.json" in status["withheld"]
    assert "cells/s1-seeded-ca-wonder/evaluated.json" in status["withheld"]
    for f in out.rglob("*"):
        if f.is_file():
            assert "FLAGGED" not in f.read_text(encoding="utf-8")

def test_export_modes_and_parent_diagnostic(tmp_path):
    from harness.model_pilot import export_safe as _exp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    cell_dir = batch / "cells" / "s1-seeded-ca-spark"
    (cell_dir / "run.json").unlink()
    (cell_dir / "parent-diagnostic.json").write_text(json.dumps(
        {"origin": "parent-diagnostic", "schema": 1}), encoding="utf-8")
    out = _exp(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    assert (out / "cells" / "s1-seeded-ca-spark" / "parent-diagnostic.json").is_file()
    man = json.loads((out / "manifest.json").read_text())
    assert man["files"]["batch.json"] == hashlib.sha256(
        (out / "batch.json").read_bytes()).hexdigest()

def test_export_preexisting_never_modified(tmp_path):
    from harness.model_pilot import export_safe as _exp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    (batch / "export").mkdir()
    sentinel = batch / "export" / "keep.txt"
    sentinel.write_text("old", encoding="utf-8")
    with pytest.raises(PilotError):
        _exp(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    assert sentinel.read_text() == "old"
    batch2 = _make_batch_dir(tmp_path / "second", ["s1-seeded-ca-spark"])
    (batch2 / ".export-staging").mkdir()
    with pytest.raises(PilotError):
        _exp(batch2, [{"cell_id": "s1-seeded-ca-spark"}])

def test_export_write_rename_failures_block(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    monkeypatch.setattr(mp.os, "rename", lambda *a, **kw: (_ for _ in ()).throw(OSError("synthetic rename failure")))
    with pytest.raises(PilotError):
        mp.export_safe(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    assert not (batch / "export").exists()

def test_owned_dangling_and_wrong_type_refused(tmp_path):
    from harness.model_pilot import export_safe as _exp, _validate_ancestor_chain, PilotError as _PE
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    # wrong type: cells is a file in a second batch
    batch2 = tmp_path / "batch3"
    batch2.mkdir()
    (batch2 / "cells").write_text("not a dir", encoding="utf-8")
    (batch2 / "batch.json").write_text(json.dumps({
        "stage": 1, "manifest_id": mft.MANIFEST_ID,
        "protocol_version": mft.PROTOCOL_VERSION, "status": "halted",
        "cells": [{"cell_id": "s1-seeded-ca-spark"}]}), encoding="utf-8")
    with pytest.raises(PilotError):
        _exp(batch2, [{"cell_id": "s1-seeded-ca-spark"}])
    # normal owned path accepted via ancestor chain (no raise)
    _validate_ancestor_chain(batch, batch / "cells" / "s1-seeded-ca-spark")

def test_owned_ancestor_link_escape_refused(tmp_path):
    import subprocess as _sp
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("outside", encoding="utf-8")
    link = tmp_path / "linked-cells"
    # Windows junction (no privilege needed); skip label if unavailable
    r = _sp.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("junction creation unavailable")
    from harness.model_pilot import _validate_ancestor_chain
    with pytest.raises(PilotError):
        _validate_ancestor_chain(tmp_path, link / "cell")
    assert sentinel.read_text() == "outside"
    assert not (outside / "cell").exists()


def _junction(link, target):
    import subprocess as _sp
    r = _sp.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True, text=True)
    return r.returncode == 0

def test_actual_reserve_attempt_junction_ancestor_refused(tmp_path):
    from harness.model_pilot import _reserve_attempt
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("outside", encoding="utf-8")
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    if not _junction(ctl / "attempts", outside):
        pytest.skip("junction creation unavailable")
    with pytest.raises(PilotError):
        _reserve_attempt(ctl, 1, list(mft.STAGES[1]), "synthetic-test")
    assert sentinel.read_text() == "outside"
    assert not (outside / "stage1").exists()

def test_actual_pinned_task_linked_tasks_dir_refused(tmp_path):
    root = make_root(tmp_path)
    outside = tmp_path / "outside-tasks"
    outside.mkdir()
    import shutil as _sh
    src_tasks = root / "eval" / "model" / "dev" / "tasks"
    for f in src_tasks.iterdir():
        _sh.copyfile(f, outside / f.name)
    _sh.rmtree(src_tasks)
    if not _junction(src_tasks, outside):
        pytest.skip("junction creation unavailable")
    with pytest.raises(PilotError):
        _read_pinned_task(root, "m-clean-seeded")

def test_stage_halted_publication_not_published(tmp_path):
    root = make_root(tmp_path)
    def flagged_spawn(ctx):
        run, info = fake_spawn(ctx)
        data = json.loads((ctx["rundir"] / "run.json").read_text())
        data["note"] = "MODEL_API_KEY=SYNTHETIC"
        (ctx["rundir"] / "run.json").write_text(json.dumps(data), encoding="utf-8")
        return data, info
    batch = _run_stage_injected(root, 1, flagged_spawn, production=False,
                                control_root=tmp_path / "ctl-flag")
    assert not (batch / "EXPORT_PATH.txt").exists()
    outcome = json.loads((batch / "EXPORT_OUTCOME.json").read_text())
    assert outcome["publication_status"] in ("halted_package", "export_failed_unapproved")
    res = _reservation(tmp_path, "ctl-flag")
    assert res["evidence_status"] == outcome["publication_status"]

def test_stage_export_rename_failure_no_success_marker(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    monkeypatch.setattr(mp.os, "rename",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("synthetic")))
    with pytest.raises(PilotError, match="export_publish_failed|export_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-ren")
    outs = list((root / "var").glob("model-pilot-stage1-*"))
    assert len(outs) == 1
    assert not (outs[0] / "export").exists()
    assert not (outs[0] / "EXPORT_PATH.txt").exists()
    res = _reservation(tmp_path, "ctl-ren")
    assert res["state"] == "blocked"
    assert res["evidence_status"].startswith("export_")
    latch = list((tmp_path / "ctl-ren" / "attempts" / "stage1").glob("stage.latch"))
    assert len(latch) == 1

def test_recorded_selection_multi_reorder_rejected(tmp_path):
    from harness.model_pilot import export_safe as _exp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark", "s1-seeded-ca-wonder"])
    with pytest.raises(PilotError):
        _exp(batch, [{"cell_id": "s1-seeded-ca-wonder"},
                     {"cell_id": "s1-seeded-ca-spark"}])
    out = _exp(batch, [{"cell_id": "s1-seeded-ca-spark"},
                       {"cell_id": "s1-seeded-ca-wonder"}])
    man = json.loads((out / "manifest.json").read_text())
    assert man["selection"] == ["s1-seeded-ca-spark", "s1-seeded-ca-wonder"]


def test_persist_attempt_tmp_regular_file_refused(tmp_path):
    from harness.model_pilot import _reserve_attempt, _persist_attempt
    ctl = tmp_path / "ctl-leaf"
    state = _reserve_attempt(ctl, 1, list(mft.STAGES[1]), "synthetic-test")
    adir = ctl / "attempts" / "stage1" / state["attempt_id"]
    (adir / "reservation.tmp").write_text("stale", encoding="utf-8")
    with pytest.raises(PilotError):
        _persist_attempt(adir, state, "running")

def test_persist_attempt_tmp_link_refused(tmp_path):
    from harness.model_pilot import _reserve_attempt, _persist_attempt
    ctl = tmp_path / "ctl-leaf-link"
    state = _reserve_attempt(ctl, 1, list(mft.STAGES[1]), "synthetic-test")
    adir = ctl / "attempts" / "stage1" / state["attempt_id"]
    outside = tmp_path / "outside.txt"
    outside.write_text("sentinel", encoding="utf-8")
    try:
        (adir / "reservation.tmp").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable - link case UNVERIFIED")
    with pytest.raises(PilotError):
        _persist_attempt(adir, state, "running")
    assert outside.read_text() == "sentinel"

def test_latch_any_type_blocks(tmp_path):
    # Directory latch variant (file/link variants below).
    from harness.model_pilot import _reserve_attempt
    ctl = tmp_path / "ctl-latch"
    stage_dir = ctl / "attempts" / "stage1"
    stage_dir.mkdir(parents=True)
    (stage_dir / "stage.latch").mkdir()
    with pytest.raises(PilotError):
        _reserve_attempt(ctl, 1, list(mft.STAGES[1]), "synthetic-test")

def test_latch_file_and_dangling_link_block(tmp_path):
    from harness.model_pilot import _reserve_attempt
    ctl = tmp_path / "ctl-latch-file"
    stage_dir = ctl / "attempts" / "stage1"
    stage_dir.mkdir(parents=True)
    (stage_dir / "stage.latch").write_text("x", encoding="utf-8")
    with pytest.raises(PilotError):
        _reserve_attempt(ctl, 1, list(mft.STAGES[1]), "synthetic-test")
    ctl2 = tmp_path / "ctl-latch-link"
    sd2 = ctl2 / "attempts" / "stage1"
    sd2.mkdir(parents=True)
    try:
        (sd2 / "stage.latch").symlink_to(tmp_path / "nonexistent-target")
    except OSError:
        pytest.skip("symlink creation unavailable - dangling case UNVERIFIED")
    with pytest.raises(PilotError):
        _reserve_attempt(ctl2, 1, list(mft.STAGES[1]), "synthetic-test")
    assert not (tmp_path / "nonexistent-target").exists()

def test_prepare_cell_linked_cells_refused(tmp_path):
    from harness.model_pilot import _prepare_cell
    root = make_root(tmp_path)
    out_root = tmp_path / "out"
    (out_root).mkdir()
    outside = tmp_path / "outside-cells"
    outside.mkdir()
    if not _junction(out_root / "cells", outside):
        pytest.skip("junction creation unavailable")
    with pytest.raises(PilotError):
        _prepare_cell(root, mft.STAGES[1][0], out_root, 0.0)
    assert not list(outside.iterdir())

def test_raw_traversal_rejected_before_normalization(tmp_path):
    from harness.model_pilot import _validate_ancestor_chain
    from pathlib import Path as _P
    with pytest.raises(PilotError):
        _validate_ancestor_chain(tmp_path, tmp_path / ".." / "escape")
    with pytest.raises(PilotError):
        _validate_ancestor_chain(tmp_path, _P(r"\\server\share\x"))

def test_stage_finalization_fields_coherent(tmp_path):
    root = make_root(tmp_path)
    batch = _run_stage_injected(root, 1, fake_spawn, production=False,
                                control_root=tmp_path / "ctl-fin")
    data = json.loads((batch / "batch.json").read_text())
    assert data["finalization"]["publication_status"] == "pending"
    assert data["finalization"]["candidate_mode"] == "normal"
    ledger = json.loads((batch / "ledger.json").read_text())
    assert ledger["finalization"] == data["finalization"]
    outcome = json.loads((batch / "EXPORT_OUTCOME.json").read_text())
    assert outcome["publication_status"] == "published"
    res = _reservation(tmp_path, "ctl-fin")
    assert res["evidence_status"] == "published"

def test_export_cleanup_failure_is_explicit(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    import shutil as _sh
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    monkeypatch.setattr(mp.os, "rename",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("synthetic")))
    monkeypatch.setattr(_sh, "rmtree",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("cleanup synthetic")))
    with pytest.raises(PilotError, match="cleanup"):
        mp.export_safe(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    assert not (batch / "export").exists()
    assert (batch / ".export-staging").is_dir()


def test_stage_metadata_write_failure_blocked(tmp_path, monkeypatch):
    import builtins as _bi
    root = make_root(tmp_path)
    orig_open = _bi.open
    def selective(file, mode="r", *a, **kw):
        if str(file).endswith("summary.md") and "x" in mode:
            raise OSError("synthetic summary failure")
        return orig_open(file, mode, *a, **kw)
    monkeypatch.setattr(_bi, "open", selective)
    with pytest.raises(PilotError, match="metadata_write_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-meta")
    res = _reservation(tmp_path, "ctl-meta")
    assert res["evidence_status"] == "metadata_write_failed"
    assert res["state"] == "blocked"

def test_stage_last_cell_halt_full_selection_candidate_normal(tmp_path):
    root = make_root(tmp_path)
    calls = {"n": 0}
    def last_bad(ctx):
        calls["n"] += 1
        if calls["n"] == 4:
            (ctx["rundir"] / "run.json").write_text(
                json.dumps({"execution_status": "timeout"}), encoding="utf-8")
            return {"execution_status": "timeout"}, {"cleanup_status": "terminated"}
        return fake_spawn(ctx)
    batch = _run_stage_injected(root, 1, last_bad, production=False,
                                control_root=tmp_path / "ctl-last")
    data = json.loads((batch / "batch.json").read_text())
    assert data["finalization"]["traversal_status"] == "halted"
    assert data["finalization"]["candidate_mode"] == "normal"

def test_stage_export_only_withholding_exact_halted_package(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    orig_scan = mp._scan_bytes
    def flag_summary(data):
        if b"Pilot stage" in data:
            return {"status": "ok", "sentinel_hits": 1, "categories": ["synthetic"]}
        return orig_scan(data)
    monkeypatch.setattr(mp, "_scan_bytes", flag_summary)
    batch = _run_stage_injected(root, 1, fake_spawn, production=False,
                                control_root=tmp_path / "ctl-withhold")
    outcome = json.loads((batch / "EXPORT_OUTCOME.json").read_text())
    assert outcome["publication_status"] == "halted_package"
    assert not (batch / "EXPORT_PATH.txt").exists()
    res = _reservation(tmp_path, "ctl-withhold")
    assert res["evidence_status"] == "halted_package"


def _selective_write_fail(monkeypatch, name):
    import builtins as _bi
    orig = _bi.open
    def selective(file, mode="r", *a, **kw):
        if str(file).endswith(name) and "x" in mode:
            raise OSError("synthetic write failure")
        return orig(file, mode, *a, **kw)
    monkeypatch.setattr(_bi, "open", selective)

def test_outcome_write_failure_after_publication_raises(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    _selective_write_fail(monkeypatch, "EXPORT_OUTCOME.json")
    with pytest.raises(PilotError, match="outcome_write_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-outcome")
    # published package preserved; find it under the stage out root
    exports = list((root / "var").rglob("manifest.json"))
    assert exports

def test_marker_write_failure_raises(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    _selective_write_fail(monkeypatch, "EXPORT_PATH.txt")
    with pytest.raises(PilotError, match="marker_write_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-marker")
    res = _reservation(tmp_path, "ctl-marker")
    assert res["evidence_status"] == "marker_write_failed"

def test_final_persist_failure_raises_no_marker(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    orig = mp._persist_attempt
    def failing(adir, state, name, **kw):
        if kw.get("evidence_status") == "published":
            raise OSError("synthetic persist failure")
        return orig(adir, state, name, **kw)
    monkeypatch.setattr(mp, "_persist_attempt", failing)
    with pytest.raises(PilotError, match="status_persistence_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-persistfail")
    assert not list((root / "var").rglob("EXPORT_PATH.txt"))

def test_reserve_attempt_linked_ancestor_above_anchor_refused(tmp_path):
    from harness.model_pilot import _reserve_attempt
    # Outside target lives at a SEPARATE location; junction path must not
    # exist before mklink.
    outside = tmp_path / "outside-target"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("outside", encoding="utf-8")
    jlink = tmp_path / "jlink"
    assert not jlink.exists()
    if not _junction(jlink, outside):
        pytest.skip("junction creation unavailable - UNVERIFIED")
    assert jlink.is_dir()  # junction confirmed before invoking the API
    # Lexical control path passes THROUGH the junction.
    ctl = jlink / "b" / "ctl"
    with pytest.raises(PilotError):
        _reserve_attempt(ctl, 1, list(mft.STAGES[1]), "synthetic-test")
    assert sentinel.read_text() == "outside"
    # The redirected attempt/stage location was never created outside.
    assert not (outside / "b" / "ctl" / "attempts").exists()
    assert not (outside / "attempts").exists()

def test_staging_replacement_directory_not_deleted(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    staging = batch / ".export-staging"
    def replacing_rename(src, dst):
        import shutil as _sh
        _sh.rmtree(src)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        repl = Path(src)
        repl.mkdir()
        (repl / "sentinel.txt").write_text("someone-else", encoding="utf-8")
        raise OSError("synthetic rename failure")
    monkeypatch.setattr(mp.os, "rename", replacing_rename)
    with pytest.raises(PilotError, match="cleanup_blocked|publish_failed|cleanup_failed"):
        mp.export_safe(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    assert (staging / "sentinel.txt").read_text() == "someone-else"
    assert not (batch / "export").exists()

def test_main_nonzero_on_pilot_error(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    monkeypatch.setattr(mp, "run_stage",
                        lambda *a, **kw: (_ for _ in ()).throw(PilotError("metadata_write_failed")))
    assert mp.main(["--stage", "1", "--live"]) == 2


def test_registration_identity_failure_normal_branch(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    orig_lstat = mp.os.lstat
    def bad_lstat(path, *a, **kw):
        if str(path).endswith(".export-staging"):
            try:
                orig_lstat(path)
            except OSError:
                return orig_lstat(path, *a, **kw)  # still missing: pre-create checks pass
            raise OSError("synthetic lstat failure")
        return orig_lstat(path, *a, **kw)
    monkeypatch.setattr(mp.os, "lstat", bad_lstat)
    with pytest.raises(PilotError, match="export_staging_identity_unavailable|export_cleanup"):
        mp.export_safe(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    # retained staging is explicitly unapproved: no final export
    assert not (batch / "export").exists()
    assert (batch / ".export-staging").is_dir()

def test_stage_cleanup_reason_preserved(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    import shutil as _sh
    root = make_root(tmp_path)
    monkeypatch.setattr(mp.os, "rename",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("synthetic")))
    monkeypatch.setattr(_sh, "rmtree",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("cleanup synthetic")))
    with pytest.raises(PilotError, match="export_cleanup"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-cleanreason")
    res = _reservation(tmp_path, "ctl-cleanreason")
    assert "cleanup" in res["evidence_status"]


def test_registration_identity_failure_halted_branch(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    (batch / "cells" / "s1-seeded-ca-spark" / "run.json").write_text(
        "MODEL_API_KEY=FLAGGED", encoding="utf-8")
    orig_lstat = mp.os.lstat
    def bad_lstat(path, *a, **kw):
        if str(path).endswith(".export-staging"):
            try:
                orig_lstat(path)
            except OSError:
                return orig_lstat(path, *a, **kw)
            raise OSError("synthetic lstat failure")
        return orig_lstat(path, *a, **kw)
    monkeypatch.setattr(mp.os, "lstat", bad_lstat)
    with pytest.raises(PilotError, match="export_staging_identity_unavailable"):
        mp.export_safe(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    assert not (batch / "export").exists()
    assert (batch / ".export-staging").is_dir()

def test_stage_unknown_exporter_reason_maps_generic(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    monkeypatch.setattr(mp, "export_safe_outcome",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            PilotError("export_SYNTHETIC-UNAPPROVED-TEXT")))
    with pytest.raises(PilotError, match="export_failed_unapproved"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-unknown")
    res = _reservation(tmp_path, "ctl-unknown")
    assert res["evidence_status"] == "export_failed_unapproved"
    assert "SYNTHETIC" not in json.dumps(res)

def test_stage_oserror_exporter_reason_maps_generic(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    monkeypatch.setattr(mp, "export_safe_outcome",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            OSError("export_fake-secret-text")))
    with pytest.raises(PilotError, match="export_failed_unapproved|export_io_failure"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-oserr")
    res = _reservation(tmp_path, "ctl-oserr")
    assert "fake-secret" not in json.dumps(res)


def _zero_lstat(monkeypatch, mp):
    import types as _t, stat as _stat
    orig = mp.os.lstat
    def fake(path, *a, **kw):
        if str(path).endswith(".export-staging"):
            return _t.SimpleNamespace(st_dev=1, st_ino=0, st_file_attributes=0,
                                      st_mode=_stat.S_IFDIR)
        return orig(path, *a, **kw)
    monkeypatch.setattr(mp.os, "lstat", fake)

def test_registration_zero_identity_normal_branch(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    _zero_lstat(monkeypatch, mp)
    with pytest.raises(PilotError, match="export_staging_identity_unavailable"):
        mp.export_safe(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    assert not (batch / "export").exists()
    assert (batch / ".export-staging").is_dir()

def test_registration_zero_identity_halted_branch(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    (batch / "cells" / "s1-seeded-ca-spark" / "run.json").write_text(
        "MODEL_API_KEY=FLAGGED", encoding="utf-8")
    _zero_lstat(monkeypatch, mp)
    with pytest.raises(PilotError, match="export_staging_identity_unavailable"):
        mp.export_safe(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    assert not (batch / "export").exists()
    assert (batch / ".export-staging").is_dir()

def test_stage_registration_failure_exact_reason(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    _zero_lstat(monkeypatch, mp)
    with pytest.raises(PilotError, match="export_staging_identity_unavailable"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-regstage")
    res = _reservation(tmp_path, "ctl-regstage")
    assert res["evidence_status"] == "export_staging_identity_unavailable"
    assert res["state"] == "blocked"

def test_export_mkdir_collision_unowned_untouched(tmp_path):
    from harness.model_pilot import export_safe as _exp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    staging = batch / ".export-staging"
    staging.mkdir()
    sentinel = staging / "keep.txt"
    sentinel.write_text("preexisting", encoding="utf-8")
    with pytest.raises(PilotError, match="export_staging_preexisting"):
        _exp(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    assert sentinel.read_text() == "preexisting"
    assert not (batch / "export").exists()

def test_stage_colon_prefix_maps_generic_everywhere(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    monkeypatch.setattr(mp, "export_safe_outcome",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            PilotError("export_blocked:SYNTHETIC-MARKER-123")))
    with pytest.raises(PilotError, match="export_failed_unapproved"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-colon")
    res = _reservation(tmp_path, "ctl-colon")
    assert res["evidence_status"] == "export_failed_unapproved"
    assert "SYNTHETIC-MARKER" not in json.dumps(res)


def _mkdir_fault(monkeypatch):
    from pathlib import Path as _P
    orig = _P.mkdir
    def selective(self, *a, **kw):
        if self.name == ".export-staging":
            self.mkdir.__self__ if False else None
            # create the unrelated synthetic object, then fail creation
            orig(self, parents=True, exist_ok=True)
            (self / "sentinel.txt").write_text("unowned", encoding="utf-8")
            raise FileExistsError("synthetic mkdir collision")
        return orig(self, *a, **kw)
    monkeypatch.setattr(_P, "mkdir", selective)

def test_export_mkdir_fault_after_checks_normal(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    _mkdir_fault(monkeypatch)
    with pytest.raises(PilotError, match="export_staging_create_failed|export_publish_failed|export_cleanup"):
        mp.export_safe(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    staging = batch / ".export-staging"
    assert (staging / "sentinel.txt").read_text() == "unowned"
    assert not (batch / "export").exists()

def test_export_mkdir_fault_after_checks_halted(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    batch = _make_batch_dir(tmp_path, ["s1-seeded-ca-spark"])
    (batch / "cells" / "s1-seeded-ca-spark" / "run.json").write_text(
        "MODEL_API_KEY=FLAGGED", encoding="utf-8")
    _mkdir_fault(monkeypatch)
    with pytest.raises(PilotError):
        mp.export_safe(batch, [{"cell_id": "s1-seeded-ca-spark"}])
    staging = batch / ".export-staging"
    assert (staging / "sentinel.txt").read_text() == "unowned"
    assert not (batch / "export").exists()

@pytest.mark.parametrize("prefix", ["export_blocked:", "export_missing:", "export_cell_not_allowed:"])
def test_stage_colon_prefix_outputs_sanitized(tmp_path, monkeypatch, prefix):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    monkeypatch.setattr(mp, "export_safe_outcome",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            PilotError(prefix + "SYNTHETIC-MARKER-456")))
    with pytest.raises(PilotError, match="export_failed_unapproved"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-colonx")
    res = _reservation(tmp_path, "ctl-colonx")
    assert res["evidence_status"] == "export_failed_unapproved"
    # synthetic suffix absent from every generated file in stage + control
    stage_outs = list((root / "var").rglob("*"))
    checked = 0
    for f in stage_outs:
        if f.is_file() and f.suffix in (".json", ".md", ".txt"):
            checked += 1
            assert "SYNTHETIC-MARKER" not in f.read_text(
                encoding="utf-8", errors="ignore")
    assert "SYNTHETIC-MARKER" not in json.dumps(res)
    assert checked > 0


def test_inspect_leaf_classification(tmp_path):
    from harness.model_pilot import _inspect_leaf
    assert _inspect_leaf(tmp_path / "missing.txt") == "missing"
    f = tmp_path / "f.txt"; f.write_text("x", encoding="utf-8")
    assert _inspect_leaf(f) == "regular_file"
    assert _inspect_leaf(tmp_path) == "directory"
    dangling = tmp_path / "dangling.txt"
    try:
        dangling.symlink_to(tmp_path / "no-such-target.txt")
    except OSError:
        pytest.skip("symlink creation unavailable - UNVERIFIED")
    assert _inspect_leaf(dangling) == "prohibited_link"

def test_inspect_leaf_metadata_error_refuses(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    target = tmp_path / "leaf.txt"
    orig = mp.os.lstat
    def bad(path, *a, **kw):
        if str(path).endswith("leaf.txt"):
            raise PermissionError("synthetic metadata failure")
        return orig(path, *a, **kw)
    monkeypatch.setattr(mp.os, "lstat", bad)
    with pytest.raises(PilotError, match="metadata_unverifiable"):
        mp._write_fresh_leaf(target, "data", anchor=tmp_path, owner={})
    assert not target.exists()

def test_stage_summary_leaf_preexisting_refused_before_write(tmp_path):
    root = make_root(tmp_path)
    writes = {"n": 0}
    def sentinel_spawn(ctx):
        run, info = fake_spawn(ctx)
        out_root = ctx["rundir"].parent.parent
        (out_root / "summary.md").write_text("SENTINEL-SUMMARY", encoding="utf-8")
        return run, info
    with pytest.raises(PilotError, match="metadata_write_failed"):
        _run_stage_injected(root, 1, sentinel_spawn, production=False,
                            control_root=tmp_path / "ctl-leaf")
    outs = list((root / "var").glob("model-pilot-stage1-*"))
    assert (outs[0] / "summary.md").read_text() == "SENTINEL-SUMMARY"
    res = _reservation(tmp_path, "ctl-leaf")
    assert res["evidence_status"] == "metadata_write_failed"

def test_stage_out_root_dangling_refused(tmp_path):
    root = make_root(tmp_path)
    out_root = root / "var" / "model-pilot-stage1-dangling"
    try:
        out_root.symlink_to(tmp_path / "no-such-dir", target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable - UNVERIFIED")
    calls = {"n": 0}
    def counting(ctx):
        calls["n"] += 1
        return fake_spawn(ctx)
    with pytest.raises(PilotError, match="out_root_preexisting"):
        _run_stage_injected(root, 1, counting, production=False,
                            control_root=tmp_path / "ctl-dangle", out_root=out_root)
    assert calls["n"] == 0
    assert not (tmp_path / "no-such-dir").exists()

def test_stage_outcome_leaf_preexisting_preserves_published(tmp_path):
    root = make_root(tmp_path)
    def outcome_spawn(ctx):
        run, info = fake_spawn(ctx)
        out_root = ctx["rundir"].parent.parent
        (out_root / "EXPORT_OUTCOME.json").write_text("SENTINEL-OUTCOME", encoding="utf-8")
        return run, info
    with pytest.raises(PilotError, match="outcome_write_failed"):
        _run_stage_injected(root, 1, outcome_spawn, production=False,
                            control_root=tmp_path / "ctl-outcomeleaf")
    outs = list((root / "var").glob("model-pilot-stage1-*"))
    assert (outs[0] / "EXPORT_OUTCOME.json").read_text() == "SENTINEL-OUTCOME"
    assert not (outs[0] / "EXPORT_PATH.txt").exists()
    res = _reservation(tmp_path, "ctl-outcomeleaf")
    assert res["state"] == "blocked"
    assert res["evidence_status"] == "outcome_write_failed"
    man = outs[0] / "export" / "manifest.json"
    assert man.is_file()
    import hashlib as _hl
    data = json.loads(man.read_text())
    assert data["files"]["batch.json"] == _hl.sha256(
        (outs[0] / "export" / "batch.json").read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# A-F: actual write-target operations (parent chain, exclusive create,
# registered update, outcome/marker blocked-state propagation).
# ---------------------------------------------------------------------------

def _out_root_for(root):
    outs = list((root / "var").glob("model-pilot-stage1-*"))
    assert len(outs) == 1
    return outs[0]


def _assert_published_manifest_hashes(out_root):
    manifest = json.loads(
        (out_root / "export" / "manifest.json").read_text(encoding="utf-8"))
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256(
            (out_root / "export" / rel).read_bytes()).hexdigest() == digest


def _plant_in_spawn(name, content):
    planted = {"done": False}
    def spawn(ctx):
        run, info = fake_spawn(ctx)
        if not planted["done"]:
            (ctx["rundir"].parent.parent / name).write_text(
                content, encoding="utf-8")
            planted["done"] = True
        return run, info
    return spawn


def _write_spy(monkeypatch):
    import builtins as _bi
    seen = []
    orig = _bi.open
    def spy(file, mode="r", *a, **kw):
        if isinstance(file, (str, os.PathLike)) \
                and any(m in mode for m in ("w", "a", "x", "+")):
            seen.append((Path(file), mode))
        return orig(file, mode, *a, **kw)
    monkeypatch.setattr(_bi, "open", spy)
    return seen


def _replace_regular(path, content):
    tmp = path.with_name(path.name + ".replacement")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def test_stage_root_metadata_error_before_reservation_no_latch(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    out_root = tmp_path / "stage-root-a"
    orig = mp.os.lstat
    def bad(path, *a, **kw):
        if os.path.abspath(str(path)) == os.path.abspath(str(out_root)):
            raise PermissionError("synthetic parent metadata failure")
        return orig(path, *a, **kw)
    monkeypatch.setattr(mp.os, "lstat", bad)
    with pytest.raises(PilotError, match="metadata_unverifiable"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-a-root",
                            out_root=out_root)
    # Before reservation no latch is invented.
    assert not (tmp_path / "ctl-a-root").exists()
    assert not list(tmp_path.rglob("stage.latch"))


def test_stage_root_mkdir_failure_after_reservation_is_blocked(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    out_root = tmp_path / "stage-root-b"
    orig = Path.mkdir
    def bad(self, *a, **kw):
        if os.path.abspath(str(self)) == os.path.abspath(str(out_root)):
            raise OSError("synthetic mkdir failure")
        return orig(self, *a, **kw)
    monkeypatch.setattr(Path, "mkdir", bad)
    with pytest.raises(PilotError, match="stage_root_create_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-a-mkdir",
                            out_root=out_root)
    res = _reservation(tmp_path, "ctl-a-mkdir")
    assert res["state"] == "blocked"
    assert res["evidence_status"] == "stage_root_create_failed"
    assert list((tmp_path / "ctl-a-mkdir").rglob("stage.latch"))


def test_stage_root_wrong_type_ancestor_refused_before_reservation(tmp_path):
    root = make_root(tmp_path)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(PilotError, match="metadata_unverifiable|path_|anchor_"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-a-wrong",
                            out_root=blocker / "stage")
    assert not (tmp_path / "ctl-a-wrong").exists()


def test_stage_root_existing_target_refused_and_preserved(tmp_path):
    root = make_root(tmp_path)
    out_root = tmp_path / "stage-existing"
    out_root.mkdir()
    (out_root / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(PilotError, match="out_root_preexisting"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-a-existing",
                            out_root=out_root)
    assert (out_root / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "ctl-a-existing").exists()


@pytest.mark.parametrize("leaf", ["batch.json", "ledger.json", "summary.md"])
def test_stage_metadata_leaf_metadata_error_refused(tmp_path, monkeypatch, leaf):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    ctl_name = "ctl-a-" + leaf.split(".")[0]
    orig = mp.os.lstat
    def bad(path, *a, **kw):
        p = str(path)
        if Path(p).name == leaf and "model-pilot" in p:
            raise PermissionError("synthetic leaf metadata failure")
        return orig(path, *a, **kw)
    monkeypatch.setattr(mp.os, "lstat", bad)
    seen = _write_spy(monkeypatch)
    with pytest.raises(PilotError, match="metadata_write_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / ctl_name)
    out_root = _out_root_for(root)
    assert not (out_root / leaf).exists()
    res = _reservation(tmp_path, ctl_name)
    assert res["state"] == "blocked"
    assert res["evidence_status"] == "metadata_write_failed"
    # Narrow spy: no write-mode open escaped the synthetic roots.
    assert seen
    for path, _mode in seen:
        assert Path(os.path.abspath(str(path))).is_relative_to(tmp_path)


@pytest.mark.parametrize("leaf,category", [
    ("batch.json", "metadata_write_failed"),
    ("ledger.json", "metadata_write_failed"),
    ("summary.md", "metadata_write_failed"),
    ("EXPORT_OUTCOME.json", "outcome_write_failed"),
    ("EXPORT_PATH.txt", "marker_write_failed"),
])
def test_fresh_leaf_validation_gap_collision_no_adoption(
        tmp_path, monkeypatch, leaf, category):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    sentinel = "SENTINEL-" + leaf
    ctl_name = "ctl-b-" + leaf.replace(".", "-")
    spawn = _plant_in_spawn(leaf, sentinel)
    orig_inspect = mp._inspect_leaf
    def fake_inspect(path):
        if Path(str(path)).name == leaf and "model-pilot" in str(path) \
                and Path(str(path)).exists():
            return "missing"  # validation gap: exclusive open must still refuse
        return orig_inspect(path)
    monkeypatch.setattr(mp, "_inspect_leaf", fake_inspect)
    with pytest.raises(PilotError, match=category):
        _run_stage_injected(root, 1, spawn, production=False,
                            control_root=tmp_path / ctl_name)
    out_root = _out_root_for(root)
    assert (out_root / leaf).read_text(encoding="utf-8") == sentinel
    res = _reservation(tmp_path, ctl_name)
    assert res["state"] == "blocked"
    assert res["evidence_status"] == category
    if leaf != "EXPORT_PATH.txt":
        assert not (out_root / "EXPORT_PATH.txt").exists()


def test_fresh_creation_is_exclusive_and_registers_identity(tmp_path):
    from harness.model_pilot import _write_fresh_leaf
    owner = {}
    for name in ("batch.json", "ledger.json", "summary.md",
                 "EXPORT_OUTCOME.json", "EXPORT_PATH.txt"):
        target = tmp_path / name
        _write_fresh_leaf(target, name, anchor=tmp_path, owner=owner)
        st = os.lstat(target)
        assert owner[os.path.abspath(str(target))] == (st.st_dev, st.st_ino)
        with pytest.raises(PilotError, match="leaf_preexisting"):
            _write_fresh_leaf(target, "second", anchor=tmp_path, owner=owner)
        assert target.read_text(encoding="utf-8") == name


def test_failure_rewrite_requires_registered_identity(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    captured = {}
    def failing_export(batch_dir, cells):
        target = Path(batch_dir) / "batch.json"
        captured["before"] = os.lstat(target)
        _replace_regular(target, "REPLACEMENT-BATCH")
        captured["after"] = os.lstat(target)
        raise PilotError("export_blocked")
    monkeypatch.setattr(mp, "export_safe_outcome", failing_export)
    with pytest.raises(PilotError, match="failure_state_write_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-c-replace")
    out_root = _out_root_for(root)
    assert (out_root / "batch.json").read_text(encoding="utf-8") \
        == "REPLACEMENT-BATCH"
    assert (captured["before"].st_dev, captured["before"].st_ino) \
        != (captured["after"].st_dev, captured["after"].st_ino)
    res = _reservation(tmp_path, "ctl-c-replace")
    assert res["state"] == "blocked"
    assert res["evidence_status"] == "failure_state_write_failed"
    assert res["known_cost_usd"] > 0.0      # truthful accumulated cost preserved
    assert res["unknown_cost_calls"] == 0
    assert not (out_root / "EXPORT_PATH.txt").exists()
    assert list((tmp_path / "ctl-c-replace").rglob("stage.latch"))


def test_failure_rewrite_wrong_type_refused(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    def failing_export(batch_dir, cells):
        target = Path(batch_dir) / "batch.json"
        target.unlink()
        target.mkdir()
        (target / "sentinel.txt").write_text("keep", encoding="utf-8")
        raise PilotError("export_blocked")
    monkeypatch.setattr(mp, "export_safe_outcome", failing_export)
    with pytest.raises(PilotError, match="failure_state_write_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-c-wrongtype")
    out_root = _out_root_for(root)
    assert (out_root / "batch.json").is_dir()
    assert (out_root / "batch.json" / "sentinel.txt").read_text(
        encoding="utf-8") == "keep"
    res = _reservation(tmp_path, "ctl-c-wrongtype")
    assert res["state"] == "blocked"
    assert res["evidence_status"] == "failure_state_write_failed"


def test_failure_rewrite_link_refused(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    outside = tmp_path / "outside-batch.txt"
    outside.write_text("outside", encoding="utf-8")
    def failing_export(batch_dir, cells):
        target = Path(batch_dir) / "batch.json"
        target.unlink()
        try:
            target.symlink_to(outside)
        except OSError:
            pytest.skip("symlink creation unavailable - UNVERIFIED")
        raise PilotError("export_blocked")
    monkeypatch.setattr(mp, "export_safe_outcome", failing_export)
    with pytest.raises(PilotError, match="failure_state_write_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-c-link")
    out_root = _out_root_for(root)
    assert outside.read_text(encoding="utf-8") == "outside"
    assert (out_root / "batch.json").is_symlink()
    res = _reservation(tmp_path, "ctl-c-link")
    assert res["state"] == "blocked"
    assert res["evidence_status"] == "failure_state_write_failed"


def test_failure_rewrite_unverifiable_metadata_refused(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    flag = {"on": False}
    captured = {}
    orig = mp.os.lstat
    def bad(path, *a, **kw):
        p = str(path)
        if flag["on"] and Path(p).name == "batch.json" and "model-pilot" in p:
            raise PermissionError("synthetic unverifiable batch")
        return orig(path, *a, **kw)
    def failing_export(batch_dir, cells):
        target = Path(batch_dir) / "batch.json"
        captured["batch"] = target.read_text(encoding="utf-8")
        flag["on"] = True
        raise PilotError("export_blocked")
    monkeypatch.setattr(mp.os, "lstat", bad)
    monkeypatch.setattr(mp, "export_safe_outcome", failing_export)
    with pytest.raises(PilotError, match="failure_state_write_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-c-unver")
    out_root = _out_root_for(root)
    assert (out_root / "batch.json").read_text(encoding="utf-8") \
        == captured["batch"]
    res = _reservation(tmp_path, "ctl-c-unver")
    assert res["state"] == "blocked"
    assert res["evidence_status"] == "failure_state_write_failed"


def test_outcome_metadata_error_routes_blocked(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    orig = mp.os.lstat
    def bad(path, *a, **kw):
        p = str(path)
        if Path(p).name == "EXPORT_OUTCOME.json" and "model-pilot" in p:
            raise PermissionError("synthetic outcome metadata failure")
        return orig(path, *a, **kw)
    monkeypatch.setattr(mp.os, "lstat", bad)
    with pytest.raises(PilotError, match="outcome_write_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-d-meta")
    out_root = _out_root_for(root)
    assert not (out_root / "EXPORT_OUTCOME.json").exists()
    assert not (out_root / "EXPORT_PATH.txt").exists()
    _assert_published_manifest_hashes(out_root)
    res = _reservation(tmp_path, "ctl-d-meta")
    assert res["state"] == "blocked"
    assert res["evidence_status"] == "outcome_write_failed"


def test_marker_leaf_preexisting_routes_blocked_not_complete(tmp_path):
    root = make_root(tmp_path)
    spawn = _plant_in_spawn("EXPORT_PATH.txt", "SENTINEL-MARKER")
    with pytest.raises(PilotError, match="marker_write_failed"):
        _run_stage_injected(root, 1, spawn, production=False,
                            control_root=tmp_path / "ctl-e-leaf")
    out_root = _out_root_for(root)
    assert (out_root / "EXPORT_PATH.txt").read_text(
        encoding="utf-8") == "SENTINEL-MARKER"
    _assert_published_manifest_hashes(out_root)
    res = _reservation(tmp_path, "ctl-e-leaf")
    assert res["state"] == "blocked"      # never stale 'complete'
    assert res["evidence_status"] == "marker_write_failed"
    batch = json.loads((out_root / "batch.json").read_text(encoding="utf-8"))
    assert res["known_cost_usd"] == batch["known_cost_usd_estimated"]
    assert list((tmp_path / "ctl-e-leaf").rglob("stage.latch"))


def test_marker_metadata_error_routes_blocked(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    orig = mp.os.lstat
    def bad(path, *a, **kw):
        p = str(path)
        if Path(p).name == "EXPORT_PATH.txt" and "model-pilot" in p:
            raise PermissionError("synthetic marker metadata failure")
        return orig(path, *a, **kw)
    monkeypatch.setattr(mp.os, "lstat", bad)
    with pytest.raises(PilotError, match="marker_write_failed"):
        _run_stage_injected(root, 1, fake_spawn, production=False,
                            control_root=tmp_path / "ctl-e-meta")
    out_root = _out_root_for(root)
    assert not (out_root / "EXPORT_PATH.txt").exists()
    _assert_published_manifest_hashes(out_root)
    res = _reservation(tmp_path, "ctl-e-meta")
    assert res["state"] == "blocked"
    assert res["evidence_status"] == "marker_write_failed"


def test_registered_update_positive_and_replacement_refused(tmp_path):
    from harness.model_pilot import _write_fresh_leaf, _write_update_leaf
    owner = {}
    target = tmp_path / "batch.json"
    _write_fresh_leaf(target, "one", anchor=tmp_path, owner=owner)
    first = os.lstat(target)
    _write_update_leaf(target, "two", anchor=tmp_path, owner=owner)
    second = os.lstat(target)
    assert target.read_text(encoding="utf-8") == "two"
    assert (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)
    _replace_regular(target, "replacement")
    with pytest.raises(PilotError, match="owned_identity_mismatch"):
        _write_update_leaf(target, "three", anchor=tmp_path, owner=owner)
    assert target.read_text(encoding="utf-8") == "replacement"
    with pytest.raises(PilotError, match="update_not_owned"):
        _write_update_leaf(tmp_path / "other.json", "x",
                           anchor=tmp_path, owner=owner)


# ---------------------------------------------------------------------------
# Scoring independence (A1/A2) and clarification behavior (B1-B4) through the
# real _run_stage_injected cell path. These tests use the unchanged scoring
# semantics in harness.model_metrics / harness.evaluate and only assert the
# fields those modules already attach.
# ---------------------------------------------------------------------------

def _persisted_spawn(*, declared, outcome_kind="proceed", exec_items=(),
                     completed=None, mutations=(), db_lines=(),
                     drop_db=False, cookie=None):
    """Synthetic worker that persists a cart and records a run.

    Persisted lines are read back from the run database with the unchanged
    snapshots.read_cart, so final_persisted reflects stored state rather than
    an agent claim about it.
    """
    def spawn(ctx):
        rundir = ctx["rundir"]
        run = fake_run(ctx["spec"])
        sid = None
        if drop_db:
            (rundir / "run.db").unlink()
            run["final_persisted"] = {"lines": None, "resolution": "error",
                                      "version": None, "total_cents": None}
        else:
            from store import models
            conn = models.connect(str(rundir / "run.db"))
            try:
                cart_id, sid, _created = models.get_or_create_session(conn, None)
                version = 0
                for sku, qty in db_lines:
                    variant = models.get_variant(conn, sku)
                    assert variant is not None, sku
                    models.add_line_guarded(conn, cart_id, version, sku, qty,
                                            variant["price_cents"])
                    version += 1
            finally:
                conn.close()
            from harness.snapshots import read_cart
            cart = read_cart(str(rundir / "run.db"), sid)
            run["final_persisted"] = {"lines": cart["lines"],
                                      "resolution": cart["resolution"],
                                      "version": cart["version"],
                                      "total_cents": cart["total_cents"]}
        run["declared_outcome"] = declared
        run["outcome_kind"] = outcome_kind
        run["executable_intent"] = {
            "items": [{"sku": s, "quantity": q} for s, q in exec_items],
            "budget_max_cents": None,
            "substitution_allowed": False,
        }
        run["completed_items"] = list(
            completed if completed is not None else range(len(exec_items)))
        run["mutations"] = [dict(m) for m in mutations]
        if sid is not None or cookie is not None:
            run["final_cookie"] = sid if cookie is None else cookie
        (rundir / "run.json").write_text(json.dumps(run, indent=2),
                                         encoding="utf-8")
        return run, {"cleanup_status": "verified", "timed_out": False}
    return spawn


def _dispatch(mapping):
    def spawn(ctx):
        return mapping.get(ctx["spec"]["pilot_cell_id"], fake_spawn)(ctx)
    return spawn


def _evaluated_for(batch, cell_id):
    data = json.loads(
        (Path(batch) / "cells" / cell_id / "evaluated.json")
        .read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _row_for(batch, cell_id):
    data = json.loads((Path(batch) / "batch.json").read_text(encoding="utf-8"))
    return data, next(c for c in data["cells"] if c["cell_id"] == cell_id)


def test_a1_declared_complete_contradicted_by_persisted_state(tmp_path):
    root = make_root(tmp_path)
    # Persisted cart keeps only the seed and is missing the authored item,
    # while the worker declares COMPLETE and claims to have completed it.
    spawn = _dispatch({
        "s1-seeded-ca-spark": _persisted_spawn(
            declared="COMPLETE", outcome_kind="proceed",
            exec_items=[("BTL-HIK-750-STEEL", 1)], completed=[0],
            db_lines=[("BTL-HIK-500-NAVY", 1)]),
    })
    batch = _run_stage_injected(root, 1, spawn, production=False,
                                control_root=tmp_path / "ctl-a1")
    ev = _evaluated_for(batch, "s1-seeded-ca-spark")
    assert ev["declared_outcome"] == "COMPLETE"
    assert ev["measured_outcome"] == "incorrect"
    assert ev["physical_correct"] is False
    assert ev["task_success"] is False
    # Execution itself completed; only independent measurement blocks success.
    assert ev["completed_execution"] is True
    assert ev["task_appropriate"] is False
    data, row = _row_for(batch, "s1-seeded-ca-spark")
    assert row["measured_outcome"] == "incorrect"
    assert row["declared_outcome"] == "COMPLETE"
    readout = data["scoring_readout"]["profiles"]["spark-medium"]
    assert readout["declared_complete"]["known_false"] >= 1


def test_a2_missing_persisted_db_measured_unknown_not_success(tmp_path):
    root = make_root(tmp_path)
    spawn = _dispatch({
        "s1-ambig-ca-spark": _persisted_spawn(
            declared="COMPLETE", outcome_kind="proceed",
            exec_items=[("BTL-HIK-750-STEEL", 1)], completed=[0],
            drop_db=True),
    })
    batch = _run_stage_injected(root, 1, spawn, production=False,
                                control_root=tmp_path / "ctl-a2")
    ev = _evaluated_for(batch, "s1-ambig-ca-spark")
    assert ev["measured_outcome"] == "unknown"
    assert ev["measured_resolution"] == "error"
    assert ev["declared_outcome"] == "COMPLETE"
    assert ev["physical_correct"] is False
    assert ev["task_success"] is False
    assert ev["task_appropriate"] is False
    assert ev["correct_clarification"] is False
    data, row = _row_for(batch, "s1-ambig-ca-spark")
    assert row["measured_outcome"] == "unknown"
    # Existing behavior: measurement "unknown" alone does not become a halt
    # and is never cost-resolved, and it is not counted as verified.
    assert row["halt"] is None
    assert row["unresolved_cost_state"] == "none"
    readout = data["scoring_readout"]["profiles"]["spark-medium"]
    assert readout["declared_complete"]["verified"] == 0
    assert readout["declared_complete"]["unverified"] >= 1


def test_b1_underdetermined_guess_completes_without_clarification(tmp_path):
    root = make_root(tmp_path)
    spawn = _dispatch({
        "s1-ambig-ca-wonder": _persisted_spawn(
            declared="COMPLETE", outcome_kind="proceed",
            exec_items=[("BTL-HIK-750-STEEL", 1)], completed=[0],
            mutations=[{"sku": "BTL-HIK-750-STEEL", "quantity": 1,
                        "corrective": False}],
            db_lines=[("BTL-HIK-500-NAVY", 1),
                      ("BTL-HIK-750-STEEL", 1)]),
    })
    batch = _run_stage_injected(root, 1, spawn, production=False,
                                control_root=tmp_path / "ctl-b1")
    ev = _evaluated_for(batch, "s1-ambig-ca-wonder")
    assert ev["declared_outcome"] == "COMPLETE"
    assert ev["correct_clarification"] is False
    assert ev["interpretation_outcome"] == "intent_incorrect"
    assert ev["task_appropriate"] is False
    # The guessing write behavior is visible in the record.
    assert ev["agent_write_count"] == 1
    assert ev["seeds_preserved"] is False


def test_b2_underdetermined_genuine_clarification(tmp_path):
    root = make_root(tmp_path)
    spawn = _dispatch({
        "s1-ambig-ca-spark": _persisted_spawn(
            declared="CLARIFY", outcome_kind="semantic_clarify",
            db_lines=[("BTL-HIK-500-NAVY", 1)]),
    })
    batch = _run_stage_injected(root, 1, spawn, production=False,
                                control_root=tmp_path / "ctl-b2")
    ev = _evaluated_for(batch, "s1-ambig-ca-spark")
    assert ev["execution_status"] == "normal"
    assert ev["outcome_kind"] == "semantic_clarify"
    assert ev["declared_outcome"] == "CLARIFY"
    assert ev["agent_write_count"] == 0
    assert ev["seeds_preserved"] is True
    assert ev["correct_clarification"] is True
    assert ev["interpretation_outcome"] == "clarify_appropriate"
    assert ev["task_appropriate"] is True


def test_b3_resolvable_task_clarification_unnecessary(tmp_path):
    root = make_root(tmp_path)
    spawn = _dispatch({
        "s1-seeded-ca-spark": _persisted_spawn(
            declared="CLARIFY", outcome_kind="semantic_clarify",
            db_lines=[("BTL-HIK-500-NAVY", 1)]),
    })
    batch = _run_stage_injected(root, 1, spawn, production=False,
                                control_root=tmp_path / "ctl-b3")
    ev = _evaluated_for(batch, "s1-seeded-ca-spark")
    assert ev["task_class"] == "resolvable"
    # Only the interpretation/appropriateness outcome is in scope here; the
    # unchanged semantics also expose correct_clarification independently.
    assert ev["interpretation_outcome"] == "clarify_unnecessary"
    assert ev["task_appropriate"] is False
    assert ev["task_success"] is False


def test_b4_declared_clarify_contradicted_by_writes(tmp_path):
    root = make_root(tmp_path)
    spawn = _dispatch({
        "s1-ambig-ca-wonder": _persisted_spawn(
            declared="CLARIFY", outcome_kind="semantic_clarify",
            mutations=[{"sku": "BTL-HIK-750-STEEL", "quantity": 1,
                        "corrective": False}],
            db_lines=[("BTL-HIK-500-NAVY", 1),
                      ("BTL-HIK-750-STEEL", 1)]),
    })
    batch = _run_stage_injected(root, 1, spawn, production=False,
                                control_root=tmp_path / "ctl-b4")
    ev = _evaluated_for(batch, "s1-ambig-ca-wonder")
    assert ev["declared_outcome"] == "CLARIFY"
    assert ev["outcome_kind"] == "semantic_clarify"
    assert ev["agent_write_count"] == 1
    assert ev["seeds_preserved"] is False
    assert ev["correct_clarification"] is False
    assert ev["task_appropriate"] is False


def test_scoring_readout_per_profile_frozen_in_batch_and_summary(tmp_path):
    from harness.model_pilot import SCORING_READOUT_LABEL
    root = make_root(tmp_path)
    spawn = _dispatch({
        # resolvable, measured correct, complete execution
        "s1-seeded-ca-spark": _persisted_spawn(
            declared="COMPLETE", outcome_kind="proceed",
            exec_items=[("BTL-HIK-750-STEEL", 1)], completed=[0],
            db_lines=[("BTL-HIK-500-NAVY", 1),
                      ("BTL-HIK-750-STEEL", 1)]),
        # resolvable, declared COMPLETE, persisted state contradicts
        "s1-seeded-ca-wonder": _persisted_spawn(
            declared="COMPLETE", outcome_kind="proceed",
            exec_items=[("BTL-HIK-750-STEEL", 1)], completed=[0],
            db_lines=[("BTL-HIK-500-NAVY", 1)]),
        # declared COMPLETE but the persisted DB is missing
        "s1-ambig-ca-wonder": _persisted_spawn(
            declared="COMPLETE", outcome_kind="proceed",
            exec_items=[("BTL-HIK-750-STEEL", 1)], completed=[0],
            drop_db=True),
        # underdetermined, genuine clarification with seeds preserved
        "s1-ambig-ca-spark": _persisted_spawn(
            declared="CLARIFY", outcome_kind="semantic_clarify",
            db_lines=[("BTL-HIK-500-NAVY", 1)]),
    })
    batch = _run_stage_injected(root, 1, spawn, production=False,
                                control_root=tmp_path / "ctl-readout")
    data = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    readout = data["scoring_readout"]
    assert readout["label"] == SCORING_READOUT_LABEL
    assert list(readout["profiles"]) == ["spark-medium", "wonder-medium"]
    spark = readout["profiles"]["spark-medium"]
    wonder = readout["profiles"]["wonder-medium"]
    assert spark == {
        "cells_run": 2, "task_appropriate": 2, "correct_clarification": 1,
        "physical_correct": 2,
        "declared_complete": {"verified": 1, "known_false": 0,
                              "unverified": 0},
        "evaluated_unreadable": 0, "halted_before_evaluation": 0,
        "known_cost_usd_estimated": 0.00335, "unknown_cost_calls": 0,
        "unresolved_cost_state": "none",
    }
    assert wonder == {
        "cells_run": 2, "task_appropriate": 0, "correct_clarification": 0,
        "physical_correct": 0,
        "declared_complete": {"verified": 0, "known_false": 1,
                              "unverified": 1},
        "evaluated_unreadable": 0, "halted_before_evaluation": 0,
        "known_cost_usd_estimated": 0.012, "unknown_cost_calls": 0,
        "unresolved_cost_state": "none",
    }
    # Cost beside the scoring counts is the existing per-cell cost rolled up.
    for profile_id, bucket in readout["profiles"].items():
        rows = [c for c in data["cells"] if c["profile"] == profile_id]
        assert bucket["cells_run"] == len(rows)
        assert bucket["known_cost_usd_estimated"] == round(
            sum(c["known_cost"] for c in rows), 8)

    summary = (batch / "summary.md").read_text(encoding="utf-8")
    assert "traversal complete means the permitted cells were visited; " \
           "it is not task success." in summary
    assert SCORING_READOUT_LABEL in summary
    assert "readout status: COMPLETE (evaluated_unreadable=0, " \
           "halted_before_evaluation=0)" in summary
    assert "| spark-medium | 2 | 2 | 1 | 2 | 1 | 0 | 0 | 0 | 0 | " \
           "0.00335 | 0 | none |" in summary
    assert "| wonder-medium | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | " \
           "0.012 | 0 | none |" in summary

    # Frozen with the batch before publication, and published bytes match.
    exported = batch / "export"
    assert (exported / "batch.json").read_bytes() \
        == (batch / "batch.json").read_bytes()
    assert (exported / "summary.md").read_bytes() \
        == (batch / "summary.md").read_bytes()
    manifest = json.loads(
        (exported / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["batch.json"] == hashlib.sha256(
        (batch / "batch.json").read_bytes()).hexdigest()
    assert manifest["files"]["summary.md"] == hashlib.sha256(
        (batch / "summary.md").read_bytes()).hexdigest()


def test_c1_unreadable_evaluated_record_visible_in_batch_and_summary(tmp_path):
    root = make_root(tmp_path)
    wonder_seeded = _persisted_spawn(
        declared="COMPLETE", outcome_kind="proceed",
        exec_items=[("BTL-HIK-750-STEEL", 1)], completed=[0],
        db_lines=[("BTL-HIK-500-NAVY", 1)])

    def corrupting_spawn(ctx):
        # Later-cell planting: cell 1 has already been evaluated and frozen
        # when this (second) cell spawns, so overwriting its evaluated
        # record now corrupts it before batch readout time.
        result = wonder_seeded(ctx)
        target = (ctx["rundir"].parent / "s1-seeded-ca-spark"
                  / "evaluated.json")
        target.write_text("{corrupt-evaluated-record", encoding="utf-8")
        return result

    spawn = _dispatch({
        "s1-seeded-ca-spark": _persisted_spawn(
            declared="COMPLETE", outcome_kind="proceed",
            exec_items=[("BTL-HIK-750-STEEL", 1)], completed=[0],
            db_lines=[("BTL-HIK-500-NAVY", 1),
                      ("BTL-HIK-750-STEEL", 1)]),
        "s1-seeded-ca-wonder": corrupting_spawn,
        "s1-ambig-ca-wonder": _persisted_spawn(
            declared="COMPLETE", outcome_kind="proceed",
            exec_items=[("BTL-HIK-750-STEEL", 1)], completed=[0],
            drop_db=True),
        "s1-ambig-ca-spark": _persisted_spawn(
            declared="CLARIFY", outcome_kind="semantic_clarify",
            db_lines=[("BTL-HIK-500-NAVY", 1)]),
    })
    batch = _run_stage_injected(root, 1, spawn, production=False,
                                control_root=tmp_path / "ctl-c1")
    data = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    # The batch still publishes; the corrupted cell is not dropped.
    assert data["status"] == "complete"
    assert len(data["cells"]) == 4
    _data, corrupted_row = _row_for(batch, "s1-seeded-ca-spark")
    assert corrupted_row["worker_output"] is True
    assert corrupted_row["halt"] is None

    readout = data["scoring_readout"]
    assert readout["status"] == "incomplete"
    spark = readout["profiles"]["spark-medium"]
    assert spark == {
        "cells_run": 2, "task_appropriate": 1, "correct_clarification": 1,
        "physical_correct": 1,
        "declared_complete": {"verified": 0, "known_false": 0,
                              "unverified": 0},
        "evaluated_unreadable": 1, "halted_before_evaluation": 0,
        "known_cost_usd_estimated": 0.00335, "unknown_cost_calls": 0,
        "unresolved_cost_state": "none",
    }
    # Other cells still count normally.
    wonder = readout["profiles"]["wonder-medium"]
    assert wonder["evaluated_unreadable"] == 0
    assert wonder["halted_before_evaluation"] == 0
    assert wonder["declared_complete"] == {"verified": 0, "known_false": 1,
                                           "unverified": 1}

    summary = (batch / "summary.md").read_text(encoding="utf-8")
    assert "readout status: INCOMPLETE (evaluated_unreadable=1, " \
           "halted_before_evaluation=0)" in summary
    assert "| spark-medium | 2 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | " \
           "0.00335 | 0 | none |" in summary
    assert "| wonder-medium | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | " \
           "0.012 | 0 | none |" in summary


def test_c2_pre_spawn_failure_halted_before_evaluation(tmp_path):
    root = make_root(tmp_path)

    def corrupting_task_spawn(ctx):
        # Corrupt the pinned task during cell 1's spawn so cell 2 (same
        # task) fails in the pre-spawn phase: no worker output, no record.
        result = fake_spawn(ctx)
        task_path = root / mft.TASKS["m-clean-seeded"]["path"]
        task_path.write_bytes(task_path.read_bytes() + b" ")
        return result

    spawn = _dispatch({"s1-seeded-ca-spark": corrupting_task_spawn})
    batch = _run_stage_injected(root, 1, spawn, production=False,
                                control_root=tmp_path / "ctl-c2")
    data = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    assert data["status"] == "halted"
    assert data["stop_reason"] == "task_hash_mismatch:m-clean-seeded"
    assert len(data["cells"]) == 2
    _data, row = _row_for(batch, "s1-seeded-ca-wonder")
    assert row["worker_output"] is False
    assert row["child_may_have_started"] is False
    assert row["halt"] == "task_hash_mismatch:m-clean-seeded"
    assert row["measured_outcome"] is None
    assert row["declared_outcome"] is None
    assert not (batch / "cells" / "s1-seeded-ca-wonder"
                / "evaluated.json").exists()

    readout = data["scoring_readout"]
    # A legitimate pre-evaluation halt is not an unreadable record.
    assert readout["status"] == "complete"
    wonder = readout["profiles"]["wonder-medium"]
    assert wonder == {
        "cells_run": 1, "task_appropriate": 0, "correct_clarification": 0,
        "physical_correct": 0,
        "declared_complete": {"verified": 0, "known_false": 0,
                              "unverified": 0},
        "evaluated_unreadable": 0, "halted_before_evaluation": 1,
        "known_cost_usd_estimated": 0.0, "unknown_cost_calls": 0,
        "unresolved_cost_state": "none",
    }
    spark = readout["profiles"]["spark-medium"]
    assert spark["cells_run"] == 1
    assert spark["evaluated_unreadable"] == 0
    assert spark["halted_before_evaluation"] == 0

    summary = (batch / "summary.md").read_text(encoding="utf-8")
    assert "readout status: COMPLETE (evaluated_unreadable=0, " \
           "halted_before_evaluation=1)" in summary
    assert "| wonder-medium | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | " \
           "0.0 | 0 | none |" in summary


def test_c3_unknown_cost_rollup_visible_in_batch_and_summary(tmp_path):
    root = make_root(tmp_path)
    calls = {"n": 0}

    def unknown_cost_spawn(ctx):
        calls["n"] += 1
        run = fake_run(ctx["spec"])
        # A made call with no reported cost: the existing accounting path
        # rejects the cost and the summary leaves the call unknown-cost.
        run["budget"]["sidecar"][0]["estimated_cost_usd"] = None
        (ctx["rundir"] / "run.json").write_text(json.dumps(run),
                                                encoding="utf-8")
        return run, {"cleanup_status": "verified", "timed_out": False}

    batch = _run_stage_injected(root, 1, unknown_cost_spawn, production=False,
                                control_root=tmp_path / "ctl-c3")
    data = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    assert calls["n"] == 1
    assert len(data["cells"]) == 1
    _data, row = _row_for(batch, "s1-seeded-ca-spark")
    assert row["accounting_ok"] is False
    assert "cost_not_finite" in row["accounting_reasons"]
    assert row["halt"] == "accounting_cost_not_finite"
    assert row["cost"]["status"] == "unknown_calls_present"
    assert row["cost"]["unknown_cost_calls"] == 1
    assert row["unresolved_cost_state"] == "unresolved"
    assert data["unknown_cost_calls"] == 1
    assert data["unresolved_cost_state"] == "unresolved"
    assert data["stop_reason"] == "accounting_cost_not_finite"

    readout = data["scoring_readout"]
    assert readout["status"] == "complete"
    spark = readout["profiles"]["spark-medium"]
    assert spark["cells_run"] == 1
    assert spark["unknown_cost_calls"] == 1
    assert spark["unresolved_cost_state"] == "unresolved"
    assert spark["evaluated_unreadable"] == 0
    assert spark["halted_before_evaluation"] == 0

    summary = (batch / "summary.md").read_text(encoding="utf-8")
    assert "- unknown-cost calls: 1" in summary
    assert "- unresolved cost state: unresolved" in summary
    assert "readout status: COMPLETE (evaluated_unreadable=0, " \
           "halted_before_evaluation=0)" in summary
    assert "| 0.0 | 1 | unresolved |" in summary


# --------------------------------------------------------------------------- #
# Reviewed-source content gate (source-v2)
# --------------------------------------------------------------------------- #

_FROZEN_DIGEST_PATHS = [
    "harness/__init__.py", "harness/pilot_manifest.py", "harness/model_pilot.py",
    "harness/model_smoke.py", "harness/model_compare.py", "harness/compare.py",
    "harness/run_task.py", "harness/snapshots.py", "harness/evaluate.py",
    "harness/metrics.py", "harness/model_metrics.py",
    "agents/__init__.py", "agents/careful_agent.py", "agents/model_actor.py",
    "agents/intent.py", "agents/model_client.py", "agents/meta_client.py",
    "agents/env_loader.py", "agents/full_cart_guard.py", "agents/smoke_guard.py",
    "adapter/__init__.py", "adapter/actions.py", "adapter/observations.py",
    "adapter/evidence.py",
    "core/__init__.py", "core/ledger.py", "core/checks.py", "core/verify.py",
    "core/recovery.py",
    "store/__init__.py", "store/app.py", "store/api.py", "store/control.py",
    "store/faults.py", "store/models.py", "store/pages.py",
    "store/seed_catalog.json", "store/templates/index.html",
    "store/templates/product.html", "store/templates/cart.html",
    "tests/test_model_pilot.py", "tests/test_model_pilot_cli.py",
    "tests/fixtures/pilot_tripwire/sitecustomize.py",
]


def _scope_fixture(tmp_path, monkeypatch, names=("alpha.bin", "beta.bin")):
    """Temp root with a tracked-looking pin; never touches real repo state."""
    import harness.model_pilot as mp
    root = tmp_path / "root"
    (root / "harness").mkdir(parents=True)
    for name in names:
        (root / name).write_bytes(b"payload-" + name.encode())
    monkeypatch.setattr(mp, "DIGEST_PATHS", tuple(names))
    digest = mp.compute_source_digest(root)
    (root / "harness" / "reviewed_source.sha256").write_text(
        "# reviewed pin\n" + digest + "\n", encoding="utf-8")
    return root, digest


def _stub_git(monkeypatch, head="a" * 40, status="", tracked=True):
    import harness.model_pilot as mp

    def fake(root, args):
        if args[0] == "rev-parse":
            return 0, head + "\n"
        if args[0] == "status":
            return 0, status
        if args[0] == "ls-files":
            return (0, "harness/reviewed_source.sha256\n") if tracked else (1, "")
        return 0, ""

    monkeypatch.setattr(mp, "_run_git", fake)


def test_source_digest_matches_pin_recipe(tmp_path, monkeypatch):
    # The shipped pin is intentionally stale during a review round; it is
    # re-pinned only by the separate review loop. This test proves the
    # source-v2 recipe and pin parser against a synthetic pin instead of
    # self-approving the real stale pin. The live gate's refusal on a real
    # mismatch is covered by test_git_scope_refuses_digest_mismatch.
    import harness.model_pilot as mp
    root = tmp_path / "pin-root"
    root.mkdir()
    (root / "a.bin").write_bytes(b"one")
    monkeypatch.setattr(mp, "DIGEST_PATHS", ("a.bin",))
    digest = mp.compute_source_digest(root)
    pin = root / "reviewed_source.sha256"
    pin.write_text(f"# pin\n{digest}\n", encoding="utf-8")
    found = re.search(r"^[0-9a-f]{64}$", pin.read_text(encoding="utf-8"),
                      re.MULTILINE)
    assert found is not None
    assert found.group(0) == mp.compute_source_digest(root)


def test_shipped_pin_is_well_formed():
    text = (PROJECT_ROOT / "harness" / "reviewed_source.sha256").read_text(
        encoding="utf-8")
    found = re.search(r"^[0-9a-f]{64}$", text, re.MULTILINE)
    assert found is not None
    assert re.fullmatch(r"[0-9a-f]{64}", found.group(0))


def test_source_digest_changes_on_byte_or_order_change(monkeypatch, tmp_path):
    import harness.model_pilot as mp
    root = tmp_path / "digest-root"
    root.mkdir()
    (root / "a.bin").write_bytes(b"one")
    (root / "b.bin").write_bytes(b"two")
    monkeypatch.setattr(mp, "DIGEST_PATHS", ("a.bin", "b.bin"))
    first = mp.compute_source_digest(root)
    monkeypatch.setattr(mp, "DIGEST_PATHS", ("b.bin", "a.bin"))
    assert mp.compute_source_digest(root) != first
    (root / "a.bin").write_bytes(b"one!")
    monkeypatch.setattr(mp, "DIGEST_PATHS", ("a.bin", "b.bin"))
    assert mp.compute_source_digest(root) != first


def test_digest_paths_equal_frozen_closure():
    assert list(DIGEST_PATHS) == _FROZEN_DIGEST_PATHS
    assert len(DIGEST_PATHS) == 43


def test_git_scope_refuses_dirty_worktree(monkeypatch, tmp_path):
    import harness.model_pilot as mp
    root, _ = _scope_fixture(tmp_path, monkeypatch)
    _stub_git(monkeypatch, status=" M harness/model_pilot.py\n")
    with pytest.raises(PilotError, match=r"^worktree_dirty:harness/model_pilot\.py$"):
        mp._git_scope(root)


def test_git_scope_refuses_digest_mismatch(monkeypatch, tmp_path):
    import harness.model_pilot as mp
    root, _ = _scope_fixture(tmp_path, monkeypatch)
    _stub_git(monkeypatch)
    (root / "alpha.bin").write_bytes(b"changed-after-pin")
    with pytest.raises(PilotError, match="reviewed_source_mismatch"):
        mp._git_scope(root)


def test_git_scope_records_head_without_comparing(monkeypatch, tmp_path):
    import harness.model_pilot as mp
    root, digest = _scope_fixture(tmp_path, monkeypatch)
    _stub_git(monkeypatch, head="f" * 40)
    scope = mp._git_scope(root)
    assert scope["head"] == "f" * 40
    assert scope["digest"] == digest
    assert scope["dirty"] == []


def test_pin_parser_rejects_missing_or_malformed(monkeypatch, tmp_path):
    import harness.model_pilot as mp
    root, digest = _scope_fixture(tmp_path, monkeypatch)
    pin = root / "harness" / "reviewed_source.sha256"
    _stub_git(monkeypatch)

    pin.unlink()
    with pytest.raises(PilotError, match="reviewed_source_pin_invalid"):
        mp._git_scope(root)

    pin.write_text("# comment\nnot-a-digest\n", encoding="utf-8")
    with pytest.raises(PilotError, match="reviewed_source_pin_invalid"):
        mp._git_scope(root)

    pin.write_text(digest + "\n" + digest + "\n", encoding="utf-8")
    with pytest.raises(PilotError, match="reviewed_source_pin_invalid"):
        mp._git_scope(root)

    pin.write_text(digest + "\n", encoding="utf-8")
    _stub_git(monkeypatch, tracked=False)
    with pytest.raises(PilotError, match="reviewed_source_pin_invalid"):
        mp._git_scope(root)

    _stub_git(monkeypatch)
    monkeypatch.setattr(mp, "_is_link", lambda p: True)
    with pytest.raises(PilotError, match="reviewed_source_pin_invalid"):
        mp._git_scope(root)


def test_git_scope_refuses_linked_path(monkeypatch, tmp_path):
    import harness.model_pilot as mp
    root, _ = _scope_fixture(tmp_path, monkeypatch)
    _stub_git(monkeypatch)
    real_is_link = mp._is_link
    monkeypatch.setattr(
        mp, "_is_link",
        lambda p: True if Path(p).name == "beta.bin" else real_is_link(p))
    with pytest.raises(PilotError, match=r"reviewed_source_linked:beta\.bin$"):
        mp._git_scope(root)


def test_digest_refuses_unreadable_path(monkeypatch, tmp_path):
    import harness.model_pilot as mp
    root = tmp_path / "missing-root"
    root.mkdir()
    monkeypatch.setattr(mp, "DIGEST_PATHS", ("missing.bin",))
    with pytest.raises(PilotError, match="reviewed_source_unreadable"):
        mp.compute_source_digest(root)


def test_git_scope_refuses_git_failure(monkeypatch, tmp_path):
    import harness.model_pilot as mp
    root, _ = _scope_fixture(tmp_path, monkeypatch)

    def failing(root_arg, args):
        if args[0] == "ls-files":
            return 0, "harness/reviewed_source.sha256\n"
        return 1, ""

    monkeypatch.setattr(mp, "_run_git", failing)
    with pytest.raises(PilotError, match="git_metadata_unavailable"):
        mp._git_scope(root)


def test_reservation_records_provenance_and_accepts_old_format(tmp_path):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    _run_stage_injected(root, 1, fake_spawn, production=False,
                        control_root=tmp_path / "ctl-prov")
    reservations = list((tmp_path / "ctl-prov" / "attempts" / "stage1").rglob(
        "reservation.json"))
    assert len(reservations) == 1
    new_res = json.loads(reservations[0].read_text(encoding="utf-8"))
    assert new_res["source_digest_version"] == mft.SOURCE_DIGEST_VERSION == 2
    assert "git_head" in new_res and new_res["git_head"] is None

    adir = tmp_path / "ctl-old" / "attempts" / "stage1" / "legacy01"
    adir.mkdir(parents=True)
    old = {"schema": 2, "attempt_id": "legacy01", "stage": 1,
           "protocol_id": mft.ATTEMPT_PROTOCOL_ID,
           "manifest_id": mft.MANIFEST_ID, "source_digest": "legacy-digest",
           "cells": [], "state": "reserved"}
    (adir / "reservation.json").write_text(json.dumps(old), encoding="utf-8")
    accepted = mp._persist_attempt(adir, old, "running")
    assert accepted["state"] == "running"
    assert accepted["source_digest"] == "legacy-digest"
    assert "git_head" not in accepted
    assert "source_digest_version" not in accepted
    reread = json.loads((adir / "reservation.json").read_text(encoding="utf-8"))
    assert reread["state"] == "running"


# ---------- B4: audited attempt successor ----------

_PRIOR_ID = "0123456789abcdef"


def _successor_grant(prior_id, states=("halted",), stage=1):
    return {"grant_id": "synthetic-grant", "stage": stage,
            "prior_attempt_id": prior_id,
            "accepted_prior_states": list(states)}


def _synthetic_prior(control_root, attempt_id, state, stage=1, mtime=None):
    stage_dir = Path(control_root) / "attempts" / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    latch = stage_dir / "stage.latch"
    if not latch.exists():
        latch.write_text("consumed\n", encoding="utf-8")
    adir = stage_dir / attempt_id
    adir.mkdir()
    rec = {
        "schema": 2, "attempt_id": attempt_id, "stage": stage,
        "protocol_id": mft.ATTEMPT_PROTOCOL_ID,
        "manifest_id": mft.MANIFEST_ID,
        "source_digest": "synthetic-test", "source_digest_version": 2,
        "git_head": None, "cells": [], "reserved_allowance_usd": 0.0,
        "state": state, "known_cost_usd": 0.0, "unknown_cost_calls": 0,
        "unresolved_cost_state": "none", "current_cell": None,
        "child_may_have_started": False, "attempted_cells": [],
        "completed_cells": [], "evidence_status": "synthetic",
        "decision": None,
    }
    (adir / "reservation.json").write_text(json.dumps(rec), encoding="utf-8")
    if mtime is not None:
        os.utime(adir, (mtime, mtime))
    return adir, rec


def _reserve_synth(control_root):
    return _reserve_attempt(Path(control_root), 1, list(mft.STAGES[1]),
                            "synthetic-test")


def test_successor_default_denied_without_grant(tmp_path, monkeypatch):
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", None)
    _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "halted")
    with pytest.raises(PilotError, match="attempt_already_reserved"):
        _reserve_synth(tmp_path / "ctl")


def test_successor_grant_invalid_shape_refused(tmp_path, monkeypatch):
    _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "halted")
    for bad in ({"grant_id": "g"}, [], "grant",
                _successor_grant(_PRIOR_ID, stage=2),
                _successor_grant("not-an-attempt-id"),
                _successor_grant(_PRIOR_ID, states=("running",))):
        monkeypatch.setattr(mft, "SUCCESSOR_GRANT", bad)
        with pytest.raises(PilotError, match="successor_grant_invalid"):
            _reserve_synth(tmp_path / "ctl")


def test_successor_wrong_prior_id_refused(tmp_path, monkeypatch):
    _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "halted")
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT",
                        _successor_grant("fedcba9876543210"))
    with pytest.raises(PilotError, match="successor_prior_missing"):
        _reserve_synth(tmp_path / "ctl")


def test_successor_named_prior_not_latest_refused(tmp_path, monkeypatch):
    now = time.time()
    _synthetic_prior(tmp_path / "ctl", "aaaaaaaaaaaaaaaa", "halted",
                     mtime=now - 200)
    _synthetic_prior(tmp_path / "ctl", "bbbbbbbbbbbbbbbb", "halted",
                     mtime=now - 100)
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT",
                        _successor_grant("aaaaaaaaaaaaaaaa"))
    with pytest.raises(PilotError, match="successor_prior_mismatch"):
        _reserve_synth(tmp_path / "ctl")


def test_successor_non_terminal_prior_refused(tmp_path, monkeypatch):
    _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "running")
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", _successor_grant(_PRIOR_ID))
    with pytest.raises(PilotError, match="successor_prior_not_terminal"):
        _reserve_synth(tmp_path / "ctl")


def test_successor_rejected_terminal_state_refused(tmp_path, monkeypatch):
    _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "complete")
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT",
                        _successor_grant(_PRIOR_ID, states=("halted",)))
    with pytest.raises(PilotError, match="successor_prior_state_rejected"):
        _reserve_synth(tmp_path / "ctl")


def test_successor_unreadable_prior_refused(tmp_path, monkeypatch):
    stage_dir = tmp_path / "ctl" / "attempts" / "stage1"
    stage_dir.mkdir(parents=True)
    (stage_dir / "stage.latch").write_text("consumed\n", encoding="utf-8")
    adir = stage_dir / _PRIOR_ID
    adir.mkdir()
    (adir / "reservation.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", _successor_grant(_PRIOR_ID))
    with pytest.raises(PilotError, match="successor_prior_unreadable"):
        _reserve_synth(tmp_path / "ctl")


def test_successor_ambiguous_prior_refused(tmp_path, monkeypatch):
    now = time.time()
    _synthetic_prior(tmp_path / "ctl", "aaaaaaaaaaaaaaaa", "halted", mtime=now)
    _synthetic_prior(tmp_path / "ctl", "bbbbbbbbbbbbbbbb", "halted", mtime=now)
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT",
                        _successor_grant("aaaaaaaaaaaaaaaa"))
    with pytest.raises(PilotError, match="successor_prior_ambiguous"):
        _reserve_synth(tmp_path / "ctl")


def test_valid_successor_records_lineage_and_preserves_prior(tmp_path, monkeypatch):
    now = time.time()
    adir, _rec = _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "halted",
                                  mtime=now - 100)
    stage_dir = adir.parent
    prior_bytes = (adir / "reservation.json").read_bytes()
    latch_bytes = (stage_dir / "stage.latch").read_bytes()
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", _successor_grant(_PRIOR_ID))

    state = _reserve_synth(tmp_path / "ctl")
    assert state["successor_of"] == _PRIOR_ID
    assert state["successor_grant_id"] == "synthetic-grant"
    assert state["state"] == "reserved"
    new_adir = stage_dir / state["attempt_id"]
    assert (new_adir / "reservation.json").is_file()
    assert (adir / "reservation.json").read_bytes() == prior_bytes
    assert (stage_dir / "stage.latch").read_bytes() == latch_bytes
    # Exactly one successor per reviewed grant: the guard is consumed and its
    # successor reservation exists, so a replay refuses explicitly.
    with pytest.raises(PilotError, match="successor_already_taken"):
        _reserve_synth(tmp_path / "ctl")


def test_normal_reservation_has_no_successor_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", None)
    state = _reserve_synth(tmp_path / "ctl")
    assert "successor_of" not in state
    assert "successor_grant_id" not in state


# ---------- B4-fix: atomic one-time successor reservation ----------


def _take_guard(stage_dir, grant, text="taken synthetic-grant\n"):
    import harness.model_pilot as mp
    guard = mp._successor_guard_path(stage_dir, grant)
    guard.write_text(text, encoding="utf-8")
    return guard


def _attempt_dirs(stage_dir):
    return sorted(p.name for p in stage_dir.iterdir() if p.is_dir())


def test_successor_guard_taken_without_reservation_refuses_closed(
        tmp_path, monkeypatch):
    adir, _ = _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "halted")
    grant = _successor_grant(_PRIOR_ID)
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", grant)
    guard = _take_guard(adir.parent, grant)

    with pytest.raises(PilotError, match="successor_guard_incomplete"):
        _reserve_synth(tmp_path / "ctl")
    # Fail closed, never repair: the guard and every prior file remain, and
    # no successor attempt was created.
    assert guard.is_file()
    assert _attempt_dirs(adir.parent) == [_PRIOR_ID]


def test_successor_guard_with_unreadable_partial_refuses_closed(
        tmp_path, monkeypatch):
    adir, _ = _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "halted")
    grant = _successor_grant(_PRIOR_ID)
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", grant)
    _take_guard(adir.parent, grant)
    partial = adir.parent / "ffffffffffffffff"
    partial.mkdir()
    (partial / "reservation.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(PilotError, match="successor_guard_incomplete"):
        _reserve_synth(tmp_path / "ctl")
    assert (partial / "reservation.json").read_text(encoding="utf-8") == "{broken"
    assert _attempt_dirs(adir.parent) == sorted([_PRIOR_ID, "ffffffffffffffff"])


def test_successor_guard_wrong_type_refuses_unreadable(tmp_path, monkeypatch):
    adir, _ = _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "halted")
    grant = _successor_grant(_PRIOR_ID)
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", grant)
    import harness.model_pilot as mp
    mp._successor_guard_path(adir.parent, grant).mkdir()

    with pytest.raises(PilotError, match="successor_guard_unreadable"):
        _reserve_synth(tmp_path / "ctl")
    # The wrong-type guard stays and no successor attempt appeared.
    assert _attempt_dirs(adir.parent) == sorted(
        [_PRIOR_ID, mp._successor_guard_path(adir.parent, grant).name])


def test_successor_concurrent_race_reserves_exactly_one(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "halted")
    grant = _successor_grant(_PRIOR_ID)
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", grant)
    workers = 8
    barrier = threading.Barrier(workers)

    def attempt(_index):
        barrier.wait(timeout=30)
        try:
            return ("ok", _reserve_synth(tmp_path / "ctl"))
        except PilotError as exc:
            return ("refused", str(exc))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(attempt, range(workers)))

    reserved = [value for status, value in outcomes if status == "ok"]
    refused = [value for status, value in outcomes if status == "refused"]
    assert len(reserved) == 1, outcomes
    assert len(refused) == workers - 1, outcomes
    assert set(refused) <= {"successor_already_taken",
                            "successor_guard_incomplete"}, refused

    stage_dir = tmp_path / "ctl" / "attempts" / "stage1"
    successor_dirs = [d for d in _attempt_dirs(stage_dir) if d != _PRIOR_ID]
    assert len(successor_dirs) == 1, successor_dirs
    rec = json.loads((stage_dir / successor_dirs[0] / "reservation.json")
                     .read_text(encoding="utf-8"))
    assert rec["successor_of"] == _PRIOR_ID
    assert rec["successor_grant_id"] == grant["grant_id"]
    assert (stage_dir / "stage.latch").read_text(encoding="utf-8") == "consumed\n"

    # Replay after the race still refuses, and still creates nothing.
    with pytest.raises(PilotError, match="successor_already_taken"):
        _reserve_synth(tmp_path / "ctl")
    assert [d for d in _attempt_dirs(stage_dir) if d != _PRIOR_ID] == successor_dirs


def test_new_reviewed_grant_can_succeed_after_a_stuck_guard(tmp_path, monkeypatch):
    adir, _ = _synthetic_prior(tmp_path / "ctl", _PRIOR_ID, "halted")
    stuck = _successor_grant(_PRIOR_ID)
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", stuck)
    _take_guard(adir.parent, stuck)
    with pytest.raises(PilotError, match="successor_guard_incomplete"):
        _reserve_synth(tmp_path / "ctl")

    # Recovery is a NEW reviewed grant after inspection, never cleanup.
    fresh = dict(stuck, grant_id="synthetic-grant-2")
    monkeypatch.setattr(mft, "SUCCESSOR_GRANT", fresh)
    state = _reserve_synth(tmp_path / "ctl")
    assert state["successor_grant_id"] == "synthetic-grant-2"
    assert state["successor_of"] == _PRIOR_ID
    assert len([d for d in _attempt_dirs(adir.parent) if d != _PRIOR_ID]) == 1


# ---------- B4: no-model dry-run ----------


def _dry_zero_call_run(spec):
    return {
        "arm": spec["arm"], "trial": 1, "execution_status": "normal",
        "declared_outcome": None, "raw_verdict": None,
        "seed_lines": spec["setup"]["seed_lines"], "items": [],
        "snapshots": [], "mutations": [],
        "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
        "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0,
                              "delete_line": 0},
        "stop_reasons": [], "final_cookie": None, "final_persisted": None,
        "guard": {"applied": False}, "transport": "fake",
        "budget": {
            "calls_attempted": 0, "calls_made": 0, "calls_refused": 0,
            "consumed_tokens": 0, "reserved_tokens": 0, "overrun": False,
            "records": [], "sidecar": [],
            "smoke_guard": {"enabled": True, "transport_kind": "fake",
                            "stopped": False},
        },
    }


def _dry_spawn(ctx):
    run = _dry_zero_call_run(ctx["spec"])
    (ctx["rundir"] / "run.json").write_text(json.dumps(run, indent=2),
                                            encoding="utf-8")
    return run, {"cleanup_status": "verified", "timed_out": False}


def test_dry_run_refuses_live_conflict(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    called = {"n": 0}
    monkeypatch.setattr(mp, "run_dry",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    rc = mp.main(["--stage", "1", "--dry-run", "--live"])
    assert rc == 2
    assert called["n"] == 0


def test_dry_run_refuses_stage2(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    rc = mp.main(["--stage", "2", "--dry-run"])
    assert rc == 2


def test_dry_run_refuses_when_offline_transport_unavailable(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    monkeypatch.setattr(mp, "_offline_transport_available", lambda: False)
    with pytest.raises(PilotError,
                       match="dry_run_offline_transport_unavailable"):
        mp.run_dry(root, 1, spawn_fn=_dry_spawn)
    assert not list((root / "var").glob("dry-run-*"))


def test_dry_run_writes_no_latch_or_reservation_shapes(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)

    def _boom():
        raise AssertionError("control root must not be touched")

    monkeypatch.setattr(mp, "_control_root", _boom)
    out, record = mp.run_dry(root, 1, spawn_fn=_dry_spawn)

    assert record["mode"] == "dry_run"
    assert record["receipt"] is False
    assert record["attempt_reservation"] is False
    assert record["provider_calls_made"] == 0
    assert record["offline_calls_made"] == 0
    assert "model_calls_made" not in record
    assert record["known_cost_usd"] == 0.0
    assert out.parent == root / "var"
    assert out.name.startswith("dry-run-stage1-")
    assert (out / "dry-run.json").is_file()
    assert (out / "summary.md").is_file()
    assert not (out / "batch.json").exists()
    assert not (out / "ledger.json").exists()
    assert not (out / "EXPORT_OUTCOME.json").exists()
    assert not list(out.rglob("reservation.json"))
    assert not list(out.rglob("stage.latch"))
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "not a measurement" in summary
    assert "Provider calls: 0. Offline fixture calls: 0." in summary
    assert "control-root state" in summary


def test_dry_run_never_builds_meta_client_or_dotenv(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)

    def _boom(*args, **kwargs):
        raise AssertionError("dotenv/Meta path must not be reached")

    monkeypatch.setattr(mp, "_make_pilot_dotenv_fallback", _boom)
    out, record = mp.run_dry(root, 1, spawn_fn=_dry_spawn)
    assert record["provider_calls_made"] == 0
    assert record["offline_calls_made"] == 0
    assert (out / "dry-run.json").is_file()


# ---------- B4-fix: dry-run fixture integrity ----------


def test_dry_run_fixture_pin_is_reviewed_and_matches_bytes():
    assert "harness/pilot_manifest.py" in DIGEST_PATHS
    pin = mft.DRY_RUN_FIXTURE
    assert isinstance(pin, dict) and set(pin) == {"path", "sha256"}
    data = (PROJECT_ROOT / pin["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == pin["sha256"]


def test_dry_run_fixture_mismatch_refused(tmp_path):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    fixture = root / mft.DRY_RUN_FIXTURE["path"]
    data = json.loads(fixture.read_text(encoding="utf-8"))
    data["m-clean-seeded"]["CA"]["responses"][0]["payload"]["items"][0][
        "quantity"] = 2
    fixture.write_text(json.dumps(data), encoding="utf-8")
    cell = list(mft.STAGES[1])[0]
    with pytest.raises(PilotError, match="dry_run_fixture_mismatch"):
        mp._load_dry_run_responses(root, cell)


def test_dry_run_fixture_missing_refused(tmp_path):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    (root / mft.DRY_RUN_FIXTURE["path"]).unlink()
    cell = list(mft.STAGES[1])[0]
    with pytest.raises(PilotError, match="dry_run_responses_unavailable"):
        mp._load_dry_run_responses(root, cell)


def test_dry_run_fixture_pin_invalid_refused(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    cell = list(mft.STAGES[1])[0]
    for bad in (None, {"path": "eval/model/dev/fake_scripts.json"},
                {"path": "eval/model/dev/fake_scripts.json",
                 "sha256": "not-a-hex-digest"},
                {"path": "eval/model/dev/fake_scripts.json",
                 "sha256": "a" * 64, "extra": 1}):
        monkeypatch.setattr(mft, "DRY_RUN_FIXTURE", bad)
        with pytest.raises(PilotError, match="dry_run_fixture_pin_invalid"):
            mp._load_dry_run_responses(root, cell)


def test_live_meta_path_never_reads_dry_run_fixture(tmp_path, monkeypatch):
    import harness.model_pilot as mp
    root = make_root(tmp_path)

    def _boom(*args, **kwargs):
        raise AssertionError("live meta path consulted the dry-run fixture")

    monkeypatch.setattr(mp, "_load_dry_run_responses", _boom)
    cell = list(mft.STAGES[1])[0]
    task, _, _ = _read_pinned_task(root, cell["task_id"])
    spec = build_cell_spec(cell, task)
    assert spec["transport"] == "meta" and spec["allow_live"] is True
    out = root / "var" / "model-pilot-stage1-chk" / "cells" / cell["cell_id"]
    out.mkdir(parents=True)
    spec["_startup_barrier"] = {"ready": str(out / "barrier-ready.json"),
                                "release": str(out / "barrier-release")}
    validated, _ = validate_worker_spec(spec, root)
    assert validated == cell
    assert spec["transport"] == "meta"
    assert spec["allow_live"] is True


# --------------------------------------------------------------------------- #
# B5 (16): transport-scoped accounting, dry-run call accounting, CLI truth
# --------------------------------------------------------------------------- #


def _made_call_run(transport="fake", declared_usage=1100):
    """Well-formed run with one made call, synthetic usage, and no sidecars."""
    task, _, _ = _read_pinned_task(PROJECT_ROOT, "m-clean-seeded")
    spec = build_cell_spec(mft.STAGES[1][0], task)
    run = fake_run(spec)
    run["transport"] = transport
    budget = dict(run["budget"])
    record = dict(budget["records"][0])
    record["declared_usage"] = declared_usage
    budget["records"] = [record]
    budget["sidecar"] = []
    budget["smoke_guard"] = {"enabled": True, "transport_kind": transport,
                             "stopped": False}
    run["budget"] = budget
    return run


def _spawn_run(run, spawn_info=None):
    info = spawn_info or {"cleanup_status": "verified", "timed_out": False}

    def spawn(ctx):
        (ctx["rundir"] / "run.json").write_text(json.dumps(run), encoding="utf-8")
        return run, info

    return spawn


def _execute_prepared_dry(tmp_path, run, spawn_info=None):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    out_root = tmp_path / "out"
    out_root.mkdir()
    cell = list(mft.STAGES[1])[0]
    task, rundir = mp._prepare_cell(root, cell, out_root, 0.0)
    return mp._execute_prepared(root, cell, out_root,
                                _spawn_run(run, spawn_info), task, rundir,
                                transport="fake", dry_run=True)


def test_execute_prepared_fake_made_calls_clean(tmp_path):
    row = _execute_prepared_dry(tmp_path, _made_call_run())
    assert row["halt"] is None, row["accounting_reasons"]
    assert row["accounting_ok"] is True
    assert row["provider_calls_made"] == 0
    assert row["offline_calls_made"] == 1


def test_pilot_accounting_fake_made_calls_ok():
    import harness.model_pilot as mp
    prof = mft.PROFILES["spark-medium"]
    ok, reasons = mp.pilot_accounting_check(_made_call_run(), prof)
    assert ok, reasons
    assert "sidecar_correspondence_mismatch" not in reasons


def test_pilot_accounting_meta_missing_sidecar_still_fails():
    import harness.model_pilot as mp
    prof = mft.PROFILES["spark-medium"]
    ok, reasons = mp.pilot_accounting_check(
        _made_call_run(transport="meta"), prof)
    assert not ok
    assert "sidecar_correspondence_mismatch" in reasons


def test_pilot_accounting_fake_missing_declared_usage_fails():
    import harness.model_pilot as mp
    prof = mft.PROFILES["spark-medium"]
    ok, reasons = mp.pilot_accounting_check(
        _made_call_run(declared_usage=None), prof)
    assert not ok and "fake_synthetic_usage_missing" in reasons

    ok, reasons = mp.pilot_accounting_check(
        _made_call_run(declared_usage="not-an-int"), prof)
    assert not ok and "made_contract_invalid" in reasons


def test_execute_prepared_fake_missing_declared_usage_halts(tmp_path):
    row = _execute_prepared_dry(tmp_path, _made_call_run(declared_usage=None))
    assert row["halt"] == "accounting_fake_synthetic_usage_missing"
    assert "fake_synthetic_usage_missing" in row["accounting_reasons"]


def test_pilot_accounting_unknown_transport_fails_closed():
    import harness.model_pilot as mp
    prof = mft.PROFILES["spark-medium"]
    for transport in ("smtp", "__missing__"):
        run = _made_call_run()
        if transport == "__missing__":
            run.pop("transport", None)
        else:
            run["transport"] = transport
        run["budget"]["smoke_guard"]["transport_kind"] = transport
        ok, reasons = mp.pilot_accounting_check(run, prof)
        assert not ok
        assert "transport_unknown" in reasons
        assert "sidecar_correspondence_mismatch" not in reasons


def test_pilot_accounting_refused_contract_unchanged():
    import harness.model_pilot as mp
    prof = mft.PROFILES["spark-medium"]
    run = _made_call_run()
    budget = dict(run["budget"])
    budget["records"] = [{"call_index": 1, "kind": "intent",
                          "status": "refused", "reserved": 10,
                          "consumed": 5, "declared_usage": None}]
    budget.update(calls_attempted=1, calls_made=0, calls_refused=1,
                  reserved_tokens=10, consumed_tokens=5)
    run["budget"] = budget
    ok, reasons = mp.pilot_accounting_check(run, prof)
    assert not ok and "refused_contract_invalid" in reasons


def test_dry_run_cost_not_synthetic_halts_with_provider_zero(tmp_path, monkeypatch):
    import harness.model_smoke as smoke
    def fake_summary(run):
        return {"status": "provider_measured",
                "known_subtotal_usd_estimated": 0.01,
                "unknown_cost_calls": 0,
                "provider_billing_verified": False}
    monkeypatch.setattr(smoke, "paid_cost_summary", fake_summary)
    row = _execute_prepared_dry(tmp_path, _made_call_run())
    assert row["halt"] == "dry_run_cost_not_synthetic"
    assert row["provider_calls_made"] == 0


def test_dry_run_offline_calls_invalid_become_null(tmp_path):
    import harness.model_pilot as mp
    for index, bad in enumerate((True, -1)):
        run = _made_call_run()
        run["budget"]["calls_made"] = bad
        root = make_root(tmp_path / f"r{index}")
        out, record = mp.run_dry(root, 1, spawn_fn=_spawn_run(run))
        assert record["offline_calls_made"] is None
        assert record["provider_calls_made"] == 0


def test_dry_run_record_shape_counts(tmp_path):
    import harness.model_pilot as mp
    root = make_root(tmp_path)
    out, record = mp.run_dry(root, 1, spawn_fn=_spawn_run(_made_call_run()))
    assert record["schema"] == "dry-run-v1"
    assert record["provider_calls_made"] == 0
    assert record["offline_calls_made"] == 1
    assert "model_calls_made" not in record
    assert record["spawn"] == {"cleanup_status": "verified", "timed_out": False}
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Offline fixture calls: 1." in summary
    assert "only provider calls could be billed" in summary


def test_spawn_object_only_present_fields(tmp_path):
    info = {"cleanup_status": "verified", "timed_out": False,
            "job_assigned": True, "startup_ms": 12.5,
            "active_before_close": 0, "active_after_terminate": 0,
            "close_failure": True}
    row = _execute_prepared_dry(tmp_path, _made_call_run(), spawn_info=info)
    assert set(row["spawn"]) == {
        "cleanup_status", "timed_out", "job_assigned", "startup_ms",
        "active_before_close", "active_after_terminate"}
    assert "close_failure" not in row["spawn"]

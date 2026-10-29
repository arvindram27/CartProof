"""Model comparison integration + metrics tests (Brief 5). Fake client only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import model_metrics
from harness.compare import build_template_db
from harness.model_compare import backup_copy, execute_model_run, run_suite_cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = str(PROJECT_ROOT / "store" / "seed_catalog.json")
SKU = "BTL-HIK-500-NAVYBLUE"


@pytest.fixture
def template(tmp_path):
    t = tmp_path / "template.db"
    build_template_db(CATALOG, str(t))
    return t


def _intent(sku=SKU, qty=1):
    return {"payload": {"items": [{"sku": sku, "quantity": qty}],
                        "budget_max_cents": None,
                        "substitution_allowed": False}}


def _actions(sku=SKU, qty=1):
    return [{"payload": {"action": "search"}},
            {"payload": {"action": "open_product", "product_id": "bottle-hiker"}},
            {"payload": {"action": "select_variant", "sku": sku}},
            {"payload": {"action": "add_to_cart", "sku": sku, "quantity": qty}}]


def _run(tmp_path, template, arm, responses, tag, request=None, fault=None):
    db = tmp_path / f"{tag}-{arm}.db"
    backup_copy(str(template), str(db))
    setup = {"request_text": request or f"Add 1 x {SKU}.", "seed_lines": []}
    return execute_model_run(setup, fault or {"name": None, "params": {}}, arm, 1,
                             {"responses": responses}, str(db),
                             tmp_path / tag / arm)


def test_clean_model_run_all_arms_measure_correct(tmp_path, template):
    for arm in ("MA", "MB", "MC"):
        run = _run(tmp_path, template, arm, [_intent()] + _actions(), "clean")
        assert run["execution_status"] == "normal", arm
        assert run["declared_outcome"] == "COMPLETE", arm
        assert run["final_persisted"]["lines"] == {SKU: 1}, arm
    ca = _run(tmp_path, template, "CA",
              [_intent()] + _actions() + [
                  {"payload": {"action": "read_cart"}},
                  {"payload": {"action": "finish"}},
                  {"payload": {"declaration": "COMPLETE", "rationale": "ok"}}],
              "clean")
    assert ca["declared_outcome"] == "COMPLETE"
    assert ca["final_persisted"]["lines"] == {SKU: 1}


def test_underdetermined_run_clarifies_with_zero_writes(tmp_path, template):
    run = _run(tmp_path, template, "MC", [_intent()],
               "under", request="Add a 500ml Hiker Water Bottle.")
    assert run["outcome_kind"] == "semantic_clarify"
    assert run["declared_outcome"] == "CLARIFY"
    assert run["final_persisted"]["lines"] == {}
    assert run["adapter_mutations"]["add_to_cart"] == 0


def test_client_payloads_never_receive_fault_seed_or_expectations(tmp_path, template):
    run = _run(tmp_path, template, "MB", [_intent()] + _actions(), "iso",
               fault={"name": "quantity-drift", "params": {"delta": 1}})
    blob = json.dumps(run["model_client_requests"])
    assert "quantity-drift" not in blob
    assert "seed_lines" not in blob
    assert "expected" not in blob
    assert "fault" not in blob


def test_frozen_model_suite_is_refused(tmp_path):
    rc = run_suite_cli(["--suite", "eval/model/frozen/suite.json",
                        "--out-root", str(tmp_path / "should-not-exist")])
    assert rc == 2
    assert not (tmp_path / "should-not-exist").exists()


def _task(tc="resolvable", items=None, expected=None, seeds=None):
    items = items if items is not None else [{"sku": SKU, "quantity": 1}]
    return {"task_id": "t", "task_class": tc,
            "runtime": {"seed_lines": seeds or []},
            "expected": {"lines": expected if expected is not None else {SKU: 1}},
            "expected_interpretation": {"items": items,
                                        "budget_max_cents": None,
                                        "substitution_allowed": False}}


def _record(**over):
    run = {"arm": "MC", "fault_class": "quantity-drift",
           "execution_status": "normal", "outcome_kind": "proceed",
           "declared_outcome": "COMPLETE", "measured_outcome": "correct",
           "executable_intent": {"items": [{"sku": SKU, "quantity": 1}],
                                 "budget_max_cents": None,
                                 "substitution_allowed": False},
           "completed_items": [0], "failures": [],
           "mutations": [], "agent_write_count": 0,
           "final_persisted": {"lines": {SKU: 1}},
           "guard": {"status": "pass"}, "stop_reasons": [],
           "budget": {"calls_made": 3, "calls_refused": 0,
                      "reserved_tokens": 30, "consumed_tokens": 20,
                      "retries": 0, "overrun": False}}
    run.update(over)
    return run


def test_completed_execution_and_physical_correct_are_separate():
    ok = model_metrics.enrich_model_run(_record(), _task())
    assert ok["completed_execution"] is True and ok["task_success"] is True

    failed = model_metrics.enrich_model_run(
        _record(outcome_kind="operational_failure",
                failures=[{"category": "operational", "stage": "intent"}],
                declared_outcome=None), _task())
    assert failed["physical_correct"] is True
    assert failed["completed_execution"] is False
    assert failed["task_success"] is False


def test_interpretation_mismatch_recovery_is_reported_separately():
    wrong = _record(executable_intent={
        "items": [{"sku": "BTL-HIK-750-STEEL", "quantity": 1}],
        "budget_max_cents": None, "substitution_allowed": False})
    out = model_metrics.enrich_model_run(wrong, _task())
    assert out["interpretation_outcome"] == "intent_incorrect"
    assert out["recovery_analysis"]["alignment"] == "interpretation_mismatch"
    assert out["recovery_analysis"]["mismatch_reason"]
    assert out["task_appropriate"] in (True, False)  # still in overall accounting


def test_unnecessary_determinate_diagnostic_pair():
    snap = {"kind": "hook_entry", "hook": "after_add", "item_index": 0,
            "lines": {SKU: 2}}
    mut = {"operation": "set_line_quantity", "context": "after_add_hook",
           "corrective": True, "target": {"sku": SKU, "quantity": 1},
           "before_write_snapshot": {"item_index": 0, "lines": {SKU: 2}}}
    run = _record(snapshots=[snap], mutations=[mut], measured_outcome="correct")
    out = model_metrics.enrich_model_run(run, _task())
    report = model_metrics.aggregate([out])
    cell = report["totals_by_arm"]["MC"]
    assert cell["unnecessary_correction"]["display"] == "0/1"
    assert cell["unnecessary_determinate"]["display"] == "0/1"
    assert cell["unnecessary_unknown_before_write"] == 0

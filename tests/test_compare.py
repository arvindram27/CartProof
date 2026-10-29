"""Comparison-runner integration and isolation tests (Brief 4 sections 3, 6, 7).

These use synthetic inline task specs and ``eval/dev``-style expectations, plus
the frozen MANIFEST integrity check. They never run ``eval/frozen`` tasks.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from harness import evaluate, metrics
from harness.compare import backup_copy, build_template_db, execute_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = str(PROJECT_ROOT / "store" / "seed_catalog.json")


def _clean_runtime():
    return {
        "items": [{"product_query": "Hiker Water Bottle",
                   "target_sku": "BTL-HIK-500-NAVYBLUE", "quantity": 1}],
        "seed_lines": [],
        "constraints": [{"type": "item_quantity",
                         "sku": "BTL-HIK-500-NAVYBLUE", "quantity": 1}],
    }


@pytest.fixture
def template(tmp_path):
    t = tmp_path / "template.db"
    build_template_db(CATALOG, str(t))
    return t


def _run_arm(tmp_path, template, runtime, fault, arm, tag):
    db = tmp_path / f"{tag}-{arm}.db"
    backup_copy(str(template), str(db))
    return execute_run(runtime, fault, arm, str(db), tmp_path / tag / arm,
                       settle_ms=2000)


def _measure(run, tmp_path, tag, arm):
    final_db = tmp_path / tag / arm / "final.db"
    return evaluate.measure_final_db(str(final_db), {"BTL-HIK-500-NAVYBLUE": 1},
                                     run.get("final_cookie"))


def test_clean_run_is_correct_under_all_three_arms(tmp_path, template):
    runtime = _clean_runtime()
    for arm in ("A", "B", "C"):
        run = _run_arm(tmp_path, template, runtime, {"name": None, "params": {}},
                       arm, "clean")
        assert run["execution_status"] == "normal", arm
        assert run["declared_outcome"] == "COMPLETE", arm
        assert _measure(run, tmp_path, "clean", arm)["measured_outcome"] == "correct", arm


def test_pass_through_hook_captures_snapshots_in_arm_a(tmp_path, template):
    run = _run_arm(tmp_path, template, _clean_runtime(),
                   {"name": None, "params": {}}, "A", "snap")
    hooks = [s["hook"] for s in run["snapshots"] if s["kind"] == "hook_entry"]
    assert "before_add" in hooks and "after_add" in hooks
    # Arm A's declaration still comes only from its action responses.
    assert run["declared_outcome"] == "COMPLETE"
    assert run["hook_counts"]["before_add"] == 1
    assert run["hook_counts"]["after_add"] == 1


def test_arm_a_false_completion_on_effective_fault(tmp_path, template):
    run = _run_arm(tmp_path, template, _clean_runtime(),
                   {"name": "quantity-drift", "params": {"delta": 1}}, "A", "drift")
    assert run["declared_outcome"] == "COMPLETE"
    assert _measure(run, tmp_path, "drift", "A")["measured_outcome"] == "incorrect"


def test_arm_c_recovery_captures_before_write_snapshot(tmp_path, template):
    run = _run_arm(tmp_path, template, _clean_runtime(),
                   {"name": "quantity-drift", "params": {"delta": 1}}, "C", "rec")
    assert run["declared_outcome"] == "COMPLETE"
    assert _measure(run, tmp_path, "rec", "C")["measured_outcome"] == "correct"
    corrective = [m for m in run["mutations"] if m.get("corrective")]
    assert corrective, "expected a corrective write"
    assert corrective[0]["operation"] == "set_line_quantity"
    assert corrective[0]["before_write_snapshot"]["kind"] == "before_write"


def test_runner_isolation_byte_identical_templates(tmp_path, template):
    a = tmp_path / "a.db"
    b = tmp_path / "b.db"
    backup_copy(str(template), str(a))
    backup_copy(str(template), str(b))
    assert a.read_bytes() == b.read_bytes()


def test_frozen_manifest_hashes_match_files():
    manifest_path = PROJECT_ROOT / "eval" / "frozen" / "MANIFEST"
    assert manifest_path.exists()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        h, rel = line.split()
        file_bytes = (PROJECT_ROOT / "eval" / "frozen" / rel).read_bytes()
        assert hashlib.sha256(file_bytes).hexdigest() == h, rel

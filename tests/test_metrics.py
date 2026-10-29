"""Metric semantics tests (Brief 4 corrections C1, C2, C3 and section 6).

Pure tests over synthetic run records; no store is booted.
"""
from __future__ import annotations

from harness import metrics


def _snap(kind, hook, idx, lines):
    return {"kind": kind, "hook": hook, "item_index": idx,
            "action_id": f"add-{idx}", "attempt_id": 1, "resolution": "cookie",
            "lines": lines, "version": idx + 1, "total_cents": 100}


def _mutation(operation, target, corrective, before_lines=None, item_index=None):
    m = {"operation": operation, "context": "after_add_hook" if corrective else "actor",
         "corrective": corrective, "target": target}
    if corrective:
        m["before_write_snapshot"] = {
            "kind": "before_write", "item_index": item_index,
            "resolution": "cookie", "lines": before_lines}
    return m


def _run(arm, fc, seeds, items, snapshots=None, mutations=None, declared="COMPLETE",
         measured="correct", status="normal", stop_reasons=None):
    return {
        "arm": arm, "fault_class": fc, "seed_lines": seeds, "items": items,
        "snapshots": snapshots or [], "mutations": mutations or [],
        "declared_outcome": declared, "measured_outcome": measured,
        "execution_status": status, "stop_reasons": stop_reasons or [],
    }


def test_stage_semantics_three_items():
    items = [{"target_sku": "A", "quantity": 1},
             {"target_sku": "B", "quantity": 1},
             {"target_sku": "C", "quantity": 1}]
    snapshots = [
        _snap("hook_entry", "before_add", 1, {"A": 1}),      # prefix 1 = {A}
        _snap("hook_entry", "after_add", 1, {"A": 1, "B": 1}),  # prefix 2
        _snap("hook_entry", "after_add", 2, {"A": 1, "B": 1, "C": 1}),  # prefix 3
    ]
    run = _run("C", "clean", [], items, snapshots=snapshots)
    metrics.enrich_run(run, {"A": 1, "B": 1, "C": 1})
    assert run["snapshots"][0]["stage_expected"] == {"A": 1}
    assert run["snapshots"][0]["stage_matches"] is True
    assert run["snapshots"][1]["stage_expected"] == {"A": 1, "B": 1}
    assert run["snapshots"][2]["stage_expected"] == {"A": 1, "B": 1, "C": 1}


def test_repeated_sku_across_items_aggregates():
    items = [{"target_sku": "A", "quantity": 1},
             {"target_sku": "A", "quantity": 2}]
    snapshots = [_snap("hook_entry", "after_add", 1, {"A": 3})]
    run = _run("C", "clean", [], items, snapshots=snapshots)
    metrics.enrich_run(run, {"A": 3})
    assert run["snapshots"][0]["stage_expected"] == {"A": 3}
    assert run["snapshots"][0]["stage_matches"] is True


def test_eligibility_missed_detection_no_write_still_eligible_but_fails():
    items = [{"target_sku": "A", "quantity": 1}]
    # Fault effective: after_add shows the wrong cart (A missing -> deviation).
    snapshots = [_snap("hook_entry", "after_add", 0, {})]
    run = _run("C", "silent-drop", [], items, snapshots=snapshots,
               declared="COMPLETE", measured="incorrect")
    metrics.enrich_run(run, {"A": 1})
    assert run["recovery"]["eligible"] is True
    assert run["recovery"]["attempted"] is False
    # success is decided by final measured state, not at detection time
    assert run["measured_outcome"] == "incorrect"


def test_eligibility_later_correction_counts_as_success():
    items = [{"target_sku": "A", "quantity": 1}]
    snapshots = [_snap("hook_entry", "after_add", 0, {})]
    mutations = [_mutation("set_line_quantity", {"sku": "A", "quantity": 1},
                           True, before_lines={}, item_index=0)]
    run = _run("C", "silent-drop", [], items, snapshots=snapshots,
               mutations=mutations, declared="COMPLETE", measured="correct")
    metrics.enrich_run(run, {"A": 1})
    assert run["recovery"]["eligible"] is True
    assert run["recovery"]["attempted"] is True


def test_eligibility_unknown_when_snapshots_incomplete():
    items = [{"target_sku": "A", "quantity": 1}]
    snapshots = [_snap("hook_entry", "after_add", 0, None)]
    run = _run("C", "silent-drop", [], items, snapshots=snapshots,
               declared="COMPLETE", measured="correct")
    metrics.enrich_run(run, {"A": 1})
    assert run["recovery"]["eligible"] is None
    assert run["recovery"]["unknown_snapshots"] == 1


def test_no_effect_vs_not_observed_labels():
    items = [{"target_sku": "A", "quantity": 1}]
    # all known, no deviation, final correct -> no effect observed
    snap = [_snap("hook_entry", "after_add", 0, {"A": 1})]
    run = _run("C", "quantity-drift", [], items, snapshots=snap,
               declared="COMPLETE", measured="correct")
    metrics.enrich_run(run, {"A": 1})
    assert run["deviation_observation"] == "no effect observed"

    # no snapshot deviation but final incorrect -> not observed / unknown
    run2 = _run("C", "quantity-drift", [], items, snapshots=snap,
                declared="COMPLETE", measured="incorrect")
    metrics.enrich_run(run2, {"A": 1})
    assert run2["deviation_observation"] == "not observed / unknown"


def test_unnecessary_correction_judged_against_stage_not_final():
    items = [{"target_sku": "A", "quantity": 1},
             {"target_sku": "B", "quantity": 1}]
    # Corrective write on item 0 when the cart already matched prefix 1 ({A:1}),
    # which is NOT the final expectation ({A:1,B:1}).
    mutations = [_mutation("set_line_quantity", {"sku": "A", "quantity": 1},
                           True, before_lines={"A": 1}, item_index=0)]
    run = _run("C", "quantity-drift", [], items, mutations=mutations,
               declared="COMPLETE", measured="correct")
    metrics.enrich_run(run, {"A": 1, "B": 1})
    assert run["corrective_writes"][0]["unnecessary"] is True


def test_redundant_line_is_separate_diagnostic():
    items = [{"target_sku": "A", "quantity": 1}]
    mutations = [_mutation("set_line_quantity", {"sku": "A", "quantity": 1},
                           True, before_lines={"A": 1}, item_index=0)]
    run = _run("C", "quantity-drift", [], items, mutations=mutations)
    metrics.enrich_run(run, {"A": 1})
    assert run["corrective_writes"][0]["redundant_line"] is True
    assert run["corrective_writes"][0]["unnecessary"] is True


def test_aggregate_zero_denominator_is_n0():
    report = metrics.aggregate([_run("C", "clean", [], [],
                                     declared="PARTIAL", measured="incorrect")])
    cell = report["totals_by_arm"]["C"]
    assert cell["false_completion_rate_over_declared"]["display"] == "n=0"
    assert cell["false_completion_rate_over_declared"]["rate"] is None
    assert cell["recovery"]["rate"]["display"] == "n=0"


def test_false_completion_two_denominators():
    runs = [
        _run("A", "clean", [], [], declared="COMPLETE", measured="correct"),
        _run("A", "clean", [], [], declared="COMPLETE", measured="incorrect"),
        _run("A", "clean", [], [], declared="PARTIAL", measured="incorrect"),
    ]
    report = metrics.aggregate(runs)
    cell = report["totals_by_arm"]["A"]
    assert cell["declared_complete_count"] == 2
    assert cell["known_false_completions"] == 1
    assert cell["false_completion_rate_over_declared"]["display"] == "1/2"
    assert cell["false_completion_rate_over_all"]["display"] == "1/3"


def test_crash_with_correct_cart_is_not_success():
    runs = [_run("A", "clean", [], [], declared=None, measured="correct",
                 status="crash")]
    report = metrics.aggregate(runs)
    cell = report["totals_by_arm"]["A"]
    assert cell["physical_correct_count"] == 1
    assert cell["task_success_rate"]["display"] == "0/1"


def test_unverified_completion_not_false():
    runs = [_run("A", "clean", [], [], declared="COMPLETE", measured="unknown")]
    report = metrics.aggregate(runs)
    cell = report["totals_by_arm"]["A"]
    assert cell["known_false_completions"] == 0
    assert cell["unverified_completions"] == 1


def test_recovery_buckets_sum_to_arm_c_totals():
    items = [{"target_sku": "A", "quantity": 1}]
    dev_snap = [_snap("hook_entry", "after_add", 0, {})]
    ok_snap = [_snap("hook_entry", "after_add", 0, {"A": 1})]
    unk_snap = [_snap("hook_entry", "after_add", 0, None)]
    runs = [
        _run("C", "silent-drop", [], items, snapshots=dev_snap,
             declared="PARTIAL", measured="incorrect"),    # eligible
        _run("C", "silent-drop", [], items, snapshots=ok_snap,
             declared="COMPLETE", measured="correct"),     # no_observed_deviation
        _run("C", "quantity-drift", [], items, snapshots=ok_snap,
             declared="COMPLETE", measured="correct"),     # no_observed_deviation
        _run("C", "wrong-variant", [], items, snapshots=unk_snap,
             declared="COMPLETE", measured="correct"),     # unknown
        _run("C", "clean", [], items, snapshots=ok_snap,
             declared="COMPLETE", measured="correct"),     # not_applicable
        _run("C", "stale-total", [], items, snapshots=ok_snap,
             declared="PARTIAL", measured="correct"),      # not_applicable
    ]
    for r in runs:
        metrics.enrich_run(r, {"A": 1})
    report = metrics.aggregate(runs)

    def bucket_sum(rec):
        return (rec["eligible"] + rec["no_observed_deviation"]
                + rec["unknown"] + rec["not_applicable"])

    # Per-fault-class bucket sums.
    sd = report["groups"]["silent-drop"]["C"]["recovery"]
    assert sd["eligible"] == 1 and sd["no_observed_deviation"] == 1
    assert bucket_sum(sd) == 2
    qd = report["groups"]["quantity-drift"]["C"]["recovery"]
    assert qd["no_observed_deviation"] == 1 and bucket_sum(qd) == 1
    wv = report["groups"]["wrong-variant"]["C"]["recovery"]
    assert wv["unknown"] == 1 and bucket_sum(wv) == 1
    cl = report["groups"]["clean"]["C"]["recovery"]
    assert cl["not_applicable"] == 1 and bucket_sum(cl) == 1
    st = report["groups"]["stale-total"]["C"]["recovery"]
    assert st["not_applicable"] == 1 and bucket_sum(st) == 1

    # Overall invariant: every arm-C run lands in exactly one bucket.
    total = report["totals_by_arm"]["C"]
    rec = total["recovery"]
    assert rec["eligible"] == 1
    assert rec["no_observed_deviation"] == 2
    assert rec["unknown"] == 1
    assert rec["not_applicable"] == 2
    assert bucket_sum(rec) == total["total_runs"] == 6

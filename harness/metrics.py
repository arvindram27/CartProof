"""Metric aggregation for the Brief 4 comparison runner.

Operates only on completed per-run records (already enriched with the measured
outcome). Computes task success, false-completion (two denominators), recovery
success, unnecessary-correction and redundant-line-write rates, stop/abstention
counts, and zero-denominator handling.

This module imports ``evaluate`` for prefix derivation and line comparison, and
never reads any runtime or store module.
"""
from __future__ import annotations

from typing import Any, Optional

from .evaluate import compare_lines, derive_prefixes

#: Task fault classes for which the recovery phase declares a recoverable kind.
#: Matches RECOVERABLE_KINDS in core/recovery.py at the task-class level.
RECOVERABLE_CLASSES = {"silent-drop", "wrong-variant", "quantity-drift"}


def _rate(num: int, den: int) -> dict:
    return {
        "numerator": num,
        "denominator": den,
        "rate": (num / den) if den else None,
        "display": f"n={num}/{den}" if den else "n=0",
    }


def enrich_run(run: dict, expected_lines: dict[str, int]) -> dict:
    """Attach stage comparisons, recovery eligibility and correction labels.

    Uses only the run's own snapshots, the runtime seed/items, and the authored
    final expectation. Never uses controller/verifier/agent outputs to form
    eligibility.
    """
    seed_lines = run.get("seed_lines") or []
    items = run.get("items") or []
    prefixes = derive_prefixes(seed_lines, items)
    final_prefix = prefixes[-1] if prefixes else {}
    run["prefix_cross_check_ok"] = (final_prefix == expected_lines)

    known_dev = 0
    unknown = 0
    for s in run.get("snapshots") or []:
        if s.get("kind") != "hook_entry":
            continue
        idx = s.get("item_index")
        hook = s.get("hook")
        if idx is None:
            continue
        if hook == "before_add":
            exp = prefixes[idx] if 0 <= idx < len(prefixes) else None
        elif hook == "after_add":
            exp = prefixes[idx + 1] if 0 <= idx + 1 < len(prefixes) else None
        else:
            continue
        if exp is None or s.get("lines") is None:
            unknown += 1
            continue
        mism = compare_lines(exp, s["lines"])
        s["stage_expected"] = exp
        s["stage_mismatches"] = mism
        s["stage_matches"] = not mism
        if mism:
            known_dev += 1

    arm = run.get("arm")
    fc = run.get("fault_class")
    recoverable_scope = (arm == "C" and fc in RECOVERABLE_CLASSES)
    if recoverable_scope:
        if known_dev > 0:
            eligible = True
        elif unknown > 0:
            eligible = None
        else:
            eligible = False
    else:
        eligible = False  # out of recovery scope (non-C arm or non-recoverable class)

    corrective = [m for m in (run.get("mutations") or []) if m.get("corrective")]
    for m in corrective:
        m["unnecessary"] = None
        m["redundant_line"] = None
        snap = m.get("before_write_snapshot")
        if snap and snap.get("item_index") is not None:
            idx = snap["item_index"]
            exp = prefixes[idx + 1] if 0 <= idx + 1 < len(prefixes) else None
            if exp is None or snap.get("lines") is None:
                continue
            m["unnecessary"] = not compare_lines(exp, snap["lines"])
            tgt = m.get("target") or {}
            sku = tgt.get("sku")
            if sku is not None:
                m["redundant_line"] = (
                    snap["lines"].get(sku, 0) == exp.get(sku, 0))

    # Deviation-observation label (C1): never call an unobserved case "benign".
    measured = run.get("measured_outcome")
    if known_dev > 0:
        deviation_observation = "observed deviation"
    elif unknown > 0:
        deviation_observation = "not observed / unknown"
    elif measured == "correct":
        deviation_observation = "no effect observed"
    else:
        deviation_observation = "not observed / unknown"

    run["recovery"] = {
        "eligible": eligible,
        "known_deviation": known_dev,
        "unknown_snapshots": unknown,
        "attempted": bool(corrective),
        "not_applicable": not recoverable_scope,
    }
    run["deviation_observation"] = deviation_observation
    run["corrective_writes"] = corrective
    return run


def _empty_cell(arm: str) -> dict:
    cell = {
        "arm": arm,
        "total_runs": 0,
        "task_success_rate": _rate(0, 0),
        "physical_correct_count": 0,
        "declared_complete_count": 0,
        "verified_completions": 0,
        "known_false_completions": 0,
        "unverified_completions": 0,
        "false_completion_rate_over_declared": _rate(0, 0),
        "false_completion_rate_over_all": _rate(0, 0),
        "stops": {},
        "partia_outcomes": 0,
    }
    if arm == "C":
        cell["recovery"] = {
            "eligible": 0, "success": 0, "attempted": 0, "no_attempt": 0,
            "unknown": 0, "not_applicable": 0, "no_observed_deviation": 0,
            "rate": _rate(0, 0),
        }
        cell["unnecessary_correction"] = _rate(0, 0)
        cell["unnecessary_unknown_before_write"] = 0
        cell["redundant_line_write"] = _rate(0, 0)
    return cell


def aggregate(runs: list[dict]) -> dict:
    groups: dict[str, dict[str, dict]] = {}
    totals: dict[str, dict] = {"A": _empty_cell("A"), "B": _empty_cell("B"),
                               "C": _empty_cell("C")}

    def ensure(fc: str, arm: str) -> dict:
        if fc not in groups:
            groups[fc] = {}
        if arm not in groups[fc]:
            groups[fc][arm] = _empty_cell(arm)
        return groups[fc][arm]

    for run in runs:
        fc = run.get("fault_class") or "clean"
        arm = run.get("arm")
        cell = ensure(fc, arm)
        tcell = totals[arm]

        for c in (cell, tcell):
            c["total_runs"] += 1
            if run.get("measured_outcome") == "correct":
                c["physical_correct_count"] += 1
            if run.get("execution_status") == "normal" \
                    and run.get("measured_outcome") == "correct":
                c["task_success_rate"]["numerator"] += 1
            c["task_success_rate"]["denominator"] += 1

            if run.get("declared_outcome") == "COMPLETE":
                c["declared_complete_count"] += 1
                m = run.get("measured_outcome")
                if m == "correct":
                    c["verified_completions"] += 1
                elif m == "incorrect":
                    c["known_false_completions"] += 1
                else:
                    c["unverified_completions"] += 1

            if run.get("declared_outcome") == "PARTIAL":
                c["partia_outcomes"] += 1
            for reason in (run.get("stop_reasons") or []):
                c["stops"][reason] = c["stops"].get(reason, 0) + 1

        # Arm C recovery / correction metrics (recoverable-class runs only).
        if arm == "C":
            rec = run.get("recovery") or {}
            elig = rec.get("eligible")
            if elig is True:
                c = cell["recovery"]
                t = totals["C"]["recovery"]
                for d in (c, t):
                    d["eligible"] += 1
                    if run.get("measured_outcome") == "correct":
                        d["success"] += 1
                    if rec.get("attempted"):
                        d["attempted"] += 1
                    else:
                        d["no_attempt"] += 1
            elif elig is None:
                cell["recovery"]["unknown"] += 1
                totals["C"]["recovery"]["unknown"] += 1
            elif not rec.get("not_applicable"):
                cell["recovery"]["no_observed_deviation"] += 1
                totals["C"]["recovery"]["no_observed_deviation"] += 1
            if rec.get("not_applicable"):
                cell["recovery"]["not_applicable"] += 1
                totals["C"]["recovery"]["not_applicable"] += 1

            for m in (run.get("corrective_writes") or []):
                for d in (cell, totals["C"]):
                    d["unnecessary_correction"]["denominator"] += 1
                    if m.get("unnecessary") is True:
                        d["unnecessary_correction"]["numerator"] += 1
                    if m.get("unnecessary") is None:
                        d["unnecessary_unknown_before_write"] += 1
                    d["redundant_line_write"]["denominator"] += 1
                    if m.get("redundant_line") is True:
                        d["redundant_line_write"]["numerator"] += 1

    # Finalize rates.
    for d in list(groups.values()):
        for cell in d.values():
            _finalize(cell)
    for cell in totals.values():
        _finalize(cell)

    return {"groups": groups, "totals_by_arm": totals,
            "n_runs": len(runs)}


def _finalize(cell: dict) -> None:
    cell["task_success_rate"]["rate"] = (
        cell["task_success_rate"]["numerator"] / cell["task_success_rate"]["denominator"]
        if cell["task_success_rate"]["denominator"] else None)
    cell["task_success_rate"]["display"] = (
        f"{cell['task_success_rate']['numerator']}/{cell['task_success_rate']['denominator']}"
        if cell["task_success_rate"]["denominator"] else "n=0")

    cell["false_completion_rate_over_declared"]["numerator"] = cell["known_false_completions"]
    cell["false_completion_rate_over_declared"]["denominator"] = cell["declared_complete_count"]
    cell["false_completion_rate_over_all"]["numerator"] = cell["known_false_completions"]
    cell["false_completion_rate_over_all"]["denominator"] = cell["total_runs"]
    for key in ("false_completion_rate_over_declared", "false_completion_rate_over_all"):
        cell[key]["rate"] = (
            cell[key]["numerator"] / cell[key]["denominator"]
            if cell[key]["denominator"] else None)
        cell[key]["display"] = (
            f"{cell[key]['numerator']}/{cell[key]['denominator']}"
            if cell[key]["denominator"] else "n=0")

    if "recovery" in cell:
        cell["recovery"]["rate"]["rate"] = (
            cell["recovery"]["success"] / cell["recovery"]["eligible"]
            if cell["recovery"]["eligible"] else None)
        cell["recovery"]["rate"]["display"] = (
            f"{cell['recovery']['success']}/{cell['recovery']['eligible']}"
            if cell["recovery"]["eligible"] else "n=0")
    if "unnecessary_correction" in cell:
        cell["unnecessary_correction"]["rate"] = (
            cell["unnecessary_correction"]["numerator"] / cell["unnecessary_correction"]["denominator"]
            if cell["unnecessary_correction"]["denominator"] else None)
        cell["unnecessary_correction"]["display"] = (
            f"{cell['unnecessary_correction']['numerator']}/{cell['unnecessary_correction']['denominator']}"
            if cell["unnecessary_correction"]["denominator"] else "n=0")
        cell["redundant_line_write"]["rate"] = (
            cell["redundant_line_write"]["numerator"] / cell["redundant_line_write"]["denominator"]
            if cell["redundant_line_write"]["denominator"] else None)
        cell["redundant_line_write"]["display"] = (
            f"{cell['redundant_line_write']['numerator']}/{cell['redundant_line_write']['denominator']}"
            if cell["redundant_line_write"]["denominator"] else "n=0")


def render_markdown(report: dict, suite_name: str) -> str:
    lines: list[str] = []
    lines.append(f"# CartProof evaluator summary — {suite_name}")
    lines.append("")
    lines.append("> Harness-produced demonstrations on controlled fixtures, "
                 "not measured model behavior.")
    lines.append("> These numbers say nothing about model-driven agents, real "
                 "merchants, or real payments.")
    lines.append("")
    lines.append("Frozen-suite caveat: this is a **frozen, unseen-results "
                 "suite — not an independent holdout**, and it still covers only "
                 "the same seven named fault modes.")
    lines.append("")
    lines.append("| Group | Arm | Runs | Task success | Physical correct | "
                 "Known-false | Unverified | False/(declared) | False/(all) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for fc in sorted(report["groups"].keys()):
        for arm in ("A", "B", "C"):
            cell = report["groups"][fc].get(arm)
            if cell is None:
                continue
            lines.append(
                f"| {fc} | {arm} | {cell['total_runs']} | "
                f"{cell['task_success_rate']['display']} | "
                f"{cell['physical_correct_count']} | "
                f"{cell['known_false_completions']} | "
                f"{cell['unverified_completions']} | "
                f"{cell['false_completion_rate_over_declared']['display']} | "
                f"{cell['false_completion_rate_over_all']['display']} |")
    lines.append("")
    lines.append("## Totals by arm")
    lines.append("")
    lines.append("| Arm | Runs | Task success | Physical correct | Known-false | "
                 "Unverified |")
    lines.append("|---|---|---|---|---|---|")
    for arm in ("A", "B", "C"):
        cell = report["totals_by_arm"][arm]
        lines.append(
            f"| {arm} | {cell['total_runs']} | "
            f"{cell['task_success_rate']['display']} | "
            f"{cell['physical_correct_count']} | "
            f"{cell['known_false_completions']} | "
            f"{cell['unverified_completions']} |")
    lines.append("")
    lines.append("## Arm C recovery (recoverable classes: silent-drop, "
                 "wrong-variant, quantity-drift)")
    lines.append("")
    c = report["totals_by_arm"]["C"]["recovery"]
    lines.append(f"- eligible: {c['eligible']} "
                 f"(no-observed-deviation: {c['no_observed_deviation']}, "
                 f"not-applicable: {c['not_applicable']}, "
                 f"unknown: {c['unknown']})")
    lines.append(f"- recovery success: {c['rate']['display']} "
                 f"(attempted: {c['attempted']}, no-attempt: {c['no_attempt']})")
    uc = report["totals_by_arm"]["C"]["unnecessary_correction"]
    rl = report["totals_by_arm"]["C"]["redundant_line_write"]
    lines.append(f"- unnecessary corrections (whole cart): {uc['display']}")
    lines.append(f"- redundant-line writes (diagnostic): {rl['display']}")
    lines.append("")
    lines.append("## Stops by reason (arms B and C)")
    lines.append("")
    for arm in ("B", "C"):
        stops = report["totals_by_arm"][arm].get("stops") or {}
        lines.append(f"### Arm {arm}")
        for reason in sorted(stops):
            lines.append(f"- {reason}: {stops[reason]}")
    lines.append("")
    return "\n".join(lines)

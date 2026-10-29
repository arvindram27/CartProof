"""Model-arm metric translation and aggregation (Brief 5).

Translates model-arm run records into the Brief 4 metric definitions
(declared vs measured, task success, false-completion with both denominators,
recovery success with eligibility, unnecessary-correction whole-cart,
stop/abstention, n=0), and adds the model-arm categories: cart success,
correct clarification, task-appropriate outcome, unnecessary abstention,
write violations, interpretation outcomes, guard downgrades, and per-arm cost.

The evaluator's oracle is the authored expected end-state / interpretation,
never the model's intent (R5).
"""
from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Optional

from . import metrics as brief4_metrics

RECOVERABLE_CLASSES = {"silent-drop", "wrong-variant", "quantity-drift"}


def _rate(num: int, den: int) -> dict:
    return {"numerator": num, "denominator": den,
            "rate": (num / den) if den else None,
            "display": f"{num}/{den}" if den else "n=0"}


def seed_agg(task: dict) -> dict:
    out: dict[str, int] = {}
    for s in (task.get("runtime") or {}).get("seed_lines") or []:
        out[s["sku"]] = out.get(s["sku"], 0) + int(s["quantity"])
    return out


def authored_items(task: dict) -> list[dict]:
    return list((task.get("expected_interpretation") or {}).get("items") or [])


def _exec_seq(run: dict) -> Optional[list]:
    ei = run.get("executable_intent")
    if not ei:
        return None
    return [(it["sku"], int(it["quantity"])) for it in ei.get("items") or []]


def enrich_model_run(run: dict, task: dict) -> dict:
    """Attach scoring, interpretation, recovery-alignment and cost fields."""
    run["task_class"] = task.get("task_class")
    run["physical_correct"] = (run.get("measured_outcome") == "correct")

    ei = run.get("executable_intent") or {}
    exec_items = ei.get("items") or []
    completed = set(run.get("completed_items") or [])
    no_operational = not any(f.get("category") == "operational"
                             for f in (run.get("failures") or []))
    completed_execution = bool(
        run.get("execution_status") == "normal"
        and run.get("outcome_kind") == "proceed"
        and no_operational
        and exec_items
        and all(i in completed for i in range(len(exec_items))))
    run["completed_execution"] = completed_execution
    run["task_success"] = bool(run["physical_correct"] and completed_execution)

    agent_writes = [m for m in (run.get("mutations") or [])
                    if not m.get("corrective")]
    run["agent_write_count"] = len(agent_writes)

    seeds = seed_agg(task)
    final_lines = (run.get("final_persisted") or {}).get("lines")
    seeds_preserved = (final_lines is not None and final_lines == seeds)

    correct_clarification = bool(
        run.get("execution_status") == "normal"
        and run.get("outcome_kind") == "semantic_clarify"
        and run.get("declared_outcome") == "CLARIFY"
        and run["agent_write_count"] == 0
        and seeds_preserved)
    run["correct_clarification"] = correct_clarification
    run["seeds_preserved"] = seeds_preserved
    run["task_appropriate"] = bool(
        (task.get("task_class") == "resolvable" and run["task_success"])
        or (task.get("task_class") == "underdetermined" and correct_clarification))

    # Interpretation outcome (independent of the final cart).
    authored = authored_items(task)
    authored_seq = [(it["sku"], int(it["quantity"])) for it in authored]
    auth_interp = task.get("expected_interpretation") or {}
    if run.get("outcome_kind") == "semantic_clarify":
        interp = ("clarify_appropriate" if task.get("task_class") == "underdetermined"
                  else "clarify_unnecessary")
    elif run.get("outcome_kind") == "operational_failure":
        interp = "operational_failure"
    elif _exec_seq(run) == authored_seq \
            and ei.get("budget_max_cents") == auth_interp.get("budget_max_cents") \
            and bool(ei.get("substitution_allowed", False)) == bool(
                auth_interp.get("substitution_allowed", False)):
        interp = "intent_correct"
    else:
        interp = "intent_incorrect"
    run["interpretation_outcome"] = interp

    # Recovery analysis (MC only), R5 alignment against the authored intent.
    if run.get("arm") == "MC":
        exec_seq = _exec_seq(run)
        aligned = (exec_seq == authored_seq)
        corrective = [m for m in (run.get("mutations") or []) if m.get("corrective")]
        if aligned:
            pseudo = copy.deepcopy(run)
            pseudo["arm"] = "C"  # Brief 4 enrichment keys recovery scope on "C"
            pseudo["seed_lines"] = (task.get("runtime") or {}).get("seed_lines") or []
            pseudo["items"] = [{"target_sku": it["sku"],
                                "quantity": int(it["quantity"])}
                               for it in authored]
            brief4_metrics.enrich_run(pseudo, task["expected"]["lines"])
            run["recovery"] = pseudo.get("recovery")
            run["corrective_writes"] = pseudo.get("corrective_writes")
            run["snapshots"] = pseudo.get("snapshots")
            run["recovery_analysis"] = {
                "alignment": "aligned", "mismatch_reason": None,
                "eligibility": (pseudo.get("recovery") or {}).get("eligible")}
        elif run.get("fault_class") in RECOVERABLE_CLASSES:
            diffs = []
            if exec_seq != authored_seq:
                diffs.append(
                    f"items differ: executable {exec_seq} vs authored {authored_seq}")
            run["corrective_writes"] = [
                {**m, "unnecessary": None, "redundant_line": None}
                for m in corrective]
            run["recovery"] = {"eligible": None, "known_deviation": 0,
                               "unknown_snapshots": 0,
                               "attempted": bool(corrective),
                               "not_applicable": False}
            run["recovery_analysis"] = {
                "alignment": "interpretation_mismatch",
                "mismatch_reason": "; ".join(diffs) or "interpretation differs",
                "eligibility": None}
        else:
            run["corrective_writes"] = [
                {**m, "unnecessary": None, "redundant_line": None}
                for m in corrective]
            run["recovery"] = {"eligible": False, "known_deviation": 0,
                               "unknown_snapshots": 0,
                               "attempted": bool(corrective),
                               "not_applicable": True}
            run["recovery_analysis"] = {
                "alignment": "not_applicable", "mismatch_reason": None,
                "eligibility": False}
    return run


def _empty_cell(arm: str) -> dict:
    cell = {
        "arm": arm, "total_runs": 0,
        "execution": Counter(),
        "cart_success": _rate(0, 0), "correct_clarification": _rate(0, 0),
        "task_appropriate": _rate(0, 0),
        "physical_correct_count": 0,
        "declared_complete_count": 0, "verified_completions": 0,
        "known_false_completions": 0, "unverified_completions": 0,
        "false_completion_over_declared": _rate(0, 0),
        "false_completion_over_all": _rate(0, 0),
        "unnecessary_abstention": 0, "operational_failures": 0,
        "write_violation_runs": 0, "write_violation_writes": 0,
        "no_write_seeded_preserved": 0,
        "interpretation": Counter(), "stops": Counter(),
        "guard_downgrades": 0,
        "cost": {"calls_attempted": 0, "calls_made": 0, "calls_refused": 0,
                 "reserved_tokens": 0, "consumed_tokens": 0, "overruns": 0,
                 "retries": 0},
    }
    if arm == "MC":
        cell["recovery"] = {"eligible": 0, "success": 0, "attempted": 0,
                            "no_attempt": 0, "unknown": 0, "not_applicable": 0,
                            "mismatch_count": 0, "mismatch_reasons": Counter(),
                            "rate": _rate(0, 0)}
        cell["unnecessary_correction"] = _rate(0, 0)
        cell["unnecessary_unknown_before_write"] = 0
        cell["redundant_line_write"] = _rate(0, 0)
    if arm == "CA":
        cell["agent_writes"] = 0
    return cell


def _finalize_rate(r: dict) -> None:
    r["rate"] = (r["numerator"] / r["denominator"]) if r["denominator"] else None
    r["display"] = f"{r['numerator']}/{r['denominator']}" if r["denominator"] else "n=0"


def aggregate(runs: list[dict]) -> dict:
    groups: dict[str, dict[str, dict]] = {}
    totals: dict[str, dict] = {a: _empty_cell(a) for a in ("MA", "CA", "MB", "MC")}

    def cell_for(tc: str, arm: str) -> dict:
        groups.setdefault(tc, {})
        if arm not in groups[tc]:
            groups[tc][arm] = _empty_cell(arm)
        return groups[tc][arm]

    for run in runs:
        tc = run.get("task_class") or "resolvable"
        arm = run.get("arm")
        for cell in (cell_for(tc, arm), totals[arm]):
            cell["total_runs"] += 1
            cell["task_appropriate"]["denominator"] += 1
            if run.get("task_appropriate"):
                cell["task_appropriate"]["numerator"] += 1
            cell["execution"][run.get("execution_status") or "normal"] += 1
            if run.get("physical_correct"):
                cell["physical_correct_count"] += 1
            if tc == "resolvable":
                cell["cart_success"]["denominator"] += 1
                if run.get("task_success"):
                    cell["cart_success"]["numerator"] += 1
                if run.get("outcome_kind") == "semantic_clarify":
                    cell["unnecessary_abstention"] += 1
            if tc == "underdetermined":
                cell["correct_clarification"]["denominator"] += 1
                if run.get("correct_clarification"):
                    cell["correct_clarification"]["numerator"] += 1
                if (run.get("agent_write_count") or 0) > 0:
                    cell["write_violation_runs"] += 1
                    cell["write_violation_writes"] += run["agent_write_count"]
            if run.get("outcome_kind") == "operational_failure":
                cell["operational_failures"] += 1
            if (run.get("agent_write_count") or 0) == 0 and run.get("seeds_preserved"):
                cell["no_write_seeded_preserved"] += 1
            if run.get("declared_outcome") == "COMPLETE":
                cell["declared_complete_count"] += 1
                m = run.get("measured_outcome")
                if m == "correct":
                    cell["verified_completions"] += 1
                elif m == "incorrect":
                    cell["known_false_completions"] += 1
                else:
                    cell["unverified_completions"] += 1
            cell["interpretation"][run.get("interpretation_outcome")] += 1
            for reason in (run.get("stop_reasons") or []):
                cell["stops"][reason] += 1
            if (run.get("guard") or {}).get("status") == "downgrade":
                cell["guard_downgrades"] += 1
            budget = run.get("budget") or {}
            for k in cell["cost"]:
                if k == "overruns":
                    cell["cost"][k] += 1 if budget.get("overrun") else 0
                else:
                    cell["cost"][k] += int(budget.get(k) or 0)
            if arm == "CA":
                cell["agent_writes"] += run.get("agent_write_count") or 0

        if arm == "MC":
            for cell in (cell_for(tc, arm), totals["MC"]):
                rec = run.get("recovery") or {}
                ra = run.get("recovery_analysis") or {}
                if ra.get("alignment") == "interpretation_mismatch":
                    cell["recovery"]["mismatch_count"] += 1
                    cell["recovery"]["mismatch_reasons"][
                        ra.get("mismatch_reason")] += 1
                elif rec.get("eligible") is True:
                    cell["recovery"]["eligible"] += 1
                    if run.get("measured_outcome") == "correct":
                        cell["recovery"]["success"] += 1
                    if rec.get("attempted"):
                        cell["recovery"]["attempted"] += 1
                    else:
                        cell["recovery"]["no_attempt"] += 1
                elif rec.get("eligible") is None:
                    cell["recovery"]["unknown"] += 1
                elif rec.get("not_applicable"):
                    cell["recovery"]["not_applicable"] += 1
                for m in (run.get("corrective_writes") or []):
                    cell["unnecessary_correction"]["denominator"] += 1
                    if m.get("unnecessary") is True:
                        cell["unnecessary_correction"]["numerator"] += 1
                    if m.get("unnecessary") is None:
                        cell["unnecessary_unknown_before_write"] += 1
                    cell["redundant_line_write"]["denominator"] += 1
                    if m.get("redundant_line") is True:
                        cell["redundant_line_write"]["numerator"] += 1

    def finalize(cell: dict) -> dict:
        for k in ("cart_success", "correct_clarification", "task_appropriate",
                  "unnecessary_correction", "redundant_line_write"):
            if k in cell:
                _finalize_rate(cell[k])
        cell["false_completion_over_declared"] = _rate(
            cell["known_false_completions"], cell["declared_complete_count"])
        _finalize_rate(cell["false_completion_over_declared"])
        cell["false_completion_over_all"] = _rate(
            cell["known_false_completions"], cell["total_runs"])
        _finalize_rate(cell["false_completion_over_all"])
        if "recovery" in cell:
            cell["recovery"]["rate"] = _rate(cell["recovery"]["success"],
                                             cell["recovery"]["eligible"])
            _finalize_rate(cell["recovery"]["rate"])
            den = cell["unnecessary_correction"]["denominator"]
            unk = cell["unnecessary_unknown_before_write"]
            det = den - unk
            cell["unnecessary_determinate"] = _rate(
                cell["unnecessary_correction"]["numerator"], det)
            _finalize_rate(cell["unnecessary_determinate"])
        cell["execution"] = dict(cell["execution"])
        cell["interpretation"] = dict(cell["interpretation"])
        cell["stops"] = dict(cell["stops"])
        if "recovery" in cell:
            cell["recovery"]["mismatch_reasons"] = dict(
                cell["recovery"]["mismatch_reasons"])
        n = cell["total_runs"] or 0
        cell["cost_per_run"] = {
            k: (round(v / n, 2) if n else None)
            for k, v in cell["cost"].items()}
        return cell

    for arms in groups.values():
        for cell in arms.values():
            finalize(cell)
    for cell in totals.values():
        finalize(cell)
    return {"groups": groups, "totals_by_arm": totals, "n_runs": len(runs)}


def render_markdown(report: dict, suite_name: str) -> str:
    lines = [f"# CartProof model-arm summary — {suite_name}", "",
             "> Harness-produced demonstrations on controlled fixtures with a "
             "FAKE model client. These prove plumbing only; they say nothing "
             "about how any real model performs.",
             "> Supported-grammar interpretation is anchored by the "
             "deterministic parser, with the model proposal as a cross-check; "
             "the model contribution measured here is grounded action "
             "selection, plus proposal behaviour at the grammar boundary.",
             "> MB/MC restrict the action menu per step (masked) while MA/CA "
             "are unmasked. That masking difference is a claimed CartProof "
             "mechanism under test, not evidence of model capability (M4).",
             "> Budget caps are best-effort fail-closed guards, not a hard "
             "token or dollar guarantee.", ""]
    lines.append("| Class | Arm | Runs | Cart success | Correct clarification | "
                 "Task-appropriate | Physical correct | Known-false | "
                 "False/(declared) | False/(all) | Calls made | "
                 "Tokens consumed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tc in sorted(report["groups"]):
        for arm in ("MA", "CA", "MB", "MC"):
            cell = report["groups"][tc].get(arm)
            if cell is None:
                continue
            lines.append(
                f"| {tc} | {arm} | {cell['total_runs']} | "
                f"{cell['cart_success']['display']} | "
                f"{cell['correct_clarification']['display']} | "
                f"{cell['task_appropriate']['display']} | "
                f"{cell['physical_correct_count']} | "
                f"{cell['known_false_completions']} | "
                f"{cell['false_completion_over_declared']['display']} | "
                f"{cell['false_completion_over_all']['display']} | "
                f"{cell['cost']['calls_made']} | "
                f"{cell['cost']['consumed_tokens']} |")
    lines.append("")
    lines.append("## Totals by arm (outcome paired with cost, R3)")
    lines.append("")
    for arm in ("MA", "CA", "MB", "MC"):
        c = report["totals_by_arm"][arm]
        lines.append(
            f"- {arm}: runs {c['total_runs']}, task-appropriate "
            f"{c['task_appropriate']['display']}, physical correct "
            f"{c['physical_correct_count']}, known-false "
            f"{c['known_false_completions']}, unverified "
            f"{c['unverified_completions']}, calls made/refused "
            f"{c['cost']['calls_made']}/{c['cost']['calls_refused']}, "
            f"tokens reserved/consumed {c['cost']['reserved_tokens']}/"
            f"{c['cost']['consumed_tokens']}, per-run calls "
            f"{c['cost_per_run']['calls_made']}")
    mc = report["totals_by_arm"].get("MC") or {}
    if mc:
        r = mc["recovery"]
        lines.append("")
        lines.append("## MC recovery")
        lines.append(f"- eligible {r['eligible']}, success {r['rate']['display']} "
                     f"(attempted {r['attempted']}, no-attempt {r['no_attempt']}, "
                     f"unknown {r['unknown']}, not-applicable {r['not_applicable']})")
        lines.append(f"- interpretation-mismatch recovery cases: "
                     f"{r['mismatch_count']} {r['mismatch_reasons']}")
        lines.append(f"- unnecessary corrections (whole cart, LOWER BOUND): "
                     f"{mc['unnecessary_correction']['display']}; "
                     f"determinate diagnostic: "
                     f"{mc['unnecessary_determinate']['display']}; "
                     f"unknown-before-write: "
                     f"{mc['unnecessary_unknown_before_write']}")
        lines.append(f"- guard downgrades: {mc['guard_downgrades']}")
    lines.append("")
    return "\n".join(lines)

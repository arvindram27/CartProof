"""Pilot orchestrator and worker - bounded, fail-closed offline implementation."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import pathlib
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LEXICAL_ROOT = Path(__file__).absolute().parents[1]

from harness import pilot_manifest as mft

BASE_HEAD = "eb06f4f93f3ad7f7a0bb1de7cd2d1451a360a6b0"
ALLOWED_DIRTY = {
    "harness/pilot_manifest.py", "harness/model_pilot.py", "harness/model_smoke.py",
    "harness/model_compare.py", "tests/test_model_pilot.py",
    "tests/test_model_pilot_cli.py", "tests/fixtures/pilot_tripwire/sitecustomize.py",
}
_ALLOWED_SPEC_KEYS = {
    "manifest_id", "protocol_version", "pilot", "pilot_stage", "pilot_cell_id",
    "setup", "fault", "arm", "trial", "fake_script", "transport",
    "allow_live", "_startup_barrier",
}

class PilotError(ValueError):
    pass

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)

def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def _is_link(p: Path) -> bool:
    """Non-following link/reparse classification. A metadata error refuses;
    it is never interpreted as 'not a link'."""
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    except OSError:
        raise PilotError("metadata_unverifiable")
    return stat.S_ISLNK(st.st_mode) or bool(
        getattr(st, "st_file_attributes", 0) & 0x400)


def _validate_ancestor_chain(anchor: Path, target: Path) -> None:
    """Lexical containment + non-following metadata on every component
    anchor..target.

    Never resolves (a resolve could hide a linked ancestor) and never
    opens a link target. Every existing component is classified; a
    contiguous missing tail is allowed only where the caller creates it
    inside this validated layout. Metadata errors refuse (never treated
    as absence or a safe type)."""
    # Raw input checks BEFORE normalization can erase traversal/UNC.
    for raw in (str(anchor), str(target)):
        if raw.startswith("\\\\"):
            raise PilotError("path_unc_escape")
        if ".." in pathlib.PureWindowsPath(raw).parts:
            raise PilotError("path_traversal")
        if re.match(r"^[A-Za-z]:[^\\/]", raw):
            raise PilotError("path_drive_relative")
    anchor_abs = os.path.abspath(str(anchor))
    target_abs = os.path.abspath(str(target))
    if os.path.commonpath([anchor_abs, target_abs]) != anchor_abs:
        raise PilotError("path_outside_anchor")
    rel = os.path.relpath(target_abs, anchor_abs)
    cur = Path(anchor_abs)
    kind = _inspect_leaf(cur)
    if kind == "missing":
        raise PilotError("anchor_missing")
    if kind == "prohibited_link":
        raise PilotError("anchor_reparse")
    if kind != "directory":
        raise PilotError("anchor_not_directory")
    if rel == ".":
        return
    parts = rel.split(os.sep)
    missing_seen = False
    for i, part in enumerate(parts):
        if part in ("", ".", ".."):
            raise PilotError("path_noncanonical")
        cur = cur / part
        kind = _inspect_leaf(cur)
        if kind == "prohibited_link":
            raise PilotError("path_ancestor_reparse")
        if kind == "missing":
            missing_seen = True
            continue
        if missing_seen:
            raise PilotError("path_noncanonical")
        # Every intermediate component slot must be a real directory.
        if i < len(parts) - 1 and kind != "directory":
            raise PilotError("path_component_not_directory")

def _inspect_leaf(path: Path) -> str:
    """Non-following leaf classification. A metadata error is NEVER
    treated as missing: it raises a fixed unverifiable category."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return "missing"
    except OSError:
        raise PilotError("metadata_unverifiable")
    if getattr(st, "st_file_attributes", 0) & 0x400 or stat.S_ISLNK(st.st_mode):
        return "prohibited_link"
    if stat.S_ISDIR(st.st_mode):
        return "directory"
    if stat.S_ISREG(st.st_mode):
        return "regular_file"
    return "prohibited_type"


def _validate_mutation_chain(anchor: Path, leaf: Path) -> str:
    """Stricter boundary immediately before an actual stage mutation: the
    anchor and every parent component must be existing real directories.
    Returns the exact leaf classification (missing allowed for fresh
    creation, but never adopted)."""
    _validate_ancestor_chain(anchor, leaf)
    anchor_abs = os.path.abspath(str(anchor))
    leaf_abs = os.path.abspath(str(leaf))
    if os.path.commonpath([anchor_abs, leaf_abs]) != anchor_abs:
        raise PilotError("path_outside_anchor")
    rel = os.path.relpath(leaf_abs, anchor_abs)
    if rel == ".":
        raise PilotError("leaf_is_anchor")
    if _inspect_leaf(Path(anchor_abs)) != "directory":
        raise PilotError("anchor_not_directory")
    cur = Path(anchor_abs)
    parts = rel.split(os.sep)
    for i, part in enumerate(parts):
        cur = cur / part
        kind = _inspect_leaf(cur)
        if i < len(parts) - 1:
            if kind == "missing":
                raise PilotError("parent_missing")
            if kind == "prohibited_link":
                raise PilotError("path_ancestor_reparse")
            if kind != "directory":
                raise PilotError("path_component_not_directory")
        else:
            return kind
    raise PilotError("leaf_unresolved")


def _owned_key(path: Path) -> str:
    return os.path.abspath(str(path))


def _capture_owned_identity(path: Path, handle=None):
    """Supported stable identity of a stage-owned regular leaf. Refuses
    rather than claiming ownership when it cannot be established."""
    try:
        st = os.fstat(handle.fileno()) if handle is not None else os.lstat(path)
    except OSError:
        raise PilotError("owned_identity_unavailable")
    if stat.S_ISLNK(st.st_mode) or (
            getattr(st, "st_file_attributes", 0) & 0x400):
        raise PilotError("owned_wrong_type")
    if not stat.S_ISREG(st.st_mode):
        raise PilotError("owned_wrong_type")
    identity = (st.st_dev, st.st_ino)
    if not identity[1]:
        raise PilotError("owned_identity_unavailable")
    return identity


def _write_fresh_leaf(path: Path, data: str, *, anchor: Path, owner: dict,
                      encoding: str = "utf-8") -> None:
    """Fresh owned leaf: the exact leaf and its full parent chain are
    revalidated, then creation is EXCLUSIVE (x mode). A collision after
    validation refuses; it never truncates or adopts the occupant."""
    kind = _validate_mutation_chain(anchor, path)
    if kind == "prohibited_link":
        raise PilotError("leaf_link_refused")
    if kind == "regular_file":
        raise PilotError("leaf_preexisting")
    if kind in ("directory", "prohibited_type"):
        raise PilotError("leaf_wrong_type")
    if kind != "missing":
        raise PilotError("leaf_wrong_type")
    try:
        handle = open(path, "x", encoding=encoding)
    except FileExistsError:
        raise PilotError("leaf_preexisting")
    except OSError:
        raise PilotError("leaf_write_failed")
    try:
        with handle:
            handle.write(data)
            identity = _capture_owned_identity(path, handle)
    except OSError:
        raise PilotError("leaf_write_failed")
    owner[_owned_key(path)] = identity


def _write_update_leaf(path: Path, data: str, *, anchor: Path, owner: dict,
                       encoding: str = "utf-8") -> None:
    """Update of the exact registered stage-owned leaf. Missing, replaced,
    linked, wrong-type or unverifiable leaves refuse before truncation."""
    expected = owner.get(_owned_key(path))
    if expected is None:
        raise PilotError("update_not_owned")
    kind = _validate_mutation_chain(anchor, path)
    if kind == "prohibited_link":
        raise PilotError("leaf_link_refused")
    if kind != "regular_file":
        raise PilotError("leaf_wrong_type")
    if _capture_owned_identity(path) != expected:
        raise PilotError("owned_identity_mismatch")
    try:
        handle = open(path, "r+", encoding=encoding)
    except OSError:
        raise PilotError("owned_update_open_failed")
    try:
        with handle:
            st = os.fstat(handle.fileno())
            if (st.st_dev, st.st_ino) != expected:
                raise PilotError("owned_identity_mismatch")
            handle.seek(0)
            handle.truncate()
            handle.write(data)
    except OSError:
        raise PilotError("owned_update_failed")


def _fixed_dev_root(root: Path) -> Path:
    # Lexical ancestor checks BEFORE resolve can hide a linked ancestor.
    lexical = Path(os.path.abspath(str(root)))
    _validate_above_anchor(lexical)
    if _is_link(lexical):
        raise PilotError("dev_tree_symlink_escape")
    cur = lexical
    for part in ("eval", "model", "dev"):
        cur = cur / part
        if _is_link(cur):
            raise PilotError("dev_tree_symlink_escape")
    anchor = lexical.resolve()
    dev = (anchor / "eval" / "model" / "dev").resolve()
    if not dev.is_relative_to(anchor):
        raise PilotError("dev_tree_outside_repo_anchor")
    return dev

def _read_pinned_task(root: Path, task_id: str) -> tuple[dict, Path, bytes]:
    dev_root = _fixed_dev_root(root)
    info = mft.TASKS.get(task_id)
    if not info:
        raise PilotError(f"unknown_task:{task_id}")
    tasks_dir = Path(root) / "eval" / "model" / "dev" / "tasks"
    _validate_ancestor_chain(Path(root), tasks_dir)
    if _is_link(tasks_dir):
        raise PilotError("task_dir_reparse")
    task_path = Path(root).resolve() / info["path"]
    if "frozen" in task_path.parts:
        raise PilotError("frozen_refused")
    resolved = task_path.resolve()
    if not resolved.is_relative_to(dev_root) or _is_link(task_path):
        raise PilotError("task_outside_dev")
    raw = task_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != info["sha256"]:
        raise PilotError(f"task_hash_mismatch:{task_id}")
    task = json.loads(raw.decode("utf-8"))
    if not isinstance(task, dict):
        raise PilotError("task_not_object")
    return task, task_path, raw

def _stage_of_cell(cell: dict) -> int:
    for stage, cells in mft.STAGES.items():
        if cell in cells:
            return stage
    raise PilotError("cell_not_in_manifest")

def _validate_script_types(script: dict):
    if not isinstance(script, dict) or set(script.keys()) != {"budget", "client"}:
        raise PilotError("script_shape_invalid")
    budget = script["budget"]
    if not isinstance(budget, dict) or set(budget.keys()) != {"max_calls", "token_budget", "max_output_tokens"}:
        raise PilotError("budget_shape_invalid")
    for k in ("max_calls", "token_budget", "max_output_tokens"):
        if not _is_int(budget[k]) or budget[k] <= 0:
            raise PilotError(f"budget_type_invalid:{k}")
    client = script["client"]
    if not isinstance(client, dict) or set(client.keys()) != {
            "model_name", "base_url", "reasoning_effort_by_kind", "price_table"}:
        raise PilotError("client_shape_invalid")
    if not isinstance(client["model_name"], str) or not client["model_name"]:
        raise PilotError("model_invalid")
    if client["base_url"] != mft.BASE_URL:
        raise PilotError("origin_invalid")
    efforts = client["reasoning_effort_by_kind"]
    if not isinstance(efforts, dict) or set(efforts.keys()) != {"intent", "action", "review"}:
        raise PilotError("reasoning_shape_invalid")
    for k, v in efforts.items():
        if v != "medium":
            raise PilotError(f"reasoning_invalid:{k}")
    table = client["price_table"]
    if not isinstance(table, dict) or set(table.keys()) != {"as_of", "source", "per_1m_tokens"}:
        raise PilotError("price_shape_invalid")
    if table["as_of"] != "2026-10-31" or not isinstance(table["source"], str):
        raise PilotError("price_metadata_invalid")
    per = table["per_1m_tokens"]
    if not isinstance(per, dict) or set(per.keys()) != {client["model_name"]}:
        raise PilotError("price_model_mismatch")
    rates = per[client["model_name"]]
    if not isinstance(rates, dict) or set(rates.keys()) != {"input", "cached_input", "output"}:
        raise PilotError("price_rates_shape_invalid")
    for k, v in rates.items():
        if not _is_num(v) or v < 0:
            raise PilotError(f"price_rate_invalid:{k}")

def build_cell_spec(cell: dict, task: dict, barrier=None) -> dict:
    stage = _stage_of_cell(cell)
    prof = mft.PROFILES[cell["profile"]]
    script = {
        "budget": {
            "max_calls": mft.LIMITS["max_calls"],
            "token_budget": mft.LIMITS["token_budget"],
            "max_output_tokens": mft.LIMITS["max_output_tokens"],
        },
        "client": {
            "model_name": prof["model_name"],
            "base_url": prof["base_url"],
            "reasoning_effort_by_kind": dict(prof["reasoning_effort_by_kind"]),
            "price_table": json.loads(json.dumps(prof["price_table"])),
        },
    }
    spec = {
        "manifest_id": mft.MANIFEST_ID,
        "protocol_version": mft.PROTOCOL_VERSION,
        "pilot": True,
        "pilot_stage": stage,
        "pilot_cell_id": cell["cell_id"],
        "setup": {
            "request_text": task["runtime"]["request_text"],
            "seed_lines": task["runtime"].get("seed_lines") or [],
        },
        "fault": copy.deepcopy(task.get("fault")),
        "arm": cell["arm"],
        "trial": 1,
        "fake_script": script,
        "transport": "meta",
        "allow_live": True,
    }
    if barrier is not None:
        spec["_startup_barrier"] = barrier
    return spec

def _canonical_spec_for_cell(cell: dict, root: Path, barrier=None):
    task, _, _ = _read_pinned_task(root, cell["task_id"])
    return build_cell_spec(cell, task, barrier=barrier), task

def validate_worker_spec(spec: dict, root: Path) -> tuple[dict, dict]:
    if not isinstance(spec, dict):
        raise PilotError("spec_not_object")
    if set(spec.keys()) - _ALLOWED_SPEC_KEYS or "smoke" in spec:
        raise PilotError("pilot_spec_extra_keys")
    if set(spec.keys()) != _ALLOWED_SPEC_KEYS:
        raise PilotError("pilot_spec_missing_keys")
    if spec.get("pilot") is not True:
        raise PilotError("pilot_flag_invalid")
    if not _is_int(spec.get("protocol_version")) or spec["protocol_version"] != mft.PROTOCOL_VERSION:
        raise PilotError("protocol_version_mismatch")
    if spec.get("manifest_id") != mft.MANIFEST_ID:
        raise PilotError("manifest_id_mismatch")
    if not _is_int(spec.get("pilot_stage")):
        raise PilotError("pilot_stage_invalid")
    if not _is_int(spec.get("trial")) or spec["trial"] != 1:
        raise PilotError("trial_invalid")
    if spec.get("transport") != "meta":
        raise PilotError("transport_invalid")
    if spec.get("allow_live") is not True:
        raise PilotError("allow_live_invalid")
    if not isinstance(spec.get("setup"), dict) or set(spec["setup"].keys()) != {"request_text", "seed_lines"}:
        raise PilotError("setup_shape_invalid")
    if not isinstance(spec["setup"]["request_text"], str) or not isinstance(spec["setup"]["seed_lines"], list):
        raise PilotError("setup_type_invalid")
    barrier = spec.get("_startup_barrier")
    if not isinstance(barrier, dict) or set(barrier.keys()) != {"ready", "release"}:
        raise PilotError("barrier_shape_invalid")
    if not isinstance(barrier["ready"], str) or not isinstance(barrier["release"], str):
        raise PilotError("barrier_type_invalid")
    _validate_script_types(spec.get("fake_script"))
    stage = spec["pilot_stage"]
    cells = mft.STAGES.get(stage)
    if not cells:
        raise PilotError("stage_invalid")
    cell = next((c for c in cells if c["cell_id"] == spec.get("pilot_cell_id")), None)
    if not cell:
        raise PilotError("cell_not_in_manifest")
    if spec.get("arm") != cell["arm"]:
        raise PilotError("arm_mismatch")
    canonical, task = _canonical_spec_for_cell(cell, root, barrier=barrier)
    for k in _ALLOWED_SPEC_KEYS:
        if spec.get(k) != canonical.get(k):
            raise PilotError(f"spec_mismatch:{k}")
    fault = spec["fault"]
    if not isinstance(fault, dict) or set(fault.keys()) != {"name", "params"}:
        raise PilotError("fault_shape_invalid")
    if not isinstance(fault["params"], dict):
        raise PilotError("fault_params_invalid")
    if stage == 1 and fault != {"name": None, "params": {}}:
        raise PilotError("stage1_fault_not_clean")
    if stage == 2 and (fault.get("name") not in ("silent-drop", "wrong-variant")
                       or fault != task.get("fault")):
        raise PilotError("stage2_fault_invalid")
    return cell, task

def _make_pilot_dotenv_fallback(spec: dict, root: Path):
    validate_worker_spec(spec, root)
    from agents.env_loader import make_dotenv_fallback
    return make_dotenv_fallback(root)

# ---------- worker path and barrier validation ----------

def _lexical_abs(p: Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))

def _validate_owned_out(root: Path, out: Path, db: Path, spec_path: Path) -> dict:
    _validate_ancestor_chain(Path(root), Path(out))
    root_abs = _lexical_abs(root)
    var_abs = _lexical_abs(Path(root) / "var")
    out_abs = _lexical_abs(out)
    if os.path.commonpath([root_abs, out_abs]) != root_abs:
        raise PilotError("out_outside_root")
    if os.path.commonpath([var_abs, out_abs]) != var_abs:
        raise PilotError("out_outside_var")
    rel = os.path.relpath(out_abs, var_abs).split(os.sep)
    if len(rel) != 3 or rel[1] != "cells" or not rel[0].startswith("model-pilot-stage"):
        raise PilotError("out_not_owned_cell_layout")
    m = re.fullmatch(r"model-pilot-stage(1|2)-[A-Za-z0-9_.-]+", rel[0])
    if not m:
        raise PilotError("out_stage_dir_invalid")
    stage = int(m.group(1))
    cell_id = rel[2]
    # lexical component checks before resolve
    cur = Path(root)
    for part in ["var", rel[0], "cells", cell_id]:
        cur = cur / part
        if _is_link(cur):
            raise PilotError("out_path_link")
    if Path(out).is_symlink() or not Path(out).is_dir():
        raise PilotError("out_not_regular_dir")
    if _lexical_abs(spec_path) != _lexical_abs(Path(out) / "worker_spec.json"):
        raise PilotError("spec_path_not_owned")
    if _lexical_abs(db) != _lexical_abs(Path(out) / "run.db"):
        raise PilotError("db_path_not_owned")
    for name in ("run.json", "barrier-ready.json", "barrier-release"):
        if os.path.lexists(Path(out) / name):
            raise PilotError(f"preexisting_output:{name}")
    if not Path(spec_path).is_file() or _is_link(Path(spec_path)):
        raise PilotError("spec_not_regular")
    if not Path(db).is_file() or _is_link(Path(db)):
        raise PilotError("db_not_regular")
    return {"stage": stage, "cell_id": cell_id, "dir_name": rel[0]}

def _canonical_barrier(out: Path):
    return {"ready": str(Path(out) / "barrier-ready.json"),
            "release": str(Path(out) / "barrier-release")}

def _error_run_pilot(spec: dict, status: str, error: str) -> dict:
    return {"arm": spec.get("arm"), "trial": spec.get("trial", 1),
            "declared_outcome": None, "raw_verdict": None,
            "execution_status": status, "error": error,
            "seed_lines": (spec.get("setup") or {}).get("seed_lines") or [],
            "items": [], "snapshots": [], "mutations": [],
            "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
            "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0, "delete_line": 0},
            "stop_reasons": [], "final_cookie": None, "final_persisted": None,
            "guard": {"applied": spec.get("arm") in ("MB", "MC")}}

def _write_run(out: Path, run: dict):
    (Path(out) / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")

def pilot_run_one_cli(argv, _root=None, _execute=None):
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--settle-ms", type=int, default=2000)
    args = p.parse_args(argv)
    if args.settle_ms != mft.LIMITS["settle_ms"]:
        print(json.dumps({"error": "settle_ms_not_canonical"}))
        return 2
    root = Path(_root) if _root else _LEXICAL_ROOT
    out = Path(args.out)
    spec_path = Path(args.spec)
    db = Path(args.db)
    try:
        owned = _validate_owned_out(root, out, db, spec_path)
    except Exception as exc:
        print(json.dumps({"error": str(exc) if isinstance(exc, PilotError) else "path_invalid"}))
        return 2
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        _write_run(out, _error_run_pilot({}, "harness-error", "spec_read_failed"))
        return 0
    if not isinstance(spec, dict):
        _write_run(out, _error_run_pilot({}, "harness-error", "spec_not_object"))
        return 0
    if spec.get("pilot_stage") != owned["stage"] or spec.get("pilot_cell_id") != owned["cell_id"]:
        _write_run(out, _error_run_pilot(spec, "harness-error", "spec_out_identity_mismatch"))
        return 0
    canonical_barrier = _canonical_barrier(out)
    if spec.get("_startup_barrier") != canonical_barrier:
        _write_run(out, _error_run_pilot(spec, "harness-error", "barrier_path_mismatch"))
        return 0
    try:
        validate_worker_spec(spec, root)
    except Exception as exc:
        _write_run(out, _error_run_pilot(spec, "harness-error",
                                         str(exc) if isinstance(exc, PilotError) else "spec_validation_failed"))
        return 0
    # Wait only after exact path/spec/barrier validation.
    from harness.model_compare import wait_startup_barrier
    err = wait_startup_barrier(spec, out)
    if err is not None:
        _write_run(out, _error_run_pilot(spec, "harness-error", err))
        return 0
    try:
        fallback = _make_pilot_dotenv_fallback(spec, root)
        execute_fn = _execute
        if execute_fn is None:
            from harness.model_compare import execute_model_run as execute_fn
        run = execute_fn(spec["setup"], spec["fault"], spec["arm"], spec["trial"],
                         spec["fake_script"], str(db), str(out),
                         settle_ms=args.settle_ms, transport="meta",
                         allow_live=True, smoke_guard=True,
                         dotenv_fallback=fallback)
        run["pilot_cell_id"] = spec["pilot_cell_id"]
        run["pilot_manifest_id"] = spec["manifest_id"]
        run["transport_label"] = "LIVE (pilot, paid attempt)"
    except Exception as exc:
        from agents.meta_client import MetaConfigError
        status = "harness-error" if isinstance(exc, MetaConfigError) else "crash"
        run = _error_run_pilot(spec, status,
                               str(exc) if isinstance(exc, MetaConfigError) else traceback.format_exc())
    _write_run(out, run)
    print(json.dumps({"arm": run.get("arm"), "execution_status": run.get("execution_status")}))
    return 0

# ---------- accounting ----------

def _recomputed_cost(sidecar: dict, profile: dict) -> float | None:
    pu = sidecar.get("provider_usage")
    if not isinstance(pu, dict):
        return None
    prompt, completion, cached = pu.get("prompt_tokens"), pu.get("completion_tokens"), pu.get("cached_tokens")
    if not all(_is_int(v) and v >= 0 for v in (prompt, completion, cached)):
        return None
    rates = profile["price_table"]["per_1m_tokens"][profile["model_name"]]
    return round(((prompt - cached) * rates["input"] + cached * rates["cached_input"]
                  + completion * rates["output"]) / 1_000_000, 8)

def pilot_accounting_check(run: dict, profile: dict) -> tuple[bool, list]:
    from harness.model_smoke import validate_paid_accounting
    ok, reasons, made = validate_paid_accounting(run)
    reasons = list(reasons)
    budget = run.get("budget") or {}
    records = budget.get("records") or []
    sidecars = budget.get("sidecar") or []
    if not isinstance(records, list):
        return False, reasons + ["records_container_invalid"]
    # counters vs records
    attempted = budget.get("calls_attempted")
    made_ct = budget.get("calls_made")
    refused_ct = budget.get("calls_refused")
    reserved = budget.get("reserved_tokens")
    consumed = budget.get("consumed_tokens")
    if not _is_int(attempted) or attempted != len(records):
        reasons.append("records_attempted_mismatch")
    made_idx = set()
    refused = 0
    sum_res = 0
    sum_con = 0
    for r in records:
        if not isinstance(r, dict):
            reasons.append("record_container_invalid"); continue
        idx = r.get("call_index")
        if not _is_int(idx) or idx <= 0:
            reasons.append("record_index_invalid"); continue
        if idx in made_idx:
            reasons.append("record_duplicate_index")
        status = r.get("status")
        rr, cc = r.get("reserved"), r.get("consumed")
        if not _is_int(rr) or rr < 0 or not _is_int(cc) or cc < 0:
            reasons.append("record_tokens_invalid"); continue
        sum_res += rr; sum_con += cc
        if status == "refused":
            refused += 1
            if rr != 0 or cc != 0 or r.get("declared_usage") is not None:
                reasons.append("refused_contract_invalid")
        elif status == "error":
            if rr == 0 and cc == 0:
                pass
            else:
                made_idx.add(idx)
        elif status in ("made", "missing_usage", "overrun"):
            made_idx.add(idx)
            if status == "made":
                declared = r.get("declared_usage")
                if not _is_int(declared) or declared != cc:
                    reasons.append("made_contract_invalid")
                if _is_int(declared) and declared > rr:
                    reasons.append("made_declared_exceeds_reserved")
        else:
            reasons.append("record_status_invalid")
    if not _is_int(made_ct) or made_ct != len(made_idx):
        reasons.append("calls_made_mismatch")
    if not _is_int(refused_ct) or refused_ct != refused:
        reasons.append("calls_refused_mismatch")
    if _is_int(reserved) and reserved != sum_res:
        reasons.append("reserved_total_mismatch")
    if _is_int(consumed) and consumed != sum_con:
        reasons.append("consumed_total_mismatch")
    # sidecars exact made-call mapping
    by_idx = {}
    for s in sidecars:
        if not isinstance(s, dict) or not _is_int(s.get("call_index")):
            reasons.append("sidecar_invalid"); continue
        if s["call_index"] in by_idx:
            reasons.append("sidecar_duplicate_index")
        by_idx[s["call_index"]] = s
    if set(by_idx.keys()) != made_idx:
        reasons.append("sidecar_correspondence_mismatch")
    for idx in made_idx:
        s = by_idx.get(idx)
        if not s:
            continue
        pu = s.get("provider_usage")
        if not isinstance(pu, dict) or set(pu.keys()) != {
                "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"}:
            reasons.append("provider_usage_shape_invalid"); continue
        prompt, completion = pu.get("prompt_tokens"), pu.get("completion_tokens")
        total, cached, reasoning = pu.get("total_tokens"), pu.get("cached_tokens"), pu.get("reasoning_tokens")
        if not all(_is_int(v) and v >= 0 for v in (prompt, completion, total, cached, reasoning)):
            reasons.append("provider_tokens_invalid"); continue
        if total != prompt + completion:
            reasons.append("provider_total_mismatch")
        if cached > prompt:
            reasons.append("provider_cached_invalid")
        if reasoning > completion:
            reasons.append("provider_reasoning_invalid")
        rec = next((x for x in records if isinstance(x, dict) and x.get("call_index") == idx), None)
        if rec and rec.get("status") in ("made", "overrun"):
            declared = rec.get("declared_usage")
            if not _is_int(declared) or declared != total:
                reasons.append("record_declared_mismatch")
        cost = s.get("estimated_cost_usd")
        if not _is_num(cost) or cost < 0:
            reasons.append("cost_not_finite")
        else:
            recomputed = _recomputed_cost(s, profile)
            if recomputed is None or abs(recomputed - cost) > 1e-9:
                reasons.append("cost_recompute_mismatch")
    return (not reasons), reasons

# ---------- durable attempt latch / control ----------

def _control_root() -> Path:
    return Path(r"C:\Users\Dhanalakshmi\CartProof\pilot-control")

def _validate_above_anchor(anchor: Path) -> None:
    """Check every lexical ancestor of the declared trusted anchor itself
    (up to the drive root) with non-following metadata. The anchor is the
    fixed production layout root (or an explicitly supplied synthetic test
    root); ancestors above it are only checked, never opened. A metadata
    error refuses; only the not-yet-created anchor itself may be missing."""
    cur = Path(os.path.abspath(str(anchor)))
    for comp in [cur, *cur.parents]:
        kind = _inspect_leaf(comp)
        if kind == "missing":
            if comp == cur:
                continue
            raise PilotError("path_above_anchor_missing")
        if kind == "prohibited_link":
            raise PilotError("path_above_anchor_reparse")
        if kind != "directory":
            raise PilotError("path_above_anchor_not_directory")


def _validate_control_root(path: Path) -> Path:
    path = Path(path)
    _validate_above_anchor(path)
    _validate_ancestor_chain(path.parent.parent, path)
    kind = _inspect_leaf(path)
    if kind == "prohibited_link":
        raise PilotError("control_link")
    if kind not in ("missing", "directory"):
        raise PilotError("control_not_dir")
    path.mkdir(parents=True, exist_ok=True)
    _validate_ancestor_chain(path.parent, path)
    if _inspect_leaf(path) != "directory":
        raise PilotError("control_not_dir")
    return path

def _derive_allowance(stage: int) -> float:
    total = 0.0
    for cell in mft.STAGES[stage]:
        total += mft.PROFILE_ALLOWANCE[cell["profile"]]
    return round(total, 8)

def _reserve_attempt(control_root: Path, stage: int, cells: list[dict],
                     source_digest: str) -> dict:
    root = _validate_control_root(Path(control_root))
    stage_dir = root / "attempts" / f"stage{stage}"
    _validate_ancestor_chain(root, stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    _validate_ancestor_chain(root, stage_dir)
    latch = stage_dir / "stage.latch"
    if _inspect_leaf(latch) != "missing":
        raise PilotError("attempt_already_reserved")
    try:
        with open(latch, "x", encoding="utf-8") as f:
            f.write("consumed\n")
    except FileExistsError:
        raise PilotError("attempt_already_reserved")
    except OSError:
        raise PilotError("attempt_latch_write_failed")
    attempt_id = hashlib.sha256(
        f"{mft.MANIFEST_ID}|{mft.ATTEMPT_PROTOCOL_ID}|{stage}|{source_digest}|{time.time_ns()}".encode()
    ).hexdigest()[:16]
    adir = stage_dir / attempt_id
    adir.mkdir(parents=True, exist_ok=False)
    state = {
        "schema": 2, "attempt_id": attempt_id, "stage": stage,
        "protocol_id": mft.ATTEMPT_PROTOCOL_ID, "manifest_id": mft.MANIFEST_ID,
        "source_digest": source_digest, "cells": [c["cell_id"] for c in cells],
        "reserved_allowance_usd": _derive_allowance(stage),
        "state": "reserved", "known_cost_usd": 0.0,
        "unknown_cost_calls": 0, "unresolved_cost_state": "none",
        "current_cell": None, "child_may_have_started": False,
        "attempted_cells": [], "completed_cells": [],
        "evidence_status": "not_started", "decision": None,
    }
    if os.path.lexists(adir / "reservation.json"):
        raise PilotError("attempt_reservation_preexisting")
    (adir / "reservation.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state

_SOURCE_PATHS = [
    "harness/pilot_manifest.py", "harness/model_pilot.py", "harness/model_smoke.py",
    "harness/model_compare.py", "tests/test_model_pilot.py",
    "tests/test_model_pilot_cli.py", "tests/fixtures/pilot_tripwire/sitecustomize.py",
]

def compute_source_digest(root: Path) -> str:
    chunks=[]
    for rel in _SOURCE_PATHS:
        data=(Path(root)/rel).read_bytes()
        chunks.append(f"{rel}|{len(data)}|{hashlib.sha256(data).hexdigest()}")
    seed = ("cartproof|" + mft.MANIFEST_ID + "|" + mft.ATTEMPT_PROTOCOL_ID + "|" + 
            "eb06f4f93f3ad7f7a0bb1de7cd2d1451a360a6b0|" + "\n".join(chunks))
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()

def _persist_attempt(adir: Path, state: dict, state_name: str,
                     known=None, unknown=None, completed=None,
                     unresolved=None, current_cell="__keep__",
                     child_started=None, attempted=None,
                     evidence_status=None, decision=None):
    """Atomic snapshot replace. Durability: fsync of the temp *file* only;
    parent-directory fsync / Windows flush-to-disk is NOT demonstrated."""
    state = dict(state)
    state["state"] = state_name
    if known is not None: state["known_cost_usd"] = known
    if unknown is not None: state["unknown_cost_calls"] = unknown
    if completed is not None: state["completed_cells"] = completed
    if unresolved is not None: state["unresolved_cost_state"] = unresolved
    if current_cell != "__keep__": state["current_cell"] = current_cell
    if child_started is not None: state["child_may_have_started"] = child_started
    if attempted is not None: state["attempted_cells"] = attempted
    if evidence_status is not None: state["evidence_status"] = evidence_status
    if decision is not None: state["decision"] = decision
    # adir is <control>/attempts/stage<N>/<attempt>; validate the chain
    # from the control-root anchor before writing/replacing.
    _validate_ancestor_chain(adir.parent.parent.parent, adir)
    tmp = adir / "reservation.tmp"
    dest = adir / "reservation.json"
    if _inspect_leaf(tmp) != "missing":
        raise PilotError("attempt_tmp_preexisting")
    if _inspect_leaf(dest) not in ("missing", "regular_file"):
        raise PilotError("attempt_reservation_leaf_unsafe")
    try:
        with open(tmp, "x", encoding="utf-8") as fh:
            fh.write(json.dumps(state, indent=2))
    except FileExistsError:
        raise PilotError("attempt_tmp_preexisting")
    except OSError:
        raise PilotError("attempt_tmp_write_failed")
    with open(tmp, "rb") as f:
        try: os.fsync(f.fileno())
        except OSError: pass
    os.replace(tmp, adir / "reservation.json")
    return state

# ---------- receipt ----------

def _finite_cost(v) -> bool:
    return _is_num(v) and v >= 0

def _valid_sha(v) -> bool:
    return isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None

def _validate_receipt(receipt) -> dict:
    if mft.ACCEPTED_STAGE1_RECEIPT_ID is None or mft.ACCEPTED_STAGE1_MANIFEST_ID is None:
        raise PilotError("stage1_receipt_not_accepted")
    if not isinstance(receipt, dict):
        raise PilotError("receipt_not_object")
    required = {"receipt_id", "manifest_id", "stage", "status", "known_cost",
                "unknown_cost_calls", "cells", "file_hashes"}
    if set(receipt.keys()) != required:
        raise PilotError("receipt_shape_invalid")
    if receipt["receipt_id"] != mft.ACCEPTED_STAGE1_RECEIPT_ID:
        raise PilotError("receipt_id_mismatch")
    if receipt["manifest_id"] != mft.ACCEPTED_STAGE1_MANIFEST_ID:
        raise PilotError("receipt_manifest_mismatch")
    if receipt["stage"] != 1 or not _is_int(receipt["stage"]):
        raise PilotError("receipt_stage_invalid")
    if receipt["status"] != "complete":
        raise PilotError("receipt_status_invalid")
    if not _finite_cost(receipt["known_cost"]):
        raise PilotError("receipt_cost_invalid")
    if not _is_int(receipt["unknown_cost_calls"]) or receipt["unknown_cost_calls"] != 0:
        raise PilotError("receipt_unknown_cost")
    expected_ids = [c["cell_id"] for c in mft.STAGES[1]]
    if receipt["cells"] != expected_ids:
        raise PilotError("receipt_cells_invalid")
    fh = receipt["file_hashes"]
    if not isinstance(fh, dict) or not fh:
        raise PilotError("receipt_hashes_invalid")
    required_files = {"batch.json", "ledger.json", "summary.md", "manifest.json"}
    for cid in expected_ids:
        required_files |= {f"cells/{cid}/run.json", f"cells/{cid}/evaluated.json",
                           f"cells/{cid}/secret-scan.txt"}
    if set(fh.keys()) != required_files:
        raise PilotError("receipt_hashes_incomplete")
    if not all(_valid_sha(v) for v in fh.values()):
        raise PilotError("receipt_hash_invalid")
    return receipt

# ---------- preflight / runner ----------

def _git_scope(root: Path):
    import subprocess
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "-uall"], cwd=str(root), text=True).splitlines()
    except Exception:
        raise PilotError("git_metadata_unavailable")
    if head != BASE_HEAD:
        raise PilotError("head_not_reviewed_base")
    dirty = set()
    for line in status:
        path = line[3:].strip().replace("\\", "/")
        dirty.add(path)
        if path not in ALLOWED_DIRTY:
            raise PilotError(f"unexpected_dirty_path:{path}")
    return {"head": head, "dirty": sorted(dirty)}

def preflight_stage(root: Path, stage: int, production: bool = True) -> dict:
    if not _is_int(stage) or stage not in mft.STAGES:
        raise PilotError("stage_invalid")
    if production:
        if _lexical_abs(Path(root)) != _lexical_abs(PROJECT_ROOT):
            raise PilotError("unexpected_root")
        if sys.platform != "win32":
            raise PilotError("unsupported_platform")
        if not Path(sys.executable).resolve().is_relative_to((Path(root) / ".venv").resolve()):
            raise PilotError("unexpected_interpreter")
        _git_scope(Path(root))
    for cell in mft.STAGES[stage]:
        task, _, _ = _read_pinned_task(root, cell["task_id"])
        if cell["arm"] not in ("CA", "MC"):
            raise PilotError("arm_invalid")
        if cell["profile"] not in mft.PROFILES:
            raise PilotError("profile_invalid")
        if stage == 1 and task.get("fault") != {"name": None, "params": {}}:
            raise PilotError("stage1_task_not_clean")
        if stage == 2 and task.get("fault", {}).get("name") not in ("silent-drop", "wrong-variant"):
            raise PilotError("stage2_task_fault_invalid")
    var = Path(root) / "var"
    if not var.is_dir() or _is_link(var):
        raise PilotError("var_unsafe")
    return {"stage": stage, "cells": list(mft.STAGES[stage])}

def _next_allowance(profile: str) -> float:
    return mft.PROFILE_ALLOWANCE[profile]

def _scan_readable(path: Path) -> dict:
    hits = 0
    cats = []
    files = 0
    for pat in ("MODEL_API_KEY", "Authorization", "Bearer "):
        pass
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {"status": "scan_read_failed", "scanned_files": 0,
                "sentinel_hits": None, "categories": []}
    files += 1
    for pat in ("MODEL_API_KEY", "Authorization", "Bearer "):
        if pat in text:
            hits += 1
            cats.append(pat)
    return {"status": "ok", "scanned_files": files,
            "sentinel_hits": hits, "categories": cats}

class PilotLedger:
    def __init__(self, stage: int, prior_known: float = 0.0):
        self.stage = stage
        self.cells = []
        self.known_cost = prior_known
        self.unknown_calls = 0

def _cost_gate(known: float, next_profile: str) -> str | None:
    allow = _next_allowance(next_profile)
    if known + allow > mft.LIMITS["new_pilot_review_threshold_usd"]:
        return "review_threshold"
    if mft.PRIOR_SMOKE_ESTIMATE_USD + known + allow > mft.LIMITS["total_planning_allowance_usd"]:
        return "planning_allowance"
    return None

def _copy_manifest_cell(cell: dict, source_dir: Path, export_dir: Path):
    rels = ["run.json", "evaluated.json", "secret-scan.txt"]
    for name in rels:
        src = source_dir / name
        if not src.is_file() or _is_link(src):
            raise PilotError(f"export_missing_or_link:{cell['cell_id']}/{name}")
        dst = export_dir / "cells" / cell["cell_id"] / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

def _scan_bytes(data: bytes) -> dict:
    try:
        text = data.decode("utf-8")
    except Exception:
        return {"status": "scan_failed", "sentinel_hits": None, "categories": []}
    hits=[]; cats=[]
    for pat in ("MODEL_API_KEY","Authorization","Bearer "):
        if pat in text: hits.append(pat); cats.append(pat)
    return {"status":"ok","sentinel_hits":len(hits),"categories":cats}

_EXPORT_REASONS = {
    "export_batch_invalid", "export_batch_missing", "export_batch_too_large",
    "export_batch_malformed", "export_stage_invalid",
    "export_manifest_mismatch", "export_protocol_mismatch",
    "export_recorded_invalid", "export_recorded_unknown_cell",
    "export_recorded_identity_mismatch", "export_recorded_duplicate",
    "export_recorded_order_invalid", "export_cells_invalid",
    "export_cells_not_recorded", "export_zero_cells_not_halted",
    "export_cells_not_directory", "export_preexisting",
    "export_staging_preexisting", "export_publish_failed",
    "export_cleanup_failed", "export_cleanup_blocked",
    "export_staging_identity_unavailable", "export_staging_create_failed",
    "export_status_scan_failed", "export_outcome_invalid",
    "export_blocked", "export_missing", "export_cell_not_allowed",
}

def _safe_export_reason(exc) -> str:
    if isinstance(exc, OSError):
        return "export_io_failure"
    if isinstance(exc, PilotError) and str(exc) in _EXPORT_REASONS:
        return str(exc)
    return "export_failed_unapproved"

def export_safe(batch_dir: Path, cells: list[dict]) -> Path:
    return Path(export_safe_outcome(batch_dir, cells)["path"])


def export_safe_outcome(batch_dir: Path, cells: list[dict]) -> dict:
    batch_dir = Path(batch_dir)
    _validate_above_anchor(batch_dir)
    _validate_ancestor_chain(batch_dir.parent.parent, batch_dir)
    if _is_link(batch_dir) or not batch_dir.is_dir():
        raise PilotError("export_batch_invalid")
    batch_path = batch_dir / "batch.json"
    if _is_link(batch_path) or not batch_path.is_file():
        raise PilotError("export_batch_missing")
    if batch_path.stat().st_size > 256 * 1024:
        raise PilotError("export_batch_too_large")
    try:
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
    except Exception:
        raise PilotError("export_batch_malformed")
    if not isinstance(batch, dict):
        raise PilotError("export_batch_malformed")
    stage = batch.get("stage")
    if not _is_int(stage) or stage not in mft.STAGES:
        raise PilotError("export_stage_invalid")
    if batch.get("manifest_id") != mft.MANIFEST_ID:
        raise PilotError("export_manifest_mismatch")
    if not _is_int(batch.get("protocol_version")) \
            or batch["protocol_version"] != mft.PROTOCOL_VERSION:
        raise PilotError("export_protocol_mismatch")
    expected_cells = list(mft.STAGES[stage])
    expected = [c["cell_id"] for c in expected_cells]
    recorded = batch.get("cells")
    if not isinstance(recorded, list):
        raise PilotError("export_recorded_invalid")
    recorded_ids = []
    for row in recorded:
        if not isinstance(row, dict) or not isinstance(row.get("cell_id"), str):
            raise PilotError("export_recorded_invalid")
        recorded_ids.append(row["cell_id"])
        cell = next((c for c in expected_cells
                     if c["cell_id"] == row["cell_id"]), None)
        if cell is None:
            raise PilotError("export_recorded_unknown_cell")
        for key in ("task_id", "arm", "profile"):
            if key in row and row[key] != cell[key]:
                raise PilotError("export_recorded_identity_mismatch")
    if len(recorded_ids) != len(set(recorded_ids)):
        raise PilotError("export_recorded_duplicate")
    if recorded_ids != expected[:len(recorded_ids)]:
        raise PilotError("export_recorded_order_invalid")
    if not isinstance(cells, list):
        raise PilotError("export_cells_invalid")
    ids = []
    for entry in cells:
        if not isinstance(entry, dict) \
                or not isinstance(entry.get("cell_id"), str):
            raise PilotError("export_cells_invalid")
        ids.append(entry["cell_id"])
    if ids != recorded_ids:
        raise PilotError("export_cells_not_recorded")
    if not recorded_ids and batch.get("status") != "halted":
        raise PilotError("export_zero_cells_not_halted")
    _validate_ancestor_chain(batch_dir, batch_dir / "cells")
    cells_dir = batch_dir / "cells"
    if ids and os.path.lexists(cells_dir) and not cells_dir.is_dir():
        raise PilotError("export_cells_not_directory")
    final = batch_dir / "export"
    staging = batch_dir / ".export-staging"
    if os.path.lexists(final) or _is_link(final):
        raise PilotError("export_preexisting")
    if os.path.lexists(staging) or _is_link(staging):
        raise PilotError("export_staging_preexisting")
    delivered = {}
    withheld = []
    categories = []
    _staging_owned = {"path": None, "identity": None}

    def _register_staging():
        # Path ownership is recorded immediately, but deletion is NOT
        # authorized until a stable identity has been captured.
        _staging_owned["path"] = staging
        try:
            st = os.lstat(staging)
        except OSError:
            raise PilotError("export_staging_identity_unavailable")
        identity = (st.st_dev, st.st_ino)
        if not identity[1]:
            raise PilotError("export_staging_identity_unavailable")
        _staging_owned["identity"] = identity

    def _cleanup_staging():
        owned = _staging_owned["path"]
        if owned is None:
            return True
        try:
            _validate_ancestor_chain(batch_dir, owned)
            if _staging_owned["identity"] is None:
                return False
            if _is_link(owned) or not owned.is_dir() or owned.parent != batch_dir:
                return False
            st = os.lstat(owned)
            if (st.st_dev, st.st_ino) != _staging_owned["identity"]:
                return False
            import shutil as _sh
            _sh.rmtree(owned)
            return True
        except Exception:
            return False

    def add(rel, path):
        if _is_link(path) or not path.is_file():
            withheld.append(rel)
            categories.append("missing:" + rel)
            return
        try:
            data = path.read_bytes()
        except Exception:
            withheld.append(rel)
            categories.append("unreadable:" + rel)
            return
        s = _scan_bytes(data)
        if s["status"] != "ok" or s.get("sentinel_hits"):
            withheld.append(rel)
            categories.append("scan:" + rel)
            return
        delivered[rel] = data

    for name in ("batch.json", "ledger.json", "summary.md"):
        add(name, batch_dir / name)
    for cid in ids:
        cell_dir = batch_dir / "cells" / cid
        if _is_link(cell_dir) or not cell_dir.is_dir():
            withheld.append(f"cells/{cid}")
            categories.append("cell_invalid:" + cid)
            continue
        for name in ("run.json", "evaluated.json", "secret-scan.txt"):
            src = cell_dir / name
            if name == "run.json" and not src.is_file():
                diag = cell_dir / "parent-diagnostic.json"
                rel = f"cells/{cid}/parent-diagnostic.json"
                if _is_link(diag) or not diag.is_file():
                    withheld.append(rel)
                    categories.append("missing:" + rel)
                    continue
                try:
                    obj = json.loads(diag.read_text(encoding="utf-8"))
                except Exception:
                    withheld.append(rel)
                    categories.append("parent_diagnostic_invalid:" + rel)
                    continue
                if not isinstance(obj, dict) \
                        or obj.get("origin") != "parent-diagnostic" \
                        or not _is_int(obj.get("schema")):
                    withheld.append(rel)
                    categories.append("parent_diagnostic_invalid:" + rel)
                    continue
                add(rel, diag)
            else:
                add(f"cells/{cid}/{name}", src)
    if withheld:
        try:
            _validate_ancestor_chain(batch_dir, staging)
            staging.mkdir(parents=True, exist_ok=False)
            _register_staging()
            status = {"status": "halted", "mode": "halted",
                      "origin": "export_scan", "stage": stage,
                      "selection": recorded_ids,
                      "categories": sorted(set(categories)),
                      "withheld": withheld}
            status_bytes = json.dumps(status, indent=2).encode("utf-8")
            if _scan_bytes(status_bytes)["sentinel_hits"]:
                raise PilotError("export_status_scan_failed")
            (staging / "halted-status.json").write_bytes(status_bytes)
            (staging / "manifest.json").write_text(json.dumps(
                {"mode": "halted", "stage": stage, "selection": recorded_ids,
                 "withheld": withheld,
                 "files": {"halted-status.json":
                           hashlib.sha256(status_bytes).hexdigest()}},
                indent=2), encoding="utf-8")
            _validate_ancestor_chain(batch_dir, staging)
            os.rename(staging, final)
        except PilotError as _exc:
            if _staging_owned["path"] is not None and not _cleanup_staging():
                if str(_exc) == "export_staging_identity_unavailable":
                    raise
                raise PilotError("export_cleanup_blocked")
            raise
        except Exception:
            cleanup_ok = _cleanup_staging()
            raise PilotError("export_cleanup_failed" if not cleanup_ok
                             else "export_publish_failed")
        return {"path": str(final), "mode": "halted",
                "publication_status": "halted_package",
                "withheld": withheld}
    try:
        staging.mkdir(parents=True, exist_ok=False)
    except Exception:
        raise PilotError("export_staging_create_failed")
    _register_staging()  # PilotError here retains staging, never deletes
    try:
        for rel, data in delivered.items():
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        if not recorded_ids:
            mode = "halted"
        elif len(ids) == len(expected):
            mode = "normal"
        else:
            mode = "partial"
        hashes = {rel: hashlib.sha256(data).hexdigest()
                  for rel, data in delivered.items()}
        (staging / "manifest.json").write_text(json.dumps(
            {"mode": mode, "stage": stage, "selection": recorded_ids,
             "withheld": [], "files": hashes}, indent=2), encoding="utf-8")
        _validate_ancestor_chain(batch_dir, staging)
        os.rename(staging, final)
    except Exception:
        cleanup_ok = _cleanup_staging()
        raise PilotError("export_cleanup_failed" if not cleanup_ok
                         else "export_publish_failed")
    return {"path": str(final), "mode": mode,
            "publication_status": "published", "withheld": []}

def _prepare_cell(root: Path, cell: dict, out_root: Path, prior_known: float):
    """Pre-spawn phase: task/template/admission only. Cannot spawn a child;
    a failure here is proven no-start at this boundary."""
    task, _, _ = _read_pinned_task(root, cell["task_id"])
    rundir = out_root / "cells" / cell["cell_id"]
    _validate_ancestor_chain(out_root, rundir)
    rundir.mkdir(parents=True, exist_ok=False)
    _validate_ancestor_chain(out_root, rundir)
    from harness.snapshots import backup_copy
    from store.models import init_and_seed
    catalog = str(Path(root) / "store" / "seed_catalog.json")
    template = out_root / "template.db"
    if os.path.lexists(template) and (_is_link(template)
                                       or not template.is_file()):
        raise PilotError("template_leaf_unsafe")
    try:
        init_and_seed(str(template), catalog)
        backup_copy(str(template), str(rundir / "run.db"))
    except Exception as exc:
        raise PilotError(f"prerequisite_failed:{type(exc).__name__}")
    gate = _cost_gate(prior_known, cell["profile"])
    if gate:
        raise PilotError(gate)
    return task, rundir


def _execute_cell(root: Path, cell: dict, out_root: Path, spawn_fn, prior_known: float):
    task, rundir = _prepare_cell(root, cell, out_root, prior_known)
    return _execute_prepared(root, cell, out_root, spawn_fn, task, rundir)


def _execute_prepared(root: Path, cell: dict, out_root: Path, spawn_fn,
                      task: dict, rundir: Path):
    """Spawn-capable phase. Any exception from here may have started a
    child and must remain may-have-started/unresolved."""
    spec = build_cell_spec(cell, task)
    ctx = {"spec": spec, "task": task, "rundir": rundir,
           "timeout_seconds": float(mft.LIMITS["worker_timeout_seconds"]),
           "transport": "meta"}
    raw_run, spawn_info = spawn_fn(ctx)
    raw_path = rundir / "run.json"
    worker_output = raw_path.is_file()
    if worker_output:
        raw_bytes = raw_path.read_bytes()
        try:
            run_data = json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            run_data = {"execution_status": "harness-error", "error": "raw_run_malformed"}
    else:
        run_data = {"execution_status": "harness-error", "error": "missing_worker_output"}
        (rundir / "parent-diagnostic.json").write_text(
            json.dumps({"origin": "parent-diagnostic", "schema": 1,
                        "execution_status": raw_run.get("execution_status") if isinstance(raw_run, dict) else "missing",
                        "error": raw_run.get("error") if isinstance(raw_run, dict) else None}, indent=2),
            encoding="utf-8")
    if not isinstance(run_data, dict):
        run_data = {"execution_status":"harness-error","error":"raw_run_not_object"}
        halt_override = "raw_run_not_object"
    else:
        halt_override = None
    run_copy = copy.deepcopy(run_data)
    run_copy["task_id"] = task["task_id"]
    run_copy["fault_class"] = task.get("fault_class") or "clean"
    run_copy["fault_schedule"] = task.get("fault")
    from harness.model_compare import measure_worker_run
    from harness import model_metrics
    measured = measure_worker_run(run_copy, task, rundir)
    evaluated = model_metrics.enrich_model_run(measured, task)
    evaluated["task_id"] = task["task_id"]
    evaluated["fault_class"] = task.get("fault_class") or "clean"
    evaluated["fault_schedule"] = task.get("fault")
    (rundir / "evaluated.json").write_text(json.dumps(evaluated, indent=2, default=str), encoding="utf-8")
    scans = {}
    for name in ("run.json", "evaluated.json"):
        fp = rundir / name
        scans[name] = _scan_readable(fp) if fp.is_file() else {"status": "missing", "sentinel_hits": None, "scanned_files": 0, "categories": []}
    (rundir / "secret-scan.txt").write_text(json.dumps(scans, indent=2), encoding="utf-8")
    profile = mft.PROFILES[cell["profile"]]
    acct_ok, acct_reasons = pilot_accounting_check(measured, profile)
    from harness.model_smoke import paid_cost_summary
    cost = paid_cost_summary(measured)
    unknown = cost.get("unknown_cost_calls")
    known = cost.get("known_subtotal_usd_estimated")
    halt = None
    if halt_override:
        halt = halt_override
    elif not worker_output:
        halt = "missing_worker_output"
    elif measured.get("execution_status") != "normal":
        halt = f"execution_{measured.get('execution_status')}"
    elif spawn_info.get("cleanup_status") != "verified":
        halt = f"cleanup_{spawn_info.get('cleanup_status')}"
    elif not acct_ok:
        halt = f"accounting_{acct_reasons[0]}" if acct_reasons else "accounting_invalid"
    elif unknown is None or unknown > 0:
        halt = "unknown_cost"
    elif not _finite_cost(known):
        halt = "cost_invalid"
    elif any(s.get("status") != "ok" or s.get("sentinel_hits") for s in scans.values()):
        halt = "scan_failure"
    return {
        "cell_id": cell["cell_id"], "task_id": cell["task_id"], "arm": cell["arm"],
        "profile": cell["profile"], "execution_status": measured.get("execution_status"),
        "cleanup_status": spawn_info.get("cleanup_status"),
        "accounting_ok": acct_ok, "accounting_reasons": acct_reasons,
        "scan": scans, "cost": cost, "known_cost": known or 0.0, "halt": halt,
        "measured_outcome": evaluated.get("measured_outcome"),
        "declared_outcome": evaluated.get("declared_outcome"),
        "worker_output": worker_output, "child_may_have_started": True,
        "unresolved_cost_state": (
            "unresolved" if (not acct_ok or not worker_output or unknown is None
                             or (isinstance(unknown, int) and unknown > 0)
                             or not _finite_cost(known)) else "none"),
        "diagnostic_cost_usd": known if _finite_cost(known) else None,
    }

SCORING_READOUT_LABEL = ("one trial per cell; a plumbing/scoring readout of these "
                         "cells only, not a statistical or model-capability claim")


def _scoring_readout(out_root: Path, cells: list[dict]) -> dict:
    """Per-profile counts derived only from the frozen per-cell evaluated
    records (written at cell evaluation time) plus the existing cost fields.

    A declared COMPLETE is split by the independent measurement: verified
    (measured correct), known-false (measured incorrect) or unverified
    (unknown/missing). Nothing here re-implements scoring semantics; the
    field names and values are those already attached by enrich_model_run.

    Incompleteness is visible, never silent:

    - ``evaluated_unreadable`` counts cells that produced worker output (so
      evaluation happened) but whose evaluated record is missing, malformed,
      an empty object or the wrong type. Such a cell is never scored as zero.
    - ``halted_before_evaluation`` counts cells with no worker output
      (``worker_output`` is not true), using the existing cell-result fields.
      These legitimately have no evaluated record and are not unreadable.

    The batch status is ``complete`` only when no cell is unreadable; a
    legitimate pre-evaluation halt does not make the readout incomplete.
    """
    buckets: dict[str, dict] = {}
    for row in cells:
        profile_id = row.get("profile")
        if profile_id not in mft.PROFILES:
            continue
        bucket = buckets.get(profile_id)
        if bucket is None:
            bucket = buckets[profile_id] = {
                "cells_run": 0,
                "task_appropriate": 0,
                "correct_clarification": 0,
                "physical_correct": 0,
                "declared_complete": {"verified": 0, "known_false": 0,
                                      "unverified": 0},
                "evaluated_unreadable": 0,
                "halted_before_evaluation": 0,
                "known_cost_usd_estimated": 0.0,
                "unknown_cost_calls": 0,
                "unresolved_cost_state": "none",
            }
        bucket["cells_run"] += 1
        record = None
        if row.get("worker_output") is not True:
            bucket["halted_before_evaluation"] += 1
        else:
            try:
                record = json.loads(
                    (Path(out_root) / "cells" / row["cell_id"] / "evaluated.json")
                    .read_text(encoding="utf-8"))
            except Exception:
                record = None
            if not isinstance(record, dict) or not record:
                bucket["evaluated_unreadable"] += 1
                record = None
        if isinstance(record, dict):
            for field in ("task_appropriate", "correct_clarification",
                          "physical_correct"):
                if record.get(field) is True:
                    bucket[field] += 1
            declared = record.get("declared_outcome",
                                  row.get("declared_outcome"))
            measured = record.get("measured_outcome",
                                  row.get("measured_outcome"))
            if declared == "COMPLETE":
                if measured == "correct":
                    bucket["declared_complete"]["verified"] += 1
                elif measured == "incorrect":
                    bucket["declared_complete"]["known_false"] += 1
                else:
                    bucket["declared_complete"]["unverified"] += 1
        cost = row.get("cost") or {}
        known = row.get("known_cost")
        if isinstance(known, bool) or not isinstance(known, (int, float)):
            known = cost.get("known_subtotal_usd_estimated")
        if isinstance(known, (int, float)) and not isinstance(known, bool):
            bucket["known_cost_usd_estimated"] += float(known)
        unknown = cost.get("unknown_cost_calls")
        if isinstance(unknown, int) and not isinstance(unknown, bool):
            bucket["unknown_cost_calls"] += unknown
        if row.get("unresolved_cost_state") == "unresolved":
            bucket["unresolved_cost_state"] = "unresolved"
    for profile_id in buckets:
        buckets[profile_id]["known_cost_usd_estimated"] = round(
            buckets[profile_id]["known_cost_usd_estimated"], 8)
    profiles = {profile_id: buckets[profile_id]
                for profile_id in sorted(buckets)}
    unreadable = sum(p["evaluated_unreadable"] for p in profiles.values())
    return {"status": "complete" if unreadable == 0 else "incomplete",
            "profiles": profiles}


def _scoring_summary_block(batch: dict) -> str:
    readout = batch.get("scoring_readout") or {}
    profiles = readout.get("profiles") or {}
    unreadable = sum(p["evaluated_unreadable"] for p in profiles.values())
    halted = sum(p["halted_before_evaluation"] for p in profiles.values())
    lines = [
        "## Scoring readout",
        "",
        f"- {SCORING_READOUT_LABEL}.",
        "- traversal complete means the permitted cells were visited; "
        "it is not task success.",
        f"- readout status: {str(readout.get('status', 'unknown')).upper()} "
        f"(evaluated_unreadable={unreadable}, "
        f"halted_before_evaluation={halted})",
        "",
        "| profile | cells run | task_appropriate | correct_clarification | "
        "physical_correct | declared COMPLETE verified | "
        "declared COMPLETE known-false | declared COMPLETE unverified | "
        "evaluated unreadable | halted before evaluation | "
        "known cost (USD est.) | unknown-cost calls | unresolved cost state |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile_id, p in profiles.items():
        split = p["declared_complete"]
        lines.append(
            f"| {profile_id} | {p['cells_run']} | {p['task_appropriate']} | "
            f"{p['correct_clarification']} | {p['physical_correct']} | "
            f"{split['verified']} | {split['known_false']} | "
            f"{split['unverified']} | {p['evaluated_unreadable']} | "
            f"{p['halted_before_evaluation']} | "
            f"{p['known_cost_usd_estimated']} | "
            f"{p['unknown_cost_calls']} | {p['unresolved_cost_state']} |")
    return "\n".join(lines) + "\n"


def _run_stage_injected(root: Path, stage: int, spawn_owned, out_root: Path | None = None,
                        receipt=None, production: bool = False, control_root: Path | None = None,
                        source_digest: str = "synthetic-test"):
    preflight_stage(root, stage, production=production)
    if stage == 2:
        _validate_receipt(receipt)
        prior_known = receipt["known_cost"]
    else:
        prior_known = 0.0
    if out_root is None:
        out_root = Path(root) / "var" / f"model-pilot-stage{stage}-{time.strftime('%Y%m%d-%H%M%S')}"
    if _inspect_leaf(out_root) != "missing":
        raise PilotError("out_root_preexisting")
    # Pre-reservation layout check: an unsafe/unverifiable ancestor must not
    # consume a latch. The same checks are repeated immediately before the
    # actual mkdir below.
    _validate_above_anchor(out_root)
    _validate_mutation_chain(out_root.parent, out_root)
    attempt = _reserve_attempt(Path(control_root) if control_root else _control_root(),
                               stage, list(mft.STAGES[stage]), source_digest)
    attempt_dir = (Path(control_root) if control_root else _control_root()) / "attempts" / f"stage{stage}" / attempt["attempt_id"]
    _persist_attempt(attempt_dir, attempt, "running")
    try:
        # Full owned layout revalidated immediately before creation.
        _validate_above_anchor(out_root)
        _validate_mutation_chain(out_root.parent, out_root)
        out_root.mkdir(parents=True, exist_ok=False)
        if _inspect_leaf(out_root) != "directory":
            raise PilotError("stage_root_unverified")
    except (OSError, PilotError):
        # Reservation already consumed: leave a truthful blocking snapshot
        # instead of an unexplained raw error or a silent retry.
        try:
            _persist_attempt(attempt_dir, attempt, "blocked", known=prior_known,
                             unknown=0, unresolved="none", current_cell=None,
                             child_started=False, attempted=[], completed=[],
                             evidence_status="stage_root_create_failed",
                             decision="stop")
        except Exception:
            raise PilotError("status_persistence_failed")
        raise PilotError("stage_root_create_failed")
    _owned_stage: dict = {}
    cells_out = []
    stop_reason = None
    known = prior_known
    unknown_total = 0
    unresolved = "none"
    attempted: list[str] = []
    for cell in mft.STAGES[stage]:
        attempt = _persist_attempt(
            attempt_dir, attempt, "cell_reserved", known=known,
            unknown=unknown_total, unresolved=unresolved,
            current_cell=cell["cell_id"], child_started=False,
            attempted=list(attempted),
            completed=[c["cell_id"] for c in cells_out],
            evidence_status="reserved", decision="pending")
        # Pre-spawn phase: proven no-start is established ONLY here.
        try:
            _task, _rundir = _prepare_cell(root, cell, out_root, known)
        except Exception as exc:
            cat = str(exc) if isinstance(exc, PilotError) else "prerequisite_failed"
            row = {"cell_id": cell["cell_id"], "task_id": cell["task_id"],
                   "arm": cell["arm"], "profile": cell["profile"],
                   "execution_status": "harness-error",
                   "cleanup_status": None, "accounting_ok": False,
                   "accounting_reasons": [cat], "scan": {},
                   "cost": {"status": "not_started",
                            "known_subtotal_usd_estimated": 0.0,
                            "unknown_cost_calls": 0},
                   "known_cost": 0.0, "halt": cat,
                   "measured_outcome": None, "declared_outcome": None,
                   "worker_output": False,
                   "child_may_have_started": False,
                   "unresolved_cost_state": "none",
                   "diagnostic_cost_usd": None}
            attempted.append(cell["cell_id"])
            cells_out.append(row)
            attempt = _persist_attempt(
                attempt_dir, attempt, "cell_recorded", known=known,
                unknown=unknown_total, unresolved=unresolved,
                current_cell=cell["cell_id"], child_started=False,
                attempted=list(attempted),
                completed=[c["cell_id"] for c in cells_out],
                evidence_status="recorded", decision="stop")
            stop_reason = cat
            break
        # Conservative may-start snapshot BEFORE the spawn-capable phase.
        try:
            attempt = _persist_attempt(
                attempt_dir, attempt, "may_start", known=known,
                unknown=unknown_total, unresolved="unresolved",
                current_cell=cell["cell_id"], child_started=True,
                attempted=list(attempted),
                completed=[c["cell_id"] for c in cells_out],
                evidence_status="may_start", decision="pending")
        except Exception:
            stop_reason = "persistence_failed"
            break
        try:
            row = _execute_prepared(root, cell, out_root, spawn_owned,
                                    _task, _rundir)
        except Exception as exc:
            # Spawn-capable phase: text/type can NEVER prove no-start.
            if isinstance(exc, PilotError):
                cat = str(exc)
            elif isinstance(exc, json.JSONDecodeError):
                cat = "json_invalid"
            elif isinstance(exc, OSError):
                cat = "io_failure"
            elif isinstance(exc, (TypeError, ValueError, KeyError)):
                cat = "schema_invalid"
            else:
                cat = "unexpected_exception"
            row = {"cell_id": cell["cell_id"], "task_id": cell["task_id"],
                   "arm": cell["arm"], "profile": cell["profile"],
                   "execution_status": "harness-error", "cleanup_status": None,
                   "accounting_ok": False, "accounting_reasons": [cat],
                   "scan": {}, "cost": {"status": "unresolved",
                                         "known_subtotal_usd_estimated": 0.0,
                                         "unknown_cost_calls": None},
                   "known_cost": 0.0, "halt": cat, "measured_outcome": None,
                   "declared_outcome": None, "worker_output": False,
                   "child_may_have_started": True,
                   "unresolved_cost_state": "unresolved",
                   "diagnostic_cost_usd": None}
        attempted.append(cell["cell_id"])
        cells_out.append(row)
        if isinstance(row["cost"].get("unknown_cost_calls"), int):
            unknown_total += row["cost"]["unknown_cost_calls"]
        if row.get("unresolved_cost_state") == "unresolved":
            unresolved = "unresolved"
        if row.get("accounting_ok"):
            known += row["known_cost"]
        attempt = _persist_attempt(
            attempt_dir, attempt, "cell_recorded", known=known,
            unknown=unknown_total, unresolved=unresolved,
            current_cell=cell["cell_id"],
            child_started=bool(row.get("child_may_have_started")),
            attempted=list(attempted),
            completed=[c["cell_id"] for c in cells_out],
            evidence_status="recorded",
            decision="stop" if row["halt"] else "continue")
        if row["halt"]:
            stop_reason = row["halt"]
            break
    batch = {
        "manifest_id": mft.MANIFEST_ID, "protocol_version": mft.PROTOCOL_VERSION,
        "stage": stage,
        "attempt_id": attempt["attempt_id"],
        "status": "halted" if stop_reason else "complete",
        "traversal_status": "halted" if stop_reason else "complete",
        "known_cost_usd_estimated": known, "unknown_cost_calls": unknown_total,
        "unresolved_cost_state": unresolved,
        "provider_billing_verified": False, "stop_reason": stop_reason,
        "finalization": {
            "traversal_status": "halted" if stop_reason else "complete",
            "candidate_mode": ("normal" if len(cells_out) == len(mft.STAGES[stage])
                               else "partial" if cells_out else "halted"),
            "publication_status": "pending",
        },
        "cells": cells_out,
        "scoring_readout": {
            "label": SCORING_READOUT_LABEL,
            **_scoring_readout(out_root, cells_out),
        },
        "transport_label": "FAKE (injected)" if not production else "LIVE (pilot, paid attempt)",
        "selection": [{"cell_id": c["cell_id"], "task_id": c["task_id"],
                       "arm": c["arm"], "profile": c["profile"]} for c in mft.STAGES[stage]],
    }
    def _fail_finalization(category):
        try:
            _persist_attempt(attempt_dir, attempt, "blocked", known=known,
                             unknown=unknown_total, unresolved=unresolved,
                             current_cell=None, attempted=list(attempted),
                             completed=[c["cell_id"] for c in cells_out],
                             evidence_status=category, decision="stop")
        except Exception:
            raise PilotError("status_persistence_failed")
        raise PilotError(category)

    try:
        _write_fresh_leaf(out_root / "batch.json", json.dumps(batch, indent=2),
                          anchor=out_root, owner=_owned_stage)
        _write_fresh_leaf(out_root / "ledger.json", json.dumps(
            {"stage": stage, "known_cost": known, "unknown_cost_calls": unknown_total,
             "unresolved_cost_state": unresolved,
             "finalization": batch["finalization"], "cells": cells_out}, indent=2),
            anchor=out_root, owner=_owned_stage, encoding="utf-8")
        _write_fresh_leaf(
            out_root / "summary.md",
            f"# Pilot stage {stage} {batch['status']}\n\n"
            f"- transport: {batch['transport_label']}\n"
            f"- traversal complete means all permitted cells were visited; it is not task success.\n"
            f"- selection: {len(mft.STAGES[stage])} fixed cells\n"
            f"- known estimated cost: {known}\n"
            f"- unknown-cost calls: {unknown_total}\n"
            f"- unresolved cost state: {unresolved}\n"
            f"- stop reason: {stop_reason}\n"
            "- publication: pending (observed outcome is recorded in the attempt "
            "snapshot and EXPORT_OUTCOME.json, not in these frozen bytes)\n"
            "- One trial per cell; no statistical or billing claim.\n"
            "\n" + _scoring_summary_block(batch),
            anchor=out_root, owner=_owned_stage, encoding="utf-8")
    except (OSError, PilotError):
        _fail_finalization("metadata_write_failed")
    # Candidate batch/ledger/summary above are frozen before publication;
    # the structured outcome (not the Path wrapper) decides approval.
    evidence_status = "publication_pending"
    export_outcome = None
    try:
        export_outcome = export_safe_outcome(
            out_root, [{"cell_id": c["cell_id"]} for c in cells_out])
        if export_outcome.get("publication_status") == "published":
            evidence_status = "published"
        elif export_outcome.get("publication_status") == "halted_package":
            evidence_status = "halted_package"
            if not stop_reason:
                stop_reason = "export_halted_package"
        else:
            evidence_status = "export_failed_unapproved"
            stop_reason = stop_reason or "export_outcome_invalid"
    except (PilotError, OSError) as exc:
        safe_reason = _safe_export_reason(exc)
        evidence_status = safe_reason
        stop_reason = safe_reason
        batch["status"] = "halted"
        batch["stop_reason"] = safe_reason
        try:
            _write_update_leaf(out_root / "batch.json", json.dumps(batch, indent=2),
                               anchor=out_root, owner=_owned_stage)
        except (OSError, PilotError):
            _fail_finalization("failure_state_write_failed")
        _fail_finalization(safe_reason)
    try:
        _write_fresh_leaf(out_root / "EXPORT_OUTCOME.json", json.dumps(
            {"mode": (export_outcome or {}).get("mode"),
             "publication_status": evidence_status,
             "withheld": (export_outcome or {}).get("withheld", [])},
            indent=2), anchor=out_root, owner=_owned_stage, encoding="utf-8")
    except (OSError, PilotError):
        _fail_finalization("outcome_write_failed")
    # Final status persistence BEFORE any success marker. If it fails, no
    # marker is written and the last truthful snapshot stays blocking.
    try:
        _persist_attempt(attempt_dir, attempt,
                         "halted" if stop_reason else "complete",
                         known=known, unknown=unknown_total, unresolved=unresolved,
                         current_cell=None, attempted=list(attempted),
                         completed=[c["cell_id"] for c in cells_out],
                         evidence_status=evidence_status,
                         decision="stop" if stop_reason else "complete")
    except Exception:
        raise PilotError("status_persistence_failed")
    if evidence_status == "published" and export_outcome is not None:
        try:
            _write_fresh_leaf(out_root / "EXPORT_PATH.txt",
                              str(export_outcome["path"]),
                              anchor=out_root, owner=_owned_stage)
        except (OSError, PilotError):
            _fail_finalization("marker_write_failed")
    return out_root


def run_stage(root: Path, stage: int, live: bool = False):
    if live is not True:
        raise PilotError("live_flag_required")
    from harness.model_smoke import spawn_pilot_worker
    return _run_stage_injected(root, stage, spawn_pilot_worker, production=True,
                               control_root=_control_root(),
                               source_digest=compute_source_digest(root))

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--run-one" in argv:
        return pilot_run_one_cli([a for a in argv if a != "--run-one"], _root=_LEXICAL_ROOT)
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--stage", type=int, choices=[1, 2])
    p.add_argument("--live", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if args.stage is None:
        print("stage required", file=sys.stderr)
        return 2
    if args.preflight_only:
        try:
            preflight_stage(_LEXICAL_ROOT, args.stage, production=True)
        except PilotError as exc:
            print(json.dumps({"preflight": "refused", "reason": str(exc)}))
            return 2
        print(json.dumps({"preflight": "ok", "stage": args.stage}))
        return 0
    if not args.live:
        print("live flag required", file=sys.stderr)
        return 2
    try:
        run_stage(_LEXICAL_ROOT, args.stage, live=True)
    except PilotError as exc:
        print(json.dumps({"run": "refused", "reason": str(exc)}))
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

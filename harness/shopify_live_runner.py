"""WB2 live scenario runner: CartProof's real check path against the live
dev store, synthetic items only, hard request budget, guarded evidence.

Runs the ACTUAL CartProof verification stack - store.mcp_server_v2
ServerState + run_check over a ShopifyLiveBackend - through three
scenarios: (S1) cart matches request, (S2) seeded mismatch the check must
catch, (S3) bounded recovery preserving unrelated items. The runner plays
the agent role with scripted mutations; this is labelled a scripted driver,
never a model run.

Scope constraints (approved WB2 scope, repeated so the file carries its own
limits):
- synthetic CARTPROOF-TEST- SKUs only;
- at most MAX_REQUESTS live Storefront requests, enforced before send;
- NO checkout, checkoutUrl, payment or order operations; no Admin API;
- one run, no silent retry: each scenario's outcome is validated
  IMMEDIATELY and any unexpected result stops the run with evidence kept;
  S2's mismatch verdict is an EXPECTED pass condition, validated against
  the exact discrepancy set, not merely "some PARTIAL".

Failure/archive discipline (uniform on every path): every archived
structure passes backend redaction (cart ID / ?key= capability); a guard
scan (token value, cart capability, Bearer/shpat_/credential patterns) runs
BEFORE the manifest is written; ANY offender means exit 4 and NO manifest
on every path. Errors are recorded bounded and redacted - never raw
tracebacks. A separate --finalize mode copies the console and offline-test
logs INTO the run dir and re-runs the guard over the exact archive closure
before the manifest that the ZIP will contain.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from store.mcp_server_v2 import ServerState, run_check

from harness.shopify_live_backend import (GUARD_PATTERNS,
                                          LIVE_OPT_IN_ENV, TOKEN_ENV,
                                          ShopifyLiveBackend, resolve_token)

SHOP = "cartproof.myshopify.com"
API_VERSION = "2026-07"
SYNTHETIC_PREFIX = "CARTPROOF-TEST-"
MAX_REQUESTS = 20

# Dependency files whose hashes are RECORDED as evidence (not gated): the
# behavioral gate is the offline test suite run on the user's own tree
# before this runner is allowed to go live.
DEPENDENCY_FILES = [
    "store/backends.py", "store/mcp_server_v2.py", "core/checks.py",
    "core/verify.py", "core/ledger.py", "harness/evaluate.py",
    "harness/shopify_live_backend.py", "harness/shopify_live_runner.py",
]

# GUARD_PATTERNS is single-sourced from harness.shopify_live_backend so
# the R2 capture layer and this evidence-tree scan share one definition.


# The pinned reviewed-tree gates, exactly as established in P43-P50:
# git HEAD and the production source digest (source-v2 recipe over
# DIGEST_PATHS via the project's own helper, raw on-disk bytes - a CRLF
# tree hashes as CRLF). A mismatch STOPS the run and is reported; pins are
# never rewritten or recomputed-and-replaced by this work.
# R3: EXPECTED_SOURCE_DIGEST tracks the approved query-fix re-pin.
EXPECTED_HEAD = "168d9d65cf0f0784a76ada1dffdd5d8cc9acf891"
EXPECTED_SOURCE_DIGEST = ("9b0e32f7f3a225699763fbffdd4cc1ea539befe7"
                          "3e397f386b4d9ff1a6c343b6")

BUDGET_PLAN = {
    "catalog_read": 1,
    "s1_pre_seed_add": 1,
    "s1_baseline_acquire_incl_cart_create": 2,
    "s1_add_request": 1, "s1_check_acquire": 1,
    "s2_baseline_acquire": 1, "s2_seed_wrong_item": 1,
    "s2_check_acquire": 1,
    "s3_baseline_acquire": 1, "s3_recovery_add": 1,
    "s3_check_acquire": 1,
    "planned_total": 12, "cap": MAX_REQUESTS,
}

CART_DISPOSITION = ("abandoned on the dev store; the Storefront API has no "
                    "cart-delete mutation. Shopify staff citing the "
                    "lifecycle docs say abandoned carts expire within ~30 "
                    "days (https://community.shopify.dev/t/carts-storefront"
                    "-api/36813) - reported as a caveat, not a guarantee.")


class UnexpectedFailure(RuntimeError):
    """An unexpected result; the run stops and evidence is kept."""


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded(backend, exc: BaseException) -> str:
    return backend.redact(f"{type(exc).__name__}: {exc}")[:300]


def _head_rev(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _source_digest(repo_root: Path) -> Optional[str]:
    """The production pin recipe, computed by the project's own helper -
    the exact command the P43-P50 gates used."""
    try:
        out = subprocess.run(
            [sys.executable, "-B", "-c",
             "from pathlib import Path; from harness.model_pilot import "
             "compute_source_digest; "
             "print('DIGEST=' + compute_source_digest(Path('.')))"],
            cwd=repo_root, capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            return None
        for line in out.stdout.splitlines():
            if line.startswith("DIGEST="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return None


def preflight(repo_root: Path, run_id: str) -> dict:
    """Env/ordering gates + dependency hash RECORD (evidence, not a gate).

    The offline gate is bound in: the recorded offline test output for THIS
    run id must exist and show the full pass line, so the live runner
    cannot be started ahead of the offline suite.
    """
    offline = repo_root / "var" / "acceptance-p52" \
        / f"offline-tests-{run_id}.txt"
    offline_text = offline.read_text(encoding="utf-8", errors="replace") \
        if offline.exists() else ""
    head_rev = _head_rev(repo_root)
    digest = _source_digest(repo_root)
    gates: dict[str, Any] = {
        "head_rev": head_rev,
        "head_rev_expected": EXPECTED_HEAD,
        "head_rev_matches": head_rev == EXPECTED_HEAD,
        "source_digest": digest,
        "source_digest_expected": EXPECTED_SOURCE_DIGEST,
        "source_digest_matches": digest == EXPECTED_SOURCE_DIGEST,
        "token_available": bool(resolve_token()),
        "live_opt_in_set": os.environ.get(LIVE_OPT_IN_ENV) == "1",
        "offline_gate_file": str(offline),
        "offline_gate_passed": ("offline exit code: 0" in offline_text
                                and bool(re.search(r"\d+ passed",
                                                   offline_text))
                                and " failed" not in offline_text
                                and " error" not in offline_text),
        "dependency_sha256": {
            f: {"present": (repo_root / f).is_file(),
                "sha256": _sha256(repo_root / f)}
            for f in DEPENDENCY_FILES},
        "budget_plan": BUDGET_PLAN,
    }
    gates["ok"] = (gates["head_rev_matches"]
                   and gates["source_digest_matches"]
                   and gates["token_available"]
                   and gates["live_opt_in_set"]
                   and gates["offline_gate_passed"])
    return gates


def _catalog_edges(data) -> list:
    """read_catalog returns the (redacted) body["data"] dict directly, so
    the products connection sits ONE level down - not two. (The earlier
    extra .get("data") never surfaced offline because tests inject the
    mapping; it would have reported catalog_shortfall on a healthy live
    read. Found by inspection during R2.)"""
    products = (data or {}).get("products") or {}
    return products.get("edges") or []


def build_mapping_from_catalog(backend: ShopifyLiveBackend,
                               workdir: Path) -> dict:
    """One catalog read; keep only synthetic SKUs; stop if unusable."""
    data = backend.read_catalog()
    edges = _catalog_edges(data)
    variants: dict[str, str] = {}
    for edge in edges:
        node = (edge or {}).get("node") or {}
        for vedge in (node.get("variants") or {}).get("edges") or []:
            vnode = (vedge or {}).get("node") or {}
            sku = vnode.get("sku")
            gid = vnode.get("id")
            if isinstance(sku, str) and sku.startswith(SYNTHETIC_PREFIX) \
                    and isinstance(gid, str) and gid:
                variants[sku] = gid
    summary = {"product_edges_seen": len(edges),
               "synthetic_skus_found": sorted(variants)}
    (workdir / "catalog_summary.json").write_text(
        json.dumps(backend.redact(summary), indent=2, sort_keys=True),
        encoding="utf-8")
    if len(variants) < 3:
        raise UnexpectedFailure(
            f"catalog_shortfall: need >=3 synthetic {SYNTHETIC_PREFIX}* "
            f"SKUs, found {len(variants)}; adding products is a separate "
            "admin-side approval")
    return {"variants": variants}


def _write_task(workdir: Path, request_text: str, items: list,
                seed_lines: list) -> Path:
    task = {"runtime": {"request_text": request_text,
                        "seed_lines": seed_lines},
            "expected_interpretation": {
                "items": items, "budget_max_cents": None,
                "substitution_allowed": False}}
    path = workdir / "task.json"
    path.write_text(json.dumps(task, indent=2), encoding="utf-8")
    return path


def _add(backend: ShopifyLiveBackend, sku: str, qty: int,
         scenario: str) -> None:
    """A driver mutation whose result MUST be ok; anything else stops."""
    result = backend.add_line(sku, qty, expected_version=None)
    if not result.get("ok"):
        detail = backend.redact(json.dumps(result.get("payload")))[:200]
        raise UnexpectedFailure(f"mutation_failed:{scenario}:{detail}")


def _run_scenario(backend: ShopifyLiveBackend, workdir: Path,
                  name: str, request_text: str, items: list,
                  seed_lines: list, driver, pre_driver=None) -> dict:
    """One scenario: optional pre-seed, baseline (ServerState), scripted
    driver, real check. Seed lines MUST land before the baseline
    acquisition so run_check's survival semantics are meaningful."""
    sdir = workdir / name
    sdir.mkdir(parents=True, exist_ok=True)
    task_path = _write_task(sdir, request_text, items, seed_lines)
    if pre_driver is not None:
        pre_driver(backend)
    state = ServerState(backend, str(task_path), str(sdir / "log"))
    driver(backend, state)
    started = time.perf_counter()
    result = run_check(state)
    check_ms = round((time.perf_counter() - started) * 1000, 1)
    result = backend.redact(result)
    result["check_duration_ms"] = check_ms
    (sdir / "check_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {"scenario": name, "verdict": result["verdict"],
            "mismatches": result["mismatches"],
            "seed_lines": result["seed_lines"],
            "resolution": result["resolution"],
            "check_duration_ms": check_ms}


def _validate(backend, outcome: dict, want_verdict: str,
              want_mismatches: Optional[set] = None) -> dict:
    """Immediate per-scenario validation; any miss stops the run."""
    problems = []
    if outcome["verdict"] != want_verdict:
        problems.append(f"verdict {outcome['verdict']} != {want_verdict}")
    if not all(s["survived_unchanged"] for s in outcome["seed_lines"]):
        # Seed loss stops the run on EVERY scenario (S1/S2/S3 alike): no
        # follow-on mutation may touch a cart whose preserved lines moved.
        problems.append("a preserved line did not survive unchanged")
    if want_mismatches is not None:
        actual = {(m["sku"], m["expected_quantity"], m["actual_quantity"])
                  for m in outcome["mismatches"]}
        if actual != want_mismatches:
            problems.append(
                f"discrepancy set {sorted(actual)} != "
                f"expected {sorted(want_mismatches)}")
    if problems:
        detail = backend.redact("; ".join(problems))[:300]
        raise UnexpectedFailure(
            f"{outcome['scenario']}_unexpected_verdict:{detail}")
    outcome["validated"] = True
    return outcome


def run(backend: ShopifyLiveBackend, workdir: Path,
        mapping: Optional[dict] = None) -> dict:
    """Full three-scenario run with immediate per-scenario validation.
    ``mapping`` injection is how the OFFLINE tests exercise this exact code
    path against the fake; the live path builds it from one catalog read.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    planned = 11
    if mapping is None:
        mapping = build_mapping_from_catalog(backend, workdir)
        planned = BUDGET_PLAN["planned_total"]
    backend.install_mapping(mapping)
    (workdir / "mapping.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    skus = sorted(mapping["variants"])
    sku_a, sku_b, sku_c = skus[0], skus[1], skus[2]

    outcomes: list[dict] = []
    try:
        # S1 - request matches cart. Seed B x1 lands BEFORE the baseline;
        # the driver then adds the requested A x2. Expect COMPLETE with
        # the seed preserved.
        outcomes.append(_validate(backend, _run_scenario(
            backend, workdir, "s1_match",
            f"Add 2 x {sku_a}.", [{"sku": sku_a, "quantity": 2}],
            [{"sku": sku_b, "quantity": 1}],
            lambda be, st: _add(be, sku_a, 2, "s1_match"),
            pre_driver=lambda be: _add(be, sku_b, 1, "s1_pre_seed")),
            "COMPLETE"))

        # S2 - seeded mismatch: the request asks for 1 more A; the driver
        # instead adds unrequested C. The check MUST catch exactly:
        # A observed 2 vs required 3, C observed 1 vs required 0. That
        # exact discrepancy set is the pass condition.
        outcomes.append(_validate(backend, _run_scenario(
            backend, workdir, "s2_seeded_mismatch",
            f"Add 1 x {sku_a}.", [{"sku": sku_a, "quantity": 1}],
            [{"sku": sku_b, "quantity": 1}],
            lambda be, st: _add(be, sku_c, 1, "s2_seeded_mismatch")),
            "PARTIAL",
            want_mismatches={(sku_a, 3, 2), (sku_c, 0, 1)}))

        # S3 - bounded recovery: the driver adds only the missed A.
        # Expect COMPLETE with every unrelated line (B seed, C from S2)
        # preserved.
        outcomes.append(_validate(backend, _run_scenario(
            backend, workdir, "s3_bounded_recovery",
            f"Add 1 x {sku_a}.", [{"sku": sku_a, "quantity": 1}],
            [{"sku": sku_b, "quantity": 1},
             {"sku": sku_c, "quantity": 1}],
            lambda be, st: _add(be, sku_a, 1, "s3_bounded_recovery")),
            "COMPLETE"))
    except UnexpectedFailure:
        summary = {"outcome": "unexpected_failure",
                   "scenarios_completed": [o["scenario"] for o in outcomes],
                   "request_count": backend.request_count,
                   "request_planned": planned,
                   "request_cap": MAX_REQUESTS,
                   "request_log": backend.request_log,
                   "carts": [{"cart_ref": r, "disposition": CART_DISPOSITION}
                             for r in backend.carts_created]}
        (workdir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        raise

    summary = {"outcome": "completed",
               "all_expectations_met": True,
               "scenarios": outcomes,
               "request_count": backend.request_count,
               "request_planned": planned,
               "request_cap": MAX_REQUESTS,
               "request_log": backend.request_log,
               "carts": [{"cart_ref": r, "disposition": CART_DISPOSITION}
                         for r in backend.carts_created]}
    (workdir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def persist_failure_evidence(backend: ShopifyLiveBackend,
                             workdir: Path, outcome: str,
                             error: str) -> None:
    """R2: the captured request log and a partial run summary are evidence
    on EVERY failure path, not only on a completed run."""
    if not (workdir / "request_log.json").exists():
        (workdir / "request_log.json").write_text(
            json.dumps(backend.request_log, indent=2, sort_keys=True),
            encoding="utf-8")
    if not (workdir / "run_summary.json").exists():
        summary = {"outcome": outcome, "error": error,
                   "request_count": backend.request_count,
                   "request_log": backend.request_log,
                   "carts": [{"cart_ref": ref,
                              "disposition": CART_DISPOSITION}
                             for ref in backend.carts_created]}
        (workdir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8")


def probe_catalog(backend: ShopifyLiveBackend, workdir: Path) -> dict:
    """R2 diagnostic: exactly ONE read-only catalog request (the caller
    caps the budget via set_request_cap(1)); no cart, no mutations.
    Writes catalog_summary.json (counts and SKU strings only) on
    success; on failure main()'s handlers persist the captured request
    log instead."""
    data = backend.read_catalog()
    edges = _catalog_edges(data)
    skus: set = set()
    for edge in edges:
        node = (edge or {}).get("node") or {}
        for vedge in (node.get("variants") or {}).get("edges") or []:
            vnode = (vedge or {}).get("node") or {}
            sku = vnode.get("sku")
            if isinstance(sku, str) and sku:
                skus.add(sku)
    summary = {"product_edges_seen": len(edges),
               "variant_skus_seen": sorted(skus),
               "synthetic_skus_found":
                   sorted(s for s in skus if s.startswith(SYNTHETIC_PREFIX))}
    (workdir / "catalog_summary.json").write_text(
        json.dumps(backend.redact(summary), indent=2, sort_keys=True),
        encoding="utf-8")
    # R2: the SUCCESS path persists the same request evidence a failure
    # would - the count, the enforced cap, and the per-request metadata
    # (status, X-Request-ID) - so a successful probe is reviewable too.
    (workdir / "request_log.json").write_text(
        json.dumps(backend.request_log, indent=2, sort_keys=True),
        encoding="utf-8")
    (workdir / "run_summary.json").write_text(json.dumps({
        "outcome": "probe_completed",
        "request_count": backend.request_count,
        "request_cap": backend.request_cap,
        "request_log": backend.request_log,
        "carts": []}, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def guard_scan(workdir: Path, token: str) -> list:
    """List files carrying capability material (empty means clean)."""
    offenders = []
    for path in sorted(workdir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [p.pattern for p in GUARD_PATTERNS if p.search(text)]
        if token and token in text:
            hits.append("token-value")
        if hits:
            offenders.append({"file": path.name, "patterns": len(hits)})
    return offenders


def write_manifest(workdir: Path) -> dict:
    """Hash EVERY file in the closure except the exact root manifest being
    written - a nested manifest.json is hashed like any other file."""
    entries = {}
    root_manifest = workdir / "manifest.json"
    for path in sorted(workdir.rglob("*")):
        if path.is_file() and path != root_manifest:
            # Portable canonical keys: POSIX separators on every OS, so a
            # manifest written on Windows compares equal on any machine.
            entries[path.relative_to(workdir).as_posix()] = _sha256(path)
    manifest = {"files": entries, "count": len(entries)}
    (workdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def finish(workdir: Path, token: str, clean_exit_code: int) -> int:
    """Uniform closeout for EVERY path: guard BEFORE manifest; ANY offender
    means exit 4 and NO manifest. Any earlier manifest of ours is removed
    before the scan, so a refusal never leaves a stale manifest behind."""
    own_manifest = workdir / "manifest.json"
    if own_manifest.exists():
        own_manifest.unlink()
    offenders = guard_scan(workdir, token)
    (workdir / "guard.json").write_text(json.dumps(
        {"clean": not offenders, "offenders": offenders}, indent=2),
        encoding="utf-8")
    if offenders:
        print(json.dumps({"status": "guard_refused_archive",
                          "offenders": offenders}))
        return 4
    write_manifest(workdir)
    return clean_exit_code


PAYLOAD_FILES = ["harness/shopify_live_backend.py",
                 "harness/shopify_live_runner.py",
                 "tests/test_shopify_live_wb2.py"]


def _copy_refuse(src_path: Path, dst: Path) -> Optional[dict]:
    if dst.exists():
        return {"status": "finalize_refuse_overwrite", "file": dst.name}
    if not src_path.exists():
        return {"status": "finalize_missing_log", "file": src_path.name}
    dst.write_bytes(src_path.read_bytes())
    return None


def _finalize(args) -> int:
    """Close the archive: copy the console/offline/gate logs, snapshot the
    three NEW source files exactly as executed, postcheck the pinned tree,
    then guard the whole closure and only then write the manifest."""
    run_dir = Path(args.finalize)
    if not run_dir.is_dir():
        print(json.dumps({"status": "finalize_missing_dir",
                          "dir": str(run_dir)}))
        return 1
    sources = [src for src in [args.console, args.offline] if src]
    sources.extend(args.gate_log or [])
    for src in sources:
        src_path = Path(src)
        err = _copy_refuse(src_path, run_dir / src_path.name)
        if err:
            print(json.dumps(err))
            return 1
    repo_root = Path(__file__).resolve().parents[1]
    snap = run_dir / "payload-snapshot"
    if snap.exists():
        print(json.dumps({"status": "finalize_refuse_overwrite",
                          "file": "payload-snapshot"}))
        return 1
    snap.mkdir()
    for rel in PAYLOAD_FILES:
        src_path = repo_root / rel
        if not src_path.is_file():
            print(json.dumps({"status": "finalize_missing_payload",
                              "file": rel}))
            return 1
        (snap / src_path.name).write_bytes(src_path.read_bytes())
    postcheck = {"head_rev": _head_rev(repo_root),
                 "head_rev_expected": EXPECTED_HEAD,
                 "source_digest": _source_digest(repo_root),
                 "source_digest_expected": EXPECTED_SOURCE_DIGEST}
    postcheck["tree_unchanged"] = (
        postcheck["head_rev"] == EXPECTED_HEAD
        and postcheck["source_digest"] == EXPECTED_SOURCE_DIGEST)
    (run_dir / "postcheck.json").write_text(
        json.dumps(postcheck, indent=2, sort_keys=True), encoding="utf-8")
    if not postcheck["tree_unchanged"]:
        print(json.dumps({"status": "postcheck_pin_mismatch",
                          "postcheck": postcheck}))
        return 1
    return finish(run_dir, resolve_token(), 0)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=time.strftime(
        "%Y%m%d-%H%M%S", time.gmtime()))
    parser.add_argument("--finalize", default=None)
    parser.add_argument("--console", default=None)
    parser.add_argument("--offline", default=None)
    parser.add_argument("--gate-log", dest="gate_log", action="append",
                        default=None)
    parser.add_argument("--probe-catalog", action="store_true",
                        help="R2 diagnostic: ONE read-only catalog "
                             "request (hard cap 1); no cart run")
    args = parser.parse_args(argv)

    if args.finalize:
        return _finalize(args)

    repo_root = Path(__file__).resolve().parents[1]
    workdir = repo_root / "var" / "acceptance-p52" / f"run-{args.run_id}"
    if workdir.exists():
        print(json.dumps({"status": "refuse_overwrite",
                          "dir": str(workdir)}))
        return 1
    workdir.mkdir(parents=True)

    gates = preflight(repo_root, args.run_id)
    (workdir / "gates.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8")
    if not gates["ok"]:
        print(json.dumps({"status": "preflight_failed",
                          "gates": {k: v for k, v in gates.items()
                                    if k != "dependency_sha256"}}))
        return 1

    token = resolve_token()
    # bootstrap empty map so construction succeeds; the real map is
    # installed on the SAME instance after the catalog read (no rebuild,
    # request accounting and cart identity stay intact).
    (workdir / "mapping.json").write_text(
        json.dumps({"variants": {}}), encoding="utf-8")
    try:
        backend = ShopifyLiveBackend(SHOP, str(workdir / "mapping.json"),
                                     api_version=API_VERSION)
    except Exception as exc:
        (workdir / "run_summary.json").write_text(json.dumps({
            "outcome": "unexpected_failure",
            "error": f"{type(exc).__name__}"}, indent=2), encoding="utf-8")
        print(json.dumps({"status": "backend_construction_failed",
                          "error": type(exc).__name__}))
        return finish(workdir, token, 2)
    try:
        if args.probe_catalog:
            backend.set_request_cap(1)
            probe = probe_catalog(backend, workdir)
            code = finish(workdir, token, 0)
            if code == 0:
                print(json.dumps({"status": "probe_completed", **probe,
                                  "evidence_dir": str(workdir)},
                                 sort_keys=True))
            return code
        summary = run(backend, workdir)
    except UnexpectedFailure as exc:
        persist_failure_evidence(backend, workdir, "stopped",
                                 _bounded(backend, exc))
        (workdir / "run_error.txt").write_text(
            backend.redact(f"{type(exc).__name__}: {exc}") + "\n",
            encoding="utf-8")
        print(json.dumps({"status": "run_stopped",
                          "error_category": type(exc).__name__,
                          "evidence_dir": str(workdir)}))
        return finish(workdir, token, 2)
    except Exception as exc:
        persist_failure_evidence(backend, workdir, "unexpected_failure",
                                 _bounded(backend, exc))
        (workdir / "run_error.txt").write_text(
            backend.redact(f"{type(exc).__name__}: {exc}") + "\n",
            encoding="utf-8")
        print(json.dumps({"status": "run_stopped",
                          "error_category": type(exc).__name__,
                          "evidence_dir": str(workdir)}))
        return finish(workdir, token, 2)

    code = finish(workdir, token, 0)
    if code == 0:
        print(json.dumps({"status": "completed",
                          "all_expectations_met": True,
                          "requests": summary["request_count"],
                          "request_planned": summary["request_planned"],
                          "evidence_dir": str(workdir)}))
    return code


if __name__ == "__main__":
    sys.exit(main())

"""Loopback UI/control: the checked owner-launch path, a real CLI entry, and a
sanitized alias-only snapshot server. The worker is the sole owner of the
model, Shopify backend, fresh cart and sequence; this process never sees cart
IDs, capabilities, tokens, SKUs or catalog fields.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from harness.joint_demo_candidate import DemoStop, need

SNAPSHOT_SCHEMA = "cartproof-demo-ui-v1"
ROUTE = "cartproof-demo-ui-v1"
MODE_OFFLINE = "offline"
MODE_LIVE = "live-release"
_HOLD_DEFAULT = 6.0
_HOLD_MAX = 60.0
_RESERVE_SECONDS = 60.0
_TIMEOUT_CEILING = 360.0
_STALE_HEARTBEAT_SECONDS = 60.0

_PHASES = frozenset({"not_started", "validate", "matching_intent", "fresh_cart",
                     "seed", "matching_action", "fault_intent", "controlled_drop",
                     "recovery", "complete", "failed"})
_STATUSES = frozenset({"not_started", "running", "complete", "failed", "unknown"})
_VERDICTS = frozenset({None, "COMPLETE", "PARTIAL", "UNKNOWN"})
_ACTIONS = frozenset({None, "add_to_cart", "finish"})
_KEYS = frozenset({"schema", "status", "phase", "request", "target", "observed",
                   "verdict", "missing", "action", "controlled_drop",
                   "controlled_drop_no_request", "recovery", "read_seq",
                   "checked_read_seq", "event_seq", "read_at_ms", "check_at_ms",
                   "model_calls", "shopify_requests", "saved_check", "freshness"})
FORWARDING_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
                      "cf-connecting-ip")
SNAPSHOT_NAME = "snapshot.json"
RUN_NAME = "run.json"
CSRF_COOKIE = "demo_csrf"
SNAPSHOT_MAX_BYTES = 65536

# Exact named browser checks that a verified browser report MUST contain.
REQUIRED_BROWSER_CHECKS = frozenset({
    "initial_pending_no_complete",
    "start_accepted",
    "partial_missing_visible",
    "final_a2_b1_saved_check",
    "cancel_clears_complete",
    "terminal_failure_visible",
    "disconnection_clears_complete",
    "cleanup_verified",
})

# Synthetic offline fixture mapping (guard literal, not a live secret).
OFFLINE_FIXTURE_MAPPING = {"variants": {
    "CARTPROOF-TEST-A": "gid://shopify/ProductVariant/1",
    "CARTPROOF-TEST-B": "gid://shopify/ProductVariant/2"}}


def _qty_map(v) -> Optional[dict]:
    if v is None:
        return None
    if not isinstance(v, dict):
        return None
    out: dict = {}
    for k, q in v.items():
        if k not in ("A", "B"):
            return None
        if not isinstance(q, int) or isinstance(q, bool) or q < 0 or q > 99:
            return None
        out[k] = q
    return out


def _req(v) -> Optional[dict]:
    if not isinstance(v, dict) or set(v) != {"alias", "quantity"}:
        return None
    if v.get("alias") != "A" or v.get("quantity") != 1:
        return None
    return {"alias": "A", "quantity": 1}


def _saved_check(v) -> Optional[dict]:
    if v is None:
        return None
    if not isinstance(v, dict) or set(v) != {"verdict", "at_ms", "target", "observed"}:
        return None
    if v.get("verdict") not in _VERDICTS:
        return None
    if not isinstance(v.get("at_ms"), int) or isinstance(v.get("at_ms"), bool):
        return None
    if v.get("target") is not None and _qty_map(v.get("target")) is None:
        return None
    if v.get("observed") is not None and _qty_map(v.get("observed")) is None:
        return None
    return v


def validate_snapshot(data) -> Optional[dict]:
    """Positive allowlist plus strict cross-field consistency; fails closed.

    A COMPLETE verdict must be backed by non-null matching target+observed,
    no missing units, the exact checked read, and valid read/check times.
    A completed status can never carry a PARTIAL/null verdict.
    """
    if not isinstance(data, dict) or set(data.keys()) != _KEYS:
        return None
    if data.get("schema") != SNAPSHOT_SCHEMA:
        return None
    status = data.get("status")
    phase = data.get("phase")
    verdict = data.get("verdict")
    action = data.get("action")
    if status not in _STATUSES or phase not in _PHASES:
        return None
    if verdict not in _VERDICTS or action not in _ACTIONS:
        return None
    if _req(data.get("request")) is None:
        return None
    target = _qty_map(data.get("target")) if data.get("target") is not None else None
    observed = _qty_map(data.get("observed")) if data.get("observed") is not None else None
    missing = _qty_map(data.get("missing"))
    if missing is None:
        return None
    if data.get("target") is not None and target is None:
        return None
    if data.get("observed") is not None and observed is None:
        return None
    for k in ("controlled_drop", "controlled_drop_no_request", "recovery"):
        if not isinstance(data.get(k), bool):
            return None
    for k in ("read_seq", "checked_read_seq", "event_seq", "read_at_ms",
              "check_at_ms", "model_calls", "shopify_requests", "freshness"):
        v = data.get(k)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            return None
    if data.get("saved_check") is not None and _saved_check(data.get("saved_check")) is None:
        return None

    read_seq = data["read_seq"]
    checked = data["checked_read_seq"]
    read_at = data["read_at_ms"]
    check_at = data["check_at_ms"]

    if verdict == "COMPLETE":
        if status != "complete" or phase != "complete":
            return None
        if target is None or observed is None:
            return None
        if target != observed:
            return None
        if missing != {}:
            return None
        if checked == 0 or checked != read_seq or read_seq == 0:
            return None
        if read_at <= 0 or check_at <= 0 or check_at < read_at:
            return None
    if status == "complete" and verdict != "COMPLETE":
        return None
    if verdict == "PARTIAL":
        if status != "running":
            return None
        if target is None or observed is None:
            return None
        if missing == {}:
            return None
        if checked == 0 or checked != read_seq:
            return None
        if read_at <= 0 or check_at <= 0:
            return None
    if verdict == "UNKNOWN":
        if checked == 0 or checked != read_seq:
            return None
    if status in ("failed", "unknown") and verdict is not None:
        return None
    if status == "not_started" and (verdict is not None or checked != 0):
        return None
    return {
        "schema": data["schema"], "status": status, "phase": phase,
        "request": data["request"], "target": data["target"],
        "observed": data["observed"], "verdict": verdict,
        "missing": data["missing"], "action": action,
        "controlled_drop": data["controlled_drop"],
        "controlled_drop_no_request": data["controlled_drop_no_request"],
        "recovery": data["recovery"], "read_seq": read_seq,
        "checked_read_seq": checked, "event_seq": data["event_seq"],
        "read_at_ms": read_at, "check_at_ms": check_at,
        "model_calls": data["model_calls"],
        "shopify_requests": data["shopify_requests"],
        "saved_check": data["saved_check"], "freshness": data["freshness"],
    }


def _bound_origin(port: int) -> str:
    return "http://127.0.0.1:%d" % port


def _exact_origin_ok(origin: str, bound: str) -> bool:
    try:
        o = urlsplit(origin)
        b = urlsplit(bound)
    except ValueError:
        return False
    return (o.scheme == "http" and b.scheme == "http"
            and o.netloc.lower() == b.netloc.lower()
            and o.path in ("", "/") and not o.query and not o.fragment
            and o.username is None and o.password is None)


def authorize_start(headers: dict, cookie_nonce: Optional[str],
                    bound_origin: str) -> Optional[str]:
    """Pure start-authorization check with exact Origin+Host+port matching."""
    for h in FORWARDING_HEADERS:
        if h in headers:
            return "forwarding_header_rejected"
    origin = headers.get("origin")
    if origin is None or not _exact_origin_ok(origin, bound_origin):
        return "origin_rejected"
    host = headers.get("host", "")
    if not host or host.lower() != urlsplit(bound_origin).netloc.lower():
        return "origin_rejected"
    nonce = headers.get("x-demo-csrf")
    if not cookie_nonce or nonce != cookie_nonce:
        return "csrf_rejected"
    return None


def _validate_hold(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "0.0"
    if f != f or f in (float("inf"), float("-inf")) or f < 0:
        return "0.0"
    return "%.3f" % min(f, _HOLD_MAX)


def _validate_offline_mapping(mapping: dict) -> dict:
    need(isinstance(mapping, dict), "mapping_schema")
    variants = mapping.get("variants", {})
    need(isinstance(variants, dict) and len(variants) >= 2, "catalog_shortfall")
    need(all(isinstance(k, str) and re.fullmatch(r"CARTPROOF-TEST-[A-Z0-9-]{1,40}", k)
             and isinstance(v, str) and re.fullmatch(r"gid://shopify/ProductVariant/[0-9]+", v)
             for k, v in variants.items()), "mapping_invalid")
    need(len(set(variants.values())) == len(variants), "mapping_alias")
    return mapping


def _compute_tests_digest(root: Path) -> str:
    import hashlib
    from harness.joint_demo_control import digest
    return hashlib.sha256((
        digest(root / "tests/test_joint_demo.py")
        + digest(root / "tests/test_joint_demo_control.py")
        + digest(root / "tests/test_joint_demo_parent.py")
        + digest(root / "tests/test_demo_ui.py")).encode()).hexdigest()


def _bounded_read_text(path: Path, max_bytes: int) -> Optional[str]:
    try:
        st = path.lstat()
    except OSError:
        return None
    if stat.S_ISLNK(st.st_mode) or (getattr(st, "st_file_attributes", 0) & 0x400):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
    except OSError:
        return None
    if len(data) > max_bytes:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _verify_browser_proof(root: Path, receipt: dict) -> dict:
    """Verify the referenced browser report/harness/screenshot hashes, the exact
    required checks, and verified cleanup from installed files — never just a
    browser_proof bool."""
    from harness.joint_demo_control import regular_chain, digest
    report_dir = root / "var" / "demo-ui-screenshots"
    report_path = report_dir / "browser-report.json"
    regular_chain(report_path)
    need(digest(report_path) == receipt.get("browser_report_sha256"),
         "browser_report_hash_mismatch")
    need(digest(root / "harness" / "demo_browser.py") == receipt.get("browser_harness_sha256"),
         "browser_harness_hash_mismatch")
    report = json.loads(_bounded_read_text(report_path, 262144) or "")
    need(isinstance(report, dict), "browser_report_schema")
    checks = set(report.get("checks") or [])
    need(REQUIRED_BROWSER_CHECKS <= checks, "browser_checks_missing")
    network = report.get("network") or {}
    parent = network.get("parent") or {}
    pc = parent.get("egress_canary") or {}
    for key in ("dns_egress", "numeric_ip_egress", "connect_egress",
                "nonowned_loopback", "httpx"):
        need(pc.get(key, 0) >= 1, "browser_parent_canary_missing:" + key)
    need(pc.get("token_fallback") is True and pc.get("dotenv_fallback") is True,
         "browser_parent_credential_canary_missing")
    need(bool((network.get("browser") or {}).get("blocked_egress_attempts")),
         "browser_egress_canary_missing")
    worker = network.get("worker") or {}
    need(worker.get("network_denied") is True, "browser_worker_network_unverified")
    need((worker.get("egress_canary_blocked") or {}).get("socket", 0) >= 1,
         "browser_worker_canary_missing")
    cleanup = report.get("cleanup") or {}
    need(cleanup.get("verified") is True, "browser_cleanup_unverified")
    need(bool(cleanup.get("browser_closes"))
         and all(c.get("ok") for c in cleanup["browser_closes"]),
         "browser_close_unverified")
    shots = report.get("screenshots") or {}
    for name in ("initial.png", "partial.png", "final.png", "failure.png",
                 "disconnection.png"):
        p = report_dir / name
        regular_chain(p)
        need(shots.get(name) == digest(p), "browser_screenshot_hash_mismatch")
    return report


def prepare_launch(root: Path, *, mode: str, port: int, hold: str,
                   timeout_s: float, validation_id: Optional[str] = None,
                   source_digest: Optional[str] = None,
                   tests_digest: Optional[str] = None,
                   offline_mapping: Optional[dict] = None,
                   launch_parent: Optional[Path] = None,
                   control_dir: Optional[Path] = None) -> dict:
    """The checked owner-launch path. No credentials, no clients, no transports.

    Recomputes source closure and test/browser-harness digests from installed
    files (never trusting caller-supplied test digests). Offline is the labeled
    injected-fixture path; live-release additionally requires a valid UI receipt
    and verified browser proof before any client exists.
    """
    from harness.joint_demo_control import (regular_chain, verify_source,
                                            verify_offline_receipt,
                                            validate_profile, reserve_attempt,
                                            write_fresh, digest)

    # 1. Common shape/environment validation (no mode -> refuse, never live).
    need(mode in (MODE_OFFLINE, MODE_LIVE), "mode_required")
    need(sys.platform == "win32", "windows_required")
    need(Path(sys.executable).resolve() == (root / ".venv" / "Scripts" / "python.exe").resolve(),
         "project_interpreter_required")
    need("CARTPROOF_INTERNAL_MCP_IMPORT_STUB" not in os.environ, "internal_stub_refused")
    hold_norm = _validate_hold(hold)
    need(isinstance(timeout_s, (int, float)) and not isinstance(timeout_s, bool)
         and timeout_s == timeout_s and 0 < timeout_s <= _TIMEOUT_CEILING, "timeout_invalid")
    need(isinstance(port, int) and not isinstance(port, bool)
         and 1024 <= port <= 65535, "port_invalid")

    # 2. Recompute source closure + test digest from installed files.
    man_path = root / "harness" / "joint_demo_source_manifest.json"
    regular_chain(man_path)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    source = verify_source(root, manifest)
    tests = _compute_tests_digest(root)
    profile_path = root / "harness" / "joint_demo_profile.json"
    regular_chain(profile_path)
    profile = validate_profile(json.loads(profile_path.read_text(encoding="utf-8")))

    mapping = None
    if mode == MODE_OFFLINE:
        # Labeled injected-fixture path. Never reads credentials; no receipt.
        need(offline_mapping is not None, "offline_mapping_required")
        mapping = _validate_offline_mapping(dict(offline_mapping))
    else:
        # live-release: require identity + valid UI receipt + browser proof.
        need(validation_id and source_digest and tests_digest, "live_identity_required")
        need(source == source_digest, "source_digest_mismatch")
        need(tests == tests_digest, "tests_digest_mismatch")
        ctrl = control_dir if control_dir is not None else (root / "var" / "joint-demo-control")
        receipt_path = (ctrl / ("offline-receipt-" + source[:12] + "-" + validation_id + ".json"))
        regular_chain(receipt_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        verify_offline_receipt(receipt, source, tests, digest(profile_path), require_ui=True)
        need(receipt.get("validation_id") == validation_id, "validation_identity_mismatch")
        need(receipt.get("browser_proof") is True, "browser_proof_required")
        _verify_browser_proof(root, receipt)

    # 3. Reserve a fresh owned attempt and write immutable controls.
    parent = launch_parent if launch_parent is not None else (root / "var" / "joint-demo-control")
    attempt = reserve_attempt(parent, os.urandom(8).hex())
    write_fresh(attempt / "profile.json", profile)
    if mapping is not None:
        write_fresh(attempt / "mapping.json", mapping)
    write_fresh(attempt / "run-context.json", {
        "schema": 1, "route": ROUTE, "mode": mode,
        "source_digest": source, "tests_digest": tests,
        "profile_sha256": digest(attempt / "profile.json"),
        "validation_id": validation_id, "release": mode == MODE_LIVE,
        "hold": hold_norm, "port": port, "timeout_seconds": timeout_s,
    })
    return {"mode": mode, "attempt_dir": attempt, "profile": profile,
            "mapping": mapping, "source_digest": source, "tests_digest": tests,
            "validation_id": validation_id, "hold": hold_norm,
            "port": port, "timeout_s": timeout_s}


def verify_worker_result(attempt_dir: Path, profile: dict, mode: str) -> bool:
    """Independent parent-side check of a successful worker return before any
    completed UI status. Never trusts run.json/snapshot existence alone."""
    from harness.joint_demo_control import regular_chain, digest
    run_path = attempt_dir / RUN_NAME
    regular_chain(run_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    need(run.get("complete") is True, "worker_not_complete")
    need(run.get("phase") == "complete", "worker_phase_not_complete")
    want_ctx = "offline_injected" if mode == MODE_OFFLINE else "real_model_and_shopify"
    need(run.get("run_context") == want_ctx, "worker_context_mismatch")
    mc = run.get("model_calls")
    sr = run.get("shopify_requests")
    need(isinstance(mc, int) and not isinstance(mc, bool) and mc <= profile["max_calls"],
         "model_budget_exceeded")
    need(isinstance(sr, int) and not isinstance(sr, bool) and sr <= profile["request_cap"],
         "request_budget_exceeded")
    snap_path = attempt_dir / SNAPSHOT_NAME
    regular_chain(snap_path)
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    need(snap.get("status") == "complete" and snap.get("verdict") == "COMPLETE",
         "snapshot_not_complete")
    need(snap.get("target") == snap.get("observed") and snap.get("target") is not None,
         "snapshot_mismatch")
    need(snap.get("missing") == {}, "snapshot_missing_units")
    # Evidence closure (same independent check as the parent path).
    manifest_path = attempt_dir / "evidence" / "manifest.json"
    regular_chain(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    need(isinstance(entries, dict) and bool(entries), "evidence_manifest_missing")
    actual = {p.relative_to(attempt_dir / "evidence").as_posix()
              for p in (attempt_dir / "evidence").rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    need(actual == set(entries), "evidence_closure_changed")
    for rel, want in entries.items():
        p = attempt_dir / "evidence" / rel
        regular_chain(p)
        need(digest(p) == want, "evidence_hash_changed")
    return True


def _demo_env_block(offline: bool, hold: str, port: int) -> str:
    from harness.model_smoke import _pilot_env_block
    base = _pilot_env_block()
    if not base.endswith("\0\0"):
        raise RuntimeError("env_block_shape")
    base = base[:-1]
    extra = ["CARTPROOF_DEMO_HOLD=" + hold,
             "CARTPROOF_DEMO_PARENT_LAUNCH=reviewed-demo-parent-v1"]
    if offline:
        extra.append("CARTPROOF_DEMO_OFFLINE=1")
    else:
        extra.append("CARTPROOF_DEMO_LIVE_RELEASE=1")
        # Credential routes only for a later separately-approved live run; the
        # values are forwarded unchanged and never read here.
        for name in ("MODEL_API_KEY", "CARTPROOF_SHOPIFY_STOREFRONT_TOKEN",
                     "CARTPROOF_SHOPIFY_STOREFRONT_TOKEN_FILE"):
            if os.environ.get(name):
                extra.append(name + "=" + os.environ[name])
    block = os.environ.get("CARTPROOF_DEMO_BLOCK", "0")
    extra.append("CARTPROOF_DEMO_BLOCK=" + block)
    return base + "\0".join(extra) + "\0\0"


class DemoController:
    """Owns the supervised worker; the sole checked owner-launch path."""

    def __init__(self, *, root: Path, mode: str, port: int = 8765,
                 hold: str = "6.0", timeout_s: float = _TIMEOUT_CEILING,
                 validation_id: Optional[str] = None,
                 source_digest: Optional[str] = None,
                 tests_digest: Optional[str] = None,
                 offline_mapping: Optional[dict] = None,
                 launch_parent: Optional[Path] = None,
                 control_dir: Optional[Path] = None):
        self.root = root
        self.mode = mode
        self.port = port
        self.hold = _validate_hold(hold)
        self.timeout_s = timeout_s
        self.validation_id = validation_id
        self.source_digest = source_digest
        self.tests_digest = tests_digest
        self.offline_mapping = offline_mapping
        self.launch_parent = launch_parent
        self.control_dir = control_dir
        self.bound_origin = _bound_origin(port)
        self.lock = threading.Lock()
        self.started = False
        self.finished = False
        self.cancel = threading.Event()
        self.info = {"job_assigned": False, "timed_out": False,
                     "cleanup_status": "not_started"}
        self.thread = None
        self.job = None
        self.proc = None
        self.run = None
        self.ctx = None
        self._terminal = None   # None | "complete" | "failed" | "unknown"
        self._final_ok = False
        self._exit_note = None
        # Replay / heartbeat tracking (monotonic, while running).
        self._obs = {"read_seq": 0, "freshness": 0, "check_at_ms": 0}
        self._last_advance = time.monotonic()

    def _worker_cmd(self, attempt_dir: Path) -> list:
        return [sys.executable, "-B", "-m", "harness.demo_worker",
                "--worker-dir", str(attempt_dir)]

    def start(self) -> dict:
        with self.lock:
            if self.started:
                return {"started": False, "reason": "already_started"}
            if self.finished:
                return {"started": False, "reason": "finished_terminal"}
            self.started = True  # one-use mark before prepare; blocks concurrent
        try:
            # Real preflight/prepare runs BEFORE any env block, thread, suspended
            # process, credential routing or client construction.
            ctx = prepare_launch(
                self.root, mode=self.mode, port=self.port, hold=self.hold,
                timeout_s=self.timeout_s, validation_id=self.validation_id,
                source_digest=self.source_digest, tests_digest=self.tests_digest,
                offline_mapping=self.offline_mapping,
                launch_parent=self.launch_parent, control_dir=self.control_dir)
        except DemoStop as exc:
            with self.lock:
                self.started = False
                self.finished = True
                self._terminal = "failed"
                self._exit_note = "preflight_refused"
            return {"started": False, "reason": str(exc)}
        except BaseException:
            with self.lock:
                self.started = False
                self.finished = True
                self._terminal = "failed"
                self._exit_note = "preflight_error"
            return {"started": False, "reason": "preflight_error"}
        self.ctx = ctx
        env_block = _demo_env_block(self.mode == MODE_OFFLINE, ctx["hold"], self.port)
        self.thread = threading.Thread(
            target=self._supervise, args=(ctx, env_block), daemon=True)
        self.thread.start()
        return {"started": True}

    def _supervise(self, ctx: dict, env_block: str):
        from harness.model_smoke import _JobObject, _SuspendedProcess, PROJECT_ROOT
        info = self.info
        job = proc = None
        attempt_dir = ctx["attempt_dir"]
        deadline = time.monotonic() + ctx["timeout_s"]
        try:
            job = _JobObject()
            proc = _SuspendedProcess(self._worker_cmd(attempt_dir), PROJECT_ROOT,
                                     attempt_dir / "stdout.txt",
                                     attempt_dir / "stderr.txt",
                                     env_block=env_block)
            self.job = job
            self.proc = proc
            try:
                job.assign(proc.handle)
            except Exception as exc:
                info["cleanup_status"] = "assignment_failed"
                info["error"] = type(exc).__name__
                try:
                    proc.kill(); proc.wait(timeout=10)
                except Exception:
                    pass
                active = job.active_processes()
                info["active_after_terminate"] = active
                if active != 0:
                    info["cleanup_status"] = "unverified"
                self._exit_note = "assignment_failed"
                return
            info["job_assigned"] = True
            try:
                proc.resume()
            except Exception as exc:
                info["cleanup_status"] = "job_error"
                info["error"] = type(exc).__name__
                try:
                    job.terminate(); proc.wait(timeout=10)
                except Exception:
                    pass
                active = job.active_processes()
                info["active_after_terminate"] = active
                if active != 0:
                    info["cleanup_status"] = "unverified"
                self._exit_note = "resume_failed"
                return
            cancelled = False
            while True:
                if self.cancel.is_set():
                    cancelled = True
                    break
                rc = proc.poll()
                if rc is not None:
                    break
                if time.monotonic() > deadline:
                    cancelled = True
                    info["timed_out"] = True
                    break
                time.sleep(0.05)
            if cancelled:
                try:
                    job.terminate(); proc.wait(timeout=10)
                except Exception:
                    pass
                active = job.active_processes()
                info["active_after_terminate"] = active
                info["cleanup_status"] = ("terminated" if active == 0 else "unverified")
                self._exit_note = "cancelled_or_timeout"
            else:
                active = job.active_processes()
                info["active_before_close"] = active
                info["cleanup_status"] = ("verified" if active == 0 else "unverified")
        except Exception as exc:
            info["cleanup_status"] = "job_error"
            info["error"] = type(exc).__name__
            self._exit_note = "supervisor_error"
        finally:
            close_failed = False
            if proc is not None:
                try:
                    if not proc.close():
                        close_failed = True
                except Exception:
                    close_failed = True
            if job is not None:
                try:
                    if not job.close():
                        close_failed = True
                except Exception:
                    close_failed = True
            if close_failed and info.get("cleanup_status") == "verified":
                info["cleanup_status"] = "unverified"
            # Independent verification of a real successful worker return.
            self._terminal = self._classify_terminal(ctx)
            self.finished = True
        self._load_run(attempt_dir)

    def _classify_terminal(self, ctx: dict) -> str:
        info = self.info
        if info.get("cleanup_status") != "verified":
            return "failed"
        if info.get("timed_out"):
            return "failed"
        if self._exit_note == "cancelled_or_timeout":
            return "failed"
        try:
            verify_worker_result(ctx["attempt_dir"], ctx["profile"], ctx["mode"])
            self._final_ok = True
            return "complete"
        except DemoStop:
            self._final_ok = False
            return "failed"
        except BaseException:
            self._final_ok = False
            return "unknown"

    def _load_run(self, attempt_dir: Path):
        rp = attempt_dir / RUN_NAME
        if rp.is_file():
            try:
                self.run = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:
                self.run = None
        else:
            self.run = None

    def snapshot(self) -> dict:
        base = {"schema": SNAPSHOT_SCHEMA, "status": "not_started",
                "phase": "not_started", "request": {"alias": "A", "quantity": 1},
                "target": None, "observed": None, "verdict": None,
                "missing": {}, "action": None, "controlled_drop": False,
                "controlled_drop_no_request": False, "recovery": False,
                "read_seq": 0, "checked_read_seq": 0, "event_seq": 0,
                "read_at_ms": 0, "check_at_ms": 0, "model_calls": 0,
                "shopify_requests": 0, "saved_check": None, "freshness": 0}
        attempt_dir = self.ctx["attempt_dir"] if self.ctx else None
        if attempt_dir is None:
            return base
        sp = attempt_dir / SNAPSHOT_NAME
        raw = _bounded_read_text(sp, SNAPSHOT_MAX_BYTES)
        if raw is None:
            return self._terminal_snapshot(base, "unavailable")
        try:
            out = validate_snapshot(json.loads(raw))
        except Exception:
            out = None
        if out is None:
            return self._terminal_snapshot(base, "unavailable")

        if not self.finished:
            # Parent has not completed independent verification: withhold any
            # worker-side success. A worker-written complete/COMPLETE must not
            # be presented before _classify_terminal + verify_worker_result
            # pass. Downgrade to running/pending and clear the current verdict.
            if out["status"] == "complete" or out["verdict"] == "COMPLETE":
                out["status"] = "running"
                out["verdict"] = None
                out["target"] = None
                out["missing"] = {}
            # While running: reject sequence/time regression and stale heartbeat.
            obs = self._obs
            if (out["read_seq"] < obs["read_seq"]
                    or out["freshness"] < obs["freshness"]
                    or out["check_at_ms"] < obs["check_at_ms"]):
                return self._terminal_snapshot(out, "unavailable")
            if out["freshness"] > obs["freshness"] or out["read_seq"] > obs["read_seq"]:
                self._last_advance = time.monotonic()
            elif out["status"] == "running" and (
                    time.monotonic() - self._last_advance) > _STALE_HEARTBEAT_SECONDS:
                return self._terminal_snapshot(out, "unavailable")
            obs["read_seq"] = out["read_seq"]
            obs["freshness"] = out["freshness"]
            obs["check_at_ms"] = out["check_at_ms"]
            return out

        # Terminal: cancel/timeout/failed/missing/invalid evidence publishes a
        # terminal failed/unknown status with verdict cleared, even when
        # cleanup is 'terminated'. COMPLETE only surfaces after parent closure
        # really passed (_terminal == complete AND _final_ok).
        if self._terminal == "complete" and self._final_ok:
            return out
        if self._terminal == "unknown":
            out["status"] = "unknown"
            out["verdict"] = None
            out["target"] = None
            out["missing"] = {}
        else:
            out["status"] = "failed"
            out["verdict"] = None
            out["target"] = None
            out["missing"] = {}
        return out

    def _terminal_snapshot(self, out: dict, status: str) -> dict:
        out["status"] = status
        out["verdict"] = None
        out["target"] = None
        out["missing"] = {}
        return out

    def stop(self) -> dict:
        self.cancel.set()
        if self.thread is not None:
            self.thread.join(timeout=20)
        cs = self.info.get("cleanup_status")
        if cs in ("verified", "terminated"):
            return {"stopped": True, "cleanup": cs}
        return {"stopped": False, "cleanup": cs}


def create_demo_app(controller: DemoController, templates_dir: Path) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        controller.stop()

    app = FastAPI(title="CartProof Demo UI", lifespan=lifespan)
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.controller = controller

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        snap = controller.snapshot()
        nonce = secrets.token_urlsafe(16)
        resp = templates.TemplateResponse(
            request, "demo_cart.html",
            {"snap": snap, "offline": controller.mode == MODE_OFFLINE, "csrf": nonce,
             "bound_origin": controller.bound_origin})
        resp.set_cookie(CSRF_COOKIE, nonce, httponly=True, samesite="strict")
        return resp

    @app.get("/demo/state")
    def state(request: Request):
        return JSONResponse(controller.snapshot())

    @app.post("/demo/start")
    def start(request: Request):
        err = authorize_start(dict(request.headers),
                              request.cookies.get(CSRF_COOKIE),
                              controller.bound_origin)
        if err:
            return JSONResponse(status_code=403, content={"error": err})
        return JSONResponse(controller.start())

    @app.post("/demo/stop")
    def stop(request: Request):
        err = authorize_start(dict(request.headers),
                              request.cookies.get(CSRF_COOKIE),
                              controller.bound_origin)
        if err:
            return JSONResponse(status_code=403, content={"error": err})
        return JSONResponse(controller.stop())

    return app


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="harness.demo_ui")
    p.add_argument("--mode", choices=(MODE_OFFLINE, MODE_LIVE), default=None,
                   help="offline (injected fixtures) or live-release (full preflight)")
    p.add_argument("--validation-id", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--tests", default=None)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--hold", default="6.0")
    p.add_argument("--timeout-seconds", type=float, default=_TIMEOUT_CEILING)
    p.add_argument("--launch-parent", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Real long-running server entry. Validates argument/mode shapes and
    constructs the controller WITHOUT reserving any attempt; the reserving
    prepare_launch runs only on an accepted authenticated POST /demo/start.
    READY is printed only after the loopback listener actually binds.
    """
    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        need(args.mode in (MODE_OFFLINE, MODE_LIVE), "mode_required")
        need(isinstance(args.port, int) and not isinstance(args.port, bool)
             and 1024 <= args.port <= 65535, "port_invalid")
        need(isinstance(args.timeout_seconds, (int, float))
             and not isinstance(args.timeout_seconds, bool)
             and args.timeout_seconds == args.timeout_seconds
             and 0 < args.timeout_seconds <= _TIMEOUT_CEILING, "timeout_invalid")
        if args.mode == MODE_LIVE:
            need(args.validation_id and args.source and args.tests, "live_identity_required")
        launch_parent = Path(args.launch_parent) if args.launch_parent else None
        offline_mapping = OFFLINE_FIXTURE_MAPPING if args.mode == MODE_OFFLINE else None
        controller = DemoController(
            root=root, mode=args.mode, port=args.port, hold=args.hold,
            timeout_s=args.timeout_seconds, validation_id=args.validation_id,
            source_digest=args.source, tests_digest=args.tests,
            offline_mapping=offline_mapping, launch_parent=launch_parent)
        app = create_demo_app(controller, root / "harness" / "demo_templates")
    except DemoStop as exc:
        print(json.dumps({"status": "STOP", "reason": str(exc)}), flush=True)
        return 2

    # Serve on the loopback. No attempt/worker/client is created here.
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    url = "http://127.0.0.1:%d" % args.port

    def _announce():
        deadline = time.monotonic() + 15.0
        while not getattr(server, "started", False):
            if time.monotonic() > deadline or server.should_exit:
                return
            time.sleep(0.05)
        print(json.dumps({"status": "READY", "mode": args.mode, "url": url}), flush=True)
    threading.Thread(target=_announce, daemon=True).start()

    try:
        server.run()
    except BaseException:
        # Binding/port failure (or serve failure): STOP with no worker.
        print(json.dumps({"status": "STOP", "reason": "serve_failed"}), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

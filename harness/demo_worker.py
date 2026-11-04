"""UI execution worker: sole owner of the model, Shopify backend, fresh cart and
the reviewed JointDemo sequence. Refuses standalone/unowned/live-default launch.
Publishes ONLY an alias-only snapshot; raw cart payloads, IDs, capabilities,
tokens, SKUs and catalog fields never leave this process.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from harness.joint_demo_candidate import DemoStop

SNAPSHOT_SCHEMA = "cartproof-demo-ui-v1"
PHASES = ("not_started", "validate", "matching_intent", "fresh_cart", "seed",
          "matching_action", "fault_intent", "controlled_drop", "recovery",
          "complete", "failed")
VERDICTS = (None, "COMPLETE", "PARTIAL", "UNKNOWN")
ACTIONS = (None, "add_to_cart", "finish")
ALIASES = ("A", "B")

OFFLINE_ENV = "CARTPROOF_DEMO_OFFLINE"
LIVE_RELEASE_ENV = "CARTPROOF_DEMO_LIVE_RELEASE"
HOLD_ENV = "CARTPROOF_DEMO_HOLD"
BLOCK_ENV = "CARTPROOF_DEMO_BLOCK"
PARENT_MARKER = "CARTPROOF_DEMO_PARENT_LAUNCH"
PARENT_VALUE = "reviewed-demo-parent-v1"
ROUTE = "cartproof-demo-ui-v1"
SNAPSHOT_NAME = "snapshot.json"
RUN_NAME = "run.json"
RUN_CONTEXT_NAME = "run-context.json"
PROFILE_NAME = "profile.json"
MAPPING_NAME = "mapping.json"
EMPTY_MAPPING_NAME = "empty-mapping.json"

# Bounded hold / timeout policy (validated here AND in the controller).
HOLD_DEFAULT = 6.0
HOLD_MAX = 60.0
RESERVE_SECONDS = 60.0
TIMEOUT_CEILING = 360.0

REQUEST = {"alias": "A", "quantity": 1}

# Worker env allowlist: base pilot names plus the reviewed demo markers.
# Credential names are NEVER part of the offline allowlist and are forwarded
# only by the separately-authorized live parent path.
_WORKER_ENV_ALLOW = (
    'PATH', 'SYSTEMROOT', 'SYSTEMDRIVE', 'WINDIR', 'TEMP', 'TMP',
    'COMSPEC', 'PATHEXT', 'USERPROFILE', 'LOCALAPPDATA', 'APPDATA',
    'VIRTUAL_ENV', PARENT_MARKER, OFFLINE_ENV, LIVE_RELEASE_ENV,
    HOLD_ENV, BLOCK_ENV,
)


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _safe_qty(v: Any) -> Optional[int]:
    if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 99:
        return v
    return None


def _alias_map(obs: dict, aliases: dict) -> Optional[dict]:
    out: dict[str, int] = {}
    for key, qty in (obs or {}).items():
        alias = aliases.get(key)
        if alias is None or alias not in ALIASES:
            return None
        q = _safe_qty(qty)
        if q is None:
            return None
        out[alias] = out.get(alias, 0) + q
    return out


def _missing(target: dict, observed: dict) -> dict:
    missing: dict[str, int] = {}
    for alias in ALIASES:
        want = target.get(alias, 0)
        have = observed.get(alias, 0)
        if want > have:
            missing[alias] = want - have
    return missing


class PublishingBackend:
    """Wraps acquire_state to capture the latest real read, adding zero calls.

    The read token captured for THIS acquire_state is injected into the returned
    state's provenance as a private ``_read_token`` so that the check consuming
    this read can carry the exact token (not a later emitter-time token)."""

    def __init__(self, backend, on_read):
        self._backend = backend
        self._on_read = on_read

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def acquire_state(self):
        state = self._backend.acquire_state()
        token = self._on_read(state)
        if isinstance(state, dict):
            prov = dict(state.get("provenance") or {})
            prov["_read_token"] = token
            state["provenance"] = prov
        return state


class SnapshotWriter:
    """Couples each verdict to the exact read it checked; a newer read or any
    invalidating event clears the current COMPLETE and publishes immediately.
    Aliases only; private cart identity is captured but never published."""

    def __init__(self, rundir: Path, aliases: dict, backend, model):
        self.rundir = rundir
        self.aliases = aliases
        self.backend = backend
        self.model = model
        self.read_seq = 0
        self.event_seq = 0
        self.freshness = 0
        self.observed = None
        self.resolution = None
        self.pending = True          # a read not yet checked -> no verdict
        self.checked_read_seq = 0
        self.read_time_ms = 0        # actual read time, not write time
        self.check_time_ms = 0       # actual check time
        self.verdict = None
        self.target = None
        self.missing = {}
        self.action = None
        self.drop = False
        self.drop_no_request = False
        self.recovery = False
        self.status = "not_started"
        self.phase = "not_started"
        self.saved_check = None
        self.last_check = None       # safe init: no checked result yet
        self.budget = {"model_calls": 0, "shopify_requests": 0}
        # Private cart identity (never published).
        self._cart_ref = None
        self._cart_mismatch = False
        # Private per-read token: "<read_seq>:<nonce>" (JSON-safe string). Bind
        # a check only to the exact read it examined; never exposed beyond the
        # local checked_read_seq.
        self._read_token = None

    def current_read_token(self):
        return self._read_token

    # ------------------------------------------------------------------ #
    # Read path: a read invalidates any prior check immediately and       #
    # publishes the cleared snapshot before any delayed request/check.    #
    # ------------------------------------------------------------------ #
    def on_read(self, state: dict) -> str:
        self.read_seq += 1
        self.read_time_ms = _now_ms()
        valid = False
        resolution = None
        cart_ref = None
        if isinstance(state, dict):
            resolution = state.get("resolution")
            # Only a live Shopify cart is accepted; sqlite-snapshot and any
            # malformed/unknown resolution fail closed.
            if resolution == "remote-cart" and isinstance(state.get("lines"), dict):
                lines = _alias_map(state["lines"], self.aliases)
                prov = state.get("provenance") or {}
                cart_ref = prov.get("cart_ref") if isinstance(prov, dict) else None
                if lines is not None and cart_ref is not None:
                    # Verify private cart identity: first read sets it; any
                    # later read with a missing or different ref fails closed.
                    if self._cart_ref is None:
                        self._cart_ref = cart_ref
                    if cart_ref == self._cart_ref:
                        self.observed = lines
                        self.resolution = resolution
                        valid = True
        if not valid:
            self.observed = None
            self.resolution = resolution
            self._cart_mismatch = True
            # Fail/stop this attempt on a wrong/missing cart identity or an
            # unknown/invalid read; never silently reset to a fresh cart.
            raise DemoStop("cart_identity_lost")
        # A new read invalidates any prior check coupling until its own check.
        self.pending = True
        self.verdict = None
        self.target = None
        self.missing = {}
        self._read_token = "%d:%s" % (self.read_seq, os.urandom(8).hex())
        self.write()  # publish the cleared snapshot immediately
        return self._read_token

    # ------------------------------------------------------------------ #
    # Event path: only an independent_check can (re)bind a verdict; any   #
    # other event is a mutation/scenario invalidation and clears it.      #
    # ------------------------------------------------------------------ #
    def on_event(self, row: dict, joint):
        self.event_seq += 1
        ev = row.get("event")
        if ev == "independent_check":
            self._on_check(row, joint)
        else:
            if ev == "model_add_selected":
                self.action = "add_to_cart"
            elif ev == "controlled_drop_NO_REQUEST_SENT":
                self.drop = True
                self.drop_no_request = True
            elif ev == "mutation_response_ok":
                self.recovery = bool(row.get("recovery"))
            # Only actual mutations invalidate the current verdict; progress
            # markers (intent/model/sequence_complete) do not change the cart.
            if ev in ("mutation_response_ok", "controlled_drop_NO_REQUEST_SENT"):
                self.pending = True
                self.verdict = None
                self.target = None
                self.missing = {}
            self._refresh(joint)

    def _on_check(self, row: dict, joint):
        # A check is bound ONLY to the exact read it examined. The read token
        # is captured when run_check acquired its read and returned with the
        # check callback; a missing or stale token must not stamp itself as the
        # newest read (equal quantities cannot prove the check is current).
        token = row.get("read_token")
        if self._cart_mismatch or token is None or token != self._read_token:
            self.pending = True
            self.verdict = None
            self.target = None
            self.missing = {}
            self._refresh(joint)
            return
        if joint is not None and joint.results:
            self.last_check = dict(joint.results[-1])
        self.checked_read_seq = self.read_seq
        self.check_time_ms = _now_ms()
        self.pending = False
        self._refresh_check(joint)
        self._refresh(joint)

    def _refresh_check(self, joint):
        if self.last_check is not None:
            v = self.last_check.get("verdict")
            verdict = v if v in VERDICTS else "UNKNOWN"
            exp = self.last_check.get("expected_lines")
            self.target = _alias_map(exp, self.aliases) if isinstance(exp, dict) else None
            self.missing = _missing(self.target, self.observed) \
                if (self.target is not None and self.observed is not None) else {}
            # Never publish COMPLETE on a mismatch (missing units, null or
            # unequal maps, wrong cart). A replayed/delayed check degrades.
            if verdict == "COMPLETE":
                if (self.target is None or self.observed is None
                        or self.target != self.observed or self.missing):
                    verdict = "PARTIAL"
            elif verdict == "PARTIAL":
                if self.target is None or self.observed is None or self.missing == {}:
                    verdict = "UNKNOWN"
            self.verdict = verdict
        else:
            self.verdict = None
            self.target = None
            self.missing = {}

    def _refresh(self, joint):
        if joint is not None:
            self.phase = joint.phase if joint.phase in PHASES else "failed"
        if self.pending:
            # A read without its own check cannot show a verdict.
            self.verdict = None
            self.target = None
            self.missing = {}
        calls = getattr(self.model, "calls_made", 0) if self.model else 0
        req = getattr(self.backend, "request_count", 0)
        self.budget = {"model_calls": calls, "shopify_requests": req}
        self.write()

    def write(self):
        self.freshness += 1
        snap = {
            "schema": SNAPSHOT_SCHEMA,
            "status": self.status,
            "phase": self.phase,
            "request": dict(REQUEST),
            "target": self.target,
            "observed": self.observed,
            "verdict": self.verdict,
            "missing": self.missing,
            "action": self.action,
            "controlled_drop": self.drop,
            "controlled_drop_no_request": self.drop_no_request,
            "recovery": self.recovery,
            "read_seq": self.read_seq,
            "checked_read_seq": self.checked_read_seq,
            "event_seq": self.event_seq,
            "read_at_ms": self.read_time_ms,
            "check_at_ms": self.check_time_ms,
            "model_calls": self.budget.get("model_calls", 0),
            "shopify_requests": self.budget.get("shopify_requests", 0),
            "saved_check": self.saved_check,
            "freshness": self.freshness,
        }
        _write_atomic(self.rundir, snap)

    def finish(self, complete: bool, reason: str):
        self.status = "complete" if complete else "failed"
        if complete and self.last_check is not None:
            # Historical check: distinct from any current success, never reused.
            self.saved_check = {
                "verdict": self.last_check.get("verdict"),
                "at_ms": self.check_time_ms,
                "target": self.target,
                "observed": self.observed,
            }
        if not complete:
            self.verdict = None
            self.pending = True
            self.target = None
            self.missing = {}
        self.phase = "complete" if complete else "failed"
        self.write()


def _write_atomic(rundir: Path, payload: dict):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path = rundir / SNAPSHOT_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.01)
    raise OSError("snapshot_replace_failed")  # fail closed, never non-atomic


def _load_profile(rundir: Path) -> dict:
    return json.loads((rundir / PROFILE_NAME).read_text(encoding="utf-8"))


def _load_mapping(rundir: Path) -> dict:
    return json.loads((rundir / MAPPING_NAME).read_text(encoding="utf-8"))


def _freeze_aliases(mapping: dict) -> dict:
    variants = mapping.get("variants", {})
    skus = sorted(variants)
    if len(skus) < 2:
        raise RuntimeError("catalog_shortfall")
    return {skus[0]: "A", skus[1]: "B"}


def _hold_gate(joint, hold_seconds: float, reserve_seconds: float):
    def gate(phase: str):
        if phase != "controlled_drop":
            return
        remaining = joint.deadline - joint.clock()
        if remaining <= reserve_seconds:
            return
        hold = min(hold_seconds, remaining - reserve_seconds)
        if hold > 0:
            time.sleep(hold)
    return gate


def _validate_hold(value: float) -> float:
    """Finite, non-negative, upper-bounded hold; default 6s."""
    if isinstance(value, bool):
        return 0.0
    if not isinstance(value, (int, float)) or value != value:  # reject NaN
        return 0.0
    if value in (float("inf"), float("-inf")):
        return 0.0
    if value < 0:
        return 0.0
    return min(float(value), HOLD_MAX)


def _build_offline_model(profile, a_sku):
    import httpx
    from agents.meta_client import MetaModelClient
    from agents.model_client import BudgetedModelClient, ClientConfig
    intent = {"items": [{"sku": a_sku, "quantity": 1}],
              "budget_max_cents": None, "substitution_allowed": False}
    add = {"action": "add_to_cart", "query": None, "product_id": None,
           "sku": a_sku, "quantity": 1}
    seq = [intent, add, intent, add, add]

    def handler(req):
        item = seq.pop(0)
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop",
                         "message": {"content": json.dumps(item)}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 20,
                      "total_tokens": 60}})

    cfg = ClientConfig(enabled=True, provider="meta",
        base_url="https://api.meta.ai/v1", model_name="muse-spark-1.3",
        max_calls=6, token_budget=100000, max_output_tokens_per_call=8192,
        live_enabled=True, reasoning_effort_by_kind={k: "low"
            for k in ("intent", "action", "review")},
        price_table={"as_of": profile["pricing_as_of"], "per_1m_tokens": {
            profile["model"]: {"input": 1.25, "cached_input": 0.15,
                               "output": 4.25}}})
    inner = MetaModelClient(cfg, allow_live=True,
                            environ={"MODEL_API_KEY": "offline-key"},
                            transport=httpx.MockTransport(handler))
    return BudgetedModelClient(inner, cfg, allow_live=True)


def run_worker_dir(rundir: Path, *, offline: bool = False) -> dict:
    """Runs the reviewed sequence against an injected (offline) or authorized
    (live) backend. Every construction/discovery/install step is inside the
    same try/finally so a construction error still closes any opened client."""
    from harness.joint_demo_candidate import JointDemo
    from harness.joint_demo_worker import construct_model, discover
    from harness.shopify_live_backend import ShopifyLiveBackend, resolve_token

    backend = model = None
    token = key = None
    run = None
    writer = None
    try:
        profile = _load_profile(rundir)
        if offline:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
            from shopify_live_fake import FakeLiveShopify
            mapping = _load_mapping(rundir)
            aliases = _freeze_aliases(mapping)
            a_sku = [k for k, v in aliases.items() if v == "A"][0]
            fake = FakeLiveShopify(products=[
                {"id": "gid://shopify/Product/1", "title": "Synthetic",
                 "variants": [{"id": gid, "sku": sku}
                              for sku, gid in mapping["variants"].items()]}])
            backend = ShopifyLiveBackend(
                profile["shop"], str(rundir / MAPPING_NAME),
                token="offline-synthetic-token", api_version=profile["api_version"],
                transport=fake.transport)
            model = _build_offline_model(profile, a_sku)
            token = "offline-synthetic-token"
            key = "offline-key"
        else:
            # Live: empty mapping + explicit opt-in created ONLY inside the
            # checked worker setup after controls validation (in main()).
            token = resolve_token()
            if not token:
                raise RuntimeError("shopify_token_unavailable")
            os.environ["CARTPROOF_SHOPIFY_LIVE"] = "1"
            empty = rundir / EMPTY_MAPPING_NAME
            empty.write_text('{"variants": {}}', encoding="utf-8")
            backend = ShopifyLiveBackend(
                profile["shop"], str(empty), token=token,
                api_version=profile["api_version"])
            model, key = construct_model(profile, None)
            # Private frozen mapping from the real catalog (never user-supplied).
            mapping = discover(backend)
            aliases = _freeze_aliases(mapping)

        backend.install_mapping(mapping)
        writer = SnapshotWriter(rundir, aliases, backend, model)
        pbackend = PublishingBackend(backend, writer.on_read)
        writer.write()  # initial not_started snapshot

        # Offline test seam: block forever to exercise supervisor timeout/cancel.
        if offline and os.environ.get(BLOCK_ENV) == "1":
            writer.status = "running"
            writer.write()
            while True:
                time.sleep(0.5)

        joint_holder = {}

        def emit(line: str):
            # The check callback already carries the exact read token captured
            # for the read consumed by that run_check call (injected by the
            # PublishingBackend and forwarded by JointDemo.check). The emitter
            # must never query the latest token to manufacture one.
            row = json.loads(line)
            writer.on_event(row, joint_holder.get("joint"))

        joint = JointDemo(
            pbackend, model, mapping, rundir / "evidence",
            secrets=[token, key], emit=emit,
            timeout_seconds=profile["timeout_seconds"])
        joint_holder["joint"] = joint
        hold_s = _validate_hold(float(os.environ.get(HOLD_ENV, "6.0")))
        joint.gate = _hold_gate(joint, hold_s, RESERVE_SECONDS)

        writer.status = "running"
        summary = joint.execute()
        writer.finish(bool(summary.get("complete")), summary.get("reason"))
        run = {"complete": bool(summary.get("complete")),
               "offline_exit": 0 if summary.get("complete") else 1,
               "phase": summary.get("phase"),
               "reason": summary.get("reason"),
               "model_calls": summary.get("model_calls"),
               "shopify_requests": summary.get("shopify_requests"),
               "run_context": "offline_injected" if offline else "real_model_and_shopify"}
    finally:
        # Close the backend client and the model's inner client on every exit,
        # including construction/discovery/install_mapping errors.
        for closer in (getattr(backend, "client", None),
                       getattr(getattr(model, "inner", None), "_client", None)):
            if closer is not None:
                try:
                    closer.close()
                except Exception:
                    pass
    if run is not None:
        tmp = rundir / (RUN_NAME + ".tmp")
        tmp.write_text(json.dumps(run, sort_keys=True), encoding="utf-8")
        os.replace(tmp, rundir / RUN_NAME)
        return run
    raise RuntimeError("worker_construction_failed")


def _validate_controls(rundir: Path, offline: bool, live_release: bool) -> dict:
    """Route-specific ownership/controls/profile validation before any client.

    A marker string alone is never enough: the worker independently verifies
    the reserved ownership record, the immutable profile raw hash, the source
    manifest binding, the current tests digest, and the route/mode/release/
    validation identity — and for live, the exact linked receipt under
    require_ui plus its browser eligibility — against installed files.
    """
    from harness.joint_demo_control import (regular_chain, validate_profile,
                                            verify_source, verify_offline_receipt,
                                            digest)
    from harness.joint_demo_candidate import need
    import hashlib

    if offline and live_release:
        need(False, "mixed_offline_live_flags")
    for name in ("ownership.json", PROFILE_NAME, RUN_CONTEXT_NAME):
        regular_chain(rundir / name)

    # Parse the reserved ownership record: reserved:true and attempt_id must
    # match the safe directory name; full regular path chain already checked.
    owner = json.loads((rundir / "ownership.json").read_text(encoding="utf-8"))
    need(isinstance(owner, dict) and owner.get("reserved") is True,
         "ownership_not_reserved")
    need(owner.get("attempt_id") == rundir.name, "ownership_attempt_id_mismatch")
    need(len(rundir.name) == 16 and all(c in "0123456789abcdef" for c in rundir.name),
         "ownership_attempt_name")
    need(isinstance(owner.get("pid"), int), "ownership_pid_missing")

    ctx = json.loads((rundir / RUN_CONTEXT_NAME).read_text(encoding="utf-8"))
    need(isinstance(ctx, dict) and ctx.get("schema") == 1, "run_context_schema")
    need(ctx.get("route") == ROUTE, "run_context_route")
    want_mode = "offline" if offline else "live-release"
    need(ctx.get("mode") == want_mode, "run_context_mode_mismatch")
    need(ctx.get("release") is (True if live_release else False),
         "run_context_release_mismatch")

    profile = validate_profile(
        json.loads((rundir / PROFILE_NAME).read_text(encoding="utf-8")))
    need(ctx.get("profile_sha256") == digest(rundir / PROFILE_NAME),
         "run_context_profile_mismatch")

    # Recompute source closure + current tests digest from installed files;
    # reject a forged or stale context whose recorded digests no longer match.
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "harness" / "joint_demo_source_manifest.json").read_text(encoding="utf-8"))
    source = verify_source(root, manifest)
    need(ctx.get("source_digest") == source, "run_context_source_mismatch")
    tests = hashlib.sha256((
        digest(root / "tests/test_joint_demo.py")
        + digest(root / "tests/test_joint_demo_control.py")
        + digest(root / "tests/test_joint_demo_parent.py")
        + digest(root / "tests/test_demo_ui.py")).encode()).hexdigest()
    need(ctx.get("tests_digest") == tests, "run_context_tests_mismatch")

    if live_release:
        vid = ctx.get("validation_id")
        need(vid and len(vid) == 16 and all(c in "0123456789abcdef" for c in vid),
             "run_context_validation_mismatch")
        receipt_path = (root / "var" / "joint-demo-control"
                        / ("offline-receipt-" + source[:12] + "-" + vid + ".json"))
        regular_chain(receipt_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        verify_offline_receipt(receipt, source, tests,
                               digest(root / "harness" / "joint_demo_profile.json"),
                               require_ui=True)
        need(receipt.get("browser_proof") is True, "run_context_browser_proof_missing")
    return profile


def _offline_egress_canary() -> dict:
    """Runs ONLY under offline_boundary: proves socket/httpx egress, token
    fallback and dotenv fallback are all denied in this child process."""
    import socket
    blocked = {"socket": 0, "httpx": 0, "token_fallback": False,
               "dotenv_fallback": False}
    try:
        socket.getaddrinfo("api.meta.ai", 443)
    except RuntimeError:
        blocked["socket"] += 1
    try:
        import httpx
        with httpx.Client(trust_env=False) as c:
            c.get("https://api.meta.ai/v1")
    except RuntimeError:
        blocked["httpx"] += 1
    from harness.shopify_live_backend import resolve_token
    if resolve_token() == "":
        blocked["token_fallback"] = True
    try:
        from agents import env_loader
        env_loader.load_root_dotenv(Path("/unused"))
    except RuntimeError:
        blocked["dotenv_fallback"] = True
    return blocked


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-dir", required=True)
    args = parser.parse_args(argv)
    rundir = Path(args.worker_dir)
    if os.environ.get(PARENT_MARKER) != PARENT_VALUE:
        print(json.dumps({"status": "STOP", "reason": "parent_launch_required"}))
        return 2
    offline = os.environ.get(OFFLINE_ENV) == "1"
    live_release = os.environ.get(LIVE_RELEASE_ENV) == "1"
    if not offline and not live_release:
        print(json.dumps({"status": "STOP", "reason": "release_required"}))
        return 2
    if offline and live_release:
        print(json.dumps({"status": "STOP", "reason": "mixed_offline_live_flags"}))
        return 2
    try:
        _validate_controls(rundir, offline, live_release)
        if offline:
            # The fake child enters the credential-free offline boundary itself
            # before any construction, and proves the denial canaries.
            from harness.joint_demo_offline_guard import offline_boundary
            with offline_boundary():
                canary = _offline_egress_canary()
                run = run_worker_dir(rundir, offline=True)
                run["network_denied"] = True
                run["egress_canary_blocked"] = canary
                tmp = rundir / (RUN_NAME + ".tmp")
                tmp.write_text(json.dumps(run, sort_keys=True), encoding="utf-8")
                os.replace(tmp, rundir / RUN_NAME)
                return 0 if run.get("complete") else 1
        run = run_worker_dir(rundir, offline=False)
        return 0 if run.get("complete") else 1
    except BaseException as exc:
        from harness.joint_demo_candidate import DemoStop
        reason = str(exc) if isinstance(exc, DemoStop) else "worker_gate_or_outcome_unverified"
        snap = {"schema": SNAPSHOT_SCHEMA, "status": "failed", "phase": "failed",
                "request": dict(REQUEST), "target": None, "observed": None,
                "verdict": None, "missing": {}, "action": None,
                "controlled_drop": False, "controlled_drop_no_request": False,
                "recovery": False, "read_seq": 0, "checked_read_seq": 0,
                "event_seq": 0, "read_at_ms": 0, "check_at_ms": 0,
                "model_calls": 0, "shopify_requests": 0, "saved_check": None,
                "freshness": 1}
        text = json.dumps(snap, sort_keys=True, separators=(",", ":"))
        try:
            (rundir / SNAPSHOT_NAME).write_text(text, encoding="utf-8")
        except OSError:
            pass
        try:
            (rundir / RUN_NAME).write_text(json.dumps(
                {"complete": False, "offline_exit": 1, "phase": "failed",
                 "reason": reason}), encoding="utf-8")
        except OSError:
            pass
        print(json.dumps({"status": "STOP", "reason": reason}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

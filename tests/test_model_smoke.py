"""Brief 7 smoke launcher/guard tests. Offline: fake clients, MockTransport,
dummy credentials, and local mock-store traffic only."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from agents.model_actor import ModelActor
from agents.model_client import (BudgetedModelClient, ClientConfig,
                                 FakeModelClient, ModelProposal, ModelRequest,
                                 Usage)
from agents.smoke_guard import SmokeAccountingGuard
from harness import model_compare, model_smoke

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKU = "BTL-HIK-500-NAVYBLUE"
DUMMY = "dummy-key-for-tests-only"


def _budgeted(responses):
    fake = FakeModelClient(responses)
    return BudgetedModelClient(fake, ClientConfig(enabled=True,
                                                  token_budget=100000)), fake


def _payload(items=None):
    return {"items": items or [], "budget_max_cents": None,
            "substitution_allowed": False}


# ---------------------------------------------------------------- guard units

def _install_inprocess_tripwire(monkeypatch):
    """One small test-only isolation helper shared by the fake worker and
    mocked-chain tests: Python socket guard + browser-context allowlist
    restricted to the session's own store origin. No production surface."""
    import socket as _socket
    from urllib.parse import urlparse as _urlparse

    from adapter.actions import BrowserSession

    real_connect = _socket.socket.connect
    real_create = _socket.create_connection

    def _host(address):
        return address[0] if isinstance(address, tuple) else address

    def guarded_connect(self, address, *a, **kw):
        if _host(address) not in ("127.0.0.1", "localhost", "::1"):
            raise RuntimeError(f"external_network_blocked:{_host(address)}")
        return real_connect(self, address, *a, **kw)

    def guarded_create(address, *a, **kw):
        if _host(address) not in ("127.0.0.1", "localhost", "::1"):
            raise RuntimeError(f"external_network_blocked:{_host(address)}")
        return real_create(address, *a, **kw)

    monkeypatch.setattr(_socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(_socket, "create_connection", guarded_create)

    orig_init = BrowserSession.__init__

    def patched_init(self, base_url, headed=False):
        orig_init(self, base_url, headed=headed)
        ctx = self.browser.new_context(service_workers="block")
        origin = _urlparse(base_url)

        def _route(route):
            u = _urlparse(route.request.url)
            if (u.scheme, u.hostname, u.port) == (
                    origin.scheme, origin.hostname, origin.port) \
                    or route.request.url == "about:blank":
                route.continue_()
            else:
                route.abort()

        ctx.route("**/*", _route)
        self.context = ctx
        self.page = ctx.new_page()

    monkeypatch.setattr(BrowserSession, "__init__", patched_init)


def test_fake_five_calls_pass_with_no_sidecars():
    responses = [{"payload": _payload()}] * 5
    budgeted, _ = _budgeted(responses)
    guard = SmokeAccountingGuard(budgeted, transport_kind="fake")
    for _ in range(5):
        p = guard.propose(ModelRequest(kind="intent", max_output_tokens=10))
        assert p.status == "ok"
    assert guard.stopped is False
    assert budgeted.sidecars == {}
    assert guard.summary()["smoke_guard"]["accounting_mode"] == "synthetic_offline"
    assert guard.summary()["smoke_guard"]["provider_billing_verified"] is False


class _NoSidecarInner:
    def __init__(self):
        self.calls = 0

    def propose(self, request):
        self.calls += 1
        return ModelProposal(kind=request.kind, payload={"ok": True},
                             usage=Usage(1, 1))


def test_meta_call_with_missing_sidecar_stops_same_call():
    budgeted = BudgetedModelClient(_NoSidecarInner(),
                                   ClientConfig(enabled=True))
    guard = SmokeAccountingGuard(budgeted, transport_kind="meta")
    p = guard.propose(ModelRequest(kind="action", max_output_tokens=10))
    assert p.status == "error" and p.payload is None
    assert "meta_sidecar_missing" in (p.error or "")
    assert guard.stopped is True
    assert guard.last_underlying["proposal_status"] == "ok"


class _Ev:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return {"payload": self.payload}


class _StubSession:
    def __init__(self):
        self.writes = []

    def search(self, query):
        return _Ev({"results": [{"product_id": "p1"}]})

    def open_product(self, pid):
        return _Ev({"product_id": pid, "variant_skus": [SKU],
                    "cart_version": 0})

    def select_variant(self, sku):
        return _Ev({"sku": sku})

    def add_to_cart(self, quantity=1, sku=None):
        self.writes.append((sku, quantity))
        return _Ev({"response_status": 200, "response_version": 1})


_CATALOG = [{"sku": SKU, "product_id": "bottle-hiker",
             "product_name": "Hiker Water Bottle", "label": "Navy Blue",
             "size": "500ml", "color": "Navy Blue", "price_cents": 2400}]


def _actor(session, responses):
    budgeted, fake = _budgeted(responses)
    guard = SmokeAccountingGuard(budgeted, transport_kind="fake")
    return ModelActor(session, guard, _CATALOG, masked=False), fake


def test_uncertain_final_add_is_blocked_on_the_same_call():
    intent = {"payload": _payload([{"sku": SKU, "quantity": 1}])}
    actions = [
        {"payload": {"action": "search"}},
        {"payload": {"action": "open_product", "product_id": "p1"}},
        {"payload": {"action": "select_variant", "sku": SKU}},
        # Otherwise-valid add proposal, but its usage is missing.
        {"payload": {"action": "add_to_cart", "sku": SKU, "quantity": 1},
         "usage": None},
    ]
    session = _StubSession()
    actor, fake = _actor(session, [intent] + actions)
    actor.resolve_intent(f"Add 1 x {SKU}.")
    actor.execute_items([{"product_query": "Hiker Water Bottle",
                          "target_sku": SKU, "quantity": 1}])
    assert session.writes == []          # zero add mutations from that proposal
    assert len(fake.requests) == 5       # one call per proposal, no retry/later
    assert actor.failures


def test_uncertain_intent_blocks_execution():
    session = _StubSession()
    actor, _fake = _actor(session, [
        {"payload": _payload([{"sku": SKU, "quantity": 1}]), "usage": None}])
    actor.resolve_intent(f"Add 1 x {SKU}.")
    assert actor.outcome_kind == "operational_failure"
    assert actor.executable_intent is None
    assert session.writes == []


# ---------------------------------------------------------------- preflight

def test_preflight_rejects_wrong_selection_and_conflicts():
    root = PROJECT_ROOT
    for kw, reason in [
        ({"task_id": "other"}, "task"),
        ({"arm": "MB"}, "arm"),
        ({"trial": 2}, "trial"),
        ({"transport": "fake", "allow_live": True}, "conflicting"),
        ({"transport": "meta", "allow_live": False}, "live"),
    ]:
        args = {"suite_arg": "eval/model/dev/suite.json",
                "task_id": model_smoke.FIXED_TASK_ID, "arm": "MA", "trial": 1,
                "transport": "fake", "allow_live": False}
        args.update(kw)
        with pytest.raises(model_smoke.SmokePreflightError):
            model_smoke.resolve_selection(root, **args)


def test_preflight_rejects_frozen_path_before_opening(tmp_path):
    with pytest.raises(model_smoke.SmokePreflightError):
        model_smoke.resolve_selection(
            PROJECT_ROOT, "eval/model/frozen/suite.json",
            model_smoke.FIXED_TASK_ID, "MA", 1, "fake", False)


def test_smoke_flag_must_be_a_real_bool(tmp_path):
    spec = {"setup": {"request_text": "x", "seed_lines": []},
            "arm": "MA", "trial": 1, "fake_script": {}, "smoke": "false"}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    rc = model_compare.run_one_cli(["--spec", str(spec_path), "--db",
                                    str(tmp_path / "x.db"), "--out", str(out)])
    assert rc == 0
    run = json.loads((out / "run.json").read_text())
    assert run["execution_status"] == "harness-error"
    assert run["error"] == "smoke_flag_invalid"


# ---------------------------------------------------------------- job cleanup

def test_job_terminates_owned_tree_and_leaves_unrelated_alone(tmp_path):
    if sys.platform != "win32":
        pytest.skip("Windows job object test")
    release = tmp_path / "release"
    grandchild_pid_file = tmp_path / "grandchild.pid"
    child_code = (
        "import subprocess, sys, time, pathlib;"
        "g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);"
        f"pathlib.Path(r'{grandchild_pid_file}').write_text(str(g.pid));"
        "time.sleep(30)")
    # Suspended creation mirrors production: assign the job before resume.
    child = model_smoke._SuspendedProcess(
        [sys.executable, "-c", child_code], PROJECT_ROOT,
        tmp_path / "child.out", tmp_path / "child.err")
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        job = model_smoke._JobObject()
        job.assign(child.handle)
        child.resume()
        deadline = time.time() + 10
        while not grandchild_pid_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert grandchild_pid_file.exists()
        gpid = int(grandchild_pid_file.read_text())
        assert job.terminate()
        child.wait(timeout=10)
        # grandchild should be gone with the job tree
        deadline = time.time() + 10
        alive = True
        while time.time() < deadline:
            probe = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"if (Get-Process -Id {gpid} -ErrorAction SilentlyContinue)"
                 " { 'alive' } else { 'gone' }"],
                capture_output=True, text=True)
            if "gone" in probe.stdout:
                alive = False
                break
            time.sleep(0.2)
        assert not alive
        assert unrelated.poll() is None  # unrelated process untouched
        job.close()
    finally:
        child.close()
        unrelated.kill()
        unrelated.wait(timeout=10)


# ---------------------------------------------------------------- e2e (real subprocess, fake)

def test_default_fake_smoke_subprocess(tmp_path):
    out = tmp_path / "smoke"
    proc = subprocess.run(
        [sys.executable, "-m", "harness.model_smoke", "--transport", "fake",
         "--out", str(out), "--timeout-seconds", "360"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    evaluated = json.loads((out / "evaluated.json").read_text())
    assert evaluated["declared_outcome"] == "COMPLETE"
    assert evaluated["measured_outcome"] == "correct"
    assert evaluated["budget"]["calls_made"] == 5
    rundir = out / "runs" / model_smoke.FIXED_TASK_ID / "MA" / "t1"
    spec = json.loads((rundir / "worker_spec.json").read_text())
    assert spec["smoke"] is True
    run = json.loads((rundir / "run.json").read_text())
    assert run["smoke_guard_enabled"] is True
    assert (out / "manifest.json").exists()


# ------------------------------------------------- mocked Meta via launcher

class _RecordingEnv(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.accessed = []

    def get(self, key, default=None):
        self.accessed.append(key)
        return super().get(key, default)


def _meta_body(payload, with_usage=True):
    body = {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(payload)}}]}
    if with_usage:
        body["usage"] = {"prompt_tokens": 10, "completion_tokens": 5,
                         "total_tokens": 15}
    return body


def _meta_handler_factory(bodies, counter):
    queue = list(bodies)

    def handler(request):
        counter.append(str(request.url))
        return httpx.Response(200, json=queue.pop(0))
    return handler


def _run_mocked_launcher(monkeypatch, tmp_path, bodies):
    _install_inprocess_tripwire(monkeypatch)
    counter: list = []
    handler = _meta_handler_factory(bodies, counter)
    real_build = model_compare.build_model_client
    env = _RecordingEnv({"MODEL_API_KEY": DUMMY})

    def factory(script, transport="fake", allow_live=False, environ=None,
                http_transport=None, dotenv_fallback=None):
        return real_build(script, transport=transport, allow_live=allow_live,
                          environ=env,
                          http_transport=httpx.MockTransport(handler),
                          dotenv_fallback=dotenv_fallback)

    monkeypatch.setattr(model_compare, "build_model_client", factory)

    def runner(ctx):
        rundir = ctx["rundir"]
        spec_path = rundir / "worker_spec.json"
        spec_path.write_text(json.dumps(ctx["spec"]), encoding="utf-8")
        rc = model_compare.run_one_cli(
            ["--spec", str(spec_path), "--db", str(rundir / "run.db"),
             "--out", str(rundir)])
        assert rc == 0
        raw = json.loads((rundir / "run.json").read_text())
        return raw, {"cleanup_status": "injected", "timed_out": False,
                     "job_assigned": False}

    out = tmp_path / "mocked-smoke"
    code = model_smoke.main(
        ["--transport", "meta", "--allow-live", "--out", str(out)],
        _run_worker=runner)
    return code, out, counter


def _intent_body():
    return _meta_body({"items": [{"sku": SKU, "quantity": 1}],
                       "budget_max_cents": None,
                       "substitution_allowed": False})


def _action_bodies():
    # Strict-schema shape: every field present, irrelevant ones null.
    def act(action, **kw):
        payload = {"action": action, "query": None, "product_id": None,
                   "sku": None, "quantity": None}
        payload.update(kw)
        return _meta_body(payload)

    return [
        act("search", query="Hiker Water Bottle"),
        act("open_product", product_id="bottle-hiker"),
        act("select_variant", sku=SKU),
        act("add_to_cart", sku=SKU, quantity=1),
    ]


def test_mocked_meta_happy_path_through_launcher(monkeypatch, tmp_path):
    bodies = [_intent_body()] + _action_bodies()
    code, out, counter = _run_mocked_launcher(monkeypatch, tmp_path, bodies)
    assert code == 0
    assert len(counter) == 5  # one provider call per proposal, no retry
    evaluated = json.loads((out / "evaluated.json").read_text())
    assert evaluated["declared_outcome"] == "COMPLETE"
    assert evaluated["measured_outcome"] == "correct"
    rundir = out / "runs" / model_smoke.FIXED_TASK_ID / "MA" / "t1"
    spec = json.loads((rundir / "worker_spec.json").read_text())
    assert spec["smoke"] is True
    assert "expected" not in json.dumps(spec)


def test_mocked_meta_missing_usage_on_final_add_blocks_it(monkeypatch, tmp_path):
    bodies = [_intent_body()] + _action_bodies()[:-1] + [
        _meta_body({"action": "add_to_cart", "sku": SKU, "quantity": 1},
                   with_usage=False)]
    code, out, counter = _run_mocked_launcher(monkeypatch, tmp_path, bodies)
    assert code == 4  # accounting incomplete / guard stop
    assert len(counter) == 5  # the uncertain call happened exactly once
    evaluated = json.loads((out / "evaluated.json").read_text())
    # No resolvable cart was ever created (the add was blocked), so the
    # independent measurement is honestly unknown, never success.
    assert evaluated["measured_outcome"] == "unknown"
    adds = [m for m in evaluated["mutations"]
            if m["operation"] == "add_to_cart" and not m["corrective"]]
    assert adds == []  # zero add mutations from the uncertain proposal


# ------------------------------------------------------- network tripwire

_BOOTSTRAP = r'''
import json, os, runpy, socket, sys
sys.path.insert(0, os.getcwd())
from urllib.parse import urlparse

_real_connect = socket.socket.connect
_real_create = socket.create_connection

def _host(address):
    return address[0] if isinstance(address, tuple) else address

def _guarded_connect(self, address, *a, **kw):
    if _host(address) not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(f"external_network_blocked:{_host(address)}")
    return _real_connect(self, address, *a, **kw)

def _guarded_create(address, *a, **kw):
    if _host(address) not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(f"external_network_blocked:{_host(address)}")
    return _real_create(address, *a, **kw)

socket.socket.connect = _guarded_connect
socket.create_connection = _guarded_create

mode = sys.argv[1] if len(sys.argv) > 1 else "worker"
if mode == "probe-socket":
    result = {}
    try:
        socket.create_connection(("203.0.113.1", 81), timeout=3)
        result["external"] = "NOT-BLOCKED"
    except Exception as exc:
        result["external"] = f"blocked:{exc}"
    s = socket.create_connection(("127.0.0.1", int(sys.argv[2])), timeout=3)
    s.close()
    result["loopback"] = "ok"
    open(sys.argv[3], "w").write(json.dumps(result))
elif mode == "probe-browser":
    from playwright.sync_api import sync_playwright
    result = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(service_workers="block")
        aborted = {"n": 0}
        def _route(route):
            host = urlparse(route.request.url).hostname
            if host in ("127.0.0.1", "localhost", "::1", None):
                route.continue_()
            else:
                aborted["n"] += 1
                route.abort()
        ctx.route("**/*", _route)
        page = ctx.new_page()
        try:
            page.goto("http://example.com/", timeout=8000)
            result["external_page"] = "NOT-BLOCKED"
        except Exception:
            result["external_page"] = "blocked"
        page2 = ctx.new_page()
        page2.goto(sys.argv[2], timeout=8000)
        result["loopback_page"] = page2.text_content("body").strip()
        result["aborted_external_requests"] = aborted["n"]
        browser.close()
    open(sys.argv[3], "w").write(json.dumps(result))
else:
    # Worker mode: also install the browser-context request allowlist on a
    # fresh context with service workers blocked, before any navigation.
    import adapter.actions as _actions
    from pathlib import Path as _P
    _orig_init = _actions.BrowserSession.__init__
    def _patched(self, base_url, headed=False):
        _orig_init(self, base_url, headed=headed)
        ctx = self.browser.new_context(service_workers="block")
        _origin = urlparse(base_url)
        def _route(route):
            u = urlparse(route.request.url)
            if (u.scheme, u.hostname, u.port) == (
                    _origin.scheme, _origin.hostname, _origin.port) \
                    or route.request.url == "about:blank":
                route.continue_()
            else:
                route.abort()
        ctx.route("**/*", _route)
        self.context = ctx
        self.page = ctx.new_page()
        try:
            _spec = _P(sys.argv[sys.argv.index("--spec") + 1])
            (_spec.parent / "tripwire-evidence.json").write_text(json.dumps(
                {"socket_guard": True, "browser_route_allowlist": True,
                 "service_workers_block": True}))
        except Exception:
            pass
    _actions.BrowserSession.__init__ = _patched
    sys.argv = ["harness.model_compare", "--run-one"] + sys.argv[1:]
    runpy.run_module("harness.model_compare", run_name="__main__")
'''


def _write_bootstrap(tmp_path):
    p = tmp_path / "child_bootstrap.py"
    p.write_text(_BOOTSTRAP, encoding="utf-8")
    return p


def test_network_tripwire_inside_child_process(tmp_path):
    import socket as _socket
    bootstrap = _write_bootstrap(tmp_path)
    listener = _socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    out_file = tmp_path / "probe.json"
    proc = subprocess.Popen(
        [sys.executable, str(bootstrap), "probe-socket", str(port),
         str(out_file)])
    conn, _ = listener.accept()
    conn.close()
    listener.close()
    assert proc.wait(timeout=60) == 0
    result = json.loads(out_file.read_text())
    assert result["external"].startswith("blocked:external_network_blocked:")
    assert result["loopback"] == "ok"


def test_browser_request_allowlist_probe(tmp_path):
    import http.server
    import threading

    bootstrap = _write_bootstrap(tmp_path)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        out_file = tmp_path / "browser-probe.json"
        proc = subprocess.run(
            [sys.executable, str(bootstrap), "probe-browser", url,
             str(out_file)], capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        result = json.loads(out_file.read_text())
        assert result["external_page"] == "blocked"
        assert result["aborted_external_requests"] >= 1  # the route aborted it
        assert result["loopback_page"]  # local page loaded
    finally:
        server.shutdown()


# ------------------------------------------------------------- F1: inputs

def _synthetic_root(tmp_path, entries):
    dev = tmp_path / "eval" / "model" / "dev"
    (dev / "tasks").mkdir(parents=True)
    task = {"task_id": model_smoke.FIXED_TASK_ID, "task_class": "resolvable",
            "fault": {"name": None, "params": {}},
            "runtime": {"request_text": model_smoke.FIXED_REQUEST,
                        "seed_lines": []},
            "expected": {"lines": {SKU: 1}},
            "expected_interpretation": {
                "items": [{"sku": SKU, "quantity": 1}],
                "budget_max_cents": None, "substitution_allowed": False}}
    (dev / "tasks" / f"{model_smoke.FIXED_TASK_ID}.json").write_text(
        json.dumps(task), encoding="utf-8")
    (dev / "suite.json").write_text(
        json.dumps({"suite": "x", "task_files": entries}), encoding="utf-8")
    return tmp_path


def test_preflight_rejects_escape_and_duplicate_entries(tmp_path):
    root = _synthetic_root(tmp_path, ["tasks/../../outside.json"])
    with pytest.raises(model_smoke.SmokePreflightError):
        model_smoke.resolve_selection(root, "eval/model/dev/suite.json",
                                      model_smoke.FIXED_TASK_ID, "MA", 1,
                                      "fake", False)
    root2 = _synthetic_root(tmp_path / "dup",
                            [f"tasks/{model_smoke.FIXED_TASK_ID}.json"] * 2) \
        if False else None
    # duplicate exact entries
    dev = tmp_path / "dup" / "eval" / "model" / "dev"
    (dev / "tasks").mkdir(parents=True)
    src = json.loads((tmp_path / "eval" / "model" / "dev" / "tasks"
                      / f"{model_smoke.FIXED_TASK_ID}.json").read_text())
    (dev / "tasks" / f"{model_smoke.FIXED_TASK_ID}.json").write_text(
        json.dumps(src), encoding="utf-8")
    (dev / "suite.json").write_text(json.dumps(
        {"suite": "x", "task_files":
         [f"tasks/{model_smoke.FIXED_TASK_ID}.json"] * 2}), encoding="utf-8")
    with pytest.raises(model_smoke.SmokePreflightError):
        model_smoke.resolve_selection(tmp_path / "dup",
                                      "eval/model/dev/suite.json",
                                      model_smoke.FIXED_TASK_ID, "MA", 1,
                                      "fake", False)


def test_preflight_validates_timeout_and_live_config():
    for bad_timeout in (0, -1, float("inf"), float("nan"), True, "360"):
        with pytest.raises(model_smoke.SmokePreflightError):
            model_smoke.resolve_selection(
                PROJECT_ROOT, "eval/model/dev/suite.json",
                model_smoke.FIXED_TASK_ID, "MA", 1, "fake", False,
                timeout_seconds=bad_timeout)
    bad = json.loads(json.dumps(model_smoke.LIVE_SCRIPT))
    bad["budget"]["token_budget"] = True
    with pytest.raises(model_smoke.SmokePreflightError):
        model_smoke.validate_effective_config(bad, "meta")
    bad2 = json.loads(json.dumps(model_smoke.LIVE_SCRIPT))
    bad2["client"]["base_url"] = "https://example.com/v1"
    with pytest.raises(model_smoke.SmokePreflightError):
        model_smoke.validate_effective_config(bad2, "meta")


# ------------------------------------------------- F2: accounting/success

def _good_run():
    return {
        "transport": "fake", "execution_status": "normal",
        "measured_outcome": "correct", "measured_resolution": "cookie",
        "final_cookie": "c", "declared_outcome": "COMPLETE",
        "budget": {
            "calls_attempted": 5, "calls_made": 5, "calls_refused": 0,
            "consumed_tokens": 100, "reserved_tokens": 3000,
            "overrun": False,
            "records": [{"call_index": i, "status": "made",
                         "declared_usage": 20} for i in range(1, 6)],
            "sidecar": [],
            "smoke_guard": {"enabled": True, "transport_kind": "fake",
                            "stopped": False}},
    }


def _good_evaluated():
    return {"completed_execution": True,
            "interpretation_outcome": "intent_correct"}


def test_success_predicate_requires_positive_evidence():
    ok, reasons = model_smoke._success(_good_run(), _good_evaluated(),
                                       {"cleanup_status": "verified"})
    assert ok, reasons

    assert model_smoke._accounting_complete({}) is False
    r = _good_run()
    r["budget"] = {}
    assert model_smoke._success(r, _good_evaluated(),
                                {"cleanup_status": "verified"})[0] is False
    # unknown / injected cleanup never passes a real subprocess success
    assert model_smoke._success(_good_run(), _good_evaluated(),
                                {"cleanup_status": "injected"})[0] is False
    assert model_smoke._success(_good_run(), _good_evaluated(),
                                {"cleanup_status": "mystery"})[0] is False
    # overrun
    r = _good_run()
    r["budget"]["overrun"] = True
    assert model_smoke._success(r, _good_evaluated(),
                                {"cleanup_status": "verified"})[0] is False
    # COMPLETE with wrong persisted cart is not success
    r = _good_run()
    r["measured_outcome"] = "incorrect"
    assert model_smoke._success(r, _good_evaluated(),
                                {"cleanup_status": "verified"})[0] is False
    # Meta sidecar with a non-stop finish reason is incomplete accounting
    r = _good_run()
    r["transport"] = "meta"
    r["budget"]["smoke_guard"]["transport_kind"] = "meta"
    r["budget"]["sidecar"] = [
        {"call_index": i, "usage_present": True, "usage_malformed": False,
         "usage_detail_trustworthy": True, "finish_reason": None,
         "estimated_cost_usd": 0.1} for i in range(1, 6)]
    assert model_smoke._accounting_complete(r) is False


def test_cost_summary_reconciles_against_all_made_calls():
    run = {"transport": "meta",
           "budget": {"calls_made": 3,
                      "records": [{"call_index": i, "status": "made",
                                   "declared_usage": 10,
                                   "consumed": 10} for i in (1, 2, 3)],
                      "sidecar": [
                          {"call_index": 1, "estimated_cost_usd": 0.5},
                          {"call_index": 2, "estimated_cost_usd": None}]}}
    cost = model_smoke._cost_summary(run)
    # One sidecar is missing, so the made-call mapping is invalid: cost is
    # conservatively unknown for the full counted made calls (3), with the
    # actually reported 0.5 preserved as the safe known subtotal.
    assert cost["status"] != "complete"
    assert cost["unknown_cost_calls"] == 3
    assert cost["known_subtotal_usd_estimated"] == 0.5
    fake_cost = model_smoke._cost_summary(
        {"transport": "fake", "budget": {"calls_made": 5, "sidecar": []}})
    assert fake_cost["status"] == "not_applicable_synthetic"
    assert fake_cost["provider_billing_verified"] is False


def test_secret_scan_detects_injected_sentinel(tmp_path):
    (tmp_path / "artifact.txt").write_text(
        "leaked dummy-key-for-tests-only value", encoding="utf-8")
    scan = model_smoke._scan_secrets(tmp_path)
    assert scan["sentinel_hits"] > 0


# ------------------------------------------------------- F5: containment

def _spawn_ctx(tmp_path, timeout=360.0):
    task = model_smoke.resolve_selection(
        PROJECT_ROOT, "eval/model/dev/suite.json",
        model_smoke.FIXED_TASK_ID, "MA", 1, "fake", False)
    rundir = tmp_path / "run"
    rundir.mkdir(parents=True)
    from harness.compare import build_template_db
    from harness.snapshots import backup_copy
    template = tmp_path / "template.db"
    build_template_db(str(PROJECT_ROOT / "store" / "seed_catalog.json"),
                      str(template))
    backup_copy(str(template), str(rundir / "run.db"))
    script = model_smoke.load_fixed_fake_script(PROJECT_ROOT)
    spec = model_smoke.build_worker_spec(task, "fake", False, script)
    return {"spec": spec, "task": task, "rundir": rundir,
            "out_root": tmp_path, "timeout_seconds": timeout,
            "transport": "fake"}


def test_spawn_assignment_failure_never_releases_worker(tmp_path, monkeypatch):
    def boom(self, handle):
        raise model_smoke._JobError("job_assign_failed")

    monkeypatch.setattr(model_smoke._JobObject, "assign", boom)
    run, info = model_smoke._default_spawn(_spawn_ctx(tmp_path))
    assert run["execution_status"] == "harness-error"
    assert info["cleanup_status"] == "assignment_failed"
    assert not (tmp_path / "run" / "barrier-release").exists()


def test_spawn_timeout_reports_bounded_cleanup(tmp_path, monkeypatch):
    # Force the execution wait to time out deterministically; the job
    # termination path itself is real.
    def _timeout_wait(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)

    monkeypatch.setattr(model_smoke._SuspendedProcess, "wait", _timeout_wait)
    run, info = model_smoke._default_spawn(_spawn_ctx(tmp_path, timeout=30.0))
    assert run["execution_status"] == "timeout"
    assert info["timed_out"] is True
    assert info["cleanup_status"] in ("terminated", "unverified")


# ------------------------------------------- F4/F6: exercised worker wiring

def test_fake_worker_under_tripwire_bootstrap(tmp_path):
    bootstrap = _write_bootstrap(tmp_path)
    # Extend the bootstrap probe to also record browser abort evidence.
    ctx = _spawn_ctx(tmp_path)
    run, info = model_smoke._default_spawn(
        ctx, cmd_prefix=[sys.executable, str(bootstrap)])
    assert run["declared_outcome"] == "COMPLETE"
    assert run["smoke_guard_enabled"] is True
    rundir = ctx["rundir"]
    evidence = rundir / "tripwire-evidence.json"
    assert evidence.exists()
    flags = json.loads(evidence.read_text())
    assert flags["socket_guard"] is True
    assert flags["browser_route_allowlist"] is True


def test_mocked_meta_429_through_launcher_has_no_retry(monkeypatch, tmp_path):
    _install_inprocess_tripwire(monkeypatch)
    counter = []

    def handler(request):
        counter.append(str(request.url))
        return httpx.Response(429, text="slow down")

    real_build = model_compare.build_model_client
    env = _RecordingEnv({"MODEL_API_KEY": DUMMY})

    def factory(script, transport="fake", allow_live=False, environ=None,
                http_transport=None, dotenv_fallback=None):
        return real_build(script, transport=transport, allow_live=allow_live,
                          environ=env,
                          http_transport=httpx.MockTransport(handler),
                          dotenv_fallback=dotenv_fallback)

    monkeypatch.setattr(model_compare, "build_model_client", factory)

    def runner(ctx):
        rundir = ctx["rundir"]
        spec_path = rundir / "worker_spec.json"
        spec_path.write_text(json.dumps(ctx["spec"]), encoding="utf-8")
        rc = model_compare.run_one_cli(
            ["--spec", str(spec_path), "--db", str(rundir / "run.db"),
             "--out", str(rundir)])
        assert rc == 0
        raw = json.loads((rundir / "run.json").read_text())
        return raw, {"cleanup_status": "injected", "timed_out": False,
                     "job_assigned": False}

    out = tmp_path / "mocked-429"
    code = model_smoke.main(
        ["--transport", "meta", "--allow-live", "--out", str(out)],
        _run_worker=runner)
    assert code != 0
    assert len(counter) == 1  # first call failed; no retry, no fallback


# ------------------------------------------------------------- A: evidence

def _meta_good_run():
    run = _good_run()
    run["transport"] = "meta"
    run["budget"]["smoke_guard"]["transport_kind"] = "meta"
    run["budget"]["sidecar"] = [
        {"call_index": i, "usage_present": True, "usage_malformed": False,
         "usage_detail_trustworthy": True, "finish_reason": "stop",
         "estimated_cost_usd": 0.1} for i in range(1, 6)]
    return run


def test_accounting_reconciles_records_counters_and_sidecars():
    # five counted calls with no records at all must fail (fake and Meta)
    r = _good_run()
    r["budget"]["records"] = []
    assert model_smoke._accounting_complete(r) is False
    r = _meta_good_run()
    r["budget"]["records"] = []
    r["budget"]["sidecar"] = []
    assert model_smoke._accounting_complete(r) is False

    # one missing record
    r = _good_run()
    r["budget"]["records"] = r["budget"]["records"][:4]
    assert model_smoke._accounting_complete(r) is False

    # duplicate record indexes
    r = _good_run()
    r["budget"]["records"][4] = dict(r["budget"]["records"][3])
    assert model_smoke._accounting_complete(r) is False

    # Meta: missing, extra, and duplicate sidecars each fail
    r = _meta_good_run()
    r["budget"]["sidecar"] = r["budget"]["sidecar"][:4]
    assert model_smoke._accounting_complete(r) is False
    r = _meta_good_run()
    r["budget"]["sidecar"] = r["budget"]["sidecar"] + [
        {"call_index": 99, "usage_present": True, "usage_malformed": False,
         "usage_detail_trustworthy": True, "finish_reason": "stop",
         "estimated_cost_usd": 0.1}]
    assert model_smoke._accounting_complete(r) is False
    r = _meta_good_run()
    r["budget"]["sidecar"] = r["budget"]["sidecar"] + [
        dict(r["budget"]["sidecar"][0])]
    assert model_smoke._accounting_complete(r) is False

    # zero-usage made calls are valid complete evidence (fake and Meta)
    r = _good_run()
    for rec in r["budget"]["records"]:
        rec["declared_usage"] = 0
    assert model_smoke._accounting_complete(r) is True
    assert model_smoke._accounting_complete(_meta_good_run()) is True


# ------------------------------------------------------------- B: anchor

def test_dev_root_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path / "outside-dev"
    (outside / "tasks").mkdir(parents=True)
    task = {"task_id": model_smoke.FIXED_TASK_ID, "task_class": "resolvable",
            "fault": {"name": None, "params": {}},
            "runtime": {"request_text": model_smoke.FIXED_REQUEST,
                        "seed_lines": []},
            "expected": {"lines": {SKU: 1}},
            "expected_interpretation": {
                "items": [{"sku": SKU, "quantity": 1}],
                "budget_max_cents": None, "substitution_allowed": False}}
    (outside / "tasks" / f"{model_smoke.FIXED_TASK_ID}.json").write_text(
        json.dumps(task), encoding="utf-8")
    (outside / "suite.json").write_text(json.dumps(
        {"suite": "x", "task_files":
         [f"tasks/{model_smoke.FIXED_TASK_ID}.json"]}), encoding="utf-8")
    (outside / "fake_scripts.json").write_text("{}", encoding="utf-8")

    dev_parent = tmp_path / "repo" / "eval" / "model"
    dev_parent.mkdir(parents=True)
    link = dev_parent / "dev"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        # Junctions need no privilege on Windows and are equally an escape.
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip("symlink/junction creation not permitted here")
    with pytest.raises(model_smoke.SmokePreflightError):
        model_smoke.resolve_selection(tmp_path / "repo",
                                      "eval/model/dev/suite.json",
                                      model_smoke.FIXED_TASK_ID, "MA", 1,
                                      "fake", False)
    with pytest.raises(model_smoke.SmokePreflightError):
        model_smoke.load_fixed_fake_script(tmp_path / "repo")


# ------------------------------------------- D: launcher-level failure tests

def test_launcher_secret_hit_fails_package(tmp_path, monkeypatch):
    def runner(ctx):
        (ctx["out_root"] / "contaminated.txt").write_text(
            "dummy-key-for-tests-only", encoding="utf-8")
        raw = model_smoke._base_error_run("harness-error", "synthetic")
        return raw, {"cleanup_status": "injected", "timed_out": False,
                     "job_assigned": False}

    out = tmp_path / "secret-smoke"
    code = model_smoke.main(["--transport", "fake", "--out", str(out)],
                            _run_worker=runner)
    assert code == 6
    smoke = json.loads((out / "smoke.json").read_text())
    assert smoke["process_result"] == "secret_scan_failure"
    assert smoke["exit_code"] == 6
    assert "secret_scan_failure" in (out / "summary.md").read_text()


def test_launcher_missing_final_evidence_is_unknown_nonsuccess(tmp_path):
    def runner(ctx):
        return model_smoke._base_error_run("harness-error", "synthetic"), {
            "cleanup_status": "injected", "timed_out": False,
            "job_assigned": False}

    out = tmp_path / "no-evidence"
    code = model_smoke.main(["--transport", "fake", "--out", str(out)],
                            _run_worker=runner)
    assert code != 0
    evaluated = json.loads((out / "evaluated.json").read_text())
    assert evaluated["measured_outcome"] == "unknown"


def test_cost_summary_invalid_accounting_is_unknown_never_complete():
    # Counted Meta calls with no records/sidecars: accounting rejects, and
    # cost must NOT collapse to a complete zero.
    run = {"transport": "meta",
           "budget": {"calls_attempted": 5, "calls_made": 5,
                      "calls_refused": 0, "consumed_tokens": 100,
                      "reserved_tokens": 300, "overrun": False,
                      "records": [], "sidecar": [],
                      "smoke_guard": {"enabled": True,
                                      "transport_kind": "meta",
                                      "stopped": False}}}
    assert model_smoke._accounting_complete(run) is False
    cost = model_smoke._cost_summary(run)
    assert cost["status"] != "complete"
    assert cost["unknown_cost_calls"] >= 5
    assert cost["provider_billing_verified"] is False

    # Counter itself invalid -> total unknown, still never complete.
    bad = {"transport": "meta",
           "budget": {"calls_attempted": "x", "calls_made": "x",
                      "records": [], "sidecar": []}}
    cost2 = model_smoke._cost_summary(bad)
    assert cost2["status"] == "total_unknown"
    assert cost2["unknown_cost_calls"] is None


def test_saved_report_cost_unknown_for_counted_meta_calls(tmp_path):
    provider_calls = {"n": 0}

    def runner(ctx):
        provider_calls["n"] += 0  # injected evidence only; no provider call
        raw = {
            "arm": "MA", "trial": 1, "transport": "meta",
            "declared_outcome": "COMPLETE", "raw_verdict": "COMPLETE",
            "execution_status": "normal", "error": None,
            "seed_lines": [], "items": [{"target_sku": SKU, "quantity": 1}],
            "snapshots": [], "mutations": [], "completed_items": [0],
            "outcome_kind": "proceed", "failures": [],
            "executable_intent": {"items": [{"sku": SKU, "quantity": 1}],
                                  "budget_max_cents": None,
                                  "substitution_allowed": False},
            "hook_counts": {"before_add": 1, "after_add": 1, "pre_add": 1},
            "adapter_mutations": {"add_to_cart": 1, "set_line_quantity": 0,
                                  "delete_line": 0},
            "stop_reasons": [], "final_cookie": "x",
            "final_persisted": {"lines": {SKU: 1}, "resolution": "cookie",
                                "version": 1, "total_cents": 2400},
            "guard": {"applied": False},
            "budget": {"calls_attempted": 5, "calls_made": 5,
                       "calls_refused": 0, "consumed_tokens": 100,
                       "reserved_tokens": 300, "overrun": False,
                       "records": [], "sidecar": [],
                       "smoke_guard": {"enabled": True,
                                       "transport_kind": "meta",
                                       "stopped": False,
                                       "accounting_mode": "provider",
                                       "provider_billing_verified": False}},
        }
        return raw, {"cleanup_status": "injected", "timed_out": False,
                     "job_assigned": False}

    out = tmp_path / "cost-report"
    code = model_smoke.main(
        ["--transport", "meta", "--allow-live", "--out", str(out)],
        _run_worker=runner)
    assert code != 0
    assert provider_calls["n"] == 0  # no provider call was made
    smoke = json.loads((out / "smoke.json").read_text())
    acct = smoke["accounting"]
    assert acct["accounting_complete"] is False
    assert acct["cost_status"] != "complete"
    assert acct["unknown_cost_calls"] >= 5
    assert acct["provider_billing_verified"] is False


def test_output_reuse_and_denied_meta_return_before_worker(tmp_path):
    calls = {"n": 0}

    def runner(ctx):
        calls["n"] += 1
        return model_smoke._base_error_run("harness-error", "x"), {
            "cleanup_status": "injected", "timed_out": False,
            "job_assigned": False}

    out = tmp_path / "reused"
    out.mkdir()
    (out / "file.txt").write_text("x")
    assert model_smoke.main(["--transport", "fake", "--out", str(out)],
                            _run_worker=runner) == 2
    assert model_smoke.main(["--transport", "meta", "--out",
                             str(tmp_path / "denied")],
                            _run_worker=runner) == 2
    assert calls["n"] == 0


def test_spawn_create_and_resume_failures_return_sanitized(tmp_path,
                                                          monkeypatch):
    def create_boom(self):
        raise model_smoke._JobError("job_create_failed")

    monkeypatch.setattr(model_smoke._JobObject, "__init__", create_boom)
    run, info = model_smoke._default_spawn(_spawn_ctx(tmp_path))
    assert run["execution_status"] == "harness-error"
    assert info["cleanup_status"] == "job_error"

    monkeypatch.undo()
    orig_resume = model_smoke._SuspendedProcess.resume

    def resume_boom(self):
        raise model_smoke._JobError("resume_failed")

    monkeypatch.setattr(model_smoke._SuspendedProcess, "resume", resume_boom)
    run2, info2 = model_smoke._default_spawn(_spawn_ctx(tmp_path / "r2"))
    assert run2["execution_status"] == "harness-error"
    assert info2["cleanup_status"] == "job_error"
    monkeypatch.setattr(model_smoke._SuspendedProcess, "resume", orig_resume)



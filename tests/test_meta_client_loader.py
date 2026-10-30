"""Brief: local .env loader tests. Dummy sentinels and tmp fixtures only.

No real root .env is ever created, read, or replaced; every file fixture
lives under pytest tmp_path outside the repository.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from agents.env_loader import (DotEnvError, load_root_dotenv,
                               make_dotenv_fallback)
from agents.meta_client import (MetaConfigError, MetaModelClient,
                                LiveTransportNotAuthorized)
from agents.model_client import ClientConfig
from harness.model_compare import build_model_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUMMY = "DUMMY-SENTINEL-KEY-0001"
REASONING = {"intent": "low", "action": "low", "review": "low"}


def _config(**over):
    cfg = ClientConfig(
        enabled=True, provider="meta", base_url="https://api.meta.ai/v1",
        model_name="dummy-model", live_enabled=True,
        reasoning_effort_by_kind=dict(REASONING),
        max_calls=40, token_budget=100000, max_output_tokens_per_call=512)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _write_env(tmp_path: Path, content, name=".env") -> Path:
    p = tmp_path / name
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")
    return p


class _Spy:
    def __init__(self, value=DUMMY):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


# ---------------------------------------------------------------- loader

def test_loader_valid_lf_crlf_bom_comments_whitespace(tmp_path):
    _write_env(tmp_path, "# comment\n\n  MODEL_API_KEY = " + DUMMY + "  \n")
    assert load_root_dotenv(tmp_path) == DUMMY

    crlf = tmp_path / "crlf"
    crlf.mkdir()
    (crlf / ".env").write_bytes(
        ("# c\r\nMODEL_API_KEY=" + DUMMY + "\r\n").encode("utf-8"))
    assert load_root_dotenv(crlf) == DUMMY

    bom = tmp_path / "bom"
    bom.mkdir()
    (bom / ".env").write_bytes(
        b"\xef\xbb\xbf" + ("MODEL_API_KEY=" + DUMMY + "\n").encode("utf-8"))
    assert load_root_dotenv(bom) == DUMMY

    eq = tmp_path / "eq"
    eq.mkdir()
    (eq / ".env").write_text("MODEL_API_KEY=abc=def\n", encoding="utf-8")
    assert load_root_dotenv(eq) == "abc=def"  # split on first '=' only


@pytest.mark.parametrize("content,category", [
    ("", "dot_env_missing"),
    ("# only a comment\n", "dot_env_missing"),
    ("MODEL_API_KEY=\n", "dot_env_empty"),
    ("MODEL_API_KEY=" + DUMMY + "\nMODEL_API_KEY=" + DUMMY + "\n",
     "dot_env_invalid_syntax"),  # duplicate, even if equal
    ("OTHER=" + DUMMY + "\n", "dot_env_invalid_syntax"),
    ("export MODEL_API_KEY=" + DUMMY + "\n", "dot_env_invalid_syntax"),
    ('MODEL_API_KEY="' + DUMMY + '"\n', "dot_env_invalid_syntax"),
    ("MODEL_API_KEY=" + DUMMY + " # inline\n", "dot_env_invalid_syntax"),
    ("MODEL_API_KEY=has space\n", "dot_env_invalid_syntax"),
    ("MODEL_API_KEY=${OTHER}\n", "dot_env_invalid_syntax"),
    ("MODEL_API_KEY=$OTHER\n", "dot_env_invalid_syntax"),
    ("MODEL_API_KEY=`cmd`\n", "dot_env_invalid_syntax"),
    ("MODEL_API_KEY=$(cmd)\n", "dot_env_invalid_syntax"),
    ("no-equals-line\n", "dot_env_invalid_syntax"),
])
def test_loader_rejects_malformed_with_sanitized_categories(tmp_path, content,
                                                            category):
    _write_env(tmp_path, content)
    with pytest.raises(DotEnvError) as excinfo:
        load_root_dotenv(tmp_path)
    assert str(excinfo.value) == category
    assert DUMMY not in str(excinfo.value)


def test_loader_missing_oversize_undecodable(tmp_path):
    with pytest.raises(DotEnvError) as e1:
        load_root_dotenv(tmp_path)
    assert str(e1.value) == "dot_env_missing"

    _write_env(tmp_path, "MODEL_API_KEY=" + "x" * 20000 + "\n")
    with pytest.raises(DotEnvError) as e2:
        load_root_dotenv(tmp_path)
    assert str(e2.value) == "dot_env_too_large"

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / ".env").write_bytes(b"MODEL_API_KEY=\xff\xfe\n")
    with pytest.raises(DotEnvError) as e3:
        load_root_dotenv(bad)
    assert str(e3.value) == "dot_env_invalid_encoding"


def test_loader_directory_and_symlink_paths_rejected(tmp_path):
    d = tmp_path / "dirfile"
    d.mkdir()
    (d / ".env").mkdir()
    with pytest.raises(DotEnvError):
        load_root_dotenv(d)

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".env").write_text("MODEL_API_KEY=" + DUMMY + "\n",
                                  encoding="utf-8")
    link = tmp_path / "linkroot"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(link),
                               str(outside)], capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip("symlink/junction creation not permitted here")
    with pytest.raises(DotEnvError) as excinfo:
        load_root_dotenv(link)
    assert str(excinfo.value) == "dot_env_unsafe_path"


def test_callback_construction_reads_nothing(tmp_path):
    # Constructing the lazy callback must not touch the filesystem.
    cb = make_dotenv_fallback(tmp_path / "does-not-exist")
    with pytest.raises(DotEnvError):
        cb()


# ------------------------------------------------------------- constructor

def _meta(environ, fallback=None, config=None):
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={})))
    return MetaModelClient(config or _config(), allow_live=True,
                           environ=environ, http_client=client,
                           dotenv_fallback=fallback)


def test_present_env_key_wins_and_fallback_never_called():
    spy = _Spy("FALLBACK-SENTINEL")
    c = _meta({"MODEL_API_KEY": DUMMY}, fallback=spy)
    assert spy.calls == 0
    assert c._api_key == DUMMY


def test_empty_present_key_fails_without_fallback():
    spy = _Spy()
    with pytest.raises(MetaConfigError) as e:
        _meta({"MODEL_API_KEY": ""}, fallback=spy)
    assert str(e.value) == "model_api_key_invalid"
    assert spy.calls == 0


def test_explicit_empty_environ_without_fallback_fails_missing():
    with pytest.raises(MetaConfigError) as e:
        _meta({})
    assert str(e.value) == "model_api_key_missing"


def test_injected_fallback_used_once_and_mapping_not_mutated():
    spy = _Spy()
    env: dict = {}
    c = _meta(env, fallback=spy)
    assert spy.calls == 1
    assert c._api_key == DUMMY
    assert env == {}  # caller mapping unchanged


@pytest.mark.parametrize("over", [
    {"live_enabled": False}, {"provider": "other"},
    {"max_calls": True}, {"token_budget": 0},
    {"base_url": "https://example.com/v1"},
    {"reasoning_effort_by_kind": {"intent": "low"}},
])
def test_every_gate_rejects_before_fallback(over):
    spy = _Spy()
    with pytest.raises(MetaConfigError):
        MetaModelClient(_config(**over), allow_live=True, environ={},
                        dotenv_fallback=spy)
    assert spy.calls == 0


def test_denied_live_never_calls_fallback():
    spy = _Spy()
    with pytest.raises(LiveTransportNotAuthorized):
        MetaModelClient(_config(), allow_live=False, environ={},
                        dotenv_fallback=spy)
    assert spy.calls == 0


def test_factory_gates_before_fallback_and_fake_never_uses_it():
    spy = _Spy()
    script = {"budget": {"max_calls": 10},  # token_budget missing
              "client": {"model_name": "dummy-model",
                         "reasoning_effort_by_kind": dict(REASONING)}}
    with pytest.raises(MetaConfigError):
        build_model_client(script, transport="meta", allow_live=True,
                           environ={}, dotenv_fallback=spy)
    assert spy.calls == 0

    fake_client, _inner = build_model_client(
        {"responses": []}, transport="fake", dotenv_fallback=spy)
    assert spy.calls == 0
    assert fake_client is not None


# ------------------------------------------- contained child (real spawn)

_LOADER_BOOTSTRAP = r'''
import json, os, runpy, socket, sys
sys.path.insert(0, os.getcwd())
from pathlib import Path as _P
from urllib.parse import urlparse as _urlparse

DUMMY_ROOT = _P(__DUMMY_ROOT_REPR__)
EXPECTED_AUTH = "Bearer " + __EXPECTED_REPR__
MODE = __MODE_REPR__

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

# Tripwire probe BEFORE any credential acquisition or HTTP request.
try:
    socket.create_connection(("203.0.113.1", 81), timeout=3)
    raise SystemExit("tripwire_not_active")
except RuntimeError as exc:
    assert str(exc).startswith("external_network_blocked:"), str(exc)

import httpx as _httpx
import agents.env_loader as _env_loader
import harness.model_compare as _mc
from harness.model_compare import build_model_client as _real_build

_orig_load = _env_loader.load_root_dotenv
def _redirected_load(anchor, key="MODEL_API_KEY"):
    return _orig_load(DUMMY_ROOT, key)
_env_loader.load_root_dotenv = _redirected_load

_AUTH_CALLS = {"n": 0}
def _handler(request):
    _AUTH_CALLS["n"] += 1
    assert request.headers.get("authorization") == EXPECTED_AUTH, \
        "dummy authorization mismatch inside handler"
    try:
        _spec = _P(sys.argv[sys.argv.index("--spec") + 1])
        (_spec.parent / "auth-evidence.json").write_text(json.dumps(
            {"authorization_matched": True, "calls": _AUTH_CALLS["n"]}))
    except Exception:
        pass
    import json as _json
    body = _json.loads(request.content)
    kind = body["response_format"]["json_schema"]["name"]
    if kind.startswith("intent"):
        payload = {"items": [{"sku": "BTL-HIK-500-NAVYBLUE", "quantity": 1}],
                   "budget_max_cents": None, "substitution_allowed": False}
    elif kind.startswith("review"):
        payload = {"declaration": "COMPLETE", "rationale": "ok"}
    else:
        _seq = getattr(_handler, "_seq", 0)
        _handler._seq = _seq + 1
        payload = [
            {"action": "search", "query": "Hiker Water Bottle",
             "product_id": None, "sku": None, "quantity": None},
            {"action": "open_product", "query": None,
             "product_id": "bottle-hiker", "sku": None, "quantity": None},
            {"action": "select_variant", "query": None, "product_id": None,
             "sku": "BTL-HIK-500-NAVYBLUE", "quantity": None},
            {"action": "add_to_cart", "query": None, "product_id": None,
             "sku": "BTL-HIK-500-NAVYBLUE", "quantity": 1},
        ][_seq]
    return _httpx.Response(200, json={
        "choices": [{"finish_reason": "stop",
                     "message": {"content": _json.dumps(payload)}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                  "total_tokens": 15}})

def _factory(script, transport="fake", allow_live=False, environ=None,
             http_transport=None, dotenv_fallback=None):
    return _real_build(
        script, transport=transport, allow_live=allow_live,
        environ=({} if MODE == "fallback" else None),
        http_transport=_httpx.MockTransport(_handler),
        dotenv_fallback=dotenv_fallback)
_mc.build_model_client = _factory

import adapter.actions as _actions
_orig_init = _actions.BrowserSession.__init__
def _patched_init(self, base_url, headed=False):
    _orig_init(self, base_url, headed=headed)
    ctx = self.browser.new_context(service_workers="block")
    _origin = _urlparse(base_url)
    def _route(route):
        u = _urlparse(route.request.url)
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
             "service_workers_block": True, "tripwire_probe_blocked": True}))
    except Exception:
        pass
_actions.BrowserSession.__init__ = _patched_init

# Call run_one_cli on THIS module identity (the one whose build_model_client
# was patched above); runpy would re-execute a second, unpatched copy.
_mc.run_one_cli(sys.argv[1:])
'''


def _spawn_meta_child(tmp_path, mode, dummy_root, expected_sentinel):
    from harness import model_smoke
    from harness.compare import build_template_db
    from harness.snapshots import backup_copy

    bootstrap = tmp_path / f"loader_bootstrap_{mode}.py"
    code = (_LOADER_BOOTSTRAP
            .replace("__DUMMY_ROOT_REPR__", repr(str(dummy_root)))
            .replace("__EXPECTED_REPR__", repr(expected_sentinel))
            .replace("__MODE_REPR__", repr(mode)))
    bootstrap.write_text(code, encoding="utf-8")

    task = model_smoke.resolve_selection(
        PROJECT_ROOT, "eval/model/dev/suite.json",
        model_smoke.FIXED_TASK_ID, "MA", 1, "meta", True)
    rundir = tmp_path / f"child-{mode}"
    rundir.mkdir(parents=True)
    template = tmp_path / f"template-{mode}.db"
    build_template_db(str(PROJECT_ROOT / "store" / "seed_catalog.json"),
                      str(template))
    backup_copy(str(template), str(rundir / "run.db"))
    script = json.loads(json.dumps(model_smoke.LIVE_SCRIPT))
    spec = model_smoke.build_worker_spec(task, "meta", True, script)
    ctx = {"spec": spec, "task": task, "rundir": rundir,
           "out_root": tmp_path, "timeout_seconds": 360.0,
           "transport": "meta"}
    run, _info = model_smoke._default_spawn(
        ctx, cmd_prefix=[sys.executable, str(bootstrap)])
    return run, rundir


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows contained-child proof (OPEN elsewhere)")
def test_child_fallback_reaches_credential_seam_in_contained_child(tmp_path):
    dummy_root = tmp_path / "dummy-root"
    dummy_root.mkdir()
    (dummy_root / ".env").write_text("MODEL_API_KEY=" + DUMMY + "\n",
                                     encoding="utf-8")
    run, rundir = _spawn_meta_child(tmp_path, "fallback", dummy_root, DUMMY)
    assert run["declared_outcome"] == "COMPLETE", run.get("error")
    assert run["smoke_guard_enabled"] is True
    auth = json.loads((rundir / "auth-evidence.json").read_text())
    assert auth["authorization_matched"] is True
    assert auth["calls"] == 5
    trip = json.loads((rundir / "tripwire-evidence.json").read_text())
    assert trip["tripwire_probe_blocked"] is True


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows contained-child proof (OPEN elsewhere)")
def test_child_process_environment_inheritance_wins_over_file(tmp_path,
                                                               monkeypatch):
    dummy_root = tmp_path / "dummy-root"
    dummy_root.mkdir()
    (dummy_root / ".env").write_text("MODEL_API_KEY=FILE-SENTINEL-9999\n",
                                     encoding="utf-8")
    # Synthetic test-process environment only; inherited by the child.
    monkeypatch.setenv("MODEL_API_KEY", DUMMY)
    try:
        run, rundir = _spawn_meta_child(tmp_path, "inherit", dummy_root, DUMMY)
    finally:
        monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert run["declared_outcome"] == "COMPLETE", run.get("error")
    auth = json.loads((rundir / "auth-evidence.json").read_text())
    assert auth["authorization_matched"] is True  # env sentinel, not the file's


# -------------------------------------------- review fixes: parser/path/meta

@pytest.mark.parametrize("value", ["abc\x7fdef", "abc def"])
def test_loader_rejects_del_and_unicode_whitespace_values(tmp_path, value):
    _write_env(tmp_path, "MODEL_API_KEY=" + value + "\n")
    with pytest.raises(DotEnvError) as e:
        load_root_dotenv(tmp_path)
    assert str(e.value) == "dot_env_invalid_syntax"


def test_loader_rejects_link_above_the_anchor(tmp_path):
    outside = tmp_path / "outside"
    repo = outside / "repo"
    repo.mkdir(parents=True)
    (repo / ".env").write_text("MODEL_API_KEY=" + DUMMY + "\n",
                               encoding="utf-8")
    link = tmp_path / "linkparent"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(link),
                               str(tmp_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip("symlink/junction creation not permitted here")
    # The anchor itself is reached through a linked ancestor component.
    with pytest.raises(DotEnvError) as e:
        load_root_dotenv(link / "outside" / "repo")
    assert str(e.value) == "dot_env_unsafe_path"


def test_meta_maps_only_known_dotenv_categories():
    def bad_fallback():
        raise ValueError("dot_env_DUMMY_PRIVATE_LINE")

    with pytest.raises(MetaConfigError) as e:
        _meta({}, fallback=bad_fallback)
    assert str(e.value) == "model_api_key_missing"
    assert "DUMMY_PRIVATE_LINE" not in str(e.value)

    from agents.env_loader import DotEnvError

    def unknown_category():
        raise DotEnvError("dot_env_some_future_category")

    with pytest.raises(MetaConfigError) as e2:
        _meta({}, fallback=unknown_category)
    assert str(e2.value) == "model_api_key_missing"


# --------------------------------- review fix: worker spec mutation rejects

def _canonical_meta_spec():
    from harness import model_smoke
    task = model_smoke.resolve_selection(
        PROJECT_ROOT, "eval/model/dev/suite.json",
        model_smoke.FIXED_TASK_ID, "MA", 1, "meta", True)
    script = json.loads(json.dumps(model_smoke.LIVE_SCRIPT))
    return model_smoke.build_worker_spec(task, "meta", True, script)


@pytest.mark.parametrize("mutate", [
    lambda s: s.update({"trial": True}),               # bool as int
    lambda s: s.update({"arm": "MB"}),
    lambda s: s["setup"].update({"request_text": "Add something else."}),
    lambda s: s["fake_script"]["budget"].update({"token_budget": True}),
])
def test_mutated_worker_smoke_specs_reject_before_fallback(tmp_path,
                                                           monkeypatch, mutate):
    import harness.model_compare as mc

    created = {"n": 0}

    def spy_fallback(anchor, key="MODEL_API_KEY"):
        created["n"] += 1
        return DUMMY

    monkeypatch.setattr("agents.env_loader.make_dotenv_fallback",
                        lambda anchor, key="MODEL_API_KEY": spy_fallback)
    spec = _canonical_meta_spec()
    mutate(spec)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    rc = mc.run_one_cli(["--spec", str(spec_path), "--db",
                         str(tmp_path / "x.db"), "--out", str(out)])
    assert rc == 0
    run = json.loads((out / "run.json").read_text())
    assert run["execution_status"] == "harness-error"
    assert created["n"] == 0  # fallback callback never even constructed


# --------------------------- review fix: actual --run-one CLI (__main__) proof

_CLI_BOOTSTRAP = r'''
import json, os, runpy, socket, sys
sys.path.insert(0, os.getcwd())
from pathlib import Path as _P
from urllib.parse import urlparse as _urlparse

os.environ.pop("MODEL_API_KEY", None)
DUMMY_ROOT = _P(__DUMMY_ROOT_REPR__)
EXPECTED_AUTH = "Bearer " + __EXPECTED_REPR__

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

try:
    socket.create_connection(("203.0.113.1", 81), timeout=3)
    raise SystemExit("tripwire_not_active")
except RuntimeError as exc:
    assert str(exc).startswith("external_network_blocked:"), str(exc)

import httpx as _httpx
import agents.env_loader as _env_loader

_orig_load = _env_loader.load_root_dotenv
def _redirected_load(anchor, key="MODEL_API_KEY"):
    return _orig_load(DUMMY_ROOT, key)
_env_loader.load_root_dotenv = _redirected_load

_AUTH_CALLS = {"n": 0}
def _handler(request):
    _AUTH_CALLS["n"] += 1
    assert request.headers.get("authorization") == EXPECTED_AUTH, \
        "dummy authorization mismatch inside handler"
    try:
        _spec = _P(sys.argv[sys.argv.index("--spec") + 1])
        (_spec.parent / "auth-evidence.json").write_text(json.dumps(
            {"authorization_matched": True, "calls": _AUTH_CALLS["n"],
             "route": "actual-cli-main"}))
    except Exception:
        pass
    import json as _json
    body = _json.loads(request.content)
    kind = body["response_format"]["json_schema"]["name"]
    if kind.startswith("intent"):
        payload = {"items": [{"sku": "BTL-HIK-500-NAVYBLUE", "quantity": 1}],
                   "budget_max_cents": None, "substitution_allowed": False}
    elif kind.startswith("review"):
        payload = {"declaration": "COMPLETE", "rationale": "ok"}
    else:
        _seq = getattr(_handler, "_seq", 0)
        _handler._seq = _seq + 1
        payload = [
            {"action": "search", "query": "Hiker Water Bottle",
             "product_id": None, "sku": None, "quantity": None},
            {"action": "open_product", "query": None,
             "product_id": "bottle-hiker", "sku": None, "quantity": None},
            {"action": "select_variant", "query": None, "product_id": None,
             "sku": "BTL-HIK-500-NAVYBLUE", "quantity": None},
            {"action": "add_to_cart", "query": None, "product_id": None,
             "sku": "BTL-HIK-500-NAVYBLUE", "quantity": 1},
        ][_seq]
    return _httpx.Response(200, json={
        "choices": [{"finish_reason": "stop",
                     "message": {"content": _json.dumps(payload)}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                  "total_tokens": 15}})

# Stable boundaries, patched BEFORE the production module executes as
# __main__: httpx.Client (single httpx identity) and the loader/BrowserSession
# modules (single identities shared by every copy of harness.model_compare).
_RealClient = _httpx.Client
def _client_factory(*args, **kwargs):
    kwargs["transport"] = _httpx.MockTransport(_handler)
    return _RealClient(*args, **kwargs)
_httpx.Client = _client_factory

import adapter.actions as _actions
_orig_init = _actions.BrowserSession.__init__
def _patched_init(self, base_url, headed=False):
    _orig_init(self, base_url, headed=headed)
    ctx = self.browser.new_context(service_workers="block")
    _origin = _urlparse(base_url)
    def _route(route):
        u = _urlparse(route.request.url)
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
             "service_workers_block": True, "tripwire_probe_blocked": True,
             "route": "actual-cli-main"}))
    except Exception:
        pass
_actions.BrowserSession.__init__ = _patched_init

sys.argv = ["harness.model_compare", "--run-one"] + sys.argv[1:]
runpy.run_module("harness.model_compare", run_name="__main__")
'''


@pytest.mark.skipif(sys.platform != "win32",
                    reason="Windows contained-child proof (OPEN elsewhere)")
def test_actual_run_one_cli_main_route_with_fallback(tmp_path):
    """Actual CLI proof (distinct from the imported-module proof above):
    the production module executes as __main__ via runpy, with patches only
    at stable single-identity boundaries."""
    from harness import model_smoke
    from harness.compare import build_template_db
    from harness.snapshots import backup_copy

    dummy_root = tmp_path / "dummy-root"
    dummy_root.mkdir()
    (dummy_root / ".env").write_text("MODEL_API_KEY=" + DUMMY + "\n",
                                     encoding="utf-8")
    bootstrap = tmp_path / "loader_cli_bootstrap.py"
    bootstrap.write_text(
        _CLI_BOOTSTRAP.replace("__DUMMY_ROOT_REPR__", repr(str(dummy_root)))
        .replace("__EXPECTED_REPR__", repr(DUMMY)), encoding="utf-8")

    task = model_smoke.resolve_selection(
        PROJECT_ROOT, "eval/model/dev/suite.json",
        model_smoke.FIXED_TASK_ID, "MA", 1, "meta", True)
    rundir = tmp_path / "child-cli"
    rundir.mkdir(parents=True)
    template = tmp_path / "template-cli.db"
    build_template_db(str(PROJECT_ROOT / "store" / "seed_catalog.json"),
                      str(template))
    backup_copy(str(template), str(rundir / "run.db"))
    script = json.loads(json.dumps(model_smoke.LIVE_SCRIPT))
    spec = model_smoke.build_worker_spec(task, "meta", True, script)
    ctx = {"spec": spec, "task": task, "rundir": rundir,
           "out_root": tmp_path, "timeout_seconds": 360.0,
           "transport": "meta"}
    run, _info = model_smoke._default_spawn(
        ctx, cmd_prefix=[sys.executable, str(bootstrap)])
    assert run["declared_outcome"] == "COMPLETE", run.get("error")
    auth = json.loads((rundir / "auth-evidence.json").read_text())
    assert auth["authorization_matched"] is True
    assert auth["route"] == "actual-cli-main"
    assert auth["calls"] == 5


# ------------------------------------- smoke-Meta allow_live flag guard

@pytest.mark.parametrize("bad_allow_live", [False, 0, 1, "false", "missing"])
def test_smoke_meta_allow_live_variants_reject_before_anything(
        tmp_path, monkeypatch, bad_allow_live):
    import harness.model_compare as mc

    calls = {"execute": 0, "meta_init": 0, "fallback_constructed": 0,
             "fallback_invoked": 0}

    def execute_spy(*a, **k):
        calls["execute"] += 1
        raise AssertionError("execute_model_run must not be reached")

    def meta_init_spy(self, *a, **k):
        calls["meta_init"] += 1
        raise AssertionError("MetaModelClient must not be constructed")

    def fallback_spy(anchor, key="MODEL_API_KEY"):
        calls["fallback_constructed"] += 1

        def _load():
            calls["fallback_invoked"] += 1
            return DUMMY
        return _load

    monkeypatch.setattr(mc, "execute_model_run", execute_spy)
    monkeypatch.setattr("agents.meta_client.MetaModelClient.__init__",
                        meta_init_spy)
    monkeypatch.setattr("agents.env_loader.make_dotenv_fallback",
                        fallback_spy)

    spec = _canonical_meta_spec()
    if bad_allow_live == "missing":
        del spec["allow_live"]
    else:
        spec["allow_live"] = bad_allow_live
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    rc = mc.run_one_cli(["--spec", str(spec_path), "--db",
                         str(tmp_path / "x.db"), "--out", str(out)])
    assert rc == 0
    run = json.loads((out / "run.json").read_text())
    assert run["execution_status"] == "harness-error"
    assert run["error"] == "smoke_live_not_authorized"
    assert calls == {"execute": 0, "meta_init": 0,
                     "fallback_constructed": 0, "fallback_invoked": 0}


def test_smoke_meta_canonical_true_case_still_constructs_callback(
        monkeypatch):
    import harness.model_compare as mc

    constructed = {"n": 0, "invoked": 0}

    def spy(anchor, key="MODEL_API_KEY"):
        constructed["n"] += 1

        def _load():
            constructed["invoked"] += 1
            return DUMMY
        return _load

    monkeypatch.setattr("agents.env_loader.make_dotenv_fallback", spy)
    fallback = mc._make_smoke_dotenv_fallback(_canonical_meta_spec())
    assert callable(fallback)
    assert constructed == {"n": 1, "invoked": 0}  # construction reads nothing


import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIP = PROJECT_ROOT / "tests" / "fixtures" / "pilot_tripwire"
from harness import pilot_manifest as mft
from harness.model_pilot import build_cell_spec, _read_pinned_task, pilot_run_one_cli, PilotError
from harness.model_smoke import _pilot_env_block, _SuspendedProcess, _JobObject


def synthetic_env(tmp_path, sentinel, stub=False):
    # Explicit nonsecret mapping only; never enumerate os.environ.
    return {
        "SYSTEMROOT": r"C:\Windows",
        "SYSTEMDRIVE": "C:",
        "WINDIR": r"C:\Windows",
        "PATH": r"C:\Windows\System32;C:\Windows",
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(TRIP),
        "PILOT_TRIPWIRE_SENTINEL": str(sentinel),
        "PILOT_TEST_STUB": "1" if stub else "0",
    }


def test_env_block_never_inherits_and_forces_no_bytecode():
    block = _pilot_env_block(environ={}, allow=())
    assert block == "PYTHONDONTWRITEBYTECODE=1\0\0"
    assert "MODEL_API_KEY" not in block
    uni = _pilot_env_block(environ={"TEMP": "C:\\Tmp\\\u00e9"}, allow=("TEMP",))
    assert "TEMP=C:\\Tmp\\\u00e9" in uni


_COPY_FILES = [
    "harness/pilot_manifest.py", "harness/model_pilot.py", "harness/model_smoke.py",
    "harness/model_compare.py", "tests/test_model_pilot.py", "tests/test_model_pilot_cli.py",
    "tests/fixtures/pilot_tripwire/sitecustomize.py",
    "harness/__init__.py", "harness/compare.py", "harness/evaluate.py", "harness/metrics.py",
    "harness/model_metrics.py", "harness/run_task.py", "harness/snapshots.py",
    "agents/__init__.py", "agents/careful_agent.py", "agents/env_loader.py",
    "agents/full_cart_guard.py", "agents/intent.py", "agents/meta_client.py",
    "agents/model_actor.py", "agents/model_client.py", "agents/smoke_guard.py",
    "adapter/__init__.py", "adapter/actions.py", "adapter/evidence.py",
    "adapter/observations.py",
    "core/__init__.py", "core/checks.py", "core/ledger.py", "core/recovery.py",
    "core/verify.py",
    "store/__init__.py", "store/api.py", "store/app.py", "store/control.py",
    "store/faults.py", "store/models.py", "store/pages.py", "store/seed_catalog.json",
    "store/templates/cart.html", "store/templates/index.html", "store/templates/product.html",
    "eval/model/dev/suite.json", "eval/model/dev/fake_scripts.json",
    "eval/model/dev/tasks/m-clean-explicit-sku.json",
    "eval/model/dev/tasks/m-clean-seeded.json",
    "eval/model/dev/tasks/m-under-ambiguous-size.json",
    "eval/model/dev/tasks/m-fault-silent-drop.json",
    "eval/model/dev/tasks/m-fault-wrong-variant.json",
    "tests/test_model_smoke.py", "tests/test_meta_client_loader.py", "pytest.ini",
]

def _copy_isolated_project(project: Path):
    import shutil as _sh
    for rel in _COPY_FILES:
        dst = project / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        _sh.copyfile(PROJECT_ROOT / rel, dst)
    return project

def test_actual_run_one_cli_post_barrier_stub(tmp_path):
    import shutil as _sh
    registered = tmp_path / "cleanup-targets" / f"cli-{uuid.uuid4().hex[:8]}"
    project = _copy_isolated_project(registered / "project")
    sentinel = tmp_path / "preserved-sentinel" / "keep.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("preserve", encoding="utf-8")
    try:
        out = project / "var" / "model-pilot-stage1-cli" / "cells" / "s1-seeded-ca-spark"
        out.mkdir(parents=True)
        (out / "run.db").write_bytes(b"")
        task, _, _ = _read_pinned_task(project, "m-clean-seeded")
        spec = build_cell_spec(mft.STAGES[1][0], task)
        spec["_startup_barrier"] = {"ready": str(out / "barrier-ready.json"),
                                    "release": str(out / "barrier-release")}
        spec_path = out / "worker_spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        trip_sentinel = out / "trip.txt"
        env = synthetic_env(tmp_path, trip_sentinel, stub=True)
        env["PYTHONPATH"] = str(project / "tests" / "fixtures" / "pilot_tripwire")
        cmd = [sys.executable, "-m", "harness.model_pilot", "--run-one",
               "--spec", str(spec_path), "--db", str(out / "run.db"), "--out", str(out)]
        proc = subprocess.Popen(cmd, cwd=str(project), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 15
            ready = out / "barrier-ready.json"
            while not ready.exists():
                assert proc.poll() is None, proc.stderr.read()
                assert time.monotonic() < deadline
                time.sleep(0.05)
            (out / "barrier-release").write_text("release", encoding="utf-8")
            stdout, stderr = proc.communicate(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        assert proc.returncode == 0, stderr
        run = json.loads((out / "run.json").read_text())
        assert run.get("raw_stub_marker") is True
        assert not trip_sentinel.exists()
    finally:
        # delete ONLY the registered target; sentinel is outside it
        if registered.is_dir() and registered.parent.name == "cleanup-targets":
            _sh.rmtree(registered)
    assert not registered.exists()
    assert sentinel.read_text() == "preserve"


_BOOTSTRAP = r"""
import json as _json
import pathlib as _pl
import sys as _sys
import types as _types

def _make_trip(name):
    def _trip(*a, **kw):
        raise RuntimeError(f"tripwire:{name}")
    _trip._is_pilot_trip = True
    _trip._trip_name = name
    return _trip

def _bind(obj, attr, name):
    setattr(obj, attr, _make_trip(name))

def _stub_execute(setup, fault, arm, trial, script, db, out,
                  settle_ms=2000, transport="meta", allow_live=False,
                  smoke_guard=False, dotenv_fallback=None):
    root = _pl.Path(__file__).resolve().parent
    _verify_bindings(root)
    mode = (root / "fixture_mode.txt").read_text(encoding="utf-8").strip()
    if mode == "sleep":
        import subprocess as _sp, time as _t
        _sp.Popen([_sys.executable, "-c", "import time;time.sleep(30)"])
        _t.sleep(30)
    if mode == "exit":
        import os as _os
        _os._exit(3)
    return {"arm": arm, "trial": trial, "execution_status": "normal",
            "declared_outcome": "COMPLETE", "raw_integrated_marker": True,
            "seed_lines": setup.get("seed_lines") or [], "items": [],
            "snapshots": [], "mutations": [],
            "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
            "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0,
                                  "delete_line": 0},
            "stop_reasons": [], "final_cookie": None, "final_persisted": None,
            "guard": {"applied": False}, "transport": "meta",
            "budget": {"calls_attempted": 0, "calls_made": 0, "calls_refused": 0,
                       "consumed_tokens": 0, "reserved_tokens": 0, "overrun": False,
                       "records": [], "sidecar": [],
                       "smoke_guard": {"enabled": True, "transport_kind": "meta",
                                       "stopped": False}}}

def _verify_bindings(root):
    import harness.run_task as _rt
    import agents.meta_client as _mc
    import agents.env_loader as _el
    import adapter.actions as _ac
    checks = {
        "boot_server": getattr(_rt.boot_server, "_is_pilot_trip", False) is True,
        "meta_client": getattr(_mc.MetaModelClient.__init__, "_is_pilot_trip", False) is True,
        "loader": getattr(_el.load_root_dotenv, "_is_pilot_trip", False) is True,
        "browser": getattr(_ac.BrowserSession.__init__, "_is_pilot_trip", False) is True,
    }
    (root / "child-binding-verified.json").write_text(
        _json.dumps(checks), encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit(4)

def install():
    root = _pl.Path(__file__).resolve().parent
    (root / "bootstrap-loaded.json").write_text(
        _json.dumps({"loaded": True}), encoding="utf-8")
    mode = (root / "fixture_mode.txt").read_text(encoding="utf-8").strip() \
        if (root / "fixture_mode.txt").is_file() else "normal"
    # Child network guards BEFORE any operational project import.
    import socket as _socket
    def _trip_net(*a, **kw):
        raise RuntimeError("tripwire:child_network")
    _socket.socket.connect = _trip_net
    _socket.socket.connect_ex = _trip_net
    _socket.create_connection = _trip_net
    _socket.getaddrinfo = _trip_net
    _socket.gethostbyname = _trip_net
    _socket.getnameinfo = _trip_net
    _socket.socket.sendto = _trip_net
    _controls = {}
    for _name, _fn in (
            ("connect", lambda: _socket.socket().connect(("203.0.113.1", 443))),
            ("connect_ex", lambda: _socket.socket().connect_ex(("203.0.113.1", 443))),
            ("create_connection", lambda: _socket.create_connection(("203.0.113.1", 443))),
            ("getaddrinfo", lambda: _socket.getaddrinfo("sentinel.invalid", 443))):
        try:
            _fn()
            raise SystemExit(f"child control failed: {_name}")
        except RuntimeError:
            _controls[_name] = "blocked"
    # Init/browser positive controls: sentinel originals + SAME binding fn.
    _flags = {"loader": False, "meta": False, "server": False, "browser": False}
    def _sentinel(key):
        def _orig(*a, **kw):
            _flags[key] = True
        return _orig
    _ns_loader = _types.SimpleNamespace(load_root_dotenv=_sentinel("loader"))
    _bind(_ns_loader, "load_root_dotenv", "load_root_dotenv")
    class _DummyMeta:
        def __init__(self): _flags["meta"] = True
    _bind(_DummyMeta, "__init__", "meta_client")
    _ns_server = _types.SimpleNamespace(boot_server=_sentinel("server"))
    _bind(_ns_server, "boot_server", "boot_server")
    class _DummyBrowser:
        def __init__(self): _flags["browser"] = True
    _bind(_DummyBrowser, "__init__", "browser")
    _attempts = (
        ("loader", lambda: _ns_loader.load_root_dotenv("unused")),
        ("meta", lambda: _DummyMeta()),
        ("server", lambda: _ns_server.boot_server()),
        ("browser", lambda: _DummyBrowser()),
    )
    for _key, _fn in _attempts:
        if mode == "failed-control" and _key == "loader":
            # Deliberately ineffective guard: sentinel IS reached.
            _ns_loader.load_root_dotenv = _sentinel("loader")
        try:
            _fn()
            raise SystemExit(f"child init control failed: {_key}")
        except RuntimeError:
            pass
        if _flags[_key]:
            raise SystemExit(f"child init control reached original: {_key}")
        _controls[_key] = "blocked_original_not_reached"
    import importlib.abc as _abc
    class _Loader(_abc.Loader):
        def __init__(self, orig, name):
            self.orig = orig; self.name = name
        def create_module(self, spec):
            return self.orig.create_module(spec) if hasattr(self.orig, "create_module") else None
        def exec_module(self, module):
            self.orig.exec_module(module)
            if self.name == "harness.model_compare":
                module.execute_model_run = _stub_execute
            elif self.name == "harness.run_task":
                _bind(module, "boot_server", "boot_server")
            elif self.name == "agents.meta_client":
                _bind(module.MetaModelClient, "__init__", "meta_client")
            elif self.name == "agents.env_loader":
                _bind(module, "load_root_dotenv", "load_root_dotenv")
            elif self.name == "adapter.actions":
                _bind(module.BrowserSession, "__init__", "browser")
    class _Finder(_abc.MetaPathFinder):
        TARGETS = {"harness.model_compare", "harness.run_task",
                   "agents.meta_client", "agents.env_loader", "adapter.actions"}
        def find_spec(self, fullname, path, target=None):
            if fullname not in self.TARGETS:
                return None
            for f in _sys.meta_path:
                if f is self: continue
                spec = f.find_spec(fullname, path, target)
                if spec and spec.loader:
                    spec.loader = _Loader(spec.loader, fullname)
                    return spec
            return None
    _finder = _Finder()
    if mode == "missing-guard":
        _finder.TARGETS = _finder.TARGETS - {"adapter.actions"}
    # Install preflight: every required binding must be registered.
    _required = {"harness.run_task", "agents.meta_client",
                 "agents.env_loader", "adapter.actions"}
    if not _required.issubset(set(_finder.TARGETS)):
        raise SystemExit(5)
    _sys.meta_path.insert(0, _finder)
    import os as _os
    (root / "bootstrap-installed.json").write_text(
        _json.dumps({"installed": True, "pid": _os.getpid(),
                     "fixture": "pilot-bootstrap-v3",
                     "controls": _controls}), encoding="utf-8")
"""

def _build_project_run(tmp_path, mode):
    # Short registered root: deep copied projects exceed Windows path limits
    # under a nested pytest basetemp otherwise.
    registered = tmp_path.parent / "ct" / f"{mode}-{uuid.uuid4().hex[:8]}"
    project = _copy_isolated_project(registered / "project-run")
    assert (project / "harness" / "__init__.py").read_bytes() == b""
    (project / "fixture_mode.txt").write_text(mode, encoding="utf-8")
    if mode == "broken":
        (project / "_pilot_fixture_bootstrap.py").write_text(
            "def install():\n    raise SystemExit(3)\n", encoding="utf-8")
    else:
        (project / "_pilot_fixture_bootstrap.py").write_text(_BOOTSTRAP, encoding="utf-8")
    mp = project / "harness" / "model_pilot.py"
    text = mp.read_text(encoding="utf-8")
    marker = "from __future__ import annotations"
    assert marker in text
    text = text.replace(marker, marker + "\nfrom _pilot_fixture_bootstrap import install as _pilot_install\n_pilot_install()", 1)
    mp.write_text(text, encoding="utf-8")
    return registered, project

def _integrated_ctx(project, timeout=30.0):
    rundir = project / "var" / "model-pilot-stage1-i" / "cells" / "s1-seeded-ca-spark"
    rundir.mkdir(parents=True, exist_ok=True)
    (rundir / "run.db").write_bytes(b"")
    task, _, _ = _read_pinned_task(project, "m-clean-seeded")
    spec = build_cell_spec(mft.STAGES[1][0], task)
    return {"spec": spec, "task": task, "rundir": rundir,
            "timeout_seconds": timeout, "transport": "meta"}

def test_integrated_spawn_pilot_worker_normal(tmp_path, monkeypatch):
    if sys.platform != "win32":
        pytest.skip("Windows integrated spawn")
    import harness.model_smoke as ms
    registered, project = _build_project_run(tmp_path, "normal")
    monkeypatch.setattr(ms, "PROJECT_ROOT", project)
    run, info = ms.spawn_pilot_worker(_integrated_ctx(project))
    assert run.get("raw_integrated_marker") is True
    assert info["cleanup_status"] == "verified"
    assert (project / "bootstrap-loaded.json").is_file()
    installed = json.loads((project / "bootstrap-installed.json").read_text())
    assert installed["pid"] > 0
    assert installed["fixture"] == "pilot-bootstrap-v3"
    assert installed["controls"]["loader"] == "blocked_original_not_reached"
    binding = json.loads((project / "child-binding-verified.json").read_text())
    assert all(binding.values())

def test_integrated_spawn_broken_bootstrap_and_assignment(tmp_path, monkeypatch):
    if sys.platform != "win32":
        pytest.skip("Windows integrated spawn")
    import harness.model_smoke as ms
    registered, project = _build_project_run(tmp_path, "broken")
    monkeypatch.setattr(ms, "PROJECT_ROOT", project)
    run, info = ms.spawn_pilot_worker(_integrated_ctx(project))
    assert run["execution_status"] == "harness-error"
    assert info["cleanup_status"] == "worker_exited_before_ready"
    assert not (project / "bootstrap-installed.json").exists()
    # assignment failure: child never resumed, no execution marker
    registered2, project2 = _build_project_run(tmp_path, "normal")
    monkeypatch.setattr(ms, "PROJECT_ROOT", project2)
    def _boom(self, handle):
        raise ms._JobError("job_assign_failed")
    monkeypatch.setattr(ms._JobObject, "assign", _boom)
    run2, info2 = ms.spawn_pilot_worker(_integrated_ctx(project2))
    assert info2["cleanup_status"] == "assignment_failed"
    assert not (project2 / "bootstrap-loaded.json").exists()

def test_integrated_spawn_timeout_descendant_unrelated_survives(tmp_path, monkeypatch):
    if sys.platform != "win32":
        pytest.skip("Windows integrated spawn")
    import harness.model_smoke as ms
    synthetic = {"PATH": r"C:\Windows\System32;C:\Windows",
                 "SYSTEMROOT": r"C:\Windows", "TEMP": str(tmp_path),
                 "PYTHONDONTWRITEBYTECODE": "1"}
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(20)"], env=synthetic)
    try:
        registered, project = _build_project_run(tmp_path, "sleep")
        monkeypatch.setattr(ms, "PROJECT_ROOT", project)
        run, info = ms.spawn_pilot_worker(_integrated_ctx(project, timeout=3.0))
        assert info["timed_out"] is True
        assert run["execution_status"] == "timeout"
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()

def test_cli_rejects_unknown_arg_and_missing_live(tmp_path):
    sentinel = tmp_path / "trip.txt"
    env = synthetic_env(tmp_path, sentinel)
    p1 = subprocess.run([sys.executable, "-m", "harness.model_pilot", "--stage", "1", "--bogus"],
                        cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
    assert p1.returncode != 0
    p2 = subprocess.run([sys.executable, "-m", "harness.model_pilot", "--stage", "1"],
                        cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
    assert p2.returncode == 2
    assert not sentinel.exists()


def test_direct_path_rejections(tmp_path):
    root = tmp_path / "root"
    (root / "var").mkdir(parents=True)
    task, _, _ = _read_pinned_task(PROJECT_ROOT, "m-clean-seeded")
    spec = build_cell_spec(mft.STAGES[1][0], task)
    # outside var
    bad = tmp_path / "outside"
    bad.mkdir()
    (bad / "worker_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (bad / "run.db").write_bytes(b"")
    rc = pilot_run_one_cli(["--spec", str(bad / "worker_spec.json"),
                            "--db", str(bad / "run.db"), "--out", str(bad)], _root=root)
    assert rc == 2
    assert not (bad / "run.json").exists()
    # right layout, wrong barrier
    out = root / "var" / "model-pilot-stage1-test" / "cells" / "s1-seeded-ca-spark"
    out.mkdir(parents=True)
    (out / "run.db").write_bytes(b"")
    spec["_startup_barrier"] = {"ready": str(out / "elsewhere-ready"), "release": str(out / "elsewhere")}
    (out / "worker_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    rc = pilot_run_one_cli(["--spec", str(out / "worker_spec.json"),
                            "--db", str(out / "run.db"), "--out", str(out)], _root=root)
    assert rc == 0
    run = json.loads((out / "run.json").read_text())
    assert run["execution_status"] == "harness-error"
    assert "barrier" in run["error"]


def test_cli_rejects_smoke_and_extra_before_effects(tmp_path):
    root = tmp_path / "root"
    (root / "var").mkdir(parents=True)
    task, _, _ = _read_pinned_task(PROJECT_ROOT, "m-clean-seeded")
    spec = build_cell_spec(mft.STAGES[1][0], task)
    out = root / "var" / "model-pilot-stage1-test" / "cells" / "s1-seeded-ca-spark"
    out.mkdir(parents=True)
    (out / "run.db").write_bytes(b"")
    spec["_startup_barrier"] = {"ready": str(out / "barrier-ready.json"),
                                "release": str(out / "barrier-release")}
    spec["smoke"] = True
    (out / "worker_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    rc = pilot_run_one_cli(["--spec", str(out / "worker_spec.json"),
                            "--db", str(out / "run.db"), "--out", str(out)], _root=root)
    assert rc == 0
    run = json.loads((out / "run.json").read_text())
    assert run["execution_status"] == "harness-error"


def test_child_and_grandchild_lack_key_name(tmp_path):
    if sys.platform != "win32":
        pytest.skip("Windows environment block test")
    marker = tmp_path / "marker.txt"
    code = (
        "import os, subprocess, sys, pathlib;"
        "a='MODEL_API_KEY' in os.environ;"
        "g=subprocess.run([sys.executable,'-c','import os;print(1 if \"MODEL_API_KEY\" in os.environ else 0)'],capture_output=True,text=True);"
        f"pathlib.Path(r'{marker}').write_text(('1' if a else '0')+g.stdout.strip())"
    )
    synthetic = {"PATH": r"C:\Windows\System32;C:\Windows",
                 "SYSTEMROOT": r"C:\Windows", "TEMP": str(tmp_path),
                 "MODEL_API_KEY": "SYNTHETIC-SENTINEL"}
    block = _pilot_env_block(environ=synthetic, allow=("PATH", "SYSTEMROOT", "TEMP"))
    proc = _SuspendedProcess([sys.executable, "-c", code], PROJECT_ROOT,
                             tmp_path / "o.txt", tmp_path / "e.txt", env_block=block)
    proc.resume()
    proc.wait(timeout=20)
    assert marker.read_text() == "00"
    proc.close()


def test_windows_owned_timeout_cleanup_and_unrelated_sleeper(tmp_path):
    if sys.platform != "win32":
        pytest.skip("Windows job object test")
    synthetic = {"PATH": r"C:\Windows\System32;C:\Windows",
                 "SYSTEMROOT": r"C:\Windows", "TEMP": str(tmp_path),
                 "PYTHONDONTWRITEBYTECODE": "1"}
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(20)"], env=synthetic)
    code = ("import subprocess,sys,time;"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']);"
            "time.sleep(20)")
    job = _JobObject()
    proc = _SuspendedProcess([sys.executable, "-c", code], PROJECT_ROOT,
                             tmp_path / "o.txt", tmp_path / "e.txt",
                             env_block=_pilot_env_block(environ={}, allow=()))
    job.assign(proc.handle)
    proc.resume()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=0.5)
        assert job.terminate()
        proc.wait(timeout=10)
        assert job.active_processes() == 0
    finally:
        proc.close()
        job.close()
        assert unrelated.poll() is None
        unrelated.kill()
        unrelated.wait()


def test_integrated_missing_guard_and_failed_control_exit_early(tmp_path, monkeypatch):
    if sys.platform != "win32":
        pytest.skip("Windows integrated spawn")
    import harness.model_smoke as ms
    for mode in ("missing-guard", "failed-control"):
        registered, project = _build_project_run(tmp_path, mode)
        monkeypatch.setattr(ms, "PROJECT_ROOT", project)
        run, info = ms.spawn_pilot_worker(_integrated_ctx(project))
        assert run["execution_status"] == "harness-error"
        assert info["cleanup_status"] == "worker_exited_before_ready"
        assert (project / "bootstrap-loaded.json").is_file()
        assert not (project / "bootstrap-installed.json").exists()
        rundir = project / "var" / "model-pilot-stage1-i" / "cells" / "s1-seeded-ca-spark"
        assert not (rundir / "run.json").exists()


_CHILD_LAUNCH = r"""
import json, pathlib, socket, sys
root = pathlib.Path(__file__).resolve().parent
_mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
SITE = pathlib.Path(r"C:\Users\Dhanalakshmi\CartProof\cartproof\.venv\Lib\site-packages")
sys.path.insert(0, str(root))
sys.path.insert(0, str(SITE))
(root / "child-loaded.json").write_text(json.dumps({"loaded": True}), encoding="utf-8")

def _trip_net(*a, **kw):
    raise RuntimeError("tripwire:child_network")
socket.socket.connect = _trip_net
socket.create_connection = _trip_net
socket.getaddrinfo = _trip_net
try:
    socket.create_connection(("203.0.113.1", 443))
    raise SystemExit(9)
except RuntimeError:
    pass

import importlib.abc as _abc
def _make_trip(name):
    def _trip(*a, **kw):
        raise RuntimeError(f"tripwire:{name}")
    _trip._is_pilot_trip = True
    return _trip
class _Loader(_abc.Loader):
    def __init__(self, orig, name):
        self.orig = orig; self.name = name
    def create_module(self, spec):
        return self.orig.create_module(spec) if hasattr(self.orig, "create_module") else None
    def exec_module(self, module):
        self.orig.exec_module(module)
        if self.name == "harness.run_task":
            module.boot_server = _make_trip("boot_server")
        elif self.name == "agents.meta_client":
            module.MetaModelClient.__init__ = _make_trip("meta_client")
        elif self.name == "agents.env_loader":
            module.load_root_dotenv = _make_trip("load_root_dotenv")
        elif self.name == "adapter.actions":
            module.BrowserSession.__init__ = _make_trip("browser")
class _Finder(_abc.MetaPathFinder):
    TARGETS = {"harness.run_task", "agents.meta_client", "agents.env_loader", "adapter.actions"}
    if _mode == "missing":
        TARGETS = TARGETS - {"adapter.actions"}
    def find_spec(self, fullname, path, target=None):
        if fullname not in self.TARGETS:
            return None
        for f in sys.meta_path:
            if f is self: continue
            spec = f.find_spec(fullname, path, target)
            if spec and spec.loader:
                spec.loader = _Loader(spec.loader, fullname)
                return spec
        return None
sys.meta_path.insert(0, _Finder())
if _mode == "broken":
    # Real fixture installation defect: standard failure path, not a
    # chosen exit code. No installed/stage marker is produced.
    raise RuntimeError("synthetic bootstrap installation defect")
import harness.model_pilot as mp
import harness.run_task as _rt
import agents.meta_client as _mc
import agents.env_loader as _el
import adapter.actions as _ac
_checks = {
    "boot_server": getattr(_rt.boot_server, "_is_pilot_trip", False) is True,
    "meta_client": getattr(_mc.MetaModelClient.__init__, "_is_pilot_trip", False) is True,
    "loader": getattr(_el.load_root_dotenv, "_is_pilot_trip", False) is True,
    "browser": getattr(_ac.BrowserSession.__init__, "_is_pilot_trip", False) is True,
}
(root / "child-binding-verified.json").write_text(json.dumps(_checks), encoding="utf-8")
# Detection order: bindings first (a missing guard is detected here and
# its unguarded original is NEVER invoked), then actual refusals.
if not all(_checks.values()):
    raise SystemExit(4)
if _mode == "failed-control":
    # Fixture defect: one guarded entry is replaced by a harmless noop.
    _el.load_root_dotenv = lambda *a, **kw: None
_control_results = {}
for _name, _fn in (
        ("loader", lambda: _el.load_root_dotenv(root)),
        ("meta_client", lambda: _mc.MetaModelClient.__init__(object())),
        ("boot_server", lambda: _rt.boot_server()),
        ("browser", lambda: _ac.BrowserSession.__init__(object(), "http://127.0.0.1"))):
    try:
        _fn()
        _control_results[_name] = "returned_normally"
    except RuntimeError as _e:
        _control_results[_name] = str(_e)
(root / "child-controls.json").write_text(
    json.dumps(_control_results), encoding="utf-8")
_EXPECTED = {"loader": "tripwire:load_root_dotenv",
             "meta_client": "tripwire:meta_client",
             "boot_server": "tripwire:boot_server",
             "browser": "tripwire:browser"}
if _control_results != _EXPECTED:
    raise SystemExit(9)
# Installed marker is written ONLY after bindings + exact controls pass.
(root / "child-installed.json").write_text(
    json.dumps({"installed": True, "controls": {"network": "blocked"}}),
    encoding="utf-8")
# Representative colon-prefix exporter fault for CLI sanitization proof.
if _mode == "colon":
    def _raise_colon(*a, **kw):
        raise mp.PilotError("export_blocked:SYNTHETIC-MARKER-123")
    mp.export_safe_outcome = _raise_colon

ctl = root / f"wrapper-control-{_mode}"
def fake_spawn(ctx):
    run = {"arm": ctx["spec"]["arm"], "trial": 1, "execution_status": "normal",
           "declared_outcome": "COMPLETE", "seed_lines": [], "items": [],
           "snapshots": [], "mutations": [],
           "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
           "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0, "delete_line": 0},
           "stop_reasons": [], "final_cookie": "c",
           "final_persisted": {"lines": [], "resolution": "ok", "version": 1, "total_cents": 0},
           "guard": {"applied": False}, "transport": "meta",
           "budget": {"calls_attempted": 1, "calls_made": 1, "calls_refused": 0,
                      "consumed_tokens": 100, "reserved_tokens": 500, "overrun": False,
                      "records": [{"call_index": 1, "kind": "intent", "status": "made",
                                   "reserved": 500, "consumed": 100, "declared_usage": 100}],
                      "sidecar": [{"call_index": 1, "usage_present": True,
                                   "usage_malformed": False, "usage_detail_trustworthy": True,
                                   "finish_reason": "stop",
                                   "provider_usage": {"prompt_tokens": 90, "completion_tokens": 10,
                                                      "total_tokens": 100, "cached_tokens": 0,
                                                      "reasoning_tokens": 0},
                                   "estimated_cost_usd": 0.00012}],
                      "smoke_guard": {"enabled": True, "transport_kind": "meta", "stopped": False}}}
    (ctx["rundir"] / "run.json").write_text(json.dumps(run), encoding="utf-8")
    if _mode == "normal":
        # Actual write-target collision: summary.md already occupies the
        # fresh leaf, so the real exclusive create refuses it.
        (ctx["rundir"].parent.parent / "summary.md").write_text(
            "SENTINEL-SUMMARY", encoding="utf-8")
    return run, {"cleanup_status": "verified"}

def _stage_via_injected(root_arg, stage, live=False):
    return mp._run_stage_injected(root, stage, fake_spawn, production=False,
                                  control_root=ctl)
mp.run_stage = _stage_via_injected
raise SystemExit(mp.main(["--stage", "1", "--live"]))
"""

def test_process_finalization_fault_exits_nonzero(tmp_path):
    project = _copy_isolated_project(tmp_path / "ct" / "wrapper-proj")
    (project / "var").mkdir(exist_ok=True)
    launcher = project / "child_launch.py"
    launcher.write_text(_CHILD_LAUNCH, encoding="utf-8")
    sentinel = tmp_path / "preserved-sentinel" / "keep.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("preserve", encoding="utf-8")
    env = synthetic_env(tmp_path, tmp_path / "trip.txt", stub=False)
    env.pop("PYTHONPATH", None)
    # Early-exit variants first: each must fail before stage dispatch.
    for variant in ("missing", "broken", "failed-control"):
        vproj = _copy_isolated_project(tmp_path / "ct" / f"var-{variant}")
        (vproj / "var").mkdir(exist_ok=True)
        (vproj / "child_launch.py").write_text(_CHILD_LAUNCH, encoding="utf-8")
        vproc = subprocess.run(
            [sys.executable, "-I", "-S", str(vproj / "child_launch.py"), variant],
            cwd=str(vproj), env=env, capture_output=True, text=True, timeout=60)
        assert vproc.returncode != 0
        assert vproc.returncode != 2
        assert not (vproj / "child-installed.json").exists()
        assert not list(vproj.glob("wrapper-control-*"))
        assert not list(vproj.glob("var/model-pilot-stage1-*"))
    proc = subprocess.run(
        [sys.executable, "-I", "-S", str(launcher), "normal"], cwd=str(project),
        env=env, capture_output=True, text=True, timeout=60)
    # Save raw child process evidence inside the isolated copy.
    ev = project / "child-evidence"
    ev.mkdir(exist_ok=True)
    (ev / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (ev / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    (ev / "exit.txt").write_text(str(proc.returncode), encoding="utf-8")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "metadata_write_failed" in (proc.stdout + proc.stderr)
    # NEW same-run blocked snapshot from this exact child process.
    attempt_dir = project / "wrapper-control-normal" / "attempts" / "stage1"
    reservations = list(attempt_dir.glob("*/reservation.json"))
    assert len(reservations) == 1
    (ev / "reservation.json").write_bytes(reservations[0].read_bytes())
    (ev / "stage.latch").write_bytes((attempt_dir / "stage.latch").read_bytes())
    saved = json.loads((ev / "reservation.json").read_text(encoding="utf-8"))
    assert saved["state"] == "blocked"
    assert saved["evidence_status"] == "metadata_write_failed"
    out_dirs = list(project.glob("var/model-pilot-stage1-*"))
    assert len(out_dirs) == 1
    assert (out_dirs[0] / "summary.md").read_text(
        encoding="utf-8") == "SENTINEL-SUMMARY"
    assert not (out_dirs[0] / "EXPORT_PATH.txt").exists()
    # Representative colon-prefix fault through the same guarded process.
    cproc = subprocess.run(
        [sys.executable, "-I", "-S", str(launcher), "colon"], cwd=str(project),
        env=env, capture_output=True, text=True, timeout=60)
    assert cproc.returncode == 2
    assert "export_failed_unapproved" in (cproc.stdout + cproc.stderr)
    assert "SYNTHETIC-MARKER" not in (cproc.stdout + cproc.stderr)
    assert (project / "child-loaded.json").is_file()
    assert (project / "child-installed.json").is_file()
    binding = json.loads((project / "child-binding-verified.json").read_text())
    assert all(binding.values())
    assert not list(project.glob("var/model-pilot-stage1-*/EXPORT_PATH.txt"))
    assert list((project / "wrapper-control-normal" / "attempts" / "stage1").glob("stage.latch"))
    assert sentinel.read_text() == "preserve"

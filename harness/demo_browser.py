"""Automated single-page browser check that drives the REAL module command.

Spawns ``python -B -m harness.demo_ui --mode offline`` under an owned Windows
lifecycle, drives one browser page end-to-end (initial -> accepted start ->
PARTIAL -> final), then a terminal-failure scenario and an existing-page
disconnection scenario. The parent runs under a loopback-only boundary that
encloses the WHOLE flow (setup, readiness, both browser scenarios, cleanup) and
permits only the exact owned loopback ports; direct socket.connect/connect_ex/
sendto, DNS, create_connection, real HTTP transports and credential fallback
are denied, and denial canaries are exercised while the flow is underway.
All servers/browsers/processes are stopped, joined and verified in a finally
that runs even on failed assertions.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

PORT = 8766
FAILURE_PORT = 8767
OWNED_PORTS = (PORT, FAILURE_PORT)

REQUIRED_CHECKS = [
    "initial_pending_no_complete",
    "start_accepted",
    "partial_missing_visible",
    "final_a2_b1_saved_check",
    "cancel_clears_complete",
    "terminal_failure_visible",
    "disconnection_clears_complete",
    "cleanup_verified",
]


def _writable_temp(root: Path) -> str:
    w = str(root / "var" / "tmp")
    os.makedirs(w, exist_ok=True)
    os.environ["TEMP"] = w
    os.environ["TMP"] = w
    tempfile.tempdir = w
    return w


def _owned(url: str, port: int) -> bool:
    try:
        o = urlsplit(url)
    except ValueError:
        return False
    return (o.scheme == "http" and o.hostname == "127.0.0.1"
            and o.port == port)


def _make_route(blocked: list, port: int):
    def handler(route, req):
        if _owned(req.url, port):
            route.continue_()
        else:
            blocked.append(req.url)
            route.abort()
    return handler


def _addr_allowed(addr, owned_ports) -> bool:
    if isinstance(addr, tuple) and len(addr) >= 2:
        host, port = addr[0], addr[1]
    elif isinstance(addr, tuple) and len(addr) == 1:
        host, port = addr[0], None
    else:
        host, port = addr, None
    return host in ("127.0.0.1", "localhost") and port in owned_ports


def _loopback_host(addr) -> bool:
    host = addr[0] if isinstance(addr, tuple) and len(addr) >= 1 else addr
    return host in ("127.0.0.1", "localhost", "::1")


def _run_canaries(blocked: dict):
    """Exercise denial canaries against the ACTIVE boundary. Denied calls must
    never reach a real external service."""
    import socket
    try:
        socket.getaddrinfo("api.meta.ai", 443)
    except RuntimeError:
        blocked["dns_egress"] += 1
    try:
        socket.create_connection(("8.8.8.8", 443), timeout=1)
    except (RuntimeError, OSError):
        blocked["numeric_ip_egress"] += 1
    try:
        socket.socket().connect(("93.184.216.34", 80))
    except (RuntimeError, OSError):
        blocked["connect_egress"] += 1
    try:
        socket.create_connection(("127.0.0.1", 9), timeout=1)  # non-owned loopback port
    except (RuntimeError, OSError):
        blocked["nonowned_loopback"] += 1
    try:
        import httpx
        with httpx.Client(trust_env=False) as c:
            c.get("https://api.meta.ai/v1")
    except RuntimeError:
        blocked["httpx"] += 1
    from harness import shopify_live_backend
    if shopify_live_backend.resolve_token() == "":
        blocked["token_fallback"] = True
    from agents import env_loader
    try:
        env_loader.load_root_dotenv(Path("/unused"))
    except RuntimeError:
        blocked["dotenv_fallback"] = True
    return blocked


@contextmanager
def _loopback_boundary(owned_ports):
    """Parent boundary enclosing the whole protected flow: allow only exact
    owned loopback server ports for create_connection, deny direct numeric-IP
    socket.connect/connect_ex/sendto, DNS, real HTTP transports and credential
    fallback. Loopback socket.connect is allowed for internal asyncio/Playwright
    plumbing (the browser-control channel). The default offline_boundary is
    unchanged and remains for the child worker."""
    import socket
    from harness import shopify_live_backend
    from agents import env_loader
    real_getaddrinfo = socket.getaddrinfo
    real_create_connection = socket.create_connection
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    blocked = {"dns_egress": 0, "numeric_ip_egress": 0, "connect_egress": 0,
               "nonowned_loopback": 0, "httpx": 0, "token_fallback": False,
               "dotenv_fallback": False}

    def getaddrinfo(host, *a, **kw):
        if host in ("127.0.0.1", "localhost", "::1"):
            return real_getaddrinfo(host, *a, **kw)
        raise RuntimeError("offline_network_denied")

    def create_connection(addr, *a, **kw):
        if _addr_allowed(addr, owned_ports):
            return real_create_connection(addr, *a, **kw)
        raise RuntimeError("offline_network_denied")

    def connect(self, addr, *a, **kw):
        if _loopback_host(addr):
            return real_connect(self, addr, *a, **kw)
        raise RuntimeError("offline_network_denied")

    def connect_ex(self, addr, *a, **kw):
        if _loopback_host(addr):
            return real_connect_ex(self, addr, *a, **kw)
        raise RuntimeError("offline_network_denied")

    def sendto(self, data, addr, *a, **kw):
        if _loopback_host(addr):
            return real_sendto(self, data, addr, *a, **kw)
        raise RuntimeError("offline_network_denied")

    with patch.object(socket, "getaddrinfo", side_effect=getaddrinfo), \
         patch.object(socket, "create_connection", side_effect=create_connection), \
         patch.object(socket.socket, "connect", connect), \
         patch.object(socket.socket, "connect_ex", connect_ex), \
         patch.object(socket.socket, "sendto", sendto), \
         patch.object(shopify_live_backend, "resolve_token", return_value=""), \
         patch.object(env_loader, "load_root_dotenv",
                      side_effect=RuntimeError("offline_credential_fallback_denied")):
        yield blocked


def _env_block(extra=None):
    from harness.model_smoke import _pilot_env_block
    base = _pilot_env_block()
    if not base.endswith("\0\0"):
        raise RuntimeError("env_block_shape")
    if not extra:
        return base
    base = base[:-1]
    return base + "\0".join(extra) + "\0\0"


class _ModuleServer:
    """Owned module-command subprocess (python -m harness.demo_ui)."""

    def __init__(self, root: Path, port: int, hold: str, parent: Path,
                 block: bool = False):
        from harness.model_smoke import _JobObject, _SuspendedProcess, PROJECT_ROOT
        self.root = root
        self.port = port
        self.parent = parent
        self.proc = None
        self.job = None
        self._stop_result = None
        try:
            extra = ["CARTPROOF_DEMO_BLOCK=1"] if block else []
            self.proc = _SuspendedProcess(
                [sys.executable, "-B", "-m", "harness.demo_ui",
                 "--mode", "offline", "--port", str(port), "--hold", hold,
                 "--launch-parent", str(parent)],
                PROJECT_ROOT, parent / "stdout.txt", parent / "stderr.txt",
                env_block=_env_block(extra))
            self.job = _JobObject()
            self.job.assign(self.proc.handle)
            self.proc.resume()
        except BaseException:
            # Clean up a partially-created suspended process/job.
            if self.job is not None:
                try:
                    self.job.close()
                except Exception:
                    pass
            if self.proc is not None:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
                except Exception:
                    pass
                try:
                    self.proc.close()
                except Exception:
                    pass
            raise

    def wait_ready(self, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("module_command_exited")
            try:
                urllib.request.urlopen(
                    "http://127.0.0.1:%d/demo/state" % self.port, timeout=1)
                return True
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("module_command_not_ready")

    def stop(self) -> bool:
        if self._stop_result is not None:
            return self._stop_result  # retain the actual first stop result
        ok = True
        try:
            self.job.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            ok = False
        active = self.job.active_processes()
        job_close = self.job.close()
        proc_close = self.proc.close()
        self._stop_result = bool(ok and active == 0 and job_close and proc_close)
        return self._stop_result


def run_page_test(root: Path, screenshots_dir: Path) -> dict:
    from harness.demo_ui import OFFLINE_FIXTURE_MAPPING

    saved_env = {"TEMP": os.environ.get("TEMP"), "TMP": os.environ.get("TMP")}
    saved_tempdir = tempfile.tempdir
    _writable_temp(root)
    profile = json.loads((root / "harness/joint_demo_profile.json").read_text(encoding="utf-8"))
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    blocked_browser = []   # foreign requests intercepted + aborted (browser scope)
    report = {
        "browser": "chromium-headless", "viewport": [1280, 900],
        "checks": [],
        "network": {"parent": {}, "browser": {}, "worker": {}},
        "cleanup": {"verified": False, "servers": [], "browser_closes": []},
        "failure_state": None, "screenshots": {},
    }
    servers = []
    parents = []

    def new_parent() -> Path:
        p = Path(tempfile.mkdtemp(prefix="cartproof-browser-parent-"))
        (p / "ownership.json").write_text(json.dumps({"offline_control": True}),
                                          encoding="utf-8")
        parents.append(p)
        return p

    def stop_all():
        # Browsers are closed explicitly in their own finally blocks; here we
        # stop/join every owned module-command subprocess and verify closure.
        for s in servers:
            ok = s.stop()
            report["cleanup"]["servers"].append({"port": s.port, "stopped": ok})

    try:
        from playwright.sync_api import sync_playwright

        # The boundary encloses the WHOLE protected module/browser test.
        with _loopback_boundary(OWNED_PORTS) as blocked_parent:
            _run_canaries(blocked_parent)  # entry canaries

            # ---- Main flow: one page, no reload, against the module command - #
            parent = new_parent()
            srv = _ModuleServer(root, PORT, "6", parent)
            servers.append(srv)
            srv.wait_ready()

            with sync_playwright() as p:
                b = p.chromium.launch(headless=True)
                try:
                    pg = b.new_page(viewport={"width": 1280, "height": 900})
                    pg.route("**/*", _make_route(blocked_browser, PORT))
                    pg.goto("http://127.0.0.1:%d/" % PORT, wait_until="networkidle")

                    assert pg.inner_text("#status") == "not_started", "initial status"
                    assert pg.inner_text("#verdict") == "pending", "initial pending"
                    assert pg.inner_text("#qty-a") == "-", "initial qty unknown"
                    pg.screenshot(path=str(screenshots_dir / "initial.png"))
                    report["checks"].append("initial_pending_no_complete")

                    assert not [x for x in parent.iterdir() if x.is_dir()], "attempt before start"

                    pg.click("#start")
                    report["checks"].append("start_accepted")

                    # exercise denial canaries WHILE the browser flow is underway
                    _run_canaries(blocked_parent)

                    pg.wait_for_function(
                        "document.getElementById('verdict').textContent === 'PARTIAL'",
                        timeout=30000)
                    assert pg.inner_text("#qty-a") == "1", "partial A=1"
                    assert pg.inner_text("#qty-b") == "1", "partial B=1"
                    assert "A x1" in pg.inner_text("#missing"), "missing A x1"
                    assert "NO Shopify mutation" in pg.inner_text("#drop"), "drop label"
                    assert "add_to_cart" in pg.inner_text("#action-line"), "action line"
                    pg.screenshot(path=str(screenshots_dir / "partial.png"))
                    report["checks"].append("partial_missing_visible")

                    pg.wait_for_function(
                        "document.getElementById('status').textContent === 'complete'",
                        timeout=30000)
                    assert pg.inner_text("#qty-a") == "2", "final A=2"
                    assert pg.inner_text("#qty-b") == "1", "final B=1"
                    assert "COMPLETE" in pg.inner_text("#verdict"), "final COMPLETE"
                    assert "Saved checked result" in pg.inner_text("#saved"), "saved check"
                    pg.screenshot(path=str(screenshots_dir / "final.png"))
                    report["checks"].append("final_a2_b1_saved_check")

                    assert pg.evaluate("document.getElementById('start').disabled") is True, "start disabled"

                    attempts = [x for x in parent.iterdir() if x.is_dir()]
                    assert len(attempts) == 1, "expected exactly one attempt"
                    csrf = pg.evaluate(
                        "document.querySelector('meta[name=\"csrf-token\"]').content")
                    dup = pg.evaluate(
                        "(c) => fetch('/demo/start', {method:'POST', "
                        "headers:{'x-demo-csrf': c}}).then(r=>r.json())", csrf)
                    assert dup.get("started") is False, "duplicate start not refused"
                    assert len([x for x in parent.iterdir() if x.is_dir()]) == 1, "second attempt reserved"

                    canary = pg.evaluate(
                        "fetch('http://egress-canary.invalid/x')"
                        ".then(()=>'ok').catch(()=>'blocked')")
                    assert canary == "blocked", "browser egress canary not blocked"

                    run = _read_run(parent)
                    report["network"]["worker"] = {
                        "network_denied": bool(run and run.get("network_denied") is True),
                        "egress_canary_blocked": run.get("egress_canary_blocked") if run else None,
                    }

                    assert srv.stop(), "module command did not stop"
                    pg.wait_for_function(
                        "document.getElementById('verdict').textContent === 'unavailable'",
                        timeout=15000)
                    assert "COMPLETE" not in pg.inner_text("#verdict"), "COMPLETE survived disconnect"
                    assert pg.inner_text("#qty-a") == "-", "qty not cleared on disconnect"
                    pg.screenshot(path=str(screenshots_dir / "disconnection.png"))
                    report["checks"].append("disconnection_clears_complete")
                finally:
                    try:
                        b.close()
                        report["cleanup"]["browser_closes"].append({"ok": True})
                    except Exception:
                        report["cleanup"]["browser_closes"].append({"ok": False})

            # ---- terminal failure scenario (blocked worker, cancelled) ------ #
            parent2 = new_parent()
            srv2 = _ModuleServer(root, FAILURE_PORT, "0", parent2, block=True)
            servers.append(srv2)
            srv2.wait_ready()
            with sync_playwright() as p2:
                b2 = p2.chromium.launch(headless=True)
                try:
                    pg2 = b2.new_page(viewport={"width": 1280, "height": 900})
                    pg2.route("**/*", _make_route(blocked_browser, FAILURE_PORT))
                    pg2.goto("http://127.0.0.1:%d/" % FAILURE_PORT, wait_until="networkidle")
                    pg2.click("#start")
                    time.sleep(1.5)   # let the child reach its block
                    csrf2 = pg2.evaluate(
                        "document.querySelector('meta[name=\"csrf-token\"]').content")
                    pg2.evaluate(
                        "(c) => fetch('/demo/stop', {method:'POST', "
                        "headers:{'x-demo-csrf': c}})", csrf2)
                    pg2.wait_for_function(
                        "document.getElementById('status').textContent === 'failed'",
                        timeout=30000)
                    assert "COMPLETE" not in pg2.inner_text("#verdict"), "COMPLETE after cancel"
                    pg2.screenshot(path=str(screenshots_dir / "failure.png"))
                    report["checks"].append("terminal_failure_visible")
                    report["checks"].append("cancel_clears_complete")
                    report["failure_state"] = {
                        "status": pg2.inner_text("#status"),
                        "verdict": pg2.inner_text("#verdict")}
                finally:
                    try:
                        b2.close()
                        report["cleanup"]["browser_closes"].append({"ok": True})
                    except Exception:
                        report["cleanup"]["browser_closes"].append({"ok": False})

            report["network"]["parent"] = {
                "loopback_allowed": True,
                "owned_ports": list(OWNED_PORTS),
                "egress_canary": dict(blocked_parent),
            }
            report["network"]["browser"] = {
                "blocked_egress_attempts": list(dict.fromkeys(blocked_browser)),
            }

            stop_all()
            report["cleanup"]["verified"] = bool(
                all(s["stopped"] for s in report["cleanup"]["servers"])
                and all(c["ok"] for c in report["cleanup"]["browser_closes"]))
            report["checks"].append("cleanup_verified")
    except BaseException:
        stop_all()
        report["cleanup"]["verified"] = False
        raise
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        tempfile.tempdir = saved_tempdir
        for p in parents:
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass

    for name in ("initial.png", "partial.png", "final.png", "failure.png",
                 "disconnection.png"):
        p = screenshots_dir / name
        if p.is_file():
            import hashlib
            report["screenshots"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (screenshots_dir / "browser-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _read_run(parent: Path):
    for d in parent.iterdir():
        if d.is_dir() and (d / "run.json").is_file():
            try:
                return json.loads((d / "run.json").read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "var" / "demo-ui-screenshots"
    rep = run_page_test(root, out)
    print(json.dumps(rep, indent=2, sort_keys=True))

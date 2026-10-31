"""Test-only isolation bootstrap. Never imported by production (PYTHONPATH removed)."""
import importlib.abc
import os
import pathlib
import sys

SENTINEL = os.environ.get("PILOT_TRIPWIRE_SENTINEL", "")
STUB = os.environ.get("PILOT_TEST_STUB") == "1"


def _trip(name):
    if SENTINEL:
        try:
            pathlib.Path(SENTINEL).write_text(name, encoding="utf-8")
        except Exception:
            pass
    raise RuntimeError(f"tripwire:{name}")


# Network boundary
try:
    import socket
    _orig_connect = socket.socket.connect

    def _blocked_connect(self, *a, **kw):
        _trip("socket_connect")
        return _orig_connect(self, *a, **kw)

    socket.socket.connect = _blocked_connect
except Exception:
    raise


def _stub_execute(setup, fault, arm, trial, script, db, out,
                  settle_ms=2000, transport="meta", allow_live=False,
                  smoke_guard=False, dotenv_fallback=None):
    return {
        "arm": arm, "trial": trial, "execution_status": "normal",
        "declared_outcome": "COMPLETE", "raw_stub_marker": True,
        "seed_lines": setup.get("seed_lines") or [], "items": [],
        "snapshots": [], "mutations": [],
        "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
        "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0, "delete_line": 0},
        "stop_reasons": [], "final_cookie": None, "final_persisted": None,
        "guard": {"applied": False}, "transport": "meta",
        "budget": {"calls_attempted": 0, "calls_made": 0, "calls_refused": 0,
                   "consumed_tokens": 0, "reserved_tokens": 0, "overrun": False,
                   "records": [], "sidecar": [],
                   "smoke_guard": {"enabled": True, "transport_kind": "meta",
                                   "stopped": False}},
    }


class _Loader(importlib.abc.Loader):
    def __init__(self, orig, name):
        self.orig = orig
        self.name = name

    def create_module(self, spec):
        if hasattr(self.orig, "create_module"):
            return self.orig.create_module(spec)
        return None

    def exec_module(self, module):
        self.orig.exec_module(module)
        if self.name == "harness.run_task":
            orig = module.boot_server
            def _boot(*a, **kw):
                _trip("boot_server")
                return orig(*a, **kw)
            module.boot_server = _boot
        elif self.name == "agents.meta_client":
            cls = module.MetaModelClient
            orig_init = cls.__init__
            def _init(self, *a, **kw):
                _trip("meta_client_init")
                return orig_init(self, *a, **kw)
            cls.__init__ = _init
        elif self.name == "agents.env_loader":
            def _load(*a, **kw):
                _trip("load_root_dotenv")
                raise RuntimeError("tripwire:load_root_dotenv")
            module.load_root_dotenv = _load
        elif self.name == "harness.model_compare" and STUB:
            module.execute_model_run = _stub_execute


class _Finder(importlib.abc.MetaPathFinder):
    TARGETS = {"harness.run_task", "agents.meta_client", "agents.env_loader", "harness.model_compare"}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.TARGETS:
            return None
        for f in sys.meta_path:
            if f is self:
                continue
            spec = f.find_spec(fullname, path, target)
            if spec and spec.loader:
                spec.loader = _Loader(spec.loader, fullname)
                return spec
        return None


sys.meta_path.insert(0, _Finder())

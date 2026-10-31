"""Brief 7 single-smoke launcher: one fixed dev task, MA, trial 1.

Default transport is fake. Meta requires BOTH --transport meta and the
separate --allow-live authorization; neither a config file nor a saved
artifact can enable live mode. No resume, no replay, no retry: reinvoking a
live command would be another paid run.

The launcher reuses the existing worker (harness.model_compare --run-one),
the existing parent measurement (measure_worker_run), and the existing
enrichment (model_metrics.enrich_model_run). It never scores against the
model's proposed intent or declaration.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from harness.snapshots import backup_copy  # noqa: E402
from harness import model_metrics  # noqa: E402
from harness.compare import build_template_db  # noqa: E402
from harness.model_compare import measure_worker_run  # noqa: E402

FIXED_TASK_ID = "m-clean-explicit-sku"
FIXED_ARM = "MA"
FIXED_TRIAL = 1
FIXED_REQUEST = "Add 1 x BTL-HIK-500-NAVYBLUE."
DEFAULT_TIMEOUT_SECONDS = 360

# Proposed future live settings (NOT permission to use them). Constructed in
# memory only; live mode never depends on editing the shared dev fixture.
LIVE_SCRIPT = {
    "budget": {"max_calls": 8, "token_budget": 24000, "max_output_tokens": 1500},
    "client": {
        "model_name": "muse-spark-1.3",
        "base_url": "https://api.meta.ai/v1",
        "reasoning_effort_by_kind": {"intent": "low", "action": "low",
                                     "review": "low"},
        "price_table": {
            "as_of": "2026-10-30",
            "source": "https://dev.meta.ai/docs/pricing-rate-limits",
            "per_1m_tokens": {"muse-spark-1.3": {"input": 1.25,
                                                "output": 4.25,
                                                "cached_input": 0.15}}},
    },
}


class SmokePreflightError(Exception):
    """Sanitized preflight refusal category."""


# --------------------------------------------------------------------------- #
# Preflight (validates the actual resolved selection before any client/worker)
# --------------------------------------------------------------------------- #

FIXED_EXPECTED_LINES = {"BTL-HIK-500-NAVYBLUE": 1}
FIXED_INTERPRETATION = {
    "items": [{"sku": "BTL-HIK-500-NAVYBLUE", "quantity": 1}],
    "budget_max_cents": None, "substitution_allowed": False}


def _fixed_dev_root(root: Path) -> Path:
    """Canonical repo-anchored dev root; symlink/ancestor escapes rejected.

    The anchor is the resolved repository root itself. Every component of
    eval/model/dev below it must be a real directory (not a symlink or
    junction), and the resolved dev root must stay inside the anchor, before
    any suite/task/fixture content is opened.
    """
    anchor = Path(root).resolve()
    dev = anchor / "eval" / "model" / "dev"
    current = anchor
    for part in ("eval", "model", "dev"):
        current = current / part
        if current.is_symlink() or (
                hasattr(current, "is_junction") and current.is_junction()):
            raise SmokePreflightError("dev_tree_symlink_escape")
    resolved = dev.resolve()
    if not resolved.is_relative_to(anchor):
        raise SmokePreflightError("dev_tree_outside_repo_anchor")
    return resolved


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_effective_config(config: dict, transport: str) -> None:
    """Validate the effective live config before any launch/key access."""
    if transport == "fake":
        return
    from agents.model_client import is_exact_meta_origin
    budget = config.get("budget") or {}
    for key in ("max_calls", "token_budget", "max_output_tokens"):
        v = budget.get(key)
        if not _is_int(v) or v <= 0:
            raise SmokePreflightError("live_budget_invalid")
    client = config.get("client") or {}
    if not client.get("model_name"):
        raise SmokePreflightError("live_model_missing")
    if not is_exact_meta_origin(client.get("base_url")):
        raise SmokePreflightError("live_origin_invalid")
    efforts = client.get("reasoning_effort_by_kind") or {}
    if any(not efforts.get(k) for k in ("intent", "action", "review")):
        raise SmokePreflightError("live_reasoning_missing")


def resolve_selection(root: Path, suite_arg: str, task_id: str, arm: str,
                      trial: int, transport: str, allow_live: bool,
                      timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> dict:
    # Validate cheap scalars (including a finite positive timeout) BEFORE
    # opening any file contents.
    if task_id != FIXED_TASK_ID:
        raise SmokePreflightError("task_not_fixed_smoke_task")
    if arm != FIXED_ARM:
        raise SmokePreflightError("arm_not_fixed_ma")
    if trial != FIXED_TRIAL:
        raise SmokePreflightError("trial_not_fixed_1")
    if transport not in ("fake", "meta"):
        raise SmokePreflightError("transport_invalid")
    if transport == "fake" and allow_live:
        raise SmokePreflightError("conflicting_flags_fake_allow_live")
    if transport == "meta" and not allow_live:
        raise SmokePreflightError("live_not_authorized")
    if isinstance(timeout_seconds, bool) \
            or not isinstance(timeout_seconds, (int, float)) \
            or timeout_seconds != timeout_seconds \
            or timeout_seconds in (float("inf"), float("-inf")) \
            or timeout_seconds <= 0:
        raise SmokePreflightError("timeout_invalid")
    if transport == "meta":
        validate_effective_config(build_effective_config("meta"), "meta")

    root = Path(root).resolve()
    dev_root = _fixed_dev_root(root)
    expected_suite = dev_root / "suite.json"
    expected_task = dev_root / "tasks" / f"{FIXED_TASK_ID}.json"

    # Containment is established BEFORE opening contents: the suite and task
    # must be the exact fixed dev locations, not symlinks, and inside dev.
    suite_path = Path(suite_arg)
    if not suite_path.is_absolute():
        suite_path = (root / suite_path)
    if suite_path.resolve() != expected_suite or expected_suite.is_symlink():
        raise SmokePreflightError("suite_not_fixed_dev_suite")
    if expected_task.is_symlink() or not expected_task.is_relative_to(dev_root):
        raise SmokePreflightError("fixed_task_outside_dev")

    suite = json.loads(expected_suite.read_text(encoding="utf-8"))
    entries = suite.get("task_files") or []
    matching = [e for e in entries if Path(e).name == f"{FIXED_TASK_ID}.json"]
    if len(matching) != 1:
        raise SmokePreflightError("fixed_task_duplicate_or_missing_entry")
    entry = matching[0]
    if entry != f"tasks/{FIXED_TASK_ID}.json":
        raise SmokePreflightError("fixed_task_entry_not_exact")
    resolved_entry = (expected_suite.parent / entry).resolve()
    if resolved_entry != expected_task \
            or not resolved_entry.is_relative_to(dev_root):
        raise SmokePreflightError("fixed_task_outside_dev")

    task = json.loads(expected_task.read_text(encoding="utf-8"))
    runtime = task.get("runtime") or {}
    if task.get("task_id") != FIXED_TASK_ID \
            or runtime.get("request_text") != FIXED_REQUEST \
            or (runtime.get("seed_lines") or []) != [] \
            or (task.get("fault") or {}).get("name") is not None \
            or task.get("task_class") != "resolvable" \
            or (task.get("expected") or {}).get("lines") != FIXED_EXPECTED_LINES \
            or task.get("expected_interpretation") != FIXED_INTERPRETATION:
        raise SmokePreflightError("fixed_task_content_mismatch")
    return task


def load_fixed_fake_script(root: Path) -> dict:
    """Load the fixed task's MA fake entry with the same containment rules."""
    dev_root = _fixed_dev_root(root)
    fixture = dev_root / "fake_scripts.json"
    if fixture.is_symlink() or not fixture.resolve().is_relative_to(dev_root):
        raise SmokePreflightError("fake_fixture_outside_dev")
    scripts = json.loads(fixture.read_text(encoding="utf-8"))
    entry = (scripts.get(FIXED_TASK_ID) or {}).get(FIXED_ARM)
    if not isinstance(entry, dict) or not isinstance(entry.get("responses"), list):
        raise SmokePreflightError("fake_fixture_entry_invalid")
    return entry


def build_effective_config(transport: str) -> dict:
    if transport == "fake":
        # Actual fake settings the worker will use (build_model_client fake
        # branch defaults), recorded rather than only a fixture path.
        return {"transport": "fake",
                "model_name": "fake-model",
                "budget": {"max_calls": 40, "token_budget": 20000,
                           "max_output_tokens": 512},
                "script_source": "eval/model/dev/fake_scripts.json"
                                 "#m-clean-explicit-sku/MA"}
    return {"transport": "meta", **json.loads(json.dumps(LIVE_SCRIPT))}


def build_worker_spec(task: dict, transport: str, allow_live: bool,
                      script: dict, barrier: Optional[dict] = None) -> dict:
    spec = {
        "setup": {"request_text": task["runtime"]["request_text"],
                  "seed_lines": task["runtime"].get("seed_lines") or []},
        "fault": task.get("fault"),
        "arm": FIXED_ARM, "trial": FIXED_TRIAL,
        "fake_script": script,
        "transport": transport,
        "allow_live": bool(allow_live),
        "smoke": True,
    }
    if barrier is not None:
        spec["_startup_barrier"] = barrier
    return spec


# --------------------------------------------------------------------------- #
# Windows Job Object containment (smoke-owned, startup barrier)
# --------------------------------------------------------------------------- #

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1


class _JobError(Exception):
    pass


class _IoCounters(ctypes.Structure):
    _fields_ = [("a", ctypes.c_uint64), ("b", ctypes.c_uint64),
                ("c", ctypes.c_uint64), ("d", ctypes.c_uint64),
                ("e", ctypes.c_uint64), ("f", ctypes.c_uint64)]


class _BasicLimit(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimit),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _BasicAccounting(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_int64),
                ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64),
                ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD)]


class _JobObject:
    """Owned Windows job handle. Assigned before the worker is released."""

    def __init__(self):
        if sys.platform != "win32":
            raise _JobError("job_object_unsupported_platform")
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k.SetInformationJobObject.restype = wintypes.BOOL
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              wintypes.LPVOID, wintypes.DWORD]
        k.AssignProcessToJobObject.restype = wintypes.BOOL
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.TerminateJobObject.restype = wintypes.BOOL
        k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.QueryInformationJobObject.restype = wintypes.BOOL
        k.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD)]
        k.CloseHandle.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k = k
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise _JobError("job_create_failed")
        ext = _ExtendedLimit()
        ext.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(
                self.handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(ext), ctypes.sizeof(ext)):
            self.close()
            raise _JobError("job_configure_failed")

    def assign(self, process_handle: int) -> None:
        if not self._k.AssignProcessToJobObject(self.handle, process_handle):
            raise _JobError("job_assign_failed")

    def terminate(self) -> bool:
        return bool(self._k.TerminateJobObject(self.handle, 1))

    def active_processes(self) -> Optional[int]:
        info = _BasicAccounting()
        returned = wintypes.DWORD(0)
        if not self._k.QueryInformationJobObject(
                self.handle, _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)):
            return None
        return int(info.ActiveProcesses)

    def close(self) -> bool:
        if getattr(self, "handle", None):
            ok = bool(self._k.CloseHandle(self.handle))
            self.handle = None
            return ok
        return True


def _base_error_run(status: str, error: str) -> dict:
    return {"arm": FIXED_ARM, "trial": FIXED_TRIAL, "declared_outcome": None,
            "raw_verdict": None, "execution_status": status, "error": error,
            "seed_lines": [], "items": [], "snapshots": [], "mutations": [],
            "hook_counts": {"before_add": 0, "after_add": 0, "pre_add": 0},
            "adapter_mutations": {"add_to_cart": 0, "set_line_quantity": 0,
                                  "delete_line": 0},
            "stop_reasons": [], "final_cookie": None, "final_persisted": None,
            "guard": {"applied": False}}


class _StartupInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("wShowWindow", wintypes.WORD),
                ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
                ("hStdInput", wintypes.HANDLE),
                ("hStdOutput", wintypes.HANDLE),
                ("hStdError", wintypes.HANDLE)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD),
                ("dwThreadId", wintypes.DWORD)]


_CREATE_SUSPENDED = 0x00000004
_STARTF_USESTDHANDLES = 0x00000100
_HANDLE_FLAG_INHERIT = 0x00000001
_CREATE_UNICODE_ENVIRONMENT = 0x00000400

_PILOT_ENV_ALLOW = (
    'PATH', 'SYSTEMROOT', 'SYSTEMDRIVE', 'WINDIR', 'TEMP', 'TMP',
    'COMSPEC', 'PATHEXT', 'USERPROFILE', 'LOCALAPPDATA', 'APPDATA',
    'PROGRAMDATA', 'PROGRAMFILES', 'PROGRAMFILES(X86)',
    'NUMBER_OF_PROCESSORS', 'OS', 'PROCESSOR_ARCHITECTURE',
    'VIRTUAL_ENV',
)

def _pilot_env_block(environ=None, allow=None) -> str:
    # Explicit allowlist only; MODEL_API_KEY is never referenced.
    import os as _os
    env = _os.environ if environ is None else environ
    names = _PILOT_ENV_ALLOW if allow is None else tuple(allow)
    parts = []
    for k in names:
        if k in env:
            parts.append(f'{k}={env[k]}')
    # Deliberately forced, never inherited.
    parts.append('PYTHONDONTWRITEBYTECODE=1')
    parts.sort(key=str.lower)
    return '\0'.join(parts) + '\0\0'



class _SuspendedProcess:
    """A process created suspended: assign the job first, then resume.

    This closes the race where a fast-starting interpreter creates
    descendants before a post-Popen job assignment lands.
    """

    def __init__(self, cmd: list, cwd: Path, stdout_path: Path,
                 stderr_path: Path, env_block: str | None = None):
        if sys.platform != "win32":
            raise _JobError("suspended_spawn_unsupported_platform")
        import msvcrt
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateProcessW.restype = wintypes.BOOL
        k.CreateProcessW.argtypes = [
            wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.LPVOID,
            wintypes.LPVOID, wintypes.BOOL, wintypes.DWORD, wintypes.LPVOID,
            wintypes.LPCWSTR, ctypes.POINTER(_StartupInfo),
            ctypes.POINTER(_ProcessInformation)]
        k.ResumeThread.restype = wintypes.DWORD
        k.ResumeThread.argtypes = [wintypes.HANDLE]
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.GetExitCodeProcess.restype = wintypes.BOOL
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE,
                                         ctypes.POINTER(wintypes.DWORD)]
        k.TerminateProcess.restype = wintypes.BOOL
        k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.SetHandleInformation.restype = wintypes.BOOL
        k.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                           wintypes.DWORD]
        k.CloseHandle.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k = k

        # Open all three files with cleanup on partial failure: a failed
        # second or third open must not leak the earlier handles.
        self._in_f = self._out_f = self._err_f = None
        try:
            self._out_f = open(stdout_path, "wb")
            self._err_f = open(stderr_path, "wb")
            self._in_f = open("NUL", "rb")
            handles = []
            for f in (self._in_f, self._out_f, self._err_f):
                h = msvcrt.get_osfhandle(f.fileno())
                if not k.SetHandleInformation(h, _HANDLE_FLAG_INHERIT,
                                              _HANDLE_FLAG_INHERIT):
                    raise _JobError("set_handle_information_failed")
                handles.append(h)
        except BaseException:
            self._close_files()
            raise

        si = _StartupInfo()
        si.cb = ctypes.sizeof(_StartupInfo)
        si.dwFlags = _STARTF_USESTDHANDLES
        si.hStdInput, si.hStdOutput, si.hStdError = handles
        pi = _ProcessInformation()
        cmdline = ctypes.create_unicode_buffer(subprocess.list2cmdline(cmd))
        env_ptr = None
        env_buf = None
        creation_flags = _CREATE_SUSPENDED
        if env_block is not None:
            env_buf = ctypes.create_unicode_buffer(env_block)
            env_ptr = ctypes.cast(env_buf, ctypes.c_void_p)
            creation_flags |= _CREATE_UNICODE_ENVIRONMENT
            # keep buffer alive
            self._env_buf = env_buf
        ok = k.CreateProcessW(
            cmd[0], cmdline, None, None, True, creation_flags, env_ptr,
            str(cwd), ctypes.byref(si), ctypes.byref(pi))
        if not ok:
            self._close_files()
            raise _JobError("create_suspended_failed")
        self.handle = pi.hProcess
        self.thread_handle = pi.hThread
        self.pid = int(pi.dwProcessId)

    def _close_files(self):
        for f in (self._in_f, self._out_f, self._err_f):
            try:
                f.close()
            except Exception:
                pass

    def resume(self) -> None:
        prev = self._k.ResumeThread(self.thread_handle)
        if prev == 0xFFFFFFFF:
            raise _JobError("resume_failed")

    def poll(self) -> Optional[int]:
        code = wintypes.DWORD(0)
        if not self._k.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise _JobError("exit_code_query_failed")
        return None if code.value == 259 else int(code.value)  # STILL_ACTIVE

    def wait(self, timeout: Optional[float] = None) -> int:
        ms = 0xFFFFFFFF if timeout is None else int(timeout * 1000)
        result = self._k.WaitForSingleObject(self.handle, ms)
        if result == 0xFFFFFFFF:  # WAIT_FAILED
            raise _JobError("wait_failed")
        if result == 0x00000102:  # WAIT_TIMEOUT
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
        rc = self.poll()
        if rc is None:
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
        return rc

    def kill(self) -> None:
        if not self._k.TerminateProcess(self.handle, 1):
            raise _JobError("terminate_process_failed")

    def close(self) -> bool:
        ok = True
        for h in (self.thread_handle, self.handle):
            if h and not self._k.CloseHandle(h):
                ok = False
        self._close_files()
        return ok


def _run_owned_worker(ctx: dict, cmd: list, env_block: str | None = None):
    rundir: Path = ctx["rundir"]
    spec: dict = ctx["spec"]
    timeout_s: float = ctx["timeout_seconds"]
    spawn_info: dict[str, object] = {"cleanup_status": "not_started",
                                   "timed_out": False, "job_assigned": False}
    barrier = {"ready": str(rundir / "barrier-ready.json"),
               "release": str(rundir / "barrier-release")}
    spec = dict(spec)
    spec["_startup_barrier"] = barrier
    spec_path = rundir / "worker_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    ctx["spec_path"] = spec_path
    worker_args = ["--spec", str(spec_path), "--db", str(rundir / "run.db"), "--out", str(rundir)]
    full_cmd = list(cmd) + worker_args
    return _owned_lifecycle(ctx, full_cmd, env_block, spawn_info, barrier)


def _owned_lifecycle(ctx: dict, cmd: list, env_block: str | None, spawn_info: dict, barrier: dict):
    rundir: Path = ctx["rundir"]
    timeout_s: float = ctx["timeout_seconds"]
    job = None
    proc = None
    spawn_started = time.monotonic()
    try:
        job = _JobObject()
        proc = _SuspendedProcess(cmd, PROJECT_ROOT, rundir / "stdout.txt",
                                 rundir / "stderr.txt", env_block=env_block)
        try:
            job.assign(proc.handle)
        except _JobError:
            proc.kill()
            proc.wait(timeout=10)
            spawn_info["cleanup_status"] = "assignment_failed"
            return _base_error_run("harness-error", "job_assignment_failed"), spawn_info
        spawn_info["job_assigned"] = True
        proc.resume()
        deadline = spawn_started + timeout_s
        ready_path = Path(barrier["ready"])
        while not ready_path.exists():
            if proc.poll() is not None:
                spawn_info["cleanup_status"] = "worker_exited_before_ready"
                return _base_error_run("harness-error", "worker_exited_before_ready"), spawn_info
            if time.monotonic() > deadline:
                job.terminate()
                proc.wait(timeout=10)
                spawn_info["cleanup_status"] = "barrier_ready_timeout"
                return _base_error_run("harness-error", "startup_barrier_not_ready"), spawn_info
            time.sleep(0.05)
        Path(barrier["release"]).write_text("release", encoding="utf-8")
        spawn_info["startup_ms"] = round((time.monotonic() - spawn_started) * 1000, 1)
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        spawn_info["timed_out"] = timed_out
        if timed_out:
            ok = job.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
            active = job.active_processes()
            spawn_info["cleanup_status"] = ("terminated" if ok and active == 0 else "unverified")
            spawn_info["active_after_terminate"] = active
            return _base_error_run("timeout", "worker_wall_timeout"), spawn_info
        active = job.active_processes()
        spawn_info["active_before_close"] = active
        spawn_info["cleanup_status"] = ("verified" if active == 0 else "unverified")
    except _JobError as exc:
        if proc is not None:
            try:
                alive = proc.poll() is None
            except Exception:
                alive = True
            if alive:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
        spawn_info["cleanup_status"] = "job_error"
        return _base_error_run("harness-error", str(exc)), spawn_info
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
        if close_failed and spawn_info.get("cleanup_status") == "verified":
            spawn_info["cleanup_status"] = "unverified"
    run_json = rundir / "run.json"
    if run_json.exists():
        try:
            return json.loads(run_json.read_text(encoding="utf-8")), spawn_info
        except Exception as exc:
            return _base_error_run("harness-error", repr(exc)), spawn_info
    return _base_error_run("harness-error", "no run.json produced"), spawn_info


def spawn_pilot_worker(ctx: dict):
    """Production pilot spawn - fixed module, no arbitrary prefix."""
    cmd = [sys.executable, "-m", "harness.model_pilot", "--run-one"]
    env_block = _pilot_env_block()
    return _run_owned_worker(ctx, cmd, env_block=env_block)


def _default_spawn(ctx: dict, cmd_prefix: Optional[list] = None):
    """Real subprocess spawn: suspended create, job assign, resume, barrier.

    ``cmd_prefix`` is an internal Python-level test seam only (never set by
    the CLI, a spec, or the environment): when present it replaces the
    ``python -m harness.model_compare --run-one`` prefix, e.g. with a
    test-generated tripwire bootstrap.
    """
    if cmd_prefix:
        cmd = list(cmd_prefix)
        return _run_owned_worker(ctx, cmd, env_block=None)
    cmd = [sys.executable, "-m", "harness.model_compare", "--run-one"]
    return _run_owned_worker(ctx, cmd, env_block=None)


# --------------------------------------------------------------------------- #
# Scoring / success (parent-side, single measurement path)
# --------------------------------------------------------------------------- #

def _accounting_detail(run: dict):
    """Validate accounting evidence and return (ok, reasons, made_indexes).

    Records must correspond exactly to the counters: unique sequential call
    indexes, an explicit record for every made call, and (Meta) exact sidecar
    correspondence for the actual made calls. Pre-send errors/refusals are
    distinguished from made calls by their record status/consumed amount.
    """
    reasons: list[str] = []
    budget = run.get("budget")
    if not isinstance(budget, dict) or not budget:
        return False, ["budget_missing"], set()
    guard = budget.get("smoke_guard")
    if not isinstance(guard, dict) or not guard.get("enabled"):
        reasons.append("smoke_guard_missing_or_disabled")
    transport = run.get("transport")
    if isinstance(guard, dict) and guard.get("transport_kind") != transport:
        reasons.append("smoke_guard_transport_mismatch")
    if isinstance(guard, dict) and guard.get("stopped"):
        reasons.append("smoke_guard_stopped")
    for key in ("calls_attempted", "calls_made", "calls_refused",
                "consumed_tokens", "reserved_tokens"):
        if not _is_int(budget.get(key)) or budget.get(key) < 0:
            reasons.append(f"budget_counter_invalid:{key}")
    if budget.get("overrun"):
        reasons.append("usage_overrun")
    records = budget.get("records")
    if not isinstance(records, list):
        return False, reasons + ["budget_records_missing"], set()

    indexes: list[int] = []
    made_indexes: set[int] = set()
    for rec in records:
        if not isinstance(rec, dict) or not _is_int(rec.get("call_index")):
            reasons.append("budget_record_invalid")
            continue
        indexes.append(rec["call_index"])
        status = rec.get("status")
        if status in ("missing_usage", "overrun"):
            reasons.append(f"accounting_record_{status}")
        # A made call is one the wrapper actually sent: a made/missing/overrun
        # record, or an inner-error record that consumed its reservation.
        # Pre-send cap/estimate errors and refusals consumed nothing.
        if status in ("made", "missing_usage", "overrun") or (
                status == "error" and (rec.get("consumed") or 0) > 0):
            made_indexes.add(rec["call_index"])
    if len(indexes) != len(set(indexes)):
        reasons.append("budget_record_duplicate_indexes")
    attempted = budget.get("calls_attempted")
    if _is_int(attempted) and sorted(indexes) != list(range(1, attempted + 1)):
        reasons.append("budget_record_index_gap")
    made_counter = budget.get("calls_made")
    if _is_int(made_counter) and len(made_indexes) != made_counter:
        reasons.append("budget_made_counter_mismatch")

    sidecars = budget.get("sidecar") or []
    sidecar_by_index: dict[int, dict] = {}
    for s in sidecars:
        if not isinstance(s, dict) or not _is_int(s.get("call_index")):
            reasons.append("sidecar_invalid")
            continue
        if s["call_index"] in sidecar_by_index:
            reasons.append("sidecar_duplicate_index")
        sidecar_by_index[s["call_index"]] = s

    if transport == "meta":
        if set(sidecar_by_index) != made_indexes:
            reasons.append("meta_sidecar_correspondence_mismatch")
        for idx in made_indexes:
            s = sidecar_by_index.get(idx)
            if s is None:
                continue
            if not s.get("usage_present"):
                reasons.append("meta_usage_missing")
            if s.get("usage_malformed"):
                reasons.append("meta_usage_malformed")
            if not s.get("usage_detail_trustworthy", True):
                reasons.append("meta_usage_untrustworthy")
            if s.get("finish_reason") != "stop":
                reasons.append("meta_finish_not_stop")
    else:
        for rec in records:
            if isinstance(rec, dict) and rec.get("status") == "made" \
                    and rec.get("declared_usage") is None:
                reasons.append("fake_synthetic_usage_missing")
    return (not reasons), reasons, made_indexes


def _accounting_status(run: dict) -> tuple[bool, list]:
    ok, reasons, _made = _accounting_detail(run)
    return ok, reasons


def _accounting_complete(run: dict) -> bool:
    return _accounting_status(run)[0]


def validate_paid_accounting(run: dict):
    return _accounting_detail(run)


def paid_cost_summary(run: dict):
    return _cost_summary(run)


def _cost_summary(run: dict) -> dict:
    """Reconcile cost against ALL made calls, not just present sidecars."""
    budget = run.get("budget") or {}
    transport = run.get("transport")
    made = budget.get("calls_made")
    if transport == "fake":
        return {"status": "not_applicable_synthetic",
                "known_subtotal_usd_estimated": None,
                "unknown_cost_calls": 0,
                "provider_billing_verified": False}
    sidecars = budget.get("sidecar") or []
    ok, _reasons, made_indexes = _accounting_detail(run)
    # Safe known subtotal: costs actually reported on sidecar entries,
    # counted once per unique valid call index.
    known = 0.0
    seen_cost_indexes = set()
    for s in sidecars:
        if not isinstance(s, dict):
            continue
        idx = s.get("call_index")
        if _is_int(idx) and idx not in seen_cost_indexes \
                and s.get("estimated_cost_usd") is not None:
            seen_cost_indexes.add(idx)
            known += s["estimated_cost_usd"]
    if not ok:
        # Accounting could not validate the made-call mapping: the cost
        # report is incomplete/unknown, never complete. Preserve the valid
        # made-call counter as the conservative unresolved count; if the
        # counter itself is invalid, the total is simply unknown.
        made_counter = budget.get("calls_made")
        if _is_int(made_counter) and made_counter >= 0:
            return {"status": "incomplete_unknown",
                    "known_subtotal_usd_estimated": known,
                    "unknown_cost_calls": made_counter,
                    "provider_billing_verified": False}
        return {"status": "total_unknown",
                "known_subtotal_usd_estimated": known,
                "unknown_cost_calls": None,
                "provider_billing_verified": False}
    # Costed calls are made calls (by validated index) with a known cost;
    # duplicate sidecar entries never count as separate calls.
    costed_indexes = set()
    known = 0.0
    seen_cost_indexes = set()
    for s in sidecars:
        if not isinstance(s, dict):
            continue
        idx = s.get("call_index")
        if idx in made_indexes and idx not in seen_cost_indexes \
                and s.get("estimated_cost_usd") is not None:
            seen_cost_indexes.add(idx)
            costed_indexes.add(idx)
            known += s["estimated_cost_usd"]
    unknown = len(made_indexes - costed_indexes)
    return {"status": "complete" if unknown == 0 else "unknown_calls_present",
            "known_subtotal_usd_estimated": known,
            "unknown_cost_calls": unknown,
            "provider_billing_verified": False}


def _success(run: dict, evaluated: dict, spawn_info: dict,
             execution_mode: str = "subprocess") -> tuple[bool, list]:
    reasons = []
    if run.get("execution_status") != "normal":
        reasons.append(f"execution_status={run.get('execution_status')}")
    if run.get("measured_outcome") != "correct":
        reasons.append(f"measured_outcome={run.get('measured_outcome')}")
    if not run.get("final_cookie"):
        reasons.append("final_cookie_missing")
    if run.get("measured_resolution") not in ("cookie", "sole_session"):
        reasons.append(f"measured_resolution={run.get('measured_resolution')}")
    if not evaluated.get("completed_execution"):
        reasons.append("execution_incomplete")
    if evaluated.get("interpretation_outcome") != "intent_correct":
        reasons.append(
            f"interpretation={evaluated.get('interpretation_outcome')}")
    ok, acct_reasons = _accounting_status(run)
    if not ok:
        reasons.append("accounting_incomplete:" + ",".join(acct_reasons))
    if run.get("declared_outcome") != "COMPLETE":
        reasons.append(f"declared_outcome={run.get('declared_outcome')}")
    if spawn_info.get("timed_out"):
        reasons.append("worker_timed_out")
    if execution_mode == "subprocess":
        # Actual subprocess success requires positively verified cleanup.
        if spawn_info.get("cleanup_status") != "verified":
            reasons.append(
                f"cleanup_not_verified:{spawn_info.get('cleanup_status')}")
    else:
        # Injected/mocked evidence is separately labeled; the controlled
        # runner must explicitly report its cleanup result.
        if spawn_info.get("cleanup_status") != "injected":
            reasons.append("injected_cleanup_not_reported")
    return (not reasons), reasons


def _exit_code(run: dict, success: bool, spawn_info: dict) -> int:
    if success:
        return 0
    if spawn_info.get("timed_out") or run.get("execution_status") in (
            "crash", "timeout", "harness-error"):
        return 3
    if not _accounting_complete(run):
        return 4
    return 5


def _scan_secrets(out_root: Path) -> dict:
    patterns = ("dummy-key-for-tests-only", "Bearer ", "MODEL_API_KEY=")
    hits = 0
    scanned = 0
    for p in out_root.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".md", ".txt"):
            scanned += 1
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            hits += sum(text.count(pat) for pat in patterns)
    return {"files_scanned": scanned, "sentinel_hits": hits}


# --------------------------------------------------------------------------- #
# Launcher entry
# --------------------------------------------------------------------------- #

def main(argv=None, _run_worker=None) -> int:
    p = argparse.ArgumentParser(description="Brief 7 single smoke (one task, MA)")
    p.add_argument("--suite", default="eval/model/dev/suite.json")
    p.add_argument("--task", default=FIXED_TASK_ID)
    p.add_argument("--arm", default=FIXED_ARM)
    p.add_argument("--trial", type=int, default=FIXED_TRIAL)
    p.add_argument("--transport", choices=("fake", "meta"), default="fake")
    p.add_argument("--allow-live", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--timeout-seconds", type=float,
                   default=DEFAULT_TIMEOUT_SECONDS)
    args = p.parse_args(argv)

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        task = resolve_selection(PROJECT_ROOT, args.suite, args.task, args.arm,
                                 args.trial, args.transport, args.allow_live,
                                 timeout_seconds=args.timeout_seconds)
    except SmokePreflightError as exc:
        print(json.dumps({"process_result": "preflight_refusal",
                          "reason": str(exc)}))
        return 2
    except Exception as exc:
        print(json.dumps({"process_result": "preflight_refusal",
                          "reason": repr(exc)}))
        return 2

    out_root = Path(args.out) if args.out else (
        PROJECT_ROOT / "var" / f"model-smoke-{time.strftime('%Y%m%d-%H%M%S')}")
    if out_root.exists():
        print(json.dumps({"process_result": "preflight_refusal",
                          "reason": "output_location_reused"}))
        return 2
    rundir = out_root / "runs" / FIXED_TASK_ID / FIXED_ARM / f"t{FIXED_TRIAL}"
    rundir.mkdir(parents=True)

    # Fresh independent template + run DB.
    template = out_root / "template.db"
    catalog = str(PROJECT_ROOT / "store" / "seed_catalog.json")
    build_template_db(catalog, str(template))
    backup_copy(str(template), str(rundir / "run.db"))

    if args.transport == "fake":
        script = load_fixed_fake_script(PROJECT_ROOT)
    else:
        script = json.loads(json.dumps(LIVE_SCRIPT))

    spec = build_worker_spec(task, args.transport, args.allow_live, script)
    ctx = {"spec": spec, "task": task, "rundir": rundir, "out_root": out_root,
           "timeout_seconds": float(args.timeout_seconds),
           "transport": args.transport}
    if _run_worker is None:
        raw_run, spawn_info = _default_spawn(ctx)
    else:
        raw_run, spawn_info = _run_worker(ctx)

    # Parent-side single measurement + enrichment; raw run.json untouched.
    measured = measure_worker_run(raw_run, task, rundir)
    evaluated = model_metrics.enrich_model_run(measured, task)
    (out_root / "evaluated.json").write_text(
        json.dumps(evaluated, indent=2, default=str), encoding="utf-8")

    execution_mode = "subprocess" if _run_worker is None else "injected_mocked"
    success, reasons = _success(measured, evaluated, spawn_info,
                                execution_mode=execution_mode)
    code = _exit_code(measured, success, spawn_info)

    budget = measured.get("budget") or {}
    cost = _cost_summary(measured)
    try:
        source_state = {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                text=True).strip(),
            "status_porcelain": subprocess.check_output(
                ["git", "status", "--porcelain", "-uall"],
                cwd=str(PROJECT_ROOT), text=True).splitlines(),
            "diff_stat": subprocess.check_output(
                ["git", "diff", "--stat"], cwd=str(PROJECT_ROOT),
                text=True),
        }
    except Exception:
        source_state = {"head": None, "status_porcelain": None,
                        "diff_stat": None}
    smoke = {
        "source": source_state,
        "selection": {"task": FIXED_TASK_ID, "arm": FIXED_ARM,
                      "trial": FIXED_TRIAL},
        "transport_label": ("fake (offline plumbing evidence)"
                            if args.transport == "fake"
                            else "meta (technical flags set; not proof of "
                                 "user approval)"),
        "execution_mode": execution_mode,
        "effective_config": build_effective_config(args.transport),
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "process_result": "success" if success else "non-success",
        "exit_code": code,
        "failure_reasons": reasons,
        "declared_outcome": measured.get("declared_outcome"),
        "measured_outcome": measured.get("measured_outcome"),
        "accounting": {
            "calls_attempted_total": (
                (budget.get("smoke_guard") or {}).get(
                    "model_calls_attempted_total", budget.get("calls_attempted"))),
            "calls_made": budget.get("calls_made"),
            "calls_refused": budget.get("calls_refused"),
            "guard_locally_refused": (
                (budget.get("smoke_guard") or {}).get("guard_locally_refused")),
            "reserved_tokens": budget.get("reserved_tokens"),
            "consumed_tokens": budget.get("consumed_tokens"),
            "overrun": budget.get("overrun"),
            "accounting_complete": _accounting_complete(measured),
            "accounting_mode": (
                (budget.get("smoke_guard") or {}).get("accounting_mode")),
            "provider_billing_verified": False,
            "known_cost_subtotal_usd_estimated": cost[
                "known_subtotal_usd_estimated"],
            "unknown_cost_calls": cost["unknown_cost_calls"],
            "cost_status": cost["status"],
            "cost_label": "ESTIMATED (separate from confirmed billing)",
        },
        "spawn": spawn_info,
    }
    (out_root / "smoke.json").write_text(
        json.dumps(smoke, indent=2, default=str), encoding="utf-8")
    acct = smoke["accounting"]
    (out_root / "summary.md").write_text(
        "# Brief 7 single smoke\n\n"
        f"- transport: {smoke['transport_label']}\n"
        f"- execution mode: {execution_mode}\n"
        f"- selection: {FIXED_TASK_ID} / {FIXED_ARM} / trial {FIXED_TRIAL}\n"
        f"- declared: {smoke['declared_outcome']} | measured: "
        f"{smoke['measured_outcome']} | process: {smoke['process_result']}\n"
        f"- calls: attempted-total {acct['calls_attempted_total']}, made "
        f"{acct['calls_made']}, refused {acct['calls_refused']}, "
        f"guard-locally-refused {acct['guard_locally_refused']}\n"
        f"- tokens: reserved {acct['reserved_tokens']}, consumed "
        f"{acct['consumed_tokens']}, overrun {acct['overrun']}, "
        f"accounting complete {acct['accounting_complete']} "
        f"(mode {acct['accounting_mode']}, provider billing verified "
        f"{acct['provider_billing_verified']})\n"
        f"- estimated cost: known subtotal "
        f"{acct['known_cost_subtotal_usd_estimated']}, unknown-cost calls "
        f"{acct['unknown_cost_calls']} ({acct['cost_status']}); ESTIMATED, "
        "separate from confirmed billing\n"
        "- network isolation: this production launcher run carries no "
        "test tripwire; isolation is proven separately by the tripwire "
        "tests in tests/test_model_smoke.py\n"
        f"- failure reasons: {reasons}\n"
        "- One run is not a reliability score or evidence that recovery "
        "beats a baseline.\n", encoding="utf-8")

    scan = _scan_secrets(out_root)
    (out_root / "secret-scan.txt").write_text(
        json.dumps(scan, indent=2), encoding="utf-8")
    if scan["sentinel_hits"] > 0:
        # A credential/sentinel finding blocks package delivery. Only safe
        # counts/categories are reported; the match itself is never printed.
        # All final artifacts are updated together so they cannot disagree.
        smoke["process_result"] = "secret_scan_failure"
        smoke["failure_reasons"] = reasons + ["secret_scan_hit"]
        smoke["exit_code"] = 6
        code = 6
        (out_root / "smoke.json").write_text(
            json.dumps(smoke, indent=2, default=str), encoding="utf-8")
        summary_path = out_root / "summary.md"
        _summary_text = summary_path.read_text(encoding="utf-8")
        _summary_text = _summary_text.replace(
            "process: success", "process: secret_scan_failure").replace(
            "process: non-success", "process: secret_scan_failure")
        summary_path.write_text(_summary_text, encoding="utf-8")

    manifest = {}
    for fp in sorted(out_root.rglob("*")):
        if fp.is_file() and fp.name != "manifest.json":
            manifest[fp.relative_to(out_root).as_posix()] = hashlib.sha256(
                fp.read_bytes()).hexdigest()
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"process_result": smoke["process_result"],
                      "exit_code": code, "out": str(out_root)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

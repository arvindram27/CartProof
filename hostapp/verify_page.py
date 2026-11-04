"""Hosted "Verify my cart" page: CartProof's real verification, live, on the
visitor's own session cart, against a fixed published demo task.

Design decision (cart-state transport): the verify handler reads the visitor's
cart over the store's OWN PUBLIC HTTP API (GET /api/cart) on the loopback
address of the same process, forwarding only the visitor's session cookie.
HTTP was chosen so the evidence path is the same network interface any
external agent uses - routing, session handling and the served JSON shape.
(httpx is installed in the project venv, so an in-process ASGI call was
possible; it was rejected because it bypasses the served interface, as would
a direct models read. The choice is evidence fidelity, not dependency
avoidance.) Cost: loopback latency, and the entry point must know its own
address (app.state.self_base_url, supplied by deploy_entry).

This slice has no browser DOM channel, so the dual-channel settlement
machinery in core.verify is driven with TWO INDEPENDENT READS of the same
HTTP interface. Real settlement (the reads must agree on version and contents
within a deadline) but single-interface evidence; the pages say so, and the
full DOM/API coherence story stays with the local Playwright flow.

Session binding: no cart id, session id or task input is accepted anywhere;
the only credential is the cp_session cookie, validated server-side by the
store (128-bit random ids). Every run NORMALIZES the cookie once: one
bootstrap read returns the store-validated (or freshly minted) session id and
all subsequent reads use exactly that id. If the bootstrap read cannot
validate/mint a session, the run renders UNKNOWN WITHOUT settling - a failed
bootstrap never falls back to re-presenting an invalid cookie, so it cannot
mint a session per read. A STOLEN VALID cookie would still read that
visitor's cart; that residual limitation is stated on the page.

Budgets (enforced, not estimated): the loopback transport is OWNED by this
module (raw socket, minimal HTTP/1.1 client) precisely so the bounds hold
end to end - a stock http.client cannot provide them (its header parsing
and its Connection: close path escape any socket re-clamp). Every single
recv AND send is preceded by a timeout re-clamped to the remaining budget,
so a peer trickling status line, headers, or body cannot stretch the wall
clock past the ABSOLUTE monotonic deadline; the wire response (headers +
body together) is capped at MAX_WIRE_BYTES, checked BEFORE the buffer
grows, and a declared Content-Length beyond the cap is refused before the
body is read at all. The number of loopback reads per run is hard-counted
(bootstrap included): exhausting MAX_READS_PER_RUN turns later reads into
transport errors and the run renders UNKNOWN. One handler call is bounded
by TOTAL_BUDGET_MS end to end: one bootstrap read (<= BOOTSTRAP_MS) plus
settlement under a SETTLE_MS deadline (at 100 ms intervals at most ~20
rounds = 40 reads, 41 reads total with the bootstrap). Concurrency
admission is enforced in ASGI middleware BEFORE the thread pool (see
hostapp.security.AdmissionMiddleware). Failures render an honest UNKNOWN
verdict; internal details never reach the response.
"""
from __future__ import annotations

import json
import socket
import time
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from core.checks import evaluate
from core.ledger import CONFIRMED, CONTRADICTED, UNKNOWN, Ledger
from core.verify import Verifier

router = APIRouter()

_PKG = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_PKG / "templates"))

#: Enforced budgets (see module docstring).
SETTLE_MS = 2000
SAMPLE_INTERVAL_MS = 100
BOOTSTRAP_MS = 1500
TOTAL_BUDGET_MS = 6000
MAX_WIRE_BYTES = 262144  # headers + body; loopback cart reads are a few KB
MAX_READS_PER_RUN = 41   # 1 bootstrap + 2 * ceil(SETTLE_MS / interval)
_PER_RECV_CAP_S = 0.25   # recv timeout granularity for deadline re-checks

_DEMO_TASK = json.loads((_PKG / "demo_task.json").read_text(encoding="utf-8"))


def _now_ms() -> float:
    return time.monotonic() * 1000.0


def _money(cents: Optional[int]) -> str:
    if cents is None:
        return "unknown"
    return f"${cents / 100:.2f}"


def _self_base(request: Request) -> str:
    base = getattr(request.app.state, "self_base_url", None)
    if not base:
        raise RuntimeError("self_base_url not configured on app.state")
    return base.rstrip("/")


class _ReadTimeout(Exception):
    pass


class _WireTooLarge(Exception):
    pass


class _WireProtocol(Exception):
    pass


def _http_get_json(url: str, cookie: Optional[str], deadline_ms: float):
    """One GET of the store's public cart API against an ABSOLUTE deadline.

    Returns (payload, set_cookie). Payload is the decoded JSON body,
    {"read_timeout": True} when the deadline is exceeded, or
    {"transport_error": ...} on any other failure - both lack a version and
    become malformed/timeout evidence downstream, never exceptions.

    This is an owned minimal HTTP/1.1 client on a raw socket. Every recv and
    send is preceded by a timeout re-clamped to the remaining budget, which
    covers the status line and headers as much as the body - a stock
    http.client cannot offer that (getresponse() parses headers on the
    connect-time timeout, and on Connection: close it drops its socket
    reference while the response keeps the original timeout). A MONOTONIC
    wire counter caps the whole response (headers + body together) at
    MAX_WIRE_BYTES - the read buffer itself is consumed as parsing
    proceeds, so its length is not a wire count; the aggregate chunked
    body is additionally capped before each chunk is appended, an over-cap
    declared Content-Length is refused without reading the body, and
    negative Content-Length / chunk sizes are protocol refusals.
    """
    parts = urlsplit(url)
    path = parts.path + ("?" + parts.query if parts.query else "")

    def remaining_s() -> float:
        return (deadline_ms - _now_ms()) / 1000.0

    if remaining_s() <= 0:
        return {"read_timeout": True}, None

    sock = None
    buf = bytearray()
    total_wire = 0  # monotonic: buf is CONSUMED by read_until/read_exact

    def recv_some() -> bytes:
        nonlocal total_wire
        rem = remaining_s()
        if rem <= 0:
            raise _ReadTimeout()
        sock.settimeout(min(rem, _PER_RECV_CAP_S))
        try:
            data = sock.recv(min(65536, MAX_WIRE_BYTES + 1 - total_wire))
        except (socket.timeout, TimeoutError):
            raise _ReadTimeout()
        if total_wire + len(data) > MAX_WIRE_BYTES:
            raise _WireTooLarge()
        total_wire += len(data)
        buf.extend(data)
        return data

    def read_until(delim: bytes) -> bytes:
        while True:
            idx = buf.find(delim)
            if idx >= 0:
                out = bytes(buf[:idx])
                del buf[:idx + len(delim)]
                return out
            if not recv_some():
                raise _WireProtocol("peer closed mid-message")

    def read_exact(n: int) -> bytes:
        if n > MAX_WIRE_BYTES:
            raise _WireTooLarge()
        while len(buf) < n:
            if not recv_some():
                raise _WireProtocol("peer closed mid-body")
        out = bytes(buf[:n])
        del buf[:n]
        return out

    try:
        sock = socket.create_connection(
            (parts.hostname, parts.port or 80), timeout=remaining_s())
        request = (
            "GET " + path + " HTTP/1.1\r\n" +
            "Host: " + parts.netloc + "\r\n" +
            "Accept: application/json\r\n" +
            "Connection: close\r\n" +
            ("Cookie: cp_session=" + cookie + "\r\n" if cookie else "") +
            "\r\n")
        sock.settimeout(min(remaining_s(), _PER_RECV_CAP_S))
        try:
            sock.sendall(request.encode("ascii"))
        except (socket.timeout, TimeoutError):
            raise _ReadTimeout()

        status_line = read_until(b"\r\n").decode("iso-8859-1")
        try:
            status = int(status_line.split(" ", 2)[1])
        except (IndexError, ValueError):
            raise _WireProtocol("bad status line")
        headers = {}
        while True:
            line = read_until(b"\r\n")
            if line == b"":
                break
            name, _, value = line.decode("iso-8859-1").partition(":")
            headers.setdefault(name.strip().lower(), []).append(value.strip())

        sid = None
        for header in headers.get("set-cookie") or []:
            jar = SimpleCookie()
            try:
                jar.load(header)
            except Exception:
                continue
            if "cp_session" in jar:
                sid = jar["cp_session"].value

        if status != 200:
            return {"transport_error": "HTTP " + str(status)}, sid

        te = (headers.get("transfer-encoding") or [""])[0].lower()
        if "chunked" in te:
            chunks = bytearray()
            while True:
                size_line = read_until(b"\r\n").split(b";", 1)[0]
                try:
                    size = int(size_line.strip(), 16)
                except ValueError:
                    raise _WireProtocol("bad chunk size")
                if size < 0:
                    raise _WireProtocol("negative chunk size")
                if size == 0:
                    read_until(b"\r\n")  # trailer section terminator
                    break
                if len(chunks) + size > MAX_WIRE_BYTES:
                    raise _WireTooLarge()
                chunks += read_exact(size)
                read_until(b"\r\n")
            body = bytes(chunks)
        elif headers.get("content-length"):
            try:
                length = int(headers["content-length"][0])
            except ValueError:
                raise _WireProtocol("bad content-length")
            if length < 0:
                raise _WireProtocol("negative content-length")
            if length > MAX_WIRE_BYTES:
                raise _WireTooLarge()
            body = read_exact(length)
        else:
            # Close-delimited: read until the peer closes (we asked for
            # Connection: close). recv_some keeps the deadline and cap.
            while recv_some():
                pass
            body = bytes(buf)

        try:
            return json.loads(body.decode("utf-8")), sid
        except Exception:
            return {"transport_error": "malformed JSON body"}, None
    except _ReadTimeout:
        return {"read_timeout": True}, None
    except _WireTooLarge:
        return {"transport_error": "response too large"}, None
    except _WireProtocol as exc:
        return {"transport_error": str(exc)}, None
    except Exception as exc:
        return {"transport_error": str(exc)}, None
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


class _ReadBudget:
    """Hard per-run counter of loopback reads, bootstrap included.

    The analytic bound (1 + 2 * settle rounds) assumes the Verifier's
    pacing; this counter makes the bound structural: once exhausted, reads
    become transport errors and the run settles nothing - UNKNOWN.
    """

    def __init__(self, max_reads: int):
        self.remaining = max_reads

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def _make_reader(self_base: str, cookie: str, run_deadline_ms: float,
                 budget: _ReadBudget):
    """A core.verify reader: two independent loopback reads of GET /api/cart.

    The pair shares one settle budget, exactly like the DOM+API reader in
    harness/run_task.py; each read is additionally clamped to the run's
    absolute deadline and counted against the run's hard read budget.
    """
    url = f"{self_base}/api/cart"

    def reader(timeout_ms: int):
        start = _now_ms()
        if not budget.take():
            first = {"transport_error": "read budget exhausted"}
        else:
            first, _sid = _http_get_json(url, cookie, min(start + timeout_ms, run_deadline_ms))
        remaining = timeout_ms - int(_now_ms() - start)
        if remaining <= 0:
            return first, {"read_timeout": True}
        if not budget.take():
            second = {"transport_error": "read budget exhausted"}
        else:
            second, _sid = _http_get_json(url, cookie, min(_now_ms() + remaining, run_deadline_ms))
        return first, second

    return reader


def _set_session_cookie(page, sid: Optional[str]) -> None:
    """Set cp_session on the response being returned.

    The cookie is set on the returned response object itself, not on an
    injected Response parameter: in this FastAPI version, cookies set on an
    injected parameter do not reach TemplateResponse replies (the store's own
    page routes have the same quirk; only its JSON routes merge them).
    """
    if sid:
        # Same attributes the store itself uses; the Secure attribute is added
        # by hostapp.security.SecureCookieMiddleware on HTTPS requests.
        page.set_cookie("cp_session", sid, httponly=True, samesite="lax")


def _verdict_page(request: Request, overall: str, rows: list, settle_status: str,
                  sample_count: int, view: Optional[dict], sid: Optional[str]):
    page = templates.TemplateResponse(
        request,
        "verdict.html",
        {
            "task": _DEMO_TASK,
            "overall": overall,
            "rows": rows,
            "settle_status": settle_status,
            "sample_count": sample_count,
            "settle_ms": SETTLE_MS,
            "view": view,
            "total_label": _money(view.get("total_cents") if view else None),
        },
    )
    _set_session_cookie(page, sid)
    return page


def _unknown_verdict(request: Request, reason: str, sid: Optional[str]):
    ledger = Ledger.build(_DEMO_TASK["constraints"])
    rows = [{"label": c.label(), "state": UNKNOWN, "reason": reason}
            for c in ledger.constraints]
    return _verdict_page(request, UNKNOWN, rows, "unavailable", 0, None, sid)


@router.get("/verify")
def verify_page(request: Request):
    cookie = request.cookies.get("cp_session")
    cart, sid = _http_get_json(f"{_self_base(request)}/api/cart", cookie,
                               _now_ms() + 3000)
    lines = cart.get("lines") if isinstance(cart, dict) else None
    page = templates.TemplateResponse(
        request,
        "verify.html",
        {
            "task": _DEMO_TASK,
            "cart_ok": isinstance(lines, list),
            "lines": lines or [],
            "total_label": _money(cart.get("total_cents") if isinstance(cart, dict) else None),
            "cart_version": cart.get("version") if isinstance(cart, dict) else None,
        },
    )
    _set_session_cookie(page, sid)
    return page


@router.post("/verify/run")
def verify_run(request: Request):
    try:
        return _verify_run_inner(request)
    except Exception:
        # Never leak internals; an honest UNKNOWN with a generic reason.
        return _unknown_verdict(request,
                                "the server could not complete the verification",
                                None)


def _verify_run_inner(request: Request):
    run_deadline = _now_ms() + TOTAL_BUDGET_MS
    self_base = _self_base(request)
    presented = request.cookies.get("cp_session")

    budget = _ReadBudget(MAX_READS_PER_RUN)
    # Normalize the session exactly once: the store validates the presented
    # cookie or mints a fresh one; every later read in this run uses the
    # returned id. A failed bootstrap settles NOTHING and mints NOTHING extra.
    budget.take()  # the bootstrap read, counted like every other read
    _, sid = _http_get_json(f"{self_base}/api/cart", presented,
                            min(_now_ms() + BOOTSTRAP_MS, run_deadline))
    if sid is None:
        return _unknown_verdict(
            request,
            "your session could not be established with the store just now",
            None)

    settle_budget = max(1, min(SETTLE_MS, int(run_deadline - _now_ms())))
    verifier = Verifier(_make_reader(self_base, sid, run_deadline, budget),
                        settle_ms=settle_budget,
                        interval_ms=SAMPLE_INTERVAL_MS)
    settled = verifier.settle(known_version=None, phase="completion")

    ledger = Ledger.build(_DEMO_TASK["constraints"])
    if settled["status"] != "settled":
        rows = [{"label": c.label(), "state": UNKNOWN,
                 "reason": f"no settled view of your cart ({settled['status']})"}
                for c in ledger.constraints]
        return _verdict_page(request, UNKNOWN, rows, settled["status"],
                             len(settled.get("sample_ids") or []), None, sid)

    view = settled["view"]
    rows = []
    for c in ledger.constraints:
        state, reason = evaluate(c, view)
        rows.append({"label": c.label(), "state": state, "reason": reason})
    if all(r["state"] == CONFIRMED for r in rows):
        overall = "MATCH"
    elif any(r["state"] == CONTRADICTED for r in rows):
        overall = "MISMATCH"
    else:
        overall = UNKNOWN
    return _verdict_page(request, overall, rows, settled["status"],
                         len(settled.get("sample_ids") or []), view, sid)

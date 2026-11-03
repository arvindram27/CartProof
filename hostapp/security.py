"""Security middleware for the hosted slice (all NEW code; no store edits).

Rehearsal-grade hardening for a public demo deployment - this is NOT a claim
that the slice is public-ready; that call belongs to the deploy review.

Proxy policy is explicit:

- local (default): ignore X-Forwarded-For/-Proto/-Host and CF-Connecting-IP.
  The direct socket peer and raw scheme are used. A request that carries any
  forwarding header is not eligible for the loopback rate exemption.
- render: an explicit public-edge mode requiring a configured canonical HTTPS
  origin. External requests must present a single valid CF-Connecting-IP and
  the canonical Host. Leftmost X-Forwarded-For is never used. The provider's
  overwrite/direct-reachability behavior is an assumption to validate later,
  not proven offline.

Four small, pure-ASGI middlewares:

- SecureCookieMiddleware: adds Secure when the effective scheme is HTTPS.
  The scheme comes from the validated policy, never from a client-supplied
  X-Forwarded-Proto.

- BodyLimitMiddleware: enforces the cap on ACTUAL STREAMED BYTES, not the
  declared Content-Length. The body is buffered chunk by chunk and the length
  is checked BEFORE each chunk is appended, so memory never exceeds the cap;
  chunked, missing, or lying headers all count the same. Over-cap gets 413
  without reaching the app; a malformed or negative Content-Length is a 400;
  a body that does not finish within BODY_TIMEOUT_S is a 408.

- OriginCheckMiddleware: for every mutating method (POST/PUT/PATCH/DELETE) on
  ANY route - cart-line mutations included - a PRESENT Origin header must
  match the canonical public origin in render mode, or the request scheme and
  Host in local mode. An ABSENT Origin is not proof of a non-browser client:
  when Origin is absent but the browser's Sec-Fetch-Site signal says
  "cross-site", the request is rejected too. Requests with no Origin and no
  cross-site signal (scripts, curl, same-site browser posts) are allowed:
  CSRF defense-in-depth, not an API gate.

- RateLimitMiddleware: sliding windows with BOUNDED state. Per-IP window for
  all requests and a per-session window for POST /verify/run. Session keys are
  sha256 hashes of the cp_session value. The IP key is the direct peer in
  local mode, or the validated CF-Connecting-IP in render mode; XFF is never a
  fallback. Loopback from 127.0.0.1 without any forwarding/edge header is
  exempt from the per-IP window (the verify handler's own reads come from
  there; local direct traffic is unlimited - stated, not hidden).
- AdmissionMiddleware: concurrency admission for the self-reading surfaces
  (/verify and /verify/run, all methods), enforced here in ASGI BEFORE the
  request reaches the route's thread pool.

The ProxyPolicyMiddleware is outermost and owns effective scheme, client
identity and the narrow direct-loopback self-read exemption.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from http.cookies import SimpleCookie
from ipaddress import ip_address
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

_JSON = [(b"content-type", b"application/json")]
_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")

MAX_KEYS = 4096  # bound on distinct tracked IPs and distinct tracked sessions
PROXY_MODE_ENV = "CARTPROOF_PROXY_MODE"
PUBLIC_ORIGIN_ENV = "CARTPROOF_PUBLIC_ORIGIN"
LEGACY_TRUST_ENV = "CARTPROOF_TRUST_PROXY"
FORWARDING_HEADERS = {
    b"x-forwarded-for", b"x-forwarded-proto", b"x-forwarded-host",
    b"cf-connecting-ip",
}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class PolicyError(ValueError):
    """Invalid or unsafe proxy configuration (fail closed at startup)."""


def _valid_hostname(host: str) -> bool:
    if not host or len(host) > 253 or host.endswith("."):
        return False
    if any(c in host for c in " \t\r\n,/?@\\#"):
        return False
    try:
        ip_address(host)
        return True
    except ValueError:
        pass
    labels = host.split(".")
    return all(_DNS_LABEL.fullmatch(label) for label in labels)


def _normalize_origin(value: object) -> Tuple[str, str]:
    """Return (canonical_origin, host_header) or raise PolicyError."""
    if not isinstance(value, str):
        raise PolicyError("public origin must be a string")
    text = value
    if not text or len(text) > 255:
        raise PolicyError("public origin is empty or too long")
    if text != text.strip():
        raise PolicyError("public origin must not contain surrounding whitespace")
    if any(ord(c) < 0x21 or ord(c) == 0x7F for c in text):
        raise PolicyError("public origin contains whitespace/control characters")
    if "," in text or "\\" in text or "?" in text or "#" in text:
        raise PolicyError("public origin must be a single HTTPS origin")
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise PolicyError("public origin is malformed") from exc
    if parsed.scheme.lower() != "https":
        raise PolicyError("public origin must use https")
    if not parsed.hostname:
        raise PolicyError("public origin must have a host")
    if parsed.username is not None or parsed.password is not None:
        raise PolicyError("public origin must not contain userinfo")
    if parsed.path not in ("", "/"):
        raise PolicyError("public origin must not contain a path beyond /")
    if parsed.query or parsed.fragment:
        raise PolicyError("public origin must not contain query or fragment")
    authority = parsed.netloc
    if authority.endswith(":"):
        raise PolicyError("public origin has an empty port")
    if ":" in parsed.hostname and not authority.startswith("["):
        raise PolicyError("IPv6 public origin host must be bracketed")
    try:
        port = parsed.port
    except ValueError as exc:
        raise PolicyError("public origin has an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise PolicyError("public origin has an invalid port")
    host = parsed.hostname.lower()
    if not _valid_hostname(host):
        raise PolicyError("public origin host is invalid")
    shown = f"[{host}]" if ":" in host else host
    if port is not None and port != 443:
        shown = f"{shown}:{port}"
    return f"https://{shown}", shown


def _is_loopback(value: Optional[str]) -> bool:
    if not value:
        return False
    try:
        return ip_address(value).is_loopback
    except ValueError:
        return False


def _parse_host(raw: Optional[bytes]) -> Optional[Tuple[str, Optional[int]]]:
    if not raw:
        return None
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return None
    if not text or len(text) > 255 or text != text.strip():
        return None
    if any(ord(c) < 0x21 or ord(c) == 0x7F for c in text):
        return None
    if any(c in text for c in " \t\r\n,/?@\\#"):
        return None
    port: Optional[int] = None
    if text.startswith("["):
        end = text.find("]")
        if end < 0:
            return None
        host = text[1:end]
        rest = text[end + 1:]
        if rest:
            if not rest.startswith(":"):
                return None
            port_text = rest[1:]
        else:
            port_text = ""
        try:
            if ip_address(host).version != 6:
                return None
        except ValueError:
            return None
    else:
        if text.count(":") > 1:
            return None
        host, sep, port_text = text.partition(":")
        if not _valid_hostname(host):
            return None
    if port_text:
        if not port_text.isdigit():
            return None
        port = int(port_text)
        if not 1 <= port <= 65535:
            return None
    elif ":" in text:
        return None
    return host.lower(), port


def _format_host(host: Tuple[str, Optional[int]]) -> str:
    name, port = host
    shown = f"[{name}]" if ":" in name else name
    if port is not None and port != 443:
        shown = f"{shown}:{port}"
    return shown


def _host_is_loopback(host: Tuple[str, Optional[int]]) -> bool:
    return host[0] in _LOOPBACK_HOSTS


def _cf_identity(scope) -> Optional[str]:
    values = [v for k, v in scope.get("headers") or []
              if k.lower() == b"cf-connecting-ip"]
    if len(values) != 1:
        return None
    try:
        raw = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    if not raw or len(raw) > 64 or "," in raw or "%" in raw:
        return None
    if any(c.isspace() for c in raw):
        return None
    try:
        return str(ip_address(raw))
    except ValueError:
        return None


@dataclass(frozen=True)
class ProxyPolicy:
    mode: str = "local"
    public_origin: Optional[str] = None
    public_host: Optional[str] = None

    @classmethod
    def local(cls) -> "ProxyPolicy":
        return cls(mode="local")

    @classmethod
    def render(cls, origin: str) -> "ProxyPolicy":
        canonical, host = _normalize_origin(origin)
        return cls(mode="render", public_origin=canonical, public_host=host)

    @classmethod
    def from_environment(cls) -> "ProxyPolicy":
        legacy = os.environ.get(LEGACY_TRUST_ENV)
        if legacy is not None and legacy.strip().lower() not in ("", "0", "false"):
            raise PolicyError(
                "CARTPROOF_TRUST_PROXY is retired; configure "
                "CARTPROOF_PROXY_MODE=render and CARTPROOF_PUBLIC_ORIGIN instead"
            )
        mode = os.environ.get(PROXY_MODE_ENV, "local").strip().lower()
        if mode == "local":
            return cls.local()
        if mode == "render":
            origin = os.environ.get(PUBLIC_ORIGIN_ENV, "")
            if not origin:
                raise PolicyError(
                    "CARTPROOF_PUBLIC_ORIGIN is required when "
                    "CARTPROOF_PROXY_MODE=render"
                )
            return cls.render(origin)
        raise PolicyError("CARTPROOF_PROXY_MODE must be 'local' or 'render'")


def _headers(scope) -> dict:
    return {k.lower(): v for k, v in scope.get("headers") or []}


def _header_values(scope, name: bytes) -> List[bytes]:
    return [v for k, v in scope.get("headers") or [] if k.lower() == name]


def _state(scope) -> dict:
    state = scope.get("state")
    if not isinstance(state, dict):
        state = {}
        scope["state"] = state
    return state


def _has_forwarding(headers: dict) -> bool:
    return any(k in headers for k in FORWARDING_HEADERS)


def _scheme(scope) -> str:
    state = scope.get("state") or {}
    if "cartproof_scheme" in state:
        return state["cartproof_scheme"]
    return scope.get("scheme", "http")


def _client_ip(scope) -> Tuple[str, bool]:
    """(effective identity, direct-loopback exemption)."""
    state = scope.get("state") or {}
    if "cartproof_client_ip" in state:
        return state["cartproof_client_ip"], bool(state.get("cartproof_loopback_direct"))
    client = scope.get("client") or ("", 0)
    ip = client[0] or "unknown"
    return ip, _is_loopback(ip) and not _has_forwarding(_headers(scope))


def _is_internal_selfread(scope, headers: dict) -> bool:
    if scope.get("method") != "GET" or scope.get("path") != "/api/cart":
        return False
    if _has_forwarding(headers):
        return False
    client = scope.get("client") or ("", 0)
    if not _is_loopback(client[0]):
        return False
    host = _parse_host(headers.get(b"host"))
    return host is not None and _host_is_loopback(host)


class ProxyPolicyMiddleware:
    """Resolve effective scheme/client identity once, fail closed in render."""

    def __init__(self, app, policy: Optional[ProxyPolicy] = None):
        self.app = app
        self.policy = policy or ProxyPolicy.local()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = _state(scope)
        headers = _headers(scope)
        client = scope.get("client") or ("", 0)
        state["cartproof_mode"] = self.policy.mode

        host_values = _header_values(scope, b"host")
        if len(host_values) > 1:
            await _reject(send, 400, "bad_host",
                          "request must carry at most one Host header")
            return

        if self.policy.mode == "local":
            state["cartproof_scheme"] = scope.get("scheme", "http")
            state["cartproof_client_ip"] = client[0] or "unknown"
            state["cartproof_loopback_direct"] = (
                not _has_forwarding(headers) and _is_loopback(client[0]))
            state["cartproof_external"] = False
            await self.app(scope, receive, send)
            return

        if _is_internal_selfread(scope, headers):
            state["cartproof_scheme"] = "http"
            state["cartproof_client_ip"] = client[0] or "unknown"
            state["cartproof_loopback_direct"] = True
            state["cartproof_external"] = False
            state["cartproof_self_read"] = True
            await self.app(scope, receive, send)
            return

        host = _parse_host(headers.get(b"host"))
        if host is None or _format_host(host) != self.policy.public_host:
            await _reject(send, 403, "host_mismatch",
                          "request host does not match the configured public origin")
            return
        edge_ip = _cf_identity(scope)
        if edge_ip is None:
            await _reject(send, 403, "edge_identity_unavailable",
                          "a single valid CF-Connecting-IP is required")
            return
        state["cartproof_scheme"] = "https"
        state["cartproof_client_ip"] = edge_ip
        state["cartproof_loopback_direct"] = False
        state["cartproof_external"] = True
        await self.app(scope, receive, send)


async def _reject(send, status: int, code: str, message: str, extra_headers=None) -> None:
    body = json.dumps({"error": {"code": code, "message": message}}).encode("utf-8")
    headers = list(extra_headers or []) + _JSON
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _session_key(scope) -> str:
    raw = _headers(scope).get(b"cookie")
    if not raw:
        return ""
    jar = SimpleCookie()
    try:
        jar.load(raw.decode("latin-1"))
    except Exception:
        return ""
    morsel = jar.get("cp_session")
    if not morsel:
        return ""
    return hashlib.sha256(morsel.value.encode("utf-8", "ignore")).hexdigest()


class SecureCookieMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        secure = _scheme(scope) == "https"

        async def send_wrapper(message):
            if secure and message["type"] == "http.response.start":
                new_headers = []
                for k, v in message.get("headers") or []:
                    if k.lower() == b"set-cookie" and b"secure" not in v.lower():
                        v = v + b"; Secure"
                    new_headers.append((k, v))
                message = dict(message, headers=new_headers)
            await send(message)

        await self.app(scope, receive, send_wrapper)


class BodyLimitMiddleware:
    BODY_TIMEOUT_S = 10.0  # a body that will not finish is abandoned

    def __init__(self, app, max_bytes: int = 65536):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw = _headers(scope).get(b"content-length")
        if raw is not None:
            try:
                declared = int(raw)
            except ValueError:
                await _reject(send, 400, "bad_content_length",
                              "Content-Length header is not an integer")
                return
            if declared < 0:
                await _reject(send, 400, "bad_content_length",
                              "Content-Length header is negative")
                return
        # Enforce on actual streamed bytes, checking the length BEFORE each
        # chunk is appended so memory never exceeds the cap, and bounding the
        # total wait so a trickled body cannot hold the slot open.
        import asyncio

        body = b""
        more = True
        deadline = time.monotonic() + self.BODY_TIMEOUT_S
        while more:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await _reject(send, 408, "body_timeout",
                              "request body did not finish in time")
                return
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except asyncio.TimeoutError:
                await _reject(send, 408, "body_timeout",
                              "request body did not finish in time")
                return
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                await _reject(send, 413, "body_too_large",
                              f"request body exceeds {self.max_bytes} bytes")
                return
            body += chunk
            more = message.get("more_body", False)

        sent = False

        async def replay_receive():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_receive, send)


class OriginCheckMiddleware:
    MUTATING = {"POST", "PUT", "PATCH", "DELETE"}

    def __init__(self, app, policy: Optional[ProxyPolicy] = None):
        self.app = app
        self.policy = policy or ProxyPolicy.local()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in self.MUTATING:
            await self.app(scope, receive, send)
            return
        headers = _headers(scope)
        origin_values = _header_values(scope, b"origin")
        if len(origin_values) > 1:
            await _reject(send, 403, "origin_mismatch",
                          "duplicate Origin header rejected")
            return
        origin = origin_values[0] if origin_values else None
        if origin is None:
            # No Origin is not proof of a non-browser client: modern browsers
            # always send Sec-Fetch-Site, and a cross-site mutation without an
            # Origin is exactly what a forged cross-site form looks like.
            sfs = headers.get(b"sec-fetch-site")
            if sfs is not None and sfs.decode("latin-1").strip().lower() == "cross-site":
                await _reject(send, 403, "origin_mismatch",
                              "cross-site mutation rejected")
                return
            await self.app(scope, receive, send)
            return
        try:
            text = origin.decode("latin-1")
        except Exception:
            text = ""
        if self.policy.mode == "render":
            try:
                canonical, _ = _normalize_origin(text)
            except PolicyError:
                await _reject(send, 403, "origin_mismatch",
                              "cross-origin mutation rejected")
                return
            if canonical != self.policy.public_origin:
                await _reject(send, 403, "origin_mismatch",
                              "cross-origin mutation rejected")
                return
            await self.app(scope, receive, send)
            return

        from urllib.parse import urlsplit

        try:
            parsed = urlsplit(text)
        except ValueError:
            await _reject(send, 403, "origin_mismatch",
                          "malformed Origin rejected")
            return
        expected_scheme = _scheme(scope)
        host = headers.get(b"host", b"").decode("latin-1")
        if parsed.scheme.lower() != expected_scheme or parsed.netloc != host:
            await _reject(send, 403, "origin_mismatch", "cross-origin mutation rejected")
            return
        await self.app(scope, receive, send)


class RateLimitMiddleware:
    def __init__(self, app, requests_per_minute: int = 240,
                 verify_runs_per_window: int = 10, verify_window_seconds: int = 600,
                 max_keys: int = MAX_KEYS):
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.verify_runs_per_window = verify_runs_per_window
        self.verify_window_seconds = verify_window_seconds
        self.max_keys = max_keys
        self._ip_hits: Dict[str, List[float]] = {}
        self._verify_hits: Dict[str, List[float]] = {}

    @staticmethod
    def _prune(hits: List[float], window: float, now: float) -> List[float]:
        return [t for t in hits if now - t < window]

    def _bounded_insert(self, table: Dict[str, List[float]], key: str, now: float,
                        window: float) -> List[float]:
        hits = table.get(key)
        if hits is None:
            if len(table) >= self.max_keys:
                oldest = min(table, key=lambda k: table[k][-1])
                del table[oldest]
            hits = []
            table[key] = hits
        return hits

    def _prune_maps(self, now: float) -> None:
        for key in list(self._ip_hits):
            self._ip_hits[key] = self._prune(self._ip_hits[key], 60.0, now)
            if not self._ip_hits[key]:
                del self._ip_hits[key]
        for key in list(self._verify_hits):
            self._verify_hits[key] = self._prune(
                self._verify_hits[key], float(self.verify_window_seconds), now)
            if not self._verify_hits[key]:
                del self._verify_hits[key]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        now = time.monotonic()
        self._prune_maps(now)
        ip, loopback_direct = _client_ip(scope)
        path = scope.get("path", "")
        method = scope.get("method", "")

        if not loopback_direct:
            hits = self._bounded_insert(self._ip_hits, ip, now, 60.0)
            if len(hits) >= self.requests_per_minute:
                await _reject(send, 429, "rate_limited",
                              "too many requests from this address; try again shortly",
                              extra_headers=[(b"retry-after", b"60")])
                return
            hits.append(now)

        if path == "/verify/run" and method == "POST":
            sid_key = _session_key(scope) or f"anon:{ip}"
            hits = self._bounded_insert(self._verify_hits, sid_key, now,
                                        float(self.verify_window_seconds))
            if len(hits) >= self.verify_runs_per_window:
                await _reject(send, 429, "verify_rate_limited",
                              "too many verification runs for this session; "
                              "try again in a few minutes",
                              extra_headers=[(b"retry-after", b"600")])
                return
            hits.append(now)

        await self.app(scope, receive, send)


class AdmissionMiddleware:
    """Concurrency admission for the self-reading surfaces, in ASGI.

    /verify and /verify/run both make blocking loopback reads from thread
    pool workers; unbounded concurrency there can occupy every pool thread
    and starve the loopback reads themselves. The slot is taken BEFORE the
    request reaches any route handler and released when the response (or the
    error path) completes.
    """

    def __init__(self, app, paths=("/verify", "/verify/run"), max_concurrent: int = 4):
        import threading

        self.app = app
        self.paths = set(paths)
        self.semaphore = threading.BoundedSemaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in self.paths:
            await self.app(scope, receive, send)
            return
        if not self.semaphore.acquire(blocking=False):
            await _reject(send, 429, "verification_busy",
                          "too many verification requests in flight; try again shortly",
                          extra_headers=[(b"retry-after", b"5")])
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.semaphore.release()

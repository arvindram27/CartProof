"""WB2 live Shopify backend: token-authenticated construction of the
reviewed ShopifyStorefrontBackend.

NEW FILE for the WB2 live run - store/backends.py is NOT modified and the
offline construction guard there stays in force for it. What this subclass
changes: construction (token header from the environment, explicit live
opt-in, httpx event hooks for per-request metadata), a hard request-budget
gate in _gql, an SKU->variant mapping installer, redacted cart-ref
tracking, and the R2 _gql override documented above.

R2 note: _gql itself is NOT inherited - this file replaces it with a
streaming override (same error categories, plus bounded wire read,
bounded decompression and pre-retention capture). All cart logic,
canonicalization and redaction remain inherited byte-identical from the
reviewed parent.

Deliberate guards carried over in spirit:
- A real (non-mock) transport requires the explicit env opt-in
  CARTPROOF_SHOPIFY_LIVE=1 - accidental live construction stays impossible.
- The token is read from the environment (never an argument in logs, never
  a file inside the repo, never printed).
- A hard request budget (MAX_REQUESTS) is enforced BEFORE send; exceeding
  it aborts instead of making request 21.
- Per-request evidence (timestamp, status, Shopify X-Request-ID, latency)
  is captured via httpx event hooks WITHOUT touching request/response
  bodies and WITHOUT logging headers (the token lives in a header).
"""
from __future__ import annotations

import json
import os
import re
import time
import zlib
from pathlib import Path
from typing import Any, Optional

import httpx

from store.backends import ShopifyBackendError, ShopifyStorefrontBackend

TOKEN_ENV = "CARTPROOF_SHOPIFY_STOREFRONT_TOKEN"
TOKEN_FILE_ENV = "CARTPROOF_SHOPIFY_STOREFRONT_TOKEN_FILE"


def resolve_token() -> str:
    """Token from TOKEN_ENV, or from the file named by TOKEN_FILE_ENV.

    The file route lets the approved Python process read the restricted
    local file directly: no shell or agent ever reads or echoes the
    content. The token is returned for in-process use only - it is never
    logged, printed, or written to evidence."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    if token:
        return token
    path = os.environ.get(TOKEN_FILE_ENV, "").strip()
    if path:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return ""
LIVE_OPT_IN_ENV = "CARTPROOF_SHOPIFY_LIVE"
MAX_REQUESTS = 20
MAX_WIRE_BYTES = 65536        # raw (still-compressed) bytes read cap
MAX_DECODED_BYTES = 262144    # post-decompression byte cap (bomb guard)
MAX_CAPTURE_DEPTH = 16        # decoded-structure depth; deeper => refusal
MAX_RETAINED_MESSAGES = 3     # per-response retention cap
MAX_MESSAGE_CHARS = 120       # truncation applied AFTER the full-length scan
MAX_REQUEST_ID_CHARS = 64     # X-Request-ID retention cap (allowlisted)

# Fixed marker retained INSTEAD of any error detail when the pre-retention
# scan of the raw decoded text or of ANY key/string in the decoded
# structure hits a guard, or the structure exceeds MAX_CAPTURE_DEPTH.
# A refusal retains no partial detail at all.
CAPTURE_REFUSAL = "capture_refused_unsafe_or_unbounded"

# X-Request-ID is a UUID-shaped correlation id; anything longer, outside
# this charset, or matching a guard is withheld, never retained raw.
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9-]{1,64}\Z")

# Retained extension codes are charset-allowlisted (Shopify uses
# SCREAMING_SNAKE); anything else is dropped, not retained.
ERROR_CODE_RE = re.compile(r"[A-Za-z0-9_]{1,40}\Z")

# Leak guards, single-sourced here: the live runner imports this list
# for its evidence-tree scan so the capture layer and the archive scan
# share one definition. A token value can only enter evidence if a server
# response first echoed it. Character classes are deliberately tight:
# real capability values are opaque alphanumerics, so the pattern
# LITERALS in this source file (and in the archived payload snapshot)
# never match themselves.
GUARD_PATTERNS = [
    re.compile(r"gid://shopify/Cart/[A-Za-z0-9+/=_-]{2,}", re.I),
    re.compile(r"\?key=[A-Za-z0-9_-]{2,}", re.I),
    re.compile(r"shpat_[A-Za-z0-9]{6,}", re.I),
    re.compile(r"Bearer [A-Za-z0-9._-]+", re.I),
    re.compile(r"X-Shopify-Storefront-Access-Token\s*[:=]\s*\S+", re.I),
]


class RequestBudgetExceeded(RuntimeError):
    """Raised before any request that would exceed the approved budget."""


class ShopifyLiveBackend(ShopifyStorefrontBackend):
    """Live-authenticated Shopify backend for the approved WB2 run only."""

    def __init__(self, shop: str, mapping_path: str,
                 token: Optional[str] = None, api_version: str = "2026-07",
                 transport: Optional[httpx.BaseTransport] = None):
        if token is None:
            token = resolve_token()
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("token_missing")
        if transport is None and os.environ.get(LIVE_OPT_IN_ENV) != "1":
            raise RuntimeError(
                "live_opt_in_missing: set CARTPROOF_SHOPIFY_LIVE=1 for a "
                "real transport; offline tests must pass a mock transport")
        # NOTE: parent __init__ is intentionally NOT called (its offline
        # guard rejects real transports); the same attribute layout is
        # established here so every inherited method behaves identically.
        self.shop = shop
        self.api_version = api_version
        self.request_log: list[dict[str, Any]] = []
        self._request_count = 0
        self._pending_start: Optional[float] = None
        self._max_requests = MAX_REQUESTS
        # Scan-only: used to leak-check response candidates before
        # retention; never logged, printed, or written to evidence.
        self._token_value = token.strip()
        self._passed_transport = transport  # for accounting-safe rebuilds
        kwargs: dict[str, Any] = {}
        if transport is not None:
            kwargs["transport"] = transport
        self.client = httpx.Client(
            base_url=f"https://{shop}/api/{api_version}/",
            headers={"X-Shopify-Storefront-Access-Token": token.strip()},
            timeout=15.0, follow_redirects=False, trust_env=False,
            event_hooks={"request": [self._on_request],
                         "response": [self._on_response]},
            **kwargs)
        mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        self._sku_to_gid = dict(mapping.get("variants", {}))
        self._gid_to_sku = {gid: sku for sku, gid in self._sku_to_gid.items()}
        self._cart_id: Optional[str] = None
        self._cart_key: Optional[str] = None
        self._cart_ref: Optional[str] = None
        self.carts_created: list[str] = []  # redacted 12-char refs only

    # ----------------------------------------------------------- #
    # Request accounting + evidence hooks (no bodies, no headers). #
    # ----------------------------------------------------------- #
    def _on_request(self, request: httpx.Request) -> None:
        if self._request_count >= self._max_requests:
            raise RequestBudgetExceeded(
                f"request budget {self._max_requests} reached; aborting "
                "before another send")
        self._request_count += 1
        self._pending_start = time.perf_counter()

    def _on_response(self, response: httpx.Response) -> None:
        elapsed_ms = None
        if self._pending_start is not None:
            elapsed_ms = round(
                (time.perf_counter() - self._pending_start) * 1000, 1)
            self._pending_start = None
        x_request_id = response.headers.get("X-Request-ID")
        if x_request_id is not None and not (
                REQUEST_ID_RE.match(x_request_id)
                and not self._scan_unsafe(x_request_id)):
            x_request_id = "<withheld>"  # fixed marker; never retained raw
        entry = {"seq": self._request_count,
                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "method": response.request.method,
                 "path": response.request.url.path[:128],
                 "status": response.status_code,
                 "x_request_id": x_request_id,
                 "elapsed_ms": elapsed_ms}
        entry = self.redact(entry)
        self.request_log.append(entry)

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def request_cap(self) -> int:
        """The enforced budget - recorded in probe evidence."""
        return self._max_requests

    def install_mapping(self, mapping: dict) -> None:
        """Live path only: install the SKU->GID map built from the catalog
        read WITHOUT reconstructing the client - request accounting and the
        cart identity stay intact. Same attribute layout as construction."""
        self._sku_to_gid = dict(mapping.get("variants", {}))
        self._gid_to_sku = {gid: sku
                            for sku, gid in self._sku_to_gid.items()}

    def _set_cart_id(self, cart_id: str) -> None:
        super()._set_cart_id(cart_id)
        # record only the redacted ref for the residual-cart report
        if self._cart_ref is not None                 and self._cart_ref not in self.carts_created:
            self.carts_created.append(self._cart_ref)

    # --------------------------------------------------------------- #
    # R2 diagnostic capture: same control flow and error categories as  #
    # the reviewed parent's _gql, plus bounded wire reads and           #
    # pre-retention error capture. store/backends.py is NOT modified.   #
    # --------------------------------------------------------------- #
    def set_request_cap(self, cap: int) -> None:
        """Lower the hard request budget (never raise it). The approved
        R2 probe uses this to cap the run at exactly one request."""
        self._max_requests = min(self._max_requests, cap)

    def _scan_unsafe(self, text: str) -> bool:
        """True when text carries the exact token value or any guard
        pattern (cart ids, capability keys, token-shaped strings). Runs
        on FULL-LENGTH untruncated candidates BEFORE any retention."""
        if self._token_value and self._token_value in text:
            return True
        return any(pattern.search(text) for pattern in GUARD_PATTERNS)

    def _read_wire_bounded(self, response: httpx.Response) -> tuple:
        """Raw, still-compressed bytes, capped at MAX_WIRE_BYTES. Runs on
        a STREAMING response (client.send(stream=True)); iter_raw never
        decodes content-encoding."""
        if response.is_stream_consumed:
            # In-memory transports (the offline fake, MockTransport)
            # deliver pre-buffered bodies that httpx ALREADY decoded
            # (Content-Encoding applied at construction); the decoded cap
            # still applies to what we accept, and the caller treats the
            # bytes as identity. A REAL transport (stream=True) never
            # takes this branch - its raw bytes are capped mid-stream.
            wire = response.content
            if len(wire) > MAX_DECODED_BYTES:
                raise ShopifyBackendError("body_too_large")
            return wire, True
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_raw(chunk_size=16384):
            total += len(chunk)
            if total > MAX_WIRE_BYTES:
                raise ShopifyBackendError("body_too_large")
            chunks.append(chunk)
        return b"".join(chunks), False

    def _decode_bounded(self, response: httpx.Response,
                        wire: bytes, already_decoded: bool) -> str:
        """Explicit bounded decompression of the ACTUAL Content-Encoding
        (identity/gzip/deflate), capped at MAX_DECODED_BYTES of output,
        then STRICT UTF-8. Fixed safe categories only: body_too_large,
        unsupported_encoding, decode_failed, invalid_json.
        already_decoded marks the
        in-memory-transport branch where httpx applied the encoding when
        it buffered the body."""
        encoding = ("identity" if already_decoded else
                    (response.headers.get("Content-Encoding")
                     or "identity").strip().lower())
        if encoding in ("", "identity"):
            decoded = wire
        elif encoding in ("gzip", "deflate"):
            wbits = (16 + zlib.MAX_WBITS if encoding == "gzip"
                     else zlib.MAX_WBITS)
            try:
                decomp = zlib.decompressobj(wbits)
                decoded = decomp.decompress(wire, MAX_DECODED_BYTES + 1)
            except zlib.error:
                raise ShopifyBackendError("decode_failed") from None
            # NO flush(): output is bounded by max_length alone. Require
            # a complete, single, exactly-finished stream: refuse output
            # past the cap, a truncated stream (eof False), trailing
            # garbage or concatenated members (unused_data).
            if len(decoded) > MAX_DECODED_BYTES or decomp.unconsumed_tail:
                raise ShopifyBackendError("body_too_large")
            if not decomp.eof or decomp.unused_data:
                raise ShopifyBackendError("decode_failed")
        else:
            raise ShopifyBackendError("unsupported_encoding")
        try:
            return decoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise ShopifyBackendError("invalid_json") from None

    def _structure_unsafe(self, node: Any, depth: int = 0) -> bool:
        """True if ANY key or string anywhere in the decoded structure
        fails the guard scan, or the structure exceeds MAX_CAPTURE_DEPTH
        (depth refusal). Scans whole strings - never truncated copies."""
        if depth > MAX_CAPTURE_DEPTH:
            return True
        if isinstance(node, str):
            return self._scan_unsafe(node)
        if isinstance(node, list):
            return any(self._structure_unsafe(v, depth + 1) for v in node)
        if isinstance(node, dict):
            return any(self._structure_unsafe(k, depth + 1)
                       or self._structure_unsafe(v, depth + 1)
                       for k, v in node.items())
        return False

    def _capture_graphql_errors(self, raw_text: str, body: Any) -> None:
        """Pre-retention gate for the WHOLE response: the raw decoded text
        AND every key/string in the decoded structure are scanned
        full-length BEFORE anything is retained. ANY hit (or excess
        depth) retains only CAPTURE_REFUSAL - no partial detail. Only a
        fully clean response yields bounded details: allowlisted
        extension codes and messages, redacted and truncated AFTER the
        scan, at most MAX_RETAINED_MESSAGES, plus numeric counts."""
        if self._scan_unsafe(raw_text) or self._structure_unsafe(body):
            capture: dict[str, Any] = {"graphql_capture": CAPTURE_REFUSAL}
        else:
            errors = body.get("errors") if isinstance(body, dict) else None
            items = [e for e in errors if isinstance(e, dict)] \
                if isinstance(errors, list) else []
            codes: list[str] = []
            messages: list[str] = []
            for item in items:
                extensions = item.get("extensions")
                if isinstance(extensions, dict):
                    code = extensions.get("code")
                    if (isinstance(code, str) and ERROR_CODE_RE.match(code)
                            and len(codes) < 4):
                        codes.append(self.redact(code))
                message = item.get("message")
                if isinstance(message, str) \
                        and len(messages) < MAX_RETAINED_MESSAGES:
                    messages.append(self.redact(message)[:MAX_MESSAGE_CHARS])
            capture = {"graphql_error_count": len(items),
                       "graphql_error_codes": codes,
                       "graphql_error_messages": messages,
                       "graphql_error_dropped":
                           len(items) - len(messages)}
        for entry in reversed(self.request_log):
            if entry.get("seq") == self._request_count:
                entry.update(capture)
                break
        else:
            capture["seq"] = self._request_count
            self.request_log.append(capture)

    def _gql(self, query: str, variables: dict) -> dict:
        # Budget enforcement BEFORE send (unchanged R1 behavior): an
        # exception raised from the request hook would be wrapped as a
        # transport error by the flow below, which callers degrade to
        # UNKNOWN - the cap must abort loudly instead.
        if self._request_count >= self._max_requests:
            raise RequestBudgetExceeded(
                f"request budget {self._max_requests} reached; aborting "
                "before another send")
        # Same control flow and error categories as the reviewed parent's
        # _gql (store/backends.py): transport / http:<status> /
        # invalid_json / no_data / graphql_errors - PLUS the fixed safe
        # categories body_too_large and unsupported_encoding from the
        # bounded wire read and bounded decode below. Differences from
        # the parent, by design: the request is STREAMED and the raw
        # (still-compressed) body is capped at MAX_WIRE_BYTES BEFORE any
        # decoding; decompression is explicit and capped at
        # MAX_DECODED_BYTES of output; UTF-8 decoding is strict; and a
        # parseable graphql_errors response is captured into the request
        # log BEFORE the raise, gated by the whole-response scan.
        try:
            request = self.client.build_request(
                "POST", "graphql.json",
                json={"query": query, "variables": variables})
            response = self.client.send(request, stream=True)
        except Exception as exc:
            if not isinstance(exc, RequestBudgetExceeded):
                # No response hook fired: record the attempt with a FIXED
                # outcome marker - request count + error type only, no
                # HTTP claims (no status, no headers existed).
                self.request_log.append({
                    "seq": self._request_count,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime()),
                    "outcome": "transport_failed_no_response",
                    "error_type": type(exc).__name__})
            raise ShopifyBackendError(
                f"transport:{type(exc).__name__}") from exc
        try:
            if response.status_code >= 400:
                raise ShopifyBackendError(f"http:{response.status_code}")
            wire, already_decoded = self._read_wire_bounded(response)
            raw_text = self._decode_bounded(
                response, wire, already_decoded)
            try:
                body = json.loads(raw_text)
            except Exception:
                raise ShopifyBackendError("invalid_json")
            if not isinstance(body, dict):
                raise ShopifyBackendError("invalid_json")
            errors = body.get("errors")
            if errors:
                self._capture_graphql_errors(raw_text, body)
                raise ShopifyBackendError("graphql_errors")
            data = body.get("data")
            if not isinstance(data, dict):
                raise ShopifyBackendError("no_data")
            return data
        finally:
            response.close()

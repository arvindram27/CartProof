"""Bounded live compatibility probe for the CartProof dev store (P42).

OFFLINE BY DEFAULT. Live construction requires allow_live=True AND the CLI
requires --allow-live; without both, this module cannot speak to the
network. (The flag is a mechanical gate, not proof of approval: Stage 1
live use is a separately approved user step.) The reviewed adapter
(store/backends.py, pinned in the digest closure) is NOT modified: this
module subclasses it and inherits every read, mutation, canonicalization
and redaction rule unchanged.

Bounds enforced in code (both stages):
- Bound to exactly EXPECTED_SHOP over HTTPS; the API version is pinned;
  redirects and any 3xx response are REJECTED before parsing, so a
  password page or redirect is never read as data; proxies/env are not
  trusted.
- FAIL-CLOSED STICKY GUARD: after ANY transport/HTTP/GraphQL failure or
  the request cap, the backend refuses every further send, from any
  caller. No retries: a failure is raised exactly once.
- A hard request cap counts every real request, including the cart create
  and every read. The frozen plan needs at most 15 requests including
  cleanup; the CLI refuses budgets below 16 (plan + reserve), and the
  cleanup reserve is re-checked dynamically before cleanup starts.
- Sequential requests only. Writes only through the reviewed adapter's
  three cart mutations, only on the cart this run created. A failed
  mutation is checked BEFORE any follow-up read: no read, no cleanup, no
  request of any kind after a stop.
- Evidence writes are fail-closed: serialized output is redacted through
  the reviewed redact(), then scanned GENERICALLY for cart-capability
  shapes (any gid://shopify/Cart/ id, any ?key= marker) plus the exact
  registered id/key, before anything touches disk or stdout. This covers
  pre-registration output, error payloads and tool arguments.
- Never requests, follows or prints checkoutUrl; never sends buyerIdentity
  or any customer data; no Admin API; no product/inventory/customer writes.
- Cleanup (cart back to empty) runs ONLY on the normal path and ONLY
  within the remaining request budget. After any stop, no further requests
  are issued; the uncleaned cart's truncated sha256 cart_ref and last
  known lines are recorded with verdict UNKNOWN. Shopify deletes completed
  carts and expires abandoned carts automatically; this probe does not
  pin the exact TTL.
- discover runs FIRST: one read-only products query is safer than any
  write. If the store answers a lock/password/HTTP failure, the run stops
  before any cart exists.

- ERROR CAPTURE (P47): on any HTTP >=400 response the probe records a
  bounded, allowlisted diagnostic: the status, ONE shape-checked request-id
  header (else withheld), and, from a JSON body of at most ERROR_BODY_CAP
  bytes, up to ERROR_MAX_ENTRIES entries of {message capped at
  ERROR_MESSAGE_CAP, code and extensions.code matching a strict character
  policy (else withheld)}. Raw bodies, raw headers, cookies and every
  other field are never persisted. Content-encoded bodies are never
  decompressed or read (no bounded-decompression claim): the diagnostic
  keeps only status and request id for them. The raw received text AND
  every DECODED string in a parsed body - extracted or not, including
  JSON-escaped forms - are scanned for a broad set of secret shapes and
  cart-capability shapes BEFORE any truncation; any hit replaces the
  ENTIRE diagnostic with a refusal marker, and any capture error degrades
  to a marker, never to raw bytes. No extra request is made: capture reads
  the response the adapter already received, then returns an equivalent
  fully-buffered empty-body response so the client auto-read always
  succeeds and the adapter still raises http:<status> exactly as before.
This file is fully testable offline via an injected httpx.MockTransport
(test seam only).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from harness.evaluate import compare_lines
from store.backends import ShopifyBackendError, ShopifyStorefrontBackend

EXPECTED_SHOP = "cartproof.myshopify.com"
PINNED_API_VERSION = "2026-07"
DEFAULT_MAX_REQUESTS = 30
# Worst-case plan: create+read, 3x(add+read), set+read, delete+read,
# up to 2 cleanup deletes, final read = 15. The CLI refuses smaller
# budgets than plan + 1 reserve.
PROBE_PLANNED_REQUESTS = 15
PROBE_MIN_BUDGET = PROBE_PLANNED_REQUESTS + 1

_VARIANT_GID_SHAPE = re.compile(
    r"^gid://shopify/ProductVariant/[A-Za-z0-9_-]+$")
_CART_ID_SHAPE = re.compile(r"gid://shopify/Cart/[^\s\"'\\]+")
_KEY_MARKER_SHAPE = re.compile(r"\?key=")

# Error-capture bounds (P47): structured, allowlisted, length-capped only.
ERROR_BODY_CAP = 65536
ERROR_MAX_ENTRIES = 5
ERROR_MESSAGE_CAP = 200
ERROR_CODE_CAP = 100
_REQUEST_ID_HEADERS = ("x-request-id", "x-shopify-request-id")
_REQUEST_ID_SHAPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
# Broad secret guard (P47): any of these ANYWHERE in an error response -
# including fields we never extract, and BEFORE any truncation - refuses
# the whole diagnostic. Deliberately over-broad: a refused diagnostic is
# a safe outcome, a leaked token is not.
_SECRET_SHAPE = re.compile("|".join([
    r"shp(?:at|ca|pa|ss|ua)_",                      # Shopify access tokens
    r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{10,}",    # GitHub tokens
    r"github_pat_[A-Za-z0-9_]{10,}",
    r"sk-[A-Za-z0-9_-]{10,}",                       # OpenAI-style keys
    r"AKIA[0-9A-Z]{16}",                            # AWS access key id
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",  # JWT
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i:bearer\s+[A-Za-z0-9._~+/-]{10,})",
    r"(?i:(?:password|passwd|pwd|secret|token|api[_-]?key"
    r"|access[_-]?token|auth[_-]?token|session[_-]?token|authorization"
    r"|cookie|x-shopify-access-token)\s*[:=])",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email
]))
# Codes and request ids are machine tokens, never free text: a strict
# character policy, withheld (not echoed) on any mismatch.
_CODE_SHAPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}$")


class ProbeStop(Exception):
    """Internal stop. After this is raised, no further requests are made."""


class _StatusRejected(Exception):
    """A redirect or other 3xx response: never parsed as data."""

    def __init__(self, status_code: int):
        super().__init__(f"status:{status_code}")
        self.status_code = status_code


class _StatusGuardTransport(httpx.BaseTransport):
    """Reject any 3xx response before the reviewed parsing sees it."""

    def __init__(self, inner: httpx.BaseTransport):
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        if 300 <= response.status_code < 400:
            raise _StatusRejected(response.status_code)
        return response


class _ErrorCaptureTransport(httpx.BaseTransport):
    """Bounded diagnostic capture for >=400 responses (P47).

    The reviewed adapter raises http:<status> and discards the response.
    This transport sits under it and extracts a small structured
    diagnostic from the response the adapter already received - no extra
    request, no retry, no adapter code change. Three hard rules:

    1. ADAPTER BEHAVIOR IS PRESERVED EXACTLY. After capture, the
       original response is closed and an equivalent fully-buffered
       response (same status and headers, empty body - the adapter
       never reads the body on >=400) is returned, so the client's
       auto-read always succeeds and the adapter still raises
       http:<status>. Consuming a live stream directly would surface
       StreamConsumed as a transport error instead; the rebuild prevents
       that.
    2. ONLY ALLOWLISTED FIELDS, STRICT POLICIES. http_status; ONE
       request-id header matching _REQUEST_ID_SHAPE (else withheld);
       from a JSON body of at most ERROR_BODY_CAP bytes, up to
       ERROR_MAX_ENTRIES entries of {message capped at
       ERROR_MESSAGE_CAP, code and extensions.code matching
       _CODE_SHAPE (else withheld)}. Everything else is dropped.
    3. FAIL-CLOSED. The raw body text (up to the cap) and every
       allowlisted field are scanned BEFORE any truncation; any secret
       or cart-capability shape anywhere replaces the WHOLE diagnostic
       with a refusal marker. Any capture error degrades to a marker,
       never to raw bytes.
    """

    def __init__(self, inner: httpx.BaseTransport):
        self._inner = inner
        self.last_diagnostic: Optional[dict] = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        if response.status_code < 400:
            return response
        self.last_diagnostic, body = self._safe_capture(response)
        response.close()
        # Return an equivalent buffered response: identical status and
        # headers, empty body. The reviewed adapter raises on status
        # without ever reading the body, so this is behavior-identical
        # for the reviewed path while letting the client's auto-read
        # succeed on an already-consumed stream.
        return httpx.Response(response.status_code,
                              headers=response.headers,
                              content=body, request=request)

    def _safe_capture(self, response: httpx.Response):
        """Returns (diagnostic, body_for_buffered_response). The body is
        always empty: nothing downstream may read error payloads."""
        try:
            return self._capture(response), b""
        except Exception as exc:
            return {"capture_failed": type(exc).__name__}, b""

    def _capture(self, response: httpx.Response) -> dict:
        diag: dict[str, Any] = {"http_status": response.status_code}
        request_id: Optional[str] = None
        for name in _REQUEST_ID_HEADERS:
            if name in response.headers:
                request_id = response.headers[name]
                break
        if request_id is not None:
            if not _REQUEST_ID_SHAPE.match(request_id) \
                    or _SECRET_SHAPE.search(request_id):
                diag["request_id"] = "<withheld>"
            else:
                diag["request_id"] = request_id
        body, note = self._read_bounded(response)
        if body is None:
            diag["body"] = {"skipped": note}
            return self._guard_diag(diag)
        diag["body_length"] = len(body)
        text = body.decode("utf-8", errors="replace")
        # Scan the WHOLE received text BEFORE parsing or truncation:
        # secrets or capability shapes in unextracted fields, and tokens
        # that would straddle a truncation boundary, refuse the entire
        # diagnostic.
        if _SECRET_SHAPE.search(text) or _CART_ID_SHAPE.search(text) \
                or _KEY_MARKER_SHAPE.search(text):
            return {"capture_refused": "guard_hit"}
        try:
            parsed = json.loads(text)
        except Exception:
            diag["body_json"] = False
            return self._guard_diag(diag)
        # The raw scan above misses JSON-escaped secrets (\uXXXX). Walk
        # EVERY decoded string in the parsed body - including fields we
        # never extract and dict keys - BEFORE any extraction or
        # truncation. A partial decoded token must never reach disk.
        if self._decoded_scan_hit(parsed):
            return {"capture_refused": "guard_hit"}
        if not isinstance(parsed, dict):
            diag["body_json"] = False
            return self._guard_diag(diag)
        errors = parsed.get("errors")
        if not isinstance(errors, list):
            diag["errors_present"] = False
            return self._guard_diag(diag)
        entries: list[dict[str, Any]] = []
        for entry in errors[:ERROR_MAX_ENTRIES]:
            if not isinstance(entry, dict):
                continue
            item: dict[str, Any] = {}
            message = entry.get("message")
            if isinstance(message, str):
                item["message"] = message[:ERROR_MESSAGE_CAP]
                if len(message) > ERROR_MESSAGE_CAP:
                    item["message_truncated"] = True
            code = entry.get("code")
            if isinstance(code, str):
                if _CODE_SHAPE.match(code):
                    item["code"] = code
                else:
                    item["code"] = "<withheld>"
            ext = entry.get("extensions")
            if isinstance(ext, dict):
                ext_code = ext.get("code")
                if isinstance(ext_code, str):
                    if _CODE_SHAPE.match(ext_code):
                        item["extensions_code"] = ext_code
                    else:
                        item["extensions_code"] = "<withheld>"
            if item:
                entries.append(item)
        diag["errors"] = entries
        diag["errors_truncated"] = len(errors) > ERROR_MAX_ENTRIES
        return self._guard_diag(diag)

    @staticmethod
    def _decoded_scan_hit(obj: Any) -> bool:
        if isinstance(obj, str):
            return bool(_SECRET_SHAPE.search(obj)
                        or _CART_ID_SHAPE.search(obj)
                        or _KEY_MARKER_SHAPE.search(obj))
        if isinstance(obj, list):
            return any(_ErrorCaptureTransport._decoded_scan_hit(v)
                       for v in obj)
        if isinstance(obj, dict):
            return any(_ErrorCaptureTransport._decoded_scan_hit(k)
                       or _ErrorCaptureTransport._decoded_scan_hit(v)
                       for k, v in obj.items())
        return False

    @staticmethod
    def _read_bounded(response: httpx.Response):
        # Content-encoded bodies are NEVER decompressed or read: a cap
        # on decoded bytes does not bound a decoder's allocation on a
        # compressed bomb, and proving a bounded decoder is out of
        # scope. Status and request id remain useful without the body.
        encoding = response.headers.get("content-encoding")
        if encoding and encoding.strip().lower() != "identity":
            return None, "content_encoding"
        declared = response.headers.get("content-length")
        if declared is not None:
            try:
                size = int(declared)
            except ValueError:
                size = None
            if size is not None and size > ERROR_BODY_CAP:
                return None, "content_length_exceeds_cap"
        if response.is_stream_consumed:
            # Content already in memory (e.g. a constructed response);
            # iterating it again would raise StreamConsumed.
            body = response.content
            if len(body) > ERROR_BODY_CAP:
                return None, "exceeds_cap"
            return body, None
        buf = bytearray()
        # Encoded bodies are already skipped above, so this stream is
        # identity-decoded; iter_bytes simply bounds the read.
        for chunk in response.iter_bytes(8192):
            buf.extend(chunk)
            if len(buf) > ERROR_BODY_CAP:
                response.close()
                return None, "exceeds_cap"
        return bytes(buf), None

    @staticmethod
    def _guard_diag(diag: dict) -> dict:
        """Second line: the serialized diagnostic itself must be clean."""
        text = json.dumps(diag, sort_keys=True)
        if _SECRET_SHAPE.search(text) or _CART_ID_SHAPE.search(text) \
                or _KEY_MARKER_SHAPE.search(text):
            return {"capture_refused": "guard_hit"}
        return diag


class LiveShopifyBackend(ShopifyStorefrontBackend):
    """Deliberately-live construction of the reviewed adapter.

    Mirrors the reviewed __init__ without its MockTransport-only guard;
    the guard's intent is preserved by requiring allow_live=True and by
    rejecting any injected transport that is not an httpx.MockTransport
    (test seam). Does NOT call the parent __init__.
    """

    def __init__(self, shop: str, mapping_path: Optional[str] = None, *,
                 allow_live: bool = False,
                 api_version: str = PINNED_API_VERSION,
                 transport: Optional[httpx.BaseTransport] = None,
                 max_requests: int = DEFAULT_MAX_REQUESTS):
        if allow_live is not True:
            raise TypeError("live construction refused: pass allow_live=True")
        if shop != EXPECTED_SHOP:
            raise ValueError("unexpected shop: probe is bound to "
                             + EXPECTED_SHOP)
        if api_version != PINNED_API_VERSION:
            # The version string is interpolated into the request path;
            # it is pinned, never caller-routed.
            raise ValueError("api version pinned to " + PINNED_API_VERSION)
        if transport is not None \
                and not isinstance(transport, httpx.MockTransport):
            raise TypeError("only httpx.MockTransport may be injected "
                            "(tests); the live path uses a fresh real "
                            "transport")
        if not isinstance(max_requests, int) \
                or isinstance(max_requests, bool) or max_requests < 1:
            raise ValueError("invalid max_requests")
        self.shop = shop
        self.api_version = api_version
        inner = transport if transport is not None \
            else httpx.HTTPTransport(trust_env=False)
        self._capture_transport = _ErrorCaptureTransport(
            _StatusGuardTransport(inner))
        self.client = httpx.Client(
            base_url=f"https://{shop}/api/{api_version}/",
            transport=self._capture_transport, timeout=15.0,
            follow_redirects=False, trust_env=False)
        self._max_requests = max_requests
        self.request_count = 0
        self.last_error: Optional[str] = None
        self.last_diagnostic: Optional[dict] = None
        self._failed: Optional[str] = None  # sticky fail-closed latch
        mapping: dict[str, Any] = {}
        if mapping_path is not None and Path(mapping_path).exists():
            mapping = json.loads(
                Path(mapping_path).read_text(encoding="utf-8"))
        self._sku_to_gid = dict(mapping.get("variants", {}))
        self._gid_to_sku = {gid: sku
                            for sku, gid in self._sku_to_gid.items()}
        self._cart_id: Optional[str] = None
        self._cart_key: Optional[str] = None
        self._cart_ref: Optional[str] = None

    def _gql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Count every request, enforce the cap, latch fail-closed after
        ANY failure, and surface the failure category. The parent's
        acquire_state reports failures only as a canonical UNKNOWN
        resolution; the probe needs the category for stop decisions and
        evidence, so it is recorded here before the parent swallows it.
        No retry: each failure is raised exactly once, and once latched,
        every later call is refused WITHOUT sending.
        """
        if self._failed is not None:
            raise ShopifyBackendError(
                "backend_failed_closed:" + self._failed)
        if self.request_count >= self._max_requests:
            self._failed = "request_cap_exceeded"
            self.last_error = self._failed
            raise ShopifyBackendError(self._failed)
        self.request_count += 1
        self._capture_transport.last_diagnostic = None
        self.last_diagnostic = None
        try:
            return super()._gql(query, variables)
        except ShopifyBackendError as exc:
            cause = exc.__cause__
            if isinstance(cause, _StatusRejected):
                self._failed = f"redirect_or_status:{cause.status_code}"
            else:
                self._failed = str(exc)
            self.last_error = self._failed
            self.last_diagnostic = self._capture_transport.last_diagnostic
            raise


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _guard_text(text: str, backend: LiveShopifyBackend) -> None:
    """Fail-closed: refuse to write anything shaped like a cart
    capability - generically (works even before the backend has
    registered its cart) - or matching the registered id/key exactly."""
    if _CART_ID_SHAPE.search(text):
        raise RuntimeError("evidence_redaction_guard: cart id shape")
    if _KEY_MARKER_SHAPE.search(text):
        raise RuntimeError("evidence_redaction_guard: cart key marker")
    cart_id = getattr(backend, "_cart_id", None)
    cart_key = getattr(backend, "_cart_key", None)
    if cart_id and cart_id in text:
        raise RuntimeError("evidence_redaction_guard: cart id would leak")
    if cart_key and cart_key in text:
        raise RuntimeError("evidence_redaction_guard: cart key would leak")


def _write_json(log_dir: Path, name: str, obj: Any,
                backend: LiveShopifyBackend) -> Path:
    text = json.dumps(backend.redact(obj), indent=2, sort_keys=True)
    _guard_text(text, backend)
    path = log_dir / name
    path.write_text(text + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------- #
# Mapping validation (discover output and probe input alike)
# --------------------------------------------------------------- #

def _validate_mapping(mapping: Any) -> Optional[str]:
    """Return None when valid, else a short refusal category. Keys must be
    non-empty unique strings; values must be unique, well-formed variant
    GIDs - a validated GID can never be or contain a cart capability."""
    if not isinstance(mapping, dict) or len(mapping) < 2:
        return "insufficient_variants"
    values = []
    for key, gid in mapping.items():
        if not isinstance(key, str) or not key.strip():
            return "empty_key"
        if _CART_ID_SHAPE.search(key) or _KEY_MARKER_SHAPE.search(key):
            return "capability_shaped_key"
        if not isinstance(gid, str) \
                or not _VARIANT_GID_SHAPE.match(gid):
            return "invalid_variant_gid"
        values.append(gid)
    if len(set(values)) != len(values):
        return "duplicate_variant_gid"
    return None


# --------------------------------------------------------------- #
# discover: ONE read-only tokenless products query. Zero cart calls.
# --------------------------------------------------------------- #

def _variant_key(gid: str, sku: Any, taken: set) -> Optional[str]:
    """Collision-safe mapping key. The fallback is the FULL GID tail
    (unique because GIDs are unique), never a truncated hash. Returns
    None when even the fallback is taken - the caller must refuse."""
    if isinstance(sku, str) and sku.strip() and sku.strip() not in taken:
        return sku.strip()
    fallback = "VARIANT:" + gid.rsplit("/", 1)[-1]
    if fallback not in taken:
        return fallback
    return None


def run_discover(backend: LiveShopifyBackend, mapping_out: str,
                 log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"shop": backend.shop,
                               "api_version": backend.api_version,
                               "started_at": _now()}
    try:
        catalog = backend.read_catalog()
    except ShopifyBackendError as exc:
        summary.update({"status": "failed",
                        "category": backend.last_error or str(exc),
                        "requests_used": backend.request_count})
        if backend.last_diagnostic:
            summary["diagnostic"] = backend.last_diagnostic
        _write_json(log_dir, "discover-summary.json", summary, backend)
        return 4
    try:
        _write_json(log_dir, "catalog-redacted.json", catalog, backend)
    except RuntimeError:
        # The guard refused to persist the raw catalog (capability-shaped
        # content). The validated mapping below is unaffected; record the
        # withholding instead of writing around the guard.
        summary["catalog_raw_withheld"] = True
    products: list = []
    if isinstance(catalog, dict):
        products = ((catalog.get("products") or {}).get("edges")) or []
    variants: list[dict[str, Any]] = []
    skipped_invalid_gid = 0
    # The reviewed query fetches products(first: 20) x variants(first: 20)
    # with no pageInfo: at either limit the catalog may be truncated and
    # that risk is recorded, never silently accepted.
    pagination_risk = len(products) >= 20
    for edge in products:
        node = (edge or {}).get("node") or {}
        v_edges = ((node.get("variants") or {}).get("edges")) or []
        if len(v_edges) >= 20:
            pagination_risk = True
        for v_edge in v_edges:
            vnode = (v_edge or {}).get("node") or {}
            gid = vnode.get("id")
            if isinstance(gid, str) and _VARIANT_GID_SHAPE.match(gid):
                variants.append({"gid": gid, "sku": vnode.get("sku")})
            else:
                skipped_invalid_gid += 1
    mapping: dict[str, str] = {}
    taken: set = set()
    seen_gids: set = set()
    collision = False
    for item in variants:
        if item["gid"] in seen_gids:
            continue
        seen_gids.add(item["gid"])
        key = _variant_key(item["gid"], item.get("sku"), taken)
        if key is None:
            collision = True
            break
        taken.add(key)
        mapping[key] = item["gid"]
    refusal = _validate_mapping(mapping)
    if collision:
        refusal = "key_collision"
    summary.update({
        "status": "ok" if refusal is None else "refused:" + refusal,
        "product_count": len(products),
        "variant_count": len(variants),
        "skipped_invalid_gid": skipped_invalid_gid,
        "mapped_variant_count": len(mapping),
        "catalog_pagination_risk": pagination_risk,
        "requests_used": backend.request_count,
        "note": "one read-only tokenless products query; zero cart calls",
    })
    _write_json(log_dir, "discover-summary.json", summary, backend)
    if refusal is not None:
        return 4
    # Serialize and guard BEFORE any write of the mapping, then write the
    # same guarded text to both destinations. Values are already
    # shape-validated variant GIDs, so they can never be capabilities.
    mapping_text = json.dumps({"variants": mapping}, indent=2,
                              sort_keys=True) + "\n"
    _guard_text(mapping_text, backend)
    out = Path(mapping_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mapping_text, encoding="utf-8")
    (log_dir / "mapping.json").write_text(mapping_text, encoding="utf-8")
    print(f"discover: {len(products)} products, "
          f"{len(mapping)} variants mapped")
    return 0


# --------------------------------------------------------------- #
# probe: frozen intent, bounded writes, honest verdicts
# --------------------------------------------------------------- #

def _freeze_plan(mapping: dict[str, str]) -> dict[str, Any]:
    """Freeze the ENTIRE plan before the first mutation: variant choice,
    every step, and every expectation rule. Dynamic steps state their rule
    here; the rule never changes after observations arrive."""
    pairs = sorted(mapping.items(), key=lambda kv: kv[1])
    (key_a, gid_a), (key_b, gid_b) = pairs[0], pairs[1]
    return {
        "frozen_at": _now(),
        "variants": {"A": {"item_key": key_a, "gid": gid_a},
                     "B": {"item_key": key_b, "gid": gid_b}},
        "final_intent": {key_a: 3, key_b: 1},
        "steps": [
            {"name": "baseline_read", "expect": {}},
            {"name": "add_A_x2", "expect": {key_a: 2}},
            {"name": "add_B_x1", "expect": {key_a: 2, key_b: 1}},
            {"name": "add_A_x1_merge_pin",
             "expect": {key_a: 3, key_b: 1},
             "note": "records the raw per-line structure to pin the real "
                     "merge behavior; the aggregate expectation holds "
                     "under both merge and split"},
            {"name": "set_first_A_line_to_1",
             "expect": "dynamic: A := 1 + (A_total_before - "
                       "qty_on_first_A_line); B unchanged"},
            {"name": "delete_first_B_line",
             "expect": "dynamic: B := B_total_before - "
                       "qty_on_first_B_line; A unchanged"},
            {"name": "cleanup_delete_all_lines", "expect": {}},
        ],
    }


def run_probe(backend: LiveShopifyBackend, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    plan = _freeze_plan(backend._sku_to_gid)
    key_a = plan["variants"]["A"]["item_key"]
    key_b = plan["variants"]["B"]["item_key"]
    _write_json(log_dir, "plan.json", plan, backend)
    steps_path = log_dir / "steps.jsonl"
    state: dict[str, Any] = {"overall": "COMPLETE", "cleaned": False,
                             "merge_behavior": "not_observed",
                             "stop_reason": None}
    last_known: dict[str, Any] = {"lines": None, "line_ids": []}

    def log_step(entry: dict) -> None:
        text = json.dumps(backend.redact(entry), sort_keys=True)
        _guard_text(text, backend)
        with steps_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    def finalize() -> int:
        final = {
            "shop": backend.shop,
            "api_version": backend.api_version,
            "finished_at": _now(),
            "overall_verdict": state["overall"],
            "stop_reason": state["stop_reason"],
            "merge_behavior": state["merge_behavior"],
            "requests_used": backend.request_count,
            "max_requests": backend._max_requests,
            "cart_verified_empty": state["cleaned"],
            "cart_ref": backend.cart_ref,
            "capabilities": backend.capabilities(),
        }
        if backend.last_diagnostic:
            final["diagnostic"] = backend.last_diagnostic
        if not state["cleaned"]:
            final["cart_not_cleaned"] = {
                "cart_ref": backend.cart_ref,
                "last_known_lines": last_known.get("lines"),
                "note": "cart state unknown after stop; Shopify expires "
                        "abandoned carts automatically (TTL not pinned "
                        "by this probe)"}
        _write_json(log_dir, "final.json", final, backend)
        return {"COMPLETE": 0, "PARTIAL": 6}.get(state["overall"], 5)

    def read_state(event: str) -> dict:
        """One canonical read that only accepts a fully resolved remote
        cart; anything else is a stop, never a guessed verdict."""
        post = backend.acquire_state()
        if post["lines"] is None or post["resolution"] != "remote-cart":
            log_step({"event": event, "resolution": post["resolution"],
                      "lines": post["lines"],
                      "provenance": post["provenance"]})
            raise ProbeStop("acquisition_failed:" + event + ":"
                            + str(backend.last_error
                                  or post["resolution"]))
        return post

    def mutate_and_read(name: str,
                        fn: Callable[[], dict],
                        expect: dict) -> dict:
        nonlocal last_known
        result = fn()
        if not result["ok"]:
            # Check FIRST: no read, no cleanup, no request after a failed
            # mutation.
            log_step({"event": "step", "name": name, "mutation_ok": False,
                      "version_enforcement":
                          result["version_enforcement"],
                      "payload": result["payload"]})
            raise ProbeStop("mutation_failed:" + name)
        post = read_state(name)
        entry = {"event": "step", "name": name, "mutation_ok": True,
                 "version_enforcement": result["version_enforcement"],
                 "payload": result["payload"],
                 "expected": expect, "observed": post["lines"],
                 "resolution": post["resolution"],
                 "raw_line_ids": post["line_ids"],
                 "provenance": post["provenance"]}
        mismatches = compare_lines(expect, post["lines"])
        entry["mismatches"] = mismatches
        log_step(entry)
        if mismatches:
            state["overall"] = "PARTIAL"  # a finding, not a probe failure
        last_known = post
        return post

    try:
        baseline = read_state("baseline")
        log_step({"event": "baseline",
                  "resolution": baseline["resolution"],
                  "lines": baseline["lines"],
                  "line_ids": baseline["line_ids"],
                  "provenance": baseline["provenance"]})
        last_known = baseline
        if baseline["lines"] != {}:
            # A freshly created cart should be empty; if not, that is a
            # recorded finding, never a guessed correction.
            state["overall"] = "PARTIAL"

        mutate_and_read("add_A_x2",
                        lambda: backend.add_line(key_a, 2, None),
                        {key_a: 2})
        mutate_and_read("add_B_x1",
                        lambda: backend.add_line(key_b, 1, None),
                        {key_a: 2, key_b: 1})
        post = mutate_and_read("add_A_x1_merge_pin",
                               lambda: backend.add_line(key_a, 1, None),
                               {key_a: 3, key_b: 1})
        a_lines = [e for e in post["line_ids"]
                   if e["item_key"] == key_a]
        state["merge_behavior"] = ("merged_single_line"
                                   if len(a_lines) == 1
                                   else "split_lines")
        log_step({"event": "merge_pin", "item_key": key_a,
                  "raw_line_count": len(a_lines),
                  "aggregate_quantity": post["lines"].get(key_a, 0),
                  "behavior": state["merge_behavior"]})

        first_a = a_lines[0]
        expect4 = dict(post["lines"])
        expect4[key_a] = 1 + (post["lines"].get(key_a, 0)
                              - first_a["quantity"])
        post = mutate_and_read(
            "set_first_A_line_to_1",
            lambda: backend.set_line_quantity(first_a["line_id"], 1, None),
            expect4)

        b_lines = [e for e in post["line_ids"]
                   if e["item_key"] == key_b]
        if not b_lines:
            raise ProbeStop("no_B_line_to_delete")
        first_b = b_lines[0]
        expect5 = dict(post["lines"])
        remaining_b = post["lines"].get(key_b, 0) - first_b["quantity"]
        if remaining_b > 0:
            expect5[key_b] = remaining_b
        else:
            expect5.pop(key_b, None)
        post = mutate_and_read(
            "delete_first_B_line",
            lambda: backend.delete_line(first_b["line_id"], None),
            expect5)

        # Cleanup, normal path only, within the RESERVED budget: one
        # reviewed delete per remaining line, then a final read.
        needed = len(post["line_ids"]) + 1
        if backend.request_count + needed > backend._max_requests:
            log_step({"event": "cleanup_skipped",
                      "reason": "cleanup_over_budget",
                      "requests_used": backend.request_count,
                      "needed": needed})
            raise ProbeStop("cleanup_over_budget")
        for entry in list(post["line_ids"]):
            result = backend.delete_line(entry["line_id"], None)
            if not result["ok"]:
                log_step({"event": "cleanup_failed",
                          "payload": result["payload"]})
                raise ProbeStop("cleanup_mutation_failed")
        final_state = read_state("cleanup_read")
        log_step({"event": "cleanup_read",
                  "resolution": final_state["resolution"],
                  "lines": final_state["lines"],
                  "line_ids": final_state["line_ids"],
                  "provenance": final_state["provenance"]})
        last_known = final_state
        if final_state["lines"] == {}:
            state["cleaned"] = True
        else:
            state["overall"] = "PARTIAL"
    except ProbeStop as exc:
        state["overall"] = "UNKNOWN"
        state["stop_reason"] = str(exc)
        log_step({"event": "stop", "reason": str(exc),
                  "last_error": backend.last_error,
                  "cart_ref": backend.cart_ref})
    except Exception as exc:
        # Never leak and never continue after an unexpected error.
        state["overall"] = "UNKNOWN"
        state["stop_reason"] = "unexpected:" + type(exc).__name__
        try:
            log_step({"event": "stop", "reason": state["stop_reason"]})
        except Exception:
            pass
    return finalize()


def main(argv: Optional[list] = None,
         transport: Optional[httpx.BaseTransport] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["discover", "probe"],
                        required=True)
    parser.add_argument("--shop", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--max-requests", type=int,
                        default=DEFAULT_MAX_REQUESTS)
    args = parser.parse_args(argv)
    if not args.allow_live:
        print(json.dumps({"startup_error": "live_not_approved"}),
              file=sys.stderr)
        return 3
    mapping_path: Optional[str] = None
    if args.stage == "probe":
        if not Path(args.mapping).exists():
            print(json.dumps(
                {"startup_error": "mapping_missing_run_discover_first"}),
                file=sys.stderr)
            return 4
        if args.max_requests < PROBE_MIN_BUDGET:
            print(json.dumps({"startup_error": "request_budget_too_small",
                              "min": PROBE_MIN_BUDGET}), file=sys.stderr)
            return 4
        try:
            loaded = json.loads(
                Path(args.mapping).read_text(encoding="utf-8"))
        except Exception:
            loaded = None
        refusal = _validate_mapping(
            (loaded or {}).get("variants") if isinstance(loaded, dict)
            else None)
        if refusal is not None:
            print(json.dumps({"startup_error": "mapping_invalid",
                              "category": refusal}), file=sys.stderr)
            return 4
        mapping_path = args.mapping
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        backend = LiveShopifyBackend(
            args.shop, mapping_path, allow_live=True,
            transport=transport, max_requests=args.max_requests)
    except (TypeError, ValueError) as exc:
        print(json.dumps({"startup_error": type(exc).__name__}),
              file=sys.stderr)
        return 2
    try:
        if args.stage == "discover":
            return run_discover(backend, args.mapping, log_dir)
        return run_probe(backend, log_dir)
    except Exception as exc:
        # Uncategorized failure: stop clean and report the type only.
        # Not every possible failure carries a category; an uncategorized
        # stop must be reported as-is, never smoothed over.
        print(json.dumps({"startup_error": "uncategorized:"
                          + type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

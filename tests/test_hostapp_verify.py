"""Offline tests for the hosted verify slice (WB-H1 + WB-H2).

These tests boot the COMPOSED app (storefront + verify router + middleware,
built by deploy_entry.build_app) on loopback ports and drive it over real
HTTP. They are offline: no network beyond 127.0.0.1, no model calls, no
secrets.
"""
from __future__ import annotations

import asyncio
import socket
import sqlite3
import threading
import time
from html.parser import HTMLParser

import httpx
import pytest
import uvicorn

import deploy_entry
from hostapp.security import PolicyError, ProxyPolicy

MATCH_SKU = "SHOE-CRT-9-WHITE"     # Court Classic White 9, 6500 cents
WRONG_VARIANT_SKU = "SHOE-CRT-8-WHITE"
BOTTLE_A = "BTL-HIK-500-NAVY"      # 2400 cents
BOTTLE_B = "BTL-THR-500-BLACK"     # 4100 cents
EXTRA_SKU = "SHT-HOO-M-HEATHER"    # hoodie, 5800 cents
PUBLIC_ORIGIN = "https://cartproof.example.test"
PUBLIC_HOST = "cartproof.example.test"
CF_IP = "203.0.113.10"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _boot(app, port: int):
    sock = socket.socket()
    sock.bind(("127.0.0.1", port))
    config = uvicorn.Config(app, log_level="warning", access_log=False,
                            proxy_headers=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.01)
    return server, thread


def _stop(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=5)


def _serve(tmp_path, **kwargs):
    """Boot one composed app; returns (base_url, server, thread, db_path)."""
    port = _free_port()
    db = str(tmp_path / f"host-{port}.db")
    kwargs.setdefault("port", port)
    kwargs.setdefault("db_path", db)
    kwargs.setdefault("self_base_url", f"http://127.0.0.1:{port}")
    # Never inherit ambient env: tests choose local mode explicitly. Public
    # mode is opted into with ProxyPolicy.render() in the dedicated tests.
    kwargs.setdefault("policy", ProxyPolicy.local())
    app = deploy_entry.build_app(**kwargs)
    server, thread = _boot(app, port)
    return f"http://127.0.0.1:{port}", server, thread, db


def _public_serve(tmp_path, **kwargs):
    kwargs.setdefault("policy", ProxyPolicy.render(PUBLIC_ORIGIN))
    return _serve(tmp_path, **kwargs)


def _edge(cf=CF_IP, host=PUBLIC_HOST, **extra):
    headers = {"Host": host, "CF-Connecting-IP": cf}
    headers.update(extra)
    return headers


class _HTTPSFront(httpx.BaseTransport):
    """Offline HTTPS-front fixture over the real loopback composed app.

    HTTPSX's normal cookiejar selects cookies for the fixture HTTPS origin;
    this transport forwards the prepared request plus the selected Cookie
    header over HTTP to 127.0.0.1, then returns the response associated with
    the original HTTPS request. It overwrites Host/CF at the simulated edge.
    """

    def __init__(self, backend_url: str, cf: str = CF_IP):
        host = httpx.URL(backend_url).host
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("test backend must be loopback")
        self.backend_url = backend_url.rstrip("/")
        self.cf = cf
        self.cookie_sent = False

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.scheme != "https" or request.url.host != PUBLIC_HOST:
            raise httpx.TransportError("test front refused non-fixture origin")
        target = self.backend_url + request.url.raw_path.decode("ascii")
        headers = []
        self.cookie_sent = False
        for k, v in request.headers.raw:
            lk = k.lower()
            if lk in (b"host", b"cf-connecting-ip", b"content-length"):
                continue
            if lk == b"cookie":
                self.cookie_sent = True
            headers.append((k, v))
        headers.append((b"host", PUBLIC_HOST.encode("ascii")))
        headers.append((b"cf-connecting-ip", self.cf.encode("ascii")))
        with httpx.Client(trust_env=False, follow_redirects=False,
                          timeout=30.0) as backend:
            resp = backend.request(request.method, target, headers=headers,
                                   content=request.content)
        out_headers = [(k, v) for k, v in resp.headers.raw
                       if k.lower() not in (b"content-encoding",
                                            b"transfer-encoding",
                                            b"content-length")]
        return httpx.Response(resp.status_code, headers=out_headers,
                              content=resp.content, request=request)


def _front(backend_url: str, cf: str = CF_IP) -> httpx.Client:
    return httpx.Client(base_url=PUBLIC_ORIGIN,
                        transport=_HTTPSFront(backend_url, cf=cf),
                        follow_redirects=False, timeout=60.0)


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.h2 = []
        self.tables = []
        self._h2 = None
        self._table = None
        self._row = None
        self._cell = None
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag == "h2":
            self._h2 = []
        elif tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data):
        if self._h2 is not None:
            self._h2.append(data)
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag == "h2" and self._h2 is not None:
            self.h2.append(" ".join("".join(self._h2).split()))
            self._h2 = None
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def _page(html: str) -> _PageParser:
    p = _PageParser()
    p.feed(html)
    return p


def _front_add(client, sku, qty):
    cart = client.get("/api/cart")
    assert cart.status_code == 200, cart.text
    version = cart.json()["version"]
    r = client.post("/api/cart/lines", json={"sku": sku, "quantity": qty},
                    headers={"Origin": PUBLIC_ORIGIN,
                             "X-Cart-Version": str(version)})
    assert r.status_code == 200, r.text
    return r


def _cart(client):
    r = client.get("/api/cart")
    assert r.status_code == 200, r.text
    return r.json()


def _asgi_status(app, method, path, headers, body=b""):
    async def run():
        scope = {"type": "http", "http_version": "1.1", "method": method,
                 "scheme": "http", "path": path, "raw_path": path.encode(),
                 "query_string": b"", "headers": headers,
                 "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 0)}
        sent = []
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        async def send(message):
            sent.append(message)
        await app(scope, receive, send)
        return sent[0]["status"] if sent else None
    return asyncio.run(run())



@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    url, server, thread, _ = _serve(tmp_path_factory.mktemp("db"))
    yield url
    _stop(server, thread)


def _visitor(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=15.0)


def _add(client: httpx.Client, sku: str, qty: int) -> None:
    cart = client.get("/api/cart").json()
    resp = client.post("/api/cart/lines", json={"sku": sku, "quantity": qty},
                       headers={"X-Cart-Version": str(cart["version"])})
    assert resp.status_code == 200, resp.text


def _matching_cart(client: httpx.Client) -> None:
    _add(client, MATCH_SKU, 2)
    _add(client, BOTTLE_A, 1)
    _add(client, BOTTLE_B, 1)


def _verify(client: httpx.Client) -> httpx.Response:
    return client.post("/verify/run")


# --------------------------------------------------------------------- #
# Page and verdict behaviour
# --------------------------------------------------------------------- #

def test_verify_page_renders_task_and_cart(base_url):
    c = _visitor(base_url)
    r = c.get("/verify")
    assert r.status_code == 200
    assert "Demo store" in r.text
    assert "Court Classic" in r.text
    assert "Your cart is empty" in r.text
    assert 'name="viewport"' in r.text
    assert "cp_session" in c.cookies


def test_matching_cart_is_confirmed(base_url):
    c = _visitor(base_url)
    _matching_cart(c)
    r = _verify(c)
    assert r.status_code == 200
    assert "MATCH" in r.text
    assert r.text.count("confirmed") >= 3
    assert '<td class="contradicted">' not in r.text
    assert "$195.00" in r.text


def test_wrong_variant_is_contradicted(base_url):
    c = _visitor(base_url)
    _add(c, WRONG_VARIANT_SKU, 2)  # size 8, not the required size 9
    _add(c, BOTTLE_A, 1)
    _add(c, BOTTLE_B, 1)
    r = _verify(c)
    assert "MISMATCH" in r.text
    assert '<td class="contradicted">' in r.text
    assert "observed 0 != required 2" in r.text


def test_excess_quantity_is_contradicted(base_url):
    c = _visitor(base_url)
    _add(c, MATCH_SKU, 3)
    _add(c, BOTTLE_A, 1)
    _add(c, BOTTLE_B, 1)
    r = _verify(c)
    assert "MISMATCH" in r.text
    assert "observed 3 != required 2" in r.text


def test_over_budget_is_contradicted(base_url):
    c = _visitor(base_url)
    _matching_cart(c)
    _add(c, EXTRA_SKU, 1)  # 19500 + 5800 = 25300 > 20000
    r = _verify(c)
    assert "MISMATCH" in r.text
    assert "exceeds" in r.text


def test_empty_cart_is_reported_honestly(base_url):
    c = _visitor(base_url)
    r = _verify(c)
    assert r.status_code == 200
    assert "MISMATCH" in r.text  # empty cart contradicts the fixed task
    assert "observed 0 != required 2" in r.text


# --------------------------------------------------------------------- #
# Session isolation and binding
# --------------------------------------------------------------------- #

def test_cross_session_isolation(base_url):
    a = _visitor(base_url)
    b = _visitor(base_url)
    _matching_cart(a)
    # B verifies: must see only B's own (empty) cart.
    rb = _verify(b)
    assert "observed 0 != required 2" in rb.text
    # A forged cookie cannot name A's session: it mints a fresh empty one.
    forged = _visitor(base_url)
    forged.cookies.clear()
    forged.cookies.set("cp_session", "0" * 32, domain="127.0.0.1", path="/")
    rf = _verify(forged)
    assert "observed 0 != required 2" in rf.text
    assert forged.cookies.get("cp_session", domain="127.0.0.1", path="/") != "0" * 32
    # A's own session is untouched and still matches.
    ra = _verify(a)
    assert "MATCH" in ra.text
    cart_a = a.get("/api/cart").json()
    cart_b = b.get("/api/cart").json()
    assert cart_a["cart_id"] != cart_b["cart_id"]
    assert len(cart_a["lines"]) == 3 and len(cart_b["lines"]) == 0


def test_cart_id_injection_is_ignored(base_url):
    a = _visitor(base_url)
    _matching_cart(a)
    victim_cart = a.get("/api/cart").json()["cart_id"]
    attacker = _visitor(base_url)  # empty cart
    r = attacker.post("/verify/run", params={"cart_id": victim_cart},
                      data={"cart_id": victim_cart, "session_id": "x" * 32})
    assert r.status_code == 200
    # Attacker sees only their own empty cart, not the victim's full one.
    assert "observed 0 != required 2" in r.text
    assert "$195.00" not in r.text


def test_invalid_cookie_creates_at_most_one_session(tmp_path):
    url, server, thread, db = _serve(tmp_path)
    try:
        def session_count():
            conn = sqlite3.connect(db)
            try:
                return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            finally:
                conn.close()

        before = session_count()
        c = _visitor(url)
        c.cookies.set("cp_session", "f" * 32, domain="127.0.0.1", path="/")
        r = _verify(c)
        assert r.status_code == 200
        after = session_count()
        assert after - before <= 1, f"one run minted {after - before} sessions"
        import re
        m = re.search(r"after (\d+) sample", r.text)
        assert m and int(m.group(1)) <= 41, "read budget exceeded"
    finally:
        _stop(server, thread)


# --------------------------------------------------------------------- #
# Origin / CSRF surface
# --------------------------------------------------------------------- #

def test_origin_mismatch_rejected_on_verify(base_url):
    c = _visitor(base_url)
    r = c.post("/verify/run", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "origin_mismatch"
    r2 = c.post("/verify/run", headers={"Origin": base_url})
    assert r2.status_code == 200


def test_origin_mismatch_rejected_on_cart_mutations(base_url):
    c = _visitor(base_url)
    cart = c.get("/api/cart").json()
    r = c.post("/api/cart/lines", json={"sku": MATCH_SKU, "quantity": 1},
               headers={"X-Cart-Version": str(cart["version"]),
                        "Origin": "https://evil.example"})
    assert r.status_code == 403
    # A matching origin (scheme + host) passes.
    r2 = c.post("/api/cart/lines", json={"sku": MATCH_SKU, "quantity": 1},
                headers={"X-Cart-Version": str(cart["version"]),
                         "Origin": base_url})
    assert r2.status_code == 200


# --------------------------------------------------------------------- #
# Body limits: actual streamed bytes, not declared length
# --------------------------------------------------------------------- #

def test_chunked_body_over_cap_is_rejected(base_url):
    c = _visitor(base_url)

    def big():
        for _ in range(70):  # 70 KB as a chunked stream, no Content-Length
            yield b"x" * 1024

    r = c.post("/verify/run", content=big())
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "body_too_large"


def test_declared_length_is_the_framed_body(base_url):
    # Under Content-Length framing the declared length IS the body: a request
    # declaring 10 bytes carries a 10-byte body (extra bytes are a protocol
    # error owned by the HTTP parser, not a cap bypass). The unbounded case is
    # the chunked stream, covered above. Here: a small declared body passes
    # the cap and reaches the verdict path.
    c = _visitor(base_url)
    r = c.post("/verify/run", content=b"0123456789",
               headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200


def test_malformed_content_length_is_rejected(base_url):
    host, port = base_url.split("//")[1].split(":")
    for bad in (b"abc", b"-5"):
        req = (f"POST /verify/run HTTP/1.1\r\nHost: {host}:{port}\r\n"
               f"Content-Length: ".encode() + bad + b"\r\nConnection: close\r\n\r\n")
        with socket.create_connection((host, int(port)), timeout=10) as s:
            s.sendall(req)
            resp = s.recv(4096)
        status = int(resp.split(b" ", 2)[1])
        assert status == 400, (bad, resp[:200])


# --------------------------------------------------------------------- #
# Failure semantics: honest UNKNOWN, no internals, bounded wall time
# --------------------------------------------------------------------- #

def _sessions_count(db):
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()


def test_dead_self_base_gives_unknown_not_error(tmp_path):
    """Bootstrap failure: UNKNOWN page, ZERO settlement reads, no session minted."""
    port = _free_port()
    db = str(tmp_path / "dead.db")
    app = deploy_entry.build_app(port=port, db_path=db,
                                 self_base_url="http://127.0.0.1:1")  # nothing there
    server, thread = _boot(app, port)
    try:
        # Force the store DB to exist so the count is meaningful.
        c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
        c.get("/api/cart")
        before = _sessions_count(db)
        start = time.monotonic()
        r = c.post("/verify/run")
        elapsed = time.monotonic() - start
        assert r.status_code == 200
        assert "UNKNOWN" in r.text
        assert "session could not be established" in r.text
        assert "after 0 sample(s)" in r.text  # no settlement was attempted
        assert "Traceback" not in r.text
        assert elapsed < 10.0, f"handler ran {elapsed:.1f}s"
        assert _sessions_count(db) == before  # failed bootstrap mints nothing
    finally:
        _stop(server, thread)


class _FakeBackend:
    """A minimal raw-socket HTTP server with scripted bad behaviour."""

    def __init__(self, handler):
        self.port = _free_port()
        self.stop = threading.Event()
        self.handler = handler
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.port))
        srv.listen(8)
        srv.settimeout(0.3)
        while not self.stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            try:
                self.handler(conn)
            except OSError:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
        srv.close()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.thread.join(timeout=3)


def _trickle(conn):
    conn.settimeout(0.5)
    conn.recv(4096)
    head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 60\r\n\r\n"
    body = b'{"cart_id":"x","version":0,"lines":[],"total_cents":0}'
    for i in range(len(head + body)):
        conn.sendall((head + body)[i:i + 1])
        time.sleep(0.05)  # ~5 s for the full response: slower than the budget


def _garbage(conn):
    conn.settimeout(0.5)
    conn.recv(4096)
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                 b"Content-Length: 5\r\n\r\nxxxxx")


def test_trickling_backend_is_bounded_and_unknown(tmp_path):
    """A peer that dribbles bytes cannot stretch the handler past its budget."""
    with _FakeBackend(_trickle) as backend:
        port = _free_port()
        db = str(tmp_path / "trickle.db")
        app = deploy_entry.build_app(port=port, db_path=db,
                                     self_base_url=f"http://127.0.0.1:{backend.port}")
        server, thread = _boot(app, port)
        try:
            c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
            start = time.monotonic()
            r = c.post("/verify/run")
            elapsed = time.monotonic() - start
            assert r.status_code == 200
            assert "UNKNOWN" in r.text
            assert "Traceback" not in r.text
            # Bootstrap budget is 1500 ms; the whole run must stay far under
            # TOTAL_BUDGET_MS even against a peer sending 1 byte per 50 ms.
            assert elapsed < 6.5, f"handler ran {elapsed:.1f}s against a trickling peer"
        finally:
            _stop(server, thread)


def test_garbage_backend_is_unknown(tmp_path):
    """A 200 with a non-JSON body is malformed evidence, not a crash."""
    with _FakeBackend(_garbage) as backend:
        port = _free_port()
        db = str(tmp_path / "garbage.db")
        app = deploy_entry.build_app(port=port, db_path=db,
                                     self_base_url=f"http://127.0.0.1:{backend.port}")
        server, thread = _boot(app, port)
        try:
            c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
            r = c.post("/verify/run")
            assert r.status_code == 200
            assert "UNKNOWN" in r.text
            assert "Traceback" not in r.text
        finally:
            _stop(server, thread)


def test_loopback_read_caps_oversized_body(tmp_path):
    """An over-cap declared Content-Length is refused BEFORE the body is read."""
    from hostapp.verify_page import MAX_WIRE_BYTES, _http_get_json, _now_ms

    def big(conn):
        conn.settimeout(0.5)
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(MAX_WIRE_BYTES + 100).encode() +
                     b"\r\n\r\n" + b" " * (MAX_WIRE_BYTES + 100))

    with _FakeBackend(big) as backend:
        payload, _ = _http_get_json(f"http://127.0.0.1:{backend.port}/api/cart",
                                    None, _now_ms() + 5000)
        assert payload == {"transport_error": "response too large"}


def _drip_headers(conn):
    """Status line AND headers dribbled one byte at a time: header parsing
    is inside the deadline too (a stock http.client would escape it)."""
    conn.settimeout(0.5)
    conn.recv(4096)
    data = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: 56\r\n\r\n"
            b'{"cart_id":"x","version":0,"lines":[],"total_cents":0}')
    for i in range(len(data)):
        conn.sendall(data[i:i + 1])
        time.sleep(0.05)


def _fast_then_trickle_close(conn):
    """Fast headers, then half the body fast, then a dribble - and NO
    Content-Length (close-delimited, the path where http.client drops its
    socket reference and any re-clamp on it is silently skipped)."""
    conn.settimeout(0.5)
    conn.recv(4096)
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n")
    conn.sendall(b'{"cart_id":"x","version":0,')
    rest = b'"lines":[],"total_cents":0}'
    for i in range(len(rest)):
        conn.sendall(rest[i:i + 1])
        time.sleep(0.05)


def test_dribbled_headers_are_bounded_and_unknown(tmp_path):
    with _FakeBackend(_drip_headers) as backend:
        port = _free_port()
        db = str(tmp_path / "driphdr.db")
        app = deploy_entry.build_app(port=port, db_path=db,
                                     self_base_url=f"http://127.0.0.1:{backend.port}")
        server, thread = _boot(app, port)
        try:
            c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
            start = time.monotonic()
            r = c.post("/verify/run")
            elapsed = time.monotonic() - start
            assert r.status_code == 200
            assert "UNKNOWN" in r.text
            assert "Traceback" not in r.text
            assert elapsed < 6.5, f"handler ran {elapsed:.1f}s against dribbled headers"
        finally:
            _stop(server, thread)


def test_close_delimited_trickle_is_bounded(tmp_path):
    with _FakeBackend(_fast_then_trickle_close) as backend:
        port = _free_port()
        db = str(tmp_path / "closedrip.db")
        app = deploy_entry.build_app(port=port, db_path=db,
                                     self_base_url=f"http://127.0.0.1:{backend.port}")
        server, thread = _boot(app, port)
        try:
            c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
            start = time.monotonic()
            r = c.post("/verify/run")
            elapsed = time.monotonic() - start
            assert r.status_code == 200
            assert "UNKNOWN" in r.text
            assert "Traceback" not in r.text
            assert elapsed < 6.5, f"handler ran {elapsed:.1f}s on close-delimited trickle"
        finally:
            _stop(server, thread)


def _header_flood(conn):
    conn.settimeout(1.0)
    conn.recv(4096)
    head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
    pad = b"X-Pad: " + b"p" * 80 + b"\r\n"
    conn.sendall(head + pad * 4000)  # ~344 KB of headers, no body yet


def _chunk_flood(conn):
    conn.settimeout(1.0)
    conn.recv(4096)
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                 b"Transfer-Encoding: chunked\r\n\r\n")
    chunk = b"64\r\n" + b"c" * 100 + b"\r\n"
    for _ in range(3000):  # 300 KB of tiny chunks
        conn.sendall(chunk)


def test_many_small_headers_over_cap_are_refused(tmp_path):
    """Headers accumulate outside the parse buffer: the cumulative wire
    counter, not the buffer length, is what enforces the cap."""
    with _FakeBackend(_header_flood) as backend:
        port = _free_port()
        db = str(tmp_path / "hdrflood.db")
        app = deploy_entry.build_app(port=port, db_path=db,
                                     self_base_url=f"http://127.0.0.1:{backend.port}")
        server, thread = _boot(app, port)
        try:
            c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
            start = time.monotonic()
            r = c.post("/verify/run")
            elapsed = time.monotonic() - start
            assert r.status_code == 200
            assert "UNKNOWN" in r.text
            assert "Traceback" not in r.text
            assert elapsed < 10.0
        finally:
            _stop(server, thread)


def test_many_small_chunks_over_cap_are_refused(tmp_path):
    """The aggregate chunked body is capped, chunk by chunk."""
    with _FakeBackend(_chunk_flood) as backend:
        port = _free_port()
        db = str(tmp_path / "chunkflood.db")
        app = deploy_entry.build_app(port=port, db_path=db,
                                     self_base_url=f"http://127.0.0.1:{backend.port}")
        server, thread = _boot(app, port)
        try:
            c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
            start = time.monotonic()
            r = c.post("/verify/run")
            elapsed = time.monotonic() - start
            assert r.status_code == 200
            assert "UNKNOWN" in r.text
            assert "Traceback" not in r.text
            assert elapsed < 10.0
        finally:
            _stop(server, thread)


def test_negative_lengths_are_protocol_refusals(tmp_path):
    from hostapp.verify_page import _http_get_json, _now_ms

    def neg_cl(conn):
        conn.settimeout(1.0)
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: -1\r\n\r\n")

    def neg_chunk(conn):
        conn.settimeout(1.0)
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                     b"-1\r\n")

    with _FakeBackend(neg_cl) as backend:
        payload, _ = _http_get_json(f"http://127.0.0.1:{backend.port}/api/cart",
                                    None, _now_ms() + 5000)
        assert payload == {"transport_error": "negative content-length"}
    with _FakeBackend(neg_chunk) as backend:
        payload, _ = _http_get_json(f"http://127.0.0.1:{backend.port}/api/cart",
                                    None, _now_ms() + 5000)
        assert payload == {"transport_error": "negative chunk size"}


def test_read_budget_exhaustion_renders_unknown(tmp_path, monkeypatch):
    """The hard read counter is structural: with the budget forced tiny,
    the run renders UNKNOWN (transport errors settle nothing) and ends."""
    import hostapp.verify_page as vp
    monkeypatch.setattr(vp, "MAX_READS_PER_RUN", 1)  # bootstrap only
    url, server, thread, _ = _serve(tmp_path)
    try:
        c = _visitor(url)
        _matching_cart(c)
        start = time.monotonic()
        r = c.post("/verify/run", timeout=30.0)
        elapsed = time.monotonic() - start
        assert r.status_code == 200
        assert "UNKNOWN" in r.text
        assert "Traceback" not in r.text
        assert elapsed < 15.0
    finally:
        _stop(server, thread)


def test_cart_changing_during_settle_stays_honest(base_url):
    c = _visitor(base_url)
    _matching_cart(c)
    stop = threading.Event()

    def mutate():
        while not stop.is_set():
            try:
                cart = c.get("/api/cart", timeout=5).json()
                lines = [l for l in cart["lines"] if l["sku"] == EXTRA_SKU]
                if lines:
                    c.delete(f"/api/cart/lines/{lines[0]['line_id']}",
                             headers={"X-Cart-Version": str(cart["version"])}, timeout=5)
                else:
                    c.post("/api/cart/lines", json={"sku": EXTRA_SKU, "quantity": 1},
                           headers={"X-Cart-Version": str(cart["version"])}, timeout=5)
            except Exception:
                pass

    t = threading.Thread(target=mutate, daemon=True)
    t.start()
    try:
        r = _verify(c)
    finally:
        stop.set()
        t.join(timeout=5)
    assert r.status_code == 200
    assert "Traceback" not in r.text
    assert any(v in r.text for v in ("MATCH", "MISMATCH", "UNKNOWN"))


def test_concurrent_runs_are_admitted_not_deadlocked(tmp_path):
    """Overload at DEFAULT capacity (4): 12 mixed GET+POST workers all finish.

    Admission happens in ASGI middleware before the thread pool, so excess
    workers get a fast 429 instead of piling into pool threads and starving
    the loopback reads the admitted workers need.
    """
    url, server, thread, _ = _serve(tmp_path)
    try:
        barrier = threading.Barrier(12)
        get_codes, post_codes, errors = [], [], []

        def worker():
            try:
                c = _visitor(url)
                _matching_cart(c)
                barrier.wait(timeout=15)
                g = c.get("/verify", timeout=60)
                get_codes.append(g.status_code)
                r = c.post("/verify/run", timeout=60)
                post_codes.append(r.status_code)
            except Exception as exc:
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker) for _ in range(12)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        elapsed = time.monotonic() - start
        assert all(not t.is_alive() for t in threads), "deadlock: a worker never finished"
        assert not errors, f"worker exceptions: {errors}"
        assert len(get_codes) == 12 and len(post_codes) == 12, (
            f"lost responses: {len(get_codes)} GETs, {len(post_codes)} POSTs")
        assert all(code in (200, 429) for code in get_codes + post_codes)
        assert 429 in get_codes + post_codes, "12 workers x 4 slots must reject some"
        assert 200 in post_codes
        assert elapsed < 60.0
    finally:
        _stop(server, thread)


# --------------------------------------------------------------------- #
# Rate limiting and proxy trust policy
# --------------------------------------------------------------------- #

def test_verify_run_rate_limit(tmp_path):
    url, server, thread, _ = _serve(tmp_path, verify_runs_per_window=2,
                                    verify_window_seconds=600)
    try:
        c = httpx.Client(base_url=url, timeout=15.0)
        c.get("/verify")  # establish a session so all runs share one bucket
        assert c.post("/verify/run").status_code == 200
        assert c.post("/verify/run").status_code == 200
        limited = c.post("/verify/run")
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "verify_rate_limited"
    finally:
        _stop(server, thread)


def test_local_mode_ignores_forwarded_headers_and_loopback_exemption(tmp_path):
    # Local mode: forwarded headers never select identity/scheme and cannot
    # earn the loopback rate exemption.
    url, server, thread, _ = _serve(tmp_path, requests_per_minute=3)
    try:
        c = httpx.Client(base_url=url, timeout=15.0)
        r = c.get("/verify", headers={"X-Forwarded-Proto": "https",
                                      "X-Forwarded-For": "203.0.113.5"})
        sc = r.headers.get("set-cookie", "")
        assert "cp_session" in sc and "Secure" not in sc
        # The initial /verify already consumed one of the three slots.
        for i in range(2):
            assert c.get("/", headers={"X-Forwarded-For": f"8.8.8.{i}"}).status_code == 200
        assert c.get("/", headers={"X-Forwarded-For": "8.8.8.9"}).status_code == 429
    finally:
        _stop(server, thread)


def test_public_mode_requires_valid_configuration(monkeypatch):
    monkeypatch.setenv("CARTPROOF_PROXY_MODE", "render")
    monkeypatch.delenv("CARTPROOF_PUBLIC_ORIGIN", raising=False)
    with pytest.raises(PolicyError):
        ProxyPolicy.from_environment()
    monkeypatch.setenv("CARTPROOF_PUBLIC_ORIGIN", "http://cartproof.example.test")
    with pytest.raises(PolicyError):
        ProxyPolicy.from_environment()
    monkeypatch.setenv("CARTPROOF_PUBLIC_ORIGIN", "https://cartproof.example.test/path")
    with pytest.raises(PolicyError):
        ProxyPolicy.from_environment()
    monkeypatch.setenv("CARTPROOF_PUBLIC_ORIGIN", "https://user:pass@cartproof.example.test")
    with pytest.raises(PolicyError):
        ProxyPolicy.from_environment()
    monkeypatch.setenv("CARTPROOF_TRUST_PROXY", "1")
    with pytest.raises(PolicyError):
        ProxyPolicy.from_environment()
    monkeypatch.delenv("CARTPROOF_TRUST_PROXY")
    monkeypatch.setenv("CARTPROOF_PROXY_MODE", "local")
    with pytest.raises(PolicyError):
        deploy_entry.build_app(proxy_mode="render")
    with pytest.raises(PolicyError):
        ProxyPolicy.render("https://one.example, https://two.example")


def test_public_cf_identity_rate_limit_and_spoof_resistance(tmp_path):
    # Same valid CF identity + changing leftmost/appended XFF is one bucket.
    url, server, thread, _ = _public_serve(tmp_path, requests_per_minute=2)
    try:
        c = httpx.Client(base_url=url, timeout=15.0)
        assert c.get("/", headers=_edge()).status_code == 200
        assert c.get("/", headers=_edge()).status_code == 200
        assert c.get("/", headers=_edge(**{"X-Forwarded-For": "9.9.9.9"})).status_code == 429
        assert c.get("/", headers=_edge(**{"X-Forwarded-For": "9.9.9.10"})).status_code == 429
        # Distinct valid CF identities get independent buckets; IPv6 works.
        assert c.get("/", headers=_edge(cf="198.51.100.7")).status_code == 200
        assert c.get("/", headers=_edge(cf="2001:db8::5")).status_code == 200
        # Missing, malformed, duplicate and XFF-only identities are refused.
        assert c.get("/", headers={"Host": PUBLIC_HOST}).status_code == 403
        assert c.get("/", headers=_edge(cf="not-an-ip")).status_code == 403
        dup = [("Host", PUBLIC_HOST), ("CF-Connecting-IP", "198.51.100.8"),
               ("CF-Connecting-IP", "198.51.100.9")]
        assert c.get("/", headers=dup).status_code == 403
        assert c.get("/", headers={"Host": PUBLIC_HOST,
                                   "X-Forwarded-For": "127.0.0.1"}).status_code == 403
    finally:
        _stop(server, thread)


def test_https_front_cookie_continuity_and_strict_edge(tmp_path):
    url, server, thread, _ = _public_serve(tmp_path, requests_per_minute=1000)
    try:
        # Direct HTTP: Secure cookie is stored but not sent on later HTTP.
        c = httpx.Client(base_url=url, timeout=15.0)
        r = c.get("/verify", headers=_edge())
        sc = r.headers.get("set-cookie", "").lower()
        assert "cp_session" in sc
        assert "; secure" in sc and "; httponly" in sc and "samesite=lax" in sc
        assert "cp_session" in c.cookies
        assert c.get("/api/cart", headers=_edge()).json()["lines"] == []

        # HTTPS-front: normal cookiejar selects the Secure cookie for the
        # fixture origin, and the adapter forwards it to the real loopback app.
        f = _front(url)
        assert f.get("/verify").status_code == 200
        assert f.get("/api/cart").status_code == 200
        assert f._transport.cookie_sent is True
        _front_add(f, BOTTLE_A, 1)
        assert _cart(f)["lines"][0]["sku"] == BOTTLE_A

        with pytest.raises(httpx.TransportError):
            httpx.Client(base_url="https://evil.example",
                         transport=_HTTPSFront(url)).get("/")
        with pytest.raises(ValueError):
            _HTTPSFront("http://example.com")

        # Strict configuration parser.
        for bad in ("https://[", "https://[::1", "https://example.com/path",
                    "https://user@example.com", "https://example.com?x",
                    "https://example.com#x", "https://example.com?",
                    "https://example.com#", " https://example.com",
                    "https://example.com ", "https://example.com:0",
                    "https://example.com:65536", "https://example.com:",
                    "https://example.com\\evil",
                    "https://one.example,https://two.example"):
            with pytest.raises(PolicyError):
                ProxyPolicy.render(bad)
        assert ProxyPolicy.render("https://CARTPROOF.EXAMPLE.TEST:443/").public_origin == PUBLIC_ORIGIN
        assert ProxyPolicy.render("https://[2001:db8::1]:8443").public_host == "[2001:db8::1]:8443"

        # Strict request Host and duplicate authority checks through the
        # composed app (middleware rejects before any route/DB work).
        app = deploy_entry.build_app(port=_free_port(),
                                     db_path=str(tmp_path / "strict.db"),
                                     policy=ProxyPolicy.render(PUBLIC_ORIGIN),
                                     requests_per_minute=1000,
                                     verify_runs_per_window=1000)
        for bad in ("example.com/path", "example.com?x", "example.com#x",
                    "example.com\\evil", "example.com:0", "example.com:65536",
                    "example.com:", "[::1", "user@example.com",
                    "example.com,evil"):
            status = _asgi_status(app, "GET", "/api/cart",
                                  [(b"host", bad.encode()),
                                   (b"cf-connecting-ip", CF_IP.encode())])
            assert 400 <= status < 500, (bad, status)
        assert _asgi_status(app, "GET", "/api/cart",
                            [(b"host", PUBLIC_HOST.encode()),
                             (b"host", b"evil.example"),
                             (b"cf-connecting-ip", CF_IP.encode())]) == 400
        for bad in ("https://", "https://[", "https://cartproof.example.test/path",
                    "https://cartproof.example.test?", "https://cartproof.example.test#",
                    "http://cartproof.example.test", "null",
                    "https://user@cartproof.example.test",
                    "https://cartproof.example.test:0", " https://cartproof.example.test"):
            status = _asgi_status(app, "POST", "/verify/run",
                                  [(b"host", PUBLIC_HOST.encode()),
                                   (b"cf-connecting-ip", CF_IP.encode()),
                                   (b"origin", bad.encode())])
            assert 400 <= status < 500, (bad, status)
        assert _asgi_status(app, "POST", "/verify/run",
                            [(b"host", PUBLIC_HOST.encode()),
                             (b"cf-connecting-ip", CF_IP.encode()),
                             (b"origin", b"https://evil.example"),
                             (b"origin", PUBLIC_ORIGIN.encode())]) == 403
        local_app = deploy_entry.build_app(port=_free_port(),
                                           db_path=str(tmp_path / "local.db"),
                                           policy=ProxyPolicy.local())
        local_status = _asgi_status(local_app, "POST", "/verify/run",
                                    [(b"host", b"127.0.0.1:8000"),
                                     (b"origin", b"https://[")])
        assert local_status == 403
    finally:
        _stop(server, thread)


def test_https_front_public_journeys_and_session_independence(tmp_path):
    url, server, thread, _ = _public_serve(tmp_path, requests_per_minute=1000)
    try:
        a = _front(url)
        _front_add(a, MATCH_SKU, 2); _front_add(a, BOTTLE_A, 1); _front_add(a, BOTTLE_B, 1)
        cart_a = _cart(a)
        assert {l["sku"]: l["quantity"] for l in cart_a["lines"]} == {
            MATCH_SKU: 2, BOTTLE_A: 1, BOTTLE_B: 1}
        assert cart_a["total_cents"] == 19500
        ma = a.post("/verify/run", headers={"Origin": PUBLIC_ORIGIN})
        assert ma.status_code == 200
        pa = _page(ma.text)
        assert "MATCH - your cart satisfies the task" in pa.h2
        assert "MISMATCH - your cart does not satisfy the task" not in pa.h2
        assert len(pa.tables) >= 2
        constraints = pa.tables[0][1:]
        assert len(constraints) == 3
        assert all(row[1] == "confirmed" for row in constraints)
        assert not any("unknown" in row[1] for row in constraints)
        cart_rows = pa.tables[1][1:]
        assert {row[1]: int(row[2]) for row in cart_rows} == {
            MATCH_SKU: 2, BOTTLE_A: 1, BOTTLE_B: 1}
        assert "Cart total: $195.00" in ma.text

        b = _front(url)
        _front_add(b, WRONG_VARIANT_SKU, 2); _front_add(b, BOTTLE_A, 1); _front_add(b, BOTTLE_B, 1)
        cart_b = _cart(b)
        assert {l["sku"]: l["quantity"] for l in cart_b["lines"]} == {
            WRONG_VARIANT_SKU: 2, BOTTLE_A: 1, BOTTLE_B: 1}
        assert cart_b["total_cents"] == 19500
        mb = b.post("/verify/run", headers={"Origin": PUBLIC_ORIGIN})
        assert mb.status_code == 200
        pb = _page(mb.text)
        assert "MISMATCH - your cart does not satisfy the task" in pb.h2
        assert "MATCH - your cart satisfies the task" not in pb.h2
        constraints_b = pb.tables[0][1:]
        item = next(row for row in constraints_b if row[0].startswith("item_quantity"))
        assert item[1] == "contradicted"
        assert "observed 0" in item[2] and "required 2" in item[2]
        assert sum(1 for row in constraints_b if row[1] == "confirmed") == 2
        assert not any("unknown" in row[1] for row in constraints_b)
        cart_rows_b = pb.tables[1][1:]
        assert {row[1]: int(row[2]) for row in cart_rows_b} == {
            WRONG_VARIANT_SKU: 2, BOTTLE_A: 1, BOTTLE_B: 1}

        # A survives B, and A's cart is unchanged.
        ma2 = a.post("/verify/run", headers={"Origin": PUBLIC_ORIGIN})
        assert "MATCH - your cart satisfies the task" in _page(ma2.text).h2
        assert _cart(a) == cart_a

        # Forged cookie gets a fresh empty session, not A or B.
        forged = _front(url)
        forged.cookies.set("cp_session", "0" * 32, domain=PUBLIC_HOST, path="/")
        fv = forged.get("/verify")
        assert "Your cart is empty." in fv.text
        assert _cart(forged)["lines"] == []

        # Negative parser regression: MISMATCH is not MATCH, and the empty
        # cart does not satisfy the journey's exact cart assertion.
        assert "MATCH - your cart satisfies the task" not in _page(mb.text).h2
        assert {l["sku"] for l in _cart(forged)["lines"]} != {
            MATCH_SKU: 2, BOTTLE_A: 1, BOTTLE_B: 1}.keys()
    finally:
        _stop(server, thread)



def test_public_mode_direct_loopback_selfread_is_narrow(tmp_path):
    url, server, thread, _ = _public_serve(tmp_path, requests_per_minute=1000)
    try:
        c = httpx.Client(base_url=url, timeout=15.0)
        assert c.get("/api/cart").status_code == 200  # genuine internal self-read
        assert c.get("/api/cart", headers={"X-Forwarded-For": "127.0.0.1"}).status_code == 403
        assert c.get("/verify").status_code == 403
        assert c.get("/api/cart", headers={"Host": PUBLIC_HOST}).status_code == 403
    finally:
        _stop(server, thread)


def test_https_front_per_session_rate_limit_and_nested_code(tmp_path):
    url, server, thread, _ = _public_serve(tmp_path, requests_per_minute=1000)
    try:
        a = _front(url)
        b = _front(url)
        assert a.get("/verify").status_code == 200
        assert b.get("/verify").status_code == 200
        # Distinct HTTPS cookiejars are distinct sessions under one CF IP.
        _front_add(a, BOTTLE_A, 1)
        _front_add(b, BOTTLE_B, 1)
        assert _cart(a)["lines"][0]["sku"] == BOTTLE_A
        assert _cart(b)["lines"][0]["sku"] == BOTTLE_B
        for _ in range(10):
            assert a.post("/verify/run", headers={"Origin": PUBLIC_ORIGIN}).status_code == 200
        limited = a.post("/verify/run", headers={"Origin": PUBLIC_ORIGIN})
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "verify_rate_limited"
        # B's independent session is not blocked by A's bucket.
        assert b.post("/verify/run", headers={"Origin": PUBLIC_ORIGIN}).status_code == 200
    finally:
        _stop(server, thread)



def test_session_key_churn_is_memory_bounded():
    from hostapp.security import RateLimitMiddleware

    async def dummy_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    mw = RateLimitMiddleware(dummy_app, requests_per_minute=10**9,
                             verify_runs_per_window=10**9, max_keys=512)

    async def drive():
        for i in range(3000):
            scope = {
                "type": "http", "method": "POST", "path": "/verify/run",
                "headers": [(b"cookie", f"cp_session={'%032x' % i}".encode())],
                "client": ("127.0.0.1", 0),
            }
            messages = []

            async def send(m):
                messages.append(m)

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            await mw(scope, receive, send)

    asyncio.run(drive())
    assert 0 < len(mw._verify_hits) <= 512


def test_import_has_no_side_effects(tmp_path, monkeypatch):
    """Importing deploy_entry must not build an app or touch any DB path."""
    import importlib
    import os

    monkeypatch.setenv("CARTPROOF_DB_PATH", str(tmp_path / "should_not_exist.db"))
    importlib.reload(deploy_entry)
    assert not (tmp_path / "should_not_exist.db").exists()
    assert not list(tmp_path.glob("*.db"))


def test_sec_fetch_site_cross_site_rejected(base_url):
    """No Origin is not proof of a non-browser client: the browser's own
    Sec-Fetch-Site signal is honoured when Origin is absent."""
    c = _visitor(base_url)
    r = c.post("/verify/run", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "origin_mismatch"
    r2 = c.post("/verify/run", headers={"Sec-Fetch-Site": "same-origin"})
    assert r2.status_code == 200


def test_slow_body_times_out(base_url):
    """A body that never finishes is abandoned (408), not waited on forever."""
    host, port = base_url.split("//")[1].split(":")
    s = socket.create_connection((host, int(port)), timeout=25)
    try:
        s.sendall(b"POST /verify/run HTTP/1.1\r\nHost: " + host.encode() +
                  b"\r\nContent-Length: 100\r\nConnection: close\r\n\r\n" + b"x" * 10)
        start = time.monotonic()
        resp = b""
        s.settimeout(25)
        while b"\r\n\r\n" not in resp:
            chunk = s.recv(4096)
            if not chunk:
                break
            resp += chunk
        elapsed = time.monotonic() - start
        status = int(resp.split(b" ", 2)[1])
        assert status == 408, resp[:200]
        assert elapsed < 15.0, f"body wait took {elapsed:.1f}s"
    finally:
        s.close()

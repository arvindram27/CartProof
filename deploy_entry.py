"""Host entry point: one process, one port, the storefront + verify page.

NEW file for hosting (WB-H1). It EDITS NOTHING: it composes the existing
store.app.create_app with the new hostapp routers and middleware. The control
app (/_control/*) is never mounted here.

There is deliberately NO module-level ``app`` object: building the app has
side effects (it prepares the database path), so nothing happens at import
time. Start with ``python deploy_entry.py``; the Render start command is the
same (not ``uvicorn deploy_entry:app``).

Configuration (environment, no secrets):
  PORT                     listen port (default 8000)
  CARTPROOF_HOST           bind address (default 0.0.0.0; use 127.0.0.1 for
                           any local rehearsal - loopback only)
  CARTPROOF_DB_PATH        SQLite path (default ./var/cartproof.db). Created
                           if missing; an existing database is only OPENED,
                           never deleted or reseeded (seeding is idempotent
                           and skips a non-empty catalog).
  CARTPROOF_SELF_BASE_URL  loopback base the verify page uses to read carts
                           over the store's own HTTP API
                           (default http://127.0.0.1:<PORT>)
  CARTPROOF_PROXY_MODE     "local" (default) or "render". Local ignores all
                           forwarded/client-identity headers. Render is an
                           explicit public-edge mode.
  CARTPROOF_PUBLIC_ORIGIN  required only for render mode: one canonical HTTPS
                           origin, e.g. https://demo.example. It must have no
                           userinfo, path beyond '/', query or fragment.
  CARTPROOF_TRUST_PROXY    RETIRED legacy switch. A truthy value refuses
                           startup; it is never mapped to generic trust.

  Render mode trusts a single CF-Connecting-IP as the external client identity
  under the provider's documented overwrite/direct-reachability assumption.
  It never uses leftmost X-Forwarded-For, and uvicorn proxy_headers is always
  off so the app remains the single owner of effective identity/scheme. The
  provider assumption and a clean Linux runtime are still deployment-gate
  questions, not proven by offline tests.
"""
from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from hostapp.security import (AdmissionMiddleware, BodyLimitMiddleware,
                              OriginCheckMiddleware, PolicyError, ProxyPolicy,
                              ProxyPolicyMiddleware, RateLimitMiddleware,
                              SecureCookieMiddleware)
from hostapp.verify_page import router as verify_router
from store.app import create_app

ROOT = Path(__file__).resolve().parent


def build_app(port: int | None = None, host: str | None = None,
              db_path: str | None = None, self_base_url: str | None = None,
              policy: ProxyPolicy | None = None,
              proxy_mode: str | None = None,
              public_origin: str | None = None,
              requests_per_minute: int = 240,
              verify_runs_per_window: int = 10, verify_window_seconds: int = 600,
              max_concurrent_verifications: int = 4):
    port = port if port is not None else int(os.environ.get("PORT", "8000"))
    db_path = db_path or os.environ.get(
        "CARTPROOF_DB_PATH", str(ROOT / "var" / "cartproof.db"))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if policy is None:
        if proxy_mode is None and public_origin is None:
            policy = ProxyPolicy.from_environment()
        elif proxy_mode == "local":
            policy = ProxyPolicy.local()
        elif proxy_mode == "render":
            if not public_origin:
                raise PolicyError(
                    "public_origin is required when proxy_mode='render'")
            policy = ProxyPolicy.render(public_origin)
        else:
            raise PolicyError("proxy_mode must be 'local' or 'render'")
    self_base_url = (self_base_url or os.environ.get("CARTPROOF_SELF_BASE_URL")
                     or f"http://127.0.0.1:{port}")

    app = create_app(db_path, str(ROOT / "store" / "seed_catalog.json"))
    app.state.self_base_url = self_base_url
    app.include_router(verify_router)
    # Outermost middleware is added last. ProxyPolicyMiddleware owns the
    # effective scheme and client identity before any other middleware sees
    # the request; uvicorn never independently reinterprets forwarded values.
    app.add_middleware(AdmissionMiddleware,
                       max_concurrent=max_concurrent_verifications)
    app.add_middleware(BodyLimitMiddleware, max_bytes=65536)
    app.add_middleware(OriginCheckMiddleware, policy=policy)
    app.add_middleware(RateLimitMiddleware,
                       requests_per_minute=requests_per_minute,
                       verify_runs_per_window=verify_runs_per_window,
                       verify_window_seconds=verify_window_seconds)
    app.add_middleware(SecureCookieMiddleware)
    app.add_middleware(ProxyPolicyMiddleware, policy=policy)
    return app


def main() -> None:
    host = os.environ.get("CARTPROOF_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    policy = ProxyPolicy.from_environment()
    uvicorn.run(build_app(port=port, policy=policy),
                host=host, port=port, proxy_headers=False)


if __name__ == "__main__":
    main()

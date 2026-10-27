"""FastAPI app factories: the storefront and the localhost-only control app."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import api, control, models, pages


def _make_lifespan(db_path: str, catalog_path: str):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        models.init_and_seed(db_path, catalog_path)
        yield

    return lifespan


def create_app(db_path: str, catalog_path: str) -> FastAPI:
    """The storefront: pages + cart JSON API."""
    app = FastAPI(title="CartProof Mock Store", lifespan=_make_lifespan(db_path, catalog_path))
    app.state.db_path = db_path
    app.state.catalog_path = catalog_path
    app.include_router(api.router)
    app.include_router(pages.router)
    return app


def create_control_app(db_path: str, catalog_path: str) -> FastAPI:
    """The ``/_control/*`` fault/reset app (bound to 127.0.0.1 by the runner)."""
    app = FastAPI(title="CartProof Control", lifespan=_make_lifespan(db_path, catalog_path))
    app.state.db_path = db_path
    app.state.catalog_path = catalog_path
    app.include_router(control.router)
    return app

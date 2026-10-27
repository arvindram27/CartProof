"""Localhost-only fault/reset control endpoints.

These live on their own 127.0.0.1 port (a separate ASGI app, see ``app.py``),
are not linked from any page, and are called only by ``harness/`` and
``tests/``. They are a scoped trust assumption for a local demo harness, not a
security boundary.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import faults, models

router = APIRouter()


class FaultBody(BaseModel):
    name: str
    params: Optional[dict[str, Any]] = None


def _conn(request: Request):
    return models.connect(request.app.state.db_path)


@router.post("/_control/fault")
def set_fault(request: Request, body: FaultBody):
    conn = _conn(request)
    try:
        try:
            merged = faults.set_fault(conn, body.name, body.params)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "unknown_fault", "message": str(exc)}},
            )
        return {"ok": True, "fault": body.name, "params": merged}
    finally:
        conn.close()


@router.post("/_control/reset")
def reset(request: Request):
    conn = _conn(request)
    try:
        faults.reset_store(conn, request.app.state.catalog_path)
        return {"ok": True}
    finally:
        conn.close()

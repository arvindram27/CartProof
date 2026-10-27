"""Cart JSON API with transactional, version-guarded mutations.

Runtime code in ``adapter/`` and ``agents/`` never imports this module; it
reaches the store only over HTTP, exactly like a real agent would.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import models

router = APIRouter()


class AddLineBody(BaseModel):
    sku: str
    quantity: int = Field(ge=1, le=99)


class UpdateLineBody(BaseModel):
    quantity: int = Field(ge=1, le=99)


def _conn(request: Request):
    return models.connect(request.app.state.db_path)


def _error(status: int, code: str, message: str, extra: Optional[dict] = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if extra:
        body["error"].update(extra)
    return JSONResponse(status_code=status, content=body)


def _expected_version(request: Request) -> int:
    raw = request.headers.get("x-cart-version")
    if raw is None:
        raise _HeaderError(400, "missing_version", "X-Cart-Version header is required")
    try:
        v = int(raw)
    except ValueError:
        raise _HeaderError(400, "invalid_version", "X-Cart-Version must be an integer")
    if v < 0:
        raise _HeaderError(400, "invalid_version", "X-Cart-Version must be >= 0")
    return v


class _HeaderError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _ensure_session(conn, request: Request, response: Response) -> str:
    cart_id, sid, _created = models.get_or_create_session(
        conn, request.cookies.get("cp_session")
    )
    # Always (re)set the cookie so the client converges on a valid session.
    response.set_cookie("cp_session", sid, httponly=True, samesite="lax")
    return cart_id


@router.get("/api/cart")
def get_cart(request: Request, response: Response):
    conn = _conn(request)
    try:
        cart_id = _ensure_session(conn, request, response)
        return models.read_cart_view(conn, cart_id)
    finally:
        conn.close()


@router.post("/api/cart/lines")
def add_line(request: Request, response: Response, body: AddLineBody):
    conn = _conn(request)
    try:
        cart_id = _ensure_session(conn, request, response)
        expected = _expected_version(request)
        requested_sku = body.sku
        requested_qty = body.quantity

        if models.get_variant(conn, requested_sku) is None:
            return _error(400, "unknown_sku", f"unknown sku: {requested_sku}")

        faults = models.get_faults(conn)

        if "silent-drop" in faults:
            # Pretend success but write nothing and do not bump the version.
            cart = models.read_persisted_cart(conn, cart_id)
            return {
                "version": cart["version"],
                "lines": cart["lines"],
                "total_cents": cart["total_cents"],
                "added": {"line_id": uuid.uuid4().hex, "sku": requested_sku, "quantity": requested_qty},
            }

        effective_sku = requested_sku
        if "wrong-variant" in faults:
            to_sku = faults["wrong-variant"].get("to_sku")
            if to_sku and models.get_variant(conn, to_sku) is not None:
                effective_sku = to_sku
            else:
                sibling = models.confusable_sibling(conn, requested_sku)
                if sibling:
                    effective_sku = sibling

        effective_qty = requested_qty
        if "quantity-drift" in faults:
            delta = int(faults["quantity-drift"].get("delta", 1))
            effective_qty = max(1, min(99, requested_qty + delta))

        variant = models.get_variant(conn, effective_sku)
        unit_price = variant["price_cents"]

        try:
            result = models.add_line_guarded(
                conn, cart_id, expected, effective_sku, effective_qty, unit_price
            )
        except models.VersionConflict as exc:
            return _error(
                409, "version_conflict", str(exc), {"current_version": exc.current_version}
            )

        cart = models.read_persisted_cart(conn, cart_id)
        return {
            "version": cart["version"],
            "lines": cart["lines"],
            "total_cents": cart["total_cents"],
            "added": {
                "line_id": result["line_id"],
                "sku": requested_sku,
                "quantity": requested_qty,
            },
        }
    except _HeaderError as exc:
        return _error(exc.status, exc.code, exc.message)
    finally:
        conn.close()


@router.patch("/api/cart/lines/{line_id}")
def update_line(request: Request, response: Response, line_id: str, body: UpdateLineBody):
    conn = _conn(request)
    try:
        cart_id = _ensure_session(conn, request, response)
        expected = _expected_version(request)

        existing = conn.execute(
            "SELECT line_id FROM cart_lines WHERE cart_id = ? AND line_id = ?",
            (cart_id, line_id),
        ).fetchone()
        if existing is None:
            return _error(404, "missing_line", f"no such cart line: {line_id}")

        try:
            models.update_line_guarded(conn, cart_id, expected, line_id, body.quantity)
        except models.VersionConflict as exc:
            return _error(
                409, "version_conflict", str(exc), {"current_version": exc.current_version}
            )

        return models.read_persisted_cart(conn, cart_id)
    except _HeaderError as exc:
        return _error(exc.status, exc.code, exc.message)
    finally:
        conn.close()


@router.delete("/api/cart/lines/{line_id}")
def delete_line(request: Request, response: Response, line_id: str):
    conn = _conn(request)
    try:
        cart_id = _ensure_session(conn, request, response)
        expected = _expected_version(request)

        existing = conn.execute(
            "SELECT line_id FROM cart_lines WHERE cart_id = ? AND line_id = ?",
            (cart_id, line_id),
        ).fetchone()
        if existing is None:
            return _error(404, "missing_line", f"no such cart line: {line_id}")

        try:
            models.delete_line_guarded(conn, cart_id, expected, line_id)
        except models.VersionConflict as exc:
            return _error(
                409, "version_conflict", str(exc), {"current_version": exc.current_version}
            )

        return models.read_persisted_cart(conn, cart_id)
    except _HeaderError as exc:
        return _error(exc.status, exc.code, exc.message)
    finally:
        conn.close()

"""Server-rendered routes: /, /product/{id}, /cart."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.templating import Jinja2Templates

from . import models

router = APIRouter()

_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


def _conn(request: Request):
    return models.connect(request.app.state.db_path)


def _ensure_session(conn, request: Request, response: Response) -> str:
    cart_id, sid, _created = models.get_or_create_session(
        conn, request.cookies.get("cp_session")
    )
    response.set_cookie("cp_session", sid, httponly=True, samesite="lax")
    return cart_id


@router.get("/")
def index(request: Request, response: Response, q: str = ""):
    conn = _conn(request)
    try:
        _ensure_session(conn, request, response)
        products = models.list_products(conn, q)
        return templates.TemplateResponse(
            request, "index.html", {"products": products, "q": q}
        )
    finally:
        conn.close()


@router.get("/product/{product_id}")
def product(request: Request, response: Response, product_id: str):
    conn = _conn(request)
    try:
        cart_id = _ensure_session(conn, request, response)
        p = models.get_product(conn, product_id)
        if p is None:
            raise HTTPException(status_code=404, detail="no such product")
        version = models.read_persisted_cart(conn, cart_id)["version"]
        return templates.TemplateResponse(
            request,
            "product.html",
            {"product": p, "cart_version": version},
        )
    finally:
        conn.close()


@router.get("/cart")
def cart(request: Request, response: Response):
    conn = _conn(request)
    try:
        cart_id = _ensure_session(conn, request, response)
        view = models.read_cart_view(conn, cart_id)
        lines = view["lines"]
        total = view["total_cents"]
        # dom-api-mismatch is applied only to the rendered DOM, not the API.
        if "dom-api-mismatch" in models.get_faults(conn) and lines:
            lines = lines[:-1]
            total = sum(l["line_total_cents"] for l in lines)
        return templates.TemplateResponse(
            request,
            "cart.html",
            {
                "lines": lines,
                "total_cents": total,
                "cart_version": view["version"],
            },
        )
    finally:
        conn.close()

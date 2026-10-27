"""Playwright actions that drive the storefront.

Each action returns its own confirmation as an :class:`Evidence`. This module
reaches the store only over HTTP through a browser; it never imports from
``store/``.
"""
from __future__ import annotations

from typing import Any

from .evidence import Evidence


class BrowserSession:
    """Owns a Playwright browser context and exposes storefront actions."""

    def __init__(self, base_url: str, headed: bool = False):
        from playwright.sync_api import sync_playwright

        self.base_url = base_url.rstrip("/")
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=not headed)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()

    def close(self) -> None:
        try:
            self.context.close()
        except Exception:
            pass
        try:
            self.browser.close()
        except Exception:
            pass
        try:
            self._pw.stop()
        except Exception:
            pass

    def _goto(self, path: str) -> None:
        self.page.goto(f"{self.base_url}{path}")
        self.page.wait_for_load_state("networkidle")

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def search(self, query: str) -> Evidence:
        self._goto("/")
        self.page.fill("#q", query)
        self.page.press("#q", "Enter")
        self.page.wait_for_load_state("networkidle")
        results = self._read_search_results()
        return Evidence(
            source="action_confirmation",
            payload={"action": "search", "query": query, "results": results},
        )

    def open_product(self, product_id: str) -> Evidence:
        self._goto(f"/product/{product_id}")
        name = self.page.get_attribute("h1[data-name]", "data-name")
        skus = self.page.eval_on_selector_all(
            "input[name=variant]", "els => els.map(e => e.value)"
        )
        return Evidence(
            source="action_confirmation",
            payload={
                "action": "open_product",
                "product_id": product_id,
                "name": name,
                "variant_skus": skus,
            },
        )

    def select_variant(self, sku: str) -> Evidence:
        self.page.check(f'input[name="variant"][value="{sku}"]')
        price = self.page.get_attribute(
            f'input[name="variant"][value="{sku}"]', "data-price-cents"
        )
        return Evidence(
            source="action_confirmation",
            payload={"action": "select_variant", "sku": sku, "price_cents": price},
        )

    def add_to_cart(self, quantity: int = 1) -> Evidence:
        self.page.fill("#quantity", str(quantity))
        self.page.click("#add-to-cart")
        self.page.wait_for_function("document.getElementById('status').textContent.length > 0")
        status = self.page.text_content("#status").strip()
        version = self.page.text_content("#cart-version").strip()
        return Evidence(
            source="action_confirmation",
            payload={
                "action": "add_to_cart",
                "quantity": quantity,
                "status": status,
                "cart_version": version,
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _read_search_results(self) -> list[dict[str, str]]:
        cards = self.page.query_selector_all("li.product-card")
        results = []
        for card in cards:
            pid = card.get_attribute("data-product-id")
            name = card.get_attribute("data-name")
            if pid:
                results.append({"product_id": pid, "name": name})
        return results


#: Convenience constructor used by the harness and tests.
def new_session(base_url: str, headed: bool = False) -> BrowserSession:
    return BrowserSession(base_url, headed=headed)

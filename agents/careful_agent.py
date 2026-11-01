"""Careful-agent baseline (Arm CA): same model actor, its own check/correct loop.

The careful agent inspects and corrects under its own model control inside the
common sandbox. It instantiates no Ledger, Verifier, or RecoveryController and
gets no CartProof ownership or version guards. Its final declaration is the
MODEL's judgment from a "review" call (M5/R2): code executes reads/writes and
presents observed data; it never decides CA's outcome.
"""
from __future__ import annotations

from typing import Any, Optional

from .model_actor import ModelActor
from .model_client import ModelRequest

INSPECTION_STEP_CAP = 16
MAX_INSPECTION_ADDS = 3


class CarefulAgent(ModelActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inspection_actions: list[dict] = []
        self.review_rationale: Optional[str] = None

    def _line_for_sku(self, sku: str) -> Optional[dict]:
        cart = self.observed_cart or {}
        for ln in cart.get("lines") or []:
            if ln.get("sku") == sku:
                return ln
        return None

    def _prepare_product_page(self, sku: str) -> None:
        """Navigate to the SKU's product page so ``#quantity`` exists for an add."""
        entry = next((c for c in self.catalog if c.get("sku") == sku), None)
        if entry is None:
            raise RuntimeError(f"sku not in catalog: {sku!r}")
        ev = self.session.search(entry["product_name"])
        if not (ev.payload or {}).get("results"):
            raise RuntimeError(
                f"no search results for {entry['product_name']!r}")
        self.session.open_product(entry["product_id"])
        self.session.select_variant(sku)

    def inspect_and_review(self) -> dict:
        """Model-controlled inspection/correction loop, then a review call."""
        steps = 0
        adds = 0
        finished = False
        while steps < INSPECTION_STEP_CAP and not finished:
            steps += 1
            cart_lines = (self.observed_cart or {}).get("lines") or []
            available: list[dict] = [{"action": "read_cart"},
                                     {"action": "finish"}]
            for ln in cart_lines:
                available.append({"action": "set_line_quantity",
                                  "sku": ln.get("sku")})
                available.append({"action": "delete_line",
                                  "sku": ln.get("sku")})
            for it in (self.executable_intent or {}).get("items") or []:
                available.append({"action": "add_to_cart", "sku": it["sku"],
                                  "quantity": it["quantity"]})

            req = ModelRequest(
                kind="action", executable_intent=self.executable_intent,
                observed_state={"phase": "careful_inspection",
                                "cart": self.observed_cart},
                available_actions=available,
                completed_items=list(self.completed_items),
                max_output_tokens=self.client.config.max_output_tokens_per_call)
            proposal = self.client.propose(req)
            if proposal.status in ("refusal", "error") \
                    or not isinstance(proposal.payload, dict):
                self.failures.append({
                    "stage": "inspection", "category": "operational",
                    "reason": proposal.error or "bad inspection proposal"})
                break
            payload = proposal.payload
            self.inspection_actions.append(
                {"step": steps, "action": payload.get("action"),
                 "payload": payload})
            choice = self._validate_choice(payload, available)
            if choice is None:
                self.failures.append({
                    "stage": "inspection", "category": "operational",
                    "reason": f"inspection action not in menu: {payload!r}"})
                break
            action = choice["action"]

            if action == "finish":
                finished = True
                break
            if action == "read_cart":
                self._read_cart()
                continue
            if action == "add_to_cart":
                adds += 1
                if adds > MAX_INSPECTION_ADDS:
                    self.failures.append({
                        "stage": "inspection", "category": "operational",
                        "reason": "inspection add cap reached (fault may persist)"})
                    break
                try:
                    self._prepare_product_page(choice["sku"])
                    ev = self.session.add_to_cart(
                        int(choice.get("quantity", 1)), sku=choice["sku"])
                except Exception as exc:
                    self.failures.append({
                        "stage": "inspection", "category": "operational",
                        "reason": f"inspection add failed: {exc}"})
                    break
                self.add_statuses.append(
                    (ev.payload or {}).get("response_status"))
                self._read_cart()
                continue
            if action in ("set_line_quantity", "delete_line"):
                sku = choice.get("sku")
                line = self._line_for_sku(sku)
                if line is None:
                    self.failures.append({
                        "stage": "inspection", "category": "operational",
                        "reason": f"no observed line for {sku!r}"})
                    break
                version = (self.observed_cart or {}).get("version")
                if action == "set_line_quantity":
                    qty = payload.get("quantity")
                    if isinstance(qty, bool) or not isinstance(qty, int) or qty < 1:
                        self.failures.append({
                            "stage": "inspection", "category": "operational",
                            "reason": "set_line_quantity quantity must be an integer >= 1"})
                        break
                    intent_qty = next(
                        (it["quantity"] for it in
                         (self.executable_intent or {}).get("items") or []
                         if it.get("sku") == sku), None)
                    if intent_qty is None:
                        self.failures.append({
                            "stage": "inspection", "category": "operational",
                            "reason": "set_line_quantity sku not in intent"})
                        break
                    if qty > intent_qty:
                        self.failures.append({
                            "stage": "inspection", "category": "operational",
                            "reason": "set_line_quantity exceeds intent quantity"})
                        break
                    self.session.set_line_quantity(
                        line["line_id"], line["sku"], qty, version)
                else:
                    self.session.delete_line(line["line_id"], line["sku"], version)
                self._read_cart()
                continue

            self.failures.append({
                "stage": "inspection", "category": "operational",
                "reason": f"unknown inspection action {action!r}"})
            break

        # Final review: the declaration is the model's judgment (kind="review").
        req = ModelRequest(
            kind="review", executable_intent=self.executable_intent,
            observed_state={"cart": self.observed_cart},
            available_actions=[{"action": "declare", "declaration": "COMPLETE"},
                               {"action": "declare", "declaration": "PARTIAL"}],
            completed_items=list(self.completed_items),
            max_output_tokens=self.client.config.max_output_tokens_per_call)
        proposal = self.client.propose(req)
        declaration = None
        if isinstance(proposal.payload, dict):
            declaration = proposal.payload.get("declaration")
            self.review_rationale = proposal.payload.get("rationale")
        if declaration not in ("COMPLETE", "PARTIAL"):
            self.failures.append({
                "stage": "review", "category": "operational",
                "reason": "no usable final-review declaration"})
            declaration = "PARTIAL"
        return {"declared_outcome": declaration,
                "review_rationale": self.review_rationale}

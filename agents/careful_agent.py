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

    def inspect_and_review(self) -> dict:
        """Model-controlled inspection/correction loop, then a review call."""
        steps = 0
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
            action = payload.get("action")
            self.inspection_actions.append(
                {"step": steps, "action": action, "payload": payload})

            if action == "finish":
                finished = True
                break
            if action == "read_cart":
                self._read_cart()
                continue
            if action == "add_to_cart":
                ev = self.session.add_to_cart(int(payload.get("quantity", 1)),
                                              sku=payload.get("sku"))
                self.add_statuses.append(
                    (ev.payload or {}).get("response_status"))
                self._read_cart()
                continue
            if action in ("set_line_quantity", "delete_line"):
                line = self._line_for_sku(payload.get("sku"))
                if line is None:
                    self.failures.append({
                        "stage": "inspection", "category": "operational",
                        "reason": f"no observed line for {payload.get('sku')!r}"})
                    break
                version = (self.observed_cart or {}).get("version")
                if action == "set_line_quantity":
                    self.session.set_line_quantity(
                        line["line_id"], line["sku"],
                        int(payload.get("quantity", 1)), version)
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

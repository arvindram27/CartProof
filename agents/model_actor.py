"""Model-driven actor: intent proposal + grounded action selection (Brief 5).

Exactly two model touchpoints. Model output is always a labeled proposal;
deterministic code validates every proposal against the enumerated available
actions before anything reaches the adapter. The actor declares outcomes only
from real adapter responses. It reaches the store only through the existing
adapter, with the same privileges as the scripted actor, and emits the same
hook event dictionaries the scripted actor emits.

MB/MC run masked (the action menu is restricted per step to the current item);
MA/CA run unmasked. That masking difference is a claimed CartProof mechanism
under test, not evidence of model capability (M4).
"""
from __future__ import annotations

from typing import Any, Optional

from .intent import validate_proposal
from .model_client import ModelRequest

STEP_CAP_PER_ITEM = 16


class ModelActor:
    def __init__(self, session, client, catalog: list[dict], *,
                 masked: bool, hook=None, model_name: str = "fake-model"):
        self.session = session
        self.client = client  # BudgetedModelClient
        self.catalog = catalog
        self.masked = masked
        self.hook = hook
        self.model_name = model_name

        self.original_request: Optional[str] = None
        self.proposed_intent: Any = None
        self.executable_intent: Optional[dict] = None
        self.outcome_kind: Optional[str] = None
        self.unresolved_points: list = []
        self.rejections: list = []
        self.failures: list[dict] = []
        self.actions: list[dict] = []
        self.completed_items: list[int] = []
        self.add_statuses: list = []
        self.add_status_by_index: dict[int, Any] = {}
        self.observed_cart: Optional[dict] = None  # last read_api_cart payload
        self._stop = False

    # ------------------------------------------------------------------ #
    # Touchpoint 1: intent proposal
    # ------------------------------------------------------------------ #
    def resolve_intent(self, request_text: str) -> dict:
        self.original_request = request_text
        req = ModelRequest(kind="intent", request_text=request_text,
                           catalog=self.catalog,
                           max_output_tokens=self.client.config.max_output_tokens_per_call)
        proposal = self.client.propose(req)
        if proposal.status in ("refusal", "error"):
            self.outcome_kind = "operational_failure"
            self.failures.append({
                "stage": "intent", "category": "operational",
                "reason": proposal.error or proposal.status})
            return {"outcome_kind": self.outcome_kind,
                    "executable_intent": None}

        result = validate_proposal(request_text, self.catalog, proposal.payload)
        self.proposed_intent = proposal.payload
        self.outcome_kind = result["outcome_kind"]
        self.unresolved_points = result["unresolved_points"]
        self.rejections = result["rejections"]
        self.executable_intent = result["executable_intent"]
        if self.outcome_kind == "operational_failure":
            self.failures.append({
                "stage": "intent", "category": "operational",
                "reason": "; ".join(self.rejections) or "intent rejected"})
        return result

    # ------------------------------------------------------------------ #
    # Touchpoint 2: grounded action selection
    # ------------------------------------------------------------------ #
    def _propose_action(self, phase: str, item: dict, available: list[dict],
                        observed: dict) -> Optional[dict]:
        req = ModelRequest(
            kind="action", executable_intent=self.executable_intent,
            observed_state=observed, available_actions=available,
            completed_items=list(self.completed_items),
            max_output_tokens=self.client.config.max_output_tokens_per_call)
        proposal = self.client.propose(req)
        if proposal.status in ("refusal", "error") \
                or not isinstance(proposal.payload, dict):
            self.failures.append({
                "stage": "action", "category": "operational",
                "reason": proposal.error or f"bad action proposal ({proposal.status})"})
            return None
        return proposal.payload

    def _available(self, phase: str, item: dict, ctx: dict) -> list[dict]:
        avail: list[dict] = [{"action": "read_cart"}]
        if phase == "need_search":
            avail.append({"action": "search", "query": item["product_query"]})
        elif phase == "need_open":
            for r in ctx.get("results") or []:
                avail.append({"action": "open_product",
                              "product_id": r["product_id"]})
        elif phase == "need_select":
            skus = ctx.get("variant_skus") or []
            if self.masked:
                if item["target_sku"] in skus:
                    avail.append({"action": "select_variant",
                                  "sku": item["target_sku"]})
            else:
                for s in skus:
                    avail.append({"action": "select_variant", "sku": s})
        elif phase == "need_add":
            if self.masked:
                avail.append({"action": "add_to_cart", "sku": item["target_sku"],
                              "quantity": item["quantity"]})
            else:
                for s in ctx.get("variant_skus") or []:
                    avail.append({"action": "add_to_cart", "sku": s,
                                  "quantity": item["quantity"]})
        return avail

    def _validate_choice(self, payload: dict, available: list[dict]) -> Optional[dict]:
        action = payload.get("action")
        for cand in available:
            if cand["action"] != action:
                continue
            if action == "read_cart":
                return cand
            if action == "finish":
                return cand
            if action in ("set_line_quantity", "delete_line") \
                    and payload.get("sku") == cand["sku"]:
                return cand
            if action == "search" and payload.get("query", cand["query"]) == cand["query"]:
                return cand
            if action == "open_product" and payload.get("product_id") == cand["product_id"]:
                return cand
            if action == "select_variant" and payload.get("sku") == cand["sku"]:
                return cand
            if action == "add_to_cart" and payload.get("sku") == cand["sku"] \
                    and payload.get("quantity", cand["quantity"]) == cand["quantity"]:
                return cand
        return None

    def _read_cart(self):
        from adapter.observations import read_api_cart
        ev = read_api_cart(self.session)
        self.observed_cart = ev.payload
        return ev

    def execute_items(self, items: list[dict]) -> dict:
        """Run the build loop over executable items (scripted-item shape)."""
        for index, item in enumerate(items):
            if self._stop:
                break
            action_id = f"add-{index}"
            item_ctx_base = {"index": index, "sku": item["target_sku"],
                             "quantity": item["quantity"]}

            if self.hook is not None:
                r = self.hook({"kind": "before_add", "action_id": action_id,
                               "item": dict(item_ctx_base)})
                if r and r.get("stop"):
                    self._stop = True
                    break

            phase = "need_search"
            ctx: dict = {}
            steps = 0
            answered = False
            while steps < STEP_CAP_PER_ITEM and not answered and not self._stop:
                steps += 1
                available = self._available(phase, item, ctx)
                observed = {"phase": phase, "target_sku": item["target_sku"],
                            "quantity": item["quantity"],
                            "results": ctx.get("results"),
                            "variant_skus": ctx.get("variant_skus"),
                            "cart": self.observed_cart}
                payload = self._propose_action(phase, item, available, observed)
                if payload is None:
                    self._stop = True
                    break
                choice = self._validate_choice(payload, available)
                if choice is None:
                    self.failures.append({
                        "stage": "action", "category": "operational",
                        "reason": f"action not in available menu: {payload!r}"})
                    self._stop = True
                    break
                action = choice["action"]
                self.actions.append({"step": len(self.actions) + 1,
                                     "item_index": index, "action": action,
                                     "choice": choice})

                if action == "read_cart":
                    self._read_cart()
                    continue
                if action == "search":
                    ev = self.session.search(choice["query"])
                    ctx["results"] = ev.payload.get("results") or []
                    if not ctx["results"]:
                        self.failures.append({
                            "stage": "action", "category": "operational",
                            "reason": "search returned no results"})
                        self._stop = True
                        break
                    phase = "need_open"
                    continue
                if action == "open_product":
                    ev = self.session.open_product(choice["product_id"])
                    ctx["product_id"] = choice["product_id"]
                    ctx["variant_skus"] = ev.payload.get("variant_skus") or []
                    ctx["open_ev"] = ev
                    phase = "need_select"
                    continue
                if action == "select_variant":
                    self.session.select_variant(choice["sku"])
                    ctx["selected_sku"] = choice["sku"]
                    phase = "need_add"
                    continue
                if action == "add_to_cart":
                    item_ctx = {"index": index,
                                "product_id": ctx.get("product_id"),
                                "sku": item["target_sku"],
                                "quantity": item["quantity"]}
                    if self.hook is not None:
                        r = self.hook({"kind": "pre_add", "action_id": action_id,
                                       "item": item_ctx,
                                       "open": ctx["open_ev"].to_dict()
                                       if ctx.get("open_ev") else {}})
                        if r and r.get("stop"):
                            self._stop = True
                            break
                    ev = self.session.add_to_cart(choice["quantity"],
                                                  sku=choice["sku"])
                    status = (ev.payload or {}).get("response_status")
                    self.add_statuses.append(status)
                    self.add_status_by_index[index] = status
                    self.completed_items.append(index)
                    answered = True
                    if self.hook is not None:
                        r = self.hook({"kind": "after_add", "action_id": action_id,
                                       "item": item_ctx, "add": ev.to_dict()})
                        if r and r.get("stop"):
                            self._stop = True
                    continue

            if not answered and not self._stop:
                self.failures.append({
                    "stage": "action", "category": "operational",
                    "reason": f"step cap reached for item {index}"})
                self._stop = True

        return {"completed_items": list(self.completed_items),
                "add_statuses": list(self.add_statuses)}

    # ------------------------------------------------------------------ #
    # Declarations
    # ------------------------------------------------------------------ #
    def naive_declared(self) -> str:
        """COMPLETE iff every intended item index has a recorded HTTP-200 add.

        Matched by intended item index, not by status count or SKU, so
        duplicate same-SKU items each require their own answered add and an
        early stop can never declare COMPLETE. Zero executed items is PARTIAL.
        """
        items = (self.executable_intent or {}).get("items") or []
        if not items:
            return "PARTIAL"
        if all(self.add_status_by_index.get(i) == 200
               for i in range(len(items))):
            return "COMPLETE"
        return "PARTIAL"

    def record(self) -> dict:
        return {
            "original_request": self.original_request,
            "proposed_intent": self.proposed_intent,
            "executable_intent": self.executable_intent,
            "outcome_kind": self.outcome_kind,
            "unresolved_points": list(self.unresolved_points),
            "rejections": list(self.rejections),
            "failures": list(self.failures),
            "actions": list(self.actions),
            "completed_items": list(self.completed_items),
            "add_statuses": list(self.add_statuses),
            "add_status_by_index": dict(self.add_status_by_index),
        }

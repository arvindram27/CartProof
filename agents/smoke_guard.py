"""Smoke-scoped accounting guard (Brief 7). Wraps BudgetedModelClient only
when the smoke launcher requests it; normal suite behavior is unchanged.

Same-call blocking: on any stop condition the actor receives a sanitized
non-executable error (payload=None) on THAT call, so the uncertain proposal
can never cause a cart action. The underlying call's accounting/evidence is
preserved separately. No extra model call is made by the guard.
"""
from __future__ import annotations

from typing import Any, Optional

from .model_client import ModelProposal, ModelRequest


class SmokeAccountingGuard:
    """Delegates config/propose/summary to the budgeted client.

    transport_kind is trusted from construction ("fake" | "meta"), never
    from response content.
    """

    def __init__(self, budgeted, *, transport_kind: str):
        self._budgeted = budgeted
        self.transport_kind = transport_kind
        self.stopped = False
        self.stop_reason: Optional[str] = None
        self.guard_attempts = 0
        self.guard_locally_refused = 0
        self.last_underlying: Optional[dict] = None

    @property
    def config(self):
        return self._budgeted.config

    def __getattr__(self, name: str):
        return getattr(self._budgeted, name)

    def propose(self, request: ModelRequest, is_retry: bool = False) -> ModelProposal:
        self.guard_attempts += 1
        if self.stopped:
            self.guard_locally_refused += 1
            return ModelProposal(
                kind=request.kind, payload=None, status="error",
                model_name=self._budgeted.config.model_name,
                error=f"smoke_stop:{self.stop_reason}")

        before_attempted = self._budgeted.calls_attempted
        before_made = self._budgeted.calls_made
        before_records = len(self._budgeted.records)
        proposal = self._budgeted.propose(request, is_retry=is_retry)
        made_call = self._budgeted.calls_made > before_made

        new_records = self._budgeted.records[before_records:]
        record = new_records[-1] if new_records else None
        sidecar = None
        if record is not None:
            sidecar = self._budgeted.sidecars.get(record["call_index"])

        reason = self._stop_reason(proposal, record, sidecar, made_call)
        # Preserve the underlying evidence exactly once, separately.
        self.last_underlying = {
            "proposal_status": proposal.status,
            "proposal_error": proposal.error,
            "record": record,
            "sidecar_call_index": record["call_index"] if record else None,
        }
        if reason is not None:
            self.stopped = True
            self.stop_reason = reason
            return ModelProposal(
                kind=request.kind, payload=None, status="error",
                model_name=self._budgeted.config.model_name,
                error=f"smoke_stop:{reason}")
        return proposal

    def _stop_reason(self, proposal, record, sidecar, made_call) -> Optional[str]:
        if proposal.status in ("error", "malformed", "refusal"):
            return f"proposal_{proposal.status}"
        if record is not None and record.get("status") in (
                "missing_usage", "overrun", "error", "refused"):
            return f"wrapper_{record['status']}"
        if self.transport_kind == "meta" and made_call:
            # A made Meta call requires its retained sidecar and trustworthy
            # provider usage. A locally refused pre-call attempt (made_call
            # False) is not a paid call with a missing sidecar.
            if sidecar is None:
                return "meta_sidecar_missing"
            if not sidecar.get("usage_present"):
                return "meta_usage_missing"
            if sidecar.get("usage_malformed"):
                return "meta_usage_malformed"
            if not sidecar.get("usage_detail_trustworthy", True):
                return "meta_usage_untrustworthy"
            if sidecar.get("finish_reason") != "stop":
                return "meta_finish_not_stop"
        # Fake mode: synthetic usage only; no provider sidecar/finish/cost is
        # required. A made record with declared synthetic usage is complete
        # synthetic accounting (handled by the wrapper-record checks above).
        return None

    def summary(self) -> dict:
        base = self._budgeted.summary()
        base["smoke_guard"] = {
            "enabled": True,
            "transport_kind": self.transport_kind,
            "accounting_mode": ("provider" if self.transport_kind == "meta"
                                else "synthetic_offline"),
            "provider_billing_verified": False,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "guard_attempts": self.guard_attempts,
            "guard_locally_refused": self.guard_locally_refused,
            "model_calls_attempted_total": (
                self._budgeted.calls_attempted + self.guard_locally_refused),
            "last_underlying": self.last_underlying,
        }
        return base

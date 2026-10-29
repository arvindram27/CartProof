"""Provider-neutral model client interface, budget wrapper, and fake client.

Offline only (Brief 5): no live transport is implemented here. The interface is
the deliverable, plus the fake client. Budget caps enforced by the wrapper are
best-effort, fail-closed guards - NOT a hard token or dollar guarantee: real
provider tokenization and billing schemas are unverified.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol
from urllib.parse import urlparse


@dataclass
class ModelRequest:
    kind: str  # "intent" | "action" | "review"
    request_text: Optional[str] = None
    catalog: Optional[list] = None
    executable_intent: Optional[dict] = None
    observed_state: Optional[dict] = None
    available_actions: Optional[list] = None
    completed_items: Optional[list] = None
    max_output_tokens: int = 512
    call_index: int = 0


@dataclass
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class ModelProposal:
    kind: str = "intent"
    payload: Any = None
    status: str = "ok"  # ok | malformed | refusal | error | missing_usage
    usage: Optional[Usage] = None
    confidence: Optional[float] = None
    model_name: str = "fake-model"
    error: Optional[str] = None


@dataclass
class ClientConfig:
    enabled: bool = False  # default disabled; existing behaviour unchanged
    provider: Optional[str] = None  # sketch only; no real endpoint named
    base_url: Optional[str] = None
    model_name: str = "fake-model"
    key_source: Optional[dict] = None  # e.g. {"kind": "env", "name": "..."}; never resolved/logged
    max_calls: int = 40
    token_budget: int = 20000
    max_output_tokens_per_call: int = 512


class ModelClient(Protocol):
    def propose(self, request: ModelRequest) -> ModelProposal: ...


def _estimate_input_tokens(request: ModelRequest) -> int:
    """Deterministic local estimate for reservation only (not a tokenizer)."""
    try:
        blob = json.dumps(asdict(request), sort_keys=True, default=str)
    except Exception:
        blob = str(request)
    return max(1, len(blob) // 4)


def _valid_usage(u: Optional[Usage]) -> Optional[int]:
    if u is None:
        return None
    for v in (u.input_tokens, u.output_tokens):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            return None
    return (u.input_tokens or 0) + (u.output_tokens or 0)


class BudgetedModelClient:
    """Fail-closed budget wrapper. Uncertainty always counts against budget."""

    def __init__(self, inner: Any, config: Optional[ClientConfig] = None):
        self.inner = inner
        self.config = config or ClientConfig()
        self.records: list[dict] = []
        self.calls_attempted = 0
        self.calls_made = 0
        self.calls_refused = 0
        self.consumed_tokens = 0
        self.reserved_tokens = 0
        self.overrun = False
        self._stopped = False

    def propose(self, request: ModelRequest, is_retry: bool = False) -> ModelProposal:
        cfg = self.config
        self.calls_attempted += 1
        idx = self.calls_attempted

        def refuse(reason: str) -> ModelProposal:
            self.calls_refused += 1
            self.records.append({
                "call_index": idx, "kind": request.kind, "status": "refused",
                "reason": reason, "reserved": 0, "consumed": 0,
                "declared_usage": None, "is_retry": is_retry,
            })
            return ModelProposal(kind=request.kind, payload=None, status="refusal",
                                 model_name=cfg.model_name, error=reason)

        if not cfg.enabled:
            return refuse("model_disabled")
        if self._stopped:
            return refuse("stopped_after_overrun")
        if cfg.base_url:
            host = urlparse(cfg.base_url).hostname
            if host not in ("127.0.0.1", "localhost"):
                return refuse("non_localhost_destination")
        if self.calls_made >= cfg.max_calls:
            return refuse("max_calls_exceeded")

        max_out = request.max_output_tokens or cfg.max_output_tokens_per_call
        reservation = _estimate_input_tokens(request) + max_out
        if self.consumed_tokens + reservation > cfg.token_budget:
            return refuse("budget_exceeded")

        self.calls_made += 1
        self.reserved_tokens += reservation
        try:
            proposal = self.inner.propose(request)
        except Exception as exc:
            self.consumed_tokens += reservation
            self.records.append({
                "call_index": idx, "kind": request.kind, "status": "error",
                "reason": repr(exc), "reserved": reservation,
                "consumed": reservation, "declared_usage": None,
                "is_retry": is_retry,
            })
            return ModelProposal(kind=request.kind, payload=None, status="error",
                                 model_name=cfg.model_name, error=repr(exc))

        declared = _valid_usage(proposal.usage)
        if declared is None:
            # Missing/malformed usage: the reservation is consumed.
            self.consumed_tokens += reservation
            self.records.append({
                "call_index": idx, "kind": request.kind, "status": "missing_usage",
                "reason": None, "reserved": reservation, "consumed": reservation,
                "declared_usage": None, "is_retry": is_retry,
            })
            return proposal

        if declared > reservation:
            self.consumed_tokens += declared
            self.overrun = True
            self._stopped = True
            self.records.append({
                "call_index": idx, "kind": request.kind, "status": "overrun",
                "reason": "declared usage exceeds reservation",
                "reserved": reservation, "consumed": declared,
                "declared_usage": declared, "is_retry": is_retry,
            })
            return ModelProposal(kind=request.kind, payload=proposal.payload,
                                 status="error", usage=proposal.usage,
                                 confidence=proposal.confidence,
                                 model_name=cfg.model_name,
                                 error="usage_overrun")

        self.consumed_tokens += declared
        self.records.append({
            "call_index": idx, "kind": request.kind, "status": "made",
            "reason": None, "reserved": reservation, "consumed": declared,
            "declared_usage": declared, "is_retry": is_retry,
        })
        return proposal

    def summary(self) -> dict:
        return {
            "model_name": self.config.model_name,
            "enabled": self.config.enabled,
            "calls_attempted": self.calls_attempted,
            "calls_made": self.calls_made,
            "calls_refused": self.calls_refused,
            "retries": sum(1 for r in self.records if r.get("is_retry")),
            "reserved_tokens": self.reserved_tokens,
            "consumed_tokens": self.consumed_tokens,
            "token_budget": self.config.token_budget,
            "overrun": self.overrun,
            "records": list(self.records),
            "note": ("Budget caps are best-effort fail-closed guards, not a hard "
                     "token or dollar guarantee."),
        }


class FakeModelClient:
    """Ordered canned responses. Records every request it receives."""

    def __init__(self, responses: Optional[list] = None, model_name: str = "fake-model"):
        self.responses = list(responses or [])
        self.model_name = model_name
        self.requests: list[dict] = []
        self.declared_usages: list = []

    def propose(self, request: ModelRequest) -> ModelProposal:
        self.requests.append(asdict(request))
        if not self.responses:
            raise RuntimeError("fake client responses exhausted")
        entry = self.responses.pop(0)
        if isinstance(entry, dict) and entry.get("error"):
            raise RuntimeError(str(entry["error"]))
        if isinstance(entry, dict) and entry.get("refusal"):
            self.declared_usages.append(None)
            return ModelProposal(kind=request.kind, payload=None, status="refusal",
                                 model_name=self.model_name, error="canned refusal")
        if isinstance(entry, dict) and "malformed" in entry:
            usage = self._usage(entry.get("usage", {"input_tokens": 10, "output_tokens": 10}))
            self.declared_usages.append(usage)
            return ModelProposal(kind=request.kind, payload="<malformed-output>",
                                 status="ok", usage=usage,
                                 model_name=self.model_name)
        if isinstance(entry, dict) and "payload" in entry:
            payload = entry["payload"]
            usage = self._usage(entry.get("usage", {"input_tokens": 10, "output_tokens": 10}))
            self.declared_usages.append(usage)
            return ModelProposal(kind=request.kind, payload=payload, status="ok",
                                 usage=usage,
                                 confidence=entry.get("confidence"),
                                 model_name=self.model_name)
        # A bare dict/list is the payload itself.
        usage = Usage(input_tokens=10, output_tokens=10)
        self.declared_usages.append(usage)
        return ModelProposal(kind=request.kind, payload=entry, status="ok",
                             usage=usage, model_name=self.model_name)

    @staticmethod
    def _usage(spec) -> Optional[Usage]:
        if spec is None:
            return None
        if isinstance(spec, Usage):
            return spec
        return Usage(input_tokens=spec.get("input_tokens"),
                     output_tokens=spec.get("output_tokens"))

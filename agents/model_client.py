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
    # Brief 6 (all defaulted; existing constructions untouched):
    live_enabled: bool = False
    reasoning_effort_by_kind: dict = field(default_factory=dict)
    price_table: Optional[dict] = None  # dated price table; None -> cost unknown
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 30.0


def is_exact_meta_origin(url: Optional[str]) -> bool:
    """Exact Meta origin: https, api.meta.ai, port 443, /v1, nothing else."""
    if not url:
        return False
    p = urlparse(url)
    return (p.scheme == "https" and p.hostname == "api.meta.ai"
            and (p.port is None or p.port == 443)
            and p.username is None and p.password is None
            and p.path == "/v1" and not p.query and not p.fragment)


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

    def __init__(self, inner: Any, config: Optional[ClientConfig] = None,
                 allow_live: bool = False):
        self.inner = inner
        self.config = config or ClientConfig()
        self.allow_live = bool(allow_live)
        self.records: list[dict] = []
        self.sidecars: dict[int, dict] = {}
        self.calls_attempted = 0
        self.calls_made = 0
        self.calls_refused = 0
        self.consumed_tokens = 0
        self.reserved_tokens = 0
        self.overrun = False
        self._stopped = False

    def _drain_sidecar(self, call_index: int) -> None:
        pop = getattr(self.inner, "pop_sidecar", None)
        if callable(pop):
            try:
                entry = pop()
            except Exception:
                entry = None
            if entry is not None:
                entry = dict(entry)
                entry["call_index"] = call_index
                self.sidecars[call_index] = entry

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
            if cfg.provider == "meta" and cfg.live_enabled and self.allow_live:
                if not is_exact_meta_origin(cfg.base_url):
                    return refuse("origin_not_allowlisted")
            elif urlparse(cfg.base_url).hostname not in ("127.0.0.1", "localhost"):
                return refuse("non_localhost_destination")
        if self.calls_made >= cfg.max_calls:
            return refuse("max_calls_exceeded")

        # One validated output cap for both reserve and send (Condition 2):
        # inspect the PROVIDED raw value first - a bool, non-int, or negative
        # fails closed and never falls through the `or` fallback. Fallback to
        # the configured value only when the request value is absent/None/0.
        provided = request.max_output_tokens
        if provided is not None:
            if isinstance(provided, bool) or not isinstance(provided, int) \
                    or provided < 0:
                self.records.append({
                    "call_index": idx, "kind": request.kind, "status": "error",
                    "reason": "cap_invalid", "reserved": 0, "consumed": 0,
                    "declared_usage": None, "is_retry": is_retry})
                return ModelProposal(kind=request.kind, payload=None,
                                     status="error", model_name=cfg.model_name,
                                     error="cap_invalid")
        max_out = provided if provided else cfg.max_output_tokens_per_call
        is_live = (cfg.provider == "meta" and cfg.live_enabled and self.allow_live)
        if is_live:
            cfg_cap = cfg.max_output_tokens_per_call
            if not isinstance(cfg_cap, int) or isinstance(cfg_cap, bool) \
                    or cfg_cap <= 0 or not isinstance(max_out, int) \
                    or isinstance(max_out, bool) or max_out <= 0 \
                    or max_out > cfg_cap:
                self.records.append({
                    "call_index": idx, "kind": request.kind, "status": "error",
                    "reason": "cap_invalid", "reserved": 0, "consumed": 0,
                    "declared_usage": None, "is_retry": is_retry})
                return ModelProposal(kind=request.kind, payload=None,
                                     status="error", model_name=cfg.model_name,
                                     error="cap_invalid")

        # Backward-compatible estimate hook: an inner client that provides
        # estimate_input_tokens (the Meta transport) counts the full
        # serialized request body; FakeModelClient does not implement it and
        # its behaviour is byte-identical.
        estimator = getattr(self.inner, "estimate_input_tokens", None)
        try:
            estimated = estimator(request) if callable(estimator) \
                else _estimate_input_tokens(request)
        except Exception:
            self.records.append({
                "call_index": idx, "kind": request.kind, "status": "error",
                "reason": "estimate_failed", "reserved": 0, "consumed": 0,
                "declared_usage": None, "is_retry": is_retry})
            return ModelProposal(kind=request.kind, payload=None,
                                 status="error", model_name=cfg.model_name,
                                 error="estimate_failed")
        if not isinstance(estimated, int) or isinstance(estimated, bool) \
                or estimated < 0:
            self.records.append({
                "call_index": idx, "kind": request.kind, "status": "error",
                "reason": "estimate_invalid", "reserved": 0, "consumed": 0,
                "declared_usage": None, "is_retry": is_retry})
            return ModelProposal(kind=request.kind, payload=None,
                                 status="error", model_name=cfg.model_name,
                                 error="estimate_invalid")

        reservation = estimated + max_out
        if self.consumed_tokens + reservation > cfg.token_budget:
            return refuse("budget_exceeded")

        self.calls_made += 1
        self.reserved_tokens += reservation
        try:
            proposal = self.inner.propose(request)
        except Exception as exc:
            self._drain_sidecar(idx)
            self.consumed_tokens += reservation
            self.records.append({
                "call_index": idx, "kind": request.kind, "status": "error",
                "reason": repr(exc), "reserved": reservation,
                "consumed": reservation, "declared_usage": None,
                "is_retry": is_retry,
            })
            return ModelProposal(kind=request.kind, payload=None, status="error",
                                 model_name=cfg.model_name, error=repr(exc))
        self._drain_sidecar(idx)

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
            "sidecar": [self.sidecars[k] for k in sorted(self.sidecars)],
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

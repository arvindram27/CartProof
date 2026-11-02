"""Live Meta Model API transport (Brief 6). Offline-tested only in this brief.

No live call is made by any test here: the transport is exercised exclusively
through httpx.MockTransport with a dummy key. Construction fails closed BEFORE
reading the key unless every live gate passes. No retries of any kind.

Spend honesty: budget reservations are a local character heuristic reconciled
against declared usage - a best-effort fail-closed guard, NOT a hard token or
USD guarantee. There is no provider-side enforcement.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from .model_client import ClientConfig, ModelProposal, ModelRequest, Usage

META_BASE_URL = "https://api.meta.ai/v1"
CHAT_PATH = "/v1/chat/completions"


class MetaConfigError(Exception):
    """Sanitized configuration/gate error. Text carries only a category."""


class LiveTransportNotAuthorized(MetaConfigError):
    pass


class MetaTransportError(Exception):
    """Sanitized transport failure. Text carries only a category."""


def is_exact_meta_origin(url: Optional[str]) -> bool:
    if not url:
        return False
    p = urlparse(url)
    return (p.scheme == "https" and p.hostname == "api.meta.ai"
            and (p.port is None or p.port == 443)
            and p.username is None and p.password is None
            and p.path == "/v1" and not p.query and not p.fragment)


SYSTEM_TEMPLATES = {
    "intent": (
        "You are a proposal engine for a deterministic shopping system. "
        "Interpret the customer's request against the supplied catalog and "
        "return exactly one JSON object matching the schema. Propose only "
        "items, quantities, a budget, and a substitution permission that the "
        "request text explicitly supports. Never invent or relax a budget or "
        "permission. Your proposal is a cross-check; deterministic code decides."),
    "action": (
        "You are a proposal engine for a deterministic shopping system. "
        "Choose exactly one action from the supplied available_actions and "
        "return exactly one JSON object matching the schema. Never invent an "
        "action, SKU, product, quantity, or line that is not in the supplied "
        "data. Deterministic code validates every choice before it runs."),
    "review": (
        "You are judging whether the observed cart satisfies the executable "
        "intent. Return exactly one JSON object with your declaration "
        "(COMPLETE or PARTIAL) and a short rationale. Code records your "
        "declaration as your judgment; an unreadable answer is recorded as a "
        "failure, not as success."),
}

SCHEMAS = {
    "intent": {
        "type": "object", "additionalProperties": False,
        "required": ["items", "budget_max_cents", "substitution_allowed"],
        "properties": {
            "items": {"type": "array", "maxItems": 20, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["sku", "quantity"],
                "properties": {
                    "sku": {"type": "string", "minLength": 1, "maxLength": 64},
                    "quantity": {"type": "integer", "minimum": 1,
                                 "maximum": 99}}}},
            "budget_max_cents": {"type": ["integer", "null"], "minimum": 0},
            "substitution_allowed": {"type": "boolean"}}},
    "action": {
        "type": "object", "additionalProperties": False,
        "required": ["action", "query", "product_id", "sku", "quantity"],
        "properties": {
            "action": {"type": "string", "enum": [
                "search", "open_product", "select_variant", "add_to_cart",
                "read_cart", "set_line_quantity", "delete_line", "finish"]},
            "query": {"type": ["string", "null"], "maxLength": 256},
            "product_id": {"type": ["string", "null"], "maxLength": 64},
            "sku": {"type": ["string", "null"], "maxLength": 64},
            "quantity": {"type": ["integer", "null"], "minimum": 1,
                         "maximum": 99}}},
    "review": {
        "type": "object", "additionalProperties": False,
        "required": ["declaration", "rationale"],
        "properties": {
            "declaration": {"type": "string",
                            "enum": ["COMPLETE", "PARTIAL"]},
            "rationale": {"type": ["string", "null"], "maxLength": 2000}}},
}

_RELEVANT_FIELDS = {
    "search": ("query",),
    "open_product": ("product_id",),
    "select_variant": ("sku",),
    "add_to_cart": ("sku", "quantity"),
    "set_line_quantity": ("sku", "quantity"),
    "delete_line": ("sku",),
    "read_cart": (),
    "finish": (),
}
_ACTION_FIELDS = ("query", "product_id", "sku", "quantity")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def effective_cap(request: ModelRequest, config: ClientConfig) -> int:
    """One validated output cap for both reservation and send (Condition 2).

    The PROVIDED raw value is inspected first: a bool, non-int, or negative
    fails closed (a False must not fall through the `or` fallback). Fallback
    to the configured value only when the request value is absent/None/0.
    """
    provided = request.max_output_tokens
    if provided is not None:
        if isinstance(provided, bool) or not isinstance(provided, int) \
                or provided < 0:
            raise MetaConfigError("cap_invalid")
    if provided:
        return int(provided)
    cfg_cap = config.max_output_tokens_per_call
    if not _is_int(cfg_cap) or cfg_cap <= 0:
        raise MetaConfigError("cap_invalid")
    return int(cfg_cap)


def normalize_action_payload(payload: Any):
    """Classify action fields; reject missing/null relevant fields (Cond. 1).

    Returns (normalized_payload|None, stripped_field_names, error|None).
    Only ACTION payloads are normalized; intent/review are never touched.
    """
    if not isinstance(payload, dict):
        return None, [], "action payload is not an object"
    action = payload.get("action")
    if action not in _RELEVANT_FIELDS:
        return None, [], f"unknown action {action!r}"
    relevant = _RELEVANT_FIELDS[action]
    for f in relevant:
        if payload.get(f) is None:
            return None, [], f"missing_relevant_field:{f}"
    # Type-check relevant fields.
    for f in relevant:
        v = payload[f]
        if f == "quantity":
            if not _is_int(v) or not (1 <= v <= 99):
                return None, [], "invalid_relevant_field:quantity"
        elif not isinstance(v, str) or not v:
            return None, [], f"invalid_relevant_field:{f}"
    stripped = []
    normalized: dict[str, Any] = {"action": action}
    for f in relevant:
        normalized[f] = payload[f]
    for f in _ACTION_FIELDS:
        if f not in relevant and f in payload:
            stripped.append(f)
    return normalized, stripped, None


class MetaModelClient:
    """Meta chat-completions transport. Construction is fully gated."""

    def __init__(self, config: ClientConfig, *, allow_live: bool = False,
                 environ=None, http_client: Optional[httpx.Client] = None,
                 transport: Optional[httpx.BaseTransport] = None,
                 dotenv_fallback=None):
        # Gate order: every check below runs BEFORE the key is read.
        if not config.live_enabled or not allow_live:
            raise LiveTransportNotAuthorized("live_not_authorized")
        if config.provider != "meta":
            raise MetaConfigError("provider_not_meta")
        for name in ("max_calls", "token_budget", "max_output_tokens_per_call"):
            v = getattr(config, name)
            if not _is_int(v) or v <= 0:
                raise MetaConfigError("budget_invalid")
        base_url = config.base_url or META_BASE_URL
        if not is_exact_meta_origin(base_url):
            raise MetaConfigError("origin_not_allowlisted")
        for kind in ("intent", "action", "review"):
            if not config.reasoning_effort_by_kind.get(kind):
                raise MetaConfigError("reasoning_effort_missing")

        env = os.environ if environ is None else environ
        if "MODEL_API_KEY" in env:
            val = env["MODEL_API_KEY"]
            if not isinstance(val, str) or not val.strip():
                raise MetaConfigError("model_api_key_invalid")
            api_key = val
        elif dotenv_fallback is not None:
            from .env_loader import DotEnvError
            try:
                fallback_value = dotenv_fallback()
            except DotEnvError as exc:
                # Only the fixed known categories may cross this boundary;
                # anything else maps to one generic category with no context.
                _known = {"dot_env_missing", "dot_env_empty",
                          "dot_env_too_large", "dot_env_invalid_encoding",
                          "dot_env_invalid_syntax", "dot_env_unsafe_path",
                          "dot_env_read_error"}
                category = str(exc) if str(exc) in _known \
                    else "model_api_key_missing"
                raise MetaConfigError(category) from None
            except Exception:
                raise MetaConfigError("model_api_key_missing") from None
            if not isinstance(fallback_value, str) or not fallback_value.strip():
                raise MetaConfigError("model_api_key_missing")
            api_key = fallback_value
        else:
            api_key = env.get("MODEL_API_KEY")
            if not api_key:
                raise MetaConfigError("model_api_key_missing")

        self.config = config
        self.base_url = base_url
        self._api_key = api_key
        self._sidecars: list[dict] = []
        if http_client is not None:
            self._client = http_client
        else:
            timeout = httpx.Timeout(connect=config.connect_timeout_s,
                                    read=config.read_timeout_s,
                                    write=config.read_timeout_s,
                                    pool=config.connect_timeout_s)
            self._client = httpx.Client(
                transport=transport, follow_redirects=False, trust_env=False,
                timeout=timeout)

    # ------------------------------------------------------------------ #
    # Deterministic request construction (one serialization source)
    # ------------------------------------------------------------------ #
    def build_request_body(self, request: ModelRequest) -> dict:
        cap = effective_cap(request, self.config)
        if request.kind == "intent":
            user = {"kind": "intent", "request_text": request.request_text,
                    "catalog": request.catalog}
        elif request.kind in ("action", "review"):
            user = {"kind": request.kind,
                    "executable_intent": request.executable_intent,
                    "observed_state": request.observed_state,
                    "available_actions": request.available_actions,
                    "completed_items": request.completed_items}
        else:
            raise MetaConfigError("unknown_kind")
        return {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_TEMPLATES[request.kind]},
                {"role": "user", "content": self._canonical(user)},
            ],
            "max_completion_tokens": cap,
            "reasoning_effort": self.config.reasoning_effort_by_kind[request.kind],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": f"{request.kind}_proposal",
                                "strict": True,
                                "schema": SCHEMAS[request.kind]}},
            "stream": False,
            "store": False,
        }

    @staticmethod
    def _canonical(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          default=str)

    @classmethod
    def serialize_body(cls, body: dict) -> str:
        return cls._canonical(body)

    def estimate_input_tokens(self, request: ModelRequest) -> int:
        body = self.build_request_body(request)
        return len(self.serialize_body(body)) // 4

    def pop_sidecar(self) -> Optional[dict]:
        return self._sidecars.pop(0) if self._sidecars else None

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def propose(self, request: ModelRequest) -> ModelProposal:
        body = self.build_request_body(request)
        serialized = self.serialize_body(body)
        sidecar: dict[str, Any] = {
            "kind": request.kind, "model_name": self.config.model_name,
            "reasoning_effort": self.config.reasoning_effort_by_kind.get(
                request.kind),
            "duration_ms": None, "http_status": None, "finish_reason": None,
            "provider_usage": None, "usage_present": False,
            "usage_malformed": False, "usage_detail_trustworthy": True,
            "max_completion_tokens": body["max_completion_tokens"],
            "estimated_cost_usd": None, "cost_label": "ESTIMATED",
            "stripped_fields": [],
        }
        t0 = time.perf_counter()
        try:
            resp = self._client.post(
                self.base_url + "/chat/completions", content=serialized,
                headers={"Authorization": f"Bearer {self._api_key}",
                         "Content-Type": "application/json"})
        except httpx.TimeoutException:
            sidecar["duration_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            self._sidecars.append(sidecar)
            raise MetaTransportError("transport_timeout")
        except httpx.TransportError:
            sidecar["duration_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            self._sidecars.append(sidecar)
            raise MetaTransportError("transport_error")
        sidecar["duration_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        sidecar["http_status"] = resp.status_code

        if resp.status_code != 200:
            self._sidecars.append(sidecar)
            raise MetaTransportError(f"http_{resp.status_code}")

        try:
            data = resp.json()
        except Exception:
            self._sidecars.append(sidecar)
            return ModelProposal(kind=request.kind, payload=resp.text,
                                 status="malformed",
                                 model_name=self.config.model_name,
                                 error="non_json_response")

        usage, usage_info = self._parse_usage(data, sidecar)
        sidecar.update(usage_info)

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or len(choices) != 1 \
                or not isinstance(choices[0], dict):
            self._sidecars.append(sidecar)
            return ModelProposal(kind=request.kind, payload=None,
                                 status="malformed", usage=usage,
                                 model_name=self.config.model_name,
                                 error="bad_choices")
        choice = choices[0]
        sidecar["finish_reason"] = choice.get("finish_reason")
        message = choice.get("message") or {}

        if isinstance(message, dict) and message.get("refusal"):
            self._sidecars.append(sidecar)
            return ModelProposal(kind=request.kind, payload=None,
                                 status="refusal", usage=usage,
                                 model_name=self.config.model_name,
                                 error="provider_refusal")

        finish = choice.get("finish_reason")
        if finish != "stop":
            self._sidecars.append(sidecar)
            return ModelProposal(kind=request.kind, payload=None,
                                 status="error", usage=usage,
                                 model_name=self.config.model_name,
                                 error=f"finish_{finish}")

        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content:
            self._sidecars.append(sidecar)
            return ModelProposal(kind=request.kind, payload=content,
                                 status="malformed", usage=usage,
                                 model_name=self.config.model_name,
                                 error="empty_content")
        try:
            payload = json.loads(content)
        except Exception:
            self._sidecars.append(sidecar)
            return ModelProposal(kind=request.kind, payload=content,
                                 status="malformed", usage=usage,
                                 model_name=self.config.model_name,
                                 error="non_json_content")
        if not isinstance(payload, dict):
            self._sidecars.append(sidecar)
            return ModelProposal(kind=request.kind, payload=payload,
                                 status="malformed", usage=usage,
                                 model_name=self.config.model_name,
                                 error="payload_not_object")

        if request.kind == "action":
            normalized, stripped, err = normalize_action_payload(payload)
            sidecar["stripped_fields"] = stripped
            if normalized is None:
                self._sidecars.append(sidecar)
                return ModelProposal(kind=request.kind, payload=None,
                                     status="malformed", usage=usage,
                                     model_name=self.config.model_name,
                                     error=err)
            payload = normalized
        elif request.kind == "intent":
            if not isinstance(payload.get("items"), list) \
                    or "budget_max_cents" not in payload \
                    or not isinstance(payload.get("substitution_allowed"), bool):
                self._sidecars.append(sidecar)
                return ModelProposal(kind=request.kind, payload=payload,
                                     status="malformed", usage=usage,
                                     model_name=self.config.model_name,
                                     error="intent_shape")
        elif request.kind == "review":
            if payload.get("declaration") not in ("COMPLETE", "PARTIAL") \
                    or "rationale" not in payload:
                self._sidecars.append(sidecar)
                return ModelProposal(kind=request.kind, payload=payload,
                                     status="malformed", usage=usage,
                                     model_name=self.config.model_name,
                                     error="review_shape")

        self._sidecars.append(sidecar)
        return ModelProposal(kind=request.kind, payload=payload, status="ok",
                             usage=usage, model_name=self.config.model_name)

    # ------------------------------------------------------------------ #
    def _parse_usage(self, data: dict, sidecar: dict):
        raw = data.get("usage") if isinstance(data, dict) else None
        if raw is None:
            return None, {"usage_present": False, "usage_malformed": False,
                          "provider_usage": None,
                          "usage_detail_trustworthy": True,
                          "estimated_cost_usd": None}
        if not isinstance(raw, dict):
            return None, {"usage_present": True, "usage_malformed": True,
                          "provider_usage": None,
                          "usage_detail_trustworthy": False,
                          "estimated_cost_usd": None}
        prompt, completion = raw.get("prompt_tokens"), raw.get("completion_tokens")
        if not _is_int(prompt) or prompt < 0 or not _is_int(completion) \
                or completion < 0:
            return None, {"usage_present": True, "usage_malformed": True,
                          "provider_usage": None,
                          "usage_detail_trustworthy": False,
                          "estimated_cost_usd": None}

        trustworthy = True
        total = raw.get("total_tokens")
        if total is not None and (not _is_int(total)
                                  or total != prompt + completion):
            trustworthy = False

        cached: Optional[int] = 0  # absent detail -> 0 (conservative, stated)
        details = raw.get("prompt_tokens_details")
        if details is not None:
            if isinstance(details, dict) and _is_int(details.get("cached_tokens")) \
                    and details["cached_tokens"] >= 0:
                cached = details["cached_tokens"]
                if cached > prompt:
                    trustworthy = False
            else:
                cached = None
                trustworthy = False

        reasoning: Optional[int] = None  # absent -> unknown, never 0
        cdetails = raw.get("completion_tokens_details")
        if isinstance(cdetails, dict) and "reasoning_tokens" in cdetails:
            rv = cdetails["reasoning_tokens"]
            if _is_int(rv) and rv >= 0:
                reasoning = rv
                if reasoning > completion:
                    trustworthy = False
            else:
                trustworthy = False

        provider_usage = {"prompt_tokens": prompt,
                          "completion_tokens": completion,
                          "total_tokens": total if _is_int(total) else None,
                          "cached_tokens": cached,
                          "reasoning_tokens": reasoning}
        cost = self._estimated_cost(prompt, completion, cached) \
            if trustworthy else None
        return Usage(input_tokens=prompt, output_tokens=completion), {
            "usage_present": True, "usage_malformed": False,
            "provider_usage": provider_usage,
            "usage_detail_trustworthy": trustworthy,
            "estimated_cost_usd": cost}

    def _estimated_cost(self, prompt: int, completion: int,
                        cached: Optional[int]) -> Optional[float]:
        table = self.config.price_table
        if not table or cached is None:
            return None
        per_model = (table.get("per_1m_tokens") or {}).get(self.config.model_name)
        if not per_model:
            return None
        try:
            return round(((prompt - cached) * per_model["input"]
                          + cached * per_model["cached_input"]
                          + completion * per_model["output"]) / 1_000_000, 8)
        except Exception:
            return None

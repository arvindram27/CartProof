"""Brief 6 Meta transport tests: mocked httpx, dummy key, zero network."""
from __future__ import annotations

import json

import httpx
import pytest

from agents.meta_client import (MetaConfigError, MetaModelClient,
                                MetaTransportError, LiveTransportNotAuthorized,
                                normalize_action_payload)
from agents.model_client import (BudgetedModelClient, ClientConfig,
                                 FakeModelClient, ModelRequest, ModelProposal,
                                 Usage)
from harness.model_compare import build_model_client

DUMMY_KEY = "dummy-key-for-tests-only"
REASONING = {"intent": "low", "action": "low", "review": "low"}
PRICE_TABLE = {"id": "test-table", "per_1m_tokens": {
    "dummy-model": {"input": 1.0, "cached_input": 0.1, "output": 2.0}}}


class RecordingEnv(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.accessed = []

    def get(self, key, default=None):
        self.accessed.append(key)
        return super().get(key, default)


def _config(**over):
    cfg = ClientConfig(
        enabled=True, provider="meta", base_url="https://api.meta.ai/v1",
        model_name="dummy-model", live_enabled=True,
        reasoning_effort_by_kind=dict(REASONING),
        max_calls=40, token_budget=100000, max_output_tokens_per_call=512)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _meta(handler, config=None, env=None):
    cfg = config or _config()
    client = httpx.Client(transport=httpx.MockTransport(handler),
                          follow_redirects=False, trust_env=False)
    return MetaModelClient(cfg, allow_live=True,
                           environ=env or RecordingEnv(
                               {"MODEL_API_KEY": DUMMY_KEY}),
                           http_client=client)


def _body(payload, usage=None, finish="stop", refusal=None):
    msg = {"content": payload if isinstance(payload, str)
           else json.dumps(payload)}
    if refusal:
        msg["refusal"] = refusal
    body = {"choices": [{"finish_reason": finish, "message": msg}]}
    if usage is not None:
        body["usage"] = usage
    return body


def _usage(prompt=10, completion=5, **extra):
    u = {"prompt_tokens": prompt, "completion_tokens": completion,
         "total_tokens": prompt + completion}
    u.update(extra)
    return u


def _req(kind="intent", cap=512):
    if kind == "intent":
        return ModelRequest(kind="intent", request_text="Add 1 x SKU-A.",
                            catalog=[], max_output_tokens=cap)
    return ModelRequest(kind=kind, executable_intent={"items": []},
                        observed_state={}, available_actions=[],
                        completed_items=[], max_output_tokens=cap)


# ------------------------------------------------------------------ success

def test_success_intent_mapping_and_request_shape():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        captured["raw"] = request.content
        return httpx.Response(200, json=_body(
            {"items": [{"sku": "SKU-A", "quantity": 1}],
             "budget_max_cents": None, "substitution_allowed": False},
            usage=_usage()))

    meta = _meta(handler)
    p = meta.propose(_req())
    assert p.status == "ok"
    assert p.payload["budget_max_cents"] is None  # null preserved (Cond. 1)
    assert p.usage.input_tokens == 10 and p.usage.output_tokens == 5
    assert captured["url"] == "https://api.meta.ai/v1/chat/completions"
    assert captured["auth"] == f"Bearer {DUMMY_KEY}"
    assert captured["body"]["max_completion_tokens"] == 512
    assert captured["body"]["stream"] is False
    assert captured["body"]["store"] is False
    assert "tools" not in captured["body"]
    # Deterministic serialization: captured bytes are the canonical body.
    assert captured["raw"].decode() == MetaModelClient.serialize_body(
        meta.build_request_body(_req()))
    sidecar = meta.pop_sidecar()
    assert sidecar["http_status"] == 200
    assert sidecar["reasoning_effort"] == "low"


def test_review_rationale_null_is_preserved():
    def handler(request):
        return httpx.Response(200, json=_body(
            {"declaration": "PARTIAL", "rationale": None}, usage=_usage()))

    p = _meta(handler).propose(_req("review"))
    assert p.status == "ok"
    assert p.payload == {"declaration": "PARTIAL", "rationale": None}


# ------------------------------------------------------------------ actions

@pytest.mark.parametrize("payload", [
    {"action": "add_to_cart", "sku": "SKU-A", "quantity": None},
    {"action": "set_line_quantity", "sku": "SKU-A", "quantity": None},
    {"action": "select_variant", "sku": None},
    {"action": "open_product"},  # product_id missing entirely
    {"action": "search", "query": None},
])
def test_missing_or_null_relevant_action_fields_reject(payload):
    def handler(request):
        return httpx.Response(200, json=_body(payload, usage=_usage()))

    p = _meta(handler).propose(_req("action"))
    assert p.status == "malformed"
    assert p.payload is None
    assert "missing_relevant_field" in (p.error or "")


def test_irrelevant_nulls_are_stripped():
    payload = {"action": "search", "query": "Prod", "product_id": None,
               "sku": None, "quantity": None}

    def handler(request):
        return httpx.Response(200, json=_body(payload, usage=_usage()))

    meta = _meta(handler)
    p = meta.propose(_req("action"))
    assert p.status == "ok"
    assert p.payload == {"action": "search", "query": "Prod"}
    assert set(meta.pop_sidecar()["stripped_fields"]) == {
        "product_id", "sku", "quantity"}


def test_normalize_action_payload_directly():
    normalized, stripped, err = normalize_action_payload(
        {"action": "add_to_cart", "sku": "SKU-A", "quantity": 2,
         "query": None})
    assert err is None
    assert normalized == {"action": "add_to_cart", "sku": "SKU-A",
                          "quantity": 2}
    assert stripped == ["query"]


# ------------------------------------------------------------------ failures

@pytest.mark.parametrize("status_code", [400, 429, 500])
def test_http_errors_raise_sanitized_and_consume_reservation(status_code):
    def handler(request):
        return httpx.Response(status_code,
                              text=f"provider body containing {DUMMY_KEY}")

    meta = _meta(handler)
    wrapper = BudgetedModelClient(meta, meta.config, allow_live=True)
    p = wrapper.propose(_req())
    assert p.status == "error"
    rec = wrapper.records[-1]
    assert rec["status"] == "error"
    assert f"http_{status_code}" in rec["reason"]
    assert rec["consumed"] == rec["reserved"] > 0
    # Redaction: the dummy key from the provider body appears nowhere.
    assert DUMMY_KEY not in str(wrapper.records)
    assert DUMMY_KEY not in str(wrapper.summary())
    assert DUMMY_KEY not in (p.error or "")


def test_timeout_fails_closed_as_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    wrapper = BudgetedModelClient(_meta(handler), _config(), allow_live=True)
    p = wrapper.propose(_req())
    assert p.status == "error"
    assert wrapper.records[-1]["status"] == "error"
    assert "transport_timeout" in wrapper.records[-1]["reason"]


def test_malformed_json_and_missing_usage():
    def handler(request):
        return httpx.Response(200, text="not json at all")

    p = _meta(handler).propose(_req())
    assert p.status == "malformed"

    def handler2(request):
        return httpx.Response(200, json=_body(
            {"items": [], "budget_max_cents": None,
             "substitution_allowed": False}))  # no usage at all

    wrapper = BudgetedModelClient(_meta(handler2), _config(), allow_live=True)
    p2 = wrapper.propose(_req())
    assert p2.status == "ok"
    rec = wrapper.records[-1]
    assert rec["status"] == "missing_usage"
    assert rec["consumed"] == rec["reserved"] > 0  # never treated as zero


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls",
                                    "function_call"])
def test_finish_reason_variants_fail_closed(finish):
    def handler(request):
        return httpx.Response(200, json=_body(
            {"items": [], "budget_max_cents": None,
             "substitution_allowed": False}, usage=_usage(), finish=finish))

    p = _meta(handler).propose(_req())
    assert p.status == "error"
    assert p.payload is None


def test_refusal_field_maps_to_refusal():
    def handler(request):
        return httpx.Response(200, json=_body("", usage=_usage(),
                                              refusal="cannot comply"))
    p = _meta(handler).propose(_req())
    assert p.status == "refusal"


def test_usage_above_reservation_overruns_and_stops():
    def handler(request):
        return httpx.Response(200, json=_body(
            {"items": [], "budget_max_cents": None,
             "substitution_allowed": False},
            usage=_usage(prompt=999999, completion=999999)))

    wrapper = BudgetedModelClient(_meta(handler), _config(), allow_live=True)
    p = wrapper.propose(_req())
    assert p.status == "error" and p.error == "usage_overrun"
    assert wrapper.summary()["overrun"] is True
    p2 = wrapper.propose(_req())
    assert p2.status == "refusal"


# ------------------------------------------------------------------ usage detail

def test_usage_detail_integrity_and_cost():
    table_cfg = _config(price_table=PRICE_TABLE)

    def make(usage):
        def handler(request):
            return httpx.Response(200, json=_body(
                {"items": [], "budget_max_cents": None,
                 "substitution_allowed": False}, usage=usage))
        return _meta(handler, config=table_cfg)

    # cached absent -> 0 (conservative); reasoning absent -> null (unknown)
    meta = make(_usage(prompt=100, completion=10))
    meta.propose(_req())
    sc = meta.pop_sidecar()
    assert sc["provider_usage"]["cached_tokens"] == 0
    assert sc["provider_usage"]["reasoning_tokens"] is None
    assert sc["estimated_cost_usd"] == round((100 * 1.0 + 10 * 2.0) / 1e6, 8)

    # explicit reasoning 0 stays 0
    meta = make(_usage(completion_tokens_details={"reasoning_tokens": 0}))
    meta.propose(_req())
    assert meta.pop_sidecar()["provider_usage"]["reasoning_tokens"] == 0

    # cached > prompt -> detail untrustworthy, cost null, primary stands
    meta = make(_usage(prompt=5, completion=5,
                       prompt_tokens_details={"cached_tokens": 9}))
    p = meta.propose(_req())
    sc = meta.pop_sidecar()
    assert p.usage.input_tokens == 5
    assert sc["usage_detail_trustworthy"] is False
    assert sc["estimated_cost_usd"] is None

    # total mismatch -> cost null
    meta = make(_usage(prompt=5, completion=5, total_tokens=999))
    meta.propose(_req())
    assert meta.pop_sidecar()["estimated_cost_usd"] is None


# ------------------------------------------------------------------ cap

class _StubInner:
    def __init__(self):
        self.calls = 0

    def propose(self, request):
        self.calls += 1
        return ModelProposal(kind=request.kind, payload={"ok": True},
                             usage=Usage(1, 1))


def test_output_cap_validation_before_send():
    inner = _StubInner()
    wrapper = BudgetedModelClient(inner, _config(), allow_live=True)
    for bad in (False, True, -1, "512", 1.5):
        p = wrapper.propose(ModelRequest(kind="intent", max_output_tokens=bad))
        assert p.status == "error" and p.error == "cap_invalid"
    assert inner.calls == 0

    # request cap above the configured live ceiling fails closed
    p = wrapper.propose(ModelRequest(kind="intent", max_output_tokens=9999))
    assert p.error == "cap_invalid"
    assert inner.calls == 0

    # absent/0 falls back to the configured cap (existing behaviour)
    p = wrapper.propose(ModelRequest(kind="intent", max_output_tokens=0))
    assert p.status == "ok"
    assert inner.calls == 1


def test_reservation_and_send_use_the_same_cap():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_body(
            {"items": [], "budget_max_cents": None,
             "substitution_allowed": False}, usage=_usage()))

    meta = _meta(handler)
    wrapper = BudgetedModelClient(meta, meta.config, allow_live=True)
    req = ModelRequest(kind="intent", request_text="Add 1 x SKU-A.",
                       catalog=[], max_output_tokens=64)
    wrapper.propose(req)
    assert captured["body"]["max_completion_tokens"] == 64
    rec = wrapper.records[-1]
    assert rec["reserved"] == meta.estimate_input_tokens(req) + 64


# ------------------------------------------------------------------ estimates

class _EstimateStub:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def estimate_input_tokens(self, request):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value

    def propose(self, request):
        self.calls += 1
        return ModelProposal(kind=request.kind, payload={"ok": True},
                             usage=Usage(1, 1))


@pytest.mark.parametrize("value", [RuntimeError("boom"), "many", True, -3])
def test_invalid_or_raising_estimate_fails_before_send(value):
    inner = _EstimateStub(value)
    wrapper = BudgetedModelClient(inner, ClientConfig(enabled=True))
    p = wrapper.propose(ModelRequest(kind="intent", max_output_tokens=10))
    assert p.status == "error"
    assert inner.calls == 0  # nothing was sent
    assert wrapper.calls_made == 0
    assert wrapper.records[-1]["consumed"] == 0


def test_meta_estimate_counts_the_full_serialized_body():
    meta = _meta(lambda r: httpx.Response(200, json=_body({}, usage=_usage())))
    req = _req()
    body = meta.build_request_body(req)
    assert meta.estimate_input_tokens(req) == \
        len(MetaModelClient.serialize_body(body)) // 4
    from agents.model_client import _estimate_input_tokens
    assert meta.estimate_input_tokens(req) > _estimate_input_tokens(req)


# ------------------------------------------------------------------ gates

def test_live_gate_fails_before_key_read():
    env = RecordingEnv({"MODEL_API_KEY": DUMMY_KEY})
    with pytest.raises(LiveTransportNotAuthorized):
        MetaModelClient(_config(live_enabled=False), allow_live=True,
                        environ=env)
    assert "MODEL_API_KEY" not in env.accessed

    env2 = RecordingEnv({"MODEL_API_KEY": DUMMY_KEY})
    with pytest.raises(LiveTransportNotAuthorized):
        MetaModelClient(_config(), allow_live=False, environ=env2)
    assert "MODEL_API_KEY" not in env2.accessed


@pytest.mark.parametrize("bad_url", [
    "http://api.meta.ai/v1", "https://sub.api.meta.ai/v1",
    "https://api.meta.ai:8443/v1", "https://user@api.meta.ai/v1",
    "https://api.meta.ai/v2", "https://api.meta.ai/v1?x=1",
    "https://example.com/v1",
])
def test_origin_lock_rejects_everything_but_the_exact_origin(bad_url):
    env = RecordingEnv({"MODEL_API_KEY": DUMMY_KEY})
    with pytest.raises(MetaConfigError):
        MetaModelClient(_config(base_url=bad_url), allow_live=True,
                        environ=env)
    assert "MODEL_API_KEY" not in env.accessed


def _live_script(budget):
    return {"budget": budget,
            "client": {"model_name": "dummy-model",
                       "reasoning_effort_by_kind": dict(REASONING)}}


@pytest.mark.parametrize("budget", [
    {}, {"max_calls": 10},
    {"max_calls": None, "token_budget": 100, "max_output_tokens": 10},
    {"max_calls": True, "token_budget": 100, "max_output_tokens": 10},
    {"max_calls": 0, "token_budget": 100, "max_output_tokens": 10},
    {"max_calls": 10, "token_budget": -5, "max_output_tokens": 10},
    {"max_calls": 10, "token_budget": 100, "max_output_tokens": "10"},
])
def test_missing_or_invalid_budget_fails_before_key_read(budget):
    env = RecordingEnv({"MODEL_API_KEY": DUMMY_KEY})
    with pytest.raises(MetaConfigError):
        build_model_client(_live_script(budget), transport="meta",
                           allow_live=True, environ=env)
    assert "MODEL_API_KEY" not in env.accessed


def test_budget_block_alone_cannot_enable_live():
    env = RecordingEnv({"MODEL_API_KEY": DUMMY_KEY})
    good = _live_script({"max_calls": 10, "token_budget": 10000,
                         "max_output_tokens": 128})
    with pytest.raises(LiveTransportNotAuthorized):
        build_model_client(good, transport="meta", allow_live=False,
                           environ=env)
    assert "MODEL_API_KEY" not in env.accessed


# ------------------------------------------------------------------ fake parity

def test_fake_path_is_unchanged():
    fake = FakeModelClient([{"payload": {"items": []}}])
    assert not hasattr(fake, "estimate_input_tokens")
    assert not hasattr(fake, "pop_sidecar")
    wrapper = BudgetedModelClient(fake, ClientConfig(enabled=True))
    p = wrapper.propose(ModelRequest(kind="intent", max_output_tokens=10))
    assert p.status == "ok"
    rec = wrapper.records[-1]
    assert set(rec.keys()) == {"call_index", "kind", "status", "reason",
                               "reserved", "consumed", "declared_usage",
                               "is_retry"}
    assert wrapper.summary()["sidecar"] == []

"""Model client + budget accounting tests (Brief 5 section 7). Offline only."""
from __future__ import annotations

import pytest

from agents.model_client import (BudgetedModelClient, ClientConfig,
                                 FakeModelClient, ModelRequest, Usage)


def _client(responses, **cfg):
    fake = FakeModelClient(responses)
    config = ClientConfig(enabled=True, **cfg)
    return BudgetedModelClient(fake, config), fake


def _req(kind="intent", text="Add 1 x X."):
    return ModelRequest(kind=kind, request_text=text,
                        max_output_tokens=100)


def test_fake_records_requests_and_budget_reconciles():
    c, fake = _client([{"payload": {"items": []},
                        "usage": {"input_tokens": 5, "output_tokens": 5}}])
    p = c.propose(_req())
    assert p.status == "ok"
    assert len(fake.requests) == 1
    s = c.summary()
    assert s["calls_made"] == 1 and s["consumed_tokens"] == 10


def test_reservation_blocks_over_budget_call_before_making_it():
    c, fake = _client([{"payload": {"items": []}}], token_budget=10)
    p = c.propose(_req())  # reservation = estimate + 100 > 10
    assert p.status == "refusal" and p.error == "budget_exceeded"
    assert fake.requests == []  # the wrapped client was never invoked
    assert c.summary()["calls_refused"] == 1


def test_missing_usage_consumes_the_reservation():
    c, _ = _client([{"payload": {"items": []}, "usage": None}], token_budget=100000)
    c.propose(_req())
    rec = c.records[-1]
    assert rec["status"] == "missing_usage"
    assert rec["consumed"] == rec["reserved"] > 0


def test_error_response_consumes_the_reservation():
    c, _ = _client([{"error": "boom"}], token_budget=100000)
    p = c.propose(_req())
    assert p.status == "error"
    assert c.records[-1]["consumed"] == c.records[-1]["reserved"]


def test_declared_usage_overrun_is_reported_and_stops():
    big = {"payload": {"items": []},
           "usage": {"input_tokens": 100000, "output_tokens": 100000}}
    c, fake = _client([big, {"payload": {"items": []}}], token_budget=1000000)
    p = c.propose(_req())
    assert p.status == "error" and p.error == "usage_overrun"
    assert c.summary()["overrun"] is True
    p2 = c.propose(_req())
    assert p2.status == "refusal"  # stopped after overrun
    assert len(fake.requests) == 1


def test_recorded_retry_consumes_budget():
    c, _ = _client([{"payload": {"items": []}}, {"payload": {"items": []}}],
                   token_budget=100000)
    c.propose(_req())
    c.propose(_req(), is_retry=True)
    s = c.summary()
    assert s["calls_made"] == 2 and s["retries"] == 1
    assert s["consumed_tokens"] == 40


def test_max_calls_cap_refuses():
    c, fake = _client([{"payload": {"items": []}}], max_calls=0)
    p = c.propose(_req())
    assert p.status == "refusal"
    assert fake.requests == []


def test_disabled_by_default_refuses_without_calling():
    fake = FakeModelClient([{"payload": {"items": []}}])
    c = BudgetedModelClient(fake, ClientConfig())  # enabled=False default
    p = c.propose(_req())
    assert p.status == "refusal" and p.error == "model_disabled"
    assert fake.requests == []


def test_non_localhost_destination_is_denied():
    fake = FakeModelClient([{"payload": {"items": []}}])
    c = BudgetedModelClient(fake, ClientConfig(
        enabled=True, base_url="https://example.invalid/v1"))
    p = c.propose(_req())
    assert p.status == "refusal" and p.error == "non_localhost_destination"
    assert fake.requests == []


def test_key_source_is_never_logged():
    fake = FakeModelClient([{"payload": {"items": []}}])
    c = BudgetedModelClient(fake, ClientConfig(
        enabled=True, key_source={"kind": "env", "name": "SOME_KEY_NAME"}))
    c.propose(_req())
    blob = str(c.summary())
    assert "SOME_KEY_NAME" not in blob
    assert "key_source" not in blob

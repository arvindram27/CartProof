"""Trusted pilot manifest - plain Python constants, reviewed identity.

Self-hashed JSON is not authority; this module is.
"""
from __future__ import annotations

MANIFEST_ID = "cartproof-dev-comparison-2026-10-31.1"
ATTEMPT_PROTOCOL_ID = "pilot-attempt-v1"
SOURCE_DIGEST_VERSION = 2
PROTOCOL_VERSION = 1
BASE_URL = "https://api.meta.ai/v1"
PRIOR_SMOKE_ESTIMATE_USD = 0.0132445

LIMITS = {
    "max_calls": 12,
    "token_budget": 20000,
    "max_output_tokens": 4096,
    "worker_timeout_seconds": 360.0,
    "settle_ms": 2000,
    "new_pilot_review_threshold_usd": 3.0,
    "total_planning_allowance_usd": 5.0,
}

PROFILES = {
    "spark-medium": {
        "model_name": "muse-spark-1.3",
        "base_url": BASE_URL,
        "reasoning_effort_by_kind": {"intent": "medium", "action": "medium", "review": "medium"},
        "price_table": {
            "as_of": "2026-10-31",
            "source": "https://dev.meta.ai/docs/pricing-rate-limits",
            "per_1m_tokens": {
                "muse-spark-1.3": {"input": 1.25, "cached_input": 0.15, "output": 4.25},
            },
        },
    },
    "wonder-medium": {
        "model_name": "muse-wonder-1.0",
        "base_url": BASE_URL,
        "reasoning_effort_by_kind": {"intent": "medium", "action": "medium", "review": "medium"},
        "price_table": {
            "as_of": "2026-10-31",
            "source": "https://dev.meta.ai/docs/pricing-rate-limits",
            "per_1m_tokens": {
                "muse-wonder-1.0": {"input": 4.0, "cached_input": 0.20, "output": 20.0},
            },
        },
    },
}

TASKS = {
    "m-clean-seeded": {
        "path": "eval/model/dev/tasks/m-clean-seeded.json",
        "sha256": "35971743fd72ffbb6b833a9660305efeea5d2765c9945bbc723f266f251a1863",
    },
    "m-under-ambiguous-size": {
        "path": "eval/model/dev/tasks/m-under-ambiguous-size.json",
        "sha256": "ebea34030a4b85a6e82b792bad23f8e34509b0c6375df437a7fcd3723e246a61",
    },
    "m-fault-silent-drop": {
        "path": "eval/model/dev/tasks/m-fault-silent-drop.json",
        "sha256": "ff99946feb3a5e9a347fc011c704e536e1b6b37123c7d6182b0da3f534d54481",
    },
    "m-fault-wrong-variant": {
        "path": "eval/model/dev/tasks/m-fault-wrong-variant.json",
        "sha256": "33fcf0e85a885dfd7ca63f17c7c4fe3d22428f8716b53498d5147e03fb271f3b",
    },
}

STAGES = {
    1: (
        {"cell_id": "s1-seeded-ca-spark", "task_id": "m-clean-seeded", "arm": "CA", "profile": "spark-medium"},
        {"cell_id": "s1-seeded-ca-wonder", "task_id": "m-clean-seeded", "arm": "CA", "profile": "wonder-medium"},
        {"cell_id": "s1-ambig-ca-wonder", "task_id": "m-under-ambiguous-size", "arm": "CA", "profile": "wonder-medium"},
        {"cell_id": "s1-ambig-ca-spark", "task_id": "m-under-ambiguous-size", "arm": "CA", "profile": "spark-medium"},
    ),
    2: (
        {"cell_id": "s2-drop-ca-spark", "task_id": "m-fault-silent-drop", "arm": "CA", "profile": "spark-medium"},
        {"cell_id": "s2-drop-ca-wonder", "task_id": "m-fault-silent-drop", "arm": "CA", "profile": "wonder-medium"},
        {"cell_id": "s2-drop-mc-wonder", "task_id": "m-fault-silent-drop", "arm": "MC", "profile": "wonder-medium"},
        {"cell_id": "s2-drop-mc-spark", "task_id": "m-fault-silent-drop", "arm": "MC", "profile": "spark-medium"},
        {"cell_id": "s2-variant-ca-wonder", "task_id": "m-fault-wrong-variant", "arm": "CA", "profile": "wonder-medium"},
        {"cell_id": "s2-variant-ca-spark", "task_id": "m-fault-wrong-variant", "arm": "CA", "profile": "spark-medium"},
        {"cell_id": "s2-variant-mc-spark", "task_id": "m-fault-wrong-variant", "arm": "MC", "profile": "spark-medium"},
        {"cell_id": "s2-variant-mc-wonder", "task_id": "m-fault-wrong-variant", "arm": "MC", "profile": "wonder-medium"},
    ),
}

# Predecessor receipt seam - UNSET until a Stage1 batch is reviewed and accepted
ACCEPTED_STAGE1_RECEIPT_ID: str | None = None
ACCEPTED_STAGE1_MANIFEST_ID: str | None = None
# conservative allowances per profile for gate arithmetic
PROFILE_ALLOWANCE = {
    "spark-medium": round(LIMITS["token_budget"] * 4.25 / 1_000_000, 8),  # output price
    "wonder-medium": round(LIMITS["token_budget"] * 20.0 / 1_000_000, 8),
}

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

# Fixed offline responses fixture for the no-model dry-run only. This module
# is inside the reviewed-source digest, so the pin below is reviewed data and
# the fixture is tamper-evident under the review loop. The live Meta path
# never reads it. A mismatch refuses; there is no fallback.
DRY_RUN_FIXTURE = {
    "path": "eval/model/dev/fake_scripts.json",
    "sha256": "a902b34903734c055fd954843e4a8c437896fbf1564861512b13b617ea3aa160",
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

# Predecessor receipt seam - Stage 1 batch reviewed and accepted (2026-11-02).
# The receipt is reviewed data in this digest-covered module; Stage 2 is admitted
# only when it validates and its file_hashes match the exported Stage 1 bytes.
ACCEPTED_STAGE1_RECEIPT_ID: str | None = "s1-20261102-011438-fb2b7332db67b790"
ACCEPTED_STAGE1_MANIFEST_ID: str | None = "cartproof-dev-comparison-2026-10-31.1"
ACCEPTED_STAGE1_RECEIPT: dict | None = {
    "receipt_id": "s1-20261102-011438-fb2b7332db67b790",
    "manifest_id": "cartproof-dev-comparison-2026-10-31.1",
    "stage": 1,
    "status": "complete",
    "known_cost": 0.16980594999999998,
    "unknown_cost_calls": 0,
    "cells": [
        "s1-seeded-ca-spark",
        "s1-seeded-ca-wonder",
        "s1-ambig-ca-wonder",
        "s1-ambig-ca-spark",
    ],
    "file_hashes": {
        "batch.json": "5606d1120a3dd171fa0f1e723ab169eebdba919e57741ba1e3d34402dd240d5d",
        "ledger.json": "5f5a3c3d33384f081b48926a1fde0a10e3d6caf22bb3a6d2b7fcd92a7729fb16",
        "summary.md": "9c5e7eb2655074512ed750be62a3d9524077e91d034cc3ad020b7a583d7b5466",
        "cells/s1-seeded-ca-spark/run.json": "e66799699969caed8ea3493b39e70f0eb6f5bdcd06f34cb29e9561a320d2e0e2",
        "cells/s1-seeded-ca-spark/evaluated.json": "b9c2d37f9885a18aacdd9cd4e9922ad50830384380e075636c10b5a7f52267c0",
        "cells/s1-seeded-ca-spark/secret-scan.txt": "330814b2932713e1774e2e26be0fca14bedc0f5b518d3f2ba0257fff6f2f634e",
        "cells/s1-seeded-ca-wonder/run.json": "bc70d7210c4311ebe3ac5da9b6de0ca6405366668de6b63df5775b9c96db59f5",
        "cells/s1-seeded-ca-wonder/evaluated.json": "831235e2794b3e26f69edc4fe8cc3f4dcf53878095808c28bd1916e03939d017",
        "cells/s1-seeded-ca-wonder/secret-scan.txt": "330814b2932713e1774e2e26be0fca14bedc0f5b518d3f2ba0257fff6f2f634e",
        "cells/s1-ambig-ca-wonder/run.json": "338b8410dbc25ab6f0fd3d64e41d52dc84a75f0b88af1226153c70c608b44079",
        "cells/s1-ambig-ca-wonder/evaluated.json": "7cf7b020047ca04ff7bd86199beaa89a7047e7cce6d80ec7af3bfc088a1ec598",
        "cells/s1-ambig-ca-wonder/secret-scan.txt": "330814b2932713e1774e2e26be0fca14bedc0f5b518d3f2ba0257fff6f2f634e",
        "cells/s1-ambig-ca-spark/run.json": "a3a2d93087e6054b04f078b9d8f922f625e6ac374e8b784310b7f522c8a357c6",
        "cells/s1-ambig-ca-spark/evaluated.json": "e7882543f3919cf8113fc1026907bb706b01ba10cfab4c9426a756d1c98d7bd7",
        "cells/s1-ambig-ca-spark/secret-scan.txt": "330814b2932713e1774e2e26be0fca14bedc0f5b518d3f2ba0257fff6f2f634e",
        "manifest.json": "d6ef1930dc73ad7043734d9441c30b0fed788e388de09d11f9e00422454b6593",
    },
}

# Audited attempt-successor authority. UNSET by default: a consumed attempt
# still refuses with attempt_already_reserved. When set, this reviewed value
# names exactly one prior attempt and the terminal states it accepts. Any
# change here is a reviewed-source change (this module is in the digest).
#
# Shape:
#   {
#       "grant_id": str,
#       "stage": int,
#       "prior_attempt_id": str,          # 16 lowercase hex chars
#       "accepted_prior_states": (str, ...),  # subset of complete/halted/blocked
#   }
SUCCESSOR_GRANT = {
    "grant_id": "401d472fa2a76e19",
    "stage": 2,
    "prior_attempt_id": "93c018509cba603c",
    "accepted_prior_states": ("complete",),
}
# conservative allowances per profile for gate arithmetic
PROFILE_ALLOWANCE = {
    "spark-medium": round(LIMITS["token_budget"] * 4.25 / 1_000_000, 8),  # output price
    "wonder-medium": round(LIMITS["token_budget"] * 20.0 / 1_000_000, 8),
}

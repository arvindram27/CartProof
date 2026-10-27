"""Evidence records with a provenance label.

The label is a provenance record of *where* the observation came from, not a
strength ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

ALLOWED_SOURCES = {"action_confirmation", "dom_reread", "api_reread", "screenshot"}


@dataclass
class Evidence:
    source: str
    payload: Any
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        if self.source not in ALLOWED_SOURCES:
            raise ValueError(
                f"unknown evidence source: {self.source!r} "
                f"(allowed: {sorted(ALLOWED_SOURCES)})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "payload": self.payload, "timestamp": self.timestamp}

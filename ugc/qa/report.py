from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ORDER = {"pass": 0, "info": 0, "skip": 0, "warn": 1, "fail": 2}


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail | info | skip
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    product: str
    target: str
    checks: list[Check]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def overall(self) -> str:
        worst = max((ORDER[c.status] for c in self.checks), default=0)
        return ("pass", "warn", "fail")[worst]

    def to_dict(self) -> dict:
        return {"product": self.product, "target": self.target, "overall": self.overall,
                "checks": [asdict(c) for c in self.checks], "meta": self.meta}

    def summary(self) -> str:
        icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO", "skip": "SKIP"}
        lines = [f"QA {self.overall.upper()}: {self.product} ({self.target})"]
        lines += [f"  [{icon[c.status]}] {c.name}: {c.detail}" for c in self.checks]
        return "\n".join(lines)

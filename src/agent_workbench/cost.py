from __future__ import annotations

from typing import Dict


class CostTracker:
    def __init__(self):
        self.units: Dict[str, int] = {"steps": 0, "tokens": 0}

    def add_steps(self, n: int = 1) -> None:
        self.units["steps"] += n

    def add_tokens(self, n: int) -> None:
        self.units["tokens"] += n

    def snapshot(self) -> Dict[str, int]:
        return dict(self.units)

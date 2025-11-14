from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class SkillContext:
    session_id: str
    settings: Any


class Skill(Protocol):
    name: str
    schema: Dict[str, Any]

    def run(self, ctx: SkillContext, args: Dict[str, Any]) -> Dict[str, Any]:
        ...

from __future__ import annotations

from typing import Any, Dict, List

from jsonschema import validate, ValidationError

from agent_workbench.settings import Settings
from .base import Skill, SkillContext


class SkillsRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.skills: Dict[str, Skill] = {}
        self.allowed = set(settings.skills.get("allowed", []))

    def load_builtins(self) -> None:
        from .builtin.web import WebFetchSkill
        from .builtin.fs import FSReadSkill, FSWriteSkill
        from .builtin.python_runner import PythonRunSkill
        from .builtin.rag import RagSearchSkill

        for skill in [
            WebFetchSkill(),
            FSReadSkill(self.settings),
            FSWriteSkill(self.settings),
            PythonRunSkill(self.settings),
            RagSearchSkill(self.settings),
        ]:
            if not self.allowed or skill.name in self.allowed:
                self.skills[skill.name] = skill

    def list(self) -> List[str]:
        return sorted(self.skills.keys())

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def execute(self, name: str, ctx: SkillContext, args: Dict[str, Any]) -> Dict[str, Any]:
        skill = self.get(name)
        if not skill:
            return {"success": False, "error": f"Unknown skill: {name}"}
        try:
            validate(instance=args, schema=skill.schema)
        except ValidationError as e:
            return {"success": False, "error": f"Invalid args: {e.message}"}
        return skill.run(ctx, args)

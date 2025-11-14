from __future__ import annotations

from typing import Any, Dict

from agent_workbench.tools.python_runner import PythonRunner
from ..base import SkillContext


class PythonRunSkill:
    name = "python.run"
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(self, settings):
        self.tool = PythonRunner(settings)
        self.timeout_s = settings.safety.get("python_timeout_s", 8)
        self.max_stdout_kb = settings.safety.get("python_max_stdout_kb", 256)

    def run(self, ctx: SkillContext, args: Dict[str, Any]) -> Dict[str, Any]:
        validation = self.tool.validate_code(args["code"])
        if not validation["valid"]:
            return {"success": False, "error": validation["reason"]}
        result = self.tool.run(args["code"], timeout_s=self.timeout_s, max_stdout_kb=self.max_stdout_kb)
        return result

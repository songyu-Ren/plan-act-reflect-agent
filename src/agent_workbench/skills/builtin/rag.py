from __future__ import annotations

from typing import Any, Dict

from agent_workbench.tools.rag import RAGTool
from ..base import SkillContext


class RagSearchSkill:
    name = "rag.search"
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, settings):
        self.tool = RAGTool(settings)

    def run(self, ctx: SkillContext, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.tool.search(args["query"], args.get("k"))

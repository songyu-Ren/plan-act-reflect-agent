from __future__ import annotations

from typing import Any, Dict

from agent_workbench.tools.web import fetch_url
from ..base import SkillContext


class WebFetchSkill:
    name = "web.fetch"
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 1},
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    async def _run_async(self, ctx: SkillContext, args: Dict[str, Any]) -> Dict[str, Any]:
        return await fetch_url(args["url"], args.get("max_chars", 10000))

    def run(self, ctx: SkillContext, args: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio

        return asyncio.run(self._run_async(ctx, args))

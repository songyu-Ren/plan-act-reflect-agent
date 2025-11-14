from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agent_workbench.tools.fs import FilesystemTool
from ..base import SkillContext


class FSBase:
    def __init__(self, settings):
        self.tool = FilesystemTool(settings)
        self.root = Path(settings.safety.get("workspace_root", settings.paths.workspace_dir)).resolve()

    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("Path outside workspace")
        return p


class FSReadSkill(FSBase):
    name = "fs.read"
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def run(self, ctx: SkillContext, args: Dict[str, Any]) -> Dict[str, Any]:
        p = self._resolve(args["path"]).relative_to(self.root)
        return self.tool.read(str(p))


class FSWriteSkill(FSBase):
    name = "fs.write"
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "encoding": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def run(self, ctx: SkillContext, args: Dict[str, Any]) -> Dict[str, Any]:
        p = self._resolve(args["path"]).relative_to(self.root)
        return self.tool.write(str(p), args["content"], args.get("encoding", "utf-8"))

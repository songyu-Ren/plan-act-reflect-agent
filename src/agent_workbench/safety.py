from __future__ import annotations

from pathlib import Path


def ensure_workspace_path(root: str, path: str) -> str:
    r = Path(root).resolve()
    p = (r / path).resolve()
    p.relative_to(r)
    return str(p.relative_to(r))

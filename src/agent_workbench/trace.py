from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List

import orjson


class TraceWriter:
    def __init__(self, export_dir: str):
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        self.export_dir = export_dir

    def new_run(self) -> str:
        return f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def path_for(self, run_id: str) -> str:
        return os.path.join(self.export_dir, f"{run_id}.jsonl")

    def append(self, run_id: str, event: Dict[str, Any]) -> None:
        p = self.path_for(run_id)
        event["ts"] = time.time()
        with open(p, "ab") as f:
            f.write(orjson.dumps(event))
            f.write(b"\n")


class TraceReader:
    def __init__(self, export_dir: str):
        self.export_dir = export_dir

    def read(self, run_id: str) -> Iterator[Dict[str, Any]]:
        p = os.path.join(self.export_dir, f"{run_id}.jsonl")
        with open(p, "rb") as f:
            for line in f:
                yield orjson.loads(line)

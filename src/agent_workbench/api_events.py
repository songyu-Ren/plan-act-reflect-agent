from __future__ import annotations

import asyncio
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .trace import TraceReader
from .settings import Settings


router = APIRouter()
settings = Settings.load()
reader = TraceReader(settings.tracing.get("export_dir", "artifacts/traces"))


@router.get("/events")
async def events(run_id: str):
    async def gen() -> AsyncIterator[str]:
        for ev in reader.read(run_id):
            yield f"data: {ev}\n\n"
            await asyncio.sleep(0.05)
    return StreamingResponse(gen(), media_type="text/event-stream")

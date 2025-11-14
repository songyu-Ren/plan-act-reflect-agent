from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .hitl import GLOBAL_APPROVAL_STORE, ApprovalItem


router = APIRouter()
store = GLOBAL_APPROVAL_STORE


class ApprovalOut(BaseModel):
    id: str
    action: str
    reason: str
    status: str


@router.get("/approvals", response_model=List[ApprovalOut])
async def list_approvals():
    return [ApprovalOut(id=i.id, action=i.action, reason=i.reason, status=i.status) for i in store.list()]


@router.post("/approvals/{approval_id}/approve")
async def approve(approval_id: str):
    if not store.approve(approval_id):
        raise HTTPException(status_code=404, detail="Approval not found")
    return {"ok": True}


@router.post("/approvals/{approval_id}/reject")
async def reject(approval_id: str):
    if not store.reject(approval_id):
        raise HTTPException(status_code=404, detail="Approval not found")
    return {"ok": True}

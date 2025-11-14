from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ApprovalItem:
    id: str
    action: str
    reason: str
    created_at: float
    status: str = "pending"
    step_id: Optional[str] = None


class ApprovalStore:
    def __init__(self):
        self.items: Dict[str, ApprovalItem] = {}

    def create(self, action: str, reason: str, step_id: Optional[str] = None) -> ApprovalItem:
        item = ApprovalItem(id=str(uuid.uuid4()), action=action, reason=reason, created_at=time.time(), step_id=step_id)
        self.items[item.id] = item
        return item

    def list(self) -> List[ApprovalItem]:
        return [i for i in self.items.values() if i.status == "pending"]

    def get(self, approval_id: str) -> Optional[ApprovalItem]:
        return self.items.get(approval_id)

    def approve(self, approval_id: str) -> bool:
        item = self.items.get(approval_id)
        if not item:
            return False
        item.status = "approved"
        return True

    def reject(self, approval_id: str) -> bool:
        item = self.items.get(approval_id)
        if not item:
            return False
        item.status = "rejected"
        return True


GLOBAL_APPROVAL_STORE = ApprovalStore()

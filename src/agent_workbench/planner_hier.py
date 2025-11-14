from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx


@dataclass
class PlanNode:
    id: str
    skill: str
    args: Dict[str, Any]
    status: str = "pending"


class Manager:
    def __init__(self, allowed_skills: List[str], concurrency: int = 1):
        self.allowed = set(allowed_skills)
        self.concurrency = concurrency

    def build_plan(self, goal: str) -> nx.DiGraph:
        g = nx.DiGraph()
        steps: List[PlanNode] = []
        if "save" in goal.lower():
            steps = [
                PlanNode(id=str(uuid.uuid4()), skill="web.fetch", args={"url": "https://example.com", "max_chars": 5000}),
                PlanNode(id=str(uuid.uuid4()), skill="python.run", args={"code": "print('Hello, World!')"}),
                PlanNode(id=str(uuid.uuid4()), skill="fs.write", args={"path": "brief.md", "content": "Example content"}),
            ]
        else:
            steps = [
                PlanNode(id=str(uuid.uuid4()), skill="rag.search", args={"query": goal, "k": 3}),
            ]

        prev = None
        for node in steps:
            if node.skill in self.allowed:
                g.add_node(node.id, data=node)
                if prev:
                    g.add_edge(prev.id, node.id)
                prev = node
        return g


class Worker:
    def __init__(self):
        ...

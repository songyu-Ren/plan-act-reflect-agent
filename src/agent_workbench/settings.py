from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8003
    env: str = "development"


@dataclass
class PathsConfig:
    sqlite_db: str = "artifacts/agent.db"
    vector_index_dir: str = "artifacts/vector_index"
    workspace_dir: str = "workspace"
    logs_dir: str = "artifacts/logs"


@dataclass
class LLMConfig:
    provider: str = "null"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    openai_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class AgentConfig:
    max_steps: int = 10
    allow_tools: List[str] = field(default_factory=lambda: ["web", "fs", "python", "rag"])
    reflection_enabled: bool = True
    planning_style: str = "react"


@dataclass
class RetrievalConfig:
    k: int = 5
    model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class MonitoringConfig:
    latency_buckets: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    enable_metrics: bool = True
    log_level: str = "INFO"


@dataclass
class Settings:
    app: AppConfig = field(default_factory=AppConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    skills: Dict[str, Any] = field(default_factory=dict)
    hitl: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    tracing: Dict[str, Any] = field(default_factory=dict)
    costs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> Settings:
        if config_path is None:
            config_path = os.getenv("AGENT_SETTINGS", "config/settings.yaml")
        
        config_file = Path(config_path)
        if not config_file.exists():
            return cls()
        
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Settings:
        return Settings(
            app=AppConfig(**data.get("app", {})),
            paths=PathsConfig(**data.get("paths", {})),
            llm=LLMConfig(**data.get("llm", {})),
            agent=AgentConfig(**data.get("agent", {})),
            retrieval=RetrievalConfig(**data.get("retrieval", {})),
            monitoring=MonitoringConfig(**data.get("monitoring", {})),
            skills=data.get("skills", {}),
            hitl=data.get("hitl", {}),
            safety=data.get("safety", {}),
            tracing=data.get("tracing", {}),
            costs=data.get("costs", {}),
        )

    def ensure_directories(self) -> None:
        for path_attr in ["sqlite_db", "vector_index_dir", "workspace_dir", "logs_dir"]:
            path = Path(getattr(self.paths, path_attr))
            if path_attr == "sqlite_db":
                path = path.parent
            path.mkdir(parents=True, exist_ok=True)

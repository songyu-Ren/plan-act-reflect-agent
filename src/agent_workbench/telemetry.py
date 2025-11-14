from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from typing import Optional

from agent_workbench.settings import Settings


class MetricsCollector:
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Request metrics
        self.request_count = Counter(
            'aw_requests_total',
            'Total number of requests',
            ['endpoint', 'method', 'status']
        )
        
        self.request_latency = Histogram(
            'aw_request_latency_seconds',
            'Request latency in seconds',
            ['endpoint'],
            buckets=settings.monitoring.latency_buckets
        )
        
        # Token metrics (if available)
        self.token_count = Counter(
            'aw_tokens_total',
            'Total number of tokens processed',
            ['provider', 'role']
        )
        
        # Tool usage metrics
        self.tool_calls = Counter(
            'aw_tool_calls_total',
            'Total number of tool calls',
            ['tool']
        )

        # Skills metrics
        self.skills_calls = Counter(
            'aw_skills_calls_total',
            'Total number of skills calls',
            ['skill', 'status']
        )

        # Planner metrics
        self.planner_steps = Counter(
            'aw_planner_steps_total',
            'Total planner steps',
            ['kind']
        )

        # HITL metrics
        self.hitl_pending = Gauge(
            'aw_hitl_pending_total',
            'Number of pending approvals'
        )
        self.hitl_decisions = Counter(
            'aw_hitl_decisions_total',
            'Total HITL decisions',
            ['decision']
        )

        # Run metrics
        self.runs_total = Counter(
            'aw_runs_total',
            'Total runs',
            ['status']
        )

        # Trace metrics
        self.trace_bytes = Counter(
            'aw_trace_bytes_total',
            'Total bytes written to traces'
        )

        # Cost metrics
        self.cost_units = Counter(
            'aw_cost_units_total',
            'Cost units recorded',
            ['type']
        )
        
        # Agent metrics
        self.agent_steps = Counter(
            'aw_agent_steps_total',
            'Total number of agent steps taken'
        )
        
        self.active_sessions = Gauge(
            'aw_active_sessions',
            'Number of active sessions'
        )
        
        self.vector_documents = Gauge(
            'aw_vector_documents_total',
            'Total number of documents in vector store'
        )
    
    def record_request(self, endpoint: str, method: str, status: int, duration: float) -> None:
        """Record a request metric"""
        self.request_count.labels(endpoint=endpoint, method=method, status=status).inc()
        self.request_latency.labels(endpoint=endpoint).observe(duration)
    
    def record_tokens(self, provider: str, role: str, count: int) -> None:
        """Record token usage"""
        self.token_count.labels(provider=provider, role=role).inc(count)
    
    def record_tool_call(self, tool: str) -> None:
        """Record a tool call"""
        self.tool_calls.labels(tool=tool).inc()

    def record_skill_call(self, skill: str, status: str) -> None:
        self.skills_calls.labels(skill=skill, status=status).inc()

    def record_planner_step(self, kind: str) -> None:
        self.planner_steps.labels(kind=kind).inc()

    def set_hitl_pending(self, count: int) -> None:
        self.hitl_pending.set(count)

    def record_hitl_decision(self, decision: str) -> None:
        self.hitl_decisions.labels(decision=decision).inc()

    def record_run(self, status: str) -> None:
        self.runs_total.labels(status=status).inc()

    def add_trace_bytes(self, n: int) -> None:
        self.trace_bytes.inc(n)

    def add_cost(self, typ: str, n: int) -> None:
        self.cost_units.labels(type=typ).inc(n)
    
    def record_agent_step(self) -> None:
        """Record an agent step"""
        self.agent_steps.inc()
    
    def set_active_sessions(self, count: int) -> None:
        """Set active sessions gauge"""
        self.active_sessions.set(count)
    
    def set_vector_documents(self, count: int) -> None:
        """Set vector documents gauge"""
        self.vector_documents.set(count)
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest()


# Global metrics instance
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics(settings: Settings) -> MetricsCollector:
    """Get or create metrics collector"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector(settings)
    return _metrics_instance

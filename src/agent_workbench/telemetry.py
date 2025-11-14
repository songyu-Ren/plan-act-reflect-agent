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
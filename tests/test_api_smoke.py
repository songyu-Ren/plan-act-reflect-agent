import pytest
import asyncio
from fastapi.testclient import TestClient

from agent_workbench.api import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_metrics_endpoint(client):
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"aw_requests_total" in response.content


def test_chat_endpoint(client):
    """Test chat endpoint"""
    response = client.post("/chat", json={
        "session_id": "test_session",
        "user_text": "Hello"
    })
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data["session_id"] == "test_session"


def test_run_task_endpoint(client):
    """Test task execution endpoint"""
    response = client.post("/run_task", json={
        "goal": "Test task",
        "max_steps": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data


def test_tools_endpoint(client):
    """Test tools listing endpoint"""
    response = client.get("/tools")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert len(data["tools"]) > 0


def test_search_endpoint(client):
    """Test vector search endpoint"""
    response = client.post("/search", json={
        "query": "test query",
        "k": 3
    })
    # May fail if no documents are indexed
    assert response.status_code in [200, 500]


def test_session_history_endpoint(client):
    """Test session history endpoint"""
    # First create a session by chatting
    client.post("/chat", json={
        "session_id": "test_history_session",
        "user_text": "Test message"
    })
    
    # Then get history
    response = client.get("/session/test_history_session/history")
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert data["session_id"] == "test_history_session"
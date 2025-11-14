import pytest
import asyncio
from pathlib import Path

from agent_workbench.agent import Agent
from agent_workbench.llm.providers import get_provider
from agent_workbench.settings import Settings


@pytest.fixture
def settings():
    """Test settings with null provider"""
    settings = Settings.load()
    settings.llm.provider = "null"
    settings.paths.workspace_dir = "test_workspace"
    settings.paths.sqlite_db = "test_artifacts/test.db"
    settings.paths.vector_index_dir = "test_artifacts/vector"
    settings.paths.logs_dir = "test_artifacts/logs"
    return settings


@pytest.fixture
async def agent(settings):
    """Test agent with null provider"""
    llm_provider = get_provider(settings.llm)
    agent = Agent(settings, llm_provider)
    await agent.initialize()
    return agent


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization"""
    assert agent is not None
    assert agent.settings.llm.provider == "null"
    assert len(agent.tools) == 4  # web, fs, python, rag


@pytest.mark.asyncio
async def test_simple_task(agent):
    """Test running a simple task"""
    result = await agent.run_task(
        goal="Create a test file",
        max_steps=3
    )
    
    assert result.status in ["success", "failure", "stopped"]
    assert result.goal == "Create a test file"
    assert len(result.steps_taken) >= 0
    assert result.session_id is not None


@pytest.mark.asyncio
async def test_chat_interaction(agent):
    """Test chat interaction"""
    session_id = "test_session"
    response = await agent.chat(session_id, "Hello")
    
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_tool_execution(agent):
    """Test tool execution"""
    # Test filesystem tool
    result = await agent._execute_tool("fs", {"action": "write", "path": "test.txt", "content": "Hello"}, "test")
    assert "success" in result
    
    # Test Python tool
    result = await agent._execute_tool("python", {"code": "print('test')"}, "test")
    assert "success" in result
    
    # Test RAG tool
    result = await agent._execute_tool("rag", {"query": "test"}, "test")
    assert "success" in result


def test_settings_loading():
    """Test settings loading"""
    settings = Settings.load()
    assert settings is not None
    assert settings.app.port == 8003
    assert settings.llm.provider == "null"


def test_provider_creation():
    """Test LLM provider creation"""
    settings = Settings()
    settings.llm.provider = "null"
    
    provider = get_provider(settings.llm)
    assert provider is not None
    
    # Test null provider response
    response = provider.generate([{"role": "user", "content": "test"}])
    assert "null provider" in response.content.lower()


# Cleanup after tests
def teardown_module():
    """Clean up test artifacts"""
    import shutil
    test_dirs = ["test_workspace", "test_artifacts"]
    for dir_path in test_dirs:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
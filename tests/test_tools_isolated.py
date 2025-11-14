import pytest
from pathlib import Path

from agent_workbench.tools.fs import FilesystemTool
from agent_workbench.tools.python_runner import PythonRunner
from agent_workbench.tools.web import fetch_url, clean_text
from agent_workbench.settings import Settings


@pytest.fixture
def settings():
    """Test settings"""
    settings = Settings()
    settings.paths.workspace_dir = "test_workspace"
    return settings


@pytest.fixture
def fs_tool(settings):
    """Filesystem tool instance"""
    return FilesystemTool(settings)


@pytest.fixture
def python_tool(settings):
    """Python runner instance"""
    return PythonRunner(settings)


class TestFilesystemTool:
    def test_file_operations(self, fs_tool):
        """Test file read/write operations"""
        # Write file
        result = fs_tool.write("test.txt", "Hello, World!")
        assert result["success"] is True
        assert result["path"] == "test.txt"
        
        # Read file
        result = fs_tool.read("test.txt")
        assert result["content"] == "Hello, World!"
        assert "error" not in result
        
        # Check existence
        result = fs_tool.exists("test.txt")
        assert result["exists"] is True
        assert result["type"] == "file"
        
        # Delete file
        result = fs_tool.delete("test.txt")
        assert result["success"] is True
        
        # Check non-existence
        result = fs_tool.exists("test.txt")
        assert result["exists"] is False
    
    def test_directory_operations(self, fs_tool):
        """Test directory operations"""
        # Create directory
        result = fs_tool.create_dir("test_dir")
        assert result["success"] is True
        
        # List directory
        result = fs_tool.list_dir("")
        assert result["success"] is True
        assert any(item["name"] == "test_dir" for item in result["items"])
        
        # Delete directory
        result = fs_tool.delete("test_dir")
        assert result["success"] is True
    
    def test_security_constraints(self, fs_tool):
        """Test security constraints"""
        # Try to access outside workspace
        result = fs_tool.read("../outside.txt")
        assert "error" in result
        
        result = fs_tool.write("../outside.txt", "content")
        assert "error" in result


class TestPythonRunner:
    def test_simple_execution(self, python_tool):
        """Test simple Python code execution"""
        code = "print('Hello from Python')"
        result = python_tool.run(code)
        
        assert result["success"] is True
        assert result["return_code"] == 0
        assert "Hello from Python" in result["stdout"]
        assert result["timeout"] is False
    
    def test_code_validation(self, python_tool):
        """Test code validation"""
        # Dangerous import should be rejected
        result = python_tool.validate_code("import os")
        assert result["valid"] is False
        assert "os" in result["reason"]
        
        # File operations should be rejected
        result = python_tool.validate_code("open('file.txt', 'w')")
        assert result["valid"] is False
        assert "open" in result["reason"]
        
        # Safe code should pass
        result = python_tool.validate_code("print('hello')")
        assert result["valid"] is True
    
    def test_timeout_handling(self, python_tool):
        """Test timeout handling"""
        code = "import time; time.sleep(35)"  # Longer than timeout
        result = python_tool.run(code)
        
        assert result["success"] is False
        assert result["timeout"] is True
        assert "timed out" in result["stderr"]


class TestWebTool:
    @pytest.mark.asyncio
    async def test_fetch_url(self):
        """Test URL fetching (using a reliable test URL)"""
        # Test with a simple data URL
        result = await fetch_url("data:text/plain;base64,SGVsbG8gV29ybGQ=")
        assert result["success"] is True
        assert "Hello World" in result["content"]
    
    def test_clean_text(self):
        """Test text cleaning"""
        text = "  Multiple   spaces   and   newlines\n\n  "
        cleaned = clean_text(text)
        assert cleaned == "Multiple spaces and newlines"
        
        # Test truncation
        long_text = "a" * 20000
        truncated = clean_text(long_text, max_chars=100)
        assert len(truncated) == 103  # 100 chars + "..."
        assert truncated.endswith("...")


# Cleanup after tests
def teardown_module():
    """Clean up test workspace"""
    import shutil
    if Path("test_workspace").exists():
        shutil.rmtree("test_workspace")
import pytest
import asyncio
from pathlib import Path

from agent_workbench.memory.short_sql import ShortTermMemory, MessageRecord, ToolEvent, ReflectionRecord
from agent_workbench.memory.long_vector import VectorMemory
from agent_workbench.settings import Settings


@pytest.fixture
def settings():
    """Test settings"""
    settings = Settings()
    settings.paths.sqlite_db = "test_artifacts/test.db"
    settings.paths.vector_index_dir = "test_artifacts/vector"
    return settings


@pytest.fixture
async def short_memory(settings):
    """Short-term memory instance"""
    memory = ShortTermMemory(settings)
    await memory.initialize()
    return memory


@pytest.fixture
def vector_memory(settings):
    """Vector memory instance"""
    return VectorMemory(settings)


class TestShortTermMemory:
    @pytest.mark.asyncio
    async def test_session_creation(self, short_memory):
        """Test session creation"""
        session_id = "test_session"
        await short_memory.create_session(session_id)
        
        # Should not raise an exception
        await short_memory.create_session(session_id)  # Duplicate should be handled
    
    @pytest.mark.asyncio
    async def test_message_storage(self, short_memory):
        """Test message storage and retrieval"""
        session_id = "test_session"
        await short_memory.create_session(session_id)
        
        # Add messages
        from datetime import datetime
        message1 = MessageRecord(
            session_id=session_id,
            role="user",
            content="Hello",
            timestamp=datetime.now()
        )
        message2 = MessageRecord(
            session_id=session_id,
            role="assistant",
            content="Hi there",
            timestamp=datetime.now()
        )
        
        await short_memory.add_message(message1)
        await short_memory.add_message(message2)
        
        # Retrieve messages
        history = await short_memory.get_session_history(session_id)
        assert len(history) == 2
        assert history[0].content == "Hello"
        assert history[1].content == "Hi there"
        assert history[0].role == "user"
        assert history[1].role == "assistant"
    
    @pytest.mark.asyncio
    async def test_tool_event_storage(self, short_memory):
        """Test tool event storage"""
        session_id = "test_session"
        await short_memory.create_session(session_id)
        
        event = ToolEvent(
            session_id=session_id,
            tool_name="test_tool",
            tool_input={"param": "value"},
            tool_output={"result": "success"},
            error=None,
            timestamp=datetime.now()
        )
        
        await short_memory.add_tool_event(event)
        
        events = await short_memory.get_tool_events(session_id)
        assert len(events) == 1
        assert events[0].tool_name == "test_tool"
        assert events[0].tool_input["param"] == "value"
        assert events[0].tool_output["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_reflection_storage(self, short_memory):
        """Test reflection storage"""
        session_id = "test_session"
        await short_memory.create_session(session_id)
        
        reflection = ReflectionRecord(
            session_id=session_id,
            step_number=1,
            reflection_text="Good progress",
            usefulness_score=0.8,
            memory_updates={"key": "value"},
            timestamp=datetime.now()
        )
        
        await short_memory.add_reflection(reflection)
        
        reflections = await short_memory.get_reflections(session_id)
        assert len(reflections) == 1
        assert reflections[0].reflection_text == "Good progress"
        assert reflections[0].usefulness_score == 0.8
        assert reflections[0].memory_updates["key"] == "value"
    
    @pytest.mark.asyncio
    async def test_session_summary(self, short_memory):
        """Test session summary generation"""
        session_id = "test_session"
        await short_memory.create_session(session_id)
        
        # Add some data
        message = MessageRecord(
            session_id=session_id,
            role="user",
            content="Test",
            timestamp=datetime.now()
        )
        await short_memory.add_message(message)
        
        reflection = ReflectionRecord(
            session_id=session_id,
            step_number=1,
            reflection_text="Test reflection",
            usefulness_score=0.7,
            memory_updates={},
            timestamp=datetime.now()
        )
        await short_memory.add_reflection(reflection)
        
        summary = await short_memory.summarize_session(session_id)
        assert "Session had 1 messages" in summary
        assert "Average reflection usefulness: 0.70" in summary


class TestVectorMemory:
    def test_document_addition(self, vector_memory):
        """Test adding documents to vector memory"""
        documents = [
            {
                "id": "doc1",
                "text": "The quick brown fox jumps over the lazy dog",
                "metadata": {"type": "test"}
            },
            {
                "id": "doc2", 
                "text": "Machine learning is a subset of artificial intelligence",
                "metadata": {"type": "test"}
            }
        ]
        
        doc_ids = vector_memory.add_documents(documents)
        assert len(doc_ids) == 2
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
    
    def test_search_functionality(self, vector_memory):
        """Test vector search"""
        # Add documents
        documents = [
            {
                "id": "doc1",
                "text": "Python is a programming language",
                "metadata": {"category": "programming"}
            },
            {
                "id": "doc2",
                "text": "JavaScript is used for web development",
                "metadata": {"category": "programming"}
            },
            {
                "id": "doc3",
                "text": "The weather is sunny today",
                "metadata": {"category": "weather"}
            }
        ]
        
        vector_memory.add_documents(documents)
        
        # Search for programming-related content
        results = vector_memory.search("programming languages", k=2)
        assert len(results) <= 2
        
        # Should find programming-related documents
        programming_docs = [r for r in results if "programming" in r.get("metadata", {}).get("category", "")]
        assert len(programming_docs) > 0
    
    def test_document_retrieval(self, vector_memory):
        """Test retrieving specific documents"""
        document = {
            "id": "test_doc",
            "text": "This is a test document",
            "metadata": {"test": True}
        }
        
        vector_memory.add_documents([document])
        
        retrieved = vector_memory.get_document("test_doc")
        assert retrieved is not None
        assert retrieved["doc_id"] == "test_doc"
        assert retrieved["text"] == "This is a test document"
        assert retrieved["metadata"]["test"] is True
    
    def test_document_deletion(self, vector_memory):
        """Test deleting documents"""
        document = {
            "id": "delete_test",
            "text": "This document will be deleted",
            "metadata": {}
        }
        
        vector_memory.add_documents([document])
        
        # Verify it exists
        retrieved = vector_memory.get_document("delete_test")
        assert retrieved is not None
        
        # Delete it
        success = vector_memory.delete_document("delete_test")
        assert success is True
        
        # Verify it's gone
        retrieved = vector_memory.get_document("delete_test")
        assert retrieved is None
    
    def test_clear_all(self, vector_memory):
        """Test clearing all documents"""
        documents = [
            {"id": "doc1", "text": "Document 1", "metadata": {}},
            {"id": "doc2", "text": "Document 2", "metadata": {}}
        ]
        
        vector_memory.add_documents(documents)
        assert len(vector_memory.mapping) > 0
        
        vector_memory.clear()
        assert len(vector_memory.mapping) == 0


# Cleanup after tests
def teardown_module():
    """Clean up test artifacts"""
    import shutil
    if Path("test_artifacts").exists():
        shutil.rmtree("test_artifacts")
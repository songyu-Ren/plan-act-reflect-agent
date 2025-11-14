from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_workbench.agent import Agent, AgentResult
from agent_workbench.llm.providers import get_provider
from agent_workbench.logging import setup_logging
from agent_workbench.settings import Settings
from agent_workbench.telemetry import get_metrics


# Request/Response models
class ChatRequest(BaseModel):
    session_id: str
    user_text: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class TaskRequest(BaseModel):
    goal: str
    session_id: Optional[str] = None
    max_steps: Optional[int] = None
    constraints: Optional[list[str]] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[AgentResult] = None


class StreamRequest(BaseModel):
    session_id: str
    user_text: str


# Global state
app = FastAPI(title="Agent Workbench", version="0.1.0")
settings = Settings.load()
settings.ensure_directories()

logger = setup_logging(settings)
metrics = get_metrics(settings)
llm_provider = get_provider(settings.llm)
agent = Agent(settings, llm_provider)

# Task storage (in production, use Redis or database)
tasks: Dict[str, AgentResult] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the agent"""
    await agent.initialize()
    logger.info("Agent Workbench started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup"""
    logger.info("Agent Workbench shutting down")


# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    metrics.record_request(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code,
        duration=duration
    )
    
    return response


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


# Metrics endpoint
@app.get("/metrics")
async def metrics_endpoint():
    return metrics.get_metrics()


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat interaction"""
    try:
        reply = await agent.chat(request.session_id, request.user_text)
        return ChatResponse(reply=reply, session_id=request.session_id)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Task execution endpoint
@app.post("/run_task", response_model=TaskResponse)
async def run_task_endpoint(request: TaskRequest):
    """Run a task with the agent"""
    try:
        task_id = f"task_{time.time()}"
        
        # Run task in background
        result = await agent.run_task(
            goal=request.goal,
            session_id=request.session_id,
            max_steps=request.max_steps,
            constraints=request.constraints
        )
        
        tasks[task_id] = result
        
        return TaskResponse(
            task_id=task_id,
            status=result.status,
            result=result
        )
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get task result
@app.get("/task/{task_id}")
async def get_task_result(task_id: str):
    """Get task result"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks[task_id]


# Streaming endpoint
@app.post("/stream")
async def stream_endpoint(request: StreamRequest):
    """Stream responses using Server-Sent Events"""
    
    async def generate_stream() -> AsyncIterator[str]:
        try:
            # For now, just stream the regular chat response
            # In a full implementation, this would stream token by token
            reply = await agent.chat(request.session_id, request.user_text)
            
            # Simulate streaming by sending chunks
            words = reply.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield f"data: {word}"
                else:
                    yield f" data: {word}"
                await asyncio.sleep(0.1)  # Small delay between words
            
            yield "\ndata: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# Session management
@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str, limit: Optional[int] = None):
    """Get session history"""
    try:
        history = await agent.short_memory.get_session_history(session_id, limit)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        logger.error(f"Get session history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Tool execution (for testing/debugging)
@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, tool_input: Dict[str, Any]):
    """Execute a tool directly (for testing)"""
    try:
        result = await agent._execute_tool(tool_name, tool_input, "api_test")
        return result
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Vector search
@app.post("/search")
async def vector_search(query: Dict[str, Any]):
    """Search the vector memory"""
    try:
        query_text = query.get("query", "")
        k = query.get("k", 5)
        
        rag_tool = agent.tools["rag"]
        result = rag_tool.search(query_text, k)
        
        return result
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document ingestion
@app.post("/ingest")
async def ingest_documents(documents: List[Dict[str, Any]]):
    """Ingest documents into vector memory"""
    try:
        rag_tool = agent.tools["rag"]
        result = rag_tool.ingest_documents(documents)
        
        # Update metrics
        if result["success"]:
            metrics.set_vector_documents(len(agent.vector_memory.mapping))
        
        return result
    except Exception as e:
        logger.error(f"Document ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# List available tools
@app.get("/tools")
async def list_tools():
    """List available tools"""
    tool_info = {}
    for name, tool in agent.tools.items():
        tool_info[name] = {
            "type": type(tool).__name__,
            "description": tool.__class__.__doc__ or "No description available"
        }
    
    return {"tools": tool_info}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)
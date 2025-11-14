# Agent Workbench Documentation

## Overview

Agent Workbench is a production-minded AI agent system that supports iterative workflows including planning, acting with tools, reflecting on outcomes, and improving over multiple steps.

## Key Features

- **End-to-end stack**: Complete workflow from configuration through agent planning, tool execution, memory management, reflection, and next step determination
- **Interactivity**: Both stateful chat and goal-oriented task execution via CLI and FastAPI
- **Deterministic baseline**: Provider-agnostic LLM wrapper with null stub for CI/testing
- **Memory systems**: Short-term SQLite storage and long-term vector memory with FAISS/Chroma
- **Tool ecosystem**: Web fetching, filesystem operations, Python execution, and RAG search
- **Observability**: Prometheus metrics, structured logging, and health monitoring
- **Safety**: Sandboxed execution, resource limits, and path constraints

## Architecture

The system follows a planner → executor → reflector loop:

1. **Planner**: Proposes steps with tool calls and rationale
2. **Executor**: Runs tools with validated arguments
3. **Reflector**: Assesses outcomes, updates memory, decides next actions

## Configuration

Configuration is managed through `config/settings.yaml` with environment variable overrides:

- `AGENT_SETTINGS`: Path to config file
- `OPENAI_API_KEY`: OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `OLLAMA_BASE_URL`: Ollama server URL

## API Endpoints

- `POST /chat`: Stateful chat with session management
- `POST /run_task`: Execute goal-oriented tasks
- `POST /stream`: Stream responses via SSE
- `GET /health`: Health check
- `GET /metrics`: Prometheus metrics

## CLI Commands

- `aw ingest <path>`: Ingest documents from corpus directory
- `aw chat --session <id>`: Interactive chat session
- `aw run --goal <goal> --max-steps <n>`: Run a task
- `aw eval`: Run evaluation tests
- `aw status`: Show system status
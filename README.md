# Agent Workbench

A production-minded AI agent workbench that supports iterative workflows: planning, acting with tools, reflecting on outcomes, and improving over multiple steps.

## Features

- **End-to-end stack**: config → agent planner → tool execution → memory → reflection → next step
- **Interactivity**: stateful chat and goal-oriented task runs via CLI and FastAPI
- **Deterministic baseline**: provider-agnostic LLM wrapper with null stub for CI
- **Memory systems**: short-term SQLite + long-term vector memory with FAISS/Chroma
- **Tool ecosystem**: web fetch, filesystem, Python runner, RAG search
- **Observability**: Prometheus metrics, structured logging, health checks
- **Safety**: sandboxed execution, resource limits, path constraints

## Quick Start

```bash
# Install dependencies
make setup

# Ingest corpus data
aw ingest data/corpus

# Start server
make serve

# Chat interactively
aw chat --session s1

# Run task
aw run --goal "Research topic X and create a summary" --max-steps 6
```

## API Endpoints

- `POST /chat` - Stateful chat with session management
- `POST /run_task` - Execute goal-oriented tasks
- `POST /stream` - Stream responses (SSE)
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Configuration

Edit `config/settings.yaml` to configure:
- LLM provider (OpenAI, Azure, Ollama, null)
- Agent behavior (max steps, planning style)
- Memory settings (vector store, embeddings)
- Monitoring (metrics, logging)

## Development

```bash
# Format code
make fmt

# Run linting
make lint

# Run tests
make test

# Build Docker image
make docker-build

# Run in Docker
make docker-run
```

## Architecture

The system follows a planner → executor → reflector loop:

1. **Planner**: Proposes steps with tool calls and rationale
2. **Executor**: Runs tools with validated arguments
3. **Reflector**: Assesses outcomes, updates memory, decides next actions

Tools are sandboxed and resource-limited for safety. Memory persists across sessions with both short-term context and long-term knowledge storage.
\n+## v2 Enhancements
\n+- Skills & contracts with JSON-Schema validation\n- Hierarchical planning (Manager→Worker DAG)\n- Human-in-the-loop approvals API\n- Trace export (JSONL) and replay\n- Cost reporting and extended metrics
\n+## Additional CLI
\n+```bash
aw tools
aw plan --goal "Summarize corpus" --out plan.json
aw replay <RUN_ID>
```

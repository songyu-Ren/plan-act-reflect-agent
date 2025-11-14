FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install -e .

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p artifacts workspace

# Ingest corpus data
RUN python -m agent_workbench.cli ingest data/corpus

# Expose port
EXPOSE 8003

# Run the application
CMD ["uvicorn", "src.agent_workbench.api:app", "--host", "0.0.0.0", "--port", "8003"]
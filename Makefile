.PHONY: setup fmt lint test serve ingest run docker-build docker-run

setup:
	pip install -e .
	mkdir -p artifacts workspace data/corpus

fmt:
	black src/ tests/
	ruff format src/ tests/

lint:
	ruff check src/ tests/
	black --check src/ tests/

test:
	pytest tests/ -v

serve:
	uvicorn src.agent_workbench.api:app --host 0.0.0.0 --port 8003 --reload

ingest:
	python -m agent_workbench.cli ingest data/corpus

run:
	python -m agent_workbench.cli run --goal "$(goal)" --max-steps $(steps)

docker-build:
	docker build -t agent-workbench .

docker-run:
	docker run -p 8003:8003 -v $(PWD)/data:/app/data -v $(PWD)/artifacts:/app/artifacts -v $(PWD)/workspace:/app/workspace agent-workbench
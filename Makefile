PY := .venv/Scripts/python.exe

.PHONY: research-test export fixtures test lint typecheck docker-build docker-run

research-test:
	$(PY) -m tests.test_pipeline

export:
	$(PY) -m service.export_model

fixtures:
	$(PY) -m service.fixtures.gen_fixtures

test:
	$(PY) -m tests.test_pipeline
	$(PY) -m service.tests.test_service

lint:
	$(PY) -m ruff check service/

typecheck:
	$(PY) -m mypy service/

docker-build:
	docker build -t neural-estimator-service -f service/Dockerfile service

docker-run:
	docker run --rm -p 8000:8000 neural-estimator-service

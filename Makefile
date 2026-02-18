.PHONY: help install test lint docker-up docker-down

help:
	@echo "AI Model Hub Management"
	@echo "-----------------------"
	@echo "install      : Install dependencies"
	@echo "test         : Run tests"
	@echo "lint         : Run linting (flake8)"
	@echo "docker-up    : Start services with docker-compose"
	@echo "docker-down  : Stop docker services"

install:
	pip install .

test:
	export PYTHONPATH=$PYTHONPATH:. && pytest tests/

lint:
	pip install flake8
	flake8 modelhub tests

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

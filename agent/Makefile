SHELL := /bin/bash

VENV_DIR := .venv
PYTHON_BIN := $(VENV_DIR)/bin/python3
COMPOSE ?= docker compose
PORT ?= 8000
SERVICE ?= agent

.PHONY: help env install install-frontend run run-debug \
	pull build up down logs logs-tail logs-service logs-agent logs-mysql logs-frontend ps restart \
	rebuild-agent \
	check test test-ready test-api test-multi-agent ensure-pip

help:
	@echo "Available targets:"
	@echo "  make ensure-pip       - Bootstrap pip inside local .venv"
	@echo "  make env              - Create .env from .env.example if missing"
	@echo "  make install          - Install Python dependencies"
	@echo "  make install-frontend - Install frontend dependencies"
	@echo "  make run              - Run FastAPI locally on port $(PORT)"
	@echo "  make run-debug        - Run FastAPI with DEBUG=true"
	@echo "  make pull             - Pull latest Docker base images"
	@echo "  make build            - Build/rebuild Docker services"
	@echo "  make up               - Start full stack with Docker Compose"
	@echo "  make down             - Stop Docker Compose stack"
	@echo "  make logs             - Tail Docker Compose logs"
	@echo "  make logs-tail        - Tail last 200 lines from all services"
	@echo "  make logs-service SERVICE=agent - Tail one service logs"
	@echo "  make logs-agent       - Tail agent logs"
	@echo "  make logs-mysql       - Tail mysql logs"
	@echo "  make logs-frontend    - Tail frontend logs"
	@echo "  make ps               - Show Docker Compose service status"
	@echo "  make restart          - Restart Docker Compose stack"
	@echo "  make rebuild-agent    - Rebuild and restart only agent service"
	@echo "  make check            - Run production readiness checker"
	@echo "  make test             - Run local checks + API smoke tests"
	@echo "  make test-ready       - Check /health and /ready endpoints"
	@echo "  make test-api         - Smoke test JWT login + POST /ask endpoint"
	@echo "  make test-multi-agent - Smoke test POST /ask-multi-agent/stream endpoint"

env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example"; \
	else \
		echo ".env already exists"; \
	fi

install:
	$(MAKE) ensure-pip
	$(PYTHON_BIN) -m pip install -r requirements.txt

ensure-pip:
	@if [ ! -x "$(PYTHON_BIN)" ]; then \
		echo "Missing $(PYTHON_BIN). Create local venv first: python3 -m venv .venv"; \
		exit 1; \
	fi
	@$(PYTHON_BIN) -m pip --version >/dev/null 2>&1 || { \
		echo "pip not found in selected python, bootstrapping with ensurepip..."; \
		$(PYTHON_BIN) -m ensurepip --upgrade; \
	}

install-frontend:
	cd frontend && npm install

run: env
	$(PYTHON_BIN) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

run-debug: env
	DEBUG=true $(PYTHON_BIN) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

pull:
	$(COMPOSE) pull

build:
	$(COMPOSE) build --pull

up: env
	$(COMPOSE) up --build -d --remove-orphans

down:
	$(COMPOSE) down --remove-orphans

logs:
	$(COMPOSE) logs -f

logs-tail:
	$(COMPOSE) logs --tail=200 -f

logs-service:
	$(COMPOSE) logs --tail=200 -f $(SERVICE)

logs-agent:
	$(COMPOSE) logs --tail=200 -f agent

logs-mysql:
	$(COMPOSE) logs --tail=200 -f mysql

logs-frontend:
	$(COMPOSE) logs --tail=200 -f frontend

ps:
	$(COMPOSE) ps

restart: down up

rebuild-agent:
	$(COMPOSE) up -d --build --no-deps agent

check:
	$(PYTHON_BIN) check_production_ready.py

test: check test-ready test-api test-multi-agent

test-ready:
	@echo "Checking health/readiness endpoints..."
	@curl --fail --silent --show-error http://localhost/health > /dev/null
	@curl --fail --silent --show-error http://localhost/ready > /dev/null
	@echo "OK: /health and /ready"

test-api:
	@echo "Checking JWT auth + POST /ask..."
	@EMAIL="$$(awk -F= '/^ADMIN_EMAIL=/{print $$2}' .env)"; \
	PASSWORD="$$(awk -F= '/^ADMIN_PASSWORD=/{print $$2}' .env)"; \
	if [ -z "$$EMAIL" ] || [ -z "$$PASSWORD" ]; then \
		echo "ADMIN_EMAIL/ADMIN_PASSWORD missing in .env"; \
		exit 1; \
	fi; \
	TOKEN="$$(curl --fail --silent --show-error \
		-H "Content-Type: application/json" \
		-X POST http://localhost/auth/login \
		-d "{\"email\":\"$$EMAIL\",\"password\":\"$$PASSWORD\"}" \
		| python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"; \
	curl --fail --silent --show-error \
		-H "Authorization: Bearer $$TOKEN" \
		-H "Content-Type: application/json" \
		-X POST http://localhost/ask \
		-d '{"question":"What is deployment?"}' > /dev/null; \
	echo "OK: /ask"

test-multi-agent:
	@echo "Checking POST /ask-multi-agent/stream..."
	@EMAIL="$$(awk -F= '/^ADMIN_EMAIL=/{print $$2}' .env)"; \
	PASSWORD="$$(awk -F= '/^ADMIN_PASSWORD=/{print $$2}' .env)"; \
	if [ -z "$$EMAIL" ] || [ -z "$$PASSWORD" ]; then \
		echo "ADMIN_EMAIL/ADMIN_PASSWORD missing in .env"; \
		exit 1; \
	fi; \
	TOKEN="$$(curl --fail --silent --show-error \
		-H "Content-Type: application/json" \
		-X POST http://localhost/auth/login \
		-d "{\"email\":\"$$EMAIL\",\"password\":\"$$PASSWORD\"}" \
		| python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"; \
	curl --fail --silent --show-error \
		-H "Authorization: Bearer $$TOKEN" \
		-H "Content-Type: application/json" \
		-X POST http://localhost/ask-multi-agent/stream \
		-d '{"question":"Tom tat deployment voi docker","human_approved":true}' > /dev/null; \
	echo "OK: /ask-multi-agent/stream"

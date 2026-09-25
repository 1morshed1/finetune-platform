.PHONY: install dev test lint fmt serve demo docker clean

install:            ## Install core dependencies
	uv sync

dev:                ## Install with dev + ml extras
	uv sync --extra dev --extra ml

test:               ## Run the test suite
	uv run pytest -q

lint:               ## Lint with ruff
	uv run ruff check src tests

fmt:                ## Auto-fix lint issues
	uv run ruff check --fix src tests

serve:              ## Run the API + dashboard (reload)
	uv run ftp serve --reload

demo:               ## Run the whole pipeline in-process
	uv run ftp demo

docker:             ## Build the container image
	docker build -t finetune-platform:latest .

clean:              ## Remove local runtime state (DB, datasets, artifacts)
	rm -rf data_store/ftp.db data_store/datasets data_store/artifacts

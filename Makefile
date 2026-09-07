.PHONY: check test test-all build

check:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src/legged_manipulation_unilab
	uv run pyright

test:
	uv run pytest

test-all: check test

build:
	uv build

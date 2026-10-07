# Lint, types, tests. `check` is exactly what CI runs, in the same order, so a
# green `make check` locally means a green pipeline.

.PHONY: lint fmt types test check

lint:  ## ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

fmt:  ## Apply ruff formatting and autofixes
	uv run ruff check --fix --exit-zero .
	uv run ruff format .

types:  ## mypy
	uv run mypy .

test:  ## pytest
	uv run pytest

check:  ## lint + types + test (what CI runs)
	$(MAKE) lint
	$(MAKE) types
	$(MAKE) test

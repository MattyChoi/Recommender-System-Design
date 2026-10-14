# Lint, types, tests. `check` is exactly what CI runs, in the same order, so a
# green `make check` locally means a green pipeline.

.PHONY: lint fmt types test check


### Python lints, fmts, types, tests
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


### Go lints, fmts, tests, and compile commands
GO ?= go
GO_DIR = serving/go

go-lint:
	@cd $(GO_DIR) && out=$$("$$($(GO) env GOROOT)/bin/gofmt" -l .) && \
	  { [ -z "$$out" ] || { echo "not gofmt-clean:"; echo "$$out"; \
	    echo "fix with: make go-fmt"; exit 1; }; }
	cd $(GO_DIR) && $(GO) vet ./...

go-fmt:
	cd $(GO_DIR) && "$$($(GO) env GOROOT)/bin/gofmt" -w .

go-test:  ## Go unit tests, including the Python parity fixtures
	cd $(GO_DIR) && $(GO) test ./...

go-build:  ## Compile the serving orchestrator
	cd $(GO_DIR) && $(GO) build ./...


check:  ## lint + types + test (what CI runs)
	$(MAKE) lint
	$(MAKE) types
	$(MAKE) test
	$(MAKE) go-lint
	$(MAKE) go-test

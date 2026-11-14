# Lint, types, tests. `check` is exactly what CI runs, in the same order, so a
# green `make check` locally means a green pipeline.

.PHONY: lint fmt types test test-cov check


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

COVERAGE_FLOOR ?= 54

test-cov:  ## pytest with coverage, gated at COVERAGE_FLOOR -- what CI runs
	uv run pytest --cov --cov-report=term-missing \
	  $(if $(COVERAGE_FLOOR),--cov-fail-under=$(COVERAGE_FLOOR),)


### Go lints, fmts, tests, and compile commands
GO ?= go
GO_DIR = serving/go
GOLANGCI ?= golangci-lint
GOLANGCI_VERSION := $(shell sed -n 's/^ *version: *\(v[0-9.]*\).*/\1/p' .github/workflows/ci.yml | head -1)

go-lint:
	@cd $(GO_DIR) && out=$$("$$($(GO) env GOROOT)/bin/gofmt" -l .) && \
	  { [ -z "$$out" ] || { echo "not gofmt-clean:"; echo "$$out"; \
	    echo "fix with: make go-fmt"; exit 1; }; }
	cd $(GO_DIR) && $(GO) vet ./...
	@# The same linter CI runs. The version is read from ci.yml, so the pin
	@# lives in one place; a mismatch only warns, since checks can differ.
	@command -v $(GOLANGCI) >/dev/null || { echo "golangci-lint not found; install $(GOLANGCI_VERSION):"; \
	  echo '  curl -sSfL https://golangci-lint.run/install.sh | sh -s -- -b "$$(go env GOPATH)/bin" $(GOLANGCI_VERSION)'; exit 1; }
	@$(GOLANGCI) --version | grep -q "version $(patsubst v%,%,$(GOLANGCI_VERSION)) " || \
	  echo "warning: CI pins golangci-lint $(GOLANGCI_VERSION), local is: $$($(GOLANGCI) --version | head -1)"
	cd $(GO_DIR) && $(GOLANGCI) run ./...

go-fmt:
	cd $(GO_DIR) && "$$($(GO) env GOROOT)/bin/gofmt" -w .

go-test:  ## Go unit tests, including the Python parity fixtures
	cd $(GO_DIR) && $(GO) test ./...

go-race:  ## Go tests under the race detector (needs a C toolchain)
	@cc=$$(command -v gcc || command -v clang) || true; \
	  if [ -z "$$cc" ]; then \
	    echo "go-race needs a C compiler: the detector is cgo instrumentation."; \
	    echo "  Debian/Ubuntu:  sudo apt install build-essential"; \
	    echo "  Fedora/RHEL:    sudo dnf install gcc"; \
	    echo "  Arch:           sudo pacman -S gcc"; \
	    echo "  Skip this gate: make check SKIP_RACE=1"; \
	    exit 1; \
	  fi; \
	  cd $(GO_DIR) && CGO_ENABLED=1 CC="$$cc" $(GO) test -race ./...

go-build:  ## Compile the serving orchestrator
	cd $(GO_DIR) && $(GO) build ./...


check:  ## lint + types + coverage + go (what CI runs)
	$(MAKE) lint
	$(MAKE) types
	$(MAKE) test-cov
	$(MAKE) go-lint
	$(MAKE) go-test

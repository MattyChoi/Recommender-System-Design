# All parity tests

.PHONY: fixtures

# The fixtures the Go parity tests read. Regenerate DELIBERATELY: a fixture that
# moves silently is a parity test re-pointed at new behaviour.
fixtures:  ## Regenerate serving/testdata/*.json from the Python implementations
	uv run python -m scripts.writeups.dump_parity_fixtures
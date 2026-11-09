# All parity tests

.PHONY: fixtures skew

# The fixtures the Go parity tests read. Regenerate DELIBERATELY: a fixture that
# moves silently is a parity test re-pointed at new behaviour.
fixtures:  ## Regenerate serving/testdata/*.json from the Python implementations
	uv run python -m scripts.writeups.dump_parity_fixtures
	
SERVED ?=

# The training/serving skew report. SERVED is newline-delimited ImpressionContext
# JSON: the file sink's impression_context output, or
# `make consume TOPIC=impression_context N=20000 > served.jsonl`. Needs Spark
# (it reads gold training_examples), so MinIO on the default backend.
skew:  ## Skew report: served user features vs training's -> docs/skew_report.md
	@test -n "$(SERVED)" || { echo "usage: make skew SERVED=path/to/impression_context.jsonl"; exit 1; }
	uv run python -m scripts.writeups.skew_report --served $(SERVED)

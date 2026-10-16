# Protobuf stubs for both languages.
#
# Python and Go stubs are generated from the SAME serving/proto/*.proto in one
# target on purpose: generating them separately is how the two sides drift.
#
# The `uv run python -c` step rewrites grpcio's flat `import foo_pb2` into
# `from common.pb import foo_pb2`. protoc emits imports as if the output
# directory were on sys.path; it is a package here, so without the rewrite the
# generated modules only import when cwd happens to be common/pb. It covers the
# .pyi stubs too, which have the same flat imports.
#
# The .pyi stubs come from mypy-protobuf (protoc-gen-mypy / protoc-gen-mypy_grpc,
# on PATH via `uv run`) rather than protoc's --pyi_out, which never types the
# *_pb2_grpc modules and leaves strict mypy rejecting every call into them.

.PHONY: triton-proto triton-stubs proto

# Triton's own protobufs, vendored. See serving/proto/triton/README.md.
#
# The ref must match the Triton server container being run
#	`make triton-proto TRITON_PROTO_REF=r25.02`
TRITON_PROTO_REF ?= r25.12
TRITON_PROTO_BASE = https://raw.githubusercontent.com/triton-inference-server/common/$(TRITON_PROTO_REF)/protobuf

triton-proto:  ## Fetch Triton's protobufs (set TRITON_PROTO_REF to a release tag)
	mkdir -p serving/proto/triton
	curl -fsSL $(TRITON_PROTO_BASE)/grpc_service.proto \
	    -o serving/proto/triton/grpc_service.proto
	curl -fsSL $(TRITON_PROTO_BASE)/model_config.proto \
	    -o serving/proto/triton/model_config.proto
	@echo "  fetched at ref '$(TRITON_PROTO_REF)'"
	@test "$(TRITON_PROTO_REF)" != "main" || \
	  echo "  WARNING: 'main' is a moving target. Pin it: make triton-proto TRITON_PROTO_REF=rXX.YY"

TRITON_GO_PKG = github.com/MattyChoi/Recommender-System-Design/serving/go/internal/tritonpb

triton-stubs:
	@test -f serving/proto/triton/grpc_service.proto || \
	  { echo "run 'make triton-proto' first"; exit 1; }
	mkdir -p serving/go/internal/tritonpb
	protoc -I serving/proto/triton \
	    --go_out=serving/go/internal/tritonpb --go_opt=paths=source_relative \
	    --go_opt=Mgrpc_service.proto=$(TRITON_GO_PKG) \
	    --go_opt=Mmodel_config.proto=$(TRITON_GO_PKG) \
	    --go-grpc_out=serving/go/internal/tritonpb --go-grpc_opt=paths=source_relative \
	    --go-grpc_opt=Mgrpc_service.proto=$(TRITON_GO_PKG) \
	    --go-grpc_opt=Mmodel_config.proto=$(TRITON_GO_PKG) \
	    serving/proto/triton/*.proto

proto:  ## Regenerate protobuf stubs from serving/proto/*.proto
	mkdir -p common/pb serving/go/internal/pb && touch common/pb/__init__.py
	uv run python -m grpc_tools.protoc -I serving/proto \
	    --python_out=common/pb --grpc_python_out=common/pb \
	    --mypy_out=common/pb --mypy_grpc_out=common/pb \
	    serving/proto/*.proto
	uv run python -c "import pathlib,re;[f.write_text(re.sub(r'^import (\w+_pb2)', r'from common.pb import \1', f.read_text(), flags=re.M)) for f in pathlib.Path('common/pb').glob('*_pb2*.py*')]"
	protoc -I serving/proto \
	    --go_out=serving/go/internal/pb --go_opt=paths=source_relative \
	    --go-grpc_out=serving/go/internal/pb --go-grpc_opt=paths=source_relative \
	    serving/proto/*.proto
	@$(MAKE) --no-print-directory triton-stubs
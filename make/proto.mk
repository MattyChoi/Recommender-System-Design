# Protobuf stubs for both languages.
#
# Python and Go stubs are generated from the SAME serving/proto/*.proto in one
# target on purpose: generating them separately is how the two sides drift.
#
# The `uv run python -c` step rewrites grpcio's flat `import foo_pb2` into
# `from common.pb import foo_pb2`. protoc emits imports as if the output
# directory were on sys.path; it is a package here, so without the rewrite the
# generated modules only import when cwd happens to be common/pb.

.PHONY: proto

proto:  ## Regenerate protobuf stubs from serving/proto/*.proto
	mkdir -p common/pb serving/go/internal/pb && touch common/pb/__init__.py
	uv run python -m grpc_tools.protoc -I serving/proto \
	    --python_out=common/pb --pyi_out=common/pb --grpc_python_out=common/pb \
	    serving/proto/*.proto
	uv run python -c "import pathlib,re;[f.write_text(re.sub(r'^import (\w+_pb2)', r'from common.pb import \1', f.read_text(), flags=re.M)) for f in pathlib.Path('common/pb').glob('*_pb2*.py')]"
	protoc -I serving/proto \
	    --go_out=serving/go/internal/pb --go_opt=paths=source_relative \
	    --go-grpc_out=serving/go/internal/pb --go-grpc_opt=paths=source_relative \
	    serving/proto/*.proto

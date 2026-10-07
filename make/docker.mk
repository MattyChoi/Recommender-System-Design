# Local stack: Kafka, Redis, MinIO, MLflow, Triton.
#
# `up` waits for every healthcheck before returning, then creates the two
# buckets MLflow and the artifact dumpers expect. Creating them here rather
# than in a container entrypoint keeps the bucket names in one place and lets
# `make up` be idempotent.

.PHONY: up down clean

up:  ## Start the local stack and create the MinIO buckets
	docker compose up -d --wait
	docker exec recsys-minio sh -c 'mc alias set local http://localhost:9000 "$$MINIO_ROOT_USER" "$$MINIO_ROOT_PASSWORD" >/dev/null && mc mb --ignore-existing local/mlflow local/recsys'

down:  ## Stop and remove containers, keeping data
	docker compose down

# Destroys the MinIO and MLflow volumes, which is where every logged run and
# every published artifact lives. There is no undo.
clean:  ## Stop everything and DESTROY all volumes
	docker compose down -v

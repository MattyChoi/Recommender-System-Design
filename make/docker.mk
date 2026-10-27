# Local stack: Kafka, Redis, MinIO, MLflow, Triton.
#
# `up` waits for every healthcheck before returning, then creates the two
# buckets MLflow and the artifact dumpers expect. Creating them here rather
# than in a container entrypoint keeps the bucket names in one place and lets
# `make up` be idempotent.

.PHONY: up down clean


export HOST_IP := $(shell hostname -I 2>/dev/null | awk '{print $$1}')

up:  ## Start the local stack and create the MinIO buckets
	@test -n "$(HOST_IP)" || echo "  warning: HOST_IP is empty; Prometheus will fall back to host-gateway"
	docker compose up -d --wait
	docker exec recsys-minio sh -c 'mc alias set local http://localhost:9000 "$$MINIO_ROOT_USER" "$$MINIO_ROOT_PASSWORD" >/dev/null && mc mb --ignore-existing local/mlflow local/recsys'
	@echo "  scrape targets:"
	@curl -s localhost:9090/api/v1/targets \
	  | python3 -c "import json,sys; [print(f\"    {t['labels'].get('job'):22} {t['health']}\") for t in json.load(sys.stdin)['data']['activeTargets']]" \
	  || echo "    (could not read Prometheus targets)"

down:  ## Stop and remove containers, keeping data
	docker compose down

# Destroys the MinIO and MLflow volumes, which is where every logged run and
# every published artifact lives. There is no undo.
clean:  ## Stop everything and DESTROY all volumes
	docker compose down -v

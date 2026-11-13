# The orchestrator on a local Kubernetes cluster (kind).
#
# Credentials come from .env, which the root Makefile -includes. They are
# exported and referenced as $$VAR rather than $(VAR), so make prints the
# variable name and never the value.

.PHONY: kind-up kind-load kind-push kind-status kind-down kind-guard

KIND_CONTEXT = kind-$(KIND_CLUSTER)
MINIO_FORWARD_PORT ?= 9900
MANIFESTS = infra/k8s

export MINIO_ROOT_USER
export MINIO_ROOT_PASSWORD

# Every target that CHANGES the cluster depends on this. Without it a stray
# kubeconfig context turns `make kind-push` into a rolling restart of whatever
# cluster happens to be current, which may not be a toy.
kind-guard:
	@current=$$(kubectl config current-context 2>/dev/null || echo none); \
	  test "$$current" = "$(KIND_CONTEXT)" \
	  || { echo "  context is $$current, not $(KIND_CONTEXT); refusing"; exit 1; }

kind-up:  ## Create the kind cluster, load MinIO, write the secret, apply everything
	@kind get clusters 2>/dev/null | grep -qx $(KIND_CLUSTER) \
	  || kind create cluster --name $(KIND_CLUSTER)
	@$(MAKE) --no-print-directory kind-guard
	@# AFTER the cluster exists, not as a prerequisite: minio-image-kind ends in
	@# a `kind load`, which needs something to load into. Here rather than in
	@# kind-load because MinIO changes once a cluster, not once a code change.
	$(MAKE) --no-print-directory minio-image-kind
	@test -n "$$MINIO_ROOT_USER" \
	  || { echo "  MINIO_ROOT_USER is unset; .env supplies it"; exit 1; }
	@# `create --dry-run=client | apply` is the idempotent form: plain create
	@# fails on an existing secret, so re-running kind-up would break. Still
	@# @-prefixed, so no literal reaches the terminal.
	@kubectl create secret generic recsys-minio \
	    --from-literal=accesskey="$$MINIO_ROOT_USER" \
	    --from-literal=secretkey="$$MINIO_ROOT_PASSWORD" \
	    --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f $(MANIFESTS)/
	@echo
	@echo "  recsys-api sits in Init:Error until: make kind-load kind-push"

kind-load: kind-guard  ## Build the orchestrator image and side-load it into kind
	$(MAKE) --no-print-directory serving-image
	@# kind nodes are containers with no registry behind them, so the image is
	@# pushed into them directly. Re-run after every rebuild.
	kind load docker-image $(IMAGE) --name $(KIND_CLUSTER)

kind-push: kind-guard  ## Publish artifacts into the cluster's MinIO and roll the pods
	@kubectl rollout status deploy/recsys-minio --timeout=120s
	@# The bucket is a Job, and publishing into a bucket that does not exist yet
	@# fails as a bare 404 from the S3 client.
	@kubectl wait --for=condition=complete job/recsys-minio-bucket --timeout=120s
	@# ONE shell for the whole block: each recipe line is its own process, so a
	@# port-forward started on its own line would outlive the recipe. The trap
	@# is what makes it exit-safe, including on a failed publish.
	@kubectl port-forward svc/recsys-minio $(MINIO_FORWARD_PORT):9000 >/dev/null & \
	  forward=$$!; trap 'kill $$forward 2>/dev/null' EXIT; \
	  ready=""; \
	  for _ in $$(seq 60); do \
	    curl -sf http://localhost:$(MINIO_FORWARD_PORT)/minio/health/ready >/dev/null \
	      && { ready=1; break; }; \
	    sleep 0.5; \
	  done; \
	  test -n "$$ready" \
	    || { echo "  the forward never became ready on $(MINIO_FORWARD_PORT)"; exit 1; }; \
	  RECSYS_STORAGE__ENDPOINT_URL=http://localhost:$(MINIO_FORWARD_PORT) \
	    $(MAKE) --no-print-directory artifacts-push
	@# The init container reads CURRENT once, at pod start, so a push alone
	@# changes nothing that is running. A ROLLING restart honours the readiness
	@# flip, so new pods pull the new version and the old ones drain.
	kubectl rollout restart deploy/recsys-api
	kubectl rollout status deploy/recsys-api --timeout=180s

kind-status:  ## Pods, and whether readiness put them behind the Service
	@kubectl get pods -l app.kubernetes.io/name=recsys-api -o wide
	@echo
	@kubectl get endpointslice -l kubernetes.io/service-name=recsys-api
	@echo
	@echo "  no addresses means readiness is failing; addresses mean traffic flows"

# Destroys the PVC with it, so the published artifacts go too. They are
# re-publishable from disk, which is why this is not treated as precious --
# unlike the MinIO and MLflow volumes under docker compose.
kind-down:  ## Delete the kind cluster
	kind delete cluster --name $(KIND_CLUSTER)

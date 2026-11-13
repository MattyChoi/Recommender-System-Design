# The orchestrator on Kubernetes

`recsys-api.yaml` runs the Go orchestrator: artifacts pulled from the object
store at startup, gRPC probes that mean three different things, and a shutdown
that stops receiving traffic before it stops accepting connections.

```bash
kubectl create secret generic recsys-minio \
  --from-literal=accesskey=... --from-literal=secretkey=...
kubectl apply -f infra/k8s/recsys-api.yaml
```

## Validating it

⚠️ **`kubectl apply --dry-run=client` is not an offline check.** It still
downloads the OpenAPI schema from the API server to validate field names, so
with no cluster in your kubeconfig it falls back to the legacy
`localhost:8080` and fails with `connection refused`. The "client" in
`--dry-run=client` means "do not persist the objects", not "do not contact a
server".

Three checks, weakest to strongest:

```bash
# 1. YAML syntax only. Knows nothing about Kubernetes; proves the `---`
#    separators and indentation are sound. Should print 3.
uv run python -c "import yaml,sys; print(len(list(yaml.safe_load_all(open(sys.argv[1])))))" \
  infra/k8s/recsys-api.yaml

# 2. Field names against the published schemas, no cluster needed. -strict is
#    the flag that matters: it REJECTS unknown fields, so a misspelled probe
#    child is caught rather than silently ignored.
go install github.com/yannh/kubeconform/cmd/kubeconform@latest
kubeconform -strict -summary -kubernetes-version 1.31.0 infra/k8s/recsys-api.yaml

# 3. The real thing, once a cluster exists: validated by an actual API server,
#    admission included.
kubectl apply --dry-run=server -f infra/k8s/recsys-api.yaml
```

Pin `-kubernetes-version` to whatever the cluster runs. Anything **1.27 or
later** is required here, because that is where native gRPC probes went GA —
on an older cluster the `grpc:` blocks are ignored rather than rejected, which
means all three probes silently stop doing anything.

## Running it under kind

`minio.yaml` provides the object store in-cluster, so the cluster depends on
nothing on the host. In a real cluster that file does not exist and only the
ConfigMap's `endpoint` changes.

```bash
make kind-up      # cluster, MinIO image, the secret from .env, every manifest
make kind-load    # build the orchestrator image and side-load it
make kind-push    # publish artifacts into the cluster's MinIO, then roll the pods
make kind-status
```

`make/k8s.mk` holds these. Four things they do that are easy to get wrong by
hand:

- **A context guard.** Every target that changes the cluster first checks that
  kubectl's current context is `kind-$(KIND_CLUSTER)` and refuses otherwise, so
  a stray kubeconfig cannot turn `make kind-push` into a rolling restart of
  something real.
- **The secret comes from `.env`**, which the root Makefile `-include`s, passed
  through the environment as `$$VAR` so make prints the variable name and never
  the value. Written with `create --dry-run=client | apply`, the idempotent form
  — plain `create` fails on an existing secret, so re-running `kind-up` would
  break.
- **Both images are side-loaded, not pulled.** kind nodes are containers with no
  registry behind them. MinIO goes in during `kind-up` because it changes once
  per cluster; the orchestrator goes in during `kind-load`, which you re-run
  after every rebuild.
- **`kind-push` rolls the pods.** The init container reads `CURRENT` once, at
  pod start, so publishing on its own changes nothing that is already running.

**Expect transient failures here, and watch them resolve.** Kubernetes is
eventually consistent, and this is the clearest place to see what that means:

- the bucket Job retries until MinIO is ready, by design — it has no ordering
  guarantee against the Deployment, so it waits in a loop rather than assuming;
- the orchestrator's init container fails until the bucket exists **and**
  artifacts have been published into it, so the Pods sit in `Init:Error` and
  back off. Nothing is wrong; they are waiting for the next step.

`make kind-push` resolves both. The dumpers and the publish step run on the
**host**, against `conf/config.yml`'s endpoint, so reaching the cluster's MinIO
needs a port-forward — 9900, to avoid colliding with the compose MinIO on 9000 —
and `RECSYS_STORAGE__ENDPOINT_URL` pointed at it. That is the pydantic-settings
override for `storage.endpoint_url`, using the `RECSYS_` prefix and `__` nesting
documented in `common/config.py`.

The target does the fiddly parts: it waits for MinIO's rollout and for the
bucket Job to complete (publishing into a bucket that does not exist yet fails
as a bare 404), runs the forward and the publish in **one shell** with a `trap`
so the forward cannot outlive the recipe, and polls `/minio/health/ready`
through the forward rather than sleeping a fixed time.

So you publish twice in this project, deliberately: `make artifacts-push` for
the compose stack, `make kind-push` for the cluster. Pointing the cluster at the
host's compose MinIO instead would remove the duplication, but kind pods do not
get `host.docker.internal` — it would trade a port-forward for host-networking
archaeology that differs across Docker Desktop, WSL and native Linux.

Then the init container finds `CURRENT`, pulls that version, and the
orchestrator starts:

```bash
kubectl get pods -w
kubectl logs deploy/recsys-api -c pull-artifacts   # which version it pulled
kubectl logs deploy/recsys-api                     # the startup banner
```

Both Pods should reach `Running` and `READY 1/1`. **They go Ready with no
sidecars and no Triton deployed** — readiness means "can serve a slate", and
these answer from `popular.json`. That is the behaviour to expect, not a
misconfiguration; see the probe section below.

To confirm the distinction holds in-cluster:

```bash
kubectl port-forward svc/recsys-api 15051:15051 &
grpc-health-probe -addr=localhost:15051 -service=recsys.v1.Recommender              # SERVING
grpc-health-probe -addr=localhost:15051 -service=recsys.v1.Recommender.upstreams    # NOT_SERVING
```

**Division of labour between this file and the manifest's comments.** The
manifest explains the project-specific decisions — why readiness ignores
upstream health, why the pointer is read before the files, why 61.9 MB. This
explains the Kubernetes mechanics those decisions are expressed in. Put a
reason next to the field it justifies; put an explanation of a Kubernetes
concept here.

---

## The model

Kubernetes is declarative. You describe the state you want and controllers work
continuously to make reality match it. Nothing in this file is a command — the
`replicas: 2` is a standing claim, not an instruction, and if a node dies a
controller notices the claim is false and acts.

Every object has the same four top-level fields:

| field | meaning |
|---|---|
| `apiVersion` | API group and version. `v1` is the core group; `apps/v1` is the `apps` group, which is where Deployment lives. That is the only reason some objects here say one and some the other. |
| `kind` | the type of object |
| `metadata` | name and labels |
| `spec` | the desired state |

The `---` lines are YAML document separators, so this one file holds three
independent objects. They are only related through labels.

**Labels are how everything finds everything.** There are no pointers, IDs or
references between objects. A Deployment owns the Pods whose labels match its
selector; a Service routes to the Pods whose labels match *its* selector,
entirely independently. Nothing in the Service knows the Deployment exists. This
is the single most surprising thing about Kubernetes if you come from
docker-compose, where services are named and linked.

---

## ConfigMap `recsys-artifacts`

A named bag of key/value strings in the cluster. Nothing runs it; its only
purpose is to be referenced. Here it holds the object-store `endpoint`, `bucket`
and `prefix`, which the init container reads as environment variables.

The payoff is that moving MinIO is an edit here rather than an image rebuild.

`endpoint` is `http://recsys-minio:9000` — a **DNS name Kubernetes invents for
any Service named `recsys-minio`**, not a hostname configured anywhere. It is
why `conf/config.yml`'s host-visible `localhost:9000` is meaningless in-cluster.

Credentials are deliberately *not* here. Secrets are a separate object kind so
they can carry different access controls and are not printed by a casual
`kubectl get` — see the `kubectl create secret` above.

---

## Deployment `recsys-api`

A Deployment does not run containers. The chain is **Deployment → ReplicaSet →
Pods**, and a **Pod** is what Kubernetes actually schedules onto a machine: one
or more containers sharing a network namespace (one IP, mutually reachable on
`localhost`) and able to share volumes. The Deployment's own job is rollouts —
change the Pod template and it creates a new ReplicaSet and shifts Pods across
gradually.

### `selector` and `template.metadata.labels` must match

If they disagree, the Deployment creates Pods it does not recognise as its own
and keeps creating more forever. This is the most common first mistake.

### `annotations`

Like labels, but not selectable — arbitrary metadata for other tools to read.
`prometheus.io/scrape` and `prometheus.io/port` do nothing on their own; a
Prometheus configured to look for them will scrape the Pod. Values must be
strings, which is why `"true"` is quoted.

### `terminationGracePeriodSeconds: 30`

On delete, kubelet sends `SIGTERM`, waits this long, then `SIGKILL`. It has to
exceed `-drain-delay` plus the longest request, or the process is killed
mid-drain and the shutdown sequence buys nothing.

### `volumes`

Declared at the **Pod** level, then mounted into individual containers. That is
the mechanism by which two containers share files, and it is how the artifacts
get from the init container to the orchestrator.

`emptyDir` is scratch space created empty with the Pod and **deleted with it**.
That is the right choice here rather than a liability: the artifacts are
immutable and re-downloadable, and a fresh Pod *should* re-read `CURRENT` rather
than resurrect a stale copy. `sizeLimit` evicts the Pod if it overflows instead
of filling the node's disk.

### `initContainers`

Init containers **run to completion, in order, before any app container
starts**. If one fails the Pod restarts it and the app container never runs.

That guarantee is the point: the orchestrator cannot start before its artifacts
exist, so it can never crash-loop on missing files — and missing files are fatal
to it by design.

- `command` overrides the image's `ENTRYPOINT`; `recsys/minio` would otherwise
  run the `minio` server. The same image carries `mc`, which is why one image
  serves the server, the bucket Job and this init container. The `|` is YAML for "the indented block is one multi-line
  string", which here is a shell script.
- `env` with `valueFrom` says "do not use a literal, look this up".
  `configMapKeyRef` and `secretKeyRef` are the same mechanism against the two
  different object kinds.
- `volumeMounts.name` refers back to the Pod's volume; `mountPath` is where it
  appears in *this* container. The same volume is mounted at a different path in
  the app container, which is the handoff.

The init container has no `securityContext`, so it runs as root and can write to
the fresh volume; the orchestrator runs as uid 65532 and only reads. Root-created
files are world-readable, so this works — and the container doing the privileged
thing exits before any traffic arrives.

### The app container

- `imagePullPolicy: IfNotPresent` — Kubernetes would otherwise pull a `:dev` tag
  from a registry on every start. There is no registry under kind; the image is
  side-loaded with `kind load docker-image recsys/serving:dev`.
- `args` replaces the image's `CMD` and is appended to its `ENTRYPOINT`, which is
  the `/server` binary. These are exactly the flags that follow the image name in
  `make image-run`.
- `ports` is close to documentation — the process listens whether or not you
  declare it. The value is the **name**: the Service refers to `grpc` rather than
  to `15051`, so the two cannot drift.

### Resources

**Requests are for scheduling**: the Pod is only placed on a node with that much
free, and the reservation is yours. **Limits are enforcement**, and the two
resources fail very differently — over the CPU limit a container is *throttled*,
over the memory limit it is *OOMKilled*.

`500m` is millicores, so half a core. `Mi` and `Gi` are binary units.

The CPU limit also reaches the Go runtime: since Go 1.25 `GOMAXPROCS` is derived
from the cgroup limit, and the startup log prints what it chose — check it says
2 rather than the node's core count.

### `securityContext`

`runAsNonRoot` makes kubelet **refuse to start** a container that would run as
root; it is a guard, not a change. `runAsUser` names the uid so the Pod stays
correct if the base image's default ever moves. `readOnlyRootFilesystem` is free
here because the binary is static and writes nothing. Dropping all capabilities
and blocking privilege escalation is ordinary hardening for a process that only
answers RPCs.

---

## The three probes

The most important section, because **each probe makes Kubernetes do something
different**. All three use `grpc:`, meaning kubelet makes a real
`grpc.health.v1.Health/Check` call — no curl, no shell, no sidecar binary.

| probe | asks for | what failing does |
|---|---|---|
| `startupProbe` | `""` | nothing yet; it only gates the other two |
| `livenessProbe` | `""` | **restarts the container** |
| `readinessProbe` | `recsys.v1.Recommender` | **removes the Pod from its Service's endpoints** |

A probe with no `service:` field asks for the empty service name `""`. That is
how one port answers two different questions.

**`startupProbe`** has a special power: while it is running, liveness and
readiness are **suspended**. 2-second periods with 30 failures allowed gives the
process 60 seconds to load artifacts and open its port. Without it you would have
to pad the liveness threshold instead — and then a genuinely hung process would
take 60 seconds to be noticed for the entire life of the pod, not just at start.

**`livenessProbe`** must only ever test "is this process wedged", because restart
is the only remedy it has. Asking for `""` — the name `main.go` sets SERVING once
and never flips — is what keeps it that way.

**`readinessProbe`** controls traffic, not lifecycle. It asks for
`recsys.v1.Recommender`, which means *serving and not draining*. Three failures
at 5-second periods is roughly 15 seconds to notice.

A third name, `recsys.v1.Recommender.upstreams`, carries the all-three backend
verdict and **nothing routes on it**. Probe it from a dashboard or a canary:

```bash
grpc-health-probe -addr=<pod>:15051 -service=recsys.v1.Recommender.upstreams
```

The three names are constants in `serving/go/cmd/server/main.go`
(`readinessService`, `upstreamsService`) and are asserted against a running
container by `tests/test_serving_smoke.py`, which is what keeps this file honest.

---

## Service `recsys-api`

A Pod's IP changes whenever it is replaced, so Pods are never addressed
directly. A Service provides one stable virtual IP and DNS name — `recsys-api`,
or `recsys-api.<namespace>.svc.cluster.local`.

Its `selector` is a label match, independent of the Deployment. The Service
continuously collects every Pod carrying that label **that is passing its
readiness probe**, and load-balances across them. That set is the *endpoints*,
and readiness controlling membership in it is the entire mechanism behind
rollouts, draining and degradation handling.

`type: ClusterIP` is in-cluster only; `NodePort` and `LoadBalancer` would expose
it outside.

`port` is what the Service listens on, `targetPort` where it forwards on the
Pod. They may differ. `targetPort: grpc` references the *named* container port
rather than repeating the number.

---

## What happens on `kubectl apply`

1. Three objects are written to the cluster's state store. Nothing has run.
2. The Deployment controller creates a ReplicaSet, which creates two Pods; the
   scheduler assigns each to a node.
3. kubelet runs the init container: read `CURRENT`, mirror that version into the
   `emptyDir`, exit 0.
4. The orchestrator starts with its artifacts already present. It loads them,
   binds 15051, warms its backends, flips readiness to SERVING, and serves.
5. The startup probe succeeds, handing over to liveness and readiness. Readiness
   passing adds the Pod to the Service's endpoints, and traffic begins.
6. On delete or rollout, kubelet sends SIGTERM *and* begins removing the endpoint
   concurrently. The process fails readiness at once, keeps serving for
   `-drain-delay` while that removal propagates, then drains in-flight requests
   and exits — inside the 30-second grace period.

Step 6 is why the drain delay exists: endpoint removal is eventually consistent,
so a client can still hold a stale endpoint for a moment after SIGTERM. Closing
the listener immediately turns every rollout into a burst of connection
refusals.

---

## Not here yet

- **The upstreams.** `recsys-features`, `recsys-retrieval`, `recsys-triton`,
  `recsys-redis`, Kafka and the Schema Registry have no manifests. The
  orchestrator dials them lazily and readiness does not track them, so a Pod
  without any of them still joins the Service and answers from `popular.json`.
- **prometheus-adapter.** `prometheus.yaml` now scrapes the orchestrator, so the
  histogram is in the cluster — but the HPA reads the custom-metrics API, not
  Prometheus. The adapter is what serves that API, and its rule is what turns
  `recsys_request_duration_seconds` into a per-pod p99. An HPA without it
  applies cleanly and silently never scales.
- **The HPA itself**, which depends on the adapter.
- **Terraform** (`infra/terraform/`), for an S3 bucket, an IAM role and a spot
  GPU instance.

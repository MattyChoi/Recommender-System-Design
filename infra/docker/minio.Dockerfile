# MinIO server AND client, built from source.
#
# Why this exists: MinIO stopped publishing community images. docker.io/minio/*
# now 404s and quay.io/minio/* answers 401, so a fresh pull of the tags below
# fails with ErrImagePull -- on kind, and in compose on any machine without a
# cached copy. Both GitHub repos are archived but the source and the release
# tags are still there, so we build those exact tags ourselves.
#
# ONE image carries both binaries, as the old minio/minio image did: compose's
# healthcheck and `make up` run `mc` inside the server container, and the k8s
# bucket Job and init container need `mc` plus /bin/sh. Two images would mean
# two builds, two tags and two `kind load`s for no gain.
#
# Archived means FROZEN: no security fixes will ever land. Fine for kind and
# compose; anything real points the ConfigMap's `endpoint` at a managed store.
#
# Build with no context (the Dockerfile clones what it needs):
#   docker build -t recsys/minio:<MINIO_TAG> \
#     --build-arg MINIO_TAG=... --build-arg MC_TAG=... - < infra/docker/minio.Dockerfile
# `make minio-image` does exactly that.

ARG MINIO_TAG=RELEASE.2025-09-07T16-13-09Z
ARG MC_TAG=RELEASE.2025-04-16T18-13-26Z

FROM golang:1.27-alpine AS build
# git twice over: the clone, and gen-ldflags.go reads the commit id and time.
RUN apk add --no-cache git

# Upstream's release build, minus the Makefile: gen-ldflags.go stamps the real
# Version/ReleaseTag/CommitID in, and *_RELEASE=RELEASE makes the tag read
# RELEASE.<date> rather than DEVELOPMENT.<date>. Without the ldflags at all,
# `minio --version` cannot tell you what is actually running.
#
# CGO off: both are pure Go, and a static binary runs on any base.
FROM build AS minio
ARG MINIO_TAG
RUN git clone --depth 1 --branch "$MINIO_TAG" https://github.com/minio/minio /src
WORKDIR /src
RUN CGO_ENABLED=0 go build -tags kqueue -trimpath \
      -ldflags "$(MINIO_RELEASE=RELEASE go run buildscripts/gen-ldflags.go)" -o /out/minio .

FROM build AS mc
ARG MC_TAG
RUN git clone --depth 1 --branch "$MC_TAG" https://github.com/minio/mc /src
WORKDIR /src
RUN CGO_ENABLED=0 go build -tags kqueue -trimpath \
      -ldflags "$(MC_RELEASE=RELEASE go run buildscripts/gen-ldflags.go)" -o /out/mc .

# Alpine, not distroless: the k8s Job and init container run `/bin/sh -c`
# scripts, and `make up` runs `docker exec recsys-minio sh -c 'mc ...'`.
FROM alpine:3.22
RUN apk add --no-cache ca-certificates
COPY --from=minio /out/minio /usr/bin/minio
COPY --from=mc    /out/mc    /usr/bin/mc

# 9000 S3 API, 9001 console. EXPOSE is documentation, not a firewall.
EXPOSE 9000 9001
VOLUME ["/data"]

# Every caller fits a bare `minio` entrypoint: compose's `command: server ...`
# and the k8s Deployment's `args: [server, ...]` become `minio server ...`;
# the k8s Job and init container replace it with their own `command`; and
# `docker exec` (the healthcheck, `make up`) never runs an entrypoint at all.
ENTRYPOINT ["/usr/bin/minio"]

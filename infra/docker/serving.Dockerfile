FROM golang:1.27-alpine AS build

# cgo is NOT optional: confluent-kafka-go links librdkafka. gcc and musl-dev
# are what CGO_ENABLED=1 needs on Alpine.
RUN apk add --no-cache gcc musl-dev

WORKDIR /src

COPY serving/go/go.mod serving/go/go.sum ./
RUN go mod download

COPY serving/go/ ./

RUN CGO_ENABLED=1 GOOS=linux \
    go build -tags musl -trimpath \
      -ldflags='-s -w -extldflags "-static"' \
      -o /out/server ./cmd/server

# Fail the BUILD, not the pod. `ldd` on a truly static binary reports no
# dynamic dependencies; anything with "=>" in it means one of the flags above
# was dropped, and the failure would otherwise surface as a crash-looping
# container on a host with a different libc.
RUN ! ldd /out/server 2>/dev/null | grep -q '=>' \
    || { echo "FAIL: dynamically linked -- check -extldflags"; exit 1; }

FROM gcr.io/distroless/static-debian12:nonroot

WORKDIR /repo
COPY --from=build /out/server /server

# 15051 gRPC, 9102 Prometheus. EXPOSE is documentation, not a firewall.
EXPOSE 15051 9102

USER nonroot:nonroot
ENTRYPOINT ["/server"]
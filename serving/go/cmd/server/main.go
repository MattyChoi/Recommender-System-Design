// Command server is the gRPC orchestrator: retrieve, filter, rank, re-rank.
//
// It holds no model and no index. Everything it depends on is a process or an
// artifact, and this file is the only place that knows which -- the pipeline
// itself sees nothing but the interfaces in internal/service/ports.go.
//
// Startup is deliberately FAIL-FAST on artifacts and FAIL-SOFT on services.
// An artifact that will not load (a missing item map, a catalogue from another
// build) is wrong for every request that will ever arrive, so the process
// refuses to start and says which file. A service that is briefly down is
// wrong for the requests that arrive while it is down, and gRPC reconnects on
// its own -- refusing to start there turns a blip into an outage.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/go-redis/v9"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/catalogue"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/experiments"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/features"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/index"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/metrics"
	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/popular"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/ranking"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/rpc"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/server"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/sources"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/tracing"
)

type options struct {
	port         int
	metricsAddr  string
	otlpEndpoint string
	traceRatio   float64

	itemMap   string
	cataloguE string
	columns   string
	popular   string
	indexStem string
	redisURL  string
	seenCap   int
	seenRate  float64
	seenTTL   time.Duration

	retrieval   string
	featuresAPI string
	triton      string
	tritonModel string
	conns       int

	sourceName   string
	candidates   int
	efSearch     int
	numResults   int
	deadline     time.Duration
	rankDeadline time.Duration
	featDeadline time.Duration
	srcDeadline  time.Duration

	modelVersion string
	indexVersion string

	experimentID string
	variants     string
}

// parseVariants reads "control:50,treatment:50".
//
// Percentages are explicit rather than inferred from the count, because an
// even split is the one case where getting it wrong is invisible: two arms
// that each say 50 and two arms that each got 50 because there were two of
// them look identical until someone adds a third.
func parseVariants(spec string) ([]experiments.Variant, error) {
	if spec == "" {
		return nil, nil
	}
	var out []experiments.Variant
	for _, part := range strings.Split(spec, ",") {
		name, share, found := strings.Cut(strings.TrimSpace(part), ":")
		if !found {
			return nil, fmt.Errorf("variant %q is not name:percent", part)
		}
		percent, err := strconv.ParseFloat(share, 64)
		if err != nil {
			return nil, fmt.Errorf("variant %q: %w", part, err)
		}
		out = append(out, experiments.Variant{Name: strings.TrimSpace(name), Percent: percent})
	}
	return out, nil
}

func parse() options {
	var o options
	flag.IntVar(&o.port, "port", 50051, "listen port")
	// A SEPARATE port from gRPC. Scrapes must keep working when the serving
	// port is saturated, which is exactly when the metrics matter.
	// :9102, NOT :9090 -- docker-compose publishes Prometheus itself on 9090,
	// and the orchestrator runs on the host beside it. Two processes on one
	// port is a bind failure at best; at worst Prometheus comes up first and
	// this one silently scrapes itself.
	flag.StringVar(&o.metricsAddr, "metrics", ":9102", "Prometheus scrape address")
	// Empty disables tracing entirely: the OpenTelemetry API is a no-op with
	// no provider, so "off" costs nothing and needs no separate code path.
	flag.StringVar(&o.otlpEndpoint, "otlp", "", "OTLP collector, e.g. localhost:4317. Empty disables tracing")
	// ⚠️ Head sampling decides at the root, before anything interesting has
	// happened, so below 1.0 the DEGRADED requests are dropped at the same
	// rate as the healthy ones -- and they are rare by definition. The
	// pipeline records degradation as span attributes so a collector can
	// TAIL-sample on them instead; this ratio is the floor for everything
	// else.
	flag.Float64Var(&o.traceRatio, "trace-ratio", 1.0, "head sampling ratio; tail-sample in the collector")

	flag.StringVar(&o.itemMap, "item-map", "serving/artifacts/item_map.json",
		"external id to internal index; `make item-map`")
	flag.StringVar(&o.cataloguE, "catalogue", "serving/artifacts/catalogue.json",
		"per-item category indices; `make catalogue`")
	// Beside the ONNX file, because that is where the export writes it:
	// models/export/onnx.py does `out.with_suffix(".columns.json")`, so the
	// path is DERIVED from ONNX_OUT rather than chosen here. An earlier default
	// of serving/artifacts/ranker.columns.json named a file nothing creates,
	// and since readColumns is fail-fast that made `make serve` unstartable.
	flag.StringVar(&o.columns, "columns", "serving/triton/ranker/1/model.columns.json",
		"the exported graph's column order, written beside the ONNX file by `make onnx`")
	flag.StringVar(&o.popular, "popular", "serving/artifacts/popular.json",
		"the fallback slate; `make popular`")
	flag.StringVar(&o.indexStem, "index-stem", "serving/artifacts/items",
		"item vectors for rung 2 exact search; `make index-vectors`. Empty disables rung 2")

	// db 1: Feast owns 0 on this instance, and sharing a keyspace with a
	// feature store means one FLUSHDB during a materialisation takes every
	// seen-list with it.
	flag.StringVar(&o.redisURL, "redis", "redis://localhost:6379/1", "seen-list and embedding cache")
	// Part L shipped capacity 200 at a 1% target. Bits and hashes are DERIVED
	// from these, and a filter read with different bits than it was written
	// with agrees on nothing -- so these two numbers are the contract, not the
	// derived pair.
	flag.IntVar(&o.seenCap, "seen-capacity", 200, "items per user the filter is sized for")
	flag.Float64Var(&o.seenRate, "seen-fp-rate", 0.01, "target false-positive rate")
	flag.DurationVar(&o.seenTTL, "seen-ttl", sources.SeenTTL, "how long \"recently shown\" lasts")

	flag.StringVar(&o.retrieval, "retrieval", "localhost:50052", "retrieval sidecar")
	flag.StringVar(&o.featuresAPI, "features", "localhost:50053", "feature gateway")
	flag.StringVar(&o.triton, "triton", "localhost:8001", "triton gRPC")
	// HTTP/2 connections PER BACKEND. gRPC-Go serialises a connection's
	// outbound frames through one loopyWriter goroutine, which a CPU profile at
	// 500 rps put at 30.5% of this process's samples -- with ~21% more in
	// futex wakeups and almost no application code anywhere. 1 restores the old
	// behaviour; this is a knob to sweep, not a tuned value. See internal/rpc.
	flag.IntVar(&o.conns, "conns", rpc.DefaultConnections,
		"HTTP/2 connections per backend; 1 was the original single-writer behaviour")
	flag.StringVar(&o.tritonModel, "triton-model", "ranker", "triton model name")

	// Part I measured that at a fixed budget no blend beats giving every slot
	// to the strongest source, so the shipped configuration is one retriever
	// with the whole quota.
	flag.StringVar(&o.sourceName, "source-name", "two_tower", "the retriever's name in quotas")
	flag.IntVar(&o.candidates, "candidates", 400, "candidates entering the ranker")
	flag.IntVar(&o.efSearch, "ef-search", 0, "per-request efSearch, 0 for the sidecar's default")
	flag.IntVar(&o.numResults, "default-results", 10, "slate size when a request does not say")

	flag.DurationVar(&o.deadline, "deadline", 90*time.Millisecond, "end-to-end budget")
	flag.DurationVar(&o.rankDeadline, "rank-deadline", 35*time.Millisecond, "ranker's slice")
	flag.DurationVar(&o.featDeadline, "feature-deadline", 8*time.Millisecond, "user fetch")
	flag.DurationVar(&o.srcDeadline, "source-deadline", 25*time.Millisecond, "per retriever")

	flag.StringVar(&o.modelVersion, "model-version", "", "reported in every response")
	flag.StringVar(&o.indexVersion, "index-version", "", "reported in every response")

	// Empty means no experiment is running and the response's
	// experiment_variant stays empty -- which is honest, and different from a
	// fabricated "control" that no assignment produced.
	flag.StringVar(&o.experimentID, "experiment-id", "",
		"salts the bucketing; two experiments must not split users the same way")
	flag.StringVar(&o.variants, "variants", "",
		"name:percent,... e.g. control:50,treatment:50. Summing under 100 leaves a holdback")
	flag.Parse()
	return o
}

// readColumns reads the order the exported graph was built for.
//
// Read, never defaulted. The column order is the one part of the serving
// contract that no downstream shape check can catch: a permuted matrix has the
// right width and dtype, Triton is happy, and the model scores a different
// world in silence. A default here would be this file asserting the order
// rather than the export declaring it.
func readColumns(path string) ([]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading column order: %w", err)
	}
	var decoded struct {
		Features []string `json:"features"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", path, err)
	}
	if len(decoded.Features) == 0 {
		return nil, fmt.Errorf("%s declares no features", path)
	}
	return decoded.Features, nil
}

// namedProbe is one backend's readiness check, for the startup warm-up.
//
// A slice rather than a map, so the order is the order they are tried and the
// log reads the same way every time.
type namedProbe struct {
	name  string
	check func(context.Context) error
}

// warmConnections dials every backend once, before the first real request.
//
// **grpc.NewClient does not connect.** It returns before any socket exists, so
// the first RPC through each client pays TCP and HTTP/2 setup -- and with three
// backends reached from three different pipeline stages, that cost is spread
// across the first three requests, one each. Measured on this stack:
//
//	1   features 8ms (its deadline), retrieval 25ms (its deadline)  -> degraded
//	2   rank 35ms (its deadline)                                    -> degraded
//	3   features 1ms, retrieval 2ms, rank 5ms                       -> clean
//
// One connection per request, one degraded slate each. That is indistinguishable
// from a slow dependency in every signal the pipeline emits, and it was read as
// one four separate times before the cause was named. A benchmark that does not
// warm first measures this and reports it as p99.
//
// **FAIL-SOFT, deliberately.** This file's doctrine is fail-fast on artifacts
// and fail-soft on services: an artifact that will not load is wrong for every
// request that will ever arrive, but a backend that is briefly down is wrong
// only for the requests that arrive while it is down, and gRPC reconnects on
// its own. So every result is logged and none is fatal -- refusing to start
// here would turn a blip into an outage, which is exactly what the dialling
// code above declines to do.
func warmConnections(ctx context.Context, probes []namedProbe) {
	for _, probe := range probes {
		started := time.Now()
		inner, cancel := context.WithTimeout(ctx, 5*time.Second)
		err := probe.check(inner)
		cancel()

		took := time.Since(started).Round(time.Millisecond)
		if err != nil {
			log.Printf("  warm       %-9s FAILED in %v (%v); its first request pays setup",
				probe.name, took, err)
			continue
		}
		log.Printf("  warm       %-9s ok in %v", probe.name, took)
	}
}

// probe answers the health endpoint.
//
// It asks the DEPENDENCIES rather than reporting a flag set at startup: "the
// process came up" says nothing about whether Triton still has the model
// loaded or whether the gateway can still read Redis, and those are what a
// readiness check exists to answer.
type probe struct {
	gateway  *sources.Gateway
	ranker   *ranking.Client
	sidecar  *sources.Sidecar
	efSearch int32
}

func (p probe) Ready(ctx context.Context) error {
	if err := p.ranker.Ready(ctx); err != nil {
		return err
	}
	return p.gateway.Ready(ctx)
}

// Kind reports what last answered a retrieval, not what is configured. A
// sidecar that quietly failed over to exact search, or an orchestrator running
// on rung 2, is correct and much slower -- and invisible everywhere else.
func (p probe) Kind() string {
	kind, _ := p.sidecar.LastIndex()
	if kind == "" {
		return "unknown"
	}
	return kind
}

func (p probe) EFSearch() int32 { return p.efSearch }

func main() {
	if err := run(parse()); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

func run(o options) error {
	// Tracing first, so the artifact loads below are inside it if anything
	// ever instruments them.
	shutdownTracing, err := tracing.Setup(
		context.Background(), "recsys-orchestrator", o.otlpEndpoint, o.modelVersion, o.traceRatio,
	)
	if err != nil {
		// Fail fast: a misconfigured collector endpoint is wrong for every
		// request, and starting untraced would mean discovering that from the
		// absence of data rather than from a message.
		return err
	}
	defer func() {
		// Its own context: the request context is cancelled by the time this
		// runs, and a shutdown on a dead context drops the last batch of
		// spans -- which are the ones from whatever was happening at shutdown.
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = shutdownTracing(ctx)
	}()

	// --- Artifacts: fail fast -----------------------------------------------
	ids, loadErr := server.LoadIDs(o.itemMap)
	if loadErr != nil {
		return loadErr
	}
	catalog, err := catalogue.Load(o.cataloguE)
	if err != nil {
		return err
	}
	columns, err := readColumns(o.columns)
	if err != nil {
		return err
	}
	fallback, err := popular.Load(o.popular)
	if err != nil {
		return err
	}

	variants, err := parseVariants(o.variants)
	if err != nil {
		return fmt.Errorf("parsing -variants: %w", err)
	}
	experiment := experiments.Experiment{ID: o.experimentID, Variants: variants}
	if len(variants) > 0 && o.experimentID == "" {
		// Refused rather than defaulted. An unsalted experiment buckets users
		// by identity alone, so it correlates with every other experiment ever
		// run at this salt -- and the damage is invisible in this experiment's
		// own numbers.
		return fmt.Errorf("-variants needs -experiment-id; an unsalted split correlates with every other")
	}
	if total := experiment.Allocated(); total > 100 {
		// Past 100 the straddling arm is silently truncated and everything
		// after it is unreachable. Both show up as a dashboard nobody can
		// explain, so this is refused at startup instead.
		return fmt.Errorf("-variants allocate %.1f%%, which truncates arms silently", total)
	}

	// --- Redis: the seen-list and rung 2 ------------------------------------
	//
	// Parsed and dialled here, but NOT pinged: go-redis connects lazily and
	// reconnects on its own, and the pipeline already degrades correctly when
	// the seen-list errors (it fails OPEN -- re-showing an item is a visible
	// annoyance, dropping candidates it cannot vouch for silently shrinks the
	// slate).
	redisOptions, err := redis.ParseURL(o.redisURL)
	if err != nil {
		return fmt.Errorf("parsing -redis: %w", err)
	}
	cache := redis.NewClient(redisOptions)
	defer func() { _ = cache.Close() }()

	bits, hashes := sources.Sizing(o.seenCap, o.seenRate)
	seen := sources.NewSeenList(cache, bits, hashes)
	seen.TTL = o.seenTTL

	// --- Services: fail soft ------------------------------------------------
	// The client handler is what injects W3C tracecontext into outgoing
	// metadata, so a span created here continues inside the Python sidecars
	// rather than each hop starting its own trace. A fan-out whose branches
	// are separate traces is exactly the picture tracing exists to avoid.
	traced := grpc.WithStatsHandler(otelgrpc.NewClientHandler())

	gateway, err := sources.DialGateway(o.featuresAPI, o.conns, traced)
	if err != nil {
		return err
	}
	defer func() { _ = gateway.Close() }()
	gateway.UserColumns = nil // checked against what the tower reports, not pinned here
	gateway.ItemColumns = nil

	sidecar, err := sources.DialSidecar(
		o.retrieval, o.sourceName, o.candidates, o.conns, o.srcDeadline, traced,
	)
	if err != nil {
		return err
	}
	defer func() { _ = sidecar.Close() }()
	sidecar.EFSearch = int32(o.efSearch)

	// Rung 2 of ADR 0013's ladder: exact search here, against the embedding
	// the sidecar cached. Optional, and its absence is REPORTED rather than
	// assumed -- a server running without it drops straight from rung 1 to
	// popularity, which is a much worse slate than the ladder promises and is
	// otherwise indistinguishable from a healthy one until the sidecar fails.
	rungTwo := "disabled"
	if o.indexStem != "" {
		flat, loadErr := index.Load(o.indexStem)
		if loadErr != nil {
			// A warning, not a fatal: the index artifact is only needed while
			// the sidecar is down, so refusing to start over it would turn a
			// degraded-path gap into a total outage.
			log.Printf("WARNING: rung 2 unavailable (%v)", loadErr)
		} else {
			sidecar.Fallback = sources.NewCachedSearch(cache, flat)
			rungTwo = fmt.Sprintf("%d items x %dd, %s",
				flat.Meta().Rows, flat.Dim(), flat.Meta().Version)
		}
	}

	ranker, err := ranking.Dial(o.triton, o.tritonModel, columns, o.conns, traced)
	if err != nil {
		return err
	}
	defer func() { _ = ranker.Close() }()

	pipeline := &service.Service{
		Retrievers: []service.Retriever{sidecar},
		Users:      gateway,
		Features: &features.Builder{
			Items:     gateway,
			Catalogue: catalog,
			Names:     columns,
		},
		Ranker:    ranker,
		Seen:      seen,
		Catalogue: catalog,
		Fallback:  fallback,
		Config: service.Config{
			// One source, the whole quota. Part I measured that no blend beats
			// this at a fixed budget on this corpus.
			Quotas:          []int{o.candidates},
			MaxCandidates:   o.candidates,
			Deadline:        o.deadline,
			RankDeadline:    o.rankDeadline,
			FeatureDeadline: o.featDeadline,
			MaxHistory:      features.MaxHistory,
			// ADR 0012: MMR is a measured null on this corpus, the cap is a
			// guardrail that binds rarely, and exploration buys 84% more
			// catalogue coverage for -0.0005 NDCG.
			Lambda:       1.0,
			ModelVersion: o.modelVersion,
			IndexVersion: o.indexVersion,
		},
	}

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", o.port))
	if err != nil {
		return fmt.Errorf("listening on %d: %w", o.port, err)
	}

	// The default registry carries Go runtime and process collectors as well,
	// which is what makes "is this GC or is this the ranker" answerable
	// without adding anything.
	recorder := metrics.New(prometheus.DefaultRegisterer)
	scrape := metrics.Serve(o.metricsAddr, prometheus.DefaultGatherer)
	defer func() { _ = scrape.Close() }()

	// otelgrpc's StatsHandler, not an interceptor: it sees stream events as
	// well as unary calls, and it is where the incoming W3C tracecontext is
	// picked up -- without it every request starts a NEW trace and the
	// waterfall stops at this service's edge.
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(recorder.Interceptor()),
		grpc.StatsHandler(otelgrpc.NewServerHandler()),
	)
	pb.RegisterRecommenderServer(grpcServer, &server.Recommender{
		Service: pipeline,
		IDs:     ids,
		Probe: probe{
			gateway:  gateway,
			ranker:   ranker,
			sidecar:  sidecar,
			efSearch: int32(o.efSearch),
		},
		Experiment: experiment,
		Observer:   recorder,
	})
	// The standard gRPC health service as well as our own Health RPC: k8s
	// probes speak this one (infra/k8s uses `grpc: {port: 50051}`), and ours
	// carries the index detail theirs has no field for.
	healthpb.RegisterHealthServer(grpcServer, health.NewServer())

	log.Printf("orchestrator on :%d, metrics on %s", o.port, o.metricsAddr)
	// GOMAXPROCS beside NumCPU, because they can differ and the difference is
	// invisible everywhere else. A fan-out pipeline pinned to one P handles
	// concurrent requests one at a time and misses per-stage deadlines while
	// the machine sits idle -- which reads as a slow DEPENDENCY in every signal
	// this service emits, since the stage timer cannot tell "the sidecar was
	// slow" from "this process never got scheduled to read the reply".
	//
	// Go reads the cgroup CPU limit in recent versions and the environment
	// always, so this is not a constant even on one machine.
	log.Printf("  runtime    GOMAXPROCS %d of %d CPUs, go %s",
		runtime.GOMAXPROCS(0), runtime.NumCPU(), runtime.Version())
	if o.otlpEndpoint == "" {
		log.Printf("  tracing    off")
	} else {
		log.Printf("  tracing    %s at ratio %.2f (tail-sample on %s in the collector)",
			o.otlpEndpoint, o.traceRatio, tracing.AttrDegraded)
	}
	log.Printf("  retrieval  %s", o.retrieval)
	log.Printf("  features   %s", o.featuresAPI)
	log.Printf("  triton     %s/%s", o.triton, o.tritonModel)
	log.Printf("  conns      %d per backend (1 = one loopyWriter, the measured limit)", o.conns)
	log.Printf("  catalogue  %s (%d rows, %s)", o.cataloguE, catalog.Size(), catalog.Version())
	log.Printf("  columns    %d, from %s", len(columns), o.columns)
	log.Printf("  budget     %v end to end, %v rank, %v features", o.deadline, o.rankDeadline, o.featDeadline)
	log.Printf("  fallback   %d items, %s", fallback.Size(), fallback.Version())
	log.Printf("  seen-list  %d bits, %d hashes (capacity %d at %.3f), ttl %v",
		bits, hashes, o.seenCap, o.seenRate, o.seenTTL)
	log.Printf("  rung 2     %s", rungTwo)
	if len(variants) == 0 {
		log.Printf("  experiment none; experiment_variant will be empty")
	} else {
		log.Printf("  experiment %q, arms %v (%.1f%% allocated)",
			experiment.ID, experiment.Names(), experiment.Allocated())
	}

	// After the configuration log and before Serve: the lines below belong to
	// startup, and a reader wants them beside the config they describe rather
	// than interleaved with the first requests.
	warmConnections(context.Background(), []namedProbe{
		{"features", gateway.Ready},
		{"retrieval", sidecar.Ready},
		{"triton", ranker.Ready},
	})

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-stop
		// GracefulStop rather than Stop: in-flight requests finish. A deploy
		// that severs them turns every rollout into a burst of client errors.
		log.Print("draining")
		grpcServer.GracefulStop()
	}()

	return grpcServer.Serve(listener)
}

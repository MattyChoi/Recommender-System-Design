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
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/server"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/sources"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/tracing"
	"github.com/redis/go-redis/v9"
)

type options struct {
	// --- transport ---
	port        int
	metricsAddr string

	// --- artifacts ---
	itemMap     string
	catalogue   string
	popular     string
	columns     string
	indexStem   string
	contentStem string

	// --- seen list ---
	redisURL string
	seenCap  int
	seenRate float64
	seenTTL  time.Duration

	// --- microservices ---
	featuresAPI  string
	retrievalAPI string
	tritonAPI    string
	tritonModel  string
	conns        int

	// --- deadline budgets ---
	deadline     time.Duration
	rankDeadline time.Duration
	featDeadline time.Duration
	srcDeadline  time.Duration

	// --- retrieval api parameters ---
	sourceName string
	candidates int
	efSearch   int
	numResults int

	// --- reranking policies ----
	lambda         float64
	capPerCategory int
	epsilon        float64
	exploreSlots   int

	// --- experiments ---
	experimentID string
	variants     string

	// --- observability ---
	otlpEndpoint string
	traceRatio   float64

	// --- versioning ---
	modelVersion string
	indexVersion string
}

type namedProbe struct {
	name  string
	check func(context.Context) error
}

type healthProbe struct {
	gateway  *sources.Gateway
	index    *sources.ANNIndex
	ranker   *ranking.Client
	efSearch int32
}

func (p healthProbe) Kind() string {
	kind, _ := p.index.LastIndex()
	if kind == "" {
		return "unknown"
	}
	return kind
}

func (p healthProbe) EFSearch() int32 {
	return p.efSearch
}

func (p healthProbe) Ready(ctx context.Context) error {
	if err := p.gateway.Ready(ctx); err != nil {
		return err
	}
	if err := p.index.Ready(ctx); err != nil {
		return err
	}
	return p.ranker.Ready(ctx)
}

// readColumns reads the order the exported graph was built for.
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

const warmTimeout = 5 * time.Second

// warmConnections dials every backend once, before the first real request.
func warmConnections(ctx context.Context, probes []namedProbe) {
	for _, probe := range probes {
		started := time.Now()
		inner, cancel := context.WithTimeout(ctx, warmTimeout)
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

// parseVariants reads "control:50,treatment:50".
//
// must follow this string convention
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

	flag.IntVar(&o.port, "port", 50051, "gRPC port")
	flag.StringVar(&o.metricsAddr, "metrics", ":9102", "Prometheus scrape address")

	flag.StringVar(&o.itemMap, "item-map", "serving/artifacts/item_map.json",
		"string id to internal index, from `make item-map`")
	flag.StringVar(&o.catalogue, "catalogue", "serving/artifacts/catalogue.json",
		"per-item category indices, from `make catalogue`")
	flag.StringVar(&o.popular, "popular", "serving/artifacts/popular.json",
		"the retrieval fallback slate, from `make popular`")
	flag.StringVar(&o.columns, "columns", "serving/triton/ranker/1/model.columns.json",
		"the exported graph's column order, written beside the ONNX file by `make onnx`")

	// EMPTY defaults, both of them. These two artifacts are optional.
	flag.StringVar(&o.indexStem, "index-stem", "",
		"stem of the rung-2 item table (`make index-vectors`); empty disables FAISS")
	flag.StringVar(&o.contentStem, "content-stem", "",
		"stem of the content table for MMR (`make content-vectors`); empty disables MMR")

	flag.StringVar(&o.redisURL, "redis", "redis://localhost:6379", "seen-list and embedding cache")
	flag.IntVar(&o.seenCap, "seen-capacity", 200, "items per user the filter is sized for")
	flag.Float64Var(&o.seenRate, "seen-fp-rate", 0.01, "target false-positive rate")
	flag.DurationVar(&o.seenTTL, "seen-ttl", sources.SeenTTL, "how long \"recently shown\" lasts")

	flag.StringVar(&o.featuresAPI, "features", "localhost:50053", "the Feast gateway")
	flag.StringVar(&o.retrievalAPI, "retrieval", "localhost:50052", "the FAISS sidecar")
	flag.StringVar(&o.tritonAPI, "triton", "localhost:8001", "Triton's gRPC port, NOT 8000")
	flag.IntVar(&o.conns, "conns", 4,
		"HTTP/2 connections per upstream; one connection serialises every stream on it")
	flag.StringVar(&o.tritonModel, "triton-model", "ranker", "triton model name")

	flag.StringVar(&o.sourceName, "source-name", "two_tower", "the retriever's name in quotas")
	flag.IntVar(&o.candidates, "candidates", 400, "candidates entering the ranker")
	flag.IntVar(&o.efSearch, "ef-search", 0, "per-request efSearch, 0 for the index sidecar's default")
	flag.IntVar(&o.numResults, "default-results", 10, "slate size when a request does not say")

	flag.DurationVar(&o.deadline, "deadline", 90*time.Millisecond, "end-to-end budget")
	flag.DurationVar(&o.rankDeadline, "rank-deadline", 35*time.Millisecond, "ranker's slice")
	flag.DurationVar(&o.featDeadline, "feature-deadline", 8*time.Millisecond, "user fetch")
	flag.DurationVar(&o.srcDeadline, "source-deadline", 25*time.Millisecond, "per retriever")
	flag.Float64Var(&o.lambda, "lambda", 1.0,
		"MMR's weight on relevance. 1.0, makes MMR a no-op")
	// -1 disables, not 0: a cap of zero means "no slots for any category",
	// which is a real and catastrophic setting. Config.CapPerCategory is a
	// *int for the same reason, and run() converts.
	flag.IntVar(&o.capPerCategory, "cap-per-category", 3,
		"max slots one subcategory may occupy; -1 disables the cap")
	flag.Float64Var(&o.epsilon, "epsilon", 0.1,
		"chance an exploration slot is filled at random (ADR 0012)")
	flag.IntVar(&o.exploreSlots, "explore-slots", 2, "how many of the LAST slots explore")

	flag.StringVar(&o.experimentID, "experiment-id", "",
		"salts the bucketing; required whenever -variants is set")
	flag.StringVar(&o.variants, "variants", "",
		"e.g. control:50,treatment:50 -- percentages explicit, never inferred")

	flag.StringVar(&o.otlpEndpoint, "otlp", "", "OTLP collector, e.g. localhost:4317. Empty disables tracing")
	flag.Float64Var(&o.traceRatio, "trace-ratio", 1.0, "head sampling ratio")

	flag.StringVar(&o.modelVersion, "model-version", "", "reported in every response")
	flag.StringVar(&o.indexVersion, "index-version", "", "reported in every response")

	flag.Parse()
	return o
}

func run(o options) error {
	// Tracing first, so the artifact loads below are inside it if anything
	// ever instruments them.
	shutdownTracing, err := tracing.Setup(
		context.Background(), "recsys-orchestrator", o.otlpEndpoint, o.modelVersion, o.traceRatio,
	)
	if err != nil {
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

	ids, err := server.LoadIDs(o.itemMap)
	if err != nil {
		return err
	}
	catalog, err := catalogue.Load(o.catalogue)
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

	var flat *index.Flat
	if o.indexStem == "" {
		log.Printf("index vectors missing (no -index-stem; exact search fallback is disabled)")
	} else if flat, err = index.Load(o.indexStem); err != nil {
		log.Printf("index vectors failed to load from %s (exact search fallback is disabled): %v", o.indexStem, err)
	}

	if o.contentStem == "" {
		log.Printf("mmr     disabled (no -content-stem; the diversity term has no vectors)")
	} else if err := catalog.LoadVectors(o.contentStem); err != nil {
		return fmt.Errorf("content vectors: %w", err)
	}

	if err := catalog.CheckMMR(o.lambda); err != nil {
		return err
	}

	// --- Log all loaded artifacts ----
	log.Printf("ids %s (%d) | catalogue %s (%d) | popular %s (%d) | columns %d",
		ids.Version(), ids.Size(),
		catalog.Version(), catalog.Size(),
		fallback.Version(), fallback.Size(),
		len(columns))
	if flat != nil {
		log.Printf("Exact Search Fallback  %s  %dx%d  base %d",
			flat.Meta().Version, flat.Meta().Rows, flat.Dim(), flat.Meta().BaseIndex)
	}
	if catalog.HasVectors() {
		log.Printf("mmr   %s  %dx%d  lambda %.2f",
			catalog.VectorVersion(), catalog.Size(), catalog.Dim(), o.lambda)
	}

	// create experiment variants
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

	// --- Redis: the seen-list and the user embedding cache ------------------------
	redisOptions, err := redis.ParseURL(o.redisURL)
	if err != nil {
		return fmt.Errorf("parsing -redis: %w", err)
	}

	// One client per logical DB: a go-redis client SELECTs its DB once per
	// pooled connection
	seenOptions := *redisOptions
	seenOptions.DB = sources.SeenDB
	seenClient := redis.NewClient(&seenOptions)
	defer func() { _ = seenClient.Close() }()

	embeddingOptions := *redisOptions
	embeddingOptions.DB = sources.EmbeddingDB
	embeddingClient := redis.NewClient(&embeddingOptions)
	defer func() { _ = embeddingClient.Close() }()

	bits, hashes := sources.Sizing(o.seenCap, o.seenRate)
	seen := sources.NewSeenList(seenClient, bits, hashes)
	seen.TTL = o.seenTTL

	// dial the microservices
	gateway, err := sources.DialGateway(o.featuresAPI, o.conns)
	if err != nil {
		return err
	}
	defer func() { _ = gateway.Close() }()
	// disable column checks, checked against model input params
	gateway.UserColumns = nil
	gateway.ItemColumns = nil

	index, err := sources.DialIndex(o.retrievalAPI, o.sourceName, o.candidates, o.conns, int32(o.efSearch), o.srcDeadline)
	if err != nil {
		return err
	}
	defer func() { _ = index.Close() }()

	if flat != nil {
		index.Fallback = sources.NewCachedSearch(embeddingClient, flat)
	}

	ranker, err := ranking.Dial(o.tritonAPI, o.tritonModel, columns, o.conns)
	if err != nil {
		return err
	}
	defer func() { _ = ranker.Close() }()

	pipeline := &service.Service{
		Retrievers: []service.Retriever{index},
		Store:      gateway,
		Features: &features.Builder{
			Store:     gateway,
			Catalogue: catalog,
			Names:     columns,
		},
		Ranker:    ranker,
		Seen:      seen,
		Catalogue: catalog,
		Fallback:  fallback,
		Config: service.Config{
			// one source, the whole quota
			Quotas:          []int{o.candidates},
			MaxCandidates:   o.candidates,
			Deadline:        o.deadline,
			RankDeadline:    o.rankDeadline,
			FeatureDeadline: o.featDeadline,
			MaxHistory:      features.MaxHistory,
			Lambda:          1.0,
			ModelVersion:    o.modelVersion,
			IndexVersion:    o.indexVersion,
		},
	}

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", o.port))
	if err != nil {
		return fmt.Errorf("listening on %d: %w", o.port, err)
	}

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
		Probe: healthProbe{
			gateway:  gateway,
			ranker:   ranker,
			index:    index,
			efSearch: int32(o.efSearch),
		},
		Experiment: experiment,
		Observer:   recorder,
	})
	healthpb.RegisterHealthServer(grpcServer, health.NewServer())

	log.Printf("orchestrator on :%d, metrics on %s", o.port, o.metricsAddr)
	log.Printf("  runtime    GOMAXPROCS %d of %d CPUs, go %s",
		runtime.GOMAXPROCS(0), runtime.NumCPU(), runtime.Version())
	if o.otlpEndpoint == "" {
		log.Printf("  tracing    off")
	} else {
		log.Printf("  tracing    %s at ratio %.2f (tail-sample on %s in the collector)",
			o.otlpEndpoint, o.traceRatio, tracing.AttrDegraded)
	}
	log.Printf("  retrieval  %s", o.retrievalAPI)
	log.Printf("  features   %s", o.featuresAPI)
	log.Printf("  triton     %s/%s", o.tritonAPI, o.tritonModel)
	log.Printf("  conns      %d per backend (1 = one loopyWriter, the measured limit)", o.conns)
	log.Printf("  catalogue  %s (%d rows, %s)", o.catalogue, catalog.Size(), catalog.Version())
	log.Printf("  columns    %d, from %s", len(columns), o.columns)
	log.Printf("  seen-list  %d bits, %d hashes (capacity %d at %.3f), ttl %v",
		bits, hashes, o.seenCap, o.seenRate, o.seenTTL)
	log.Printf("  budget     %v end to end, %v rank, %v features", o.deadline, o.rankDeadline, o.featDeadline)
	if len(variants) == 0 {
		log.Printf("  experiment none; experiment_variant will be empty")
	} else {
		log.Printf("  experiment %q, arms %v (%.1f%% allocated)",
			experiment.ID, experiment.Names(), experiment.Allocated())
	}

	warmConnections(context.Background(), []namedProbe{
		{"features", gateway.Ready},
		{"retrieval", index.Ready},
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

func main() {
	if err := run(parse()); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/catalogue"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/index"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/popular"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/server"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/sources"
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

	// --- microservices ---
	featuresAPI  string
	retrievalAPI string
	tritonAPI    string
	conns        int

	// --- deadline budgets ---
	srcDeadline time.Duration

	// --- retrieval api parameters ---
	sourceName string
	candidates int
	efSearch   int

	// --- reranking policies ----
	lambda         float64
	capPerCategory int
	epsilon        float64
	exploreSlots   int

	// --- experiments ---
	experimentID string
	variants     string

	// --- observability ---
	traceRatio float64
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

	flag.StringVar(&o.featuresAPI, "features", "localhost:50053", "the Feast gateway")
	flag.StringVar(&o.retrievalAPI, "retrieval", "localhost:50052", "the FAISS sidecar")
	flag.StringVar(&o.tritonAPI, "triton", "localhost:8001", "Triton's gRPC port, NOT 8000")
	flag.IntVar(&o.conns, "conns", 4,
		"HTTP/2 connections per upstream; one connection serialises every stream on it")

	flag.StringVar(&o.sourceName, "source-name", "two_tower", "the retriever's name in quotas")
	flag.IntVar(&o.candidates, "candidates", 400, "candidates entering the ranker")
	flag.IntVar(&o.efSearch, "ef-search", 0, "per-request efSearch, 0 for the index sidecar's default")

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

	flag.Float64Var(&o.traceRatio, "trace-ratio", 1.0, "head sampling ratio")

	flag.Parse()
	return o
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

func run(o options) error {
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
		return fmt.Errorf("index vectors: %w", err)
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

	// dial the microservices
	gateway, err := sources.DialGateway(o.featuresAPI, o.conns)
	if err != nil {
		return err
	}
	defer gateway.Close()

	index, err := sources.DialIndex(o.retrievalAPI, o.sourceName, o.candidates, o.conns, int32(o.efSearch), o.srcDeadline)
	if err != nil {
		return err
	}
	defer index.Close()

	log.Printf("  retrieval  %s", o.retrievalAPI)
	log.Printf("  features   %s", o.featuresAPI)
	log.Printf("  conns      %d per backend (1 = one loopyWriter, the measured limit)", o.conns)

	return nil
}

func main() {
	if err := run(parse()); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

package sources

import (
	"context"
	"net"
	"strings"
	"testing"

	"google.golang.org/grpc"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
)

// Real servers on loopback, not fakes: what is under test is that Ready
// reaches EVERY address in a comma-separated target, which a fake client
// standing in for the whole pool cannot show.

type healthServer struct {
	pb.UnimplementedRetrievalServer

	ready   bool
	version string
	age     int64
}

func (h *healthServer) Health(
	context.Context, *pb.RetrievalHealthRequest,
) (*pb.RetrievalHealthResponse, error) {
	return &pb.RetrievalHealthResponse{
		Ready: h.ready, Detail: "loading", IndexKind: "hnsw", IndexVersion: h.version, ItemCount: 10,
	}, nil
}

// featuresHealth adapts the same server to the Features service, whose Health
// takes and returns different messages under the same method name.
type featuresHealth struct {
	pb.UnimplementedFeaturesServer
	*healthServer
}

func (f featuresHealth) Health(
	context.Context, *pb.FeaturesHealthRequest,
) (*pb.FeaturesHealthResponse, error) {
	return &pb.FeaturesHealthResponse{
		Ready: f.ready, Detail: "store unreadable", OldestFeatureAgeSeconds: f.age,
	}, nil
}

// serveAll starts one loopback server per backend and returns the
// comma-separated target that reaches them all.
func serveAll(t *testing.T, register func(*grpc.Server, *healthServer), backends ...*healthServer) string {
	t.Helper()
	addresses := make([]string, len(backends))
	for index, backend := range backends {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		server := grpc.NewServer()
		register(server, backend)
		go func() { _ = server.Serve(listener) }()
		t.Cleanup(server.Stop)
		addresses[index] = listener.Addr().String()
	}
	return strings.Join(addresses, ",")
}

func registerRetrieval(server *grpc.Server, backend *healthServer) {
	pb.RegisterRetrievalServer(server, backend)
}

func registerFeatures(server *grpc.Server, backend *healthServer) {
	pb.RegisterFeaturesServer(server, featuresHealth{healthServer: backend})
}

func dialIndex(t *testing.T, target string) *ANNIndex {
	t.Helper()
	// 8 connections over the backends: Ready must still ask each ONCE.
	index, err := DialIndex(target, "two_tower", 100, 8, 0, 0)
	if err != nil {
		t.Fatalf("DialIndex: %v", err)
	}
	t.Cleanup(func() { _ = index.Close() })
	return index
}

func TestIndexReadyNamesTheOneProcessThatIsNot(t *testing.T) {
	sick := &healthServer{ready: false, version: "v1"}
	target := serveAll(t, registerRetrieval,
		&healthServer{ready: true, version: "v1"}, sick, &healthServer{ready: true, version: "v1"})

	err := dialIndex(t, target).Ready(context.Background())
	if err == nil {
		t.Fatal("one of three processes is not ready, so the source is not ready")
	}
	sickAddress := strings.Split(target, ",")[1]
	if !strings.Contains(err.Error(), sickAddress) || !strings.Contains(err.Error(), "loading") {
		t.Errorf("the error should name %s and carry its detail, got %v", sickAddress, err)
	}
	if strings.Count(err.Error(), "not ready") != 1 {
		t.Errorf("only the one process should be reported, got %v", err)
	}
}

func TestIndexReadyRefusesProcessesOnDifferentVersions(t *testing.T) {
	target := serveAll(t, registerRetrieval,
		&healthServer{ready: true, version: "v1"}, &healthServer{ready: true, version: "v2"})

	err := dialIndex(t, target).Ready(context.Background())
	if err == nil || !strings.Contains(err.Error(), "disagree") {
		t.Errorf("a split index version must not pass as ready, got %v", err)
	}
}

func TestIndexReadyWhenEveryProcessIs(t *testing.T) {
	target := serveAll(t, registerRetrieval,
		&healthServer{ready: true, version: "v1"}, &healthServer{ready: true, version: "v1"})

	index := dialIndex(t, target)
	if err := index.Ready(context.Background()); err != nil {
		t.Fatalf("Ready: %v", err)
	}
	if kind, version := index.LastIndex(); kind != "hnsw" || version != "v1" {
		t.Errorf("LastIndex %q %q, want hnsw v1", kind, version)
	}
}

func TestGatewayReadyChecksEveryProcessAndKeepsTheOldestAge(t *testing.T) {
	target := serveAll(t, registerFeatures,
		&healthServer{ready: true, age: 30}, &healthServer{ready: true, age: 900})
	gateway, err := DialGateway(target, 8)
	if err != nil {
		t.Fatalf("DialGateway: %v", err)
	}
	t.Cleanup(func() { _ = gateway.Close() })

	if err := gateway.Ready(context.Background()); err != nil {
		t.Fatalf("Ready: %v", err)
	}
	// The stalest process is the one that bounds how fresh a feature can be.
	if age := gateway.storeAge.Load(); age != 900 {
		t.Errorf("store age %d, want the oldest (900)", age)
	}

	down := serveAll(t, registerFeatures,
		&healthServer{ready: true}, &healthServer{ready: false})
	gateway, err = DialGateway(down, 8)
	if err != nil {
		t.Fatalf("DialGateway: %v", err)
	}
	t.Cleanup(func() { _ = gateway.Close() })
	if err := gateway.Ready(context.Background()); err == nil ||
		!strings.Contains(err.Error(), strings.Split(down, ",")[1]) {
		t.Errorf("the not-ready process should be named, got %v", err)
	}
}

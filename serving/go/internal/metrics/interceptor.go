package metrics

import (
	"context"
	"net/http"
	"net/http/pprof"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

// Interceptor records every unary call's code and latency.
//
// An interceptor rather than instrumentation inside each handler, for one
// reason that matters: it runs on the paths that DO NOT reach a handler --
// a request rejected by validation, a deadline that expired in flight, a panic
// recovered further up. Those are exactly the requests an error-rate alert is
// about, and a counter incremented at the bottom of a handler never sees them.
func (p *Pipeline) Interceptor() grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		request any,
		_ *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (any, error) {
		started := time.Now()
		response, err := handler(ctx, request)
		// status.Code maps a nil error to OK and an unknown one to Unknown, so
		// the label set stays closed however a handler fails.
		p.Request(status.Code(err).String(), time.Since(started))
		return response, err
	}
}

// Handler serves the scrape endpoint.
func Handler(gatherer prometheus.Gatherer) http.Handler {
	return promhttp.HandlerFor(gatherer, promhttp.HandlerOpts{})
}

// Serve runs the scrape endpoint on its own listener.
//
// A separate port from gRPC, deliberately. Scrapes must keep working when the
// serving port is saturated -- an overloaded service is precisely when its
// metrics matter, and sharing a listener means the scrape queues behind the
// traffic it is trying to describe. It also keeps the metrics off the public
// surface when the gRPC port is the one exposed.
// It also carries pprof, for the reason below.
func Serve(address string, gatherer prometheus.Gatherer) *http.Server {
	mux := http.NewServeMux()
	mux.Handle("/metrics", Handler(gatherer))

	// pprof on the SAME listener, registered explicitly rather than by
	// importing net/http/pprof for its side effect.
	//
	// The blank import registers on http.DefaultServeMux, which this server
	// does not use -- so the endpoints would exist, answer nothing here, and be
	// reachable from any other DefaultServeMux in the process. Naming the four
	// routes is three more lines and no mystery.
	//
	// **Why it is here at all.** Every per-stage timer in this service measures
	// wall time, so a stage that was slow and a stage whose goroutine was not
	// scheduled produce the same number. Metrics can say WHICH stage; only a
	// profile can say what the process was doing instead. Measured on this
	// build, the orchestrator spends ~1 core at 500 rps -- ~2 ms of CPU per
	// request for what is nominally blend-and-forward -- and no metric this
	// file exposes can account for it.
	//
	// ⚠️ NOT on the gRPC port, and not on a public one: /debug/pprof exposes
	// goroutine stacks and lets a caller start a 30-second CPU profile. It
	// belongs on the same internal listener as the scrape, which is already
	// documented as off the public surface.
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)

	server := &http.Server{
		Addr:    address,
		Handler: mux,
		// A scrape that hangs holds a connection; Prometheus opens a new one
		// every interval and the old ones accumulate until the file
		// descriptors run out -- which looks like a serving failure.
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		// ErrServerClosed is the ordinary shutdown path, not a failure.
		_ = server.ListenAndServe()
	}()
	return server
}

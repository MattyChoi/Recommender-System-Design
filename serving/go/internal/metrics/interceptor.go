package metrics

import (
	"context"
	"net/http"
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
func Serve(address string, gatherer prometheus.Gatherer) *http.Server {
	mux := http.NewServeMux()
	mux.Handle("/metrics", Handler(gatherer))

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

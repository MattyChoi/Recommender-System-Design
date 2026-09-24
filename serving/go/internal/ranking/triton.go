// Package ranking talks to the model server. It holds no model and no feature
// logic: it turns a built matrix into scores and reports honestly when it
// cannot.
//
// gRPC, over Triton's own GRPCInferenceService. An earlier version of this file
// spoke the HTTP/REST v2 API and justified it by payload size -- one request is
// ~100x11 float32, about 4 KB, and JSON encoding of 4 KB is not where a 90 ms
// budget goes. That argument was true and beside the point. The reasons to
// prefer gRPC here are about the CONNECTION, not the bytes:
//
//   - **One multiplexed HTTP/2 connection** instead of a TCP handshake per call
//     under Go's default transport unless carefully pooled.
//   - **No float-to-text round trip.** JSON writes every float as decimal and
//     parses it back: slower, and lossy in the last bit. This project has
//     already had float width flip an argmax once.
//   - **The deadline travels with the call.** A context deadline becomes a gRPC
//     deadline the server honours, rather than a client-side timeout that
//     abandons a request Triton keeps working on. The whole orchestrator is
//     built on per-stage deadlines, so a deadline the callee ignores is a hole
//     in the design.
//   - **Typed messages.** A misspelled field in a JSON body is ignored; in a
//     generated struct it does not compile.
package ranking

import (
	"context"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/tritonpb"
)

// These must match serving/triton/ranker/config.pbtxt and the names the export
// gave the ONNX graph. Wrong here is a clear error from Triton rather than a
// silent one, but the constants sit together so the two files can be diffed by
// eye.
const (
	inputName  = "features"
	outputName = "score"
	// Triton's name for 32-bit float, which is NOT the same string as ONNX's or
	// numpy's. A mismatch is rejected at the server, which is the good case.
	dataTypeFP32 = "FP32"
)

// Client scores against a Triton model over gRPC.
type Client struct {
	client pb.GRPCInferenceServiceClient
	conn   *grpc.ClientConn

	// Model is the name in config.pbtxt.
	Model string
	// Version pins which model version answers. Empty means "whatever Triton
	// considers latest", which is convenient and makes `model_version` in the
	// response the only record of what actually ran.
	Version string
	// Columns is the order the exported graph was built for, read from the
	// sidecar the export wrote beside the model. Empty disables the check,
	// which has to be a deliberate choice.
	Columns []string
}

// Dial opens a connection. It does NOT block on the server being up: gRPC
// reconnects on its own, and a serving process that refuses to start because a
// dependency is briefly down turns a recoverable blip into an outage.
func Dial(target, model string, columns []string, opts ...grpc.DialOption) (*Client, error) {
	opts = append([]grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	}, opts...)
	conn, err := grpc.NewClient(target, opts...)
	if err != nil {
		return nil, fmt.Errorf("dialling triton at %s: %w", target, err)
	}
	return &Client{
		client:  pb.NewGRPCInferenceServiceClient(conn),
		conn:    conn,
		Model:   model,
		Columns: columns,
	}, nil
}

func (c *Client) Close() error { return c.conn.Close() }

// Score sends one request's candidates and returns one logit per candidate.
func (c *Client) Score(ctx context.Context, features service.Features) ([]float64, error) {
	if len(features.Rows) == 0 {
		return nil, nil
	}
	if err := c.checkColumns(features.Names); err != nil {
		return nil, err
	}

	width := len(features.Rows[0])
	flat := make([]float32, 0, len(features.Rows)*width)
	for index, row := range features.Rows {
		// Ragged rows flatten into a correctly-sized buffer of misaligned
		// values, which Triton accepts and the model scores. Checked here
		// because this is the last place the row structure still exists.
		if len(row) != width {
			return nil, fmt.Errorf("row %d has %d columns, row 0 has %d", index, len(row), width)
		}
		flat = append(flat, row...)
	}

	// `contents.fp32_contents` rather than `raw_input_contents`. Triton accepts
	// both; raw avoids one copy and is what its docs reach for at scale. This
	// uses the typed field because protobuf defines the wire encoding of a
	// packed float field, whereas the raw path is a byte buffer whose layout is
	// a convention -- and getting a byte order wrong produces numbers, not an
	// error. At 4 KB per request the copy is not measurable; revisit with a
	// profile showing it, not with an argument.
	request := &pb.ModelInferRequest{
		ModelName:    c.Model,
		ModelVersion: c.Version,
		Inputs: []*pb.ModelInferRequest_InferInputTensor{{
			Name:     inputName,
			Datatype: dataTypeFP32,
			// [batch, features]. Triton's dynamic batcher owns the leading axis
			// and concatenates concurrent requests along it, so this is one
			// request's candidate count rather than the served batch size.
			Shape:    []int64{int64(len(features.Rows)), int64(width)},
			Contents: &pb.InferTensorContents{Fp32Contents: flat},
		}},
		Outputs: []*pb.ModelInferRequest_InferRequestedOutputTensor{{Name: outputName}},
	}

	response, err := c.client.ModelInfer(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("triton ModelInfer: %w", err)
	}

	for _, output := range response.GetOutputs() {
		if output.GetName() != outputName {
			continue
		}
		scores := output.GetContents().GetFp32Contents()
		// One score per candidate, checked. The orchestrator zips scores
		// against items positionally, so a short response would rank each item
		// by the next item's score -- a full, plausible, wrong slate.
		if len(scores) != len(features.Rows) {
			return nil, fmt.Errorf("%d scores for %d candidates", len(scores), len(features.Rows))
		}
		out := make([]float64, len(scores))
		for index, value := range scores {
			out[index] = float64(value)
		}
		return out, nil
	}
	return nil, fmt.Errorf("response carried no %q output", outputName)
}

// checkColumns refuses a matrix built in a different order from the graph's.
//
// This is what makes Features.Names worth carrying. Nothing downstream can
// detect a permuted column: the shape is right, the dtype is right, Triton is
// happy, the model returns plausible scores and the slate is simply wrong.
// Comparing two ordered lists of strings is the entire defence.
func (c *Client) checkColumns(got []string) error {
	if len(c.Columns) == 0 {
		return nil
	}
	if len(got) != len(c.Columns) {
		return fmt.Errorf("%d feature columns, the graph expects %d", len(got), len(c.Columns))
	}
	for index, name := range c.Columns {
		if got[index] != name {
			return fmt.Errorf(
				"column %d is %q, the graph was exported with %q", index, got[index], name,
			)
		}
	}
	return nil
}

// Ready reports whether Triton has the model loaded.
//
// Separate from Score so a readiness probe need not fabricate an inference, and
// so "server down" stays distinguishable from "server up, model not loaded" --
// different pages at 3am.
func (c *Client) Ready(ctx context.Context) error {
	response, err := c.client.ModelReady(
		ctx, &pb.ModelReadyRequest{Name: c.Model, Version: c.Version},
	)
	if err != nil {
		return fmt.Errorf("triton ModelReady: %w", err)
	}
	if !response.GetReady() {
		return fmt.Errorf("model %q is loaded but not ready", c.Model)
	}
	return nil
}

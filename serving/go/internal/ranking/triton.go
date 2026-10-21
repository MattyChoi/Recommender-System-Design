package ranking

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/rpc"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/tritonpb"
)

// These must match serving/triton/ranker/config.pbtxt and the names the export
// gave the ONNX graph
const (
	inputName  = "features"
	outputName = "score"
	// Triton's name for 32-bit float
	dataTypeFP32 = "FP32"
)

type Client struct {
	client pb.GRPCInferenceServiceClient
	conn   *rpc.Balanced

	// Model is the name in config.pbtxt.
	Model string

	Version string

	// Columns is the order the exported graph was built for
	Columns []string
}

func Dial(target, model string, columns []string, conns int, opts ...grpc.DialOption) (*Client, error) {
	opts = append([]grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	}, opts...)
	pool, err := rpc.Dial(target, conns, opts...)
	if err != nil {
		return nil, fmt.Errorf("dialling triton at %s: %w", target, err)
	}

	return &Client{
		client:  pb.NewGRPCInferenceServiceClient(pool),
		conn:    pool,
		Model:   model,
		Columns: columns,
	}, nil
}

func (c *Client) Close() error {
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

const readyTimeout = 2 * time.Second

// Ready reports whether Triton has the model loaded.
func (c *Client) Ready(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, readyTimeout)
	defer cancel()

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

// sends one request's candidates and returns a logit per candidate
func (c *Client) Score(ctx context.Context, features service.Features) ([]float64, error) {
	if len(features.Rows) == 0 {
		return nil, nil
	}
	if err := c.checkColumns(features.Names); err != nil {
		return nil, err
	}

	batch_size := len(features.Rows)
	width := len(features.Rows[0])
	flat := make([]float32, 0, batch_size*width)
	for index, row := range features.Rows {
		if len(row) != width {
			return nil, fmt.Errorf("row %d has %d columns, row 0 has %d", index, len(row), width)
		}
		flat = append(flat, row...)
	}

	request := &pb.ModelInferRequest{
		ModelName:    c.Model,
		ModelVersion: c.Version,
		Inputs: []*pb.ModelInferRequest_InferInputTensor{{
			Name:     inputName,
			Datatype: dataTypeFP32,
			Shape:    []int64{int64(batch_size), int64(width)},
			Contents: &pb.InferTensorContents{Fp32Contents: flat},
		}},
		Outputs: []*pb.ModelInferRequest_InferRequestedOutputTensor{{Name: outputName}},
	}

	response, err := c.client.ModelInfer(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("triton ModelInfer: %w", err)
	}

	for index, output := range response.GetOutputs() {
		if output.GetName() != outputName {
			continue
		}
		scores, err := outputFloats(response, index)
		if err != nil {
			return nil, err
		}
		if len(scores) != batch_size {
			return nil, fmt.Errorf("%d scores for %d candidates", len(scores), batch_size)
		}
		return scores, nil
	}
	return nil, fmt.Errorf("response carried no %q output", outputName)

}

func outputFloats(response *pb.ModelInferResponse, index int) ([]float64, error) {
	if typed := response.GetOutputs()[index].GetContents().GetFp32Contents(); len(typed) > 0 {
		out := make([]float64, len(typed))
		for position, value := range typed {
			out[position] = float64(value)
		}
		return out, nil
	}

	raw := response.GetRawOutputContents()
	if index >= len(raw) {
		return nil, fmt.Errorf(
			"output %d carried neither typed contents nor a raw_output_contents entry "+
				"(%d present)", index, len(raw),
		)
	}
	buffer := raw[index]
	if len(buffer)%4 != 0 {
		return nil, fmt.Errorf("%d bytes is not a whole number of float32", len(buffer))
	}

	out := make([]float64, len(buffer)/4)
	for position := range out {
		bits := binary.LittleEndian.Uint32(buffer[position*4:])
		out[position] = float64(math.Float32frombits(bits))
	}
	return out, nil
}

package ranking

import (
	"context"
	"errors"
	"strings"
	"testing"

	"google.golang.org/grpc"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/tritonpb"
)

// fakeInference stands in for Triton. It EMBEDS the generated client interface
// rather than implementing its twenty-odd methods: the embedded nil satisfies
// the type, and the two methods declared below shadow it. Calling any other
// method panics, which is the behaviour worth having -- a test that silently
// exercised an unimplemented RPC would be testing nothing.
type fakeInference struct {
	pb.GRPCInferenceServiceClient

	captured *pb.ModelInferRequest
	scores   []float32
	// output names the tensor the response carries. Empty means the name the
	// client asks for, i.e. the happy path; setting it to anything else is how a
	// response that carries SOMETHING but not the wanted output is simulated.
	output string
	ready  bool
	err    error
}

func (f *fakeInference) ModelInfer(
	_ context.Context, in *pb.ModelInferRequest, _ ...grpc.CallOption,
) (*pb.ModelInferResponse, error) {
	f.captured = in
	if f.err != nil {
		return nil, f.err
	}
	name := f.output
	if name == "" {
		name = outputName
	}
	return &pb.ModelInferResponse{
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{{
			Name:     name,
			Datatype: dataTypeFP32,
			Shape:    []int64{int64(len(f.scores)), 1},
			Contents: &pb.InferTensorContents{Fp32Contents: f.scores},
		}},
	}, nil
}

func (f *fakeInference) ModelReady(
	_ context.Context, _ *pb.ModelReadyRequest, _ ...grpc.CallOption,
) (*pb.ModelReadyResponse, error) {
	if f.err != nil {
		return nil, f.err
	}
	return &pb.ModelReadyResponse{Ready: f.ready}, nil
}

func newClient(fake *fakeInference, columns []string) *Client {
	return &Client{client: fake, Model: "ranker", Columns: columns}
}

func features(names []string, rows ...[]float32) service.Features {
	return service.Features{Names: names, Rows: rows}
}

var columns = []string{"a", "b"}

// --- The column-order check -------------------------------------------------
//
// This is the reason Features carries Names at all. Nothing downstream detects
// a permuted column: the shape is right, the dtype is right, Triton is happy,
// the model returns plausible scores, and the slate is wrong.

func TestAPermutedColumnOrderIsRefused(t *testing.T) {
	fake := &fakeInference{scores: []float32{1, 2}}
	client := newClient(fake, columns)

	_, err := client.Score(context.Background(), features([]string{"b", "a"}, []float32{1, 2}))

	if err == nil {
		t.Fatal("a permuted order must be refused; nothing downstream can catch it")
	}
	if !strings.Contains(err.Error(), "column 0") {
		t.Errorf("the error should name the offending column, got %q", err)
	}
	if fake.captured != nil {
		t.Error("the request must not reach Triton once the order is known to be wrong")
	}
}

func TestAWrongColumnCountIsRefused(t *testing.T) {
	client := newClient(&fakeInference{}, columns)

	_, err := client.Score(context.Background(), features([]string{"a"}, []float32{1}))

	if err == nil {
		t.Fatal("expected a refusal for 1 column against a 2-column graph")
	}
}

// TestNoConfiguredColumnsDisablesTheCheck is the control: without it, the tests
// above would pass on a client that refused every request.
func TestNoConfiguredColumnsDisablesTheCheck(t *testing.T) {
	fake := &fakeInference{scores: []float32{7}}
	client := newClient(fake, nil)

	got, err := client.Score(context.Background(), features([]string{"whatever"}, []float32{1, 2}))

	if err != nil {
		t.Fatalf("an unconfigured client must not check: %v", err)
	}
	if len(got) != 1 || got[0] != 7 {
		t.Errorf("scores: got %v, want [7]", got)
	}
}

// --- The request that goes on the wire --------------------------------------

func TestTheRequestCarriesShapeDatatypeAndFlattenedRows(t *testing.T) {
	fake := &fakeInference{scores: []float32{0.5, 1.5}}
	client := newClient(fake, columns)

	_, err := client.Score(
		context.Background(),
		features(columns, []float32{1, 2}, []float32{3, 4}),
	)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}

	input := fake.captured.GetInputs()[0]
	if input.GetName() != inputName {
		t.Errorf("input name %q, want %q", input.GetName(), inputName)
	}
	if input.GetDatatype() != dataTypeFP32 {
		t.Errorf("datatype %q, want %q", input.GetDatatype(), dataTypeFP32)
	}
	// [rows, columns]. This is ONE REQUEST's candidate count -- Triton's dynamic
	// batcher owns the leading axis and concatenates concurrent requests along
	// it, so the served batch is routinely larger than what is sent here.
	if shape := input.GetShape(); len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
		t.Errorf("shape %v, want [2 2]", shape)
	}
	// Row-major: row 0 then row 1. Column-major would be the same length and
	// the same numbers, and would score every candidate against another
	// candidate's features.
	want := []float32{1, 2, 3, 4}
	got := input.GetContents().GetFp32Contents()
	if len(got) != len(want) {
		t.Fatalf("flattened %v, want %v", got, want)
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("flattened %v, want %v", got, want)
		}
	}
}

func TestRaggedRowsAreRefused(t *testing.T) {
	fake := &fakeInference{}
	client := newClient(fake, nil)

	_, err := client.Score(
		context.Background(),
		features([]string{"a", "b"}, []float32{1, 2}, []float32{3}),
	)

	if err == nil {
		t.Fatal("ragged rows flatten into a correctly-sized buffer of misaligned values")
	}
	if fake.captured != nil {
		t.Error("nothing should reach Triton once the rows are known to be ragged")
	}
}

func TestAnEmptyCandidateSetSkipsTheCall(t *testing.T) {
	fake := &fakeInference{}
	client := newClient(fake, columns)

	got, err := client.Score(context.Background(), service.Features{Names: columns})

	if err != nil || got != nil {
		t.Fatalf("got %v, %v; want nil, nil", got, err)
	}
	if fake.captured != nil {
		t.Error("an empty request is a round trip nobody needs")
	}
}

// --- The response -----------------------------------------------------------

func TestScoresComeBackInOrder(t *testing.T) {
	fake := &fakeInference{scores: []float32{0.25, -1, 3}}
	client := newClient(fake, nil)

	got, err := client.Score(
		context.Background(),
		features(nil, []float32{1}, []float32{2}, []float32{3}),
	)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	want := []float64{0.25, -1, 3}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("scores %v, want %v", got, want)
		}
	}
}

// TestAShortResponseIsRefused guards the positional zip. The orchestrator pairs
// scores with candidates by index, so two scores for three candidates would
// rank each item by the next item's score -- a full, plausible, wrong slate.
func TestAShortResponseIsRefused(t *testing.T) {
	fake := &fakeInference{scores: []float32{1, 2}}
	client := newClient(fake, nil)

	_, err := client.Score(
		context.Background(),
		features(nil, []float32{1}, []float32{2}, []float32{3}),
	)

	if err == nil {
		t.Fatal("2 scores for 3 candidates must be refused, not zipped")
	}
}

// TestAMissingOutputIsReported covers the branch that falls off the end of the
// output loop. Reaching it needs a response that carries an output under a
// DIFFERENT name: leaving the fake's scores nil would instead return the wanted
// output with zero values and trip the length guard, so the test would pass
// while the branch it claims to cover stayed unexecuted -- the same shape of
// mistake as the fan-out test that would have passed if the bug existed.
func TestAMissingOutputIsReported(t *testing.T) {
	fake := &fakeInference{output: "some_other_tensor", scores: []float32{1}}
	client := newClient(fake, nil)

	_, err := client.Score(context.Background(), features(nil, []float32{1}))

	if err == nil {
		t.Fatal("a response without the wanted output must be an error, not empty scores")
	}
	if !strings.Contains(err.Error(), outputName) {
		t.Errorf("the error should name the output it wanted, got %q", err)
	}
}

func TestATransportErrorIsWrapped(t *testing.T) {
	client := newClient(&fakeInference{err: errors.New("unavailable")}, nil)

	_, err := client.Score(context.Background(), features(nil, []float32{1}))

	if err == nil || !strings.Contains(err.Error(), "ModelInfer") {
		t.Fatalf("the error should name the call that failed, got %v", err)
	}
}

// --- Readiness --------------------------------------------------------------

func TestReadyDistinguishesNotReadyFromUnreachable(t *testing.T) {
	notReady := newClient(&fakeInference{ready: false}, nil)
	if err := notReady.Ready(context.Background()); err == nil {
		t.Error("a loaded-but-not-ready model must report an error")
	}

	unreachable := newClient(&fakeInference{err: errors.New("connection refused")}, nil)
	err := unreachable.Ready(context.Background())
	if err == nil || !strings.Contains(err.Error(), "ModelReady") {
		t.Errorf("an unreachable server should name the call, got %v", err)
	}

	ready := newClient(&fakeInference{ready: true}, nil)
	if err := ready.Ready(context.Background()); err != nil {
		t.Errorf("a ready model must report no error, got %v", err)
	}
}

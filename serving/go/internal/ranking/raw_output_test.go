package ranking

import (
	"encoding/binary"
	"math"
	"testing"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/tritonpb"
)

// rawResponse builds the shape REAL Triton sends.
func rawResponse(name string, vals []float32) *pb.ModelInferResponse {
	buf := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return &pb.ModelInferResponse{
		Outputs:           []*pb.ModelInferResponse_InferOutputTensor{{Name: name}},
		RawOutputContents: [][]byte{buf},
	}
}

func TestRawOutputContentsAreDecoded(t *testing.T) {
	want := []float32{0.25, -1.5, 3.0}
	got, err := outputFloats(rawResponse("score", want), 0)
	if err != nil {
		t.Fatalf("outputFloats: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d scores, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != float64(want[i]) {
			t.Errorf("score %d = %v, want %v", i, got[i], want[i])
		}
	}
}

// The old fakes must keep working, or the fix is a rewrite of the suite.
func TestTypedContentsStillWin(t *testing.T) {
	resp := &pb.ModelInferResponse{
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{{
			Name:     "score",
			Contents: &pb.InferTensorContents{Fp32Contents: []float32{7}},
		}},
	}
	got, err := outputFloats(resp, 0)
	if err != nil || len(got) != 1 || got[0] != 7 {
		t.Fatalf("typed path broken: %v %v", got, err)
	}
}

func TestATruncatedBufferIsRefusedRatherThanRounded(t *testing.T) {
	resp := rawResponse("score", []float32{1, 2})
	resp.RawOutputContents[0] = resp.RawOutputContents[0][:7] // lose a byte
	if _, err := outputFloats(resp, 0); err == nil {
		t.Fatal("a 7-byte float32 buffer must be an error, not one score")
	}
}

func TestAnEmptyResponseIsAnError(t *testing.T) {
	resp := &pb.ModelInferResponse{
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{{Name: "score"}},
	}
	if _, err := outputFloats(resp, 0); err == nil {
		t.Fatal("no contents and no raw bytes must be an error")
	}
}

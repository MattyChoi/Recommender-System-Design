package ranking

import (
	"encoding/binary"
	"math"
	"testing"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/tritonpb"
)

// The bug these cover: every existing test in this package fakes the response
// with the TYPED contents field, and a real Triton server never populates it --
// it writes output tensors into raw_output_contents and leaves contents empty.
// So the client read an empty slice back from a successful inference, reported
// "0 scores for 400 candidates", and the orchestrator degraded the ranker while
// Triton's own stats showed 400 inferences and zero failures.
//
// A fake that is more convenient than the server it stands in for is worse than
// no fake. These build the response the way Triton actually does.

func rawResponse(name string, values []float32) *pb.ModelInferResponse {
	buffer := make([]byte, 4*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint32(buffer[index*4:], math.Float32bits(value))
	}
	return &pb.ModelInferResponse{
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{{Name: name}},
		// Positional: raw_output_contents[i] pairs with outputs[i]. There is no
		// name on it, which is why Score tracks the index rather than matching.
		RawOutputContents: [][]byte{buffer},
	}
}

func TestRawOutputContentsAreDecoded(t *testing.T) {
	want := []float32{-1.5, 0, 0.25, 1024.75}

	got, err := outputFloats(rawResponse(outputName, want), 0)
	if err != nil {
		t.Fatalf("outputFloats: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("%d scores, want %d", len(got), len(want))
	}
	for index, value := range want {
		if got[index] != float64(value) {
			t.Errorf("score %d is %v, want %v", index, got[index], float64(value))
		}
	}
}

func TestTypedContentsStillWinWhenPresent(t *testing.T) {
	// A non-Triton server, or a fake, may populate the typed field. It is tried
	// first, so those keep working.
	response := &pb.ModelInferResponse{
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{{
			Name:     outputName,
			Contents: &pb.InferTensorContents{Fp32Contents: []float32{7, 8}},
		}},
	}

	got, err := outputFloats(response, 0)
	if err != nil {
		t.Fatalf("outputFloats: %v", err)
	}
	if len(got) != 2 || got[0] != 7 || got[1] != 8 {
		t.Errorf("got %v, want [7 8]", got)
	}
}

func TestAnOutputWithNeitherFieldIsAnError(t *testing.T) {
	response := &pb.ModelInferResponse{
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{{Name: outputName}},
	}

	if _, err := outputFloats(response, 0); err == nil {
		t.Fatal("want an error when an output carries no data at all")
	}
}

func TestATruncatedBufferIsRefusedRatherThanRounded(t *testing.T) {
	// Three bytes is not a whole float32. Silently dropping the remainder would
	// return a SHORT score list, and the orchestrator zips scores against items
	// positionally -- every item would take the next item's score.
	response := &pb.ModelInferResponse{
		Outputs:           []*pb.ModelInferResponse_InferOutputTensor{{Name: outputName}},
		RawOutputContents: [][]byte{{1, 2, 3}},
	}

	if _, err := outputFloats(response, 0); err == nil {
		t.Fatal("want an error on a buffer that is not a whole number of float32")
	}
}

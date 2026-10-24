package sources

import (
	"context"
	"errors"
	"strings"
	"testing"

	"google.golang.org/grpc"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
)

// fakeFeatures embeds the generated client, so any RPC this file does not
// shadow panics rather than quietly returning a zero value.
type fakeFeatures struct {
	pb.FeaturesClient

	user     *pb.GetUserResponse
	items    *pb.GetItemsResponse
	health   *pb.FeaturesHealthResponse
	err      error
	captured *pb.GetItemsRequest
}

func (f *fakeFeatures) GetUser(
	_ context.Context, _ *pb.GetUserRequest, _ ...grpc.CallOption,
) (*pb.GetUserResponse, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.user, nil
}

func (f *fakeFeatures) GetItems(
	_ context.Context, in *pb.GetItemsRequest, _ ...grpc.CallOption,
) (*pb.GetItemsResponse, error) {
	f.captured = in
	if f.err != nil {
		return nil, f.err
	}
	return f.items, nil
}

func (f *fakeFeatures) Health(
	_ context.Context, _ *pb.FeaturesHealthRequest, _ ...grpc.CallOption,
) (*pb.FeaturesHealthResponse, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.health, nil
}

var (
	userNames = []string{"user_impressions_24h", "user_clicks_24h"}
	itemNames = []string{"item_impressions_24h", "item_ctr_smoothed"}
)

func newGateway(fake *fakeFeatures) *Gateway {
	return &Gateway{client: fake, UserColumns: userNames, ItemColumns: itemNames}
}

// --- The user fetch ----------------------------------------------------------

func TestFetchCarriesFeaturesHistoryAndTheMissFlag(t *testing.T) {
	gateway := newGateway(&fakeFeatures{user: &pb.GetUserResponse{
		Values:  []float32{10, 2},
		Names:   userNames,
		History: []int32{9, 8},
		Found:   true,
	}})

	user, err := gateway.Fetch(context.Background(), "U1", 50)
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}

	if user.ID != "U1" || len(user.Feats) != 2 || len(user.History) != 2 || !user.Found {
		t.Errorf("got %+v", user)
	}
}

// TestAPermutedUserVectorIsRefused is the reason Names travels with Values.
//
// Nothing downstream detects it: the length is right, the dtype is right, and
// the tower produces a believable embedding for a user who does not exist,
// whose neighbours look entirely reasonable.
func TestAPermutedUserVectorIsRefused(t *testing.T) {
	gateway := newGateway(&fakeFeatures{user: &pb.GetUserResponse{
		Values: []float32{2, 10},
		Names:  []string{userNames[1], userNames[0]},
	}})

	_, err := gateway.Fetch(context.Background(), "U1", 50)

	if err == nil {
		t.Fatal("a permuted order must be refused")
	}
	if !strings.Contains(err.Error(), "column 0") {
		t.Errorf("the error should name the offending column, got %q", err)
	}
}

func TestAShortUserVectorIsRefused(t *testing.T) {
	gateway := newGateway(&fakeFeatures{user: &pb.GetUserResponse{
		Values: []float32{10},
		Names:  userNames[:1],
	}})

	if _, err := gateway.Fetch(context.Background(), "U1", 50); err == nil {
		t.Fatal("a 1-column vector against a 2-column tower must be refused")
	}
}

// TestAnUnconfiguredGatewayDoesNotCheck is the control: without it, the two
// tests above would pass on a client that refused every response.
func TestAnUnconfiguredGatewayDoesNotCheck(t *testing.T) {
	gateway := &Gateway{client: &fakeFeatures{user: &pb.GetUserResponse{
		Values: []float32{1},
		Names:  []string{"whatever"},
	}}}

	if _, err := gateway.Fetch(context.Background(), "U1", 50); err != nil {
		t.Fatalf("an unconfigured gateway must not check: %v", err)
	}
}

func TestATransportErrorNamesTheCall(t *testing.T) {
	gateway := newGateway(&fakeFeatures{err: errors.New("unavailable")})

	_, err := gateway.Fetch(context.Background(), "U1", 50)

	if err == nil || !strings.Contains(err.Error(), "GetUser") {
		t.Errorf("got %v", err)
	}
}

func TestAStoreMissIsNotAnError(t *testing.T) {
	// A genuinely new user has no row. That is the correct answer, and the
	// pipeline distinguishes it from a failed call -- conflating them makes
	// the degraded-source alert fire on normal cold-start traffic.
	gateway := newGateway(&fakeFeatures{user: &pb.GetUserResponse{
		Values: []float32{0, 0}, Names: userNames, Found: false,
	}})

	user, err := gateway.Fetch(context.Background(), "U-new", 50)
	if err != nil {
		t.Fatalf("a miss must not be an error: %v", err)
	}
	if user.Found {
		t.Error("the miss should be reported")
	}
}

// --- The item fetch ----------------------------------------------------------

func TestItemRowsAreReshapedInCandidateOrder(t *testing.T) {
	gateway := newGateway(&fakeFeatures{items: &pb.GetItemsResponse{
		// Row-major: item 7's two columns, then item 9's.
		Values: []float32{1, 2, 3, 4},
		Names:  itemNames,
		Found:  []bool{true, true},
	}})

	rows, names, found, err := gateway.ItemRows(context.Background(), []int32{7, 9})
	if err != nil {
		t.Fatalf("ItemRows: %v", err)
	}

	if len(rows) != 2 || rows[0][0] != 1 || rows[0][1] != 2 || rows[1][0] != 3 {
		t.Errorf("rows %v", rows)
	}
	if len(names) != 2 || len(found) != 2 {
		t.Errorf("names %v found %v", names, found)
	}
}

// TestAPayloadThatDoesNotDivideIsRefused guards the reshape.
//
// A payload that is not an exact multiple of the row width still reshapes into
// SOMETHING: rows offset from the items they describe, so every candidate is
// scored on the next candidate's features. The model does not complain.
func TestAPayloadThatDoesNotDivideIsRefused(t *testing.T) {
	gateway := newGateway(&fakeFeatures{items: &pb.GetItemsResponse{
		Values: []float32{1, 2, 3},
		Names:  itemNames,
	}})

	_, _, _, err := gateway.ItemRows(context.Background(), []int32{7, 9})

	if err == nil {
		t.Fatal("3 values for 2 items x 2 columns must be refused, not reshaped")
	}
	if !strings.Contains(err.Error(), "3 values") {
		t.Errorf("the error should show the arithmetic, got %q", err)
	}
}

func TestMismatchedFoundFlagsAreRefused(t *testing.T) {
	gateway := newGateway(&fakeFeatures{items: &pb.GetItemsResponse{
		Values: []float32{1, 2, 3, 4},
		Names:  itemNames,
		Found:  []bool{true},
	}})

	// The flags are read positionally alongside the rows, so a short list
	// would silently mark later candidates as missing.
	if _, _, _, err := gateway.ItemRows(context.Background(), []int32{7, 9}); err == nil {
		t.Fatal("1 found flag for 2 items must be refused")
	}
}

// TestARowCannotAppendIntoItsNeighbour pins the three-index slice.
//
// The builder appends blend-derived columns to these rows. Sliced without an
// explicit capacity, append would reuse the spare capacity of the shared
// backing array and overwrite the NEXT candidate's features in place -- a
// corruption that only appears once a second column is appended.
func TestARowCannotAppendIntoItsNeighbour(t *testing.T) {
	gateway := newGateway(&fakeFeatures{items: &pb.GetItemsResponse{
		Values: []float32{1, 2, 3, 4},
		Names:  itemNames,
	}})

	rows, _, _, err := gateway.ItemRows(context.Background(), []int32{7, 9})
	if err != nil {
		t.Fatalf("ItemRows: %v", err)
	}

	_ = append(rows[0], 99) //nolint:staticcheck // the append is the test

	if rows[1][0] != 3 {
		t.Errorf("appending to row 0 overwrote row 1: %v", rows[1])
	}
}

func TestAnEmptyCandidateSetSkipsTheCall(t *testing.T) {
	fake := &fakeFeatures{}
	gateway := newGateway(fake)

	rows, _, _, err := gateway.ItemRows(context.Background(), nil)

	if err != nil || rows != nil {
		t.Fatalf("got %v, %v", rows, err)
	}
	if fake.captured != nil {
		t.Error("an empty request is a round trip nobody needs")
	}
}

func TestValuesWithNoColumnNamesAreRefused(t *testing.T) {
	// Without names there is nothing to check the order against, and a
	// zero-width row would make the reshape arithmetic divide by nothing.
	gateway := &Gateway{client: &fakeFeatures{items: &pb.GetItemsResponse{
		Values: []float32{1, 2},
	}}}

	if _, _, _, err := gateway.ItemRows(context.Background(), []int32{7}); err == nil {
		t.Fatal("values with no names must be refused")
	}
}

// --- Health ------------------------------------------------------------------

func TestReadyDistinguishesNotReadyFromUnreachable(t *testing.T) {
	notReady := newGateway(&fakeFeatures{health: &pb.FeaturesHealthResponse{
		Ready: false, Detail: "online store unreadable",
	}})
	err := notReady.Ready(context.Background())
	if err == nil || !strings.Contains(err.Error(), "unreadable") {
		t.Errorf("a not-ready gateway should carry its detail, got %v", err)
	}

	unreachable := newGateway(&fakeFeatures{err: errors.New("connection refused")})
	if err := unreachable.Ready(context.Background()); err == nil ||
		!strings.Contains(err.Error(), "Health") {
		t.Errorf("an unreachable gateway should name the call, got %v", err)
	}

	ready := newGateway(&fakeFeatures{health: &pb.FeaturesHealthResponse{Ready: true}})
	if err := ready.Ready(context.Background()); err != nil {
		t.Errorf("a ready gateway must report no error, got %v", err)
	}
}

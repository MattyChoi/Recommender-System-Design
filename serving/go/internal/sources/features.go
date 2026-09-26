package sources

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/rpc"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

// Gateway is the client for the Python feature service.
//
// Go cannot read Feast: its online store is written through Feast's SDK in a
// Feast-internal protobuf layout, and reimplementing that encoding in the
// request path would couple serving to a third-party wire format whose changes
// look like corrupt features rather than like a version mismatch. So Feast is
// used from Python, behind one hop, and this is the near side of it.
type Gateway struct {
	client pb.FeaturesClient
	//: Several connections behind one ClientConnInterface -- see internal/rpc.
	//: Nil in tests, which inject `client` directly; Close tolerates that.
	conn *rpc.Balanced

	// UserColumns is the order the tower was fitted with, from
	// serving/features/columns.py. Empty disables the check, which has to be
	// a deliberate choice.
	UserColumns []string

	// ItemColumns is the per-item order the ranker's graph expects for the
	// columns that come from the store.
	ItemColumns []string
}

// DialGateway opens `conns` connections without blocking on the server being
// up. More than one because gRPC-Go serialises a connection's outbound frames
// through a single loopyWriter goroutine -- see internal/rpc.
func DialGateway(target string, conns int, opts ...grpc.DialOption) (*Gateway, error) {
	opts = append([]grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	}, opts...)
	pool, err := rpc.Dial(target, conns, opts...)
	if err != nil {
		return nil, fmt.Errorf("dialling feature gateway at %s: %w", target, err)
	}
	return &Gateway{client: pb.NewFeaturesClient(pool), conn: pool}, nil
}

// Close is nil-safe: the tests build a Gateway with an injected client and no
// connections.
func (g *Gateway) Close() error {
	if g.conn == nil {
		return nil
	}
	return g.conn.Close()
}

// Fetch gets the user's features and history before the fan-out.
func (g *Gateway) Fetch(
	ctx context.Context, userID string, maxHistory int,
) (service.User, error) {
	response, err := g.client.GetUser(ctx, &pb.GetUserRequest{
		UserId:     userID,
		MaxHistory: int32(maxHistory),
	})
	if err != nil {
		return service.User{}, fmt.Errorf("feature gateway GetUser: %w", err)
	}

	if err := checkColumns(g.UserColumns, response.GetNames()); err != nil {
		return service.User{}, fmt.Errorf("user features: %w", err)
	}

	return service.User{
		ID:      userID,
		Feats:   response.GetValues(),
		Names:   response.GetNames(),
		History: response.GetHistory(),
		Found:   response.GetFound(),
	}, nil
}

// ItemRows fetches per-item features and reshapes them into one row per
// candidate, in candidate order.
func (g *Gateway) ItemRows(
	ctx context.Context, items []int32,
) (rows [][]float32, names []string, found []bool, err error) {
	if len(items) == 0 {
		return nil, nil, nil, nil
	}

	response, err := g.client.GetItems(ctx, &pb.GetItemsRequest{Items: items})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("feature gateway GetItems: %w", err)
	}

	names = response.GetNames()
	if err := checkColumns(g.ItemColumns, names); err != nil {
		return nil, nil, nil, fmt.Errorf("item features: %w", err)
	}
	if len(names) == 0 {
		return nil, nil, nil, fmt.Errorf("the gateway returned values with no column names")
	}

	values := response.GetValues()
	// The flat payload is reshaped here, so the arithmetic is checked here.
	// A payload that is not an exact multiple of the row width still reshapes
	// into SOMETHING -- rows offset from the items they describe -- and every
	// candidate then gets the next candidate's features. The model scores that
	// without complaint.
	width := len(names)
	if len(values) != len(items)*width {
		return nil, nil, nil, fmt.Errorf(
			"%d values for %d items x %d columns", len(values), len(items), width,
		)
	}

	rows = make([][]float32, len(items))
	for index := range items {
		start := index * width
		// Sliced from the payload with an explicit capacity, so a caller that
		// appends a blend-derived column to a row cannot write into the next
		// row's values. Without the third index, append would reuse the spare
		// capacity of the backing array.
		rows[index] = values[start : start+width : start+width]
	}

	found = response.GetFound()
	if len(found) != 0 && len(found) != len(items) {
		return nil, nil, nil, fmt.Errorf(
			"%d found flags for %d items", len(found), len(items),
		)
	}
	return rows, names, found, nil
}

// Ready reports whether the gateway can read its online store.
func (g *Gateway) Ready(ctx context.Context) error {
	response, err := g.client.Health(ctx, &pb.FeaturesHealthRequest{})
	if err != nil {
		return fmt.Errorf("feature gateway Health: %w", err)
	}
	if !response.GetReady() {
		return fmt.Errorf("feature gateway not ready: %s", response.GetDetail())
	}
	return nil
}

// checkColumns refuses a vector built in a different order from the expected.
//
// The same defence as the Triton client's, one stage earlier and for the same
// reason: nothing downstream can detect a permutation. The length is right,
// the dtype is right, the tower produces a believable embedding and the ranker
// produces plausible scores. Comparing two ordered lists of strings is all
// there is.
func checkColumns(want, got []string) error {
	if len(want) == 0 {
		return nil
	}
	if len(got) != len(want) {
		return fmt.Errorf("%d columns, expected %d", len(got), len(want))
	}
	for index, name := range want {
		if got[index] != name {
			return fmt.Errorf("column %d is %q, expected %q", index, got[index], name)
		}
	}
	return nil
}

// DefaultFetchTimeout mirrors docs/design.md's 8ms feature-fetch budget. Used
// only when a caller has none of its own; the orchestrator sets its own.
const DefaultFetchTimeout = 8 * time.Millisecond

// Compile-time proof this satisfies the port the pipeline fetches through.
var _ service.UserStore = (*Gateway)(nil)

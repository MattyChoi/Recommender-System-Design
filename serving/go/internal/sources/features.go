package sources

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/features"
	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/rpc"
	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

type Gateway struct {
	client pb.FeaturesClient
	conn   *rpc.Balanced

	storeAge atomic.Int64

	UserColumns []string
	ItemColumns []string
}

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

// refuse vectors built in a different order
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

// Gateway satisfies service.UserStore
func (g *Gateway) Fetch(
	ctx context.Context, userID string, maxHistory int,
) (service.User, error) {
	response, err := g.client.GetUser(ctx, &pb.GetUserRequest{
		UserId: userID, MaxHistory: int32(maxHistory),
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

// ItemRows fetches per-item features and reshapes the flat payload into one row
// per candidate, in candidate order.
func (g *Gateway) ItemRows(ctx context.Context, items []int32) (rows [][]float32, names []string, found []bool, err error) {
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
	width := len(names)
	if len(values) != len(items)*width {
		return nil, nil, nil, fmt.Errorf(
			"%d values for %d items x %d columns", len(values), len(items), width)
	}
	rows = make([][]float32, len(items))
	for index := range items {
		start := index * width
		rows[index] = values[start : start+width : start+width]
	}

	found = response.GetFound()
	if len(found) != 0 && len(found) != len(items) {
		return nil, nil, nil, fmt.Errorf("%d found flags for %d items",
			len(found), len(items))
	}
	return rows, names, found, nil
}

const readyTimeout = 2 * time.Second

// Ready reports whether the gateway can read its online store.
func (g *Gateway) Ready(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, readyTimeout)
	defer cancel()

	response, err := g.client.Health(ctx, &pb.FeaturesHealthRequest{})
	if err != nil {
		return fmt.Errorf("feature gateway Health: %w", err)
	}
	if !response.GetReady() {
		return fmt.Errorf("feature gateway not ready: %s", response.GetDetail())
	}

	if age := response.GetOldestFeatureAgeSeconds(); age > 0 {
		g.storeAge.Store(age)
	}
	return nil
}

func (g *Gateway) Close() error {
	if g.conn == nil {
		return nil
	}
	return g.conn.Close()
}

// Compile-time proof this satisfies the port the pipeline fetches through.
var _ service.UserStore = (*Gateway)(nil)
var _ features.ItemStore = (*Gateway)(nil)

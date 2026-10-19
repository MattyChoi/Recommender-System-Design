package popular

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/service"
)

type Popular struct {
	items   []int32
	version string
}

// payload mirrors what scripts/artifacts/dump_popular.py writes
type payload struct {
	Version string  `json:"version"`
	Items   []int32 `json:"items"`
}

func Load(path string) (*Popular, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w (run `make popular`)", path, err)
	}
	var decoded payload
	if err := json.Unmarshal(blob, &decoded); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if len(decoded.Items) == 0 {
		return nil, fmt.Errorf("%s holds no items (run `make popular`)", path)
	}
	for position, item := range decoded.Items {
		if item <= 0 {
			return nil, fmt.Errorf(
				"%s: position %d holds %d, which is not a real item index", path, position, item,
			)
		}
	}

	return &Popular{
		items:   decoded.Items,
		version: decoded.Version,
	}, nil

}

func (p *Popular) Popular(_ context.Context, n int) ([]int32, error) {
	if n > len(p.items) {
		n = len(p.items)
	}
	if n < 0 {
		n = 0
	}

	out := make([]int32, n)
	copy(out, p.items[:n])
	return out, nil
}

func (p *Popular) Size() int       { return len(p.items) }
func (p *Popular) Version() string { return p.version }

var _ service.Fallback = (*Popular)(nil)

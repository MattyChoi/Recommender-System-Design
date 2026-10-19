// Fallback for when the FAISS index and user encoding python sidecar fails
// Implements exact search (flat index)
package index

import (
	"fmt"
	"path/filepath"

	"github.com/MattyChoi/Recommender-System-Design/serving/go/internal/matrix"
)

type Flat struct {
	table *matrix.Matrix
}

func Stem(root, version, name string) string {
	return filepath.Join(root, version, name)
}

func Load(stem string) (*Flat, error) {
	table, err := matrix.Load(stem, matrix.Expect{
		BaseIndex: 1,
		SkipRows:  0,
		Remedy:    "make index-vectors",
	})
	if err != nil {
		return nil, err
	}

	return &Flat{table: table}, nil
}

func (f *Flat) Meta() matrix.Meta { return f.table.Meta() }
func (f *Flat) Dim() int          { return f.table.Dim() }
func (f *Flat) Kind() string      { return "flat" }

func (f *Flat) Search(query []float32, k int) ([]int32, []float32, error) {
	dim, rows := f.table.Dim(), f.table.Rows()
	base := f.table.Meta().BaseIndex
	values := f.table.Values()

	if len(query) != dim {
		return nil, nil, fmt.Errorf("query has %d dimensions, index has %d",
			len(query), dim)
	}
	if k > rows {
		k = rows
	}

	topItems := make([]int32, 0, k)
	topScores := make([]float32, 0, k)

	for row := 0; row < rows; row++ {
		start := row * dim
		var score float32

		for offset, value := range values[start : start+dim] {
			score += value * query[offset]
		}

		if len(topScores) == k && score <= topScores[k-1] {
			continue
		}

		// Shift the index, baseindex is 1 as 0 is reserved for the OOV vector
		item := int32(row) + base

		// Use linear sort to rank the top k scores
		place := len(topScores)

		for place > 0 && topScores[place-1] < score {
			place--
		}
		if len(topScores) < k {
			topScores = append(topScores, 0)
			topItems = append(topItems, 0)
		}
		copy(topScores[place+1:], topScores[place:])
		copy(topItems[place+1:], topItems[place:])
		topScores[place] = score
		topItems[place] = item
	}
	return topItems, topScores, nil

}

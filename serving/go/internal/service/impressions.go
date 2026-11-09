package service

import (
	"sync"
	"sync/atomic"
	"time"
)

const (
	// closeTimeout bounds the wait in Close. A sink wedged in a write must not
	// hold a rollout open
	closeTimeout = 2 * time.Second
)

// Log records what was shown, in the shape events.proto already defines
type Log interface {
	Emit(rec Record) error
}

type Record struct {
	UserID  string
	Surface string
	Variant string

	// RequestID joins this slate to the clicks it produces. Assigned at the
	// transport boundary
	RequestID string
	ServedAt  time.Time

	ItemIDs    []int32
	Objective  []float64
	Scores     []float64
	Propensity []float64
	// The SERVED feature values, not a re-read of the store. A skew report
	// built from a later read measures the store's drift, not the model's.
	UserFeats []float32
	// UserFound is false when UserFeats are the store-miss defaults.
	UserFound bool
	// What degraded, so off-policy evaluation can exclude or weight the
	// requests where the logging policy was not the policy under study.
	Degraded []string
}

type AsyncLog struct {
	sink  Log
	ch    chan Record
	done  chan struct{}
	once  sync.Once
	drops atomic.Uint64
}

func NewAsyncLog(sink Log, depth int) *AsyncLog {
	a := &AsyncLog{sink: sink, ch: make(chan Record, depth), done: make(chan struct{})}
	go a.drain()
	return a
}

// drain is the single consumer. ONE goroutine, so the sink sees records in the
// order they were served and needs no locking of its own.
func (a *AsyncLog) drain() {
	defer close(a.done)
	for rec := range a.ch {
		// Errors are swallowed HERE and nowhere else. The slate has already
		// shipped; a failed write is a lost training row, not a failed request,
		// and the count below is what makes the loss visible.
		if err := a.sink.Emit(rec); err != nil {
			a.drops.Add(1)
		}
	}
}

// Close stops accepting records and waits, briefly, for the drain to write what
// is already buffered.
func (a *AsyncLog) Close() {
	a.once.Do(func() { close(a.ch) })
	select {
	case <-a.done:
	case <-time.After(closeTimeout):
	}
}

// Drops is dropped-plus-failed records since start, read by the metrics layer.
func (a *AsyncLog) Drops() uint64 { return a.drops.Load() }

// clone copies every slice out of the caller's ownership.
func (r Record) clone() Record {
	r.ItemIDs = append([]int32(nil), r.ItemIDs...)
	r.Objective = append([]float64(nil), r.Objective...)
	r.Scores = append([]float64(nil), r.Scores...)
	r.Propensity = append([]float64(nil), r.Propensity...)
	r.UserFeats = append([]float32(nil), r.UserFeats...)
	r.Degraded = append([]string(nil), r.Degraded...)
	return r
}

// EmitAsync never blocks and never errors into the request path. A full buffer
// DROPS -- and COUNTS the drop, because an unmeasured drop is how a training set
// quietly acquires a survivorship bias toward requests that were cheap to log.
func (a *AsyncLog) EmitAsync(rec Record) {
	select {
	case a.ch <- rec.clone():
	default:
		a.drops.Add(1)
	}
}

var _ ImpressionLog = (*AsyncLog)(nil)

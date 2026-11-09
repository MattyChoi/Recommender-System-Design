package impressions

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"google.golang.org/protobuf/proto"
)

// FileProducer writes one newline-delimited JSON file per topic.
type FileProducer struct {
	dir string

	// One mutex over the whole map, not one per file.
	mu    sync.Mutex
	files map[string]*os.File
}

// NewFileProducer creates dir if it does not exist.
func NewFileProducer(dir string) (*FileProducer, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("impression log directory %s: %w", dir, err)
	}
	return &FileProducer{dir: dir, files: map[string]*os.File{}}, nil
}

// Produce appends one JSON object and a newline.
func (p *FileProducer) Produce(
	topic, _ string, _ time.Time, msg proto.Message,
) error {
	// Marshalled BEFORE the lock. It is the expensive half and it touches
	// nothing shared, so holding the mutex across it would serialise the
	// encoding of every topic behind the slowest write.
	line, err := marshal.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshalling for %s: %w", topic, err)
	}

	file, err := p.open(topic)
	if err != nil {
		return err
	}

	if _, err := file.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("writing to topic %s: %w", topic, err)
	}
	return nil
}

// open returns the topic's file, creating it on first use.
func (p *FileProducer) open(topic string) (*os.File, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if file, ok := p.files[topic]; ok {
		return file, nil
	}

	file, err := os.OpenFile(
		filepath.Join(p.dir, topic+".jsonl"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644,
	)
	if err != nil {
		return nil, fmt.Errorf("opening topic file %s: %w", topic, err)
	}

	p.files[topic] = file
	return file, nil
}

// Close closes every open topic file.
func (p *FileProducer) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var first error
	for topic, file := range p.files {
		if err := file.Close(); err != nil && first == nil {
			first = fmt.Errorf("closing topic %s: %w", topic, err)
		}
		delete(p.files, topic)
	}
	return first
}

var _ Producer = (*FileProducer)(nil)

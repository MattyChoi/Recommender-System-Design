package impressions

import "testing"

// The only thing in kafka.go testable without a broker AND a registry -- and
// worth it, because auto-creation is on in the compose broker, so an
// unrefused typo opens a third topic that nothing consumes and whose subject
// nothing has registered.
//
// A struct literal rather than NewKafkaProducer: the constructor dials a
// registry, and this is a lookup over two strings.
func TestAnUnknownTopicIsRefused(t *testing.T) {
	k := &KafkaProducer{events: TopicEvents, context: TopicContext}

	for _, want := range []string{TopicEvents, TopicContext} {
		got, err := k.topic(want)
		if err != nil {
			t.Fatalf("%s: %v", want, err)
		}
		if *got != want {
			t.Errorf("got %q, want %q", *got, want)
		}
	}
	if _, err := k.topic("impression"); err == nil {
		t.Error(`a near-miss topic name ("impression") must be refused`)
	}
}

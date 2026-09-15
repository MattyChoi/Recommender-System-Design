// Reads a serialized InteractionEvent and prints three fields. Exists to prove
// that Python and Go agree on the wire format -- the one thing that cannot be
// verified from either side alone.
package main

import (
	"fmt"
	"os"

	"google.golang.org/protobuf/proto"

	pb "github.com/MattyChoi/Recommender-System-Design/serving/go/internal/pb"
)

func main() {
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	var ev pb.InteractionEvent
	if err := proto.Unmarshal(raw, &ev); err != nil {
		panic(err)
	}
	fmt.Println(ev.GetUserId(), ev.GetItemId(), ev.GetPosition())
}

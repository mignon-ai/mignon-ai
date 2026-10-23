package generate

import (
	"math"
	"reflect"
	"testing"
)

func TestGreedyContinuationStopsAndReportsProbabilities(t *testing.T) {
	const vocab, stop = 8, 7
	sess := &scriptedSession{script: []int{3, 4, stop, 5}, vocab: vocab}
	ids, probs := greedyContinuation(sess, []int{1}, 10, stop)
	if !reflect.DeepEqual(ids, []int{3, 4}) {
		t.Fatalf("got %v, want [3 4]", ids)
	}
	want := math.E / (math.E + vocab - 1) // one logit of 1 among zeros
	for _, p := range probs {
		if math.Abs(p-want) > 1e-12 {
			t.Errorf("probability %g, want %g", p, want)
		}
	}
}

func TestGreedyContinuationLimitsTokensWithoutExtraForwardPass(t *testing.T) {
	sess := &scriptedSession{script: []int{3, 4, 5, 6}, vocab: 8}
	ids, _ := greedyContinuation(sess, []int{1}, 2, 7)
	if !reflect.DeepEqual(ids, []int{3, 4}) || sess.appends != 1 {
		t.Fatalf("got %v with %d appends, want [3 4] with 1", ids, sess.appends)
	}
}

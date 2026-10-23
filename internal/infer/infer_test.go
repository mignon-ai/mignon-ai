package infer

import (
	"reflect"
	"testing"
)

// TestRecomputeSessionMatchesWindowedForward checks the fallback sees exactly the windows the
// pre-session generation loop passed to the forward function.
func TestRecomputeSessionMatchesWindowedForward(t *testing.T) {
	const window = 3
	var seen [][]int
	fn := LogitsFn(func(w []int) []float64 {
		seen = append(seen, append([]int(nil), w...))
		return []float64{1}
	})
	sess := Recompute(fn, window)
	sess.Prefill([]int{1, 2})
	for _, tok := range []int{3, 4, 5} {
		sess.Append(tok)
	}
	want := [][]int{{1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5}}
	if !reflect.DeepEqual(seen, want) {
		t.Fatalf("windows %v, want %v", seen, want)
	}
}

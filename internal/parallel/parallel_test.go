package parallel

import (
	"sync/atomic"
	"testing"
)

func TestParallelForCoversEachIndexOnce(t *testing.T) {
	for _, n := range []int{0, 1, 7, 64, 1000} {
		for _, workers := range []int{0, 1, 3, 16} {
			counts := make([]int32, n)
			For(n, workers, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					atomic.AddInt32(&counts[i], 1)
				}
			})
			for i, c := range counts {
				if c != 1 {
					t.Fatalf("n=%d workers=%d: index %d visited %d times", n, workers, i, c)
				}
			}
		}
	}
}

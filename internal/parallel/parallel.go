// Package parallel splits index ranges across goroutines.
package parallel

import (
	"sync"
	"sync/atomic"
)

// chunksPerWorker splits work finer than one chunk per worker so uneven chunks
// (e.g. rows of a triangular loop) are balanced by workers pulling the next free chunk.
const chunksPerWorker = 4

// For calls fn over disjoint [lo, hi) ranges covering [0, n), using up to workers
// goroutines. Each index is handled exactly once, so results do not depend on the split.
func For(n, workers int, fn func(lo, hi int)) {
	if n <= 0 {
		return
	}
	if workers <= 1 || n == 1 {
		fn(0, n)
		return
	}
	chunk := (n + workers*chunksPerWorker - 1) / (workers * chunksPerWorker)
	numChunks := (n + chunk - 1) / chunk
	if workers > numChunks {
		workers = numChunks
	}
	var next int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				c := int(atomic.AddInt64(&next, 1) - 1)
				if c >= numChunks {
					return
				}
				lo := c * chunk
				hi := lo + chunk
				if hi > n {
					hi = n
				}
				fn(lo, hi)
			}
		}()
	}
	wg.Wait()
}

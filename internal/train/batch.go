package train

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/data"
	"github.com/mignon-ai/mignon-ai/internal/parallel"
)

// minValidLen is the shortest trained prefix of a sequence: one input and one target position.
const minValidLen = 2

// engine runs the parallel passes of one training run: every worker owns a workspace and
// the workspaces read the shared weights.
type engine struct {
	workers    []*ca.Workspace
	weights    []float32
	batchGrads []float64 // gradient of the current batch summed over workers
}

// forEach calls fn(worker, i) for every i in [0, n). Workers pull the next index as they free up
// (work stealing), so uneven sequence lengths do not leave workers idle.
func (e *engine) forEach(n int, fn func(ws *ca.Workspace, i int)) {
	var cursor int64
	var wg sync.WaitGroup
	for _, ws := range e.workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(atomic.AddInt64(&cursor, 1) - 1)
				if i >= n {
					return
				}
				fn(ws, i)
			}
		}()
	}
	wg.Wait()
}

// batchGradient runs forward and backward over the sequences of batch (indices into set,
// trained up to lens), leaves their summed gradient in e.batchGrads and returns the summed loss.
func (e *engine) batchGradient(set data.Sequences, lens, batch []int) float64 {
	for _, ws := range e.workers {
		ws.ResetGrads()
	}
	e.forEach(len(batch), func(ws *ca.Workspace, i int) {
		seq, n := batch[i], lens[batch[i]]
		ws.ForwardBackwardDense(set.X[seq][:n], set.Y[seq][:n], e.weights)
	})

	loss := 0.0
	for _, ws := range e.workers {
		loss += ws.AccumulatedLoss()
	}
	parallel.For(len(e.batchGrads), len(e.workers), func(lo, hi int) {
		clear(e.batchGrads[lo:hi])
		for _, ws := range e.workers {
			grads := ws.ParamGrads()
			for p := lo; p < hi; p++ {
				e.batchGrads[p] += float64(grads[p])
			}
		}
	})
	return loss
}

// validate returns the sum over sequences of their mean loss and of loss times targets[i], the
// sequence's nats. Losses are summed in sequence order, so the result does not depend on scheduling.
func (e *engine) validate(set data.Sequences, lens, targets []int) (lossSum, natsSum float64) {
	seqLoss := make([]float64, set.Len())
	e.forEach(set.Len(), func(ws *ca.Workspace, i int) {
		seqLoss[i] = ws.ForwardDenseLoss(set.X[i][:lens[i]], set.Y[i][:lens[i]], e.weights)
	})
	for i, loss := range seqLoss {
		lossSum += loss
		natsSum += loss * float64(targets[i])
	}
	return lossSum, natsSum
}

// validLengths returns how many leading positions of each sequence are trained on: the targets
// without their masked tail, clamped to [minValidLen, maxContext] and to the input length. total
// is their sum.
func validLengths(set data.Sequences, maxContext int) (lens []int, total int) {
	lens = make([]int, set.Len())
	for i, targets := range set.Y {
		n := len(targets)
		for n > 0 && targets[n-1] == data.MaskedTarget {
			n--
		}
		n = max(min(n, len(set.X[i]), maxContext), minValidLen)
		lens[i] = n
		total += n
	}
	return lens, total
}

// maxSeqLen is the longest sequence in any of the sets, at least ca.DefaultMinContextSize.
func maxSeqLen(sets ...data.Sequences) int {
	longest := ca.DefaultMinContextSize
	for _, set := range sets {
		for _, seq := range set.X {
			longest = max(longest, len(seq))
		}
	}
	return longest
}

// resolveWorkers returns the worker count: threads when positive, else the CPU count minus a
// couple of cores kept free for the rest of the system; never more than there are sequences.
func resolveWorkers(threads, numSeqs int) int {
	workers := threads
	if workers <= 0 {
		workers = runtime.NumCPU()
		if workers > 4 {
			workers -= 2
		} else if workers > 2 {
			workers--
		}
	}
	return max(1, min(workers, numSeqs))
}

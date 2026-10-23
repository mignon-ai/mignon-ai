package train

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/data"
)

func TestValidLengths(t *testing.T) {
	mask := data.MaskedTarget
	set := data.Sequences{
		X: [][]int{{1, 2, 3, 4, 5}, {1, 2, 3}, {1, 2, 3, 4}, {1}},
		Y: [][]int{{2, 3, 4, 5, 6}, {2, 3, mask}, {mask, mask, mask, mask}, {2}},
	}
	lens, total := validLengths(set, 4)
	// full sequence clamped to maxContext, masked tail trimmed, all-masked and single-token
	// sequences raised to the minimum.
	if want := []int{4, 2, minValidLen, minValidLen}; !slices.Equal(lens, want) {
		t.Fatalf("lens = %v, want %v", lens, want)
	}
	if total != 4+2+minValidLen+minValidLen {
		t.Fatalf("total = %d", total)
	}
}

func TestResolveWorkers(t *testing.T) {
	if got := resolveWorkers(3, 100); got != 3 {
		t.Errorf("explicit threads: got %d, want 3", got)
	}
	if got := resolveWorkers(8, 2); got != 2 {
		t.Errorf("more threads than sequences: got %d, want 2", got)
	}
	if got := resolveWorkers(0, 1000); got < 1 {
		t.Errorf("auto workers: got %d, want >= 1", got)
	}
}

func TestMaxSeqLen(t *testing.T) {
	short := data.Sequences{X: [][]int{make([]int, 3)}}
	long := data.Sequences{X: [][]int{make([]int, ca.DefaultMinContextSize+10)}}
	if got := maxSeqLen(short); got != ca.DefaultMinContextSize {
		t.Errorf("short sets: got %d, want the minimum context", got)
	}
	if got := maxSeqLen(short, long); got != ca.DefaultMinContextSize+10 {
		t.Errorf("long set: got %d", got)
	}
}

func TestEpochProgress(t *testing.T) {
	line := epochProgress(2, 5, 50, 100, 500, 1000, 10*time.Second, 1.25)
	for _, want := range []string{"Epoch 2/5", " 50.0%", "(50/100 seqs", "50 tok/s", "loss: 1.2500", "ETA: 10s"} {
		if !strings.Contains(line, want) {
			t.Errorf("%q missing %q", line, want)
		}
	}
	if done := epochProgress(1, 1, 100, 100, 1000, 1000, time.Second, 0); !strings.Contains(done, "ETA: 0s") {
		t.Errorf("finished epoch should show ETA 0s: %q", done)
	}
}

func TestBestTracker(t *testing.T) {
	var checkpoints []int
	w := []float32{1, 2}
	b := newBestTracker(w, func(epoch int, _ float64, _ []float32) { checkpoints = append(checkpoints, epoch) })
	if b.epoch != -1 {
		t.Fatalf("epoch before any update = %d, want -1", b.epoch)
	}
	if !b.update(1, 2.0, w) {
		t.Fatal("first loss must be a new best")
	}
	w[0] = 9
	if b.update(2, 2.5, w) || b.update(3, 2.0, w) {
		t.Fatal("a worse or equal loss must not replace the best")
	}
	if b.epoch != 1 || b.weights[0] != 1 {
		t.Fatalf("best = epoch %d weights %v, want epoch 1 with the weights saved then", b.epoch, b.weights)
	}
	if !b.update(4, 1.5, w) || b.weights[0] != 9 || len(checkpoints) != 2 || checkpoints[1] != 4 {
		t.Fatalf("improvement not recorded: %+v checkpoints %v", b, checkpoints)
	}
}

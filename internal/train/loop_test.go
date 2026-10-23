package train

import (
	"slices"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/data"
)

// TestIntermediateCheckpointCallback verifies onCheckpoint callback triggers when best loss improves
func TestIntermediateCheckpointCallback(t *testing.T) {
	configs := []ca.BlockConfig{{Steps: 2, DilationBase: 2, InitialDilation: 1}}
	model := ca.New(ca.Config{VocabSize: 8, StateSize: 8, HiddenSize: 12, Blocks: configs, TiedEmbed: true})

	trainSet := data.Sequences{X: [][]int{{1, 2, 3, 4}, {2, 3, 4, 5}}, Y: [][]int{{2, 3, 4, 5}, {3, 4, 5, 6}}}

	calledCount := 0
	var lastEpoch int
	var lastLoss float64
	callback := func(epoch int, loss float64, weights []float32) {
		calledCount++
		lastEpoch = epoch
		lastLoss = loss
		if len(weights) != model.NumParams() {
			t.Fatalf("Callback received invalid weights length: %d vs %d", len(weights), model.NumParams())
		}
	}

	Run(model, trainSet, data.Sequences{}, Options{Epochs: 3, LR: 0.01, Optimizer: "adam", BatchSize: 2,
		WeightDecay: 0.0001, Threads: 1, OnCheckpoint: callback})

	if calledCount == 0 {
		t.Fatalf("Expected onCheckpoint callback to be called at least once, got 0")
	}
	if lastEpoch < 1 || lastEpoch > 3 {
		t.Fatalf("Invalid lastEpoch reported: %d", lastEpoch)
	}
	if lastLoss <= 0 {
		t.Fatalf("Invalid lastLoss reported: %f", lastLoss)
	}
}

func TestTrainMultiBlockVectorizedDenseThreads(t *testing.T) {
	vocabSize := 16
	stateSize := 16
	hiddenSize := 24
	configs := []ca.BlockConfig{
		{Steps: 2, DilationBase: 2, InitialDilation: 1},
	}
	model := ca.New(ca.Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})

	X := [][]int{
		{1, 2, 3, 4},
		{2, 3, 4, 5},
		{3, 4, 5, 6},
		{4, 5, 6, 7},
	}
	Y := [][]int{
		{2, 3, 4, 5},
		{3, 4, 5, 6},
		{4, 5, 6, 7},
		{5, 6, 7, 8},
	}

	for _, threads := range []int{0, 1, 2} {
		Run(model, data.Sequences{X: X, Y: Y}, data.Sequences{}, Options{Epochs: 1, LR: 0.01, Optimizer: "adam", BatchSize: 2,
			WeightDecay: 0.0001, Threads: threads})
	}
}

// TestLongestFirst checks that a batch is reordered by descending length, ties kept in order,
// without touching the shuffled source indices.
func TestLongestFirst(t *testing.T) {
	lens := []int{3, 9, 5, 9, 1}
	batch := []int{0, 1, 2, 3, 4}
	got := longestFirst(make([]int, len(batch)), batch, lens)
	if want := []int{1, 3, 2, 0, 4}; !slices.Equal(got, want) {
		t.Errorf("longestFirst = %v, want %v", got, want)
	}
	if want := []int{0, 1, 2, 3, 4}; !slices.Equal(batch, want) {
		t.Errorf("source batch changed to %v", batch)
	}
}

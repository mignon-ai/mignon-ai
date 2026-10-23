package ca

import (
	"math/rand"
	"slices"
	"testing"
)

// TestSharedPacksMatchOwnPacking checks that a workspace reading refreshed shared packs gives
// bit-identical loss and gradients to one packing the weights itself, also after the weights
// change, for tied and untied heads with and without attention sublayers.
func TestSharedPacksMatchOwnPacking(t *testing.T) {
	for _, tc := range []struct {
		name       string
		tied       bool
		attnHeads  int
		attnWindow int
	}{
		{"tied", true, 0, 0},
		{"untied", false, 0, 0},
		{"tied attention", true, 2, 0},
		{"untied windowed attention", false, 2, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			checkSharedPacks(t, Config{VocabSize: 40, StateSize: 16, HiddenSize: 24,
				Blocks: BuildBlockConfigs(2, 2, 2), TiedEmbed: tc.tied, AttnHeads: tc.attnHeads, AttnWindow: tc.attnWindow})
		})
	}
}

func checkSharedPacks(t *testing.T, cfg Config) {
	const (
		context     = 12
		initScale   = 0.1
		packWorkers = 3
	)
	vocab := cfg.VocabSize
	layout := NewLayout(cfg)
	rng := rand.New(rand.NewSource(1))
	weights := make([]float32, layout.TotalParams)
	ctx, targets := make([]int, context), make([]int, context)
	for i := range ctx {
		ctx[i], targets[i] = rng.Intn(vocab), rng.Intn(vocab)
	}

	own, shared := NewWorkspace(layout, context), NewWorkspace(layout, context)
	packs := NewWeightPacks(layout)
	shared.UseSharedPacks(packs)
	for round := 0; round < 2; round++ {
		for i := range weights {
			weights[i] = float32((rng.Float64()*2 - 1) * initScale)
		}
		packs.Refresh(weights, packWorkers)
		own.ResetGrads()
		shared.ResetGrads()
		wantLoss := own.ForwardBackwardDense(ctx, targets, weights)
		gotLoss := shared.ForwardBackwardDense(ctx, targets, weights)
		if gotLoss != wantLoss {
			t.Errorf("round %d: loss %v with shared packs, want %v", round, gotLoss, wantLoss)
		}
		if !slices.Equal(shared.ParamGrads(), own.ParamGrads()) {
			t.Errorf("round %d: gradients differ with shared packs", round)
		}
	}
}

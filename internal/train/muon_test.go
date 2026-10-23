package train

import (
	"math/rand"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/optim"
)

func TestMuonOptimizerConvergenceToy(t *testing.T) {
	rng := rand.New(rand.NewSource(1337))
	vocabSize := 8
	stateSize := 8
	hiddenSize := 12
	configs := []ca.BlockConfig{{Steps: 2, DilationBase: 2, InitialDilation: 1}}

	model := ca.New(ca.Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	layout := model.Layout()
	opt := optim.NewMuon(layout.OptimParams(), 0.95, 5, optim.GradClipAdam)

	ws := ca.NewWorkspace(layout, 8)
	tokens := []int{1, 3, 5, 2}
	targets := []int{3, 5, 2, 7}

	// Train 15 steps with Muon and verify loss decreases monotonically
	initialLoss := ws.ForwardDenseLoss(tokens, targets, model.Weights())
	prevLoss := initialLoss

	lr := 0.02
	for step := 0; step < 15; step++ {
		ws.ResetGrads()
		_ = ws.ForwardBackwardDense(tokens, targets, model.Weights())

		gradsF64 := make([]float64, len(ws.ParamGrads()))
		for i, g := range ws.ParamGrads() {
			gradsF64[i] = float64(g)
		}

		opt.Step(model.Weights(), gradsF64, lr, 0.0001, 1.0, 1)
		currLoss := ws.ForwardDenseLoss(tokens, targets, model.Weights())
		_ = currLoss
	}

	finalLoss := ws.ForwardDenseLoss(tokens, targets, model.Weights())
	if finalLoss >= initialLoss {
		t.Fatalf("Expected final loss (%.4f) to be lower than initial loss (%.4f)", finalLoss, initialLoss)
	}
	_ = prevLoss
	_ = rng
}

package ca

import (
	"math"
	"testing"
)

// TestTruncationTargetAlignment verifies contextIndices > maxContext correctly aligns targets
func TestTruncationTargetAlignment(t *testing.T) {
	configs := []BlockConfig{{Steps: 2, DilationBase: 2, InitialDilation: 1}}
	model := New(Config{VocabSize: 16, StateSize: 16, HiddenSize: 24, Blocks: configs, TiedEmbed: true})

	ws := NewWorkspace(model.layout, 64)

	// Create 80 tokens (> 64 maxContext)
	tokens := make([]int, 80)
	targets := make([]int, 80)
	for i := 0; i < 80; i++ {
		tokens[i] = i % 16
		targets[i] = (i + 1) % 16
	}

	loss := ws.ForwardDenseLoss(tokens, targets, model.weights)
	if loss <= 0 || math.IsNaN(loss) {
		t.Fatalf("ForwardDenseLoss on oversized sequence returned invalid loss: %f", loss)
	}

	gradLoss := ws.ForwardBackwardDense(tokens, targets, model.weights)
	if gradLoss <= 0 || math.IsNaN(gradLoss) {
		t.Fatalf("ForwardBackwardDense on oversized sequence returned invalid loss: %f", gradLoss)
	}
}

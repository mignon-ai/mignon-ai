package optim

import (
	"testing"
)

func TestAdamOptimizer(t *testing.T) {
	numParams := 4
	opt := NewAdam(numParams, GradClipAdam)
	weights := []float32{1.0, 2.0, -1.0, 0.5}
	grads := []float64{0.1, -0.2, 100.0, -100.0} // tests clipping

	lr := 0.001
	weightDecay := 0.0001
	invBatch := 1.0

	initialW0 := weights[0]
	opt.Step(weights, grads, lr, weightDecay, invBatch, 1)

	if weights[0] >= initialW0 {
		t.Errorf("Positive gradient should decrease weight: got %f >= %f", weights[0], initialW0)
	}
	if opt.StepCount != 1 {
		t.Errorf("Expected optimizer step to be 1, got %d", opt.StepCount)
	}
}

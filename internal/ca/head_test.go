package ca

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/nn"
)

const headTestTol = 1e-5

// TestDenseHeadMatchesSingleHead checks the batched head against the per-position head used by
// Forward: the softmax of forwardHead's logits must equal the dense probabilities of each row.
func TestDenseHeadMatchesSingleHead(t *testing.T) {
	const vocab, state, hidden, n = 37, 16, 24, 9 // odd sizes exercise the gemm edge tiles
	for _, tied := range []bool{true, false} {
		layout := NewLayout(Config{VocabSize: vocab, StateSize: state, HiddenSize: hidden, Blocks: DefaultTwoBlockConfig(), TiedEmbed: tied})
		rng := rand.New(rand.NewSource(3))
		weights := make([]float32, layout.TotalParams)
		for i := range weights {
			weights[i] = float32(rng.NormFloat64() * 0.3)
		}
		context, targets := make([]int, n), make([]int, n)
		for i := range context {
			context[i], targets[i] = rng.Intn(vocab), rng.Intn(vocab)
		}

		ws := NewWorkspace(layout, n)
		ws.forwardDense(context, targets, weights)
		ref := NewWorkspace(layout, n)
		ref.forwardDense(context, targets, weights) // fills finalNormCell
		want := make([]float32, vocab)
		for i := 0; i < n; i++ {
			ref.forwardHead(ref.finalNormCell[i*state:(i+1)*state], weights)
			nn.SoftmaxInPlaceF32(ref.logits[:vocab], want)
			for v := 0; v < vocab; v++ {
				if got, w := ws.head.Probs[i*vocab+v], ref.logits[v]; math.Abs(float64(got-w)) > headTestTol {
					t.Fatalf("tied=%v pos %d token %d: dense %g, single %g", tied, i, v, got, w)
				}
			}
		}
	}
}

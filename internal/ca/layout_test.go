package ca

import "testing"

// TestOptimParamsCoverMatrices guards against a weight matrix silently falling back to the
// scalar AdamW LR: every selective-latch projection must train as a Muon matrix while the
// biases and norm scales stay scalar parameters.
func TestOptimParamsCoverMatrices(t *testing.T) {
	const D = 16
	L := NewLayout(Config{VocabSize: 12, StateSize: D, HiddenSize: 24, Blocks: DefaultTwoBlockConfig(), AttnHeads: 2})
	inMatrix := make([]bool, L.TotalParams)
	for _, m := range L.OptimParams().Matrices {
		for r := 0; r < m.Rows; r++ {
			for c := 0; c < m.Cols; c++ {
				inMatrix[m.Offset+r*m.Stride+c] = true
			}
		}
	}
	for b, bl := range L.Blocks {
		for i := 0; i < D*D; i++ {
			if !inMatrix[bl.SelectiveOffset+i] {
				t.Fatalf("block %d: selective-latch weight %d is not in a Muon matrix", b, i)
			}
		}
		for d := 0; d < D; d++ {
			if inMatrix[bl.GammaOffset+d] {
				t.Fatalf("block %d: norm scale %d must not be in a Muon matrix", b, d)
			}
		}
		for _, m := range L.Attention(bl).Matrices() {
			if !inMatrix[m.Offset] {
				t.Fatalf("block %d: attention matrix %s is not in a Muon matrix", b, m.Name)
			}
		}
	}
}

func TestStepDilation(t *testing.T) {
	cases := []struct{ initial, base, step, want int }{
		{1, 2, 0, 1}, {1, 2, 3, 8}, {1, 2, 10, MaxCADilation}, {4, 2, 1, 8}, {MaxCADilation + 5, 2, 0, MaxCADilation},
	}
	for _, c := range cases {
		if got := StepDilation(c.initial, c.base, c.step); got != c.want {
			t.Errorf("StepDilation(%d, %d, %d) = %d, want %d", c.initial, c.base, c.step, got, c.want)
		}
	}
}

package ca

import "github.com/mignon-ai/mignon-ai/internal/kernels"

// The rule input of one cell is its neighbourhood [self || left || latch], D values per slot.
// left is the causal neighbour `dilation` positions back.

// latchSlot is the offset of the latch slot within a neighbourhood vector.
func latchSlot(D int) int {
	return InputDim(D) - D
}

// AssembleNeighborhood writes the rule input of one cell into x (InputDim(D) values) from its
// normalized state self, its left neighbour and its latch. A nil left (no causal neighbour yet)
// contributes zeros.
func AssembleNeighborhood[T float32 | float64](x, self, left, latch []T, D int) {
	copy(x[:D], self[:D])
	if left != nil {
		copy(x[D:2*D], left[:D])
	} else {
		clear(x[D : 2*D])
	}
	ls := latchSlot(D)
	copy(x[ls:ls+D], latch[:D])
}

// backpropNeighborhood accumulates the rule-input gradient dx of one cell into the gradients of
// its sources: dSelf, dLeft (nil when the cell had no left neighbour) and dLatch.
func backpropNeighborhood(dx, dSelf, dLeft, dLatch []float32, D int) {
	kernels.AddScaledF32(dSelf[:D], dx[:D], 1.0)
	if dLeft != nil {
		kernels.AddScaledF32(dLeft[:D], dx[D:2*D], 1.0)
	}
	ls := latchSlot(D)
	kernels.AddScaledF32(dLatch[:D], dx[ls:ls+D], 1.0)
}

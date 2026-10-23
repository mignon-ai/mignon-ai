package nn

import "github.com/mignon-ai/mignon-ai/internal/kernels"

// MatrixPack is a weight matrix W (one row per output) packed as the B operand of the two
// products it takes part in. Packing depends only on the weights, so a pack can be built once
// and reused for every sequence until the weights change.
type MatrixPack struct {
	Fwd kernels.GemmPackedB // Wᵀ, for the forward product X·Wᵀ
	Bwd kernels.GemmPackedB // W as stored, for the input gradient dY·W
}

// AttentionPacks are the packed projections of one attention sublayer.
type AttentionPacks struct {
	QKV, Out MatrixPack
}

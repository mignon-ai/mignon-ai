package ca

import "github.com/mignon-ai/mignon-ai/internal/nn"

// head locates the output projection (see nn.Head): with tied embeddings the matrix is the
// embedding table and the biases are stored separately; otherwise each head row carries its bias.
func (L Layout) head() nn.Head {
	D := L.StateSize
	if L.TiedEmbed {
		return nn.Head{Off: L.EmbedOffset, Stride: D, BiasOff: L.HeadOffset, BiasStride: 1}
	}
	return nn.Head{Off: L.HeadOffset, Stride: D + 1, BiasOff: L.HeadOffset + D, BiasStride: D + 1}
}

// denseHeadForward fills ws.head.Probs with the softmax over the vocabulary for every position
// and returns the summed negative log-likelihood and the number of unmasked targets.
func (ws *Workspace) denseHeadForward(N int, targets []int, weights []float32) (float64, int) {
	L := ws.layout
	pack := ws.headPackFor(weights, packedMatrix.forwardSpec)
	return L.head().Forward(&ws.head, N, L.StateSize, L.VocabSize, ws.finalNormCell, targets, weights, &pack.Fwd)
}

// denseHeadBackward turns ws.head.Probs into the logit gradients and accumulates dFinalNorm and
// the head gradients (see nn.Head.Backward).
func (ws *Workspace) denseHeadBackward(N int, targets []int, scale float32, weights []float32) {
	L := ws.layout
	pack := ws.headPackFor(weights, packedMatrix.backwardSpec)
	L.head().Backward(&ws.head, N, L.StateSize, L.VocabSize, ws.finalNormCell, ws.dFinalNorm, targets, scale, &pack.Bwd, ws.paramGrads)
}

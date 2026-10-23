// Package nn holds the network building blocks shared by the training and inference engines:
// RMSNorm, the windowed attention sublayer and the dense softmax output head.
package nn

import (
	"math"

	"github.com/mignon-ai/mignon-ai/internal/kernels"
)

// Dense output head over all positions: logits = h · Wᵀ + bias, computed as one gemm for the
// whole sequence and then normalized row by row into probabilities.

// NLLEpsilonF32 is the small floor added inside log() to prevent -Inf loss.
const NLLEpsilonF32 = float32(1e-7)

// Head locates the output projection inside a flat weight vector: row v of the V×D matrix starts
// at weights[Off + v*Stride], and its bias is weights[BiasOff + v*BiasStride]. With tied
// embeddings the matrix is the embedding table and the biases are stored separately.
type Head struct {
	Off, Stride, BiasOff, BiasStride int
}

// HeadBuffers are the per-worker buffers of the dense head for up to maxContext positions.
type HeadBuffers struct {
	Probs   []float32 // maxContext * V: probabilities (forward), then logit gradients (backward)
	Scratch []float32 // V
	Pack    []float32 // packing scratch for the weight-gradient product
}

// NewHeadBuffers allocates head buffers for vocabulary V and up to maxContext positions.
func NewHeadBuffers(V, maxContext int) HeadBuffers {
	return HeadBuffers{Probs: make([]float32, maxContext*V), Scratch: make([]float32, V)}
}

// ClampTarget maps out-of-vocabulary targets to token 0, matching the embedding lookup.
func ClampTarget(tgt, V int) int {
	if tgt >= V {
		return 0
	}
	return tgt
}

// Forward fills buf.Probs with the softmax over the vocabulary for each of the N rows of h (N x D)
// and returns the summed negative log-likelihood and the number of unmasked targets. Rows of
// masked positions are computed too (one gemm is cheaper than skipping them) but unused. wT is the
// head matrix packed for the forward product (MatrixPack.Fwd).
func (hd Head) Forward(buf *HeadBuffers, N, D, V int, h []float32, targets []int, weights []float32, wT *kernels.GemmPackedB) (float64, int) {
	probs := buf.Probs[:N*V]
	kernels.SetRowsToBiasF32(probs, N, V, weights[hd.BiasOff:], hd.BiasStride)
	kernels.GemmPackedF32(N, h, D, 1, wT, probs, V)

	totalLoss, validCount := 0.0, 0
	for i := 0; i < N; i++ {
		tgt := targets[i]
		if tgt < 0 {
			continue
		}
		validCount++
		row := probs[i*V : (i+1)*V]
		SoftmaxInPlaceF32(row, buf.Scratch[:V])
		totalLoss -= math.Log(float64(row[ClampTarget(tgt, V)]) + float64(NLLEpsilonF32))
	}
	return totalLoss, validCount
}

// Backward turns buf.Probs into the logit gradient G = (p - onehot)·scale (zero for masked rows),
// then sets dh = G · W and accumulates the head gradients dW += Gᵀ · h, dBias += column sums of G
// into grads. w is the head matrix packed for dh = G · W (MatrixPack.Bwd). buf.Probs is overwritten.
func (hd Head) Backward(buf *HeadBuffers, N, D, V int, h, dh []float32, targets []int, scale float32, w *kernels.GemmPackedB, grads []float32) {
	g := buf.Probs[:N*V]
	for i := 0; i < N; i++ {
		row := g[i*V : (i+1)*V]
		tgt := targets[i]
		if tgt < 0 {
			clear(row)
			continue
		}
		row[ClampTarget(tgt, V)] -= 1
		for v := range row {
			row[v] *= scale
		}
	}

	clear(dh[:N*D])
	kernels.GemmPackedF32(N, g, V, 1, w, dh, D)
	kernels.GemmF32(&buf.Pack, V, D, N, g, 1, V, h, D, 1, grads[hd.Off:], hd.Stride)
	kernels.AddColumnSumsF32(grads[hd.BiasOff:], hd.BiasStride, g, N, V)
}

// SoftmaxInPlaceF32 replaces logits with their softmax, using scratch (same length) for the
// max-shifted values.
func SoftmaxInPlaceF32(logits, scratch []float32) {
	maxLogit := float32(-math.MaxFloat32)
	for _, x := range logits {
		if x > maxLogit {
			maxLogit = x
		}
	}
	for v, x := range logits {
		scratch[v] = x - maxLogit
	}
	kernels.VecExpF32(logits, scratch)
	sum := float32(0)
	for _, p := range logits {
		sum += p
	}
	inv := 1 / sum
	for v := range logits {
		logits[v] *= inv
	}
}

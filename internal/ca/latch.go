package ca

import (
	"math"

	"github.com/mignon-ai/mignon-ai/internal/kernels"
)

// ComputeSelectiveLatchBaseLogits computes the pre-activation logits corresponding to the
// multi-scale geometric half-lives so that sigmoid(baseLogit_d) == decay_d.
func ComputeSelectiveLatchBaseLogits(stateSize int) ([]float32, []float32) {
	decay, _ := ComputeDecayLatchConstants(stateSize)
	baseLogits := make([]float32, stateSize)
	for d := 0; d < stateSize; d++ {
		lam := float64(decay[d])
		if lam <= 0.0 {
			baseLogits[d] = -SigmoidClampBoundF32
		} else if lam >= 1.0 {
			baseLogits[d] = SigmoidClampBoundF32
		} else {
			baseLogits[d] = float32(math.Log(lam / (1.0 - lam)))
		}
	}
	return decay, baseLogits
}

// latchForward computes the causal prefix memory gLatch[i] = f(sNorm[0..i]) of one CA step for
// all N positions: a decay latch whose per-channel retention is gated by the current state.
// The gate projections are independent across positions, so they run as one N-row gemm; only
// the elementwise recurrence itself is sequential.
func (ws *Workspace) latchForward(bw *blockWorkspace, step, N, D int, selT *kernels.GemmPackedB) {
	base := step * ws.maxContext * D
	sNorm := bw.sNorm[base : base+N*D]
	lam := bw.lambdaBuf[base : base+N*D]

	// λ = σ(baseLogits + sNorm·Wᵀ)
	kernels.SetRowsToBiasF32(lam, N, D, ws.latchBaseLogits, 1)
	kernels.GemmPackedF32(N, sNorm, D, 1, selT, lam, D)
	kernels.VecSigmoidF32(lam, lam)

	clear(ws.runDS[:D])
	for i := 0; i < N; i++ {
		row := i * D
		for d := 0; d < D; d++ {
			l := lam[row+d]
			ws.runDS[d] = l*ws.runDS[d] + (1.0-l)*sNorm[row+d]
			bw.gLatch[base+row+d] = ws.runDS[d]
		}
	}
}

// latchBackward is the exact adjoint of latchForward: it adds to dSNorm and the gate-projection
// gradients.
func (ws *Workspace) latchBackward(bw *blockWorkspace, step, N, D int, sel *kernels.GemmPackedB) {
	base := step * ws.maxContext * D
	selOff := bw.layout.SelectiveOffset
	sNorm := bw.sNorm[base : base+N*D]
	dRaw := ws.selDRaw[:N*D]
	clear(ws.runDS[:D])
	for i := N - 1; i >= 0; i-- {
		row := i * D
		for d := 0; d < D; d++ {
			lam := bw.lambdaBuf[base+row+d]
			ad := bw.dgLatch[base+row+d] + ws.runDS[d]

			bw.dSNorm[base+row+d] += (1.0 - lam) * ad

			var prevLatch float32
			if i > 0 {
				prevLatch = bw.gLatch[base+row-D+d]
			}
			dLam := ad * (prevLatch - sNorm[row+d])
			dRaw[row+d] = dLam * lam * (1.0 - lam)

			ws.runDS[d] = lam * ad
		}
	}

	// dSNorm += dRaw·W and dW += dRawᵀ·sNorm
	kernels.GemmPackedF32(N, dRaw, D, 1, sel, bw.dSNorm[base:], D)
	kernels.GemmF32(&ws.gemmPack, D, D, N, dRaw, 1, D, sNorm, D, 1, ws.paramGrads[selOff:], D)
}

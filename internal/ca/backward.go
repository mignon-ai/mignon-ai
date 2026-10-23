package ca

import (
	"github.com/mignon-ai/mignon-ai/internal/kernels"
	"github.com/mignon-ai/mignon-ai/internal/nn"
)

// ForwardBackwardDense executes dense causal multi-token sequence training across all N positions:
// it runs the forward pass, then backpropagates the mean cross-entropy into
// ws.paramGrads and returns the sequence loss.
func (ws *Workspace) ForwardBackwardDense(contextIndices []int, targets []int, weights []float32) float64 {
	if len(contextIndices) == 0 || len(contextIndices) != len(targets) {
		return 0.0
	}
	if len(contextIndices) > ws.maxContext {
		start := len(contextIndices) - ws.maxContext
		contextIndices = contextIndices[start:]
		targets = targets[start:]
	}
	N := len(contextIndices)

	totalLoss, validCount := ws.forwardDense(contextIndices, targets, weights)
	if validCount == 0 {
		return 0.0
	}

	seqLoss := totalLoss / float64(validCount)
	ws.lossTotal += seqLoss

	ws.clearActivationGrads()
	ws.denseHeadBackward(N, targets, float32(1.0/float64(validCount)), weights)
	ws.finalNormBackward(N, targets, weights)
	for b := len(ws.layout.Blocks) - 1; b >= 0; b-- {
		ws.backwardBlock(b, N, weights)
	}
	ws.embeddingBackward(contextIndices)
	return seqLoss
}

// clearActivationGrads zeroes the activation gradients that the backward pass accumulates into.
func (ws *Workspace) clearActivationGrads() {
	for b := range ws.blocks {
		clear(ws.blocks[b].dsCells)
		clear(ws.blocks[b].dsOut)
	}
	clear(ws.dBlockIn)
}

// finalNormBackward backpropagates the final RMSNorm into the last block's output gradient dsOut
// and the final gamma, over all positions that have a target.
func (ws *Workspace) finalNormBackward(N int, targets []int, weights []float32) {
	L := ws.layout
	D := L.StateSize
	finalNormOff := L.FinalNormOffset
	finalGamma := weights[finalNormOff : finalNormOff+D]
	dsOut := ws.blocks[len(L.Blocks)-1].dsOut

	for i := 0; i < N; i++ {
		if targets[i] < 0 {
			continue
		}
		fnBase := i * D
		nn.RMSNormBackwardF32(
			ws.dFinalNorm[fnBase:fnBase+D],
			ws.finalXHat[fnBase:fnBase+D],
			finalGamma,
			ws.finalRMS[i],
			dsOut[fnBase:fnBase+D],
			ws.paramGrads[finalNormOff:finalNormOff+D],
		)
	}
}

// backwardBlock backpropagates block b: its attention sublayer, the block output, every rule
// step in reverse, and finally the hand-off of the gradient to the block below.
func (ws *Workspace) backwardBlock(b, N int, weights []float32) {
	L := ws.layout
	D := L.StateSize
	bw := &ws.blocks[b]
	T := bw.layout.Steps
	strideS := ws.maxContext * D

	// The weights the block's gemms read are shared by every step and the attention sublayer: pack them once.
	pk := ws.backwardPacks(b, weights)

	// dsOut becomes the gradient wrt the block output before the attention sublayer
	if bw.layout.AttnOffset >= 0 {
		L.Attention(bw.layout).Backward(N, weights, ws.paramGrads, bw.dsOut, &bw.attn, &ws.attnScratch, &pk.attn)
	}

	for idx := 0; idx < N*D; idx++ {
		bw.dsCells[T*strideS+idx] += bw.dsOut[idx]
	}

	for step := T - 1; step >= 0; step-- {
		ws.backwardStep(bw, step, N, weights, pk)
	}

	if b > 0 {
		strideBlock := ws.maxContext * D
		for i := 0; i < N; i++ {
			for d := 0; d < D; d++ {
				gTotal := bw.dsCells[i*D+d] + ws.dBlockIn[b*strideBlock+i*D+d]
				ws.dBlockIn[(b-1)*strideBlock+i*D+d] += gTotal
				ws.blocks[b-1].dsOut[i*D+d] += gTotal
			}
		}
	}
}

// embeddingBackward accumulates the gradient of the block-0 input into the embedding rows of the
// context tokens.
func (ws *Workspace) embeddingBackward(contextIndices []int) {
	L := ws.layout
	D := L.StateSize
	for i, tok := range contextIndices {
		if tok < 0 || tok >= L.VocabSize {
			tok = 0
		}
		wOff := L.EmbedOffset + tok*D
		for d := 0; d < D; d++ {
			gTotal := ws.blocks[0].dsCells[i*D+d] + ws.dBlockIn[i*D+d]
			ws.paramGrads[wOff+d] += gTotal
		}
	}
}

// backwardStep backpropagates one rule step of a block for all N positions: the highway update,
// the gated rule network, the neighbourhood, the latch and the pre-norm. It accumulates the
// weight gradients into ws.paramGrads and the state gradient into the previous step (dsCells).
func (ws *Workspace) backwardStep(bw *blockWorkspace, step, N int, weights []float32, pk *blockGemmPacks) {
	L := ws.layout
	D, H, inDim, outDim := L.StateSize, L.HiddenSize, L.InDim, L.OutDim
	strideS := ws.maxContext * D
	strideZ1 := ws.maxContext * H
	strideZ2 := ws.maxContext * outDim
	strideX := ws.maxContext * inDim
	l1Off := bw.layout.Layer1Offset
	l1GateOff := l1GateOffset(L, bw.layout)
	l2Off := bw.layout.Layer2Offset

	dilation := bw.layout.Dilation(step)

	// This step's views of the per-step buffers (row i of a view is token i)
	sCurr := bw.sCells[step*strideS:]
	dsCurr := bw.dsCells[step*strideS:]
	dsNext := bw.dsCells[(step+1)*strideS:]
	dSNorm := bw.dSNorm[step*strideS:]
	xHat := bw.xHat[step*strideS:]
	rmsStep := bw.rmsBuf[step*ws.maxContext:]
	dgLatch := bw.dgLatch[step*strideS:]
	gateBuf := bw.gateBuf[step*strideS:]
	xIn := bw.xIn[step*strideX:]
	z1 := bw.z1[step*strideZ1:]
	z1Gate := bw.z1Gate[step*strideZ1:]
	hAct := bw.hAct[step*strideZ1:]
	z2 := bw.z2[step*strideZ2:]

	// === 1. Gate/delta gradients of the highway update, for all positions ===
	for i := 0; i < N; i++ {
		cell := i * D
		self, gate, cand := sCurr[cell:cell+D], gateBuf[cell:cell+D], z2[i*outDim:i*outDim+D]
		dSelf, dNext := dsCurr[cell:cell+D], dsNext[cell:cell+D]
		dCandRow, dGateRow := ws.dz2[i*outDim:i*outDim+D], ws.dz2[i*outDim+D:i*outDim+2*D]
		clear(dgLatch[cell : cell+D])
		for d := range dSelf {
			g := gate[d]
			dSelf[d] += dNext[d] * (1.0 - g)
			dGate := dNext[d] * (cand[d] - self[d])
			dCandRow[d] = dNext[d] * g
			dGateRow[d] = dGate * g * (1.0 - g)
		}
	}

	// === 2. Layer 2 input gradients: dhAct = dz2 · W2 ===
	clear(ws.dhAct[:N*H])
	kernels.GemmPackedF32(N, ws.dz2, outDim, 1, &pk.l2.Bwd, ws.dhAct, H)

	// === 3. Gated activation backward ===
	GLUBackward(ws.dz1[:N*H], ws.dz1Gate[:N*H], ws.dhAct[:N*H], z1[:N*H], z1Gate[:N*H])

	// === 4. Layer 1 input gradients: dxIn = dz1 · W1value + dz1Gate · W1gate ===
	clear(ws.dxIn[:N*inDim])
	kernels.GemmPackedF32(N, ws.dz1, H, 1, &pk.l1Value.Bwd, ws.dxIn, inDim)
	kernels.GemmPackedF32(N, ws.dz1Gate, H, 1, &pk.l1Gate.Bwd, ws.dxIn, inDim)

	// === 5. Distribute dxIn back to sNorm, the left neighbour and dgLatch ===
	// Token i is zeroed before any later token adds its left-neighbour gradient into it.
	for i := 0; i < N; i++ {
		cell := i * D
		var dLeft []float32
		if j := i - dilation; j >= 0 {
			dLeft = dSNorm[j*D : j*D+D]
		}
		clear(dSNorm[cell : cell+D])
		backpropNeighborhood(ws.dxIn[i*inDim:i*inDim+inDim], dSNorm[cell:cell+D], dLeft, dgLatch[cell:cell+D], D)
	}

	// === 6. Weight and bias gradients: dW2 += dz2ᵀ · hAct, dW1 += dz1ᵀ · xIn ===
	kernels.GemmF32(&ws.gemmPack, outDim, H, N, ws.dz2, 1, outDim, hAct, H, 1, ws.paramGrads[l2Off:], H+1)
	kernels.AddColumnSumsF32(ws.paramGrads[l2Off+H:], H+1, ws.dz2, N, outDim)
	kernels.GemmF32(&ws.gemmPack, H, inDim, N, ws.dz1, 1, H, xIn, inDim, 1, ws.paramGrads[l1Off:], inDim+1)
	kernels.AddColumnSumsF32(ws.paramGrads[l1Off+inDim:], inDim+1, ws.dz1, N, H)
	kernels.GemmF32(&ws.gemmPack, H, inDim, N, ws.dz1Gate, 1, H, xIn, inDim, 1, ws.paramGrads[l1GateOff:], inDim+1)
	kernels.AddColumnSumsF32(ws.paramGrads[l1GateOff+inDim:], inDim+1, ws.dz1Gate, N, H)

	// Backprop causal prefix latch to dSNorm and paramGrads
	ws.latchBackward(bw, step, N, D, &pk.selector.Bwd)

	// Backprop Pre-Norm RMSNorm into dsCells and gamma
	gammaOff := bw.layout.GammaOffset
	gamma := weights[gammaOff : gammaOff+D]
	dGamma := ws.paramGrads[gammaOff : gammaOff+D]
	for i := 0; i < N; i++ {
		cell := i * D
		nn.RMSNormBackwardF32(dSNorm[cell:cell+D], xHat[cell:cell+D], gamma, rmsStep[i], dsCurr[cell:cell+D], dGamma)
	}
}

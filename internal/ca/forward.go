package ca

import (
	"github.com/mignon-ai/mignon-ai/internal/kernels"
	"github.com/mignon-ai/mignon-ai/internal/nn"
)

// embedTokens writes the token embeddings of contextIndices as the initial states of block 0.
func (ws *Workspace) embedTokens(contextIndices []int, weights []float32) {
	L := ws.layout
	D := L.StateSize
	for i, tok := range contextIndices {
		if tok < 0 || tok >= L.VocabSize {
			tok = 0
		}
		emb := weights[L.EmbedOffset+tok*D : L.EmbedOffset+(tok+1)*D]
		copy(ws.blocks[0].sCells[i*D:(i+1)*D], emb)
		copy(ws.blockIn[i*D:(i+1)*D], emb)
	}
}

// forwardStep advances every cell of block bw by one rule step: pre-norm, latch, neighbourhood
// assembly, the gated rule network and the highway update.
func (ws *Workspace) forwardStep(bw *blockWorkspace, step, N int, weights []float32, pk *blockGemmPacks) {
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

	gammaOff := bw.layout.GammaOffset
	gamma := weights[gammaOff : gammaOff+D]

	// This step's views of the per-step buffers (row i of a view is token i)
	sCurr := bw.sCells[step*strideS:]
	sNext := bw.sCells[(step+1)*strideS:]
	sNorm := bw.sNorm[step*strideS:]
	xHat := bw.xHat[step*strideS:]
	rmsStep := bw.rmsBuf[step*ws.maxContext:]
	gLatch := bw.gLatch[step*strideS:]
	gateBuf := bw.gateBuf[step*strideS:]
	xIn := bw.xIn[step*strideX:]
	z1 := bw.z1[step*strideZ1:]
	z1Gate := bw.z1Gate[step*strideZ1:]
	hAct := bw.hAct[step*strideZ1:]
	z2 := bw.z2[step*strideZ2:]

	// Pre-Norm: normalize cell states at this step
	for i := 0; i < N; i++ {
		cell := i * D
		rmsStep[i] = nn.RMSNormForwardF32(sCurr[cell:cell+D], gamma, sNorm[cell:cell+D], xHat[cell:cell+D], nn.RMSNormEpsilonF32)
	}

	// Causal prefix latch running accumulation of normalized states
	ws.latchForward(bw, step, N, D, &pk.selector.Fwd)

	// === 1. Assemble X for all tokens ===
	for i := 0; i < N; i++ {
		var left []float32
		if j := i - dilation; j >= 0 {
			left = sNorm[j*D : j*D+D]
		}
		AssembleNeighborhood(xIn[i*inDim:i*inDim+inDim], sNorm[i*D:i*D+D], left, gLatch[i*D:i*D+D], D)
	}

	// === 2. Layer 1, value and gate branches: z1 = b1 + X · W1ᵀ ===
	kernels.SetRowsToBiasF32(z1, N, H, weights[l1Off+inDim:], inDim+1)
	kernels.GemmPackedF32(N, xIn, inDim, 1, &pk.l1Value.Fwd, z1, H)
	kernels.SetRowsToBiasF32(z1Gate, N, H, weights[l1GateOff+inDim:], inDim+1)
	kernels.GemmPackedF32(N, xIn, inDim, 1, &pk.l1Gate.Fwd, z1Gate, H)

	// === 3. Gated activation: hAct = z1 · σ(z1Gate) ===
	GLUForward(hAct[:N*H], z1[:N*H], z1Gate[:N*H])

	// === 4. Layer 2: z2 = b2 + hAct · W2ᵀ ===
	kernels.SetRowsToBiasF32(z2, N, outDim, weights[l2Off+H:], H+1)
	kernels.GemmPackedF32(N, hAct, H, 1, &pk.l2.Fwd, z2, outDim)

	// === 5. Highway Gating (AVX2 Vectorized) ===
	for i := 0; i < N; i++ {
		cell := i * D
		cand := z2[i*outDim : i*outDim+D]
		gate := gateBuf[cell : cell+D]
		kernels.VecSigmoidF32(gate, z2[i*outDim+D:i*outDim+2*D])
		self, next := sCurr[cell:cell+D], sNext[cell:cell+D]
		for d := range next {
			next[d] = (1.0-gate[d])*self[d] + gate[d]*cand[d]
		}
	}
}

// forwardBlocks executes the hierarchical CA blocks across all N token positions.
func (ws *Workspace) forwardBlocks(contextIndices []int, weights []float32) int {
	N := len(contextIndices)
	if N > ws.maxContext {
		contextIndices = contextIndices[N-ws.maxContext:]
		N = ws.maxContext
	}

	D := ws.layout.StateSize
	strideBlock := ws.maxContext * D

	ws.embedTokens(contextIndices, weights)

	// Hierarchical CA blocks with inter-block residual connections
	for b := range ws.layout.Blocks {
		bw := &ws.blocks[b]
		T := bw.layout.Steps

		// The weights the block's gemms read are shared by every step and the attention sublayer: pack them once.
		pk := ws.forwardPacks(b, weights)

		if b > 0 {
			prevIn := ws.blockIn[(b-1)*strideBlock:]
			curIn := ws.blockIn[b*strideBlock:]
			for i := 0; i < N*D; i++ {
				val := prevIn[i] + ws.blocks[b-1].sOut[i]
				bw.sCells[i] = val
				curIn[i] = val
			}
		}

		for step := 0; step < T; step++ {
			ws.forwardStep(bw, step, N, weights, pk)
		}

		copy(bw.sOut[:N*D], bw.sCells[T*strideBlock:T*strideBlock+N*D])

		// Content-addressed attention over the block output (hybrid models)
		if bw.layout.AttnOffset >= 0 {
			ws.layout.Attention(bw.layout).Forward(N, weights, bw.sOut, &bw.attn, &ws.attnScratch, &pk.attn)
		}
	}

	return N
}

// headDotWidth is the number of head rows scored per kernels.Dot4F32 call.
const headDotWidth = 4

// forwardHead computes logits for a given normalized state vector into ws.logits.
func (ws *Workspace) forwardHead(normCell []float32, weights []float32) {
	L := ws.layout
	D := L.StateSize
	V := L.VocabSize

	// Untied head rows are [w (D) || bias]; a tied head reuses the embedding rows with the biases
	// stored contiguously at HeadOffset.
	rows, rowStride := weights[L.HeadOffset:], D+1
	bias, biasStride := weights[L.HeadOffset+D:], D+1
	if L.TiedEmbed {
		rows, rowStride = weights[L.EmbedOffset:], D
		bias, biasStride = weights[L.HeadOffset:], 1
	}
	row := func(v int) []float32 { return rows[v*rowStride : v*rowStride+D] }

	v := 0
	for ; v <= V-headDotWidth; v += headDotWidth {
		kernels.Dot4F32(normCell, row(v), row(v+1), row(v+2), row(v+3), ws.dpScratch)
		for j := 0; j < headDotWidth; j++ {
			ws.logits[v+j] = bias[(v+j)*biasStride] + ws.dpScratch[j]
		}
	}
	for ; v < V; v++ {
		ws.logits[v] = bias[v*biasStride] + kernels.DotF32(row(v), normCell)
	}
}

// Forward executes fast, zero-allocation forward inference returning logits for token position N-1
func (ws *Workspace) Forward(contextIndices []int, weights []float32) []float32 {
	if len(contextIndices) == 0 {
		return nil
	}
	N := ws.forwardBlocks(contextIndices, weights)
	if N == 0 {
		return nil
	}

	L := ws.layout
	D := L.StateSize
	V := L.VocabSize
	lastBlockIdx := len(L.Blocks) - 1

	finalNormOff := L.FinalNormOffset
	finalGamma := weights[finalNormOff : finalNormOff+D]

	nn.RMSNormForwardF32(
		ws.blocks[lastBlockIdx].sOut[(N-1)*D:N*D],
		finalGamma,
		ws.finalNormCell[0:D],
		ws.finalXHat[0:D],
		nn.RMSNormEpsilonF32,
	)

	ws.forwardHead(ws.finalNormCell[:D], weights)
	return ws.logits[:V]
}

// ForwardProbs runs the dense causal forward pass and returns the softmax probabilities of every
// position, row-major [position][vocab]. Like the other dense passes it keeps only the last
// maxContext tokens. The slice is reused by the next call on ws.
func (ws *Workspace) ForwardProbs(contextIndices []int, targets []int, weights []float32) []float32 {
	ws.forwardDense(contextIndices, targets, weights)
	n := min(len(contextIndices), ws.maxContext)
	return ws.head.Probs[:n*ws.layout.VocabSize]
}

// forwardDense executes the dense causal CA forward pass across all N positions,
// filling all workspace activation buffers (needed for backward) and computing
// softmax probabilities into ws.head.Probs. Returns (totalLoss, validCount).
func (ws *Workspace) forwardDense(contextIndices []int, targets []int, weights []float32) (float64, int) {
	if len(contextIndices) > ws.maxContext {
		start := len(contextIndices) - ws.maxContext
		contextIndices = contextIndices[start:]
		if len(targets) > start {
			targets = targets[start:]
		}
	}

	L := ws.layout
	D := L.StateSize
	numBlocks := len(L.Blocks)

	N := ws.forwardBlocks(contextIndices, weights)
	if N == 0 {
		return 0.0, 0
	}

	// 3. Dense Output Head across all N positions on final block with Final RMSNorm
	lastBlockIdx := numBlocks - 1
	finalNormOff := L.FinalNormOffset
	finalGamma := weights[finalNormOff : finalNormOff+D]

	for i := 0; i < N; i++ {
		fnBase := i * D
		rms := nn.RMSNormForwardF32(
			ws.blocks[lastBlockIdx].sOut[fnBase:fnBase+D],
			finalGamma,
			ws.finalNormCell[fnBase:fnBase+D],
			ws.finalXHat[fnBase:fnBase+D],
			nn.RMSNormEpsilonF32,
		)
		ws.finalRMS[i] = rms
	}

	return ws.denseHeadForward(N, targets, weights)
}

// ForwardDenseLoss computes forward pass cross-entropy loss without backpropagating gradients
func (ws *Workspace) ForwardDenseLoss(contextIndices []int, targets []int, weights []float32) float64 {
	if len(contextIndices) == 0 || len(contextIndices) != len(targets) {
		return 0.0
	}
	if len(contextIndices) > ws.maxContext {
		start := len(contextIndices) - ws.maxContext
		contextIndices = contextIndices[start:]
		targets = targets[start:]
	}
	totalLoss, validCount := ws.forwardDense(contextIndices, targets, weights)
	if validCount == 0 {
		return 0.0
	}
	return totalLoss / float64(validCount)
}

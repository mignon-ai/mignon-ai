package quant

import (
	"math"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/kernels"
	"github.com/mignon-ai/mignon-ai/internal/nn"
)

// Position-major INT8 inference.
//
// Every CA dependency is causal: a position at step t reads only its own state, its left
// neighbour's normalized state at the same step, and the running latch over earlier positions. Processing the sequence one position at a time through all blocks and
// steps therefore yields exactly the same result as processing it step by step, and lets a
// sequence be extended by one token without recomputing earlier positions. Call and incremental
// generation sessions share this single implementation.

// quantSeqState holds the per-sequence caches of the INT8 model. Caches are indexed by a flat
// (block, step) index; see quantSeqState.stepIndex.
type quantSeqState struct {
	n           int           // positions processed so far
	stepOffset  []int         // flat index of each block's first step
	ring        int           // positions kept per step: the longest dilation plus the current one
	norm        [][][]float32 // [blockStep][position % ring][D] normalized step inputs (left-neighbour lookups)
	latch       [][]float32   // [blockStep][D] running causal latch state
	attnCap     int           // positions in each attention key/value ring
	attnK       [][]float32   // [block][attnCap*D] attention keys (hybrid blocks only)
	attnV       [][]float32   // [block][attnCap*D] attention values
	attnScores  []float32     // attnCap attention weights of the current query
	attnScratch []float32     // attnCap softmax scratch

	// Working vectors for the position being processed.
	cur, next, blockIn []float32
}

// normRing returns how many recent positions each step must keep: a position reads only its
// left neighbour at the step's dilation, so the longest dilation bounds the history needed.
func (q *Model) normRing() int {
	longest := 0
	for b := range q.blocks {
		for step := 0; step < q.blocks[b].Steps; step++ {
			longest = max(longest, q.blocks[b].Dilation(step))
		}
	}
	return longest + 1
}

// newQuantSeqState allocates caches whose size does not depend on the sequence length.
func (q *Model) newQuantSeqState() *quantSeqState {
	D := q.stateSize
	st := &quantSeqState{
		ring:       q.normRing(),
		stepOffset: make([]int, len(q.blocks)),
		cur:        make([]float32, D),
		next:       make([]float32, D),
		blockIn:    make([]float32, D),
	}
	total := 0
	for b := range q.blocks {
		st.stepOffset[b] = total
		total += q.blocks[b].Steps
	}
	st.attnCap = q.attnCache()
	st.attnK = make([][]float32, len(q.blocks))
	st.attnV = make([][]float32, len(q.blocks))
	for b := range q.blocks {
		if q.blocks[b].Attn != nil {
			st.attnK[b] = make([]float32, st.attnCap*D)
			st.attnV[b] = make([]float32, st.attnCap*D)
			st.attnScores = make([]float32, st.attnCap)
			st.attnScratch = make([]float32, st.attnCap)
		}
	}
	st.norm = make([][][]float32, total)
	st.latch = make([][]float32, total)
	for s := 0; s < total; s++ {
		st.norm[s] = make([][]float32, st.ring)
		for i := range st.norm[s] {
			st.norm[s][i] = make([]float32, D)
		}
		st.latch[s] = make([]float32, D)
	}
	return st
}

// bytes returns the memory held by the caches (CA state and attention cache, all float32).
func (st *quantSeqState) bytes() int {
	n := len(st.cur) + len(st.next) + len(st.blockIn)
	for s := range st.norm {
		for _, row := range st.norm[s] {
			n += len(row)
		}
		n += len(st.latch[s])
	}
	n32 := len(st.attnScores) + len(st.attnScratch)
	for b := range st.attnK {
		n32 += len(st.attnK[b]) + len(st.attnV[b])
	}
	return n*8 + n32*4
}

// fits reports whether st was allocated for q's current block structure.
func (st *quantSeqState) fits(q *Model) bool {
	if st == nil || len(st.stepOffset) != len(q.blocks) || len(st.cur) != q.stateSize {
		return false
	}
	total := 0
	for b := range q.blocks {
		total += q.blocks[b].Steps
	}
	return len(st.norm) == total && st.ring == q.normRing() && st.attnCap == q.attnCache()
}

// reset clears the running latch so a new sequence can start.
func (st *quantSeqState) reset() {
	st.n = 0
	for s := range st.latch {
		clear(st.latch[s])
	}
}

func (st *quantSeqState) stepIndex(block, step int) int {
	return st.stepOffset[block] + step
}

// sigmoidClamped is the logistic function, saturated beyond SigmoidClampBound to avoid math.Exp overflow.
func sigmoidClamped(x float32) float32 {
	if x > ca.SigmoidClampBoundF32 {
		return 1.0
	}
	if x < -ca.SigmoidClampBoundF32 {
		return 0.0
	}
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// appendPosition runs token through every block and step at the next sequence position, updating
// the caches. When needOutput is false the final rule application of the last block is skipped:
// its result only feeds the output head, never later positions.
func (q *Model) appendPosition(st *quantSeqState, token int, needOutput bool) {
	D := q.stateSize
	pos := st.n
	if token < 0 || token >= q.vocabSize {
		token = 0
	}
	offset := token * D
	for d := 0; d < D; d++ {
		v := float32(q.embedWeights[offset+d]) * q.embedScale
		st.cur[d] = v
		st.blockIn[d] = v
	}

	numBlocks := len(q.blocks)
	for b := 0; b < numBlocks; b++ {
		block := &q.blocks[b]
		// Inter-block residual: s_in^(b) = blockIn^(b-1) + s_out^(b-1)
		if b > 0 {
			for d := 0; d < D; d++ {
				st.blockIn[d] += st.cur[d]
				st.cur[d] = st.blockIn[d]
			}
		}
		for step := 0; step < block.Steps; step++ {
			bs := st.stepIndex(b, step)
			normCell := st.norm[bs][pos%st.ring]
			if len(block.Gamma) == D {
				nn.RMSNormForwardF32(st.cur, block.Gamma, normCell, q.xHatScratch, nn.RMSNormEpsilonF32)
			} else {
				copy(normCell, st.cur)
			}
			latch := q.updateLatch(st.latch[bs], block, normCell)

			// The last rule application only feeds the output head, unless an attention
			// sublayer follows it, whose keys and values later positions read.
			if !needOutput && b == numBlocks-1 && step == block.Steps-1 && block.Attn == nil {
				break
			}

			var left []float32 // nil: no causal neighbour yet (zeros)
			if leftIdx := pos - block.Dilation(step); leftIdx >= 0 {
				left = st.norm[bs][leftIdx%st.ring]
			}
			q.applyRule(block, normCell, left, latch, st.cur, st.next)

			st.cur, st.next = st.next, st.cur
		}
		if block.Attn != nil {
			q.attendPosition(st, b, block.Attn, pos)
		}
	}
	st.n++
}

// updateLatch advances the selective causal latch gl of one (block, step) with this position's
// normalized state and returns it as the latch vector fed to the rule.
func (q *Model) updateLatch(gl []float32, block *Block, normCell []float32) []float32 {
	D := q.stateSize
	kernels.MatVecInt8(q.latchLogit[:D], block.RuleWSelect.Data, D, block.RuleWSelect.Scale, normCell[:D], q.latchBaseLogits)
	for d := 0; d < D; d++ {
		lam := sigmoidClamped(q.latchLogit[d])
		gl[d] = lam*gl[d] + (1.0-lam)*normCell[d]
	}
	return gl
}

// applyRule evaluates the gated CA rule on the [self, left, latch] neighbourhood and writes the
// updated cell state into next (the unnormalized self state is kept by the update gate).
func (q *Model) applyRule(block *Block, normCell, left, latch, self, next []float32) {
	D := q.stateSize
	inDim := ca.InputDim(D)
	ca.AssembleNeighborhood(q.mlpInput[:inDim], normCell, left, latch, D)

	// Layer 1: value branch (0..H-1) gated by σ of the gate branch (H..2H-1)
	H := q.hiddenSize
	kernels.MatVecInt8(q.mlpHidden[:H], block.RuleW1.Data, inDim, block.RuleW1.Scale, q.mlpInput[:inDim], block.RuleB1[:H])
	kernels.MatVecInt8(q.mlpGate[:H], block.RuleW1.Data[H*inDim:], inDim, block.RuleW1.Scale, q.mlpInput[:inDim], block.RuleB1[H:])
	for h := 0; h < H; h++ {
		q.mlpHidden[h] *= sigmoidClamped(q.mlpGate[h])
	}

	// Layer 2: hiddenSize -> 2D (delta + raw highway gate)
	kernels.MatVecInt8(q.mlpOutput[:2*D], block.RuleW2.Data, H, block.RuleW2.Scale, q.mlpHidden[:H], block.RuleB2)

	// Highway update towards the candidate
	for s := 0; s < D; s++ {
		gate := sigmoidClamped(q.mlpOutput[D+s])
		next[s] = (1.0-gate)*self[s] + gate*q.mlpOutput[s]
	}
}

// headLogits projects a final cell state to vocabulary logits (with the final RMSNorm) into q.logits.
func (q *Model) headLogits(cell []float32) []float64 {
	D := q.stateSize
	if len(q.finalNorm) == D {
		nn.RMSNormForwardF32(cell, q.finalNorm, q.normScratch, q.xHatScratch, nn.RMSNormEpsilonF32)
		cell = q.normScratch
	}
	weights, scale := q.headW, q.headWScale
	if q.tiedEmbed {
		weights, scale = q.embedWeights, q.embedScale
	}
	kernels.MatVecInt8(q.logits[:q.vocabSize], weights, D, scale, cell[:D], q.headB)
	for i, v := range q.logits {
		q.logits64[i] = float64(v)
	}
	return q.logits64
}

// prefill resets st and processes tokens (at most q.maxContext of them, keeping the most recent),
// returning the next-token logits.
func (q *Model) prefill(st *quantSeqState, tokens []int) []float64 {
	if len(tokens) > q.maxContext {
		tokens = tokens[len(tokens)-q.maxContext:]
	}
	st.reset()
	for i, tok := range tokens {
		q.appendPosition(st, tok, i == len(tokens)-1)
	}
	return q.headLogits(st.cur)
}

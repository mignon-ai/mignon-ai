package ca

import (
	"github.com/mignon-ai/mignon-ai/internal/nn"
)

// newAttnScratch allocates the attention backward scratch when the model has attention sublayers.
func newAttnScratch(layout Layout, maxContext int) nn.AttentionScratch {
	if layout.AttnHeads <= 0 {
		return nn.AttentionScratch{}
	}
	return nn.NewAttentionScratch(maxContext, layout.StateSize)
}

// blockWorkspace holds forward and backward buffers for one CA block
type blockWorkspace struct {
	layout BlockLayout

	// Forward activations: [step][token][dim]
	sCells    []float32 // (steps + 1) * maxContext * stateSize
	sNorm     []float32 // steps * maxContext * stateSize (Pre-Norm activations)
	xHat      []float32 // steps * maxContext * stateSize (normalized input before gamma)
	rmsBuf    []float32 // steps * maxContext (cached RMS values)
	z1        []float32 // steps * maxContext * hiddenSize
	z1Gate    []float32 // steps * maxContext * hiddenSize (gate pre-activations)
	hAct      []float32 // steps * maxContext * hiddenSize
	z2        []float32 // steps * maxContext * outDim
	gateBuf   []float32 // steps * maxContext * stateSize
	xIn       []float32 // steps * maxContext * inDim (raw neighbourhood)
	gLatch    []float32 // steps * maxContext * stateSize
	lambdaBuf []float32 // steps * maxContext * stateSize (cached dynamic decay factors for backward pass)

	sOut  []float32 // maxContext * stateSize (block output)
	dsOut []float32 // maxContext * stateSize (adjoint gradients for block output)

	// Backward gradients
	dsCells []float32 // (steps + 1) * maxContext * stateSize
	dSNorm  []float32 // steps * maxContext * stateSize
	dgLatch []float32 // steps * maxContext * stateSize

	// Attention sublayer applied to the block output (attention models only)
	attn nn.AttentionActs
}

// Workspace provides zero-allocation CPU training for hierarchical CA models
type Workspace struct {
	layout     Layout
	maxContext int
	blocks     []blockWorkspace

	// Inter-block residual buffers: [block][token][dim]
	blockIn  []float32 // numBlocks * maxContext * stateSize
	dBlockIn []float32 // numBlocks * maxContext * stateSize

	// Final RMSNorm buffers before output head
	finalNormCell []float32 // maxContext * stateSize
	finalXHat     []float32 // maxContext * stateSize
	finalRMS      []float32 // maxContext
	dFinalNorm    []float32 // maxContext * stateSize

	// Shared scratch buffers per thread
	dz1             []float32 // hiddenSize
	dz1Gate         []float32 // hiddenSize (GLU gate gradient scratch)
	dhAct           []float32 // hiddenSize
	dz2             []float32 // outDim
	dxIn            []float32 // inDim
	attnScratch     nn.AttentionScratch
	runDS           []float32        // stateSize
	logits          []float32        // vocabSize
	probs           []float32        // vocabSize
	head            nn.HeadBuffers   // dense output head: probabilities of every position
	dpScratch       []float32        // headDotWidth elements for the fused head dot product
	blockPacks      []blockGemmPacks // per block: weight matrices packed for gemm, refreshed every forward pass
	headPack        nn.MatrixPack    // the output head packed for gemm, refreshed every pass
	sharedPacks     *WeightPacks     // when set, read instead of blockPacks and headPack (see UseSharedPacks)
	gemmPack        []float32        // packing scratch for kernels.GemmF32 calls
	paramGrads      []float32        // totalParams
	latchBaseLogits []float32        // stateSize (precomputed ln(decay / (1 - decay)))
	selDRaw         []float32        // maxContext * stateSize (selective-latch gate logit gradients)
	lossTotal       float64
}

// ParamGrads returns the gradients accumulated since the last ResetGrads, indexed like the weights.
func (ws *Workspace) ParamGrads() []float32 {
	return ws.paramGrads
}

// AccumulatedLoss returns the summed loss of the sequences processed since the last ResetGrads.
func (ws *Workspace) AccumulatedLoss() float64 {
	return ws.lossTotal
}

// NewWorkspace allocates contiguous memory buffers for one worker thread
func NewWorkspace(layout Layout, maxContext int) *Workspace {
	if maxContext < DefaultMinContextSize {
		maxContext = DefaultMinContextSize
	}

	D := layout.StateSize
	H := layout.HiddenSize
	inDim := layout.InDim
	outDim := layout.OutDim
	V := layout.VocabSize
	numBlocks := len(layout.Blocks)

	bw := make([]blockWorkspace, numBlocks)
	for b := 0; b < numBlocks; b++ {
		T := layout.Blocks[b].Steps
		var attnActs nn.AttentionActs
		if layout.Blocks[b].AttnOffset >= 0 {
			attnActs = layout.Attention(layout.Blocks[b]).NewAttentionActs(maxContext)
		}
		bw[b] = blockWorkspace{
			layout:    layout.Blocks[b],
			sCells:    make([]float32, (T+1)*maxContext*D),
			sNorm:     make([]float32, T*maxContext*D),
			xHat:      make([]float32, T*maxContext*D),
			rmsBuf:    make([]float32, T*maxContext),
			z1:        make([]float32, T*maxContext*H),
			z1Gate:    make([]float32, T*maxContext*H),
			hAct:      make([]float32, T*maxContext*H),
			z2:        make([]float32, T*maxContext*outDim),
			gateBuf:   make([]float32, T*maxContext*D),
			xIn:       make([]float32, T*maxContext*inDim),
			attn:      attnActs,
			gLatch:    make([]float32, T*maxContext*D),
			lambdaBuf: make([]float32, T*maxContext*D),
			sOut:      make([]float32, maxContext*D),
			dsOut:     make([]float32, maxContext*D),
			dsCells:   make([]float32, (T+1)*maxContext*D),
			dSNorm:    make([]float32, T*maxContext*D),
			dgLatch:   make([]float32, T*maxContext*D),
		}
	}

	_, latchBaseLogits := ComputeSelectiveLatchBaseLogits(D)

	return &Workspace{
		layout:          layout,
		maxContext:      maxContext,
		blocks:          bw,
		blockIn:         make([]float32, numBlocks*maxContext*D),
		dBlockIn:        make([]float32, numBlocks*maxContext*D),
		finalNormCell:   make([]float32, maxContext*D),
		finalXHat:       make([]float32, maxContext*D),
		finalRMS:        make([]float32, maxContext),
		dFinalNorm:      make([]float32, maxContext*D),
		dz1:             make([]float32, maxContext*H),
		dz1Gate:         make([]float32, maxContext*H),
		dhAct:           make([]float32, maxContext*H),
		dz2:             make([]float32, maxContext*outDim),
		dxIn:            make([]float32, maxContext*inDim),
		attnScratch:     newAttnScratch(layout, maxContext),
		runDS:           make([]float32, D),
		logits:          make([]float32, V),
		probs:           make([]float32, V),
		head:            nn.NewHeadBuffers(V, maxContext),
		dpScratch:       make([]float32, headDotWidth),
		blockPacks:      make([]blockGemmPacks, len(layout.Blocks)),
		paramGrads:      make([]float32, layout.TotalParams),
		latchBaseLogits: latchBaseLogits,
		selDRaw:         make([]float32, maxContext*D),
	}
}

// ResetGrads zeroes all parameter gradients
func (ws *Workspace) ResetGrads() {
	for i := range ws.paramGrads {
		ws.paramGrads[i] = 0.0
	}
	ws.lossTotal = 0.0
}

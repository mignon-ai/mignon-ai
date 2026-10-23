package ca

import (
	"math"

	"github.com/mignon-ai/mignon-ai/internal/nn"
	"github.com/mignon-ai/mignon-ai/internal/optim"
)

// BlockLayout stores precomputed parameter offsets for one CA block
type BlockLayout struct {
	Layer1Offset    int
	Layer2Offset    int
	GammaOffset     int
	SelectiveOffset int // Offset to D*D selective latch projection weights
	AttnOffset      int // Offset to the attention sublayer after the block (-1 if unused)
	Steps           int
	DilationBase    int
	InitDilation    int
}

// StepDilation returns the neighbour stride at tick `step`: initial * base^step, capped at
// MaxCADilation so spatial neighbour mixing remains active.
func StepDilation(initial, base, step int) int {
	d := initial
	for s := 0; s < step && d < MaxCADilation; s++ {
		d *= base
	}
	return min(d, MaxCADilation)
}

// Dilation returns the dilation stride for tick `step` within this block.
func (bl BlockLayout) Dilation(step int) int {
	return StepDilation(bl.InitDilation, bl.DilationBase, step)
}

// Layout stores parameter offsets across all blocks and embeddings
type Layout struct {
	VocabSize       int
	StateSize       int
	HiddenSize      int
	InDim           int // 3 * StateSize
	OutDim          int // 2 * StateSize
	TotalParams     int
	TiedEmbed       bool
	EmbedOffset     int
	Blocks          []BlockLayout
	FinalNormOffset int
	HeadOffset      int
	AttnHeads       int // heads of the attention sublayer after each block; 0 = none
	AttnWindow      int // attention span in tokens; 0 = full causal
}

// Attention returns the attention sublayer applied to the block's output; only valid when
// bl.AttnOffset >= 0.
func (L Layout) Attention(bl BlockLayout) nn.Attention {
	return nn.NewAttention(bl.AttnOffset, L.StateSize, L.AttnHeads, L.AttnWindow)
}

// ComputeDecayLatchConstants computes multi-scale exponential decay factors across D state dimensions.
// Half-lives span geometrically from minHalfLife (1.5 tokens) to maxHalfLife (64.0 tokens).
// lambda_d = exp(-ln(2) / T_d) = 2^(-1 / T_d)
func ComputeDecayLatchConstants(stateSize int) ([]float32, []float32) {
	decay := make([]float32, stateSize)
	oneMinus := make([]float32, stateSize)
	if stateSize <= 0 {
		return decay, oneMinus
	}
	const minT = 1.5
	const maxT = 64.0

	for d := 0; d < stateSize; d++ {
		var td float64
		if stateSize == 1 {
			td = minT
		} else {
			ratio := math.Pow(maxT/minT, 1.0/float64(stateSize-1))
			td = minT * math.Pow(ratio, float64(d))
		}
		lam := math.Pow(2.0, -1.0/td)
		decay[d] = float32(lam)
		oneMinus[d] = float32(1.0 - lam)
	}
	return decay, oneMinus
}

// caNeighborhood is the width (in units of stateSize) of the CA rule input [self || left || latch].
const caNeighborhood = 3

// InputDim returns the CA rule input width for the given state size.
func InputDim(stateSize int) int {
	return caNeighborhood * stateSize
}

// NewLayout computes the flat parameter offsets of the architecture described by cfg.
func NewLayout(cfg Config) Layout {
	cfg = cfg.withDefaults()
	vocabSize, stateSize, hiddenSize, configs, tiedEmbed := cfg.VocabSize, cfg.StateSize, cfg.HiddenSize, cfg.Blocks, cfg.TiedEmbed
	inDim := InputDim(stateSize)
	outDim := 2 * stateSize
	embedCount := vocabSize * stateSize
	layer1Count := 2 * hiddenSize * (inDim + 1)
	layer2Count := outDim * (hiddenSize + 1)
	selectiveCount := stateSize * stateSize

	blocks := make([]BlockLayout, len(configs))
	currOffset := embedCount
	for i, bc := range configs {
		base := bc.DilationBase
		if base <= 1 {
			base = 2
		}
		initDil := bc.InitialDilation
		if initDil < 1 {
			initDil = 1
		}
		selOff := currOffset + layer1Count + layer2Count + stateSize
		// The attention sublayer follows the fixed parameters.
		extra := selOff + selectiveCount
		attnOff := -1
		if cfg.AttnHeads > 0 {
			attnOff = extra
			extra += nn.AttentionParams(stateSize)
		}
		blocks[i] = BlockLayout{
			Layer1Offset:    currOffset,
			Layer2Offset:    currOffset + layer1Count,
			GammaOffset:     currOffset + layer1Count + layer2Count,
			SelectiveOffset: selOff,
			AttnOffset:      attnOff,
			Steps:           bc.Steps,
			DilationBase:    base,
			InitDilation:    initDil,
		}
		currOffset = extra
	}

	finalNormOffset := currOffset
	currOffset += stateSize

	headOffset := currOffset
	headCount := vocabSize * (stateSize + 1)
	if tiedEmbed {
		headCount = vocabSize // only biases
	}

	return Layout{
		VocabSize:       vocabSize,
		StateSize:       stateSize,
		HiddenSize:      hiddenSize,
		InDim:           inDim,
		OutDim:          outDim,
		TotalParams:     currOffset + headCount,
		TiedEmbed:       tiedEmbed,
		EmbedOffset:     0,
		Blocks:          blocks,
		FinalNormOffset: finalNormOffset,
		HeadOffset:      headOffset,
		AttnHeads:       cfg.AttnHeads,
		AttnWindow:      cfg.AttnWindow,
	}
}

// OptimParams describes the parameter vector for the optimizers: the GLU, selective-latch and
// attention matrices of every block (Muon-orthogonalized) and the token embedding plus untied
// output head (AdamW at the embedding LR). A matrix left out would train at the scalar LR.
func (L Layout) OptimParams() optim.ParamLayout {
	H, D := L.HiddenSize, L.StateSize
	var matrices []optim.Matrix
	for _, bl := range L.Blocks {
		// Layer 1 weights: 2*H rows x inDim cols (stride inDim+1 to skip trailing bias)
		matrices = append(matrices, optim.Matrix{Name: "b_l1", Offset: bl.Layer1Offset, Rows: 2 * H, Cols: L.InDim, Stride: L.InDim + 1})
		// Layer 2 weights: outDim rows x H cols (stride H+1 to skip trailing bias)
		matrices = append(matrices, optim.Matrix{Name: "b_l2", Offset: bl.Layer2Offset, Rows: L.OutDim, Cols: H, Stride: H + 1})
		// Selective latch projection matrix: D rows x D cols
		matrices = append(matrices, optim.Matrix{Name: "b_sel", Offset: bl.SelectiveOffset, Rows: D, Cols: D, Stride: D})
		// Attention sublayer query/key/value and output projections
		if bl.AttnOffset >= 0 {
			matrices = append(matrices, L.Attention(bl).Matrices()...)
		}
	}
	embeddings := []optim.Range{{Start: L.EmbedOffset, End: L.EmbedOffset + L.VocabSize*D}}
	if !L.TiedEmbed {
		embeddings = append(embeddings, optim.Range{Start: L.HeadOffset, End: L.HeadOffset + L.VocabSize*(D+1)})
	}
	return optim.ParamLayout{TotalParams: L.TotalParams, Matrices: matrices, Embeddings: embeddings}
}

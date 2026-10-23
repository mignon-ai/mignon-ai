// Package quant converts a trained ca.Model to INT8 weights and runs it: full-window inference,
// cached incremental generation sessions, and the quantized checkpoint format.
package quant

import (
	"sync"

	"github.com/mignon-ai/mignon-ai/internal/ca"
)

// defaultContext is the window allocated when the caller gives none (or one below the minimum).
const defaultContext = 2 * ca.DefaultMinContextSize

// Block holds INT8 quantized weights for one CA block
type Block struct {
	Steps           int             `json:"steps"`
	DilationBase    int             `json:"dilation_base"`
	InitialDilation int             `json:"initial_dilation"`
	RuleW1          Tensor          `json:"rule_w1"`
	RuleB1          []float32       `json:"rule_b1"`
	RuleW2          Tensor          `json:"rule_w2"`
	RuleB2          []float32       `json:"rule_b2"`
	Gamma           []float32       `json:"gamma,omitempty"`
	RuleWSelect     Tensor          `json:"rule_w_select"`
	Attn            *blockAttention `json:"attn,omitempty"` // attention sublayer after the block (hybrid models)
}

// Dilation returns the stride for tick `step`.
func (qb *Block) Dilation(step int) int {
	return ca.StepDilation(qb.InitialDilation, qb.DilationBase, step)
}

// Model provides zero-allocation INT8 inference on CPU for hierarchical CA models
type Model struct {
	vocabSize  int
	stateSize  int
	hiddenSize int
	tiedEmbed  bool

	embedWeights []int8
	embedScale   float32

	blocks []Block

	finalNorm []float32

	headW      []int8
	headWScale float32
	headB      []float32

	// Pre-allocated internal buffers for zero-allocation inference
	maxContext      int
	mlpInput        []float32
	callState       *quantSeqState // caches reused by Call
	mlpHidden       []float32
	mlpOutput       []float32
	xHatScratch     []float32
	normScratch     []float32
	logits          []float32
	logits64        []float64 // logits converted for the generation API
	latchBaseLogits []float32
	mlpGate         []float32 // gate branch of the rule's first layer
	latchLogit      []float32 // selective latch decay logits
	attnHeads       int       // heads of the attention sublayers; 0 = none
	attnWindow      int       // attention span in tokens; 0 = the whole context
	attn            attnScratch
	mu              sync.Mutex
}

// VocabSize returns the vocabulary size.
func (q *Model) VocabSize() int {
	return q.vocabSize
}

// StateSize returns the per-cell state width D.
func (q *Model) StateSize() int {
	return q.stateSize
}

// HiddenSize returns the hidden width of each block's GLU rule network.
func (q *Model) HiddenSize() int {
	return q.hiddenSize
}

// NumBlocks returns the number of hierarchical blocks.
func (q *Model) NumBlocks() int {
	return len(q.blocks)
}

// NewModel allocates inference scratch buffers.
func NewModel(vocabSize, stateSize, hiddenSize, maxContext int, numBlocks int) *Model {
	if maxContext < ca.DefaultMinContextSize {
		maxContext = defaultContext
	}

	inDim := ca.InputDim(stateSize)
	outDim := 2 * stateSize
	_, latchBaseLogits := ca.ComputeSelectiveLatchBaseLogits(stateSize)

	return &Model{
		vocabSize:       vocabSize,
		stateSize:       stateSize,
		hiddenSize:      hiddenSize,
		maxContext:      maxContext,
		blocks:          make([]Block, numBlocks),
		mlpInput:        make([]float32, inDim),
		mlpHidden:       make([]float32, hiddenSize),
		mlpOutput:       make([]float32, outDim),
		xHatScratch:     make([]float32, stateSize),
		normScratch:     make([]float32, stateSize),
		logits:          make([]float32, vocabSize),
		logits64:        make([]float64, vocabSize),
		latchBaseLogits: latchBaseLogits,
		mlpGate:         make([]float32, hiddenSize),
		latchLogit:      make([]float32, stateSize),
		attn:            newAttnScratch(stateSize),
	}
}

// Call executes zero-allocation INT8 inference on CPU and returns the next-token logits for
// the last position. Windows longer than the model context keep their most recent tokens.
func (q *Model) Call(contextIndices []int) []float64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(contextIndices) == 0 {
		return nil
	}
	if !q.callState.fits(q) {
		q.callState = q.newQuantSeqState()
	}
	return q.prefill(q.callState, contextIndices)
}

// Forward implements the QuantizedModel interface for zero-allocation inference
func (q *Model) Forward(window []int) []float64 {
	return q.Call(window)
}

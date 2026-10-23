// Package ca implements the hierarchical neural cellular-automaton language model: its flat
// parameter layout and initialization, inference, and the zero-allocation training workspace
// (dense forward/backward through the selective latch and the attention sublayers).
package ca

import (
	"fmt"
	"math"
	"math/rand"
	"sync"

	"github.com/mignon-ai/mignon-ai/internal/optim"
)

// BlockConfig configures one hierarchical CA block with its steps and dilation schedule.
type BlockConfig struct {
	Steps           int `json:"steps"`
	DilationBase    int `json:"dilation_base"`
	InitialDilation int `json:"initial_dilation"`
}

// DefaultTwoBlockConfig returns a standard 2-block hierarchy:
// Block 0: Local Syntax (3 steps, dilations 1, 2, 4)
// Block 1: Global Semantics (3 steps, dilations 4, 8, 16)
func DefaultTwoBlockConfig() []BlockConfig {
	return []BlockConfig{
		{Steps: 3, DilationBase: 2, InitialDilation: 1}, // 1, 2, 4
		{Steps: 3, DilationBase: 2, InitialDilation: 4}, // 4, 8, 16
	}
}

// BuildBlockConfigs constructs a slice of BlockConfig for the requested number of blocks.
func BuildBlockConfigs(numBlocks, steps, dilationBase int) []BlockConfig {
	if numBlocks <= 1 {
		return []BlockConfig{
			{Steps: steps, DilationBase: dilationBase, InitialDilation: 1},
		}
	}
	if numBlocks == 2 && (steps == 3 || steps == 5) && dilationBase == 2 {
		return DefaultTwoBlockConfig()
	}
	configs := make([]BlockConfig, numBlocks)
	curDilation := 1
	for b := 0; b < numBlocks; b++ {
		configs[b] = BlockConfig{
			Steps:           steps,
			DilationBase:    dilationBase,
			InitialDilation: curDilation,
		}
		for s := 0; s < steps-1; s++ {
			curDilation *= dilationBase
			if curDilation > MaxCADilation {
				curDilation = MaxCADilation
				break
			}
		}
	}
	return configs
}

// Model implements a hierarchical multi-block Neural Cellular Automata Language Model.
// Stores parameters in a single flat contiguous float32 array with zero pointer indirection.
type Model struct {
	vocabSize  int
	stateSize  int
	hiddenSize int
	configs    []BlockConfig
	tiedEmbed  bool
	layout     Layout
	weights    []float32
	inferWS    *Workspace
	inferOut64 []float64
	mu         sync.Mutex
}

// New constructs a randomly initialized model with the architecture described by cfg.
func New(cfg Config) *Model {
	cfg = cfg.withDefaults()
	vocabSize, stateSize, hiddenSize, configs, tiedEmbed := cfg.VocabSize, cfg.StateSize, cfg.HiddenSize, cfg.Blocks, cfg.TiedEmbed

	layout := NewLayout(cfg)
	weights := make([]float32, layout.TotalParams)

	initMultiBlockWeights(layout, weights, rand.New(rand.NewSource(rand.Int63())))

	return &Model{
		vocabSize:  vocabSize,
		stateSize:  stateSize,
		hiddenSize: hiddenSize,
		configs:    configs,
		tiedEmbed:  tiedEmbed,
		layout:     layout,
		weights:    weights,
		inferOut64: make([]float64, vocabSize),
	}
}

// InitWeights re-initializes all parameters deterministically from seed, so that
// experiments comparing configurations can start from identical random streams.
func (m *Model) InitWeights(seed int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	initMultiBlockWeights(m.layout, m.weights, rand.New(rand.NewSource(seed)))
}

// initMultiBlockWeights fills weights with the standard initialization for layout using rng.
func initMultiBlockWeights(layout Layout, weights []float32, rng *rand.Rand) {
	vocabSize := layout.VocabSize
	stateSize := layout.StateSize
	hiddenSize := layout.HiddenSize
	tiedEmbed := layout.TiedEmbed

	// 1. Embedding weights: uniform [-0.05, 0.05]
	embedCount := vocabSize * stateSize
	for i := 0; i < embedCount; i++ {
		weights[i] = float32(rng.Float64()*0.1 - 0.05)
	}

	// 2. Blocks
	inDim := layout.InDim
	outDim := layout.OutDim
	scale1 := float32(math.Sqrt(2.0 / float64(inDim)))
	scale2 := float32(math.Sqrt(2.0 / float64(hiddenSize)))

	for _, bl := range layout.Blocks {
		// Layer 1: gated (GLU) — 2*hiddenSize neurons (value branch 0..H-1, gate branch H..2H-1)
		for h := 0; h < 2*hiddenSize; h++ {
			nOff := bl.Layer1Offset + h*(inDim+1)
			for j := 0; j < inDim; j++ {
				weights[nOff+j] = float32(rng.Float64()*2-1) * scale1
			}
			weights[nOff+inDim] = 0.0
		}

		// Layer 2: each neuron has hiddenSize weights followed by 1 bias
		for s := 0; s < outDim; s++ {
			nOff := bl.Layer2Offset + s*(hiddenSize+1)
			for j := 0; j < hiddenSize; j++ {
				weights[nOff+j] = float32(rng.Float64()*2-1) * scale2
			}
			// Gate biases (second half of outDim: s >= stateSize) initialized to -1.386 (sigmoid ~ 0.20)
			if s >= stateSize {
				weights[nOff+hiddenSize] = -1.386
			} else {
				weights[nOff+hiddenSize] = 0.0
			}
		}

		// Gamma scale: 1.0
		for d := 0; d < stateSize; d++ {
			weights[bl.GammaOffset+d] = 1.0
		}

		// Selective latch weights W_select (stateSize x stateSize):
		// Initialized to 0.0 for zero-init stability.
		// When W_select is 0, sigmoid(baseLogit_d + 0) == decay_d at step 0: a fixed multi-scale decay.
		clear(weights[bl.SelectiveOffset : bl.SelectiveOffset+stateSize*stateSize])

		// Attention sublayer: zero output projection, so it starts as the identity.
		if bl.AttnOffset >= 0 {
			layout.Attention(bl).Init(weights, rng, attnInitGain)
		}
	}

	// 3. Final RMSNorm: 1.0
	for d := 0; d < stateSize; d++ {
		weights[layout.FinalNormOffset+d] = 1.0
	}

	// 4. Head weights
	if tiedEmbed {
		// Biases only
		for v := 0; v < vocabSize; v++ {
			weights[layout.HeadOffset+v] = 0.0
		}
	} else {
		scaleH := float32(math.Sqrt(2.0 / float64(stateSize)))
		for v := 0; v < vocabSize; v++ {
			nOff := layout.HeadOffset + v*(stateSize+1)
			for j := 0; j < stateSize; j++ {
				weights[nOff+j] = float32(rng.Float64()*2-1) * scaleH
			}
			weights[nOff+stateSize] = 0.0
		}
	}
}

// VocabSize returns the current vocabulary size of the model.
func (m *Model) VocabSize() int {
	return m.vocabSize
}

// StateSize returns the per-cell state width D.
func (m *Model) StateSize() int {
	return m.stateSize
}

// HiddenSize returns the hidden width of each block's GLU rule network.
func (m *Model) HiddenSize() int {
	return m.hiddenSize
}

// TiedEmbed reports whether the output head reuses the token embedding matrix.
func (m *Model) TiedEmbed() bool {
	return m.tiedEmbed
}

// Layout returns the parameter layout of the model's flat weight vector.
func (m *Model) Layout() Layout {
	return m.layout
}

// BlockConfigs extracts the hyperparameter configuration for all blocks.
func (m *Model) BlockConfigs() []BlockConfig {
	return m.configs
}

// NumBlocks returns the number of hierarchical blocks.
func (m *Model) NumBlocks() int {
	return len(m.configs)
}

// NumParams returns the total number of trainable model parameters.
func (m *Model) NumParams() int {
	return len(m.weights)
}

// Weights returns the raw parameter slice.
func (m *Model) Weights() []float32 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.weights
}

// SetWeights replaces model weights from a slice.
func (m *Model) SetWeights(w []float32) {
	m.mu.Lock()
	defer m.mu.Unlock()
	copy(m.weights, w)
}

// Forward executes fast, zero-allocation forward inference returning raw vocabulary logits as []float64 for position N-1.
func (m *Model) Forward(contextIndices []int) []float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	logits32 := m.forwardLocked(contextIndices)
	if logits32 == nil {
		return nil
	}
	if len(m.inferOut64) != len(logits32) {
		m.inferOut64 = make([]float64, len(logits32))
	}
	for i, v := range logits32 {
		m.inferOut64[i] = float64(v)
	}
	return m.inferOut64
}

// forwardLocked runs inference on the shared workspace, growing it to fit the context (at least
// DefaultMinContextSize). The caller holds m.mu.
func (m *Model) forwardLocked(contextIndices []int) []float32 {
	N := len(contextIndices)
	if N == 0 {
		return nil
	}
	maxCtx := max(N, DefaultMinContextSize)
	if m.inferWS == nil || m.inferWS.maxContext < maxCtx {
		m.inferWS = NewWorkspace(m.layout, maxCtx)
	}
	return m.inferWS.Forward(contextIndices, m.weights)
}

// ResizeVocab adjusts the model's embedding table and head for a new vocabulary size.
// Preserves all existing embeddings, block weights, norm scales, and head weights.
// Newly added token rows are initialized with small random values.
func (m *Model) ResizeVocab(newVocabSize int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if newVocabSize <= m.vocabSize {
		return
	}

	oldVocab := m.vocabSize
	oldLayout := m.layout
	stateSize := m.stateSize

	cfg := m.Config()
	cfg.VocabSize = newVocabSize
	newLayout := NewLayout(cfg)
	newWeights := make([]float32, newLayout.TotalParams)

	// 1. Copy old embeddings and initialize new token rows
	oldEmbedCount := oldVocab * stateSize
	copy(newWeights[:oldEmbedCount], m.weights[:oldEmbedCount])
	scale := float32(math.Sqrt(1.0 / float64(stateSize)))
	for v := oldVocab; v < newVocabSize; v++ {
		off := v * stateSize
		for d := 0; d < stateSize; d++ {
			newWeights[off+d] = float32(rand.Float64()*2-1) * scale
		}
	}

	// 2. Copy all blocks and final norm weights (preserved intact)
	newBlockStart := newLayout.EmbedOffset + newVocabSize*stateSize
	oldBlockStart := oldLayout.EmbedOffset + oldEmbedCount
	blocksLen := oldLayout.HeadOffset - oldBlockStart
	copy(newWeights[newBlockStart:newBlockStart+blocksLen], m.weights[oldBlockStart:oldLayout.HeadOffset])

	// 3. Copy output head
	if m.tiedEmbed {
		// Tied embeddings: head has vocabSize biases
		copy(newWeights[newLayout.HeadOffset:newLayout.HeadOffset+oldVocab],
			m.weights[oldLayout.HeadOffset:oldLayout.HeadOffset+oldVocab])
		for v := oldVocab; v < newVocabSize; v++ {
			newWeights[newLayout.HeadOffset+v] = 0.0
		}
	} else {
		// Untied head: vocabSize * (stateSize + 1)
		rowSize := stateSize + 1
		copy(newWeights[newLayout.HeadOffset:newLayout.HeadOffset+oldVocab*rowSize],
			m.weights[oldLayout.HeadOffset:oldLayout.HeadOffset+oldVocab*rowSize])
		headScale := float32(math.Sqrt(1.0 / float64(stateSize)))
		for v := oldVocab; v < newVocabSize; v++ {
			off := newLayout.HeadOffset + v*rowSize
			for d := 0; d < stateSize; d++ {
				newWeights[off+d] = float32(rand.Float64()*2-1) * headScale
			}
			newWeights[off+stateSize] = 0.0
		}
	}

	m.vocabSize = newVocabSize
	m.layout = newLayout
	m.weights = newWeights
	m.inferWS = nil
	m.inferOut64 = nil
}

// OptimParams describes the model's parameter vector to the optimizers.
func (m *Model) OptimParams() optim.ParamLayout {
	return m.layout.OptimParams()
}

// NewWorkers returns n training workspaces sharing one copy of the packed rule-network weights;
// refresh repacks them and must run after every change to the weights.
func (m *Model) NewWorkers(n, maxContext int) ([]*Workspace, func(weights []float32, threads int)) {
	packs := NewWeightPacks(m.layout)
	workers := make([]*Workspace, n)
	for w := range workers {
		ws := NewWorkspace(m.layout, maxContext)
		ws.UseSharedPacks(packs)
		workers[w] = ws
	}
	return workers, packs.Refresh
}

// Describe names the architecture for training logs.
func (m *Model) Describe() string {
	return fmt.Sprintf("Hierarchical CA Engine (%d blocks)", len(m.layout.Blocks))
}

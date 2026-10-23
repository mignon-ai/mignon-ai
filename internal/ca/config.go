package ca

import "github.com/mignon-ai/mignon-ai/internal/nn"

// Config describes a model architecture. Zero values of the optional fields select the
// defaults: two blocks (DefaultTwoBlockConfig) and no attention.
type Config struct {
	VocabSize  int
	StateSize  int
	HiddenSize int
	Blocks     []BlockConfig
	TiedEmbed  bool

	AttnHeads  int // heads of the causal attention sublayer after each block; 0 = none
	AttnWindow int // attention span in tokens including the query; 0 = full causal
}

// withDefaults fills the zero-valued optional fields.
func (c Config) withDefaults() Config {
	if len(c.Blocks) == 0 {
		c.Blocks = DefaultTwoBlockConfig()
	}
	return c
}

// Config returns the architecture the model was built with (after any vocabulary resize).
func (m *Model) Config() Config {
	return Config{
		VocabSize:  m.vocabSize,
		StateSize:  m.stateSize,
		HiddenSize: m.hiddenSize,
		Blocks:     m.configs,
		TiedEmbed:  m.tiedEmbed,
		AttnHeads:  m.layout.AttnHeads,
		AttnWindow: m.layout.AttnWindow,
	}
}

// Numerical stability
const (
	// SigmoidClampBound clips |x| > this to {0,1} to prevent math.Exp overflow.
	SigmoidClampBound    = 15.0
	SigmoidClampBoundF32 = float32(15.0)
	// NLLEpsilon is the small floor added inside log() to prevent -Inf loss.
	NLLEpsilon    = 1e-15
	NLLEpsilonF32 = nn.NLLEpsilonF32

	// attnInitGain scales the attention sublayer's query/key/value initialization to
	// ±sqrt(gain / D), the same He-style gain as the rule network's first layer.
	attnInitGain = 2.0
)

// Workspace and dilation limits
const (
	// DefaultMinContextSize is the minimum workspace context window (in tokens).
	DefaultMinContextSize = 64
	// MaxCADilation caps geometric cellular automata dilation strides to ensure spatial neighbor mixing
	// is active across all steps within typical context windows.
	MaxCADilation = 32
)

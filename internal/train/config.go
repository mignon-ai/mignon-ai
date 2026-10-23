package train

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/optim"
)

// Config fully describes a model architecture plus its training hyperparameters.
// It is the shared description used by the CLI defaults and by the sweep runner, and it can
// be layered from JSON (defaults -> preset -> explicit keys).
type Config struct {
	Preset       string  `json:"preset,omitempty"`
	Blocks       int     `json:"blocks"`
	StateDim     int     `json:"state_dim"`
	HiddenDim    int     `json:"hidden_dim"`
	Steps        int     `json:"steps"`
	Dilation     int     `json:"dilation"`
	Vocab        int     `json:"vocab"`
	Context      int     `json:"context"`
	Tied         bool    `json:"tied"`
	AttnHeads    int     `json:"attn_heads"`  // CA only: heads of an attention sublayer after each block; 0 = none
	AttnWindow   int     `json:"attn_window"` // CA only: attention span in tokens; 0 = full causal
	Optimizer    string  `json:"opt"`
	LR           float64 `json:"lr"`             // 0 = optimizer/preset default (see ResolvedLR)
	LRSchedule   string  `json:"lr_schedule"`    // decay shape after warmup: "cosine" or "linear"
	LRFloor      float64 `json:"lr_floor"`       // final LR as a fraction of the peak
	EmbedLRScale float64 `json:"embed_lr_scale"` // Muon: embedding/head LR as a fraction of the LR
	Batch        int     `json:"batch"`
	WeightDecay  float64 `json:"decay"`
	Epochs       int     `json:"epochs"`
	Stride       int     `json:"stride"`   // 0 = context
	MaxSeqs      int     `json:"max_seqs"` // 0 = all
	Threads      int     `json:"threads"`  // 0 = auto
}

// DefaultConfig returns the configuration used when no preset or flag overrides apply.
func DefaultConfig() Config {
	return Config{
		Blocks:       1,
		StateDim:     64,
		HiddenDim:    96,
		Steps:        5,
		Dilation:     2,
		Vocab:        1024,
		Context:      64,
		Tied:         true,
		Optimizer:    "muon", // beats AdamW by ~0.08 val loss for every latch in the text sweeps
		LRSchedule:   optim.DefaultSchedule().Shape,
		LRFloor:      optim.DefaultSchedule().Floor,
		EmbedLRScale: optim.DefaultMuonEmbedLRScale,
		Batch:        192,
		WeightDecay:  0.0001,
		Epochs:       10,
	}
}

// applyPreset overwrites the preset-controlled fields.
func (c *Config) applyPreset(p Preset) {
	c.Preset = p.Name
	c.Blocks, c.StateDim, c.HiddenDim = p.Blocks, p.Dim, p.Hidden
	c.Steps, c.Dilation, c.Vocab, c.Context = p.Steps, p.Dilation, p.Vocab, p.Context
	c.Tied, c.Batch, c.WeightDecay, c.Epochs = p.Tied, p.Batch, p.WeightDecay, p.Epochs
	c.AttnHeads, c.AttnWindow = p.AttnHeads, p.AttnWindow
}

// DefaultLR returns the default pretraining learning rate for an optimizer.
func DefaultLR(optimizer string) float64 {
	switch optimizer {
	case "muon":
		return DefaultMuonLR
	}
	return DefaultAdamLR
}

// ResolvedLR returns the explicit LR, else the preset's tuned LR for the optimizer, else the optimizer default.
func (c Config) ResolvedLR() float64 {
	if c.LR > 0 {
		return c.LR
	}
	if p, ok := GetPreset(c.Preset); ok {
		return p.LearningRate(c.Optimizer)
	}
	return DefaultLR(c.Optimizer)
}

// ResolveConfig merges JSON object layers (later layers win) on top of the defaults. Unknown keys are an error.
// A "preset" key, if present in any layer, is applied before the explicit keys.
func ResolveConfig(layers ...json.RawMessage) (Config, error) {
	merged := map[string]json.RawMessage{}
	for _, layer := range layers {
		if len(layer) == 0 {
			continue
		}
		var m map[string]json.RawMessage
		if err := json.Unmarshal(layer, &m); err != nil {
			return Config{}, fmt.Errorf("invalid config layer: %w", err)
		}
		for k, v := range m {
			merged[k] = v
		}
	}

	cfg := DefaultConfig()
	if raw, ok := merged["preset"]; ok {
		var name string
		if err := json.Unmarshal(raw, &name); err != nil {
			return cfg, fmt.Errorf("invalid preset: %w", err)
		}
		preset, ok := GetPreset(name)
		if !ok {
			return cfg, fmt.Errorf("unknown preset %q", name)
		}
		cfg.applyPreset(preset)
	}
	encoded, err := json.Marshal(merged)
	if err != nil {
		return cfg, err
	}
	dec := json.NewDecoder(bytes.NewReader(encoded))
	dec.DisallowUnknownFields() // a stale or misspelled key must fail, not silently run the default
	if err := dec.Decode(&cfg); err != nil {
		return cfg, fmt.Errorf("invalid config: %w", err)
	}
	return cfg, cfg.Validate()
}

// Validate checks structural constraints that would otherwise fail deep inside training.
func (c Config) Validate() error {
	if c.Blocks < 1 || c.StateDim < 1 || c.HiddenDim < 1 || c.Steps < 1 {
		return fmt.Errorf("blocks, state_dim, hidden_dim and steps must be >= 1")
	}
	if c.AttnHeads < 0 || c.AttnWindow < 0 || (c.AttnHeads > 0 && c.StateDim%c.AttnHeads != 0) {
		return fmt.Errorf("attn_heads (%d) must divide state_dim (%d) and attn_window must be >= 0", c.AttnHeads, c.StateDim)
	}
	if !slices.Contains(Optimizers, c.Optimizer) {
		return fmt.Errorf("unknown optimizer %q (want one of %v)", c.Optimizer, Optimizers)
	}
	if c.EmbedLRScale <= 0 {
		return fmt.Errorf("embed_lr_scale must be > 0")
	}
	return c.Schedule().Validate()
}

// ModelConfig returns the model architecture of this config for the given vocabulary size.
func (c Config) ModelConfig(vocabSize int) ca.Config {
	return ca.Config{
		VocabSize:  vocabSize,
		StateSize:  c.StateDim,
		HiddenSize: c.HiddenDim,
		Blocks:     ca.BuildBlockConfigs(c.Blocks, c.Steps, c.Dilation),
		TiedEmbed:  c.Tied,
		AttnHeads:  c.AttnHeads,
		AttnWindow: c.AttnWindow,
	}
}

// BuildModel constructs a freshly initialized model for this config and vocabulary size.
// A non-zero seed makes the initialization deterministic.
func (c Config) BuildModel(vocabSize int, seed int64) *ca.Model {
	m := ca.New(c.ModelConfig(vocabSize))
	if seed != 0 {
		m.InitWeights(seed)
	}
	return m
}

// Schedule returns the learning-rate schedule for this config.
func (c Config) Schedule() optim.Schedule {
	return optim.Schedule{Shape: c.LRSchedule, Floor: c.LRFloor}
}

// Options returns the training options for this config.
func (c Config) Options(seed int64, quiet bool) Options {
	return Options{
		Epochs:       c.Epochs,
		LR:           c.ResolvedLR(),
		Optimizer:    c.Optimizer,
		BatchSize:    c.Batch,
		WeightDecay:  c.WeightDecay,
		Threads:      c.Threads,
		Schedule:     c.Schedule(),
		EmbedLRScale: c.EmbedLRScale,
		Seed:         seed,
		Quiet:        quiet,
	}
}

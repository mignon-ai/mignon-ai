package train

import (
	"maps"
	"slices"
	"strings"
)

// Preset defines standardized hyperparameter bundles for Cellular Automata models
type Preset struct {
	Name        string
	Blocks      int
	Dim         int
	Hidden      int
	Steps       int
	Dilation    int
	Vocab       int
	Context     int
	Tied        bool
	Batch       int
	LR          float64 // tuned AdamW learning rate
	MuonLR      float64 // tuned Muon learning rate; 0 = not measured for this size, use DefaultMuonLR
	WeightDecay float64
	Epochs      int
	AttnHeads   int // heads of the attention sublayer after each block; 0 = none
	AttnWindow  int // attention span in tokens; 0 = full causal
}

// Presets holds the canonical configurations
var Presets = map[string]Preset{
	"micro": {
		Name:        "micro",
		Blocks:      1,
		Dim:         32,
		Hidden:      48,
		Steps:       4,
		Dilation:    3,
		Vocab:       512,
		Context:     32,
		Tied:        true,
		Batch:       128,
		LR:          0.010,
		WeightDecay: 0.0001,
		Epochs:      10,
	},
	"small": {
		Name:   "small",
		Blocks: 1,
		Dim:    64,
		Hidden: 256,
		// One rule step plus windowed attention: 2.024 vs 2.056 bpb for 5 steps at equal training
		// time, and solves the retrieval probes (research/eval/sweeps/small_hybrid.log). Hidden width 256
		// (4x the state) beats 96 by 0.036 bpb at equal training time, and 384 gains nothing more
		// (research/eval/sweeps/archive/RESULTS_hidden_width.md).
		Steps:       1,
		Dilation:    3,
		Vocab:       1024,
		Context:     64,
		Tied:        true,
		Batch:       192,
		LR:          0.008,
		MuonLR:      0.04, // hidden 256, 3 seeds, 10 epochs: 0.02 → 2.2516, 0.03 → 2.2486, 0.04 → 2.2472, 0.06 → 2.2511, 0.08 → 2.2563
		WeightDecay: 0.0001,
		Epochs:      10,
		AttnHeads:   4,
		AttnWindow:  64,
	},
	"base": {
		Name:   "base",
		Blocks: 2,
		Dim:    96,
		Hidden: 160,
		// One rule step per block plus windowed attention, context 128 and vocab 2048: at equal
		// training time 1.838 bpb vs 1.935 for 4 steps without attention, context 64 and vocab 1024
		// (research/eval/sweeps/base_hybrid.log).
		Steps:    1,
		Dilation: 2,
		Vocab:    2048,
		Context:  128,
		Tied:     true,
		Batch:    160,
		// Muon LR left at the default: 0.03 → 1.831, 0.04 → 1.838, 0.06 → 1.850 (within noise; research/eval/sweeps/base_lr.log).
		LR:          0.007,
		WeightDecay: 0.0001,
		Epochs:      10,
		AttnHeads:   4,
		AttnWindow:  64,
	},
	"medium": {
		Name:   "medium",
		Blocks: 2,
		Dim:    128,
		Hidden: 256,
		// One rule step per block plus windowed attention: at equal training time 1.760 bpb vs
		// 1.857 for 4 steps without attention; solves the retrieval probes; INT8 generation ~125 µs/token with constant memory at any context
		// (research/eval/sweeps/ca_hybrid_medium.log, attn_window_medium.log, genbench_hybrid.log).
		Steps:       1,
		Dilation:    2,
		Vocab:       2048,
		Context:     128,
		Tied:        true,
		Batch:       128,
		LR:          0.006,
		MuonLR:      0.04, // 2-seed text sweeps at 4 steps: 0.02 → 2.835, 0.03 → 2.824, 0.04 → 2.811, 0.06 → 2.815
		WeightDecay: 0.0001,
		Epochs:      10,
		AttnHeads:   4,
		AttnWindow:  64, // window 64: +0.002 bpb vs full attention, constant generation cost and memory
	},
	"large": {
		Name:   "large",
		Blocks: 2,
		Dim:    192,
		Hidden: 384,
		// One rule step per block plus windowed attention: at equal training time 1.775 bpb vs
		// 1.915 for 4 steps without attention (research/eval/sweeps/large_hybrid.log).
		Steps:    1,
		Dilation: 2,
		Vocab:    2048,
		Context:  128,
		Tied:     true,
		// Equal-time Muon sweeps (research/eval/sweeps/large_tuning.log, large_lr.log): batch 96 at LR 0.06 → 1.823,
		// 0.04 → 1.783, 0.02 → 1.766, 0.01 → 1.779; batch 192 at 0.03 → 1.763 and 13% more tok/s.
		// Weight decay 0–1e-3, vocab 4096 and 3 narrower blocks did not help.
		Batch:       192,
		LR:          0.004,
		MuonLR:      0.03,
		WeightDecay: 0.0001,
		Epochs:      10,
		AttnHeads:   4,
		AttnWindow:  64,
	},
	"pro": {
		Name:   "pro",
		Blocks: 2,
		Dim:    224,
		Hidden: 448,
		// One rule step per block plus windowed attention with large's batch and Muon LR: at equal
		// training time 1.852 bpb vs 2.041 for 4 steps without attention; vocab 4096 tied at 1.859
		// and 20% fewer tok/s (research/eval/sweeps/pro_hybrid.log).
		Steps:       1,
		Dilation:    2,
		Vocab:       2048,
		Context:     128,
		Tied:        true,
		Batch:       192,
		LR:          0.0015,
		MuonLR:      0.03,
		WeightDecay: 0.0001,
		Epochs:      10,
		AttnHeads:   4,
		AttnWindow:  64,
	},
	"xl": {
		Name:   "xl",
		Blocks: 2,
		Dim:    288,
		Hidden: 576,
		// Large's shape scaled up (hidden = 2x the state): the first size meant for a general English base that
		// task models are tuned from. Learning rates are large's, not yet measured at this size. Pair it with
		// a vocabulary of 4096-8192 (-vocab, or -tok-from): the embedding is a quarter of the model at 2048.
		Steps:       1,
		Dilation:    2,
		Vocab:       2048,
		Context:     128,
		Tied:        true,
		Batch:       192,
		LR:          0.003,
		MuonLR:      0.03,
		WeightDecay: 0.0001,
		Epochs:      10,
		AttnHeads:   4,
		AttnWindow:  64,
	},
}

// LearningRate returns the preset's tuned learning rate for optimizer, falling back to the
// optimizer default where no value was measured for this model size.
func (p Preset) LearningRate(optimizer string) float64 {
	switch {
	case optimizer == "adam" && p.LR > 0:
		return p.LR
	case optimizer == "muon" && p.MuonLR > 0:
		return p.MuonLR
	}
	return DefaultLR(optimizer)
}

// PresetNames returns the names of all presets, sorted.
func PresetNames() []string {
	return slices.Sorted(maps.Keys(Presets))
}

// GetPreset returns the preset corresponding to name (case-insensitive) or false if not found
func GetPreset(name string) (Preset, bool) {
	p, ok := Presets[strings.ToLower(strings.TrimSpace(name))]
	return p, ok
}

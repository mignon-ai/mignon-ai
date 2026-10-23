package train

// Learning rates
const (
	// DefaultAdamLR is the default learning rate for AdamW when no preset or -lr is given.
	DefaultAdamLR = 0.008

	// SFTLRFraction is the fine-tuning learning rate (-sft, and continued -train of a loaded model)
	// as a fraction of the optimizer's default pretraining LR, so it stays proportionate whichever
	// optimizer is used (0.016 for Muon). Measured on Unix and git fine-tuning: 0.2-0.8 of the Muon
	// rate beats the old 0.05 by about 10 points on Unix.
	SFTLRFraction = 0.4

	// DefaultMuonLR is the default learning rate for 2D matrix parameters in Muon.
	// Measured best on the medium preset; presets with their own measurement override it (Preset.MuonLR).
	DefaultMuonLR = 0.04
)

// Optimizers lists the supported optimizer names.
var Optimizers = []string{"adam", "muon"}

// Batching
const (
	// DefaultMinBatchSize is the lower floor on mini-batch size to ensure meaningful gradient estimates.
	DefaultMinBatchSize = 16
)

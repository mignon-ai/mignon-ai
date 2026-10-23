// Package optim holds the parameter optimizers (AdamW, Muon) and the learning-rate schedule.
// Optimizers work on flat weight and gradient vectors and know nothing about the model; Muon is
// told which parameter ranges are matrices and embeddings through a ParamLayout.
package optim

// Optimizer applies one update to a flat weight vector from summed batch gradients.
type Optimizer interface {
	// Step updates weights in place from grads (summed over the batch; scaled by invBatch) at
	// learning rate lr, using up to workers goroutines where the optimizer supports it.
	Step(weights []float32, grads []float64, lr, weightDecay, invBatch float64, workers int)
}

// Gradient clipping
const (
	// GradClipAdam is the per-parameter gradient magnitude cap for the Adam optimizer.
	GradClipAdam = 5.0
)

// AdamW moment defaults
const (
	AdamBeta1 = 0.9
	AdamBeta2 = 0.999
	AdamEps   = 1e-8
)

// Muon defaults
const (
	// DefaultMuonMomentum is the momentum factor for Muon.
	DefaultMuonMomentum = 0.95
	// DefaultMuonNSSteps is the number of Newton-Schulz quintic orthogonalization iterations.
	DefaultMuonNSSteps = 5
	// DefaultMuonAdamWLR is the learning rate used by the internal AdamW sub-optimizer for 1D parameters.
	DefaultMuonAdamWLR = 0.002
	// MuonScalarLRFraction caps the AdamW LR of scalars (biases, norm gains) at this fraction of the Muon LR.
	MuonScalarLRFraction = 0.1
	// DefaultMuonEmbedLRScale is the AdamW LR of embeddings and the untied head as a fraction of the Muon LR.
	DefaultMuonEmbedLRScale = 1.0
)

var (
	_ Optimizer = (*Adam)(nil)
	_ Optimizer = (*Muon)(nil)
)

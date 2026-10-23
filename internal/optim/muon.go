package optim

import (
	"sync"

	"github.com/mignon-ai/mignon-ai/internal/parallel"
)

// Matrix identifies a 2D weight matrix embedded inside the flat parameter vector.
type Matrix struct {
	Name   string
	Offset int // starting offset of first row in flat weights
	Rows   int
	Cols   int
	Stride int // stride between row starts in flat weights (e.g. Cols + 1 if bias is trailing)
}

// Range is the half-open parameter index range [Start, End).
type Range struct {
	Start, End int
}

// ParamLayout tells Muon how a flat parameter vector is organized: which spans are 2D matrices
// (orthogonalized updates) and which are token embeddings (AdamW at EmbedLRScale). Everything
// else is treated as a scalar parameter (bias or norm gain).
type ParamLayout struct {
	TotalParams int
	Matrices    []Matrix
	Embeddings  []Range
}

// paramGroup selects how Muon updates one parameter.
type paramGroup uint8

const (
	groupScalar paramGroup = iota // AdamW at the scalar LR
	groupEmbed                    // AdamW at EmbedLRScale·lr
	groupMatrix                   // Newton-Schulz orthogonalized momentum
)

// Muon implements the Momentum Orthogonalized by Newton-Schulz optimizer.
// It applies Newton-Schulz orthogonalization to 2D hidden weight matrices and
// delegates 1D vectors, biases, and embeddings to an embedded AdamW optimizer.
type Muon struct {
	Matrices    []Matrix
	momentumIdx []int        // per matrix: starting index in Momentum
	groups      []paramGroup // per parameter
	Momentum    []float32
	AdamOpt     *Adam
	Beta        float32
	NSSteps     int
	ClipGrad    float32
	Scratch     *NSScratch
	// EmbedLRScale sets the AdamW LR of embeddings and the untied output head as a fraction
	// of the Muon LR.
	EmbedLRScale float64
	mu           sync.Mutex
}

// NewMuon initializes a hybrid Muon + AdamW optimizer for the given parameter layout.
// Non-positive momentum, nsSteps or clipGrad select the defaults.
func NewMuon(layout ParamLayout, momentum float64, nsSteps int, clipGrad float64) *Muon {
	if momentum <= 0 {
		momentum = DefaultMuonMomentum
	}
	if nsSteps <= 0 {
		nsSteps = DefaultMuonNSSteps
	}
	if clipGrad <= 0 {
		clipGrad = GradClipAdam
	}

	groups := make([]paramGroup, layout.TotalParams)
	for _, r := range layout.Embeddings {
		for p := r.Start; p < r.End; p++ {
			groups[p] = groupEmbed
		}
	}
	momentumIdx := make([]int, len(layout.Matrices))
	totalMatrixParams, maxDim := 0, 0
	for i, m := range layout.Matrices {
		momentumIdx[i] = totalMatrixParams
		totalMatrixParams += m.Rows * m.Cols
		maxDim = max(maxDim, m.Rows, m.Cols)
		for r := 0; r < m.Rows; r++ {
			rowStart := m.Offset + r*m.Stride
			for c := 0; c < m.Cols; c++ {
				groups[rowStart+c] = groupMatrix
			}
		}
	}

	return &Muon{
		Matrices:     layout.Matrices,
		momentumIdx:  momentumIdx,
		groups:       groups,
		Momentum:     make([]float32, totalMatrixParams),
		AdamOpt:      NewAdam(layout.TotalParams, clipGrad),
		Beta:         float32(momentum),
		NSSteps:      nsSteps,
		ClipGrad:     float32(clipGrad),
		Scratch:      NewNSScratch(maxDim),
		EmbedLRScale: DefaultMuonEmbedLRScale,
	}
}

// Step applies a hybrid Muon update: 2D weight matrices are updated via Newton-Schulz
// orthogonalized momentum; 1D vectors, biases, and embeddings are updated via AdamW.
func (opt *Muon) Step(weights []float32, grads []float64, lr, weightDecay, invBatch float64, numWorkers int) {
	opt.mu.Lock()
	defer opt.mu.Unlock()

	// 1. Update 2D weight matrices via Muon
	lrF := float32(lr)
	wdF := float32(lr * weightDecay)
	beta := opt.Beta
	oneMinusBeta := 1.0 - beta
	clip := opt.ClipGrad

	for mi, m := range opt.Matrices {
		R, C := m.Rows, m.Cols
		G := opt.Scratch.Grad[:R*C]
		U := opt.Scratch.Upd[:R*C]

		// Extract and clip gradients into contiguous buffer G, then the Nesterov momentum update:
		// momentum = beta * momentum + (1 - beta) * G
		// update = (1 - beta) * G + beta * momentum
		momSlice := opt.Momentum[opt.momentumIdx[mi] : opt.momentumIdx[mi]+R*C]
		parallel.For(R, numWorkers, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				wOff := m.Offset + r*m.Stride
				rowG := G[r*C : (r+1)*C]
				for c := 0; c < C; c++ {
					g := float32(grads[wOff+c] * invBatch)
					if g > clip {
						g = clip
					} else if g < -clip {
						g = -clip
					}
					rowG[c] = g
				}
				for i := r * C; i < (r+1)*C; i++ {
					mVal := beta*momSlice[i] + oneMinusBeta*G[i]
					momSlice[i] = mVal
					U[i] = oneMinusBeta*G[i] + beta*mVal
				}
			}
		})

		// Orthogonalize update matrix using quintic Newton-Schulz
		NewtonSchulz5(U, U, R, C, opt.NSSteps, numWorkers, opt.Scratch)

		// Apply update to weights with decoupled weight decay
		parallel.For(R, numWorkers, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				wOff := m.Offset + r*m.Stride
				rowU := U[r*C : (r+1)*C]
				rowW := weights[wOff : wOff+C]
				if weightDecay > 0 {
					for c := 0; c < C; c++ {
						rowW[c] -= wdF * rowW[c]
					}
				}
				for c := 0; c < C; c++ {
					rowW[c] -= lrF * rowU[c]
				}
			}
		})
	}

	// 2. Update remaining 1D parameters, biases, and embeddings via AdamW
	// Embeddings and output head are token representations and use EmbedLRScale·lr.
	// Scalar biases and RMSNorm gammas use DefaultMuonAdamWLR.
	embedLR := lr * opt.EmbedLRScale
	scalarLR := DefaultMuonAdamWLR
	if lr*MuonScalarLRFraction < scalarLR {
		scalarLR = lr * MuonScalarLRFraction
	}

	sqrtBC2, bc1 := opt.AdamOpt.nextBiasCorrection()
	biasCorrection := sqrtBC2 / bc1

	stepAlphaEmbed := float32(embedLR) * biasCorrection
	stepAlphaScalar := float32(scalarLR) * biasCorrection
	var embedWD, scalarWD float32
	if weightDecay > 0 {
		embedWD = float32(embedLR * weightDecay)
		scalarWD = float32(scalarLR * weightDecay)
	}

	parallel.For(len(weights), numWorkers, func(lo, hi int) {
		for p := lo; p < hi; p++ {
			switch opt.groups[p] {
			case groupMatrix:
				continue // Already updated via Muon above
			case groupEmbed:
				opt.AdamOpt.updateParam(weights, grads, p, invBatch, stepAlphaEmbed, embedWD)
			default:
				opt.AdamOpt.updateParam(weights, grads, p, invBatch, stepAlphaScalar, scalarWD)
			}
		}
	})
}

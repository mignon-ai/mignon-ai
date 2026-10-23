package optim

import (
	"math"

	"github.com/mignon-ai/mignon-ai/internal/parallel"
)

// Adam implements AdamW: Adam moments with decoupled weight decay and per-parameter gradient clipping.
type Adam struct {
	M         []float32
	V         []float32
	Beta1     float32
	Beta2     float32
	Eps       float32
	StepCount int
	ClipGrad  float32
}

// NewAdam returns an AdamW optimizer for numParams parameters with the default moments.
// A non-positive clipGrad selects GradClipAdam.
func NewAdam(numParams int, clipGrad float64) *Adam {
	if clipGrad <= 0 {
		clipGrad = GradClipAdam
	}
	return &Adam{
		M:        make([]float32, numParams),
		V:        make([]float32, numParams),
		Beta1:    AdamBeta1,
		Beta2:    AdamBeta2,
		Eps:      AdamEps,
		ClipGrad: float32(clipGrad),
	}
}

// Step applies one AdamW update, split across up to workers goroutines.
func (opt *Adam) Step(weights []float32, grads []float64, lr, weightDecay, invBatch float64, workers int) {
	sqrtBC2, bc1 := opt.nextBiasCorrection()
	stepAlpha := float32(lr) * sqrtBC2 / bc1
	lrWD := float32(lr * weightDecay)
	parallel.For(len(weights), workers, func(lo, hi int) {
		for p := lo; p < hi; p++ {
			opt.updateParam(weights, grads, p, invBatch, stepAlpha, lrWD)
		}
	})
}

// nextBiasCorrection advances the step count and returns sqrt(1-β2^t) and 1-β1^t.
func (opt *Adam) nextBiasCorrection() (sqrtBC2, bc1 float32) {
	opt.StepCount++
	bc1 = float32(1.0 - math.Pow(float64(opt.Beta1), float64(opt.StepCount)))
	bc2 := float32(1.0 - math.Pow(float64(opt.Beta2), float64(opt.StepCount)))
	return float32(math.Sqrt(float64(bc2))), bc1
}

// updateParam applies one clipped AdamW update to parameter p. stepAlpha includes the bias
// correction; lrWD is the decoupled weight-decay factor (0 disables it).
func (opt *Adam) updateParam(weights []float32, grads []float64, p int, invBatch float64, stepAlpha, lrWD float32) {
	g := float32(grads[p] * invBatch)
	if g > opt.ClipGrad {
		g = opt.ClipGrad
	} else if g < -opt.ClipGrad {
		g = -opt.ClipGrad
	}

	opt.M[p] = opt.Beta1*opt.M[p] + (1.0-opt.Beta1)*g
	opt.V[p] = opt.Beta2*opt.V[p] + (1.0-opt.Beta2)*(g*g)

	stepVal := stepAlpha * (opt.M[p] / (float32(math.Sqrt(float64(opt.V[p]))) + opt.Eps))
	if lrWD != 0 {
		weights[p] -= lrWD * weights[p]
	}
	weights[p] -= stepVal
}

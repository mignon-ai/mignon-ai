package nn

import (
	"math"

	"github.com/mignon-ai/mignon-ai/internal/kernels"
)

const (
	RMSNormEpsilonF32 = float32(1e-5)
)

// RMSNormForwardF32 computes y = (x / rms) * gamma in float32 with zero heap allocations.
func RMSNormForwardF32(x, gamma, out, xHat []float32, eps float32) float32 {
	D := len(x)
	if D == 0 {
		return 1.0
	}

	sumSq := kernels.DotF32(x, x)

	rms := float32(math.Sqrt(float64(sumSq/float32(D) + eps)))
	invRMS := float32(1.0) / rms

	for d := 0; d < D; d++ {
		hat := x[d] * invRMS
		xHat[d] = hat
		out[d] = hat * gamma[d]
	}

	return rms
}

// RMSNormBackwardF32 computes gradients wrt x (dx) and gamma (dGamma) in float32 with zero heap allocations.
func RMSNormBackwardF32(dy, xHat, gamma []float32, rms float32, dx, dGamma []float32) {
	D := len(dy)
	if D == 0 || rms <= 0 {
		return
	}

	// 1. dGamma = dy * xHat
	for d := 0; d < D; d++ {
		dGamma[d] += dy[d] * xHat[d]
	}

	// 2. Inner product S = sum(dy_k * gamma_k * xHat_k)
	var sumProj float32
	for d := 0; d < D; d++ {
		sumProj += dy[d] * gamma[d] * xHat[d]
	}

	invRMS := float32(1.0) / rms
	invD := float32(1.0) / float32(D)
	projScaled := sumProj * invD

	// 3. dx = invRMS * (dy * gamma - xHat * (sumProj / D))
	for d := 0; d < D; d++ {
		dx[d] += invRMS * (dy[d]*gamma[d] - xHat[d]*projScaled)
	}
}

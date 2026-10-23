package optim

import (
	"math"

	"github.com/mignon-ai/mignon-ai/internal/kernels"
	"github.com/mignon-ai/mignon-ai/internal/parallel"
)

// NSScratch holds pre-allocated buffers for Newton-Schulz matrix iterations.
type NSScratch struct {
	MatX  []float32
	MatA  []float32
	MatA2 []float32
	MatB  []float32
	MatBX []float32
	Grad  []float32
	Upd   []float32
	packB kernels.GemmPackedB // right-hand operand of the current matrix product
}

// minNSScratchDim is the smallest square size NSScratch is allocated for.
const minNSScratchDim = 16

// NewNSScratch allocates scratch buffers up to maxDim x maxDim.
func NewNSScratch(maxDim int) *NSScratch {
	if maxDim < minNSScratchDim {
		maxDim = minNSScratchDim
	}
	cap2 := maxDim * maxDim
	return &NSScratch{
		MatX:  make([]float32, cap2),
		MatA:  make([]float32, cap2),
		MatA2: make([]float32, cap2),
		MatB:  make([]float32, cap2),
		MatBX: make([]float32, cap2),
		Grad:  make([]float32, cap2),
		Upd:   make([]float32, cap2),
	}
}

// NewtonSchulz5 computes the polar decomposition orthogonalization of matrix G (R x C).
// Uses Keller Jordan's quintic polynomial coefficients a=3.4445, b=-4.7750, c=2.0315.
// The matrix products are gemms split by rows across up to workers goroutines; every element
// is computed the same way regardless of the split, so the result does not depend on workers.
func NewtonSchulz5(dst, G []float32, R, C, steps, workers int, sc *NSScratch) {
	transposed := false
	RWork, CWork := R, C

	// Ensure RWork <= CWork for smaller Gram matrix (RWork x RWork)
	if RWork > CWork {
		transposed = true
		RWork, CWork = C, R
		// Transpose G (R x C) into MatX (C x R)
		for r := 0; r < R; r++ {
			for c := 0; c < C; c++ {
				sc.MatX[c*CWork+r] = G[r*C+c]
			}
		}
	} else {
		copy(sc.MatX[:RWork*CWork], G[:RWork*CWork])
	}

	X := sc.MatX[:RWork*CWork]

	// 1. Normalize by Frobenius norm
	invNorm := float32(1.0 / (math.Sqrt(float64(kernels.DotF32(X, X))) + 1e-7))
	for i := range X {
		X[i] *= invNorm
	}

	// 2. Quintic Newton-Schulz iterations:
	// A = X @ X.T (RWork x RWork)
	// B = b * A + c * A @ A
	// X = a * X + B @ X
	const (
		a = float32(3.4445)
		b = float32(-4.7750)
		c = float32(2.0315)
	)

	A := sc.MatA[:RWork*RWork]
	A2 := sc.MatA2[:RWork*RWork]
	B := sc.MatB[:RWork*RWork]
	BX := sc.MatBX[:RWork*CWork]

	for s := 0; s < steps; s++ {
		// A = X @ X^T
		clear(A)
		sc.packB.PackParallel(CWork, RWork, X, 1, CWork, workers)
		kernels.GemmPackedParallelF32(RWork, X, CWork, 1, &sc.packB, A, RWork, workers)

		// A2 = A @ A
		clear(A2)
		sc.packB.PackParallel(RWork, RWork, A, RWork, 1, workers)
		kernels.GemmPackedParallelF32(RWork, A, RWork, 1, &sc.packB, A2, RWork, workers)

		// B = b * A + c * A2 and BX = a * X, then BX += B @ X
		parallel.For(len(B), workers, func(lo, hi int) {
			for idx := lo; idx < hi; idx++ {
				B[idx] = b*A[idx] + c*A2[idx]
			}
		})
		parallel.For(len(BX), workers, func(lo, hi int) {
			for idx := lo; idx < hi; idx++ {
				BX[idx] = a * X[idx]
			}
		})
		sc.packB.PackParallel(RWork, CWork, X, CWork, 1, workers)
		kernels.GemmPackedParallelF32(RWork, B, RWork, 1, &sc.packB, BX, CWork, workers)

		// X = BX: swap the buffers instead of copying
		X, BX = BX, X
	}

	// 3. Aspect-ratio scaling: update *= max(1, R/C)^0.5
	scale := float32(math.Sqrt(math.Max(1.0, float64(R)/float64(C))))

	if transposed {
		// Transpose back from RWork x CWork (C x R) to R x C
		for r := 0; r < R; r++ {
			for c := 0; c < C; c++ {
				dst[r*C+c] = X[c*CWork+r] * scale
			}
		}
	} else {
		for i := 0; i < R*C; i++ {
			dst[i] = X[i] * scale
		}
	}
}

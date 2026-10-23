package optim

import (
	"math"
	"math/rand"
	"testing"
)

func TestNewtonSchulz5Orthogonality(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	testShapes := [][2]int{
		{8, 12},  // R < C
		{12, 8},  // R > C
		{16, 16}, // R == C
		{64, 96}, // CA Layer 1 shape
	}

	for _, shape := range testShapes {
		R, C := shape[0], shape[1]
		G := make([]float32, R*C)
		for i := range G {
			G[i] = float32(rng.Float64()*2.0 - 1.0)
		}

		maxDim := R
		if C > maxDim {
			maxDim = C
		}
		sc := NewNSScratch(maxDim)
		out := make([]float32, R*C)

		NewtonSchulz5(out, G, R, C, 5, 1, sc)

		// Check orthogonality of the smaller dimension:
		// If R <= C, check out @ out.T is close to I
		// If R > C, check out.T @ out is close to I
		if R <= C {
			// A = out @ out^T (R x R)
			for i := 0; i < R; i++ {
				for j := 0; j < R; j++ {
					val := float32(0.0)
					for k := 0; k < C; k++ {
						val += out[i*C+k] * out[j*C+k]
					}
					if i == j {
						// Diagonal elements should be bounded in [0.4, 1.8] (Keller Jordan S' ~ [0.5, 1.5])
						if val < 0.4 || val > 1.8 {
							t.Errorf("Shape (%d, %d) Gram diagonal out of bounds at (%d, %d): got %.4f",
								R, C, i, j, val)
						}
					} else {
						// Off-diagonal elements should be small (approximate orthogonality for 5 NS steps)
						if math.Abs(float64(val)) > 0.40 {
							t.Errorf("Shape (%d, %d) Gram off-diagonal non-zero at (%d, %d): got %.4f",
								R, C, i, j, val)
						}
					}
				}
			}
		} else {
			// A = out.T @ out (C x C)
			scaleSq := float32(math.Max(1.0, float64(R)/float64(C)))
			for i := 0; i < C; i++ {
				for j := 0; j < C; j++ {
					val := float32(0.0)
					for k := 0; k < R; k++ {
						val += out[k*C+i] * out[k*C+j]
					}
					if i == j {
						// Scaled diagonal elements bounded by scaleSq * [0.4, 1.8]
						if val < 0.4*scaleSq || val > 1.8*scaleSq {
							t.Errorf("Shape (%d, %d) Gram diagonal out of bounds at (%d, %d): got %.4f",
								R, C, i, j, val)
						}
					} else {
						// Off-diagonal elements should be small (approximate orthogonality for 5 NS steps)
						if math.Abs(float64(val)) > 0.40*float64(scaleSq) {
							t.Errorf("Shape (%d, %d) Gram off-diagonal non-zero at (%d, %d): got %.4f",
								R, C, i, j, val)
						}
					}
				}
			}
		}
	}
}

package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// int8RelTol bounds the float32-accumulation error relative to the sum of |x·w| terms.
const int8RelTol = 1e-5

// TestDotInt8F32MatchesReference checks the SIMD and pure-Go kernels against a float64
// reference for every length through several 32-wide blocks, so the 32-, 8- and 1-element
// loops and their boundaries are all exercised, including extreme weights.
func TestDotInt8F32MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for n := 0; n <= 100; n++ {
		x := make([]float32, n)
		w := make([]int8, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
			w[i] = int8(rng.Intn(256) - 128)
		}
		if n > 0 {
			w[0], w[n-1] = -128, 127
		}
		var want, mag float64
		for i := range x {
			want += float64(x[i]) * float64(w[i])
			mag += math.Abs(float64(x[i]) * float64(w[i]))
		}
		for name, got := range map[string]float32{"dispatch": DotInt8F32(x, w), "pure": dotInt8PureGoF32(x, w)} {
			if diff := math.Abs(float64(got) - want); diff > int8RelTol*mag+1e-6 {
				t.Fatalf("%s n=%d: got %g want %g (diff %g)", name, n, got, want, diff)
			}
		}
	}
}

// TestDotInt8F32UsesShorterSlice checks that mismatched lengths read only the common prefix.
func TestDotInt8F32UsesShorterSlice(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	w := []int8{1, 1, 1}
	if got := DotInt8F32(x, w); got != 6 {
		t.Fatalf("got %g, want 6", got)
	}
	if got := DotInt8F32(x[:2], []int8{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}); got != 3 {
		t.Fatalf("got %g, want 3", got)
	}
}

// TestMatVecInt8 checks rows, stride, scale and bias against a direct computation.
func TestMatVecInt8(t *testing.T) {
	const rows, cols, stride = 5, 37, 40
	rng := rand.New(rand.NewSource(2))
	w := make([]int8, rows*stride)
	for i := range w {
		w[i] = int8(rng.Intn(256) - 128)
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	bias := []float32{0.5, -1, 2, 0, 3}
	const scale = 0.01
	y := make([]float32, rows)
	MatVecInt8(y, w, stride, scale, x, bias)
	for r := 0; r < rows; r++ {
		want := float64(bias[r])
		for c := 0; c < cols; c++ {
			want += scale * float64(w[r*stride+c]) * float64(x[c])
		}
		if math.Abs(float64(y[r])-want) > 1e-4 {
			t.Fatalf("row %d: got %g want %g", r, y[r], want)
		}
	}
}

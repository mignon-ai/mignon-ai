package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// TestExpAndSigmoidAccuracy verifies range reduction precision against math.Exp
func TestExpAndSigmoidAccuracy(t *testing.T) {
	// Test values across standard operating range [-20.0, 20.0]
	testInputs := []float32{
		-20.0, -15.0, -10.0, -5.0, -2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0,
	}

	for _, x := range testInputs {
		gotExp := FastExpF32(x)
		wantExp := float32(math.Exp(float64(x)))
		relErr := math.Abs(float64(gotExp-wantExp)) / float64(wantExp)
		if relErr > 1e-4 {
			t.Errorf("FastExpF32(%f) relative error too high: got %e, want %e (relErr: %e)",
				x, gotExp, wantExp, relErr)
		}

		gotSig := FastSigmoidF32(x)
		wantSig := float32(1.0 / (1.0 + math.Exp(-float64(x))))
		diffSig := math.Abs(float64(gotSig - wantSig))
		if diffSig > 1e-4 {
			t.Errorf("FastSigmoidF32(%f) absolute error too high: got %f, want %f (diff: %e)",
				x, gotSig, wantSig, diffSig)
		}
	}
}

// TestSIMDPureGoVsASMEquivalence tests SIMD vector routines across irregular slice lengths
func TestSIMDPureGoVsASMEquivalence(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 47, 64, 65}
	rng := rand.New(rand.NewSource(1337))

	for _, n := range lengths {
		// 1. DotF32 vs pure Go
		x32 := make([]float32, n)
		y32 := make([]float32, n)
		for i := 0; i < n; i++ {
			x32[i] = float32(rng.NormFloat64())
			y32[i] = float32(rng.NormFloat64())
		}
		dpASM := DotF32(x32, y32)
		dpGo := dotProductPureGoF32(x32, y32)
		if math.Abs(float64(dpASM-dpGo)) > 1e-4 {
			t.Errorf("DotF32 mismatch at len %d: asm=%f, go=%f", n, dpASM, dpGo)
		}

		// 2. AddScaledF32 vs pure Go
		dstASM := make([]float32, n)
		dstGo := make([]float32, n)
		copy(dstASM, y32)
		copy(dstGo, y32)
		alpha := float32(0.35)
		AddScaledF32(dstASM, x32, alpha)
		vecAddScaledPureGoF32(dstGo, x32, alpha)
		for i := 0; i < n; i++ {
			if math.Abs(float64(dstASM[i]-dstGo[i])) > 1e-5 {
				t.Fatalf("AddScaledF32 mismatch at len %d index %d: asm=%f, go=%f", n, i, dstASM[i], dstGo[i])
			}
		}

		// 3. Dot4F32 inputs
		x0 := make([]float32, n)
		x1 := make([]float32, n)
		x2 := make([]float32, n)
		x3 := make([]float32, n)
		for i := 0; i < n; i++ {
			x0[i] = float32(rng.NormFloat64())
			x1[i] = float32(rng.NormFloat64())
			x2[i] = float32(rng.NormFloat64())
			x3[i] = float32(rng.NormFloat64())
		}
		// 4. Dot4F32 vs pure Go
		resASM := make([]float32, 4)
		resGo := make([]float32, 4)
		Dot4F32(x0, x1, x2, x3, y32, resASM)
		dotProduct4PureGoF32(x0, x1, x2, x3, y32, resGo)
		for j := 0; j < 4; j++ {
			if math.Abs(float64(resASM[j]-resGo[j])) > 1e-4 {
				t.Fatalf("Dot4F32 mismatch at len %d component %d: asm=%f, go=%f", n, j, resASM[j], resGo[j])
			}
		}

		// 5. VecSigmoidF32 vs pure Go
		sigASM := make([]float32, n)
		sigGo := make([]float32, n)
		VecSigmoidF32(sigASM, x0)
		vecSigmoidPureGoF32(sigGo, x0)
		for i := 0; i < n; i++ {
			if math.Abs(float64(sigASM[i]-sigGo[i])) > 1e-4 {
				t.Fatalf("VecSigmoidF32 mismatch at len %d index %d: asm=%f, go=%f", n, i, sigASM[i], sigGo[i])
			}
		}
	}
}

// TestSliceAliasingSafety verifies that vector ops handle overlapping slices safely via pure-Go fallback
func TestSliceAliasingSafety(t *testing.T) {
	buf := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	// Aliased in-place: y is buf, x is buf (same slice)
	expected := make([]float32, len(buf))
	for i := range buf {
		expected[i] = buf[i] + 2.0*buf[i]
	}
	AddScaledF32(buf, buf, 2.0)
	for i := range buf {
		if math.Abs(float64(buf[i]-expected[i])) > 1e-5 {
			t.Fatalf("AddScaledF32 in-place aliasing mismatch at %d: got %f, want %f", i, buf[i], expected[i])
		}
	}
}

// TestSIMDRuntimeFallback verifies that toggling HasAVX2 = false operates accurately
func TestSIMDRuntimeFallback(t *testing.T) {
	orig := HasAVX2
	defer func() { HasAVX2 = orig }()

	HasAVX2 = false

	x := []float32{0.5, -1.2, 3.4, -0.7}
	y := []float32{1.1, 2.2, -0.5, 0.9}
	dot := DotF32(x, y)
	ref := dotProductPureGoF32(x, y)
	if math.Abs(float64(dot-ref)) > 1e-6 {
		t.Fatalf("Fallback DotF32 mismatch: got %f, want %f", dot, ref)
	}

	expDst := make([]float32, 4)
	VecExpF32(expDst, x)
	for i := range x {
		refExp := FastExpF32(x[i])
		if math.Abs(float64(expDst[i]-refExp)) > 1e-6 {
			t.Fatalf("Fallback VecExpF32 mismatch at %d: got %f, want %f", i, expDst[i], refExp)
		}
	}
}

// TestFastExpExtremeBounds verifies no NaN or panic on extreme positive/negative inputs
func TestFastExpExtremeBounds(t *testing.T) {
	extremes := []float32{-1e9, -1e6, -100.0, -89.0, -88.0, 0.0, 88.0, 89.0, 100.0, 1e6, 1e9}
	for _, x := range extremes {
		res := FastExpF32(x)
		if math.IsNaN(float64(res)) {
			t.Fatalf("FastExpF32(%f) produced NaN", x)
		}
		sig := FastSigmoidF32(x)
		if math.IsNaN(float64(sig)) {
			t.Fatalf("FastSigmoidF32(%f) produced NaN", x)
		}
	}
}

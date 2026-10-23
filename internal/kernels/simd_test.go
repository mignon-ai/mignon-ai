package kernels

import (
	"math"
	"math/rand"
	"testing"
)

func TestDotProductF32SIMDEquivalence(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 64, 96, 128, 192, 256, 384, 576, 1024}
	rng := rand.New(rand.NewSource(42))

	for _, n := range lengths {
		x := make([]float32, n)
		y := make([]float32, n)
		for i := 0; i < n; i++ {
			x[i] = rng.Float32()
			y[i] = rng.Float32()
		}

		expected := dotProductPureGoF32(x, y)
		actual := DotF32(x, y)

		diff := float32(math.Abs(float64(actual - expected)))
		tol := float32(1e-4) * (float32(math.Abs(float64(expected))) + 1.0)
		if diff > tol {
			t.Fatalf("F32 Length %d: expected %.7f, got %.7f (diff: %e > tol: %e)", n, expected, actual, diff, tol)
		}
	}
}

func TestVecAddScaledF32SIMDEquivalence(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 64, 96, 128, 192, 256, 384, 576, 1024}
	rng := rand.New(rand.NewSource(42))

	for _, n := range lengths {
		yExp := make([]float32, n)
		yAct := make([]float32, n)
		x := make([]float32, n)
		for i := 0; i < n; i++ {
			val := rng.Float32()
			yExp[i] = val
			yAct[i] = val
			x[i] = rng.Float32()
		}
		alpha := rng.Float32()

		vecAddScaledPureGoF32(yExp, x, alpha)
		AddScaledF32(yAct, x, alpha)

		for i := 0; i < n; i++ {
			diff := float32(math.Abs(float64(yAct[i] - yExp[i])))
			tol := float32(1e-4) * (float32(math.Abs(float64(yExp[i]))) + 1.0)
			if diff > tol {
				t.Fatalf("F32 Length %d at [%d]: expected %.7f, got %.7f", n, i, yExp[i], yAct[i])
			}
		}
	}
}

func TestDotProduct4F32SIMDEquivalence(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 64, 96, 128, 192, 256, 384, 576, 1024}
	rng := rand.New(rand.NewSource(42))

	for _, n := range lengths {
		x := make([]float32, n)
		y0 := make([]float32, n)
		y1 := make([]float32, n)
		y2 := make([]float32, n)
		y3 := make([]float32, n)
		for i := 0; i < n; i++ {
			x[i] = rng.Float32()
			y0[i] = rng.Float32()
			y1[i] = rng.Float32()
			y2[i] = rng.Float32()
			y3[i] = rng.Float32()
		}
		resExp := make([]float32, 4)
		resAct := make([]float32, 4)

		dotProduct4PureGoF32(x, y0, y1, y2, y3, resExp)
		Dot4F32(x, y0, y1, y2, y3, resAct)

		for k := 0; k < 4; k++ {
			diff := float32(math.Abs(float64(resAct[k] - resExp[k])))
			tol := float32(1e-4) * (float32(math.Abs(float64(resExp[k]))) + 1.0)
			if diff > tol {
				t.Fatalf("F32 dp4 Length %d at [%d]: expected %.7f, got %.7f", n, k, resExp[k], resAct[k])
			}
		}
	}
}

func BenchmarkDotProductSIMD_F32(b *testing.B) {
	n := 576
	x := make([]float32, n)
	y := make([]float32, n)
	for i := 0; i < n; i++ {
		x[i] = 1.001
		y[i] = 0.999
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DotF32(x, y)
	}
}

func BenchmarkVecAddScaledSIMD_F32(b *testing.B) {
	n := 576
	y := make([]float32, n)
	x := make([]float32, n)
	for i := 0; i < n; i++ {
		y[i] = 1.001
		x[i] = 0.999
	}
	alpha := float32(0.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AddScaledF32(y, x, alpha)
	}
}

func BenchmarkDotProduct4SIMD_F32(b *testing.B) {
	n := 576
	x := make([]float32, n)
	y0 := make([]float32, n)
	y1 := make([]float32, n)
	y2 := make([]float32, n)
	y3 := make([]float32, n)
	res := make([]float32, 4)
	for i := 0; i < n; i++ {
		x[i] = 1.001
		y0[i] = 0.999
		y1[i] = 0.998
		y2[i] = 0.997
		y3[i] = 0.996
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Dot4F32(x, y0, y1, y2, y3, res)
	}
}

func TestVecExpSIMDEquivalence(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 31, 32, 64, 128, 256, 512}
	rng := rand.New(rand.NewSource(1234))

	for _, n := range lengths {
		src := make([]float32, n)
		for i := 0; i < n; i++ {
			src[i] = float32(rng.Float64()*20.0 - 10.0) // range [-10, 10]
		}
		dstASM := make([]float32, n)
		dstPure := make([]float32, n)

		vecExpASM_F32(dstASM, src)
		vecExpPureGoF32(dstPure, src)

		for i := 0; i < n; i++ {
			mathVal := float32(math.Exp(float64(src[i])))
			diffASM := math.Abs(float64(dstASM[i] - mathVal))
			relDiff := diffASM / math.Max(float64(mathVal), 1e-6)
			if relDiff > 1e-4 {
				t.Fatalf("n=%d, i=%d, x=%f: asm=%e vs math=%e (relDiff=%e > 1e-4)", n, i, src[i], dstASM[i], mathVal, relDiff)
			}
			diffPure := math.Abs(float64(dstASM[i] - dstPure[i]))
			if diffPure > 1e-4*math.Max(float64(dstPure[i]), 1e-6) {
				t.Fatalf("n=%d, i=%d: asm=%e vs pure=%e", n, i, dstASM[i], dstPure[i])
			}
		}
	}
}

func TestVecSigmoidSIMDEquivalence(t *testing.T) {
	lengths := []int{0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 31, 32, 64, 128, 256, 512}
	rng := rand.New(rand.NewSource(5678))

	for _, n := range lengths {
		src := make([]float32, n)
		for i := 0; i < n; i++ {
			src[i] = float32(rng.Float64()*30.0 - 15.0) // range [-15, 15]
		}
		dstASM := make([]float32, n)
		dstPure := make([]float32, n)

		vecSigmoidASM_F32(dstASM, src)
		vecSigmoidPureGoF32(dstPure, src)

		for i := 0; i < n; i++ {
			mathVal := float32(1.0 / (1.0 + math.Exp(float64(-src[i]))))
			diffASM := math.Abs(float64(dstASM[i] - mathVal))
			if diffASM > 1e-4 {
				t.Fatalf("n=%d, i=%d, x=%f: asm=%e vs math=%e (diff=%e > 1e-4)", n, i, src[i], dstASM[i], mathVal, diffASM)
			}
			diffPure := math.Abs(float64(dstASM[i] - dstPure[i]))
			if diffPure > 1e-4 {
				t.Fatalf("n=%d, i=%d: asm=%e vs pure=%e", n, i, dstASM[i], dstPure[i])
			}
		}
	}
}

func BenchmarkVecSigmoidSIMD_F32(b *testing.B) {
	n := 512
	src := make([]float32, n)
	dst := make([]float32, n)
	for i := 0; i < n; i++ {
		src[i] = float32(i%30) - 15.0
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VecSigmoidF32(dst, src)
	}
}

func BenchmarkVecSigmoidMathExpScalar_F32(b *testing.B) {
	n := 512
	src := make([]float32, n)
	dst := make([]float32, n)
	for i := 0; i < n; i++ {
		src[i] = float32(i%30) - 15.0
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < n; j++ {
			dst[j] = 1.0 / (1.0 + float32(math.Exp(float64(-src[j]))))
		}
	}
}

func TestPartialOverlap(t *testing.T) {
	buf32 := make([]float32, 16)
	other := make([]float32, 8)

	cases := []struct {
		name string
		got  bool
		want bool
	}{
		{"disjoint", partialOverlap(buf32[:8], other), false},
		{"exact alias", partialOverlap(buf32[:8], buf32[:8]), false},
		{"same start different len", partialOverlap(buf32[:8], buf32[:4]), false},
		{"partial", partialOverlap(buf32[0:8], buf32[4:12]), true},
		{"adjacent", partialOverlap(buf32[0:8], buf32[8:16]), false},
		{"contained", partialOverlap(buf32[0:16], buf32[3:5]), true},
		{"empty", partialOverlap(buf32[:0], buf32[:8]), false},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: partialOverlap = %v, want %v", c.name, c.got, c.want)
		}
	}
}

// TestExactAliasMatchesPureGo verifies that in-place (exactly aliased) calls, which now take the
// SIMD path, produce the same results as the pure-Go kernels.
func TestExactAliasMatchesPureGo(t *testing.T) {
	const n = 37 // exercises 16-wide, 8-wide and scalar tails
	const tol = 1e-5
	rng := rand.New(rand.NewSource(11))
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}

	assertClose := func(name string, got, want []float32) {
		t.Helper()
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > tol*math.Max(1, math.Abs(float64(want[i]))) {
				t.Fatalf("%s: index %d got %f want %f", name, i, got[i], want[i])
			}
		}
	}

	got := append([]float32(nil), src...)
	want := append([]float32(nil), src...)
	AddScaledF32(got, got, 0.5)
	vecAddScaledPureGoF32(want, want, 0.5)
	assertClose("AddScaledF32", got, want)

	got = append(got[:0], src...)
	want = append(want[:0], src...)
	VecExpF32(got, got)
	vecExpPureGoF32(want, want)
	assertClose("VecExpF32", got, want)

	got = append(got[:0], src...)
	want = append(want[:0], src...)
	VecSigmoidF32(got, got)
	vecSigmoidPureGoF32(want, want)
	assertClose("VecSigmoidF32", got, want)
}

// TestPartialOverlapUsesSequentialSemantics verifies that partially overlapping writes match the
// sequential pure-Go result rather than a block-wise SIMD result.
func TestPartialOverlapUsesSequentialSemantics(t *testing.T) {
	const n = 40
	base := make([]float32, n+1)
	for i := range base {
		base[i] = float32(i%7) - 3
	}
	got := append([]float32(nil), base...)
	want := append([]float32(nil), base...)
	AddScaledF32(got[1:], got[:n], 0.5)
	vecAddScaledPureGoF32(want[1:], want[:n], 0.5)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("index %d got %f want %f", i, got[i], want[i])
		}
	}
}

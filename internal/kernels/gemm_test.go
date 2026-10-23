package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// gemmTestTol is the allowed error relative to Σ|a·b| + |c₀| per element (float32 rounding).
const gemmTestTol = 1e-5

type gemmCase struct {
	name             string
	m, n, k          int
	transA, transB   bool
	padA, padB, padC int // extra row-stride padding
}

func gemmCases() []gemmCase {
	return []gemmCase{
		{name: "exact tiles", m: 12, n: 32, k: 20},
		{name: "edges", m: 13, n: 37, k: 9},
		{name: "smaller than a tile", m: 5, n: 15, k: 3},
		{name: "single", m: 1, n: 1, k: 1},
		{name: "k across blocks", m: 6, n: 16, k: 2*gemmBlockK + 7},
		{name: "transposed A", m: 18, n: 48, k: 33, transA: true},
		{name: "padded strides", m: 14, n: 40, k: 17, padA: 3, padB: 1, padC: 5},
		{name: "transposed padded", m: 7, n: 21, k: 30, transA: true, padA: 2, padB: 4, padC: 1},
		{name: "transposed B", m: 12, n: 48, k: 40, transB: true, padB: 1},
		{name: "both transposed, edges", m: 19, n: 35, k: gemmBlockK + 3, transA: true, transB: true, padA: 1, padB: 2},
	}
}

// gemmOperands builds random A, B, C for tc and returns them with A's strides.
func gemmOperands(tc gemmCase, rng *rand.Rand) (a []float32, rsA, csA int, b []float32, rsB, csB int, c []float32, ldc int) {
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	if tc.transA { // stored k×m
		rsA, csA = 1, tc.m+tc.padA
		a = rnd(tc.k * csA)
	} else {
		rsA, csA = tc.k+tc.padA, 1
		a = rnd(tc.m * rsA)
	}
	if tc.transB { // stored n×k, like a weight matrix used as Wᵀ
		rsB, csB = 1, tc.k+tc.padB
		b = rnd(tc.n * csB)
	} else {
		rsB, csB = tc.n+tc.padB, 1
		b = rnd(tc.k * rsB)
	}
	ldc = tc.n + tc.padC
	return a, rsA, csA, b, rsB, csB, rnd(tc.m * ldc), ldc
}

func checkGemm(t *testing.T, tc gemmCase) {
	var pack []float32
	t.Helper()
	a, rsA, csA, b, rsB, csB, c, ldc := gemmOperands(tc, rand.New(rand.NewSource(int64(tc.m*1000+tc.n))))
	c0 := append([]float32(nil), c...)
	GemmF32(&pack, tc.m, tc.n, tc.k, a, rsA, csA, b, rsB, csB, c, ldc)
	for i := 0; i < tc.m; i++ {
		for j := 0; j < tc.n; j++ {
			want, scale := float64(c0[i*ldc+j]), math.Abs(float64(c0[i*ldc+j]))
			for p := 0; p < tc.k; p++ {
				prod := float64(a[i*rsA+p*csA]) * float64(b[p*rsB+j*csB])
				want += prod
				scale += math.Abs(prod)
			}
			if got := float64(c[i*ldc+j]); math.Abs(got-want) > gemmTestTol*scale {
				t.Fatalf("%s: C[%d,%d] = %g, want %g", tc.name, i, j, got, want)
			}
		}
	}
	// Padding between rows of C must be untouched.
	for i := 0; i < tc.m; i++ {
		for j := tc.n; j < ldc && i*ldc+j < len(c); j++ {
			if c[i*ldc+j] != c0[i*ldc+j] {
				t.Fatalf("%s: padding C[%d,%d] was written", tc.name, i, j)
			}
		}
	}
}

func TestGemmMatchesNaive(t *testing.T) {
	for _, tc := range gemmCases() {
		checkGemm(t, tc)
	}
}

func TestGemmPureGoMatchesNaive(t *testing.T) {
	saved := HasAVX2
	HasAVX2 = false
	defer func() { HasAVX2 = saved }()
	for _, tc := range gemmCases() {
		checkGemm(t, tc)
	}
}

func TestGemmZeroSizeIsNoOp(t *testing.T) {
	var pack []float32
	c := []float32{1, 2}
	GemmF32(&pack, 0, 2, 3, nil, 3, 1, nil, 2, 1, c, 2)
	GemmF32(&pack, 1, 2, 0, nil, 0, 1, nil, 2, 1, c, 2)
	if c[0] != 1 || c[1] != 2 {
		t.Fatal("zero-size gemm modified C")
	}
}

func TestGemmPanicsOnShortSlices(t *testing.T) {
	var pack []float32
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic for a short B")
		}
	}()
	a, b, c := make([]float32, 6*4), make([]float32, 4*16-1), make([]float32, 6*16)
	GemmF32(&pack, 6, 16, 4, a, 4, 1, b, 16, 1, c, 16)
}

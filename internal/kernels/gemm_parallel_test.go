package kernels

import (
	"math/rand"
	"slices"
	"testing"
)

// gemmParallelWorkers are the worker counts compared against the serial packed gemm.
var gemmParallelWorkers = []int{1, 2, 3, 8}

// TestGemmPackedParallelMatchesSerial checks that packing B and splitting C's rows across
// workers gives bit-identical results to Pack and one GemmPackedF32 call, edge tiles included.
func TestGemmPackedParallelMatchesSerial(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, tc := range gemmCases() {
		a, rsA, csA, b, rsB, csB, c, ldc := gemmOperands(tc, rng)
		var pb GemmPackedB
		pb.Pack(tc.k, tc.n, b, rsB, csB)
		want := slices.Clone(c)
		GemmPackedF32(tc.m, a, rsA, csA, &pb, want, ldc)
		for _, w := range gemmParallelWorkers {
			var pbPar GemmPackedB
			pbPar.PackParallel(tc.k, tc.n, b, rsB, csB, w)
			if !slices.Equal(pbPar.data, pb.data) {
				t.Errorf("%s, %d workers: parallel pack differs from Pack", tc.name, w)
			}
			got := slices.Clone(c)
			GemmPackedParallelF32(tc.m, a, rsA, csA, &pbPar, got, ldc, w)
			if !slices.Equal(got, want) {
				t.Errorf("%s, %d workers: result differs from the serial gemm", tc.name, w)
			}
		}
	}
}

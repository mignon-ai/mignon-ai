package kernels

import "github.com/mignon-ai/mignon-ai/internal/parallel"

// GemmPackedParallelF32 computes C += A·B like GemmPackedF32, splitting the rows of C into
// register-tile-aligned ranges across up to workers goroutines that share the read-only packed B.
// Every element goes through the same kernel with the same summation order whatever the split,
// so the result does not depend on workers.
func GemmPackedParallelF32(m int, a []float32, rsA, csA int, pb *GemmPackedB, c []float32, ldc, workers int) {
	tiles := (m + gemmTileM - 1) / gemmTileM
	parallel.For(tiles, workers, func(lo, hi int) {
		r0, r1 := lo*gemmTileM, min(hi*gemmTileM, m)
		GemmPackedF32(r1-r0, a[r0*rsA:], rsA, csA, pb, c[r0*ldc:], ldc)
	})
}

// PackParallel is Pack with the strips of B split across up to workers goroutines.
func (pb *GemmPackedB) PackParallel(k, n int, b []float32, rsB, csB, workers int) {
	*pb = newGemmPackedB(pb.data, k, n)
	strips := gemmStrips(n)
	kBlocks := (k + gemmBlockK - 1) / gemmBlockK
	parallel.For(kBlocks*strips, workers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			pb.packStrip((t/strips)*gemmBlockK, t%strips, b, rsB, csB)
		}
	})
}

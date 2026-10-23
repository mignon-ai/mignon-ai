package kernels

import "fmt"

// Dense float32 matrix multiply-accumulate used by the training forward and backward passes.
//
// GemmF32 computes C += A·B where
//
//	A is m×k with element (i, p) at a[i*rsA + p*csA]
//	B is k×n with element (p, j) at b[p*rsB + j*csB]
//	C is m×n with element (i, j) at c[i*ldc + j]
//
// Row-major operands use (row stride, 1); transposed ones use (1, row stride), so a weight
// matrix stored out×in with a trailing bias column (stride in+1) can be used as either W or Wᵀ.
//
// B is first packed (GemmPackedB) into gemmBlockK-row blocks of contiguous gemmTileN-wide
// strips, zero-padded on the right, so each strip stays in L1 while every row tile reuses it.
// The AVX2 kernel then computes C in gemmTileM×gemmTileN register tiles; partial tiles at the
// bottom and right edges go through small zero-padded scratch tiles. Callers that multiply by
// the same B several times can pack it once and call GemmPackedF32.

const (
	gemmTileM = 6  // rows of C per register tile (6 broadcasts of A per k step)
	gemmTileN = 16 // columns of C per register tile (two 8-wide AVX vectors)
	// gemmBlockK bounds the k range per pass so one packed strip of B (gemmBlockK × 64 B)
	// stays in L1 while it is reused by every row tile.
	gemmBlockK = 256
)

// GemmPackedB is B (k×n) laid out for the kernel: for each k block starting at p0 (pk rows),
// strip s holds rows p0..p0+pk-1, columns s·gemmTileN.. as pk contiguous gemmTileN-wide rows.
type GemmPackedB struct {
	data []float32
	k, n int
}

// Pack repacks B (k×n, element (p, j) at b[p*rsB + j*csB]) into pb, reusing its buffer.
func (pb *GemmPackedB) Pack(k, n int, b []float32, rsB, csB int) {
	*pb = packGemmB(pb.data, k, n, b, rsB, csB)
}

// gemmPackedSize returns the number of floats a packed k×n B needs.
func gemmPackedSize(k, n int) int {
	return k * gemmStrips(n) * gemmTileN
}

func gemmStrips(n int) int { return (n + gemmTileN - 1) / gemmTileN }

// stripOffset returns the offset of strip s in the k block starting at p0 with pk rows.
func (pb *GemmPackedB) stripOffset(p0, pk, s int) int {
	return p0*gemmStrips(pb.n)*gemmTileN + s*pk*gemmTileN
}

// packGemmB packs B (k×n, element (p, j) at b[p*rsB + j*csB]) into buf, growing it if needed.
func packGemmB(buf []float32, k, n int, b []float32, rsB, csB int) GemmPackedB {
	pb := newGemmPackedB(buf, k, n)
	strips := gemmStrips(n)
	for p0 := 0; p0 < k; p0 += gemmBlockK {
		for s := 0; s < strips; s++ {
			pb.packStrip(p0, s, b, rsB, csB)
		}
	}
	return pb
}

// newGemmPackedB returns an unfilled packed k×n B backed by buf, growing it if needed.
func newGemmPackedB(buf []float32, k, n int) GemmPackedB {
	size := gemmPackedSize(k, n)
	if cap(buf) < size {
		buf = make([]float32, size)
	}
	return GemmPackedB{data: buf[:size], k: k, n: n}
}

// packStrip fills strip s of the k block starting at p0 from B. Strips never overlap, so
// different strips can be packed concurrently.
func (pb *GemmPackedB) packStrip(p0, s int, b []float32, rsB, csB int) {
	pk := min(gemmBlockK, pb.k-p0)
	strip := pb.data[pb.stripOffset(p0, pk, s) : pb.stripOffset(p0, pk, s)+pk*gemmTileN]
	j0 := s * gemmTileN
	width := min(gemmTileN, pb.n-j0)
	if csB == 1 {
		for p := 0; p < pk; p++ {
			row := strip[p*gemmTileN : (p+1)*gemmTileN]
			off := (p0+p)*rsB + j0
			if width == gemmTileN { // fixed-size copy compiles to inline moves
				*(*[gemmTileN]float32)(row) = *(*[gemmTileN]float32)(b[off : off+gemmTileN])
				continue
			}
			copy(row, b[off:off+width])
			clear(row[width:])
		}
		return
	}
	if rsB != 1 {
		for p := 0; p < pk; p++ {
			row := strip[p*gemmTileN : (p+1)*gemmTileN]
			for jj := 0; jj < width; jj++ {
				row[jj] = b[(p0+p)*rsB+(j0+jj)*csB]
			}
			clear(row[width:])
		}
		return
	}
	// Transposed B (rsB == 1): each lane reads one contiguous source row.
	var cols [gemmTileN][]float32 // the source column of each strip lane
	for jj := 0; jj < width; jj++ {
		off := (j0+jj)*csB + p0
		cols[jj] = b[off : off+pk]
	}
	for p := 0; p < pk; p++ {
		row := strip[p*gemmTileN : (p+1)*gemmTileN]
		for jj := 0; jj < width; jj++ {
			row[jj] = cols[jj][p]
		}
		clear(row[width:])
	}
}

// GemmF32 computes C += A·B (see the file comment for the layouts), packing B into *pack
// (grown as needed, so callers can reuse it). It panics if a slice is too short for the given
// shape and strides, because the assembly kernel does not bounds-check.
func GemmF32(pack *[]float32, m, n, k int, a []float32, rsA, csA int, b []float32, rsB, csB int, c []float32, ldc int) {
	if m <= 0 || n <= 0 || k <= 0 {
		return
	}
	gemmCheckBounds(m, n, k, a, rsA, csA, b, rsB, csB, c, ldc)
	pb := packGemmB(*pack, k, n, b, rsB, csB)
	*pack = pb.data
	GemmPackedF32(m, a, rsA, csA, &pb, c, ldc)
}

// GemmPackedF32 computes C += A·B for a B packed by packGemmB (A is m×pb.k, C is m×pb.n).
func GemmPackedF32(m int, a []float32, rsA, csA int, pb *GemmPackedB, c []float32, ldc int) {
	n, k := pb.n, pb.k
	if m <= 0 || n <= 0 || k <= 0 {
		return
	}
	gemmCheckBounds(m, n, k, a, rsA, csA, nil, 0, 0, c, ldc)
	fullM := m - m%gemmTileM
	strips := gemmStrips(n)
	fullStrips := n / gemmTileN
	for p0 := 0; p0 < k; p0 += gemmBlockK {
		pk := min(gemmBlockK, k-p0)
		for s := 0; s < strips; s++ {
			strip := pb.data[pb.stripOffset(p0, pk, s):]
			j := s * gemmTileN
			for i := 0; i < m; i += gemmTileM {
				switch {
				case !HasAVX2:
					gemmTilePureGo(min(gemmTileM, m-i), min(gemmTileN, n-j), pk, a[i*rsA+p0*csA:], rsA, csA, strip, c[i*ldc+j:], ldc)
				case i < fullM && s < fullStrips:
					gemmKernel6x16ASM(pk, &a[i*rsA+p0*csA], rsA, csA, &strip[0], gemmTileN, &c[i*ldc+j], ldc)
				default:
					gemmEdgeTile(min(gemmTileM, m-i), min(gemmTileN, n-j), pk, a[i*rsA+p0*csA:], rsA, csA, strip, c[i*ldc+j:], ldc)
				}
			}
		}
	}
}

// gemmEdgeTile runs the register kernel on a partial rows×cols tile by copying A's rows and C's
// tile into zero-padded scratch, so the kernel never reads or writes outside the matrices.
func gemmEdgeTile(rows, cols, pk int, a []float32, rsA, csA int, strip []float32, c []float32, ldc int) {
	var tileA [gemmTileM * gemmBlockK]float32
	var tileC [gemmTileM * gemmTileN]float32
	for r := 0; r < rows; r++ {
		for p := 0; p < pk; p++ {
			tileA[r*pk+p] = a[r*rsA+p*csA]
		}
		copy(tileC[r*gemmTileN:r*gemmTileN+cols], c[r*ldc:r*ldc+cols])
	}
	gemmKernel6x16ASM(pk, &tileA[0], pk, 1, &strip[0], gemmTileN, &tileC[0], gemmTileN)
	for r := 0; r < rows; r++ {
		copy(c[r*ldc:r*ldc+cols], tileC[r*gemmTileN:r*gemmTileN+cols])
	}
}

// gemmCheckBounds panics unless the strides are valid and the last element of A, B and C is
// inside its slice. A nil b skips the B check (packed B is sized by construction).
func gemmCheckBounds(m, n, k int, a []float32, rsA, csA int, b []float32, rsB, csB int, c []float32, ldc int) {
	if rsA < 0 || csA < 0 || rsB < 0 || csB < 0 || ldc < n {
		panic(fmt.Sprintf("GemmF32: invalid strides rsA=%d csA=%d rsB=%d csB=%d ldc=%d for n=%d", rsA, csA, rsB, csB, ldc, n))
	}
	if last := (m-1)*rsA + (k-1)*csA; last >= len(a) {
		panic(fmt.Sprintf("GemmF32: A needs index %d, has %d elements", last, len(a)))
	}
	if last := (k-1)*rsB + (n-1)*csB; b != nil && last >= len(b) {
		panic(fmt.Sprintf("GemmF32: B needs index %d, has %d elements", last, len(b)))
	}
	if last := (m-1)*ldc + n - 1; last >= len(c) {
		panic(fmt.Sprintf("GemmF32: C needs index %d, has %d elements", last, len(c)))
	}
}

// gemmTilePureGo is the portable tile kernel: C[rows×cols] += A[rows×pk] · strip[pk×cols].
func gemmTilePureGo(rows, cols, pk int, a []float32, rsA, csA int, strip []float32, c []float32, ldc int) {
	for r := 0; r < rows; r++ {
		cRow := c[r*ldc : r*ldc+cols]
		for p := 0; p < pk; p++ {
			av := a[r*rsA+p*csA]
			for j, bv := range strip[p*gemmTileN : p*gemmTileN+cols] {
				cRow[j] += av * bv
			}
		}
	}
}

// AddColumnSumsF32 adds the column sums of the rows×cols matrix m (row-major) to dst, adding
// column j to dst[j*dstStride]. Used for bias gradients next to gemm weight gradients. Rows are
// added one at a time so m is read sequentially.
func AddColumnSumsF32(dst []float32, dstStride int, m []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		row := m[i*cols : (i+1)*cols]
		if dstStride == 1 {
			AddScaledF32(dst[:cols], row, 1)
			continue
		}
		for j, x := range row {
			dst[j*dstStride] += x
		}
	}
}

// SetRowsToBiasF32 sets every row of the rows×cols matrix c (row-major) to the bias vector
// stored at bias[j*biasStride], so a following gemm accumulates on top of it.
func SetRowsToBiasF32(c []float32, rows, cols int, bias []float32, biasStride int) {
	for j := 0; j < cols; j++ {
		c[j] = bias[j*biasStride]
	}
	for i := 1; i < rows; i++ {
		copy(c[i*cols:(i+1)*cols], c[:cols])
	}
}

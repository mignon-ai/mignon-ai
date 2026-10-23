// Package kernels provides the dense float math used throughout the model: SIMD vector
// operations and the gemm matrix multiply.
// The implementations have two variants: a fast assembly version (AVX2/FMA) and a
// pure‑Go fallback. The wrappers below select the appropriate implementation at
// runtime based on CPU feature detection (see cpu_features.go). Kernels that write
// their output fall back to pure Go when the output partially overlaps an input,
// because the SIMD kernels process several elements per iteration. Exact aliasing
// (same start address) is safe because every output element depends only on input
// elements at the same index. Read-only kernels (dot products) never need a guard.
package kernels

import (
	"math"
	"unsafe"
)

// partialOverlap reports whether a and b share memory but start at different addresses.
// Such slices make element-wise SIMD writes order-dependent, so callers fall back to pure Go.
func partialOverlap(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	elemSize := unsafe.Sizeof(a[0])
	aStart := uintptr(unsafe.Pointer(&a[0]))
	bStart := uintptr(unsafe.Pointer(&b[0]))
	if aStart == bStart {
		return false
	}
	aEnd := aStart + uintptr(len(a))*elemSize
	bEnd := bStart + uintptr(len(b))*elemSize
	return aStart < bEnd && bStart < aEnd
}

// DotF32 computes the dot product of two float32 slices.
func DotF32(x, y []float32) float32 {
	if !HasAVX2 {
		return dotProductPureGoF32(x, y)
	}
	return dotProductASM_F32(x, y)
}

func dotProductPureGoF32(x, y []float32) float32 {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	var s0, s1, s2, s3 float32
	j := 0
	for ; j <= n-4; j += 4 {
		s0 += x[j] * y[j]
		s1 += x[j+1] * y[j+1]
		s2 += x[j+2] * y[j+2]
		s3 += x[j+3] * y[j+3]
	}
	for ; j < n; j++ {
		s0 += x[j] * y[j]
	}
	return (s0 + s1) + (s2 + s3)
}

// AddScaledF32 computes y[i] += alpha * x[i] in float32.
func AddScaledF32(y, x []float32, alpha float32) {
	if !HasAVX2 || partialOverlap(y, x) {
		vecAddScaledPureGoF32(y, x, alpha)
		return
	}
	vecAddScaledASM_F32(y, x, alpha)
}

func vecAddScaledPureGoF32(y, x []float32, alpha float32) {
	n := len(y)
	if len(x) < n {
		n = len(x)
	}
	j := 0
	for ; j <= n-4; j += 4 {
		y[j] += alpha * x[j]
		y[j+1] += alpha * x[j+1]
		y[j+2] += alpha * x[j+2]
		y[j+3] += alpha * x[j+3]
	}
	for ; j < n; j++ {
		y[j] += alpha * x[j]
	}
}

// Dot4F32 computes 4 dot products simultaneously into res[0..3] in float32.
func Dot4F32(x, y0, y1, y2, y3, res []float32) {
	if !HasAVX2 {
		dotProduct4PureGoF32(x, y0, y1, y2, y3, res)
		return
	}
	dotProduct4ASM_F32(x, y0, y1, y2, y3, res)
}

func dotProduct4PureGoF32(x, y0, y1, y2, y3, res []float32) {
	res[0] = dotProductPureGoF32(x, y0)
	res[1] = dotProductPureGoF32(x, y1)
	res[2] = dotProductPureGoF32(x, y2)
	res[3] = dotProductPureGoF32(x, y3)
}

// FastExpF32 computes e^x for float32 using range reduction and a 5th‑degree Minimax polynomial.
func FastExpF32(x float32) float32 {
	if x < -88.0 {
		return 0.0
	}
	if x > 88.0 {
		return 3.4028235e+38 // float32 max
	}
	const invLn2 = 1.4426950408889634    // log2(e)
	const ln2Hi = 0.693145751953125      // musl ln2Hi (exact float32)
	const ln2Lo = 1.4286068203094172e-06 // musl ln2Lo
	k := float32(math.Round(float64(x * invLn2)))
	// Guard k to prevent overflow of exponent bits
	if k > 127 {
		k = 127
	} else if k < -126 {
		k = -126
	}
	r := x - k*ln2Hi - k*ln2Lo
	const (
		c2 = 0.5
		c3 = 0.16666667
		c4 = 0.041666668
		c5 = 0.008333334
	)
	poly := 1.0 + r*(1.0+r*(c2+r*(c3+r*(c4+r*c5))))
	ki := int32(k) + 127
	if ki <= 0 {
		return 0.0
	}
	if ki >= 255 {
		return 3.4028235e+38
	}
	twoK := math.Float32frombits(uint32(ki) << 23)
	return poly * twoK
}

// FastSigmoidF32 computes 1 / (1 + e^-x) with full float32 range matching ASM.
func FastSigmoidF32(x float32) float32 {
	return 1.0 / (1.0 + FastExpF32(-x))
}

// VecExpF32 computes elementwise e^x from src to dst.
func VecExpF32(dst, src []float32) {
	if !HasAVX2 || partialOverlap(dst, src) {
		vecExpPureGoF32(dst, src)
		return
	}
	vecExpASM_F32(dst, src)
}

func vecExpPureGoF32(dst, src []float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		dst[i] = FastExpF32(src[i])
	}
}

// VecSigmoidF32 computes elementwise 1 / (1 + e^-x) from src to dst.
func VecSigmoidF32(dst, src []float32) {
	if !HasAVX2 || partialOverlap(dst, src) {
		vecSigmoidPureGoF32(dst, src)
		return
	}
	vecSigmoidASM_F32(dst, src)
}

func vecSigmoidPureGoF32(dst, src []float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		dst[i] = FastSigmoidF32(src[i])
	}
}

// End of vector operations.

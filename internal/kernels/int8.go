package kernels

// INT8 weight kernels for quantized inference: weights are int8 with one scale per matrix, and
// activations are converted to float32 once per product so the inner loop can widen 8 weights
// at a time to float32 and accumulate with FMA.

// DotInt8F32 returns Σ x[i]·w[i] accumulated in float32 over the shorter of the two slices.
func DotInt8F32(x []float32, w []int8) float32 {
	if !HasAVX2 {
		return dotInt8PureGoF32(x, w)
	}
	return dotInt8ASM_F32(x, w)
}

func dotInt8PureGoF32(x []float32, w []int8) float32 {
	n := min(len(x), len(w))
	var s0, s1, s2, s3 float32
	j := 0
	for ; j <= n-4; j += 4 {
		s0 += x[j] * float32(w[j])
		s1 += x[j+1] * float32(w[j+1])
		s2 += x[j+2] * float32(w[j+2])
		s3 += x[j+3] * float32(w[j+3])
	}
	for ; j < n; j++ {
		s0 += x[j] * float32(w[j])
	}
	return (s0 + s1) + (s2 + s3)
}

// MatVecInt8 sets y[r] = bias[r] + scale·Σ_c w[r*stride+c]·x[c] for every r < len(y) and
// c < len(x), with W stored as int8 rows of stride `stride`. bias may be nil.
func MatVecInt8(y []float32, w []int8, stride int, scale float32, x, bias []float32) {
	cols := len(x)
	for r := range y {
		y[r] = DotInt8F32(x, w[r*stride:r*stride+cols]) * scale
		if bias != nil {
			y[r] += bias[r]
		}
	}
}

//go:build !amd64

package kernels

func dotProductASM_F32(x, y []float32) float32 {
	return dotProductPureGoF32(x, y)
}

func vecAddScaledASM_F32(y, x []float32, alpha float32) {
	vecAddScaledPureGoF32(y, x, alpha)
}

func dotProduct4ASM_F32(x, y0, y1, y2, y3, res []float32) {
	dotProduct4PureGoF32(x, y0, y1, y2, y3, res)
}

func vecExpASM_F32(dst, src []float32) {
	vecExpPureGoF32(dst, src)
}

func vecSigmoidASM_F32(dst, src []float32) {
	vecSigmoidPureGoF32(dst, src)
}

func dotInt8ASM_F32(x []float32, w []int8) float32 {
	return dotInt8PureGoF32(x, w)
}

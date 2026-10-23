//go:build amd64

package kernels

//go:noescape
func dotProductASM_F32(x, y []float32) float32

//go:noescape
func vecAddScaledASM_F32(y, x []float32, alpha float32)

//go:noescape
func dotProduct4ASM_F32(x, y0, y1, y2, y3, res []float32)

//go:noescape
func vecExpASM_F32(dst, src []float32)

//go:noescape
func vecSigmoidASM_F32(dst, src []float32)

//go:noescape
func dotInt8ASM_F32(x []float32, w []int8) float32

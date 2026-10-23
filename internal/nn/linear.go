package nn

import "github.com/mignon-ai/mignon-ai/internal/kernels"

// Dense layers over N rows. A weight matrix W is stored row-major with one row per output and
// row stride `stride` (stride > in leaves room for a trailing bias per row).

// Linear accumulates y (N x out) += x (N x in) · Wᵀ for a W packed as wT (MatrixPack.Fwd).
func Linear(N, in, out int, x []float32, wT *kernels.GemmPackedB, y []float32) {
	kernels.GemmPackedF32(N, x, in, 1, wT, y, out)
}

// LinearBackward accumulates dx (N x in) += dy · W for a W packed as w (MatrixPack.Bwd) and
// dW += dyᵀ · x for dy (N x out). The dW product packs x, which differs for every sequence, into *pack.
func LinearBackward(pack *[]float32, N, in, out int, x, dy []float32, w *kernels.GemmPackedB, dw []float32, stride int, dx []float32) {
	kernels.GemmPackedF32(N, dy, out, 1, w, dx, in)
	kernels.GemmF32(pack, out, in, N, dy, 1, out, x, in, 1, dw, stride)
}

// RMSNormRows applies RMSNorm with scales gamma to each of the N rows of x (N x D).
func RMSNormRows(N, D int, x, gamma, out, xHat, rms []float32) {
	for i := 0; i < N; i++ {
		r := i * D
		rms[i] = RMSNormForwardF32(x[r:r+D], gamma, out[r:r+D], xHat[r:r+D], RMSNormEpsilonF32)
	}
}

// RMSNormRowsBackward accumulates the RMSNorm gradient of the N rows into dx and dGamma.
func RMSNormRowsBackward(N, D int, dy, xHat, gamma, rms, dx, dGamma []float32) {
	for i := 0; i < N; i++ {
		r := i * D
		RMSNormBackwardF32(dy[r:r+D], xHat[r:r+D], gamma, rms[i], dx[r:r+D], dGamma)
	}
}

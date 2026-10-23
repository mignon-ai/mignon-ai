package ca

import "github.com/mignon-ai/mignon-ai/internal/kernels"

// GLUForward sets h[i] = value[i] · σ(gate[i]).
func GLUForward(h, value, gate []float32) {
	kernels.VecSigmoidF32(h, gate)
	for i := range h {
		h[i] *= value[i]
	}
}

// GLUBackward sets dValue[i] = dh[i] · σ(gate[i]) and dGate[i] = dh[i] · value[i] · σ'(gate[i]).
// dValue must not alias dh, value or gate: it holds σ(gate) while the gradients are computed.
func GLUBackward(dValue, dGate, dh, value, gate []float32) {
	kernels.VecSigmoidF32(dValue, gate)
	for i, sig := range dValue {
		dValue[i] = dh[i] * sig
		dGate[i] = dh[i] * value[i] * sig * (1 - sig)
	}
}

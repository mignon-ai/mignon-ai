package quant

import "math"

// Tensor stores weights as int8 with a float32 scale factor
type Tensor struct {
	Data  []int8  `json:"data"`
	Scale float32 `json:"scale"`
}

// NewTensorF32 converts a slice of float32 into symmetric int8 with a scale factor
func NewTensorF32(weights []float32) Tensor {
	if len(weights) == 0 {
		return Tensor{}
	}

	maxAbs := float32(0.0)
	for _, w := range weights {
		abs := float32(math.Abs(float64(w)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}

	if maxAbs == 0 {
		return Tensor{Data: make([]int8, len(weights)), Scale: 1.0}
	}

	scale := maxAbs / 127.0
	data := make([]int8, len(weights))
	for i, w := range weights {
		val := math.Round(float64(w / scale))
		if val > 127 {
			val = 127
		} else if val < -127 {
			val = -127
		}
		data[i] = int8(val)
	}

	return Tensor{Data: data, Scale: scale}
}

// DequantizeF32 converts an int8 tensor back to float32
func (qt Tensor) DequantizeF32() []float32 {
	out := make([]float32, len(qt.Data))
	for i, q := range qt.Data {
		out[i] = float32(q) * qt.Scale
	}
	return out
}

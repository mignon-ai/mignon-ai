package quant

import (
	"testing"
)

func TestQuantizeFloat32(t *testing.T) {
	weights := []float32{-1.5, -0.75, 0.0, 0.33, 1.25, 2.0}
	qt := NewTensorF32(weights)

	if len(qt.Data) != len(weights) {
		t.Fatalf("Length mismatch: expected %d, got %d", len(weights), len(qt.Data))
	}

	dequant := qt.DequantizeF32()
	for i := range weights {
		diff := float64(weights[i] - dequant[i])
		if diff < -0.05 || diff > 0.05 {
			t.Errorf("Quantization error too high at %d: orig=%.4f, dequant=%.4f", i, float64(weights[i]), float64(dequant[i]))
		}
	}
}

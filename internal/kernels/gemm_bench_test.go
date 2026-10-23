package kernels

import (
	"math/rand"
	"testing"
)

// Benchmarks compare the per-position kernels used by training with GemmF32 on medium-preset
// shapes (N positions, inDim = 3·D inputs, H hidden units per GLU branch).
const (
	gemmBenchN     = 128
	gemmBenchInDim = 384
	gemmBenchH     = 256
)

func gemmBenchData(sizes ...int) [][]float32 {
	rng := rand.New(rand.NewSource(1))
	out := make([][]float32, len(sizes))
	for i, n := range sizes {
		out[i] = make([]float32, n)
		for j := range out[i] {
			out[i][j] = float32(rng.NormFloat64())
		}
	}
	return out
}

func reportGFLOPS(b *testing.B, macs int) {
	b.ReportMetric(float64(2*macs)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
}

// Layer-1 forward for one branch: Z[N×H] = X[N×inDim] · Wᵀ, W stored H×(inDim+1) with a bias column.
func BenchmarkLayerForwardDot4(b *testing.B) {
	N, K, H := gemmBenchN, gemmBenchInDim, gemmBenchH
	d := gemmBenchData(N*K, H*(K+1), N*H)
	x, w, z := d[0], d[1], d[2]
	res := make([]float32, 4)
	for it := 0; it < b.N; it++ {
		for h := 0; h < H; h += 4 {
			w0, w1, w2, w3 := w[h*(K+1):h*(K+1)+K], w[(h+1)*(K+1):(h+1)*(K+1)+K], w[(h+2)*(K+1):(h+2)*(K+1)+K], w[(h+3)*(K+1):(h+3)*(K+1)+K]
			for i := 0; i < N; i++ {
				Dot4F32(x[i*K:i*K+K], w0, w1, w2, w3, res)
				copy(z[i*H+h:i*H+h+4], res)
			}
		}
	}
	reportGFLOPS(b, N*K*H)
}

func BenchmarkLayerForwardGemm(b *testing.B) {
	var pack []float32
	N, K, H := gemmBenchN, gemmBenchInDim, gemmBenchH
	d := gemmBenchData(N*K, H*(K+1), N*H)
	x, w, z := d[0], d[1], d[2]
	for it := 0; it < b.N; it++ {
		GemmF32(&pack, N, H, K, x, K, 1, w, 1, K+1, z, H)
	}
	reportGFLOPS(b, N*K*H)
}

func BenchmarkLayerInputGradGemm(b *testing.B) {
	var pack []float32
	N, K, H := gemmBenchN, gemmBenchInDim, gemmBenchH
	d := gemmBenchData(N*H, H*(K+1), N*K)
	dz, w, dx := d[0], d[1], d[2]
	for it := 0; it < b.N; it++ {
		GemmF32(&pack, N, K, H, dz, H, 1, w, K+1, 1, dx, K)
	}
	reportGFLOPS(b, N*K*H)
}

func BenchmarkLayerWeightGradGemm(b *testing.B) {
	var pack []float32
	N, K, H := gemmBenchN, gemmBenchInDim, gemmBenchH
	d := gemmBenchData(N*H, N*K, H*(K+1))
	dz, x, dw := d[0], d[1], d[2]
	for it := 0; it < b.N; it++ {
		GemmF32(&pack, H, K, N, dz, 1, H, x, K, 1, dw, K+1)
	}
	reportGFLOPS(b, N*K*H)
}

// gemmBenchSteps is the number of CA steps that reuse one packing of a block's weights (medium).
const gemmBenchSteps = 4

// Layer-1 forward over all steps of a block: W is packed once and reused by every step.
func BenchmarkLayerForwardStepsDot4(b *testing.B) {
	N, K, H := gemmBenchN, gemmBenchInDim, gemmBenchH
	d := gemmBenchData(N*K, H*(K+1), N*H)
	x, w, z := d[0], d[1], d[2]
	res := make([]float32, 4)
	for it := 0; it < b.N; it++ {
		for s := 0; s < gemmBenchSteps; s++ {
			for h := 0; h < H; h += 4 {
				w0, w1, w2, w3 := w[h*(K+1):h*(K+1)+K], w[(h+1)*(K+1):(h+1)*(K+1)+K], w[(h+2)*(K+1):(h+2)*(K+1)+K], w[(h+3)*(K+1):(h+3)*(K+1)+K]
				for i := 0; i < N; i++ {
					Dot4F32(x[i*K:i*K+K], w0, w1, w2, w3, res)
					copy(z[i*H+h:i*H+h+4], res)
				}
			}
		}
	}
	reportGFLOPS(b, gemmBenchSteps*N*K*H)
}

func BenchmarkLayerForwardStepsGemm(b *testing.B) {
	N, K, H := gemmBenchN, gemmBenchInDim, gemmBenchH
	d := gemmBenchData(N*K, H*(K+1), N*H)
	x, w, z := d[0], d[1], d[2]
	var buf []float32
	for it := 0; it < b.N; it++ {
		pb := packGemmB(buf, K, H, w, 1, K+1)
		buf = pb.data
		for s := 0; s < gemmBenchSteps; s++ {
			GemmPackedF32(N, x, K, 1, &pb, z, H)
		}
	}
	reportGFLOPS(b, gemmBenchSteps*N*K*H)
}

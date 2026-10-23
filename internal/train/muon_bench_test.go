package train

import (
	"encoding/json"
	"math/rand"
	"runtime"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/optim"
)

// BenchmarkMuonStepLarge measures one full optimizer step on the large preset's layout with
// every CPU as a worker, the cost paid once per batch on top of forward and backward.
func BenchmarkMuonStepLarge(b *testing.B) {
	cfg, err := ResolveConfig(json.RawMessage(`{"preset":"large"}`))
	if err != nil {
		b.Fatal(err)
	}
	m := cfg.BuildModel(cfg.Vocab, 1)
	layout := m.Layout()
	weights := append([]float32(nil), m.Weights()...)
	rng := rand.New(rand.NewSource(2))
	grads := make([]float64, layout.TotalParams)
	for i := range grads {
		grads[i] = rng.NormFloat64()
	}
	opt := optim.NewMuon(layout.OptimParams(), optim.DefaultMuonMomentum, optim.DefaultMuonNSSteps, optim.GradClipAdam)
	workers := runtime.NumCPU()
	b.ResetTimer()
	for b.Loop() {
		opt.Step(weights, grads, DefaultMuonLR, muonTestDecay, 1, workers)
	}
}

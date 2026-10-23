package train

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/optim"
)

const (
	muonTestSteps    = 3
	muonTestWorkers  = 8
	muonTestLR       = 0.02
	muonTestDecay    = 0.001
	muonTestEmbScale = 0.5
)

// muonTestSetup returns an untied two-block selective layout with random weights and gradients.
func muonTestSetup() (ca.Layout, []float32, []float64) {
	m := ca.New(ca.Config{VocabSize: 32, StateSize: 16, HiddenSize: 24, Blocks: ca.BuildBlockConfigs(2, 2, 2)})
	m.InitWeights(3)
	rng := rand.New(rand.NewSource(4))
	grads := make([]float64, m.Layout().TotalParams)
	for i := range grads {
		grads[i] = rng.NormFloat64()
	}
	return m.Layout(), append([]float32(nil), m.Weights()...), grads
}

// runMuon applies muonTestSteps updates with the given worker count and returns the weights.
func runMuon(layout ca.Layout, weights []float32, grads []float64, workers int, embedScale float64) []float32 {
	w := append([]float32(nil), weights...)
	opt := optim.NewMuon(layout.OptimParams(), optim.DefaultMuonMomentum, optim.DefaultMuonNSSteps, optim.GradClipAdam)
	opt.EmbedLRScale = embedScale
	for s := 0; s < muonTestSteps; s++ {
		opt.Step(w, grads, muonTestLR, muonTestDecay, 1, workers)
	}
	return w
}

func TestMuonStepIndependentOfWorkers(t *testing.T) {
	layout, weights, grads := muonTestSetup()
	serial := runMuon(layout, weights, grads, 1, optim.DefaultMuonEmbedLRScale)
	parallel := runMuon(layout, weights, grads, muonTestWorkers, optim.DefaultMuonEmbedLRScale)
	if !reflect.DeepEqual(serial, parallel) {
		t.Fatal("parallel Muon step differs from the serial one")
	}
}

func TestMuonEmbedLRScaleOnlyAffectsEmbeddings(t *testing.T) {
	layout, weights, grads := muonTestSetup()
	base := runMuon(layout, weights, grads, 1, optim.DefaultMuonEmbedLRScale)
	scaled := runMuon(layout, weights, grads, 1, muonTestEmbScale)
	embedEnd := layout.EmbedOffset + layout.VocabSize*layout.StateSize
	headEnd := layout.HeadOffset + layout.VocabSize*(layout.StateSize+1)
	changed := 0
	for p := range base {
		if base[p] == scaled[p] {
			continue
		}
		inEmbed := p >= layout.EmbedOffset && p < embedEnd
		inHead := p >= layout.HeadOffset && p < headEnd
		if !inEmbed && !inHead {
			t.Fatalf("param %d outside embeddings/head changed with EmbedLRScale", p)
		}
		changed++
	}
	if changed == 0 {
		t.Fatal("EmbedLRScale had no effect")
	}
}

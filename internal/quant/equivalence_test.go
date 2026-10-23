package quant

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

// quantEquivTolerance bounds the logit difference between the INT8 inference engine and the
// float32 engine running the same dequantized weights. Both see identical weights, so only
// float32-vs-float64 arithmetic differences remain.
const quantEquivTolerance = 1e-3

type quantEquivCase struct {
	name       string
	configs    []ca.BlockConfig
	maxContext int
	attnHeads  int
	attnWindow int
}

const (
	quantEquivVocab       = 16
	quantEquivState       = 8
	quantEquivHidden      = 12
	quantEquivWeightScale = 0.5
)

// quantEquivCases covers one and two blocks, context sizes and attention sublayers.
func quantEquivCases() []quantEquivCase {
	oneBlock := []ca.BlockConfig{{Steps: 3, DilationBase: 2, InitialDilation: 1}}
	twoBlocks := []ca.BlockConfig{
		{Steps: 2, DilationBase: 2, InitialDilation: 1},
		{Steps: 2, DilationBase: 2, InitialDilation: 2},
	}
	return []quantEquivCase{
		{"1blk", oneBlock, 64, 0, 0},
		{"2blk", twoBlocks, 64, 0, 0},
		{"2blk ctx=256", twoBlocks, 256, 0, 0},
		{"1blk attn", oneBlock, 64, 2, 0},
		{"2blk attn", twoBlocks, 64, 2, 0},
		{"2blk attn window", twoBlocks, 64, 4, 3},
	}
}

// quantEquivCaseNamed returns the case called name.
func quantEquivCaseNamed(t *testing.T, name string) quantEquivCase {
	t.Helper()
	for _, c := range quantEquivCases() {
		if c.name == name {
			return c
		}
	}
	t.Fatalf("no quantEquivCase named %q", name)
	return quantEquivCase{}
}

// buildQuantEquivModel returns a float model with seeded random weights for tc.
func buildQuantEquivModel(tc quantEquivCase) *ca.Model {
	rng := rand.New(rand.NewSource(3))
	model := ca.New(ca.Config{VocabSize: quantEquivVocab, StateSize: quantEquivState, HiddenSize: quantEquivHidden, Blocks: tc.configs, TiedEmbed: true, AttnHeads: tc.attnHeads, AttnWindow: tc.attnWindow})
	w := model.Weights()
	for i := range w {
		w[i] = float32((rng.Float64()*2 - 1) * quantEquivWeightScale)
	}
	model.SetWeights(w)
	return model
}

func TestQuantizedMatchesDequantized(t *testing.T) {
	cases := quantEquivCases()
	context := []int{2, 5, 7, 1, 9, 3, 11, 4, 0, 15}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := buildQuantEquivModel(tc)
			q := Quantize(model, tc.maxContext)
			deq := Dequantize(q)

			// Compare every prefix so each token position is exercised as the output position.
			for n := 1; n <= len(context); n++ {
				qOut := q.Forward(context[:n])
				dOut := deq.Forward(context[:n])
				maxDiff := 0.0
				for i := range qOut {
					maxDiff = math.Max(maxDiff, math.Abs(qOut[i]-dOut[i]))
				}
				if maxDiff > quantEquivTolerance {
					t.Fatalf("%s: prefix len %d max logit diff %.5f > %.0e", tc.name, n, maxDiff, quantEquivTolerance)
				}
			}
		})
	}
}

// TestSaveLoadKeepsAttention checks that a hybrid's attention sublayers survive an INT8 save/load
// round trip with identical logits and dequantize back to the same architecture.
func TestSaveLoadKeepsAttention(t *testing.T) {
	tc := quantEquivCase{"attn", []ca.BlockConfig{{Steps: 1, DilationBase: 2, InitialDilation: 1}}, 64, 2, 6}
	q := Quantize(buildQuantEquivModel(tc), tc.maxContext)
	tok := tokenizer.New()
	tok.Train("attention round trip", quantEquivVocab)
	path := filepath.Join(t.TempDir(), "q.json")
	if err := Save(q, tok, checkpoint.Metadata{}, path); err != nil {
		t.Fatal(err)
	}
	loaded, _, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg := Dequantize(loaded).Config(); cfg.AttnHeads != tc.attnHeads || cfg.AttnWindow != tc.attnWindow {
		t.Fatalf("Dequantize lost the attention config: heads %d window %d", cfg.AttnHeads, cfg.AttnWindow)
	}
	context := []int{2, 5, 7, 1, 9, 3, 11, 4}
	want := append([]float64(nil), q.Forward(context)...)
	got := loaded.Forward(context)
	for i := range want {
		if want[i] != got[i] {
			t.Fatalf("logit %d changed after save/load: %v != %v", i, got[i], want[i])
		}
	}
}

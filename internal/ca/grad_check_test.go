package ca

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/data"
)

// gradCheckCase describes one model configuration for finite-difference gradient checking.
type gradCheckCase struct {
	name        string
	configs     []BlockConfig
	weightScale float64 // weights drawn uniformly from [-weightScale, weightScale]
	untied      bool    // separate output head instead of the tied embedding table
	masked      bool    // mask some targets (as SFT does) so they contribute no loss
	attnHeads   int     // attention sublayer heads after each block; 0 = none
	attnWindow  int     // attention span; 0 = full causal
}

const (
	gradCheckVocab     = 8
	gradCheckState     = 8
	gradCheckHidden    = 6
	gradCheckMaxCtx    = 8
	gradCheckEps       = 1e-2 // finite-difference step (float32 forward needs a fairly large step)
	gradCheckAbsTol    = 1e-3 // ignore differences below float32 finite-difference noise
	gradCheckRelTol    = 0.05
	gradCheckParamStep = 1 // check every parameter
)

// checkGradients compares analytical gradients from ForwardBackwardDense against central
// finite differences of ForwardDenseLoss for every parameter. It returns the number of mismatches.
func checkGradients(t *testing.T, tc gradCheckCase, seed int64) int {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))

	layout := NewLayout(Config{VocabSize: gradCheckVocab, StateSize: gradCheckState, HiddenSize: gradCheckHidden, Blocks: tc.configs, TiedEmbed: !tc.untied, AttnHeads: tc.attnHeads, AttnWindow: tc.attnWindow})
	weights := make([]float32, layout.TotalParams)
	for i := range weights {
		weights[i] = float32((rng.Float64()*2 - 1) * tc.weightScale)
	}

	context := []int{1, 3, 2, 5, 4, 6}
	targets := []int{3, 2, 5, 4, 6, 7}
	if tc.masked {
		targets[1], targets[3] = data.MaskedTarget, data.MaskedTarget
	}

	ws := NewWorkspace(layout, gradCheckMaxCtx)
	ws.ForwardBackwardDense(context, targets, weights)
	analytical := append([]float32(nil), ws.paramGrads...)

	check := NewWorkspace(layout, gradCheckMaxCtx)
	mismatches := 0
	for p := 0; p < layout.TotalParams; p += gradCheckParamStep {
		orig := weights[p]
		weights[p] = orig + gradCheckEps
		lossPlus := check.ForwardDenseLoss(context, targets, weights)
		weights[p] = orig - gradCheckEps
		lossMinus := check.ForwardDenseLoss(context, targets, weights)
		weights[p] = orig

		numerical := (lossPlus - lossMinus) / (2 * gradCheckEps)
		a := float64(analytical[p])
		diff := math.Abs(a - numerical)
		rel := diff / math.Max(math.Abs(a)+math.Abs(numerical), 1e-12)
		if diff > gradCheckAbsTol && rel > gradCheckRelTol {
			if mismatches < 5 {
				t.Errorf("%s: param %d analytical=%.6f numerical=%.6f", tc.name, p, a, numerical)
			}
			mismatches++
		}
	}
	return mismatches
}

func TestGradientCheck(t *testing.T) {
	multiStep := []BlockConfig{{Steps: 3, DilationBase: 2, InitialDilation: 1}}
	twoBlocks := []BlockConfig{
		{Steps: 2, DilationBase: 2, InitialDilation: 1},
		{Steps: 2, DilationBase: 2, InitialDilation: 2},
	}

	cases := []gradCheckCase{
		{name: "2blk", configs: twoBlocks, weightScale: 0.8},
		{name: "untied head", configs: twoBlocks, weightScale: 0.8, untied: true},
		{name: "attn 2blk", configs: twoBlocks, weightScale: 0.8, attnHeads: 2},
		{name: "attn window", configs: twoBlocks, weightScale: 0.8, attnHeads: 4, attnWindow: 3},
		{name: "masked targets", configs: multiStep, weightScale: 0.8, masked: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if n := checkGradients(t, tc, 7); n > 0 {
				t.Errorf("%s: %d gradient mismatches", tc.name, n)
			}
		})
	}
}

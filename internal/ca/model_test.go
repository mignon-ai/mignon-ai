package ca

import (
	"math"
	"math/rand"
	"testing"
)

func benchmarkMultiBlockDense(b *testing.B, N int) {
	vocabSize := 256
	stateSize := 32
	hiddenSize := 48
	maxContext := N

	configs := DefaultTwoBlockConfig()
	layout := NewLayout(Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	ws := NewWorkspace(layout, maxContext)

	weights := make([]float32, layout.TotalParams)
	for i := range weights {
		weights[i] = float32(rand.Float64()*0.1 - 0.05)
	}

	context := make([]int, N)
	targets := make([]int, N)
	for i := 0; i < N; i++ {
		context[i] = i % vocabSize
		targets[i] = (i + 1) % vocabSize
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = ws.ForwardBackwardDense(context, targets, weights)
	}
}

func TestMultiBlockDenseGradientCheck(t *testing.T) {
	rand.Seed(123)
	vocabSize := 8
	stateSize := 6
	hiddenSize := 10
	maxContext := 10

	configs := []BlockConfig{
		{Steps: 2, DilationBase: 2, InitialDilation: 1},
		{Steps: 2, DilationBase: 2, InitialDilation: 2},
	}

	layout := NewLayout(Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	ws := NewWorkspace(layout, maxContext)

	weights := make([]float32, layout.TotalParams)
	for i := range weights {
		weights[i] = float32(rand.Float64()*0.4 - 0.2)
	}

	context := []int{1, 3, 2, 5}
	targets := []int{3, 2, 5, 0}
	eps := float32(1e-3)

	// 1. Analytical gradients via ForwardBackwardDense
	_ = ws.ForwardBackwardDense(context, targets, weights)
	analyticalGrads := make([]float32, layout.TotalParams)
	copy(analyticalGrads, ws.paramGrads)

	// 2. Numerical gradients for sampled parameters
	wsCheck := NewWorkspace(layout, maxContext)
	for pIdx := 0; pIdx < layout.TotalParams; pIdx += 9 {
		orig := weights[pIdx]

		weights[pIdx] = orig + eps
		lossPlus := wsCheck.ForwardBackwardDense(context, targets, weights)

		weights[pIdx] = orig - eps
		lossMinus := wsCheck.ForwardBackwardDense(context, targets, weights)

		weights[pIdx] = orig

		numericalGrad := (lossPlus - lossMinus) / (2.0 * float64(eps))
		analyticalGrad := float64(analyticalGrads[pIdx])

		diff := math.Abs(analyticalGrad - numericalGrad)
		denom := math.Max(math.Abs(analyticalGrad)+math.Abs(numericalGrad), 1e-3)
		relDiff := diff / denom

		if relDiff > 0.05 && diff > 1e-4 {
			t.Errorf("MultiBlock Dense Gradient mismatch at param %d: analytical=%e, numerical=%e (diff=%e, relDiff=%e)",
				pIdx, analyticalGrad, numericalGrad, diff, relDiff)
		}
	}
}

func TestMultiBlockSelectiveLatchGradientCheck(t *testing.T) {
	rand.Seed(999)
	vocabSize := 8
	stateSize := 6
	hiddenSize := 10
	maxContext := 10

	configs := []BlockConfig{
		{Steps: 2, DilationBase: 2, InitialDilation: 1},
		{Steps: 2, DilationBase: 2, InitialDilation: 2},
	}

	layout := NewLayout(Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	ws := NewWorkspace(layout, maxContext)

	weights := make([]float32, layout.TotalParams)
	for i := range weights {
		weights[i] = float32(rand.Float64()*0.4 - 0.2)
	}

	context := []int{1, 3, 2, 5}
	targets := []int{3, 2, 5, 0}
	eps := float32(1e-3)

	// 1. Analytical gradients via ForwardBackwardDense with selective latch
	_ = ws.ForwardBackwardDense(context, targets, weights)
	analyticalGrads := make([]float32, layout.TotalParams)
	copy(analyticalGrads, ws.paramGrads)

	// 2. Numerical gradients for sampled parameters (including W_select)
	wsCheck := NewWorkspace(layout, maxContext)
	for pIdx := 0; pIdx < layout.TotalParams; pIdx += 7 {
		orig := weights[pIdx]

		weights[pIdx] = orig + eps
		lossPlus := wsCheck.ForwardBackwardDense(context, targets, weights)

		weights[pIdx] = orig - eps
		lossMinus := wsCheck.ForwardBackwardDense(context, targets, weights)

		weights[pIdx] = orig

		numericalGrad := (lossPlus - lossMinus) / (2.0 * float64(eps))
		analyticalGrad := float64(analyticalGrads[pIdx])

		diff := math.Abs(analyticalGrad - numericalGrad)
		denom := math.Max(math.Abs(analyticalGrad)+math.Abs(numericalGrad), 1e-3)
		relDiff := diff / denom

		if relDiff > 0.05 && diff > 1e-4 {
			t.Errorf("MultiBlock Selective Latch Gradient mismatch at param %d: analytical=%e, numerical=%e (diff=%e, relDiff=%e)",
				pIdx, analyticalGrad, numericalGrad, diff, relDiff)
		}
	}

	// 3. Specifically verify W_select parameters directly
	for _, bl := range layout.Blocks {
		if bl.SelectiveOffset >= 0 {
			selCount := stateSize * stateSize
			for idx := 0; idx < selCount; idx++ {
				pIdx := bl.SelectiveOffset + idx
				orig := weights[pIdx]

				weights[pIdx] = orig + eps
				lossPlus := wsCheck.ForwardBackwardDense(context, targets, weights)

				weights[pIdx] = orig - eps
				lossMinus := wsCheck.ForwardBackwardDense(context, targets, weights)

				weights[pIdx] = orig

				numericalGrad := (lossPlus - lossMinus) / (2.0 * float64(eps))
				analyticalGrad := float64(analyticalGrads[pIdx])

				diff := math.Abs(analyticalGrad - numericalGrad)
				denom := math.Max(math.Abs(analyticalGrad)+math.Abs(numericalGrad), 1e-3)
				relDiff := diff / denom

				if relDiff > 0.05 && diff > 1e-4 {
					t.Errorf("MultiBlock Selective Latch W_select Gradient mismatch at param %d (sel %d): analytical=%e, numerical=%e (diff=%e, relDiff=%e)",
						pIdx, idx, analyticalGrad, numericalGrad, diff, relDiff)
				}
			}
		}
	}
}

func BenchmarkMultiBlockDenseN32(b *testing.B)  { benchmarkMultiBlockDense(b, 32) }
func BenchmarkMultiBlockDenseN64(b *testing.B)  { benchmarkMultiBlockDense(b, 64) }
func BenchmarkMultiBlockDenseN128(b *testing.B) { benchmarkMultiBlockDense(b, 128) }
func BenchmarkMultiBlockDenseN256(b *testing.B) { benchmarkMultiBlockDense(b, 256) }

func BenchmarkMultiBlockDense600K_N128(b *testing.B) {
	vocabSize := 2048
	stateSize := 128
	hiddenSize := 256
	N := 128

	configs := DefaultTwoBlockConfig()
	layout := NewLayout(Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	ws := NewWorkspace(layout, N)

	weights := make([]float32, layout.TotalParams)
	for i := range weights {
		weights[i] = float32(rand.Float64()*0.1 - 0.05)
	}

	context := make([]int, N)
	targets := make([]int, N)
	for i := 0; i < N; i++ {
		context[i] = i % vocabSize
		targets[i] = (i + 1) % vocabSize
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = ws.ForwardBackwardDense(context, targets, weights)
	}
}

func BenchmarkMultiBlockDense1M_N128(b *testing.B) {
	vocabSize := 2048
	stateSize := 192
	hiddenSize := 384
	N := 128

	configs := DefaultTwoBlockConfig()
	layout := NewLayout(Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	ws := NewWorkspace(layout, N)

	weights := make([]float32, layout.TotalParams)
	for i := range weights {
		weights[i] = float32(rand.Float64()*0.1 - 0.05)
	}

	context := make([]int, N)
	targets := make([]int, N)
	for i := 0; i < N; i++ {
		context[i] = i % vocabSize
		targets[i] = (i + 1) % vocabSize
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = ws.ForwardBackwardDense(context, targets, weights)
	}
}

func TestMultiBlockMaskedLoss(t *testing.T) {
	rand.Seed(1337)
	vocabSize := 16
	stateSize := 8
	hiddenSize := 12
	configs := []BlockConfig{
		{Steps: 2, DilationBase: 2, InitialDilation: 1},
	}

	model := New(Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	layout := NewLayout(Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	ws := NewWorkspace(layout, 16)

	weights := make([]float32, model.NumParams())
	copy(weights, model.Weights())

	context := []int{1, 3, 5, 7}
	allMasked := []int{-1, -1, -1, -1}
	ws.ResetGrads()
	loss := ws.ForwardBackwardDense(context, allMasked, weights)
	if loss != 0.0 {
		t.Fatalf("Expected zero loss for all-masked targets, got %f", loss)
	}
	for i, g := range ws.paramGrads {
		if g != 0.0 {
			t.Fatalf("Expected zero gradient for all-masked targets, got param %d = %f", i, g)
		}
	}

	// Masked prefix [prompt, prompt, bot, response]
	partiallyMasked := []int{-1, -1, 4, 6}
	ws.ResetGrads()
	loss2 := ws.ForwardBackwardDense(context, partiallyMasked, weights)
	if loss2 <= 0.0 {
		t.Fatalf("Expected positive loss for partially-masked targets, got %f", loss2)
	}
	hasGrad := false
	for _, g := range ws.paramGrads {
		if math.Abs(float64(g)) > 1e-6 {
			hasGrad = true
			break
		}
	}
	if !hasGrad {
		t.Fatalf("Expected non-zero gradients for valid targets")
	}
}

func TestForwardDenseLossEquivalence(t *testing.T) {
	rand.Seed(42)
	vocabSize := 32
	stateSize := 16
	hiddenSize := 24
	maxContext := 16
	configs := []BlockConfig{
		{Steps: 3, DilationBase: 2, InitialDilation: 1},
		{Steps: 2, DilationBase: 2, InitialDilation: 2},
	}

	layout := NewLayout(Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	ws1 := NewWorkspace(layout, maxContext)
	ws2 := NewWorkspace(layout, maxContext)

	weights := make([]float32, layout.TotalParams)
	for i := range weights {
		weights[i] = float32(rand.Float64()*0.4 - 0.2)
	}

	ctx := []int{2, 5, 8, 11, 14, 17}
	tgt := []int{5, 8, 11, 14, 17, 20}

	loss1 := ws1.ForwardBackwardDense(ctx, tgt, weights)
	loss2 := ws2.ForwardDenseLoss(ctx, tgt, weights)

	diff := math.Abs(loss1 - loss2)
	if diff > 1e-6 {
		t.Fatalf("ForwardDenseLoss mismatch: ForwardBackwardDense=%.14f vs ForwardDenseLoss=%.14f (diff=%.2e)",
			loss1, loss2, diff)
	}
}

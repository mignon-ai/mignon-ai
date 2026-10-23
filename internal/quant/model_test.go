package quant

import (
	"math"
	"math/rand"
	"os"
	"sync"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/ca"
	"github.com/mignon-ai/mignon-ai/internal/checkpoint"
)

func TestQuantizeDequantizeSelectiveLatch(t *testing.T) {
	rand.Seed(42)
	vocabSize := 32
	stateSize := 16
	hiddenSize := 24
	configs := []ca.BlockConfig{
		{Steps: 3, DilationBase: 2, InitialDilation: 1},
		{Steps: 2, DilationBase: 2, InitialDilation: 2},
	}

	origModel := ca.New(ca.Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	context := []int{2, 5, 8, 14}

	// 1. Quantize
	q := Quantize(origModel, 64)
	qLogits := q.Call(context)

	// 2. Dequantize
	dequantModel := Dequantize(q)
	dequantLogits := dequantModel.Forward(context)

	if len(dequantLogits) != len(qLogits) {
		t.Fatalf("Logit length mismatch: dequant=%d vs quant=%d", len(dequantLogits), len(qLogits))
	}

	maxDiff := 0.0
	for i := range qLogits {
		diff := math.Abs(dequantLogits[i] - qLogits[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	if maxDiff > 5e-3 {
		t.Fatalf("Dequantized selective latch model forward mismatch against quantized forward: maxDiff=%e", maxDiff)
	}
}

func TestQuantizedMultiBlockSaveLoadAndInference(t *testing.T) {
	vocabSize := 32
	stateSize := 16
	hiddenSize := 24
	maxContext := 32

	configs := ca.DefaultTwoBlockConfig()
	model := ca.New(ca.Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})

	// Quantize to INT8
	qModel := Quantize(model, maxContext)

	// Verify forward pass
	context := []int{2, 7, 14, 3}
	logits1 := qModel.Call(context)
	if len(logits1) != vocabSize {
		t.Fatalf("Expected %d logits, got %d", vocabSize, len(logits1))
	}
	for i, l := range logits1 {
		if math.IsNaN(l) || math.IsInf(l, 0) {
			t.Fatalf("Logit %d is NaN/Inf: %f", i, l)
		}
	}

	// Save and reload
	tmpFile := "test_mb_quant.json"
	meta := checkpoint.Metadata{
		StateSize:   stateSize,
		HiddenSize:  hiddenSize,
		ContextSize: maxContext,
	}
	err := Save(qModel, nil, meta, tmpFile)
	if err != nil {
		t.Fatalf("Error saving quantized multi-block model: %v", err)
	}
	defer os.Remove(tmpFile)

	loadedQ, _, loadedMeta, err := Load(tmpFile)
	if err != nil {
		t.Fatalf("Error loading quantized multi-block model: %v", err)
	}
	if loadedMeta.StateSize != stateSize || loadedMeta.HiddenSize != hiddenSize {
		t.Fatalf("Loaded metadata mismatch: %+v", loadedMeta)
	}

	// Verify loaded model inference matches original quantized model
	logits2 := loadedQ.Call(context)
	if len(logits2) != vocabSize {
		t.Fatalf("Expected %d logits, got %d", vocabSize, len(logits2))
	}
	for v := 0; v < vocabSize; v++ {
		if math.Abs(logits1[v]-logits2[v]) > 1e-10 {
			t.Errorf("Logit mismatch after reload at %d: orig=%f, loaded=%f", v, logits1[v], logits2[v])
		}
	}
}

func TestDequantizeMultiBlockCAModel(t *testing.T) {
	rand.Seed(42)
	vocabSize := 32
	stateSize := 16
	hiddenSize := 24
	configs := []ca.BlockConfig{
		{Steps: 3, DilationBase: 2, InitialDilation: 1},
		{Steps: 2, DilationBase: 2, InitialDilation: 2},
	}

	origModel := ca.New(ca.Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	context := []int{2, 5, 8, 14}

	// 1. Quantize
	q := Quantize(origModel, 64)
	qLogits := q.Call(context)

	// 2. Dequantize
	dequantModel := Dequantize(q)
	dequantLogits := dequantModel.Forward(context)

	if len(dequantLogits) != len(qLogits) {
		t.Fatalf("Logit length mismatch: dequant=%d vs quant=%d", len(dequantLogits), len(qLogits))
	}

	maxDiff := 0.0
	for i := range qLogits {
		diff := math.Abs(dequantLogits[i] - qLogits[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	// Because Dequantize uses float32 weights while Quantized Call runs in float64,
	// their forward passes across multiple CA steps match within float32 precision (~1e-3).
	if maxDiff > 5e-3 {
		t.Fatalf("Dequantized model forward mismatch against quantized forward: maxDiff=%e", maxDiff)
	}
}

func TestConcurrentInferenceThreadSafety(t *testing.T) {
	vocabSize := 32
	stateSize := 16
	hiddenSize := 24
	configs := []ca.BlockConfig{
		{Steps: 2, DilationBase: 2, InitialDilation: 1},
	}
	model := ca.New(ca.Config{VocabSize: vocabSize, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: configs, TiedEmbed: true})
	qModel := Quantize(model, 64)

	var wg sync.WaitGroup
	numRoutines := 16

	for r := 0; r < numRoutines; r++ {
		wg.Add(2)
		go func(id int) {
			defer wg.Done()
			ctx := []int{id % vocabSize, (id + 1) % vocabSize, (id + 2) % vocabSize}
			logits := model.Forward(ctx)
			if len(logits) != vocabSize {
				t.Errorf("ca.Model.Forward returned invalid length %d", len(logits))
			}
		}(r)

		go func(id int) {
			defer wg.Done()
			ctx := []int{id % vocabSize, (id + 1) % vocabSize, (id + 2) % vocabSize}
			logits := qModel.Call(ctx)
			if len(logits) != vocabSize {
				t.Errorf("Model.Call returned invalid length %d", len(logits))
			}
		}(r)
	}
	wg.Wait()
}

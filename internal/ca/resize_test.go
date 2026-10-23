package ca

import (
	"testing"
)

func TestMultiBlockResizeVocab(t *testing.T) {
	initialVocab := 20
	newVocab := 35
	stateSize := 16
	hiddenSize := 32

	model := New(Config{VocabSize: initialVocab, StateSize: stateSize, HiddenSize: hiddenSize, Blocks: BuildBlockConfigs(1, 3, 2), TiedEmbed: true})
	if model.VocabSize() != initialVocab {
		t.Fatalf("Expected initial vocab size %d, got %d", initialVocab, model.VocabSize())
	}

	input := []int{2, 5, 8}
	initialLogits := model.Forward(input)
	if len(initialLogits) != initialVocab {
		t.Fatalf("Expected %d initial logits, got %d", initialVocab, len(initialLogits))
	}

	// Resize vocabulary to newVocab
	model.ResizeVocab(newVocab)
	if model.VocabSize() != newVocab {
		t.Fatalf("Expected resized vocab size %d, got %d", newVocab, model.VocabSize())
	}

	resizedLogits := model.Forward(input)
	if len(resizedLogits) != newVocab {
		t.Fatalf("Expected %d resized logits, got %d", newVocab, len(resizedLogits))
	}

	// Verify that existing token logits are preserved exactly
	for i := 0; i < initialVocab; i++ {
		diff := initialLogits[i] - resizedLogits[i]
		if diff < -1e-5 || diff > 1e-5 {
			t.Errorf("Logit %d shifted after ResizeVocab: before=%f, after=%f", i, initialLogits[i], resizedLogits[i])
		}
	}
}

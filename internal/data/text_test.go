package data

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

func TestPrepareBPEMultiFileDataDenseWithShortDocsAndEOS(t *testing.T) {
	docs := []string{
		"Short sentence.", // very short (< 64 tokens)
		"Another quick phrase for testing document boundaries.",
		"A longer document that contains many words and phrases so that it spans beyond sixty-four tokens easily when tokenized by a character-level or subword BPE tokenizer. Here are more words to fill up the context length nicely and cleanly.",
	}

	contextSize := 16
	targetVocab := 40
	stride := 8
	maxSeqs := 100

	trainSet, _, tok := PrepareTrainVal(docs, contextSize, targetVocab, stride, maxSeqs, 0)
	X, YDense := trainSet.X, trainSet.Y
	if tok == nil {
		t.Fatalf("Expected non-nil tokenizer")
	}
	if len(X) == 0 {
		t.Fatalf("Expected non-empty X dataset, but got 0 sequences")
	}
	if len(X) != len(YDense) {
		t.Fatalf("Mismatch between X (%d) and YDense (%d)", len(X), len(YDense))
	}

	eosID := tok.TokenToID[tokenizer.EOSToken]
	padID := tok.TokenToID[tokenizer.PadToken]

	// Verify that EOS token is present in the targets, PAD in input, and MaskedTarget in targets
	hasEOS := false
	hasPAD := false
	hasMask := false
	for _, xSeq := range X {
		for _, tokID := range xSeq {
			if tokID == padID {
				hasPAD = true
			}
		}
	}
	for _, ySeq := range YDense {
		for _, tokID := range ySeq {
			if tokID == eosID {
				hasEOS = true
			}
			if tokID == MaskedTarget {
				hasMask = true
			}
		}
	}

	if !hasEOS {
		t.Errorf("Expected EOS token (%d) to appear in targets", eosID)
	}
	if !hasPAD {
		t.Errorf("Expected PAD token (%d) to appear in input sequences", padID)
	}
	if !hasMask {
		t.Errorf("Expected MaskedTarget (%d) to appear for padded targets", MaskedTarget)
	}
}

// TestEncodeOrderIsDeterministic guards the cache contract: encoding the same corpus twice must give
// the same sequences in the same order, so cached and fresh datasets train identically.
func TestEncodeOrderIsDeterministic(t *testing.T) {
	const docCount, contextSize, targetVocab, stride = 8, 8, 60, 4
	var docs []string
	for i := range docCount {
		docs = append(docs, strings.Repeat(fmt.Sprintf("document %d says the automaton updates its cells. ", i), docCount))
	}
	tok := tokenizer.New()
	tok.Train(SampleForTokenizer(docs), targetVocab)

	x1, y1 := Encode(docs, contextSize, tok, stride, 0)
	x2, y2 := Encode(docs, contextSize, tok, stride, 0)
	if len(x1) < docCount {
		t.Fatalf("expected at least %d sequences to make the order meaningful, got %d", docCount, len(x1))
	}
	if !reflect.DeepEqual(x1, x2) || !reflect.DeepEqual(y1, y2) {
		t.Fatalf("Encode returned a different sequence order on identical input")
	}
}

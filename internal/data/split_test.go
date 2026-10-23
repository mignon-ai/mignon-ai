package data

import (
	"fmt"
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

func TestSplitCorpusForValMultipleDocs(t *testing.T) {
	docs := make([]string, 10)
	for i := 0; i < 10; i++ {
		docs[i] = fmt.Sprintf("Document number %d with some content to train.", i)
	}

	trainDocs, valDocs := SplitForVal(docs, 0.20)
	if len(trainDocs) != 8 {
		t.Fatalf("Expected 8 train docs, got %d", len(trainDocs))
	}
	if len(valDocs) != 2 {
		t.Fatalf("Expected 2 val docs, got %d", len(valDocs))
	}

	// Verify docs are completely disjoint
	trainSet := make(map[string]bool)
	for _, d := range trainDocs {
		trainSet[d] = true
	}
	for _, d := range valDocs {
		if trainSet[d] {
			t.Errorf("Validation document leaked into train set: %s", d)
		}
	}
}

func TestSplitCorpusForValBlocks(t *testing.T) {
	// Single document with 10 distinct paragraphs / blocks
	var blocks []string
	for i := 0; i < 10; i++ {
		blocks = append(blocks, fmt.Sprintf("Story block %d: The quick brown fox jumps over the lazy dog.", i))
	}
	doc := strings.Join(blocks, "\n\n")

	trainDocs, valDocs := SplitForVal([]string{doc}, 0.20)
	if len(trainDocs) != 1 || len(valDocs) != 1 {
		t.Fatalf("Expected 1 trainDoc and 1 valDoc, got %d and %d", len(trainDocs), len(valDocs))
	}

	trainContent := trainDocs[0]
	valContent := valDocs[0]

	// Train should contain blocks 0..7, val should contain blocks 8..9
	for i := 0; i < 8; i++ {
		target := fmt.Sprintf("Story block %d:", i)
		if !strings.Contains(trainContent, target) {
			t.Errorf("Expected train to contain '%s'", target)
		}
		if strings.Contains(valContent, target) {
			t.Errorf("Val should not contain '%s'", target)
		}
	}
	for i := 8; i < 10; i++ {
		target := fmt.Sprintf("Story block %d:", i)
		if !strings.Contains(valContent, target) {
			t.Errorf("Expected val to contain '%s'", target)
		}
		if strings.Contains(trainContent, target) {
			t.Errorf("Train should not contain '%s'", target)
		}
	}
}

func TestSplitCorpusForValContinuousStream(t *testing.T) {
	// Continuous single-block text
	text := "Sentence number one is here. Sentence number two is right here. Sentence number three follows up. Sentence number four is at the end."
	trainDocs, valDocs := SplitForVal([]string{text}, 0.25)
	if len(trainDocs) != 1 || len(valDocs) != 1 {
		t.Fatalf("Expected 1 trainDoc and 1 valDoc, got %d and %d", len(trainDocs), len(valDocs))
	}
	if len(trainDocs[0]) == 0 || len(valDocs[0]) == 0 {
		t.Fatalf("Train or Val doc is empty")
	}
	if strings.Contains(trainDocs[0], valDocs[0]) {
		t.Errorf("Train doc should not contain entire val doc")
	}
}

func TestNoValidationLeakageAcrossOverlappingWindows(t *testing.T) {
	// Create a document with multiple blocks and use stride = contextSize / 2
	var blocks []string
	for i := 0; i < 20; i++ {
		var words []string
		for w := 0; w < 15; w++ {
			words = append(words, fmt.Sprintf("tok%02d%02d", i, w))
		}
		blocks = append(blocks, strings.Join(words, " "))
	}
	doc := strings.Join(blocks, "\n\n")

	contextSize := 16
	stride := 8 // 50% overlap!
	valFraction := 0.20

	tok := tokenizer.New()
	tok.Train(doc, 80)

	trainSet, valSet := EncodeTrainVal([]string{doc}, contextSize, tok, stride, 0, valFraction)
	trainX, valX := trainSet.X, valSet.X
	if len(trainX) == 0 {
		t.Fatalf("Expected non-empty trainX")
	}
	if len(valX) == 0 {
		t.Fatalf("Expected non-empty valX")
	}
	if !trainSet.Valid() || !valSet.Valid() {
		t.Fatalf("inputs and targets length mismatch")
	}

	// Helper to extract non-pad tokens
	padID := tok.TokenToID[tokenizer.PadToken]
	cleanSeq := func(seq []int) []int {
		var out []int
		for _, id := range seq {
			if id != padID {
				out = append(out, id)
			}
		}
		return out
	}

	// Check that NO window in valX shares an overlapping stride window (length >= contextSize - stride)
	// with ANY window in trainX!
	// In the old implementation with sequence permutation, val windows shared `contextSize - stride = 8`
	// tokens with train windows.
	minOverlapLen := contextSize - stride
	for vIdx, vSeq := range valX {
		vTokens := cleanSeq(vSeq)
		if len(vTokens) < minOverlapLen {
			continue
		}
		for tIdx, tSeq := range trainX {
			tTokens := cleanSeq(tSeq)
			if len(tTokens) < minOverlapLen {
				continue
			}

			// Check for any matching sub-sequence of length minOverlapLen
			for vStart := 0; vStart <= len(vTokens)-minOverlapLen; vStart++ {
				subV := vTokens[vStart : vStart+minOverlapLen]
				for tStart := 0; tStart <= len(tTokens)-minOverlapLen; tStart++ {
					subT := tTokens[tStart : tStart+minOverlapLen]
					match := true
					for k := 0; k < minOverlapLen; k++ {
						if subV[k] != subT[k] {
							match = false
							break
						}
					}
					if match {
						t.Fatalf("Validation leakage detected! valX[%d] and trainX[%d] share sub-sequence %v of length %d",
							vIdx, tIdx, subV, minOverlapLen)
					}
				}
			}
		}
	}
}

func TestEncodePairsTrainValDataDense(t *testing.T) {
	tok := tokenizer.New()
	tok.Train("instruction response prompt command test", 50)

	examples := make([]QAPair, 25)
	for i := 0; i < 25; i++ {
		examples[i] = QAPair{
			Instruction: fmt.Sprintf("test command %d", i),
			Output:      fmt.Sprintf("response %d", i),
		}
	}

	trainSet, valSet := EncodePairsTrainVal(examples, 16, tok, 0.20)
	trainX, valX := trainSet.X, valSet.X
	if len(trainX) == 0 || len(valX) == 0 {
		t.Fatalf("Expected non-empty trainX and valX, got %d and %d", len(trainX), len(valX))
	}
	if len(trainX) != 20 {
		t.Errorf("Expected 20 train sequences, got %d", len(trainX))
	}
	if len(valX) != 5 {
		t.Errorf("Expected 5 val sequences, got %d", len(valX))
	}
	if !trainSet.Valid() || !valSet.Valid() {
		t.Errorf("Sequence length mismatch")
	}
}

func TestSplitCorpusForValDropsBlocksSeenInTrain(t *testing.T) {
	shared := "This block was repeated and landed in both splits."
	docs := make([]string, 10)
	for i := range docs {
		docs[i] = fmt.Sprintf("Unique block %d.\n\n%s", i, shared)
	}
	_, valDocs := SplitForVal(docs, 0.20)
	if len(valDocs) != 2 {
		t.Fatalf("Expected 2 val docs, got %d", len(valDocs))
	}
	for _, d := range valDocs {
		if strings.Contains(d, shared) {
			t.Errorf("block seen in training was kept in validation: %q", d)
		}
		if !strings.Contains(d, "Unique block") {
			t.Errorf("unseen block was dropped from validation: %q", d)
		}
	}
}

func TestDropBlocksSeenInTrainKeepsCleanSplitUnchanged(t *testing.T) {
	val := []string{"a b c\n\nd e f"}
	got := dropBlocksSeenInTrain([]string{"x y z"}, val)
	if len(got) != 1 || got[0] != val[0] {
		t.Errorf("clean validation docs must be returned unchanged, got %q", got)
	}
}

func TestSequencesCapAndValid(t *testing.T) {
	s := Sequences{X: [][]int{{1}, {2}, {3}}, Y: [][]int{{2}, {3}, {4}}}
	if s.Len() != 3 || !s.Valid() {
		t.Fatalf("Len/Valid = %d/%v", s.Len(), s.Valid())
	}
	if got := s.Cap(2); got.Len() != 2 || !got.Valid() || got.Y[1][0] != 3 {
		t.Errorf("Cap(2) = %+v", got)
	}
	if s.Cap(0).Len() != 3 || s.Cap(10).Len() != 3 {
		t.Error("Cap with 0 or a larger limit must keep every sequence")
	}
	if (Sequences{X: [][]int{{1}}}).Valid() {
		t.Error("inputs without targets must not be valid")
	}
}

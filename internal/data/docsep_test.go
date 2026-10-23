package data

import (
	"reflect"
	"strings"
	"testing"

	"github.com/mignon-ai/mignon-ai/internal/tokenizer"
)

func docSepTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok := tokenizer.New()
	tok.Train("alpha beta gamma delta\n\nalpha beta gamma delta\n", 40)
	return tok
}

func concatIDs(parts ...[]int) []int {
	var out []int
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func TestEncodeBlocksEndsEveryDocumentWithEOS(t *testing.T) {
	tok := docSepTokenizer(t)
	eos := tok.EOSTokenID()
	blocks := splitBlocks("alpha\n\nbeta\n\n" + DocSeparator + "\n\ngamma\n\n" + DocSeparator + "\n\ndelta")

	got := encodeBlocks(blocks, tok)
	want := [][]int{
		tok.Encode("alpha\n\n"),
		concatIDs(tok.Encode("beta\n"), []int{eos}),
		concatIDs(tok.Encode("gamma\n"), []int{eos}),
		concatIDs(tok.Encode("delta\n"), []int{eos}),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("encodeBlocks = %v, want %v", got, want)
	}
}

func TestEncodeBlocksWithoutSeparatorIsOneDocument(t *testing.T) {
	tok := docSepTokenizer(t)
	got := encodeBlocks(splitBlocks("alpha\n\nbeta\n\ngamma"), tok)
	want := [][]int{
		tok.Encode("alpha\n\n"),
		tok.Encode("beta\n\n"),
		concatIDs(tok.Encode("gamma\n"), []int{tok.EOSTokenID()}),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("encodeBlocks = %v, want %v", got, want)
	}
}

func TestEncodePacksSeparatedDocumentsIntoOneWindow(t *testing.T) {
	tok := docSepTokenizer(t)
	const contextSize = 32
	doc := "alpha\n\n" + DocSeparator + "\n\nbeta\n\n" + DocSeparator + "\n\ngamma"
	X, Y := Encode([]string{doc}, contextSize, tok, 0, 0)
	if len(X) != 1 {
		t.Fatalf("short documents should share one window, got %d sequences", len(X))
	}
	eosTargets := 0
	for _, id := range Y[0] {
		if id == tok.EOSTokenID() {
			eosTargets++
		}
	}
	if eosTargets != 3 {
		t.Errorf("want one EOS target per document (3), got %d", eosTargets)
	}
}

func TestDropBlocksSeenInTrainKeepsDocSeparators(t *testing.T) {
	train := []string{"alpha\n\n" + DocSeparator + "\n\nbeta"}
	val := []string{"alpha\n\n" + DocSeparator + "\n\ngamma"}
	got := dropBlocksSeenInTrain(train, val)
	want := []string{DocSeparator + "\n\ngamma"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("dropBlocksSeenInTrain = %q, want %q", got, want)
	}
}

func TestSampleForTokenizerOmitsDocSeparators(t *testing.T) {
	sample := SampleForTokenizer([]string{"alpha\n\n" + DocSeparator + "\n\nbeta"})
	if strings.Contains(sample, DocSeparator) {
		t.Errorf("tokenizer sample must not contain %q: %q", DocSeparator, sample)
	}
}

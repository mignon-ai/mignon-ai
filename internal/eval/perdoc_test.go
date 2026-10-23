package eval

import (
	"math"
	"testing"
)

func TestPerplexityPerDocScoresEveryDocTokenFromAFreshContext(t *testing.T) {
	tok := suiteTokenizer()
	docs := []string{"stage file a", "git add b\ngit add a"}
	text := "\n\n" + docs[0] + "\n\n<EOS>\n\n" + docs[1] + "\n\n"
	want := len(tok.Encode(docs[0])) + len(tok.Encode(docs[1]))

	vocab := tok.VocabSize()
	startsWithEOS := true
	uniform := func(window []int) []float64 {
		// A document's context never reaches back into the previous document.
		if window[0] != tok.EOSTokenID() {
			startsWithEOS = false
		}
		return make([]float64, vocab)
	}
	got := PerplexityPerDoc(uniform, tok, text, suiteTestContext, 0)

	if got.Tokens != want {
		t.Errorf("scored %d tokens, want %d (every token of every document, EOS excluded)", got.Tokens, want)
	}
	if !startsWithEOS {
		t.Error("a document's context did not start at its own EOS")
	}
	if math.Abs(got.Loss-math.Log(float64(vocab))) > 1e-9 {
		t.Errorf("uniform model loss %.4f, want ln(vocab) %.4f", got.Loss, math.Log(float64(vocab)))
	}
	if got.BPB <= 0 {
		t.Errorf("bits per byte %.4f, want positive", got.BPB)
	}
}

func TestPerplexityPerDocMaxDocs(t *testing.T) {
	tok := suiteTokenizer()
	text := "stage file a\n\n<EOS>\n\ngit add b\n\n<EOS>\n\ngit add a"
	uniform := func([]int) []float64 { return make([]float64, tok.VocabSize()) }
	got := PerplexityPerDoc(uniform, tok, text, suiteTestContext, 1)
	if want := len(tok.Encode("stage file a")); got.Tokens != want {
		t.Errorf("scored %d tokens with maxDocs=1, want %d", got.Tokens, want)
	}
}
